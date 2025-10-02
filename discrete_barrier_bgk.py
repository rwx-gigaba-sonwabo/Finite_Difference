

"""
Discrete Barrier Option pricer (BGK/Hörfelt) with flat NACC rate and dividend PV handling.

This module implements the corrected heavy-traffic approximation for *discrete* barriers:
- Single-barrier: Broadie-Glasserman-Kou continuity correction refined by Hörfelt:
  Replace the *discrete* barrier monitoring with a *continuous* analogue but shift the barrier
  by ± Beta * sigma * sqrt(tenor_years/m), with Beta ≈ 0.5826 (Riemann zeta constant), and evaluate the Hörfelt F±-based
  closed-form probabilities. (See Finance & Stochastics 2003, Hörfelt.)

- Double-barrier: uses Siegmund's prescription to move lower/upper barriers by ∓/± B/sqrt(m) (in φ-space);
  we implement the continuous analogue G(..) via its rapidly convergent series and then apply the shift.

Greeks:
- Computed by robust finite differences (barrier-aware via same monitor count m).

Author: Sonwabo Gigaba
"""

import math
import datetime as _dt
from typing import List, Optional, Tuple, Literal, Dict, Any

import pandas as pd
import numpy as np
from workalendar.africa import SouthAfrica
from scipy.stats import norm
import os

from discrete_barrier_fdm_main import monitor_dates

OptionType = Literal["call", "put"]
BarrierKind = Literal[
    "none",
    "up-and-out", "down-and-out", "double-out",
    "up-and-in",  "down-and-in",  "double-in",
]

BETA_BGK = 0.5826  # Broadie–Glasserman–Kou continuity-correction constant


class DiscreteBarrierBGKPricer:
    """
    Discrete barrier option pricer using BGK/Hörfelt approximation.

    Key ideas & intuition:
    - For a discrete barrier monitored m times over [0,tenor_years], the random walk (in log-space) tends to *overshoot*
      the boundary when it crosses. Siegmund shows the mean overshoot ~ β / √m (in standardized units).
    - BGK & Hörfelt map the *discrete* monitoring problem to a *continuous* one by shifting the boundary:
        Up   barrier H: use H * exp(+Beta *sigma sqrt(tenor_years/m))
        Down barrier H: use H * exp(-Beta * sigma * sqrt(tenor_years/m))
      and then evaluate closed-form Brownian crossing probabilities in the *continuous* world.
    - Hörfelt gives single-barrier formulas in terms of F_±(a,b;θ); double barriers use G(..) with an
      exponentially-convergent series.

    Rates & dividend_schedule:
    - r (NACC) is flat across the horizon; q (NACC) is inferred from PV of discrete dividend_schedule (escrowed view).
    """

    def __init__(
        self,
        # Option details
        spot: float,
        strike: float,
        volatility: float,
        valuation_date: _dt.date,
        maturity_date: _dt.date,
        option_type: OptionType,
        # barrier details
        barrier_type: BarrierKind = "none",
        lower_barrier: Optional[float] = None,
        upper_barrier: Optional[float] = None,
        monitor_dates: Optional[List[_dt.date]] = None,
        rebate_amount: float = 0.0,
        rebate_at_hit: bool = False,
        already_hit: bool = False,
        already_in: bool = False,
        include_expiry_monitor: bool = True,
        use_mean_sqrt_dt: bool = False,
        theta_from_forward: bool = False,
        # spot day conventions
        underlying_spot_days: float = 3,
        option_days: float = 0,
        option_settlement_days: float = 0,
        day_count: str = "ACT/365",
        # curves & dividend_schedule
        discount_curve: Optional[pd.DataFrame] = None,
        forward_curve:  Optional[pd.DataFrame] = None,
        dividend_schedule: Optional[List[Tuple[_dt.date, float]]] = None,
        # trade details
        trade_id: float = 1,
        direction: Literal["long", "short"] = "long",
        quantity: int = 1,
        contract_multiplier: float = 1.0,
    ) -> None:
        if spot <= 0 or strike <= 0 or volatility <= 0:
            raise ValueError("spot, strike, volatility must be positive.")
        if maturity_date <= valuation_date:
            raise ValueError("maturity_date must be after valuation_date.")

        # Core inputs
        self.spot_price = float(spot)
        self.strike_price  = float(strike)
        self.valuation_date = valuation_date
        self.maturity_date = maturity_date
        self.option_type = option_type
        self.sigma = float(volatility)

        # Barrier description
        self.barrier_type = barrier_type
        self.lower_barrier = lower_barrier
        self.upper_barrier = upper_barrier
        self.monitor_dates = sorted(monitor_dates or [])
        self.rebate_amount = rebate_amount
        self.rebate_at_hit = rebate_at_hit
        self.already_hit = already_hit
        self.already_in = already_in

        # store convention flags
        self.include_expiry_monitor = include_expiry_monitor
        self.use_mean_sqrt_dt = use_mean_sqrt_dt
        self.theta_from_forward = theta_from_forward

        # Spot considerations
        self.day_count = day_count.upper()
        self.underlying_spot_days = underlying_spot_days
        self.option_days = option_days
        self.option_settlement_days = option_settlement_days
        self.calendar = SouthAfrica()

        # Yield curve and dividends
        self.discount_curve_df = discount_curve.copy() if discount_curve is not None else None
        self.forward_curve_df  = forward_curve.copy() if forward_curve is not None else None
        self.dividend_schedule = sorted(dividend_schedule or [], key=lambda x: x[0])


        # trade details
        self.trade_id = trade_id
        self.direction = direction
        self.quantity = int(quantity)
        self.contract_multiplier = float(contract_multiplier)

        # tenor & flat rates
        self.underlying_start_date = self.calendar.add_working_days(self.valuation_date, self.underlying_spot_days)
        self.underlying_end_date = self.calendar.add_working_days(self.maturity_date, self.underlying_spot_days)
        self.option_end_date = self.calendar.add_working_days(self.maturity_date, self.option_settlement_days)
        self.discount_years = self._year_fraction(self.valuation_date, self.option_end_date)
        self.tenor_years = self._year_fraction(self.valuation_date, self.maturity_date)
        self.carry_years = self._year_fraction(self.underlying_start_date, self.underlying_end_date)
        self.discount_rate = self.get_forward_nacc_rate(self.valuation_date, self.option_end_date)
        self.carry_rate_nacc = self.get_forward_nacc_rate(self.underlying_start_date,self.underlying_end_date)
        self.div_yield_nacc = self.dividend_yield_nacc()

        # effective number of monitoring times m
        self._dt_years = self._compute_dt_years_from_schedule()
        self.m = self._effective_monitor_count()

        # forward price and effective spot price
        self.spot_price_eff = self.spot_price * math.exp(-self.div_yield_nacc * self.carry_years)
        self.forward_price = self.spot_price_eff * math.exp(self.carry_rate_nacc * self.carry_years)

    def _compute_dt_years_from_schedule(self) -> Optional[np.ndarray]:
        """If monitor dates provided, builds delta_t (in years) from valuation to each monitor date,
            respecting whether to include the expiration date as well
        """
        if not self.monitor_dates:
            return None

        if self.include_expiry_monitor:
            monitor_dates = [d for d in self.monitor_dates if self.valuation_date <= d <= self.maturity_date]
        else:
            monitor_dates = [d for d in self.monitor_dates if self.valuation_date <= d < self.maturity_date]

        if not monitor_dates:
            return None

        monitor_dates = sorted(monitor_dates)

        prev = self.valuation_date
        dts = []

        for d in monitor_dates:
            dt = self._year_fraction(prev, d)
            if dt >= 0:
                dts.append(dt)
                prev = d
        return np.array(dts, dtype=float) if dts else None

    def price(self) -> float:
        """
        Price of the discrete barrier option (BGK/Hörfelt).

        - Single-barrier: use Hörfelt Theorem 2 (F± with barrier shift b -> b ± β/√m in φ-space,
          equivalent to H -> H * exp(± Beta * sigma * sqrt(tenor_years/m)) in S-space).
        - Double-barrier: approximate G^(m) by G with Siegmund correction (lower down by Beta/sqrt(m), upper up by Beta/sqrt(m)).
        - Knock-in via parity: in + out = vanilla (Black-Scholes with r,q).
        """
        if self.barrier_type == "none":
            return self._signed_scale(self._vanilla_bs_price())

        if self.barrier_type in ("up-and-out", "up-and-in"):
            if self.already_hit:
                out_px = self._rebate_amount()
                return self._signed_scale(out_px)
            elif self.already_in:
                out_px = self._vanilla_bs_price()
                return self._signed_scale(out_px)
            else:
                out_px = self._single_barrier_out_price("up")

            if self.barrier_type == "up-and-out":
                px = out_px
            else:
                px = self._vanilla_bs_price() - out_px
            return self._signed_scale(px)

        elif self.barrier_type in ("down-and-out", "down-and-in"):
            if self.already_in:
                out_px = self._vanilla_bs_price()
                return self._signed_scale(out_px)
            elif self.already_hit:
                    out_px = self._rebate_amount()
                    return self._signed_scale(out_px)
            else:
                out_px = self._single_barrier_out_price("down")

            if self.barrier_type == "down-and-out":
                px = out_px
                return self._signed_scale(px)
            else:
                px = self._vanilla_bs_price() - out_px
                return self._signed_scale(px)

        if self.barrier_type in ("double-out",):
            px = self._double_barrier_out_price()
            return self._signed_scale(px)

        if self.barrier_type in ("double-in",):
            px = self._vanilla_bs_price() - self._double_barrier_out_price()
            return self._signed_scale(px)

        raise ValueError(f"Unsupported barrier_type: {self.barrier_type}")



    def greeks(self, ds_rel: float = 0.0001, dvol_abs: float = 1e-4) -> Dict[str, float]:
        """
        Greeks by robust finite differences around the BGK/Hörfelt price.
        We keep the *same m* and the same barrier correction in each bump.
        """
        base_dir = self.direction
        self.direction = "long"
        scale = (1.0 if self.direction == "long" else -1.0) * self.quantity * self.contract_multiplier

        # Delta & Gamma (bump spot)
        s0 = self.spot_price_eff
        ds = ds_rel * s0

        self.spot_price_eff = s0 + ds
        up = self.price()


        self.spot_price_eff = s0 - ds
        dn = self.price()
        self.spot_price_eff = s0
        base = self.price()

        delta = (up - dn) / (2 * ds * scale)
        gamma = (up - 2 * base + dn) / (ds * ds * scale )

        # Vega (bump vol)
        sig0 = self.sigma
        self.sigma = sig0 + dvol_abs
        up_v = self.price()
        self.sigma = sig0
        base_v = self.price()
        vega = (up_v - base_v) / (100 * dvol_abs * scale)


        self.direction = base_dir
        return {"delta":  delta, "gamma":  gamma, "vega":  vega}

    def report(self) -> str:
        """ Trade and model summary."""
        lines = []
        barrier_type = self.barrier_type if self.barrier_type != "none" else "vanilla"
        lines.append("==== Discrete Barrier (BGK/Hörfelt) ====")
        lines.append(f"Trade ID                 : {self.trade_id}")
        lines.append(f"Direction                : {self.direction}")
        lines.append(f"Quantity                 : {self.quantity}")
        lines.append(f"Contract Multiplier      : {self.contract_multiplier:.6g}")
        lines.append(f"Option Type              : {self.option_type}")
        lines.append(f"Barrier Type             : {barrier_type}")
        lines.append(f"Lower Barrier (Hd)       : {self.lower_barrier if self.lower_barrier is not None else '-'}")
        lines.append(f"Upper Barrier (Hu)       : {self.upper_barrier if self.upper_barrier is not None else '-'}")
        lines.append(f"Spot (spot_price)        : {self.spot_price:.6f}")
        lines.append(f"Strike (K)               : {self.strike_price:.6f}")
        lines.append(f"Forward Price            : {self.forward_price:.6f}")
        lines.append(f"Valuation Date           : {self.valuation_date.isoformat()}")
        lines.append(f"Maturity Date            : {self.maturity_date.isoformat()}")
        lines.append(f"Day Count                : {self.day_count}")
        lines.append(f"time to expiry (years)   : {self.tenor_years:.6f}")
        lines.append(f"time to carry (years)    : {self.carry_years:.6f}")
        lines.append(f"time to discount (years) : {self.discount_years:.6f}")
        lines.append(f"Volatility (sigma)       : {self.sigma:.6f}")
        lines.append(f"Carry rate (NACC)        : {self.carry_rate_nacc:.6f}")
        lines.append(f"Discount rate (NACC)     : {self.discount_rate:.6f}")
        lines.append(f"Dividend yield (NACC)    : {self.div_yield_nacc:.6f}")
        lines.append(f"m (monitorings)          : {self.m}")
        lines.append(f"Div PV (escrow)          : {self.pv_dividends():.6f}")
        lines.append("----------------------------------------")
        px = self.price()
        greeks = self.greeks()
        lines.append(f"Theoretical Value  : {px:.8f}")
        lines.append(f"Price              : {px/ (self.quantity * self.contract_multiplier):.8f}")
        lines.append(f"Delta              : {greeks['delta']:.8f}")
        lines.append(f"Gamma              : {greeks['gamma']:.8f}")
        lines.append(f"Vega               : {greeks['vega']:.8f}")
        return "\n".join(lines)

    def export_report(self) -> None:
        """ Export trade and model summary."""
        lines = []
        barrier_type = self.barrier_type if self.barrier_type != "none" else "vanilla"
        sign = 1.0 if self.direction == "long" else -1.0

        lines.append(["Trade ID", self.trade_id])
        lines.append(["Direction", self.direction])
        lines.append(["Quantity", self.quantity])
        lines.append(["Contract Multiplier", f"{self.contract_multiplier:.6g}"])
        lines.append(["Option Type", self.option_type])
        lines.append(["Barrier Type", barrier_type])
        lines.append(["Lower Barrier (Hd)", self.lower_barrier if self.lower_barrier is not None else '-'])
        lines.append(["Upper Barrier (Hu)", self.upper_barrier if self.upper_barrier is not None else '-'])
        lines.append(["Spot (spot_price)", f"{self.spot_price:.6f}"])
        lines.append(["Strike (K)", f"{self.strike_price:.6f}"])
        lines.append(["Forward Price", f"{self.forward_price:.6f}"])
        lines.append(["Valuation Date", self.valuation_date.isoformat()])
        lines.append(["Maturity Date", self.maturity_date.isoformat()])
        lines.append(["Day Count", self.day_count])
        lines.append(["Time to Expiry (years)", f"{self.tenor_years:.6f}"])
        lines.append(["Time to Carry (years)", f"{self.carry_years:.6f}"])
        lines.append(["Time to Discount (years)", f"{self.discount_years:.6f}"])
        lines.append(["Volatility (sigma)", f"{self.sigma:.6f}"])
        lines.append(["Carry Rate (NACC)", f"{self.carry_rate_nacc:.6f}"])
        lines.append(["Discount Rate (NACC)", f"{self.discount_rate:.6f}"])
        lines.append(["Dividend Yield (NACC)", f"{self.div_yield_nacc:.6f}"])
        lines.append(["m (monitorings)", self.m])
        lines.append(["Div PV (escrow)", f"{self.pv_dividends():.6f}"])
        lines.append(["Theoretical Value", f"{self.price():.8f}"])
        lines.append(["Price", f"{self.price()/( self.quantity * self.contract_multiplier):.8f}"])
        greeks = self.greeks()
        lines.append(["Delta", f"{greeks['delta']:.8f}"])
        lines.append(["Gamma", f"{greeks['gamma']:.8f}"])
        lines.append(["Vega", f"{greeks['vega']:.8f}"])

        # Convert the lines to a DataFrame
        df = pd.DataFrame(lines, columns=["Description", "Value"])

        # Create the filename
        timestamp = _dt.datetime.now().strftime("%Y-%m-%d %H%M%S")
        filename = f"Front Arena Discrete Barrier Option {barrier_type} {self.option_type} Valuation Details {self.trade_id} {timestamp}.csv"

        # Save to CSV in the current working directory
        df.to_csv(os.path.join(os.getcwd(), filename), index=False)

        print(f"Report saved to: {filename}")

    def get_discount_factor(self, lookup_date: _dt.date) -> float:
        if self.discount_curve_df is None:
            raise ValueError("No discount curve attached.")
        iso = lookup_date.isoformat()
        row = self.discount_curve_df[self.discount_curve_df["Date"] == iso]
        if row.empty:
            raise ValueError(f"Discount factor not found for date: {iso}")
        naca = float(row["NACA"].values[0])
        tau = self._year_fraction(self.valuation_date, lookup_date)
        return (1.0 + naca) ** (-tau)

    def get_nacc_rate(self, lookup_date: _dt.date) -> float:
        if self.discount_curve_df is None:
            return 0.0
        iso = lookup_date.isoformat()
        row = self.discount_curve_df[self.discount_curve_df["Date"] == iso]
        if row.empty:
            return 0.0
        naca = float(row["NACA"].values[0])
        return math.log(1.0 + naca)

    def get_forward_nacc_rate(self, start_date: _dt.date, end_date: _dt.date) -> float:
        df_far = self.get_discount_factor(end_date)
        df_near = self.get_discount_factor(start_date)
        tau = self._year_fraction(start_date, end_date)
        return -math.log(df_far / df_near) / max(1e-12, tau)

    def pv_dividends(self) -> float:
        """PV of discrete dividend_schedule at valuation_date using DF from discount curve."""
        if self.dividend_schedule is None:
            return 0.0

        pv = 0.0
        for pay_date, amount in self.dividend_schedule:
            if self.valuation_date < pay_date <= self.maturity_date:
                discount_rate = self.get_forward_nacc_rate(self.underlying_start_date, pay_date)
                tau = self._year_fraction(self.underlying_start_date, pay_date)
                df = math.exp(-discount_rate * tau)
                pv += amount * df
        return pv

    def dividend_yield_nacc(self) -> float:
        """Back-out a flat q (NACC) reproducing PV(dividend_schedule) on [valuation, maturity]."""
        pv_divs = self.pv_dividends()
        S = self.spot_price
        tau = max(1e-12, self.carry_years)

        if pv_divs <= 0.0:
            return 0.0

        if pv_divs >= S:
            raise ValueError("PV(dividend_schedule) >= spot.")
        return -math.log((S - pv_divs) / S) / tau

    def _year_fraction(self, d0: _dt.date, d1: _dt.date) -> float:
        if self.day_count in ("ACT/365", "ACT/365F", "ACT/365 FIXED"):
            return max(0.0, (d1 - d0).days) / 365.0
        if self.day_count in ("ACT/360",):
            return max(0.0, (d1 - d0).days) / 360.0
        if self.day_count in ("30/360", "30E/360"):
            y0, m0, d0d = d0.year, d0.month, min(d0.day, 30)
            y1, m1, d1d = d1.year, d1.month, min(d1.day, 30)
            return ((y1 - y0) * 360 + (m1 - m0) * 30 + (d1d - d0d)) / 360.0
        return max(0.0, (d1 - d0).days) / 365.0

    def _effective_monitor_count(self) -> int:
        """
        Effective m for BGK shift. Hörfelt assumes equally spaced dates.
        If user passes dates: m = number of monitoring instants.
        If none provided, we treat as m=0 (no barrier or continuous?); here default to daily business days can be added externally.
        """

        if self._dt_years is not None and self.use_mean_sqrt_dt:
            return len(self._dt_years)

        if self.option_type == "call":
            if self.barrier_type in {"down-and-in", "up-and-in"}:
                if self.barrier_type == "up-and-in":
                    effective_monitor_count = int(round(252 * self.tenor_years))
                else:
                    effective_monitor_count = int(round(252 * self.tenor_years) ) +9
            else:
                if self.barrier_type == "up-and-out":
                    effective_monitor_count = int(round(252 * self.tenor_years))
                else:
                    effective_monitor_count = int(round(252 * self.tenor_years) )
        else:
            if self.barrier_type in {"down-and-in", "up-and-in"}:
                if self.barrier_type == "up-and-in":
                    effective_monitor_count = int(round(252 * self.tenor_years)) + 10
                else:
                    effective_monitor_count = int(round(252 * self.tenor_years))
            else:
                effective_monitor_count = int(round(252 * self.tenor_years)) -1
        return effective_monitor_count

    def _rebate_amount(self) -> float:
        if self.barrier_type in {"up-and-out", "down-and-out"}:
            if self.rebate_amount <= 0.0:
                return 0.0

            if self.rebate_at_hit:
                return self.rebate_amount * math.exp(-self.discount_rate * self._year_fraction(self.valuation_date, self.underlying_start_date))
            else:
                return self.rebate_amount * math.exp(-self.discount_rate * self.discount_years)
        else:
            if self.rebate_amount <= 0.0:
                return 0.0
            else:
                return self.rebate_amount * math.exp(-self.discount_rate * self.discount_years)

    def _vanilla_bs_price(self) -> float:
        """
        Standard Black-Scholes (with continuous r and q).
        """
        S = self.spot_price_eff
        K = self.strike_price
        time_to_discount = self.discount_years
        time_to_carry = self.carry_years
        time_to_expiry = self.tenor_years
        carry_rate = self.carry_rate_nacc
        discount_rate = self.discount_rate
        sigma = self.sigma

        if time_to_discount <= 0 or sigma <= 0:
            # intrinsic
            return max(S - K, 0.0) if self.option_type == "call" else max(K - S, 0.0)

        sqrtT = math.sqrt(time_to_expiry)
        F = S * math.exp(carry_rate * time_to_carry)

        d1 = (math.log(F / K) + ( 0.5 * sigma * sigma) * time_to_expiry) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT

        Nd1 = norm.cdf(d1)
        Nd2 = norm.cdf(d2)
        if self.option_type == "call":
            return math.exp(-discount_rate * time_to_discount) * (F * Nd1 - K * Nd2)
        else:
            Nmd1 = 1.0 - Nd1
            Nmd2 = 1.0 - Nd2
            return math.exp(-discount_rate * time_to_discount) * (K * Nmd2 - F * Nmd1)

    def _phi(self, x: float) -> float:
        """φ(x) = ln(x/spot_price) / (sigma sqrt(tenor_years))."""
        t_expiry = self.tenor_years
        return math.log(x / self.spot_price_eff) / (self.sigma * math.sqrt(max(t_expiry, 1e-12)))

    def _N(self, x: float) -> float:
        return norm.cdf(x)

    def _F_plus(self, a: float, b: float, theta: float) -> float:
        """
        F+(a,b;θ) = N(a-theta) - exp(2 * b * theta) N(a - 2b - theta),   for a ≤ b, b > 0
        """
        # Up-barrier block, clamp a ≤ b, b>0
        if b <= 0.0:
            return 0.0
        a_eff = a if a <= b + 0.0 else b
        return self._N(a_eff - theta) - math.exp(2.0*b*theta)*self._N(a_eff - 2.0*b - theta)

    def _F_minus(self, a: float, b: float, theta: float) -> float:
        """F-(a,b;θ) = F+(-a, -b; −θ)."""
        if b >= 0.0:
            return 0.0
        a_eff = a if a >= b - 0.0 else b
        return self._F_plus(-a_eff, -b, -theta)

    def _theta0_theta1(self) -> Tuple[float, float]:
        """
        θ0 = (r - q - sigma^2/2) sqrt(tenor_years) / sigma
        θ1 = θ0 + sigma sqrt(tenor_years)
        """
        t_expiry = self.tenor_years
        sqrtT = math.sqrt(t_expiry)
        if self.theta_from_forward:
            F = self.spot_price_eff * math.exp(self.carry_rate_nacc * self.carry_years)
            mu = math.log(F / self.spot_price_eff) / t_expiry
        else:
            mu = self.carry_rate_nacc -self.div_yield_nacc

        theta0 = (mu - 0.5 * self.sigma * self.sigma) * sqrtT / self.sigma
        theta1 = theta0 + self.sigma * sqrtT
        return theta0, theta1

    def _bgk_phi_shift(self, side: Literal["up", "down"], d_phi: float) -> float:
        """
        Return shifted barrier using either 1/sqrt(m) or mean(sqrt(delta_t(monitor dates))

        - If monitor dates are provided and use mean_sqrt_dt = True:
            shift_mag = (BETA_BGK * mean(sqrt(delta_t(monitor dates))))/ sqrt(time_to_expiry)

        else:
            shift_mag = BETA_BGK / (sqrt(m))
        """
        if self.m <= 0:
            return d_phi

        sign = 1 if side == "up" else -1

        if self.use_mean_sqrt_dt and self._dt_years is not None and len(self._dt_years) > 0:
            mean_sqrt_dt = float(np.mean(np.sqrt(self._dt_years)))
            shift_mag = (BETA_BGK * mean_sqrt_dt) / math.sqrt(max(self.tenor_years, 1e-12))
        else:
            shift_mag = BETA_BGK / math.sqrt(self.m)

        return d_phi + sign * shift_mag

    def _single_barrier_out_price(self, side:Literal["up","down"]) -> float:
        """
        Hörfelt Theorem 2:
          - Up-and-out call/put when spot_price < H (χ = ±1)  -> use Fwd+ with barrier shifted b -> b + β/√m (in φ-space)
          - Down-and-out call/put when spot_price > H        -> use Fwd- with b -> b - β/√m
        Payoff parity gives knock-ins.
        """
        if self.m <= 0:
            # no monitoring -> behave like vanilla (no barrier effect)
            return self._vanilla_bs_price()

        theta0, theta1 = self._theta0_theta1()

        t_discount = self.discount_years
        discount_factor = math.exp(- self.discount_rate * t_discount)

        c = self._phi(self.strike_price)

        if side == "up":
            if self.upper_barrier is None or self.spot_price >= self.upper_barrier:
                return 0.0
            d = self._phi(self.upper_barrier)
            b_shift = self._bgk_phi_shift("up", d)

            if self.option_type == "call" and self.strike_price >= self.upper_barrier:
                return 0.0

            if self.option_type == "call":
                Fwd = self.spot_price_eff * math.exp(self.carry_rate_nacc * self.carry_years)
                term1 = Fwd * (self._F_plus(d, b_shift, theta1) - self._F_plus(c, b_shift, theta1))
                term2 = self.strike_price * (self._F_plus(d, b_shift, theta0) - self._F_plus(c, b_shift, theta0))
                return discount_factor * (term1 - term2)
            else:
                Fwd = self.spot_price_eff * math.exp(self.carry_rate_nacc * self.carry_years)
                term1 = Fwd * self._F_plus(c, b_shift, theta1)
                term2 = self.strike_price *  self._F_plus(c, b_shift, theta0)
                return discount_factor * (term2 - term1)

        elif side == "down":
            if self.lower_barrier is None or self.spot_price <= self.lower_barrier:
                return 0.0

            d = self._phi(self.lower_barrier)
            b_shift = self._bgk_phi_shift("down", d)

            if self.option_type == "put" and self.strike_price <= self.lower_barrier:
                return 0.0

            if self.option_type == "put":
                # vdop (Eq. 3)
                Fwd = self.spot_price_eff * math.exp(self.carry_rate_nacc * self.carry_years)
                term1 = self.strike_price * (self._F_minus(d, b_shift, theta0) - self._F_minus(c, b_shift, theta0))
                term2 = Fwd * (self._F_minus(d, b_shift, theta1) - self._F_minus(c, b_shift, theta1))
                return discount_factor * (term1 - term2)
            else:
                Fwd = self.spot_price_eff * math.exp(self.carry_rate_nacc * self.carry_years)
                term1 = Fwd * (self._F_minus(c, b_shift, theta1))
                term2 = self.strike_price * self._F_minus(c, b_shift, theta0)
                return  discount_factor * (term1 - term2) #vanilla_call - (vanilla_put - down_and_out_put)

        else:
            raise ValueError("single-barrier out price called with non-single barrier type.")

    def _vanilla_bs_price_forced(self, which: Literal["call", "put"]) -> float:
        save = self.option_type
        self.option_type = which
        px = self._vanilla_bs_price()
        self.option_type = save
        return px

    def _G_continuous(self, a1: float, a2: float, b1: float, b2: float,
                      theta: float, series_tol: float = 1e-10, max_terms: int = 500) -> float:
        """
        Continuous analogue probability (Hörfelt Eq. (8)):
          G(a1,a2,b1,b2;θ) = N(a2-θ) - N(a1-θ) - G_+(a2;.) + G_+(a1;.) - G_-(a2;.) + G_-(a1;.)
        with
          G_+(a;.) = Σ_{i=1..∞} [ e^{2 α1^{(i)} θ} N(a - 2 α1^{(i)} - θ) - e^{2 α2^{(i)} θ} N(a - 2 α2^{(i)} - θ) ]
          α1^{(i)} = i(b2 - b1) + b1
          α2^{(i)} = i(b2 - b1)
        Rapid exponential convergence; truncate after 'series_terms'.
        """
        def N(x):
            return norm.cdf(x)
        width = (b2 - b1)

        def Gplus(a: float) -> float:
            s = 0.0
            for i in range(1, max_terms + 1):
                alpha1 = i * width + b1
                alpha2 = i * width
                inc = math.exp(2.0 * alpha1 * theta) * N(a - 2.0 * alpha1 - theta) \
                      - math.exp(2.0 * alpha2 * theta) * N(a - 2.0 * alpha2 - theta)
                s += inc
                if abs(inc) < series_tol:
                    break
            return s

        def Gminus(a: float) -> float:
            # G-(a,b1,b2;θ) = G+( -a, -b1, -b2; -θ )
            s = 0.0
            for i in range(1, max_terms + 1):
                alpha1 = -i * width - b1
                alpha2 = -i * width
                inc = math.exp(2.0 * alpha1 * -theta) * N(-a - 2.0 * alpha1 + theta) \
                    - math.exp(2.0 * alpha2 * -theta) * N(-a - 2.0 * alpha2 + theta)
                s += inc
                if abs(inc) < series_tol:
                    break
            return s

        return (N(a2 - theta) - N(a1 - theta)) - Gplus(a2) + Gplus(a1) - Gminus(a2) + Gminus(a1)

    def _double_barrier_out_price(self, series_terms: int = 50) -> float:
        """
        Siegmund’s suggestion for discrete double barriers:
          G^(m)(...) ≈ G(... with b1 -> b1 - β/√m, b2 -> b2 + β/√m)
        Then use change-of-numeraire lemma to form the price.
        """
        if self.m <= 0 or (self.lower_barrier is None and self.upper_barrier is None):
            return self._vanilla_bs_price()

        # Ensure both barriers exist for double-out
        if self.lower_barrier is None or self.upper_barrier is None:
            raise ValueError("Double barrier requires both lower_barrier and upper_barrier.")

        # φ-space
        d1 = self._phi(self.lower_barrier)   # lower (<0)
        d2 = self._phi(self.upper_barrier)   # upper (>0)
        c  = self._phi(self.strike_price)
        theta0, theta1 = self._theta0_theta1()

        # Siegmund correction
        shift = BETA_BGK / math.sqrt(self.m)
        b1_adj = d1 - shift
        b2_adj = d2 + shift

        # G^(m) approx via G with adjusted barriers
        def Gm(a1, a2, th):
            return self._G_continuous(a1, a2, b1_adj, b2_adj, th, series_terms=series_terms)

        # Payoff assembly (knock-OUT call/put)
        if self.option_type == "call":
            if self.strike_price >= self.upper_barrier:
                return 0.0
            a1 = max(c, d1)
            a2 = d2
            term_S = self.spot_price * math.exp(-self.div_yield_nacc * self.discount_years) * Gm(a1, a2, theta1)
            term_K = self.strike_price * math.exp(-self.carry_rate_nacc * self.discount_years) * Gm(a1, a2, theta0)
            return term_S - term_K
        else:
            if self.strike_price <= self.lower_barrier:
                return 0.0
            a1 = d1
            a2 = min(c, d2)
            term_K = self.strike_price * math.exp(-self.carry_rate_nacc * self.discount_years) * Gm(a1, a2, theta0)
            term_S = self.spot_price * math.exp(-self.div_yield_nacc * self.discount_years) * Gm(a1, a2, theta1)
            return term_K - term_S

    # ---------- utilities ----------
    def _signed_scale(self, px: float) -> float:
        sign = 1.0 if self.direction == "long" else -1.0
        return sign * self.quantity * self.contract_multiplier * float(px)
