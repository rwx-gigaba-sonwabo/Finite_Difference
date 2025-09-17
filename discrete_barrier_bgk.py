import math
import datetime as dt
from typing import List, Optional, Tuple, Literal, Dict, Any

import pandas as pd

"""
Discrete Barrier Option pricer (BGK/Hörfelt) with flat NACC rate and dividend PV handling.

This module implements the corrected heavy-traffic approximation for *discrete* barriers:
- Single-barrier: Broadie-Glasserman-Kou continuity correction refined by Hörfelt:
  Replace the *discrete* barrier monitoring with a *continuous* analogue but shift the barrier
  by ± Beta * sigma * sqrt(T/m), with Beta ≈ 0.5826 (Riemann zeta constant), and evaluate the Hörfelt F±-based
  closed-form probabilities. (See Finance & Stochastics 2003, Hörfelt.)

- Double-barrier: uses Siegmund's prescription to move lower/upper barriers by ∓/± B/sqrt(m) (in φ-space);
  we implement the continuous analogue G(..) via its rapidly convergent series and then apply the shift.

Greeks:
- Computed by robust finite differences (barrier-aware via same monitor count m).

Author: Sonwabo Gigaba
"""
from __future__ import annotations

import math
import datetime as _dt
from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal, Dict, Any

import pandas as pd


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
    - For a discrete barrier monitored m times over [0,T], the random walk (in log-space) tends to *overshoot*
      the boundary when it crosses. Siegmund shows the mean overshoot ~ β / √m (in standardized units).
    - BGK & Hörfelt map the *discrete* monitoring problem to a *continuous* one by shifting the boundary:
        Up   barrier H: use H * exp(+Beta *sigma sqrt(T/m))
        Down barrier H: use H * exp(-Beta * sigma * sqrt(T/m))
      and then evaluate closed-form Brownian crossing probabilities in the *continuous* world.
    - Hörfelt gives single-barrier formulas in terms of F_±(a,b;θ); double barriers use G(..) with an
      exponentially-convergent series.

    Rates & dividends:
    - r (NACC) is flat across the horizon; q (NACC) is inferred from PV of discrete dividends (escrowed view).
    """

    def __init__(
        self,
        spot: float,
        strike: float,
        valuation_date: _dt.date,
        maturity_date: _dt.date,
        option_type: OptionType,
        barrier_type: BarrierKind = "none",
        lower_barrier: Optional[float] = None,
        upper_barrier: Optional[float] = None,
        monitor_dates: Optional[List[_dt.date]] = None,
        # curves & dividends
        discount_curve: Optional[pd.DataFrame] = None,
        forward_curve:  Optional[pd.DataFrame] = None,
        dividend_schedule: Optional[List[Tuple[_dt.date, float]]] = None,
        # model parameters
        volatility: float = 0.2, 
        day_count: str = "ACT/365",
        # trade details
        trade_id: str = "T-0001",
        direction: Literal["long", "short"] = "long",
        quantity: int = 1,
        contract_multiplier: float = 1.0,
    ) -> None:
        if spot <= 0 or strike <= 0 or volatility <= 0:
            raise ValueError("spot, strike, volatility must be positive.")
        if maturity_date <= valuation_date:
            raise ValueError("maturity_date must be after valuation_date.")

        # store
        self.S0 = float(spot)
        self.K  = float(strike)
        self.val_date = valuation_date
        self.mat_date = maturity_date
        self.opt_type = option_type
        self.barrier_type = barrier_type
        self.Hd = lower_barrier
        self.Hu = upper_barrier
        self.monitor_dates = sorted(monitor_dates or [])
        self.discount_curve_df = discount_curve.copy() if discount_curve is not None else None
        self.forward_curve_df  = forward_curve.copy() if forward_curve is not None else None
        self.dividends = sorted(dividend_schedule or [], key=lambda x: x[0])
        self.sigma = float(volatility)
        self.day_count = day_count.upper()

        # trade details
        self.trade_id = trade_id
        self.direction = direction
        self.quantity = int(quantity)
        self.contract_multiplier = float(contract_multiplier)

        # tenor & flat rates
        self.T = self._year_fraction(self.val_date, self.mat_date)
        self.r_nacc = self._flat_r_nacc_from_curve()  # continuous-compounded short rate
        self.q_nacc = self._flat_q_nacc_from_dividends()  # effective continuous dividend yield

        # effective number of monitoring times m
        self.m = self._effective_monitor_count()

    def price(self) -> float:
        """
        Price of the discrete barrier option (BGK/Hörfelt).

        - Single-barrier: use Hörfelt Theorem 2 (F± with barrier shift b -> b ± β/√m in φ-space,
          equivalent to H -> H * exp(± Beta * sigma * sqrt(T/m)) in S-space).
        - Double-barrier: approximate G^(m) by G with Siegmund correction (lower down by Beta/sqrt(m), upper up by Beta/sqrt(m)).
        - Knock-in via parity: in + out = vanilla (Black-Scholes with r,q).
        """
        if self.barrier_type == "none":
            return self._signed_scale(self._vanilla_bs_price())

        if self.barrier_type in ("up-and-out", "down-and-out"):
            px = self._single_barrier_out_price()
        elif self.barrier_type in ("up-and-in", "down-and-in"):
            vanilla = self._vanilla_bs_price()
            out_equiv = self._single_barrier_out_price()  # same parameters, out-type
            px = vanilla - out_equiv
        elif self.barrier_type in ("double-out",):
            px = self._double_barrier_out_price()
        elif self.barrier_type in ("double-in",):
            vanilla = self._vanilla_bs_price()
            out_equiv = self._double_barrier_out_price()
            px = vanilla - out_equiv
        else:
            raise ValueError(f"Unsupported barrier_type: {self.barrier_type}")

        return self._signed_scale(px)

    def greeks(self, ds_rel: float = 1e-4, dvol_abs: float = 1e-4) -> Dict[str, float]:
        """
        Greeks by robust finite differences around the BGK/Hörfelt price.
        We keep the *same m* and the same barrier correction in each bump.
        """
        base_dir = self.direction
        self.direction = "long"

        # Delta & Gamma (bump spot)
        s0 = self.S0
        ds = max(1e-8, ds_rel * s0)
        self.S0 = s0 + ds; up = self.price()
        self.S0 = s0 - ds; dn = self.price()
        self.S0 = s0
        base = self.price()

        delta = (up - dn) / (2 * ds)
        gamma = (up - 2 * base + dn) / (ds * ds)

        # Vega (bump vol)
        sig0 = self.sigma
        self.sigma = sig0 + dvol_abs; upv = self.price()
        self.sigma = sig0 - dvol_abs; dnv = self.price()
        self.sigma = sig0
        vega = (upv - dnv) / (2 * dvol_abs)

        self.direction = base_dir
        scale = (1.0 if self.direction == "long" else -1.0) * self.quantity * self.contract_multiplier
        return {"delta": scale * delta, "gamma": scale * gamma, "vega": scale * vega}

    def report(self) -> str:
        """Human-friendly trade and model summary."""
        lines = []
        lines.append("==== Discrete Barrier (BGK/Hörfelt) ====")
        lines.append(f"Trade ID           : {self.trade_id}")
        lines.append(f"Direction          : {self.direction}")
        lines.append(f"Quantity           : {self.quantity}")
        lines.append(f"Contract Multiplier: {self.contract_multiplier:.6g}")
        lines.append(f"Option Type        : {self.opt_type}")
        lines.append(f"Barrier Type       : {self.barrier_type}")
        lines.append(f"Lower Barrier (Hd) : {self.Hd if self.Hd is not None else '-'}")
        lines.append(f"Upper Barrier (Hu) : {self.Hu if self.Hu is not None else '-'}")
        lines.append(f"Spot (S0)          : {self.S0:.6f}")
        lines.append(f"Strike (K)         : {self.K:.6f}")
        lines.append(f"Valuation Date     : {self.val_date.isoformat()}")
        lines.append(f"Maturity Date      : {self.mat_date.isoformat()}")
        lines.append(f"Day Count          : {self.day_count}")
        lines.append(f"T (years)          : {self.T:.6f}")
        lines.append(f"Volatility (sigma) : {self.sigma:.6f}")
        lines.append(f"Flat r (NACC)      : {self.r_nacc:.6f}")
        lines.append(f"Flat q (NACC)      : {self.q_nacc:.6f}")
        lines.append(f"m (monitorings)    : {self.m}")
        lines.append(f"Div PV (escrow)    : {self._pv_dividends():.6f}")
        lines.append("----------------------------------------")
        px = self.price()
        greeks = self.greeks()
        lines.append(f"Price              : {px:.8f}")
        lines.append(f"Delta              : {greeks['delta']:.8f}")
        lines.append(f"Gamma              : {greeks['gamma']:.8f}")
        lines.append(f"Vega               : {greeks['vega']:.8f}")
        return "\n".join(lines)

    def _normalize_curve_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Date" not in df.columns or "NACA" not in df.columns:
            raise ValueError("Curve DataFrame must have columns: 'Date', 'NACA'.")
        out = df.copy()
        if not pd.api.types.is_string_dtype(out["Date"]):
            out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
        return out

    def _discount_factor_from_curve(self, lookup_date: _dt.date) -> float:
        if self.discount_curve_df is None:
            raise ValueError("No discount curve provided.")
        df = self._normalize_curve_df(self.discount_curve_df)
        iso = lookup_date.isoformat()
        row = df[df["Date"] == iso]
        if row.empty:
            raise ValueError(f"Discount curve DF not found for date: {iso}")
        naca = float(row["NACA"].values[0])
        tau = self._year_fraction(self.val_date, lookup_date)
        return (1.0 + naca) ** (-tau)

    def _flat_r_nacc_from_curve(self) -> float:
        """Flat continuous-compounded short rate r over [0,T] implied by DF(0,T)."""
        if self.discount_curve_df is None:
            return 0.0
        DF_T = self._discount_factor_from_curve(self.mat_date)
        return -math.log(max(DF_T, 1e-16)) / max(self.T, 1e-12)

    def _pv_dividends(self) -> float:
        """PV of discrete dividends at valuation using DF from discount curve."""
        if not self.dividends:
            return 0.0
        pv = 0.0
        for pay_date, amount in self.dividends:
            if self.val_date < pay_date <= self.mat_date:
                DF = self._discount_factor_from_curve(pay_date)
                pv += float(amount) * DF
        return pv

    def _flat_q_nacc_from_dividends(self) -> float:
        """
        Back out a flat dividend yield q (NACC) such that escrowed dividends are reproduced:
        S0_effective = S0 - PV(divs) = S0 * e^{-q T}  ->  q = -ln( (S0 - PVdivs) / S0 ) / T
        """
        pvD = self._pv_dividends()
        if pvD <= 0.0:
            return 0.0
        if pvD >= self.S0:
            # defensive: clamp to very high q to avoid negative inside log
            return 50.0
        return -math.log((self.S0 - pvD) / self.S0) / max(self.T, 1e-12)

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
        return max(0, len(self.monitor_dates))

    def _vanilla_bs_price(self) -> float:
        """
        Standard Black-Scholes (with continuous r and q).
        """
        S, K, T, r, q, sig = self.S0, self.K, self.T, self.r_nacc, self.q_nacc, self.sigma
        if T <= 0 or sig <= 0:
            # intrinsic
            return max(S - K, 0.0) if self.opt_type == "call" else max(K - S, 0.0)
        sqrtT = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * sqrtT)
        d2 = d1 - sig * sqrtT
        Nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
        Nd2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))
        if self.opt_type == "call":
            return S * math.exp(-q * T) * Nd1 - K * math.exp(-r * T) * Nd2
        else:
            Nmd1 = 1.0 - Nd1; Nmd2 = 1.0 - Nd2
            return K * math.exp(-r * T) * Nmd2 - S * math.exp(-q * T) * Nmd1

    # ---------- Hörfelt single-barrier pricing (Theorem 2) ----------
    def _phi(self, x: float) -> float:
        """φ(x) = ln(x/S0) / (sigma sqrt(T))."""
        return math.log(x / self.S0) / (self.sigma * math.sqrt(max(self.T, 1e-12)))

    def _N(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _F_plus(self, a: float, b: float, theta: float) -> float:
        """
        F+(a,b;θ) = N(a-theta) - exp(2 * b * theta) N(a - 2b - theta),   for a ≤ b, b > 0
        """
        return self._N(a - theta) - math.exp(2.0 * b * theta) * self._N(a - 2.0 * b - theta)

    def _F_minus(self, a: float, b: float, theta: float) -> float:
        """F-(a,b;θ) = F+(-a, -b; −θ)."""
        return self._F_plus(-a, -b, -theta)

    def _theta0_theta1(self) -> Tuple[float, float]:
        """
        θ0 = (r - q - sigma^2/2) sqrt(T) / sigma
        θ1 = θ0 + sigma sqrt(T)
        """
        sqrtT = math.sqrt(max(self.T, 1e-12))
        theta0 = (self.r_nacc - self.q_nacc - 0.5 * self.sigma * self.sigma) * sqrtT / self.sigma
        theta1 = theta0 + self.sigma * sqrtT
        return theta0, theta1

    def _single_barrier_out_price(self) -> float:
        """
        Hörfelt Theorem 2:
          - Up-and-out call/put when S0 < H (χ = ±1)  -> use F+ with barrier shifted b -> b + β/√m (in φ-space)
          - Down-and-out call/put when S0 > H        -> use F- with b -> b - β/√m
        Payoff parity gives knock-ins.
        """
        if self.m <= 0:
            # no monitoring -> behave like vanilla (no barrier effect)
            return self._vanilla_bs_price()

        theta0, theta1 = self._theta0_theta1()

        # φ-space levels
        c = self._phi(self.K)

        if self.barrier_type == "up-and-out":
            if self.Hu is None or self.S0 >= self.Hu:
                return 0.0
            d = self._phi(self.Hu)
            b_shift = d + BETA_BGK / math.sqrt(self.m)  # upward shift in φ
            # Option sign χ handled by payoff composition below
            if self.opt_type == "call":
                # vuoc (Eq. 2)
                term1 = self.S0 * math.exp(-self.q_nacc * self.T) * (self._F_plus(d, b_shift, theta1) - self._F_plus(c, b_shift, theta1))
                term2 = self.K  * math.exp(-self.r_nacc * self.T) * (self._F_plus(d, b_shift, theta0) - self._F_plus(c, b_shift, theta0))
                return term1 - term2
            else:
                # up-and-out put via call-put parity on F terms (χ=-1). Using same structure with θ0/θ1 swapped signs
                # A straightforward route: vanilla put - up-and-in put; but Hörfelt gives uo-put by sign change.
                # We obtain put-out by vanilla - (up-and-in put); compute up-and-in by vanilla - up-and-out put of put-call parity pair:
                vanilla_put = self._vanilla_bs_price()
                # up-and-in put = vanilla put - up-and-out put; but we need up-and-out put itself.
                # For stability, price up-and-out CALL via formula and then use parity:
                uo_call = self.S0 * math.exp(-self.q_nacc * self.T) * (self._F_plus(d, b_shift, theta1) - self._F_plus(c, b_shift, theta1)) \
                          - self.K  * math.exp(-self.r_nacc * self.T) * (self._F_plus(d, b_shift, theta0) - self._F_plus(c, b_shift, theta0))
                vanilla_call = self._vanilla_bs_price_forced("call")
                uo_put = vanilla_put - (vanilla_call - uo_call)  # out put = vanilla put - in put; in put = vanilla call - out call (symmetry)
                return uo_put

        elif self.barrier_type == "down-and-out":
            if self.Hd is None or self.S0 <= self.Hd:
                return 0.0
            d = self._phi(self.Hd)
            b_shift = d - BETA_BGK / math.sqrt(self.m)  # downward shift in φ
            if self.opt_type == "put":
                # vdop (Eq. 3)
                term1 = self.K  * math.exp(-self.r_nacc * self.T) * (self._F_minus(d, b_shift, theta0) - self._F_minus(c, b_shift, theta0))
                term2 = self.S0 * math.exp(-self.q_nacc * self.T) * (self._F_minus(d, b_shift, theta1) - self._F_minus(c, b_shift, theta1))
                return term1 - term2
            else:
                # down-and-out call via parity (mirror logic to the up-case)
                vanilla_call = self._vanilla_bs_price_forced("call")
                # down-and-in call = vanilla call - down-and-out call; but we need down-and-out call directly.
                # Use Hörfelt symmetry: compute down-and-out put using formula and then infer down-and-out call via parity with vanilla.
                dop_put = self.K  * math.exp(-self.r_nacc * self.T) * (self._F_minus(d, b_shift, theta0) - self._F_minus(c, b_shift, theta0)) \
                          - self.S0 * math.exp(-self.q_nacc * self.T) * (self._F_minus(d, b_shift, theta1) - self._F_minus(c, b_shift, theta1))
                vanilla_put = self._vanilla_bs_price_forced("put")
                do_call = vanilla_call - (vanilla_put - dop_put)  # out call = vanilla call - in call; in call = vanilla put - out put
                return do_call

        else:
            raise ValueError("single-barrier out price called with non-single barrier type.")

    def _vanilla_bs_price_forced(self, which: Literal["call", "put"]) -> float:
        save = self.opt_type
        self.opt_type = which
        px = self._vanilla_bs_price()
        self.opt_type = save
        return px

    def _G_continuous(self, a1: float, a2: float, b1: float, b2: float, theta: float, series_terms: int = 50) -> float:
        """
        Continuous analogue probability (Hörfelt Eq. (8)):
          G(a1,a2,b1,b2;θ) = N(a2-θ) - N(a1-θ) - G_+(a2;.) + G_+(a1;.) - G_-(a2;.) + G_-(a1;.)
        with
          G_+(a;.) = Σ_{i=1..∞} [ e^{2 α1^{(i)} θ} N(a - 2 α1^{(i)} - θ) - e^{2 α2^{(i)} θ} N(a - 2 α2^{(i)} - θ) ]
          α1^{(i)} = i(b2 - b1) + b1
          α2^{(i)} = i(b2 - b1)
        Rapid exponential convergence; truncate after 'series_terms'.
        """
        def N(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

        def Gplus(a: float) -> float:
            total = 0.0
            width = (b2 - b1)
            for i in range(1, series_terms + 1):
                alpha1 = i * width + b1
                alpha2 = i * width
                total += math.exp(2.0 * alpha1 * theta) * N(a - 2.0 * alpha1 - theta)
                total -= math.exp(2.0 * alpha2 * theta) * N(a - 2.0 * alpha2 - theta)
            return total

        def Gminus(a: float) -> float:
            # G-(a,b1,b2;θ) = G+( -a, -b1, -b2; -θ )
            a_m = -a; b1_m = -b1; b2_m = -b2; th_m = -theta
            width = (b2_m - b1_m)
            total = 0.0
            def N(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
            for i in range(1, series_terms + 1):
                alpha1 = i * width + b1_m
                alpha2 = i * width
                total += math.exp(2.0 * alpha1 * th_m) * N(a_m - 2.0 * alpha1 - th_m)
                total -= math.exp(2.0 * alpha2 * th_m) * N(a_m - 2.0 * alpha2 - th_m)
            return total

        return (N(a2 - theta) - N(a1 - theta)) - Gplus(a2) + Gplus(a1) - Gminus(a2) + Gminus(a1)

    def _double_barrier_out_price(self, series_terms: int = 50) -> float:
        """
        Siegmund’s suggestion for discrete double barriers:
          G^(m)(...) ≈ G(... with b1 -> b1 - β/√m, b2 -> b2 + β/√m)
        Then use change-of-numeraire lemma to form the price.
        """
        if self.m <= 0 or (self.Hd is None and self.Hu is None):
            return self._vanilla_bs_price()

        # Ensure both barriers exist for double-out
        if self.Hd is None or self.Hu is None:
            raise ValueError("Double barrier requires both lower_barrier and upper_barrier.")

        # φ-space
        d1 = self._phi(self.Hd)   # lower (<0)
        d2 = self._phi(self.Hu)   # upper (>0)
        c  = self._phi(self.K)
        theta0, theta1 = self._theta0_theta1()

        # Siegmund correction
        shift = BETA_BGK / math.sqrt(self.m)
        b1_adj = d1 - shift
        b2_adj = d2 + shift

        # G^(m) approx via G with adjusted barriers
        def Gm(a1, a2, th):
            return self._G_continuous(a1, a2, b1_adj, b2_adj, th, series_terms=series_terms)

        # Payoff assembly (knock-OUT call/put)
        if self.opt_type == "call":
            if self.K >= self.Hu:
                return 0.0
            a1 = max(c, d1)
            a2 = d2
            term_S = self.S0 * math.exp(-self.q_nacc * self.T) * Gm(a1, a2, theta1)
            term_K = self.K  * math.exp(-self.r_nacc * self.T) * Gm(a1, a2, theta0)
            return term_S - term_K
        else:
            if self.K <= self.Hd:
                return 0.0
            a1 = d1
            a2 = min(c, d2)
            term_K = self.K  * math.exp(-self.r_nacc * self.T) * Gm(a1, a2, theta0)
            term_S = self.S0 * math.exp(-self.q_nacc * self.T) * Gm(a1, a2, theta1)
            return term_K - term_S

    # ---------- utilities ----------
    def _signed_scale(self, px: float) -> float:
        sign = 1.0 if self.direction == "long" else -1.0
        return sign * self.quantity * self.contract_multiplier * float(px)
