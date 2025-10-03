import math
import datetime as _dt
from typing import List, Optional, Tuple, Literal, Dict, Any
import pandas as pd

# ---------------------------- Utility ----------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

# ---------------------------- Class ----------------------------

OptionType = Literal["call", "put"]
BarrierKind = Literal[
    "none",
    "up-and-out", "down-and-out", "double-out",
    "up-and-in",  "down-and-in",  "double-in",
]

BETA_BGK = 0.5826  # Broadie–Glasserman–Kou continuity-correction constant
EPS = 1e-12


class DiscreteBarrierBGKPricer:
    """
    Discrete barrier option pricer using BGK/Hörfelt approximation in a Black-76 forward layout.
    Includes:
      - include_expiry_monitor (bool)
      - use_mean_sqrt_dt (bool)       -> mean √Δt shift for irregular schedules
      - theta_from_forward (bool)     -> drift from ln(F0/S0)/T vs carry (r - q)
      - rebate_at_hit with hazard PV
      - already_hit + barrier_hit_date exact discounting
    """

    def __init__(
        self,
        *,
        # Core
        spot: float,
        strike: float,
        valuation_date: _dt.date,
        maturity_date: _dt.date,
        option_type: OptionType,
        barrier_type: BarrierKind = "none",
        lower_barrier: Optional[float] = None,
        upper_barrier: Optional[float] = None,
        monitor_dates: Optional[List[_dt.date]] = None,
        # Rebates & status
        rebate_amount: float = 0.0,
        rebate_at_hit: bool = False,
        already_hit: bool = False,
        barrier_hit_date: Optional[_dt.date] = None,
        # Curves & dividends
        discount_curve: Optional[pd.DataFrame] = None,   # for pricing leg discounting
        forward_curve:  Optional[pd.DataFrame] = None,   # for carry leg (can be same as discount)
        dividend_schedule: Optional[List[Tuple[_dt.date, float]]] = None,
        # Model
        volatility: float = 0.2,
        day_count: str = "ACT/365",
        # FA-alignment knobs
        include_expiry_monitor: bool = True,
        use_mean_sqrt_dt: bool = False,
        theta_from_forward: bool = False,  # False => μ = r - q (PDE carry), to match FA
        # Trade/meta
        trade_id: str = "T-0001",
        direction: Literal["long", "short"] = "long",
        quantity: int = 1,
        contract_multiplier: float = 1.0,
    ) -> None:
        if spot <= 0 or strike <= 0 or volatility <= 0:
            raise ValueError("spot, strike, volatility must be positive.")
        if maturity_date <= valuation_date:
            raise ValueError("maturity_date must be after valuation_date.")

        # Inputs
        self.spot_price = float(spot)
        self.strike_price = float(strike)
        self.valuation_date = valuation_date
        self.maturity_date = maturity_date
        self.option_type = option_type
        self.barrier_type = barrier_type
        self.lower_barrier = lower_barrier
        self.upper_barrier = upper_barrier
        self.monitor_dates = sorted(monitor_dates or [])
        self.rebate_amount = float(rebate_amount)
        self.rebate_at_hit = bool(rebate_at_hit)
        self.already_hit = bool(already_hit)
        self.barrier_hit_date = barrier_hit_date

        self.discount_curve_df = discount_curve.copy() if discount_curve is not None else None
        self.forward_curve_df  = forward_curve.copy() if forward_curve is not None else None
        self.dividend_schedule = sorted(dividend_schedule or [], key=lambda x: x[0])

        self.sigma = float(volatility)
        self.day_count = day_count.upper()

        self.include_expiry_monitor = include_expiry_monitor
        self.use_mean_sqrt_dt = use_mean_sqrt_dt
        self.theta_from_forward = theta_from_forward

        self.trade_id = trade_id
        self.direction = direction
        self.quantity = int(quantity)
        self.contract_multiplier = float(contract_multiplier)

        # Times
        self.tenor_years = self._year_fraction(self.valuation_date, self.maturity_date)  # T
        self.discount_years = self.tenor_years  # settle at expiry; adjust if needed

        # Curves -> flat rates
        DF_T_disc = self._df_from_curve(self.discount_curve_df, self.maturity_date) if self.discount_curve_df is not None else 1.0
        self.discount_rate = -math.log(max(DF_T_disc, 1e-16)) / max(self.discount_years, EPS)

        # Carry curve: if not provided, fall back to discount curve
        carry_curve = self.forward_curve_df if self.forward_curve_df is not None else self.discount_curve_df
        DF_T_carry = self._df_from_curve(carry_curve, self.maturity_date) if carry_curve is not None else 1.0
        self.carry_rate_nacc = -math.log(max(DF_T_carry, 1e-16)) / max(self.tenor_years, EPS)

        # Dividend flat q via PV escrow
        self.div_yield_nacc = self._dividend_yield_nacc()

        # Effective S for forward calc (Black-76 presentation)
        self.spot_price_eff = self.spot_price * math.exp(-self.div_yield_nacc * self.tenor_years)
        self.forward_price = self.spot_price_eff * math.exp(self.carry_rate_nacc * self.tenor_years)

        # Monitoring Δt (years) from schedule
        self._dt_years = self._compute_dt_years_from_schedule()
        # Effective m
        self.m = len(self._dt_years) if self._dt_years is not None else self._heuristic_m()

    # -------------------------- Public API --------------------------

    def price(self) -> float:
        """Total price incl. rebate leg (if any)."""
        if self.barrier_type == "none":
            px = self._vanilla_b76()
            return self._signed_scale(px)

        if self.barrier_type in ("up-and-out", "down-and-out"):
            side = "up" if "up" in self.barrier_type else "down"
            px_core = self._single_barrier_out_price(side)
            px_reb = self._rebate_leg()  # handles at-hit vs at-expiry
            return self._signed_scale(px_core + px_reb)

        if self.barrier_type in ("up-and-in", "down-and-in"):
            # same-type parity: IN = Vanilla - OUT
            side = "up" if "up" in self.barrier_type else "down"
            px_out = self._single_barrier_out_price(side)
            px_van = self._vanilla_b76()
            return self._signed_scale(px_van - px_out)

        if self.barrier_type in ("double-out",):
            px_core = self._double_barrier_out_price()
            px_reb = self._rebate_leg()  # if you attach rebates to doubles
            return self._signed_scale(px_core + px_reb)

        if self.barrier_type in ("double-in",):
            px_out = self._double_barrier_out_price()
            px_van = self._vanilla_b76()
            return self._signed_scale(px_van - px_out)

        raise ValueError(f"Unsupported barrier_type: {self.barrier_type}")

    def greeks(self, ds_rel: float = 1e-4, dvol_abs: float = 1e-4) -> Dict[str, float]:
        """Robust finite-difference Greeks under spot convention."""
        saved_dir = self.direction
        self.direction = "long"

        s0 = self.spot_price
        ds = max(1e-8, ds_rel * s0)

        self.spot_price = s0 + ds; self._refresh_for_spot_change()
        up = self.price()
        self.spot_price = s0 - ds; self._refresh_for_spot_change()
        dn = self.price()
        self.spot_price = s0;      self._refresh_for_spot_change()
        base = self.price()

        delta = (up - dn) / (2 * ds)
        gamma = (up - 2 * base + dn) / (ds * ds)

        sig0 = self.sigma
        self.sigma = sig0 + dvol_abs; upv = self.price()
        self.sigma = sig0 - dvol_abs; dnv = self.price()
        self.sigma = sig0
        vega = (upv - dnv) / (2 * dvol_abs)

        self.direction = saved_dir
        scale = (1.0 if self.direction == "long" else -1.0) * self.quantity * self.contract_multiplier
        return {"delta": scale * delta, "gamma": scale * gamma, "vega": scale * vega}

    def report(self) -> str:
        """Model summary + concise barrier-hit summary."""
        lines = []
        lines.append("==== Discrete Barrier (BGK/Hörfelt) — Black-76 layout ====")
        lines.append(f"Trade ID           : {self.trade_id}")
        lines.append(f"Direction          : {self.direction}")
        lines.append(f"Quantity           : {self.quantity}")
        lines.append(f"Contract Multiplier: {self.contract_multiplier:.6g}")
        lines.append(f"Option Type        : {self.option_type}")
        lines.append(f"Barrier Type       : {self.barrier_type}")
        lines.append(f"Lower Barrier (Hd) : {self.lower_barrier if self.lower_barrier is not None else '-'}")
        lines.append(f"Upper Barrier (Hu) : {self.upper_barrier if self.upper_barrier is not None else '-'}")
        lines.append(f"Spot (S0)          : {self.spot_price:.8f}")
        lines.append(f"Strike (K)         : {self.strike_price:.8f}")
        lines.append(f"Valuation Date     : {self.valuation_date.isoformat()}")
        lines.append(f"Maturity Date      : {self.maturity_date.isoformat()}")
        lines.append(f"Day Count          : {self.day_count}")
        lines.append(f"T (years)          : {self.tenor_years:.8f}")
        lines.append(f"Volatility (sigma) : {self.sigma:.8f}")
        lines.append(f"Discount r (NACC)  : {self.discount_rate:.8f}")
        lines.append(f"Carry (r_a) NACC   : {self.carry_rate_nacc:.8f}")
        lines.append(f"Dividend y (q)     : {self.div_yield_nacc:.8f}")
        lines.append(f"F0 (forward)       : {self.forward_price:.8f}")
        lines.append(f"m (monitors)       : {self.m}")
        lines.append(f"include_expiry_mon : {self.include_expiry_monitor}")
        lines.append(f"use_mean_sqrt_dt   : {self.use_mean_sqrt_dt}")
        lines.append(f"theta_from_forward : {self.theta_from_forward}")
        lines.append("----------------------------------------")

        px = self.price()
        greeks = self.greeks()
        lines.append(f"Price              : {px:.10f}")
        lines.append(f"Delta              : {greeks['delta']:.10f}")
        lines.append(f"Gamma              : {greeks['gamma']:.10f}")
        lines.append(f"Vega               : {greeks['vega']:.10f}")

        # Barrier-hit summary
        try:
            mets = self.barrier_hit_metrics()
            lines.append("---- Barrier hit summary ----")
            if mets.get('hazard'):
                lines.append(f"P(hit by last monitor): {mets['P_hit']:.6%}")
                lines.append(f"Expected hit date     : {mets['expected_hit_date'] or '-'}")
                lines.append(f"Mode hit date         : {mets['mode_hit_date'] or '-'}")
                top3 = sorted(mets['hazard'], key=lambda x: x[1], reverse=True)[:3]
                for d, p, DF, contrib in top3:
                    lines.append(f"  {d.isoformat()}: p={p:.4%}, DF={DF:.6f}, rebate PV contrib={contrib:.6f}")
                lines.append(f"Rebate PV at hit (expected): {mets['rebate_pv_at_hit']:.6f}")
            else:
                lines.append("No schedule/single barrier not set: hazard summary not available.")
        except Exception as e:
            lines.append(f"(hazard summary unavailable: {e})")

        return "\n".join(lines)

    def report_hazard_table(self, max_rows: int = 20) -> str:
        """Optional: detailed per-date hazard table."""
        mets = self.barrier_hit_metrics()
        lines = []
        lines.append("=== Barrier hit hazard table ===")
        if not mets.get('hazard'):
            lines.append("No hazard entries (no schedule or not a single barrier).")
            return "\n".join(lines)

        lines.append(f"P(hit by last monitor): {mets['P_hit']:.6%}")
        lines.append(f"Expected hit date     : {mets['expected_hit_date'] or '-'}")
        lines.append(f"Mode hit date         : {mets['mode_hit_date'] or '-'}")
        lines.append(f"Rebate PV at hit (expected): {mets['rebate_pv_at_hit']:.6f}")
        lines.append("")
        lines.append(f"{'Date':<12} {'p_i':>10} {'DF_i':>12} {'PV contrib':>14}")
        lines.append("-" * 52)

        for i, (d, p, DF, contrib) in enumerate(mets['hazard']):
            if i >= max_rows:
                lines.append(f"... ({len(mets['hazard'])-max_rows} more rows)")
                break
            lines.append(f"{d.isoformat():<12} {p:>9.4%} {DF:>12.6f} {contrib:>14.6f}")

        return "\n".join(lines)

    # ------------------- Curves / dividends / time -------------------

    def _normalize_curve(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        cols = {c.lower(): c for c in out.columns}
        return out, cols

    def _df_from_curve(self, curve: Optional[pd.DataFrame], date: _dt.date) -> float:
        """Supports ('date','df') or ('Date','NACA'). If none, returns 1.0."""
        if curve is None or curve.empty:
            return 1.0
        df, cols = self._normalize_curve(curve)
        cl = {k: v for k, v in cols.items()}
        # direct DF table
        if "date" in cl and "df" in cl:
            # pick last row with date <= target, else first
            s_dates = pd.to_datetime(df[cl["date"]]).dt.date
            mask = s_dates <= date
            row = df[mask].tail(1) if mask.any() else df.head(1)
            return float(row[cl["df"]].values[0])
        # NACA table
        if "date" in cl and "naca" in cl:
            s_dates = pd.to_datetime(df[cl["date"]]).dt.date
            mask = s_dates == date
            if not mask.any():
                mask = s_dates < date
            row = df[mask].tail(1) if mask.any() else df.head(1)
            naca = float(row[cl["naca"]].values[0])
            tau = self._year_fraction(self.valuation_date, date)
            return (1.0 + naca) ** (-tau)
        raise ValueError("Curve must have ('date','df') or ('Date','NACA').")

    def _pv_dividends(self) -> float:
        if not self.dividend_schedule:
            return 0.0
        pv = 0.0
        for pay_date, amt in self.dividend_schedule:
            if self.valuation_date < pay_date <= self.maturity_date:
                DF = self._df_from_curve(self.discount_curve_df, pay_date)
                pv += float(amt) * DF
        return pv

    def _dividend_yield_nacc(self) -> float:
        """Flat q such that S0 * e^{-qT} = S0 - PV(divs)."""
        pvD = self._pv_dividends()
        if pvD <= 0.0:
            return 0.0
        if pvD >= self.spot_price:
            return 50.0
        return -math.log((self.spot_price - pvD) / self.spot_price) / max(self.tenor_years, EPS)

    def _year_fraction(self, d0: _dt.date, d1: _dt.date) -> float:
        days = max(0, (d1 - d0).days)
        if self.day_count in ("ACT/365", "ACT/365F"):
            return days / 365.0
        if self.day_count in ("ACT/360",):
            return days / 360.0
        if self.day_count in ("30/360", "30E/360"):
            y0, m0, dd0 = d0.year, d0.month, min(d0.day, 30)
            y1, m1, dd1 = d1.year, d1.month, min(d1.day, 30)
            return ((y1 - y0) * 360 + (m1 - m0) * 30 + (dd1 - dd0)) / 360.0
        return days / 365.0

    # ------------------- Monitoring schedule -------------------

    def _compute_dt_years_from_schedule(self) -> Optional[List[float]]:
        if not self.monitor_dates:
            return None
        if self.include_expiry_monitor:
            mons = [d for d in self.monitor_dates if self.valuation_date < d <= self.maturity_date]
        else:
            mons = [d for d in self.monitor_dates if self.valuation_date < d < self.maturity_date]
        if not mons:
            return None
        mons = sorted(mons)
        prev = self.valuation_date
        dts = []
        for d in mons:
            dt = self._year_fraction(prev, d)
            if dt > 0:
                dts.append(dt)
                prev = d
        return dts if dts else None

    def _heuristic_m(self) -> int:
        """Fallback if no schedule provided. Daily for puts; weekly for calls (FA-aligned heuristic)."""
        days_per_year = 252
        if self.option_type == "put":
            return max(1, int(round(days_per_year * self.tenor_years)))            # daily
        else:
            return max(1, int(round(days_per_year * self.tenor_years) / 5))       # ~weekly

    # ------------------- Vanilla Black-76 -------------------

    def _vanilla_b76(self) -> float:
        T, sig = self.tenor_years, self.sigma
        if T <= 0.0 or sig <= 0.0:
            intr = max(self.spot_price - self.strike_price, 0.0) if self.option_type == "call" else max(self.strike_price - self.spot_price, 0.0)
            return intr
        DF = math.exp(-self.discount_rate * self.discount_years)
        F0 = self.forward_price
        st = sig * math.sqrt(T)
        d1 = (math.log(F0 / self.strike_price) + 0.5 * sig * sig * T) / st
        d2 = d1 - st
        if self.option_type == "call":
            return DF * (F0 * _norm_cdf(d1) - self.strike_price * _norm_cdf(d2))
        else:
            return DF * (self.strike_price * _norm_cdf(-d2) - F0 * _norm_cdf(-d1))

    # ------------------- φ / θ / F+ / F- -------------------

    def _phi_at(self, x: float, T: float) -> float:
        return math.log(max(x, EPS) / self.spot_price_eff) / (self.sigma * math.sqrt(max(T, EPS)))

    def _phi(self, x: float) -> float:
        return self._phi_at(x, self.tenor_years)

    def _thetas_at(self, T: float) -> Tuple[float, float]:
        sqrtT = math.sqrt(max(T, EPS))
        if self.theta_from_forward:
            mu = math.log(self.forward_price / self.spot_price_eff) / max(self.tenor_years, EPS)
        else:
            mu = (self.carry_rate_nacc - self.div_yield_nacc)
        theta0 = (mu - 0.5 * self.sigma * self.sigma) * sqrtT / self.sigma
        theta1 = theta0 + self.sigma * sqrtT
        return theta0, theta1

    def _theta0_theta1(self) -> Tuple[float, float]:
        return self._thetas_at(self.tenor_years)

    def _F_plus(self, a: float, b: float, theta: float) -> float:
        """Up-barrier block with clamping a<=b."""
        if b <= 0.0:
            return 0.0
        a_eff = a if a <= b else b
        return _norm_cdf(a_eff - theta) - math.exp(2.0 * b * theta) * _norm_cdf(a_eff - 2.0 * b - theta)

    def _F_minus(self, a: float, b: float, theta: float) -> float:
        """Down-barrier block with clamping a>=b."""
        if b >= 0.0:
            return 0.0
        a_eff = a if a >= b else b
        # symmetry
        return self._F_plus(-a_eff, -b, -theta)

    # ------------------- BGK φ-shift -------------------

    def _bgk_phi_shift(self, side: Literal["up", "down"], d_phi: float) -> float:
        if self.m <= 0:
            return d_phi
        sign = +1.0 if side == "up" else -1.0
        if self.use_mean_sqrt_dt and self._dt_years:
            mean_sqrt_dt = sum(math.sqrt(dt) for dt in self._dt_years) / len(self._dt_years)
            shift_mag = (BETA_BGK * mean_sqrt_dt) / math.sqrt(max(self.tenor_years, EPS))
        else:
            shift_mag = BETA_BGK / math.sqrt(self.m)
        return d_phi + sign * shift_mag

    def _bgk_phi_shift_at(self, side: Literal["up", "down"], d_phi: float, T: float, m_t: int) -> float:
        if m_t <= 0:
            return d_phi
        sign = +1.0 if side == "up" else -1.0
        if self.use_mean_sqrt_dt and self._dt_years and m_t > 0:
            partial = self._dt_years[:m_t]
            mean_sqrt_dt = sum(math.sqrt(dt) for dt in partial) / len(partial)
            shift_mag = (BETA_BGK * mean_sqrt_dt) / math.sqrt(max(T, EPS))
        else:
            shift_mag = BETA_BGK / math.sqrt(m_t)
        return d_phi + sign * shift_mag

    # ------------------- Single-barrier OUT -------------------

    def _single_barrier_out_price(self, side: Literal["up", "down"]) -> float:
        if self.m <= 0:
            return self._vanilla_b76()

        # immediate KO checks
        if side == "up":
            if self.upper_barrier is None or self.spot_price >= self.upper_barrier:
                return 0.0
        else:
            if self.lower_barrier is None or self.spot_price <= self.lower_barrier:
                return 0.0

        theta0, theta1 = self._theta0_theta1()
        c = self._phi(self.strike_price)
        DF = math.exp(-self.discount_rate * self.discount_years)
        F0 = self.forward_price

        if side == "up":
            d = self._phi(self.upper_barrier)
            b = self._bgk_phi_shift("up", d)
            if self.option_type == "call" and self.upper_barrier is not None and self.strike_price >= self.upper_barrier:
                return 0.0
            if self.option_type == "call":
                return DF * ( F0 * ( self._F_plus(d,b,theta1) - self._F_plus(c,b,theta1) )
                              - self.strike_price * ( self._F_plus(d,b,theta0) - self._F_plus(c,b,theta0) ) )
            else:  # up-and-out put (direct)
                return DF * ( self.strike_price * self._F_plus(c,b,theta0) - F0 * self._F_plus(c,b,theta1) )

        else:  # down
            d = self._phi(self.lower_barrier)
            b = self._bgk_phi_shift("down", d)
            if self.option_type == "put" and self.lower_barrier is not None and self.strike_price <= self.lower_barrier:
                return 0.0
            if self.option_type == "put":
                return DF * ( self.strike_price * ( self._F_minus(d,b,theta0) - self._F_minus(c,b,theta0) )
                              - F0 * ( self._F_minus(d,b,theta1) - self._F_minus(c,b,theta1) ) )
            else:  # down-and-out call (direct)
                return DF * ( F0 * self._F_minus(c,b,theta1) - self.strike_price * self._F_minus(c,b,theta0) )

    # ------------------- Double-barrier OUT (series) -------------------

    def _G_continuous(self, a1: float, a2: float, b1: float, b2: float, theta: float, series_terms: int = 50) -> float:
        # simple symmetric image-series (good convergence for moderate series_terms)
        def N(x): return _norm_cdf(x)
        total = N(a2 - theta) - N(a1 - theta)
        L = (b2 - b1)
        for k in range(1, series_terms + 1):
            shift = 2.0 * k * L
            total += (N(a2 - theta - shift) - N(a1 - theta - shift))
            total -= (N(a2 - theta + shift) - N(a1 - theta + shift))
        return total

    def _double_barrier_out_price(self, series_terms: int = 50) -> float:
        if self.m <= 0 or (self.lower_barrier is None and self.upper_barrier is None):
            return self._vanilla_b76()
        if self.lower_barrier is None or self.upper_barrier is None:
            raise ValueError("Double barrier requires both lower_barrier and upper_barrier.")

        d1 = self._phi(self.lower_barrier)
        d2 = self._phi(self.upper_barrier)
        c  = self._phi(self.strike_price)
        theta0, theta1 = self._theta0_theta1()

        # Siegmund correction for discrete: widen by ±shift
        if self.use_mean_sqrt_dt and self._dt_years:
            mean_sqrt_dt = sum(math.sqrt(dt) for dt in self._dt_years) / len(self._dt_years)
            shift = (BETA_BGK * mean_sqrt_dt) / math.sqrt(max(self.tenor_years, EPS))
        else:
            shift = BETA_BGK / math.sqrt(self.m)

        b1_adj = d1 - shift
        b2_adj = d2 + shift

        def Gm(a1, a2, th):
            return self._G_continuous(a1, a2, b1_adj, b2_adj, th, series_terms=series_terms)

        DF = math.exp(-self.discount_rate * self.discount_years)
        F0 = self.forward_price

        if self.option_type == "call":
            if self.strike_price >= self.upper_barrier:
                return 0.0
            a1 = max(c, d1); a2 = d2
            return DF * ( F0 * Gm(a1,a2,theta1) - self.strike_price * Gm(a1,a2,theta0) )
        else:
            if self.strike_price <= self.lower_barrier:
                return 0.0
            a1 = d1; a2 = min(c, d2)
            return DF * ( self.strike_price * Gm(a1,a2,theta0) - F0 * Gm(a1,a2,theta1) )

    # ------------------- Barrier survival / hazards -------------------

    def _survival_prob_to(self, side: Literal["up","down"], T: float, m_t: int) -> float:
        """S(T) under BGK for sub-horizon T with first m_t monitors."""
        theta0, _ = self._thetas_at(T)
        if side == "up":
            d = self._phi_at(self.upper_barrier, T)
            b = self._bgk_phi_shift_at("up", d, T, m_t)
            return self._F_plus(b, b, theta0)
        else:
            d = self._phi_at(self.lower_barrier, T)
            b = self._bgk_phi_shift_at("down", d, T, m_t)
            return self._F_minus(b, b, theta0)

    def barrier_hit_metrics(self) -> Dict[str, Any]:
        """
        Builds per-date hazard curve and PV of rebate-at-hit:
          hazards: list of (date, p_i, DF_i, PV_contrib_i)
          P_hit  = sum p_i
          survival_to_T = S(T_last)
        """
        # Only single-barrier types are meaningful here
        if self.barrier_type not in {"up-and-out","down-and-out","up-and-in","down-and-in"}:
            return {'P_hit': 0.0, 'survival_to_T': 1.0, 'hazard': [], 'expected_hit_date': None,
                    'mode_hit_date': None, 'rebate_pv_at_hit': 0.0}

        if not self._dt_years:
            return {'P_hit': 0.0, 'survival_to_T': 1.0, 'hazard': [], 'expected_hit_date': None,
                    'mode_hit_date': None, 'rebate_pv_at_hit': 0.0}

        if "up" in self.barrier_type:
            side = "up"
        else:
            side = "down"

        # build monitor list per flag
        if self.include_expiry_monitor:
            mons = [d for d in self.monitor_dates if self.valuation_date < d <= self.maturity_date]
        else:
            mons = [d for d in self.monitor_dates if self.valuation_date < d < self.maturity_date]
        if not mons:
            return {'P_hit': 0.0, 'survival_to_T': 1.0, 'hazard': [], 'expected_hit_date': None,
                    'mode_hit_date': None, 'rebate_pv_at_hit': 0.0}

        # cumulative horizons
        cumulative_T = []
        acc = 0.0
        for dt in self._dt_years:
            acc += dt
            cumulative_T.append(acc)

        hazards = []
        S_prev = 1.0
        total_hit_prob = 0.0
        pv_rebate = 0.0

        for k, (T_k, d_k) in enumerate(zip(cumulative_T, mons), start=1):
            S_k = self._survival_prob_to(side, T_k, k)
            p_k = max(0.0, S_prev - S_k)  # hazard mass in (t_{k-1}, t_k]
            # discount factor at date
            DF_k = self._df_from_curve(self.discount_curve_df, d_k)
            contrib = self.rebate_amount * DF_k * p_k
            hazards.append((d_k, p_k, DF_k, contrib))
            pv_rebate += contrib
            total_hit_prob += p_k
            S_prev = S_k

        # expected/mode dates (conditional on hit)
        expected_date = None
        mode_date = None
        if total_hit_prob > 0.0:
            weights = [h[1] / total_hit_prob for h in hazards]
            ords = [h[0].toordinal() for h in hazards]
            exp_ord = sum(w * o for w, o in zip(weights, ords))
            expected_date = _dt.date.fromordinal(int(round(exp_ord)))
            mode_date = max(hazards, key=lambda x: x[1])[0]

        return {
            'P_hit': float(total_hit_prob),
            'survival_to_T': float(S_prev),
            'hazard': hazards,
            'expected_hit_date': expected_date,
            'mode_hit_date': mode_date,
            'rebate_pv_at_hit': float(pv_rebate),
        }

    # ------------------- Rebate leg -------------------

    def _rebate_leg(self) -> float:
        """PV of rebate leg for OUT styles; 0 otherwise."""
        if self.rebate_amount <= 0.0:
            return 0.0
        if self.barrier_type not in {"up-and-out", "down-and-out", "double-out"}:
            return 0.0

        if self.rebate_at_hit:
            if self.already_hit:
                # exact discounting to provided hit date (or value date if absent)
                hit_date = self.barrier_hit_date or self.valuation_date
                DF = self._df_from_curve(self.discount_curve_df, hit_date)
                return self.rebate_amount * DF
            else:
                mets = self.barrier_hit_metrics()
                return mets['rebate_pv_at_hit']
        else:
            DF_T = math.exp(-self.discount_rate * self.discount_years)
            return self.rebate_amount * DF_T

    # ------------------- Helpers -------------------

    def _refresh_for_spot_change(self) -> None:
        """Recompute derived quantities after S bump (for Greeks)."""
        self.spot_price_eff = self.spot_price * math.exp(-self.div_yield_nacc * self.tenor_years)
        self.forward_price = self.spot_price_eff * math.exp(self.carry_rate_nacc * self.tenor_years)

    def _signed_scale(self, px: float) -> float:
        sgn = 1.0 if self.direction == "long" else -1.0
        return sgn * self.quantity * self.contract_multiplier * float(px)
