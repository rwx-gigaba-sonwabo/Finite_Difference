import math
import datetime as _dt
from typing import List, Optional, Tuple, Literal, Dict, Any
import pandas as pd
import numpy as np


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


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
        # Pricing method: "bgk" | "mc" | "auto"
        # "auto" → BGK when monitoring freq >= bgk_min_freq dates/year, else Monte Carlo
        pricing_method: Literal["bgk", "mc", "auto"] = "auto",
        bgk_min_freq: float = 20.0,       # monitoring dates/year threshold; >= → BGK, < → MC
        mc_n_paths: int = 100_000,        # MC simulation paths
        mc_seed: Optional[int] = 42,      # RNG seed (None = non-reproducible)
        mc_use_antithetic: bool = True,   # antithetic variates for variance reduction
        # Settlement lags (business days) — mirrors fd_american_equity pattern
        underlying_spot_days: int = 0,     # spot lag for carry/forward period
        option_days: int = 0,              # settlement start lag for discount period
        option_settlement_days: int = 0,   # settlement end lag at option maturity
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
        self.forward_curve_df = forward_curve.copy() if forward_curve is not None else None
        self.dividend_schedule = sorted(dividend_schedule or [], key=lambda x: x[0])

        self.sigma = float(volatility)
        self.day_count = day_count.upper()
        self._year_denominator = self._infer_denominator(self.day_count)

        # Settlement lag parameters (mirrors fd_american_equity)
        self.underlying_spot_days  = int(underlying_spot_days)
        self.option_days           = int(option_days)
        self.option_settlement_days = int(option_settlement_days)

        self.include_expiry_monitor = include_expiry_monitor
        self.use_mean_sqrt_dt = use_mean_sqrt_dt
        self.theta_from_forward = theta_from_forward

        self.pricing_method = pricing_method
        self.bgk_min_freq = float(bgk_min_freq)
        self.mc_n_paths = int(mc_n_paths)
        self.mc_seed = mc_seed
        self.mc_use_antithetic = bool(mc_use_antithetic)
        self._last_mc_std_error: float = 0.0   # updated after each MC call for diagnostics

        self.trade_id = trade_id
        self.direction = direction
        self.quantity = int(quantity)
        self.contract_multiplier = float(contract_multiplier)

        # Settlement dates (business-day adjusted, mirrors fd_american_equity)
        _need_cal = (self.underlying_spot_days > 0
                     or self.option_days > 0
                     or self.option_settlement_days > 0)
        if _need_cal:
            try:
                from workalendar.africa import SouthAfrica as _Cal
                _cal = _Cal()
                self.carry_start_date    = _cal.add_working_days(self.valuation_date, self.underlying_spot_days)
                self.carry_end_date      = _cal.add_working_days(self.maturity_date,  self.underlying_spot_days)
                self.discount_start_date = _cal.add_working_days(self.valuation_date, self.option_days)
                self.discount_end_date   = _cal.add_working_days(self.maturity_date,  self.option_settlement_days)
            except ImportError as _exc:
                raise RuntimeError(
                    "workalendar is required for non-zero settlement lags. "
                    "Install with: pip install workalendar"
                ) from _exc
        else:
            # No lags: carry and discount windows equal the option life
            self.carry_start_date    = self.valuation_date
            self.carry_end_date      = self.maturity_date
            self.discount_start_date = self.valuation_date
            self.discount_end_date   = self.maturity_date

        # Three separate time measures (mirrors fd_american_equity exactly)
        #   time_to_expiry  → σ√T for vol; simulation horizon; monitor schedule
        #   time_to_carry   → forward price / cost-of-carry calculation
        #   time_to_discount → discount-factor calculation
        self.time_to_expiry   = self._year_fraction(self.valuation_date,    self.maturity_date)
        self.time_to_carry    = self._year_fraction(self.carry_start_date,  self.carry_end_date)
        self.time_to_discount = self._year_fraction(self.discount_start_date, self.discount_end_date)

        # Backward-compat aliases so all existing formula code is unaffected
        self.tenor_years    = self.time_to_expiry
        self.discount_years = self.time_to_discount

        # Forward NACC rates over each window (mirrors fd_american_equity's
        # get_forward_nacc_rate pattern: -ln(DF_far/DF_near) / tau)
        self.discount_rate_nacc = (
            self.get_forward_nacc_rate(self.discount_start_date, self.discount_end_date)
            if self.discount_curve_df is not None else 0.0
        )
        self.discount_rate = self.discount_rate_nacc   # backward-compat alias

        _carry_curve_df = self.forward_curve_df if self.forward_curve_df is not None else self.discount_curve_df
        self.carry_rate_nacc = (
            self.get_forward_nacc_rate(self.carry_start_date, self.carry_end_date,
                                       curve_df=_carry_curve_df)
            if _carry_curve_df is not None else self.discount_rate_nacc
        )

        # Dividend flat q via PV escrow (unchanged logic)
        self.div_yield_nacc = self._dividend_yield_nacc()

        # Effective S and forward price — use time_to_carry for the carry window
        self.spot_price_eff = self.spot_price * math.exp(-self.div_yield_nacc * self.time_to_carry)
        self.forward_price  = self.spot_price_eff * math.exp(self.carry_rate_nacc * self.time_to_carry)

        # Monitoring Δt (years) from schedule
        self._dt_years = self._compute_dt_years_from_schedule()
        # Effective m
        self.m = len(self._dt_years) if self._dt_years is not None else self._heuristic_m()

    def price(self) -> float:
        """Total price incl. rebate leg (if any)."""
        if self.barrier_type == "none":
            px = self._vanilla_b76()
            return self._signed_scale(px)

        # ---- Route to Monte Carlo or BGK ----
        if self._select_method() == "mc":
            return self._signed_scale(self._price_via_mc())

        # ---- BGK path (original analytic logic) ----
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
        lines.append(f"Spot lag (carry)   : {self.underlying_spot_days}bd  "
                     f"({self.carry_start_date.isoformat()} -> {self.carry_end_date.isoformat()})")
        lines.append(f"Settlement (disc)  : {self.option_days}bd start / "
                     f"{self.option_settlement_days}bd end  "
                     f"({self.discount_start_date.isoformat()} -> {self.discount_end_date.isoformat()})")
        lines.append(f"T expiry (years)   : {self.time_to_expiry:.8f}")
        lines.append(f"T carry (years)    : {self.time_to_carry:.8f}")
        lines.append(f"T discount (years) : {self.time_to_discount:.8f}")
        lines.append(f"Volatility (sigma) : {self.sigma:.8f}")
        lines.append(f"Discount r (NACC)  : {self.discount_rate_nacc:.8f}  (fwd over discount window)")
        lines.append(f"Carry r (NACC)     : {self.carry_rate_nacc:.8f}  (fwd over carry window)")
        lines.append(f"Dividend y (q)     : {self.div_yield_nacc:.8f}")
        lines.append(f"F0 (forward)       : {self.forward_price:.8f}")
        lines.append(f"m (monitors)       : {self.m}")
        lines.append(f"include_expiry_mon : {self.include_expiry_monitor}")
        lines.append(f"use_mean_sqrt_dt   : {self.use_mean_sqrt_dt}")
        lines.append(f"theta_from_forward : {self.theta_from_forward}")
        # Pricing method
        selected = self._select_method()
        lines.append(f"pricing_method     : {self.pricing_method}  →  selected: {selected.upper()}")
        lines.append(f"bgk_min_freq       : {self.bgk_min_freq:.1f} dates/yr  "
                     f"(mon freq = {self.m / max(self.tenor_years, EPS):.1f}/yr)")
        if selected == "mc":
            lines.append(f"mc_n_paths         : {self.mc_n_paths:,}")
            lines.append(f"mc_seed            : {self.mc_seed}")
            lines.append(f"mc_use_antithetic  : {self.mc_use_antithetic}")
        lines.append("----------------------------------------")

        px = self.price()
        greeks = self.greeks()
        lines.append(f"Price              : {px:.10f}")
        if selected == "mc":
            lines.append(f"MC std error       : {self._last_mc_std_error:.2e}")
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

    def get_discount_factor(self, lookup_date: _dt.date,
                             curve_df: Optional[pd.DataFrame] = None) -> float:
        """
        Discount factor from valuation_date to lookup_date.

        Mirrors fd_american_equity's get_discount_factor() but delegates to
        _df_from_curve() so that both ('Date','NACA') and ('date','df') curve
        formats are supported with graceful fallback (nearest prior date).

        Parameters
        ----------
        lookup_date : date
        curve_df : DataFrame, optional
            Override the instance discount_curve_df (e.g. to use forward curve).
        """
        crv = curve_df if curve_df is not None else self.discount_curve_df
        return self._df_from_curve(crv, lookup_date)

    def get_forward_nacc_rate(self, start_date: _dt.date, end_date: _dt.date,
                               curve_df: Optional[pd.DataFrame] = None) -> float:
        """
        Continuously-compounded forward rate between start_date and end_date.

        Mirrors fd_american_equity's get_forward_nacc_rate():
            r_fwd = -ln(DF(end) / DF(start)) / tau(start, end)

        When start_date == valuation_date, DF(start) = 1 and this collapses to
        the standard terminal spot rate. With settlement lags start_date differs
        from valuation_date, giving the true forward rate over the settlement
        window — the same rate that enters the PDE / Black-76 formula.

        Parameters
        ----------
        start_date, end_date : date
        curve_df : DataFrame, optional
            Override the instance discount_curve_df (e.g. to use forward curve).
        """
        df_far  = self.get_discount_factor(end_date,   curve_df)
        df_near = self.get_discount_factor(start_date, curve_df)
        tau = self._year_fraction(start_date, end_date)
        if tau <= EPS:
            return 0.0
        return -math.log(max(df_far / max(df_near, 1e-16), 1e-16)) / tau

    def _pv_dividends(self) -> float:
        if not self.dividend_schedule:
            return 0.0
        pv = 0.0
        for pay_date, amt in self.dividend_schedule:
            if self.valuation_date < pay_date <= self.maturity_date:
                DF = self.get_discount_factor(pay_date)   # uses discount_curve_df
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

    def _infer_denominator(self, day_count: str) -> int:
        """Year denominator for ACT/* conventions (mirrors fd_american_equity)."""
        if day_count in ("ACT/365", "ACT/365F"):
            return 365
        if day_count == "ACT/360":
            return 360
        if day_count == "ACT/364":
            return 364
        if day_count in ("30/360", "30E/360", "BOND", "US30/360"):
            return 360
        return 365   # safe default

    def _year_fraction(self, d0: _dt.date, d1: _dt.date) -> float:
        """Year fraction between two dates under self.day_count (mirrors fd_american_equity)."""
        if d1 <= d0:
            return 0.0
        if self.day_count in ("ACT/365", "ACT/365F", "ACT/360", "ACT/364"):
            return (d1 - d0).days / float(self._year_denominator)
        if self.day_count in ("30/360", "30E/360", "BOND", "US30/360"):
            y0, m0, dd0 = d0.year, d0.month, d0.day
            y1, m1, dd1 = d1.year, d1.month, d1.day
            dd0 = min(dd0, 30)
            if dd0 == 30:
                dd1 = min(dd1, 30)
            return ((y1 - y0) * 360 + (m1 - m0) * 30 + (dd1 - dd0)) / 360.0
        return (d1 - d0).days / 365.0

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
            # Derive drift from the actual forward over the carry window
            # (F = S_eff * exp(carry * T_carry) => ln(F/S_eff)/T_carry = carry_rate_nacc)
            mu = math.log(self.forward_price / self.spot_price_eff) / max(self.time_to_carry, EPS)
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

    # ------------------- Method selection (BGK vs Monte Carlo) -------------------

    def _select_method(self) -> str:
        """
        Return "bgk" or "mc".

        Mirrors the RiskFlow intuition:
          - Frequent monitoring (dense dates, daily-ish) → BGK correction to continuous
            formula is accurate (Broadie–Glasserman–Kou 1997 convergence).
          - Infrequent monitoring (few dates per year, monthly/quarterly) → the BGK
            shift breaks down; Monte Carlo handles discrete hitting exactly.

        Threshold: bgk_min_freq monitoring dates per year (default 20 ≈ fortnightly).
        """
        if self.pricing_method in ("bgk", "mc"):
            return self.pricing_method
        # "auto"
        if self.m <= 0:
            return "bgk"
        freq = self.m / max(self.tenor_years, EPS)
        return "bgk" if freq >= self.bgk_min_freq else "mc"

    # ------------------- Monte Carlo helpers -------------------

    def _mc_monitoring_times(self) -> List[float]:
        """Cumulative monitoring times (years from valuation date)."""
        if self._dt_years:
            acc, times = 0.0, []
            for dt in self._dt_years:
                acc += dt
                times.append(round(acc, 12))
            return times
        # Fallback: evenly-spaced m points
        T, m = self.tenor_years, max(1, self.m)
        return [round(T * k / m, 12) for k in range(1, m + 1)]

    def _mc_out_price(self, effective_barrier_type: Optional[str] = None) -> float:
        """
        Vectorised Monte Carlo price for a discretely monitored OUT-type option.

        Parameters:
        
        - effective_barrier_type : str, optional
            Override the instance barrier_type.  Used when in-type parity requires
            pricing the corresponding out-type (e.g. "up-and-out" for "up-and-in").

        Returns:
        - The raw discounted present value (no direction/quantity scaling).
            Side-effect: updates self._last_mc_std_error.
        """
        btype = effective_barrier_type or self.barrier_type
        T = self.tenor_years
        DF_T = math.exp(-self.discount_rate * self.discount_years)

        # GBM parameters under risk-neutral measure
        mu  = self.carry_rate_nacc - self.div_yield_nacc   # spot drift
        sig = self.sigma
        S0  = self.spot_price
        K   = self.strike_price
        Hu  = self.upper_barrier
        Hd  = self.lower_barrier

        # ------ Build time grid ------
        mon_times = self._mc_monitoring_times()   # cumulative monitoring times

        # Merge monitoring times and maturity, deduplicate, keep sorted
        raw = [0.0] + mon_times
        if not mon_times or abs(mon_times[-1] - T) > 1e-10:
            raw.append(T)
        time_points = sorted(set(round(t, 10) for t in raw))

        mon_set = {round(t, 10) for t in mon_times}
        is_mon  = [round(tp, 10) in mon_set for tp in time_points]  # includes t=0 (False)

        dts = np.diff(time_points)
        n_steps = len(dts)
        sqrt_dts = np.sqrt(np.maximum(dts, 0.0))

        # ------ Simulate ------
        rng = np.random.default_rng(self.mc_seed)
        n_half = max(1, self.mc_n_paths // 2) if self.mc_use_antithetic else self.mc_n_paths
        Z  = rng.standard_normal((n_half, n_steps))
        if self.mc_use_antithetic:
            Z = np.concatenate([Z, -Z], axis=0)  # antithetic pairs
        n_sim = Z.shape[0]

        # Log-spot increments and cumulative path
        log_drift = (mu - 0.5 * sig * sig) * dts         # (n_steps,)
        log_vol   = sig * sqrt_dts                         # (n_steps,)
        log_incs  = log_drift[None, :] + log_vol[None, :] * Z  # (n_sim, n_steps)
        log_S = np.log(S0) + np.concatenate(
            [np.zeros((n_sim, 1)), np.cumsum(log_incs, axis=1)], axis=1
        )  # (n_sim, n_time_points)
        S_paths = np.exp(log_S)

        # ------ Barrier knockout tracking ------
        alive = np.ones(n_sim, dtype=bool)
        rebate_pv = np.zeros(n_sim)   # PV of rebate-at-hit (accrued per path)

        for col, (tp, mon_flag) in enumerate(zip(time_points, is_mon)):
            if col == 0 or not mon_flag:
                continue   # skip t=0 and non-monitoring time points

            S_k = S_paths[:, col]

            newly_ko = np.zeros(n_sim, dtype=bool)
            if btype in ("up-and-out", "double-out") and Hu is not None:
                newly_ko |= (S_k >= Hu)
            if btype in ("down-and-out", "double-out") and Hd is not None:
                newly_ko |= (S_k <= Hd)
            newly_ko &= alive   # only freshly knocked

            alive[newly_ko] = False

            # Rebate paid at first touch
            if self.rebate_at_hit and self.rebate_amount > 0.0 and newly_ko.any():
                DF_k = math.exp(-self.discount_rate * tp)
                rebate_pv[newly_ko] = self.rebate_amount * DF_k   # present-value rebate

        # ------ Terminal payoff ------
        S_mat = S_paths[:, -1]
        if self.option_type == "call":
            intrinsic = np.maximum(S_mat - K, 0.0)
        else:
            intrinsic = np.maximum(K - S_mat, 0.0)

        option_payoff = np.where(alive, intrinsic, 0.0)

        # ------ Combine option + rebate ------
        if self.rebate_amount > 0.0 and self.rebate_at_hit:
            # rebate_pv already in present-value terms; option payoff needs DF_T
            mc_price = DF_T * float(np.mean(option_payoff)) + float(np.mean(rebate_pv))
            # std error: approximate as option part only (rebate part is deterministic per path)
            se_payoff = np.std(option_payoff, ddof=1) * DF_T / math.sqrt(n_sim)
        elif self.rebate_amount > 0.0:
            # Rebate paid at expiry to knocked-out paths
            rebate_payoff = np.where(~alive, self.rebate_amount, 0.0)
            total_payoff  = option_payoff + rebate_payoff
            mc_price = DF_T * float(np.mean(total_payoff))
            se_payoff = np.std(total_payoff, ddof=1) * DF_T / math.sqrt(n_sim)
        else:
            mc_price  = DF_T * float(np.mean(option_payoff))
            se_payoff = np.std(option_payoff, ddof=1) * DF_T / math.sqrt(n_sim)

        self._last_mc_std_error = float(se_payoff)
        return mc_price

    def _price_via_mc(self) -> float:
        """
        Route all barrier types to Monte Carlo.
        Returns raw price (no direction/quantity scaling).

        - OUT types    → _mc_out_price() directly.
        - IN types     → vanilla Black-76 minus _mc_out_price() (put-call parity).
        - Vanilla      → _vanilla_b76() (exact, no MC needed).
        - already_hit  → 0 or discounted rebate (as for the BGK path).
        """
        if self.barrier_type == "none":
            return self._vanilla_b76()

        # Immediate knockout if spot is already beyond barrier
        if self.barrier_type in ("up-and-out", "double-out"):
            if self.upper_barrier is not None and self.spot_price >= self.upper_barrier:
                return 0.0
        if self.barrier_type in ("down-and-out", "double-out"):
            if self.lower_barrier is not None and self.spot_price <= self.lower_barrier:
                return 0.0

        # Already hit (e.g. trade carried over a knocked-out barrier date)
        if self.already_hit:
            hit_date = self.barrier_hit_date or self.valuation_date
            DF = self._df_from_curve(self.discount_curve_df, hit_date)
            return self.rebate_amount * DF if self.rebate_amount > 0.0 else 0.0

        if self.barrier_type in ("up-and-out", "down-and-out", "double-out"):
            return self._mc_out_price()

        if self.barrier_type in ("up-and-in", "down-and-in"):
            # Parity: IN = Vanilla − OUT (same-direction OUT type)
            out_type = "up-and-out" if "up" in self.barrier_type else "down-and-out"
            vanilla  = self._vanilla_b76()
            out_px   = self._mc_out_price(effective_barrier_type=out_type)
            return vanilla - out_px

        if self.barrier_type == "double-in":
            vanilla = self._vanilla_b76()
            out_px  = self._mc_out_price(effective_barrier_type="double-out")
            return vanilla - out_px

        raise ValueError(f"Unsupported barrier_type for MC: {self.barrier_type}")

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
            DF_k = self.get_discount_factor(d_k)
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
        self.spot_price_eff = self.spot_price * math.exp(-self.div_yield_nacc * self.time_to_carry)
        self.forward_price  = self.spot_price_eff * math.exp(self.carry_rate_nacc * self.time_to_carry)

    def _signed_scale(self, px: float) -> float:
        sgn = 1.0 if self.direction == "long" else -1.0
        return sgn * self.quantity * self.contract_multiplier * float(px)
