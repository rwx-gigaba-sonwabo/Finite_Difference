import math
import datetime as _dt
from typing import List, Optional, Tuple, Literal, Dict

import pandas as pd

"""
Discrete Barrier Option pricer (BGK/Hörfelt) with flat NACC rate and dividend PV handling.

Update (paper-faithful BGK + Black-76 presentation):
- Monitoring m includes maturity; BGK shift uses m monitors over [0,T].
- F⁺/F⁻ blocks use correct domains with clamping of the terminal cut to the barrier.
- Single-barrier OUT formulas implemented directly for all 4 cases (UO-call, UO-put, DO-put, DO-call).
- Knock-IN built via same-type parity: IN = Vanilla - OUT (same option type).

Reference intuition:
- Discrete monitoring ≈ continuous monitoring with φ-space barrier shift b = d ± β/√m
  (β ≈ 0.5826; '+' for up, '−' for down). Black-76 is a presentation: prefactors S0 e^{-qT} and K e^{-rT};
  θ0, θ1 and the BGK shift depend on (r − q, σ, T) only.
"""

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
    Discrete barrier option pricer using BGK/Hörfelt approximation (Black-76 presentation).
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

        # Monitoring dates: ensure maturity included for BGK scaling
        md = sorted(monitor_dates or [])
        if md and md[-1] < self.mat_date:
            md.append(self.mat_date)
        self.monitor_dates = md

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
        self.r_nacc = self._flat_r_nacc_from_curve()      # r (NACC)
        self.q_nacc = self._flat_q_nacc_from_dividends()  # q (NACC)

        # effective number of monitoring times m (count in (val, T], include T)
        self.m = self._effective_monitor_count()

    # ---------------- public API ----------------
    def price(self) -> float:
        """
        Price of the discrete barrier option (BGK/Hörfelt).
        - Single-barrier: continuous Hörfelt blocks with BGK φ-shift.
        - Double-barrier: Siegmund-corrected continuous series.
        - Knock-in via same-type parity: IN = Vanilla - OUT.
        """
        if self.barrier_type == "none":
            return self._signed_scale(self._vanilla_bs_price())

        if self.barrier_type in ("up-and-out", "down-and-out"):
            px = self._single_barrier_out_price()

        elif self.barrier_type in ("up-and-in", "down-and-in"):
            vanilla = self._vanilla_bs_price()
            out_equiv = self._single_barrier_out_price()  # OUT of same type
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

        # Delta & Gamma (spot bumps; log-space size via relative)
        s0 = self.S0
        ds = max(1e-8, ds_rel * s0)
        self.S0 = s0 + ds; up = self.price()
        self.S0 = s0 - ds; dn = self.price()
        self.S0 = s0
        base = self.price()

        delta = (up - dn) / (2 * ds)
        gamma = (up - 2 * base + dn) / (ds * ds)

        # Vega (vol bump)
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
        lines.append("==== Discrete Barrier (BGK/Hörfelt) — Black-76 presentation ====")
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

    # ---------------- curves / time utils ----------------
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
            return 50.0  # defensive clamp
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
        Effective m for BGK shift. Assume dates ~equally spaced; count those strictly after val and ≤ T.
        """
        if not self.monitor_dates:
            return 0
        return sum(1 for d in self.monitor_dates if self.val_date < d <= self.mat_date)

    # ---------------- vanilla (Black-76 presentation) ----------------
    def _vanilla_bs_price(self) -> float:
        """
        Vanilla Black–Scholes/Black-76 (with continuous r, q).
        Price = S e^{-qT} N(d1) - K e^{-rT} N(d2)  for call; put analogously.
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
            Nmd1 = 1.0 - Nd1
            Nmd2 = 1.0 - Nd2
            return K * math.exp(-r * T) * Nmd2 - S * math.exp(-q * T) * Nmd1

    # ---------------- Hörfelt single-barrier blocks ----------------
    def _phi(self, x: float) -> float:
        """φ(x) = ln(x/S0) / (σ√T)."""
        return math.log(max(x, EPS) / self.S0) / (self.sigma * math.sqrt(max(self.T, EPS)))

    def _N(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _theta0_theta1(self) -> Tuple[float, float]:
        """
        θ0 = (r - q - σ²/2) √T / σ ;  θ1 = θ0 + σ √T
        """
        sqrtT = math.sqrt(max(self.T, EPS))
        theta0 = (self.r_nacc - self.q_nacc - 0.5 * self.sigma * self.sigma) * sqrtT / self.sigma
        theta1 = theta0 + self.sigma * sqrtT
        return theta0, theta1

    # ---- Correct F⁺ / F⁻ with domain & clamping ----
    def _F_plus(self, a: float, b: float, theta: float) -> float:
        """
        Up-barrier block: F⁺(a,b;θ) = N(a-θ) - e^{2 b θ} N(a - 2 b - θ)
        Domain: b > 0. We clamp a to min(a,b) (joint event {X_T ≤ a, max ≤ b}).
        """
        if b <= 0.0:
            return 0.0
        a_eff = a if a <= b + 0.0 else b
        return self._N(a_eff - theta) - math.exp(2.0 * b * theta) * self._N(a_eff - 2.0 * b - theta)

    def _F_minus(self, a: float, b: float, theta: float) -> float:
        """
        Down-barrier block: F⁻(a,b;θ) = F⁺(-a, -b; -θ)
        Domain: b < 0. We clamp a to max(a,b) (joint event {X_T ≥ a, min ≥ b}).
        """
        if b >= 0.0:
            return 0.0
        a_eff = a if a >= b - 0.0 else b
        # compute via symmetry on clamped values
        return self._F_plus(-a_eff, -b, -theta)

    # ---------------- Single-barrier OUT (paper-faithful, all 4 cases) ----------------
    def _single_barrier_out_price(self) -> float:
        """
        Hörfelt single-barrier OUT with BGK φ-shift:
          up:   b = d + β/√m  (d=φ(Hu)>0)
          down: b = d - β/√m  (d=φ(Hd)<0)
        Direct formulas (no cross-type parity detours):
          UO call:  S e^{-qT}[F⁺(d,b;θ1)-F⁺(c,b;θ1)] − K e^{-rT}[F⁺(d,b;θ0)-F⁺(c,b;θ0)]
          UO put:   K e^{-rT} F⁺(c,b;θ0) − S e^{-qT} F⁺(c,b;θ1)
          DO put:   K e^{-rT}[F⁻(d,b;θ0)-F⁻(c,b;θ0)] − S e^{-qT}[F⁻(d,b;θ1)-F⁻(c,b;θ1)]
          DO call:  S e^{-qT} F⁻(c,b;θ1) − K e^{-rT} F⁻(c,b;θ0)
        """
        if self.m <= 0:
            # no monitoring -> vanilla (no barrier effect)
            return self._vanilla_bs_price()

        # Immediate KO checks
        if self.barrier_type == "up-and-out" and (self.Hu is None or self.S0 >= self.Hu):
            return 0.0
        if self.barrier_type == "down-and-out" and (self.Hd is None or self.S0 <= self.Hd):
            return 0.0

        theta0, theta1 = self._theta0_theta1()
        c = self._phi(self.K)
        DF_S = math.exp(-self.q_nacc * self.T) * self.S0
        DF_K = math.exp(-self.r_nacc * self.T) * self.K

        if self.barrier_type == "up-and-out":
            # economic zero only for UO-call if K >= Hu
            if self.opt_type == "call" and self.Hu is not None and self.K >= self.Hu:
                return 0.0
            d = self._phi(self.Hu)
            b = d + BETA_BGK / math.sqrt(self.m)
            if self.opt_type == "call":
                return DF_S * ( self._F_plus(d, b, theta1) - self._F_plus(c, b, theta1) ) \
                     - DF_K * ( self._F_plus(d, b, theta0) - self._F_plus(c, b, theta0) )
            else:
                # DIRECT UO put
                return DF_K * self._F_plus(c, b, theta0) - DF_S * self._F_plus(c, b, theta1)

        elif self.barrier_type == "down-and-out":
            # economic zero only for DO-put if K <= Hd
            if self.opt_type == "put" and self.Hd is not None and self.K <= self.Hd:
                return 0.0
            d = self._phi(self.Hd)
            b = d - BETA_BGK / math.sqrt(self.m)
            if self.opt_type == "put":
                return DF_K * ( self._F_minus(d, b, theta0) - self._F_minus(c, b, theta0) ) \
                     - DF_S * ( self._F_minus(d, b, theta1) - self._F_minus(c, b, theta1) )
            else:
                # DIRECT DO call
                return DF_S * self._F_minus(c, b, theta1) - DF_K * self._F_minus(c, b, theta0)

        else:
            raise ValueError("single-barrier out price called with non-single barrier type.")

    # ---------------- Double-barrier OUT (unchanged, with Siegmund correction) ----------------
    def _G_continuous(self, a1: float, a2: float, b1: float, b2: float, theta: float, series_terms: int = 50) -> float:
        """
        Continuous analogue probability (Hörfelt Eq. (8)), series form (rapid convergence).
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
            a_m = -a; b1_m = -b1; b2_m = -b2; th_m = -theta
            width = (b2_m - b1_m)
            total = 0.0
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
        Then form the OUT price by change-of-numeraire lemma.
        """
        if self.m <= 0 or (self.Hd is None and self.Hu is None):
            return self._vanilla_bs_price()

        # Require both barriers for double-out
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
        DF_S = math.exp(-self.q_nacc * self.T) * self.S0
        DF_K = math.exp(-self.r_nacc * self.T) * self.K

        if self.opt_type == "call":
            if self.K >= self.Hu:
                return 0.0
            a1 = max(c, d1)
            a2 = d2
            return DF_S * Gm(a1, a2, theta1) - DF_K * Gm(a1, a2, theta0)
        else:
            if self.K <= self.Hd:
                return 0.0
            a1 = d1
            a2 = min(c, d2)
            return DF_K * Gm(a1, a2, theta0) - DF_S * Gm(a1, a2, theta1)

    # ---------------- scaling ----------------
    def _signed_scale(self, px: float) -> float:
        sign = 1.0 if self.direction == "long" else -1.0
        return sign * self.quantity * self.contract_multiplier * float(px)
