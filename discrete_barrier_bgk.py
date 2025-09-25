import math
import datetime as _dt
from typing import List, Optional, Tuple, Literal, Dict

import pandas as pd

OptionType = Literal["call", "put"]
BarrierKind = Literal[
    "none",
    "up-and-out", "down-and-out", "double-out",
    "up-and-in",  "down-and-in",  "double-in",
]

BETA_BGK = 0.5826
EPS = 1e-12


class DiscreteBarrierBGKPricer:
    """
    Discrete barrier option pricer (BGK/Hörfelt) in a Black-76 forward layout.

    Key changes from spot-BS layout:
      - Two discount curves:
          * asset_discount_curve  => DF_a(T) multiplies the forward leg
          * strike_discount_curve => DF_k(T) multiplies the strike leg
      - Forward F0:
          * If forward_input is given: use it; under bumps, either keep it "constant" or
            rescale with S0 preserving implied mu = ln(F0_ref/S0_ref)/T ("recompute").
          * Else compute F0 = S0 * exp( (r_asset - q) * T ) where r_asset is the flat NACC
            implied by DF_a(T).
      - All barrier formulas replace S*e^{-qT} with DF_a*F0, and K*e^{-rT} with DF_k*K.
      - Thetas (θ0, θ1): by default computed from the *implied mu from F0* (theta_from_forward=True);
        optionally from (r_asset - q) if theta_from_forward=False.
    """

    def __init__(
        self,
        *,
        spot: float,
        strike: float,
        valuation_date: _dt.date,
        maturity_date: _dt.date,
        option_type: OptionType,
        barrier_type: BarrierKind = "none",
        lower_barrier: Optional[float] = None,
        upper_barrier: Optional[float] = None,
        monitor_dates: Optional[List[_dt.date]] = None,

        # Curves (two curves; can be the same)
        asset_discount_curve: Optional[pd.DataFrame] = None,   # DF for asset leg
        strike_discount_curve: Optional[pd.DataFrame] = None,  # DF for strike leg

        # Dividends (escrow mapping to a flat q)
        dividend_schedule: Optional[List[Tuple[_dt.date, float]]] = None,

        # Model params
        volatility: float = 0.2,
        day_count: str = "ACT/365",

        # Forward control
        forward_input: Optional[float] = None,                 # market forward (optional)
        forward_sticky: Literal["recompute", "constant"] = "recompute",
        theta_from_forward: bool = True,                       # use mu = ln(F0/S0)/T for θ's

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

        # Core state
        self.S0 = float(spot); self.S0_ref = float(spot)
        self.K  = float(strike)
        self.val_date = valuation_date
        self.mat_date = maturity_date
        self.opt_type = option_type
        self.barrier_type = barrier_type
        self.Hd = lower_barrier; self.Hu = upper_barrier

        # Monitoring: ensure maturity included
        md = sorted(monitor_dates or [])
        if md and md[-1] < self.mat_date:
            md.append(self.mat_date)
        self.monitor_dates = md

        # Curves
        self.asset_curve  = asset_discount_curve.copy()  if asset_discount_curve  is not None else None
        self.strike_curve = strike_discount_curve.copy() if strike_discount_curve is not None else None

        # Dividends to flat q
        self.dividends = sorted(dividend_schedule or [], key=lambda x: x[0])

        # Vol/daycount
        self.sigma = float(volatility)
        self.day_count = day_count.upper()

        # Forward controls
        self.forward_input = float(forward_input) if forward_input is not None else None
        self.forward_sticky = forward_sticky
        self.theta_from_forward = bool(theta_from_forward)

        # Trade/meta
        self.trade_id = trade_id
        self.direction = direction
        self.quantity = int(quantity)
        self.contract_multiplier = float(contract_multiplier)

        # Time & flat rates (from curves)
        self.T = self._year_fraction(self.val_date, self.mat_date)
        self.DFa_T = self._df_from_curve(self.asset_curve, self.mat_date) if self.asset_curve is not None else 1.0
        self.DFk_T = self._df_from_curve(self.strike_curve, self.mat_date) if self.strike_curve is not None else 1.0
        self.r_asset = -math.log(max(self.DFa_T,1e-16)) / max(self.T,1e-12) if self.T>0 else 0.0

        # Flat q via escrow mapping
        self.q = self._flat_q_from_dividends()

        # Effective monitors (count in (val, T], include T)
        self.m = self._effective_monitor_count()

        # If a forward was supplied, store its implied mu at S0_ref for sticky "recompute"
        if self.forward_input is not None:
            self.mu_impl = math.log(self.forward_input / self.S0_ref) / max(self.T,1e-12)
        else:
            self.mu_impl = (self.r_asset - self.q)

    # ---------------- Public API ----------------
    def price(self) -> float:
        if self.barrier_type == "none":
            return self._signed_scale(self._vanilla_b76_price())

        if self.barrier_type in ("up-and-out","down-and-out"):
            px = self._single_barrier_out_price()

        elif self.barrier_type in ("up-and-in","down-and-in"):
            van = self._vanilla_b76_price()
            out_eq = self._single_barrier_out_price()
            px = van - out_eq  # same-type parity

        elif self.barrier_type in ("double-out",):
            px = self._double_barrier_out_price()

        elif self.barrier_type in ("double-in",):
            van = self._vanilla_b76_price()
            out_eq = self._double_barrier_out_price()
            px = van - out_eq

        else:
            raise ValueError(f"Unsupported barrier_type: {self.barrier_type}")

        return self._signed_scale(px)

    def greeks(self, ds_rel: float = 1e-4, dvol_abs: float = 1e-4) -> Dict[str,float]:
        """
        Robust FD greeks under Black-76 forward layout.
        IMPORTANT: F0 is recomputed per call via _F0_now(), honoring forward_sticky.
        """
        saved_dir = self.direction
        self.direction = "long"

        base_S = self.S0
        ds = max(1e-8, ds_rel*base_S)

        self.S0 = base_S + ds; up = self.price()
        self.S0 = base_S - ds; dn = self.price()
        self.S0 = base_S;       base = self.price()

        delta = (up - dn) / (2*ds)
        gamma = (up - 2*base + dn) / (ds*ds)

        sig0 = self.sigma
        self.sigma = sig0 + dvol_abs; upv = self.price()
        self.sigma = sig0 - dvol_abs; dnv = self.price()
        self.sigma = sig0
        vega = (upv - dnv) / (2*dvol_abs)

        self.direction = saved_dir
        scale = (1.0 if self.direction == "long" else -1.0) * self.quantity * self.contract_multiplier
        return {"delta": scale*delta, "gamma": scale*gamma, "vega": scale*vega}

    def report(self) -> str:
        lines = []
        lines.append("==== Discrete Barrier (BGK/Hörfelt) — Black-76 Forward Layout ====")
        lines.append(f"Trade ID           : {self.trade_id}")
        lines.append(f"Direction          : {self.direction}")
        lines.append(f"Quantity           : {self.quantity}")
        lines.append(f"Contract Multiplier: {self.contract_multiplier:.6g}")
        lines.append(f"Option Type        : {self.opt_type}")
        lines.append(f"Barrier Type       : {self.barrier_type}")
        lines.append(f"Lower Barrier (Hd) : {self.Hd if self.Hd is not None else '-'}")
        lines.append(f"Upper Barrier (Hu) : {self.Hu if self.Hu is not None else '-'}")
        lines.append(f"Spot (S0)          : {self.S0:.8f}")
        lines.append(f"Strike (K)         : {self.K:.8f}")
        lines.append(f"T (years)          : {self.T:.8f}")
        lines.append(f"Volatility (sigma) : {self.sigma:.8f}")
        lines.append(f"DF_asset(T)        : {self.DFa_T:.8f}")
        lines.append(f"DF_strike(T)       : {self.DFk_T:.8f}")
        F0 = self._F0_now()
        lines.append(f"Forward F0         : {F0:.8f}  (mode={'input' if self.forward_input is not None else 'derived'}, sticky={self.forward_sticky})")
        mu = math.log(F0/self.S0)/max(self.T,1e-12)
        lines.append(f"mu (ln(F0/S0)/T)   : {mu:.8f}  (theta_from_forward={self.theta_from_forward})")
        lines.append(f"flat q (from divs) : {self.q:.8f}")
        lines.append(f"m (monitors incl T): {self.m}")
        lines.append("----------------------------------------")
        px = self.price()
        greeks = self.greeks()
        lines.append(f"Price              : {px:.10f}")
        lines.append(f"Delta              : {greeks['delta']:.10f}")
        lines.append(f"Gamma              : {greeks['gamma']:.10f}")
        lines.append(f"Vega               : {greeks['vega']:.10f}")
        return "\n".join(lines)

    # ---------------- Time/curves/dividends ----------------
    def _year_fraction(self, d0: _dt.date, d1: _dt.date) -> float:
        if self.day_count in ("ACT/365","ACT/365F"):
            return max(0,(d1-d0).days)/365.0
        if self.day_count in ("ACT/360",):
            return max(0,(d1-d0).days)/360.0
        # simple fallback
        return max(0,(d1-d0).days)/365.0

    def _df_from_curve(self, curve: Optional[pd.DataFrame], date: _dt.date) -> float:
        if curve is None or curve.empty:
            return 1.0
        # Accept either {'date','df'} or {'Date','NACA'} (flat-compounded)
        cols = {c.lower(): c for c in curve.columns}
        if "df" in cols and "date" in cols:
            df = curve
            row = df[df[cols["date"]] <= pd.Timestamp(date)].tail(1)
            if row.empty: row = df.head(1)
            return float(row[cols["df"]].values[0])
        elif "naca" in cols and "date" in cols:
            df = curve.copy()
            ser_date = pd.to_datetime(df[cols["date"]]).dt.date
            row = df[ser_date == date]
            if row.empty:
                # last available earlier
                idx = (pd.to_datetime(df[cols["date"]]).dt.date <= date)
                row = df[idx].tail(1)
                if row.empty: row = df.head(1)
            naca = float(row[cols["naca"]].values[0])
            tau = self._year_fraction(self.val_date, date)
            return (1.0 + naca) ** (-tau)
        else:
            raise ValueError("Curve must have ('date','df') or ('Date','NACA').")

    def _pv_dividends(self) -> float:
        if not self.dividends: return 0.0
        pv = 0.0
        for pay_date, amt in self.dividends:
            if self.val_date < pay_date <= self.mat_date:
                DFpay = self._df_from_curve(self.asset_curve or self.strike_curve, pay_date)
                pv += float(amt) * DFpay
        return pv

    def _flat_q_from_dividends(self) -> float:
        pvD = self._pv_dividends()
        if pvD <= 0.0: return 0.0
        if pvD >= self.S0: return 50.0
        return -math.log((self.S0 - pvD) / self.S0) / max(self.T,1e-12)

    def _effective_monitor_count(self) -> int:
        if not self.monitor_dates: return 0
        return sum(1 for d in self.monitor_dates if self.val_date < d <= self.mat_date)

    # ---------------- Forward & thetas ----------------
    def _F0_now(self) -> float:
        """
        Forward used in pricing prefactors DF_a*F0 and in d1/d2.
        - If forward_input is not None:
            * 'recompute': preserve implied mu at init => F0_now = S * exp(mu_impl T)
            * 'constant' : keep F0 fixed (sticky-forward)
        - Else: F0_now = S * exp((r_asset - q) T)
        """
        if self.forward_input is not None:
            if self.forward_sticky == "constant":
                return self.forward_input
            # recompute with implied mu from init
            return self.S0 * math.exp(self.mu_impl * self.T)
        # derive from asset curve + q
        return self.S0 * math.exp((self.r_asset - self.q) * self.T)

    def _thetas(self) -> Tuple[float,float]:
        """
        θ0 = (mu - 0.5σ²) √T / σ, θ1 = θ0 + σ√T
        where mu = ln(F0/S0)/T (default) or mu = r_asset - q (if theta_from_forward=False).
        """
        if self.T <= 0.0 or self.sigma <= 0.0:
            return 0.0, 0.0
        mu = (math.log(self._F0_now()/self.S0) / self.T) if self.theta_from_forward else (self.r_asset - self.q)
        sqrtT = math.sqrt(self.T)
        theta0 = (mu - 0.5*self.sigma*self.sigma) * sqrtT / self.sigma
        theta1 = theta0 + self.sigma*sqrtT
        return theta0, theta1

    # ---------------- Vanilla (Black-76 forward) ----------------
    def _vanilla_b76_price(self) -> float:
        """
        Generalized Black-76 with distinct discount factors:
          Call:  DF_k * ( F0 N(d1) - K N(d2) )
          Put :  DF_k * ( K N(-d2) - F0 N(-d1) )
        where d1 = [ln(F0/K) + 0.5 σ² T] / (σ√T), d2 = d1 - σ√T.
        """
        T, sig = self.T, self.sigma
        if T <= 0.0 or sig <= 0.0:
            return max(self.S0 - self.K, 0.0) if self.opt_type=="call" else max(self.K - self.S0, 0.0)
        DFk = self.DFk_T
        F0  = self._F0_now()
        st  = sig*math.sqrt(T)
        d1  = (math.log(F0/self.K) + 0.5*sig*sig*T) / st
        d2  = d1 - st
        N   = lambda x: 0.5*(1.0 + math.erf(x/math.sqrt(2.0)))
        if self.opt_type == "call":
            return DFk*( F0*N(d1) - self.K*N(d2) )
        else:
            return DFk*( self.K*N(-d2) - F0*N(-d1) )

    # ---------------- φ, N, F⁺/F⁻ blocks ----------------
    def _phi(self, x: float) -> float:
        return math.log(max(x, EPS) / self.S0) / (self.sigma * math.sqrt(max(self.T, EPS)))

    def _N(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _F_plus(self, a: float, b: float, theta: float) -> float:
        # Up-barrier block, clamp a ≤ b, b>0
        if b <= 0.0: return 0.0
        a_eff = a if a <= b + 0.0 else b
        return self._N(a_eff - theta) - math.exp(2.0*b*theta)*self._N(a_eff - 2.0*b - theta)

    def _F_minus(self, a: float, b: float, theta: float) -> float:
        # Down-barrier block, clamp a ≥ b, b<0
        if b >= 0.0: return 0.0
        a_eff = a if a >= b - 0.0 else b
        return self._F_plus(-a_eff, -b, -theta)

    # ---------------- Single-barrier OUT (Black-76 forward prefactors) ----------------
    def _single_barrier_out_price(self) -> float:
        if self.m <= 0:
            return self._vanilla_b76_price()

        # Immediate KO
        if self.barrier_type == "up-and-out":
            if self.Hu is None or self.S0 >= self.Hu: return 0.0
        if self.barrier_type == "down-and-out":
            if self.Hd is None or self.S0 <= self.Hd: return 0.0

        theta0, theta1 = self._thetas()
        c = self._phi(self.K)
        DF_a = self.DFa_T
        DF_k = self.DFk_T
        F0   = self._F0_now()
        S_leg = DF_a * F0
        K_leg = DF_k * self.K

        if self.barrier_type == "up-and-out":
            if self.opt_type == "call" and self.Hu is not None and self.K >= self.Hu:
                return 0.0
            d = self._phi(self.Hu); b = d + BETA_BGK / math.sqrt(self.m)
            if self.opt_type == "call":
                return S_leg*( self._F_plus(d,b,theta1) - self._F_plus(c,b,theta1) ) \
                     - K_leg*( self._F_plus(d,b,theta0) - self._F_plus(c,b,theta0) )
            else:
                # direct UO put
                return K_leg*self._F_plus(c,b,theta0) - S_leg*self._F_plus(c,b,theta1)

        elif self.barrier_type == "down-and-out":
            if self.opt_type == "put" and self.Hd is not None and self.K <= self.Hd:
                return 0.0
            d = self._phi(self.Hd); b = d - BETA_BGK / math.sqrt(self.m)
            if self.opt_type == "put":
                return K_leg*( self._F_minus(d,b,theta0) - self._F_minus(c,b,theta0) ) \
                     - S_leg*( self._F_minus(d,b,theta1) - self._F_minus(c,b,theta1) )
            else:
                # direct DO call
                return S_leg*self._F_minus(c,b,theta1) - K_leg*self._F_minus(c,b,theta0)

        else:
            raise ValueError("single-barrier out called with non-single barrier type.")

    # ---------------- Double-barrier OUT (same series, Black-76 prefactors) ----------------
    def _G_continuous(self, a1: float, a2: float, b1: float, b2: float, theta: float, series_terms: int=50) -> float:
        N = self._N
        total = N(a2 - theta) - N(a1 - theta)
        L = (b2 - b1)
        for k in range(1, series_terms+1):
            shift = 2.0*k*L
            total += (N(a2 - theta - shift) - N(a1 - theta - shift))
            total -= (N(a2 - theta + shift) - N(a1 - theta + shift))
        return total

    def _double_barrier_out_price(self, series_terms: int=50) -> float:
        if self.m <= 0 or (self.Hd is None and self.Hu is None):
            return self._vanilla_b76_price()
        if self.Hd is None or self.Hu is None:
            raise ValueError("Double barrier requires both lower_barrier and upper_barrier.")

        d1 = self._phi(self.Hd); d2 = self._phi(self.Hu); c = self._phi(self.K)
        theta0, theta1 = self._thetas()
        shift = BETA_BGK / math.sqrt(self.m)
        b1 = d1 - shift; b2 = d2 + shift

        DF_a = self.DFa_T; DF_k = self.DFk_T
        F0   = self._F0_now()
        S_leg = DF_a * F0
        K_leg = DF_k * self.K

        def Gm(a1,a2,th): return self._G_continuous(a1,a2,b1,b2,th,series_terms=series_terms)

        if self.opt_type == "call":
            if self.K >= self.Hu: return 0.0
            a1 = max(c, d1); a2 = d2
            return S_leg*Gm(a1,a2,theta1) - K_leg*Gm(a1,a2,theta0)
        else:
            if self.K <= self.Hd: return 0.0
            a1 = d1; a2 = min(c, d2)
            return K_leg*Gm(a1,a2,theta0) - S_leg*Gm(a1,a2,theta1)

    # ---------------- Scaling ----------------
    def _signed_scale(self, px: float) -> float:
        sgn = 1.0 if self.direction == "long" else -1.0
        return sgn * self.quantity * self.contract_multiplier * float(px)
