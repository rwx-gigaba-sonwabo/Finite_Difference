# fd_american_equity.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Dict, Iterable

import numpy as np
import pandas as pd

DayCount = Literal["ACT/365", "ACT/360"]
OptionType = Literal["call", "put"]
SettlementType = Literal["cash", "physical"]
Side = Literal["buy", "sell"]

# ---------- business-day + year-fraction helpers ----------

_BD = pd.tseries.offsets.BusinessDay()


def add_business_days(d: pd.Timestamp, n: int) -> pd.Timestamp:
    if n == 0:
        return pd.Timestamp(d).normalize()
    return (pd.Timestamp(d) + n * _BD).normalize()


def year_fraction(d1: pd.Timestamp, d2: pd.Timestamp, dc: DayCount) -> float:
    days = (pd.Timestamp(d2) - pd.Timestamp(d1)).days
    base = 365.0 if dc.upper().startswith("ACT/365") else 360.0
    return max(0.0, days / base)

# ---------- yield curve (NACA → DF) ----------

class YieldCurveNACA:
    """
    Zero curve given as nominal annual compounded-annually (NACA) rates vs 'Date',
    anchored at 'StartDate'. We linearly interpolate the zero rate in maturity T,
    then compute DF(StartDate->t) = (1 + r(T))^(-T).
    CSV columns required: 'YieldCurve','StartDate','Date','NACA'.
    """
    def __init__(self, start_date: pd.Timestamp, dates: pd.Series, rates_naca: pd.Series, day_count: DayCount):
        self.start = pd.Timestamp(start_date).normalize()
        df = pd.DataFrame({"Date": pd.to_datetime(dates).values, "NACA": pd.to_numeric(rates_naca).values})
        df = df.sort_values("Date").drop_duplicates("Date")
        self._dates = df["Date"].values
        self._rates = df["NACA"].values.astype(float)
        self.dc = day_count

    @staticmethod
    def from_csv(path: str, curve_name: str, day_count: DayCount) -> "YieldCurveNACA":
        raw = pd.read_csv(path)
        need = {"YieldCurve", "StartDate", "Date", "NACA"}
        missing = need.difference(raw.columns)
        if missing:
            raise ValueError(f"Curve CSV missing columns: {missing}")
        sub = raw.loc[raw["YieldCurve"].astype(str) == str(curve_name)].copy()
        if sub.empty:
            raise ValueError(f"Curve '{curve_name}' not found in {path}")
        start = pd.to_datetime(sub["StartDate"].iloc[0]).normalize()
        return YieldCurveNACA(start, sub["Date"], sub["NACA"], day_count)

    def _rate_at(self, date: pd.Timestamp) -> float:
        """Linear interpolation of NACA in maturity T (ACT/365 or ACT/360 as chosen)."""
        t = pd.Timestamp(date).normalize()
        if t <= self.start:
            return float(self._rates[0])
        T_list = np.array([year_fraction(self.start, pd.Timestamp(d), self.dc) for d in self._dates])
        T = year_fraction(self.start, t, self.dc)
        # clamp outside
        if T <= T_list[0]:
            return float(self._rates[0])
        if T >= T_list[-1]:
            return float(self._rates[-1])
        i = np.searchsorted(T_list, T)
        w = (T - T_list[i-1]) / (T_list[i] - T_list[i-1])
        return float((1.0 - w) * self._rates[i-1] + w * self._rates[i])

    def df(self, date: pd.Timestamp) -> float:
        """DF(StartDate -> date) from NACA: DF = (1 + r)^(-T)."""
        r = self._rate_at(date)
        T = year_fraction(self.start, date, self.dc)
        return float((1.0 + r) ** (-T))

    def df_ratio(self, start: pd.Timestamp, end: pd.Timestamp) -> float:
        """DF(end)/DF(start) with same base StartDate (robust via ratio)."""
        return self.df(end) / self.df(start)

    def nacc_between(self, start: pd.Timestamp, end: pd.Timestamp) -> float:
        """
        Continuous comp 'average' rate r* so that DF(end) = DF(start)*exp(-r* * YF).
        """
        yf = max(1e-12, year_fraction(start, end, self.dc))
        ratio = self.df_ratio(start, end)  # DF_e / DF_s
        return -np.log(ratio) / yf

# ---------- dividends ----------

def load_dividend_stream_csv(path: str) -> pd.DataFrame:
    """
    Columns: 'Ex Div Date','Record Date','Pay Date','Amount' (cents).
    """
    raw = pd.read_csv(path)
    need = {"Ex Div Date", "Record Date", "Pay Date", "Amount"}
    missing = need.difference(raw.columns)
    if missing:
        raise ValueError(f"Dividend CSV missing columns: {missing}")
    out = raw.copy()
    out["Ex Div Date"] = pd.to_datetime(out["Ex Div Date"]).dt.normalize()
    out["Record Date"] = pd.to_datetime(out["Record Date"]).dt.normalize()
    out["Pay Date"]    = pd.to_datetime(out["Pay Date"]).dt.normalize()
    out["Amount"]      = pd.to_numeric(out["Amount"]).astype(float) / 100.0  # cents -> currency units
    return out

# ---------- contract + market specs ----------

@dataclass
class ContractSpec:
    side: Side
    contracts: int = 1
    contract_multiplier: float = 1.0

    @property
    def sign(self) -> int:
        return +1 if self.side.lower() in ("buy", "long", "+", "b") else -1

    def scale(self, x: float) -> float:
        return self.sign * self.contracts * self.contract_multiplier * x

@dataclass
class MarketSpec:
    # instrument
    spot: float
    strike: float
    sigma: float
    valuation_date: pd.Timestamp
    expiry_date: pd.Timestamp
    option_type: OptionType
    settlement_type: SettlementType
    # conventions
    underlying_spot_days: int
    option_spot_days: int
    option_settlement_days: int
    day_count: DayCount
    # curves (as YieldCurveNACA objects)
    discount_curve: YieldCurveNACA
    forward_curve: YieldCurveNACA
    # dividends (DataFrame from loader)
    dividend_stream: Optional[pd.DataFrame] = None

@dataclass
class Numerics:
    k_domain: float = 6.0
    mu_star: float = 0.6
    m_min: int = 60
    n_min: int = 60
    use_richardson: bool = True
    rannacher_steps: int = 2

# ---------- tridiagonal solver ----------

def _thomas_tridiagonal(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    n = len(d)
    c_ = np.empty(n)
    d_ = np.empty(n)
    b0 = b[0]
    c_[0] = c[0] / b0
    d_[0] = d[0] / b0
    for i in range(1, n):
        denom = b[i] - a[i] * c_[i-1]
        c_[i] = c[i] / denom if i < n - 1 else 0.0
        d_[i] = (d[i] - a[i] * d_[i-1]) / denom
    x = np.empty(n)
    x[-1] = d_[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i + 1]
    return x

# ---------- core pricer ----------

class AmericanFDM:
    """
    American equity option FD pricer:
      - CN with Rannacher + projection
      - Separate carry vs discount windows
      - Discrete dividends via PV subtraction at discount_start
      - Optional Richardson extrapolation
    """

    def __init__(self, market: MarketSpec, contract: ContractSpec, numerics: Numerics = Numerics()):
        self.mkt = market
        self.cnt = contract
        self.num = numerics
        self._prepare_windows_and_effective_inputs()

    # windows/rates/effective spot ---------------------------------

    def _prepare_windows_and_effective_inputs(self) -> None:
        m = self.mkt

        # windows
        self.carry_start = add_business_days(m.valuation_date, m.underlying_spot_days)
        if m.settlement_type == "physical":
            self.carry_end = add_business_days(m.expiry_date, m.option_settlement_days)
        else:
            self.carry_end = add_business_days(m.expiry_date, m.underlying_spot_days)

        self.discount_start = add_business_days(m.valuation_date, m.option_spot_days)
        self.discount_end   = add_business_days(m.expiry_date,   m.option_settlement_days)

        # year-fractions
        self.T_pde   = year_fraction(self.discount_start, self.carry_end, m.day_count)
        self.T_carry = year_fraction(self.carry_start,    self.carry_end, m.day_count)
        self.T_disc  = year_fraction(self.discount_start, self.discount_end, m.day_count)

        # rates from curves (as NACC averages over each window)
        self.r_carry = m.forward_curve.nacc_between(self.carry_start, self.carry_end)
        self.r_disc  = m.discount_curve.nacc_between(self.discount_start, self.discount_end)

        # PV(dividends) to discount_start using the FORWARD curve
        self.pv_div_ds = 0.0
        if m.dividend_stream is not None and len(m.dividend_stream) > 0:
            div = m.dividend_stream
            mask = (div["Ex Div Date"] > m.valuation_date) & (div["Ex Div Date"] <= m.expiry_date)
            for pay, amt in zip(div.loc[mask, "Pay Date"], div.loc[mask, "Amount"]):
                self.pv_div_ds += float(amt) * m.forward_curve.df_ratio(self.discount_start, pay)

        # effective spot at PDE start
        self.S_eff_ds = max(1e-12, m.spot - self.pv_div_ds)

        # cost-of-carry drift and effective q for boundaries
        self.b = self.r_carry
        self.q_eff = self.r_disc - self.b

    # grid sizing ---------------------------------------------------

    def _domain_half_width(self) -> float:
        sigma = max(1e-12, self.mkt.sigma)
        return self.num.k_domain * sigma * np.sqrt(max(self.T_pde, 1e-12))

    def _build_grids(self, n_time: int) -> Tuple[np.ndarray, np.ndarray]:
        N = max(self.num.n_min, int(n_time))
        L = self._domain_half_width()
        T = max(1e-12, self.T_pde)
        dt = T / N
        dx = np.sqrt(dt / max(1e-12, self.num.mu_star))
        M = int(np.ceil(2.0 * L / dx))
        M = max(self.num.m_min, M)

        x_center = np.log(self.S_eff_ds / max(1e-12, self.mkt.strike))
        x_min = x_center - (M // 2) * dx
        x = x_min + dx * np.arange(M + 1)
        S = np.exp(x) * self.mkt.strike
        t = np.linspace(0.0, T, N + 1)
        return S, t

    # FD engine -----------------------------------------------------

    def _solve_once(self, n_time: int) -> float:
        S, t = self._build_grids(n_time)
        M = S.size - 1
        N = t.size - 1
        dt = t[1] - t[0]
        sigma2 = self.mkt.sigma ** 2

        # payoff at maturity of the PDE (tau=0 at carry_end)
        if self.mkt.option_type == "call":
            payoff = np.maximum(S - self.mkt.strike, 0.0)
        else:
            payoff = np.maximum(self.mkt.strike - S, 0.0)
        V = payoff.copy()

        def theta_for_step(k: int) -> float:
            return 1.0 if k < self.num.rannacher_steps else 0.5

        def tri_diag_coeffs(theta: float):
            dx = np.log(S[1] / S[0])
            nu = self.b - 0.5 * sigma2
            A = 0.5 * sigma2 / (dx * dx)
            B = nu / (2.0 * dx)
            a = A - B
            b = -2.0 * A - self.r_disc
            c = A + B
            Ai = -theta * dt * a
            Bi = 1.0 - theta * dt * b
            Ci = -theta * dt * c
            Ae = (1.0 + (1.0 - theta) * dt * b)
            Be = (1.0 - theta) * dt * a
            Ce = (1.0 - theta) * dt * c
            return (np.full(M-1, Ai), np.full(M-1, Bi), np.full(M-1, Ci),
                    np.full(M-1, Be), np.full(M-1, Ae), np.full(M-1, Ce))

        def left_bc(tau: float) -> float:
            if self.mkt.option_type == "call":
                return 0.0
            return self.mkt.strike * np.exp(-self.r_disc * (self.T_pde - tau))

        def right_bc(tau: float) -> float:
            if self.mkt.option_type == "call":
                return S[-1] * np.exp(-self.q_eff * (self.T_pde - tau)) \
                       - self.mkt.strike * np.exp(-self.r_disc * (self.T_pde - tau))
            return 0.0

        for k in range(N, 0, -1):
            th = theta_for_step(N - k)
            Ai, Bi, Ci, Be, Ae, Ce = tri_diag_coeffs(th)

            rhs = np.empty(M-1)
            rhs[:] = Ae * V[1:M]
            rhs[1:] += Ce[1:] * V[2:M]
            rhs[:-1] += Be[:-1] * V[0:M-1]

            rhs[0]  -= Ai[0] * left_bc(t[k])
            rhs[-1] -= Ci[-1] * right_bc(t[k])

            V[1:M] = _thomas_tridiagonal(Ai, Bi, Ci, rhs)

            # American projection
            V = np.maximum(V, payoff)

            V[0]  = left_bc(t[k-1])
            V[-1] = right_bc(t[k-1])

        # interpolate to S_eff at discount_start
        price_ds = float(np.interp(self.S_eff_ds, S, V))

        # Align PV to discount_end if PDE horizon differs
        if abs(self.T_pde - self.T_disc) > 1e-12:
            price_ds *= np.exp(self.r_disc * (self.T_pde - self.T_disc))
        return price_ds

    # public API ----------------------------------------------------

    def price(self, n_time: int) -> float:
        pN = self._solve_once(n_time)
        if not self.num.use_richardson:
            return self.cnt.scale(pN)
        half = max(self.num.n_min, int(n_time // 2))
        pH = self._solve_once(half)
        return self.cnt.scale((4.0 * pN - pH) / 3.0)

    def batch_price(self, n_list: Iterable[int]) -> Dict[int, float]:
        return {int(n): self.price(int(n)) for n in n_list}

    def greeks(self, n_time: int, dS_rel: float = 1e-4, dVol_abs: float = 1e-4) -> Dict[str, float]:
        base = self.price(n_time)
        S0 = self.mkt.spot
        dS = max(1e-12, dS_rel * S0)

        up = self._bumped(dS=+dS).price(n_time)
        dn = self._bumped(dS=-dS).price(n_time)
        delta = (up - dn) / (2.0 * dS)
        gamma = (up - 2.0 * base + dn) / (dS ** 2)

        v_up = self._bumped(dVol=+dVol_abs).price(n_time)
        vega = (v_up - base) / (dVol_abs * 100.0)

        m1 = self._bumped(valuation_bump_bd=1)
        th1 = m1.price(n_time)
        dt = year_fraction(self.mkt.valuation_date, add_business_days(self.mkt.valuation_date, 1), self.mkt.day_count)
        theta_annual = (th1 - base) / max(1e-12, dt)
        theta_daily = theta_annual / 365.0

        return {"Delta": delta, "Gamma": gamma, "Vega": vega,
                "Theta (Annual)": theta_annual, "Theta (Daily)": theta_daily}

    def _bumped(self, dS: float = 0.0, dVol: float = 0.0, valuation_bump_bd: int = 0) -> "AmericanFDM":
        m = self.mkt
        m2 = MarketSpec(
            spot=m.spot + dS,
            strike=m.strike,
            sigma=m.sigma + dVol,
            valuation_date=add_business_days(m.valuation_date, valuation_bump_bd),
            expiry_date=m.expiry_date,
            option_type=m.option_type,
            settlement_type=m.settlement_type,
            underlying_spot_days=m.underlying_spot_days,
            option_spot_days=m.option_spot_days,
            option_settlement_days=m.option_settlement_days,
            day_count=m.day_count,
            discount_curve=m.discount_curve,
            forward_curve=m.forward_curve,
            dividend_stream=m.dividend_stream,
        )
        return AmericanFDM(m2, self.cnt, self.num)
