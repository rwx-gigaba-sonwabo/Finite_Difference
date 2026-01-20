"""
mc_discrete_barrier_option.py

Discrete (date-scheduled) single-barrier option valuation via Monte Carlo,
aligned to the Riskworx / RiskFlow-style curve handling you showed:

- Curves are DAILY NACA: DataFrame with columns ["Date", "NACA"] where Date is ISO "YYYY-MM-DD".
- Discount factor:
    DF(valuation, d) = (1 + NACA(d)) ** (-tau(valuation, d))
- Forward NACC rate over [d0, d1]:
    f_nacc(d0, d1) = -ln(DF(d1)/DF(d0)) / tau(d0, d1)

- Dividends are [(pay_date, cash_amount), ...] and applied as cash drops on pay_date.
- Monitoring times are dates (not year-fractions) and barrier is checked ONLY on those dates.
- Non-uniform time steps come naturally from the union grid:
    {valuation} ∪ {div_dates} ∪ {monitor_dates} ∪ {maturity}

Barrier breach "band" (your bp interval around barrier):
- Down barrier: hit if S <= B + band
- Up barrier:   hit if S >= B - band
where band = max(abs_tol, B * tol_bps * 1e-4)

Supports:
- barrier_type in:
    "none",
    "down-and-out", "up-and-out",
    "down-and-in",  "up-and-in"
- rebate for knock-out:
    - paid at maturity (default) or paid at hit time (rebate_at_hit=True)

Notes:
- Underlying is GBM with deterministic per-step carry extracted from the forward curve:
    ln S_{i+1} = ln S_i + (carry_i - 0.5 σ^2) Δt + σ sqrt(Δt) Z
  where carry_i = forward_curve.get_forward_nacc_rate(d_i, d_{i+1})
- Discounting uses DF(valuation, maturity) for terminal payoffs.
- If you need day-count conventions to match your FD code, set day_count consistently.

Dependencies: numpy, pandas
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


OptionType = Literal["call", "put"]
BarrierType = Literal["none", "down-and-out", "up-and-out", "down-and-in", "up-and-in"]


class NacaCurve:
    """
    Daily NACA curve with exact-date lookup.

    Expected df columns:
      - "Date" : "YYYY-MM-DD"
      - "NACA" : numeric (e.g. 0.085 for 8.5% NACA)
    """

    def __init__(
        self,
        discount_curve_df: pd.DataFrame,
        valuation_date: dt.date,
        day_count: str = "ACT/365F",
        date_col: str = "Date",
        naca_col: str = "NACA",
    ) -> None:
        self.valuation_date = valuation_date
        self.day_count = day_count
        self._year_denominator = self._infer_denominator(day_count)

        df = discount_curve_df.copy()
        if date_col not in df.columns or naca_col not in df.columns:
            raise ValueError(f"Curve df must have columns [{date_col}, {naca_col}].")

        df[date_col] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
        df[naca_col] = pd.to_numeric(df[naca_col], errors="coerce")

        if df[naca_col].isna().any():
            bad = df[df[naca_col].isna()].head(5)
            raise ValueError(f"Found non-numeric NACA values. Examples:\n{bad}")

        self._df = df[[date_col, naca_col]].rename(columns={date_col: "Date", naca_col: "NACA"})
        # For faster exact lookups
        self._map: Dict[str, float] = dict(zip(self._df["Date"].values, self._df["NACA"].values))

    @classmethod
    def from_csv(
        cls,
        path: str,
        valuation_date: dt.date,
        day_count: str = "ACT/365F",
        date_col: str = "Date",
        naca_col: str = "NACA",
        date_format: Optional[str] = None,
    ) -> "NacaCurve":
        df = pd.read_csv(path)
        if date_format is not None:
            df[date_col] = pd.to_datetime(df[date_col], format=date_format)
        return cls(df, valuation_date=valuation_date, day_count=day_count, date_col=date_col, naca_col=naca_col)

    def _infer_denominator(self, day_count: str) -> int:
        if day_count in ("ACT/365", "ACT/365F"):
            return 365
        if day_count in ("ACT/360", "ACT/364"):
            return 360 if day_count == "ACT/360" else 364
        if day_count in ("30/360", "BOND", "US30/360"):
            return 360
        return 365

    def year_fraction(self, start_date: dt.date, end_date: dt.date) -> float:
        if end_date <= start_date:
            return 0.0
        if self.day_count in ("ACT/365", "ACT/365F", "ACT/360", "ACT/364"):
            return (end_date - start_date).days / float(self._year_denominator)

        if self.day_count in ("30/360", "BOND", "US30/360"):
            y1, m1, d1 = start_date.year, start_date.month, start_date.day
            y2, m2, d2 = end_date.year, end_date.month, end_date.day
            d1 = min(d1, 30)
            if d1 == 30:
                d2 = min(d2, 30)
            days = (y2 - y1) * 360 + (m2 - m1) * 30 + (d2 - d1)
            return days / 360.0

        return (end_date - start_date).days / 365.0

    def get_naca(self, lookup_date: dt.date) -> float:
        iso = lookup_date.isoformat()
        if iso not in self._map:
            raise ValueError(f"NACA not found for date: {iso}")
        return float(self._map[iso])

    def get_discount_factor(self, lookup_date: dt.date) -> float:
        """
        DF(valuation, lookup_date) using the curve's DAILY NACA at lookup_date:
            DF = (1 + NACA(lookup_date)) ** (-tau(valuation, lookup_date))
        """
        naca = self.get_naca(lookup_date)
        tau = self.year_fraction(self.valuation_date, lookup_date)
        return (1.0 + naca) ** (-tau)

    def get_forward_nacc_rate(self, start_date: dt.date, end_date: dt.date) -> float:
        """
        Forward NACC rate (continuous) implied by curve DFs:
            f = -ln(DF(end)/DF(start)) / tau(start,end)
        """
        df_far = self.get_discount_factor(end_date)
        df_near = self.get_discount_factor(start_date)
        tau = self.year_fraction(start_date, end_date)
        return -math.log(df_far / df_near) / max(1e-12, tau)


@dataclass(frozen=True)
class BarrierSpec:
    barrier_type: BarrierType
    level: Optional[float] = None
    tol_bps: float = 0.0       # e.g. 1.0 means 1bp band
    abs_tol: float = 0.0       # absolute band (added via max)


@dataclass(frozen=True)
class RebateSpec:
    amount: float = 0.0
    rebate_at_hit: bool = False   # if True, PV rebate at hit date; else pay at maturity


@dataclass(frozen=True)
class MCConfig:
    n_paths: int = 200_000
    seed: int = 42
    antithetic: bool = True
    chunk_size: int = 50_000
    dividend_before_monitor: bool = True
    spot_floor: float = 1e-12     # prevent negative/zero after dividend


def _vanilla_payoff(sT: np.ndarray, strike: float, option_type: OptionType) -> np.ndarray:
    if option_type == "call":
        return np.maximum(sT - strike, 0.0)
    return np.maximum(strike - sT, 0.0)


def _barrier_band(level: float, tol_bps: float, abs_tol: float) -> float:
    return max(abs_tol, abs(level) * (tol_bps * 1e-4))


def _build_event_grid(
    valuation: dt.date,
    maturity: dt.date,
    dividends: Sequence[Tuple[dt.date, float]],
    monitor_dates: Sequence[dt.date],
    include_maturity_monitor: bool = True,
) -> Tuple[List[dt.date], Dict[dt.date, float], set]:
    """
    Returns:
      - grid dates (sorted unique), including valuation and maturity
      - div_map: date -> total cash dividend
      - monitor_set: set of monitoring dates
    """
    if maturity <= valuation:
        raise ValueError("maturity must be after valuation.")

    div_map: Dict[dt.date, float] = {}
    for d, amt in dividends:
        if valuation < d <= maturity and float(amt) != 0.0:
            div_map[d] = div_map.get(d, 0.0) + float(amt)

    monitor_set = {d for d in monitor_dates if valuation < d <= maturity}
    if include_maturity_monitor:
        monitor_set.add(maturity)

    grid = sorted({valuation, maturity, *div_map.keys(), *monitor_set})
    if grid[0] != valuation:
        grid = [valuation] + grid

    return grid, div_map, monitor_set


def price_discrete_barrier_mc(
    *,
    spot: float,
    strike: float,
    vol: float,
    option_type: OptionType,
    valuation: dt.date,
    maturity: dt.date,
    discount_curve: NacaCurve,
    forward_curve: Optional[NacaCurve] = None,
    dividends: Sequence[Tuple[dt.date, float]] = (),
    monitor_dates: Sequence[dt.date] = (),
    barrier: BarrierSpec = BarrierSpec("none"),
    rebate: RebateSpec = RebateSpec(),
    cfg: MCConfig = MCConfig(),
    include_maturity_monitor: bool = True,
) -> Dict[str, object]:
    """
    Monte Carlo pricer for discretely monitored single-barrier options (single underlying).

    Parameters mirror your FD workflow:
      - Curves extracted like your get_discount_factor / get_forward_nacc_rate methods
      - Dividends and monitoring dates are calendar dates (dt.date)
    """
    if spot <= 0.0 or strike <= 0.0:
        raise ValueError("spot and strike must be positive.")
    if vol < 0.0:
        raise ValueError("vol must be non-negative.")

    fwd_curve = forward_curve or discount_curve

    grid, div_map, monitor_set = _build_event_grid(
        valuation=valuation,
        maturity=maturity,
        dividends=dividends,
        monitor_dates=monitor_dates,
        include_maturity_monitor=include_maturity_monitor,
    )

    n_steps = len(grid) - 1
    if n_steps <= 0:
        raise ValueError("Event grid has no steps.")

    # Precompute per-step dt, drift, diffusion.
    # Drift uses the forward curve implied continuous forward over each interval.
    dt_arr = np.empty(n_steps, dtype=float)
    drift = np.empty(n_steps, dtype=float)
    diff = np.empty(n_steps, dtype=float)

    for i in range(n_steps):
        d0, d1 = grid[i], grid[i + 1]
        tau = discount_curve.year_fraction(d0, d1)
        dt_arr[i] = tau

        carry = fwd_curve.get_forward_nacc_rate(d0, d1)
        drift[i] = (carry - 0.5 * vol * vol) * tau
        diff[i] = vol * math.sqrt(max(tau, 0.0))

    # Discount factor to maturity
    df_T = discount_curve.get_discount_factor(maturity)

    # Barrier settings
    bt = barrier.barrier_type
    if bt != "none":
        if barrier.level is None or barrier.level <= 0.0:
            raise ValueError("barrier.level must be provided and positive for barrier options.")
        band = _barrier_band(barrier.level, barrier.tol_bps, barrier.abs_tol)
    else:
        band = 0.0

    # RNG and path count
    rng = np.random.default_rng(cfg.seed)
    n_paths = int(cfg.n_paths)
    if n_paths <= 0:
        raise ValueError("cfg.n_paths must be positive.")
    use_anti = bool(cfg.antithetic)
    n_obs = n_paths // 2 if use_anti else n_paths
    if use_anti and n_obs <= 0:
        raise ValueError("With antithetic=True, set n_paths >= 2.")

    chunk = max(1, int(cfg.chunk_size))

    def barrier_breached(s: np.ndarray) -> np.ndarray:
        if bt in ("down-and-out", "down-and-in"):
            return s <= (barrier.level + band)
        if bt in ("up-and-out", "up-and-in"):
            return s >= (barrier.level - band)
        return np.zeros_like(s, dtype=bool)

    def simulate_payoffs(Z: np.ndarray) -> np.ndarray:
        """
        Z shape: (n_sim, n_steps)
        Returns discounted payoff array of length n_sim.
        """
        n_sim = Z.shape[0]
        s = np.full(n_sim, float(spot), dtype=float)

        # State
        if bt in ("down-and-out", "up-and-out"):
            alive = np.ones(n_sim, dtype=bool)
            hit_step = np.full(n_sim, -1, dtype=int)  # step index (1..n_steps)
        elif bt in ("down-and-in", "up-and-in"):
            hit = np.zeros(n_sim, dtype=bool)
        else:
            alive = None
            hit = None

        for i in range(n_steps):
            # Evolve GBM to grid[i+1]
            s *= np.exp(drift[i] + diff[i] * Z[:, i])

            d1 = grid[i + 1]

            # Dividend/monitor ordering at same date matters.
            if cfg.dividend_before_monitor:
                if d1 in div_map:
                    s = np.maximum(s - div_map[d1], cfg.spot_floor)

            if d1 in monitor_set and bt != "none":
                breached = barrier_breached(s)

                if bt in ("down-and-out", "up-and-out"):
                    if rebate.rebate_at_hit and rebate.amount != 0.0:
                        newly = alive & breached
                        # record first hit time as step index (1-based)
                        hit_step[newly] = i + 1
                    alive &= ~breached
                else:
                    hit |= breached

            if not cfg.dividend_before_monitor:
                if d1 in div_map:
                    s = np.maximum(s - div_map[d1], cfg.spot_floor)

        sT = s
        vanilla = _vanilla_payoff(sT, strike, option_type)

        if bt == "none":
            return df_T * vanilla

        if bt in ("down-and-out", "up-and-out"):
            payoff = np.zeros_like(vanilla)

            # survivors get vanilla at maturity
            payoff[alive] = df_T * vanilla[alive]

            if rebate.amount != 0.0:
                knocked = ~alive
                if rebate.rebate_at_hit:
                    # PV rebate at first hit date (requires recorded hit_step).
                    # If a path was knocked out but hit_step not recorded (e.g. rebate_at_hit False),
                    # it won't happen here; but for safety:
                    idx = np.where(hit_step >= 1)[0]
                    for j in idx:
                        hit_date = grid[hit_step[j]]
                        payoff[j] = rebate.amount * discount_curve.get_discount_factor(hit_date)
                else:
                    payoff[knocked] = rebate.amount * df_T

            return payoff

        # knock-in
        return df_T * vanilla * hit

    # Streaming estimation (avoid holding all payoffs)
    sum_p = 0.0
    sum_p2 = 0.0
    obs_done = 0

    while obs_done < n_obs:
        m = min(chunk, n_obs - obs_done)
        Z = rng.standard_normal(size=(m, n_steps))

        if use_anti:
            p = 0.5 * (simulate_payoffs(Z) + simulate_payoffs(-Z))
        else:
            p = simulate_payoffs(Z)

        sum_p += float(np.sum(p))
        sum_p2 += float(np.sum(p * p))
        obs_done += m

    n = float(n_obs)
    price = sum_p / n
    var = max(0.0, (sum_p2 / n) - price * price)
    stderr = math.sqrt(var / n)

    return {
        "price": float(price),
        "stderr": float(stderr),
        "ci_95": (float(price - 1.96 * stderr), float(price + 1.96 * stderr)),
        "n_observations": int(n_obs),
        "antithetic": bool(use_anti),
        "grid_points": int(len(grid)),
        "steps": int(n_steps),
        "barrier_type": bt,
        "barrier_level": barrier.level,
        "barrier_band": float(band),
        "dividend_before_monitor": bool(cfg.dividend_before_monitor),
    }


# -------------------------
# Example main script usage
# -------------------------
if __name__ == "__main__":
    # ---- Load curves (daily NACA) ----
    # Example path; replace with yours
    discount_curve_df = pd.read_csv(r"C:\Finite Difference (Equity Americans)\Finite_Difference-main\ZAR-SWAP-28072025.csv")
    discount_curve_df["Date"] = pd.to_datetime(discount_curve_df["Date"], format="%Y/%m/%d").dt.strftime("%Y-%m-%d")

    # Optional separate forward curve; otherwise discount_curve used
    forward_curve_df = discount_curve_df.copy()

    # ---- Dates ----
    valuation = dt.date(2025, 7, 28)
    maturity = dt.date(2026, 7, 28)

    # ---- Wrap curves ----
    discount_curve = NacaCurve(discount_curve_df, valuation_date=valuation, day_count="ACT/365F")
    forward_curve = NacaCurve(forward_curve_df, valuation_date=valuation, day_count="ACT/365F")

    # ---- Dividends (pay_date, amount) ----
    divs = [
        (dt.date(2025, 9, 12), 7.63),
        (dt.date(2026, 4, 10), 8.0115),
    ]

    # ---- Monitoring dates (weekly example) ----
    monitor_dates = [valuation + dt.timedelta(days=7 * i) for i in range(1, 27)]

    # ---- Barrier spec (example) ----
    barrier = BarrierSpec(
        barrier_type="down-and-out",
        level=95.0,
        tol_bps=1.0,   # 1bp band around barrier
        abs_tol=0.0,
    )

    rebate = RebateSpec(amount=0.0, rebate_at_hit=False)

    cfg = MCConfig(
        n_paths=200_000,
        seed=42,
        antithetic=True,
        chunk_size=50_000,
        dividend_before_monitor=True,  # flip if RiskFlow checks barrier before dividend on same date
    )

    out = price_discrete_barrier_mc(
        spot=100.0,
        strike=100.0,
        vol=0.25,
        option_type="call",
        valuation=valuation,
        maturity=maturity,
        discount_curve=discount_curve,
        forward_curve=forward_curve,
        dividends=divs,
        monitor_dates=monitor_dates,
        barrier=barrier,
        rebate=rebate,
        cfg=cfg,
        include_maturity_monitor=True,  # set False if maturity is NOT a monitoring observation in your definition
    )

    print(out)
