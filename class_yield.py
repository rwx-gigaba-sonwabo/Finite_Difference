import datetime as dt
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Literal, Tuple, Dict

import numpy as np
import pandas as pd


class NacaCurve:
    def __init__(self, discount_curve_df: pd.DataFrame, valuation_date: dt.date, day_count: str = "ACT/365F"):
        self.discount_curve_df = discount_curve_df.copy()
        self.discount_curve_df["Date"] = pd.to_datetime(self.discount_curve_df["Date"]).dt.strftime("%Y-%m-%d")
        self.valuation_date = valuation_date
        self.day_count = day_count
        self._year_denominator = self._infer_denominator(day_count)

    def _infer_denominator(self, day_count: str) -> int:
        if day_count in ("ACT/365", "ACT/365F"):
            return 365
        if day_count in ("ACT/360", "ACT/364"):
            return 360 if day_count == "ACT/360" else 364
        if day_count in ("30/360", "BOND", "US30/360"):
            return 360
        return 365

    def _year_fraction(self, start_date: dt.date, end_date: dt.date) -> float:
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

    def get_discount_factor(self, lookup_date: dt.date) -> float:
        iso = lookup_date.isoformat()
        row = self.discount_curve_df[self.discount_curve_df["Date"] == iso]
        if row.empty:
            raise ValueError(f"Discount factor not found for date: {iso}")
        naca = float(row["NACA"].values[0])
        tau = self._year_fraction(self.valuation_date, lookup_date)
        return (1.0 + naca) ** (-tau)

    def get_forward_nacc_rate(self, start_date: dt.date, end_date: dt.date) -> float:
        df_far = self.get_discount_factor(end_date)
        df_near = self.get_discount_factor(start_date)
        tau = self._year_fraction(start_date, end_date)
        return -math.log(df_far / df_near) / max(1e-12, tau)


OptionType = Literal["call", "put"]
BarrierType = Literal["none", "down-and-out", "up-and-out", "down-and-in", "up-and-in"]


@dataclass(frozen=True)
class BarrierSpec:
    barrier_type: BarrierType
    level: Optional[float] = None
    tol_bps: float = 0.0
    abs_tol: float = 0.0


@dataclass(frozen=True)
class RebateSpec:
    amount: float = 0.0
    at_hit: bool = False  # False => maturity rebate


def _vanilla_payoff(sT: np.ndarray, k: float, opt_type: OptionType) -> np.ndarray:
    if opt_type == "call":
        return np.maximum(sT - k, 0.0)
    return np.maximum(k - sT, 0.0)


def price_discrete_barrier_mc(
    spot: float,
    strike: float,
    vol: float,
    option_type: OptionType,
    valuation: dt.date,
    maturity: dt.date,
    discount_curve: NacaCurve,
    forward_curve: Optional[NacaCurve],
    dividends: Sequence[Tuple[dt.date, float]],
    monitor_dates: Sequence[dt.date],
    barrier: BarrierSpec,
    rebate: RebateSpec = RebateSpec(),
    n_paths: int = 200_000,
    seed: int = 42,
    antithetic: bool = True,
    dividend_before_monitor: bool = True,
):
    fwd_curve = forward_curve or discount_curve

    # --- build event grid ---
    div_map: Dict[dt.date, float] = {}
    for d, a in dividends:
        if valuation < d <= maturity and a != 0.0:
            div_map[d] = div_map.get(d, 0.0) + float(a)

    mon_set = {d for d in monitor_dates if valuation < d <= maturity}
    # ensure maturity is included if you want monitoring at maturity:
    mon_set.add(maturity)

    grid = sorted({valuation, maturity, *div_map.keys(), *mon_set})
    if grid[0] != valuation:
        grid = [valuation] + grid

    # precompute dt, drift, diffusion for each interval
    n_steps = len(grid) - 1
    dt_arr = np.empty(n_steps)
    drift = np.empty(n_steps)
    diff = np.empty(n_steps)

    for i in range(n_steps):
        d0, d1 = grid[i], grid[i + 1]
        tau = discount_curve._year_fraction(d0, d1)
        dt_arr[i] = tau
        carry = fwd_curve.get_forward_nacc_rate(d0, d1)
        drift[i] = (carry - 0.5 * vol * vol) * tau
        diff[i] = vol * math.sqrt(max(tau, 0.0))

    # barrier band
    if barrier.barrier_type != "none":
        if barrier.level is None:
            raise ValueError("Barrier level required.")
        band = max(barrier.abs_tol, abs(barrier.level) * (barrier.tol_bps * 1e-4))
    else:
        band = 0.0

    rng = np.random.default_rng(seed)
    n_obs = n_paths // 2 if antithetic else n_paths

    payoffs = np.empty(n_obs)

    def simulate(Z: np.ndarray) -> np.ndarray:
        s = np.full(Z.shape[0], spot, dtype=float)
        bt = barrier.barrier_type

        if bt in ("down-and-out", "up-and-out"):
            alive = np.ones(Z.shape[0], dtype=bool)
            hit_idx = np.full(Z.shape[0], -1, dtype=int)
        elif bt in ("down-and-in", "up-and-in"):
            hit = np.zeros(Z.shape[0], dtype=bool)
        else:
            alive = None
            hit = None

        for i in range(n_steps):
            s *= np.exp(drift[i] + diff[i] * Z[:, i])

            d1 = grid[i + 1]

            # dividend vs monitoring ordering
            if dividend_before_monitor:
                if d1 in div_map:
                    s = np.maximum(s - div_map[d1], 1e-12)

            if d1 in mon_set and bt != "none":
                if bt in ("down-and-out", "down-and-in"):
                    breached = s <= (barrier.level + band)
                else:
                    breached = s >= (barrier.level - band)

                if bt in ("down-and-out", "up-and-out"):
                    if rebate.at_hit and rebate.amount != 0.0:
                        newly = alive & breached
                        hit_idx[newly] = i + 1  # grid index
                    alive &= ~breached
                else:
                    hit |= breached

            if not dividend_before_monitor:
                if d1 in div_map:
                    s = np.maximum(s - div_map[d1], 1e-12)

        sT = s
        vanilla = _vanilla_payoff(sT, strike, option_type)
        dfT = discount_curve.get_discount_factor(maturity)

        if bt == "none":
            return dfT * vanilla

        if bt in ("down-and-out", "up-and-out"):
            out = np.zeros_like(vanilla)
            out[alive] = dfT * vanilla[alive]
            if rebate.amount != 0.0:
                if rebate.at_hit:
                    knocked = hit_idx >= 0
                    if np.any(knocked):
                        # PV at hit date
                        for j in np.where(knocked)[0]:
                            out[j] = rebate.amount * discount_curve.get_discount_factor(grid[hit_idx[j]])
                else:
                    out[~alive] = rebate.amount * dfT
            return out

        # knock-in
        return dfT * vanilla * hit

    # run MC
    for start in range(0, n_obs, 50_000):
        m = min(50_000, n_obs - start)
        Z = rng.standard_normal(size=(m, n_steps))
        if antithetic:
            p = 0.5 * (simulate(Z) + simulate(-Z))
        else:
            p = simulate(Z)
        payoffs[start : start + m] = p

    price = float(np.mean(payoffs))
    stderr = float(np.std(payoffs, ddof=1) / math.sqrt(len(payoffs)))
    return {
        "price": price,
        "stderr": stderr,
        "ci95": (price - 1.96 * stderr, price + 1.96 * stderr),
        "n_obs": int(len(payoffs)),
        "antithetic": antithetic,
        "grid_points": len(grid),
    }
