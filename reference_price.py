# xva_engine/reference_price.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from xva_engine.config import SamplingConvention


@dataclass(frozen=True)
class FixingSchedule:
    """
    Defines a fixing (sampling) period [start_day, end_day] in days-from-value-date,
    plus a sampling convention.

    Notes
    -----
    - BULLET: single sample at end_day (typical forward/futures settlement style)
    - DAILY/WEEKLY/MONTHLY: multiple sample days inside [start_day, end_day]
    - offset_days shifts all sample days (useful if your fixings reference a shifted date)
    """
    start_day: int
    end_day: int
    convention: SamplingConvention = SamplingConvention.DAILY
    offset_days: int = 0

    def sample_days(self) -> np.ndarray:
        start = int(self.start_day) + int(self.offset_days)
        end = int(self.end_day) + int(self.offset_days)

        if end < start:
            raise ValueError("FixingSchedule end_day must be >= start_day (after offset).")

        if self.convention == SamplingConvention.BULLET:
            return np.array([end], dtype=float)

        if self.convention == SamplingConvention.DAILY:
            return np.arange(start, end + 1, 1, dtype=float)

        if self.convention == SamplingConvention.WEEKLY:
            return np.arange(start, end + 1, 7, dtype=float)

        if self.convention == SamplingConvention.MONTHLY:
            # Lightweight approximation: 30-day step.
            return np.arange(start, end + 1, 30, dtype=float)

        raise ValueError(f"Unsupported convention: {self.convention}")


class ReferencePrice:
    """
    Reference price = deterministic functional of the simulated forward curve.

    Core idea
    ---------
    You simulate the forward curve nodes F(t, T_j). The trade's reference price is
    constructed from those simulated forwards, using a fixing schedule.

    Settlement lag convention (your requirement)
    --------------------------------------------
    If your convention is:

        "For a maturity/fixing date T, the forward price used is the curve evaluated at T + 2."

    then for every fixing sample day 'd' we look up the forward at:

        lookup_day = d + settlement_lag_days

    and if lookup_day lies between curve nodes, we **linearly interpolate** in tenor.

    Realised fixings
    ----------------
    realised_fixings is a dict keyed by sample day (int days-from-base) -> fixing value.
    - If a sample day is in realised_fixings and d <= scen_day, we treat it as known (deterministic).
    - Otherwise, we use the simulated curve for that sample day (via lookup_day = d + lag).

    Parameters
    ----------
    fixing_schedule:
        Fixing schedule defining sample days.
    settlement_lag_days:
        Settlement lag applied to each sample day when querying the curve (default 2).
        If your curve tenors ALREADY include spot/settlement lag (e.g., first tenor = 2),
        you likely want settlement_lag_days=0.
    realised_fixings:
        Optional realised fixings keyed by sample day (int).
    """

    def __init__(
        self,
        fixing_schedule: FixingSchedule,
        settlement_lag_days: int = 2,
        realised_fixings: Optional[Dict[int, float]] = None,
    ) -> None:
        self.fixing_schedule = fixing_schedule
        self.settlement_lag_days = int(settlement_lag_days)
        self.realised_fixings = realised_fixings or {}

    @staticmethod
    def _interp_curve_linear(
        tenor_days: np.ndarray,
        scen_curve: torch.Tensor,
        query_days: np.ndarray,
    ) -> torch.Tensor:
        """
        Linear interpolation of a simulated curve slice across tenor.

        Parameters
        ----------
        tenor_days : np.ndarray
            1D sorted array of curve node tenors (days-from-base), length n_tenors.
        scen_curve : torch.Tensor
            Simulated curve slice at scenario time: shape (n_tenors, n_sims).
        query_days : np.ndarray
            Days to query, shape (m,).

        Returns
        -------
        torch.Tensor
            Interpolated forwards: shape (m, n_sims).
        """
        td = np.asarray(tenor_days, dtype=float)
        if td.ndim != 1:
            raise ValueError("tenor_days must be 1D.")
        if scen_curve.ndim != 2:
            raise ValueError("scen_curve must have shape (n_tenors, n_sims).")
        if scen_curve.shape[0] != td.size:
            raise ValueError("tenor_days length must match scen_curve first dimension.")
        if td.size < 2:
            # Degenerate curve: return the single node for all queries
            return scen_curve[0:1, :].expand(int(query_days.size), scen_curve.shape[1])

        x = np.asarray(query_days, dtype=float)

        # Flat extrapolation beyond endpoints (common in curve usage)
        x = np.clip(x, td[0], td[-1])

        # Find bracketing indices:
        # j = first index with td[j] >= x ; i=j-1
        j = np.searchsorted(td, x, side="left")
        j = np.clip(j, 1, td.size - 1)
        i = j - 1

        t0 = td[i]
        t1 = td[j]
        denom = (t1 - t0)
        # denom should never be 0 if td strictly increasing; guard anyway.
        denom = np.where(denom == 0.0, 1.0, denom)
        w = (x - t0) / denom  # in [0,1]

        # Gather left/right curves and blend
        left = scen_curve[i, :]   # (m, n_sims) via advanced indexing
        right = scen_curve[j, :]  # (m, n_sims)

        w_t = torch.as_tensor(w, dtype=scen_curve.dtype, device=scen_curve.device).unsqueeze(1)
        return (1.0 - w_t) * left + w_t * right

    def compute(
        self,
        scen_index: int,
        scen_day: float,
        scen_curve: torch.Tensor,
        tenor_days: np.ndarray,
    ) -> torch.Tensor:
        """
        Compute the reference price at scenario time t = scen_day for all simulations.

        Parameters
        ----------
        scen_index : int
            Index into scenario grid (kept for API consistency; not required for computation).
        scen_day : float
            Scenario day in days-from-base.
        scen_curve : torch.Tensor
            Simulated forward curve at this scenario step:
            shape (n_tenors, n_sims) representing F(t, T_j) across nodes.
        tenor_days : np.ndarray
            Curve node tenors (days-from-base), aligned to scen_curve.

        Returns
        -------
        torch.Tensor
            Reference price per simulation: shape (n_sims,).
        """
        _ = scen_index  # explicitly unused; kept to match caller signature

        if scen_curve.ndim != 2:
            raise ValueError("scen_curve must have shape (n_tenors, n_sims).")

        sample_days = self.fixing_schedule.sample_days()

        # Split samples into realised vs future relative to scen_day
        realised_days = [d for d in sample_days if d <= scen_day and int(d) in self.realised_fixings]
        future_days = [d for d in sample_days if d > scen_day or int(d) not in self.realised_fixings]

        pieces: list[torch.Tensor] = []

        # 1) Realised component (deterministic)
        if realised_days:
            realised_vals = torch.as_tensor(
                [self.realised_fixings[int(d)] for d in realised_days],
                dtype=scen_curve.dtype,
                device=scen_curve.device,
            )
            # Broadcast scalar mean to (n_sims,)
            pieces.append(realised_vals.mean().expand(scen_curve.shape[1]))

        # 2) Future component (model-implied from simulated curve)
        if future_days:
            # Apply settlement lag: query at (fixing day + lag)
            query_days = np.asarray(future_days, dtype=float) + float(self.settlement_lag_days)

            # Linear interpolation between curve nodes at query_days
            sampled = self._interp_curve_linear(
                tenor_days=np.asarray(tenor_days, dtype=float),
                scen_curve=scen_curve,
                query_days=query_days,
            )  # (n_future, n_sims)

            pieces.append(sampled.mean(dim=0))

        if not pieces:
            return torch.zeros(scen_curve.shape[1], dtype=scen_curve.dtype, device=scen_curve.device)

        # Weighted average across realised/future samples (pro-rata by number of samples)
        n_total = len(sample_days)
        out = torch.zeros(scen_curve.shape[1], dtype=scen_curve.dtype, device=scen_curve.device)

        if realised_days:
            out += pieces[0] * (len(realised_days) / n_total)
            if future_days:
                out += pieces[1] * (len(future_days) / n_total)
        else:
            out += pieces[0]

        return out
