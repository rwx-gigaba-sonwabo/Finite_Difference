from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from xva_engine.config import SamplingConvention


@dataclass(frozen=True)
class FixingSchedule:
    """
    Fixing (sampling) period [start_day, end_day] in days-from-value-date.

    Conventions:
    - BULLET: single sample at end_day
    - DAILY/WEEKLY/MONTHLY: multiple samples in the window
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
            return np.array([float(end)], dtype=float)

        if self.convention == SamplingConvention.DAILY:
            return np.arange(start, end + 1, 1, dtype=float)

        if self.convention == SamplingConvention.WEEKLY:
            return np.arange(start, end + 1, 7, dtype=float)

        if self.convention == SamplingConvention.MONTHLY:
            # lightweight approximation: 30-day step
            return np.arange(start, end + 1, 30, dtype=float)

        raise ValueError(f"Unsupported convention: {self.convention}")


class ReferencePrice:
    """
    Build reference price from a simulated forward curve slice F(t, T_j).

    Key conventions supported:
    - Settlement lag: query forwards at (fixing_day + settlement_lag_days)
    - Linear interpolation in tenor between curve nodes
    - Realised fixings: dict[sample_day(int)] -> value; used when sample_day <= scen_day
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
        scen_curve: torch.Tensor,  # (n_tenors, n_sims)
        query_days: np.ndarray,    # (m,)
    ) -> torch.Tensor:
        td = np.asarray(tenor_days, dtype=float)
        if td.ndim != 1:
            raise ValueError("tenor_days must be 1D.")
        if scen_curve.ndim != 2:
            raise ValueError("scen_curve must have shape (n_tenors, n_sims).")
        if scen_curve.shape[0] != td.size:
            raise ValueError("tenor_days length must match scen_curve first dimension.")
        if td.size < 2:
            return scen_curve[0:1, :].expand(int(query_days.size), scen_curve.shape[1])

        x = np.asarray(query_days, dtype=float)
        x = np.clip(x, td[0], td[-1])  # flat extrapolation

        j = np.searchsorted(td, x, side="left")
        j = np.clip(j, 1, td.size - 1)
        i = j - 1

        t0 = td[i]
        t1 = td[j]
        denom = np.where((t1 - t0) == 0.0, 1.0, (t1 - t0))
        w = (x - t0) / denom  # in [0,1]

        left = scen_curve[i, :]   # (m, n_sims)
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
        _ = scen_index  # kept for API consistency

        sample_days = self.fixing_schedule.sample_days()

        realised_days = [d for d in sample_days if d <= scen_day and int(d) in self.realised_fixings]
        future_days = [d for d in sample_days if d > scen_day or int(d) not in self.realised_fixings]

        parts: list[torch.Tensor] = []

        # realised component
        if realised_days:
            realised_vals = torch.as_tensor(
                [self.realised_fixings[int(d)] for d in realised_days],
                dtype=scen_curve.dtype,
                device=scen_curve.device,
            )
            parts.append(realised_vals.mean().expand(scen_curve.shape[1]))

        # future component from curve, using settlement lag and linear interpolation
        if future_days:
            query_days = np.asarray(future_days, dtype=float) + float(self.settlement_lag_days)
            sampled = self._interp_curve_linear(
                tenor_days=np.asarray(tenor_days, dtype=float),
                scen_curve=scen_curve,
                query_days=query_days,
            )  # (n_future, n_sims)
            parts.append(sampled.mean(dim=0))

        if not parts:
            return torch.zeros(scen_curve.shape[1], dtype=scen_curve.dtype, device=scen_curve.device)

        # pro-rata mix if both realised and future exist
        n_total = len(sample_days)
        out = torch.zeros(scen_curve.shape[1], dtype=scen_curve.dtype, device=scen_curve.device)

        if realised_days:
            out += parts[0] * (len(realised_days) / n_total)
            if future_days:
                out += parts[1] * (len(future_days) / n_total)
        else:
            out += parts[0]

        return out
