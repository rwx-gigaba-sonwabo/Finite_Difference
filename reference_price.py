from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import torch

from xva_engine.config import SamplingConvention


@dataclass(frozen=True)
class FixingSchedule:
    """
    Defines a fixing (sampling) period [start_day, end_day] in days from base date,
    plus a sampling convention.

    Example:
        monthly averaging over a month -> DAILY samples inside [start, end]
        futures-settlement style -> BULLET sample at end_day only
    """
    start_day: int
    end_day: int
    convention: SamplingConvention = SamplingConvention.DAILY
    offset_days: int = 0  # optional sample-date shift

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
            # Approximate monthly as 30-day step for a lightweight mimic.
            return np.arange(start, end + 1, 30, dtype=float)

        raise ValueError(f"Unsupported convention: {self.convention}")


class ReferencePrice:
    """
    Reference price = deterministic functional of the simulated forward curve.

    RiskFlow notion:
    - you do NOT simulate reference prices directly.
    - you simulate ForwardPrice curves, then compute reference prices by sampling
      the curve at specific "reference/delivery dates" per sample date.

    This simplified implementation:
    - maps each sample day t_i to a delivery tenor day T'(t_i) using:
        T'(t_i) = min{delivery_day >= t_i}
      (flat-left / closest delivery proxy in spirit)
    - computes arithmetic average over samples, mixing realised fixings if provided.
    """
    def __init__(
        self,
        fixing_schedule: FixingSchedule,
        delivery_days: np.ndarray,
        realised_fixings: Optional[Dict[int, float]] = None,
    ) -> None:
        self.fixing_schedule = fixing_schedule
        self.delivery_days = np.asarray(delivery_days, dtype=float)
        if self.delivery_days.ndim != 1:
            raise ValueError("delivery_days must be 1D.")
        self.realised_fixings = realised_fixings or {}

    def _map_sample_to_delivery(self, sample_day: float) -> float:
        idx = np.searchsorted(self.delivery_days, sample_day, side="left")
        if idx >= self.delivery_days.size:
            # If sample beyond last delivery node, clamp to last node (flat extrapolation style).
            return float(self.delivery_days[-1])
        return float(self.delivery_days[idx])

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
            index into scenario grid (0..n_steps-1)
        scen_day : float
            scenario day in days-from-base
        scen_curve : torch.Tensor
            simulated forward curve at this scenario index:
            shape (n_tenors, n_sims)  (this is F(t, T_j) across nodes)
        tenor_days : np.ndarray
            curve node tenors (days-from-base), aligned to scen_curve

        Returns
        -------
        torch.Tensor
            reference price per simulation: shape (n_sims,)
        """
        sample_days = self.fixing_schedule.sample_days()
        # Split samples into "already realised" and "future" relative to scen_day
        realised = [d for d in sample_days if d <= scen_day and int(d) in self.realised_fixings]
        future = [d for d in sample_days if d > scen_day or int(d) not in self.realised_fixings]

        values = []

        # Add realised fixings as constants (broadcast)
        if realised:
            # Use the provided realised fixing values (already "fact")
            realised_vals = torch.as_tensor(
                [self.realised_fixings[int(d)] for d in realised],
                dtype=scen_curve.dtype,
                device=scen_curve.device,
            )
            # mean over realised samples (same for every scenario path)
            values.append(realised_vals.mean().expand(scen_curve.shape[1]))

        # Add future samples by sampling the simulated curve
        if future:
            mapped_delivery = [self._map_sample_to_delivery(d) for d in future]
            # For each mapped delivery day, find nearest curve node index
            # (in a full implementation, you might interpolate; we snap to nearest node for simplicity)
            tenor_days_arr = np.asarray(tenor_days, dtype=float)
            idxs = [int(np.argmin(np.abs(tenor_days_arr - md))) for md in mapped_delivery]
            sampled = scen_curve[idxs, :]  # (n_future, n_sims)
            values.append(sampled.mean(dim=0))

        if not values:
            # No samples? return zeros
            return torch.zeros(scen_curve.shape[1], dtype=scen_curve.dtype, device=scen_curve.device)

        # If both realised and future exist, average them with correct weights
        # (RiskFlow uses prorating by number of samples; we do the same)
        n_total = len(sample_days)
        out = torch.zeros(scen_curve.shape[1], dtype=scen_curve.dtype, device=scen_curve.device)

        if realised:
            out += values[0] * (len(realised) / n_total)
            if future:
                out += values[1] * (len(future) / n_total)
        else:
            out += values[0]  # only future samples

        return out
