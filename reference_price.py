from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import matplotlib.pyplot as plt

from xva_engine.config import SamplingConvention
from xva_engine.utils.dates import to_date, day_offset, ensure_dates


@dataclass(frozen=True)
class FixingSchedule:
    start_date: date
    end_date: date
    convention: SamplingConvention = SamplingConvention.DAILY
    offset_days: int = 0

    def sample_dates(self) -> list[date]:
        start = self.start_date
        end = self.end_date
        if end < start:
            raise ValueError("FixingSchedule end_date must be >= start_date.")

        if self.convention == SamplingConvention.BULLET:
            return [end]

        if self.convention == SamplingConvention.DAILY:
            step = 1
        elif self.convention == SamplingConvention.WEEKLY:
            step = 7
        elif self.convention == SamplingConvention.MONTHLY:
            step = 30  # lightweight approximation
        else:
            raise ValueError(f"Unsupported convention: {self.convention}")

        out = []
        d = start
        while d <= end:
            out.append(d)
            d = d.fromordinal(d.toordinal() + step)
        return out

    def sample_days(self, base_date: date) -> np.ndarray:
        return np.asarray(
            [day_offset(base_date, d) + int(self.offset_days) for d in self.sample_dates()],
            dtype=float,
        )


class ReferencePrice:
    def __init__(
        self,
        *,
        fixing_schedule: FixingSchedule,
        delivery_dates: Sequence[date],
        realised_fixings: Optional[Dict[date, float]] = None,
    ) -> None:
        self.fixing_schedule = fixing_schedule
        self.delivery_dates = ensure_dates(delivery_dates)
        if not self.delivery_dates:
            raise ValueError("delivery_dates must be non-empty.")
        self.realised_fixings = {to_date(k): float(v) for k, v in (realised_fixings or {}).items()}

    def _map_sample_to_delivery(self, sample_date: date) -> date:
        for d in self.delivery_dates:
            if d >= sample_date:
                return d
        return self.delivery_dates[-1]

    def compute(
        self,
        *,
        base_date: date,
        scen_date: date,
        scen_curve: torch.Tensor,     # (n_tenors, n_sims)
        tenor_dates: Sequence[date],  # curve node dates T_j
    ) -> torch.Tensor:
        scen_date = to_date(scen_date)
        tenor_dates = ensure_dates(tenor_dates)

        if scen_curve.ndim != 2:
            raise ValueError("scen_curve must have shape (n_tenors, n_sims).")
        if len(tenor_dates) != scen_curve.shape[0]:
            raise ValueError("tenor_dates length must match scen_curve first dimension.")

        sample_dates = self.fixing_schedule.sample_dates()

        realised = [d for d in sample_dates if (d <= scen_date) and (d in self.realised_fixings)]
        future = [d for d in sample_dates if (d > scen_date) or (d not in self.realised_fixings)]

        values: list[torch.Tensor] = []

        if realised:
            realised_vals = torch.as_tensor(
                [self.realised_fixings[d] for d in realised],
                dtype=scen_curve.dtype,
                device=scen_curve.device,
            )
            values.append(realised_vals.mean().expand(scen_curve.shape[1]))

        if future:
            mapped_delivery = [self._map_sample_to_delivery(d) for d in future]
            tenor_days_arr = np.asarray([day_offset(base_date, d) for d in tenor_dates], dtype=float)
            mapped_days = np.asarray([day_offset(base_date, d) for d in mapped_delivery], dtype=float)

            # snap (replace with interpolation in production)
            idxs = [int(np.argmin(np.abs(tenor_days_arr - md))) for md in mapped_days]
            sampled = scen_curve[idxs, :]  # (n_future, n_sims)
            values.append(sampled.mean(dim=0))

        if not values:
            return torch.zeros(scen_curve.shape[1], dtype=scen_curve.dtype, device=scen_curve.device)

        n_total = len(sample_dates)
        out = torch.zeros(scen_curve.shape[1], dtype=scen_curve.dtype, device=scen_curve.device)

        if realised:
            out += values[0] * (len(realised) / n_total)
            if future:
                out += values[1] * (len(future) / n_total)
        else:
            out += values[0]

        return out

    def plot_schedule(self, *, value_date: Optional[date] = None, title: str = "Fixing schedule") -> None:
        fix_dates = self.fixing_schedule.sample_dates()
        y = np.zeros(len(fix_dates))

        plt.figure()
        plt.plot(fix_dates, y, marker="o", linestyle="None")
        plt.axvline(self.fixing_schedule.start_date, linestyle="--", label="fix start")
        plt.axvline(self.fixing_schedule.end_date, linestyle="--", label="fix end")
        if value_date is not None:
            plt.axvline(value_date, linestyle=":", label="value date")
        plt.yticks([])
        plt.title(title)
        plt.legend()
        plt.tight_layout()
