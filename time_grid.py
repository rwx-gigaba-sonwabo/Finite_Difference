from __future__ import annotations


from dataclasses import dataclass
from datetime import date
import numpy as np

from xva_engine.utils.dates import add_days


@dataclass(frozen=True)
class TimeGrid:
    base_date: date
    scen_days: np.ndarray  # (n_times,)

    @classmethod
    def from_horizon(cls, base_date: date, dt_days: int, horizon_days: int) -> "TimeGrid":
        if dt_days <= 0:
            raise ValueError("dt_days must be positive.")
        if horizon_days <= 0:
            raise ValueError("horizon_days must be positive.")

        days = np.arange(0, horizon_days + dt_days, dt_days, dtype=float)
        if days[-1] > horizon_days:
            days[-1] = float(horizon_days)
        return cls(base_date=base_date, scen_days=days)

    @property
    def n_steps(self) -> int:
        return int(self.scen_days.size)

    def year_fractions(self, days_in_year: float) -> np.ndarray:
        return self.scen_days / float(days_in_year)

    def dates(self) -> list[date]:
        return [add_days(self.base_date, d) for d in self.scen_days]