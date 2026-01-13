from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimeGrid:
    """
    Scenario time grid in DAYS from base date.

    RiskFlow uses a "scenario time grid" in day units, and CSForwardPriceModel
    converts to year fractions using DAYS_IN_YEAR.
    """
    scen_days: np.ndarray  # shape (m,)

    @classmethod
    def regular(cls, dt_days: int, horizon_days: int) -> "TimeGrid":
        if dt_days <= 0:
            raise ValueError("dt_days must be positive.")
        if horizon_days <= 0:
            raise ValueError("horizon_days must be positive.")
        days = np.arange(0, horizon_days + dt_days, dt_days, dtype=float)
        if days[-1] > horizon_days:
            days[-1] = float(horizon_days)
        return cls(scen_days=days)

    @property
    def n_steps(self) -> int:
        return int(self.scen_days.size)

    def year_fractions(self, days_in_year: float) -> np.ndarray:
        return self.scen_days / days_in_year
