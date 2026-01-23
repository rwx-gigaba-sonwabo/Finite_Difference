from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from xva_engine.reference_price import ReferencePrice
from xva_engine.config import DiscountingConfig


@dataclass(frozen=True)
class CommodityForward:
    """
    Simple commodity forward.

    IMPORTANT:
    - maturity_day is the CASHFLOW/SETTLEMENT day (days-from-value-date)
      so that discounting uses DF(t, cashflow_day).
    """
    maturity_day: int
    strike: float
    notional: float
    reference_price: ReferencePrice
    discounting: DiscountingConfig

    @staticmethod
    def discount_factor(t_day: float, T_day: float, days_in_year: float, r: float) -> float:
        tau = max((T_day - t_day) / float(days_in_year), 0.0)
        return float(np.exp(-r * tau))

    def mtm(
        self,
        scen_index: int,
        scen_day: float,
        scen_curve: torch.Tensor,
        tenor_days: np.ndarray,
        days_in_year: float,
    ) -> torch.Tensor:
        ref = self.reference_price.compute(
            scen_index=scen_index,
            scen_day=scen_day,
            scen_curve=scen_curve,
            tenor_days=tenor_days,
        )

        df = self.discount_factor(
            t_day=scen_day,
            T_day=float(self.maturity_day),
            days_in_year=float(days_in_year),
            r=float(self.discounting.rate),
        )

        return df * float(self.notional) * (ref - float(self.strike))
