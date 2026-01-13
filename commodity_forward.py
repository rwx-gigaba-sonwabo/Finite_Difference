from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from xva_engine.reference_price import ReferencePrice
from xva_engine.config import DiscountingConfig


@dataclass(frozen=True)
class CommodityForward:
    """
    Simple commodity forward:

        Payoff at maturity: N * (RefPrice(T) - K)

    In simulation (for exposure profiles), at each scenario time t:
        MTM(t) = DF(t,T) * N * (RefPrice(t;T) - K)

    For a plain forward on a single delivery date, RefPrice(t;T) can just be F(t,T).
    For energy-style averaging, RefPrice is computed from a fixing schedule over forward samples.
    """
    maturity_day: int
    strike: float
    notional: float
    reference_price: ReferencePrice
    discounting: DiscountingConfig

    def discount_factor(self, t_day: float, T_day: float, days_in_year: float) -> float:
        tau = max((T_day - t_day) / days_in_year, 0.0)
        r = self.discounting.rate
        return float(np.exp(-r * tau))

    def mtm(
        self,
        scen_index: int,
        scen_day: float,
        scen_curve: torch.Tensor,
        tenor_days: np.ndarray,
        days_in_year: float,
    ) -> torch.Tensor:
        """
        Returns MTM per simulation: shape (n_sims,)
        """
        ref = self.reference_price.compute(
            scen_index=scen_index,
            scen_day=scen_day,
            scen_curve=scen_curve,
            tenor_days=tenor_days,
        )
        df = self.discount_factor(scen_day, float(self.maturity_day), days_in_year)
        return df * self.notional * (ref - float(self.strike))
