from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from xva_engine.config import CounterpartyConfig


@dataclass(frozen=True)
class ExposureProfile:
    """
    Exposure profile on a grid of times:
    - ee: Expected positive exposure at each time (discounted-to-0 or not, depending on convention)
    - pfe: q-quantile exposure at each time
    """
    times_days: np.ndarray
    ee: np.ndarray
    pfe: np.ndarray


class XvaCalculator:
    """
    Minimal CVA/PFE engine for a single counterparty.

    - exposure(t) per scenario = max(MTM(t), 0)
    - EE(t) = average across scenarios
    - PFE_q(t) = quantile across scenarios
    - CVA = LGD * sum( 0.5*(EE*(t_i)+EE*(t_{i-1})) * (S(t_{i-1})-S(t_i)) )
      where EE* is discounted-to-0 exposure if you choose that convention.

    This matches the discrete estimator described in the RiskFlow Credit_Monte_Carlo docs.
    """
    def __init__(
        self,
        counterparty: CounterpartyConfig,
        days_in_year: float,
        pfe_quantile: float = 0.95,
        discount_to_zero: bool = True,
        flat_discount_rate: float = 0.0,
    ) -> None:
        self.cp = counterparty
        self.days_in_year = float(days_in_year)
        self.q = float(pfe_quantile)
        self.discount_to_zero = bool(discount_to_zero)
        self.flat_discount_rate = float(flat_discount_rate)

    def _survival(self, t_years: np.ndarray) -> np.ndarray:
        h = self.cp.hazard_rate
        return np.exp(-h * t_years)

    def _df0(self, t_years: np.ndarray) -> np.ndarray:
        r = self.flat_discount_rate
        return np.exp(-r * t_years)

    def build_exposure_profile(
        self,
        times_days: np.ndarray,
        mtm_paths: torch.Tensor,
    ) -> ExposureProfile:
        """
        Parameters
        ----------
        times_days : np.ndarray
            scenario grid in days, shape (n_steps,)
        mtm_paths : torch.Tensor
            MTM(t) per time and scenario, shape (n_steps, n_sims)

        Returns
        -------
        ExposureProfile
        """
        times_years = times_days / self.days_in_year
        mtm = mtm_paths.detach().cpu().numpy()  # (n_steps, n_sims)

        exposure = np.maximum(mtm, 0.0)  # positive exposure only

        if self.discount_to_zero:
            df0 = self._df0(times_years).reshape(-1, 1)
            exposure = exposure * df0  # discounted exposure (E*)

        ee = exposure.mean(axis=1)
        pfe = np.quantile(exposure, self.q, axis=1)

        return ExposureProfile(times_days=times_days, ee=ee, pfe=pfe)

    def cva_from_ee(self, times_days: np.ndarray, ee_star: np.ndarray) -> float:
        """
        Discrete CVA integral with deterministic hazard:
            CVA = LGD * sum_i 0.5*(EE_{i-1}+EE_i) * (S_{i-1}-S_i)
        """
        times_years = times_days / self.days_in_year
        S = self._survival(times_years)
        lgd = 1.0 - self.cp.recovery

        cva = 0.0
        for i in range(1, len(times_days)):
            avg_ee = 0.5 * (ee_star[i - 1] + ee_star[i])
            dp = S[i - 1] - S[i]
            cva += lgd * avg_ee * dp

        return float(cva)
