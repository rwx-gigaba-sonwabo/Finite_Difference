from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from xva_engine.config import CounterpartyConfig


@dataclass(frozen=True)
class ExposureProfile:
    """
    Exposure profile on a grid of times (days-from-value-date):
    - ee: expected positive exposure
    - pfe: q-quantile positive exposure
    """
    times_days: np.ndarray
    ee: np.ndarray
    pfe: np.ndarray


class XvaCalculator:
    """
    Minimal CVA/PFE calculator with deterministic hazard and optional deflation to t=0.
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
        return np.exp(-float(self.cp.hazard_rate) * t_years)

    def _df0(self, t_years: np.ndarray) -> np.ndarray:
        return np.exp(-float(self.flat_discount_rate) * t_years)

    def build_exposure_profile(self, times_days: np.ndarray, mtm_paths: torch.Tensor) -> ExposureProfile:
        times_days = np.asarray(times_days, dtype=float)
        if mtm_paths.ndim != 2 or mtm_paths.shape[0] != times_days.size:
            raise ValueError("mtm_paths must be (n_steps, n_sims) aligned to times_days.")

        times_years = times_days / self.days_in_year
        mtm = mtm_paths.detach().cpu().numpy()
        exposure = np.maximum(mtm, 0.0)

        if self.discount_to_zero:
            df0 = self._df0(times_years).reshape(-1, 1)
            exposure = exposure * df0

        ee = exposure.mean(axis=1)
        pfe = np.quantile(exposure, self.q, axis=1)
        return ExposureProfile(times_days=times_days, ee=ee, pfe=pfe)

    def cva_from_ee(self, times_days: np.ndarray, ee_star: np.ndarray) -> float:
        times_days = np.asarray(times_days, dtype=float)
        ee_star = np.asarray(ee_star, dtype=float)
        if times_days.size != ee_star.size:
            raise ValueError("times_days and ee_star must have same length.")

        t_years = times_days / self.days_in_year
        S = self._survival(t_years)
        lgd = 1.0 - float(self.cp.recovery)

        cva = 0.0
        for i in range(1, len(times_days)):
            avg_ee = 0.5 * (ee_star[i - 1] + ee_star[i])
            dp = S[i - 1] - S[i]
            cva += lgd * avg_ee * dp
        return float(cva)
