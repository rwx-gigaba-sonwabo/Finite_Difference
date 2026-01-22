from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from xva_engine.config import CounterpartyConfig
from xva_engine.utils.dates import add_days


@dataclass(frozen=True)
class ExposureProfile:
    times_days: np.ndarray
    times_dates: Optional[list[date]]
    ee: np.ndarray
    pfe: np.ndarray
    ee_star: Optional[np.ndarray] = None
    pfe_star: Optional[np.ndarray] = None


class XvaCalculator:
    def __init__(
        self,
        *,
        counterparty: CounterpartyConfig,
        days_in_year: float = 365.0,
        discount_rate: float = 0.0,
    ) -> None:
        self.cp = counterparty
        self.days_in_year = float(days_in_year)
        self.discount_rate = float(discount_rate)

    def _survival(self, t_years: np.ndarray) -> np.ndarray:
        return np.exp(-float(self.cp.hazard_rate) * t_years)

    def _df0t(self, t_years: np.ndarray) -> np.ndarray:
        return np.exp(-float(self.discount_rate) * t_years)

    def build_exposure_profile(
        self,
        *,
        base_date: date,
        times_days: np.ndarray,
        mtm_paths: torch.Tensor,   # (n_times, n_sims)
        pfe_q: float = 0.95,
        discount_to_zero: bool = True,
    ) -> ExposureProfile:
        times_days = np.asarray(times_days, dtype=float)
        if mtm_paths.ndim != 2 or mtm_paths.shape[0] != times_days.size:
            raise ValueError("mtm_paths must be (n_times, n_sims) and match times_days length.")

        e = torch.clamp(mtm_paths, min=0.0)
        ee = e.mean(dim=1).cpu().numpy()
        pfe = torch.quantile(e, q=float(pfe_q), dim=1).cpu().numpy()

        ee_star = None
        pfe_star = None
        if discount_to_zero:
            t_years = times_days / self.days_in_year
            df = self._df0t(t_years)
            ee_star = ee * df
            pfe_star = pfe * df

        times_dates = [add_days(base_date, d) for d in times_days]
        return ExposureProfile(times_days, times_dates, ee, pfe, ee_star, pfe_star)

    def cva_from_ee(self, *, times_days: np.ndarray, ee_star: np.ndarray) -> float:
        times_days = np.asarray(times_days, dtype=float)
        ee_star = np.asarray(ee_star, dtype=float)
        if times_days.size != ee_star.size:
            raise ValueError("times_days and ee_star must have same length.")

        t_years = times_days / self.days_in_year
        S = self._survival(t_years)
        lgd = 1.0 - float(self.cp.recovery)

        cva = 0.0
        for i in range(1, times_days.size):
            avg_ee = 0.5 * (ee_star[i - 1] + ee_star[i])
            dp = S[i - 1] - S[i]
            cva += lgd * avg_ee * dp
        return float(cva)

    @staticmethod
    def plot_exposure(profile: ExposureProfile, *, title: str = "Exposure profile", discounted: bool = False) -> None:
        x = profile.times_dates if profile.times_dates is not None else profile.times_days

        if discounted:
            if profile.ee_star is None or profile.pfe_star is None:
                raise ValueError("Discounted series not available.")
            ee, pfe = profile.ee_star, profile.pfe_star
            suffix = " (discounted)"
        else:
            ee, pfe = profile.ee, profile.pfe
            suffix = ""

        plt.figure()
        plt.plot(x, ee, label="EE" + suffix)
        plt.plot(x, pfe, label="PFE" + suffix)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
