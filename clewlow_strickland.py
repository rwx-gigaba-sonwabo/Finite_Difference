from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class CSParams:
    """
    Clewlow–Strickland parameters for the *historical / real-world* calibration:

    - alpha: mean reversion speed
    - sigma: "reversion volatility" (OU-style parameter used in RiskFlow)
    - mu: drift term used in historical simulation (RiskFlow stores it as 'Drift')
    """
    alpha: float
    sigma: float
    mu: float


class CSForwardCurveSimulator:
    """
    Mimics RiskFlow CSForwardPriceModel:

    For each delivery date T (curve node), simulate F(t, T) via:
        dF/F = mu dt + sigma exp(-alpha (T - t)) dW

    Implementation detail (RiskFlow):
    - build maturity-dependent cumulative variance 'var'
    - convert to per-step vol via sqrt(diff(var))
    - simulate:
        F = F0 * exp(drift + cumsum(vol * Z))
    where drift = mu * t - 0.5 * var  (historical mode)
    """
    def __init__(
        self,
        params: CSParams,
        days_in_year: float,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.params = params
        self.days_in_year = float(days_in_year)
        self.device = torch.device(device)
        self.dtype = dtype

    def _riskflow_dt_matrix_days(
        self,
        scen_days: np.ndarray,
        tenor_days: np.ndarray,
    ) -> np.ndarray:
        """
        RiskFlow dt construction (per tenor):
            tenor_rel = tenor_days (since base date = 0)
            delta = clip(tenor_rel, scen_days[i-1], scen_days[i]) - scen_days[i-1]
            dt = insert 0 at top then / DAYS_IN_YEAR

        This ensures variance stops accumulating once the curve node has matured.
        """
        tenor_rel = tenor_days.reshape(1, -1)  # (1, n_tenors)
        start = scen_days[:-1].reshape(-1, 1)
        end = scen_days[1:].reshape(-1, 1)
        delta = np.clip(tenor_rel, start, end) - start  # (n_steps-1, n_tenors)
        dt_days = np.insert(delta, 0, 0.0, axis=0)      # (n_steps, n_tenors)
        return dt_days

    def simulate(
        self,
        initial_curve: np.ndarray,
        tenor_days: np.ndarray,
        scen_days: np.ndarray,
        z: torch.Tensor,
        risk_neutral: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        initial_curve : np.ndarray
            F(0, T) at each curve node. Shape (n_tenors,)
        tenor_days : np.ndarray
            Delivery dates as days-from-base. Shape (n_tenors,)
        scen_days : np.ndarray
            Scenario grid in days-from-base. Shape (n_steps,)
        z : torch.Tensor
            Normal shocks for the single CS factor.
            Shape (n_steps, n_sims)  (RiskFlow uses one factor, broadcast across tenors)
        risk_neutral : bool
            If True, sets mu = 0.0 (RiskFlow implied mode style).

        Returns
        -------
        torch.Tensor
            Simulated forward curves:
            shape (n_steps, n_tenors, n_sims)
        """
        if initial_curve.ndim != 1:
            raise ValueError("initial_curve must be 1D: (n_tenors,).")
        if tenor_days.ndim != 1:
            raise ValueError("tenor_days must be 1D: (n_tenors,).")
        if scen_days.ndim != 1:
            raise ValueError("scen_days must be 1D: (n_steps,).")
        if z.ndim != 2 or z.shape[0] != scen_days.size:
            raise ValueError("z must be shape (n_steps, n_sims) aligned to scen_days.")

        n_steps = scen_days.size
        n_tenors = tenor_days.size
        n_sims = z.shape[1]

        # Build matrices consistent with RiskFlow
        dt_days = self._riskflow_dt_matrix_days(scen_days, tenor_days)  # (n_steps, n_tenors)
        dt = dt_days / self.days_in_year                               # years
        t_cum = dt.cumsum(axis=0)                                       # (n_steps, n_tenors)

        # tenors matrix: time-to-maturity at each scenario time
        # RiskFlow: tenors = (tenor_excel - (scen_day + base_excel)).clip(0,inf) / DAYS_IN_YEAR
        tenors = (tenor_days.reshape(1, -1) - scen_days.reshape(-1, 1)).clip(0.0, np.inf) / self.days_in_year

        alpha = float(self.params.alpha)
        sigma = float(self.params.sigma)
        mu = 0.0 if risk_neutral else float(self.params.mu)

        # Variance adjustment (OU-style)
        var_adj = (1.0 - np.exp(-2.0 * alpha * t_cum)) / (2.0 * alpha)
        var = (sigma**2) * np.exp(-2.0 * alpha * tenors) * var_adj  # cumulative log-variance

        # Per-step vol (RiskFlow: sqrt(diff(insert(var,0,0))))
        var0 = np.insert(var, 0, 0.0, axis=0)
        delta_var = np.diff(var0, axis=0)
        delta_var = np.maximum(delta_var, 0.0)
        vol = np.sqrt(delta_var)  # (n_steps, n_tenors)

        # Drift term (RiskFlow: mu*t - 0.5*var). In implied mode, mu=0 so drift=-0.5*var.
        drift = mu * t_cum - 0.5 * var  # (n_steps, n_tenors)

        # Torch tensors
        init = torch.as_tensor(initial_curve, dtype=self.dtype, device=self.device).view(1, n_tenors, 1)
        drift_t = torch.as_tensor(drift, dtype=self.dtype, device=self.device).view(n_steps, n_tenors, 1)
        vol_t = torch.as_tensor(vol, dtype=self.dtype, device=self.device).view(n_steps, n_tenors, 1)

        z_t = z.to(device=self.device, dtype=self.dtype).view(n_steps, 1, n_sims)
        z_portion = z_t * vol_t  # (n_steps, n_tenors, n_sims)

        return init * torch.exp(drift_t + torch.cumsum(z_portion, dim=0))
