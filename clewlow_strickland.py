# xva_engine/models/clewlow_strickland.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

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

        dt_days = self._riskflow_dt_matrix_days(scen_days, tenor_days)  # (n_steps, n_tenors)
        dt = dt_days / self.days_in_year                               # years
        t_cum = dt.cumsum(axis=0)                                       # (n_steps, n_tenors)

        tenors = (tenor_days.reshape(1, -1) - scen_days.reshape(-1, 1)).clip(0.0, np.inf)
        tenors = tenors / self.days_in_year  # (n_steps, n_tenors)

        alpha = float(self.params.alpha)
        sigma = float(self.params.sigma)
        mu = 0.0 if risk_neutral else float(self.params.mu)

        var_adj = (1.0 - np.exp(-2.0 * alpha * t_cum)) / (2.0 * alpha)
        var = (sigma**2) * np.exp(-2.0 * alpha * tenors) * var_adj

        var0 = np.insert(var, 0, 0.0, axis=0)
        delta_var = np.diff(var0, axis=0)
        delta_var = np.maximum(delta_var, 0.0)
        vol = np.sqrt(delta_var)

        drift = mu * t_cum - 0.5 * var

        init = torch.as_tensor(initial_curve, dtype=self.dtype, device=self.device).view(1, n_tenors, 1)
        drift_t = torch.as_tensor(drift, dtype=self.dtype, device=self.device).view(n_steps, n_tenors, 1)
        vol_t = torch.as_tensor(vol, dtype=self.dtype, device=self.device).view(n_steps, n_tenors, 1)

        z_t = z.to(device=self.device, dtype=self.dtype).view(n_steps, 1, n_sims)
        z_portion = z_t * vol_t

        return init * torch.exp(drift_t + torch.cumsum(z_portion, dim=0))


def _norm_icdf(u: torch.Tensor) -> torch.Tensor:
    """
    RiskFlow-style normal inverse CDF:
        sqrt(2) * erfinv(2u - 1)
    """
    return 1.4142135623730951 * torch.erfinv(2.0 * u - 1.0)


def _draw_sobol_normals(
    *,
    n_steps: int,
    n_sims: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    fast_forward: int = 0,
) -> torch.Tensor:
    """
    Standalone Sobol->Normal generator for diagnosis runs.
    Shape: (n_steps, n_sims) for 1 factor.
    """
    engine = torch.quasirandom.SobolEngine(dimension=1, scramble=True, seed=seed)
    if fast_forward > 0:
        engine.fast_forward(fast_forward)

    u = engine.draw(n_steps * n_sims, dtype=dtype).to(device)  # (n_steps*n_sims, 1)
    eps = torch.finfo(u.dtype).eps
    u = 0.5 + (1.0 - eps) * (u - 0.5)  # avoid exact 0/1

    z = _norm_icdf(u)  # (n_steps*n_sims, 1)
    z = z.view(n_steps, n_sims)
    return z


def _plot_curves(
    times_days: np.ndarray,
    curves: torch.Tensor,
    tenor_days: np.ndarray,
    tenor_index: int,
    n_paths: int,
    seed: int,
    use_years: bool,
    days_in_year: float,
    title: str,
) -> None:
    """
    Lightweight plotting util (kept inside file intentionally for "diagnostic-only" usage).
    """
    import matplotlib.pyplot as plt  # local import for optional dependency

    if curves.ndim != 3:
        raise ValueError("curves must be (n_steps, n_tenors, n_sims).")
    n_steps, n_tenors, n_sims = curves.shape
    if not (0 <= tenor_index < n_tenors):
        raise ValueError(f"tenor_index must be in [0, {n_tenors-1}].")

    x = times_days.astype(float)
    if use_years:
        x = x / float(days_in_year)

    f = curves[:, tenor_index, :].detach().cpu().numpy()  # (n_steps, n_sims)

    rng = np.random.default_rng(seed)
    n_paths = int(min(n_paths, n_sims))
    idx = rng.choice(n_sims, size=n_paths, replace=False)

    plt.figure()
    plt.plot(x, f[:, idx], linewidth=0.8)

    mean = f.mean(axis=1)
    plt.plot(x, mean, linewidth=2.0, label="Mean")

    lo = np.quantile(f, 0.05, axis=1)
    hi = np.quantile(f, 0.95, axis=1)
    plt.plot(x, lo, linestyle="--", linewidth=1.5, label="Q05")
    plt.plot(x, hi, linestyle="--", linewidth=1.5, label="Q95")

    t_label = "years" if use_years else "days"
    tenor_label = f"{tenor_days[tenor_index]:.0f}d"
    plt.title(title or f"CS Simulated Forwards (tenor={tenor_label})")
    plt.xlabel(f"Time ({t_label})")
    plt.ylabel("Forward price")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    """
    Diagnostic runner.

    Purpose:
    - mimic RiskFlow CS forward-curve simulation using your calibrated (alpha, sigma, mu)
    - generate Sobol-based normal shocks with a seed
    - plot a subset of simulated forward paths for a chosen delivery node

    Usage example:
        python xva_engine/models/clewlow_strickland.py
    """

    # -----------------------
    # 1) User inputs (replace with your calibrated parameters)
    # -----------------------
    ALPHA = 1.20   # mean reversion speed
    SIGMA = 0.35   # reversion volatility
    MU = 0.02      # historical drift (set 0.0 for risk-neutral)

    DAYS_IN_YEAR = 365.0

    # Scenario grid
    DT_DAYS = 3
    HORIZON_DAYS = 365

    # Simulation config (RiskFlow-like)
    NUM_SIMS = 25_000
    SOBOL_SEED = 7
    FAST_FORWARD = 0

    # Forward curve nodes (delivery dates in days from base)
    # Use multiple nodes to visualise Samuelson damping across maturities.
    TENOR_DAYS = np.array([30.0, 90.0, 180.0, 365.0], dtype=float)

    # Initial forward curve F(0,T) at those nodes (toy example)
    INITIAL_CURVE = np.array([2000.0, 2005.0, 2015.0, 2030.0], dtype=float)

    # Plot controls
    TENOR_INDEX_TO_PLOT = 1     # pick a node, e.g. 90d
    N_PATHS_TO_PLOT = 60
    PLOT_X_IN_YEARS = True

    # Historical vs risk-neutral toggle
    RISK_NEUTRAL = False  # True -> mu=0, like implied mode

    # -----------------------
    # 2) Build time grid
    # -----------------------
    scen_days = np.arange(0, HORIZON_DAYS + DT_DAYS, DT_DAYS, dtype=float)
    if scen_days[-1] > HORIZON_DAYS:
        scen_days[-1] = float(HORIZON_DAYS)
    n_steps = scen_days.size

    # -----------------------
    # 3) Generate 1-factor Sobol normals
    # -----------------------
    device = torch.device("cpu")
    dtype = torch.float64

    z = _draw_sobol_normals(
        n_steps=n_steps,
        n_sims=NUM_SIMS,
        seed=SOBOL_SEED,
        fast_forward=FAST_FORWARD,
        device=device,
        dtype=dtype,
    )  # (n_steps, n_sims)

    # -----------------------
    # 4) Simulate forward curves with RiskFlow-style variance increments
    # -----------------------
    params = CSParams(alpha=ALPHA, sigma=SIGMA, mu=MU)
    sim = CSForwardCurveSimulator(params=params, days_in_year=DAYS_IN_YEAR, device=device, dtype=dtype)

    curves = sim.simulate(
        initial_curve=INITIAL_CURVE,
        tenor_days=TENOR_DAYS,
        scen_days=scen_days,
        z=z,
        risk_neutral=RISK_NEUTRAL,
    )  # (n_steps, n_tenors, n_sims)

    # -----------------------
    # 5) Plot selected tenor paths
    # -----------------------
    title = "Clewlow–Strickland Forward Simulation (diagnostic)"
    _plot_curves(
        times_days=scen_days,
        curves=curves,
        tenor_days=TENOR_DAYS,
        tenor_index=TENOR_INDEX_TO_PLOT,
        n_paths=N_PATHS_TO_PLOT,
        seed=123,
        use_years=PLOT_X_IN_YEARS,
        days_in_year=DAYS_IN_YEAR,
        title=title,
    )

    # Optional: quick printout for sanity
    # Mean terminal forward by tenor:
    terminal = curves[-1, :, :].mean(dim=1).detach().cpu().numpy()
    print("Mean terminal forward by tenor node:", dict(zip(TENOR_DAYS.astype(int), terminal.round(4))))
