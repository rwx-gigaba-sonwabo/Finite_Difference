# gbm_asset_price_diagnostic.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def norm_icdf(u: torch.Tensor) -> torch.Tensor:
    """
    RiskFlow-style inverse normal CDF:
        sqrt(2) * erfinv(2u - 1)
    """
    return 1.4142135623730951 * torch.erfinv(2.0 * u - 1.0)


def sobol_normals_time_as_dimension(
    n_steps: int,
    n_sims: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    fast_forward: int = 0,
) -> torch.Tensor:
    """
    Correct Sobol usage for time stepping:
      - dimension = n_steps
      - draw = n_sims
      - return Z with shape (n_steps, n_sims)
    """
    engine = torch.quasirandom.SobolEngine(dimension=n_steps, scramble=True, seed=seed)
    if fast_forward > 0:
        engine.fast_forward(fast_forward)

    u = engine.draw(n_sims, dtype=dtype).to(device)  # (n_sims, n_steps)
    eps = torch.finfo(u.dtype).eps
    u = 0.5 + (1.0 - eps) * (u - 0.5)

    z = norm_icdf(u)  # (n_sims, n_steps)
    return z.transpose(0, 1).contiguous()  # (n_steps, n_sims)


@dataclass(frozen=True)
class GBMParams:
    """
    Historical / real-world GBM parameters (RiskFlow-style):
    - mu: arithmetic drift in dS/S = mu dt + sigma dW
    - sigma: volatility
    """
    mu: float
    sigma: float


class GBMSimulator:
    """
    GBM simulation consistent with RiskFlow-style parameter usage:

      dS(t)/S(t) = mu dt + sigma dW(t)

    Discrete exact scheme:
      S_{t+dt} = S_t * exp((mu - 0.5*sigma^2)dt + sigma*sqrt(dt)*Z)

    Notes:
    - If you want risk-neutral pricing, set mu = r - q (or 0 for FX forward measure),
      but for historical calibration, mu is taken from data.
    """

    def __init__(
        self,
        params: GBMParams,
        days_in_year: float = 365.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.params = params
        self.days_in_year = float(days_in_year)
        self.device = torch.device(device)
        self.dtype = dtype

    def simulate(
        self,
        s0: float,
        scen_days: np.ndarray,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        s0:
            Initial spot value S(0).
        scen_days:
            Scenario grid in days-from-base, shape (n_steps,)
        z:
            Standard normals, shape (n_steps, n_sims)

        Returns
        -------
        torch.Tensor
            Simulated spot paths, shape (n_steps, n_sims)
        """
        scen_days = np.asarray(scen_days, dtype=float)
        if scen_days.ndim != 1:
            raise ValueError("scen_days must be 1D.")
        if z.ndim != 2 or z.shape[0] != scen_days.size:
            raise ValueError("z must be (n_steps, n_sims) aligned to scen_days.")

        mu = float(self.params.mu)
        sigma = float(self.params.sigma)

        t = scen_days / self.days_in_year  # years
        dt = np.diff(t, prepend=t[0])      # dt[0]=0

        dt_t = torch.as_tensor(dt, dtype=self.dtype, device=self.device).view(-1, 1)
        z_t = z.to(self.device, self.dtype)

        drift = (mu - 0.5 * sigma * sigma) * dt_t
        diff = sigma * torch.sqrt(torch.clamp(dt_t, min=0.0)) * z_t

        log_increments = drift + diff
        log_path = torch.cumsum(log_increments, dim=0)

        return float(s0) * torch.exp(log_path)

    @staticmethod
    def sanity_check_z(z: torch.Tensor) -> None:
        """
        Quick Sobol/normal diagnostics.
        """
        m = z.mean(dim=1).detach().cpu().numpy()
        s = z.std(dim=1, unbiased=False).detach().cpu().numpy()
        print("Z mean range:", float(m.min()), float(m.max()))
        print("Z std  range:", float(s.min()), float(s.max()))
        if np.max(np.abs(m)) > 5e-3:
            print("WARNING: per-step Z mean bias detected (bad Sobol shape/reshaping).")

    def sanity_check_mean(
        self,
        paths: torch.Tensor,
        s0: float,
        scen_days: np.ndarray,
        tol: float = 0.02,
    ) -> None:
        """
        Check the mean path against the theoretical expectation under GBM:
            E[S(t)] = S0 * exp(mu * t)

        This is the cleanest check to catch drift/RNG bias issues.
        """
        scen_days = np.asarray(scen_days, dtype=float)
        t = scen_days / self.days_in_year

        empirical = paths.mean(dim=1).detach().cpu().numpy()
        target = float(s0) * np.exp(float(self.params.mu) * t)

        rel_err = (empirical - target) / np.maximum(target, 1e-12)

        max_abs = float(np.max(np.abs(rel_err)))
        print(f"Max |relative mean error|: {max_abs:.4%}")
        if max_abs > tol:
            print("WARNING: Mean check failing (likely RNG bias or unit mismatch).")

    def sanity_check_variance(
        self,
        paths: torch.Tensor,
        s0: float,
        scen_days: np.ndarray,
        tol_abs: float = 5e-3,
    ) -> None:
        """
        Check Var[log(S(t)/S0)] against sigma^2 * t (GBM log variance).
        """
        scen_days = np.asarray(scen_days, dtype=float)
        t = scen_days / self.days_in_year

        log_ratio = torch.log(paths / float(s0))
        emp_var = log_ratio.var(dim=1, unbiased=False).detach().cpu().numpy()
        target = (float(self.params.sigma) ** 2) * t

        diff = emp_var - target
        max_abs = float(np.max(np.abs(diff)))
        print(f"Max abs log-variance error: {max_abs:.6f}")
        if max_abs > tol_abs:
            print("WARNING: Variance check failing (dt scaling or sigma units mismatch).")


def plot_paths(
    scen_days: np.ndarray,
    paths: torch.Tensor,
    n_paths: int = 100,
    use_years: bool = True,
    days_in_year: float = 365.0,
    title: str = "GBM spot simulation (diagnostic)",
) -> None:
    import matplotlib.pyplot as plt

    x = scen_days.astype(float)
    if use_years:
        x = x / float(days_in_year)

    s = paths.detach().cpu().numpy()
    n_steps, n_sims = s.shape

    idx = np.random.default_rng(123).choice(n_sims, size=min(n_paths, n_sims), replace=False)

    plt.figure()
    plt.plot(x, s[:, idx], linewidth=0.8)
    plt.plot(x, s.mean(axis=1), linewidth=2.0, label="Mean")
    plt.plot(x, np.quantile(s, 0.05, axis=1), linestyle="--", label="Q05")
    plt.plot(x, np.quantile(s, 0.95, axis=1), linestyle="--", label="Q95")
    plt.title(title)
    plt.xlabel("Time (years)" if use_years else "Time (days)")
    plt.ylabel("Spot")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    """
    Diagnostic runner for GBM (RiskFlow-style FX spot calibration/simulation).

    Replace MU and SIGMA with the values you compute from historical calibration
    (e.g., from calc_statistics on log prices / log returns).

    This script:
      - builds a scenario grid in days
      - draws Sobol-based normals with time steps as Sobol dimensions
      - simulates GBM spot paths
      - runs sanity checks (mean and variance)
      - plots paths with mean and 5-95% bands
    """

    # -----------------------
    # 1) Inputs (replace with your calibrated values)
    # -----------------------
    MU = 0.04       # annual drift in dS/S
    SIGMA = 0.12    # annual vol

    S0 = 18.50      # initial spot (e.g., FX spot)

    DAYS_IN_YEAR = 365.0

    DT_DAYS = 1
    HORIZON_DAYS = 365

    NUM_SIMS = 50_000
    SOBOL_SEED = 7
    FAST_FORWARD = 0

    # -----------------------
    # 2) Scenario grid
    # -----------------------
    scen_days = np.arange(0, HORIZON_DAYS + DT_DAYS, DT_DAYS, dtype=float)
    if scen_days[-1] > HORIZON_DAYS:
        scen_days[-1] = float(HORIZON_DAYS)
    n_steps = scen_days.size

    # -----------------------
    # 3) RNG
    # -----------------------
    device = torch.device("cpu")
    dtype = torch.float64

    z = sobol_normals_time_as_dimension(
        n_steps=n_steps,
        n_sims=NUM_SIMS,
        seed=SOBOL_SEED,
        device=device,
        dtype=dtype,
        fast_forward=FAST_FORWARD,
    )

    GBMSimulator.sanity_check_z(z)

    # -----------------------
    # 4) Simulate
    # -----------------------
    sim = GBMSimulator(params=GBMParams(mu=MU, sigma=SIGMA), days_in_year=DAYS_IN_YEAR, device=device, dtype=dtype)
    paths = sim.simulate(s0=S0, scen_days=scen_days, z=z)

    # -----------------------
    # 5) Sanity checks
    # -----------------------
    sim.sanity_check_mean(paths=paths, s0=S0, scen_days=scen_days, tol=0.01)
    sim.sanity_check_variance(paths=paths, s0=S0, scen_days=scen_days, tol_abs=1e-2)

    # -----------------------
    # 6) Plot
    # -----------------------
    plot_paths(
        scen_days=scen_days,
        paths=paths,
        n_paths=120,
        use_years=True,
        days_in_year=DAYS_IN_YEAR,
        title="GBM Spot Simulation (RiskFlow-style diagnostic)",
    )
