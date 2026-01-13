# xva_engine/engine.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from xva_engine.config import SimulationConfig, CounterpartyConfig, DiscountingConfig
from xva_engine.rng import SobolNormalRng
from xva_engine.timegrid import TimeGrid
from xva_engine.models.clewlow_strickland import CSForwardCurveSimulator, CSParams
from xva_engine.products.commodity_forward import CommodityForward
from xva_engine.xva.cva import XvaCalculator, ExposureProfile


@dataclass(frozen=True)
class RunResult:
    times_days: np.ndarray
    mtm_paths: torch.Tensor
    exposure_profile: ExposureProfile
    cva: float


class CommodityXvaEngine:
    """
    A minimal RiskFlow-like XVA engine slice for commodities:
    - simulate CS forward curve (one factor)
    - compute reference prices
    - value forward contract on each profile date
    - compute EE/PFE and CVA
    """
    def __init__(
        self,
        sim_cfg: SimulationConfig,
        cs_params: CSParams,
        initial_curve: np.ndarray,
        tenor_days: np.ndarray,
        discounting: DiscountingConfig,
        counterparty: CounterpartyConfig,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.sim_cfg = sim_cfg
        self.cs_params = cs_params
        self.initial_curve = np.asarray(initial_curve, dtype=float)
        self.tenor_days = np.asarray(tenor_days, dtype=float)
        self.discounting = discounting
        self.counterparty = counterparty

        self.device = torch.device(device)
        self.dtype = dtype

        self.time_grid = TimeGrid.regular(sim_cfg.dt_days, sim_cfg.horizon_days)

        self.rng = SobolNormalRng(
            seed=sim_cfg.seed,
            fast_forward=sim_cfg.fast_forward,
            device=self.device,
            dtype=self.dtype,
        )
        self.simulator = CSForwardCurveSimulator(
            params=cs_params,
            days_in_year=sim_cfg.days_in_year,
            device=self.device,
            dtype=self.dtype,
        )

        # In RiskFlow, exposure is often handled in deflated terms; we provide the same option.
        self.xva = XvaCalculator(
            counterparty=counterparty,
            days_in_year=sim_cfg.days_in_year,
            pfe_quantile=0.95,
            discount_to_zero=True,
            flat_discount_rate=discounting.rate,
        )

    def run_forward_cva(
        self,
        trade: CommodityForward,
        risk_neutral: bool = False,
    ) -> RunResult:
        times_days = self.time_grid.scen_days
        n_steps = times_days.size
        n_sims = self.sim_cfg.num_sims

        # RiskFlow: one factor for CSForwardPriceModel
        z = self.rng.draw_normals(dimension=1, n=n_steps * n_sims).view(1, n_steps, n_sims)[0]
        # z shape: (n_steps, n_sims)

        curves = self.simulator.simulate(
            initial_curve=self.initial_curve,
            tenor_days=self.tenor_days,
            scen_days=times_days,
            z=z,
            risk_neutral=risk_neutral,
        )
        # curves shape: (n_steps, n_tenors, n_sims)

        # Revalue the trade at each time step
        mtm_paths = torch.zeros((n_steps, n_sims), dtype=self.dtype, device=self.device)
        for i, t_day in enumerate(times_days):
            scen_curve = curves[i, :, :]  # (n_tenors, n_sims)
            mtm_paths[i, :] = trade.mtm(
                scen_index=i,
                scen_day=float(t_day),
                scen_curve=scen_curve,
                tenor_days=self.tenor_days,
                days_in_year=self.sim_cfg.days_in_year,
            )

        profile = self.xva.build_exposure_profile(times_days=times_days, mtm_paths=mtm_paths)
        cva = self.xva.cva_from_ee(times_days=times_days, ee_star=profile.ee)

        return RunResult(
            times_days=times_days,
            mtm_paths=mtm_paths,
            exposure_profile=profile,
            cva=cva,
        )
