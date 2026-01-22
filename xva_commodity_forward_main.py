# main_example_gold_forward.py
from __future__ import annotations

import numpy as np

from xva_engine.config import SimulationConfig, CounterpartyConfig, DiscountingConfig, SamplingConvention
from xva_engine.models.clewlow_strickland import CSParams
from xva_engine.reference_price import FixingSchedule, ReferencePrice
from xva_engine.products.commodity_forward import CommodityForward
from xva_engine.engine import CommodityXvaEngine


def main() -> None:
    # ----------------------------
    # 1) Inputs you already have from historical calibration
    # ----------------------------
    # Example placeholders (replace with your fitted values):
    alpha = 1.2     # mean reversion speed
    sigma = 0.35    # reversion volatility
    mu = 0.02       # real-world drift (RiskFlow 'Drift' in historical mode)

    cs_params = CSParams(alpha=alpha, sigma=sigma, mu=mu)

    # ----------------------------
    # 2) Forward curve representation (F(0,T) + curve nodes)
    # ----------------------------
    # For a single gold forward settling in 90 days, you minimally need a node at T=90.
    # If you have multiple curve points, provide them here.
    tenor_days = np.array([90.0], dtype=float)
    initial_curve = np.array([2000.0], dtype=float)  # F(0,90) = 2000

    # ----------------------------
    # 3) Fixing / reference price rule
    # ----------------------------
    # For a plain forward settling on a single settlement date: "bullet" at maturity.
    fixing = FixingSchedule(
        start_day=90,
        end_day=90,
        convention=SamplingConvention.BULLET,
        offset_days=0,
    )
    ref_price = ReferencePrice(
        fixing_schedule=fixing,
        delivery_days=tenor_days,  # maps the fixing to the nearest delivery node
        realised_fixings=None,     # you can supply realised fixings as {day: value}
    )

    # ----------------------------
    # 4) Trade definition
    # ----------------------------
    strike = 1995.0
    notional = 1_000_000.0

    discounting = DiscountingConfig(rate=0.05)
    trade = CommodityForward(
        maturity_day=90,
        strike=strike,
        notional=notional,
        reference_price=ref_price,
        discounting=discounting,
    )

    # ----------------------------
    # 5) Credit + Simulation config (match RiskFlow knobs here)
    # ----------------------------
    sim_cfg = SimulationConfig(
        num_sims=50_000,
        seed=7,
        fast_forward=0,
        dt_days=3,          # example: 3-day grid step (your question)
        horizon_days=365,
        days_in_year=365.0,
    )

    counterparty = CounterpartyConfig(hazard_rate=0.02, recovery=0.4)

    engine = CommodityXvaEngine(
        sim_cfg=sim_cfg,
        cs_params=cs_params,
        initial_curve=initial_curve,
        tenor_days=tenor_days,
        discounting=discounting,
        counterparty=counterparty,
        device="cpu",
    )

    # For CVA pricing, you will typically want risk_neutral=True (mu=0).
    # For historical PFE-style risk metrics, risk_neutral=False uses your fitted mu.
    result = engine.run_forward_cva(trade=trade, risk_neutral=True)

    print(f"CVA (discounted EE*, flat hazard): {result.cva:,.2f}")
    print("EE* first 5 points:", result.exposure_profile.ee[:5])
    print("PFE(95%) first 5 points:", result.exposure_profile.pfe[:5])


if __name__ == "__main__":
    main()
