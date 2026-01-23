"""
main_run_commodity_forward_cva.py

End-to-end runner:
- Define trade dates as real datetime.date objects (value date, fixing schedule, maturity).
- Build an initial commodity forward curve F(0,T) on tenor (delivery) dates.
- Simulate forward curves with Clewlow–Strickland (RiskFlow-like one-factor).
- Revalue the commodity forward on each scenario date -> MTM paths.
- Build exposure profiles (EE/PFE) and compute CVA.
- Plot:
    (1) Fixing schedule + value date + maturity date (as datetimes)
    (2) EE/PFE profiles (discounted-to-0 and undiscounted)
    (3) Optional: initial forward curve

Assumes your package layout matches the imports used in your uploaded files:
    xva_engine.engine
    xva_engine.config
    xva_engine.reference_price
    xva_engine.products.commodity_forward
    xva_engine.models.clewlow_strickland
    xva_engine.xva.cva

If your module names differ (e.g., time_grid vs timegrid), keep the package imports
aligned with what xva_engine/engine.py currently uses.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, timedelta
import calendar
import numpy as np
import matplotlib.pyplot as plt

# --- Your project imports ---
from xva_engine.config import SimulationConfig, CounterpartyConfig, DiscountingConfig, SamplingConvention
from xva_engine.models.clewlow_strickland import CSParams
from xva_engine.reference_price import FixingSchedule, ReferencePrice
from xva_engine.products.commodity_forward import CommodityForward
from xva_engine.engine import CommodityXvaEngine
from xva_engine.xva.cva import XvaCalculator


# ----------------------------
# Date helpers (real-life style)
# ----------------------------
def add_months(d: date, n: int) -> date:
    """Add n months to a date (keeps day-of-month if possible, otherwise clamps)."""
    y = d.year + (d.month - 1 + n) // 12
    m = (d.month - 1 + n) % 12 + 1
    last = calendar.monthrange(y, m)[1]
    day = min(d.day, last)
    return date(y, m, day)


def to_day(base: date, d: date) -> int:
    """Integer day offset from base date."""
    return int((d - base).days)


def day_to_date(base: date, day: float) -> date:
    return base + timedelta(days=int(round(day)))


def make_monthly_tenor_dates(value_date: date, n_months: int, anchor_day: int = 1) -> list[date]:
    """
    Build monthly tenor (delivery) dates. By default, uses the 1st of each month.
    """
    # anchor to a stable day-of-month (common for contract month nodes)
    anchored = date(value_date.year, value_date.month, min(anchor_day, calendar.monthrange(value_date.year, value_date.month)[1]))
    return [add_months(anchored, i) for i in range(1, n_months + 1)]


# ----------------------------
# Plotting helpers
# ----------------------------
def plot_fixing_schedule(
    *,
    value_date: date,
    maturity_date: date,
    fixing_dates: list[date],
    title: str = "Fixing schedule (datetime)"
) -> None:
    y = np.zeros(len(fixing_dates), dtype=float)
    plt.figure()
    plt.plot(fixing_dates, y, marker="o", linestyle="None", label="fixings")
    plt.axvline(value_date, linestyle=":", label="value date")
    plt.axvline(maturity_date, linestyle="--", label="maturity")
    plt.yticks([])
    plt.title(title)
    plt.legend()
    plt.tight_layout()


def plot_exposure_profiles(
    *,
    base_date: date,
    times_days: np.ndarray,
    profile_discounted,
    profile_undiscounted,
    title: str = "Exposure profiles"
) -> None:
    times_dates = [day_to_date(base_date, d) for d in times_days]

    plt.figure()
    plt.plot(times_dates, profile_undiscounted.ee, label="EE (undiscounted)")
    plt.plot(times_dates, profile_undiscounted.pfe, label="PFE (undiscounted)")
    plt.plot(times_dates, profile_discounted.ee, label="EE* (discounted-to-0)")
    plt.plot(times_dates, profile_discounted.pfe, label="PFE* (discounted-to-0)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()


def plot_initial_forward_curve(
    *,
    value_date: date,
    tenor_dates: list[date],
    initial_curve: np.ndarray,
    title: str = "Initial forward curve F(0,T)"
) -> None:
    plt.figure()
    plt.plot(tenor_dates, initial_curve, marker="o")
    plt.axvline(value_date, linestyle=":", label="value date")
    plt.title(title)
    plt.legend()
    plt.tight_layout()


# ----------------------------
# Main runner
# ----------------------------
def main() -> None:
    # ----------------------------
    # 1) Trade +Nor / market inputs (DATETIME objects)
    # ----------------------------
    value_date = date(2026, 1, 22)               # "today" / valuation date
    maturity_date = add_months(value_date, 6)    # 6M forward (example)

    # Choose a fixing convention
    # - BULLET: single fixing at end of period (typically maturity)
    # - DAILY/WEEKLY/MONTHLY: averaging over [fix_start, fix_end]
    convention = SamplingConvention.BULLET

    if convention == SamplingConvention.BULLET:
        fix_start_date = maturity_date
        fix_end_date = maturity_date
    else:
        # Example: average over the last calendar month before maturity (toy)
        fix_end_date = maturity_date
        fix_start_date = add_months(maturity_date, -1)

    # Delivery (tenor) dates for the curve nodes (DATETIME objects)
    tenor_dates = make_monthly_tenor_dates(value_date, n_months=24, anchor_day=1)

    # Day offsets (what your model actually uses internally)
    maturity_day = to_day(value_date, maturity_date)
    fix_start_day = to_day(value_date, fix_start_date)
    fix_end_day = to_day(value_date, fix_end_date)
    tenor_days = np.asarray([to_day(value_date, d) for d in tenor_dates], dtype=float)

    # Ensure horizon covers the latest relevant date
    horizon_days = int(max(maturity_day, fix_end_day, float(tenor_days.max())))

    # ----------------------------
    # 2) Build initial forward curve F(0,T) at curve nodes
    # ----------------------------
    # Replace this with your actual curve bootstrapping / market curve ingestion.
    # Here: a simple contango-ish term structure:
    spot = 80.0
    contango_rate = 0.06
    tau = tenor_days / 365.0
    initial_curve = spot * np.exp(contango_rate * tau)  # shape (n_tenors,)

    # ----------------------------
    # 3) Build fixing schedule + reference price object
    # ----------------------------
    fixing_schedule = FixingSchedule(
        start_day=fix_start_day,
        end_day=fix_end_day,
        convention=convention,
        offset_days=0,
    )

    # delivery_days: mapping basis for sampling simulated curve
    # Most simply: use curve node tenor days as delivery nodes.
    delivery_days = tenor_days.copy()

    # Realised fixings: dict[day -> price]. Usually empty unless value date is inside fixing window.
    # Example: if value_date already past some fixing dates, you’d populate them.
    realised_fixings: dict[float, float] = {}

    reference_price = ReferencePrice(
        fixing_schedule=fixing_schedule,
        delivery_days=delivery_days,
        realised_fixings=realised_fixings,
    )

    # ----------------------------
    # 4) Trade definition (forward)
    # ----------------------------
    strike = float(np.interp(maturity_day, tenor_days, initial_curve))  # ATM-ish strike
    notional = 1_000_000.0

    discounting = DiscountingConfig(rate=0.09)  # flat CC rate (replace with curve later)

    trade = CommodityForward(
        maturity_day=int(maturity_day),
        strike=strike,
        notional=notional,
        reference_price=reference_price,
        discounting=discounting,
    )

    # ----------------------------
    # 5) Simulation + credit configs
    # ----------------------------
    sim_cfg = SimulationConfig(
        num_sims=20_000,     # increase as needed
        seed=7,
        fast_forward=0,
        dt_days=7,           # weekly profiles
        horizon_days=horizon_days,
        days_in_year=365.0,
    )

    # Clewlow–Strickland params
    # NOTE: for XVA exposures, you usually want risk_neutral=True (engine will use mu=0).
    cs_params = CSParams(
        alpha=1.2,   # mean reversion speed (Samuelson effect)
        sigma=0.35,  # vol scale
        mu=0.02,     # historical drift knob (ignored in RN mode)
    )

    counterparty = CounterpartyConfig(
        hazard_rate=0.03,  # flat hazard (3% pa)
        recovery=0.4,
    )

    # ----------------------------
    # 6) Run engine
    # ----------------------------
    engine = CommodityXvaEngine(
        sim_cfg=sim_cfg,
        cs_params=cs_params,
        initial_curve=initial_curve,
        tenor_days=tenor_days,
        discounting=discounting,
        counterparty=counterparty,
        device="cpu",
    )

    # Risk-neutral exposure run (recommended for pricing/XVA)
    result_rn = engine.run_forward_cva(trade=trade, risk_neutral=True)

    # ----------------------------
    # 7) Build BOTH discounted and undiscounted exposure profiles for plotting
    # ----------------------------
    xva_discounted = XvaCalculator(
        counterparty=counterparty,
        days_in_year=sim_cfg.days_in_year,
        pfe_quantile=0.95,
        discount_to_zero=True,
        flat_discount_rate=discounting.rate,
    )
    prof_discounted = xva_discounted.build_exposure_profile(
        times_days=result_rn.times_days,
        mtm_paths=result_rn.mtm_paths,
    )

    xva_undiscounted = XvaCalculator(
        counterparty=counterparty,
        days_in_year=sim_cfg.days_in_year,
        pfe_quantile=0.95,
        discount_to_zero=False,
        flat_discount_rate=discounting.rate,
    )
    prof_undiscounted = xva_undiscounted.build_exposure_profile(
        times_days=result_rn.times_days,
        mtm_paths=result_rn.mtm_paths,
    )

    # CVA computed off discounted EE* in your engine convention
    cva = xva_discounted.cva_from_ee(times_days=result_rn.times_days, ee_star=prof_discounted.ee)

    # ----------------------------
    # 8) Plot: curve, fixing schedule, exposure profiles
    # ----------------------------
    plot_initial_forward_curve(
        value_date=value_date,
        tenor_dates=tenor_dates,
        initial_curve=initial_curve,
        title="Initial commodity forward curve F(0,T)",
    )

    # Fixing dates as datetime objects (for the plot)
    fixing_days = fixing_schedule.sample_days()
    fixing_dates = [day_to_date(value_date, d) for d in fixing_days]

    plot_fixing_schedule(
        value_date=value_date,
        maturity_date=maturity_date,
        fixing_dates=fixing_dates,
        title=f"Fixing schedule ({convention.value})",
    )

    plot_exposure_profiles(
        base_date=value_date,
        times_days=result_rn.times_days,
        profile_discounted=prof_discounted,
        profile_undiscounted=prof_undiscounted,
        title="Commodity Forward Exposure Profiles (RN)",
    )

    # ----------------------------
    # 9) Print a compact run summary
    # ----------------------------
    print("\n=== RUN SUMMARY ===")
    print(f"Value date:    {value_date.isoformat()}")
    print(f"Maturity date: {maturity_date.isoformat()}  (maturity_day={maturity_day})")
    print(f"Fixing window: {fix_start_date.isoformat()} -> {fix_end_date.isoformat()} "
          f"(days {fix_start_day} -> {fix_end_day}), convention={convention.value}")
    print(f"Strike:        {strike:.6f}")
    print(f"Notional:      {notional:,.0f}")
    print(f"CVA:           {cva:,.6f}")
    print("\nSimulationConfig:", asdict(sim_cfg))
    print("CSParams:", asdict(cs_params))
    print("CounterpartyConfig:", asdict(counterparty))
    print("DiscountingConfig:", asdict(discounting))

    plt.show()


if __name__ == "__main__":
    main()
