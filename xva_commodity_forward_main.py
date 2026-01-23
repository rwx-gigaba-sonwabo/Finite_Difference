"""
xva_commodity_forward_main.py

Date-driven trade specification:
- VALUE_DATE is fixed (curve anchor) = 2025-07-28
- Trade maturity is specified as a datetime.date (per commodity or common)
- The engine still works in "days from value date" internally:
    fix_end_day = (maturity_date - VALUE_DATE).days

Settlement lag handling:
- If MATURITY_IS_CASHFLOW_DATE = False:
      maturity_date is fixing/expiry T  -> cashflow = T + lag
      ReferencePrice looks up forwards at (fixing_day + lag)
      Trade.maturity_day = cashflow_day for discounting
- If MATURITY_IS_CASHFLOW_DATE = True:
      maturity_date already equals settlement/cashflow date -> lag must be 0
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from xva_engine.config import (
    SamplingConvention,
    SimulationConfig,
    CounterpartyConfig,
    DiscountingConfig,
)
from xva_engine.models.clewlow_strickland import CSParams
from xva_engine.reference_price import FixingSchedule, ReferencePrice
from xva_engine.products.commodity_forward import CommodityForward
from xva_engine.engine import CommodityXvaEngine
from xva_engine.xva.cva import XvaCalculator


# =============================================================================
# 1) GLOBAL VALUE DATE (curve anchor)
# =============================================================================
VALUE_DATE = date(2025, 7, 28)


# =============================================================================
# 2) YOUR MARKET DATA (PASTE YOUR FULL DICTS HERE)
# =============================================================================
commodities = [
    "CommodityPrice.ALUMINIUM_ALLOY",
    "CommodityPrice.BRENT_OIL_IPE",
    "CommodityPrice.COPPER",
    "CommodityPrice.CMX_GOLD",
    "CommodityPrice.CRUDE_OIL_WTI",
    "CommodityPrice.FUEL_OIL",
    "CommodityPrice.GOLD",
    "CommodityPrice.GOLD_AM",
    "CommodityPrice.GOLD_PM",
    "CommodityPrice.LEAD",
    "CommodityPrice.LOW_SULPHUR_GASOIL",
    "CommodityPrice.NICKEL",
    "CommodityPrice.PALLADIUM",
    "CommodityPrice.PALLADIUM_AM",
    "CommodityPrice.PALLADIUM_PM",
    "CommodityPrice.PLATINUM",
    "CommodityPrice.PLATINUM_AM",
    "CommodityPrice.PLATINUM_PM",
    "CommodityPrice.SILVER",
    "CommodityPrice.TIN",
    "CommodityPrice.ZINC",
]

commodity_tenors: Dict[str, List[int]] = {
    # Example only — paste your full dict
    "CommodityPrice.GOLD": [2, 35, 94, 186, 276, 367, 732],
    "CommodityPrice.BRENT_OIL_IPE": [2, 35, 65, 94, 127, 156, 186, 219, 247, 276, 308, 338, 367, 553, 732],
}

commodity_prices: Dict[str, List[float]] = {
    # Example only — paste your full dict
    "CommodityPrice.GOLD": [3307.06, 3320.38, 3345.73, 3381.77, 3414.29, 3445.39, 3550.86],
    "CommodityPrice.BRENT_OIL_IPE": [70.04, 70.04, 69.32, 68.69, 68.22, 67.94, 67.75, 67.62, 67.53, 67.47, 67.41, 67.33, 67.26, 67.05, 67.05],
}


# =============================================================================
# 3) RUN CONTROLS
# =============================================================================
# Which assets to run: short codes like ["GOLD"] or ["GOLD", "BRENT_OIL_IPE"]; [] => run all keys in commodity_prices
ASSETS_TO_RUN: List[str] = ["GOLD"]

# Trade maturity date:
# Option A: one maturity date for all assets
COMMON_MATURITY_DATE: Optional[date] = date(2026, 1, 22)

# Option B: per-asset maturity dates (override the common date if present)
MATURITY_BY_ASSET: Dict[str, date] = {
    # "CommodityPrice.GOLD": date(2026, 1, 22),
}

# Fixing convention:
FIXING_CONVENTION = SamplingConvention.BULLET   # BULLET / DAILY / WEEKLY / MONTHLY
AVERAGING_WINDOW_DAYS = 30                      # used if not BULLET (toy)

# Settlement lag:
SETTLEMENT_LAG_DAYS = 2                         # if your convention is "lookup at T+2"
MATURITY_IS_CASHFLOW_DATE = False               # True => maturity is already settlement/cashflow, so lag should be 0

# Notional / strike:
NOTIONAL = 1_000_000.0
STRIKE_MODE = "ATM"                              # ATM or CUSTOM
CUSTOM_STRIKE = 0.0

# Simulation:
NUM_SIMS = 20_000
DT_DAYS = 7
DAYS_IN_YEAR = 365.0
SEED = 7
FAST_FORWARD = 0

# Discount & credit:
DISCOUNT_RATE = 0.09
HAZARD_RATE = 0.03
RECOVERY = 0.40

# CS params:
CS_PARAMS_BY_ASSET: Dict[str, CSParams] = {}
DEFAULT_CS_PARAMS = CSParams(alpha=1.2, sigma=0.35, mu=0.02)  # mu ignored if risk_neutral=True


# =============================================================================
# 4) HELPERS
# =============================================================================
def asset_code_from_short(short: str) -> str:
    short_u = short.strip().upper()
    code = next((c for c in commodities if c.split(".")[-1].upper() == short_u), None)
    if code is None:
        raise ValueError(f"Unknown asset short name: {short}")
    return code


def days_to_dates(base: date, days: np.ndarray) -> List[date]:
    return [base + timedelta(days=int(round(d))) for d in days]


def interp_initial_forward(day: float, tenor_days: np.ndarray, initial_curve: np.ndarray) -> float:
    """Linear interpolation on initial curve nodes with flat extrapolation."""
    td = tenor_days.astype(float)
    y = initial_curve.astype(float)
    x = float(np.clip(day, td[0], td[-1]))
    return float(np.interp(x, td, y))


def plot_initial_curve(asset_code: str, tenor_days: np.ndarray, initial_curve: np.ndarray) -> None:
    tenor_dates = days_to_dates(VALUE_DATE, tenor_days)
    plt.figure()
    plt.plot(tenor_dates, initial_curve, marker="o")
    plt.axvline(VALUE_DATE, linestyle=":", label="value date")
    plt.title(f"Initial forward curve F(0,T) — {asset_code}")
    plt.legend()
    plt.tight_layout()


def plot_fixing_schedule(asset_code: str, fixing_days: np.ndarray, fix_end_day: int, cashflow_day: int) -> None:
    fixing_dates = days_to_dates(VALUE_DATE, fixing_days)
    fix_end_date = VALUE_DATE + timedelta(days=int(fix_end_day))
    cashflow_date = VALUE_DATE + timedelta(days=int(cashflow_day))

    y = np.zeros(len(fixing_dates), dtype=float)
    plt.figure()
    plt.plot(fixing_dates, y, marker="o", linestyle="None", label="fixings")
    plt.axvline(VALUE_DATE, linestyle=":", label="value date")
    plt.axvline(fix_end_date, linestyle="--", label="fix end (T)")
    plt.axvline(cashflow_date, linestyle="-.", label="cashflow/settle")
    plt.yticks([])
    plt.title(f"Fixing schedule — {asset_code}")
    plt.legend()
    plt.tight_layout()


def plot_exposure(asset_code: str, times_days: np.ndarray, prof_raw, prof_disc) -> None:
    times_dates = days_to_dates(VALUE_DATE, times_days)

    plt.figure()
    plt.plot(times_dates, prof_raw.ee, label="EE (raw)")
    plt.plot(times_dates, prof_raw.pfe, label="PFE (raw)")
    plt.title(f"Exposure profile — RAW — {asset_code}")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(times_dates, prof_disc.ee, label="EE* (discount-to-0)")
    plt.plot(times_dates, prof_disc.pfe, label="PFE* (discount-to-0)")
    plt.title(f"Exposure profile — DISCOUNTED — {asset_code}")
    plt.legend()
    plt.tight_layout()


# =============================================================================
# 5) PER-ASSET RUN
# =============================================================================
def run_asset(asset_code: str) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    if asset_code not in commodity_tenors or asset_code not in commodity_prices:
        raise KeyError(f"Missing tenors/prices for {asset_code}")

    tenor_days = np.asarray(commodity_tenors[asset_code], dtype=float)
    initial_curve = np.asarray(commodity_prices[asset_code], dtype=float)

    if tenor_days.size != initial_curve.size:
        raise ValueError(f"Tenors/prices length mismatch for {asset_code}.")
    if not np.all(np.diff(tenor_days) > 0):
        raise ValueError(f"Tenor days must be strictly increasing for {asset_code}.")

    # --- maturity date selection
    maturity_date = MATURITY_BY_ASSET.get(asset_code, COMMON_MATURITY_DATE)
    if maturity_date is None:
        raise ValueError("No maturity date provided: set COMMON_MATURITY_DATE or MATURITY_BY_ASSET[asset_code].")

    # convert maturity_date to day count from VALUE_DATE
    fix_end_day = int((maturity_date - VALUE_DATE).days)
    if fix_end_day <= 0:
        raise ValueError(f"{asset_code}: maturity_date must be after VALUE_DATE. Got fix_end_day={fix_end_day}.")

    # settlement lag policy
    lag = 0 if MATURITY_IS_CASHFLOW_DATE else int(SETTLEMENT_LAG_DAYS)

    # cashflow date/day (discounting should go to cashflow)
    cashflow_day = int(fix_end_day + lag)

    # fixing window
    if FIXING_CONVENTION == SamplingConvention.BULLET:
        fix_start_day = fix_end_day
    else:
        fix_start_day = max(0, fix_end_day - int(AVERAGING_WINDOW_DAYS))

    fixing_schedule = FixingSchedule(
        start_day=int(fix_start_day),
        end_day=int(fix_end_day),
        convention=FIXING_CONVENTION,
        offset_days=0,
    )

    # Reference price lookup uses (sample_day + settlement_lag_days)
    reference_price = ReferencePrice(
        fixing_schedule=fixing_schedule,
        settlement_lag_days=int(lag),
        realised_fixings={},
    )

    # ATM strike must match the same convention: F(0, T+lag)
    if STRIKE_MODE.upper() == "ATM":
        strike = interp_initial_forward(float(fix_end_day + lag), tenor_days, initial_curve)
    else:
        strike = float(CUSTOM_STRIKE)

    discounting = DiscountingConfig(rate=float(DISCOUNT_RATE))

    # IMPORTANT: maturity_day on the trade is the CASHFLOW day (discounting DF(t, cashflow))
    trade = CommodityForward(
        maturity_day=int(cashflow_day),
        strike=float(strike),
        notional=float(NOTIONAL),
        reference_price=reference_price,
        discounting=discounting,
    )

    # Horizon only needs to run until cashflow day (trade is dead after settlement)
    sim_cfg = SimulationConfig(
        num_sims=int(NUM_SIMS),
        seed=int(SEED),
        fast_forward=int(FAST_FORWARD),
        dt_days=int(DT_DAYS),
        horizon_days=int(cashflow_day),
        days_in_year=float(DAYS_IN_YEAR),
    )

    cs_params = CS_PARAMS_BY_ASSET.get(asset_code, DEFAULT_CS_PARAMS)
    counterparty = CounterpartyConfig(hazard_rate=float(HAZARD_RATE), recovery=float(RECOVERY))

    engine = CommodityXvaEngine(
        sim_cfg=sim_cfg,
        cs_params=cs_params,
        initial_curve=initial_curve,
        tenor_days=tenor_days,
        discounting=discounting,
        counterparty=counterparty,
        device="cpu",
    )

    # Risk-neutral exposures for CVA
    result = engine.run_forward_cva(trade=trade, risk_neutral=True)

    # Engine already produces discounted-to-0 exposures in result.exposure_profile
    prof_disc = result.exposure_profile

    # Build raw exposure profile (undiscounted) for comparison
    xva_raw = XvaCalculator(
        counterparty=counterparty,
        days_in_year=sim_cfg.days_in_year,
        pfe_quantile=0.95,
        discount_to_zero=False,
        flat_discount_rate=discounting.rate,
    )
    prof_raw = xva_raw.build_exposure_profile(times_days=result.times_days, mtm_paths=result.mtm_paths)

    cva = float(result.cva)

    # Plots
    plot_initial_curve(asset_code, tenor_days, initial_curve)
    plot_fixing_schedule(asset_code, fixing_schedule.sample_days(), fix_end_day, cashflow_day)
    plot_exposure(asset_code, result.times_days, prof_raw, prof_disc)

    # Console summary
    print("\n====================================================")
    print(f"ASSET:                  {asset_code}")
    print(f"VALUE DATE:             {VALUE_DATE.isoformat()}")
    print(f"MATURITY DATE (input):  {maturity_date.isoformat()}  -> fix_end_day={fix_end_day}")
    print(f"MATURITY_IS_CASHFLOW:   {MATURITY_IS_CASHFLOW_DATE}")
    print(f"SETTLEMENT LAG USED:    {lag} day(s)")
    print(f"CASHFLOW DAY:           {cashflow_day}  -> {(VALUE_DATE + timedelta(days=cashflow_day)).isoformat()}")
    print(f"FIXING CONVENTION:      {FIXING_CONVENTION.value}")
    print(f"STRIKE MODE / STRIKE:   {STRIKE_MODE} / {strike:.8f}")
    print(f"NOTIONAL:               {NOTIONAL:,.0f}")
    print(f"DISCOUNT RATE:          {DISCOUNT_RATE:.6f}")
    print(f"HAZARD / RECOVERY:      {HAZARD_RATE:.6f} / {RECOVERY:.2f}")
    print(f"CVA:                    {cva:,.8f}")
    print("====================================================\n")

    return cva, result.times_days, prof_disc.ee, prof_disc.pfe


# =============================================================================
# 6) ENTRYPOINT
# =============================================================================
def main() -> None:
    if not ASSETS_TO_RUN:
        asset_codes = list(commodity_prices.keys())
    else:
        asset_codes = [asset_code_from_short(s) for s in ASSETS_TO_RUN]

    cvas: Dict[str, float] = {}

    for code in asset_codes:
        cva, _, _, _ = run_asset(code)
        cvas[code] = cva

    if len(cvas) > 1:
        print("CVA ranking (highest to lowest):")
        for k, v in sorted(cvas.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {k:40s}  {v:,.8f}")

    plt.show()


if __name__ == "__main__":
    main()
