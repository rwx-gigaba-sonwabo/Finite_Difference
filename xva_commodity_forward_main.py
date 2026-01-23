"""
xva_commodity_forward_main.py

Aligned to your setup:

- You have MANY commodities.
- For each commodity you have:
    * commodity_tenors[asset_code]  = list[int] days-to-maturity measured from VALUE_DATE = 2025-07-28
    * commodity_prices[asset_code]  = list[float] initial forward prices at those tenors  (this is F(0,T))

- VALUE_DATE is fixed for the curves: 2025-07-28
- We can run XVA (EE/PFE/CVA) for:
    * a single commodity, or
    * multiple commodities (overlay exposure profiles)

This script:
1) selects asset(s)
2) builds initial curve + tenors for each
3) defines a forward trade (maturity picked from available tenors, strike ATM by default)
4) runs CS simulation + MTM + exposure + CVA using your existing engine
5) plots:
   - initial forward curve (per asset)
   - fixing schedule (value date + maturity)
   - exposure profiles (EE/PFE)
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ---- Your engine modules (match your project) ----
from xva_engine.config import (
    SimulationConfig,
    CounterpartyConfig,
    DiscountingConfig,
    SamplingConvention,
)
from xva_engine.models.clewlow_strickland import CSParams
from xva_engine.reference_price import FixingSchedule, ReferencePrice
from xva_engine.products.commodity_forward import CommodityForward
from xva_engine.engine import CommodityXvaEngine
from xva_engine.xva.cva import XvaCalculator


# =============================================================================
# 0) GLOBAL VALUE DATE (CURVE ANCHOR DATE)
# =============================================================================
VALUE_DATE = date(2025, 7, 28)  # <-- your statement: all tenors are days from here


# =============================================================================
# 1) YOUR MARKET DATA (PASTE/KEEP YOUR DICTS EXACTLY AS YOU HAVE THEM)
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

# Tenors are DAYS to maturity FROM VALUE_DATE (2025-07-28)
commodity_tenors: Dict[str, List[int]] = {
    # >>> KEEP YOUR FULL DICT HERE <<<
    # examples:
    "CommodityPrice.GOLD": [2, 35, 94, 186, 276, 367, 732],
    "CommodityPrice.BRENT_OIL_IPE": [2, 35, 65, 94, 127, 156, 186, 219, 247, 276, 308, 338, 367, 553, 732],
    # ...
}

# Initial forward curve points F(0,T) aligned to the tenors above
commodity_prices: Dict[str, List[float]] = {
    # >>> KEEP YOUR FULL DICT HERE <<<
    # examples:
    "CommodityPrice.GOLD": [3307.06, 3320.38, 3345.73, 3381.77, 3414.29, 3445.39, 3550.86],
    "CommodityPrice.BRENT_OIL_IPE": [70.04, 70.04, 69.32, 68.69, 68.22, 67.94, 67.75, 67.62, 67.53, 67.47, 67.41, 67.33, 67.26, 67.05, 67.05],
    # ...
}


# =============================================================================
# 2) USER CONTROLS
# =============================================================================
# Run a single asset by its short name (matches the suffix after the dot), or run all
ASSETS_TO_RUN = ["GOLD"]          # e.g. ["GOLD"], or ["GOLD", "BRENT_OIL_IPE"], or [] for all
TRADE_MATURITY_DAYS_TARGET = 186  # pick closest available tenor to this (e.g. ~6M)
STRIKE_MODE = "ATM"               # "ATM" or "CUSTOM"
CUSTOM_STRIKE = 0.0               # used only if STRIKE_MODE == "CUSTOM"
NOTIONAL = 1_000_000.0

FIXING_CONVENTION = SamplingConvention.BULLET  # BULLET / DAILY / WEEKLY / MONTHLY
AVERAGING_WINDOW_DAYS = 30                     # used only if not BULLET (toy averaging window)

# Simulation & XVA
NUM_SIMS = 20_000
DT_DAYS = 7                 # weekly profile dates
DAYS_IN_YEAR = 365.0
SEED = 7
FAST_FORWARD = 0

DISCOUNT_RATE = 0.09        # flat cc (placeholder)
HAZARD_RATE = 0.03          # flat hazard (placeholder)
RECOVERY = 0.40             # (placeholder)

# CS params
# If you have per-commodity calibration, you can fill this dict:
CS_PARAMS_BY_ASSET: Dict[str, CSParams] = {
    # "CommodityPrice.GOLD": CSParams(alpha=..., sigma=..., mu=...),
}
DEFAULT_CS_PARAMS = CSParams(alpha=1.2, sigma=0.35, mu=0.02)  # fallback


# =============================================================================
# 3) HELPERS
# =============================================================================
def asset_code_from_short(short: str) -> Optional[str]:
    short = short.strip().upper()
    return next((c for c in commodities if c.split(".")[-1].upper() == short), None)


def days_to_dates(base: date, days: np.ndarray) -> List[date]:
    return [base + timedelta(days=int(round(d))) for d in days]


def closest_tenor(target: float, tenors: np.ndarray) -> int:
    idx = int(np.argmin(np.abs(tenors - float(target))))
    return int(tenors[idx])


def interp_curve_at(day: float, tenor_days: np.ndarray, curve: np.ndarray) -> float:
    return float(np.interp(float(day), tenor_days.astype(float), curve.astype(float)))


def plot_fixing_schedule(value_date: date, maturity_date: date, fixing_days: np.ndarray, title: str) -> None:
    fixing_dates = days_to_dates(value_date, fixing_days)
    y = np.zeros(len(fixing_dates), dtype=float)

    plt.figure()
    plt.plot(fixing_dates, y, marker="o", linestyle="None", label="fixings")
    plt.axvline(value_date, linestyle=":", label="value date")
    plt.axvline(maturity_date, linestyle="--", label="maturity")
    plt.yticks([])
    plt.title(title)
    plt.legend()
    plt.tight_layout()


def plot_initial_curve(asset_code: str, value_date: date, tenor_days: np.ndarray, initial_curve: np.ndarray) -> None:
    tenor_dates = days_to_dates(value_date, tenor_days)
    plt.figure()
    plt.plot(tenor_dates, initial_curve, marker="o")
    plt.axvline(value_date, linestyle=":", label="value date")
    plt.title(f"Initial forward curve F(0,T) — {asset_code}")
    plt.legend()
    plt.tight_layout()


def plot_exposure_overlay(value_date: date, overlay: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]) -> None:
    """
    overlay items: (asset_code, times_days, EE, PFE) where EE/PFE are already consistent convention
    """
    plt.figure()
    for asset_code, times_days, ee, _pfe in overlay:
        times_dates = days_to_dates(value_date, times_days)
        plt.plot(times_dates, ee, label=f"EE — {asset_code}")
    plt.title("Expected Exposure (EE) overlay")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    for asset_code, times_days, _ee, pfe in overlay:
        times_dates = days_to_dates(value_date, times_days)
        plt.plot(times_dates, pfe, label=f"PFE — {asset_code}")
    plt.title("PFE overlay")
    plt.legend()
    plt.tight_layout()


# =============================================================================
# 4) MAIN RUNNER PER ASSET
# =============================================================================
def run_asset(asset_code: str) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    if asset_code not in commodity_tenors or asset_code not in commodity_prices:
        raise KeyError(f"Missing tenors/prices for {asset_code}")

    tenor_days = np.asarray(commodity_tenors[asset_code], dtype=float)
    initial_curve = np.asarray(commodity_prices[asset_code], dtype=float)

    if tenor_days.shape[0] != initial_curve.shape[0]:
        raise ValueError(f"Tenors/prices length mismatch for {asset_code}")

    # ---- Choose trade maturity from available tenors (snap to closest node)
    maturity_day = closest_tenor(TRADE_MATURITY_DAYS_TARGET, tenor_days)
    maturity_date = VALUE_DATE + timedelta(days=int(maturity_day))

    # ---- Strike
    if STRIKE_MODE.upper() == "ATM":
        strike = interp_curve_at(maturity_day, tenor_days, initial_curve)
    else:
        strike = float(CUSTOM_STRIKE)

    # ---- Fixing schedule (days-from-VALUE_DATE)
    if FIXING_CONVENTION == SamplingConvention.BULLET:
        fix_start_day = maturity_day
        fix_end_day = maturity_day
    else:
        fix_end_day = maturity_day
        fix_start_day = max(0, maturity_day - int(AVERAGING_WINDOW_DAYS))

    fixing_schedule = FixingSchedule(
        start_day=int(fix_start_day),
        end_day=int(fix_end_day),
        convention=FIXING_CONVENTION,
        offset_days=0,
    )

    # ---- Reference price mapping
    # Delivery days are your curve node tenors
    reference_price = ReferencePrice(
        fixing_schedule=fixing_schedule,
        delivery_days=tenor_days,
        realised_fixings={},  # populate if VALUE_DATE is inside fixing period and some fixings are known
    )

    # ---- Trade
    discounting = DiscountingConfig(rate=float(DISCOUNT_RATE))
    trade = CommodityForward(
        maturity_day=int(maturity_day),
        strike=float(strike),
        notional=float(NOTIONAL),
        reference_price=reference_price,
        discounting=discounting,
    )

    # ---- Simulation horizon must cover the trade maturity at least (and ideally max curve tenor you rely on)
    horizon_days = int(max(maturity_day, float(tenor_days.max())))
    sim_cfg = SimulationConfig(
        num_sims=int(NUM_SIMS),
        seed=int(SEED),
        fast_forward=int(FAST_FORWARD),
        dt_days=int(DT_DAYS),
        horizon_days=int(horizon_days),
        days_in_year=float(DAYS_IN_YEAR),
    )

    # ---- CS params (per asset if you have them)
    cs_params = CS_PARAMS_BY_ASSET.get(asset_code, DEFAULT_CS_PARAMS)

    # ---- Credit
    counterparty = CounterpartyConfig(hazard_rate=float(HAZARD_RATE), recovery=float(RECOVERY))

    # ---- Engine
    engine = CommodityXvaEngine(
        sim_cfg=sim_cfg,
        cs_params=cs_params,
        initial_curve=initial_curve,
        tenor_days=tenor_days,
        discounting=discounting,
        counterparty=counterparty,
        device="cpu",
    )

    # RN exposures for CVA are usually the correct convention:
    result = engine.run_forward_cva(trade=trade, risk_neutral=True)

    # ---- Build exposure profile in BOTH conventions for plotting
    # (Your engine’s internal XvaCalculator may already discount-to-0; we compute explicitly here.)
    xva_disc = XvaCalculator(
        counterparty=counterparty,
        days_in_year=sim_cfg.days_in_year,
        pfe_quantile=0.95,
        discount_to_zero=True,
        flat_discount_rate=discounting.rate,
    )
    prof_disc = xva_disc.build_exposure_profile(result.times_days, result.mtm_paths)

    xva_raw = XvaCalculator(
        counterparty=counterparty,
        days_in_year=sim_cfg.days_in_year,
        pfe_quantile=0.95,
        discount_to_zero=False,
        flat_discount_rate=discounting.rate,
    )
    prof_raw = xva_raw.build_exposure_profile(result.times_days, result.mtm_paths)

    cva = xva_disc.cva_from_ee(result.times_days, prof_disc.ee)

    # ---- Plots per asset
    plot_initial_curve(asset_code, VALUE_DATE, tenor_days, initial_curve)
    plot_fixing_schedule(
        VALUE_DATE,
        maturity_date,
        fixing_schedule.sample_days(),
        title=f"Fixing schedule — {asset_code} ({FIXING_CONVENTION.value})",
    )

    # Exposure plots (per asset)
    times_dates = days_to_dates(VALUE_DATE, result.times_days)
    plt.figure()
    plt.plot(times_dates, prof_raw.ee, label="EE (undiscounted)")
    plt.plot(times_dates, prof_raw.pfe, label="PFE (undiscounted)")
    plt.title(f"Exposure profile (raw) — {asset_code}")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(times_dates, prof_disc.ee, label="EE* (discount-to-0)")
    plt.plot(times_dates, prof_disc.pfe, label="PFE* (discount-to-0)")
    plt.title(f"Exposure profile (discounted-to-0) — {asset_code}")
    plt.legend()
    plt.tight_layout()

    # ---- Print summary
    print("\n====================================================")
    print(f"ASSET:          {asset_code}")
    print(f"VALUE DATE:     {VALUE_DATE.isoformat()}")
    print(f"MATURITY:       {maturity_date.isoformat()}  (day={maturity_day})")
    print(f"STRIKE:         {strike:.6f}  ({STRIKE_MODE})")
    print(f"NOTIONAL:       {NOTIONAL:,.0f}")
    print(f"CS PARAMS:      alpha={cs_params.alpha:.6f}, sigma={cs_params.sigma:.6f}, mu={cs_params.mu:.6f}")
    print(f"DISCOUNT RATE:  {DISCOUNT_RATE:.6f}")
    print(f"HAZARD RATE:    {HAZARD_RATE:.6f}  RECOVERY={RECOVERY:.2f}")
    print(f"CVA:            {cva:,.6f}")
    print("====================================================\n")

    return cva, result.times_days, prof_disc.ee, prof_disc.pfe


# =============================================================================
# 5) SCRIPT ENTRYPOINT
# =============================================================================
def main() -> None:
    # Resolve which assets to run
    if not ASSETS_TO_RUN:
        asset_codes = list(commodity_prices.keys())
    else:
        asset_codes = []
        for short in ASSETS_TO_RUN:
            code = asset_code_from_short(short)
            if code is None:
                raise ValueError(f"Unknown asset short name '{short}'. Check your commodities list.")
            asset_codes.append(code)

    overlay_items: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    cvas: Dict[str, float] = {}

    for code in asset_codes:
        cva, times_days, ee, pfe = run_asset(code)
        cvas[code] = cva
        overlay_items.append((code, times_days, ee, pfe))

    # Overlay exposure plots across assets (discount-to-0 EE*/PFE*)
    if len(overlay_items) > 1:
        plot_exposure_overlay(VALUE_DATE, overlay_items)

    # Print CVA ranking
    if len(cvas) > 1:
        print("CVA RANKING (highest to lowest):")
        for k, v in sorted(cvas.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {k:35s}  {v:,.6f}")

    plt.show()


if __name__ == "__main__":
    main()
