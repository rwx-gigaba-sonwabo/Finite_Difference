"""
xva_commodity_forward_main.py

UPDATED to align with your market-data layout:

- You have MANY commodities.
- For each commodity:
    commodity_tenors[asset_code] = list[int]  # tenor node days FROM VALUE_DATE (2025-07-28)
    commodity_prices[asset_code] = list[float]  # initial forward prices F(0, T_node)

- VALUE_DATE is fixed: 2025-07-28.
- We run:
    simulate CS forward curve -> revalue forward -> exposure profile -> CVA
- We plot:
    1) Initial forward curve per commodity
    2) Fixing schedule + value date + fixing end date + cashflow/settlement date
    3) Exposure profiles: raw (EE/PFE) and discounted-to-0 (EE*/PFE*) used for CVA
    4) Optional overlay across multiple commodities

IMPORTANT (Settlement lag):
--------------------------
Your ReferencePrice now supports settlement_lag_days (default 2) and linearly interpolates in tenor.
But if your tenor nodes ALREADY reflect settlement dates (common if the first tenor is 2 days),
then you should set settlement_lag_days = 0 to avoid double-counting.

This script provides:
- SETTLEMENT_LAG_DAYS: either explicit, or AUTO (based on min tenor)
- and it makes the trade cashflow date consistent with the lag:
      cashflow_day = fixing_end_day + settlement_lag_days
      trade.maturity_day = cashflow_day   (so discounting is to the cashflow date)

If your "maturity" in your tenors is already the settlement/cashflow date, AUTO will pick 0.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ---- project imports (match your codebase structure) ----
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
# 2) YOUR MARKET DATA (paste your full dicts exactly)
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

# Tenors are DAYS from VALUE_DATE
commodity_tenors: Dict[str, List[int]] = {
    # >>> paste your FULL dict here <<<
    "CommodityPrice.GOLD": [2, 35, 94, 186, 276, 367, 732],
    "CommodityPrice.BRENT_OIL_IPE": [2, 35, 65, 94, 127, 156, 186, 219, 247, 276, 308, 338, 367, 553, 732],
}

# Initial forward curve points F(0,T_node) at those tenors
commodity_prices: Dict[str, List[float]] = {
    # >>> paste your FULL dict here <<<
    "CommodityPrice.GOLD": [3307.06, 3320.38, 3345.73, 3381.77, 3414.29, 3445.39, 3550.86],
    "CommodityPrice.BRENT_OIL_IPE": [70.04, 70.04, 69.32, 68.69, 68.22, 67.94, 67.75, 67.62, 67.53, 67.47, 67.41, 67.33, 67.26, 67.05, 67.05],
}


# =============================================================================
# 3) RUN CONTROLS
# =============================================================================
# Which commodities to run:
# - Provide short names like ["GOLD", "BRENT_OIL_IPE"], or [] to run all in commodity_prices
ASSETS_TO_RUN: List[str] = ["GOLD"]

# Trade style:
FIXING_CONVENTION = SamplingConvention.BULLET      # BULLET / DAILY / WEEKLY / MONTHLY
AVERAGING_WINDOW_DAYS = 30                         # used if not BULLET (toy)

# Choose a trade "fixing end" day (expiry/fixing end), not necessarily a node:
TRADE_FIX_END_TARGET_DAYS = 186                    # e.g., ~6M

# Strike / notional:
STRIKE_MODE = "ATM"                                # ATM or CUSTOM
CUSTOM_STRIKE = 0.0
NOTIONAL = 1_000_000.0

# Settlement lag:
# - If you know your convention: set explicitly (0 or 2)
# - If you want AUTO: set to None and the script will infer from min tenor
SETTLEMENT_LAG_DAYS: Optional[int] = None          # None => AUTO infer

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
# If you have per-commodity calibrations, fill CS_PARAMS_BY_ASSET; otherwise default used.
CS_PARAMS_BY_ASSET: Dict[str, CSParams] = {}
DEFAULT_CS_PARAMS = CSParams(alpha=1.2, sigma=0.35, mu=0.02)  # mu ignored in RN mode


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


def infer_settlement_lag(tenor_days: np.ndarray) -> int:
    """
    Heuristic:
    - if first tenor is 2 (or <=2), it often means curve nodes are already settlement/delivery dates.
      In that case, applying +2 again would double-count -> choose 0.
    - otherwise choose 2.
    """
    mn = float(np.min(tenor_days))
    return 0 if mn <= 2.0 else 2


def pick_fix_end_day(target: float, tenor_days: np.ndarray) -> int:
    """
    You can either:
    - force exact day (target), or
    - snap to closest curve node.

    Here we keep the FIX-END day as the target (not snapped), so the reference price interpolation
    truly gets exercised. If you'd rather snap, replace with closest node selection.
    """
    return int(round(float(target)))


def interp_initial_forward(day: float, tenor_days: np.ndarray, initial_curve: np.ndarray) -> float:
    """Linear interpolation on initial curve nodes."""
    x = float(day)
    td = tenor_days.astype(float)
    y = initial_curve.astype(float)
    # clamp to support
    x = max(td[0], min(td[-1], x))
    return float(np.interp(x, td, y))


def plot_initial_curve(asset_code: str, tenor_days: np.ndarray, initial_curve: np.ndarray) -> None:
    tenor_dates = days_to_dates(VALUE_DATE, tenor_days)
    plt.figure()
    plt.plot(tenor_dates, initial_curve, marker="o")
    plt.axvline(VALUE_DATE, linestyle=":", label="value date")
    plt.title(f"Initial forward curve F(0,T) — {asset_code}")
    plt.legend()
    plt.tight_layout()


def plot_fixing_schedule(
    asset_code: str,
    fixing_days: np.ndarray,
    fix_end_day: int,
    cashflow_day: int,
) -> None:
    fixing_dates = days_to_dates(VALUE_DATE, fixing_days)
    fix_end_date = VALUE_DATE + timedelta(days=int(fix_end_day))
    cashflow_date = VALUE_DATE + timedelta(days=int(cashflow_day))

    y = np.zeros(len(fixing_dates), dtype=float)
    plt.figure()
    plt.plot(fixing_dates, y, marker="o", linestyle="None", label="fixings")
    plt.axvline(VALUE_DATE, linestyle=":", label="value date")
    plt.axvline(fix_end_date, linestyle="--", label="fix end (T)")
    plt.axvline(cashflow_date, linestyle="-.", label="cashflow/settle (T+lag)")
    plt.yticks([])
    plt.title(f"Fixing schedule — {asset_code}")
    plt.legend()
    plt.tight_layout()


def plot_exposure_profiles(
    asset_code: str,
    times_days: np.ndarray,
    prof_raw,
    prof_disc,
) -> None:
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


def plot_overlay(
    title: str,
    series: List[Tuple[str, np.ndarray, np.ndarray]],
) -> None:
    plt.figure()
    for asset_code, times_days, y in series:
        times_dates = days_to_dates(VALUE_DATE, times_days)
        plt.plot(times_dates, y, label=asset_code)
    plt.title(title)
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

    if tenor_days.ndim != 1 or initial_curve.ndim != 1:
        raise ValueError(f"Bad curve inputs for {asset_code}: must be 1D arrays.")
    if tenor_days.size != initial_curve.size:
        raise ValueError(f"Tenors/prices length mismatch for {asset_code}.")
    if not np.all(np.diff(tenor_days) > 0):
        raise ValueError(f"Tenor days must be strictly increasing for {asset_code}.")

    # Settlement lag choice
    lag = SETTLEMENT_LAG_DAYS if SETTLEMENT_LAG_DAYS is not None else infer_settlement_lag(tenor_days)

    # Fixing end day = trade expiry / end of averaging window (T)
    fix_end_day = pick_fix_end_day(TRADE_FIX_END_TARGET_DAYS, tenor_days)

    # Cashflow/settlement day = T + lag (this is what you should discount to)
    cashflow_day = int(fix_end_day + lag)

    if cashflow_day > int(tenor_days[-1]):
        print(
            f"[WARN] {asset_code}: cashflow_day={cashflow_day} exceeds max curve tenor={int(tenor_days[-1])}. "
            f"ReferencePrice will flat-extrapolate at the last node."
        )

    # Fixing window
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

    # Reference price: settlement lag applied INSIDE ReferencePrice for curve lookup
    reference_price = ReferencePrice(
        fixing_schedule=fixing_schedule,
        settlement_lag_days=int(lag),
        realised_fixings={},  # populate if you have known fixings inside the window
    )

    # Strike: ATM should match the same lookup convention => F(0, T+lag)
    if STRIKE_MODE.upper() == "ATM":
        strike = interp_initial_forward(float(fix_end_day + lag), tenor_days, initial_curve)
    else:
        strike = float(CUSTOM_STRIKE)

    # Trade cashflow maturity day MUST be the settlement/cashflow date
    discounting = DiscountingConfig(rate=float(DISCOUNT_RATE))
    trade = CommodityForward(
        maturity_day=int(cashflow_day),
        strike=float(strike),
        notional=float(NOTIONAL),
        reference_price=reference_price,
        discounting=discounting,
    )

    # Simulation horizon should cover cashflow day (exposure ends after settlement)
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

    # RN exposures recommended for CVA
    result = engine.run_forward_cva(trade=trade, risk_neutral=True)

    # Engine uses discounted-to-0 exposure (EE*) by default; we also compute raw exposure for plotting
    xva_raw = XvaCalculator(
        counterparty=counterparty,
        days_in_year=sim_cfg.days_in_year,
        pfe_quantile=0.95,
        discount_to_zero=False,
        flat_discount_rate=discounting.rate,
    )
    prof_raw = xva_raw.build_exposure_profile(times_days=result.times_days, mtm_paths=result.mtm_paths)

    xva_disc = XvaCalculator(
        counterparty=counterparty,
        days_in_year=sim_cfg.days_in_year,
        pfe_quantile=0.95,
        discount_to_zero=True,
        flat_discount_rate=discounting.rate,
    )
    prof_disc = xva_disc.build_exposure_profile(times_days=result.times_days, mtm_paths=result.mtm_paths)

    cva = xva_disc.cva_from_ee(times_days=result.times_days, ee_star=prof_disc.ee)

    # ---- plots
    plot_initial_curve(asset_code, tenor_days, initial_curve)
    plot_fixing_schedule(asset_code, fixing_schedule.sample_days(), fix_end_day, cashflow_day)
    plot_exposure_profiles(asset_code, result.times_days, prof_raw, prof_disc)

    # ---- summary
    print("\n====================================================")
    print(f"ASSET:               {asset_code}")
    print(f"VALUE DATE:          {VALUE_DATE.isoformat()}")
    print(f"FIX START (day):     {fix_start_day}")
    print(f"FIX END   (day=T):   {fix_end_day}  ({(VALUE_DATE + timedelta(days=fix_end_day)).isoformat()})")
    print(f"SETTLEMENT LAG:      {lag} day(s)")
    print(f"CASHFLOW (T+lag):    {cashflow_day}  ({(VALUE_DATE + timedelta(days=cashflow_day)).isoformat()})")
    print(f"STRIKE MODE:         {STRIKE_MODE}")
    print(f"STRIKE:              {strike:.8f}")
    print(f"NOTIONAL:            {NOTIONAL:,.0f}")
    print(f"DISCOUNT RATE:       {DISCOUNT_RATE:.6f}")
    print(f"HAZARD/RECOVERY:     {HAZARD_RATE:.6f} / {RECOVERY:.2f}")
    print(f"CS PARAMS:           alpha={cs_params.alpha:.6f}, sigma={cs_params.sigma:.6f}, mu={cs_params.mu:.6f} (mu ignored RN)")
    print(f"CVA:                 {cva:,.8f}")
    print("SimulationConfig:", asdict(sim_cfg))
    print("====================================================\n")

    # Return discounted curves for overlay
    return cva, result.times_days, prof_disc.ee, prof_disc.pfe


# =============================================================================
# 6) ENTRYPOINT
# =============================================================================
def main() -> None:
    # Resolve which assets to run
    if not ASSETS_TO_RUN:
        asset_codes = list(commodity_prices.keys())
    else:
        asset_codes = [asset_code_from_short(s) for s in ASSETS_TO_RUN]

    overlay_ee: List[Tuple[str, np.ndarray, np.ndarray]] = []
    overlay_pfe: List[Tuple[str, np.ndarray, np.ndarray]] = []
    cvas: Dict[str, float] = {}

    for code in asset_codes:
        cva, times_days, ee_star, pfe_star = run_asset(code)
        cvas[code] = cva
        overlay_ee.append((code, times_days, ee_star))
        overlay_pfe.append((code, times_days, pfe_star))

    # Overlay plots across assets (discount-to-0 series used for CVA)
    if len(asset_codes) > 1:
        plot_overlay("EE* overlay (discount-to-0)", overlay_ee)
        plot_overlay("PFE* overlay (discount-to-0)", overlay_pfe)

        print("CVA ranking (highest to lowest):")
        for k, v in sorted(cvas.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {k:40s}  {v:,.8f}")

    plt.show()


if __name__ == "__main__":
    main()
