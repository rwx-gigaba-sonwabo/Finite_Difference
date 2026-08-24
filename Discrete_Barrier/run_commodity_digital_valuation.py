"""
run_commodity_digital_valuation.py
=====================================
Base (t0) valuation of a European commodity digital option and a
commodity digital barrier option using the base_valuation.* modules
(commodity_curve, yield_curve, commodity_digital_option,
commodity_digital_barrier_option).

Unlike the scenario-cube instruments this is deterministic, single-curve
base valuation only -- no scenario cube, no Monte Carlo exposure, no XVA.
Volatility is a plain scalar float. It is the "what is this option worth
today" building block those richer workflows are layered on top of.

Sections
--------
1. Market data      forward price curve & discount curve pillars
2. Trade terms       expiry / strike / barrier / payout
3. Value             build curves -> price + Greeks (analytic & bump)
4. Report            console summary
"""
from __future__ import annotations

from datetime import date

from base_valuation.commodity_curve import build_commodity_forward_curve
from base_valuation.yield_curve import build_yield_curve
from base_valuation.commodity_digital_option import (
    value_commodity_digital_option,
    bump_and_reprice_commodity_digital_option,
)
from base_valuation.commodity_digital_barrier_option import (
    value_commodity_digital_barrier_option,
    bump_and_reprice_commodity_digital_barrier_option,
)


# ===========================================================================
# 1. MARKET DATA
# ===========================================================================
VAL_DATE = date(2025, 1, 2)

FWD_CURVE_DATES = [
    date(2025, 4, 2), date(2025, 7, 2), date(2026, 1, 2),
    date(2027, 1, 2), date(2030, 1, 2),
]
FWD_CURVE_PRICES = [78.0, 79.0, 80.0, 82.0, 88.0]   # USD / barrel

DISC_CURVE_DATES = [date(2025, 4, 2), date(2026, 1, 2), date(2027, 1, 2), date(2030, 1, 2)]
DISC_CURVE_RATES = [0.0500, 0.0500, 0.0500, 0.0500]

DAY_COUNT = "ACT/365"
VOL = 0.30  # scalar Black implied volatility

# ===========================================================================
# 2. TRADE TERMS
# ===========================================================================
MATURITY_DATE = date(2026, 1, 2)
STRIKE = 80.0
PAYOUT = 100.0

BARRIER_LEVEL = 95.0
MONITOR_START = VAL_DATE


def main() -> None:
    fwd_curve = build_commodity_forward_curve(
        val_date=VAL_DATE, curve_dates=FWD_CURVE_DATES, prices=FWD_CURVE_PRICES,
        day_count=DAY_COUNT, interpolation="forward_price",
    )
    disc_curve = build_yield_curve(
        val_date=VAL_DATE, curve_dates=DISC_CURVE_DATES, rates=DISC_CURVE_RATES,
        day_count=DAY_COUNT, interpolation="linear",
    )

    # --- European digital ---------------------------------------------
    digital = value_commodity_digital_option(
        val_date=VAL_DATE, maturity_date=MATURITY_DATE, strike=STRIKE,
        is_call=True, digital_type="cash", payout=PAYOUT,
        forward_curve=fwd_curve, discount_curve=disc_curve, vol=VOL,
        spot_days=0, settle_days=2,
    )
    digital_bump = bump_and_reprice_commodity_digital_option(
        val_date=VAL_DATE, maturity_date=MATURITY_DATE, strike=STRIKE,
        is_call=True, digital_type="cash", payout=PAYOUT,
        forward_curve=fwd_curve, discount_curve=disc_curve, vol=VOL,
        spot_days=0, settle_days=2,
    )

    print("=" * 70)
    print(f"European Commodity Digital (cash-or-nothing call)  @ {VAL_DATE}")
    print("=" * 70)
    print(f"  Maturity / Strike / Payout: {MATURITY_DATE} / {STRIKE} / {PAYOUT}")
    print(f"  Forward (spot-day adj.):    {digital.forward:.4f}   Vol: {digital.vol:.2%}")
    print(f"  T_opt / T_disc / DF:        {digital.T_opt:.4f} / {digital.T_disc:.4f} / {digital.df:.6f}")
    print("-" * 70)
    print(f"  Price:              {digital.price:>12.4f}")
    print(f"  Analytic  Delta/Gamma/Vega/Theta: "
          f"{digital.analytic_delta:.4f} / {digital.analytic_gamma:.6f} / "
          f"{digital.analytic_vega:.4f} / {digital.analytic_theta:.4f}")
    print(f"  Bump      Delta/Gamma/Vega/Theta: "
          f"{digital_bump['delta']:.4f} / {digital_bump['gamma']:.6f} / "
          f"{digital_bump['vega']:.4f} / {digital_bump['theta']:.4f}")

    # --- Digital barrier (continuous, up-and-in) -----------------------
    barrier = value_commodity_digital_barrier_option(
        val_date=VAL_DATE, maturity_date=MATURITY_DATE,
        upper_barrier=BARRIER_LEVEL, lower_barrier=None, touch="in",
        digital_type="cash", payout=PAYOUT,
        forward_curve=fwd_curve, discount_curve=disc_curve, vol=VOL,
        monitoring="continuous", cost_of_carry=None,  # curve-implied
        spot_days=0, settle_days=2,
    )
    barrier_bump = bump_and_reprice_commodity_digital_barrier_option(
        val_date=VAL_DATE, maturity_date=MATURITY_DATE,
        upper_barrier=BARRIER_LEVEL, lower_barrier=None, touch="in",
        digital_type="cash", payout=PAYOUT,
        forward_curve=fwd_curve, discount_curve=disc_curve, vol=VOL,
        monitoring="continuous", cost_of_carry=None,
        spot_days=0, settle_days=2,
    )

    print()
    print("=" * 70)
    print(f"Commodity Digital Barrier (One-Touch Up, continuous)  @ {VAL_DATE}")
    print("=" * 70)
    print(f"  Maturity / Barrier / Payout: {MATURITY_DATE} / {BARRIER_LEVEL} / {PAYOUT}")
    print(f"  Pricing method:              {barrier.pricing_method}")
    print(f"  Forward / Implied spot / b:  {barrier.forward:.4f} / "
          f"{barrier.implied_spot:.4f} / {barrier.cost_of_carry:.4%}")
    print("-" * 70)
    print(f"  Price:              {barrier.price:>12.4f}")
    print(f"  Analytic  Delta/Vega/Theta: "
          f"{barrier.analytic_delta:.4f} / {barrier.analytic_vega:.4f} / {barrier.analytic_theta:.4f}")
    print(f"  Bump      Delta/Gamma/Vega/Theta: "
          f"{barrier_bump['delta']:.4f} / {barrier_bump['gamma']:.6f} / "
          f"{barrier_bump['vega']:.4f} / {barrier_bump['theta']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()