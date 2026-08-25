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

To change the market data or trade terms, edit the CONFIG block below
directly -- every field is labelled and grouped by section (market data,
digital option terms, barrier terms).

Sections
--------
1. Config            market data pillars + trade terms (edit here)
2. Value             build curves -> price + Greeks (analytic & bump)
3. Report            console summary
"""
from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class Config:
    # --- Market data: forward price curve -----------------------------
    val_date: date = date(2025, 1, 2)
    fwd_curve_dates: list[date] = field(default_factory=lambda: [
        date(2025, 4, 2), date(2025, 7, 2), date(2026, 1, 2),
        date(2027, 1, 2), date(2030, 1, 2),
    ])
    fwd_curve_prices: list[float] = field(default_factory=lambda: [
        78.0, 79.0, 80.0, 82.0, 88.0,  # USD / barrel
    ])

    # --- Market data: discount curve -----------------------------------
    disc_curve_dates: list[date] = field(default_factory=lambda: [
        date(2025, 4, 2), date(2026, 1, 2), date(2027, 1, 2), date(2030, 1, 2),
    ])
    disc_curve_rates: list[float] = field(default_factory=lambda: [
        0.0500, 0.0500, 0.0500, 0.0500,
    ])

    day_count: str = "ACT/365"
    vol: float = 0.30  # scalar Black implied volatility

    # --- European digital option terms ----------------------------------
    maturity_date: date = date(2026, 1, 2)
    strike: float = 80.0
    payout: float = 100.0

    # --- Digital barrier option terms (One-Touch Up, continuous) --------
    barrier_level: float = 95.0

    @property
    def monitor_start(self) -> date:
        return self.val_date


# ===========================================================================
# EDIT THIS BLOCK TO CONFIGURE A RUN
# ===========================================================================
CONFIG = Config()


def main(cfg: Config = CONFIG) -> None:
    fwd_curve = build_commodity_forward_curve(
        val_date=cfg.val_date, curve_dates=cfg.fwd_curve_dates, prices=cfg.fwd_curve_prices,
        day_count=cfg.day_count, interpolation="forward_price",
    )
    disc_curve = build_yield_curve(
        val_date=cfg.val_date, curve_dates=cfg.disc_curve_dates, rates=cfg.disc_curve_rates,
        day_count=cfg.day_count, interpolation="linear",
    )

    # --- European digital ---------------------------------------------
    digital = value_commodity_digital_option(
        val_date=cfg.val_date, maturity_date=cfg.maturity_date, strike=cfg.strike,
        is_call=True, digital_type="cash", payout=cfg.payout,
        forward_curve=fwd_curve, discount_curve=disc_curve, vol=cfg.vol,
        spot_days=0, settle_days=2,
    )
    digital_bump = bump_and_reprice_commodity_digital_option(
        val_date=cfg.val_date, maturity_date=cfg.maturity_date, strike=cfg.strike,
        is_call=True, digital_type="cash", payout=cfg.payout,
        forward_curve=fwd_curve, discount_curve=disc_curve, vol=cfg.vol,
        spot_days=0, settle_days=2,
    )

    print("=" * 70)
    print(f"European Commodity Digital (cash-or-nothing call)  @ {cfg.val_date}")
    print("=" * 70)
    print(f"  Maturity / Strike / Payout: {cfg.maturity_date} / {cfg.strike} / {cfg.payout}")
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
        val_date=cfg.val_date, maturity_date=cfg.maturity_date,
        upper_barrier=cfg.barrier_level, lower_barrier=None, touch="in",
        digital_type="cash", payout=cfg.payout,
        forward_curve=fwd_curve, discount_curve=disc_curve, vol=cfg.vol,
        monitoring="continuous", cost_of_carry=None,  # curve-implied
        spot_days=0, settle_days=2,
    )
    barrier_bump = bump_and_reprice_commodity_digital_barrier_option(
        val_date=cfg.val_date, maturity_date=cfg.maturity_date,
        upper_barrier=cfg.barrier_level, lower_barrier=None, touch="in",
        digital_type="cash", payout=cfg.payout,
        forward_curve=fwd_curve, discount_curve=disc_curve, vol=cfg.vol,
        monitoring="continuous", cost_of_carry=None,
        spot_days=0, settle_days=2,
    )

    print()
    print("=" * 70)
    print(f"Commodity Digital Barrier (One-Touch Up, continuous)  @ {cfg.val_date}")
    print("=" * 70)
    print(f"  Maturity / Barrier / Payout: {cfg.maturity_date} / {cfg.barrier_level} / {cfg.payout}")
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
