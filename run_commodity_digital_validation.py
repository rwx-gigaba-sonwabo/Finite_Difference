"""
run_commodity_digital_validation.py
======================================
CSV-driven base (t0) valuation of a European commodity digital option,
built for running quick validation checks against an external pricing
system (e.g. Front Arena).

Market data (forward curve, discount curve) is loaded from CSV files --
see market_data_csv/commodity_forward_curve.csv and
market_data_csv/discount_curve_naca.csv for the expected two-column
(date, value) format.

To run a validation check: edit the CONFIG block below directly, then
run this file. Everything you'd want to change between runs -- val date,
curve files, trade terms, the vol skew, the external price to diff
against -- is a labelled field on the Config dataclass, grouped by
section. No command-line flags to remember.

Every run appends one row to results/commodity_digital_validation_log.csv
(created with a header on first run) so a sequence of validation checks
builds up into a comparable log. Set ``external_price`` to have the
script compute and report the diff against a price read off the system
you're validating.

Examples
--------
Quick sanity run with the bundled sample curves and default trade terms:
leave CONFIG as-is and run the file.

Point at your own curves, override trade terms, and diff against an
external system's price::

    CONFIG = Config(
        val_date=date(2025, 6, 2),
        forward_curve=Path("market_data_csv/brent_fwd_2025-06-02.csv"),
        discount_curve=Path("market_data_csv/zar_swap_naca_2025-06-02.csv"),
        maturity=date(2026, 6, 2), strike=82.5, is_call=True,
        digital_type="cash", payout=100.0, vol=0.28,
        spot_days=2, settle_days=2, external_price=41.90,
    )

With a skew adjustment (tight call/put-spread replication using a
(strike, vol) smile CSV -- see market_data_csv/vol_skew.csv), instead of
a flat scalar ``vol``::

    CONFIG = Config(
        strike=82.5, vol_skew_curve=Path("market_data_csv/vol_skew.csv"),
        apply_skew_adjustment=True, external_price=41.90,
    )

(``vol_skew_curve`` without ``apply_skew_adjustment`` uses the smile for
a plain vol-at-strike lookup instead of the flat ``vol`` number, without
the full replication -- the unchanged, pre-skew-flag behaviour.)
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from base_valuation.market_data_io import (
    build_forward_curve_from_csv,
    build_yield_curve_from_csv,
    build_vol_skew_from_csv,
)
from base_valuation.commodity_digital_option import (
    value_commodity_digital_option,
    bump_and_reprice_commodity_digital_option,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_FORWARD_CSV = REPO_ROOT / "market_data_csv" / "commodity_forward_curve.csv"
DEFAULT_DISCOUNT_CSV = REPO_ROOT / "market_data_csv" / "discount_curve_naca.csv"
DEFAULT_RESULTS_LOG = REPO_ROOT / "results" / "commodity_digital_validation_log.csv"


@dataclass
class Config:
    # --- Valuation date --------------------------------------------------
    val_date: date = date(2025, 1, 2)

    # --- Forward price curve (CSV) ---------------------------------------
    forward_curve: Path = DEFAULT_FORWARD_CSV
    forward_date_col: str = "date"
    forward_value_col: str = "value"
    forward_interp: str = "linear"  # "linear" | "forward_price"

    # --- Discount curve (CSV, NACA by default) ----------------------------
    discount_curve: Path = DEFAULT_DISCOUNT_CSV
    discount_date_col: str = "date"
    discount_value_col: str = "value"
    rate_convention: str = "NACA"  # "NACA" | "NACC"
    compounding_freq: int = 1  # 1=NACA, 2=NACS, 4=NACQ, 12=NACM
    discount_interp: str = "hermite_rt"
    # "linear" | "linear_rt" | "reciprocal_time" | "hermite_rt" | "stitched_linear_hermite_rt"

    # --- Trade terms -------------------------------------------------------
    maturity: date = date(2026, 1, 2)
    strike: float = 80.0
    is_call: bool = True
    digital_type: str = "cash"  # "cash" | "asset"
    payout: float = 100.0
    vol: float = 0.30  # flat scalar vol; ignored if vol_skew_curve is set
    spot_days: int = 0
    spot_calendar: str = "USD"
    settle_days: int = 2
    settlement_calendar: str = "USD"
    day_count: str = "ACT/365"

    # --- Volatility skew (optional) ----------------------------------------
    vol_skew_curve: Path | None = None  # (strike, vol) CSV; used instead of `vol` when set
    vol_skew_strike_col: str = "strike"
    vol_skew_vol_col: str = "vol"
    apply_skew_adjustment: bool = False
    # Tight call/put-spread replication using the full local slope of
    # vol_skew_curve at the strike, instead of a plain vol-at-strike lookup.
    # Requires vol_skew_curve.
    skew_strike_bump_frac: float = 1e-3
    # Relative strike bump for the replicating spread (only used with apply_skew_adjustment).

    # --- Validation / output -------------------------------------------------
    trade_id: str = ""  # free-text label written to the results log
    external_price: float | None = None  # price from the system being validated, for a diff report
    results_log: Path = DEFAULT_RESULTS_LOG
    no_log: bool = False  # skip appending to the results log

    def __post_init__(self) -> None:
        if self.apply_skew_adjustment and self.vol_skew_curve is None:
            raise ValueError("apply_skew_adjustment=True requires vol_skew_curve to be set")


# ===========================================================================
# EDIT THIS BLOCK TO CONFIGURE A RUN
# ===========================================================================
CONFIG = Config()


def append_results_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main(cfg: Config = CONFIG) -> None:
    fwd_curve = build_forward_curve_from_csv(
        cfg.forward_curve, cfg.val_date,
        date_col=cfg.forward_date_col, price_col=cfg.forward_value_col,
        day_count=cfg.day_count, interpolation=cfg.forward_interp,
    )
    disc_curve = build_yield_curve_from_csv(
        cfg.discount_curve, cfg.val_date,
        date_col=cfg.discount_date_col, rate_col=cfg.discount_value_col,
        rate_convention=cfg.rate_convention, compounding_freq=cfg.compounding_freq,
        day_count=cfg.day_count, interpolation=cfg.discount_interp,
    )
    vol_skew = None
    if cfg.vol_skew_curve is not None:
        vol_skew = build_vol_skew_from_csv(
            cfg.vol_skew_curve,
            strike_col=cfg.vol_skew_strike_col, vol_col=cfg.vol_skew_vol_col,
        )

    common_kwargs = dict(
        val_date=cfg.val_date, maturity_date=cfg.maturity, strike=cfg.strike,
        is_call=cfg.is_call, digital_type=cfg.digital_type, payout=cfg.payout,
        forward_curve=fwd_curve, discount_curve=disc_curve,
        vol=cfg.vol, vol_skew=vol_skew, apply_skew_adjustment=cfg.apply_skew_adjustment,
        skew_strike_bump_frac=cfg.skew_strike_bump_frac,
        spot_days=cfg.spot_days, spot_calendar=cfg.spot_calendar,
        settle_days=cfg.settle_days, settlement_calendar=cfg.settlement_calendar,
        day_count=cfg.day_count,
    )
    result = value_commodity_digital_option(**common_kwargs)
    bump = bump_and_reprice_commodity_digital_option(**common_kwargs)

    print("=" * 72)
    print(f"European Commodity Digital ({'Call' if cfg.is_call else 'Put'}, "
          f"{cfg.digital_type}-or-nothing)  @ {cfg.val_date}")
    print("=" * 72)
    print(f"  Forward curve:   {cfg.forward_curve}  (interp={cfg.forward_interp})")
    print(f"  Discount curve:  {cfg.discount_curve}  "
          f"(convention={cfg.rate_convention}, interp={cfg.discount_interp})")
    if vol_skew is not None:
        print(f"  Vol skew curve:  {cfg.vol_skew_curve}  "
              f"(apply_skew_adjustment={cfg.apply_skew_adjustment})")
    print(f"  Maturity / Strike / Payout: {cfg.maturity} / {cfg.strike} / {cfg.payout}")
    print(f"  Forward (spot-day adj.):    {result.forward:.6f}   Vol (at strike): {result.vol:.4%}")
    print(f"  T_opt / T_disc / DF:        {result.T_opt:.6f} / {result.T_disc:.6f} / {result.df:.8f}")
    print("-" * 72)
    print(f"  Price:              {result.price:>14.6f}")
    if result.analytic_delta is not None:
        print(f"  Analytic  Delta/Gamma/Vega/Theta: "
              f"{result.analytic_delta:.6f} / {result.analytic_gamma:.8f} / "
              f"{result.analytic_vega:.6f} / {result.analytic_theta:.6f}")
    else:
        print("  Analytic  Greeks:            n/a when skew-adjusted (see docstring)")
    print(f"  Bump      Delta/Gamma/Vega/Theta: "
          f"{bump['delta']:.6f} / {bump['gamma']:.8f} / "
          f"{bump['vega']:.6f} / {bump['theta']:.6f}")

    diff = diff_pct = None
    if cfg.external_price is not None:
        diff = result.price - cfg.external_price
        diff_pct = diff / cfg.external_price if cfg.external_price != 0.0 else float("nan")
        print("-" * 72)
        print(f"  External price:     {cfg.external_price:>14.6f}")
        print(f"  Diff (this - ext):  {diff:>14.6f}   ({diff_pct:+.4%})")
    print("=" * 72)

    if not cfg.no_log:
        row = {
            "trade_id": cfg.trade_id,
            "val_date": cfg.val_date.isoformat(),
            "maturity": cfg.maturity.isoformat(),
            "is_call": cfg.is_call,
            "digital_type": cfg.digital_type,
            "strike": cfg.strike,
            "payout": cfg.payout,
            "vol_input": cfg.vol,
            "vol_at_strike": result.vol,
            "vol_skew_curve_file": str(cfg.vol_skew_curve) if vol_skew is not None else "",
            "skew_adjusted": result.skew_adjusted,
            "forward_curve_file": str(cfg.forward_curve),
            "discount_curve_file": str(cfg.discount_curve),
            "rate_convention": cfg.rate_convention,
            "discount_interp": cfg.discount_interp,
            "forward": result.forward,
            "T_opt": result.T_opt,
            "T_disc": result.T_disc,
            "df": result.df,
            "price": result.price,
            "analytic_delta": result.analytic_delta,
            "analytic_gamma": result.analytic_gamma,
            "analytic_vega": result.analytic_vega,
            "analytic_theta": result.analytic_theta,
            "bump_delta": bump["delta"],
            "bump_gamma": bump["gamma"],
            "bump_vega": bump["vega"],
            "bump_theta": bump["theta"],
            "external_price": cfg.external_price,
            "diff": diff,
            "diff_pct": diff_pct,
        }
        append_results_log(cfg.results_log, row)
        print(f"Logged to {cfg.results_log}")


if __name__ == "__main__":
    main()
