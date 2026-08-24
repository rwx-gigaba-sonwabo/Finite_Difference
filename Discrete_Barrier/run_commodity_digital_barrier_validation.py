"""
run_commodity_digital_barrier_validation.py
==============================================
CSV-driven base (t0) valuation of a commodity digital barrier option
(One-Touch / No-Touch, single or double barrier, continuous or discrete
monitoring), built for running quick validation checks against an
external pricing system (e.g. Front Arena).

Market data (forward curve, discount curve) is loaded from CSV files --
see market_data_csv/commodity_forward_curve.csv and
market_data_csv/discount_curve_naca.csv for the expected two-column
(date, value) format. Everything else is controlled by command-line
flags so repeated validation runs don't require editing this file.

Every run appends one row to
results/commodity_digital_barrier_validation_log.csv (created with a
header on first run). Pass --external-price to have the script compute
and report the diff against a price read off the system you're validating.

Examples
--------
Quick sanity run with the bundled sample curves and default (continuous,
up-and-in, cash) trade terms::

    python run_commodity_digital_barrier_validation.py

Discretely monitored down-and-out, own curves, diff against an external
system's price::

    python run_commodity_digital_barrier_validation.py \\
        --val-date 2025-06-02 \\
        --forward-curve market_data_csv/brent_fwd_2025-06-02.csv \\
        --discount-curve market_data_csv/zar_swap_naca_2025-06-02.csv \\
        --maturity 2026-06-02 --lower-barrier 65 --touch out \\
        --monitoring discrete --cost-of-carry 0.035 \\
        --vol 0.28 --external-price 12.40

See ``python run_commodity_digital_barrier_validation.py --help`` for the
full flag list.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path

from base_valuation.market_data_io import (
    build_forward_curve_from_csv,
    build_yield_curve_from_csv,
)
from base_valuation.commodity_digital_barrier_option import (
    value_commodity_digital_barrier_option,
    bump_and_reprice_commodity_digital_barrier_option,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_FORWARD_CSV = REPO_ROOT / "market_data_csv" / "commodity_forward_curve.csv"
DEFAULT_DISCOUNT_CSV = REPO_ROOT / "market_data_csv" / "discount_curve_naca.csv"
RESULTS_LOG = REPO_ROOT / "results" / "commodity_digital_barrier_validation_log.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate a commodity digital barrier option against an external system.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--val-date", type=date.fromisoformat, default=date(2025, 1, 2))

    g_fwd = p.add_argument_group("Forward price curve (CSV)")
    g_fwd.add_argument("--forward-curve", type=Path, default=DEFAULT_FORWARD_CSV)
    g_fwd.add_argument("--forward-date-col", default="date")
    g_fwd.add_argument("--forward-value-col", default="value")
    g_fwd.add_argument("--forward-interp", default="linear", choices=["linear", "forward_price"])

    g_disc = p.add_argument_group("Discount curve (CSV, NACA by default)")
    g_disc.add_argument("--discount-curve", type=Path, default=DEFAULT_DISCOUNT_CSV)
    g_disc.add_argument("--discount-date-col", default="date")
    g_disc.add_argument("--discount-value-col", default="value")
    g_disc.add_argument("--rate-convention", default="NACA", choices=["NACA", "NACC"])
    g_disc.add_argument("--compounding-freq", type=int, default=1,
                         help="1=NACA, 2=NACS, 4=NACQ, 12=NACM")
    g_disc.add_argument(
        "--discount-interp", default="hermite_rt",
        choices=["linear", "linear_rt", "reciprocal_time", "hermite_rt", "stitched_linear_hermite_rt"],
    )

    g_trade = p.add_argument_group("Trade terms")
    g_trade.add_argument("--maturity", type=date.fromisoformat, default=date(2026, 1, 2))
    g_trade.add_argument("--monitoring-start", type=date.fromisoformat, default=None,
                          help="Defaults to --val-date")
    g_trade.add_argument("--upper-barrier", type=float, default=95.0)
    g_trade.add_argument("--lower-barrier", type=float, default=None)
    g_trade.add_argument("--touch", default="in", choices=["in", "out"])
    g_trade.add_argument("--digital-type", default="cash", choices=["cash", "asset"])
    g_trade.add_argument("--payout", type=float, default=200.0)
    g_trade.add_argument("--vol", type=float, default=0.30)
    g_trade.add_argument("--monitoring", default="continuous", choices=["continuous", "discrete"])
    g_trade.add_argument("--monitoring-calendar", default="USD")
    g_trade.add_argument("--cost-of-carry", type=float, default=None,
                          help="Flat scalar b; omit to derive from the forward curve's own term structure")
    g_trade.add_argument("--spot-days", type=int, default=0)
    g_trade.add_argument("--spot-calendar", default="USD")
    g_trade.add_argument("--settle-days", type=int, default=2)
    g_trade.add_argument("--settlement-calendar", default="USD")
    g_trade.add_argument("--day-count", default="ACT/365")
    g_trade.add_argument("--n-mc-paths", type=int, default=20_000)
    g_trade.add_argument("--mc-seed", type=int, default=0)

    g_out = p.add_argument_group("Validation / output")
    g_out.add_argument("--trade-id", default="", help="Free-text label written to the results log")
    g_out.add_argument("--external-price", type=float, default=None,
                        help="Price from the system being validated, for a diff report")
    g_out.add_argument("--results-log", type=Path, default=RESULTS_LOG)
    g_out.add_argument("--no-log", action="store_true", help="Skip appending to the results log")
    g_out.add_argument("--skip-bump", action="store_true",
                        help="Skip the bump-and-reprice cross-check (saves time on large MC runs)")

    args = p.parse_args()
    if args.upper_barrier is None and args.lower_barrier is None:
        p.error("at least one of --upper-barrier / --lower-barrier is required")
    if args.monitoring_start is None:
        args.monitoring_start = args.val_date
    return args


def append_results_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()

    fwd_curve = build_forward_curve_from_csv(
        args.forward_curve, args.val_date,
        date_col=args.forward_date_col, price_col=args.forward_value_col,
        day_count=args.day_count, interpolation=args.forward_interp,
    )
    disc_curve = build_yield_curve_from_csv(
        args.discount_curve, args.val_date,
        date_col=args.discount_date_col, rate_col=args.discount_value_col,
        rate_convention=args.rate_convention, compounding_freq=args.compounding_freq,
        day_count=args.day_count, interpolation=args.discount_interp,
    )

    common_kwargs = dict(
        val_date=args.val_date, maturity_date=args.maturity,
        upper_barrier=args.upper_barrier, lower_barrier=args.lower_barrier,
        touch=args.touch, digital_type=args.digital_type, payout=args.payout,
        forward_curve=fwd_curve, discount_curve=disc_curve, vol=args.vol,
        monitoring=args.monitoring, monitoring_calendar=args.monitoring_calendar,
        cost_of_carry=args.cost_of_carry,
        spot_days=args.spot_days, spot_calendar=args.spot_calendar,
        settle_days=args.settle_days, settlement_calendar=args.settlement_calendar,
        day_count=args.day_count, n_mc_paths=args.n_mc_paths, mc_seed=args.mc_seed,
    )

    result = value_commodity_digital_barrier_option(**common_kwargs)
    bump = None
    if not args.skip_bump:
        bump = bump_and_reprice_commodity_digital_barrier_option(**common_kwargs)

    barrier_desc = []
    if args.upper_barrier is not None:
        barrier_desc.append(f"upper={args.upper_barrier}")
    if args.lower_barrier is not None:
        barrier_desc.append(f"lower={args.lower_barrier}")

    print("=" * 72)
    print(f"Commodity Digital Barrier ({args.touch.title()}-Touch, "
          f"{', '.join(barrier_desc)}, {args.digital_type}, {args.monitoring})  @ {args.val_date}")
    print("=" * 72)
    print(f"  Forward curve:   {args.forward_curve}  (interp={args.forward_interp})")
    print(f"  Discount curve:  {args.discount_curve}  "
          f"(convention={args.rate_convention}, interp={args.discount_interp})")
    print(f"  Maturity / Payout: {args.maturity} / {args.payout}")
    print(f"  Pricing method:              {result.pricing_method}")
    print(f"  Forward / Implied spot / b:  {result.forward:.6f} / "
          f"{result.implied_spot:.6f} / {result.cost_of_carry:.6%}")
    print(f"  T_opt / T_disc / DF:         {result.T_opt:.6f} / {result.T_disc:.6f} / {result.df:.8f}")
    if result.mc_stderr is not None:
        print(f"  MC standard error:           {result.mc_stderr:.6f}")
    print("-" * 72)
    print(f"  Price:              {result.price:>14.6f}")
    if result.analytic_delta is not None:
        print(f"  Analytic  Delta/Vega/Theta: "
              f"{result.analytic_delta:.6f} / {result.analytic_vega:.6f} / {result.analytic_theta:.6f}")
    else:
        print("  Analytic  Greeks:            n/a for this configuration (see docstring)")
    if bump is not None:
        print(f"  Bump      Delta/Gamma/Vega/Theta: "
              f"{bump['delta']:.6f} / {bump['gamma']:.8f} / "
              f"{bump['vega']:.6f} / {bump['theta']:.6f}")

    diff = diff_pct = None
    if args.external_price is not None:
        diff = result.price - args.external_price
        diff_pct = diff / args.external_price if args.external_price != 0.0 else float("nan")
        print("-" * 72)
        print(f"  External price:     {args.external_price:>14.6f}")
        print(f"  Diff (this - ext):  {diff:>14.6f}   ({diff_pct:+.4%})")
    print("=" * 72)

    if not args.no_log:
        row = {
            "trade_id": args.trade_id,
            "val_date": args.val_date.isoformat(),
            "maturity": args.maturity.isoformat(),
            "upper_barrier": args.upper_barrier,
            "lower_barrier": args.lower_barrier,
            "touch": args.touch,
            "digital_type": args.digital_type,
            "monitoring": args.monitoring,
            "payout": args.payout,
            "vol": args.vol,
            "forward_curve_file": str(args.forward_curve),
            "discount_curve_file": str(args.discount_curve),
            "rate_convention": args.rate_convention,
            "discount_interp": args.discount_interp,
            "pricing_method": result.pricing_method,
            "forward": result.forward,
            "implied_spot": result.implied_spot,
            "cost_of_carry": result.cost_of_carry,
            "T_opt": result.T_opt,
            "T_disc": result.T_disc,
            "df": result.df,
            "mc_stderr": result.mc_stderr,
            "price": result.price,
            "analytic_delta": result.analytic_delta,
            "analytic_vega": result.analytic_vega,
            "analytic_theta": result.analytic_theta,
            "bump_delta": bump["delta"] if bump else None,
            "bump_gamma": bump["gamma"] if bump else None,
            "bump_vega": bump["vega"] if bump else None,
            "bump_theta": bump["theta"] if bump else None,
            "external_price": args.external_price,
            "diff": diff,
            "diff_pct": diff_pct,
        }
        append_results_log(args.results_log, row)
        print(f"Logged to {args.results_log}")


if __name__ == "__main__":
    main()
