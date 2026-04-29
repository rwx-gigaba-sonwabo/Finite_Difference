"""
GBM FX CALIBRATION RUNNER
==========================

Runs the GBM FX implied calibration for one or more FX pairs from a
RiskFlow JSON market-data file, compares the results against the
GBMAssetPriceTSModelParameters already stored in that file (i.e. what
RiskFlow's own bootstrapper produced), and optionally exports everything
to Excel or CSV.

USAGE
-----
    # Calibrate all pairs found in the JSON
    python gbm_fx_calibration_runner.py market_data.json

    # Calibrate specific pairs only
    python gbm_fx_calibration_runner.py market_data.json --pairs EUR AUD USD

    # Export to Excel
    python gbm_fx_calibration_runner.py market_data.json --export fx_calib.xlsx

    # Export to CSV folder
    python gbm_fx_calibration_runner.py market_data.json --csv-dir ./output

    # Combined
    python gbm_fx_calibration_runner.py market_data.json \\
        --pairs EUR AUD --export fx_calib.xlsx --csv-dir ./output

WHAT THE COMPARISON SHOWS
--------------------------
    Abs_Diff      = Calibrated_Vol  - RiskFlow_Vol          (vol points)
    Rel_Diff_Pct  = 100 * Abs_Diff / RiskFlow_Vol           (%)

    Columns reference the stored GBMAssetPriceTSModelParameters.{currency}
    entries in the JSON's Price Factors section.  If those entries are absent
    (e.g. the JSON was never passed through RiskFlow's bootstrapper), the
    comparison section is skipped.
"""

import sys
import os
import argparse
import json
import textwrap

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Locate and import the calibration module regardless of working directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from gbm_fx_implied_calibration import (
    bootstrap_fx_from_json,
    compare_with_riskflow,
    print_calibration_summary,
    print_comparison_tables,
    export_to_excel,
    export_to_csv,
)


# ===========================================================================
#  Core runner
# ===========================================================================

def run(json_path, fx_pairs=None, export_xlsx=None, csv_dir=None, verbose=True):
    """
    Calibrate GBM FX vol curves, compare against RiskFlow, and export.

    Parameters
    ----------
    json_path : str
        Path to RiskFlow JSON market-data file.
    fx_pairs : list of str, optional
        Currency names to calibrate, e.g. ['EUR', 'AUD'].
        None → calibrate every GBMAssetPriceTSModelPrices entry.
    export_xlsx : str, optional
        Output Excel file path.
    csv_dir : str, optional
        Output directory for CSV files.
    verbose : bool
        Whether to print detailed per-expiry tables.

    Returns
    -------
    calibrated : dict
        Keyed by currency; each value is the dict returned by
        bootstrap_fx_from_json.
    comparisons : dict
        Keyed by currency; each value is a pd.DataFrame.
    """
    # ── Step 1: validate file ─────────────────────────────────────────────
    if not os.path.isfile(json_path):
        print(f"ERROR: File not found: {json_path}")
        sys.exit(1)

    # ── Step 2: calibrate ─────────────────────────────────────────────────
    if fx_pairs:
        calibrated = {}
        for pair in fx_pairs:
            result = bootstrap_fx_from_json(json_path, fx_name=pair, verbose=verbose)
            calibrated.update(result)
    else:
        calibrated = bootstrap_fx_from_json(json_path, fx_name=None, verbose=verbose)

    if not calibrated:
        print("\nNo FX pairs calibrated — check the JSON Market Prices section.")
        return {}, {}

    # ── Step 3: summary table ─────────────────────────────────────────────
    print_calibration_summary(calibrated)

    # ── Step 4: compare against RiskFlow stored params ───────────────────
    comparisons = compare_with_riskflow(calibrated, json_path, verbose=verbose)

    if comparisons:
        print_comparison_tables(comparisons)
        _print_aggregate_comparison(comparisons)
    else:
        print("\n  No stored GBMAssetPriceTSModelParameters found for comparison.")
        print("  (Run the file through RiskFlow's bootstrapper first to populate them.)")

    # ── Step 5: export ────────────────────────────────────────────────────
    if export_xlsx:
        try:
            export_to_excel(calibrated, comparisons, export_xlsx)
        except Exception as exc:
            print(f"\n  WARNING: Excel export failed ({exc})")
            print("  Install openpyxl with:  pip install openpyxl")

    if csv_dir:
        export_to_csv(calibrated, comparisons, csv_dir)

    return calibrated, comparisons


# ---------------------------------------------------------------------------
# Aggregate comparison banner
# ---------------------------------------------------------------------------

def _print_aggregate_comparison(comparisons):
    """Print a cross-currency summary of differences."""
    print("\n" + "=" * 70)
    print("AGGREGATE COMPARISON  —  ALL PAIRS")
    print("=" * 70)
    print(f"  {'FX Pair':<20s}  {'Points':>6s}  {'Max|Abs|':>10s}  "
          f"{'Mean|Abs|':>10s}  {'Max|Rel%|':>10s}  {'Mean|Rel%|':>11s}")
    print("  " + "─" * 74)

    all_abs = []
    all_rel = []
    for currency, df in comparisons.items():
        ma  = df['Abs_Diff'].abs().max()
        mn  = df['Abs_Diff'].abs().mean()
        mr  = df['Rel_Diff_Pct'].abs().max()
        mnr = df['Rel_Diff_Pct'].abs().mean()
        all_abs.append(ma)
        all_rel.append(mr)
        print(f"  {currency:<20s}  {len(df):>6d}  {ma:>10.6f}  "
              f"{mn:>10.6f}  {mr:>10.4f}%  {mnr:>10.4f}%")

    print("  " + "─" * 74)
    if all_abs:
        print(f"  {'OVERALL':<20s}  {'':>6s}  {max(all_abs):>10.6f}  "
              f"{'':>10s}  {max(all_rel):>10.4f}%")
    print("=" * 70)


# ===========================================================================
#  Batch calibration helper (callable from other scripts)
# ===========================================================================

def calibrate_multiple(json_path, currencies, verbose=False):
    """
    Convenience wrapper: calibrate a list of currencies and return a
    combined DataFrame ready for downstream analysis.

    Parameters
    ----------
    json_path : str
    currencies : list of str
    verbose : bool

    Returns
    -------
    pd.DataFrame
        Columns: FX_Pair, Expiry, Calibrated_Vol, RiskFlow_Vol,
                 Abs_Diff, Rel_Diff_Pct
        (RiskFlow_Vol and difference columns are NaN when no stored params.)
    """
    calibrated = {}
    for ccy in currencies:
        r = bootstrap_fx_from_json(json_path, fx_name=ccy, verbose=verbose)
        calibrated.update(r)

    comparisons = compare_with_riskflow(calibrated, json_path, verbose=verbose)

    rows = []
    for ccy, p in calibrated.items():
        for exp, vol in p['Vol']:
            row = {
                'FX_Pair': ccy,
                'Expiry': exp,
                'Calibrated_Vol': vol,
                'RiskFlow_Vol': np.nan,
                'Abs_Diff': np.nan,
                'Rel_Diff_Pct': np.nan,
            }
            if ccy in comparisons:
                df_cmp = comparisons[ccy]
                # find closest stored expiry
                match = df_cmp[np.isclose(df_cmp['Expiry'], exp, atol=1e-6)]
                if not match.empty:
                    row['RiskFlow_Vol'] = match.iloc[0]['RiskFlow_Vol']
                    row['Abs_Diff'] = match.iloc[0]['Abs_Diff']
                    row['Rel_Diff_Pct'] = match.iloc[0]['Rel_Diff_Pct']
            rows.append(row)

    return pd.DataFrame(rows)


# ===========================================================================
#  Self-test: runs against a synthetic in-memory JSON
# ===========================================================================

def _self_test_runner():
    """
    Build a minimal synthetic JSON, run the calibration, and verify
    that the results are self-consistent.
    """
    import tempfile, json

    print("=" * 70)
    print("RUNNER SELF-TEST: synthetic JSON round-trip")
    print("=" * 70)

    # Build a small synthetic FXVol surface for a fake 'TST' currency
    # Moneyness: [0.9, 1.0, 1.1]   Expiry: [0.25, 0.5, 1.0, 2.0]
    moneyness = [0.9, 1.0, 1.1]
    expiries  = [0.25, 0.5, 1.0, 2.0]
    atm_base  = [0.10, 0.11, 0.12, 0.13]  # gentle upward slope → no correction

    surface_rows = []
    for i, exp in enumerate(expiries):
        atm = atm_base[i]
        # Simple skew: ±1 moneyness unit adds ±0.01 vol
        for m in moneyness:
            vol = atm + (1.0 - m) * 0.01
            surface_rows.append([m, exp, vol])

    synthetic_json = {
        "System Parameters": {"Base_Date": "2025-01-01"},
        "Price Factors": {
            "FXVol.TST.USD": {
                "Surface": {
                    "_type": "Curve",
                    "array": surface_rows
                },
                "Moneyness_Rule": "Sticky_Moneyness"
            }
        },
        "Market Prices": {
            "GBMAssetPriceTSModelPrices.TST": {
                "instrument": {
                    "Asset_Price_Volatility": "TST.USD"
                },
                "Children": []
            }
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(synthetic_json, f)
        tmp_path = f.name

    try:
        calibrated = bootstrap_fx_from_json(tmp_path, verbose=True)
        assert 'TST' in calibrated, "Expected 'TST' in calibrated results"
        assert len(calibrated['TST']['Vol']) == len(expiries), \
            f"Expected {len(expiries)} expiry points"

        # Verify ATM vols are recovered at moneyness=1.0
        for i, (exp, avg_vol) in enumerate(calibrated['TST']['Vol']):
            assert abs(exp - expiries[i]) < 1e-9, f"Expiry mismatch at index {i}"
            # avg_vol should equal atm_base[i] (no correction needed for rising vol)
            assert abs(avg_vol - atm_base[i]) < 1e-9, \
                f"Vol mismatch at expiry {exp}: got {avg_vol}, expected {atm_base[i]}"

        print("\n  PASS: all calibrated vols match ATM base vols")

        # Quick check: calibrate_multiple returns a DataFrame
        df = calibrate_multiple(tmp_path, ['TST'], verbose=False)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(expiries)
        print(f"  PASS: calibrate_multiple returned DataFrame with {len(df)} rows")

    finally:
        os.unlink(tmp_path)

    print("\n" + "=" * 70)
    print("RUNNER SELF-TEST PASSED")
    print("=" * 70)


# ===========================================================================
#  CLI
# ===========================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        prog='gbm_fx_calibration_runner.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Risk-neutral GBM FX calibration runner.

            Calibrates GBMAssetPriceTSModelParameters for FX rates from
            FXVol surfaces stored in a RiskFlow JSON market-data file,
            then compares against the parameters already stored in that
            file (if present) and reports absolute and relative differences.
        """),
        epilog=textwrap.dedent("""\
            Examples:
              %(prog)s market_data.json
              %(prog)s market_data.json --pairs EUR AUD
              %(prog)s market_data.json --export fx_calib.xlsx
              %(prog)s market_data.json --csv-dir ./output
              %(prog)s market_data.json --pairs EUR --export out.xlsx --csv-dir ./out
              %(prog)s --test
        """),
    )
    parser.add_argument(
        'json_path',
        nargs='?',
        help='Path to a RiskFlow JSON market-data file.',
    )
    parser.add_argument(
        '--pairs',
        nargs='+',
        metavar='CCY',
        help='Currency names to calibrate (e.g. EUR AUD USD). '
             'Omit to calibrate all pairs found.',
    )
    parser.add_argument(
        '--export',
        metavar='FILE.xlsx',
        help='Export results to an Excel workbook.',
    )
    parser.add_argument(
        '--csv-dir',
        metavar='DIR',
        help='Export results as CSV files in this directory.',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress per-expiry detail tables (summary still printed).',
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run the built-in self-test (no JSON file required).',
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.test:
        # Run module-level self-tests first
        from gbm_fx_implied_calibration import _self_test
        _self_test()
        print()
        _self_test_runner()
        sys.exit(0)

    if not args.json_path:
        parser.print_help()
        sys.exit(1)

    run(
        json_path=args.json_path,
        fx_pairs=args.pairs,
        export_xlsx=args.export,
        csv_dir=args.csv_dir,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
