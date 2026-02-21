"""
Main runner for the Bjerksund-Stensland (forward/Black-76) American option
pricer.

Usage
-----
Edit the TRADES list in the __main__ block to define your positions. Each
entry is a dict with the inputs accepted by
BjerksundStenslandOptionPricer.price() / .greeks(), plus optional benchmark
values for comparison and an optional `valuation_date` / `maturity_date`
pair (ACT/365) instead of a raw `T`.

Results are printed to the console and, when OUTPUT_CSV is not None, written
to a CSV file with the same column layout as run_american_scenarios.py.
"""

from __future__ import annotations

import csv
import datetime as dt
import math
import os
from typing import Any, Dict, List, Optional

from bjerksund_stensland_forward import BjerksundStenslandOptionPricer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _act365(val_date: dt.date, mat_date: dt.date) -> float:
    """Return time to expiry in years under ACT/365."""
    return max((mat_date - val_date).days / 365.0, 0.0)


def _abs_diff(model: float, bench: Optional[float]) -> Optional[float]:
    if bench is None or math.isnan(bench):
        return None
    return abs(model - bench)


def _pct_diff(model: float, bench: Optional[float]) -> Optional[float]:
    """Relative difference: |model - bench| / |bench| * 100."""
    if bench is None or math.isnan(bench) or bench == 0.0:
        return None
    return abs(model - bench) / abs(bench) * 100.0


def _fmt(val: Optional[float], decimals: int = 6) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}"


# ---------------------------------------------------------------------------
# Core scenario runner
# ---------------------------------------------------------------------------

def run_bs_scenario(trade: Dict[str, Any]) -> Dict[str, Any]:
    """
    Price one trade with the Bjerksund-Stensland pricer and compute diffs.

    Required keys in *trade*
    ------------------------
    trade_name : str
    S          : float  - spot price
    K          : float  - strike price
    r          : float  - continuously compounded risk-free rate (e.g. 0.07)
    sigma      : float  - annualised volatility (e.g. 0.25)

    Time-to-expiry - supply ONE of:
      T              : float    - years directly
      valuation_date : dt.date  (+ maturity_date)

    Optional forward specification (priority: F > q > dividends > no-div):
      F         : float  - forward price
      q         : float  - continuous dividend yield
      dividends : list of (t_years: float, amount: float)

    Optional benchmarks (any combination):
      bench_price, bench_delta, bench_gamma, bench_vega : float

    Optional greek bump sizes:
      dS     : float  (default 1e-4 - relative spot bump)
      dSigma : float  (default 1e-4 - relative vol bump)

    Returns
    -------
    Dict with model values, benchmarks and abs / pct differences.
    """
    pricer = BjerksundStenslandOptionPricer()

    # Resolve time to expiry
    if "T" in trade and trade["T"] is not None:
        T = float(trade["T"])
    elif "valuation_date" in trade and "maturity_date" in trade:
        T = _act365(trade["valuation_date"], trade["maturity_date"])
    else:
        raise ValueError(
            f"Trade '{trade.get('trade_name', '?')}' must supply either "
            "'T' or ('valuation_date', 'maturity_date')."
        )

    S = float(trade["S"])
    K = float(trade["K"])
    r = float(trade["r"])
    sigma = float(trade["sigma"])
    opt_type = trade.get("option_type", "call")
    F = trade.get("F")
    q = trade.get("q")
    dividends = trade.get("dividends")
    dS = float(trade.get("dS", 1e-4))
    dSigma = float(trade.get("dSigma", 1e-4))

    # Benchmarks (None means not provided)
    bench_price = trade.get("bench_price")
    bench_delta = trade.get("bench_delta")
    bench_gamma = trade.get("bench_gamma")
    bench_vega = trade.get("bench_vega")

    # Price
    price_result = pricer.price(S, K, T, r, sigma, opt_type, F, q, dividends)
    model_price = price_result["price"]

    # Greeks
    greek_result = pricer.greeks(
        S, K, T, r, sigma, opt_type, F, q, dividends,
        dS=dS, dSigma=dSigma,
    )
    model_delta = greek_result["delta"]
    model_gamma = greek_result["gamma"]
    model_vega = greek_result["vega"]

    return {
        # Inputs
        "trade_name": trade.get("trade_name", "unnamed"),
        "option_type": opt_type,
        "S": S,
        "K": K,
        "T": T,
        "r": r,
        "sigma": sigma,
        "early_exercise": price_result.get("early_exercise", 0.0),
        "boundary_I": price_result.get("I", 0.0),
        # Model outputs
        "model_price": model_price,
        "model_delta": model_delta,
        "model_gamma": model_gamma,
        "model_vega": model_vega,
        # Benchmarks
        "bench_price": bench_price,
        "bench_delta": bench_delta,
        "bench_gamma": bench_gamma,
        "bench_vega": bench_vega,
        # Absolute differences
        "price_abs_diff": _abs_diff(model_price, bench_price),
        "delta_abs_diff": _abs_diff(model_delta, bench_delta),
        "gamma_abs_diff": _abs_diff(model_gamma, bench_gamma),
        "vega_abs_diff": _abs_diff(model_vega, bench_vega),
        # Relative (%) differences
        "price_pct_diff": _pct_diff(model_price, bench_price),
        "delta_pct_diff": _pct_diff(model_delta, bench_delta),
        "gamma_pct_diff": _pct_diff(model_gamma, bench_gamma),
        "vega_pct_diff": _pct_diff(model_vega, bench_vega),
    }


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_all_bs_scenarios(
    trades: List[Dict[str, Any]],
    output_csv: Optional[str] = None,
    print_results: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run all trades, optionally print a formatted table, and save a CSV.

    Parameters
    ----------
    trades       : list of trade dicts (see run_bs_scenario for key ref).
    output_csv   : file path for the results CSV; skipped when None.
    print_results: whether to print the summary table to stdout.

    Returns
    -------
    list of result dicts (one per trade).
    """
    all_results: List[Dict[str, Any]] = []

    for trade in trades:
        name = trade.get("trade_name", "unnamed")
        print(f"Pricing  {name} ...", end="  ")
        result = run_bs_scenario(trade)
        all_results.append(result)
        print(
            f"Price={_fmt(result['model_price'], 4)}  "
            f"Delta={_fmt(result['model_delta'], 4)}  "
            f"Gamma={_fmt(result['model_gamma'], 6)}  "
            f"Vega={_fmt(result['model_vega'], 4)}"
        )

    if print_results:
        _print_table(all_results)

    if output_csv:
        _save_csv(all_results, output_csv)
        print(f"\nResults saved to  {os.path.abspath(output_csv)}")

    return all_results


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_table(results: List[Dict[str, Any]]) -> None:
    """Print a structured console summary."""
    sep = "=" * 80

    print(f"\n{sep}")
    print("  BJERKSUND-STENSLAND AMERICAN OPTION RESULTS")
    print(sep)

    for r in results:
        print(
            f"\n  Trade : {r['trade_name']}"
            f"  |  Type: {r['option_type'].upper()}"
            f"  |  S={r['S']}  K={r['K']}  T={r['T']:.4f}y"
            f"  r={r['r']:.4f}  vol={r['sigma']:.4f}"
        )
        if r["early_exercise"]:
            print(
                f"  *** Early-exercise optimal now"
                f"  (boundary I = {r['boundary_I']:.4f}) ***"
            )

        has_bench = any(
            r[k] is not None
            for k in ("bench_price", "bench_delta", "bench_gamma", "bench_vega")
        )

        col = 13
        header = (
            f"  {'Metric':<10}  {'Model':>{col}}  "
            f"{'Benchmark':>{col}}  {'Abs Diff':>{col}}  {'% Diff':>10}"
        )
        print(f"\n{header}")
        print(f"  {'-'*10}  {'-'*col}  {'-'*col}  {'-'*col}  {'-'*10}")

        metrics = [
            ("Price", "model_price", "bench_price",
             "price_abs_diff", "price_pct_diff"),
            ("Delta", "model_delta", "bench_delta",
             "delta_abs_diff", "delta_pct_diff"),
            ("Gamma", "model_gamma", "bench_gamma",
             "gamma_abs_diff", "gamma_pct_diff"),
            ("Vega", "model_vega", "bench_vega",
             "vega_abs_diff", "vega_pct_diff"),
        ]
        for label, mk, bk, adk, pdk in metrics:
            bench_str = _fmt(r[bk]) if has_bench else "N/A"
            abd_str = _fmt(r[adk]) if has_bench else "N/A"
            if has_bench and r[pdk] is not None:
                pct_str = f"{r[pdk]:.4f}%"
            else:
                pct_str = "N/A"
            print(
                f"  {label:<10}  {_fmt(r[mk]):>{col}}"
                f"  {bench_str:>{col}}  {abd_str:>{col}}  {pct_str:>10}"
            )

    print(f"\n{sep}\n")


def _save_csv(results: List[Dict[str, Any]], path: str) -> None:
    """Write results to a CSV file."""
    if not results:
        return

    fieldnames = [
        "trade_name", "option_type", "S", "K", "T", "r", "sigma",
        "early_exercise", "boundary_I",
        "model_price", "bench_price", "price_abs_diff", "price_pct_diff",
        "model_delta", "bench_delta", "delta_abs_diff", "delta_pct_diff",
        "model_gamma", "bench_gamma", "gamma_abs_diff", "gamma_pct_diff",
        "model_vega", "bench_vega", "vega_abs_diff", "vega_pct_diff",
    ]

    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {k: ("" if v is None else v) for k, v in row.items()}
            )


# ---------------------------------------------------------------------------
# Entry point - edit TRADES below to price your positions
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # OUTPUT_CSV: set to None to suppress file output, or a file path
    #             to save a CSV of results.
    # ------------------------------------------------------------------
    OUTPUT_CSV: Optional[str] = "bs_forward_results.csv"

    # ------------------------------------------------------------------
    # TRADES: each dict is one option to price.
    #
    # Required:
    #   trade_name, S, K, r, sigma
    #   + one of: T  OR  (valuation_date, maturity_date)
    #
    # Optional forward (priority: F > q > dividends > no-div):
    #   F          - forward price
    #   q          - continuous dividend yield
    #   dividends  - list of (time_in_years: float, amount: float)
    #
    # Optional benchmarks (for comparison):
    #   bench_price, bench_delta, bench_gamma, bench_vega
    #
    # Optional greek bumps:
    #   dS, dSigma  (defaults: 1e-4 each)
    # ------------------------------------------------------------------

    TRADES: List[Dict[str, Any]] = [
        # ---- Trade 1: simple ATM call, 1-year, no dividends ----
        {
            "trade_name": "ATM_Call_1Y",
            "option_type": "call",
            "S": 100.0,
            "K": 100.0,
            "T": 1.0,
            "r": 0.07,
            "sigma": 0.25,
            # No F, q, or dividends -> forward = S * exp(r*T)
            # No benchmarks -> comparison columns will show N/A
        },

        # ---- Trade 2: ITM put with continuous dividend yield ----
        {
            "trade_name": "ITM_Put_DivYield",
            "option_type": "put",
            "S": 110.0,
            "K": 100.0,
            "T": 0.5,
            "r": 0.06,
            "sigma": 0.30,
            "q": 0.02,         # 2% continuous dividend yield
            # Replace with your benchmark values
            "bench_price": 3.50,
            "bench_delta": -0.32,
            "bench_gamma": 0.04,
            "bench_vega": 18.0,
        },

        # ---- Trade 3: using valuation / maturity dates ----
        {
            "trade_name": "OTM_Call_Dated",
            "option_type": "call",
            "S": 95.0,
            "K": 105.0,
            "valuation_date": dt.date(2025, 8, 28),
            "maturity_date": dt.date(2026, 8, 28),
            "r": 0.075,
            "sigma": 0.22,
            # Discrete dividends: (time_in_years, cash_amount)
            "dividends": [(0.25, 1.50), (0.75, 1.50)],
        },

        # ---- Trade 4: using a direct forward price ----
        {
            "trade_name": "ATM_Put_Forward",
            "option_type": "put",
            "S": 100.0,
            "K": 100.0,
            "T": 0.25,
            "r": 0.065,
            "sigma": 0.20,
            "F": 101.0,        # forward price supplied directly
            "bench_price": 4.20,
            "bench_delta": -0.48,
            "bench_gamma": 0.055,
            "bench_vega": 9.80,
        },
    ]

    run_all_bs_scenarios(
        trades=TRADES,
        output_csv=OUTPUT_CSV,
        print_results=True,
    )
