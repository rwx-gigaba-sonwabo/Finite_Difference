"""
Main runner for the Bjerksund-Stensland (forward/Black-76) American option
pricer.

Two pricing paths are supported per trade:

  Simple path  (existing API)
    Supply T (float), r, and optionally F / q / dividends-in-years.
    Calls pricer.price() and pricer.greeks().

  Curve-based path  (new API)
    Supply valuation_date, maturity_date, discount_curve, and optionally
    forward_curve, dividend_schedule (date-based), and all three spot-day
    lags.  Calls pricer.price_from_curves() and pricer.greeks_from_curves().

The curve path mirrors AmericanFDMPricer inputs exactly and correctly
separates time_to_carry, time_to_expiry, and time_to_discount.

Usage
-----
Edit TRADES in the __main__ block.  Each entry is a dict whose keys
determine which path is taken (see "Required keys" below).  Results are
printed to the console and, when OUTPUT_CSV is not None, written to a CSV.

Required keys (all trades)
--------------------------
  trade_name  : str
  S           : float  - spot price
  K           : float  - strike price
  sigma       : float  - annualised volatility

Time specification - exactly ONE of:
  T           : float  - time to expiry in years  [simple path]
  valuation_date + maturity_date : dt.date         [curve path]

Rate / forward specification:
  Simple path (T supplied):
    r          : float   - continuously compounded risk-free rate
    F          : float   - forward price (optional, overrides q/dividends)
    q          : float   - continuous dividend yield (optional)
    dividends  : list of (t_years: float, cash: float) (optional)

  Curve path (valuation_date + maturity_date supplied):
    discount_curve : pd.DataFrame["Date" (ISO YYYY-MM-DD), "NACA"]
                     Use build_flat_curve() for a quick flat curve.
    forward_curve  : same format (optional; falls back to discount_curve)
    dividend_schedule : list of (ex_date: dt.date, cash: float) (optional)
    underlying_spot_days     : int  (default 0)
    option_days              : int  (default 0)
    option_settlement_days   : int  (default 0)
    day_count                : str  (default "ACT/365")

Optional benchmarks (any combination, any path):
  bench_price, bench_delta, bench_gamma, bench_vega : float

Optional greek bump sizes (both paths):
  dS, dSigma  (defaults: 1e-4 each)
"""

from __future__ import annotations

import csv
import datetime as dt
import math
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from bjerksund_stensland_forward import BjerksundStenslandOptionPricer


# ---------------------------------------------------------------------------
# Curve builder helper
# ---------------------------------------------------------------------------

def build_flat_curve(
    rate: float,
    val_date: dt.date,
    mat_date: dt.date,
    pad_days: int = 15,
) -> pd.DataFrame:
    """Return a flat NACA rate curve as a DataFrame suitable for
    price_from_curves() / greeks_from_curves().

    The curve covers every calendar day from val_date - 1 to
    mat_date + pad_days (inclusive), so that all business-day-rolled
    carry/discount end dates are guaranteed to be present.

    Parameters
    ----------
    rate      : flat NACA rate (e.g. 0.08 for 8% NACA).
    val_date  : option valuation date.
    mat_date  : option maturity date.
    pad_days  : extra calendar days beyond mat_date (default 15).

    Returns
    -------
    pd.DataFrame with columns:
      "Date" : ISO string "YYYY-MM-DD"
      "NACA" : float (constant = rate)
    """
    start = val_date - dt.timedelta(days=1)
    end = mat_date + dt.timedelta(days=pad_days)
    date_range = pd.date_range(start=start, end=end, freq="D")
    return pd.DataFrame({"Date": date_range.strftime("%Y-%m-%d"), "NACA": rate})


# ---------------------------------------------------------------------------
# Diff helpers
# ---------------------------------------------------------------------------

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
    """Price one trade and return a result dict with model values and diffs.

    The pricing path (simple or curve-based) is selected automatically:
      - If 'discount_curve' is present in the trade dict  -> curve path.
      - Otherwise                                          -> simple path.

    See module docstring for the full key reference.
    """
    pricer = BjerksundStenslandOptionPricer()
    name = trade.get("trade_name", "unnamed")
    opt_type = trade.get("option_type", "call")
    S = float(trade["S"])
    K = float(trade["K"])
    sigma = float(trade["sigma"])
    dS = float(trade.get("dS", 1e-4))
    dSigma = float(trade.get("dSigma", 1e-4))

    bench_price = trade.get("bench_price")
    bench_delta = trade.get("bench_delta")
    bench_gamma = trade.get("bench_gamma")
    bench_vega = trade.get("bench_vega")

    # ---- Select pricing path ----
    if "discount_curve" in trade:
        # --- Curve-based path ---
        val_date = trade["valuation_date"]
        mat_date = trade["maturity_date"]
        discount_curve = trade["discount_curve"]
        forward_curve = trade.get("forward_curve")
        div_schedule = trade.get("dividend_schedule")
        usd = int(trade.get("underlying_spot_days", 0))
        od = int(trade.get("option_days", 0))
        osd = int(trade.get("option_settlement_days", 0))
        dc = trade.get("day_count", "ACT/365")

        price_result = pricer.price_from_curves(
            S, K, val_date, mat_date, sigma, opt_type,
            discount_curve, forward_curve, div_schedule,
            usd, od, osd, dc,
        )
        greek_result = pricer.greeks_from_curves(
            S, K, val_date, mat_date, sigma, opt_type,
            discount_curve, forward_curve, div_schedule,
            usd, od, osd, dc,
            dS=dS, dSigma=dSigma,
        )

        T_exp = price_result["T_exp"]
        T_carry = price_result["T_carry"]
        T_disc = price_result["T_disc"]
        carry_rate = price_result["carry_rate"]
        disc_rate = price_result["disc_rate"]
        F_eff = price_result["F_eff"]
        b = price_result["b"]
        path = "curve"

    else:
        # --- Simple path ---
        if "T" in trade and trade["T"] is not None:
            T_exp = float(trade["T"])
        elif "valuation_date" in trade and "maturity_date" in trade:
            days = (trade["maturity_date"] - trade["valuation_date"]).days
            T_exp = max(days / 365.0, 0.0)
        else:
            raise ValueError(
                f"Trade '{name}': supply 'T', or ('valuation_date' + "
                "'maturity_date'), or 'discount_curve'."
            )

        r = float(trade["r"])
        F_arg = trade.get("F")
        q_arg = trade.get("q")
        divs_arg = trade.get("dividends")

        price_result = pricer.price(
            S, K, T_exp, r, sigma, opt_type, F_arg, q_arg, divs_arg
        )
        greek_result = pricer.greeks(
            S, K, T_exp, r, sigma, opt_type, F_arg, q_arg, divs_arg,
            dS=dS, dSigma=dSigma,
        )

        T_carry = T_exp
        T_disc = T_exp
        carry_rate = r
        disc_rate = r
        F_eff = pricer._resolve_forward(S, r, T_exp, F_arg, q_arg, divs_arg)
        b = math.log(max(F_eff, 1e-15) / max(S, 1e-15)) / max(T_exp, 1e-12)
        path = "simple"

    model_price = price_result["price"]
    model_delta = greek_result["delta"]
    model_gamma = greek_result["gamma"]
    model_vega = greek_result["vega"]

    return {
        # Inputs / resolved quantities
        "trade_name": name,
        "option_type": opt_type,
        "path": path,
        "S": S,
        "K": K,
        "T_exp": T_exp,
        "T_carry": T_carry,
        "T_disc": T_disc,
        "carry_rate": carry_rate,
        "disc_rate": disc_rate,
        "F_eff": F_eff,
        "b": b,
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
    """Run all trades, print a summary table, and optionally save a CSV.

    Parameters
    ----------
    trades       : list of trade dicts (see module docstring for keys).
    output_csv   : CSV file path; skipped when None.
    print_results: whether to print the formatted table to stdout.

    Returns
    -------
    list of result dicts, one per trade.
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
        print(f"Results saved to  {os.path.abspath(output_csv)}")

    return all_results


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_table(results: List[Dict[str, Any]]) -> None:
    """Print a structured per-trade console summary."""
    sep = "=" * 80
    print(f"\n{sep}")
    print("  BJERKSUND-STENSLAND AMERICAN OPTION RESULTS")
    print(sep)

    for r in results:
        # Header line
        path_tag = f"[{r['path']}]"
        print(
            f"\n  Trade : {r['trade_name']}  {path_tag}"
            f"  |  Type: {r['option_type'].upper()}"
            f"  |  S={r['S']}  K={r['K']}"
        )
        # Time / rate detail
        print(
            f"  T_exp={r['T_exp']:.4f}y  "
            f"T_carry={r['T_carry']:.4f}y  "
            f"T_disc={r['T_disc']:.4f}y  "
            f"carry_rate={r['carry_rate']:.4f}  "
            f"disc_rate={r['disc_rate']:.4f}"
        )
        print(
            f"  F_eff={r['F_eff']:.4f}  "
            f"b={r['b']:.4f}  "
            f"vol={r['sigma']:.4f}"
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
            ("Vega",  "model_vega",  "bench_vega",
             "vega_abs_diff",  "vega_pct_diff"),
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
    """Write results to a CSV file (DataFrames excluded from output)."""
    if not results:
        return

    fieldnames = [
        "trade_name", "option_type", "path",
        "S", "K", "T_exp", "T_carry", "T_disc",
        "carry_rate", "disc_rate", "F_eff", "b", "sigma",
        "early_exercise", "boundary_I",
        "model_price", "bench_price", "price_abs_diff", "price_pct_diff",
        "model_delta", "bench_delta", "delta_abs_diff", "delta_pct_diff",
        "model_gamma", "bench_gamma", "gamma_abs_diff", "gamma_pct_diff",
        "model_vega",  "bench_vega",  "vega_abs_diff",  "vega_pct_diff",
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
    # OUTPUT_CSV: set to None to suppress file output.
    # ------------------------------------------------------------------
    OUTPUT_CSV: Optional[str] = "bs_forward_results.csv"

    # ------------------------------------------------------------------
    # TRADES - each dict is one option to price.
    #
    # Simple path: supply T + r (no discount_curve).
    # Curve path : supply valuation_date + maturity_date + discount_curve.
    #
    # build_flat_curve(rate, val_date, mat_date) creates a daily flat
    # NACA curve that covers all dates the pricer needs to look up.
    # ------------------------------------------------------------------

    VAL = dt.date(2025, 8, 28)   # shared valuation date for examples below

    TRADES: List[Dict[str, Any]] = [

        # ==================================================================
        # SIMPLE PATH EXAMPLES
        # ==================================================================

        # ---- Simple 1: ATM call, 1-year, no dividends ----
        {
            "trade_name": "ATM_Call_1Y_simple",
            "option_type": "call",
            "S": 100.0,
            "K": 100.0,
            "T": 1.0,
            "r": 0.07,
            "sigma": 0.25,
            # No benchmarks -> comparison columns show N/A
        },

        # ---- Simple 2: ITM put, continuous dividend yield ----
        {
            "trade_name": "ITM_Put_DivYield_simple",
            "option_type": "put",
            "S": 110.0,
            "K": 100.0,
            "T": 0.5,
            "r": 0.06,
            "sigma": 0.30,
            "q": 0.02,
            "bench_price": 3.50,
            "bench_delta": -0.32,
            "bench_gamma": 0.04,
            "bench_vega": 18.0,
        },

        # ==================================================================
        # CURVE-BASED PATH EXAMPLES
        # ==================================================================

        # ---- Curve 1: ATM call, flat curves, no spot-day lags ----
        # Equivalent to simple path when underlying_spot_days = option_days
        # = option_settlement_days = 0.
        {
            "trade_name": "ATM_Call_1Y_curve",
            "option_type": "call",
            "S": 100.0,
            "K": 100.0,
            "valuation_date": VAL,
            "maturity_date": VAL + dt.timedelta(days=365),
            "sigma": 0.25,
            "discount_curve": build_flat_curve(
                0.07, VAL, VAL + dt.timedelta(days=365)
            ),
            # forward_curve omitted -> falls back to discount_curve
            "underlying_spot_days": 0,
            "option_days": 0,
            "option_settlement_days": 0,
            "day_count": "ACT/365",
        },

        # ---- Curve 2: ATM call, separate carry and discount curves,
        #               T+3 equity settlement, no option lag ----
        {
            "trade_name": "ATM_Call_SplitCurves_T3",
            "option_type": "call",
            "S": 100.0,
            "K": 100.0,
            "valuation_date": VAL,
            "maturity_date": VAL + dt.timedelta(days=365),
            "sigma": 0.25,
            "discount_curve": build_flat_curve(
                0.075, VAL, VAL + dt.timedelta(days=365)
            ),
            "forward_curve": build_flat_curve(
                0.08, VAL, VAL + dt.timedelta(days=365)
            ),
            "underlying_spot_days": 3,   # T+3 equity spot
            "option_days": 0,
            "option_settlement_days": 0,
            "day_count": "ACT/365",
        },

        # ---- Curve 3: ITM put with discrete dividends and spot-day lags ----
        {
            "trade_name": "ITM_Put_Divs_T3",
            "option_type": "put",
            "S": 110.0,
            "K": 100.0,
            "valuation_date": VAL,
            "maturity_date": dt.date(2026, 8, 28),
            "sigma": 0.28,
            "discount_curve": build_flat_curve(
                0.075, VAL, dt.date(2026, 8, 28)
            ),
            "forward_curve": build_flat_curve(
                0.08, VAL, dt.date(2026, 8, 28)
            ),
            # Discrete cash dividends: (ex-date, cash amount)
            "dividend_schedule": [
                (dt.date(2025, 11, 28), 2.00),
                (dt.date(2026, 2, 28), 2.00),
                (dt.date(2026, 5, 28), 2.00),
            ],
            "underlying_spot_days": 3,   # T+3 equity settlement
            "option_days": 0,
            "option_settlement_days": 0,
            "day_count": "ACT/365",
            "bench_price": 7.50,
            "bench_delta": -0.42,
            "bench_gamma": 0.015,
            "bench_vega": 38.0,
        },

        # ---- Curve 4: OTM call, 3-month, all spot lags set ----
        {
            "trade_name": "OTM_Call_FullLags",
            "option_type": "call",
            "S": 95.0,
            "K": 105.0,
            "valuation_date": VAL,
            "maturity_date": VAL + dt.timedelta(days=91),
            "sigma": 0.22,
            "discount_curve": build_flat_curve(
                0.08, VAL, VAL + dt.timedelta(days=91)
            ),
            "forward_curve": build_flat_curve(
                0.085, VAL, VAL + dt.timedelta(days=91)
            ),
            "underlying_spot_days": 3,   # equity T+3
            "option_days": 0,            # option settles same day
            "option_settlement_days": 1, # option delivery T+1
            "day_count": "ACT/365",
        },
    ]

    run_all_bs_scenarios(
        trades=TRADES,
        output_csv=OUTPUT_CSV,
        print_results=True,
    )
