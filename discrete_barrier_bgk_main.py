"""
discrete_barrier_bgk_main.py
=============================
Multi-scenario runner for DiscreteBarrierBGKPricer.

Supports full per-trade configuration including:
  - Barrier type  : up-and-out | down-and-out | double-out |
                    up-and-in  | down-and-in  | double-in  | none
  - Monitoring    : daily | weekly | fortnightly | monthly | quarterly,
                    or a fully custom list of dt.date objects
  - Dividend schedules : per-trade, date-based cash amounts
  - Pricing method: "bgk" | "mc" | "auto"  (per-trade MC parameters)
  - Separate discount and forward curves (flat builder or loaded CSV)
  - Settlement lags : underlying_spot_days, option_days,
                      option_settlement_days
  - Benchmark prices & Greeks for model-validation comparison

Usage
-----
Edit the TRADES list in the __main__ block.  Results are printed to the
console and, when OUTPUT_CSV is not None, written to a timestamped CSV.

Trade dict keys (all trades)
------------------------------
Required:
  trade_name        : str
  option_type       : "call" | "put"
  barrier_type      : "up-and-out" | "down-and-out" | "double-out" |
                      "up-and-in"  | "down-and-in"  | "double-in"  | "none"
  S                 : float  spot price
  K                 : float  strike price
  sigma             : float  annualised Black-Scholes volatility
  valuation_date    : dt.date
  maturity_date     : dt.date
  discount_curve    : pd.DataFrame["Date" (ISO "YYYY-MM-DD"), "NACA"]
                      Use build_flat_curve() for a quick flat curve.

Optional product:
  lower_barrier     : float  (required for down-* and double-* types)
  upper_barrier     : float  (required for up-* and double-* types)
  monitor_dates     : List[dt.date]   explicit schedule (overrides
                      monitor_frequency when supplied)
  monitor_frequency : "daily" | "weekly" | "fortnightly" | "monthly" |
                      "quarterly"  (default "weekly" when neither key given)
  rebate_amount     : float  (default 0)
  rebate_at_hit     : bool   (default False)
  already_hit       : bool   (default False)
  barrier_hit_date  : dt.date (used only when already_hit=True)

Optional market:
  forward_curve     : pd.DataFrame same format; falls back to discount_curve
  dividend_schedule : List[Tuple[dt.date, float]]  (pay_date, cash_amount)
  underlying_spot_days   : int  (default 0)
  option_days            : int  (default 0)
  option_settlement_days : int  (default 0)
  day_count         : str  (default "ACT/365")

Optional model:
  pricing_method    : "bgk" | "mc" | "auto"  (default "auto")
  bgk_min_freq      : float  monitoring-dates/year threshold  (default 20.0)
  mc_n_paths        : int    (default 100_000)
  mc_seed           : int | None  (default 42)
  mc_use_antithetic : bool   (default True)
  include_expiry_monitor : bool  (default True)
  use_mean_sqrt_dt  : bool   (default False)
  theta_from_forward: bool   (default False)

Optional trade meta:
  direction         : "long" | "short"  (default "long")
  quantity          : int    (default 1)
  contract_multiplier: float (default 1.0)

Optional benchmarks:
  bench_price, bench_delta, bench_gamma, bench_vega : float

Optional Greek bump sizes:
  dS_rel   : float  relative spot bump   (default 1e-4)
  dVol_abs : float  absolute vol bump    (default 1e-4)
"""

from __future__ import annotations

import csv
import datetime as dt
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from discrete_barrier_bgk import DiscreteBarrierBGKPricer


# ---------------------------------------------------------------------------
# Curve and schedule builders
# ---------------------------------------------------------------------------

def build_flat_curve(
    rate: float,
    val_date: dt.date,
    mat_date: dt.date,
    pad_days: int = 15,
) -> pd.DataFrame:
    """Return a flat NACA daily curve spanning val_date-1 to mat_date+pad_days.

    Parameters
    ----------
    rate     : flat NACA rate (e.g. 0.085 for 8.5% NACA).
    val_date : option valuation date.
    mat_date : option maturity date.
    pad_days : extra calendar days beyond mat_date (default 15).

    Returns
    -------
    pd.DataFrame with columns "Date" (ISO "YYYY-MM-DD") and "NACA".
    """
    start = val_date - dt.timedelta(days=1)
    end   = mat_date + dt.timedelta(days=pad_days)
    dates = pd.date_range(start=start, end=end, freq="D")
    return pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "NACA": rate})


def build_monitoring_dates(
    val_date: dt.date,
    mat_date: dt.date,
    frequency: str = "weekly",
) -> List[dt.date]:
    """Build a calendar-spaced monitoring schedule.

    Dates run from val_date + step through mat_date (inclusive),
    advancing by a fixed calendar-day step.

    Parameters
    ----------
    val_date  : valuation date (monitoring starts strictly after this).
    mat_date  : maturity date (last monitor on or before this).
    frequency : "daily"       ->  1-day step  (~252 dates/year)
                "weekly"      ->  7-day step  (~52 dates/year)
                "fortnightly" -> 14-day step  (~26 dates/year)
                "monthly"     -> 30-day step  (~12 dates/year)
                "quarterly"   -> 91-day step  (~4  dates/year)

    Returns
    -------
    Sorted List[dt.date].
    """
    step_map = {
        "daily":       1,
        "weekly":      7,
        "fortnightly": 14,
        "monthly":     30,
        "quarterly":   91,
    }
    step = step_map.get(frequency.lower())
    if step is None:
        raise ValueError(
            f"Unknown frequency {frequency!r}. "
            f"Choose from: {list(step_map)}."
        )
    dates: List[dt.date] = []
    d = val_date + dt.timedelta(days=step)
    while d <= mat_date:
        dates.append(d)
        d += dt.timedelta(days=step)
    return dates


# ---------------------------------------------------------------------------
# Diff / format helpers
# ---------------------------------------------------------------------------

def _abs_diff(model: float, bench: Optional[float]) -> Optional[float]:
    if bench is None or (isinstance(bench, float) and math.isnan(bench)):
        return None
    return model - bench


def _pct_diff(model: float, bench: Optional[float]) -> Optional[float]:
    """Relative difference: (model - bench) / |bench| * 100."""
    if bench is None or (isinstance(bench, float) and math.isnan(bench)):
        return None
    if bench == 0.0:
        return None
    return (model - bench) / abs(bench) * 100.0


def _fmt(val: Optional[float], decimals: int = 6) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}"


# ---------------------------------------------------------------------------
# Core scenario runner
# ---------------------------------------------------------------------------

def run_bgk_scenario(trade: Dict[str, Any]) -> Dict[str, Any]:
    """Price one discrete barrier trade and return a result dict.

    See module docstring for the full key reference.

    Returns
    -------
    Dict containing model outputs, resolved inputs, and benchmark diffs.
    On error, returns a dict with 'error' key set to the exception message.
    """
    name = trade.get("trade_name", "unnamed")

    try:
        # ---- Resolve monitoring dates --------------------------------
        if "monitor_dates" in trade and trade["monitor_dates"] is not None:
            mon_dates = list(trade["monitor_dates"])
        else:
            freq = trade.get("monitor_frequency", "weekly")
            mon_dates = build_monitoring_dates(
                trade["valuation_date"],
                trade["maturity_date"],
                frequency=freq,
            )

        # ---- Build pricer -------------------------------------------
        pricer = DiscreteBarrierBGKPricer(
            spot=float(trade["S"]),
            strike=float(trade["K"]),
            valuation_date=trade["valuation_date"],
            maturity_date=trade["maturity_date"],
            option_type=trade["option_type"],
            barrier_type=trade.get("barrier_type", "none"),
            lower_barrier=trade.get("lower_barrier"),
            upper_barrier=trade.get("upper_barrier"),
            monitor_dates=mon_dates,
            rebate_amount=float(trade.get("rebate_amount", 0.0)),
            rebate_at_hit=bool(trade.get("rebate_at_hit", False)),
            already_hit=bool(trade.get("already_hit", False)),
            barrier_hit_date=trade.get("barrier_hit_date"),
            discount_curve=trade["discount_curve"],
            forward_curve=trade.get("forward_curve"),
            dividend_schedule=trade.get("dividend_schedule"),
            volatility=float(trade["sigma"]),
            day_count=trade.get("day_count", "ACT/365"),
            include_expiry_monitor=bool(trade.get("include_expiry_monitor", True)),
            use_mean_sqrt_dt=bool(trade.get("use_mean_sqrt_dt", False)),
            theta_from_forward=bool(trade.get("theta_from_forward", False)),
            pricing_method=trade.get("pricing_method", "auto"),
            bgk_min_freq=float(trade.get("bgk_min_freq", 20.0)),
            mc_n_paths=int(trade.get("mc_n_paths", 100_000)),
            mc_seed=trade.get("mc_seed", 42),
            mc_use_antithetic=bool(trade.get("mc_use_antithetic", True)),
            underlying_spot_days=int(trade.get("underlying_spot_days", 0)),
            option_days=int(trade.get("option_days", 0)),
            option_settlement_days=int(trade.get("option_settlement_days", 0)),
            trade_id=name,
            direction=trade.get("direction", "long"),
            quantity=int(trade.get("quantity", 1)),
            contract_multiplier=float(trade.get("contract_multiplier", 1.0)),
        )

        # ---- Price and Greeks ---------------------------------------
        dS_rel   = float(trade.get("dS_rel",   1e-4))
        dVol_abs = float(trade.get("dVol_abs", 1e-4))

        model_price = pricer.price()
        mc_std_err  = pricer._last_mc_std_error  # 0.0 when BGK was used

        greeks      = pricer.greeks(ds_rel=dS_rel, dvol_abs=dVol_abs)
        model_delta = greeks["delta"]
        model_gamma = greeks["gamma"]
        model_vega  = greeks["vega"]

        # ---- Resolved quantities for reporting ----------------------
        selected_method = pricer._select_method().upper()
        mon_freq        = pricer.m / max(pricer.tenor_years, 1e-12)

        bench_price = trade.get("bench_price")
        bench_delta = trade.get("bench_delta")
        bench_gamma = trade.get("bench_gamma")
        bench_vega  = trade.get("bench_vega")

        return {
            # Identification
            "trade_name":       name,
            "option_type":      trade["option_type"],
            "barrier_type":     trade.get("barrier_type", "none"),
            "lower_barrier":    trade.get("lower_barrier"),
            "upper_barrier":    trade.get("upper_barrier"),
            "direction":        trade.get("direction", "long"),
            # Market inputs
            "S":                float(trade["S"]),
            "K":                float(trade["K"]),
            "sigma":            float(trade["sigma"]),
            # Time measures
            "T_exp":            pricer.time_to_expiry,
            "T_carry":          pricer.time_to_carry,
            "T_disc":           pricer.time_to_discount,
            # Rates and yield
            "carry_rate":       pricer.carry_rate_nacc,
            "disc_rate":        pricer.discount_rate_nacc,
            "div_yield":        pricer.div_yield_nacc,
            "F_eff":            pricer.forward_price,
            # Monitoring
            "m":                pricer.m,
            "mon_freq":         mon_freq,
            "pricing_method":   trade.get("pricing_method", "auto"),
            "selected_method":  selected_method,
            "bgk_min_freq":     float(trade.get("bgk_min_freq", 20.0)),
            "mc_n_paths":       int(trade.get("mc_n_paths", 100_000))
                                if selected_method == "MC" else None,
            "mc_seed":          trade.get("mc_seed", 42)
                                if selected_method == "MC" else None,
            "mc_std_error":     mc_std_err if selected_method == "MC" else None,
            # Settlement lags
            "underlying_spot_days":   int(trade.get("underlying_spot_days", 0)),
            "option_days":            int(trade.get("option_days", 0)),
            "option_settlement_days": int(trade.get("option_settlement_days", 0)),
            # Model outputs
            "model_price":      model_price,
            "model_delta":      model_delta,
            "model_gamma":      model_gamma,
            "model_vega":       model_vega,
            # Benchmarks
            "bench_price":      bench_price,
            "bench_delta":      bench_delta,
            "bench_gamma":      bench_gamma,
            "bench_vega":       bench_vega,
            # Absolute differences
            "price_abs_diff":   _abs_diff(model_price, bench_price),
            "delta_abs_diff":   _abs_diff(model_delta, bench_delta),
            "gamma_abs_diff":   _abs_diff(model_gamma, bench_gamma),
            "vega_abs_diff":    _abs_diff(model_vega,  bench_vega),
            # Relative (%) differences
            "price_pct_diff":   _pct_diff(model_price, bench_price),
            "delta_pct_diff":   _pct_diff(model_delta, bench_delta),
            "gamma_pct_diff":   _pct_diff(model_gamma, bench_gamma),
            "vega_pct_diff":    _pct_diff(model_vega,  bench_vega),
            # Error flag
            "error": None,
        }

    except Exception as exc:
        return {
            "trade_name": name,
            "error": str(exc),
            # fill remaining keys with None so CSV writer doesn't choke
            **{
                k: None for k in [
                    "option_type", "barrier_type", "lower_barrier",
                    "upper_barrier", "direction", "S", "K", "sigma",
                    "T_exp", "T_carry", "T_disc", "carry_rate", "disc_rate",
                    "div_yield", "F_eff", "m", "mon_freq", "pricing_method",
                    "selected_method", "bgk_min_freq", "mc_n_paths", "mc_seed",
                    "mc_std_error", "underlying_spot_days", "option_days",
                    "option_settlement_days", "model_price", "model_delta",
                    "model_gamma", "model_vega", "bench_price", "bench_delta",
                    "bench_gamma", "bench_vega", "price_abs_diff",
                    "delta_abs_diff", "gamma_abs_diff", "vega_abs_diff",
                    "price_pct_diff", "delta_pct_diff", "gamma_pct_diff",
                    "vega_pct_diff",
                ]
            },
        }


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_all_bgk_scenarios(
    trades: List[Dict[str, Any]],
    output_csv: Optional[str] = None,
    print_results: bool = True,
) -> List[Dict[str, Any]]:
    """Price all trades, print a formatted summary, and optionally save CSV.

    Parameters
    ----------
    trades       : list of trade dicts (see module docstring).
    output_csv   : CSV file path; when None no file is written.
    print_results: whether to print the formatted table to stdout.

    Returns
    -------
    list of result dicts, one per trade.
    """
    all_results: List[Dict[str, Any]] = []

    for trade in trades:
        name = trade.get("trade_name", "unnamed")
        print(f"Pricing  {name} ...", end="  ")
        result = run_bgk_scenario(trade)
        all_results.append(result)

        if result["error"]:
            print(f"ERROR: {result['error']}")
        else:
            mc_tag = (
                f"  MC_se={result['mc_std_error']:.2e}"
                if result["selected_method"] == "MC"
                else ""
            )
            print(
                f"[{result['selected_method']}]"
                f"  Price={_fmt(result['model_price'], 4)}"
                f"  Delta={_fmt(result['model_delta'], 4)}"
                f"  Gamma={_fmt(result['model_gamma'], 6)}"
                f"  Vega={_fmt(result['model_vega'], 4)}"
                f"{mc_tag}"
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
    sep  = "=" * 88
    sep2 = "-" * 88
    print(f"\n{sep}")
    print("  DISCRETE BARRIER BGK / MONTE CARLO OPTION RESULTS")
    print(sep)

    for r in results:
        if r["error"]:
            print(f"\n  Trade : {r['trade_name']}  *** PRICING ERROR ***")
            print(f"  {r['error']}")
            print(f"\n{sep2}")
            continue

        # ---- Header --------------------------------------------------
        low_tag = (f"  Hd={r['lower_barrier']}"
                   if r["lower_barrier"] is not None else "")
        up_tag  = (f"  Hu={r['upper_barrier']}"
                   if r["upper_barrier"] is not None else "")
        print(
            f"\n  Trade : {r['trade_name']}"
            f"  |  {r['option_type'].upper()}  {r['barrier_type'].upper()}"
            f"{low_tag}{up_tag}"
            f"  |  S={r['S']}  K={r['K']}"
            f"  |  {r['direction'].upper()}"
        )

        # ---- Time / rate block --------------------------------------
        print(
            f"  T_exp={r['T_exp']:.4f}y  "
            f"T_carry={r['T_carry']:.4f}y  "
            f"T_disc={r['T_disc']:.4f}y  "
            f"carry_r={r['carry_rate']:.4f}  "
            f"disc_r={r['disc_rate']:.4f}  "
            f"div_q={r['div_yield']:.4f}"
        )
        print(
            f"  F_eff={r['F_eff']:.4f}  "
            f"vol={r['sigma']:.4f}  "
            f"lags=({r['underlying_spot_days']}bd/"
            f"{r['option_days']}bd/{r['option_settlement_days']}bd)"
        )

        # ---- Monitoring / method block ------------------------------
        mc_detail = ""
        if r["selected_method"] == "MC":
            mc_detail = (
                f"  mc_paths={r['mc_n_paths']:,}"
                f"  seed={r['mc_seed']}"
                f"  se={r['mc_std_error']:.2e}"
            )
        print(
            f"  monitors={r['m']}  freq={r['mon_freq']:.1f}/yr  "
            f"method={r['pricing_method']} -> {r['selected_method']}"
            f"  bgk_min={r['bgk_min_freq']:.0f}/yr"
            f"{mc_detail}"
        )

        # ---- Price / Greeks matrix ----------------------------------
        has_bench = any(
            r[k] is not None
            for k in ("bench_price", "bench_delta", "bench_gamma", "bench_vega")
        )

        col = 14
        header = (
            f"  {'Metric':<10}  {'Model':>{col}}"
            + (f"  {'Benchmark':>{col}}  {'Abs Diff':>{col}}  {'% Diff':>10}"
               if has_bench else "")
        )
        print(f"\n{header}")
        underline = f"  {'-'*10}  {'-'*col}" + (
            f"  {'-'*col}  {'-'*col}  {'-'*10}" if has_bench else ""
        )
        print(underline)

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
            model_str = _fmt(r[mk])
            if has_bench:
                bench_str = _fmt(r[bk]) if r[bk] is not None else "N/A"
                abd_str   = _fmt(r[adk]) if r[adk] is not None else "N/A"
                pct_str   = (
                    f"{r[pdk]:+.4f}%"
                    if r[pdk] is not None else "N/A"
                )
                print(
                    f"  {label:<10}  {model_str:>{col}}"
                    f"  {bench_str:>{col}}  {abd_str:>{col}}  {pct_str:>10}"
                )
            else:
                print(f"  {label:<10}  {model_str:>{col}}")

    print(f"\n{sep}\n")


def _save_csv(results: List[Dict[str, Any]], path: str) -> None:
    """Write results to a CSV file (DataFrame inputs are excluded)."""
    if not results:
        return

    fieldnames = [
        "trade_name", "option_type", "barrier_type",
        "lower_barrier", "upper_barrier", "direction",
        "S", "K", "sigma",
        "T_exp", "T_carry", "T_disc",
        "carry_rate", "disc_rate", "div_yield", "F_eff",
        "m", "mon_freq", "pricing_method", "selected_method",
        "bgk_min_freq", "mc_n_paths", "mc_seed", "mc_std_error",
        "underlying_spot_days", "option_days", "option_settlement_days",
        "model_price", "bench_price", "price_abs_diff", "price_pct_diff",
        "model_delta", "bench_delta", "delta_abs_diff", "delta_pct_diff",
        "model_gamma", "bench_gamma", "gamma_abs_diff", "gamma_pct_diff",
        "model_vega",  "bench_vega",  "vega_abs_diff",  "vega_pct_diff",
        "error",
    ]

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {k: ("" if v is None else v) for k, v in row.items()}
            )


# ---------------------------------------------------------------------------
# Entry point — edit TRADES below
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    import datetime as dt

    # ------------------------------------------------------------------
    # OUTPUT_CSV: set to None to suppress CSV output.
    # ------------------------------------------------------------------
    _ts = dt.datetime.now().strftime("%Y-%m-%d %H%M%S")
    OUTPUT_CSV: Optional[str] = f"discrete_barrier_bgk_results {_ts}.csv"

    # ------------------------------------------------------------------
    # Shared market data
    # ------------------------------------------------------------------

    VAL = dt.date(2025, 7, 28)   # common valuation date
    MAT_1Y = dt.date(2026, 7, 28)
    MAT_6M = dt.date(2026, 1, 28)
    MAT_3M = dt.date(2025, 10, 28)

    # Flat NACA discount curve at 8.5%
    DISC_85 = build_flat_curve(0.085, VAL, MAT_1Y)

    # Separate flat NACA forward/carry curve at 9.1%
    FWD_91  = build_flat_curve(0.091, VAL, MAT_1Y)

    # 6-month and 3-month curves
    DISC_85_6M = build_flat_curve(0.085, VAL, MAT_6M)
    DISC_85_3M = build_flat_curve(0.085, VAL, MAT_3M)

    # JSE-style cash dividend schedule (shared across equity trades)
    JSE_DIVS: List[Tuple[dt.date, float]] = [
        (dt.date(2025, 9, 12), 7.63),
        (dt.date(2026, 4, 10), 8.0115),
    ]

    # A light dividend schedule for shorter trades (within 6-month window)
    SHORT_DIVS: List[Tuple[dt.date, float]] = [
        (dt.date(2025, 9, 12), 3.50),
    ]

    # ------------------------------------------------------------------
    # TRADES
    # ------------------------------------------------------------------

    TRADES: List[Dict[str, Any]] = [

        # ==============================================================
        # 1.  Up-and-Out Call  |  Daily monitoring  ->  BGK path
        #     252 dates/year >> bgk_min_freq (20) -> BGK selected by auto
        # ==============================================================
        {
            "trade_name":       "T01_UAO_Call_Daily_BGK",
            "option_type":      "call",
            "barrier_type":     "up-and-out",
            "S":                229.74,
            "K":                220.00,
            "upper_barrier":    270.00,
            "sigma":            0.2613190156888,
            "valuation_date":   VAL,
            "maturity_date":    MAT_1Y,
            "monitor_frequency":"daily",        # ~252 dates -> BGK auto-selected
            "discount_curve":   DISC_85,
            "forward_curve":    FWD_91,
            "dividend_schedule":JSE_DIVS,
            "pricing_method":   "auto",
            "bgk_min_freq":     20.0,
            "day_count":        "ACT/365",
            "direction":        "long",
            "quantity":         1,
            "contract_multiplier": 1.0,
            # Replace with RiskFlow / Front Arena reference values:
            "bench_price":      None,
            "bench_delta":      None,
            "bench_gamma":      None,
            "bench_vega":       None,
        },

        # ==============================================================
        # 2.  Down-and-Out Put  |  Weekly monitoring  ->  BGK path
        #     52 dates/year > bgk_min_freq (20) -> BGK auto-selected
        # ==============================================================
        {
            "trade_name":       "T02_DAO_Put_Weekly_BGK",
            "option_type":      "put",
            "barrier_type":     "down-and-out",
            "S":                229.74,
            "K":                240.00,
            "lower_barrier":    185.00,
            "sigma":            0.2613190156888,
            "valuation_date":   VAL,
            "maturity_date":    MAT_1Y,
            "monitor_frequency":"weekly",       # ~52 dates/yr -> BGK
            "discount_curve":   DISC_85,
            "forward_curve":    FWD_91,
            "dividend_schedule":JSE_DIVS,
            "pricing_method":   "auto",
            "bgk_min_freq":     20.0,
            "day_count":        "ACT/365",
            "direction":        "long",
            "bench_price":      None,
            "bench_delta":      None,
            "bench_gamma":      None,
            "bench_vega":       None,
        },

        # ==============================================================
        # 3.  Up-and-In Call  |  Monthly monitoring  ->  MC path
        #     12 dates/year < bgk_min_freq (20) -> MC auto-selected
        # ==============================================================
        {
            "trade_name":       "T03_UAI_Call_Monthly_MC",
            "option_type":      "call",
            "barrier_type":     "up-and-in",
            "S":                229.74,
            "K":                220.00,
            "upper_barrier":    260.00,
            "sigma":            0.2613190156888,
            "valuation_date":   VAL,
            "maturity_date":    MAT_1Y,
            "monitor_frequency":"monthly",      # ~12 dates/yr -> MC
            "discount_curve":   DISC_85,
            "forward_curve":    FWD_91,
            "dividend_schedule":JSE_DIVS,
            "pricing_method":   "auto",
            "bgk_min_freq":     20.0,
            "mc_n_paths":       200_000,
            "mc_seed":          123,
            "mc_use_antithetic":True,
            "day_count":        "ACT/365",
            "direction":        "long",
            "bench_price":      None,
            "bench_delta":      None,
            "bench_gamma":      None,
            "bench_vega":       None,
        },

        # ==============================================================
        # 4.  Down-and-In Put  |  Quarterly monitoring  ->  MC forced
        #     4 dates/year; pricing_method="mc" overrides auto rule
        # ==============================================================
        {
            "trade_name":       "T04_DAI_Put_Quarterly_MC",
            "option_type":      "put",
            "barrier_type":     "down-and-in",
            "S":                229.74,
            "K":                235.00,
            "lower_barrier":    190.00,
            "sigma":            0.274159019046,
            "valuation_date":   VAL,
            "maturity_date":    MAT_1Y,
            "monitor_frequency":"quarterly",    # 4 dates: forced MC anyway
            "discount_curve":   DISC_85,
            "forward_curve":    FWD_91,
            "dividend_schedule":JSE_DIVS,
            "pricing_method":   "mc",           # explicit MC override
            "mc_n_paths":       500_000,
            "mc_seed":          7,
            "mc_use_antithetic":True,
            "day_count":        "ACT/365",
            "direction":        "long",
            "bench_price":      None,
            "bench_delta":      None,
            "bench_gamma":      None,
            "bench_vega":       None,
        },

        # ==============================================================
        # 5.  Double-Out Call  |  Fortnightly monitoring  ->  BGK path
        #     ~26 dates/year > bgk_min_freq (20) -> BGK auto-selected
        #     No dividends; T+3 equity settlement lags
        # ==============================================================
        {
            "trade_name":       "T05_DblOut_Call_Fortnightly_BGK",
            "option_type":      "call",
            "barrier_type":     "double-out",
            "S":                100.00,
            "K":                100.00,
            "lower_barrier":    82.00,
            "upper_barrier":    120.00,
            "sigma":            0.22,
            "valuation_date":   VAL,
            "maturity_date":    MAT_6M,
            "monitor_frequency":"fortnightly",  # ~26 dates/yr -> BGK
            "discount_curve":   DISC_85_6M,
            "forward_curve":    build_flat_curve(0.09, VAL, MAT_6M),
            "dividend_schedule":None,
            "pricing_method":   "auto",
            "bgk_min_freq":     20.0,
            "day_count":        "ACT/365",
            "underlying_spot_days":   3,
            "option_days":            0,
            "option_settlement_days": 0,
            "direction":        "long",
            "bench_price":      None,
            "bench_delta":      None,
            "bench_gamma":      None,
            "bench_vega":       None,
        },

        # ==============================================================
        # 6.  Double-In Put  |  Weekly monitoring  ->  MC forced
        #     Same double barrier as T05 but in-type, explicit MC
        # ==============================================================
        {
            "trade_name":       "T06_DblIn_Put_Weekly_MC",
            "option_type":      "put",
            "barrier_type":     "double-in",
            "S":                100.00,
            "K":                100.00,
            "lower_barrier":    82.00,
            "upper_barrier":    120.00,
            "sigma":            0.22,
            "valuation_date":   VAL,
            "maturity_date":    MAT_6M,
            "monitor_frequency":"weekly",
            "discount_curve":   DISC_85_6M,
            "forward_curve":    build_flat_curve(0.09, VAL, MAT_6M),
            "dividend_schedule":None,
            "pricing_method":   "mc",
            "mc_n_paths":       300_000,
            "mc_seed":          99,
            "mc_use_antithetic":True,
            "underlying_spot_days":   3,
            "option_days":            0,
            "option_settlement_days": 0,
            "day_count":        "ACT/365",
            "direction":        "long",
            "bench_price":      None,
            "bench_delta":      None,
            "bench_gamma":      None,
            "bench_vega":       None,
        },

        # ==============================================================
        # 7.  Up-and-Out Call  |  Weekly  |  Rebate at hit = 5.00
        #     With separate carry and discount curves + dividends
        # ==============================================================
        {
            "trade_name":       "T07_UAO_Call_Rebate_Weekly",
            "option_type":      "call",
            "barrier_type":     "up-and-out",
            "S":                229.74,
            "K":                215.00,
            "upper_barrier":    265.00,
            "sigma":            0.2613190156888,
            "valuation_date":   VAL,
            "maturity_date":    MAT_1Y,
            "monitor_frequency":"weekly",
            "rebate_amount":    5.00,
            "rebate_at_hit":    True,
            "discount_curve":   DISC_85,
            "forward_curve":    FWD_91,
            "dividend_schedule":JSE_DIVS,
            "pricing_method":   "auto",
            "bgk_min_freq":     20.0,
            "day_count":        "ACT/365",
            "direction":        "long",
            "bench_price":      None,
            "bench_delta":      None,
            "bench_gamma":      None,
            "bench_vega":       None,
        },

        # ==============================================================
        # 8.  Down-and-Out Call  |  Custom bi-weekly dates  ->  BGK
        #     Explicit monitor_dates list overrides monitor_frequency.
        #     Single dividend; no settlement lags.
        # ==============================================================
        {
            "trade_name":       "T08_DAO_Call_CustomDates_BGK",
            "option_type":      "call",
            "barrier_type":     "down-and-out",
            "S":                229.74,
            "K":                220.00,
            "lower_barrier":    195.00,
            "sigma":            0.2613190156888,
            "valuation_date":   VAL,
            "maturity_date":    MAT_1Y,
            # Custom every-two-weeks schedule (explicit dates)
            "monitor_dates": [
                VAL + dt.timedelta(days=14 * i)
                for i in range(1, 27)          # 26 monitoring dates
            ],
            "discount_curve":   DISC_85,
            "forward_curve":    FWD_91,
            "dividend_schedule":[
                (dt.date(2025, 9, 12), 7.63),
            ],
            "pricing_method":   "auto",
            "bgk_min_freq":     20.0,
            "day_count":        "ACT/365",
            "direction":        "long",
            "bench_price":      None,
            "bench_delta":      None,
            "bench_gamma":      None,
            "bench_vega":       None,
        },

        # ==============================================================
        # 9.  Vanilla call (barrier_type="none")  |  parity / sanity
        #     Price should equal Black-76 exactly.
        # ==============================================================
        {
            "trade_name":       "T09_Vanilla_Call_Sanity",
            "option_type":      "call",
            "barrier_type":     "none",
            "S":                229.74,
            "K":                220.00,
            "sigma":            0.2613190156888,
            "valuation_date":   VAL,
            "maturity_date":    MAT_1Y,
            "discount_curve":   DISC_85,
            "forward_curve":    FWD_91,
            "dividend_schedule":JSE_DIVS,
            "pricing_method":   "auto",
            "day_count":        "ACT/365",
            "direction":        "long",
            # Vanilla benchmarks (replace with Black-76 reference):
            "bench_price":      None,
            "bench_delta":      None,
            "bench_gamma":      None,
            "bench_vega":       None,
        },

        # ==============================================================
        # 10. Up-and-Out Call  |  Weekly  |  Short direction
        #     Demonstrates sign flip: model_price should be negative.
        # ==============================================================
        {
            "trade_name":       "T10_UAO_Call_Short",
            "option_type":      "call",
            "barrier_type":     "up-and-out",
            "S":                229.74,
            "K":                220.00,
            "upper_barrier":    270.00,
            "sigma":            0.2613190156888,
            "valuation_date":   VAL,
            "maturity_date":    MAT_1Y,
            "monitor_frequency":"weekly",
            "discount_curve":   DISC_85,
            "forward_curve":    FWD_91,
            "dividend_schedule":JSE_DIVS,
            "pricing_method":   "auto",
            "bgk_min_freq":     20.0,
            "day_count":        "ACT/365",
            "direction":        "short",
            "bench_price":      None,
            "bench_delta":      None,
            "bench_gamma":      None,
            "bench_vega":       None,
        },

        # ==============================================================
        # 11. Down-and-Out Put  |  Daily  |  3-month, 1 dividend
        #     Shorter tenor to test a 3-month window.
        # ==============================================================
        {
            "trade_name":       "T11_DAO_Put_Daily_3M",
            "option_type":      "put",
            "barrier_type":     "down-and-out",
            "S":                229.74,
            "K":                240.00,
            "lower_barrier":    205.00,
            "sigma":            0.2613190156888,
            "valuation_date":   VAL,
            "maturity_date":    MAT_3M,
            "monitor_frequency":"daily",
            "discount_curve":   DISC_85_3M,
            "forward_curve":    build_flat_curve(0.091, VAL, MAT_3M),
            "dividend_schedule":SHORT_DIVS,
            "pricing_method":   "auto",
            "bgk_min_freq":     20.0,
            "day_count":        "ACT/365",
            "direction":        "long",
            "bench_price":      None,
            "bench_delta":      None,
            "bench_gamma":      None,
            "bench_vega":       None,
        },

        # ==============================================================
        # 12. Already-Hit Down-and-Out  |  barrier already crossed
        #     Returns rebate PV only (option is dead).
        # ==============================================================
        {
            "trade_name":       "T12_DAO_Call_AlreadyHit",
            "option_type":      "call",
            "barrier_type":     "down-and-out",
            "S":                175.00,            # below lower_barrier
            "K":                220.00,
            "lower_barrier":    185.00,
            "sigma":            0.2613190156888,
            "valuation_date":   VAL,
            "maturity_date":    MAT_1Y,
            "monitor_frequency":"weekly",
            "already_hit":      True,
            "barrier_hit_date": dt.date(2025, 7, 14),
            "rebate_amount":    10.00,
            "rebate_at_hit":    True,
            "discount_curve":   DISC_85,
            "forward_curve":    FWD_91,
            "dividend_schedule":JSE_DIVS,
            "pricing_method":   "bgk",
            "day_count":        "ACT/365",
            "direction":        "long",
            "bench_price":      None,
            "bench_delta":      None,
            "bench_gamma":      None,
            "bench_vega":       None,
        },

    ]

    # ------------------------------------------------------------------
    # Run all scenarios
    # ------------------------------------------------------------------
    run_all_bgk_scenarios(
        trades=TRADES,
        output_csv=OUTPUT_CSV,
        print_results=True,
    )
