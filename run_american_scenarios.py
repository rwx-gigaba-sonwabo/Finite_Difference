"""
Batch runner for American option finite-difference scenarios.

This script reads a configuration CSV of scenario inputs, instantiates
the AmericanFDMPricer for each scenario, computes price and Greeks, and
writes a results CSV with differences versus Front Arena (FA) values.

It is designed to mirror the structure of the discrete barrier scenario
runner for consistency.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from fd_american_equity import AmericanFDMPricer
from utils import create_rate_df, naca_to_nacc


def _percentage_diff(model_val: float, fa_val: Optional[float]) -> float:
    """
    Compute the absolute percentage difference between model and FA values.

    Parameters
    ----------
    model_val :
        Value produced by the validation model.
    fa_val :
        Corresponding value from Front Arena. If None, NaN is returned.

    Returns
    -------
    float
        Absolute percentage difference (|model - FA| / |FA| * 100),
        or NaN if the FA value is missing or zero.
    """
    if fa_val is None or np.isnan(fa_val) or fa_val == 0.0:
        return np.nan
    return abs(model_val - fa_val) / abs(fa_val) * 100.0


def run_american_scenario(
    scenario_name: str,
    S0: float,
    K: float,
    sigma: float,
    rate: float,
    FA_price: Optional[float],
    FA_delta: Optional[float],
    FA_gamma: Optional[float],
    FA_vega: Optional[float],
    # Base parameters that stay constant across runs
    valuation: dt.date,
    maturity: dt.date,
    opt_type: str = "call",
    trade_number: int = 201871103,
    quantity: int = 1000,
    contract_size: int = 1,
    position: str = "long",
    divs: Optional[list] = None,
    underlying_spot_days: int = 0,
    option_days: int = 0,
    option_settlement_days: int = 0,
    day_count: str = "ACT/365",
    grid_type: str = "uniform",
    num_space_nodes: int = 500,
    num_time_steps: int = 500,
    rannacher_steps: int = 2,
) -> Dict[str, Any]:
    """
    Run a single American-option scenario and return headline results.

    Layout mirrors the discrete-barrier run_scenario, but without barrier
    fields.

    Parameters
    ----------
    scenario_name :
        Identifier for the scenario (e.g. "scenario_1").
    S0 :
        Spot price of the underlying.
    K :
        Strike price.
    sigma :
        Volatility (per annum, in decimal).
    rate :
        Flat NACA rate used to generate discount and forward curves.
    FA_price, FA_delta, FA_gamma, FA_vega :
        Front Arena benchmark values for price and Greeks.
    valuation :
        Valuation date.
    maturity :
        Maturity date of the option.
    opt_type :
        Option type, "call" or "put".
    trade_number :
        Trade identifier used when instantiating the pricer.
    quantity :
        Number of contracts.
    contract_size :
        Contract multiplier.
    position :
        "long" or "short".
    divs :
        Dividend schedule as a list of (date, cash_amount) pairs.
    underlying_spot_days, option_days, option_settlement_days :
        Business-day lags for underlying spot, option spot, and settlement.
    day_count :
        Day-count convention for the FD engine (e.g. "ACT/365").
    grid_type :
        Spatial grid type ("uniform").
    num_space_nodes, num_time_steps :
        Grid dimensions.
    rannacher_steps :
        Number of Rannacher (fully implicit) steps at segment starts.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing scenario inputs, model outputs, FA
        benchmarks, and absolute / percentage differences.
    """
    if divs is None:
        divs = []

    # Convert NACA to NACC if required elsewhere; kept here for symmetry
    _nacc_rate = naca_to_nacc(rate)  # noqa: F841  # retained for debugging

    # Create flat NACA curves and convert to ISO date format expected by FDM
    discount_curve = create_rate_df(rate)
    forward_curve = create_rate_df(rate)

    discount_curve["Date"] = pd.to_datetime(
        discount_curve["Date"], format="%Y/%m/%d"
    ).dt.strftime("%Y-%m-%d")
    forward_curve["Date"] = pd.to_datetime(
        forward_curve["Date"], format="%Y/%m/%m"
    ).dt.strftime("%Y-%m-%d")

    # Instantiate American FDM pricer (same curve / date style as barrier engine)
    pricer = AmericanFDMPricer(
        spot=S0,
        strike=K,
        valuation_date=valuation,
        maturity_date=maturity,
        sigma=sigma,
        option_type=opt_type,
        discount_curve=discount_curve,
        forward_curve=forward_curve,
        dividend_schedule=divs,
        trade_id=trade_number,
        direction=position,
        quantity=quantity,
        contract_multiplier=contract_size,
        underlying_spot_days=underlying_spot_days,
        option_days=option_days,
        option_settlement_days=option_settlement_days,
        day_count=day_count,
        grid_type=grid_type,
        num_space_nodes=num_space_nodes,
        num_time_steps=num_time_steps,
        rannacher_steps=rannacher_steps,
    )

    # Price and Greeks from the FD pricer
    model_price = pricer.price_log2()
    greeks = pricer.greeks_log2()

    # Collect everything into a dict for CSV/export
    results: Dict[str, Any] = {
        "scenario_name": scenario_name,
        "S0": S0,
        "K": K,
        "sigma": sigma,
        "rate": rate,
        "model_price": model_price,
        "FA_price": FA_price if FA_price is not None else np.nan,
        "price_diff": (
            abs(model_price - FA_price) if FA_price is not None else np.nan
        ),
        "price_pct_diff": _percentage_diff(model_price, FA_price),
        "model_delta": greeks["delta"],
        "FA_delta": FA_delta if FA_delta is not None else np.nan,
        "delta_diff": (
            abs(greeks["delta"] - FA_delta) if FA_delta is not None else np.nan
        ),
        "delta_pct_diff": _percentage_diff(greeks["delta"], FA_delta),
        "model_gamma": greeks["gamma"],
        "FA_gamma": FA_gamma if FA_gamma is not None else np.nan,
        "gamma_diff": (
            abs(greeks["gamma"] - FA_gamma) if FA_gamma is not None else np.nan
        ),
        "gamma_pct_diff": _percentage_diff(greeks["gamma"], FA_gamma),
        "model_vega": greeks["vega"],
        "FA_vega": FA_vega if FA_vega is not None else np.nan,
        "vega_diff": (
            abs(greeks["vega"] - FA_vega) if FA_vega is not None else np.nan
        ),
        "vega_pct_diff": _percentage_diff(greeks["vega"], FA_vega),
    }

    return results


def run_all_american_scenarios(
    config_csv_path: str,
    output_csv_path: str,
    base_params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Read configuration CSV, run all American scenarios, and save results.

    Parameters
    ----------
    config_csv_path :
        Path to the configuration CSV with scenario inputs.
    output_csv_path :
        Path where the scenario results CSV will be written.
    base_params :
        Dictionary of base parameters passed verbatim into
        :func:`run_american_scenario` for each row.

    Expected columns in the config CSV
    ----------------------------------
    scenario_name, S0, K, sigma, rate, FA_price, FA_delta, FA_gamma, FA_vega

    Returns
    -------
    pandas.DataFrame
        DataFrame of results for all scenarios.
    """
    config_df = pd.read_csv(config_csv_path)

    all_results: list[Dict[str, Any]] = []
    for _, row in config_df.iterrows():
        print(f"\nRunning {row['scenario_name']}...")

        # Handle NaNs in FA columns
        fa_price = row["FA_price"] if pd.notna(row["FA_price"]) else None
        fa_delta = row["FA_delta"] if pd.notna(row["FA_delta"]) else None
        fa_gamma = row["FA_gamma"] if pd.notna(row["FA_gamma"]) else None
        fa_vega = row["FA_vega"] if pd.notna(row["FA_vega"]) else None

        result = run_american_scenario(
            scenario_name=row["scenario_name"],
            S0=row["S0"],
            K=row["K"],
            sigma=row["sigma"],
            rate=row["rate"],
            FA_price=fa_price,
            FA_delta=fa_delta,
            FA_gamma=fa_gamma,
            FA_vega=fa_vega,
            **base_params,
        )

        all_results.append(result)
        print(
            "  Price %Diff: "
            f"{result['price_pct_diff']:.4f}%, "
            "Delta %Diff: "
            f"{result['delta_pct_diff']:.4f}%, "
            "Gamma %Diff: "
            f"{result['gamma_pct_diff']:.4f}%, "
            "Vega %Diff: "
            f"{result['vega_pct_diff']:.4f}%"
        )

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_csv_path, index=False)
    print(f"\n✓ American option results saved to {output_csv_path}")

    return results_df


if __name__ == "__main__":
    # Example base setup (mirrors the barrier script pattern)

    valuation_date = dt.date(2025, 7, 28)
    maturity_date = dt.date(2025, 8, 28)

    # Dividend schedule: list of (date, cash_amount)
    example_divs: list[tuple[dt.date, float]] = []
    # e.g. example_divs = [(dt.date(2025, 8, 10), 1.50)]

    base_parameters: Dict[str, Any] = {
        "valuation": valuation_date,
        "maturity": maturity_date,
        "opt_type": "put",  # or "call"
        "trade_number": 201871200,
        "quantity": 1000,
        "contract_size": 1,
        "position": "long",
        "divs": example_divs,
        "underlying_spot_days": 0,
        "option_days": 0,
        "option_settlement_days": 0,
        "day_count": "ACT/365",
        "grid_type": "uniform",
        "num_space_nodes": 500,
        "num_time_steps": 500,
        "rannacher_steps": 2,
    }

    results_df = run_all_american_scenarios(
        config_csv_path="american_config_scenarios.csv",
        output_csv_path="american_scenario_results.csv",
        base_params=base_parameters,
    )

    print("\n=== All American Scenario Results ===")
    print(results_df.to_string(index=False))
