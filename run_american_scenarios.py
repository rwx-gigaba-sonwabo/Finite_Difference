import pandas as pd  # type: ignore
import numpy as np   # type: ignore
import datetime as dt
from typing import Dict, Any, Optional

from fd_american_equity import AmericanFDMPricer
from utils import create_rate_df, naca_to_nacc


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
    Layout mirrors the discrete barrier run_scenario, but without barrier fields.
    """
    if divs is None:
        divs = []

    # Convert NACA to NACC if you need it elsewhere (kept here for symmetry)
    nacc_rate = naca_to_nacc(rate)  # noqa: F841  # kept for potential logging / debugging

    # Create flat NACA curves and convert to ISO format the way your FDM expects
    discount_curve = create_rate_df(rate)
    forward_curve = create_rate_df(rate)
    discount_curve["Date"] = pd.to_datetime(
        discount_curve["Date"], format="%Y/%m/%d"
    ).dt.strftime("%Y-%m-%d")
    forward_curve["Date"] = pd.to_datetime(
        forward_curve["Date"], format="%Y/%m/%d"
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

    # --- PRICE + GREEKS ---
    # If your class methods are named differently, adjust these two lines
    model_price = pricer.price_log2()
    greeks = pricer.greeks_log2()

    # Helper: percentage difference vs FA
    def pct_diff(model_val: float, fa_val: Optional[float]) -> float:
        if fa_val is None or np.isnan(fa_val) or fa_val == 0.0:
            return np.nan
        return abs(model_val - fa_val) / abs(fa_val) * 100.0

    # Collect everything into a dict for CSV/export
    results = {
        "scenario_name": scenario_name,
        "S0": S0,
        "K": K,
        "sigma": sigma,
        "rate": rate,
        "model_price": model_price,
        "FA_price": FA_price if FA_price is not None else np.nan,
        "price_diff": abs(model_price - FA_price) if FA_price is not None else np.nan,
        "price_pct_diff": pct_diff(model_price, FA_price),
        "model_delta": greeks["delta"],
        "FA_delta": FA_delta if FA_delta is not None else np.nan,
        "delta_diff": abs(greeks["delta"] - FA_delta) if FA_delta is not None else np.nan,
        "delta_pct_diff": pct_diff(greeks["delta"], FA_delta),
        "model_gamma": greeks["gamma"],
        "FA_gamma": FA_gamma if FA_gamma is not None else np.nan,
        "gamma_diff": abs(greeks["gamma"] - FA_gamma) if FA_gamma is not None else np.nan,
        "gamma_pct_diff": pct_diff(greeks["gamma"], FA_gamma),
        "model_vega": greeks["vega"],
        "FA_vega": FA_vega if FA_vega is not None else np.nan,
        "vega_diff": abs(greeks["vega"] - FA_vega) if FA_vega is not None else np.nan,
        "vega_pct_diff": pct_diff(greeks["vega"], FA_vega),
    }

    return results


def run_all_american_scenarios(
    config_csv_path: str,
    output_csv_path: str,
    base_params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Read configuration CSV, run all American scenarios, and save results to output CSV.

    Expected columns in the config CSV:
        scenario_name, S0, K, sigma, rate,
        FA_price, FA_delta, FA_gamma, FA_vega
    """
    config_df = pd.read_csv(config_csv_path)

    all_results = []
    for _, row in config_df.iterrows():
        print(f"\nRunning {row['scenario_name']}...")

        # Handle NaNs in FA columns
        FA_price = row["FA_price"] if pd.notna(row["FA_price"]) else None
        FA_delta = row["FA_delta"] if pd.notna(row["FA_delta"]) else None
        FA_gamma = row["FA_gamma"] if pd.notna(row["FA_gamma"]) else None
        FA_vega = row["FA_vega"] if pd.notna(row["FA_vega"]) else None

        result = run_american_scenario(
            scenario_name=row["scenario_name"],
            S0=row["S0"],
            K=row["K"],
            sigma=row["sigma"],
            rate=row["rate"],
            FA_price=FA_price,
            FA_delta=FA_delta,
            FA_gamma=FA_gamma,
            FA_vega=FA_vega,
            **base_params,
        )

        all_results.append(result)
        print(
            f"  Price %Diff: {result['price_pct_diff']:.4f}%, "
            f"Delta %Diff: {result['delta_pct_diff']:.4f}%, "
            f"Gamma %Diff: {result['gamma_pct_diff']:.4f}%, "
            f"Vega %Diff: {result['vega_pct_diff']:.4f}%"
        )

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_csv_path, index=False)
    print(f"\n✓ American option results saved to {output_csv_path}")

    return results_df


if __name__ == "__main__":
    # --- Example base setup (mirror your barrier script pattern) ---

    # Dates
    valuation = dt.date(2025, 7, 28)
    maturity = dt.date(2025, 8, 28)

    # Dividend schedule: list of (date, cash_amount)
    divs = []  # e.g. [(dt.date(2025, 8, 10), 1.50)]

    base_params = {
        "valuation": valuation,
        "maturity": maturity,
        "opt_type": "put",               # or "call"
        "trade_number": 201871200,
        "quantity": 1000,
        "contract_size": 1,
        "position": "long",
        "divs": divs,
        "underlying_spot_days": 0,
        "option_days": 0,
        "option_settlement_days": 0,
        "day_count": "ACT/365",
        "grid_type": "uniform",
        "num_space_nodes": 500,
        "num_time_steps": 500,
        "rannacher_steps": 2,
    }

    # Run batch
    results = run_all_american_scenarios(
        config_csv_path="american_config_scenarios.csv",
        output_csv_path="american_scenario_results.csv",
        base_params=base_params,
    )

    print("\n=== All American Scenario Results ===")
    print(results.to_string(index=False))
