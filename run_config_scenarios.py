import pandas as pd  # type: ignore
import numpy as np # type: ignore
import datetime as dt
from discrete_barrier_fdm_pricer import DiscreteBarrierFDMPricer
from utils import create_rate_df, naca_to_nacc
from typing import Dict, Any, Optional


def run_scenario(
    scenario_name: str,
    S0: float,
    K: float,
    sigma: float,
    rate: float,
    barrier_type: str,
    upper_barrier: Optional[float],
    lower_barrier: Optional[float],
    FA_price: Optional[float],
    FA_delta: Optional[float],
    FA_gamma: Optional[float],
    FA_vega: Optional[float],
    # Base parameters that stay constant across runs
    valuation: dt.date,
    maturity: dt.date,
    monitor_dates: list,
    opt_type: str = "call",
    trade_number: int = 201871103,
    quantity: int = 1000,
    contract_size: int = 1,
    position: str = "long",
    divs: list = None,
    rebate_amount: float = 0,
    rebate_at_hit: bool = True,
    use_one_sided_greeks_near_barrier: bool = False,
    already_hit: bool = False,
    already_in: bool = False,
    underlying_spot_days: int = 0,
    option_days: int = 0,
    option_settlement_days: int = 0,
    day_count: str = "ACT/365",
    grid_type: str = "uniform",
    num_space_nodes: int = 500,
    num_time_steps: int = 500,
) -> Dict[str, Any]:
    """
    Run a single pricing scenario and return headline results.
    """
    if divs is None:
        divs = []

    # Convert NACA rate to NACC rate for the pricer
    nacc_rate = naca_to_nacc(rate)

    # Create rate curves for this scenario (using NACA rate as expected by create_rate_df)
    discount_curve = create_rate_df(rate)
    forward_curve = create_rate_df(rate)
    discount_curve["Date"] = pd.to_datetime(discount_curve["Date"], format="%Y/%m/%d").dt.strftime("%Y-%m-%d")
    forward_curve["Date"] = pd.to_datetime(forward_curve["Date"], format="%Y/%m/%d").dt.strftime("%Y-%m-%d")

    pricer = DiscreteBarrierFDMPricer(
        spot=S0,
        strike=K,
        valuation_date=valuation,
        maturity_date=maturity,
        sigma=sigma,
        option_type=opt_type,
        barrier_type=barrier_type,
        lower_barrier=lower_barrier,
        upper_barrier=upper_barrier,
        already_in=already_in,
        already_hit=already_hit,
        monitor_dates=monitor_dates,
        discount_curve=discount_curve,
        forward_curve=forward_curve,
        dividend_schedule=divs,
        trade_id=trade_number,
        direction=position,
        quantity=quantity,
        underlying_spot_days=underlying_spot_days,
        option_days=option_days,
        option_settlement_days=option_settlement_days,
        rebate_amount=rebate_amount,
        rebate_at_hit=rebate_at_hit,
        contract_multiplier=contract_size,
        use_one_sided_greeks_near_barrier=use_one_sided_greeks_near_barrier,
        num_space_nodes=num_space_nodes,
        num_time_steps=num_time_steps,
        grid_type=grid_type,
        rannacher_steps=2,
        restart_on_monitoring=False,
        mollify_final=False,
        mollify_band_nodes=2,
        day_count=day_count
    )

    # Calculate price and greeks
    model_price = pricer.price_log2()
    greeks = pricer.greeks_log2()

    # Helper function to calculate percentage difference
    def pct_diff(model_val: float, fa_val: Optional[float]) -> float:
        if fa_val is None or np.isnan(fa_val) or fa_val == 0.0:
            return np.nan
        return abs(model_val - fa_val) / abs(fa_val) * 100.0

    # Return results
    results = {
        "scenario_name": scenario_name,
        "S0": S0,
        "K": K,
        "sigma": sigma,
        "rate": rate,
        "barrier_type": barrier_type,
        "upper_barrier": upper_barrier if upper_barrier is not None else np.nan,
        "lower_barrier": lower_barrier if lower_barrier is not None else np.nan,
        "model_price": model_price,
        "FA_price": FA_price if FA_price is not None else np.nan,
        "price_diff": abs(model_price - FA_price) if FA_price is not None else np.nan,
        "price_pct_diff": pct_diff(model_price, FA_price),
        "model_delta": greeks['delta'],
        "FA_delta": FA_delta if FA_delta is not None else np.nan,
        "delta_diff": abs(greeks['delta'] - FA_delta) if FA_delta is not None else np.nan,
        "delta_pct_diff": pct_diff(greeks['delta'], FA_delta),
        "model_gamma": greeks['gamma'],
        "FA_gamma": FA_gamma if FA_gamma is not None else np.nan,
        "gamma_diff": abs(greeks['gamma'] - FA_gamma) if FA_gamma is not None else np.nan,
        "gamma_pct_diff": pct_diff(greeks['gamma'], FA_gamma),
        "model_vega": greeks['vega'],
        "FA_vega": FA_vega if FA_vega is not None else np.nan,
        "vega_diff": abs(greeks['vega'] - FA_vega) if FA_vega is not None else np.nan,
        "vega_pct_diff": pct_diff(greeks['vega'], FA_vega),
    }

    return results


def run_all_scenarios(
    config_csv_path: str,
    output_csv_path: str,
    base_params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Read configuration CSV, run all scenarios, and save results to output CSV.

    Args:
        config_csv_path: Path to configuration CSV file
        output_csv_path: Path to save results CSV file
        base_params: Dictionary of base parameters that stay constant

    Returns:
        DataFrame containing all results
    """
    # Read configuration CSV
    config_df = pd.read_csv(config_csv_path)

    # Run each scenario
    all_results = []
    for _, row in config_df.iterrows():
        print(f"\nRunning {row['scenario_name']}...")

        # Handle NaN values for barriers and FA values
        upper_barrier = row['upper_barrier'] if pd.notna(row['upper_barrier']) else None
        lower_barrier = row['lower_barrier'] if pd.notna(row['lower_barrier']) else None
        FA_price = row['FA_price'] if pd.notna(row['FA_price']) else None
        FA_delta = row['FA_delta'] if pd.notna(row['FA_delta']) else None
        FA_gamma = row['FA_gamma'] if pd.notna(row['FA_gamma']) else None
        FA_vega = row['FA_vega'] if pd.notna(row['FA_vega']) else None

        result = run_scenario(
            scenario_name=row['scenario_name'],
            S0=row['S0'],
            K=row['K'],
            sigma=row['sigma'],
            rate=row['rate'],
            barrier_type=row['barrier_type'],
            upper_barrier=upper_barrier,
            lower_barrier=lower_barrier,
            FA_price=FA_price,
            FA_delta=FA_delta,
            FA_gamma=FA_gamma,
            FA_vega=FA_vega,
            **base_params
        )

        all_results.append(result)
        print(f"  Price %Diff: {result['price_pct_diff']:.4f}%, Delta %Diff: {result['delta_pct_diff']:.4f}%, Gamma %Diff: {result['gamma_pct_diff']:.4f}%, Vega %Diff: {result['vega_pct_diff']:.4f}%")

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Save to CSV
    results_df.to_csv(output_csv_path, index=False)
    print(f"\n✓ Results saved to {output_csv_path}")

    return results_df


if __name__ == "__main__":
    # Dates
    valuation = dt.date(2025, 7, 28)
    maturity = dt.date(2025, 8, 28)
    
    # Monitoring dates
    monitor_dates = [
        dt.date(2025, 7, 28),
        dt.date(2025, 7, 29),
        dt.date(2025, 7, 30),
        dt.date(2025, 7, 31),
        dt.date(2025, 8, 1),
        dt.date(2025, 8, 4),
        dt.date(2025, 8, 5),
        dt.date(2025, 8, 6),
        dt.date(2025, 8, 7),
        dt.date(2025, 8, 8),
        dt.date(2025, 8, 11),
        dt.date(2025, 8, 12),
        dt.date(2025, 8, 13),
        dt.date(2025, 8, 14),
        dt.date(2025, 8, 15),
        dt.date(2025, 8, 18),
        dt.date(2025, 8, 19),
        dt.date(2025, 8, 20),
        dt.date(2025, 8, 21),
        dt.date(2025, 8, 22),
        dt.date(2025, 8, 25),
        dt.date(2025, 8, 26),
        dt.date(2025, 8, 27),
        dt.date(2025, 8, 28),
    ]

    # Dividend schedule
    divs = []

    # Base parameters dictionary
    base_params = {
        "valuation": valuation,
        "maturity": maturity,
        "monitor_dates": monitor_dates,
        "opt_type": "put",
        "trade_number": 201871100,
        "quantity": 1000,
        "contract_size": 1,
        "position": "long",
        "divs": divs,
        "rebate_amount": 0,
        "rebate_at_hit": True,
        "use_one_sided_greeks_near_barrier": False,
        "already_hit": False,
        "already_in": False,
        "underlying_spot_days": 0,
        "option_days": 0,
        "option_settlement_days": 0,
        "day_count": "ACT/365",
        "grid_type": "uniform",
        "num_space_nodes": 500,
        "num_time_steps": 500,
    }

    # Run all scenarios
    results = run_all_scenarios(
        config_csv_path="config_scenarios_1.csv",
        output_csv_path="scenario_results_1.csv",
        base_params=base_params
    )

    # Display results
    print("\n=== All Scenario Results ===")
    print(results.to_string(index=False))