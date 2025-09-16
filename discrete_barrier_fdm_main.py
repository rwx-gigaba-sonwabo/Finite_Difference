import pandas as pd
import datetime as dt
from discrete_barrier_fdm_pricer import DiscreteBarrierFDMPricer

# --- Load curves (daily NACA, decimal) ---
discount_curve = pd.read_csv(
    r"C:\Finite Difference (Equity Americans)\Finite_Difference-main\ZAR-SWAP-28072025.csv"
)
discount_curve["Date"] = pd.to_datetime(
    discount_curve["Date"], format="%Y/%m/%d"
).dt.strftime("%Y-%m-%d")

forward_curve = pd.read_csv(
    r"C:\Finite Difference (Equity Americans)\Finite_Difference-main\ZAR-SWAP-28072025.csv"
)
forward_curve["Date"] = pd.to_datetime(
    forward_curve["Date"], format="%Y/%m/%d"
).dt.strftime("%Y-%m-%d")

# --- Dates ---
valuation = dt.date(2025, 7, 28)
maturity  = dt.date(2026, 7, 28)

# --- Dividend schedule: list of (pay_date, cash_amount) ---
divs = [
    (dt.date(2025, 9, 12), 7.63),
    (dt.date(2026, 4, 10), 8.0115),
]

# --- Monitoring dates (e.g., weekly) ---
monitor_dates = [valuation + dt.timedelta(days=7 * i) for i in range(1, 27)]

# --- Build pricer (trade details in __init__; choose day count as needed) ---
pricer = DiscreteBarrierFDMPricer(
    spot=229.74,
    strike=190.0,
    valuation_date=valuation,
    maturity_date=maturity,
    sigma=0.274159019046,
    option_type="call",
    barrier_type="down-and-out",
    lower_barrier=180.0,
    monitor_dates=monitor_dates,
    discount_curve=discount_curve,
    forward_curve=forward_curve,     # optional; used for get_forward_curve_nacc if needed
    dividend_schedule=divs,
    # Trade details (moved into __init__)
    trade_id="ZA-DO-CALL-001",
    direction="long",
    quantity=1,
    contract_multiplier=1.0,
    # Numerical knobs
    grid_points=500,
    time_steps=500,
    rannacher_steps=2,
    restart_on_monitoring=True,
    price_extrapolation=True,
    # Day count for year fractions (matches your framework options)
    day_count="ACT/365",             # or "ACT/360", "30/360", etc.
)

if __name__ == "__main__":
    print(pricer.report())
    print("\nPrice:", pricer.price())
    print("Greeks:", pricer.greeks())