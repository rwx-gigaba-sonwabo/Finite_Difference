import datetime as dt
import pandas as pd

from discrete_barrier_bgk import DiscreteBarrierBGKPricer

# Load curves (your exact pattern)
discount_curve = pd.read_csv(r"C:\Finite Difference (Equity Americans)\Finite_Difference-main\ZAR-SWAP-28072025.csv")
discount_curve["Date"] = pd.to_datetime(discount_curve["Date"], format="%Y/%m/%d").dt.strftime("%Y-%m-%d")

forward_curve  = pd.read_csv(r"C:\Finite Difference (Equity Americans)\Finite_Difference-main\ZAR-SWAP-28072025.csv")
forward_curve["Date"] = pd.to_datetime(forward_curve["Date"], format="%Y/%m/%d").dt.strftime("%Y-%m-%d")

# Trade dates
valuation = dt.date(2025, 7, 28)
maturity  = dt.date(2026, 7, 28)

# Dividends (date, cash amount) in same units as spot
dividends = [
    (dt.date(2025, 9, 12), 7.63),
    (dt.date(2026, 4, 10), 8.0115),
]

# Monitoring schedule (example weekly; replace with your business-day schedule)
monitoring_dates = [valuation + dt.timedelta(days=7*i) for i in range(1, 27)]

# Instantiate pricer
pricer = DiscreteBarrierBGKPricer(
    trade_id="ZA-DO-CALL-001",
    direction="long",
    quantity=1,
    contract_multiplier=1.0,
    spot=229.74,
    strike=190.0,
    valuation_date=valuation,
    maturity_date=maturity,
    option_type="call",
    barrier_type="down-and-out",  # up-and-out / in variants / double-* supported
    lower_barrier=180.0,
    upper_barrier=None,
    monitoring_dates=monitoring_dates,
    discount_curve=discount_curve,
    forward_curve=forward_curve,     # optional, for QA/reporting
    dividend_schedule=dividends,
    volatility=0.274159019046,
    day_count="ACT/365",
)

print(pricer.report())