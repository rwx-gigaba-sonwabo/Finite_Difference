# discrete_barrier_main.py
import pandas as pd
import datetime as dt
from discrete_barrier_fdm_pricer import DiscreteBarrierFDMPricer

# ---- Load curves (daily NACA) ----
discount_curve = pd.read_csv(r"C:\Finite Difference (Equity Americans)\Finite_Difference-main\ZAR-SWAP-28072025.csv")
discount_curve["Date"] = pd.to_datetime(discount_curve["Date"], format="%Y/%m/%d").dt.strftime("%Y-%m-%d")

# optional separate forward curve, otherwise discount_curve will be used
forward_curve = discount_curve.copy()

# ---- Dates ----
valuation = dt.date(2025, 7, 28)
maturity  = dt.date(2026, 7, 28)

# ---- Dividends (pay_date, amount) ----
divs = [
    (dt.date(2025, 9, 12), 7.63),
    (dt.date(2026, 4, 10), 8.0115),
]

# ---- Monitoring dates (example: weekly) ----
monitor_dates = [valuation + dt.timedelta(days=7*i) for i in range(1, 27)]

# ---- Instantiate pricer ----
pricer = DiscreteBarrierFDMPricer(
    spot=229.74,
    strike=220.00,
    valuation_date=valuation,
    maturity_date=maturity,
    sigma=0.2613190156888,
    option_type="call",
    barrier_type="up-and-out",
    lower_barrier=None,
    upper_barrier=270.00,
    monitor_dates=monitor_dates,
    discount_curve=discount_curve,
    forward_curve=forward_curve,
    dividend_schedule=divs,
    grid_points=500,
    time_steps=500,
    rannacher_steps=2,
    flat_r_nacc=0.0,            # will auto-compute from curve; set explicitly to override
    day_count="ACT/365",
    mollify_final=True,
    mollify_band_nodes=2,
)

# (Optional) If you want to explicitly use curve-implied flat r:
pricer.flat_r_nacc = pricer.flat_r_from_curve()

# ---- Print a full report (includes price & Greeks) ----
pricer.print_details(
    trade_id="201871085",
    direction="long",
    quantity=1,
    contract_multiplier=1.0
)
