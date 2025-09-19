# discrete_barrier_fdm_main_CN.py
# -*- coding: utf-8 -*-

import math
import pandas as pd
import datetime as dt

from discrete_barrier_fdm_pricer_cn import DiscreteBarrierFDMPricerCN

# ------------------------------ Curves --------------------------------
# Example: load a CSV with Date (YYYY-MM-DD) and NACA columns.
discount_curve = pd.DataFrame({
    "Date": ["2025-07-28", "2025-08-28"],
    "NACA": [0.08, 0.08],
})
discount_curve["Date"] = pd.to_datetime(discount_curve["Date"]).dt.strftime("%Y-%m-%d")

def df_from_naca(valuation_date: dt.date, lookup_date: dt.date) -> float:
    row = discount_curve[discount_curve["Date"] == lookup_date.isoformat()]
    if row.empty:
        raise ValueError(f"Curve date {lookup_date.isoformat()} not in CSV.")
    naca = float(row["NACA"].values[0])
    tau = (lookup_date - valuation_date).days / 365.0
    return (1.0 + naca) ** (-tau)

# ------------------------------ Dates ---------------------------------
valuation = dt.date(2025, 7, 28)
maturity  = dt.date(2025, 8, 28)
T_years   = (maturity - valuation).days / 365.0

DF_T = df_from_naca(valuation, maturity)
r_flat = -math.log(DF_T) / max(1e-12, T_years)

# ----------------------------- Instrument -----------------------------
S0       = 229.74
K        = 220.00
sigma    = 0.2613190156888
opt_type = "call"                 # "call" or "put"

barrier_type  = "up-and-out"      # also supports KI via "up-and-in", etc.
lower_barrier = None
upper_barrier = 270.0
rebate_amount = 1.0
rebate_at_hit = True

# Monitoring dates (discrete)
monitoring_dates = [
    dt.date(2025, 7, 28),
    dt.date(2025, 8, 4),
    dt.date(2025, 8, 7),
    dt.date(2025, 8, 12),
    dt.date(2025, 8, 18),
    dt.date(2025, 8, 22),
    dt.date(2025, 8, 27),
]

# Cash dividends (optional)
dividends = []  # e.g., [(dt.date(2025, 9, 12), 7.63)]

# ------------------------------- Numerics -----------------------------
num_space_nodes = 600
num_time_steps  = 600
rannacher_steps = 2

# ------------------------------ Build/run -----------------------------
pricer = DiscreteBarrierFDMPricerCN(
    spot=S0,
    strike=K,
    valuation_date=valuation,
    maturity_date=maturity,
    volatility=sigma,
    option_type=opt_type,
    barrier_type=barrier_type,
    lower_barrier=lower_barrier,
    upper_barrier=upper_barrier,
    monitoring_dates=monitoring_dates,
    rebate_amount=rebate_amount,
    rebate_at_hit=rebate_at_hit,
    flat_rate_nacc=r_flat,
    dividend_list=dividends,
    num_space_nodes=num_space_nodes,
    num_time_steps=num_time_steps,
    rannacher_steps=rannacher_steps,
    day_count="ACT/365",
    smooth_payoff_around_strike=True,
    payoff_smoothing_half_width_nodes=2,
    use_one_sided_greeks_near_barrier=True,
    barrier_safety_cells=2,
)

pricer.print_details()
print("\\nPrice:", pricer.price())
print("Greeks:", pricer.greeks())