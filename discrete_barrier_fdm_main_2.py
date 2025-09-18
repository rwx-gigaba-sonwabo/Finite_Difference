# discrete_barrier_fdm_main.py
# -*- coding: utf-8 -*-

import pandas as pd
import datetime as dt

from discrete_barrier_fdm_pricer_2 import DiscreteBarrierFDMPricer2

# ---------------------------------------------------------------------
# 1) Load curves (CSV with columns: Date, NACA). Dates "YYYY-MM-DD".
#    If your CSV is "YYYY/MM/DD", adjust the format below accordingly.
# ---------------------------------------------------------------------
discount_curve = pd.read_csv(
    r"C:\Finite Difference (Equity Americans)\Finite_Difference-main\ZAR-SWAP-28072025.csv"
)
# Normalize date column to ISO
discount_curve["Date"] = pd.to_datetime(discount_curve["Date"]).dt.strftime("%Y-%m-%d")

# Helper: discount factor to a lookup date using the CSV NACA for that date
def df_from_naca(valuation_date: dt.date, lookup_date: dt.date) -> float:
    """Compute DF(valuation->lookup) = (1+NACA)^(-tau)."""
    row = discount_curve[discount_curve["Date"] == lookup_date.isoformat()]
    if row.empty:
        raise ValueError(f"Curve date {lookup_date.isoformat()} not found in CSV.")
    naca = float(row["NACA"].values[0])
    tau = (lookup_date - valuation_date).days / 365.0  # ACT/365 to match the pricer default
    return (1.0 + naca) ** (-tau)

# ---------------------------------------------------------------------
# 2) Trade / dates
# ---------------------------------------------------------------------
valuation = dt.date(2025, 7, 28)
maturity  = dt.date(2025, 8, 28)  # adjust if needed

T = (maturity - valuation).days / 365.0  # ACT/365

# Flat continuous short rate r from DF(T): r = -ln(DF)/T
DF_T = df_from_naca(valuation, maturity)
r_flat = - (0.0 if T <= 0 else (pd.np.log(DF_T) / T))  # pd.np to avoid extra import; or use math.log

# ---------------------------------------------------------------------
# 3) Dividends (optional) — cash amounts with calendar pay dates
#    Example list; set to [] if none.
# ---------------------------------------------------------------------
dividends = [] [
    # (dt.date(2025, 9, 12), 7.63),
    # (dt.date(2026, 4, 10), 8.0115),
]
# If no dividends, set dividends = []

# ---------------------------------------------------------------------
# 4) Monitoring dates (discrete). Example: specific dates or weekly.
#    These should be valuation < d <= maturity for KO checks.
# ---------------------------------------------------------------------
monitoring_dates = [
    dt.date(2025, 7, 28),
    dt.date(2025, 8, 4),
    dt.date(2025, 8, 7),
    dt.date(2025, 8, 12),
    dt.date(2025, 8, 18),
    dt.date(2025, 8, 22),
    dt.date(2025, 8, 27),
]

# ---------------------------------------------------------------------
# 5) Economics
# ---------------------------------------------------------------------
S0 = 229.74               # spot in the same units as strike/barriers
K  = 220.00               # strike
sigma = 0.2613190156888   # annualized vol
opt_type = "call"         # "call" or "put"

barrier_type = "up-and-out"   # "down-and-out", "up-and-out", "double-out", "down-and-in", "up-and-in", "double-in", or "none"
lower_barrier = None
upper_barrier = 270.00

# ---------------------------------------------------------------------
# 6) Numerical grid controls
# ---------------------------------------------------------------------
num_space_nodes = 600
num_time_steps  = 600
rannacher_steps = 2

# ---------------------------------------------------------------------
# 7) Build and run the pricer
# ---------------------------------------------------------------------
pricer = DiscreteBarrierFDMPricer2(
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
    flat_rate_nacc=r_flat,
    dividends=dividends,
    num_space_nodes=num_space_nodes,
    num_time_steps=num_time_steps,
    rannacher_steps=rannacher_steps,
    day_count="ACT/365",
    smooth_payoff_around_strike=True,
    payoff_smoothing_half_width_nodes=2,
)

# Pretty print all details (including BGK decision), plus price & Greeks
pricer.print_details()

# If you also want the raw dict:
px = pricer.price()
gr = pricer.greeks()
print("\nPrice:", px)
print("Greeks:", gr)
