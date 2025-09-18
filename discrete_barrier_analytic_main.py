
import pandas as pd
import datetime as dt

from discrete_barrier_analytic_pricer import DiscreteBarrierFDMPricerAnalytic

# --- Curves (daily NACA) ---
discount_curve = pd.read_csv(
    r"C:\Finite Difference (Equity Americans)\Finite_Difference-main\ZAR-SWAP-28072025.csv"
)
discount_curve["Date"] = pd.to_datetime(discount_curve["Date"], format="%Y/%m/%d").dt.strftime("%Y-%m-%d")

# Optional forward curve (not strictly needed for this build; kept for future extensions)
forward_curve = discount_curve.copy()

# --- Contract, market & monitoring schedule ---
valuation = dt.date(2025, 7, 28)
maturity  = dt.date(2025, 8, 28)

monitoring_dates = [
    dt.date(2025, 7, 28),
    dt.date(2025, 8, 4),
    dt.date(2025, 8, 7),
    dt.date(2025, 8, 12),
    dt.date(2025, 8, 18),
    dt.date(2025, 8, 22),
    dt.date(2025, 8, 27),
]

dividends = []  # e.g. [(dt.date(2025, 9, 12), 7.63), (dt.date(2026, 4, 10), 8.0115)]

pricer = DiscreteBarrierFDMPricerAnalytic(
    # trade inputs
    trade_id="201871085",
    direction="long",
    quantity=1,
    contract_multiplier=1.0,
    # product
    option_type="call",
    barrier_type="up-and-out",
    strike=220.0,
    lower_barrier=None,
    upper_barrier=270.0,
    rebate_amount=0.0,
    rebate_timing_in="expiry",
    rebate_timing_out="hit",
    barrier_status=None,
    # market
    spot=229.74,
    volatility=0.261319015688,
    valuation_date=valuation,
    maturity_date=maturity,
    monitoring_dates=monitoring_dates,
    # curves & dividends
    discount_curve=discount_curve,
    forward_curve=forward_curve,
    dividend_schedule=dividends,
    day_count="ACT/365",
    # numerics
    time_steps=600,
    space_nodes=600,
    rannacher_steps=2,
    snap_strike_and_barrier=True,
    # FIS n_lim decision
    n_desired_for_decision=400,
    n_min_steps_per_interval=1,
    n_lim_multiplier=5,
)

pricer.print_details()
