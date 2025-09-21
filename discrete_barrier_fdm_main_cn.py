import math
import datetime as dt

from discrete_barrier_fdm_pricer import DiscreteBarrierFDM

# ------------------------------ Dates ---------------------------------
valuation = dt.date(2025, 7, 28)
maturity  = dt.date(2025, 8, 28)
T_years   = (maturity - valuation).days / 365.0

# Flat r from a toy DF (here: 8% NACA ~ 7.7% NACC for 1m; we just hardcode for demo)
r_flat = 0.077  # replace with your curve bootstrapped NACC to the option maturity

# ----------------------------- Instrument -----------------------------
S0       = 229.74
K        = 220.00
sigma    = 0.2613190156888
q        = 0.0
opt_type = "call"                 # "call" or "put"

barrier_type  = "up-and-out"      # supports also *-in and double-*
lower_barrier = None
upper_barrier = 270.0
rebate_amount = 1.0
rebate_at_hit = True

# Monitoring dates (discrete, exact)
monitoring_dates = [
    dt.date(2025, 8, 4),
    dt.date(2025, 8, 7),
    dt.date(2025, 8, 12),
    dt.date(2025, 8, 18),
    dt.date(2025, 8, 22),
    dt.date(2025, 8, 27),
    dt.date(2025, 8, 28),   # maturity
]

# Status flags at t=0
already_hit = False   # for KO
already_in  = False   # for KI

# ------------------------------- Numerics -----------------------------
N = 600
M = 1200
rannacher = 2

pricer = DiscreteBarrierFDM(
    spot=S0, strike=K,
    valuation_date=valuation, maturity_date=maturity,
    volatility=sigma, flat_rate_nacc=r_flat, dividend_yield=q,
    option_type=opt_type,
    barrier_type=barrier_type, lower_barrier=lower_barrier, upper_barrier=upper_barrier,
    monitoring_dates=monitoring_dates,
    rebate_amount=rebate_amount, rebate_at_hit=rebate_at_hit,
    already_hit=already_hit, already_in=already_in,
    dividend_list=[],
    num_space_nodes=N, num_time_steps=M, rannacher_steps=rannacher,
    day_count="ACT/365",
    min_substeps_between_monitors=2,
    grid_type="uniform",   # switch to "sinh" if you want clustering near barrier
    sinh_alpha=1.5,
    use_one_sided_greeks_near_barrier=True, barrier_safety_cells=2,
)

# Print details & price
pricer.print_details()
print("\\nPrice:", pricer.price())
print("Greeks:", pricer.greeks())

# Convergence check (example)
_ = pricer.validate_convergence(
    N_list=[300, 450, 600],
    M_list=[600, 900, 1200],
)