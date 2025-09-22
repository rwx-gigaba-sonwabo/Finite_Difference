import datetime as dt
from discrete_barrier_fdm_pricer import DiscreteBarrierFDM

valuation = dt.date(2025,7,28)
maturity  = dt.date(2025,8,28)

# Market
r = 0.070538822
q = 0.0
sigma = 0.261319016

# Trade
S0 = 229.74
K  = 220.0
barrier_type  = "up-and-out"
lower_barrier = None
upper_barrier = 270.0
rebate_amount = 0.0
rebate_at_hit = True

monitoring_dates = [
    dt.date(2025,8,4), dt.date(2025,8,7), dt.date(2025,8,12),
    dt.date(2025,8,18), dt.date(2025,8,22), dt.date(2025,8,27), dt.date(2025,8,28)
]

pricer = DiscreteBarrierFDM(
    spot=S0, strike=K,
    valuation_date=valuation, maturity_date=maturity,
    volatility=sigma, flat_rate_nacc=r, dividend_yield=q,
    option_type="call",
    barrier_type=barrier_type, lower_barrier=lower_barrier, upper_barrier=upper_barrier,
    monitoring_dates=monitoring_dates, rebate_amount=rebate_amount, rebate_at_hit=rebate_at_hit,
    already_hit=False, already_in=False,
    num_time_steps=900,       # you select M; N is derived to meet Δt–Δx relation
    rannacher_steps=2,
    lambda_target=0.45,       # tune if needed (0.3–0.6 typical)
    day_count="ACT/365",
)

pricer.print_details()
print("\\nPrice:", pricer.price())
print("Greeks:", pricer.greeks())