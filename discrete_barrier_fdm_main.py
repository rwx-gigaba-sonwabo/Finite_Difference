valuation = dt.date(2025, 7, 28)
maturity  = dt.date(2025, 8, 28)

pricer = DiscreteBarrierFDMPricer(
    spot=229.74,
    strike=220.0,
    valuation_date=valuation,
    maturity_date=maturity,
    sigma=0.261319015688,             # annual vol
    option_type="call",
    barrier_type="up-and-out",
    upper_barrier=270.00,              # note: your system quotes in cents; feed S and H consistently
    monitor_dates=[                   # your discrete dates
        dt.date(2025,8,4), dt.date(2025,8,7), dt.date(2025,8,12),
        dt.date(2025,8,18), dt.date(2025,8,22), dt.date(2025,8,27),
    ],
    # flat NACA  -> NACC
    flat_r_nacc=math.log(1.0 + 0.075081),  # example: 7.5081% NACA → NACC
    day_count="ACT/365",
    grid_points=500,
    time_steps=500,
    rannacher_steps=2,
)
print("Price:", pricer.price())
print("Greeks:", pricer.greeks())
