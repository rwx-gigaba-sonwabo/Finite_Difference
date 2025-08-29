import QuantLib as ql
from vanilla_option_pricer_test import VanillaOptionPricerTest


def main_test():
    cal = ql.SouthAfrica()
    dc = ql.Actual365Fixed()

    valuation = ql.Date(28, 7, 2025)
    expiry = ql.Date(28, 10, 2025)

    S0 = 176.39
    K = 170.00
    r = 0.07104445517       # Domestic risk-free rate
    q = 0.162544154746      # Foreign risk-free rate (dividend yield)
    sigma = 0.296198831050

    opt_type = "put"  # keep your original casing here
    ex_type = "American"
    settle_type = "cash"
    trade_number = 201870944

    side = "buy"
    contracts = 1
    contract_multiplier = 1

    opt_spot_days = 0
    opt_settle_days = 0
    und_spot_days = 3

    pricer = VanillaOptionPricerTest(
        spot_price=S0,
        strike_price=K,
        risk_free_rate=r,
        volatility=sigma,
        dividend_yield=q,
        valuation_date=valuation,
        maturity_date=expiry,
        contracts=contracts,
        contract_multiplier=contract_multiplier,
        side=side,
        option_type=opt_type,
        exercise_type=ex_type,
        option_spot_days=opt_spot_days,
        option_settlement_days=opt_settle_days,
        underlying_spot_days=und_spot_days,
        settlement_type=settle_type,
        calendar=cal,
        day_counter=dc,
        trade_number=trade_number
    )

    steps = [40, 60, 100, 150, 200, 250, 300,
             350, 400, 450, 500, 1000]
    print("=== Price Convergence (Richardson) ===")
    for n, p in pricer.batch_price(steps).items():
        print(f"{n:4d} → {p:.8f}")

    print("\n=== Greeks (manual vega bump) ===")
    greeks = pricer.calculate_greeks(steps)
    for k, v in greeks.items():
        print(f"{k:<14}: {v:.8f}")

    #  option calculations
    pricer.plot_price_convergence(steps)
    pricer.export_report_fx(steps)


if __name__ == "__main__":
    main_test()
