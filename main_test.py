import QuantLib as ql
from vanilla_option_pricer_test import VanillaOptionPricerTest
import pandas as pd

def main_test():
    cal = ql.SouthAfrica()
    dc = ql.Actual365Fixed()

    valuation = ql.Date(28, 7, 2025)
    expiry = ql.Date(28, 7, 2026)

    discount_curve = pd.read_csv("C:\Finite Difference (Equity Americans)\Finite_Difference-main\ZAR-SWAP-28072025.csv")
    discount_curve['Date'] = pd.to_datetime(discount_curve['Date'], format='%Y/%m/%d').dt.strftime('%Y-%m-%d')

    forward_curve = pd.read_csv("C:\Finite Difference (Equity Americans)\Finite_Difference-main\ZAR-SWAP-28072025.csv")
    forward_curve['Date'] = pd.to_datetime(forward_curve['Date'], format='%Y/%m/%d').dt.strftime('%Y-%m-%d')

    div_schedule = [
        (ql.Date(12, ql.September, 2025), 7.63),
        (ql.Date(10, ql.April, 2026), 8.0115),
    ]

    S0 = 229.74
    K = 250.00
    sigma = 0.243373477588

    opt_type = "Call"
    ex_type = "American"
    settle_type = "cash"
    trade_number = 201871033

    side = "buy"
    contracts = 1
    contract_multiplier = 1

    opt_spot_days = 0
    opt_settle_days = 0
    und_spot_days = 3

    pricer = VanillaOptionPricerTest(
        spot_price=S0,
        strike_price=K,
        discount_curve=discount_curve,
        forward_curve=forward_curve,
        volatility=sigma,
        dividend_schedule=div_schedule,
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
    print(pricer.pv_dividends)


if __name__ == "__main__":
    main_test()
