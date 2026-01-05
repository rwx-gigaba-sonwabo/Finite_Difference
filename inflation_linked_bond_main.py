from datetime import date

from discount.discount import YieldCurve
from zero_coupon.markets.cpi_publication import CPIPublication
from zero_coupon.instruments.inflation_linked_bond import (
    InflationLinkedBond,
)
from zero_coupon.pricers.inflation_linked_bond_pricer import (
    InflationLinkedBondPricer,
)


def main():
    today = date(2025, 6, 30)
    issue_date = date(2025, 6, 19)
    zero_rates = [0.02,
                  0.0225,
                  0.025,
                  0.0275,
                  0.03,
                  0.0325,
                  0.035,
                  0.0375,
                  0.04,
                  0.0425]
    maturities = [
        date(2025, 12, 19),
        date(2026, 6, 19),
        date(2026, 12, 19),
        date(2027, 6, 19),
        date(2027, 12, 19),
        date(2028, 6, 19),
        date(2028, 12, 19),
        date(2029, 6, 19),
        date(2029, 12, 19),
        date(2030, 6, 19),
    ]
    yc = YieldCurve(zero_rates, maturities, today)

    # 2) CPI history (first-of-month)
    cpi_hist = {
        date(2025, 1, 1): 140.5,
        date(2025, 2, 1): 142.3,
        date(2025, 3, 1): 143.1,
        date(2025, 4, 1): 144.0,
        date(2025, 5, 1): 145.2,
    }
    cpi_pub = CPIPublication(cpi_hist)

    # 3) create the ILB instrument
    infl_bond = InflationLinkedBond(
        notional=1000.0,
        coupon_rate=0.0625,
        issue_date=issue_date,
        maturity_date=date(2030, 6, 19),
        real_yield_curve=yc,
        cpi_history=cpi_pub,
        frequency="semi-annual",
    )

    # 4) price & print
    pricer = InflationLinkedBondPricer(infl_bond)
    pricer.print_details()


if __name__ == "__main__":
    main()
