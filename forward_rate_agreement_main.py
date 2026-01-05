from datetime import date

from discount.discount import YieldCurve
from zero_coupon.instruments.forward_rate_agreement import ForwardRateAgreement
from zero_coupon.pricers.forward_rate_agreement_pricer import (
    ForwardRateAgreementPricer,
)


def main():
    # 1) build zero curve
    today = date(2025, 6, 19)
    settlement = date(2025, 9, 19)
    zero_rates = [0.015, 0.0175, 0.02, 0.0225]
    maturities = [
        date(2025, 9, 19),
        date(2025, 12, 19),
        date(2026, 3, 19),
        date(2026, 6, 19),
    ]
    yc = YieldCurve(zero_rates, maturities, today)

    # 2) define a 3×6 FRA
    fra = ForwardRateAgreement(
        settle_date=settlement,
        maturity_date=date(2025, 12, 19),
        position="long",
        strike_rate=0.018,
        notional=1_000_000.0,
        frequency="quarterly",
    )
    fra_pricer = ForwardRateAgreementPricer(fra, yc)
    fra_pricer.print_details()


if __name__ == "__main__":
    main()
