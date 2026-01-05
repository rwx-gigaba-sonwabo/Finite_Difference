from datetime import date

from discount.discount import YieldCurve
from zero_coupon.instruments.fixed_rate_bond import FixedRateBond
from zero_coupon.pricers.fixed_rate_bond_pricer import (
    FixedRateBondPricer,
)
from curve_importer.curve_importer import CurveImporter


def main():
    issue_date = date(2013, 7, 31)
    value_date = date(2025, 7, 16)
    maturity_date = date(2030, 1, 31)

    spot_curve = CurveImporter()
    [py_dates, curve_zero_rates] = spot_curve.load_data(r"c:\Users\SonwaboGigaba\Downloads\ZAR_SWAP_16072025.csv")

    zero_rates = [0.072292602,
                  0.073500442,
                  0.075291119,
                  0.0765714,
                  0.077320025,
                  0.077991308,
                  0.07880426,
                  0.07986765,
                  0.08119893,
                  0.08286982,
                ]
    mats = [
        date(2025, 7, 31),
        date(2026, 2, 2),
        date(2026, 7, 31),
        date(2027, 2, 1),
        date(2027, 8, 2),
        date(2028, 1, 31),
        date(2028, 7, 31),
        date(2029, 1, 31),
        date(2029, 7, 31),
        date(2030, 1, 31),
    ]
    yc = YieldCurve(zero_rates, mats, value_date)

    frb = FixedRateBond(
        notional=1_000_000,
        issue_date=issue_date,
        value_date=value_date,
        last_coupon_date=date(2025, 1, 31),
        next_coupon_date=date(2025, 7, 31),
        maturity_date=maturity_date,
        coupon_rate=0.08,
        frequency="semi-annual",
    )
    pricer = FixedRateBondPricer(frb, yc)
    pricer.print_details()


if __name__ == "__main__":
    main()
