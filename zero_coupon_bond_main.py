from datetime import date

from discount.discount import YieldCurve
from zero_coupon.instruments.zero_coupon_bond import (
    ZeroCouponBond,
)
from zero_coupon.pricers.zero_coupon_bond_pricer import (
    ZeroCouponBondPricer,
)


def main():
    # 1-year zero @ 5%
    val_date = date(2025, 6, 19)
    z_rates = [0.05]
    mats = [date(2026, 6, 19)]
    yc = YieldCurve(z_rates, mats, val_date)

    bond = ZeroCouponBond(face_value=1_000, maturity_date=mats[0])
    pricer = ZeroCouponBondPricer(bond, yc)
    pricer.print_details()


if __name__ == "__main__":
    main()
