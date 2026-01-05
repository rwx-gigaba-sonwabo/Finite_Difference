from datetime import date

import QuantLib as ql  # type: ignore

from discount.discount import YieldCurve
from cashflow.cashflow_date_engine import generate_cashflow_dates


class FixedRateBond:
    """
    Fixed-rate bond instrument using QuantLib.
    All vanilla bonds here pay semi-annual coupons by default.
    """

    def __init__(
        self,
        notional: float,
        issue_date: date,
        value_date: date,
        last_coupon_date: date,
        next_coupon_date: date,
        maturity_date: date,
        coupon_rate: float,
        frequency: str = "semi-annual",
        calendar: ql.Calendar = ql.SouthAfrica(),
        day_counter: ql.DayCounter = ql.SimpleDayCounter(),
    ):
        # Map frequency of payment periods
        freq_map = {
            "annual":   (ql.Annual,   ql.Period(ql.Annual)),
            "semi-annual": (ql.Semiannual, ql.Period(ql.Semiannual)),
            "quarterly": (ql.Quarterly, ql.Period(ql.Quarterly)),
            "monthly":  (ql.Monthly,  ql.Period(ql.Monthly)),
        }
        freq_key = frequency.lower()
        if freq_key not in freq_map:
            raise ValueError(f"Unsupported frequency '{frequency}'")
        self.ql_frequency, self.ql_tenor = freq_map[freq_key]

        # Build payment schedule
        self.issue_date = ql.Date(
            issue_date.day,
            issue_date.month,
            issue_date.year
        )
        self.value_date = ql.Date(
            value_date.day,
            value_date.month,
            value_date.year
        )
        self.ncd = next_coupon_date
        self.lcd = last_coupon_date

        self.maturity_date = ql.Date(
            maturity_date.day,
            maturity_date.month,
            maturity_date.year
        )
        self.calendar = calendar

        schedule_py_dates = generate_cashflow_dates(last_coupon_date,
                                                    maturity_date,
                                                    self.ql_tenor,
                                                    ql.SouthAfrica(),
                                                    ql.Unadjusted,
                                                    False)

        schedule_dates_ql = [
            ql.Date(d.day, d.month, d.year)
            for d in schedule_py_dates
        ]
        schedule_vector = ql.DateVector(schedule_dates_ql)
        self.schedule = ql.Schedule(
            schedule_vector,
            ql.NullCalendar(),
            ql.Unadjusted
        )
        # store plain-vanilla inputs
        self.notional = notional
        self.coupon_rate = coupon_rate
        self.day_counter = day_counter
        self.ex_coupon_period = ql.Period(10, ql.Days)

    def to_quantlib_bond(self, yield_curve: YieldCurve) -> ql.FixedRateBond:
        """
        Create a QuantLib FixedRateBond, attach a DiscountingBondEngine from
        existing YieldCurve.discount_curve handle, and return it.
        """
        bond = ql.FixedRateBond(
            0,                        # settlementDays
            100,                      # faceAmount
            self.schedule,            # schedule
            [self.coupon_rate],       # coupons
            self.day_counter,         # dayCounter (accrual)
            ql.Following,             # paymentConvention
            100.0,                    # redemption
            self.issue_date,          # issueDate
            self.calendar,            # paymentCalendar
            self.ex_coupon_period,    # exCouponPeriod
            self.calendar,            # exCouponCalendar
            ql.Preceding,            # exCouponConvention
            False                      # exCouponEndOfMonth
        )
        engine = ql.DiscountingBondEngine(yield_curve.discount_curve)
        bond.setPricingEngine(engine)
        return bond
