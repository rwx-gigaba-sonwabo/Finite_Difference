from datetime import date

from discount.discount import YieldCurve
from zero_coupon.markets.cpi_publication import CPIPublication

import QuantLib as ql  # type: ignore


class InflationLinkedBond:
    """
    Thin wrapper around QuantLib.CPIBond that:
      - takes nominal real zero rate discount-curve handle directly
      - delegates all CPI indexing and cash-flow logic to QuantLib
      - stores CPI and discount-curve handles for post-processing
    """
    def __init__(
        self,
        notional: float,
        issue_date: date,
        maturity_date: date,
        coupon_rate: float,
        real_yield_curve: YieldCurve,
        cpi_history: CPIPublication,
        frequency: str = "semi-annual",
        calendar: ql.Calendar = ql.SouthAfrica(),
        day_counter: ql.DayCounter = ql.Actual365Fixed(),
    ):
        # Map frequency of payment periods
        freq_map = {
            "annual":   (ql.Annual,   ql.Period(1, ql.Years)),
            "semi-annual": (ql.Semiannual, ql.Period(6, ql.Months)),
            "quarterly": (ql.Quarterly, ql.Period(3, ql.Months)),
            "monthly":  (ql.Monthly,  ql.Period(1, ql.Months)),
        }
        self.freq_key = frequency.lower()
        if self.freq_key not in freq_map:
            raise ValueError(f"Unsupported frequency '{frequency}'")
        self.ql_frequency, self.ql_tenor = freq_map[self.freq_key]

        # Build payment schedule
        self.issue_date = ql.Date(
            issue_date.day,
            issue_date.month,
            issue_date.year
        )
        self.maturity_date = ql.Date(
            maturity_date.day,
            maturity_date.month,
            maturity_date.year
        )
        self.calendar = calendar
        self.schedule = ql.Schedule(
            self.issue_date,
            self.maturity_date,
            self.ql_tenor,
            calendar,
            ql.ModifiedFollowing,
            ql.ModifiedFollowing,
            ql.DateGeneration.Backward,
            False,
        )

        self.notional = notional
        self.coupon_rate = coupon_rate
        self.day_counter = day_counter
        self.real_yield_curve = real_yield_curve
        self.real_discount_curve = self.real_yield_curve.discount_curve
        self.cpi_history = cpi_history
        self._lag = ql.Period(4, ql.Months)
        self.bond = self._to_quantlib_bond()

    def _ql_to_py(self, d: ql.Date) -> date:
        """Convert a QuantLib Date to Python datetime.date."""
        return date(d.year(), d.month(), d.dayOfMonth())

    def _to_quantlib_bond(self) -> ql.FixedRateBond:
        """
        Create a QuantLib FixedRateBond, attach a DiscountingBondEngine from
        existing YieldCurve.discount_curve handle, and return it.
        Though we used QuantLib FixedRateBond, it is actually a CPI-linked
        bond because real zero rates in the YieldCurve class
        """
        bond = ql.FixedRateBond(
            0,                        # settlementDays
            100.0,                    # faceAmount
            self.schedule,            # schedule
            [self.coupon_rate],       # coupons
            self.day_counter,         # dayCounter (accrual)
            ql.ModifiedFollowing,     # paymentConvention
            100.0,                    # redemption
            self.issue_date,          # issueDate
            self.calendar,            # paymentCalendar
            ql.Period(10, ql.Days),   # exCouponPeriod
            self.calendar,            # exCouponCalendar
            ql.ModifiedFollowing,     # exCouponConvention
            True                      # exCouponEndOfMonth
        )

        engine = ql.DiscountingBondEngine(self.real_discount_curve)
        bond.setPricingEngine(engine)
        return bond

    def index_ratio(self) -> float:
        """
        BESA Index Ratio = CPI(settle-lag)/CPI(issue), floored at 1.0.
        """
        settle = self.bond.settlementDate()
        settle_py = self._ql_to_py(settle)
        cpi_settle = self.cpi_history.published_cpi(settle_py)
        issue_py_date = self._ql_to_py(self.issue_date)
        cpi_issue = self.cpi_history.published_cpi(issue_py_date)
        return max(cpi_settle / cpi_issue, 1.0)
