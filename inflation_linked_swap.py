import QuantLib as ql  # type: ignore
from datetime import date


class InflationLinkedSwap:
    """
    Inflation-linked swap:
      - one leg pays a fixed rate on an inflation-indexed notional
      - the other leg pays floating (e.g. 3m Jibar) on the same notional
    """
    def __init__(
        self,
        issue_date: date,
        maturity_date: date,
        notional: float,
        fixed_rate: float,
        ibor_index: ql.IborIndex,
        zero_index: ql.ZeroInflationIndex,
        observation_lag: ql.Period,
        nominal_curve_handle: ql.YieldTermStructureHandle,
        pay_fixed_leg: bool = True,
        float_freq=ql.Semiannual,
        float_dc: ql.DayCounter = ql.Actual365Fixed(),
    ):
        self.issue_date = issue_date
        self.maturity_date = maturity_date
        self.notional = notional
        self.fixed_rate = fixed_rate
        self.ibor_index = ibor_index
        self.zero_index = zero_index
        self.observation_lag = observation_lag
        self.nominal_handle = nominal_curve_handle
        self.pay_fixed_leg = pay_fixed_leg
        self.float_freq = float_freq
        self.float_dc = float_dc

        # Compute base CPI using 4-/3-month lag + linear interpolation
        self.ql_issue_date = ql.Date(
            self.issue_date.day,
            self.issue_date.month,
            self.issue_date.year
        )
        self.base_cpi = ql.CPI.laggedFixing(
            self.zero_index,
            self.ql_issue_date,
            self.observation_lag,
            ql.CPI.Linear
        )

    def to_quantlib(self) -> ql.Swap:
        # 1) Convert maturity date to QuantLib Date
        ql_maturity_date = ql.Date(
            self.maturity_date.day,
            self.maturity_date.month,
            self.maturity_date.year
        )
        value_date = self.nominal_handle.referenceDate()

        # 2) Build fixed and cpi schedules
        float_schedule = ql.Schedule(
            value_date,
            ql_maturity_date,
            ql.Period(self.float_freq),
            ql.SouthAfrica(),                 # calendar
            ql.ModifiedFollowing,             # convention
            ql.ModifiedFollowing,             # termination convention
            ql.DateGeneration.Backward,       # date generation
            False                             # end-of-month
        )

        Inflation_schedule = float_schedule

        # 3) Floating leg cashflows
        float_leg = ql.IborLeg(
            [self.notional] * (len(float_schedule) - 1),
            # notionals: one per accrual
            float_schedule,                               # schedule
            self.ibor_index,                              # ibor index
            self.float_dc,  # day count convention
            ql.ModifiedFollowing,                         # payment convention
            [self.ibor_index.fixingDays()],               # fixing days
            [1.0],                                        # gearings
            [0.0],                                        # spreads
            [],                                           # caps
            [],                                           # floors
            False,                                        # in‐arrears
            ql.Period(),                                  # ex‐coupon period
            ql.NullCalendar(),                            # ex‐coupon calendar
            ql.Unadjusted,                                # ex‐coupon conv
            False,  # ex‐coupon end‐of‐month
            ql.SouthAfrica(),                             # payment calendar
            0,                                            # payment lag
            False                                         # withIndexedCoupons?
        )

        # 4) Inflation leg cashflows
        raw_inflation_leg = ql.CPILeg(
            [self.notional] * (len(float_schedule)-1),  # notionals
            Inflation_schedule,                         # schedule
            self.zero_index,                            # index
            self.base_cpi,                              # baseCPI
            self.observation_lag,                       # observationLag
            self.float_dc,                              # paymentDayCounter
            ql.ModifiedFollowing,                       # paymentConvention
            [self.fixed_rate],                          # fixedRates
            [],                                         # caps
            [],                                         # floors
            ql.Period(),                                # exCouponPeriod
            ql.NullCalendar(),                          # exCouponCalendar
            ql.Unadjusted,                              # exCouponConvention
            False,                                      # exCouponEndOfMonth
            ql.SouthAfrica(),                           # paymentCalendar
            True,                                       # growthOnly
            ql.CPI.Linear                               # Interpolation
        )
        # 4b) strip off the redemption cash-flow at maturity
        # Quantlib's CPILeg always adds a redemption cash-flow at the end.
        # Inflation-linked swaps are not zero-coupon swaps, thus no Notional
        # cashflow is swapped at Maturity
        inflation_leg = [
            cash_flow for cash_flow in raw_inflation_leg
            if ql.as_cpi_coupon(cash_flow) is not None
        ]

        # 5) Assemble into Swap
        # leg-0 = inflation, leg-1 = floating; payer=+1 means pay leg-0
        payer_flags = [
            self.pay_fixed_leg,       # True → pay inflation leg
            not self.pay_fixed_leg    # False → receive floating leg
        ]
        swap = ql.Swap(
            [inflation_leg, float_leg],
            payer_flags
        )
        swap.setPricingEngine(
            ql.DiscountingSwapEngine(self.nominal_handle)
        )
        return swap
