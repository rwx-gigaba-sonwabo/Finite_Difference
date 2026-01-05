import QuantLib as ql  # type: ignore
from datetime import date
import math
from typing import Union


class YieldCurve:
    def __init__(
            self,
            zero_rates,
            maturities,
            value_date,
            calendar=None,
            day_count=365
            ):
        """
        Parameters:
        - zero_rates: list of NACC zero rates (e.g., 0.03 for 3%)
        - maturity_dates: list of datetime objects
        - valuation_date: datetime object
        - calendar: calendar used to for modified following business day
          convention
        - day_count: QuantLib day count convention
        (default: Actual/365)
        """
        if len(zero_rates) != len(maturities):
            raise ValueError("Length of zero_rates and maturities must match.")

        if not all(isinstance(r, (float, int)) for r in zero_rates):
            raise TypeError(
                "zero_rates must be a list of floats "
                "(not QuoteHandle or SimpleQuote)."
            )

        if isinstance(day_count, (int, float)):
            if day_count == 365:
                self.day_count = ql.Actual365Fixed()
                self.py_day_count = day_count
            elif day_count == 360:
                self.day_count = ql.Actual360()
                self.py_day_count = day_count
            elif day_count == 365.25:
                self.day_count = ql.Actual36525()
                self.py_day_count = day_count
            else:
                raise ValueError(
                    "Unsupported day count convention. "
                    "Use 365, 360, or 365.25."
                )
        elif isinstance(day_count, ql.DayCounter):
            self.day_count = day_count
        else:
            raise TypeError(
                "day_count must be an int, float, or QuantLib DayCounter."
            )

        self.zero_rates = zero_rates
        self.maturities = [
            ql.Date(d.day, d.month, d.year) if isinstance(d, date)
            else ql.Date(d.dayOfMonth(), d.month(), d.year())
            for d in maturities
        ]
        self.calendar = calendar or ql.NullCalendar
        if isinstance(value_date, date):
            self.value_date = ql.Date(
                value_date.day,
                value_date.month,
                value_date.year
            )
        elif isinstance(value_date, ql.Date):
            self.value_date = value_date
        else:
            raise TypeError(
                ("value_date must be a datetime.datetime or "
                 "QuantLib.Date object")
            )
        self.py_value_date = value_date
        ql.Settings.instance().evaluationDate = self.value_date

        dates = [self.value_date] + self.maturities
        dfs = [1] + [
            math.exp(
                -r * self.day_count.yearFraction(self.value_date, ql_date)
            )
            for ql_date, r in zip(self.maturities, self.zero_rates)
        ]

        self.curve = ql.DiscountCurve(
                    dates,
                    dfs,
                    self.day_count)

        self.curve.enableExtrapolation()
        self.discount_curve = ql.YieldTermStructureHandle(self.curve)

    def get_discount_factor(self, date):
        """Return interpolated discount factor for the given date."""
        ql_date = ql.Date(date.day, date.month, date.year)

        if ql_date < self.value_date:
            return 1.0

        return self.discount_curve.discount(ql_date)

    def get_zero_rate(self, date):
        """Return interpolated zero rate for the given date."""
        ql_date = ql.Date(date.day, date.month, date.year)
        t = self.day_count.yearFraction(self.curve.referenceDate(), ql_date)

        return self.discount_curve.zeroRate(
                t,
                ql.Continuous,
                ql.Annual
                ).rate()

    def forward_rate(self, start_date: date, end_date: date) -> float:
        """
        Implied simple forward rate (annual) between start_date and end_date,
        using ACT/365 day count.
        f = (DF(start)/DF(end) - 1) * (365 / days)
        """
        df_start = self.get_discount_factor(start_date)
        df_end = self.get_discount_factor(end_date)
        days = (end_date - start_date).days
        if days <= 0:
            raise ValueError("end_date must be after start_date")
        return (df_start / df_end - 1) * (self.py_day_count / days)


def discount_factor(
    rate: float,
    start_date: Union[ql.Date, date],
    end_date: Union[ql.Date, date],
    method: str = "continuous",
    compounding_frequency = ql.Annual,
    day_count: Union[ql.DayCounter, float] = ql.Actual365Fixed()
) -> float:
    """
    Compute discount factor given a rate, dates, and compounding type using QuantLib.
    Parameters:
        rate (float): Interest rate (e.g., 0.05 for 5%)
        start_date (date or ql.Date): Start date
        end_date (date or ql.Date): End/maturity date
        method (str): One of ['continuous', 'simple', 'compounded', 'discount']
        compounding_frequency (QuantLib.Compounding): Frequency of compounding
        day_count (float or QuantLib.DayCounter): Day count convention
    Returns:
        float: Discount factor
    """
    # Convert Python dates to QuantLib Dates if needed
    if isinstance(start_date, date):
        start_date = ql.Date(start_date.day, start_date.month, start_date.year)
    if isinstance(end_date, date):
        end_date = ql.Date(end_date.day, end_date.month, end_date.year)

    # Set evaluation date
    ql.Settings.instance().evaluationDate = start_date

    # Choose day count convention
    if isinstance(day_count, (int, float)):
        if day_count == 365:
            dc = ql.Actual365Fixed()
        elif day_count == 360:
            dc = ql.Actual360()
        elif day_count == 365.25:
            dc = ql.Actual36525()
        else:
            raise ValueError("Unsupported day count. Use 360, 365, or 365.25.")
    elif isinstance(day_count, ql.DayCounter):
        dc = day_count
    else:
        raise TypeError("day_count must be a float or QuantLib DayCounter.")

    t = dc.yearFraction(start_date, end_date)

    if t <= 0:
        return 1.0

    method = method.lower()
    if method == "continuous":
        return ql.DiscountFactor(math.exp(-rate * t))
    elif method == "simple":
        return 1.0 / (1.0 + rate * t)
    elif method == "compounded":
        return 1.0 / (1.0 + rate / compounding_frequency) ** (compounding_frequency * t)
    elif method == "discount":
        return 1.0 - rate * t
    else:
        raise ValueError("Unsupported method. Choose 'continuous', 'simple', 'compounded', or 'discount'.")