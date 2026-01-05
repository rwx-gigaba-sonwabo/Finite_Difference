import QuantLib as ql  # type: ignore
from datetime import date
from typing import Dict, Tuple, List


class CPITermStructure:
    """
    Builds a handle to a ZeroInflationTermStructure that
    (a) interpolates your historical first-of-month CPI fixings
    (b) if given inflation swap quotes + a nominal curve, bootstraps
        a market-implied forward-CPI curve via PiecewiseZeroInflation.
    """
    def __init__(
        self,
        historical_cpi: Dict[date, float],
        inflation_zero_rates: List[Tuple[ql.Period, float]],
        nominal_curve_handle: ql.YieldTermStructureHandle,
        observation_lag: ql.Period = ql.Period(4, ql.Months),
        calendar: ql.Calendar = ql.SouthAfrica(),
        frequency=ql.Monthly,
        day_counter: ql.DayCounter = ql.Actual365Fixed(),
        availability_lag: ql.Period = ql.Period(1, ql.Months),
        currency: ql.Currency = ql.ZARCurrency(),
    ):
        # sort & store
        self.historical_cpi = historical_cpi
        self._cpi_data = sorted(self.historical_cpi.items())
        self.observation_lag = observation_lag
        self.calendar = calendar
        self.frequency = frequency
        self.day_counter = day_counter
        self.availability_lag = availability_lag
        self.currency = currency
        self.inflation_zero_rates = inflation_zero_rates
        self.nominal_curve_handle = nominal_curve_handle

    def build_handle(self) -> ql.ZeroInflationTermStructureHandle:
        """
        Return a Handle[ZeroInflationTermStructure] that
        covers both historical CPI (lagged interpolation)
        and projects future CPI by capitalizing the latest
        published index with your real zero curve.
        """
        # 1) set value date
        ql_value_date = self.nominal_curve_handle.referenceDate()

        # 2) turn zero rates and dates into QL vectors
        ql_zero_dates = []
        ql_zero_rates = []
        for mat_date, rate in self.inflation_zero_rates:
            maturity_date = ql.Date(
                mat_date.day, mat_date.month, mat_date.year
            )
            ql_zero_dates.append(maturity_date)
            ql_zero_rates.append(rate)

        latest_cpi_date = max(self.historical_cpi.keys())
        ql_latest_cpi_date = ql.Date(
            latest_cpi_date.day,
            latest_cpi_date.month,
            latest_cpi_date.year
        )
        ref_date = self.calendar.advance(
            ql_latest_cpi_date,
            self.availability_lag
        )

        # 3) build raw index with past CPI fixings
        raw_index = ql.ZeroInflationIndex(
            "CPI-SA",                                   # family name
            ql.CustomRegion("South Africa", "ZA"),      # region/calendar
            False,                                      # revised?
            self.frequency,                             # published frequency
            self.availability_lag,                      # lag
            self.currency                               # currency
        )
        for cpi_date, cpi in self._cpi_data:
            ql_cpi_date = ql.Date(cpi_date.day, cpi_date.month, cpi_date.year)
            raw_index.addFixing(ql_cpi_date, cpi)

        # 4) build helpers from with simple inflation swap quotes
        helpers = []
        for mat_date, quote in self.inflation_zero_rates:
            maturity_date = ql.Date(
                mat_date.day, mat_date.month, mat_date.year
            )
            helpers.append(
                ql.ZeroCouponInflationSwapHelper(
                    ql.QuoteHandle(ql.SimpleQuote(quote/100)),
                    self.observation_lag,
                    maturity_date,
                    self.calendar,
                    ql.ModifiedFollowing,
                    self.day_counter,
                    raw_index,
                    ql.CPI.Linear,
                    self.nominal_curve_handle
                )
            )

        # 5) Bootstrap the PieceWiseZeroInflation curve
        pw_inflation_curve = ql.PiecewiseZeroInflation(
            ql_value_date,
            self.calendar,
            self.day_counter,
            self.availability_lag,
            self.frequency,
            0.03,  # initial zero-CPI guess or initial value
            helpers
        )
        pw_inflation_curve.enableExtrapolation()

        return ql.ZeroInflationTermStructureHandle(pw_inflation_curve)

    def build_index(self) -> ql.ZeroInflationIndex:
        """
        Wrap the TS handle in QuantLib's built-in ZACPI index, seeded with
        fixings. This index will:
          - apply the 4/3m lag + linear interpolation for past dates
          - use your forward curve for future dates
        """
        # Initiate term structure handle
        term_struct_handle = self.build_handle()

        # wrap in a full ZeroInflationIndex where we can force interpolation
        zar_region = ql.CustomRegion("South Africa", "ZA")
        raw_cpi_index = ql.ZeroInflationIndex(
            "CPI-SA",
            zar_region,
            False,
            self.frequency,
            self.availability_lag,
            ql.ZARCurrency(),
            term_struct_handle
        )

        zar_cpi_index = raw_cpi_index
        # seed all historical fixings
        # (so that first-4m fixings come from your table)
        for cpi_date, cpi in self._cpi_data:
            ql_cpi_date = ql.Date(cpi_date.day, cpi_date.month, cpi_date.year)
            zar_cpi_index.addFixing(ql_cpi_date, cpi)
        return zar_cpi_index
