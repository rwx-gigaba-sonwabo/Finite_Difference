import calendar
from datetime import date, datetime
from typing import Dict, Tuple, Union

import pandas as pd
from dateutil.relativedelta import relativedelta

from curves.discount_curve import DiscountYieldCurveHandler


class BondHistoricalCPI:
    """
    Holds a history of headline CPI figures (first-of-month) and computes the
    "published" CPI for any date via BESA's 4/3-month rule with linear interpolation.

    Extension logic:
    - We assume the CPI/inflation curve discount factor behaves like:
          DF(t) = I(0) / I(t)
      so that inflation index ratios are:
          I(t2) / I(t1) = DF(t1) / DF(t2)

      Therefore for projecting monthly fixings:
          CPI_next = CPI_prev * DF(prev) / DF(next)
    """

    def __init__(
        self,
        value_date: date,
        curve_anchor_date: date,
        monthly_cpi: pd.DataFrame,
        curve: DiscountYieldCurveHandler,
        extend_cpi: int = 96,
        date_col: str = "Date",
        value_col: str = "Value",
    ):
        self.value_date = value_date
        self.curve_anchor_date = curve_anchor_date
        self.curve = curve
        self.extend_cpi = int(extend_cpi)

        self._monthly_cpi= self._df_to_fixing_map(
            monthly_cpi=monthly_cpi,
            date_col=date_col,
            value_col=value_col,
        )

        # Pre-extend fixings so published_cpi can be called safely beyond last fixing.
        if self.extend_cpi > 0:
            self._monthly_cpi = dict(self.extend_historical_cpi(self.extend_cpi))


    @staticmethod
    def _first_of_month(d: date) -> date:
        return date(d.year, d.month, 1)

    @staticmethod
    def _shift_months(d: date, months: int) -> date:
        y, m = divmod(d.month - 1 + months, 12)
        return date(d.year + y, m + 1, 1)

    @staticmethod
    def _coerce_to_date(x) -> date:
        """
        Accepts date/datetime/strings/timestamps and returns datetime.date.
        """
        if isinstance(x, date) and not isinstance(x, datetime):
            return x
        if isinstance(x, datetime):
            return x.date()
        # pandas-friendly coercion
        return pd.to_datetime(x).date()

    @staticmethod
    def _resolve_column(df: pd.DataFrame, preferred: str) -> str:
        """
        Tries to find the intended column even if the user has 'Date'/'Value'
        or other capitalization.
        """
        if preferred in df.columns:
            return preferred

        lower_map = {c.lower(): c for c in df.columns}
        if preferred.lower() in lower_map:
            return lower_map[preferred.lower()]

        # common fallbacks
        fallbacks = {
            "date": ["date", "Date", "DATE", "fixing_date", "FixingDate"],
            "value": ["value", "Value", "VALUE", "cpi", "CPI", "fixing", "Fixing"],
        }
        for cand in fallbacks.get(preferred.lower(), []):
            if cand in df.columns:
                return cand

        raise KeyError(
            f"Could not find a '{preferred}' column in monthly_cpi. "
            f"Available columns: {list(df.columns)}"
        )

    def _df_to_fixing_map(
        self,
        monthly_cpi: pd.DataFrame,
        date_col: str,
        value_col: str,
    ) -> Dict[date, float]:
        if not isinstance(monthly_cpi, pd.DataFrame):
            raise TypeError("monthly_cpi must be a pandas DataFrame.")

        dcol = self._resolve_column(monthly_cpi, date_col)
        vcol = self._resolve_column(monthly_cpi, value_col)

        df = monthly_cpi[[dcol, vcol]].copy()
        df[dcol] = df[dcol].apply(self._coerce_to_date)
        df[vcol] = pd.to_numeric(df[vcol], errors="raise")

        # Normalise to first-of-month fixings
        df[dcol] = df[dcol].apply(self._first_of_month)

        # If duplicates exist (same month), keep the last row
        df = df.sort_values(dcol).drop_duplicates(subset=[dcol], keep="last")

        fixings = dict(zip(df[dcol].tolist(), df[vcol].tolist()))

        if not fixings:
            raise ValueError("monthly_cpi is empty after processing.")

        return fixings

    def _discount_factor_for_date(self, d: date) -> float:
        """
        Uses the curve.get_discount_factor_for_date function.
        If the curve expects a datetime, we provide one.
        """
        try:
            return float(self.curve.get_discount_factor_for_date(d))
        except TypeError:
            dt = datetime(d.year, d.month, d.day)
            return float(self.curve.get_discount_factor_for_date(dt))

    def _bracket(self, d: date) -> Tuple[date, date]:
        # BESA 4/3-month bracketing
        first = self._first_of_month(d)
        j = self._shift_months(first, -4)
        j1 = self._shift_months(j, 1)
        if d.day == 1:
            return j, j
        return j, j1

    def extend_historical_cpi(self, months: int) -> Dict[date, float]:
        """
        Forecast future *first-of-month* CPI fixings for `months` ahead using the curve's
        discount factor. Returns a dict[date, float] including both historical and projected.

        Projection rule (index ratio implied by DFs):
            CPI_next = CPI_prev * DF(prev) / DF(next)
        """
        months = int(months)
        if months <= 0:
            return dict(self._monthly_cpi)

        fixings = dict(self._monthly_cpi)

        # Start from last available first-of-month fixing
        prev_date = max(fixings.keys())
        prev_date = self._first_of_month(prev_date)
        prev_cpi = float(fixings[prev_date])

        reset_date = self._shift_months(self.curve_anchor_date, -1)

        fixing_df = 1
        fixing_reset_cpi = 1
        value_date = self.value_date

        for i in range(1, months+1):
            next_date = self._shift_months(prev_date, i)

            if next_date == reset_date:
                value_date = next_date
                next_carry_date = self._shift_months(self.value_date, i)
                fixing_df = self._discount_factor_for_date(value_date)
                next_df = self._discount_factor_for_date(next_carry_date)
                prev_cpi = fixing_reset_cpi
            elif fixing_df < 1:
                next_carry_date = self._shift_months(self.value_date, i)
                next_df = self._discount_factor_for_date(next_carry_date)
            else:
                next_carry_date = value_date + relativedelta(months=i)
                next_df = self._discount_factor_for_date(next_carry_date)
                
            next_cpi = prev_cpi * (fixing_df / next_df)
            
            fixings[next_date] = float(next_cpi)
            
            fixing_reset_cpi = next_cpi
            fixing_reset_date = next_date

        return fixings

    def published_cpi(self, d: date) -> float:
        j, j1 = self._bracket(d)
        
        latest = max(self._monthly_cpi.keys())
        if j > latest or j1 > latest:
            # extend up to cover the maximum required month
            target = max(j, j1)
            # months difference in whole months from latest to target
            months_to_add = (target.year - latest.year) * 12 + (target.month - latest.month)
            if months_to_add > 0:
                self._monthly_cpi = dict(self.extend_historical_cpi(months_to_add))

        cpi_j = self._monthly_cpi[j]
        cpi_j1 = self._monthly_cpi[j1]

        if j == j1:
            return cpi_j

        D = calendar.monthrange(d.year, d.month)[1]
        fraction = (d.day - 1) / D
        return cpi_j + fraction * (cpi_j1 - cpi_j)
