import calendar
from datetime import date
from typing import Mapping, Tuple


class CPIPublication:
    """
    Holds a history of headline CPI figures (first-of-month) and
    computes the “published” CPI for any date via BESA's 4/3-month rule
    with linear interpolation.
    """

    def __init__(self, monthly_cpi: Mapping[date, float]):
        self._monthly_cpi = monthly_cpi

    @staticmethod
    def _first_of_month(d: date) -> date:
        return date(d.year, d.month, 1)

    @staticmethod
    def _shift_months(d: date, months: int) -> date:
        y, m = divmod(d.month - 1 + months, 12)
        return date(d.year + y, m + 1, 1)

    def _bracket(self, d: date) -> Tuple[date, date]:
        first = self._first_of_month(d)
        j = self._shift_months(first, -4)
        j1 = self._shift_months(j, 1)
        if d.day == 1:
            return j, j
        return j, j1

    def published_cpi(self, d: date) -> float:
        j, j1 = self._bracket(d)
        cpi_j = self._monthly_cpi[j]
        cpi_j1 = self._monthly_cpi[j1]
        if j == j1:
            return cpi_j
        D = calendar.monthrange(d.year, d.month)[1]
        fraction = (d.day - 1) / D
        return cpi_j + fraction * (cpi_j1 - cpi_j)
