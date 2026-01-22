from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Sequence, Union

try:
    import pandas as pd  # optional
except Exception:  # pragma: no cover
    pd = None


DateLikeOrTs = Union[date, datetime, "pd.Timestamp"]


def to_date(x: DateLikeOrTs) -> date:
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    if pd is not None and isinstance(x, pd.Timestamp):
        return x.date()
    raise TypeError(f"Unsupported date-like type: {type(x)}")


def day_offset(base_date: date, d: DateLikeOrTs) -> int:
    dd = to_date(d)
    return int((dd - base_date).days)


def add_days(base_date: date, days: float) -> date:
    return base_date + timedelta(days=int(round(days)))


def ensure_dates(seq: Sequence[DateLikeOrTs]) -> list[date]:
    return [to_date(x) for x in seq]
