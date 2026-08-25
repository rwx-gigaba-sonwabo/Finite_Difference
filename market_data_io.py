"""
base_valuation/market_data_io.py
===================================
CSV-driven market data loaders for base_valuation runner / validation
scripts.

Expected CSV format -- two columns, header row required::

    date,value
    2025-04-02,78.00
    2025-07-02,79.00
    ...

``date`` must be ISO format (YYYY-MM-DD), each on or after the valuation
date, strictly increasing. Column names are configurable (``date_col`` /
``value_col``) so files exported with different headers (e.g. "Date,Rate",
matching the convention already used elsewhere in this project) don't need
renaming.

``value`` means different things depending on which loader is used:

- :func:`build_forward_curve_from_csv` -- a traded forward/futures price
  for that maturity. Interpolated across maturities with **linear
  interpolation** (the market-standard convention for a futures strip) via
  :func:`base_valuation.commodity_curve.build_commodity_forward_curve`.
- :func:`build_yield_curve_from_csv` -- a zero rate for that maturity, in
  whatever compounding convention ``rate_convention`` declares. **NACA**
  (Nominal Annual Compounded Annually -- the South African market standard
  for quoting yields) is the default; rates are converted to
  continuously-compounded (NACC) before curve construction, since
  ``YieldCurve.discount_factor(t) = exp(-r(t)*t)`` throughout this codebase
  expects NACC. A configurable interpolation scheme (Hermite-RT, stitched
  linear/Hermite-RT, etc.) is then applied across maturities.
"""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import numpy as np

from base_valuation.commodity_curve import (
    CommodityForwardCurve,
    build_commodity_forward_curve,
)
from base_valuation.yield_curve import build_yield_curve
from market_data.yield_curve import YieldCurve
from market_data.vol_surface import SimpleSkew

__all__ = [
    "load_date_value_csv",
    "effective_annual_to_continuous",
    "build_forward_curve_from_csv",
    "build_yield_curve_from_csv",
    "build_vol_skew_from_csv",
]


def load_date_value_csv(
    filepath: str | Path,
    date_col: str = "date",
    value_col: str = "value",
) -> tuple[list[date], np.ndarray]:
    """Load a two-column (date, value) CSV, sorted by date ascending.

    Parameters
    ----------
    filepath : str or Path
    date_col, value_col : str
        Header names of the date and value columns (case-sensitive, but
        surrounding whitespace is stripped). Matching is case-insensitive
        as a convenience (e.g. a file with "Date,Rate" headers works with
        the defaults `date_col="date"`, `value_col="value"` only if you
        pass the actual header names -- pass `date_col="Date",
        value_col="Rate"` for that file).

    Returns
    -------
    dates : list[date]
    values : np.ndarray, shape (n,)

    Raises
    ------
    ValueError
        Missing header/columns, no data rows, unparsable date/value,
        or duplicate dates.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Market data CSV not found: {path}")

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty file or missing header row")
        fieldnames = [h.strip() for h in reader.fieldnames]
        if date_col not in fieldnames:
            raise ValueError(
                f"{path}: missing required column {date_col!r} (found {fieldnames})"
            )
        if value_col not in fieldnames:
            raise ValueError(
                f"{path}: missing required column {value_col!r} (found {fieldnames})"
            )
        reader.fieldnames = fieldnames  # use stripped names as row dict keys too
        rows = list(reader)

    if not rows:
        raise ValueError(f"{path}: no data rows")

    try:
        parsed = [
            (date.fromisoformat(row[date_col].strip()), float(row[value_col].strip()))
            for row in rows
        ]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{path}: could not parse a row ({exc})") from exc

    parsed.sort(key=lambda x: x[0])
    dates = [d for d, _ in parsed]

    if len(set(dates)) != len(dates):
        raise ValueError(f"{path}: duplicate dates found in {date_col!r} column")

    values = np.array([v for _, v in parsed], dtype=np.float64)
    return dates, values


def effective_annual_to_continuous(
    rate: float | np.ndarray, compounding_freq: int = 1,
) -> float | np.ndarray:
    """Convert a periodically-compounded nominal rate to a continuously
    compounded (NACC) rate:

        r_nacc = compounding_freq * ln(1 + rate / compounding_freq)

    ``compounding_freq=1`` is **NACA** (Nominal Annual Compounded Annually
    -- the South African market standard for quoting yields, and the
    default rate convention for :func:`build_yield_curve_from_csv`). 2, 4,
    and 12 give the semi-annual/quarterly/monthly equivalents (NACS/NACQ/
    NACM) should a curve ever be quoted that way instead.

    This is time-independent (the same conversion applies at every
    maturity): ``(1 + r/freq)^(freq*t) == exp(r_nacc * t)`` for all ``t``.
    """
    freq = float(compounding_freq)
    return freq * np.log(1.0 + np.asarray(rate, dtype=np.float64) / freq)


def build_forward_curve_from_csv(
    filepath: str | Path,
    val_date: date,
    date_col: str = "date",
    price_col: str = "value",
    day_count: str = "ACT/365",
    interpolation: str = "linear",
) -> CommodityForwardCurve:
    """Build a :class:`~base_valuation.commodity_curve.CommodityForwardCurve`
    from a (date, price) CSV of traded maturities.

    ``interpolation`` : {"linear", "forward_price"}
        ``"linear"`` (default) -- linear between quoted maturities, **flat**
        extrapolation beyond the shortest/longest traded contract (the
        literal "linear interpolation across the available maturities
        traded" behaviour).
        ``"forward_price"`` -- linear between quoted maturities, **linear**
        right-extrapolation beyond the last contract (projects the final
        segment's slope forward instead of flattening) -- switch to this
        if the system you're validating against extrapolates the strip
        rather than flattening it.
    """
    dates, prices = load_date_value_csv(filepath, date_col, price_col)
    return build_commodity_forward_curve(
        val_date=val_date, curve_dates=dates, prices=prices,
        day_count=day_count, interpolation=interpolation,
    )


def build_yield_curve_from_csv(
    filepath: str | Path,
    val_date: date,
    date_col: str = "date",
    rate_col: str = "value",
    rate_convention: str = "NACA",
    compounding_freq: int = 1,
    day_count: str = "ACT/365",
    interpolation: str = "hermite_rt",
) -> YieldCurve:
    """Build a :class:`~market_data.yield_curve.YieldCurve` from a
    (date, rate) CSV of zero-rate pillars.

    Parameters
    ----------
    rate_convention : {"NACA", "NACC"}
        ``"NACA"`` (default) -- rates in the CSV are Nominal Annual
        Compounded Annually; converted to continuously-compounded zero
        rates via :func:`effective_annual_to_continuous` before curve
        construction. ``compounding_freq`` generalises this to NACS/NACQ/
        NACM (2/4/12) if the source system ever quotes that way instead.
        ``"NACC"`` -- rates are already continuously compounded; used as-is.
    interpolation : str
        Any key in ``base_valuation.yield_curve.INTERPOLATORS``:
        ``"hermite_rt"`` (Hermite spline in r*t space -- smooth,
        arbitrage-consistent forward rates; a common default for a
        validation-grade curve), ``"stitched_linear_hermite_rt"`` (linear
        short end / Hermite long end), ``"linear_rt"``, ``"reciprocal_time"``,
        or plain ``"linear"``.
    """
    if rate_convention not in ("NACA", "NACC"):
        raise ValueError(f"rate_convention must be 'NACA' or 'NACC', got {rate_convention!r}")

    dates, rates = load_date_value_csv(filepath, date_col, rate_col)

    if rate_convention == "NACA":
        rates = effective_annual_to_continuous(rates, compounding_freq)

    return build_yield_curve(
        val_date=val_date, curve_dates=dates, rates=rates,
        day_count=day_count, interpolation=interpolation,
    )


def build_vol_skew_from_csv(
    filepath: str | Path,
    strike_col: str = "strike",
    vol_col: str = "vol",
) -> SimpleSkew:
    """Build a :class:`~market_data.vol_surface.SimpleSkew` from a
    (strike, vol) CSV of smile quotes.

    Expected CSV format -- two columns, header row required::

        strike,vol
        70.0,0.34
        80.0,0.30
        95.0,0.27

    Column names are configurable (``strike_col`` / ``vol_col``) so files
    exported with different headers don't need renaming. Rows are sorted
    by strike before construction; ``SimpleSkew`` itself then applies
    linear interpolation in strike space with flat extrapolation beyond
    the quoted range (see its docstring). This is deliberately
    single-tenor -- build one CSV (and one ``SimpleSkew``) per tenor
    bucket relevant to the trade.

    Parameters
    ----------
    filepath : str or Path
        Path to the (strike, vol) CSV.
    strike_col, vol_col : str
        Column headers to read the strike and vol from.

    Returns
    -------
    SimpleSkew

    Raises
    ------
    ValueError
        Missing header/columns, no data rows, unparsable strike/vol,
        or duplicate strikes.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Vol skew CSV not found: {path}")

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty file or missing header row")
        fieldnames = [h.strip() for h in reader.fieldnames]
        if strike_col not in fieldnames:
            raise ValueError(
                f"{path}: missing required column {strike_col!r} (found {fieldnames})"
            )
        if vol_col not in fieldnames:
            raise ValueError(
                f"{path}: missing required column {vol_col!r} (found {fieldnames})"
            )
        reader.fieldnames = fieldnames  # use stripped names as row dict keys too
        rows = list(reader)

    if not rows:
        raise ValueError(f"{path}: no data rows")

    try:
        parsed = [
            (float(row[strike_col].strip()), float(row[vol_col].strip()))
            for row in rows
        ]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{path}: could not parse a row ({exc})") from exc

    parsed.sort(key=lambda x: x[0])
    strikes = [k for k, _ in parsed]

    if len(set(strikes)) != len(strikes):
        raise ValueError(f"{path}: duplicate strikes found in {strike_col!r} column")

    vols = [v for _, v in parsed]
    return SimpleSkew(strikes=strikes, vols=vols)
