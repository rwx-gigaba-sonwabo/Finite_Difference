"""
Utility functions for rate conversions and curve construction.

These helpers are shared between the finite-difference engines and
scenario runners. They are written to be simple, transparent, and
easily auditable in a model-validation context.
"""

from __future__ import annotations

import datetime as dt

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


def nacc_to_naca(nacc_rate: float) -> float:
    """
    Convert a continuously compounded rate (NACC) to an annually
    compounded rate (NACA).

    Formula
    -------
    NACA = exp(NACC) - 1

    Parameters
    ----------
    nacc_rate :
        Continuously compounded rate (NACC).

    Returns
    -------
    float
        Annually compounded rate (NACA).

    Examples
    --------
    >>> nacc_to_naca(0.05)  # 5% NACC
    0.05127109637602412  # ~5.13% NACA
    """
    return float(np.exp(nacc_rate) - 1.0)


def naca_to_nacc(naca_rate: float) -> float:
    """
    Convert an annually compounded rate (NACA) to a continuously
    compounded rate (NACC).

    Formula
    -------
    NACC = ln(1 + NACA)

    Parameters
    ----------
    naca_rate :
        Annually compounded rate (NACA).

    Returns
    -------
    float
        Continuously compounded rate (NACC).

    Examples
    --------
    >>> naca_to_nacc(0.05)  # 5% NACA
    0.04879016416943204  # ~4.88% NACC
    """
    return float(np.log(1.0 + naca_rate))


def create_rate_df(rate: float) -> pd.DataFrame:
    """
    Create a flat NACA rate curve as a pandas DataFrame.

    The curve spans daily dates from 2025-07-28 to 2028-09-28 (inclusive),
    with a constant NACA rate in the "NACA" column and dates formatted
    as "YYYY/MM/DD". This format matches what the finite-difference
    engines expect before they convert to ISO strings.

    Parameters
    ----------
    rate :
        Constant NACA rate used for all rows.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - "Date": string, formatted "YYYY/MM/DD"
        - "NACA": float, flat rate value
    """
    start_date = dt.date(2025, 7, 28)
    end_date = dt.date(2028, 9, 28)

    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    df = pd.DataFrame(
        {
            "Date": date_range.strftime("%Y/%m/%d"),
            "NACA": rate,
        }
    )

    return df
