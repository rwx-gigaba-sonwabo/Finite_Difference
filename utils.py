import pandas as pd
import datetime as dt
import numpy as np


def nacc_to_naca(nacc_rate: float) -> float:
    """
    Convert a continuously compounded rate (NACC) to an annually compounded rate (NACA).

    Formula: NACA = exp(NACC) - 1

    Parameters:
    -----------
    nacc_rate : float
        The continuously compounded rate (NACC)

    Returns:
    --------
    float
        The annually compounded rate (NACA)

    Example:
    --------
    >>> nacc_to_naca(0.05)  # 5% NACC
    0.05127109637602412  # ~5.13% NACA
    """
    return np.exp(nacc_rate) - 1


def naca_to_nacc(naca_rate: float) -> float:
    """
    Convert an annually compounded rate (NACA) to a continuously compounded rate (NACC).

    Formula: NACC = ln(1 + NACA)

    Parameters:
    -----------
    naca_rate : float
        The annually compounded rate (NACA)

    Returns:
    --------
    float
        The continuously compounded rate (NACC)

    Example:
    --------
    >>> naca_to_nacc(0.05)  # 5% NACA
    0.04879016416943204  # ~4.88% NACC
    """
    return np.log(1 + naca_rate)


def create_rate_df(rate: float) -> None:
    """
    Create a CSV file with daily dates from 2025/07/28 to 2028/09/28 and a constant rate.

    Parameters:
    -----------
    rate : float
        The constant rate value to use for all rows
    csv_path : str
        The file path where the CSV should be saved
    """
    start_date = dt.date(2025, 7, 28)
    end_date = dt.date(2028, 9, 28)

    # Generate daily dates
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Create DataFrame
    df = pd.DataFrame({
        'Date': date_range.strftime('%Y/%m/%d'),
        'NACA': rate
    })

    return df