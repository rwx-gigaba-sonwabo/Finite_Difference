from __future__ import annotations

import json
from collections import OrderedDict
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

# ---- Robust imports (works if scripts are in same folder) ----
import utils
from stochasticprocess import PCAInterestRateCalibration  # optional (class-based)
from data_extract import extract_ada_curve_panel


# ---------------------------
# 1) Core: compute statistics
# ---------------------------
def compute_curve_statistics(
    curve_panel: pd.DataFrame,
    num_business_days: float = 252.0,
    max_alpha: float = 4.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """
    Computes RiskFlow-style OU/log statistics and correlation:
        stats, correlation, delta = utils.calc_statistics(df + force_positive, method='Log', ...)

    Returns:
        stats, correlation, delta, force_positive
    """
    df = curve_panel.copy()
    df = df.sort_index()
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if df.shape[1] < 2:
        raise ValueError("Need at least 2 tenors (columns) for PCAInterestRate calibration.")
    if df.shape[0] < 5:
        raise ValueError("Need more history (rows) to estimate OU/PCA stats reliably.")

    # Match RiskFlow positivity enforcement
    min_rate = float(df.min().min())
    force_positive = 0.0 if min_rate > 0.0 else -5.0 * min_rate

    stats, correlation, delta = utils.calc_statistics(
        df + force_positive,
        method="Log",
        num_business_days=num_business_days,
        max_alpha=max_alpha,
    )

    return stats, correlation, delta, force_positive


# -------------------------------------------------
# 2) Manual calibration (matches RiskFlow calibrate)
# -------------------------------------------------
def calibrate_pca_interest_rate_manual(
    curve_panel: pd.DataFrame,
    *,
    rate_drift_model: str = "Drift_To_Forward",
    matrix_type: str = "Covariance",
    distribution_type: str = "LogNormal",
    num_business_days: float = 252.0,
    num_factors: int = 3,
    max_alpha: float = 4.0,
) -> utils.CalibrationInfo:
    """
    Manual version of PCAInterestRateCalibration.calibrate, but works with
    curve_panel columns as float tenors (no comma-name requirement).

    Output matches the parameter structure used by PCAInterestRateModel:
      - Reversion_Speed
      - Historical_Yield (utils.Curve)
      - Yield_Volatility (utils.Curve)
      - Eigenvectors: [{'Eigenvector': utils.Curve, 'Eigenvalue': ...}, ...]
      - Rate_Drift_Model, Princ_Comp_Source, Distribution_Type
      - correlation_coef = aki.T
      - delta = transformed diff panel
    """
    stats, correlation, delta, force_positive = compute_curve_statistics(
        curve_panel,
        num_business_days=num_business_days,
        max_alpha=max_alpha,
    )

    tenors = np.array(curve_panel.columns, dtype=np.float64)

    # RiskFlow interpolation choices
    standard_deviation = stats["Reversion Volatility"].interpolate()
    covariance = (
        np.dot(
            standard_deviation.values.reshape(-1, 1),
            standard_deviation.values.reshape(1, -1),
        )
        * correlation.values
    )

    aki, evecs, evals = utils.PCA(covariance, num_factors)
    mean_reversion_speed = float(stats["Mean Reversion Speed"].mean())

    vol_curve = standard_deviation
    reversion_level = stats["Long Run Mean"].interpolate().bfill().ffill()
    correlation_coef = aki.T

    param = OrderedDict(
        {
            "Reversion_Speed": mean_reversion_speed,
            "Historical_Yield": utils.Curve([], list(zip(tenors, reversion_level.values))),
            "Yield_Volatility": utils.Curve([], list(zip(tenors, vol_curve.values))),
            "Eigenvectors": [
                OrderedDict(
                    {
                        "Eigenvector": utils.Curve([], list(zip(tenors, evec))),
                        "Eigenvalue": float(ev),
                    }
                )
                for evec, ev in zip(evecs.real.T, evals.real)
            ],
            "Rate_Drift_Model": rate_drift_model,
            "Princ_Comp_Source": matrix_type,
            "Distribution_Type": distribution_type,
        }
    )

    return utils.CalibrationInfo(param, correlation_coef, delta)


# -----------------------------------------------------
# 3) Optional: class calibration (requires comma columns)
# -----------------------------------------------------
def calibrate_pca_interest_rate_via_class(
    curve_panel: pd.DataFrame,
    curve_prefix: str,
    *,
    rate_drift_model: str = "Drift_To_Forward",
    matrix_type: str = "Covariance",
    distribution_type: str = "LogNormal",
    num_business_days: float = 252.0,
) -> utils.CalibrationInfo:
    """
    Uses your PCAInterestRateCalibration class directly.
    It expects column names where tenor is token [1] after splitting by comma.
    We therefore rename float tenor columns -> f"{prefix},{tenor}".

    This should match RiskFlow output *exactly* (up to eigenvector sign flips).
    """
    df = curve_panel.copy()
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    # Rename columns to the format expected by PCAInterestRateCalibration
    rename_map = {t: f"{curve_prefix},{float(t)}" for t in df.columns}
    df = df.rename(columns=rename_map)

    calib = PCAInterestRateCalibration(
        model="PCAInterestRateModel",
        param={
            "Rate_Drift_Model": rate_drift_model,
            "Matrix_Type": matrix_type,
            "Distribution_Type": distribution_type,
        },
    )

    return calib.calibrate(df, vol_shift=0.0, num_business_days=num_business_days)


# ---------------------------
# 4) Helpers: pretty output
# ---------------------------
def summarize_calibration(calib_info: utils.CalibrationInfo) -> None:
    p = calib_info.param
    print("\n=== PCAInterestRate Calibration Summary ===")
    print(f"Reversion_Speed: {float(p['Reversion_Speed']):.10f}")

    hy = np.asarray(p["Historical_Yield"].array)
    vol = np.asarray(p["Yield_Volatility"].array)

    print(f"Tenors: {hy[:, 0].tolist()}")
    print(f"Historical_Yield: {hy[:, 1].tolist()}")
    print(f"Yield_Volatility: {vol[:, 1].tolist()}")

    evs = p["Eigenvectors"]
    print(f"Num PCA factors: {len(evs)}")
    for i, ev in enumerate(evs, start=1):
        print(f"  PC{i}: Eigenvalue={float(ev['Eigenvalue']):.10f}")


def calibration_to_jsonable(calib_info: utils.CalibrationInfo) -> Dict[str, Any]:
    def curve_to_dict(c: utils.Curve) -> Dict[str, Any]:
        arr = np.asarray(c.array)
        return {"meta": list(c.meta), "array": arr.tolist()}

    p = calib_info.param
    out_param: Dict[str, Any] = dict(p)
    out_param["Historical_Yield"] = curve_to_dict(p["Historical_Yield"])
    out_param["Yield_Volatility"] = curve_to_dict(p["Yield_Volatility"])
    out_param["Eigenvectors"] = [
        {
            "Eigenvalue": float(x["Eigenvalue"]),
            "Eigenvector": curve_to_dict(x["Eigenvector"]),
        }
        for x in p["Eigenvectors"]
    ]
    out = {
        "param": out_param,
        "correlation": np.asarray(calib_info.correlation).tolist(),
        "delta": calib_info.delta.to_dict() if isinstance(calib_info.delta, pd.DataFrame) else calib_info.delta,
    }
    return out


# ---------------------------
# 5) Example main
# ---------------------------
if __name__ == "__main__":
    ADA_FILE = "path/to/your/archive.ada"
    CURVE_PREFIX = "InflationRate.ZA.CPI"  # prefix BEFORE the first comma in ADA headers

    # This returns a date x tenor panel with float tenor columns (per your fixed extractor)
    curve_panel = extract_ada_curve_panel(
        file_path=ADA_FILE,
        curve_prefix=CURVE_PREFIX,
        start_date="2020-01-01",
        end_date="2024-12-31",
        match_mode="equals",
        tenor_token_index=1,
    )

    # ---- Option A: manual calibration (robust to float tenor columns) ----
    calib_info = calibrate_pca_interest_rate_manual(
        curve_panel,
        rate_drift_model="Drift_To_Forward",
        matrix_type="Covariance",
        distribution_type="LogNormal",
        num_business_days=252.0,
        num_factors=3,
        max_alpha=4.0,
    )
    summarize_calibration(calib_info)

    # Save parameters if you want to diff vs RiskFlow output
    with open("inflation_pca_params_manual.json", "w", encoding="utf-8") as f:
        json.dump(calibration_to_jsonable(calib_info), f, indent=2)

    # ---- Option B: class calibration (closest to RiskFlow; needs comma columns) ----
    calib_info_class = calibrate_pca_interest_rate_via_class(
        curve_panel,
        curve_prefix=CURVE_PREFIX,
        rate_drift_model="Drift_To_Forward",
        matrix_type="Covariance",
        distribution_type="LogNormal",
        num_business_days=252.0,
    )
    summarize_calibration(calib_info_class)

    with open("inflation_pca_params_class.json", "w", encoding="utf-8") as f:
        json.dump(calibration_to_jsonable(calib_info_class), f, indent=2)

    # ---- Also show the raw stats if you want to inspect them ----
    stats, corr, delta, shift = compute_curve_statistics(curve_panel, num_business_days=252.0, max_alpha=4.0)
    print("\nForce-positive shift applied:", shift)
    print("\nStats head:\n", stats.head())
    print("\nCorrelation shape:", corr.shape)
    print("\nDelta head:\n", delta.head())

from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd


def calculate_statistics_CS(
    data: pd.DataFrame,
    method: str = "log",
    num_business_days: float = 252.0,
    smooth: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    RiskFlow-style OU/log statistics for a *curve panel* (multiple tenors).

    Parameters
    ----------
    data : pd.DataFrame
        Either:
          - columns: ['date', <tenor1>, <tenor2>, ...]
        or:
          - index is datetime, columns are tenors
    method : {"log", "diff"}
        "log": work on log(level) and treat delta as log-returns-like changes
        "diff": work on level differences
    num_business_days : float
        Annualization factor (default 252)
    smooth : float
        If > 0: remove outliers beyond smooth*std from median per tenor,
        then interpolate.

    Returns
    -------
    stats : pd.DataFrame
        Indexed by tenor, with columns:
        mean_reversion_speed, drift, mu, reversion_volatility,
        long_run_mean, volatility
    correlation : pd.DataFrame
        Tenor-by-tenor correlation matrix of delta series (used in PCA step)
    delta : pd.DataFrame
        Delta series used for covariance/correlation (your `x`)
    """
    method = method.lower().strip()
    if method not in {"log", "diff"}:
        raise ValueError("method must be 'log' or 'diff'")

    df = data.copy()

    # Accept either a 'date' column or a datetime index
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

    df = df.sort_index()
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    # All remaining columns are tenors (RiskFlow intuition: node-by-node stats)
    tenor_cols = list(df.columns)
    if len(tenor_cols) < 1:
        raise ValueError("No tenor columns found (expected >0 after cleaning).")

    # Optional outlier removal per tenor
    if smooth > 0.0:
        med = df.median(axis=0)
        sd = df.std(axis=0, ddof=0)
        mask = (df.sub(med, axis=1).abs()).le(smooth * sd, axis=1)
        df = df.where(mask)

        # interpolate through time; if index isn't datetime-like, fallback to index interpolation
        try:
            df = df.interpolate(method="time")
        except (ValueError, TypeError):
            df = df.interpolate(method="index")

        df = df.ffill().bfill()

    # Build the working series s(t) and delta x(t) = s(t+1)-s(t) (shifted to align with y=s(t))
    if method == "diff":
        s = df.astype(float)
    else:
        # log requires positivity; caller usually enforces via force_positive upstream
        s = np.log(df.astype(float).clip(lower=1e-12))

    # x corresponds to "delta" used for covariance/correlation and OU regression-like stats
    delta = s.diff(1).shift(-1)          # x_t = s_{t+1} - s_t (aligned at time t)
    y = s                                # y_t = s_t

    # drop the last row (shift(-1) makes it NaN), keep panel shape
    delta = delta.iloc[:-1]
    y = y.iloc[:-1]

    # Centered moments per tenor
    x_mean = delta.mean(axis=0, skipna=True)
    y_mean = y.mean(axis=0, skipna=True)

    x_dev = delta.sub(x_mean, axis=1)
    y_dev = y.sub(y_mean, axis=1)

    cov_xy = (x_dev * y_dev).mean(axis=0, skipna=True)
    var_y = (y_dev ** 2).mean(axis=0, skipna=True)

    # Mean reversion speed (alpha) per tenor
    eps = 1e-16
    ratio = cov_xy / var_y.clip(lower=eps)

    # RiskFlow-style transform: alpha = -N * ln(1 + cov/var)
    alpha = (-num_business_days * np.log(1.0 + ratio.clip(lower=-0.999999999))).astype(float)
    alpha = alpha.clip(lower=0.001, upper=4.0)  # same stabilisation you were doing

    # Sigma^2 per tenor (OU-consistent)
    exp_a = np.exp(-alpha / num_business_days)
    var_x = (x_dev ** 2).mean(axis=0, skipna=True)

    denom = (1.0 - np.exp(-2.0 * alpha / num_business_days)).clip(lower=1e-16)
    sigma2 = (var_x - ((1.0 - exp_a) ** 2) * var_y) * (2.0 * alpha) / denom

    sigma2 = sigma2.clip(lower=0.0)
    sigma = np.sqrt(sigma2)

    # Long-run level (theta) per tenor
    one_minus_exp = (1.0 - exp_a).clip(lower=1e-16)
    theta = y_mean + x_mean / one_minus_exp

    if method == "log":
        # RiskFlow-style adjustment for log-level mean
        log_theta = np.exp(theta + sigma2 / (4.0 * alpha).clip(lower=1e-16))
        long_run_mean = log_theta
    else:
        long_run_mean = theta

    # Vol + drift per tenor (annualised)
    volatility = delta.std(axis=0, ddof=0) * np.sqrt(num_business_days)
    drift = x_mean * num_business_days
    mu = drift + 0.5 * (volatility ** 2)

    # Correlation across tenors (this is what your PCA wants)
    correlation = delta.corr()

    stats = pd.DataFrame(
        {
            "mean_reversion_speed": alpha,
            "drift": drift,
            "mu": mu,
            "reversion_volatility": sigma,
            "long_run_mean": long_run_mean,
            "volatility": volatility,
        }
    )
    stats.index.name = "tenor"

    return stats, correlation, delta
