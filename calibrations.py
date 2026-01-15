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
