def calibrate_hw1f_interest_rate(
    curve_panel      : pd.DataFrame,
    num_business_days: float = 252.0,
    smooth           : float = 0.0,
    frequency        : int   = 1,
    max_alpha        : float = 4.0,
    rate_drift_model : str   = "Drift_To_Forward",
    distribution_type: str   = "Lognormal",
) -> tuple:
    """
    Calibrate Hull-White 1-Factor Interest Rate Model parameters
    using the Pre-Computed Statistics Averaging method.

    This mirrors the RiskFlow HullWhite1FactorInterestRateModel
    calibration procedure, which:
      1. Estimates per-tenor OU parameters via calc_statistics
      2. Averages sigma_k and alpha_k across all m tenors to
         produce a single representative (sigma, alpha) pair
         for the entire curve — i.e.:
            sigma = (1/m) * sum(sigma_k)   k = 1,...,m
            alpha = (1/m) * sum(alpha_k)   k = 1,...,m

    Args:
        curve_panel       : DataFrame with datetime index and float
                            tenor columns, values are zero rates
        num_business_days : annualisation factor (default 252)
        smooth            : outlier removal threshold in std devs.
                            0 = no smoothing
        frequency         : differencing frequency (default 1)
        max_alpha         : upper clip for mean reversion speed
        rate_drift_model  : 'Drift_To_Forward' or 'Drift_To_Blend'
        distribution_type : 'Lognormal' or 'Normal'

    Returns:
        param            : OrderedDict of calibrated parameters
        correlation_coef : cross-tenor correlation matrix
        delta            : differenced series used in calibration
    """
    import numpy as np
    import pandas as pd
    from collections import OrderedDict

    # ════════════════════════════════════════════════════════════════════════
    # STEP 1: force_positive shift
    # Mirrors RiskFlow: shift panel if any rate <= 0 so that
    # log-transform is valid
    # ════════════════════════════════════════════════════════════════════════
    min_rate     = curve_panel.min().min()
    force_positive = 0.0 if min_rate > 0.0 else -5.0 * min_rate

    if force_positive > 0.0:
        print(f"  force_positive applied: {force_positive:.6f} "
              f"(min rate was {min_rate:.6f})")

    # ════════════════════════════════════════════════════════════════════════
    # STEP 2: extract tenor array
    # ════════════════════════════════════════════════════════════════════════
    tenor = np.array(
        [float(x.split(',')[1]) if ',' in str(x) else float(x)
         for x in curve_panel.columns],
        dtype=np.float64
    )

    # ════════════════════════════════════════════════════════════════════════
    # STEP 3: run calc_statistics on shifted panel
    # Produces per-tenor: Alpha, Sigma, Reversion Volatility,
    # Long Run Mean, Volatility, Drift
    # ════════════════════════════════════════════════════════════════════════
    stats, correlation, delta = calculate_statistics_curve(
        data_frame        = curve_panel + force_positive,
        method            = 'Log',
        num_business_days = num_business_days,
        frequency         = frequency,
        max_alpha         = max_alpha,
        smooth            = smooth,
    )

    # ════════════════════════════════════════════════════════════════════════
    # STEP 4: Pre-Computed Statistics Averaging
    # Per the methodology:
    #   sigma = (1/m) * sum(sigma_k)   across all m tenors
    #   alpha = (1/m) * sum(alpha_k)   across all m tenors
    # where sigma_k is the Reversion Volatility (not raw Sigma)
    # matching what RiskFlow feeds into the simulation engine
    # ════════════════════════════════════════════════════════════════════════
    mean_reversion_speed = float(
        stats['Mean Reversion Speed'].mean()
    )
    reversion_volatility = stats['Reversion Volatility'].interpolate()

    # scalar sigma — simple average across tenors
    sigma_scalar = float(reversion_volatility.mean())

    # Vol curve — tenor-level volatility term structure
    vol_curve = reversion_volatility

    # Long run mean / historical yield — tenor level
    reversion_level = (
        stats['Long Run Mean']
        .interpolate()
        .bfill()
        .ffill()
    )

    # cross-tenor correlation
    correlation_coef = correlation

    # ════════════════════════════════════════════════════════════════════════
    # STEP 5: pack into OrderedDict matching RiskFlow JSON structure
    # ════════════════════════════════════════════════════════════════════════
    param = OrderedDict({

        # ── Scalar parameters ────────────────────────────────────────────
        'Alpha'           : mean_reversion_speed,
        'Reversion_Speed' : mean_reversion_speed,  # alias for consistency

        'Sigma'           : sigma_scalar,
        'Reversion_Volatility': sigma_scalar,       # alias

        # ── Term structure of volatility (per tenor) ─────────────────────
        'Vol_Curve': list(zip(
            tenor.tolist(),
            vol_curve.values.tolist()
        )),

        # ── Historical yield / long run mean (per tenor) ─────────────────
        'Historical_Yield': list(zip(
            tenor.tolist(),
            reversion_level.values.tolist()
        )),

        # ── Model settings ────────────────────────────────────────────────
        'Rate_Drift_Model' : rate_drift_model,
        'Distribution_Type': distribution_type,

        # ── force_positive stored for simulation engine reference ─────────
        'Force_Positive'   : force_positive,
    })

    # ════════════════════════════════════════════════════════════════════════
    # STEP 6: print summary to terminal
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  HW1F Calibration Summary")
    print(f"{'='*60}")
    print(f"  Alpha (mean reversion speed) : {mean_reversion_speed:.6f}")
    print(f"  Sigma (avg reversion vol)    : {sigma_scalar:.6f}")
    print(f"  Rate Drift Model             : {rate_drift_model}")
    print(f"  Distribution Type            : {distribution_type}")
    print(f"  Force Positive Shift         : {force_positive:.6f}")
    print(f"\n  --- Vol Curve (Reversion Volatility) ---")
    for t, v in param['Vol_Curve']:
        print(f"    Tenor {t:.4f}y : {v:.6f}")
    print(f"\n  --- Historical Yield (Long Run Mean) ---")
    for t, v in param['Historical_Yield']:
        print(f"    Tenor {t:.4f}y : {v:.6f}")

    return param, correlation_coef, delta


# ── Usage ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    ASSET = "HullWhite1FactorInterestRateModel.ZAR-SWAP"

    # 1. Run calibration
    param, correlation_coef, delta = calibrate_hw1f_interest_rate(
        curve_panel       = your_curve_panel_df,
        num_business_days = 252.0,
        smooth            = 0.0,
        rate_drift_model  = "Drift_To_Forward",
    )

    # 2. Extract from MarketData.json
    extracted = extract_hw1f_params(
        filepath    = r"C:\XVA_engine\MarketData.json",
        asset_names = ASSET
    )

    # 3. Compare
    comparison_df = compare_hw1f_params(
        calibrated_param = param,
        extracted_param  = extracted,
        asset_name       = ASSET,
        output_path      = r"C:\XVA_engine\outputs\hw1f_comparison.csv"
    )