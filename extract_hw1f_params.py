def extract_hw1f_params(filepath, asset_names):
    """
    Extract HullWhite1Factor model parameters from MarketData.json.

    Args:
        filepath    : Path to MarketData.json
        asset_names : str or list of model names
                      e.g. 'HullWhite1FactorInterestRateModel.ZAR-SWAP'
    Returns:
        dict keyed by asset name containing:
        {
            'Alpha'           : float,   # mean reversion speed
            'Sigma'           : float,   # reversion volatility (scalar)
            'Vol_Curve'       : list of [tenor, value],  # if term structure
            'Rate_Drift_Model': str,
        }
    """
    import json
    import os

    if isinstance(asset_names, str):
        asset_names = [asset_names]

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        market_data = json.load(f)

    price_models = (
        market_data
        .get('MarketData', {})
        .get('Price Models', {})
    )

    results = {}

    def unpack_curve(raw):
        if raw is None:
            return []
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            if '.Curve' in raw:
                return raw['.Curve'].get('data', [])
            if 'data' in raw:
                return raw['data']
        return []

    for asset_name in asset_names:
        if asset_name not in price_models:
            available = [k for k in price_models if 'HullWhite' in k]
            print(f"WARNING: '{asset_name}' not found.")
            print(f"  Available HW models: {available}")
            continue

        model = price_models[asset_name]

        results[asset_name] = {
            'Alpha'           : model.get('Alpha') or model.get('Reversion_Speed'),
            'Sigma'           : model.get('Sigma') or model.get('Reversion_Volatility'),
            'Vol_Curve'       : unpack_curve(model.get('Vol')),
            'Rate_Drift_Model': model.get('Rate_Drift_Model'),
        }

        r = results[asset_name]
        print(f"\n✅ Extracted '{asset_name}':")
        print(f"   Alpha (mean reversion) : {r['Alpha']}")
        print(f"   Sigma (reversion vol)  : {r['Sigma']}")
        print(f"   Vol Curve tenors       : {len(r['Vol_Curve'])}")
        print(f"   Rate Drift Model       : {r['Rate_Drift_Model']}")

    return results


def compare_hw1f_params(calibrated_param, extracted_param,
                         asset_name, output_path=None):
    """
    Compare calibrated HW1F parameters against MarketData.json parameters.

    The Pre-Computed Statistics Averaging method produces:
      - sigma : (1/m) * sum of per-tenor reversion volatilities
      - alpha : (1/m) * sum of per-tenor mean reversion speeds

    Both are scalars — so unlike PCA there is no tenor-level
    comparison for alpha and sigma. However if a Vol term structure
    is stored, that is compared tenor by tenor.

    Args:
        calibrated_param : dict from your HW1F calibration function
                           expects keys: 'Alpha', 'Sigma', 'Vol_Curve'
        extracted_param  : full dict from extract_hw1f_params()
        asset_name       : str
        output_path      : optional CSV path
    """
    import pandas as pd
    import os

    # ── Unwrap asset key ─────────────────────────────────────────────────────
    if asset_name in extracted_param:
        ext = extracted_param[asset_name]
    else:
        ext = extracted_param

    if hasattr(calibrated_param, 'param'):
        calibrated_param = calibrated_param.param
    cal = calibrated_param

    # ── Helpers ──────────────────────────────────────────────────────────────
    def to_dict(pairs):
        if not pairs:
            return {}
        return {float(p[0]): float(p[1]) for p in pairs}

    def adiff(a, b):
        return abs(float(a) - float(b)) \
               if (a is not None and b is not None) else None

    def rdiff(a, b):
        if a is None or b is None:
            return None
        denom = abs(float(b)) if abs(float(b)) > 1e-12 else 1e-12
        return abs(float(a) - float(b)) / denom * 100.0

    rows = []

    # ════════════════════════════════════════════════════════════════════════
    # 1. ALPHA — scalar mean reversion speed
    #    Per the Pre-Computed Statistics Averaging method:
    #    alpha = (1/m) * sum(alpha_k) across tenors
    # ════════════════════════════════════════════════════════════════════════
    cal_alpha = cal.get('Alpha') or cal.get('Reversion_Speed')
    ext_alpha = ext.get('Alpha') or ext.get('Reversion_Speed')

    print(f"\nAlpha  cal={cal_alpha}  ext={ext_alpha}")

    rows.append({
        'Parameter'   : 'Alpha (Mean Reversion Speed)',
        'Tenor'       : 'scalar',
        'Calibrated'  : float(cal_alpha) if cal_alpha is not None else None,
        'Extracted'   : float(ext_alpha) if ext_alpha is not None else None,
        'Abs_Diff'    : adiff(cal_alpha, ext_alpha),
        'Rel_Diff_Pct': rdiff(cal_alpha, ext_alpha),
    })

    # ════════════════════════════════════════════════════════════════════════
    # 2. SIGMA — scalar reversion volatility
    #    Per the Pre-Computed Statistics Averaging method:
    #    sigma = (1/m) * sum(sigma_k) across tenors
    # ════════════════════════════════════════════════════════════════════════
    cal_sigma = cal.get('Sigma') or cal.get('Reversion_Volatility')
    ext_sigma = ext.get('Sigma') or ext.get('Reversion_Volatility')

    print(f"Sigma  cal={cal_sigma}  ext={ext_sigma}")

    rows.append({
        'Parameter'   : 'Sigma (Reversion Volatility)',
        'Tenor'       : 'scalar',
        'Calibrated'  : float(cal_sigma) if cal_sigma is not None else None,
        'Extracted'   : float(ext_sigma) if ext_sigma is not None else None,
        'Abs_Diff'    : adiff(cal_sigma, ext_sigma),
        'Rel_Diff_Pct': rdiff(cal_sigma, ext_sigma),
    })

    # ════════════════════════════════════════════════════════════════════════
    # 3. VOL CURVE — term structure (if stored)
    # ════════════════════════════════════════════════════════════════════════
    cal_vc = to_dict(cal.get('Vol_Curve', []))
    ext_vc = to_dict(ext.get('Vol_Curve', []))

    print(f"Vol Curve: cal={len(cal_vc)} tenors  ext={len(ext_vc)} tenors")

    for t in sorted(set(cal_vc) | set(ext_vc)):
        rows.append({
            'Parameter'   : 'Vol_Curve',
            'Tenor'       : round(t, 6),
            'Calibrated'  : cal_vc.get(t),
            'Extracted'   : ext_vc.get(t),
            'Abs_Diff'    : adiff(cal_vc.get(t), ext_vc.get(t)),
            'Rel_Diff_Pct': rdiff(cal_vc.get(t), ext_vc.get(t)),
        })

    # ════════════════════════════════════════════════════════════════════════
    # 4. BUILD DATAFRAME
    # ════════════════════════════════════════════════════════════════════════
    df = pd.DataFrame(rows, columns=[
        'Parameter', 'Tenor',
        'Calibrated', 'Extracted',
        'Abs_Diff', 'Rel_Diff_Pct'
    ])

    for col in ['Calibrated', 'Extracted', 'Abs_Diff']:
        df[col] = df[col].apply(
            lambda x: round(x, 8) if isinstance(x, float) else x)
    df['Rel_Diff_Pct'] = df['Rel_Diff_Pct'].apply(
        lambda x: round(x, 4) if isinstance(x, float) else x)

    print(f"\n{'='*80}")
    print(f"  HW1F Parameter Comparison — {asset_name}")
    print(f"{'='*80}")
    print(df.to_string(index=False))

    large = df[df['Rel_Diff_Pct'].apply(
        lambda x: isinstance(x, float) and x > 1.0)]
    if not large.empty:
        print(f"\n⚠️  {len(large)} differences > 1%:")
        print(large.to_string(index=False))
    else:
        print("\n✅ All within 1% tolerance.")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✅ Saved: {output_path}")

    return df


# ── Usage ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    ASSET = "HullWhite1FactorInterestRateModel.ZAR-SWAP"

    extracted = extract_hw1f_params(
        filepath    = r"C:\XVA_engine\MarketData.json",
        asset_names = ASSET
    )

    # calibrated_param from your HW1F calibration function
    # should contain 'Alpha', 'Sigma', optionally 'Vol_Curve'
    comparison_df = compare_hw1f_params(
        calibrated_param = calibrated_hw1f_param,
        extracted_param  = extracted,
        asset_name       = ASSET,
        output_path      = r"C:\XVA_engine\outputs\hw1f_comparison.csv"
    )