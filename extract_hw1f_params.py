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
    Compare HW1F calibrated vs extracted parameters.

    RiskFlow stores ONLY two comparable parameters for HW1F:
        Alpha : scalar mean reversion speed
        Sigma : single scalar reversion volatility
                (stored as .Curve with one point at tenor 0.0)

    Calibrated-only outputs (no extracted counterpart) are reported
    in a separate derived table and exported to a second CSV sheet.
    """
    import pandas as pd
    import os

    # ── Unwrap ───────────────────────────────────────────────────────────────
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
        if isinstance(pairs, dict) and '.Curve' in pairs:
            pairs = pairs['.Curve'].get('data', [])
        if isinstance(pairs, list):
            return {float(p[0]): float(p[1]) for p in pairs}
        return {}

    def adiff(a, b):
        return abs(float(a) - float(b)) \
               if (a is not None and b is not None) else None

    def rdiff(a, b):
        if a is None or b is None:
            return None
        denom = abs(float(b)) if abs(float(b)) > 1e-12 else 1e-12
        return abs(float(a) - float(b)) / denom * 100.0

    # ════════════════════════════════════════════════════════════════════════
    # TABLE 1: DIRECTLY COMPARABLE PARAMETERS
    # Only Alpha and Sigma scalar — what RiskFlow actually stores
    # ════════════════════════════════════════════════════════════════════════
    comparable_rows = []

    # ── Alpha (scalar) ───────────────────────────────────────────────────────
    cal_alpha = cal.get('Alpha')
    ext_alpha = ext.get('Alpha')
    comparable_rows.append({
        'Parameter'   : 'Alpha (Mean Reversion Speed)',
        'Tenor'       : 'scalar',
        'Calibrated'  : float(cal_alpha) if cal_alpha is not None else None,
        'Extracted'   : float(ext_alpha) if ext_alpha is not None else None,
        'Abs_Diff'    : adiff(cal_alpha, ext_alpha),
        'Rel_Diff_Pct': rdiff(cal_alpha, ext_alpha),
    })

    # ── Sigma scalar
    # RiskFlow stores Sigma as .Curve with ONE point at tenor=0.0
    # The single value is the scalar reversion volatility for the curve
    # Calibrated scalar = average of per-tenor sigma values
    # ────────────────────────────────────────────────────────────────────────
    ext_sig_dict = to_dict(ext.get('Sigma', []))

    # Extract the single scalar from extracted
    # (tenor 0.0 or first available point)
    if ext_sig_dict:
        ext_sigma_scalar = list(ext_sig_dict.values())[0]
    else:
        ext_sigma_scalar = None

    # Calibrated scalar sigma = average of per-tenor curve
    cal_sig_dict = to_dict(cal.get('Sigma', {}))
    if cal_sig_dict:
        import numpy as np
        cal_sigma_scalar = float(np.mean(list(cal_sig_dict.values())))
    else:
        cal_sigma_scalar = cal.get('Sigma_Scalar')

    comparable_rows.append({
        'Parameter'   : 'Sigma (Reversion Volatility)',
        'Tenor'       : 'scalar',
        'Calibrated'  : round(cal_sigma_scalar, 8)
                        if cal_sigma_scalar is not None else None,
        'Extracted'   : round(ext_sigma_scalar, 8)
                        if ext_sigma_scalar is not None else None,
        'Abs_Diff'    : adiff(cal_sigma_scalar, ext_sigma_scalar),
        'Rel_Diff_Pct': rdiff(cal_sigma_scalar, ext_sigma_scalar),
    })

    # ── Lambda scalar (always 0 — confirm alignment) ─────────────────────────
    cal_lam = cal.get('Lambda', 0.0)
    ext_lam = ext.get('Lambda', 0.0)
    comparable_rows.append({
        'Parameter'   : 'Lambda',
        'Tenor'       : 'scalar',
        'Calibrated'  : float(cal_lam),
        'Extracted'   : float(ext_lam),
        'Abs_Diff'    : adiff(cal_lam, ext_lam),
        'Rel_Diff_Pct': rdiff(cal_lam, ext_lam),
    })

    comparable_df = pd.DataFrame(comparable_rows, columns=[
        'Parameter', 'Tenor',
        'Calibrated', 'Extracted',
        'Abs_Diff', 'Rel_Diff_Pct'
    ])
    for col in ['Calibrated', 'Extracted', 'Abs_Diff']:
        comparable_df[col] = comparable_df[col].apply(
            lambda x: round(x, 8) if isinstance(x, float) else x)
    comparable_df['Rel_Diff_Pct'] = comparable_df['Rel_Diff_Pct'].apply(
        lambda x: round(x, 4) if isinstance(x, float) else x)

    # ════════════════════════════════════════════════════════════════════════
    # TABLE 2: CALIBRATED-ONLY DERIVED OUTPUTS
    # Sigma term structure and Historical Yield are produced by calibration
    # but not stored in RiskFlow — reported for transparency only
    # ════════════════════════════════════════════════════════════════════════
    derived_rows = []

    # Sigma per tenor
    for t, v in sorted(cal_sig_dict.items()):
        derived_rows.append({
            'Parameter' : 'Sigma_PerTenor (Calibrated Only)',
            'Tenor'     : round(t, 6),
            'Value'     : round(v, 8),
            'Note'      : 'Not stored in RiskFlow MarketData.json'
        })

    # Historical Yield per tenor
    cal_hy = to_dict(cal.get('Historical_Yield', []))
    for t, v in sorted(cal_hy.items()):
        derived_rows.append({
            'Parameter' : 'Historical_Yield_PerTenor (Calibrated Only)',
            'Tenor'     : round(t, 6),
            'Value'     : round(v, 8),
            'Note'      : 'Not stored in RiskFlow MarketData.json'
        })

    derived_df = pd.DataFrame(derived_rows, columns=[
        'Parameter', 'Tenor', 'Value', 'Note'
    ])

    # ════════════════════════════════════════════════════════════════════════
    # PRINT
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  HW1F Comparison — {asset_name}")
    print(f"{'='*80}")
    print("\n── Directly Comparable Parameters (RiskFlow vs Calibrated) ──")
    print(comparable_df.to_string(index=False))

    large = comparable_df[comparable_df['Rel_Diff_Pct'].apply(
        lambda x: isinstance(x, float) and x > 1.0)]
    if not large.empty:
        print(f"\n⚠️  {len(large)} differences > 1%:")
        print(large.to_string(index=False))
    else:
        print("\n✅ All comparable parameters within 1% tolerance.")

    print("\n── Calibrated-Only Derived Outputs (no RiskFlow counterpart) ──")
    print(derived_df.to_string(index=False))

    # ════════════════════════════════════════════════════════════════════════
    # EXPORT — two sheets in one Excel file, or two CSVs
    # ════════════════════════════════════════════════════════════════════════
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Try Excel first (two sheets), fall back to two CSVs
        try:
            with pd.ExcelWriter(
                output_path.replace('.csv', '.xlsx'),
                engine='openpyxl'
            ) as writer:
                comparable_df.to_excel(
                    writer,
                    sheet_name='Comparable_Parameters',
                    index=False
                )
                derived_df.to_excel(
                    writer,
                    sheet_name='Calibrated_Only_Derived',
                    index=False
                )
            print(f"\n✅ Saved Excel: "
                  f"{output_path.replace('.csv', '.xlsx')}")

        except ImportError:
            # openpyxl not available — write two CSVs
            comparable_df.to_csv(output_path, index=False)
            derived_path = output_path.replace('.csv', '_derived.csv')
            derived_df.to_csv(derived_path, index=False)
            print(f"\n✅ Saved: {output_path}")
            print(f"✅ Saved: {derived_path}")

    return comparable_df, derived_df

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
