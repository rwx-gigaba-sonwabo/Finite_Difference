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

    Aligned to your calculate_statistics_curve() output which uses:
        'Alpha'  → mean reversion speed (per tenor)
        'Sigma'  → reversion volatility (per tenor)  [NOT raw diffusion sigma]
        'Long_run_mean' → long run mean (per tenor)

    And to the MarketData.json structure which stores:
        'Alpha'  → scalar float
        'Sigma'  → {'.Curve': {'meta': [], 'data': [[tenor, val], ...]}}
    """
    import numpy as np
    import pandas as pd
    from collections import OrderedDict

    # ════════════════════════════════════════════════════════════════════════
    # STEP 1: force_positive shift
    # ════════════════════════════════════════════════════════════════════════
    min_rate       = curve_panel.min().min()
    force_positive = 0.0 if min_rate > 0.0 else -5.0 * min_rate

    if force_positive > 0.0:
        print(f"  force_positive applied: {force_positive:.6f} "
              f"(min rate was {min_rate:.6f})")

    # ════════════════════════════════════════════════════════════════════════
    # STEP 2: tenor array
    # ════════════════════════════════════════════════════════════════════════
    tenor = np.array(
        [float(x.split(',')[1]) if ',' in str(x) else float(x)
         for x in curve_panel.columns],
        dtype=np.float64
    )

    # ════════════════════════════════════════════════════════════════════════
    # STEP 3: calculate_statistics_curve
    # Note: positional first argument, no 'data_frame=' keyword
    # ════════════════════════════════════════════════════════════════════════
    stats, correlation, delta = calculate_statistics_curve(
        curve_panel + force_positive,   # ← positional, no keyword
        method            = 'Log',
        num_business_days = num_business_days,
        frequency         = frequency,
        max_alpha         = max_alpha,
        smooth            = smooth,
    )

    # ════════════════════════════════════════════════════════════════════════
    # STEP 4: Pre-Computed Statistics Averaging
    # Using YOUR stat keys: 'Alpha' and 'Sigma'
    #
    # alpha = (1/m) * sum(Alpha_k)   across tenors
    # sigma = (1/m) * sum(Sigma_k)   across tenors
    #
    # where Sigma_k is the reversion volatility per tenor
    # (not the raw OU diffusion parameter)
    # ════════════════════════════════════════════════════════════════════════
    mean_reversion_speed = float(stats['Alpha'].mean())

    # Per-tenor sigma curve (reversion volatility)
    sigma_curve = stats['Sigma'].interpolate()

    # Scalar sigma — simple average across tenors
    sigma_scalar = float(sigma_curve.mean())

    # Long run mean / historical yield per tenor
    reversion_level = (
        stats['Long_run_mean']
        .interpolate()
        .bfill()
        .ffill()
    )

    # ════════════════════════════════════════════════════════════════════════
    # STEP 5: Pack into OrderedDict matching MarketData.json structure
    #
    # From Image 1, the JSON structure is:
    # {
    #   "Lambda": 0.0,
    #   "Alpha": 3.534...,          ← scalar float
    #   "Sigma": {
    #       ".Curve": {
    #           "meta": [],
    #           "data": [[0.0, 0.1056...]]
    #       }
    #   },
    #   "Quanto_FX_Correlation": 0.0,
    #   "Quanto_FX_Volatility": 0.0
    # }
    # ════════════════════════════════════════════════════════════════════════
    param = OrderedDict({

        # ── Scalar parameters ────────────────────────────────────────────
        'Lambda'          : 0.0,
        'Alpha'           : mean_reversion_speed,

        # ── Sigma stored as .Curve structure (mirrors JSON) ───────────────
        'Sigma': {
            '.Curve': {
                'meta': [],
                'data': list(zip(
                    tenor.tolist(),
                    sigma_curve.values.tolist()
                ))
            }
        },

        # ── Historical yield / long run mean (per tenor) ─────────────────
        'Historical_Yield': list(zip(
            tenor.tolist(),
            reversion_level.values.tolist()
        )),

        # ── Quanto fields (as in JSON) ────────────────────────────────────
        'Quanto_FX_Correlation': 0.0,
        'Quanto_FX_Volatility' : 0.0,

        # ── Model settings ────────────────────────────────────────────────
        'Rate_Drift_Model' : rate_drift_model,
        'Distribution_Type': distribution_type,
        'Force_Positive'   : force_positive,
    })

    # ════════════════════════════════════════════════════════════════════════
    # STEP 6: Terminal summary
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  HW1F Calibration Summary")
    print(f"{'='*60}")
    print(f"  Alpha (mean reversion speed) : {mean_reversion_speed:.6f}")
    print(f"  Sigma scalar (avg rev vol)   : {sigma_scalar:.6f}")
    print(f"  Rate Drift Model             : {rate_drift_model}")
    print(f"  Distribution Type            : {distribution_type}")
    print(f"  Force Positive Shift         : {force_positive:.6f}")
    print(f"\n  {'Tenor':>8}  {'Sigma (Rev Vol)':>16}  "
          f"{'Long Run Mean':>14}")
    print(f"  {'-'*8}  {'-'*16}  {'-'*14}")
    for t, s, l in zip(tenor,
                        sigma_curve.values,
                        reversion_level.values):
        print(f"  {t:8.4f}  {s:16.8f}  {l:14.8f}")

    return param, correlation, delta


def extract_hw1f_params(filepath, asset_names):
    """
    Extract HW1F parameters from MarketData.json.
    Handles the nested .Curve structure for Sigma as seen in Image 1.
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

    def unpack_curve(raw):
        """Handles .Curve nesting and plain lists."""
        if raw is None:
            return []
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            if '.Curve' in raw:
                return raw['.Curve'].get('data', [])
            if 'data' in raw:
                return raw['data']
        # scalar — wrap as single point at tenor 0
        try:
            return [[0.0, float(raw)]]
        except (TypeError, ValueError):
            return []

    results = {}

    for asset_name in asset_names:
        if asset_name not in price_models:
            available = [k for k in price_models if 'HullWhite' in k]
            print(f"WARNING: '{asset_name}' not found.")
            print(f"  Available HW models: {available}")
            continue

        model = price_models[asset_name]

        results[asset_name] = {
            'Lambda'               : model.get('Lambda', 0.0),
            'Alpha'                : model.get('Alpha'),
            'Sigma'                : unpack_curve(model.get('Sigma')),
            'Historical_Yield'     : unpack_curve(
                                        model.get('Historical_Yield')
                                     ),
            'Quanto_FX_Correlation': model.get('Quanto_FX_Correlation', 0.0),
            'Quanto_FX_Volatility' : model.get('Quanto_FX_Volatility', 0.0),
            'Rate_Drift_Model'     : model.get('Rate_Drift_Model'),
        }

        r = results[asset_name]
        print(f"\n✅ Extracted '{asset_name}':")
        print(f"   Lambda : {r['Lambda']}")
        print(f"   Alpha  : {r['Alpha']}")
        print(f"   Sigma  : {len(r['Sigma'])} tenor points  "
              f"first={r['Sigma'][0] if r['Sigma'] else 'EMPTY'}")

    return results


def compare_hw1f_params(calibrated_param, extracted_param,
                         asset_name, output_path=None):
    """
    Compare calibrated HW1F parameters against MarketData.json.
    Handles nested .Curve Sigma structure.
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
        # handles both list-of-pairs and .Curve nested dict
        if isinstance(pairs, dict) and '.Curve' in pairs:
            pairs = pairs['.Curve'].get('data', [])
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

    # ── 1. Lambda (scalar) ───────────────────────────────────────────────────
    cal_lam = cal.get('Lambda', 0.0)
    ext_lam = ext.get('Lambda', 0.0)
    rows.append({
        'Parameter'   : 'Lambda',
        'Tenor'       : 'scalar',
        'Calibrated'  : float(cal_lam),
        'Extracted'   : float(ext_lam),
        'Abs_Diff'    : adiff(cal_lam, ext_lam),
        'Rel_Diff_Pct': rdiff(cal_lam, ext_lam),
    })

    # ── 2. Alpha (scalar mean reversion speed) ───────────────────────────────
    cal_alpha = cal.get('Alpha')
    ext_alpha = ext.get('Alpha')
    rows.append({
        'Parameter'   : 'Alpha (Mean Reversion Speed)',
        'Tenor'       : 'scalar',
        'Calibrated'  : float(cal_alpha) if cal_alpha is not None else None,
        'Extracted'   : float(ext_alpha) if ext_alpha is not None else None,
        'Abs_Diff'    : adiff(cal_alpha, ext_alpha),
        'Rel_Diff_Pct': rdiff(cal_alpha, ext_alpha),
    })

    # ── 3. Sigma curve (tenor by tenor) ─────────────────────────────────────
    # Calibrated Sigma is stored as {'.Curve': {'data': [[t,v],...]}}
    cal_sig = to_dict(cal.get('Sigma', {}))
    ext_sig = to_dict(ext.get('Sigma', []))
    print(f"Sigma: cal={len(cal_sig)} tenors  ext={len(ext_sig)} tenors")

    for t in sorted(set(cal_sig) | set(ext_sig)):
        rows.append({
            'Parameter'   : 'Sigma (Reversion Volatility)',
            'Tenor'       : round(t, 6),
            'Calibrated'  : cal_sig.get(t),
            'Extracted'   : ext_sig.get(t),
            'Abs_Diff'    : adiff(cal_sig.get(t), ext_sig.get(t)),
            'Rel_Diff_Pct': rdiff(cal_sig.get(t), ext_sig.get(t)),
        })

    # ── 4. Historical Yield curve (tenor by tenor) ───────────────────────────
    cal_hy = to_dict(cal.get('Historical_Yield', []))
    ext_hy = to_dict(ext.get('Historical_Yield', []))
    print(f"Historical Yield: cal={len(cal_hy)} tenors  "
          f"ext={len(ext_hy)} tenors")

    for t in sorted(set(cal_hy) | set(ext_hy)):
        rows.append({
            'Parameter'   : 'Historical_Yield',
            'Tenor'       : round(t, 6),
            'Calibrated'  : cal_hy.get(t),
            'Extracted'   : ext_hy.get(t),
            'Abs_Diff'    : adiff(cal_hy.get(t), ext_hy.get(t)),
            'Rel_Diff_Pct': rdiff(cal_hy.get(t), ext_hy.get(t)),
        })

    # ── 5. Quanto fields (scalars) ───────────────────────────────────────────
    for field in ['Quanto_FX_Correlation', 'Quanto_FX_Volatility']:
        cal_v = cal.get(field, 0.0)
        ext_v = ext.get(field, 0.0)
        rows.append({
            'Parameter'   : field,
            'Tenor'       : 'scalar',
            'Calibrated'  : float(cal_v) if cal_v is not None else 0.0,
            'Extracted'   : float(ext_v) if ext_v is not None else 0.0,
            'Abs_Diff'    : adiff(cal_v, ext_v),
            'Rel_Diff_Pct': rdiff(cal_v, ext_v),
        })

    # ── 6. Build DataFrame ───────────────────────────────────────────────────
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
    print(f"  HW1F Comparison — {asset_name}")
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