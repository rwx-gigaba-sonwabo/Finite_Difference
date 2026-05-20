# ===========================================================================
# GBM FX IMPLIED CALIBRATION — EXTRACT, CALIBRATE, COMPARE
# Mirrors the structure of the PCA and HW1F comparison functions
# ===========================================================================

def extract_gbm_fx_params(filepath, fx_names=None):
    """
    Extract GBM FX Vol curve parameters from MarketData.json.
    These are the PRODUCTION parameters RiskFlow reads at runtime.

    The Vol curve stored is the integrated (average) vol curve —
    tagged 'Integrated' in RiskFlow — where each point (T, sigma_bar)
    satisfies V(T) = sigma_bar^2 * T.

    Args:
        filepath : path to MarketData.json
        fx_names : str, list, or None (extracts all FX GBM models)

    Returns:
        dict keyed by FX factor name:
        {
            'Vol_Curve'              : list of [expiry, atm_vol],
            'Integrated_Variance'    : list of [expiry, V(T)],
            'Quanto_FX_Volatility'   : list of [expiry, vol] or None,
            'Quanto_FX_Correlation'  : float or None,
        }
    """
    import json
    import os
    import numpy as np

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        market_data = json.load(f)

    price_factors = market_data.get('MarketData', {}) \
                               .get('Price Factors', {})

    # Also check top-level Price Factors (some exports differ)
    if not price_factors:
        price_factors = market_data.get('Price Factors', {})

    if isinstance(fx_names, str):
        fx_names = [fx_names]

    # ── Find all GBM FX model keys if none specified ─────────────────────────
    GBM_PREFIXES = (
        'GBMAssetPriceTSModelParameters.',
        'FxRateVol.',
        'GBMTSModelParameters.',
    )

    if fx_names is None:
        fx_names = [
            k for k in price_factors
            if any(k.startswith(p) for p in GBM_PREFIXES)
        ]
        if not fx_names:
            print("WARNING: No GBM FX model keys found in Price Factors.")
            print("Available keys (first 20):",
                  list(price_factors.keys())[:20])

    def unpack_curve(raw):
        """Handles list-of-pairs, .Curve.data nesting, or raw array."""
        if raw is None:
            return []
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            if '.Curve' in raw:
                return raw['.Curve'].get('data', [])
            if 'Curve' in raw:
                inner = raw['Curve']
                if isinstance(inner, list):
                    return inner
                if isinstance(inner, dict):
                    return inner.get('data', [])
            if 'data' in raw:
                return raw['data']
            if 'array' in raw:
                return raw['array']
        return []

    results = {}

    for name in fx_names:
        # Try the key directly, then with prefix variants
        factor_data = price_factors.get(name)
        if factor_data is None:
            for prefix in GBM_PREFIXES:
                alt = prefix + name if not name.startswith(prefix) else name
                factor_data = price_factors.get(alt)
                if factor_data is not None:
                    name = alt
                    break

        if factor_data is None:
            print(f"WARNING: '{name}' not found in Price Factors.")
            continue

        # ── Extract Vol curve ────────────────────────────────────────────────
        # RiskFlow stores the vol as tagged 'Integrated' — average vol
        # sigma_bar(T) such that V(T) = sigma_bar^2 * T
        raw_vol = (
            factor_data.get('Vol')
            or factor_data.get('Volatility')
            or factor_data.get('Curve')
        )
        vol_pairs = unpack_curve(raw_vol)

        # ── Recompute integrated variance from vol pairs ─────────────────────
        integrated_variance = []
        for pair in vol_pairs:
            T      = float(pair[0])
            vol    = float(pair[1])
            V      = vol ** 2 * T
            integrated_variance.append([T, V])

        # ── Quanto fields (FX cross-currency correction) ─────────────────────
        quanto_vol_raw  = factor_data.get('Quanto_FX_Volatility')
        quanto_corr_raw = factor_data.get('Quanto_FX_Correlation', 0.0)

        results[name] = {
            'Vol_Curve'            : vol_pairs,
            'Integrated_Variance'  : integrated_variance,
            'Quanto_FX_Volatility' : unpack_curve(quanto_vol_raw),
            'Quanto_FX_Correlation': float(quanto_corr_raw)
                                     if quanto_corr_raw is not None else 0.0,
        }

        # ── Verification print ───────────────────────────────────────────────
        r = results[name]
        print(f"\n✅ Extracted '{name}':")
        print(f"   Vol Curve           : {len(r['Vol_Curve'])} expiry points")
        if r['Vol_Curve']:
            print(f"   First point         : T={r['Vol_Curve'][0][0]:.4f}y  "
                  f"vol={r['Vol_Curve'][0][1]:.6f}")
            print(f"   Last  point         : T={r['Vol_Curve'][-1][0]:.4f}y  "
                  f"vol={r['Vol_Curve'][-1][1]:.6f}")
        print(f"   Quanto FX Corr      : {r['Quanto_FX_Correlation']:.4f}")

    return results


def calibrate_gbm_fx_params(
    vol_surface_pairs,          # list of (expiry_years, atm_vol) from market
    fx_name          = "FX",
    enforce_monotone = True,
    output_path      = None,
):
    """
    Calibrate GBM FX implied vol curve using the four-step analytic bootstrap.
    Mirrors RiskFlow's GBMAssetPriceTSModelParameters.bootstrap exactly:

      Step 1: Extract ATM vols sigma_bar(T_i) at each expiry
      Step 2: Compute V(T_i) = sigma_bar^2 * T_i  (integrated variance)
      Step 3: Enforce V non-decreasing (calendar-spread arbitrage removal)
      Step 4: Invert via Simpson quadratic to get instantaneous sigma(T_i)

    No optimiser — fully analytic.

    Args:
        vol_surface_pairs : list of (expiry_years, atm_vol) tuples
        fx_name           : label for output
        enforce_monotone  : clamp integrated variance if non-monotone
        output_path       : optional CSV path

    Returns:
        param : OrderedDict with Vol_Curve, IntegratedVariance DataFrame,
                InstantaneousVol, Drift description
    """
    import numpy as np
    import pandas as pd
    from collections import OrderedDict

    arr = np.array(sorted(vol_surface_pairs, key=lambda x: x[0]),
                   dtype=float)
    expiries = arr[:, 0]
    atm_vols = arr[:, 1]

    # ════════════════════════════════════════════════════════════════════════
    # STEP 2: Integrated variance V(T) = sigma_bar^2 * T
    # ════════════════════════════════════════════════════════════════════════
    V = atm_vols ** 2 * expiries

    # ════════════════════════════════════════════════════════════════════════
    # STEP 3: Enforce monotonicity
    # V must be non-decreasing — if V(T_i) < V(T_{i-1}) the instantaneous
    # vol would be imaginary (calendar-spread arbitrage)
    # Mirrors bootstrappers.py:557-578: clamp with warning
    # ════════════════════════════════════════════════════════════════════════
    clamped_flags = np.zeros(len(V), dtype=bool)
    V_enforced    = V.copy()
    sigma_prev    = atm_vols.copy()

    for i in range(1, len(V)):
        dt   = expiries[i] - expiries[i-1]
        # minimum variance at T_i given sigma at T_{i-1}
        M    = V_enforced[i-1] + dt * (sigma_prev[i-1] ** 2)
        if V_enforced[i] < M:
            print(f"  ⚠️  Non-monotone variance at T={expiries[i]:.4f}y — "
                  f"clamping {V_enforced[i]:.8f} → {M:.8f}")
            V_enforced[i]   = M
            clamped_flags[i] = True
            # update atm_vol to be consistent with clamped V
            sigma_prev[i]   = np.sqrt(V_enforced[i] / expiries[i])
        else:
            sigma_prev[i]   = atm_vols[i]

    # Recompute atm_vol from enforced V for output
    atm_vols_enforced = np.sqrt(V_enforced / expiries)

    # ════════════════════════════════════════════════════════════════════════
    # STEP 4: Instantaneous vol via Simpson quadratic inversion
    # For each bucket [T_{i-1}, T_i]:
    #   dV = V(T_i) - V(T_{i-1})
    #   Simpson's rule: dV = (dt/3)(u^2 + uv + v^2)
    #   where u = sigma(T_{i-1}), v = sigma(T_i) [unknown]
    #   Rearranging: dt*v^2 + dt*u*v + (dt*u^2 - dV) = 0
    #   Taking positive root: v = (-b + sqrt(b^2 - 4ac)) / 2a
    # ════════════════════════════════════════════════════════════════════════
    inst_vols  = np.zeros(len(expiries))
    # First point: instantaneous vol = ATM vol (no prior bucket)
    inst_vols[0] = atm_vols_enforced[0]

    for i in range(1, len(expiries)):
        dt   = (expiries[i] - expiries[i-1]) / 3.0  # dt/3 per Simpson
        u    = inst_vols[i-1]                        # known sigma(T_{i-1})
        dV   = V_enforced[i] - V_enforced[i-1]       # bucket variance

        # Quadratic coefficients (Table 1 from the PDF)
        a    = dt
        b    = dt * u
        c    = dt * u**2 - dV                        # = M - var_t in code

        discriminant = b**2 - 4.0 * a * c
        if discriminant < 0.0:
            print(f"  ⚠️  Negative discriminant at T={expiries[i]:.4f}y "
                  f"(disc={discriminant:.2e}) — setting inst_vol=0")
            inst_vols[i] = 0.0
        else:
            inst_vols[i] = (-b + np.sqrt(discriminant)) / (2.0 * a)

    # ════════════════════════════════════════════════════════════════════════
    # Build output structures
    # ════════════════════════════════════════════════════════════════════════
    variance_df = pd.DataFrame({
        'Expiry'                 : expiries,
        'ATM_Vol_Input'          : atm_vols,
        'ATM_Vol_Enforced'       : atm_vols_enforced,
        'Integrated_Variance'    : V_enforced,
        'Bucket_Forward_Variance': np.r_[
            V_enforced[0] / expiries[0],
            np.diff(V_enforced) / np.diff(expiries)
        ],
        'Instantaneous_Vol'      : inst_vols,
        'Variance_Clamped'       : clamped_flags,
    })

    # Vol_Curve: list of [expiry, atm_vol_enforced]
    # This is what RiskFlow stores as the 'Integrated' tagged Vol curve
    vol_curve_pairs = list(zip(
        expiries.tolist(),
        atm_vols_enforced.tolist()
    ))

    param = OrderedDict({
        'Vol_Curve'           : vol_curve_pairs,
        'Integrated_Variance' : variance_df,
        'Instantaneous_Vol'   : list(zip(
                                    expiries.tolist(),
                                    inst_vols.tolist()
                                )),
        'Drift'               : 'domestic_zero_rate - foreign_zero_rate',
        'Quanto_FX_Volatility': None,
        'Quanto_FX_Correlation': 0.0,
    })

    # ════════════════════════════════════════════════════════════════════════
    # Terminal summary
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  GBM FX Calibration Summary — {fx_name}")
    print(f"{'='*60}")
    print(f"  Expiry points     : {len(expiries)}")
    print(f"  Clamped points    : {int(clamped_flags.sum())}")
    print(f"\n  {'Expiry':>8}  {'ATM_Vol':>10}  "
          f"{'Integ_Var':>12}  {'Inst_Vol':>10}  {'Clamped':>7}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*7}")
    for _, row in variance_df.iterrows():
        print(f"  {row.Expiry:8.4f}  "
              f"{row.ATM_Vol_Enforced:10.6f}  "
              f"{row.Integrated_Variance:12.8f}  "
              f"{row.Instantaneous_Vol:10.6f}  "
              f"{'YES' if row.Variance_Clamped else 'no':>7}")

    if output_path:
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        variance_df.to_csv(output_path, index=False)
        print(f"\n✅ Saved: {output_path}")

    return param


def compare_gbm_fx_params(calibrated_param, extracted_param,
                           fx_name, output_path=None):
    """
    Compare calibrated GBM FX Vol curve against MarketData.json parameters.

    For each expiry point compares:
      - ATM Vol (integrated/average vol)
      - Integrated Variance V(T)
      - Instantaneous Vol sigma(T)

    Args:
        calibrated_param : OrderedDict from calibrate_gbm_fx_params()
        extracted_param  : full dict from extract_gbm_fx_params()
        fx_name          : str key into extracted_param
        output_path      : optional CSV path
    """
    import pandas as pd
    import os

    # ── Unwrap ───────────────────────────────────────────────────────────────
    if fx_name in extracted_param:
        ext = extracted_param[fx_name]
    else:
        ext = extracted_param

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
    # 1. ATM VOL CURVE — expiry by expiry
    # ════════════════════════════════════════════════════════════════════════
    cal_vol = to_dict(cal.get('Vol_Curve', []))
    ext_vol = to_dict(ext.get('Vol_Curve', []))
    print(f"\nVol Curve: cal={len(cal_vol)} expiries  "
          f"ext={len(ext_vol)} expiries")

    for T in sorted(set(cal_vol) | set(ext_vol)):
        rows.append({
            'Parameter'   : 'ATM_Vol (Integrated)',
            'Expiry'      : round(T, 6),
            'Calibrated'  : cal_vol.get(T),
            'Extracted'   : ext_vol.get(T),
            'Abs_Diff'    : adiff(cal_vol.get(T), ext_vol.get(T)),
            'Rel_Diff_Pct': rdiff(cal_vol.get(T), ext_vol.get(T)),
        })

    # ════════════════════════════════════════════════════════════════════════
    # 2. INTEGRATED VARIANCE — expiry by expiry
    # Derived: V(T) = vol^2 * T
    # ════════════════════════════════════════════════════════════════════════
    cal_iv = to_dict(cal.get('Integrated_Variance',[]) \
                     if isinstance(cal.get('Integrated_Variance'), list)
                     else [])
    # If calibrated returns DataFrame, convert it
    if hasattr(cal.get('Integrated_Variance'), 'iterrows'):
        cal_iv = {
            float(row.Expiry): float(row.Integrated_Variance)
            for _, row in cal['Integrated_Variance'].iterrows()
        }
    ext_iv = to_dict(ext.get('Integrated_Variance', []))

    for T in sorted(set(cal_iv) | set(ext_iv)):
        rows.append({
            'Parameter'   : 'Integrated_Variance V(T)',
            'Expiry'      : round(T, 6),
            'Calibrated'  : cal_iv.get(T),
            'Extracted'   : ext_iv.get(T),
            'Abs_Diff'    : adiff(cal_iv.get(T), ext_iv.get(T)),
            'Rel_Diff_Pct': rdiff(cal_iv.get(T), ext_iv.get(T)),
        })

    # ════════════════════════════════════════════════════════════════════════
    # 3. INSTANTANEOUS VOL — calibrated only (not stored in MarketData.json)
    # Included as a derived validation column
    # ════════════════════════════════════════════════════════════════════════
    cal_inst = to_dict(cal.get('Instantaneous_Vol', []))
    for T, v in sorted(cal_inst.items()):
        rows.append({
            'Parameter'   : 'Instantaneous_Vol sigma(T)',
            'Expiry'      : round(T, 6),
            'Calibrated'  : v,
            'Extracted'   : None,   # not stored in JSON — derived only
            'Abs_Diff'    : None,
            'Rel_Diff_Pct': None,
        })

    # ════════════════════════════════════════════════════════════════════════
    # 4. QUANTO CORRELATION — scalar
    # ════════════════════════════════════════════════════════════════════════
    cal_qc = cal.get('Quanto_FX_Correlation', 0.0)
    ext_qc = ext.get('Quanto_FX_Correlation', 0.0)
    rows.append({
        'Parameter'   : 'Quanto_FX_Correlation',
        'Expiry'      : 'scalar',
        'Calibrated'  : float(cal_qc) if cal_qc is not None else 0.0,
        'Extracted'   : float(ext_qc) if ext_qc is not None else 0.0,
        'Abs_Diff'    : adiff(cal_qc, ext_qc),
        'Rel_Diff_Pct': rdiff(cal_qc, ext_qc),
    })

    # ════════════════════════════════════════════════════════════════════════
    # 5. BUILD DATAFRAME
    # ════════════════════════════════════════════════════════════════════════
    df = pd.DataFrame(rows, columns=[
        'Parameter', 'Expiry',
        'Calibrated', 'Extracted',
        'Abs_Diff', 'Rel_Diff_Pct'
    ])

    for col in ['Calibrated', 'Extracted', 'Abs_Diff']:
        df[col] = df[col].apply(
            lambda x: round(x, 8) if isinstance(x, float) else x)
    df['Rel_Diff_Pct'] = df['Rel_Diff_Pct'].apply(
        lambda x: round(x, 4) if isinstance(x, float) else x)

    print(f"\n{'='*80}")
    print(f"  GBM FX Comparison — {fx_name}")
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

    MARKET_DATA = r"C:\XVA_engine\MarketData.json"
    FX_NAME     = "GBMAssetPriceTSModelParameters.FxRate.USD.ZAR"

    # 1. Extract from MarketData.json
    extracted = extract_gbm_fx_params(
        filepath = MARKET_DATA,
        fx_names = FX_NAME,
    )

    # 2. Build ATM vol quotes from your market data source
    #    (expiry_years, atm_vol) — e.g. from aaextract.ada or vol surface
    atm_vol_quotes = [
        (1/12,  0.145),
        (0.25,  0.150),
        (0.50,  0.158),
        (1.00,  0.165),
        (2.00,  0.172),
        (5.00,  0.180),
    ]

    # 3. Calibrate
    param = calibrate_gbm_fx_params(
        vol_surface_pairs = atm_vol_quotes,
        fx_name           = FX_NAME,
        output_path       = r"C:\XVA_engine\outputs\gbm_calibration.csv",
    )

    # 4. Compare
    comparison_df = compare_gbm_fx_params(
        calibrated_param = param,
        extracted_param  = extracted,
        fx_name          = FX_NAME,
        output_path      = r"C:\XVA_engine\outputs\gbm_comparison.csv",
    )