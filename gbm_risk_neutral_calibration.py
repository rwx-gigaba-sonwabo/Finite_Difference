"""
==============================================================================
GBM FX IMPLIED CALIBRATION  (Q-Measure / Risk-Neutral)
==============================================================================
Integrates the bootstrap logic from gbm_fx_implied_calibration.py with
the extract / compare / export pattern used for HW1F and PCA validation.

Pipeline:
  1. bootstrap_fx_from_json()   — reads JSON, extracts ATM vols via
                                   extract_atm_vols(), applies Simpson
                                   variance correction, returns calibrated
                                   Vol curves
  2. extract_gbm_fx_params()    — reads STORED GBMAssetPriceTSModelParameters
                                   from the same JSON (production values)
  3. compare_gbm_fx_params()    — expiry-by-expiry comparison of calibrated
                                   vs extracted, with abs and rel differences
  4. export_gbm_fx_results()    — writes Excel (two sheets) or two CSVs
==============================================================================
"""

import json
import os
import sys
import numpy as np
import pandas as pd


# ===========================================================================
# JSON helpers  (from uploaded file — kept verbatim)
# ===========================================================================

def load_market_data(json_path):
    with open(json_path, 'r') as fh:
        return json.load(fh)


def _curve_array(obj):
    """
    Extract numpy array from RiskFlow curve / surface object.
    Handles {'_type':'Curve','array':[...]} and plain list-of-lists.
    Also handles the .Curve nesting used in some JSON exports.
    """
    if isinstance(obj, dict):
        if obj.get('_type') == 'Curve':
            return np.array(obj['array'], dtype=float)
        if '.Curve' in obj:
            return np.array(obj['.Curve'].get('data', []), dtype=float)
        if 'array' in obj:
            return np.array(obj['array'], dtype=float)
        if 'data' in obj:
            return np.array(obj['data'], dtype=float)
    if isinstance(obj, (list, tuple)):
        return np.array(sorted(obj), dtype=float)
    return np.array(obj, dtype=float)


# ===========================================================================
# Vol surface reader  (from uploaded file)
# ===========================================================================

def read_vol_surface(price_factors, vol_name, is_fx=True):
    """
    Read FXVol or EquityPriceVol surface from price_factors dict.

    Returns np.ndarray shape (N, 3): [moneyness, expiry, vol]
    """
    prefix = 'FXVol' if is_fx else 'EquityPriceVol'
    key    = f'{prefix}.{vol_name}'
    if key not in price_factors:
        raise KeyError(f"'{key}' not found in Price Factors")
    return _curve_array(price_factors[key].get('Surface', {}))


# ===========================================================================
# ATM vol extraction  (from uploaded file — core function)
# ===========================================================================

def extract_atm_vols(surface_arr):
    """
    Interpolate to moneyness = 1.0 at every expiry node.

    Mirrors bootstrappers.py:
        mn_ix  = np.searchsorted(moneyness, 1.0)
        atm_vol = [np.interp(1.0, moneyness[mn_ix-1:mn_ix+1], row) ...]

    Parameters
    ----------
    surface_arr : np.ndarray shape (N, 3)
        Columns: [moneyness, expiry, vol]

    Returns
    -------
    expiries : np.ndarray  sorted unique expiry year-fractions
    atm_vols : np.ndarray  ATM vol at each expiry
    """
    expiries = np.unique(surface_arr[:, 1])
    atm_vols = np.empty(len(expiries))

    for i, exp in enumerate(expiries):
        slice_  = surface_arr[surface_arr[:, 1] == exp]
        mn      = slice_[:, 0]
        vl      = slice_[:, 2]
        order   = np.argsort(mn)
        atm_vols[i] = float(np.interp(1.0, mn[order], vl[order]))

    return expiries, atm_vols


# ===========================================================================
# Declining variance correction  (from uploaded file)
# ===========================================================================

def correct_declining_variance(expiries, atm_vols):
    """
    Ensure V(t) = sigma_avg(t)^2 * t is non-decreasing.
    Applies piecewise-linear exact Simpson integral correction.

    Returns
    -------
    avg_vols  : list   corrected average vols  (stored in Vol curve)
    inst_vols : list   instantaneous vols
    corrected : bool
    details   : list of dict  per-expiry diagnostics
    """
    expiries = np.asarray(expiries, dtype=float)
    atm_vols = np.asarray(atm_vols, dtype=float)
    n        = len(expiries)

    if n == 0:
        return [], [], False, []

    dt          = np.diff(np.concatenate([[0.0], expiries]))
    var_target  = expiries * atm_vols ** 2

    sig         = [float(atm_vols[0])]
    avg         = [float(atm_vols[0])]
    var_prev    = float(var_target[0])
    corrected   = False

    details = [{
        'expiry'     : expiries[0],
        'raw_atm_vol': atm_vols[0],
        'avg_vol'    : atm_vols[0],
        'inst_vol'   : atm_vols[0],
        'var_target' : var_target[0],
        'var_actual' : var_target[0],
        'clamped'    : False,
    }]

    for i in range(1, n):
        delta_t = dt[i] / 3.0
        var_t   = float(var_target[i])

        # Minimum achievable variance
        M       = var_prev + delta_t * sig[-1] ** 2
        clamped = False

        if var_t < M:
            corrected = True
            clamped   = True
            var_t     = M

        # Quadratic: a*v^2 + b*v + c = 0
        a             = delta_t
        b             = sig[-1] * delta_t
        c             = M - var_t          # <= 0; zero when clamped
        discriminant  = b * b - 4.0 * a * c
        sig_i         = (-b + np.sqrt(max(discriminant, 0.0))) / (2.0 * a)
        avg_i         = np.sqrt(var_t / expiries[i])

        sig.append(float(sig_i))
        avg.append(float(avg_i))
        var_prev = var_t

        details.append({
            'expiry'     : expiries[i],
            'raw_atm_vol': float(atm_vols[i]),
            'avg_vol'    : float(avg_i),
            'inst_vol'   : float(sig_i),
            'var_target' : float(var_target[i]),
            'var_actual' : float(var_t),
            'clamped'    : clamped,
        })

    return avg, sig, corrected, details


# ===========================================================================
# STEP 1: bootstrap_fx_from_json
# Reads Market Prices, locates FXVol surfaces, calls extract_atm_vols
# ===========================================================================

_MARKET_PRICE_TYPES = {'GBMAssetPriceTSModelPrices', 'GBMTSModelPrices'}


def bootstrap_fx_from_json(json_path, fx_name=None, verbose=True):
    """
    Full GBM FX implied calibration from a RiskFlow JSON file.

    Parameters
    ----------
    json_path : str   path to RiskFlow JSON
    fx_name   : str   optional filter, e.g. 'EUR.USD'
    verbose   : bool

    Returns
    -------
    dict keyed by currency name, each containing:
        'Vol'                   list of (expiry, avg_vol)
        'Quanto_FX_Volatility'  None
        'Quanto_FX_Correlation' 0.0
        '_vol_surface_name'     str
        '_is_fx'                bool
        '_was_corrected'        bool
        '_details'              list of per-expiry diagnostic dicts
        '_expiries'             np.ndarray
        '_atm_vols_raw'         np.ndarray  (before correction)
    """
    if verbose:
        print("=" * 70)
        print("GBM FX IMPLIED CALIBRATION  —  BOOTSTRAP FROM JSON")
        print("=" * 70)
        print(f"  File: {json_path}")

    market_data   = load_market_data(json_path)
    price_factors = market_data.get('Price Factors', {})
    market_prices = market_data.get('Market Prices', {})

    if verbose:
        print(f"  Price Factors : {len(price_factors)} entries")
        print(f"  Market Prices : {len(market_prices)} entries")

    results = {}

    for mp_name, implied_params in market_prices.items():
        parts       = mp_name.split('.')
        factor_type = parts[0]
        if factor_type not in _MARKET_PRICE_TYPES:
            continue

        currency = '.'.join(parts[1:])
        if fx_name is not None and currency.upper() != fx_name.upper():
            continue

        instrument = implied_params.get('instrument', implied_params)
        vol_name   = instrument.get('Asset_Price_Volatility', '')
        if not vol_name:
            if verbose:
                print(f"  WARNING: No Asset_Price_Volatility for {mp_name} — skipping")
            continue

        is_fx = ('FXVol.'          + vol_name) in price_factors
        is_eq = ('EquityPriceVol.' + vol_name) in price_factors

        if not is_fx and not is_eq:
            if verbose:
                print(f"  WARNING: surface not found for {vol_name} — skipping {currency}")
            continue

        surf_type = 'FXVol' if is_fx else 'EquityPriceVol'
        if verbose:
            print(f"\n  {'-'*60}")
            print(f"  Currency : {currency}")
            print(f"  Surface  : {surf_type}.{vol_name}")

        try:
            surface_arr = read_vol_surface(price_factors, vol_name, is_fx=is_fx)
        except KeyError as exc:
            if verbose:
                print(f"  ERROR: {exc} — skipping")
            continue

        # ── Core extraction: ATM vols from surface ───────────────────────────
        expiries, atm_vols_raw = extract_atm_vols(surface_arr)

        if verbose:
            print(f"\n  Raw ATM vols ({len(expiries)} expiry points):")
            print(f"  {'Expiry':>8s}  {'ATM Vol':>10s}  {'Variance':>10s}")
            print("  " + "-" * 36)
            for t, v in zip(expiries, atm_vols_raw):
                print(f"  {t:8.4f}  {v:10.6f}  {v*v*t:10.6f}")

        # ── Variance correction ──────────────────────────────────────────────
        avg_vols, inst_vols, was_corrected, details = \
            correct_declining_variance(expiries, atm_vols_raw)

        if verbose:
            if was_corrected:
                print(f"\n  NOTE: Declining variance corrected.")
            print(f"\n  Calibrated Vol curve (after correction):")
            print(f"  {'Expiry':>8s}  {'Raw ATM':>10s}  {'Avg Vol':>10s}  "
                  f"{'Inst Vol':>10s}  {'Variance':>10s}  {'Clamped':>8s}")
            print("  " + "-" * 70)
            for d in details:
                flag = '*' if d['clamped'] else ' '
                print(f"  {d['expiry']:8.4f}  {d['raw_atm_vol']:10.6f}  "
                      f"{d['avg_vol']:10.6f}  {d['inst_vol']:10.6f}  "
                      f"{d['var_actual']:10.6f}  {flag:>8s}")

        results[currency] = {
            'Vol'                   : list(zip(expiries.tolist(), avg_vols)),
            'Quanto_FX_Volatility'  : None,
            'Quanto_FX_Correlation' : 0.0,
            '_vol_surface_name'     : vol_name,
            '_is_fx'                : is_fx,
            '_was_corrected'        : was_corrected,
            '_details'              : details,
            '_expiries'             : expiries,
            '_atm_vols_raw'         : atm_vols_raw,
        }

    if not results and verbose:
        print(f"\n  WARNING: No GBM market price entries found.")
        print(f"  First 20 Market Prices keys:")
        for k in list(market_prices.keys())[:20]:
            print(f"    {k}")

    if verbose:
        print("\n" + "=" * 70)
        print(f"  Calibrated {len(results)} pair(s): "
              f"{', '.join(results.keys()) or 'none'}")
        print("=" * 70)

    return results


# ===========================================================================
# STEP 2: extract_gbm_fx_params
# Reads STORED GBMAssetPriceTSModelParameters from Price Factors
# ===========================================================================

def extract_gbm_fx_params(json_path, fx_names=None, verbose=True):
    """
    Extract stored GBM FX parameters from the Price Factors section.
    These are the PRODUCTION values RiskFlow reads at runtime.

    Parameters
    ----------
    json_path : str
    fx_names  : str, list, or None (extract all)

    Returns
    -------
    dict keyed by currency name:
        'Vol_Curve'             list of [expiry, vol]
        'Integrated_Variance'   list of [expiry, V(T)]
        'Quanto_FX_Volatility'  list or None
        'Quanto_FX_Correlation' float
    """
    market_data   = load_market_data(json_path)
    price_factors = market_data.get('Price Factors', {})

    GBM_PREFIX = 'GBMAssetPriceTSModelParameters.'

    if fx_names is None:
        fx_names = [
            k[len(GBM_PREFIX):]
            for k in price_factors
            if k.startswith(GBM_PREFIX)
        ]
        if not fx_names and verbose:
            print("WARNING: No GBMAssetPriceTSModelParameters entries found.")
            print("Available Price Factor keys (first 20):",
                  list(price_factors.keys())[:20])
    elif isinstance(fx_names, str):
        fx_names = [fx_names]

    results = {}

    for name in fx_names:
        full_key = GBM_PREFIX + name \
                   if not name.startswith(GBM_PREFIX) else name
        clean    = full_key[len(GBM_PREFIX):]

        factor_data = price_factors.get(full_key)
        if factor_data is None:
            if verbose:
                print(f"WARNING: '{full_key}' not found in Price Factors.")
            continue

        # ── Extract Vol curve ────────────────────────────────────────────────
        vol_raw  = factor_data.get('Vol')
        if vol_raw is None:
            if verbose:
                print(f"WARNING: No 'Vol' key in {full_key}")
            continue

        vol_arr  = _curve_array(vol_raw)
        if vol_arr.ndim != 2 or vol_arr.shape[1] < 2:
            if verbose:
                print(f"WARNING: Unexpected Vol shape {vol_arr.shape} for {full_key}")
            continue

        vol_pairs = [[float(r[0]), float(r[1])] for r in vol_arr]

        # Recompute integrated variance from stored vols
        integ_var = [[T, v**2 * T] for T, v in vol_pairs]

        # ── Quanto fields ────────────────────────────────────────────────────
        qv_raw  = factor_data.get('Quanto_FX_Volatility')
        qv      = _curve_array(qv_raw).tolist() \
                  if qv_raw is not None and qv_raw != 0.0 else []
        qc      = float(factor_data.get('Quanto_FX_Correlation', 0.0) or 0.0)

        results[clean] = {
            'Vol_Curve'            : vol_pairs,
            'Integrated_Variance'  : integ_var,
            'Quanto_FX_Volatility' : qv,
            'Quanto_FX_Correlation': qc,
        }

        if verbose:
            print(f"\n✅ Extracted '{full_key}':")
            print(f"   Vol Curve : {len(vol_pairs)} expiry points  "
                  f"first={vol_pairs[0]}  last={vol_pairs[-1]}")
            print(f"   Quanto FX Correlation: {qc}")

    return results


# ===========================================================================
# STEP 3: compare_gbm_fx_params
# Expiry-by-expiry comparison of calibrated vs extracted
# ===========================================================================

def compare_gbm_fx_params(calibrated, extracted, verbose=True):
    """
    Compare bootstrap-calibrated Vol curves against stored parameters.

    Parameters
    ----------
    calibrated : dict   output of bootstrap_fx_from_json()
    extracted  : dict   output of extract_gbm_fx_params()
    verbose    : bool

    Returns
    -------
    dict keyed by currency.  Each value is a pd.DataFrame:
        Expiry, RiskFlow_Vol, Calibrated_Vol,
        RiskFlow_Variance, Calibrated_Variance,
        Abs_Diff_Vol, Rel_Diff_Vol_Pct,
        Abs_Diff_Var, Rel_Diff_Var_Pct,
        Instantaneous_Vol, Clamped
    """
    comparisons = {}

    # Union of currencies present in both
    all_currencies = set(calibrated.keys()) | set(extracted.keys())

    for currency in sorted(all_currencies):

        if currency not in calibrated:
            if verbose:
                print(f"  {currency}: no calibrated result — skipping")
            continue
        if currency not in extracted:
            if verbose:
                print(f"  {currency}: not found in stored params — skipping")
            continue

        calib = calibrated[currency]
        ext   = extracted[currency]

        # ── Build arrays ─────────────────────────────────────────────────────
        ext_arr   = np.array(ext['Vol_Curve'],  dtype=float)   # [T, vol]
        cal_exp   = np.array([x[0] for x in calib['Vol']])
        cal_vol   = np.array([x[1] for x in calib['Vol']])

        ext_exp   = ext_arr[:, 0]
        ext_vol   = ext_arr[:, 1]

        # Interpolate calibrated onto extracted expiry grid
        cal_interp = np.interp(ext_exp, cal_exp, cal_vol,
                               left=cal_vol[0], right=cal_vol[-1])

        ext_var    = ext_vol   ** 2 * ext_exp
        cal_var    = cal_interp ** 2 * ext_exp

        abs_vol    = cal_interp - ext_vol
        abs_var    = cal_var    - ext_var

        with np.errstate(invalid='ignore', divide='ignore'):
            rel_vol = np.where(np.abs(ext_vol) > 1e-12,
                               100.0 * abs_vol / ext_vol, np.nan)
            rel_var = np.where(np.abs(ext_var) > 1e-12,
                               100.0 * abs_var / ext_var, np.nan)

        # Instantaneous vol from calibration details (if available)
        details    = calib.get('_details', [])
        inst_lookup = {d['expiry']: d['inst_vol'] for d in details}
        clamped_lk  = {d['expiry']: d['clamped']  for d in details}

        inst_vols  = np.array([inst_lookup.get(t, np.nan) for t in ext_exp])
        clamped    = np.array([clamped_lk.get(t,  False)  for t in ext_exp])

        df = pd.DataFrame({
            'Expiry'            : ext_exp,
            'RiskFlow_Vol'      : np.round(ext_vol,    8),
            'Calibrated_Vol'    : np.round(cal_interp, 8),
            'Abs_Diff_Vol'      : np.round(abs_vol,    8),
            'Rel_Diff_Vol_Pct'  : np.round(rel_vol,    4),
            'RiskFlow_Variance' : np.round(ext_var,    8),
            'Calibrated_Variance': np.round(cal_var,   8),
            'Abs_Diff_Var'      : np.round(abs_var,    8),
            'Rel_Diff_Var_Pct'  : np.round(rel_var,    4),
            'Instantaneous_Vol' : np.round(inst_vols,  8),
            'Variance_Clamped'  : clamped,
        })

        comparisons[currency] = df

        if verbose:
            print(f"\n{'='*76}")
            print(f"  GBM FX Comparison — {currency}")
            print(f"{'='*76}")
            print(df[[
                'Expiry','RiskFlow_Vol','Calibrated_Vol',
                'Abs_Diff_Vol','Rel_Diff_Vol_Pct'
            ]].to_string(index=False))

            large = df[df['Rel_Diff_Vol_Pct'].apply(
                lambda x: isinstance(x, float) and abs(x) > 1.0)]
            if not large.empty:
                print(f"\n  ⚠️  {len(large)} vol points > 1% relative diff:")
                print(large[['Expiry','RiskFlow_Vol',
                              'Calibrated_Vol','Rel_Diff_Vol_Pct'
                              ]].to_string(index=False))
            else:
                print(f"\n  ✅ All vol points within 1% tolerance.")

            print(f"\n  Max  |abs diff vol| : {df['Abs_Diff_Vol'].abs().max():.8f}")
            print(f"  Mean |abs diff vol| : {df['Abs_Diff_Vol'].abs().mean():.8f}")
            print(f"  Max  |rel diff vol| : {df['Rel_Diff_Vol_Pct'].abs().max():.4f}%")

    return comparisons


# ===========================================================================
# STEP 4: export_gbm_fx_results
# Two-sheet Excel or two CSVs — matches HW1F / PCA export pattern
# ===========================================================================

def export_gbm_fx_results(calibrated, comparisons, output_path,
                           verbose=True):
    """
    Export calibrated vols, comparisons, and summary.

    Excel sheets produced:
        Calibrated_Vols      — all (currency, expiry, vol, variance)
        Cmp_<currency>       — per-currency comparison DataFrame
        Summary              — aggregate diff metrics per currency

    Falls back to CSV if openpyxl is unavailable.
    """
    import os

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # ── Sheet 1: Calibrated Vols ─────────────────────────────────────────────
    cal_rows = []
    for currency, p in calibrated.items():
        for d in p.get('_details', []):
            cal_rows.append({
                'FX_Pair'          : currency,
                'Vol_Surface'      : p['_vol_surface_name'],
                'Expiry'           : d['expiry'],
                'Raw_ATM_Vol'      : round(d['raw_atm_vol'], 8),
                'Calibrated_Vol'   : round(d['avg_vol'],     8),
                'Instantaneous_Vol': round(d['inst_vol'],    8),
                'Integrated_Var'   : round(d['var_actual'],  8),
                'Variance_Clamped' : d['clamped'],
                'Corrected'        : p['_was_corrected'],
            })
    cal_df = pd.DataFrame(cal_rows)

    # ── Summary ──────────────────────────────────────────────────────────────
    summary_rows = []
    for currency, df in comparisons.items():
        summary_rows.append({
            'FX_Pair'           : currency,
            'N_Expiry_Points'   : len(df),
            'Max_Abs_Diff_Vol'  : df['Abs_Diff_Vol'].abs().max(),
            'Mean_Abs_Diff_Vol' : df['Abs_Diff_Vol'].abs().mean(),
            'Max_Rel_Diff_Pct'  : df['Rel_Diff_Vol_Pct'].abs().max(),
            'Mean_Rel_Diff_Pct' : df['Rel_Diff_Vol_Pct'].abs().mean(),
            'N_Clamped'         : int(df['Variance_Clamped'].sum()),
            'Any_Exceedance_1pct': bool(
                (df['Rel_Diff_Vol_Pct'].abs() > 1.0).any()
            ),
        })
    summary_df = pd.DataFrame(summary_rows)

    # ── Write ─────────────────────────────────────────────────────────────────
    try:
        import openpyxl  # noqa
        xlsx_path = output_path if output_path.endswith('.xlsx') \
                    else output_path.replace('.csv', '.xlsx')

        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            if not cal_df.empty:
                cal_df.to_excel(writer,
                                sheet_name='Calibrated_Vols', index=False)
            for currency, df in comparisons.items():
                sheet = f'Cmp_{currency}'[:31]
                df.to_excel(writer, sheet_name=sheet, index=False)
            if not summary_df.empty:
                summary_df.to_excel(writer,
                                    sheet_name='Summary', index=False)

        if verbose:
            print(f"\n✅ Saved Excel: {xlsx_path}")

    except ImportError:
        # Fallback: write individual CSVs
        base = output_path.replace('.xlsx', '').replace('.csv', '')
        cal_df.to_csv(f'{base}_calibrated_vols.csv',     index=False)
        summary_df.to_csv(f'{base}_summary.csv',          index=False)
        for currency, df in comparisons.items():
            safe = currency.replace('/', '-').replace('\\', '-')
            df.to_csv(f'{base}_cmp_{safe}.csv', index=False)
        if verbose:
            print(f"\n✅ Saved CSVs to: {base}_*.csv")

    return cal_df, summary_df


# ===========================================================================
# Full pipeline wrapper
# ===========================================================================

def run_gbm_fx_calibration(json_path, output_path,
                            fx_name=None, verbose=True):
    """
    Run the complete GBM FX calibration and comparison pipeline.

    1. Bootstrap calibrated vols from FXVol surfaces
    2. Extract stored production vols from GBMAssetPriceTSModelParameters
    3. Compare expiry-by-expiry
    4. Export to Excel / CSV

    Parameters
    ----------
    json_path   : str   RiskFlow JSON market data file
    output_path : str   output .xlsx or .csv path
    fx_name     : str   optional currency filter e.g. 'EUR.USD'
    verbose     : bool
    """
    # Step 1
    calibrated = bootstrap_fx_from_json(
        json_path, fx_name=fx_name, verbose=verbose
    )

    # Step 2
    extracted = extract_gbm_fx_params(
        json_path, fx_names=fx_name, verbose=verbose
    )

    # Step 3
    comparisons = compare_gbm_fx_params(
        calibrated, extracted, verbose=verbose
    )

    # Step 4
    cal_df, summary_df = export_gbm_fx_results(
        calibrated, comparisons, output_path, verbose=verbose
    )

    return calibrated, extracted, comparisons, cal_df, summary_df


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] in ('--test', '-t', 'test'):
        # Run self-test from uploaded file
        from gbm_fx_implied_calibration import _self_test
        _self_test()
        sys.exit(0)

    json_path   = sys.argv[1]
    fx_name     = sys.argv[2] if len(sys.argv) > 2 else None
    output_path = sys.argv[3] if len(sys.argv) > 3 \
                  else r'C:\XVA_engine\outputs\gbm_fx_calibration.xlsx'

    calibrated, extracted, comparisons, cal_df, summary_df = \
        run_gbm_fx_calibration(
            json_path   = json_path,
            output_path = output_path,
            fx_name     = fx_name,
            verbose     = True,
        )
