"""
GBM FX IMPLIED CALIBRATION  (Q-Measure / Risk-Neutral)

PURPOSE:
    Calibrates GBMAssetPriceTSModelParameters for FX rates from market
    FX volatility surfaces (FXVol).  This replicates — and exposes — the
    logic inside GBMAssetPriceTSModelParameters.bootstrap() in
    bootstrappers.py as a standalone, inspectable calibration script.

WHAT THIS SCRIPT DOES:
    1. Scans the Market Prices section of a RiskFlow JSON file for
       GBMAssetPriceTSModelPrices (alias GBMTSModelPrices) entries.
    2. Locates the referenced FXVol surface for each entry.
    3. Extracts the ATM (moneyness = 1.0) implied vol at every expiry node.
    4. Applies the Simpson's-rule declining-variance correction so that
       the integrated variance  V(t) = sigma_avg(t)^2 * t  is strictly
       non-decreasing.
    5. Returns the calibrated Vol curve:  [(t_1, sigma_1), ..., (t_n, sigma_n)]

CALIBRATED OUTPUT (stored as GBMAssetPriceTSModelParameters.{currency}):
    Vol                  Curve  — average ATM vol at each expiry, corrected
    Quanto_FX_Volatility None   — plain FX, no quanto adjustment
    Quanto_FX_Correlation 0.0  — plain FX

VARIANCE CORRECTION (piecewise-linear exact integral):
    Integrated variance must satisfy V(t_i) >= V(t_{i-1}).  When the raw ATM
    vol surface implies a decrease, the code fixes var_t = M (the minimum
    allowed value) and solves for the instantaneous vol sig_i via:

        (dt/3) * (sig_{i-1}^2 + sig_{i-1}*sig_i + sig_i^2) = V(t_i) - V(t_{i-1})

    This is the exact integral of a piecewise-linear sigma(t) from sig_{i-1}
    to sig_i over an interval dt, which is also the Simpson's-rule value for
    that interval.  Solving the quadratic in sig_i:

        a = dt/3
        b = sig_{i-1} * dt/3
        c = M - V(t_i)      (zero when the variance was clamped)

    sig_i = (-b + sqrt(b^2 - 4*a*c)) / (2*a)
"""

import json
import sys
import os
import numpy as np
import pandas as pd
import scipy.stats


# ---------------------------------------------------------------------------
# JSON / market-data helpers
# ---------------------------------------------------------------------------

def load_market_data(json_path):
    """Load a RiskFlow JSON file and return the parsed dict."""
    with open(json_path, 'r') as fh:
        return json.load(fh)


def _curve_array(obj):
    """
    Extract a numpy array from a raw JSON curve / surface object.

    Handles both the serialised  {'_type': 'Curve', 'array': [...]}  form
    and plain lists of lists.
    """
    if isinstance(obj, dict) and obj.get('_type') == 'Curve':
        return np.array(obj['array'], dtype=float)
    if isinstance(obj, (list, tuple)):
        return np.array(sorted(obj), dtype=float)
    return np.array(obj, dtype=float)


# ---------------------------------------------------------------------------
# FXVol surface reader
# ---------------------------------------------------------------------------

def read_vol_surface(price_factors, vol_name, is_fx=True):
    """
    Read and parse a 2-D vol surface (FXVol or EquityPriceVol) from
    the price_factors dict.

    FXVol Surface columns:  [moneyness, expiry, vol]

    Parameters
    ----------
    price_factors : dict
        The 'Price Factors' section from the market-data JSON.
    vol_name : str
        Surface name, e.g. 'EUR.USD'  (without type prefix).
    is_fx : bool
        True  -> look for  'FXVol.<vol_name>'
        False -> look for  'EquityPriceVol.<vol_name>'

    Returns
    -------
    np.ndarray  shape (N, 3)  columns [moneyness, expiry, vol]
    """
    prefix = 'FXVol' if is_fx else 'EquityPriceVol'
    key = f'{prefix}.{vol_name}'
    if key not in price_factors:
        raise KeyError(f"'{key}' not found in Price Factors")
    return _curve_array(price_factors[key].get('Surface', {}))


# ---------------------------------------------------------------------------
# ATM vol extraction
# ---------------------------------------------------------------------------

def extract_atm_vols(surface_arr):
    """
    Interpolate to moneyness = 1.0 at every expiry node.

    Mirrors the bootstrapper idiom:
        mn_ix = np.searchsorted(moneyness, 1.0)
        atm_vol = [np.interp(1.0, moneyness[mn_ix-1:mn_ix+1], row) ...]

    Parameters
    ----------
    surface_arr : np.ndarray  shape (N, 3)
        Columns: [moneyness, expiry, vol]

    Returns
    -------
    expiries : np.ndarray  sorted unique expiry year-fractions
    atm_vols : np.ndarray  ATM vol at each expiry (same length)
    """
    expiries = np.unique(surface_arr[:, 1])
    atm_vols = np.empty(len(expiries))

    for i, exp in enumerate(expiries):
        slice_ = surface_arr[surface_arr[:, 1] == exp]
        mn = slice_[:, 0]
        vl = slice_[:, 2]
        # sort by moneyness before interpolating
        order = np.argsort(mn)
        atm_vols[i] = float(np.interp(1.0, mn[order], vl[order]))

    return expiries, atm_vols


# ---------------------------------------------------------------------------
# Declining-variance correction  (replicates bootstrappers.py lines 557-577)
# ---------------------------------------------------------------------------

def correct_declining_variance(expiries, atm_vols):
    """
    Ensure V(t) = sigma_avg(t)^2 * t is non-decreasing by applying the
    piecewise-linear exact integral.

    Parameters
    ----------
    expiries : array-like   Year fractions, sorted ascending.
    atm_vols : array-like   ATM implied vols at each expiry.

    Returns
    -------
    avg_vols : list
        Corrected average vols (what gets stored in the Vol curve).
    inst_vols : list
        Instantaneous vols derived by the correction.
    corrected : bool
        True if any expiry required adjustment.
    details : list of dict
        Per-step diagnostic information.
    """
    expiries = np.asarray(expiries, dtype=float)
    atm_vols = np.asarray(atm_vols, dtype=float)
    n = len(expiries)

    if n == 0:
        return [], [], False, []

    # dt[i] = expiry[i] - expiry[i-1],  dt[0] = expiry[0] - 0
    dt = np.diff(np.concatenate([[0.0], expiries]))

    # Target integrated variance from the raw ATM vols
    var_target = expiries * atm_vols ** 2

    # Seed from the first expiry point
    sig = [float(atm_vols[0])]    # instantaneous vol
    avg = [float(atm_vols[0])]    # average vol (output)
    var_prev = float(var_target[0])
    corrected = False

    details = [{
        'expiry': expiries[0],
        'raw_atm_vol': atm_vols[0],
        'avg_vol': atm_vols[0],
        'inst_vol': atm_vols[0],
        'var_target': var_target[0],
        'var_actual': var_target[0],
        'clamped': False,
    }]

    for i in range(1, n):
        delta_t = dt[i] / 3.0          # dt / 3  (from Simpson's rule / exact integral)
        var_t = float(var_target[i])

        # Minimum achievable variance: V_prev + (dt/3) * sig_prev^2
        M = var_prev + delta_t * sig[-1] ** 2
        clamped = False

        if var_t < M:
            corrected = True
            clamped = True
            var_t = M

        # Quadratic to find instantaneous vol sig_i:
        #   delta_t * (sig_i^2 + sig_prev*sig_i + sig_prev^2) = var_t - var_prev
        # Rearranged: a*sig_i^2 + b*sig_i + c = 0
        a = delta_t
        b = sig[-1] * delta_t
        c = M - var_t                   # <= 0 always; 0 when clamped

        discriminant = b * b - 4.0 * a * c   # always >= 0 by construction
        sig_i = (-b + np.sqrt(max(discriminant, 0.0))) / (2.0 * a)

        avg_i = np.sqrt(var_t / expiries[i])

        sig.append(float(sig_i))
        avg.append(float(avg_i))
        var_prev = var_t

        details.append({
            'expiry': expiries[i],
            'raw_atm_vol': float(atm_vols[i]),
            'avg_vol': float(avg_i),
            'inst_vol': float(sig_i),
            'var_target': float(var_target[i]),
            'var_actual': float(var_t),
            'clamped': clamped,
        })

    return avg, sig, corrected, details


# ---------------------------------------------------------------------------
# Main calibration pipeline
# ---------------------------------------------------------------------------

_MARKET_PRICE_TYPES = {'GBMAssetPriceTSModelPrices', 'GBMTSModelPrices'}


def bootstrap_fx_from_json(json_path, fx_name=None, verbose=True):
    """
    Full GBM FX implied calibration pipeline from a RiskFlow JSON file.

    Steps
    -----
    1. Load JSON; locate GBMAssetPriceTSModelPrices / GBMTSModelPrices entries.
    2. For each entry load the referenced FXVol surface.
    3. Extract ATM vols; apply declining-variance correction.
    4. Return the calibrated Vol curve and diagnostics.

    Parameters
    ----------
    json_path : str
        Path to a RiskFlow JSON market-data file.
    fx_name : str, optional
        Restrict calibration to this currency (e.g. 'EUR').
        None = calibrate all entries found.
    verbose : bool
        Print detailed per-expiry tables.

    Returns
    -------
    dict
        Keyed by currency name.  Each value is a dict with:
          'Vol'                  list of (expiry, avg_vol) tuples
          'Quanto_FX_Volatility' None
          'Quanto_FX_Correlation' 0.0
          '_vol_surface_name'    str   (FXVol key used)
          '_is_fx'               bool
          '_was_corrected'       bool
          '_details'             list of per-expiry diagnostic dicts
    """
    if verbose:
        print("=" * 70)
        print("GBM FX IMPLIED CALIBRATION  —  BOOTSTRAP FROM JSON")
        print("=" * 70)
        print(f"  File   : {json_path}")

    market_data = load_market_data(json_path)
    price_factors = market_data.get('Price Factors', {})
    market_prices = market_data.get('Market Prices', {})

    if verbose:
        print(f"  Price Factors entries : {len(price_factors)}")
        print(f"  Market Prices entries : {len(market_prices)}")

    results = {}

    for mp_name, implied_params in market_prices.items():
        parts = mp_name.split('.')
        factor_type = parts[0]
        if factor_type not in _MARKET_PRICE_TYPES:
            continue

        currency = '.'.join(parts[1:])
        if fx_name is not None and currency.upper() != fx_name.upper():
            continue

        instrument = implied_params.get('instrument', implied_params)
        vol_name = instrument.get('Asset_Price_Volatility', '')
        if not vol_name:
            if verbose:
                print(f"  WARNING: No Asset_Price_Volatility for {mp_name} — skipping")
            continue

        # Determine surface type (FXVol takes priority over EquityPriceVol)
        is_fx = ('FXVol.' + vol_name) in price_factors
        is_eq = ('EquityPriceVol.' + vol_name) in price_factors

        if not is_fx and not is_eq:
            if verbose:
                print(f"  WARNING: Neither FXVol.{vol_name} nor EquityPriceVol.{vol_name} "
                      f"found in Price Factors — skipping {currency}")
            continue

        if verbose:
            surf_type = 'FXVol' if is_fx else 'EquityPriceVol'
            print(f"\n  {'-'*60}")
            print(f"  Currency  : {currency}")
            print(f"  Surface   : {surf_type}.{vol_name}")
            print(f"  {'-'*60}")

        try:
            surface_arr = read_vol_surface(price_factors, vol_name, is_fx=is_fx)
        except KeyError as exc:
            if verbose:
                print(f"  ERROR: {exc} — skipping")
            continue

        expiries, atm_vols = extract_atm_vols(surface_arr)

        if verbose:
            print(f"\n  Raw ATM vols from surface ({len(expiries)} expiry points):")
            print(f"  {'Expiry':>8s}  {'ATM Vol':>10s}  {'Variance':>10s}")
            print("  " + "-" * 36)
            for t, v in zip(expiries, atm_vols):
                print(f"  {t:8.4f}  {v:10.6f}  {v*v*t:10.6f}")

        avg_vols, inst_vols, was_corrected, details = correct_declining_variance(
            expiries, atm_vols)

        if verbose and was_corrected:
            print(f"\n  NOTE: Declining variance detected and corrected.")

        if verbose:
            print(f"\n  Calibrated Vol curve (after variance correction):")
            print(f"  {'Expiry':>8s}  {'Raw ATM':>10s}  {'Avg Vol':>10s}  "
                  f"{'Inst Vol':>10s}  {'Variance':>10s}  {'Clamped':>8s}")
            print("  " + "-" * 70)
            for d in details:
                flag = '*' if d['clamped'] else ' '
                print(f"  {d['expiry']:8.4f}  {d['raw_atm_vol']:10.6f}  "
                      f"{d['avg_vol']:10.6f}  {d['inst_vol']:10.6f}  "
                      f"{d['var_actual']:10.6f}  {flag:>8s}")

        results[currency] = {
            'Vol': list(zip(expiries, avg_vols)),
            'Quanto_FX_Volatility': None,
            'Quanto_FX_Correlation': 0.0,
            '_vol_surface_name': vol_name,
            '_is_fx': is_fx,
            '_was_corrected': was_corrected,
            '_details': details,
        }

    if not results and verbose:
        print(f"\n  WARNING: No GBMAssetPriceTSModelPrices / GBMTSModelPrices entries found.")
        print(f"  First 20 Market Prices keys:")
        for k in list(market_prices.keys())[:20]:
            print(f"    {k}")

    if verbose:
        print("\n" + "=" * 70)
        print(f"  Calibrated {len(results)} FX pair(s): {', '.join(results.keys()) or 'none'}")
        print("=" * 70)

    return results


# ---------------------------------------------------------------------------
# Comparison against stored RiskFlow GBMAssetPriceTSModelParameters
# ---------------------------------------------------------------------------

def compare_with_riskflow(calibrated, json_path, verbose=True):
    """
    Compare calibrated Vol curves against stored GBMAssetPriceTSModelParameters
    already present in the JSON's Price Factors section.

    Parameters
    ----------
    calibrated : dict
        Output of bootstrap_fx_from_json().
    json_path : str
    verbose : bool

    Returns
    -------
    dict
        Keyed by currency name.  Each value is a pd.DataFrame with columns:
          Expiry, RiskFlow_Vol, Calibrated_Vol, Abs_Diff, Rel_Diff_Pct
    """
    market_data = load_market_data(json_path)
    price_factors = market_data.get('Price Factors', {})

    comparisons = {}

    for currency, calib in calibrated.items():
        stored_key = f'GBMAssetPriceTSModelParameters.{currency}'
        if stored_key not in price_factors:
            if verbose:
                print(f"  No stored params for {currency} ({stored_key}) — skipping comparison")
            continue

        stored_raw = price_factors[stored_key]
        vol_obj = stored_raw.get('Vol')
        if vol_obj is None:
            if verbose:
                print(f"  Stored params for {currency} have no 'Vol' — skipping")
            continue

        stored_arr = _curve_array(vol_obj)
        stored_expiries = stored_arr[:, 0]
        stored_vols = stored_arr[:, 1]

        calib_exp = np.array([x[0] for x in calib['Vol']])
        calib_vol = np.array([x[1] for x in calib['Vol']])

        # Interpolate our calibrated vols to the stored expiry grid
        calib_interp = np.interp(stored_expiries, calib_exp, calib_vol,
                                 left=calib_vol[0], right=calib_vol[-1])

        abs_diff = calib_interp - stored_vols
        with np.errstate(invalid='ignore', divide='ignore'):
            rel_diff_pct = np.where(
                np.abs(stored_vols) > 1e-12,
                100.0 * abs_diff / stored_vols,
                np.nan
            )

        df = pd.DataFrame({
            'Expiry': stored_expiries,
            'RiskFlow_Vol': stored_vols,
            'Calibrated_Vol': calib_interp,
            'Abs_Diff': abs_diff,
            'Rel_Diff_Pct': rel_diff_pct,
        })
        comparisons[currency] = df

    return comparisons


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

def print_calibration_summary(results):
    """Print a compact summary table of all calibrated currencies."""
    print("\n" + "=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)
    print(f"  {'Currency':<20s}  {'Surface':<20s}  {'Points':>6s}  "
          f"{'Min Vol':>8s}  {'Max Vol':>8s}  {'Corrected':>10s}")
    print("  " + "-" * 78)
    for currency, p in results.items():
        vols = [v for _, v in p['Vol']]
        print(f"  {currency:<20s}  {p['_vol_surface_name']:<20s}  "
              f"{len(vols):>6d}  {min(vols):>8.4f}  {max(vols):>8.4f}  "
              f"{'YES' if p['_was_corrected'] else 'no':>10s}")
    print("=" * 70)


def print_comparison_tables(comparisons):
    """Print per-currency comparison tables with difference metrics."""
    for currency, df in comparisons.items():
        print(f"\n{'='*70}")
        print(f"COMPARISON vs RISKFLOW STORED PARAMS  —  {currency}")
        print(f"{'='*70}")
        print(f"  {'Expiry':>8s}  {'RiskFlow Vol':>13s}  {'Calibrated':>12s}  "
              f"{'Abs Diff':>10s}  {'Rel Diff%':>10s}")
        print("  " + "-" * 64)
        for _, row in df.iterrows():
            print(f"  {row['Expiry']:8.4f}  {row['RiskFlow_Vol']:13.6f}  "
                  f"{row['Calibrated_Vol']:12.6f}  "
                  f"{row['Abs_Diff']:+10.6f}  {row['Rel_Diff_Pct']:+10.4f}%")
        print()
        max_abs = df['Abs_Diff'].abs().max()
        mean_abs = df['Abs_Diff'].abs().mean()
        max_rel = df['Rel_Diff_Pct'].abs().max()
        mean_rel = df['Rel_Diff_Pct'].abs().mean()
        print(f"  Max  |abs diff| : {max_abs:.8f}")
        print(f"  Mean |abs diff| : {mean_abs:.8f}")
        print(f"  Max  |rel diff| : {max_rel:.6f}%")
        print(f"  Mean |rel diff| : {mean_rel:.6f}%")
        print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_to_excel(calibrated, comparisons, output_path):
    """
    Export calibrated Vol curves, comparisons, and a summary to Excel.

    Sheets produced:
      Calibrated_Vols   — all calibrated (expiry, vol) points
      Cmp_<currency>    — per-currency comparison table
      Summary           — aggregate difference metrics

    Parameters
    ----------
    calibrated : dict   Output of bootstrap_fx_from_json().
    comparisons : dict  Output of compare_with_riskflow().
    output_path : str   .xlsx file path.
    """
    try:
        import openpyxl  # noqa: F401
        engine = 'openpyxl'
    except ImportError:
        # Fall back to xlwt / xlrd / xlsxwriter if available
        engine = None

    with pd.ExcelWriter(output_path, engine=engine) as writer:
        # -- Sheet 1: all calibrated vols ----------------------------------
        rows = []
        for currency, p in calibrated.items():
            for exp, vol in p['Vol']:
                rows.append({
                    'FX_Pair': currency,
                    'Vol_Surface': p['_vol_surface_name'],
                    'Surface_Type': 'FXVol' if p['_is_fx'] else 'EquityPriceVol',
                    'Expiry': exp,
                    'Calibrated_Vol': vol,
                    'Variance': vol ** 2 * exp,
                    'Variance_Corrected': p['_was_corrected'],
                })
        if rows:
            pd.DataFrame(rows).to_excel(writer, sheet_name='Calibrated_Vols', index=False)

        # -- Sheet 2+: per-currency comparison -----------------------------
        for currency, df in comparisons.items():
            sheet = f'Cmp_{currency}'[:31]   # Excel 31-char sheet-name limit
            df.to_excel(writer, sheet_name=sheet, index=False)

        # -- Last sheet: summary -------------------------------------------
        summary = []
        for currency, df in comparisons.items():
            summary.append({
                'FX_Pair': currency,
                'N_Points': len(df),
                'Max_Abs_Diff': df['Abs_Diff'].abs().max(),
                'Mean_Abs_Diff': df['Abs_Diff'].abs().mean(),
                'Max_Rel_Diff_Pct': df['Rel_Diff_Pct'].abs().max(),
                'Mean_Rel_Diff_Pct': df['Rel_Diff_Pct'].abs().mean(),
            })
        if summary:
            pd.DataFrame(summary).to_excel(writer, sheet_name='Summary', index=False)

    print(f"\n  Exported to: {output_path}")


def export_to_csv(calibrated, comparisons, output_dir):
    """
    Export calibrated vols and comparisons to CSV files in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Calibrated vols
    rows = []
    for currency, p in calibrated.items():
        for exp, vol in p['Vol']:
            rows.append({
                'FX_Pair': currency,
                'Vol_Surface': p['_vol_surface_name'],
                'Expiry': exp,
                'Calibrated_Vol': vol,
                'Variance': vol ** 2 * exp,
            })
    if rows:
        path = os.path.join(output_dir, 'gbm_fx_calibrated_vols.csv')
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  Calibrated vols  → {path}")

    # Per-currency comparisons
    for currency, df in comparisons.items():
        safe_name = currency.replace('/', '-').replace('\\', '-')
        path = os.path.join(output_dir, f'gbm_fx_comparison_{safe_name}.csv')
        df.to_csv(path, index=False)
        print(f"  Comparison       → {path}")

    # Summary
    summary = []
    for currency, df in comparisons.items():
        summary.append({
            'FX_Pair': currency,
            'N_Points': len(df),
            'Max_Abs_Diff': df['Abs_Diff'].abs().max(),
            'Mean_Abs_Diff': df['Abs_Diff'].abs().mean(),
            'Max_Rel_Diff_Pct': df['Rel_Diff_Pct'].abs().max(),
            'Mean_Rel_Diff_Pct': df['Rel_Diff_Pct'].abs().mean(),
        })
    if summary:
        path = os.path.join(output_dir, 'gbm_fx_calibration_summary.csv')
        pd.DataFrame(summary).to_csv(path, index=False)
        print(f"  Summary          → {path}")


# ===========================================================================
#  SELF-TEST  (run without any JSON file)
# ===========================================================================

def _self_test():
    """
    Synthetic test cases that verify the calibration logic.

    Test 1 – Flat vol surface (no correction needed)
    Test 2 – Rising vol surface (mild correction expected)
    Test 3 – Strongly declining variance (correction mandatory)
    """
    print("\n" + "=" * 70)
    print("SELF-TEST: Synthetic ATM vol surfaces")
    print("=" * 70)

    def _run(label, expiries, atm_vols, expect_corrected):
        print(f"\n-- {label} ----------------------------------------------")
        avg, inst, corrected, details = correct_declining_variance(
            np.array(expiries), np.array(atm_vols))

        # Verify variance is non-decreasing after correction
        var_out = [d['var_actual'] for d in details]
        diffs = np.diff(var_out)
        monotone = np.all(diffs >= -1e-12)

        print(f"  {'Expiry':>6s}  {'Input ATM':>10s}  {'Avg Vol':>9s}  "
              f"{'Inst Vol':>9s}  {'Variance':>9s}  {'Clamped':>8s}")
        print("  " + "-" * 62)
        for d in details:
            flag = '*' if d['clamped'] else ' '
            print(f"  {d['expiry']:6.3f}  {d['raw_atm_vol']:10.6f}  "
                  f"{d['avg_vol']:9.6f}  {d['inst_vol']:9.6f}  "
                  f"{d['var_actual']:9.6f}  {flag:>8s}")

        # Cross-check: reconstruct variance from calibrated inst_vols via the
        # same piecewise-linear formula and confirm it equals var_out.
        dt = np.diff(np.concatenate([[0.0], expiries]))
        var_check = [var_out[0]]
        for i in range(1, len(expiries)):
            delta_t = dt[i] / 3.0
            increment = delta_t * (inst[i-1]**2 + inst[i-1]*inst[i] + inst[i]**2)
            var_check.append(var_check[-1] + increment)

        var_recon_ok = np.allclose(var_out, var_check, atol=1e-10)

        print()
        print(f"  Correction applied    : {'YES' if corrected else 'no'}"
              f"  (expected: {'YES' if expect_corrected else 'no'})")
        print(f"  Variance monotone     : {'PASS' if monotone else 'FAIL'}")
        print(f"  Variance reconstruction: {'PASS' if var_recon_ok else 'FAIL'}")
        assert corrected == expect_corrected, \
            f"Correction flag mismatch in '{label}'"
        assert monotone, f"Non-monotone variance in '{label}'"
        assert var_recon_ok, f"Variance reconstruction failed in '{label}'"

    # Test 1: flat vol  → no correction
    _run(
        "Test 1 – flat vol  (no correction)",
        expiries=[0.25, 0.5, 1.0, 2.0, 5.0],
        atm_vols=[0.10, 0.10, 0.10, 0.10, 0.10],
        expect_corrected=False,
    )

    # Test 2: gently rising vol → no correction
    _run(
        "Test 2 – rising vol  (no correction)",
        expiries=[0.25, 0.5, 1.0, 2.0, 5.0],
        atm_vols=[0.08, 0.10, 0.12, 0.13, 0.14],
        expect_corrected=False,
    )

    # Test 3: humped surface — peak at 1y then declining → correction needed
    _run(
        "Test 3 – humped/declining variance  (correction required)",
        expiries=[0.25, 0.50, 1.00, 2.00, 5.00],
        atm_vols=[0.10, 0.15, 0.20, 0.08, 0.05],
        expect_corrected=True,
    )

    # Test 4: single-point surface → trivially OK
    _run(
        "Test 4 – single expiry  (no correction possible)",
        expiries=[1.0],
        atm_vols=[0.12],
        expect_corrected=False,
    )

    print("\n" + "=" * 70)
    print("ALL SELF-TESTS PASSED")
    print("=" * 70)


# ===========================================================================
#  Entry point
# ===========================================================================

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] in ('--test', '-t', 'test'):
        _self_test()
        sys.exit(0)

    json_path = sys.argv[1]
    fx_name = sys.argv[2] if len(sys.argv) > 2 else None

    results = bootstrap_fx_from_json(json_path, fx_name=fx_name, verbose=True)

    if results:
        print_calibration_summary(results)
