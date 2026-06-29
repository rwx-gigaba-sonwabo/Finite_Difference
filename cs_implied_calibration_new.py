"""
==============================================================================
CS IMPLIED CALIBRATION (Q-Measure / Risk-Neutral)
==============================================================================

PURPOSE:
    Replicates what RiskFlow does when 'Model Configuration' maps
    ForwardPrice -> CSImpliedForwardPriceModel.

    In this mode, Sigma and Alpha are BOOTSTRAPPED from traded European
    commodity option prices using least-squares optimization.

    This corresponds to:
        - riskflow/bootstrappers.py   CSForwardPriceModelParameters.bootstrap()
        - riskflow/utils.py           black_european_option_price()

WHAT THIS SCRIPT DOES:
    1. Takes a set of European futures options with known market prices
    2. Defines the CS variance function V(sigma, alpha, T, S)
    3. Uses scipy.optimize.minimize to find (sigma, alpha) that best
       reproduce the observed option premiums
    4. Returns the calibrated parameters: {'Sigma': ..., 'Alpha': ...}
       (Drift is always 0 in the risk-neutral measure)

KEY DIFFERENCE FROM HISTORICAL:
    - Historical: fits to time series of past prices (what DID happen)
    - Implied: fits to current option prices (what the market EXPECTS)
    - The Sigma and Alpha values will generally be DIFFERENT numbers

THE VARIANCE FUNCTION:
    V(sigma, alpha, T, S) = sigma^2 * exp(-2*alpha*S) * B(2*alpha, T)

    where B(a, t) = (1 - exp(-a*t)) / a

    This gives the total log-variance of F(T, S) under the CS model:
    - T = option expiry time
    - S = forward settlement/delivery time
    - The effective Black vol is: sigma_black = sqrt(V / T)

PIPELINE (mirrors gbm_risk_neutral_calibration.py):
    1. bootstrap_from_json()        — reads JSON, bootstraps (Sigma, Alpha)
                                       for each commodity from market option prices
    2. extract_cs_params()          — reads STORED CSForwardPriceModelParameters
                                       from Price Factors in the same JSON
    3. compare_cs_params()          — scalar comparison of Sigma / Alpha:
                                       calibrated vs. stored, with abs and
                                       rel differences
    4. export_cs_results()          — writes Excel (multi-sheet) or CSVs
    5. run_cs_calibration()         — convenience wrapper for the full pipeline
==============================================================================
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats

# Constants — must match riskflow/utils.py exactly
EXCEL_OFFSET = pd.Timestamp('1899-12-30 00:00:00')

# Day count codes — riskflow/utils.py lines 43-45
DAYCOUNT_None   = -1
DAYCOUNT_ACT365 = 0
DAYCOUNT_ACT360 = 1


# ===========================================================================
# JSON helper  (self-contained — no cs_simulation import needed)
# ===========================================================================

def load_market_data(json_path):
    """
    Load RiskFlow JSON and unwrap the MarketData wrapper if present.

    Handles both:
      Format A (CVAMarketData): {"MarketData": {"Price Factors": {...}, ...}}
      Format B (flat):          {"Price Factors": {...}, ...}
    """
    with open(json_path, 'r') as fh:
        raw = json.load(fh)

    if "MarketData" in raw and isinstance(raw["MarketData"], dict):
        data = raw["MarketData"]
        print("  JSON format: wrapped (MarketData key found — unwrapped)")
    else:
        data = raw
        print("  JSON format: flat (no MarketData wrapper)")

    return data


# ===========================================================================
# Core model functions
# ===========================================================================

def get_day_count_accrual(reference_date, time_in_days, day_count_code):
    """
    Replicates riskflow/utils.py get_day_count_accrual() (line 2191).

    Converts a number of calendar days into a year fraction using the
    day count convention of the discount curve.

    Parameters
    ----------
    reference_date : pd.Timestamp
    time_in_days   : int or float
    day_count_code : int  — DAYCOUNT_ACT365 (0), DAYCOUNT_ACT360 (1), or
                            DAYCOUNT_None (-1)

    Returns
    -------
    float : Year fraction.
    """
    if day_count_code == DAYCOUNT_ACT360:
        return time_in_days / 360.0
    elif day_count_code == DAYCOUNT_ACT365:
        return time_in_days / 365.0
    elif day_count_code == DAYCOUNT_None:
        return time_in_days
    else:
        return time_in_days / 365.0


def black_european_option_price(F, X, r, vol, tenor, buyOrSell, callOrPut):
    """
    Replicates riskflow/utils.py black_european_option_price() (line 1165).

    Standard Black-76 formula for European options on forwards/futures.

    Parameters
    ----------
    F          : float  Forward price at delivery.
    X          : float  Strike price.
    r          : float  Risk-free rate (continuously compounded).
    vol        : float  Black implied volatility (annualised).
    tenor      : float  Time to option expiry in years.
    buyOrSell  : float  +1.0 for long, -1.0 for short.
    callOrPut  : float  +1.0 for call, -1.0 for put.

    Returns
    -------
    float : Option premium.
    """
    stddev = vol * np.sqrt(tenor)
    sign   = 1.0 if (F > 0.0 and X > 0.0) else -1.0
    d1     = (np.log(F / X) + 0.5 * stddev * stddev) / stddev
    d2     = d1 - stddev
    return buyOrSell * callOrPut * (
        F * scipy.stats.norm.cdf(callOrPut * sign * d1) -
        X * scipy.stats.norm.cdf(callOrPut * sign * d2)
    ) * np.exp(-r * tenor)


def cs_variance(sigma, alpha, T, S):
    """
    The Clewlow-Strickland variance function.

    Replicates the V() function inside bootstrappers.py (line 395).

    V = sigma^2 * exp(-2*alpha*S) * B(2*alpha, T)

    Parameters
    ----------
    sigma : float  Instantaneous OU process volatility.
    alpha : float  Mean reversion speed.
    T     : float  Time to option expiry (years).
    S     : float  Time to forward settlement/delivery (years).

    Returns
    -------
    float : Total log-variance of F(T, S).
    """
    def B(a, t):
        return (1.0 - np.exp(-a * t)) / a if a != 0 else t

    return sigma * sigma * np.exp(-2.0 * alpha * S) * B(2.0 * alpha, T)


# ===========================================================================
# Factor lookup builders
# ===========================================================================

def _curve_array(obj):
    """
    Extract a numpy float array from all known RiskFlow curve/surface formats.

    Format A: {'_type': 'Curve', 'array': [...]}            -- legacy flat JSON
    Format B: {'.Curve': {'meta': [...], 'data': [[...]]}}  -- CVAMarketData
    Format C: {'data': [[...]]}                              -- bare data wrapper
    Format D: {'array': [...]}                               -- bare array wrapper
    Format E: plain list/tuple of lists
    Format F: surface wrapper {'Surface': <any above>}
    """
    if obj is None:
        return np.array([], dtype=float).reshape(0, 2)

    # Unwrap Surface wrapper
    if isinstance(obj, dict) and 'Surface' in obj:
        obj = obj['Surface']

    if isinstance(obj, dict):
        # Format A
        if obj.get('_type') == 'Curve':
            return np.array(obj['array'], dtype=float)
        # Format B -- CVAMarketData nesting
        if '.Curve' in obj:
            inner = obj['.Curve']
            return np.array(inner.get('data', []), dtype=float)
        # Format C
        if 'data' in obj:
            return np.array(obj['data'], dtype=float)
        # Format D
        if 'array' in obj:
            return np.array(obj['array'], dtype=float)

    # Format E -- plain list/tuple
    if isinstance(obj, (list, tuple)):
        return np.array(obj, dtype=float)

    return np.array(obj, dtype=float)


def _parse_date(raw):
    """
    Parse a RiskFlow date field into a pd.Timestamp.

    Handles all formats found in CVAMarketData JSONs:
      - plain string:        "2026-02-26"
      - Excel serial int:    45678  (days since 1899-12-30)
      - dict with value key: {"_value": "2026-02-26"} or {"_value": 45678}
    """
    if raw is None:
        return None
    if isinstance(raw, pd.Timestamp):
        return raw
    # Unwrap dict  e.g. {"_value": ..., "_type": "DateOffset"}
    if isinstance(raw, dict):
        raw = (raw.get('_value') or raw.get('value') or
               raw.get('date')   or raw.get('Date') or
               next(iter(raw.values()), None))
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return EXCEL_OFFSET + pd.Timedelta(days=int(raw))
    if isinstance(raw, str):
        return pd.Timestamp(raw)
    return pd.Timestamp(str(raw))


def _build_forward_lookup(price_factors, energy_name):
    """
    Build a forward price interpolator from the Price Factors section.
    Replicates ForwardPrice.current_value(excel_date).
    Handles all CVAMarketData curve formats via _curve_array().
    """
    factor_key  = 'ForwardPrice.' + energy_name
    factor_data = price_factors[factor_key]

    arr     = _curve_array(factor_data.get('Curve', factor_data))
    tenors  = arr[:, 0]
    prices  = arr[:, 1]

    def lookup(excel_date):
        return float(np.interp(excel_date, tenors, prices))

    return lookup, factor_data


def _build_discount_lookup(price_factors, discount_rate_name):
    """
    Build a discount rate interpolator from the Price Factors section.
    Replicates InterestRate.current_value(year_fraction).
    Handles all CVAMarketData curve formats via _curve_array().
    """
    factor_key  = 'InterestRate.' + discount_rate_name
    factor_data = price_factors[factor_key]

    arr     = _curve_array(factor_data.get('Curve', factor_data))
    tenors  = arr[:, 0]
    rates   = arr[:, 1]

    dc_str = factor_data.get('Day_Count', 'ACT_365')
    if 'ACT_360' in str(dc_str).upper().replace(' ', '_'):
        day_count_code = DAYCOUNT_ACT360
    elif 'ACT_365' in str(dc_str).upper().replace(' ', '_'):
        day_count_code = DAYCOUNT_ACT365
    else:
        day_count_code = DAYCOUNT_ACT365

    def lookup(year_fraction):
        return float(np.interp(year_fraction, tenors, rates))

    return lookup, day_count_code


def _build_vol_surface_lookup(price_factors, vol_name):
    """
    Build a volatility surface interpolator from the Price Factors section.
    Replicates ForwardPriceVol.current_value([[t, d, moneyness]]).
    Handles all CVAMarketData surface formats via _curve_array().
    """
    factor_key  = 'ForwardPriceVol.' + vol_name
    factor_data = price_factors[factor_key]

    # Surface may live under a 'Surface' key or directly on factor_data
    surface_obj = factor_data.get('Surface', factor_data)
    arr         = _curve_array(surface_obj)

    arr              = np.array(arr, dtype=float)
    unique_moneyness = np.unique(arr[:, 2])

    def lookup(expiry_yf, delivery_yf, moneyness=1.0):
        m_idx     = np.argmin(np.abs(unique_moneyness - moneyness))
        m_val     = unique_moneyness[m_idx]
        slice_arr = arr[arr[:, 2] == m_val]

        if len(slice_arr) == 0:
            slice_arr = arr
        if len(slice_arr) == 1:
            return float(slice_arr[0, 3])

        unique_expiries = np.unique(slice_arr[:, 1])

        if len(unique_expiries) == 1:
            tenor_data = slice_arr[slice_arr[:, 1] == unique_expiries[0]]
            return float(np.interp(delivery_yf, tenor_data[:, 0], tenor_data[:, 3]))

        vol_at_expiries = []
        for exp in unique_expiries:
            exp_slice        = slice_arr[slice_arr[:, 1] == exp]
            vol_at_delivery  = np.interp(delivery_yf, exp_slice[:, 0], exp_slice[:, 3])
            vol_at_expiries.append(vol_at_delivery)

        return float(np.interp(expiry_yf, unique_expiries, vol_at_expiries))

    return lookup


# ===========================================================================
# STEP 1: bootstrap_from_json
# Reads Market Prices, locates CSForwardPriceModelPrices, runs optimizer
# ===========================================================================

def bootstrap_from_json(json_path, commodity_name=None, verbose=True):
    """
    Full CS implied calibration pipeline from a RiskFlow JSON file.

    Replicates CSForwardPriceModelParameters.bootstrap() at
    bootstrappers.py lines 387-489.

    Parameters
    ----------
    json_path      : str   Path to RiskFlow JSON market data file.
    commodity_name : str   Optional filter, e.g. 'BRENT.OIL'. None = all.
    verbose        : bool

    Returns
    -------
    dict mapping commodity name to:
        {
          'Sigma'          : float,
          'Alpha'          : float,
          '_options'       : list of option dicts (with per-option fit info),
          '_result'        : scipy OptimizeResult,
          '_commodity'     : str,
        }
    """
    if verbose:
        print("=" * 70)
        print("CS IMPLIED CALIBRATION — BOOTSTRAP FROM JSON")
        print("=" * 70)
        print(f"  Loading: {json_path}")

    market_data   = load_market_data(json_path)
    price_factors = market_data.get('Price Factors', {})
    market_prices = market_data.get('Market Prices', {})
    sys_params    = market_data.get('System Parameters', {})

    # Resolve Base_Date — uses module-level _parse_date()
    base_date = _parse_date(sys_params.get('Base_Date'))
    if base_date is None:
        val_config = market_data.get('Valuation Configuration', {})
        if isinstance(val_config, dict):
            base_date = _parse_date(
                val_config.get('Base_Date', val_config.get('Run_Date'))
            )

    if base_date is None:
        raise ValueError(
            "Cannot find Base_Date in System Parameters or "
            "Valuation Configuration"
        )

    vol_delta = sys_params.get('Volatility_Delta', 0.0)

    if verbose:
        print(f"  Base date          : {base_date.date()}")
        print(f"  Volatility delta   : {vol_delta}")
        print(f"  Market Prices count: {len(market_prices)}")

    results = {}

    for market_price_name, implied_params in market_prices.items():
        rate_parts  = tuple(market_price_name.split('.'))
        factor_type = rate_parts[0]

        if factor_type != 'CSForwardPriceModelPrices':
            continue

        commodity = '.'.join(rate_parts[1:])
        if (commodity_name is not None and
                commodity.upper() != commodity_name.upper()):
            continue

        if verbose:
            print(f"\n  {'='*60}")
            print(f"  Bootstrapping: {commodity}")
            print(f"  {'='*60}")

        instrument    = implied_params.get('instrument', implied_params)
        vol_name      = instrument['Forward_Volatility']
        energy_name   = instrument['Energy']
        discount_name = instrument['Discount_Rate']
        quote_type    = instrument.get('Quote_Type', 'Implied_Volatility')

        if verbose:
            print(f"  Forward Volatility : {vol_name}")
            print(f"  Energy (Forward)   : {energy_name}")
            print(f"  Discount Rate      : {discount_name}")
            print(f"  Quote Type         : {quote_type}")

        try:
            forward_lookup,  forward_data   = _build_forward_lookup(
                                                  price_factors, energy_name)
            discount_lookup, day_count_code = _build_discount_lookup(
                                                  price_factors, discount_name)
            vol_lookup = _build_vol_surface_lookup(price_factors, vol_name)
        except KeyError as e:
            print(f"  ERROR: Cannot find factor {e} in Price Factors "
                  f"— skipping {commodity}")
            continue

        options_list = instrument.get('Energy_Futures_Options', [])
        if verbose:
            print(f"  Options to calibrate: {len(options_list)}")
            print(f"\n  {'Expiry':>12s}  {'Settle':>12s}  {'T':>6s}  "
                  f"{'S':>6s}  {'Forward':>8s}  {'Strike':>8s}  "
                  f"{'r':>6s}  {'MktVol':>8s}  {'Premium':>10s}")
            print("  " + "-" * 94)

        for option in options_list:
            expiry_date = _parse_date(option['Expiry_Date'])
            t = get_day_count_accrual(
                    base_date,
                    (expiry_date - base_date).days,
                    day_count_code)

            settlement_date = _parse_date(option['Settlement_Date'])
            d = get_day_count_accrual(
                    base_date,
                    (settlement_date - base_date).days,
                    day_count_code)

            expiry_excel     = (expiry_date     - EXCEL_OFFSET).days
            settlement_excel = (settlement_date - EXCEL_OFFSET).days

            forward_at_exp    = forward_lookup(expiry_excel)
            forward_at_settle = forward_lookup(settlement_excel)
            r                 = discount_lookup(t)

            if quote_type == 'Implied_Volatility':
                quoted = option.get('Quoted_Market_Value')
                sigma  = quoted if quoted else vol_lookup(t, d, 1.0)
                sigma += vol_delta
            else:
                if verbose:
                    print(f"  WARNING: Quote type '{quote_type}' not "
                          f"supported — skipping option")
                continue

            strike = option.get('Strike') or forward_at_exp

            option['Forward'] = forward_at_settle
            option['Strike']  = strike
            option['r']       = r
            option['S']       = d
            option['T']       = t
            option['sigma']   = sigma

            cp    = 1.0 if option.get('Option_Type', 'Call') == 'Call' else -1.0
            units = option.get('Units', 1.0)
            option['Premium'] = black_european_option_price(
                option['Forward'], option['Strike'], r, sigma, t,
                units, cp)
            option['Units']  = units
            option.setdefault('Weight', 1.0)

            if verbose:
                exp_str = str(expiry_date.date())
                set_str = str(settlement_date.date())
                print(f"  {exp_str:>12s}  {set_str:>12s}  {t:6.3f}  "
                      f"{d:6.3f}  {forward_at_settle:8.4f}  "
                      f"{strike:8.4f}  {r:6.4f}  "
                      f"{sigma:8.4f}  {option['Premium']:10.4f}")

        fitted, opt_result = _run_optimizer(options_list, verbose=verbose)

        # Attach per-option CS model values for downstream comparison
        sigma_fit = fitted['Sigma']
        alpha_fit = fitted['Alpha']
        for option in options_list:
            cp           = 1.0 if option['Option_Type'] == 'Call' else -1.0
            discount     = np.exp(-option['r'] * option['T'])
            total_var    = cs_variance(sigma_fit, alpha_fit,
                                       option['T'], option['S'])
            option['cs_vol']     = np.sqrt(total_var / option['T'])
            option['cs_premium'] = black_european_option_price(
                option['Forward'], option['Strike'], 0.0,
                np.sqrt(max(total_var, 1e-12)),
                1.0, option['Units'], cp) * discount
            option['cs_error']   = (option['cs_premium'] -
                                    option['Premium']) ** 2

        results[commodity] = {
            'Sigma'     : sigma_fit,
            'Alpha'     : alpha_fit,
            '_options'  : options_list,
            '_result'   : opt_result,
            '_commodity': commodity,
        }

    if not results and verbose:
        print("\n  WARNING: No CSForwardPriceModelPrices entries found.")
        mp_types = set(k.split('.')[0] for k in market_prices)
        print(f"  Market Prices entry types found: {mp_types}")

    if verbose:
        print("=" * 70)
        print(f"  Calibrated {len(results)} commodity(ies): "
              f"{', '.join(results.keys()) or 'none'}")
        print("=" * 70)

    return results


def _run_optimizer(options, verbose=True):
    """
    Internal: run scipy minimiser to find (Sigma, Alpha).
    Replicates bootstrappers.py lines 465-468.
    """

    def calc_error(x, options):
        sigma, alpha = x
        error        = 0.0
        for option in options:
            discount      = np.exp(-option['r'] * option['T'])
            cp            = 1.0 if option['Option_Type'] == 'Call' else -1.0
            total_var     = cs_variance(sigma, alpha,
                                        option['T'], option['S'])
            total_stddev  = np.sqrt(max(total_var, 1e-12))
            model_premium = black_european_option_price(
                option['Forward'], option['Strike'], 0.0, total_stddev,
                1.0, option['Units'], cp
            ) * discount
            error += option['Weight'] * (option['Premium'] - model_premium) ** 2
        return error

    result = scipy.optimize.minimize(
        calc_error, (0.5, 0.1),
        args=(options,),
        bounds=[(0.001, 2.5), (-1, 2.0)]
    )

    fitted_sigma = result.x[0]
    fitted_alpha = result.x[1]

    if verbose:
        print()
        print("  Calibration result:")
        print(f"    Sigma (OU instantaneous vol) : {fitted_sigma:.6f}")
        print(f"    Alpha (Mean reversion speed) : {fitted_alpha:.6f}")
        print(f"    Drift                        : 0.0 (risk-neutral)")
        print(f"    Optimiser success            : {result.success}")
        print(f"    Final objective value        : {result.fun:.10f}")

        print(f"\n  Per-option fit quality:")
        print(f"  {'T':>6s}  {'S':>6s}  {'Strike':>8s}  {'MktVol':>8s}  "
              f"{'CS Vol':>8s}  {'MktPrem':>10s}  {'CSPrem':>10s}  "
              f"{'SqErr':>10s}")
        print("  " + "-" * 78)

        for option in options:
            cp            = 1.0 if option['Option_Type'] == 'Call' else -1.0
            discount      = np.exp(-option['r'] * option['T'])
            total_var     = cs_variance(fitted_sigma, fitted_alpha,
                                        option['T'], option['S'])
            cs_vol        = np.sqrt(total_var / option['T'])
            cs_premium    = black_european_option_price(
                option['Forward'], option['Strike'], 0.0,
                np.sqrt(max(total_var, 1e-12)), 1.0, option['Units'], cp
            ) * discount
            sq_err = (cs_premium - option['Premium']) ** 2
            mkt_vol = option.get('sigma', 0.0)
            print(f"  {option['T']:6.3f}  {option['S']:6.3f}  "
                  f"{option['Strike']:8.2f}  {mkt_vol:8.4f}  "
                  f"{cs_vol:8.4f}  {option['Premium']:10.4f}  "
                  f"{cs_premium:10.4f}  {sq_err:10.6f}")

    return {'Sigma': fitted_sigma, 'Alpha': fitted_alpha}, result


# ===========================================================================
# STEP 2: extract_cs_params
# Reads STORED CSForwardPriceModelParameters from Price Factors
# ===========================================================================

def extract_cs_params(json_path, commodity_names=None, verbose=True):
    """
    Extract stored CSForwardPriceModelParameters from Price Factors.

    These are the PRODUCTION values already stored in the JSON — the
    numbers RiskFlow would have written from a previous bootstrap run.

    Parameters
    ----------
    json_path        : str        Path to RiskFlow JSON.
    commodity_names  : str|list   Optional filter. None = extract all.
    verbose          : bool

    Returns
    -------
    dict mapping commodity name to:
        {
          'Sigma' : float,
          'Alpha' : float,
          'Drift' : float,   (0.0 for implied)
        }
    """
    market_data   = load_market_data(json_path)
    price_factors = (
        market_data.get('Price Factors')
        or market_data.get('PriceFactors')
        or {}
    )

    CS_PREFIX = 'CSForwardPriceModelParameters.'

    if commodity_names is None:
        commodity_names = [
            k[len(CS_PREFIX):]
            for k in price_factors
            if k.startswith(CS_PREFIX)
        ]
        if not commodity_names and verbose:
            print("WARNING: No CSForwardPriceModelParameters found.")
            cs_any = [k for k in price_factors if 'CS' in k]
            print(f"  CS-related keys found: {cs_any[:10]}")
    elif isinstance(commodity_names, str):
        commodity_names = [commodity_names]

    results = {}

    for name in commodity_names:
        full_key = (CS_PREFIX + name
                    if not name.startswith(CS_PREFIX) else name)
        clean    = full_key[len(CS_PREFIX):]

        factor_data = price_factors.get(full_key)
        if factor_data is None:
            if verbose:
                print(f"WARNING: '{full_key}' not found in Price Factors.")
            continue

        sigma = factor_data.get('Sigma')
        alpha = factor_data.get('Alpha')
        drift = factor_data.get('Drift', 0.0)

        if sigma is None or alpha is None:
            if verbose:
                print(f"WARNING: Missing Sigma or Alpha in {full_key}")
            continue

        results[clean] = {
            'Sigma': float(sigma),
            'Alpha': float(alpha),
            'Drift': float(drift) if drift is not None else 0.0,
        }

        if verbose:
            print(f"  Extracted '{full_key}': "
                  f"Sigma={float(sigma):.6f}  "
                  f"Alpha={float(alpha):.6f}  "
                  f"Drift={float(drift) if drift else 0.0:.6f}")

    return results


# ===========================================================================
# STEP 3: compare_cs_params
# Scalar comparison: calibrated vs stored Sigma / Alpha
# ===========================================================================

def compare_cs_params(calibrated, extracted, verbose=True):
    """
    Compare bootstrap-calibrated CS parameters against stored values.

    Unlike GBM (which has a vol curve to compare point-by-point), CS
    produces two scalar parameters (Sigma, Alpha) per commodity. The
    comparison DataFrame has one row per commodity.

    Parameters
    ----------
    calibrated : dict   output of bootstrap_from_json()
    extracted  : dict   output of extract_cs_params()
    verbose    : bool

    Returns
    -------
    pd.DataFrame with columns:
        Commodity,
        Stored_Sigma,  Calibrated_Sigma,  Abs_Diff_Sigma,  Rel_Diff_Sigma_Pct,
        Stored_Alpha,  Calibrated_Alpha,  Abs_Diff_Alpha,  Rel_Diff_Alpha_Pct,
        Stored_Drift,
        Optimizer_Success,  Final_Obj_Value,
        N_Options
    """
    rows = []
    all_commodities = sorted(
        set(calibrated.keys()) | set(extracted.keys())
    )

    for commodity in all_commodities:
        if commodity not in calibrated:
            if verbose:
                print(f"  {commodity}: no calibrated result — skipping")
            continue
        if commodity not in extracted:
            if verbose:
                print(f"  {commodity}: not found in stored params — "
                      f"comparison will have NaN for stored values")

        calib = calibrated[commodity]
        ext   = extracted.get(commodity, {})

        stored_sigma = ext.get('Sigma', np.nan)
        stored_alpha = ext.get('Alpha', np.nan)
        stored_drift = ext.get('Drift', np.nan)

        cal_sigma = calib['Sigma']
        cal_alpha = calib['Alpha']

        abs_sigma = cal_sigma - stored_sigma
        abs_alpha = cal_alpha - stored_alpha

        with np.errstate(invalid='ignore', divide='ignore'):
            rel_sigma = (100.0 * abs_sigma / stored_sigma
                         if abs(stored_sigma) > 1e-12 else np.nan)
            rel_alpha = (100.0 * abs_alpha / stored_alpha
                         if abs(stored_alpha) > 1e-12 else np.nan)

        opt_result  = calib.get('_result')
        obj_val     = float(opt_result.fun)    if opt_result else np.nan
        opt_success = bool(opt_result.success) if opt_result else np.nan
        n_options   = len(calib.get('_options', []))

        rows.append({
            'Commodity'           : commodity,
            'Stored_Sigma'        : round(stored_sigma, 8),
            'Calibrated_Sigma'    : round(cal_sigma,    8),
            'Abs_Diff_Sigma'      : round(abs_sigma,    8),
            'Rel_Diff_Sigma_Pct'  : round(rel_sigma,    4),
            'Stored_Alpha'        : round(stored_alpha, 8),
            'Calibrated_Alpha'    : round(cal_alpha,    8),
            'Abs_Diff_Alpha'      : round(abs_alpha,    8),
            'Rel_Diff_Alpha_Pct'  : round(rel_alpha,    4),
            'Stored_Drift'        : round(stored_drift, 8) if not np.isnan(stored_drift) else np.nan,
            'Optimizer_Success'   : opt_success,
            'Final_Obj_Value'     : round(obj_val, 10)     if not np.isnan(obj_val) else np.nan,
            'N_Options'           : n_options,
        })

    df = pd.DataFrame(rows)

    if verbose and not df.empty:
        print(f"\n{'='*76}")
        print("  CS Implied Calibration — Parameter Comparison")
        print(f"{'='*76}")
        cols_to_show = [
            'Commodity',
            'Stored_Sigma', 'Calibrated_Sigma',
            'Abs_Diff_Sigma', 'Rel_Diff_Sigma_Pct',
            'Stored_Alpha', 'Calibrated_Alpha',
            'Abs_Diff_Alpha', 'Rel_Diff_Alpha_Pct',
        ]
        print(df[cols_to_show].to_string(index=False))

        # Flag large relative differences
        sigma_breach = df[
            df['Rel_Diff_Sigma_Pct'].apply(
                lambda x: isinstance(x, float) and abs(x) > 1.0)
        ]
        alpha_breach = df[
            df['Rel_Diff_Alpha_Pct'].apply(
                lambda x: isinstance(x, float) and abs(x) > 1.0)
        ]

        if sigma_breach.empty and alpha_breach.empty:
            print("\n  All parameters within 1% tolerance.")
        else:
            if not sigma_breach.empty:
                print(f"\n  {len(sigma_breach)} Sigma point(s) > 1% diff:")
                print(sigma_breach[
                    ['Commodity', 'Stored_Sigma',
                     'Calibrated_Sigma', 'Rel_Diff_Sigma_Pct']
                ].to_string(index=False))
            if not alpha_breach.empty:
                print(f"\n  {len(alpha_breach)} Alpha point(s) > 1% diff:")
                print(alpha_breach[
                    ['Commodity', 'Stored_Alpha',
                     'Calibrated_Alpha', 'Rel_Diff_Alpha_Pct']
                ].to_string(index=False))

        if len(df) > 0:
            print(f"\n  Max |abs diff Sigma| : "
                  f"{df['Abs_Diff_Sigma'].abs().max():.8f}")
            print(f"  Max |abs diff Alpha| : "
                  f"{df['Abs_Diff_Alpha'].abs().max():.8f}")

    return df


# ===========================================================================
# STEP 4: export_cs_results
# Excel (multi-sheet) or CSV fallback
# ===========================================================================

def export_cs_results(calibrated, comparison_df, output_path, verbose=True):
    """
    Export calibrated parameters, per-option fit details, and comparison.

    Excel sheets produced:
        Comparison        — commodity-level Sigma/Alpha comparison
        Calibrated_Params — calibrated scalar parameters
        Option_Fit        — per-option fit details across all commodities
        Summary           — aggregate diff metrics

    Falls back to individual CSVs if openpyxl is unavailable.

    Parameters
    ----------
    calibrated     : dict         output of bootstrap_from_json()
    comparison_df  : pd.DataFrame output of compare_cs_params()
    output_path    : str          target .xlsx or .csv path
    verbose        : bool

    Returns
    -------
    (calibrated_params_df, option_fit_df, summary_df)
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # ── Calibrated parameters (scalar) ──────────────────────────────────────
    cal_rows = []
    for commodity, p in calibrated.items():
        cal_rows.append({
            'Commodity'         : commodity,
            'Calibrated_Sigma'  : round(p['Sigma'], 8),
            'Calibrated_Alpha'  : round(p['Alpha'], 8),
            'Drift'             : 0.0,
            'N_Options'         : len(p.get('_options', [])),
            'Optimizer_Success' : bool(p['_result'].success)
                                  if p.get('_result') else '',
            'Final_Obj_Value'   : round(float(p['_result'].fun), 10)
                                  if p.get('_result') else '',
        })
    cal_df = pd.DataFrame(cal_rows)

    # ── Per-option fit details ───────────────────────────────────────────────
    opt_rows = []
    for commodity, p in calibrated.items():
        for option in p.get('_options', []):
            opt_rows.append({
                'Commodity'    : commodity,
                'T_Expiry'     : round(option.get('T', np.nan),           6),
                'S_Settle'     : round(option.get('S', np.nan),           6),
                'Forward'      : round(option.get('Forward', np.nan),     4),
                'Strike'       : round(option.get('Strike', np.nan),      4),
                'Discount_Rate': round(option.get('r', np.nan),           6),
                'Market_Vol'   : round(option.get('sigma', np.nan),       6),
                'CS_Vol'       : round(option.get('cs_vol', np.nan),      6),
                'Market_Prem'  : round(option.get('Premium', np.nan),     6),
                'CS_Prem'      : round(option.get('cs_premium', np.nan),  6),
                'Sq_Error'     : round(option.get('cs_error', np.nan),    10),
                'Option_Type'  : option.get('Option_Type', 'Call'),
                'Units'        : option.get('Units', 1.0),
            })
    opt_df = pd.DataFrame(opt_rows)

    # ── Summary ──────────────────────────────────────────────────────────────
    summary_rows = []
    if not comparison_df.empty:
        for _, row in comparison_df.iterrows():
            summary_rows.append({
                'Commodity'              : row['Commodity'],
                'Stored_Sigma'           : row['Stored_Sigma'],
                'Calibrated_Sigma'       : row['Calibrated_Sigma'],
                'Abs_Diff_Sigma'         : row['Abs_Diff_Sigma'],
                'Rel_Diff_Sigma_Pct'     : row['Rel_Diff_Sigma_Pct'],
                'Stored_Alpha'           : row['Stored_Alpha'],
                'Calibrated_Alpha'       : row['Calibrated_Alpha'],
                'Abs_Diff_Alpha'         : row['Abs_Diff_Alpha'],
                'Rel_Diff_Alpha_Pct'     : row['Rel_Diff_Alpha_Pct'],
                'Sigma_Exceedance_1pct'  : bool(
                    isinstance(row['Rel_Diff_Sigma_Pct'], float)
                    and abs(row['Rel_Diff_Sigma_Pct']) > 1.0),
                'Alpha_Exceedance_1pct'  : bool(
                    isinstance(row['Rel_Diff_Alpha_Pct'], float)
                    and abs(row['Rel_Diff_Alpha_Pct']) > 1.0),
                'N_Options'              : row['N_Options'],
                'Optimizer_Success'      : row['Optimizer_Success'],
                'Final_Obj_Value'        : row['Final_Obj_Value'],
            })
    summary_df = pd.DataFrame(summary_rows)

    # ── Write ─────────────────────────────────────────────────────────────────
    try:
        import openpyxl  # noqa
        xlsx_path = (output_path if output_path.endswith('.xlsx')
                     else output_path.replace('.csv', '.xlsx'))

        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            if not comparison_df.empty:
                comparison_df.to_excel(
                    writer, sheet_name='Comparison', index=False)
            if not cal_df.empty:
                cal_df.to_excel(
                    writer, sheet_name='Calibrated_Params', index=False)
            if not opt_df.empty:
                opt_df.to_excel(
                    writer, sheet_name='Option_Fit', index=False)
            if not summary_df.empty:
                summary_df.to_excel(
                    writer, sheet_name='Summary', index=False)

        if verbose:
            print(f"\n  Saved Excel: {xlsx_path}")
            print(f"    Sheets: Comparison | Calibrated_Params | "
                  f"Option_Fit | Summary")

    except ImportError:
        base = output_path.replace('.xlsx', '').replace('.csv', '')
        if not comparison_df.empty:
            comparison_df.to_csv(f'{base}_comparison.csv', index=False)
        if not cal_df.empty:
            cal_df.to_csv(f'{base}_calibrated_params.csv', index=False)
        if not opt_df.empty:
            opt_df.to_csv(f'{base}_option_fit.csv', index=False)
        if not summary_df.empty:
            summary_df.to_csv(f'{base}_summary.csv', index=False)
        if verbose:
            print(f"\n  Saved CSVs to: {base}_*.csv")

    return cal_df, opt_df, summary_df


# ===========================================================================
# STEP 5: run_cs_calibration
# Full pipeline convenience wrapper
# ===========================================================================

def run_cs_calibration(json_path, output_path,
                        commodity_name=None, verbose=True):
    """
    Run the complete CS implied calibration and comparison pipeline.

    1. bootstrap_from_json()  — bootstrap (Sigma, Alpha) from market prices
    2. extract_cs_params()    — extract stored production parameters
    3. compare_cs_params()    — scalar comparison of Sigma and Alpha
    4. export_cs_results()    — export to Excel / CSV

    Parameters
    ----------
    json_path      : str   RiskFlow JSON market data file
    output_path    : str   target .xlsx or .csv path
    commodity_name : str   optional commodity filter e.g. 'BRENT.OIL'
    verbose        : bool

    Returns
    -------
    (calibrated, extracted, comparison_df, cal_df, opt_df, summary_df)
    """
    # Step 1
    calibrated = bootstrap_from_json(
        json_path, commodity_name=commodity_name, verbose=verbose
    )

    # Step 2
    extracted = extract_cs_params(
        json_path,
        commodity_names=list(calibrated.keys()) or commodity_name,
        verbose=verbose
    )

    # Step 3
    comparison_df = compare_cs_params(
        calibrated, extracted, verbose=verbose
    )

    # Step 4
    cal_df, opt_df, summary_df = export_cs_results(
        calibrated, comparison_df, output_path, verbose=verbose
    )

    return calibrated, extracted, comparison_df, cal_df, opt_df, summary_df


# ===========================================================================
# Legacy convenience: calibrate_implied (used by __main__ self-test)
# ===========================================================================

def calibrate_implied(options):
    """
    Standalone calibrator (no JSON). Used for the self-test below.

    Parameters
    ----------
    options : list of dict  — same schema as bootstrap_from_json output

    Returns
    -------
    dict with keys 'Sigma', 'Alpha'
    """
    fitted, _ = _run_optimizer(options, verbose=True)
    return fitted


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == '__main__':

    # =========================================================================
    # MODE 1: Full pipeline from a RiskFlow JSON
    # =========================================================================
    # Runs: bootstrap → extract stored → compare → export
    #
    # Usage (command line):
    #   python cs_implied_calibration.py path/to/market_data.json \
    #          path/to/output.xlsx [COMMODITY.NAME]
    #
    # Or set the constants below and run without arguments.
    # =========================================================================

    # ── Set your inputs here ─────────────────────────────────────────────────
    JSON_PATH      = r"C:\XVA_engine\MarketData.json"
    OUTPUT_PATH    = r"C:\XVA_engine\outputs\cs_implied_calibration.xlsx"

    # Set to a specific commodity to calibrate one only,
    # or None to calibrate ALL CSForwardPriceModelPrices found in the JSON.
    COMMODITY_NAME = None   # e.g. "BRENT.OIL" or None for all

    VERBOSE        = True
    # ─────────────────────────────────────────────────────────────────────────

    if len(sys.argv) > 1:
        JSON_PATH   = sys.argv[1]
        OUTPUT_PATH = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_PATH
        COMMODITY_NAME = sys.argv[3] if len(sys.argv) > 3 else None

    if os.path.exists(JSON_PATH):
        # Full pipeline
        (calibrated, extracted, comparison_df,
         cal_df, opt_df, summary_df) = run_cs_calibration(
            json_path      = JSON_PATH,
            output_path    = OUTPUT_PATH,
            commodity_name = COMMODITY_NAME,
            verbose        = VERBOSE,
        )
        sys.exit(0)

    # =========================================================================
    # MODE 2: Synthetic self-test (no JSON needed)
    # =========================================================================
    # When JSON_PATH does not exist, generates synthetic option data from
    # known CS parameters and verifies the optimizer can recover them.
    # =========================================================================

    print("\n" + "=" * 70)
    print("SELF-TEST: Synthetic data with known parameters")
    print("=" * 70)

    true_sigma = 0.35
    true_alpha = 1.2
    r          = 0.05

    option_specs = [
        {'T': 0.25, 'S': 0.25, 'Forward': 85.0, 'Strike': 85.0, 'Option_Type': 'Call'},
        {'T': 0.25, 'S': 0.50, 'Forward': 84.0, 'Strike': 84.0, 'Option_Type': 'Call'},
        {'T': 0.25, 'S': 1.00, 'Forward': 82.0, 'Strike': 82.0, 'Option_Type': 'Call'},
        {'T': 0.50, 'S': 0.50, 'Forward': 84.0, 'Strike': 84.0, 'Option_Type': 'Call'},
        {'T': 0.50, 'S': 1.00, 'Forward': 82.0, 'Strike': 82.0, 'Option_Type': 'Call'},
        {'T': 1.00, 'S': 1.00, 'Forward': 82.0, 'Strike': 82.0, 'Option_Type': 'Call'},
    ]

    options = []
    for spec in option_specs:
        total_var    = cs_variance(true_sigma, true_alpha, spec['T'], spec['S'])
        total_stddev = np.sqrt(total_var)
        cp           = 1.0
        discount     = np.exp(-r * spec['T'])

        premium  = black_european_option_price(
            spec['Forward'], spec['Strike'], 0.0, total_stddev,
            1.0, 1.0, cp
        ) * discount
        mkt_vol  = np.sqrt(total_var / spec['T'])

        options.append({
            'Forward'    : spec['Forward'],
            'Strike'     : spec['Strike'],
            'T'          : spec['T'],
            'S'          : spec['S'],
            'r'          : r,
            'Premium'    : premium,
            'Units'      : 1.0,
            'Option_Type': spec['Option_Type'],
            'Weight'     : 1.0,
            'sigma'      : mkt_vol,
        })

    print(f"\nTrue parameters: Sigma={true_sigma}, Alpha={true_alpha}")
    print(f"Number of options: {len(options)}\n")

    params = calibrate_implied(options)

    print(f"\nRecovery check:")
    print(f"  True Sigma = {true_sigma:.4f},  Fitted Sigma = {params['Sigma']:.4f}")
    print(f"  True Alpha = {true_alpha:.4f},  Fitted Alpha = {params['Alpha']:.4f}")

    # Demonstrate comparison / export with synthetic data
    synthetic_calibrated = {
        'BRENT.OIL': {
            'Sigma'     : params['Sigma'],
            'Alpha'     : params['Alpha'],
            '_options'  : options,
            '_result'   : None,
            '_commodity': 'BRENT.OIL',
        }
    }
    synthetic_extracted = {
        'BRENT.OIL': {
            'Sigma': true_sigma,
            'Alpha': true_alpha,
            'Drift': 0.0,
        }
    }
    comparison_df = compare_cs_params(
        synthetic_calibrated, synthetic_extracted, verbose=True
    )
    print("\nComparison DataFrame:")
    print(comparison_df.to_string(index=False))
