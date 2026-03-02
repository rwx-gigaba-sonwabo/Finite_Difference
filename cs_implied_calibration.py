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
==============================================================================
"""

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats

# Constants — must match riskflow/utils.py exactly
EXCEL_OFFSET = pd.Timestamp('1899-12-30 00:00:00')

# Day count codes — riskflow/utils.py lines 43-45
DAYCOUNT_None = -1
DAYCOUNT_ACT365 = 0
DAYCOUNT_ACT360 = 1


def get_day_count_accrual(reference_date, time_in_days, day_count_code):
    """
    Replicates riskflow/utils.py get_day_count_accrual() (line 2191).

    Converts a number of calendar days into a year fraction using the
    day count convention of the discount curve.

    In the bootstrapper (line 438-441), this is called as:
        t = discount.get_day_count_accrual(Base_Date, (Expiry_Date - Base_Date).days)

    The discount curve's get_day_count() method returns the day count code
    (ACT/365, ACT/360, etc.) which determines the divisor.

    Parameters
    ----------
    reference_date : pd.Timestamp
        The base/reference date.
    time_in_days : int or float
        Number of calendar days from reference_date.
    day_count_code : int
        DAYCOUNT_ACT365 (0), DAYCOUNT_ACT360 (1), or DAYCOUNT_None (-1).

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
        # Default fallback — ACT/365
        return time_in_days / 365.0


def _build_forward_lookup(price_factors, energy_name):
    """
    Build a forward price interpolator from the Price Factors section.

    Replicates how riskflow/riskfactors.py ForwardPrice (line 734) loads its
    curve and how forward.current_value(excel_date) returns the interpolated price.

    In the bootstrapper (lines 444-445):
        forward_at_exp = forward.current_value(expiry_excel)
        forward_at_settle = forward.current_value(settlement_excel)

    Parameters
    ----------
    price_factors : dict
        The 'Price Factors' section from the market data JSON.
    energy_name : str
        The energy factor name, e.g. 'BRENT.OIL'.

    Returns
    -------
    callable : Takes an excel day number, returns the interpolated forward price.
    dict : The raw factor data (for extracting currency, day count, etc.).
    """
    factor_key = 'ForwardPrice.' + energy_name
    factor_data = price_factors[factor_key]

    # Extract the curve array — same as Factor1D.__init__
    curve = factor_data['Curve']
    if isinstance(curve, dict) and curve.get('_type') == 'Curve':
        arr = curve['array']
    else:
        arr = np.array(sorted(curve))

    tenors = arr[:, 0]   # excel day numbers
    prices = arr[:, 1]   # forward prices

    def lookup(excel_date):
        return float(np.interp(excel_date, tenors, prices))

    return lookup, factor_data


def _build_discount_lookup(price_factors, discount_rate_name):
    """
    Build a discount rate interpolator from the Price Factors section.

    Replicates how riskflow/riskfactors.py InterestRate (line 640) loads its
    curve and how discount.current_value(t) returns the continuously compounded rate.

    In the bootstrapper (line 446):
        r = discount.current_value(t)

    Also provides get_day_count() for determining the year fraction convention.

    Parameters
    ----------
    price_factors : dict
        The 'Price Factors' section from the market data JSON.
    discount_rate_name : str
        The interest rate factor name, e.g. 'USD.LIBOR'.

    Returns
    -------
    callable : Takes a year fraction, returns the interpolated rate.
    int : Day count code for this curve.
    """
    factor_key = 'InterestRate.' + discount_rate_name
    factor_data = price_factors[factor_key]

    curve = factor_data['Curve']
    if isinstance(curve, dict) and curve.get('_type') == 'Curve':
        arr = curve['array']
    else:
        arr = np.array(sorted(curve))

    tenors = arr[:, 0]   # year fractions
    rates = arr[:, 1]    # continuously compounded rates

    # Determine day count convention — riskfactors.py InterestRate.get_day_count()
    # maps Day_Count field to the DAYCOUNT_* constants
    dc_str = factor_data.get('Day_Count', 'ACT_365')
    if 'ACT_360' in str(dc_str).upper().replace(' ', '_'):
        day_count_code = DAYCOUNT_ACT360
    elif 'ACT_365' in str(dc_str).upper().replace(' ', '_'):
        day_count_code = DAYCOUNT_ACT365
    else:
        day_count_code = DAYCOUNT_ACT365  # default

    def lookup(year_fraction):
        return float(np.interp(year_fraction, tenors, rates))

    return lookup, day_count_code


def _build_vol_surface_lookup(price_factors, vol_name):
    """
    Build a volatility surface interpolator from the Price Factors section.

    Replicates how riskflow/riskfactors.py ForwardPriceVol (line 1121) loads
    its surface and how vol_surface.current_value([[t, d, moneyness]]) returns
    the interpolated implied volatility.

    In the bootstrapper (line 448):
        sigma = vol_surface.current_value([[t, d, 1.0]])[0]

    The vol surface is a 3D object indexed by (expiry, tenor/delivery, moneyness).
    For CS calibration, moneyness is always 1.0 (ATM).

    Parameters
    ----------
    price_factors : dict
        The 'Price Factors' section from the market data JSON.
    vol_name : str
        The vol surface name, e.g. 'BRENT.OIL'.

    Returns
    -------
    callable : Takes (expiry_yf, delivery_yf, moneyness), returns implied vol.
    float : The Volatility_Delta shift (if any).
    """
    factor_key = 'ForwardPriceVol.' + vol_name
    factor_data = price_factors[factor_key]

    surface = factor_data['Surface']
    if isinstance(surface, dict) and surface.get('_type') == 'Curve':
        arr = surface['array']
    else:
        arr = np.array(sorted(surface))

    # Surface columns: [tenor/delivery, expiry, moneyness, vol]
    # ForwardPriceVol indices: TENOR_INDEX=0, EXPIRY_INDEX=1, MONEYNESS_INDEX=2
    # For ATM calibration (moneyness=1.0), we interpolate along expiry dimension
    # then along tenor/delivery dimension

    # Simple approach: for moneyness=1.0, extract the ATM slice and do 2D interp
    # If the surface only has one moneyness, use it directly
    unique_moneyness = np.unique(arr[:, 2])

    def lookup(expiry_yf, delivery_yf, moneyness=1.0):
        # Find closest moneyness slice
        m_idx = np.argmin(np.abs(unique_moneyness - moneyness))
        m_val = unique_moneyness[m_idx]
        slice_arr = arr[arr[:, 2] == m_val]

        if len(slice_arr) == 0:
            # Fallback: use all data
            slice_arr = arr

        if len(slice_arr) == 1:
            return float(slice_arr[0, 3])

        # Group by expiry, interpolate along tenor at each expiry, then interp along expiry
        unique_expiries = np.unique(slice_arr[:, 1])

        if len(unique_expiries) == 1:
            # Single expiry — just interpolate along tenor
            tenor_data = slice_arr[slice_arr[:, 1] == unique_expiries[0]]
            return float(np.interp(delivery_yf, tenor_data[:, 0], tenor_data[:, 3]))

        # Multiple expiries: interpolate along tenor at each expiry, then along expiry
        vol_at_expiries = []
        for exp in unique_expiries:
            exp_slice = slice_arr[slice_arr[:, 1] == exp]
            vol_at_delivery = np.interp(delivery_yf, exp_slice[:, 0], exp_slice[:, 3])
            vol_at_expiries.append(vol_at_delivery)

        return float(np.interp(expiry_yf, unique_expiries, vol_at_expiries))

    return lookup


def bootstrap_from_json(json_path, commodity_name=None):
    """
    Full implied calibration pipeline from a RiskFlow JSON file.

    Replicates the COMPLETE flow of CSForwardPriceModelParameters.bootstrap()
    at bootstrappers.py lines 387-489, including all the data preparation that
    happens BEFORE the optimizer is called:

        1. Load market data from JSON
        2. Find CSForwardPriceModelPrices entries in Market Prices
        3. Construct factor lookups: ForwardPriceVol, ForwardPrice, InterestRate
        4. For each Energy_Futures_Option:
           a) T = day_count_accrual(Base_Date, Expiry_Date - Base_Date)
           b) S = day_count_accrual(Base_Date, Settlement_Date - Base_Date)
           c) Forward = forward_curve(settlement_excel_date)
           d) Strike = forward_curve(expiry_excel_date) if ATM, else given strike
           e) r = discount_curve(T)
           f) sigma = vol_surface(T, S, 1.0) or Quoted_Market_Value
           g) Premium = Black(Forward, Strike, r, sigma, T, Units, callOrPut)
        5. Run calibrate_implied() optimizer

    This is the function you call when you want to replicate what RiskFlow does
    end-to-end when it bootstraps implied CS parameters.

    Parameters
    ----------
    json_path : str
        Path to a RiskFlow JSON file containing Market Prices with
        CSForwardPriceModelPrices entries.
    commodity_name : str, optional
        If specified, only calibrate this commodity (e.g. 'BRENT.OIL').
        If None, calibrate all CSForwardPriceModelPrices found.

    Returns
    -------
    dict : Maps commodity name to calibrated {'Sigma': ..., 'Alpha': ...}
    """
    from cs_simulation import load_market_data

    print("=" * 70)
    print("CS IMPLIED CALIBRATION — BOOTSTRAP FROM JSON")
    print("=" * 70)
    print(f"  Loading: {json_path}")

    market_data = load_market_data(json_path)
    price_factors = market_data.get('Price Factors', {})
    market_prices = market_data.get('Market Prices', {})
    sys_params = market_data.get('System Parameters', {})

    # Get Base_Date — check System Parameters first, then Valuation Configuration
    base_date = sys_params.get('Base_Date')
    if base_date is None:
        val_config = market_data.get('Valuation Configuration', {})
        if isinstance(val_config, dict):
            base_date = val_config.get('Base_Date', val_config.get('Run_Date'))
    if isinstance(base_date, str):
        base_date = pd.Timestamp(base_date)

    if base_date is None:
        raise ValueError("Cannot find Base_Date in System Parameters or Valuation Configuration")

    # Volatility delta shift — sys_params.get('Volatility_Delta', 0.0)
    vol_delta = sys_params.get('Volatility_Delta', 0.0)

    print(f"  Base date: {base_date.date()}")
    print(f"  Volatility delta: {vol_delta}")
    print(f"  Market Prices entries: {len(market_prices)}")

    results = {}

    # Iterate over Market Prices — replicates bootstrappers.py line 409
    for market_price_name, implied_params in market_prices.items():
        # Parse the rate name — replicates utils.check_rate_name()
        rate_parts = tuple(market_price_name.split('.'))
        factor_type = rate_parts[0]
        factor_name_parts = rate_parts[1:]

        if factor_type != 'CSForwardPriceModelPrices':
            continue

        commodity = '.'.join(factor_name_parts)
        if commodity_name is not None and commodity.upper() != commodity_name.upper():
            continue

        print(f"\n  {'='*60}")
        print(f"  Bootstrapping: {commodity}")
        print(f"  {'='*60}")

        instrument = implied_params.get('instrument', implied_params)

        # Extract the three factor references — bootstrappers.py lines 415-423
        vol_name = instrument['Forward_Volatility']
        energy_name = instrument['Energy']
        discount_name = instrument['Discount_Rate']
        quote_type = instrument.get('Quote_Type', 'Implied_Volatility')

        print(f"  Forward Volatility: {vol_name}")
        print(f"  Energy (Forward):   {energy_name}")
        print(f"  Discount Rate:      {discount_name}")
        print(f"  Quote Type:         {quote_type}")

        # Build lookups — replicates construct_factor() calls at lines 427-430
        try:
            forward_lookup, forward_data = _build_forward_lookup(price_factors, energy_name)
            discount_lookup, day_count_code = _build_discount_lookup(price_factors, discount_name)
            vol_lookup = _build_vol_surface_lookup(price_factors, vol_name)
        except KeyError as e:
            print(f"  ERROR: Cannot find factor {e} in Price Factors — skipping")
            continue

        # Process each option — replicates bootstrappers.py lines 437-463
        options_list = instrument.get('Energy_Futures_Options', [])
        print(f"  Options to calibrate: {len(options_list)}")

        print(f"\n  {'Expiry':>12s}  {'Settle':>12s}  {'T':>6s}  {'S':>6s}  "
              f"{'Forward':>8s}  {'Strike':>8s}  {'r':>6s}  {'MktVol':>8s}  {'Premium':>10s}")
        print("  " + "-" * 94)

        for option in options_list:
            # T = time to expiry in year fractions — line 438-439
            expiry_date = option['Expiry_Date']
            if isinstance(expiry_date, str):
                expiry_date = pd.Timestamp(expiry_date)
            days_to_expiry = (expiry_date - base_date).days
            t = get_day_count_accrual(base_date, days_to_expiry, day_count_code)

            # S = time to settlement in year fractions — line 440-441
            settlement_date = option['Settlement_Date']
            if isinstance(settlement_date, str):
                settlement_date = pd.Timestamp(settlement_date)
            days_to_settle = (settlement_date - base_date).days
            d = get_day_count_accrual(base_date, days_to_settle, day_count_code)

            # Excel dates for forward curve lookup — line 442-443
            expiry_excel = (expiry_date - EXCEL_OFFSET).days
            settlement_excel = (settlement_date - EXCEL_OFFSET).days

            # Forward prices — line 444-445
            forward_at_exp = forward_lookup(expiry_excel)
            forward_at_settle = forward_lookup(settlement_excel)

            # Discount rate — line 446
            r = discount_lookup(t)

            # Implied vol — lines 447-453
            if quote_type == 'Implied_Volatility':
                quoted = option.get('Quoted_Market_Value')
                if quoted:
                    sigma = quoted
                else:
                    # vol_surface.current_value([[t, d, 1.0]])[0] — line 448
                    sigma = vol_lookup(t, d, 1.0)
                sigma += vol_delta  # line 450
            else:
                print(f"  WARNING: Quote type '{quote_type}' not supported — skipping option")
                continue

            # Strike: ATM if not specified — line 455
            strike = option.get('Strike')
            if not strike:
                strike = forward_at_exp

            # Store computed values back into option dict (same as bootstrapper)
            option['Forward'] = forward_at_settle
            option['Strike'] = strike
            option['r'] = r
            option['S'] = d
            option['T'] = t
            option['sigma'] = sigma

            # Compute market premium — lines 461-463
            cp = 1.0 if option.get('Option_Type', 'Call') == 'Call' else -1.0
            units = option.get('Units', 1.0)
            option['Premium'] = black_european_option_price(
                option['Forward'], option['Strike'], r, sigma, t,
                units, cp)
            option['Units'] = units
            if 'Weight' not in option:
                option['Weight'] = 1.0

            exp_str = str(expiry_date.date()) if hasattr(expiry_date, 'date') else str(expiry_date)[:10]
            set_str = str(settlement_date.date()) if hasattr(settlement_date, 'date') else str(settlement_date)[:10]
            print(f"  {exp_str:>12s}  {set_str:>12s}  {t:6.3f}  {d:6.3f}  "
                  f"{forward_at_settle:8.4f}  {strike:8.4f}  {r:6.4f}  "
                  f"{sigma:8.4f}  {option['Premium']:10.4f}")

        # Run the optimizer — line 465-468
        print()
        params = calibrate_implied(options_list)
        results[commodity] = params

    if not results:
        print("\n  WARNING: No CSForwardPriceModelPrices entries found in Market Prices.")
        print("  Available market price types:")
        for k in market_prices:
            print(f"    {k}")

    print("=" * 70)
    return results


def black_european_option_price(F, X, r, vol, tenor, buyOrSell, callOrPut):
    """
    Replicates riskflow/utils.py black_european_option_price() (line 1165).

    Standard Black-76 formula for European options on forwards/futures.

    INTUITION:
        This is the Black-Scholes formula adapted for forward prices.
        The forward F already includes the cost-of-carry, so we only
        need to discount the expected payoff at the risk-free rate.

    Parameters
    ----------
    F : float
        Forward price of the underlying at delivery.
    X : float
        Strike price.
    r : float
        Risk-free interest rate (continuously compounded).
    vol : float
        Black implied volatility (annualised).
    tenor : float
        Time to option expiry in years.
    buyOrSell : float
        +1.0 for buy (long), -1.0 for sell (short).
    callOrPut : float
        +1.0 for call, -1.0 for put.

    Returns
    -------
    float : Option premium.
    """
    stddev = vol * np.sqrt(tenor)
    sign = 1.0 if (F > 0.0 and X > 0.0) else -1.0
    d1 = (np.log(F / X) + 0.5 * stddev * stddev) / stddev
    d2 = d1 - stddev
    return buyOrSell * callOrPut * (
        F * scipy.stats.norm.cdf(callOrPut * sign * d1) -
        X * scipy.stats.norm.cdf(callOrPut * sign * d2)
    ) * np.exp(-r * tenor)


def cs_variance(sigma, alpha, T, S):
    """
    The Clewlow-Strickland variance function.

    Replicates the V() function inside bootstrappers.py (line 395).

    INTUITION:
        This computes the total log-variance of the forward price F(T,S)
        from time 0 to option expiry T, where the forward settles at time S.

        V = sigma^2 * exp(-2*alpha*S) * B(2*alpha, T)

        Breaking this down:
        - sigma^2:              base instantaneous variance
        - exp(-2*alpha*S):      Samuelson damping — later delivery = less variance
        - B(2*alpha, T):        variance accumulation over the option's life,
                                 accounting for mean reversion

        B(a, t) = (1 - exp(-a*t)) / a
        This is the integrated OU variance over time t.
        When alpha=0, B(0, t) = t (standard Brownian motion).

    Parameters
    ----------
    sigma : float
        Instantaneous OU process volatility.
    alpha : float
        Mean reversion speed.
    T : float
        Time to option expiry (years).
    S : float
        Time to forward settlement/delivery (years). S >= T typically.

    Returns
    -------
    float : Total log-variance of F(T, S).
    """
    def B(a, t):
        return (1.0 - np.exp(-a * t)) / a if a != 0 else t

    return sigma * sigma * np.exp(-2.0 * alpha * S) * B(2.0 * alpha, T)


def calibrate_implied(options):
    """
    Replicates riskflow/bootstrappers.py CSForwardPriceModelParameters.bootstrap() (line 387).

    Finds the (sigma, alpha) pair that minimises the weighted sum of squared
    pricing errors across a set of European commodity options.

    HOW IT WORKS:
        1. For each option, compute the CS-model Black vol:
           sigma_black = sqrt(V(sigma, alpha, T, S) / T)    ... wait, actually
           the code passes sqrt(V) directly as the vol to black_european_option_price
           with tenor=1, because V already includes the T scaling via B().

           Actually looking more carefully: the code passes:
             vol = sqrt(V(sigma, alpha, T, S))
             tenor = 1.0   (not the actual tenor!)
           This is because V already represents the total variance (vol^2 * T),
           so sqrt(V) is the total standard deviation, and using tenor=1
           means the Black formula uses stddev = sqrt(V) * sqrt(1) = sqrt(V).

        2. Price the option using Black-76 with r=0 (risk-neutral, discounting
           done separately) and compare to the market premium.

        3. Minimise the sum of squared errors using scipy.optimize.minimize
           with bounds: sigma in [0.001, 2.5], alpha in [-1, 2.0].

    Parameters
    ----------
    options : list of dict
        Each dict must have keys:
            'Forward':     Forward price at settlement
            'Strike':      Option strike
            'T':           Time to expiry (years)
            'S':           Time to settlement (years)
            'r':           Risk-free rate
            'Premium':     Market premium to match
            'Units':       Notional/units
            'Option_Type': 'Call' or 'Put'
            'Weight':      Weight in the objective function

    Returns
    -------
    dict with keys 'Sigma', 'Alpha' (no Drift — it's always 0 for implied)
    """

    def calc_error(x, options):
        """
        Objective function: sum of weighted squared pricing errors.

        For each option:
            model_premium = Black(F, K, r=0, vol=sqrt(V(sigma,alpha,T,S)), tenor=1) * discount
            error += weight * (market_premium - model_premium)^2
        """
        sigma, alpha = x
        error = 0.0

        for option in options:
            discount = np.exp(-option['r'] * option['T'])
            cp = 1.0 if option['Option_Type'] == 'Call' else -1.0

            # Compute CS total variance and convert to total stddev
            total_var = cs_variance(sigma, alpha, option['T'], option['S'])
            total_stddev = np.sqrt(max(total_var, 1e-12))

            # Price using Black with r=0 and tenor=1 (variance already embedded in vol)
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

    # ---- Report results ----
    print("=" * 70)
    print("IMPLIED CALIBRATION RESULTS (Q-Measure / Risk-Neutral)")
    print("=" * 70)
    print(f"  Sigma (OU instantaneous vol):    {fitted_sigma:.6f}")
    print(f"  Alpha (Mean reversion speed):    {fitted_alpha:.6f}")
    print(f"  Drift:                           0.0 (risk-neutral)")
    print(f"  Optimiser success:               {result.success}")
    print(f"  Optimiser message:               {result.message}")
    print(f"  Final objective value:            {result.fun:.10f}")
    print()

    print("  Per-option fit quality:")
    print(f"  {'Expiry':>8s}  {'Settle':>8s}  {'Strike':>8s}  {'Mkt Vol':>8s}  "
          f"{'CS Vol':>8s}  {'Mkt Prem':>10s}  {'CS Prem':>10s}  {'Error':>10s}")
    print("  " + "-" * 82)

    for option in options:
        cp = 1.0 if option['Option_Type'] == 'Call' else -1.0
        discount = np.exp(-option['r'] * option['T'])

        # CS model vol and premium
        total_var = cs_variance(fitted_sigma, fitted_alpha, option['T'], option['S'])
        cs_vol = np.sqrt(total_var / option['T'])  # effective Black vol
        cs_premium = black_european_option_price(
            option['Forward'], option['Strike'], 0.0, np.sqrt(total_var),
            1.0, option['Units'], cp
        ) * discount

        err = (cs_premium - option['Premium']) ** 2
        mkt_vol = option.get('sigma', 0.0)

        print(f"  {option['T']:8.3f}  {option['S']:8.3f}  {option['Strike']:8.2f}  "
              f"{mkt_vol:8.4f}  {cs_vol:8.4f}  {option['Premium']:10.4f}  "
              f"{cs_premium:10.4f}  {err:10.6f}")

    print("=" * 70)

    return {'Sigma': fitted_sigma, 'Alpha': fitted_alpha}


if __name__ == '__main__':
    import sys

    # ====================================================================
    # MODE 1: Bootstrap from a RiskFlow JSON file (replicates full pipeline)
    # ====================================================================
    # Usage:  python cs_implied_calibration.py path/to/market_data.json [COMMODITY.NAME]
    #
    # This calls bootstrap_from_json() which replicates the complete
    # CSForwardPriceModelParameters.bootstrap() flow:
    #   1. Loads Market Prices → finds CSForwardPriceModelPrices entries
    #   2. Reads ForwardPriceVol, ForwardPrice, InterestRate from Price Factors
    #   3. Computes T, S, Forward, Strike, r, sigma, Premium for each option
    #   4. Runs the optimizer to find (Sigma, Alpha)

    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        commodity = sys.argv[2] if len(sys.argv) > 2 else None
        results = bootstrap_from_json(json_path, commodity_name=commodity)

        print("\nFinal results:")
        for name, params in results.items():
            print(f"  {name}: Sigma={params['Sigma']:.6f}, Alpha={params['Alpha']:.6f}")
        sys.exit(0)

    # ====================================================================
    # MODE 2: Synthetic self-test (no JSON needed)
    # ====================================================================
    # When run without arguments, generates synthetic option data from known
    # CS parameters and verifies the optimizer can recover them.

    print("\n" + "=" * 70)
    print("SELF-TEST: Synthetic data with known parameters")
    print("=" * 70)

    true_sigma = 0.35
    true_alpha = 1.2
    r = 0.05

    # Synthetic European call options on commodity futures
    option_specs = [
        {'T': 0.25, 'S': 0.25, 'Forward': 85.0, 'Strike': 85.0, 'Option_Type': 'Call'},
        {'T': 0.25, 'S': 0.50, 'Forward': 84.0, 'Strike': 84.0, 'Option_Type': 'Call'},
        {'T': 0.25, 'S': 1.00, 'Forward': 82.0, 'Strike': 82.0, 'Option_Type': 'Call'},
        {'T': 0.50, 'S': 0.50, 'Forward': 84.0, 'Strike': 84.0, 'Option_Type': 'Call'},
        {'T': 0.50, 'S': 1.00, 'Forward': 82.0, 'Strike': 82.0, 'Option_Type': 'Call'},
        {'T': 1.00, 'S': 1.00, 'Forward': 82.0, 'Strike': 82.0, 'Option_Type': 'Call'},
    ]

    # Compute "market" premiums using the true parameters
    options = []
    for spec in option_specs:
        total_var = cs_variance(true_sigma, true_alpha, spec['T'], spec['S'])
        total_stddev = np.sqrt(total_var)
        cp = 1.0 if spec['Option_Type'] == 'Call' else -1.0
        discount = np.exp(-r * spec['T'])

        premium = black_european_option_price(
            spec['Forward'], spec['Strike'], 0.0, total_stddev,
            1.0, 1.0, cp
        ) * discount

        mkt_vol = np.sqrt(total_var / spec['T'])

        options.append({
            'Forward': spec['Forward'],
            'Strike': spec['Strike'],
            'T': spec['T'],
            'S': spec['S'],
            'r': r,
            'Premium': premium,
            'Units': 1.0,
            'Option_Type': spec['Option_Type'],
            'Weight': 1.0,
            'sigma': mkt_vol,
        })

    print(f"\nTrue parameters: Sigma={true_sigma}, Alpha={true_alpha}")
    print(f"Number of options: {len(options)}\n")

    params = calibrate_implied(options)

    print(f"\nRecovery check:")
    print(f"  True Sigma = {true_sigma:.4f},  Fitted Sigma = {params['Sigma']:.4f}")
    print(f"  True Alpha = {true_alpha:.4f},  Fitted Alpha = {params['Alpha']:.4f}")
