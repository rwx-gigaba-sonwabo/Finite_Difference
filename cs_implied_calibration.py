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
import scipy.optimize
import scipy.stats


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

    # ---- Run the optimisation ----
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

    # ---- Show fit quality per option ----
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


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == '__main__':
    # ---- Create synthetic market data ----
    # Imagine we have European call options on Brent crude futures
    # with different expiry and settlement dates

    # True parameters we're trying to recover
    true_sigma = 0.35
    true_alpha = 1.2

    # Risk-free rate
    r = 0.05

    # Generate "market" option prices from the true CS model
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
            'sigma': mkt_vol,  # for reporting
        })

    print(f"\nTrue parameters: Sigma={true_sigma}, Alpha={true_alpha}")
    print(f"Number of options: {len(options)}\n")

    # Run the implied calibration
    params = calibrate_implied(options)

    print(f"\nRecovery check:")
    print(f"  True Sigma = {true_sigma:.4f},  Fitted Sigma = {params['Sigma']:.4f}")
    print(f"  True Alpha = {true_alpha:.4f},  Fitted Alpha = {params['Alpha']:.4f}")
