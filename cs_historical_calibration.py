"""
==============================================================================
CS HISTORICAL CALIBRATION (P-Measure)
==============================================================================

PURPOSE:
    Replicates what RiskFlow does when 'Model Configuration' maps
    ForwardPrice -> CSForwardPriceModel.

    In this mode, the model parameters (Sigma, Alpha, Drift) are estimated
    from a HISTORICAL time series of forward prices using statistical methods.

    This corresponds to:
        - riskflow/stochasticprocess.py  CSForwardPriceCalibration.calibrate()
        - riskflow/utils.py              calc_statistics()

WHAT THIS SCRIPT DOES:
    1. Takes a DataFrame of daily forward prices (columns = different tenors)
    2. Computes log-returns
    3. Estimates mean reversion speed (alpha) from autocorrelation of log prices
    4. Estimates reversion volatility (sigma) from residual variance
    5. Estimates drift (mu) from the mean log-return + Jensen's inequality correction
    6. Returns the calibrated parameters: {'Sigma': ..., 'Alpha': ..., 'Drift': ...}

THE UNDERLYING MODEL (Clewlow-Strickland):
    dF(t,T) = mu * F(t,T) * dt + sigma * exp(-alpha * (T-t)) * F(t,T) * dW(t)

    The key feature is the exp(-alpha*(T-t)) term: volatility DECAYS for
    forwards with longer time-to-delivery. This is the Samuelson effect.
==============================================================================
"""

import numpy as np
import pandas as pd


def calc_statistics(data_frame, method='Log', num_business_days=252.0, max_alpha=4.0):
    """
    Replicates riskflow/utils.py calc_statistics() (line 2076).

    This function estimates the parameters of a mean-reverting process from
    historical time series data.

    INTUITION:
        We model log-prices as an Ornstein-Uhlenbeck (OU) process:
            dX = -alpha * (X - theta) * dt + sigma * dW

        From discrete daily observations, we estimate:
            alpha:  How fast the process reverts to its mean
            sigma:  The instantaneous volatility of the OU process
            theta:  The long-run mean level
            drift:  The average daily change (annualised)

    HOW ALPHA IS ESTIMATED:
        For an OU process, X(t+dt) = X(t) + (1 - exp(-alpha*dt)) * (theta - X(t)) + noise
        So the regression coefficient of dX on X is: beta = 1 - exp(-alpha*dt)
        Solving: alpha = -log(1 + beta) * num_business_days

        In practice, we compute this from the covariance of (dX, X):
            beta = Cov(dX, X) / Var(X)

    HOW SIGMA IS ESTIMATED:
        Total variance of dX = Var(noise) + beta^2 * Var(X)
        Var(noise) = Var(dX) - beta^2 * Var(X)
        sigma^2 = Var(noise) * 2*alpha / (1 - exp(-2*alpha*dt))

        This corrects for the discrete sampling and mean reversion.

    Parameters
    ----------
    data_frame : pd.DataFrame
        Daily forward prices. Each column is a different tenor/contract.
    method : str
        'Log' for log-transform (lognormal model), 'Diff' for no transform.
    num_business_days : float
        Trading days per year (default 252).
    max_alpha : float
        Cap on estimated alpha to prevent numerical issues.

    Returns
    -------
    stats : pd.DataFrame
        Columns: Volatility, Drift, Mean Reversion Speed, Long Run Mean, Reversion Volatility
    correlation : pd.DataFrame
        Correlation matrix of the log-returns.
    data : pd.DataFrame
        The differenced (log-return) data used in the estimation.
    """

    def calc_alpha(x, y):
        """
        Estimate mean reversion speed from the regression of changes on levels.

        x = log-returns (dX)
        y = log-prices (X)

        beta = Cov(dX, X) / Var(X)
        alpha = -log(1 + beta) * days_per_year
        """
        beta = ((x - x.mean(axis=0)) * (y - y.mean(axis=0))).mean(axis=0) / \
               ((y - y.mean(axis=0)) ** 2.0).mean(axis=0)
        return (-num_business_days * np.log(1.0 + beta)).clip(0.001, max_alpha)

    def calc_sigma2(x, y, alpha):
        """
        Estimate the OU process variance parameter.

        This removes the mean-reversion component from the total variance
        and scales appropriately for continuous time.

        sigma^2 = [Var(dX) - (1-exp(-alpha/N))^2 * Var(X)] * 2*alpha / (1-exp(-2*alpha/N))
        """
        dt_factor = 1.0 - np.exp(-alpha / num_business_days)
        return (x.var(axis=0) - (dt_factor ** 2) * y.var(axis=0)) * \
               (2.0 * alpha) / (1.0 - np.exp(-2.0 * alpha / num_business_days))

    def calc_theta(x, y, alpha):
        """
        Estimate the long-run mean of the OU process.

        theta = mean(X) + mean(dX) / (1 - exp(-alpha/N))
        """
        return y.mean(axis=0) + x.mean(axis=0) / (1.0 - np.exp(-alpha / num_business_days))

    def calc_log_theta(theta, sigma2, alpha):
        """
        For the log-transform case, convert the log-space theta to price-space.

        E[exp(X)] = exp(E[X] + Var[X]/2)
        The long-run variance of the OU process is sigma^2 / (2*alpha).
        """
        return np.exp(theta + sigma2 / (4.0 * alpha))

    # ---- Step 1: Transform prices to log-prices (if method='Log') ----
    transform = {'Diff': lambda x: x, 'Log': lambda x: np.log(x.clip(0.0001, np.inf))}[method]
    transformed_df = transform(data_frame)

    # ---- Step 2: Compute daily log-returns ----
    # shift(-1) aligns today's return = tomorrow's price - today's price
    data = transformed_df.diff(1).shift(-1)

    # ---- Step 3: Estimate parameters ----
    y = transformed_df  # log-price levels
    alpha = calc_alpha(data, y)
    theta = calc_theta(data, y, alpha)
    sigma2 = calc_sigma2(data, y, alpha)

    # ---- Step 4: Convert theta to price-space for lognormal model ----
    if method == 'Log':
        theta = calc_log_theta(theta, sigma2, alpha)
        theta.replace([np.inf, -np.inf], np.nan, inplace=True)
        median = theta.median()
        theta[np.abs(theta - median) > (2 * theta.std())] = np.nan

    # ---- Step 5: Build results table ----
    stats = pd.DataFrame({
        'Volatility': data.std(axis=0) * np.sqrt(num_business_days),
        'Drift': data.mean(axis=0) * num_business_days,
        'Mean Reversion Speed': alpha,
        'Long Run Mean': theta,
        'Reversion Volatility': np.sqrt(sigma2)
    })

    correlation = data.corr()
    return stats, correlation, data


def calibrate_historical(data_frame, num_business_days=252.0):
    """
    Replicates riskflow/stochasticprocess.py CSForwardPriceCalibration.calibrate() (line 976).

    Takes the statistical estimates and packages them as CS model parameters.

    IMPORTANT:
        - 'Sigma' here is the REVERSION VOLATILITY, not the raw log-return volatility.
          It's the instantaneous vol parameter of the OU process driving the forward curve.
        - 'Alpha' is the mean reversion speed.
        - 'Drift' includes a Jensen's inequality correction: mu = drift + 0.5 * vol^2
          This converts from the arithmetic drift of log-returns to the geometric drift
          of the actual forward price process.

    Parameters
    ----------
    data_frame : pd.DataFrame
        Daily forward prices. Columns should be formatted as 'Name,Tenor'.
    num_business_days : float
        Trading days per year.

    Returns
    -------
    dict with keys 'Sigma', 'Alpha', 'Drift'
    """
    stats, correlation, delta = calc_statistics(
        data_frame, method='Log', num_business_days=num_business_days, max_alpha=5.0)

    alpha = stats['Mean Reversion Speed'].values[0]
    sigma = stats['Reversion Volatility'].values[0]
    # Jensen's correction: for a lognormal process, E[F] = exp(mu + 0.5*sigma^2)
    # so the drift of F is: mu_F = mu_logF + 0.5 * sigma_logF^2
    mu = stats['Drift'].values[0] + 0.5 * (stats['Volatility'].values[0]) ** 2

    print("=" * 60)
    print("HISTORICAL CALIBRATION RESULTS (P-Measure)")
    print("=" * 60)
    print(f"  Sigma (Reversion Volatility):  {sigma:.6f}")
    print(f"  Alpha (Mean Reversion Speed):  {alpha:.6f}")
    print(f"  Drift (mu):                    {mu:.6f}")
    print(f"  Raw Log-Return Volatility:     {stats['Volatility'].values[0]:.6f}")
    print(f"  Raw Log-Return Drift:          {stats['Drift'].values[0]:.6f}")
    print("=" * 60)

    return {'Sigma': sigma, 'Alpha': alpha, 'Drift': mu}


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == '__main__':
    # Generate synthetic forward price data for demonstration
    np.random.seed(42)
    n_days = 504  # ~2 years of daily data

    # Simulate a mean-reverting forward curve with known parameters
    true_sigma = 0.40
    true_alpha = 1.0
    true_drift = 0.02

    # Create synthetic daily forward prices for 3 tenors
    tenors = [0.25, 0.5, 1.0]  # years to delivery
    dates = pd.date_range('2022-01-01', periods=n_days, freq='B')

    prices = {}
    for tenor in tenors:
        # Simulate OU-driven lognormal forward prices
        dt = 1.0 / 252.0
        log_price = np.log(100.0)  # start at $100
        path = [np.exp(log_price)]

        for i in range(n_days - 1):
            vol_decay = true_sigma * np.exp(-true_alpha * tenor)
            dW = np.random.normal(0, np.sqrt(dt))
            log_price += (true_drift - 0.5 * vol_decay ** 2) * dt + vol_decay * dW
            path.append(np.exp(log_price))

        prices[f'BRENT,{tenor}'] = path

    df = pd.DataFrame(prices, index=dates)
    print("\nSample forward prices (first 5 rows):")
    print(df.head())
    print(f"\nTrue parameters: Sigma={true_sigma}, Alpha={true_alpha}, Drift={true_drift}")
    print()

    # Run the calibration
    params = calibrate_historical(df)
