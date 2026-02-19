"""
==============================================================================
CS SIMULATION DIAGNOSTICS
==============================================================================

PURPOSE:
    Comprehensive diagnostic tests for Clewlow-Strickland commodity forward
    price Monte Carlo simulations. Validates both RiskFlow's output and the
    independent validation script (cs_simulation.py).

DIAGNOSTICS:
    1. Martingale Tests — E[F(t,T)] vs theoretical expectation
    2. Moment Matching — mean, variance, skewness, kurtosis vs lognormal theory
    3. Tail Analysis — QQ plots, KS/AD tests, VaR/ES comparison
    4. Parameter Recovery — re-estimate sigma, alpha, drift, correlation
    5. Convergence Checks — statistics vs number of scenarios
    6. Standard Error Analysis — estimation uncertainty vs sample size
    7. Cross-Simulation Comparison — RiskFlow vs validation output

USAGE:
    from cs_simulation import run_simulation_from_json
    from cs_diagnostics import run_full_diagnostics

    simulated, precalc, metadata = run_simulation_from_json(...)
    results = run_full_diagnostics(simulated, metadata)

    # Compare with RiskFlow output:
    results = run_full_diagnostics(simulated, metadata, sim_benchmark=rf_output)

THE CS MODEL (recap):
    F(t,T) = F(0,T) * exp( drift(t,T) + cumsum(vol(t,T) * Z(t)) )

    Log-return X = log(F(t,T)/F(0,T)) ~ N(m, V) where:
        V(t,T) = sigma^2 * exp(-2*alpha*(T-t)) * (1 - exp(-2*alpha*t)) / (2*alpha)
        m = drift*t_eff - 0.5*V  (historical, t_eff = min(t, T))
        m = -0.5*V               (implied/risk-neutral)

    This gives lognormal forward prices with tenor-dependent volatility
    (Samuelson effect: nearer delivery = higher vol).
==============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

DAYS_IN_YEAR = 365.25


# =============================================================================
# 1. THEORETICAL MOMENTS — Clewlow-Strickland Lognormal Model
# =============================================================================

def cs_log_variance(sigma, alpha, T_years, t_years):
    """
    Theoretical log-variance of X = log(F(t,T)/F(0,T)) under the CS model.

    V(t, T) = sigma^2 * exp(-2*alpha*(T-t)) * (1 - exp(-2*alpha*t_eff)) / (2*alpha)

    where T = total time-to-delivery from base date (years),
          t = simulation time from base date (years),
          t_eff = min(t, T) — no variance accumulation after delivery.

    Derivation:
        Var[log F(t,T)] = sigma^2 * integral_0^{t_eff} exp(-2*alpha*(T-s)) ds
                        = sigma^2 * exp(-2*alpha*T) * [exp(2*alpha*t_eff) - 1] / (2*alpha)
                        = sigma^2 * exp(-2*alpha*(T-t_eff)) * [1 - exp(-2*alpha*t_eff)] / (2*alpha)

    Parameters
    ----------
    sigma : float — OU process volatility
    alpha : float — mean reversion speed
    T_years : float or np.ndarray — delivery time(s) from base date
    t_years : float or np.ndarray — simulation time from base date
    """
    T_arr = np.asarray(T_years, dtype=np.float64)
    t_arr = np.asarray(t_years, dtype=np.float64)
    t_eff = np.minimum(t_arr, T_arr)
    Tmt = np.maximum(T_arr - t_arr, 0.0)

    if np.abs(alpha) < 1e-10:
        return sigma ** 2 * t_eff

    return sigma ** 2 * np.exp(-2.0 * alpha * Tmt) * \
           (1.0 - np.exp(-2.0 * alpha * t_eff)) / (2.0 * alpha)


def cs_theoretical_log_moments(sigma, alpha, drift, T_years, t_years):
    """
    Theoretical mean and variance of X = log(F(t,T)/F(0,T)).

    Under CS: X ~ N(m, V) where
        V = cs_log_variance(sigma, alpha, T, t)
        m = drift * t_eff - 0.5 * V

    Returns: (m, V)
    """
    V = cs_log_variance(sigma, alpha, T_years, t_years)
    t_eff = np.minimum(np.asarray(t_years), np.asarray(T_years))
    m = drift * t_eff - 0.5 * V
    return m, V


def cs_theoretical_price_moments(F0, sigma, alpha, drift, T_years, t_years):
    """
    Theoretical price-level moments for F(t,T).

    Since X = log(F/F0) ~ N(m, V):
        E[F]    = F0 * exp(m + V/2) = F0 * exp(drift * t_eff)
        Var[F]  = E[F]^2 * (exp(V) - 1)
        Skew[F] = (exp(V) + 2) * sqrt(exp(V) - 1)       [lognormal]
        Kurt[F] = exp(4V) + 2*exp(3V) + 3*exp(2V) - 6   [excess kurtosis]

    Returns: dict
    """
    m, V = cs_theoretical_log_moments(sigma, alpha, drift, T_years, t_years)
    t_eff = np.minimum(np.asarray(t_years), np.asarray(T_years))

    price_mean = F0 * np.exp(drift * t_eff)
    price_var = price_mean ** 2 * np.maximum(np.exp(V) - 1.0, 0.0)
    price_std = np.sqrt(price_var)

    eV = np.exp(V)
    price_skew = (eV + 2.0) * np.sqrt(np.maximum(eV - 1.0, 0.0))
    price_kurt_excess = np.exp(4 * V) + 2 * np.exp(3 * V) + 3 * np.exp(2 * V) - 6.0

    return {
        'log_mean': m, 'log_var': V,
        'price_mean': price_mean, 'price_var': price_var,
        'price_std': price_std, 'price_skew': price_skew,
        'price_kurt_excess': price_kurt_excess,
    }


# =============================================================================
# HELPERS
# =============================================================================

def _get_time_tenor_arrays(metadata):
    """Extract simulation times (years) and delivery times (years) from metadata."""
    scen_time_grid = metadata['scen_time_grid']
    t_years = scen_time_grid / DAYS_IN_YEAR
    T_years = (metadata['tenors_excel'] - metadata['base_date_excel']) / DAYS_IN_YEAR
    return t_years, T_years


def _select_timesteps(n_timesteps, n_target=10):
    """Select ~n_target evenly spaced timestep indices."""
    if n_timesteps <= n_target:
        return list(range(n_timesteps))
    step = max(1, n_timesteps // n_target)
    indices = list(range(0, n_timesteps, step))
    if indices[-1] != n_timesteps - 1:
        indices.append(n_timesteps - 1)
    return indices


# =============================================================================
# 2. MARTINGALE TESTS
# =============================================================================

def martingale_test(simulated, metadata, timestep_indices=None,
                    confidence=0.95, plot=True):
    """
    Test the martingale property of simulated forward prices.

    Under risk-neutral (implied, drift=0):
        E[F(t,T)] = F(0,T) for all t <= T
        (forwards are martingales under the T-forward measure)

    Under P-measure (historical, drift != 0):
        E[F(t,T)] = F(0,T) * exp(drift * t)

    For each selected (timestep, tenor) pair, compute:
        ratio = E_sim[F(t,T)] / E_theo[F(t,T)]
    and run a two-sided t-test for statistical significance.

    Parameters
    ----------
    simulated : np.ndarray, shape [timesteps, tenors, scenarios]
    metadata : dict — from run_simulation_from_json() or equivalent
    timestep_indices : list of int, optional
    confidence : float
    plot : bool

    Returns
    -------
    results_df : pd.DataFrame
    """
    t_years, T_years = _get_time_tenor_arrays(metadata)
    params = metadata['params']
    drift = params['Drift']
    F0 = metadata['prices']
    n_scenarios = simulated.shape[2]

    if timestep_indices is None:
        timestep_indices = _select_timesteps(simulated.shape[0])

    records = []

    for t_idx in timestep_indices:
        t = t_years[t_idx]
        for tenor_idx, (T, f0) in enumerate(zip(T_years, F0)):
            if t > T + 0.01:
                continue  # past delivery

            sim_prices = simulated[t_idx, tenor_idx, :]
            sim_mean = np.mean(sim_prices)
            sim_se = np.std(sim_prices, ddof=1) / np.sqrt(n_scenarios)

            # Theoretical expectation
            t_eff = min(t, T)
            theo_mean = f0 * np.exp(drift * t_eff)

            ratio = sim_mean / theo_mean if theo_mean != 0 else np.nan
            t_stat = (sim_mean - theo_mean) / sim_se if sim_se > 0 else 0.0
            p_value = 2 * (1 - sp_stats.norm.cdf(abs(t_stat)))

            records.append({
                'timestep': t_idx,
                'sim_time_y': round(t, 4),
                'tenor_idx': tenor_idx,
                'delivery_y': round(T, 3),
                'F0': round(f0, 4),
                'E_sim': round(sim_mean, 4),
                'E_theo': round(theo_mean, 4),
                'ratio': round(ratio, 6),
                'SE': round(sim_se, 4),
                't_stat': round(t_stat, 3),
                'p_value': round(p_value, 4),
                'pass': p_value > (1 - confidence),
            })

    df = pd.DataFrame(records)

    # --- Print summary ---
    n_pass = df['pass'].sum()
    n_total = len(df)
    measure = 'risk-neutral (Q)' if abs(drift) < 1e-10 else f'P-measure (drift={drift:.4f})'

    print("=" * 80)
    print(f"MARTINGALE TEST — {measure}")
    print("=" * 80)
    print(f"  Tests passed: {n_pass}/{n_total} at {confidence*100:.0f}% confidence")
    print(f"  Scenarios: {n_scenarios}")
    print()

    # Show worst deviations
    df_sorted = df.reindex(df['ratio'].sub(1.0).abs().sort_values(ascending=False).index)
    print("  Worst deviations (ratio = E_sim / E_theo, ideal = 1.000):")
    cols = ['sim_time_y', 'delivery_y', 'F0', 'E_sim', 'E_theo', 'ratio', 'p_value']
    print(df_sorted[cols].head(10).to_string(index=False))
    print("=" * 80)

    if plot and len(df) > 0:
        _plot_martingale(df, T_years, n_scenarios, drift)

    return df


def _plot_martingale(df, T_years, n_scenarios, drift):
    """Plot martingale ratio over time for each tenor."""
    n_tenors = int(df['tenor_idx'].max()) + 1
    n_cols = min(n_tenors, 4)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4), squeeze=False)
    axes = axes.flatten()

    measure = 'Q-measure' if abs(drift) < 1e-10 else 'P-measure'
    fig.suptitle(f'Martingale Test: E[F(t,T)] / E_theo  ({measure}, N={n_scenarios})',
                 fontsize=12)

    for tenor_idx in range(n_cols):
        ax = axes[tenor_idx]
        sub = df[df['tenor_idx'] == tenor_idx]
        if sub.empty:
            continue

        ax.plot(sub['sim_time_y'], sub['ratio'], 'o-', markersize=3, color='steelblue')
        ax.axhline(1.0, color='red', linestyle='--', linewidth=1)

        # 95% confidence band
        se_ratio = sub['SE'].values / sub['E_theo'].values
        z = sp_stats.norm.ppf(0.975)
        ax.fill_between(sub['sim_time_y'], 1.0 - z * se_ratio, 1.0 + z * se_ratio,
                        alpha=0.2, color='red', label='95% CI')

        ax.set_title(f'T = {T_years[tenor_idx]:.2f}y')
        ax.set_xlabel('Simulation time (y)')
        ax.set_ylabel('E_sim / E_theo')
        ax.set_ylim(0.95, 1.05)
        if tenor_idx == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()
    plt.close()


# =============================================================================
# 3. MOMENT MATCHING
# =============================================================================

def moment_matching(simulated, metadata, timestep_indices=None, plot=True):
    """
    Compare sample moments against theoretical lognormal moments.

    For each selected (timestep, tenor):
        Log-return X = log(F(t,T)/F(0,T)):
            - Mean vs theoretical m = drift*t - 0.5*V
            - Variance vs theoretical V(t,T)
            - Skewness vs 0 (normal)
            - Excess kurtosis vs 0 (normal)

        Price level F(t,T):
            - Mean, std, skewness, kurtosis vs lognormal theory

    Returns
    -------
    log_df : pd.DataFrame — log-return moment comparison
    price_df : pd.DataFrame — price-level moment comparison
    """
    t_years, T_years = _get_time_tenor_arrays(metadata)
    params = metadata['params']
    F0 = metadata['prices']
    n_scenarios = simulated.shape[2]

    if timestep_indices is None:
        timestep_indices = _select_timesteps(simulated.shape[0], n_target=8)

    log_records = []
    price_records = []

    for t_idx in timestep_indices:
        t = t_years[t_idx]
        if t < 1e-6:
            continue

        for tenor_idx, (T, f0) in enumerate(zip(T_years, F0)):
            if t > T + 0.01:
                continue

            sim_F = simulated[t_idx, tenor_idx, :]
            log_ret = np.log(sim_F / f0)

            # Theoretical
            theo = cs_theoretical_price_moments(
                f0, params['Sigma'], params['Alpha'], params['Drift'], T, t)

            # Log-return moments
            log_records.append({
                't(y)': round(t, 4),
                'T(y)': round(T, 3),
                'mean_sim': round(np.mean(log_ret), 6),
                'mean_theo': round(float(theo['log_mean']), 6),
                'var_sim': round(np.var(log_ret, ddof=1), 6),
                'var_theo': round(float(theo['log_var']), 6),
                'skew_sim': round(float(sp_stats.skew(log_ret)), 4),
                'skew_theo': 0.0,
                'kurt_sim': round(float(sp_stats.kurtosis(log_ret)), 4),
                'kurt_theo': 0.0,
            })

            # Price-level moments
            price_records.append({
                't(y)': round(t, 4),
                'T(y)': round(T, 3),
                'mean_sim': round(np.mean(sim_F), 4),
                'mean_theo': round(float(theo['price_mean']), 4),
                'std_sim': round(np.std(sim_F, ddof=1), 4),
                'std_theo': round(float(theo['price_std']), 4),
                'skew_sim': round(float(sp_stats.skew(sim_F)), 4),
                'skew_theo': round(float(theo['price_skew']), 4),
                'kurt_sim': round(float(sp_stats.kurtosis(sim_F)), 4),
                'kurt_theo': round(float(theo['price_kurt_excess']), 4),
            })

    log_df = pd.DataFrame(log_records)
    price_df = pd.DataFrame(price_records)

    print("=" * 100)
    print("MOMENT MATCHING — LOG-RETURNS  (X = log(F(t,T)/F(0,T)) ~ N(m, V))")
    print("=" * 100)
    print(f"  Scenarios: {n_scenarios}")
    print()
    if len(log_df) > 0:
        print(log_df.to_string(index=False))
    print()

    print("=" * 100)
    print("MOMENT MATCHING — PRICE LEVELS  (F(t,T) ~ Lognormal)")
    print("=" * 100)
    if len(price_df) > 0:
        print(price_df.to_string(index=False))
    print("=" * 100)

    if plot and len(log_df) > 0:
        _plot_moments(log_df, price_df, n_scenarios)

    return log_df, price_df


def _plot_moments(log_df, price_df, n_scenarios):
    """Scatter plots: simulated vs theoretical for key moments."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f'Moment Matching: Simulated vs Theoretical (N={n_scenarios})', fontsize=13)

    for ax, xc, yc, title in [
        (axes[0, 0], 'mean_theo', 'mean_sim', 'Log-Return Mean'),
        (axes[0, 1], 'var_theo', 'var_sim', 'Log-Return Variance'),
    ]:
        x, y = log_df[xc], log_df[yc]
        ax.scatter(x, y, s=25, alpha=0.7, edgecolors='k', linewidths=0.3)
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        margin = (lims[1] - lims[0]) * 0.05
        ax.plot([lims[0] - margin, lims[1] + margin],
                [lims[0] - margin, lims[1] + margin], 'r--', linewidth=1)
        ax.set_xlabel('Theoretical')
        ax.set_ylabel('Simulated')
        ax.set_title(title)

    for ax, xc, yc, title in [
        (axes[1, 0], 'mean_theo', 'mean_sim', 'Price Mean'),
        (axes[1, 1], 'std_theo', 'std_sim', 'Price Std Dev'),
    ]:
        x, y = price_df[xc], price_df[yc]
        ax.scatter(x, y, s=25, alpha=0.7, edgecolors='k', linewidths=0.3)
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        margin = (lims[1] - lims[0]) * 0.05
        ax.plot([lims[0] - margin, lims[1] + margin],
                [lims[0] - margin, lims[1] + margin], 'r--', linewidth=1)
        ax.set_xlabel('Theoretical')
        ax.set_ylabel('Simulated')
        ax.set_title(title)

    plt.tight_layout()
    plt.show()
    plt.close()


# =============================================================================
# 4. TAIL ANALYSIS
# =============================================================================

def tail_analysis(simulated, metadata, tenor_idx=0, timestep_idx=-1, plot=True):
    """
    Quantify tail behavior vs theoretical lognormal distribution.

    Tests performed:
        - Kolmogorov-Smirnov test (log-returns vs N(m, V))
        - Anderson-Darling test
        - Quantile comparison at 1%, 5%, 10%, 90%, 95%, 99%
        - VaR and Expected Shortfall comparison in price space

    Parameters
    ----------
    simulated : np.ndarray, shape [timesteps, tenors, scenarios]
    metadata : dict
    tenor_idx : int
    timestep_idx : int — use -1 for final timestep
    plot : bool

    Returns
    -------
    results : dict
    """
    t_years, T_years = _get_time_tenor_arrays(metadata)
    params = metadata['params']
    F0 = metadata['prices']

    if timestep_idx < 0:
        timestep_idx = simulated.shape[0] + timestep_idx

    t = t_years[timestep_idx]
    T = T_years[tenor_idx]
    f0 = F0[tenor_idx]
    n_scenarios = simulated.shape[2]

    sim_F = simulated[timestep_idx, tenor_idx, :]
    log_ret = np.log(sim_F / f0)

    # Theoretical parameters
    m, V = cs_theoretical_log_moments(
        params['Sigma'], params['Alpha'], params['Drift'], T, t)
    sigma_lr = np.sqrt(V) if V > 0 else 1e-10

    # --- Kolmogorov-Smirnov test ---
    standardized = (log_ret - m) / sigma_lr
    ks_stat, ks_p = sp_stats.kstest(standardized, 'norm')

    # --- Anderson-Darling test ---
    ad_result = sp_stats.anderson(standardized, dist='norm')

    # --- Quantile comparison ---
    quantiles = [0.01, 0.05, 0.10, 0.90, 0.95, 0.99]
    tail_records = []
    for q in quantiles:
        emp_q = np.percentile(log_ret, q * 100)
        theo_q = sp_stats.norm.ppf(q, loc=m, scale=sigma_lr)

        # P(X <= empirical quantile) under theoretical distribution
        theo_prob_at_emp = sp_stats.norm.cdf(emp_q, loc=m, scale=sigma_lr)

        tail_records.append({
            'quantile': f'{q*100:.0f}%',
            'emp_logret': round(emp_q, 6),
            'theo_logret': round(theo_q, 6),
            'diff': round(emp_q - theo_q, 6),
            'emp_price': round(f0 * np.exp(emp_q), 4),
            'theo_price': round(f0 * np.exp(theo_q), 4),
            'theo_prob_at_emp': round(theo_prob_at_emp, 4),
        })

    tail_df = pd.DataFrame(tail_records)

    # --- VaR / Expected Shortfall comparison ---
    var_levels = [0.01, 0.05]
    var_records = []
    for alpha_var in var_levels:
        # Empirical VaR (price space, left tail)
        emp_var = np.percentile(sim_F, alpha_var * 100)
        # Theoretical VaR
        theo_var = f0 * np.exp(sp_stats.norm.ppf(alpha_var, loc=m, scale=sigma_lr))

        # Empirical Expected Shortfall
        mask = sim_F <= emp_var
        emp_es = np.mean(sim_F[mask]) if mask.any() else emp_var

        # Theoretical ES for lognormal: E[F | F <= VaR] = E[F]*Phi(z-sigma)/alpha
        z_alpha = sp_stats.norm.ppf(alpha_var)
        E_F = f0 * np.exp(params['Drift'] * min(t, T))
        theo_es = E_F * sp_stats.norm.cdf(z_alpha - sigma_lr) / alpha_var

        var_records.append({
            'level': f'{alpha_var*100:.0f}%',
            'VaR_emp': round(emp_var, 4),
            'VaR_theo': round(theo_var, 4),
            'VaR_ratio': round(emp_var / theo_var, 4) if theo_var != 0 else np.nan,
            'ES_emp': round(emp_es, 4),
            'ES_theo': round(theo_es, 4),
            'ES_ratio': round(emp_es / theo_es, 4) if theo_es != 0 else np.nan,
        })

    var_df = pd.DataFrame(var_records)

    # --- Print ---
    print("=" * 80)
    print(f"TAIL ANALYSIS — Tenor {tenor_idx} (T={T:.3f}y), Timestep {timestep_idx} (t={t:.3f}y)")
    print("=" * 80)
    print(f"  Scenarios: {n_scenarios},  F(0,T) = {f0:.4f}")
    print(f"  Theoretical: X ~ N({m:.6f}, {V:.6f}),  sigma_logret = {sigma_lr:.6f}")
    print()
    print(f"  Kolmogorov-Smirnov:  stat = {ks_stat:.4f},  p-value = {ks_p:.4f}  "
          f"{'[PASS]' if ks_p > 0.05 else '[FAIL at 5%]'}")
    print(f"  Anderson-Darling:    stat = {ad_result.statistic:.4f}")
    for sl, cv in zip(ad_result.significance_level, ad_result.critical_values):
        flag = ' <--' if ad_result.statistic > cv else ''
        print(f"    {sl:5.1f}% significance:  critical = {cv:.4f}{flag}")
    print()
    print("  Quantile comparison (log-returns and prices):")
    print(tail_df.to_string(index=False))
    print()
    print("  VaR / Expected Shortfall comparison (price levels):")
    print(var_df.to_string(index=False))
    print("=" * 80)

    results = {
        'ks_stat': ks_stat, 'ks_p': ks_p,
        'ad_stat': ad_result.statistic,
        'ad_critical': dict(zip(ad_result.significance_level, ad_result.critical_values)),
        'tail_df': tail_df, 'var_df': var_df,
        'log_ret': log_ret, 'theo_mean': m, 'theo_var': V,
    }

    if plot:
        _plot_tails(log_ret, m, V, f0, t, T, tenor_idx, n_scenarios)

    return results


def _plot_tails(log_ret, m, V, f0, t, T, tenor_idx, n_scenarios):
    """QQ plot, histogram, and left-tail CDF comparison."""
    sigma_lr = np.sqrt(V) if V > 0 else 1e-10
    n = len(log_ret)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Tail Analysis — Tenor {tenor_idx} (T={T:.2f}y, t={t:.2f}y, N={n_scenarios})',
                 fontsize=12)

    # QQ plot (standardized log-returns vs N(0,1))
    ax = axes[0]
    standardized = np.sort((log_ret - m) / sigma_lr)
    theoretical_q = sp_stats.norm.ppf(np.arange(1, n + 1) / (n + 1))
    ax.scatter(theoretical_q, standardized, s=1, alpha=0.3, color='steelblue')
    ax.plot([-4, 4], [-4, 4], 'r--', linewidth=1)
    ax.set_xlabel('Theoretical Quantiles N(0,1)')
    ax.set_ylabel('Sample Quantiles')
    ax.set_title('QQ Plot (Standardized Log-Returns)')
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)

    # Histogram with theoretical overlay
    ax = axes[1]
    ax.hist(log_ret, bins=80, density=True, alpha=0.6, color='steelblue', label='Simulated')
    x_range = np.linspace(m - 4 * sigma_lr, m + 4 * sigma_lr, 300)
    ax.plot(x_range, sp_stats.norm.pdf(x_range, loc=m, scale=sigma_lr),
            'r-', linewidth=2, label=f'N({m:.4f}, {V:.4f})')
    ax.set_xlabel('Log-return')
    ax.set_ylabel('Density')
    ax.set_title('Distribution Comparison')
    ax.legend(fontsize=8)

    # Left-tail CDF on log scale
    ax = axes[2]
    sorted_lr = np.sort(log_ret)
    ecdf = np.arange(1, n + 1) / n
    theo_cdf = sp_stats.norm.cdf(sorted_lr, loc=m, scale=sigma_lr)

    left_mask = ecdf < 0.15
    ax.semilogy(sorted_lr[left_mask], ecdf[left_mask], 'b.',
                markersize=2, alpha=0.5, label='Empirical')
    ax.semilogy(sorted_lr[left_mask], theo_cdf[left_mask], 'r-',
                linewidth=1, label='Theoretical')
    ax.set_xlabel('Log-return')
    ax.set_ylabel('CDF (log scale)')
    ax.set_title('Left Tail CDF')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()
    plt.close()


# =============================================================================
# 5. PARAMETER RECOVERY
# =============================================================================

def parameter_recovery(simulated, metadata, plot=True):
    """
    Recover CS model parameters from simulated paths.

    Estimates:
        Drift (mu):   From mean log-returns: mu = (E[logret] + 0.5*V) / t
        Volatility:   Raw log-return std at each (t, T) pair
        Alpha:        From Samuelson effect (vol ratio across tenors):
                      V(t,T1)/V(t,T2) = exp(-2*alpha*(T1-T2))
                      => alpha = -log(V1/V2) / (2*(T1-T2))
        Sigma:        From V(t,T) = sigma^2 * exp(-2*alpha*(T-t)) * ... solve for sigma
        Correlation:  From cross-tenor log-return correlations (multi-tenor)

    Parameters
    ----------
    simulated : np.ndarray, shape [timesteps, tenors, scenarios]
    metadata : dict

    Returns
    -------
    recovered : dict with keys sigma, alpha, drift, plus diagnostics
    """
    t_years, T_years = _get_time_tenor_arrays(metadata)
    params = metadata['params']
    F0 = metadata['prices']
    n_tenors = len(F0)
    n_timesteps = simulated.shape[0]

    # --- Compute log-return statistics at each (t, tenor) ---
    vol_surface = np.full((n_timesteps, n_tenors), np.nan)
    mean_surface = np.full((n_timesteps, n_tenors), np.nan)

    for t_idx in range(1, n_timesteps):
        t = t_years[t_idx]
        if t < 1e-6:
            continue
        for tenor_idx in range(n_tenors):
            T = T_years[tenor_idx]
            if t > T + 0.01:
                continue
            log_ret = np.log(simulated[t_idx, tenor_idx, :] / F0[tenor_idx])
            vol_surface[t_idx, tenor_idx] = np.std(log_ret, ddof=1)
            mean_surface[t_idx, tenor_idx] = np.mean(log_ret)

    # --- Recover alpha from Samuelson effect ---
    # Use the last few timesteps for reliable estimates
    last_indices = list(range(max(1, n_timesteps - 5), n_timesteps))

    alpha_estimates = []
    if n_tenors >= 2:
        for t_idx in last_indices:
            t = t_years[t_idx]
            for i in range(n_tenors):
                for j in range(i + 1, n_tenors):
                    T_i, T_j = T_years[i], T_years[j]
                    if t > min(T_i, T_j) + 0.01:
                        continue
                    v_i = vol_surface[t_idx, i]
                    v_j = vol_surface[t_idx, j]
                    if np.isnan(v_i) or np.isnan(v_j) or v_i <= 0 or v_j <= 0:
                        continue
                    T_diff = T_i - T_j
                    if abs(T_diff) < 0.01:
                        continue
                    # V(t,T_i)/V(t,T_j) = exp(-2*alpha*(T_i-T_j))
                    # (the (1-exp(-2*alpha*t))/(2*alpha) part cancels in the ratio)
                    ratio = (v_i ** 2) / (v_j ** 2)
                    if ratio > 0:
                        alpha_est = -np.log(ratio) / (2.0 * T_diff)
                        if -1 < alpha_est < 5:
                            alpha_estimates.append(alpha_est)

    alpha_recovered = np.median(alpha_estimates) if alpha_estimates else np.nan

    # --- Recover sigma from vol surface ---
    sigma_estimates = []
    alpha_use = alpha_recovered if not np.isnan(alpha_recovered) else params['Alpha']

    for t_idx in last_indices:
        t = t_years[t_idx]
        for tenor_idx in range(n_tenors):
            T = T_years[tenor_idx]
            if t > T + 0.01:
                continue
            v = vol_surface[t_idx, tenor_idx]
            if np.isnan(v) or v <= 0:
                continue
            Tmt = T - t
            t_eff = min(t, T)
            if abs(alpha_use) < 1e-10:
                denom = t_eff
            else:
                denom = np.exp(-2 * alpha_use * Tmt) * \
                        (1 - np.exp(-2 * alpha_use * t_eff)) / (2 * alpha_use)
            if denom > 1e-12:
                sigma_est = v / np.sqrt(denom)
                if 0 < sigma_est < 10:
                    sigma_estimates.append(sigma_est)

    sigma_recovered = np.median(sigma_estimates) if sigma_estimates else np.nan

    # --- Recover drift ---
    drift_estimates = []
    for t_idx in last_indices:
        t = t_years[t_idx]
        for tenor_idx in range(n_tenors):
            T = T_years[tenor_idx]
            if t > T + 0.01 or t < 0.1:
                continue
            m_obs = mean_surface[t_idx, tenor_idx]
            v_obs = vol_surface[t_idx, tenor_idx]
            if np.isnan(m_obs) or np.isnan(v_obs):
                continue
            # E[logret] = drift*t - 0.5*V, so drift = (E[logret] + 0.5*V) / t
            V_obs = v_obs ** 2
            drift_est = (m_obs + 0.5 * V_obs) / t
            drift_estimates.append(drift_est)

    drift_recovered = np.median(drift_estimates) if drift_estimates else np.nan

    # --- Recover annualised log-return volatility ---
    vol_estimates = []
    for t_idx in last_indices:
        t = t_years[t_idx]
        if t < 0.1:
            continue
        for tenor_idx in range(n_tenors):
            T = T_years[tenor_idx]
            if t > T + 0.01:
                continue
            v = vol_surface[t_idx, tenor_idx]
            if not np.isnan(v) and v > 0:
                vol_estimates.append(v / np.sqrt(t))  # annualised

    vol_recovered = np.median(vol_estimates) if vol_estimates else np.nan

    # --- Print results ---
    print("=" * 70)
    print("PARAMETER RECOVERY FROM SIMULATED PATHS")
    print("=" * 70)
    print(f"  Scenarios: {simulated.shape[2]}")
    print(f"  Timesteps used for recovery: {last_indices}")
    print()
    print(f"  {'Parameter':<30s}  {'Input':>10s}  {'Recovered':>10s}  {'Rel Err':>10s}")
    print("  " + "-" * 65)

    for name, true_val, rec_val in [
        ('Sigma (OU vol)', params['Sigma'], sigma_recovered),
        ('Alpha (mean reversion)', params['Alpha'], alpha_recovered),
        ('Drift (mu)', params['Drift'], drift_recovered),
        ('Annualised logret vol (*)', np.nan, vol_recovered),
    ]:
        if np.isnan(true_val):
            print(f"  {name:<30s}  {'n/a':>10s}  {rec_val:10.6f}  {'n/a':>10s}")
        else:
            err = (rec_val - true_val) / abs(true_val) * 100 if abs(true_val) > 1e-10 else np.nan
            err_str = f'{err:.2f}%' if not np.isnan(err) else 'n/a'
            print(f"  {name:<30s}  {true_val:10.6f}  {rec_val:10.6f}  {err_str:>10s}")

    print()
    if alpha_estimates:
        print(f"  Alpha estimates ({len(alpha_estimates)} pairs): "
              f"mean={np.mean(alpha_estimates):.4f}, "
              f"std={np.std(alpha_estimates):.4f}, "
              f"median={np.median(alpha_estimates):.4f}")
    else:
        print("  Alpha: need >= 2 tenors with different delivery dates")

    if sigma_estimates:
        print(f"  Sigma estimates ({len(sigma_estimates)} points): "
              f"mean={np.mean(sigma_estimates):.4f}, "
              f"std={np.std(sigma_estimates):.4f}")
    print("=" * 70)

    recovered = {
        'sigma': sigma_recovered,
        'alpha': alpha_recovered,
        'drift': drift_recovered,
        'vol_annualised': vol_recovered,
        'alpha_estimates': alpha_estimates,
        'sigma_estimates': sigma_estimates,
        'drift_estimates': drift_estimates,
        'vol_surface': vol_surface,
        'mean_surface': mean_surface,
    }

    if plot:
        _plot_parameter_recovery(vol_surface, t_years, T_years, params, recovered)

    return recovered


def _plot_parameter_recovery(vol_surface, t_years, T_years, params, recovered):
    """Plot vol over time and Samuelson effect."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Parameter Recovery', fontsize=12)
    n_tenors = vol_surface.shape[1]

    # --- Vol over time for each tenor ---
    ax = axes[0]
    for tenor_idx in range(min(n_tenors, 6)):
        valid = ~np.isnan(vol_surface[:, tenor_idx])
        if valid.any():
            ax.plot(t_years[valid], vol_surface[valid, tenor_idx],
                    'o-', markersize=2, label=f'T={T_years[tenor_idx]:.2f}y')

    ax.set_xlabel('Simulation time (y)')
    ax.set_ylabel('Std of log-return')
    ax.set_title('Log-Return Volatility over Time')
    ax.legend(fontsize=7)

    # --- Samuelson effect at final timestep ---
    ax = axes[1]
    t_final_idx = vol_surface.shape[0] - 1
    t_final = t_years[t_final_idx]
    vols_final = vol_surface[t_final_idx, :]
    valid = ~np.isnan(vols_final)

    if valid.any():
        Tmt = T_years[valid] - t_final
        ax.scatter(Tmt, vols_final[valid], s=50, zorder=5, label='Simulated',
                   edgecolors='k', linewidths=0.5)

        # Theoretical curve
        sigma, alpha = params['Sigma'], params['Alpha']
        Tmt_range = np.linspace(max(Tmt.min(), 0.001), Tmt.max() * 1.1, 100)
        if abs(alpha) > 1e-10:
            V_theo = sigma ** 2 * np.exp(-2 * alpha * Tmt_range) * \
                     (1 - np.exp(-2 * alpha * t_final)) / (2 * alpha)
        else:
            V_theo = sigma ** 2 * t_final * np.ones_like(Tmt_range)
        ax.plot(Tmt_range, np.sqrt(V_theo), 'r-', linewidth=2, label='Theoretical (input)')

        # Recovered curve
        sr, ar = recovered['sigma'], recovered['alpha']
        if not np.isnan(sr) and not np.isnan(ar) and abs(ar) > 1e-10:
            V_rec = sr ** 2 * np.exp(-2 * ar * Tmt_range) * \
                    (1 - np.exp(-2 * ar * t_final)) / (2 * ar)
            ax.plot(Tmt_range, np.sqrt(V_rec), 'g--', linewidth=2, label='Theoretical (recovered)')

    ax.set_xlabel('Remaining time-to-delivery (y)')
    ax.set_ylabel('Std of log-return')
    ax.set_title(f'Samuelson Effect at t = {t_final:.2f}y')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()
    plt.close()


# =============================================================================
# 5b. CORRELATION RECOVERY (multi-factor)
# =============================================================================

def correlation_recovery(simulations_dict, metadata_dict, true_correlations=None):
    """
    Recover the correlation matrix from multiple correlated factor simulations.

    Parameters
    ----------
    simulations_dict : dict of {factor_name: np.ndarray}
        Each array has shape [timesteps, tenors, scenarios].
    metadata_dict : dict of {factor_name: metadata}
    true_correlations : dict of {(name1, name2): rho}, optional
        Input correlations for comparison.

    Returns
    -------
    corr_df : pd.DataFrame — recovered vs input correlations
    """
    factor_names = list(simulations_dict.keys())
    n_factors = len(factor_names)

    if n_factors < 2:
        print("  Correlation recovery requires >= 2 factors.")
        return None

    # Use the first tenor of each factor at a mid-point timestep
    log_returns = {}
    for fname in factor_names:
        sim = simulations_dict[fname]
        meta = metadata_dict[fname]
        F0 = meta['prices'][0]
        t_idx = sim.shape[0] // 2  # mid-point
        log_returns[fname] = np.log(sim[t_idx, 0, :] / F0)

    records = []
    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            fi, fj = factor_names[i], factor_names[j]
            rho_sim = np.corrcoef(log_returns[fi], log_returns[fj])[0, 1]

            rho_true = np.nan
            if true_correlations:
                rho_true = true_correlations.get(
                    (fi, fj), true_correlations.get((fj, fi), np.nan))

            records.append({
                'Factor 1': fi,
                'Factor 2': fj,
                'rho_input': round(rho_true, 4) if not np.isnan(rho_true) else 'n/a',
                'rho_sim': round(rho_sim, 4),
                'diff': round(rho_sim - rho_true, 4) if not np.isnan(rho_true) else 'n/a',
            })

    corr_df = pd.DataFrame(records)

    print("=" * 70)
    print("CORRELATION RECOVERY")
    print("=" * 70)
    print(corr_df.to_string(index=False))
    print("=" * 70)

    return corr_df


# =============================================================================
# 6. CONVERGENCE CHECKS
# =============================================================================

def convergence_analysis(simulated, metadata, tenor_idx=0, timestep_idx=-1,
                         scenario_counts=None, plot=True):
    """
    Analyze convergence of key statistics as the number of scenarios increases.

    Computes mean, std, percentiles, and VaR at increasing scenario counts.
    Compares each to the theoretical value to quantify convergence.

    Parameters
    ----------
    simulated : np.ndarray
    metadata : dict
    tenor_idx : int
    timestep_idx : int (-1 for final)
    scenario_counts : list of int, optional
    plot : bool

    Returns
    -------
    conv_df : pd.DataFrame
    """
    t_years, T_years = _get_time_tenor_arrays(metadata)
    params = metadata['params']
    F0 = metadata['prices']

    if timestep_idx < 0:
        timestep_idx = simulated.shape[0] + timestep_idx

    t = t_years[timestep_idx]
    T = T_years[tenor_idx]
    f0 = F0[tenor_idx]
    n_total = simulated.shape[2]

    sim_F = simulated[timestep_idx, tenor_idx, :]

    if scenario_counts is None:
        candidates = [50, 100, 250, 500, 1000, 2000, 4096, 8192, 16384, 32768]
        scenario_counts = [n for n in candidates if n <= n_total]
        if n_total not in scenario_counts:
            scenario_counts.append(n_total)
        scenario_counts.sort()

    # Theoretical values
    theo = cs_theoretical_price_moments(
        f0, params['Sigma'], params['Alpha'], params['Drift'], T, t)
    m, V = cs_theoretical_log_moments(
        params['Sigma'], params['Alpha'], params['Drift'], T, t)
    sigma_lr = np.sqrt(V) if V > 0 else 1e-10

    theo_var_1pct = f0 * np.exp(sp_stats.norm.ppf(0.01, loc=m, scale=sigma_lr))
    theo_var_5pct = f0 * np.exp(sp_stats.norm.ppf(0.05, loc=m, scale=sigma_lr))

    records = []
    for N in scenario_counts:
        subset = sim_F[:N]
        records.append({
            'N': N,
            'mean': np.mean(subset),
            'mean_err%': (np.mean(subset) / float(theo['price_mean']) - 1) * 100,
            'std': np.std(subset, ddof=1),
            'std_err%': (np.std(subset, ddof=1) / float(theo['price_std']) - 1) * 100,
            'VaR_1%': np.percentile(subset, 1),
            'VaR_5%': np.percentile(subset, 5),
            'p50': np.percentile(subset, 50),
            'p95': np.percentile(subset, 95),
        })

    conv_df = pd.DataFrame(records)

    print("=" * 100)
    print(f"CONVERGENCE ANALYSIS — Tenor {tenor_idx} (T={T:.3f}y), t={t:.3f}y")
    print("=" * 100)
    print(f"  Theoretical: E[F]={float(theo['price_mean']):.4f}, "
          f"Std={float(theo['price_std']):.4f}, "
          f"VaR1%={theo_var_1pct:.4f}, VaR5%={theo_var_5pct:.4f}")
    print()
    print(conv_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print("=" * 100)

    if plot:
        _plot_convergence(conv_df, theo, theo_var_1pct, theo_var_5pct, T, t, tenor_idx)

    return conv_df


def _plot_convergence(conv_df, theo, theo_var_1pct, theo_var_5pct, T, t, tenor_idx):
    """Plot convergence curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Convergence — Tenor {tenor_idx} (T={T:.2f}y, t={t:.2f}y)', fontsize=12)

    N = conv_df['N'].values

    for ax, col, theo_val, title in [
        (axes[0, 0], 'mean', float(theo['price_mean']), 'Mean Forward Price'),
        (axes[0, 1], 'std', float(theo['price_std']), 'Std Forward Price'),
        (axes[1, 0], 'VaR_1%', theo_var_1pct, '1% VaR (Left Tail)'),
        (axes[1, 1], 'VaR_5%', theo_var_5pct, '5% VaR (Left Tail)'),
    ]:
        ax.semilogx(N, conv_df[col], 'o-', color='steelblue', markersize=4)
        ax.axhline(theo_val, color='red', linestyle='--', linewidth=1,
                   label=f'Theo = {theo_val:.2f}')
        ax.set_xlabel('N scenarios')
        ax.set_ylabel(col)
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()
    plt.close()


# =============================================================================
# 7. STANDARD ERROR ANALYSIS
# =============================================================================

def standard_error_analysis(simulated, metadata, tenor_idx=0, timestep_idx=-1,
                            scenario_counts=None, n_bootstrap=200, plot=True):
    """
    Analyze standard errors of key statistics and 1/sqrt(N) convergence.

    For each scenario count N:
        - SE(mean):  analytical = sigma / sqrt(N)
        - SE(var):   analytical = sigma^2 * sqrt(2 / (N-1))
        - SE(P5), SE(P95), SE(VaR1%): via bootstrap
        - Check SE * sqrt(N) is approximately constant

    Parameters
    ----------
    simulated : np.ndarray
    metadata : dict
    tenor_idx, timestep_idx : int
    scenario_counts : list of int
    n_bootstrap : int — bootstrap resamples for quantile SEs
    plot : bool

    Returns
    -------
    se_df : pd.DataFrame
    """
    t_years, T_years = _get_time_tenor_arrays(metadata)

    if timestep_idx < 0:
        timestep_idx = simulated.shape[0] + timestep_idx

    t = t_years[timestep_idx]
    T = T_years[tenor_idx]
    n_total = simulated.shape[2]

    sim_F = simulated[timestep_idx, tenor_idx, :]

    if scenario_counts is None:
        candidates = [100, 250, 500, 1000, 2000, 4096, 8192, 16384]
        scenario_counts = [n for n in candidates if n <= n_total]
        if n_total not in scenario_counts:
            scenario_counts.append(n_total)
        scenario_counts.sort()

    records = []
    for N in scenario_counts:
        subset = sim_F[:N]

        # Analytical SEs
        se_mean = np.std(subset, ddof=1) / np.sqrt(N)
        se_var = np.var(subset, ddof=1) * np.sqrt(2.0 / (N - 1))

        # Bootstrap for quantile SEs
        boot_p5, boot_p95, boot_var1 = [], [], []
        for _ in range(n_bootstrap):
            resample = np.random.choice(subset, size=N, replace=True)
            boot_p5.append(np.percentile(resample, 5))
            boot_p95.append(np.percentile(resample, 95))
            boot_var1.append(np.percentile(resample, 1))

        se_p5 = np.std(boot_p5)
        se_p95 = np.std(boot_p95)
        se_var1 = np.std(boot_var1)

        records.append({
            'N': N,
            'SE(mean)': round(se_mean, 4),
            'SE(var)': round(se_var, 2),
            'SE(P5)': round(se_p5, 4),
            'SE(P95)': round(se_p95, 4),
            'SE(VaR1%)': round(se_var1, 4),
            'SE(mean)*sqrtN': round(se_mean * np.sqrt(N), 4),
        })

    se_df = pd.DataFrame(records)

    print("=" * 90)
    print(f"STANDARD ERROR ANALYSIS — Tenor {tenor_idx} (T={T:.3f}y), t={t:.3f}y")
    print("=" * 90)
    print(f"  SE * sqrt(N) should be approximately constant (1/sqrt(N) convergence)")
    print()
    print(se_df.to_string(index=False))
    print("=" * 90)

    if plot:
        _plot_standard_errors(se_df, T, t, tenor_idx)

    return se_df


def _plot_standard_errors(se_df, T, t, tenor_idx):
    """Plot SE curves: log-log and SE*sqrt(N) stability."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Standard Error Analysis — Tenor {tenor_idx} (T={T:.2f}y, t={t:.2f}y)',
                 fontsize=12)

    N = se_df['N'].values

    # SE vs N (log-log)
    ax = axes[0]
    ax.loglog(N, se_df['SE(mean)'], 'o-', label='SE(Mean)', markersize=4)
    ax.loglog(N, se_df['SE(P5)'], 's-', label='SE(P5)', markersize=4)
    ax.loglog(N, se_df['SE(P95)'], '^-', label='SE(P95)', markersize=4)
    ax.loglog(N, se_df['SE(VaR1%)'], 'v-', label='SE(VaR 1%)', markersize=4)

    # 1/sqrt(N) reference line
    ref_val = se_df['SE(mean)'].iloc[0] * np.sqrt(N[0])
    ref = ref_val / np.sqrt(N)
    ax.loglog(N, ref, 'k--', linewidth=1, alpha=0.5, label='1/sqrt(N) ref')

    ax.set_xlabel('N scenarios')
    ax.set_ylabel('Standard Error')
    ax.set_title('SE vs N (log-log scale)')
    ax.legend(fontsize=8)

    # SE * sqrt(N) should be constant
    ax = axes[1]
    ax.semilogx(N, se_df['SE(mean)*sqrtN'], 'o-', markersize=4, label='SE(Mean) * sqrt(N)')
    ax.set_xlabel('N scenarios')
    ax.set_ylabel('SE * sqrt(N)')
    ax.set_title('SE * sqrt(N)  (should be constant)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()
    plt.close()


# =============================================================================
# 8. CROSS-SIMULATION COMPARISON
# =============================================================================

def compare_simulations(sim_a, sim_b, metadata,
                        labels=('Validation', 'RiskFlow'),
                        tenor_idx=0, timestep_idx=-1, plot=True):
    """
    Compare two simulation outputs side-by-side.

    Tests:
        - Two-sample KS test between empirical distributions
        - Moment comparison table
        - QQ plot (empirical A vs empirical B)
        - Path-level comparison (if same scenario count → same seed assumed)

    Parameters
    ----------
    sim_a, sim_b : np.ndarray, shape [timesteps, tenors, scenarios]
    metadata : dict
    labels : tuple of str
    tenor_idx, timestep_idx : int
    plot : bool

    Returns
    -------
    comparison : dict
    """
    t_years, T_years = _get_time_tenor_arrays(metadata)
    F0 = metadata['prices']

    if timestep_idx < 0:
        timestep_idx = min(sim_a.shape[0], sim_b.shape[0]) + timestep_idx

    t = t_years[timestep_idx]
    T = T_years[tenor_idx]
    f0 = F0[tenor_idx]

    fa = sim_a[timestep_idx, tenor_idx, :]
    fb = sim_b[timestep_idx, tenor_idx, :]

    # --- Moment comparison ---
    moments = {}
    for label, data in [(labels[0], fa), (labels[1], fb)]:
        log_ret = np.log(data / f0)
        moments[label] = {
            'mean': np.mean(data),
            'std': np.std(data, ddof=1),
            'skewness': sp_stats.skew(data),
            'kurtosis': sp_stats.kurtosis(data),
            'log_mean': np.mean(log_ret),
            'log_std': np.std(log_ret, ddof=1),
            'P1': np.percentile(data, 1),
            'P5': np.percentile(data, 5),
            'P50': np.percentile(data, 50),
            'P95': np.percentile(data, 95),
            'P99': np.percentile(data, 99),
        }

    moments_df = pd.DataFrame(moments).T

    # --- KS test ---
    ks_stat, ks_p = sp_stats.ks_2samp(fa, fb)

    # --- Path-level comparison (if same N) ---
    path_comparison = None
    if fa.shape[0] == fb.shape[0]:
        diff = fa - fb
        rel_diff = np.abs(diff) / np.maximum(np.abs(fb), 1e-10)
        path_comparison = {
            'max_abs_diff': float(np.max(np.abs(diff))),
            'mean_abs_diff': float(np.mean(np.abs(diff))),
            'median_abs_diff': float(np.median(np.abs(diff))),
            'max_rel_diff': float(np.max(rel_diff)),
            'mean_rel_diff': float(np.mean(rel_diff)),
            'pearson_corr': float(np.corrcoef(fa, fb)[0, 1]),
        }

    # --- Print ---
    print("=" * 80)
    print(f"SIMULATION COMPARISON: {labels[0]} vs {labels[1]}")
    print(f"  Tenor {tenor_idx} (T={T:.3f}y), Timestep {timestep_idx} (t={t:.3f}y)")
    print("=" * 80)
    print(f"\n  2-sample KS test: stat = {ks_stat:.4f}, p = {ks_p:.4f}  "
          f"{'[distributions match]' if ks_p > 0.05 else '[distributions DIFFER]'}")
    print()
    print("  Moment comparison:")
    print(moments_df.to_string(float_format=lambda x: f'{x:.4f}'))

    if path_comparison:
        print(f"\n  Path-level comparison (same-seed scenario matching):")
        print(f"    Max absolute diff:    {path_comparison['max_abs_diff']:.6f}")
        print(f"    Mean absolute diff:   {path_comparison['mean_abs_diff']:.6f}")
        print(f"    Median absolute diff: {path_comparison['median_abs_diff']:.6f}")
        print(f"    Max relative diff:    {path_comparison['max_rel_diff']:.2e}")
        print(f"    Mean relative diff:   {path_comparison['mean_rel_diff']:.2e}")
        print(f"    Pearson correlation:   {path_comparison['pearson_corr']:.8f}")

    print("=" * 80)

    if plot:
        _plot_comparison(fa, fb, f0, labels, T, t, tenor_idx, path_comparison is not None)

    return {
        'moments_df': moments_df,
        'ks_stat': ks_stat, 'ks_p': ks_p,
        'path_comparison': path_comparison,
    }


def _plot_comparison(fa, fb, f0, labels, T, t, tenor_idx, same_seed):
    """Histogram overlay, QQ plot, and optional path scatter."""
    n_plots = 3 if same_seed else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 5))
    fig.suptitle(f'{labels[0]} vs {labels[1]} — Tenor {tenor_idx} (T={T:.2f}y, t={t:.2f}y)',
                 fontsize=12)

    # Histogram overlay
    ax = axes[0]
    lo = min(fa.min(), fb.min())
    hi = max(fa.max(), fb.max())
    bins = np.linspace(lo, hi, 80)
    ax.hist(fa, bins=bins, density=True, alpha=0.5, label=labels[0], color='steelblue')
    ax.hist(fb, bins=bins, density=True, alpha=0.5, label=labels[1], color='coral')
    ax.set_xlabel('Forward Price')
    ax.set_ylabel('Density')
    ax.set_title('Distribution Overlay')
    ax.legend(fontsize=8)

    # QQ plot (A vs B)
    ax = axes[1]
    n_points = min(len(fa), len(fb), 2000)
    probs = np.linspace(0.1, 99.9, n_points)
    qa = np.percentile(fa, probs)
    qb = np.percentile(fb, probs)
    ax.scatter(qa, qb, s=2, alpha=0.4, color='steelblue')
    lims = [min(qa[0], qb[0]), max(qa[-1], qb[-1])]
    ax.plot(lims, lims, 'r--', linewidth=1)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title('QQ Plot')

    # Path-level scatter (if same seed)
    if same_seed and n_plots > 2:
        ax = axes[2]
        n_show = min(len(fa), len(fb), 5000)
        ax.scatter(fa[:n_show], fb[:n_show], s=1, alpha=0.1, color='steelblue')
        lims = [min(fa[:n_show].min(), fb[:n_show].min()),
                max(fa[:n_show].max(), fb[:n_show].max())]
        ax.plot(lims, lims, 'r--', linewidth=1)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title('Path-by-Path Scatter')

    plt.tight_layout()
    plt.show()
    plt.close()


# =============================================================================
# 9. FULL DIAGNOSTIC REPORT
# =============================================================================

def run_full_diagnostics(simulated, metadata, sim_benchmark=None,
                         benchmark_label='RiskFlow', plot=True):
    """
    Run all diagnostic tests on a simulation output.

    Parameters
    ----------
    simulated : np.ndarray, shape [timesteps, tenors, scenarios]
        Output from cs_simulation.run_simulation_from_json() or equivalent.
    metadata : dict
        Must contain keys: params, scen_time_grid, tenors_excel,
                          base_date_excel, prices.
    sim_benchmark : np.ndarray, optional
        A second simulation (e.g. RiskFlow output) for comparison.
    benchmark_label : str
        Label for the benchmark simulation.
    plot : bool
        Whether to generate plots.

    Returns
    -------
    results : dict with all diagnostic outputs
    """
    results = {}
    n_tenors = simulated.shape[1]

    print("\n" + "#" * 80)
    print("#  CS SIMULATION DIAGNOSTICS — FULL REPORT")
    print("#" * 80)
    print(f"#  Shape: {simulated.shape}  "
          f"[{simulated.shape[0]} timesteps, {simulated.shape[1]} tenors, "
          f"{simulated.shape[2]} scenarios]")
    print(f"#  Model params: Sigma={metadata['params']['Sigma']}, "
          f"Alpha={metadata['params']['Alpha']}, "
          f"Drift={metadata['params']['Drift']}")
    print("#" * 80 + "\n")

    # --- 1. Martingale test ---
    print("\n>>> [1/6] MARTINGALE TEST\n")
    results['martingale'] = martingale_test(simulated, metadata, plot=plot)

    # --- 2. Moment matching ---
    print("\n>>> [2/6] MOMENT MATCHING\n")
    results['log_moments'], results['price_moments'] = moment_matching(
        simulated, metadata, plot=plot)

    # --- 3. Tail analysis ---
    print("\n>>> [3/6] TAIL ANALYSIS\n")
    results['tails'] = {}
    tenor_indices = [0]
    if n_tenors > 1:
        tenor_indices.append(n_tenors - 1)
    for tidx in tenor_indices:
        print(f"\n  --- Tenor {tidx} ---")
        results['tails'][tidx] = tail_analysis(
            simulated, metadata, tenor_idx=tidx, plot=plot)

    # --- 4. Parameter recovery ---
    print("\n>>> [4/6] PARAMETER RECOVERY\n")
    results['recovery'] = parameter_recovery(simulated, metadata, plot=plot)

    # --- 5. Convergence ---
    print("\n>>> [5/6] CONVERGENCE ANALYSIS\n")
    results['convergence'] = convergence_analysis(
        simulated, metadata, tenor_idx=0, plot=plot)

    # --- 6. Standard error ---
    print("\n>>> [6/6] STANDARD ERROR ANALYSIS\n")
    results['standard_errors'] = standard_error_analysis(
        simulated, metadata, tenor_idx=0, plot=plot)

    # --- Bonus: comparison ---
    if sim_benchmark is not None:
        print(f"\n>>> [BONUS] CROSS-SIMULATION COMPARISON vs {benchmark_label}\n")
        results['comparison'] = {}
        for tidx in tenor_indices:
            print(f"\n  --- Tenor {tidx} ---")
            results['comparison'][tidx] = compare_simulations(
                simulated, sim_benchmark, metadata,
                labels=('Validation', benchmark_label),
                tenor_idx=tidx, plot=plot)

    print("\n" + "#" * 80)
    print("#  DIAGNOSTICS COMPLETE")
    print("#" * 80 + "\n")

    return results


# =============================================================================
# MAIN — EXAMPLE USAGE
# =============================================================================

if __name__ == '__main__':
    from cs_simulation import (run_simulation_standalone,
                               timestamp_to_excel_days)

    print("Running example simulation for diagnostics demo...\n")

    # Run a simulation with known parameters
    simulated, precalc, scen_time_grid = run_simulation_standalone(
        base_date_str='2024-01-15',
        delivery_dates_str=['2024-07-15', '2025-01-15', '2025-07-15', '2026-01-15'],
        forward_prices=[85.0, 83.0, 81.0, 80.0],
        sigma=0.35, alpha=1.2, drift_mu=0.02,
        num_scenarios=8192, use_antithetic=True,
    )

    # Build metadata dict (same structure as run_simulation_from_json returns)
    base_date = pd.Timestamp('2024-01-15')
    delivery_dates = [pd.Timestamp(d) for d in
                      ['2024-07-15', '2025-01-15', '2025-07-15', '2026-01-15']]

    metadata = {
        'params': {'Sigma': 0.35, 'Alpha': 1.2, 'Drift': 0.02},
        'scen_time_grid': scen_time_grid,
        'tenors_excel': np.array([timestamp_to_excel_days(d) for d in delivery_dates]),
        'base_date_excel': timestamp_to_excel_days(base_date),
        'prices': np.array([85.0, 83.0, 81.0, 80.0]),
    }

    # Run all diagnostics
    results = run_full_diagnostics(simulated, metadata, plot=True)
