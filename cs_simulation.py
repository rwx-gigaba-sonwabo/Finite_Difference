"""
==============================================================================
CS FORWARD PRICE SIMULATION
==============================================================================

PURPOSE:
    Replicates the Monte Carlo simulation mechanics of RiskFlow for the
    Clewlow-Strickland commodity forward price model.

    This script covers BOTH model variants:
        - CSForwardPriceModel (historical/P-measure with drift)
        - CSImpliedForwardPriceModel (implied/Q-measure, drift = 0)

    The simulation code is IDENTICAL for both — the only difference is the
    parameters fed in (sigma, alpha, drift).

    This corresponds to:
        - riskflow/stochasticprocess.py   CSForwardPriceModel.precalculate() + generate()
        - riskflow/calculation.py         CMC_State.reset() + get_cholesky_decomp()

WHAT THIS SCRIPT DOES:
    1. precalculate():
       - Takes the initial forward curve F(0,T) and model params (sigma, alpha, drift)
       - Computes time-to-delivery for each tenor at each simulation timestep
       - Computes the incremental volatility and cumulative drift tensors
       - These are pre-computed ONCE and reused across all MC batches

    2. generate():
       - Takes correlated random numbers (from Cholesky decomposition)
       - Scales them by the pre-computed volatility structure
       - Applies cumulative sum to build the stochastic integral
       - Multiplies by the initial curve to get simulated forward prices

    3. Full pipeline:
       - Build correlation matrix + Cholesky decomposition
       - Generate correlated random normals
       - Simulate forward curves across time and scenarios

THE MODEL (final form):
    F(t,T) = F(0,T) * exp( drift(t,T) + cumsum(vol(t,T) * Z(t)) )

    where:
        drift(t,T)  = mu * t - 0.5 * sigma^2 * exp(-2*alpha*(T-t)) * v(t)
        vol(t,T)    = incremental standard deviation at each timestep
        v(t)        = (1 - exp(-2*alpha*t)) / (2*alpha)    [OU variance]
        Z(t)        = correlated standard normal draws
==============================================================================
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

# Days in a year — matches riskflow/utils.py DAYS_IN_YEAR
DAYS_IN_YEAR = 365.25


def precalculate(initial_curve, tenors_in_days, scen_time_grid_days,
                 sigma, alpha, drift, base_date_excel, use_implied=False):
    """
    Replicates riskflow/stochasticprocess.py CSForwardPriceModel.precalculate() (line 910).

    Pre-computes the volatility and drift tensors that will be used in generate().

    STEP-BY-STEP EXPLANATION:

    1. COMPUTE TIME-TO-DELIVERY (tenors):
       For each simulation time t and each forward tenor T:
           tenors[t, T] = max(T - t, 0) / DAYS_IN_YEAR
       This gives the remaining time-to-delivery in years.
       When t > T (we've passed the delivery date), it clips to 0.

    2. COMPUTE dt (time increments):
       For each timestep, we need the time increment in years.
       But there's a subtlety: for a given tenor, once the simulation time
       passes its delivery date, the tenor is "dead" and shouldn't accumulate
       more variance. The delta/clip logic handles this.

    3. COMPUTE VARIANCE (var):
       var[t, T] = sigma^2 * exp(-2*alpha * tenors[t,T]) * v(t)

       where v(t) = (1 - exp(-2*alpha * cumsum(dt))) / (2*alpha)

       This is the CUMULATIVE log-variance of F(t, T) from time 0 to time t.
       - exp(-2*alpha * tenors): Samuelson decay — less variance for distant deliveries
       - v(t): OU variance accumulation — variance grows but saturates

    4. COMPUTE INCREMENTAL VOLATILITY (vol):
       vol[t] = sqrt( var[t] - var[t-1] )

       We need incremental (not cumulative) volatility because generate()
       uses cumsum of (vol * Z) to build paths.

       For the implied branch, negative incremental variance can occur
       (numerical issue). RiskFlow handles this by clamping to zero.

    5. COMPUTE DRIFT:
       Historical:  drift[t,T] = mu * cumsum(dt) - 0.5 * var[t,T]
       Implied:     drift[t,T] = -0.5 * var[t,T]    (mu = 0)

       The -0.5 * var term is the Jensen's inequality correction that ensures
       E[F(t,T)] = F(0,T) * exp(mu * t) under the lognormal dynamics.

    Parameters
    ----------
    initial_curve : np.ndarray, shape [num_tenors]
        F(0, T_i) — today's forward prices at each delivery date.
    tenors_in_days : np.ndarray, shape [num_tenors]
        Delivery dates as excel day numbers (absolute, not relative).
    scen_time_grid_days : np.ndarray, shape [num_timesteps]
        Simulation time points in days relative to base_date.
    sigma : float
        OU process volatility parameter.
    alpha : float
        Mean reversion speed.
    drift : float
        Drift rate (0 for implied model).
    base_date_excel : int
        Base date as excel day number.
    use_implied : bool
        If True, use torch tensors (differentiable). If False, use numpy.

    Returns
    -------
    dict with keys:
        'initial_curve': shape [1, num_tenors, 1]
        'vol':           shape [num_timesteps, num_tenors, 1]
        'drift':         shape [num_timesteps, num_tenors, 1]
    """
    num_timesteps = len(scen_time_grid_days)
    num_tenors = len(tenors_in_days)

    # ---- Step 1: Compute time-to-delivery at each simulation time ----
    # Convert absolute tenor dates to relative (from base_date)
    excel_date_time_grid = scen_time_grid_days + base_date_excel

    # tenors[t, T] = (delivery_date - simulation_date) in years, clipped at 0
    tenors = (tenors_in_days.reshape(1, -1) -
              excel_date_time_grid.reshape(-1, 1)).clip(0.0, np.inf) / DAYS_IN_YEAR

    # ---- Step 2: Compute time increments (dt) ----
    # This logic ensures that once a tenor's delivery date has passed,
    # it no longer accumulates variance.
    tenor_rel = tenors_in_days - base_date_excel
    delta = tenor_rel.reshape(1, -1).clip(
        scen_time_grid_days[:-1].reshape(-1, 1),
        scen_time_grid_days[1:].reshape(-1, 1)
    ) - scen_time_grid_days[:-1].reshape(-1, 1)
    dt = np.insert(delta, 0, 0, axis=0) / DAYS_IN_YEAR

    # ---- Step 3: Compute cumulative variance ----
    # v(t) = (1 - exp(-2*alpha*t)) / (2*alpha)  — the OU integrated variance
    cumulative_dt = dt.cumsum(axis=0)

    if not use_implied:
        # HISTORICAL (numpy) branch
        var_adj = (1.0 - np.exp(-2.0 * alpha * cumulative_dt)) / (2.0 * alpha)
        var = np.square(sigma) * np.exp(-2.0 * alpha * tenors) * var_adj

        # ---- Step 4: Incremental volatility ----
        vol = np.sqrt(np.diff(np.insert(var, 0, 0, axis=0), axis=0))

        # ---- Step 5: Drift ----
        drift_tensor = drift * cumulative_dt - 0.5 * var

        return {
            'initial_curve': initial_curve.reshape(1, -1, 1),
            'vol': np.expand_dims(vol, axis=2),
            'drift': np.expand_dims(drift_tensor, axis=2),
        }
    else:
        # IMPLIED (torch) branch — matches the `else` in precalculate
        t_cumulative_dt = torch.tensor(cumulative_dt, dtype=torch.float64)
        t_tenors = torch.tensor(tenors, dtype=torch.float64)
        t_sigma = torch.tensor(sigma, dtype=torch.float64, requires_grad=True)
        t_alpha = torch.tensor(alpha, dtype=torch.float64, requires_grad=True)

        var_adj = (1.0 - torch.exp(-2.0 * t_alpha * t_cumulative_dt)) / (2.0 * t_alpha)
        var = torch.square(t_sigma) * torch.exp(-2.0 * t_alpha * t_tenors) * var_adj

        # Incremental variance with safety clamp
        delta_var = torch.diff(torch.nn.functional.pad(var, [0, 0, 1, 0]), dim=0)
        safe_delta = torch.where(delta_var > 0.0, delta_var, torch.ones_like(delta_var))
        vol = torch.where(delta_var > 0.0, torch.sqrt(safe_delta), torch.zeros_like(delta_var))

        drift_tensor = -0.5 * var  # no mu term for implied

        return {
            'initial_curve': torch.tensor(initial_curve.reshape(1, -1, 1), dtype=torch.float64),
            'vol': torch.unsqueeze(vol, dim=2),
            'drift': torch.unsqueeze(drift_tensor, dim=2),
            # Keep references for gradient computation
            '_sigma_tensor': t_sigma,
            '_alpha_tensor': t_alpha,
        }


def build_cholesky(correlation_dict, factor_names):
    """
    Replicates riskflow/calculation.py get_cholesky_decomp() (line 820).

    Builds the Cholesky decomposition of the correlation matrix for all
    stochastic factors. This is used to generate correlated random draws.

    INTUITION:
        If we have N independent standard normals Z, then X = L * Z gives
        us N correlated normals where L * L^T = Sigma (correlation matrix).

        Example with 2 factors (BRENT and WTI) with correlation 0.85:
            Sigma = [[1.0, 0.85],
                     [0.85, 1.0]]
            L = cholesky(Sigma)

        Then X = L @ Z produces correlated draws for BRENT and WTI.

    Parameters
    ----------
    correlation_dict : dict
        Maps (factor1, factor2) -> correlation value.
        E.g., {('BRENT', 'WTI'): 0.85}
    factor_names : list of str
        Ordered list of factor names.

    Returns
    -------
    L : np.ndarray, shape [N, N]
        Lower-triangular Cholesky factor.
    """
    N = len(factor_names)
    corr_matrix = np.eye(N, dtype=np.float64)

    for i in range(N):
        for j in range(i + 1, N):
            key = (factor_names[i], factor_names[j])
            alt_key = (factor_names[j], factor_names[i])
            rho = correlation_dict.get(key, correlation_dict.get(alt_key, 0.0))
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho

    # ---- Heal if not positive definite (eigenvalue raising) ----
    eigval, eigvec = np.linalg.eig(corr_matrix)
    eigval, eigvec = np.real(eigval), np.real(eigvec)

    if (eigval < 1e-8).any():
        print("  WARNING: Correlation matrix not positive definite — raising eigenvalues")
        P_plus_B = eigvec @ np.diag(np.maximum(eigval, 1e-4)) @ eigvec.T
        diag_norm = np.diag(1.0 / np.sqrt(P_plus_B.diagonal()))
        corr_matrix = diag_norm @ P_plus_B @ diag_norm

    L = np.linalg.cholesky(corr_matrix)
    return L


def generate_random_numbers(cholesky_L, num_timesteps, batch_size, use_antithetic=False):
    """
    Replicates riskflow/calculation.py CMC_State.reset() (line 405).

    Generates correlated random numbers for all factors, all timesteps,
    and all scenarios in one shot.

    INTUITION:
        1. Draw independent standard normals: Z ~ N(0, I)
           Shape: [num_factors, num_timesteps * batch_size]

        2. Correlate them: X = L @ Z
           This gives us correlated draws across factors.

        3. Reshape to: [num_factors, num_timesteps, batch_size]
           So X[f, t, s] = correlated normal for factor f, time t, scenario s.

    ANTITHETIC SAMPLING:
        If enabled, we generate batch_size/2 draws and mirror them:
            X_full = [X, -X]
        This reduces variance because every "up" path has a matching "down" path.

    Parameters
    ----------
    cholesky_L : np.ndarray, shape [num_factors, num_factors]
    num_timesteps : int
    batch_size : int
    use_antithetic : bool

    Returns
    -------
    np.ndarray, shape [num_factors, num_timesteps, batch_size]
    """
    num_factors = cholesky_L.shape[0]
    sample_size = batch_size // 2 if use_antithetic else batch_size

    # Independent standard normals
    Z = np.random.randn(num_factors, num_timesteps * sample_size)

    # Correlate
    X = cholesky_L @ Z

    # Reshape
    X = X.reshape(num_factors, num_timesteps, sample_size)

    if use_antithetic:
        X = np.concatenate([X, -X], axis=2)

    return X


def generate_paths(precalc, random_numbers, factor_index=0):
    """
    Replicates riskflow/stochasticprocess.py CSForwardPriceModel.generate() (line 952).

    THIS IS THE CORE SIMULATION STEP.

    INTUITION:
        Starting from the initial forward curve F(0, T), we build simulated
        paths using the pre-computed drift and volatility:

        F(t, T) = F(0, T) * exp( drift[t, T] + cumsum_over_t( vol[t, T] * Z[t] ) )

        Step by step:
        1. Extract this factor's random draws: Z[t, scenario]
        2. Scale by tenor-dependent volatility: vol[t, tenor] * Z[t, scenario]
           → Shape: [timesteps, tenors, scenarios]
        3. Cumulative sum over time: this builds the stochastic integral
           ∫ sigma(s,T) dW(s) ≈ sum of vol_i * Z_i
        4. Add drift and exponentiate
        5. Multiply by initial curve

    Parameters
    ----------
    precalc : dict
        Output from precalculate().
    random_numbers : np.ndarray, shape [num_factors, num_timesteps, batch_size]
    factor_index : int
        Which row in random_numbers belongs to this factor.

    Returns
    -------
    np.ndarray, shape [num_timesteps, num_tenors, batch_size]
        Simulated forward curve surface.
    """
    initial_curve = precalc['initial_curve']
    vol = precalc['vol']
    drift = precalc['drift']

    num_timesteps = vol.shape[0]

    # Extract this factor's random draws and add a tenor dimension
    # Z shape: [timesteps, scenarios] → [timesteps, 1, scenarios]
    Z = random_numbers[factor_index, :num_timesteps, :]
    Z = np.expand_dims(Z, axis=1)

    # Scale by the volatility structure
    # vol: [timesteps, tenors, 1] * Z: [timesteps, 1, scenarios]
    # → z_portion: [timesteps, tenors, scenarios]
    z_portion = vol * Z

    # Cumulative sum over time dimension: builds the stochastic integral
    stoch_integral = np.cumsum(z_portion, axis=0)

    # Final forward prices: F(0,T) * exp(drift + stochastic_integral)
    simulated = initial_curve * np.exp(drift + stoch_integral)

    return simulated


# ============================================================================
# FULL SIMULATION PIPELINE
# ============================================================================

def run_simulation(model_type='historical',
                   sigma=0.35, alpha=1.0, drift_mu=0.02,
                   num_scenarios=4096, num_batches=1,
                   use_antithetic=False, random_seed=42):
    """
    Runs a complete CS forward price simulation, replicating the full
    RiskFlow pipeline from precalculate through to scenario generation.

    Parameters
    ----------
    model_type : str
        'historical' for CSForwardPriceModel, 'implied' for CSImpliedForwardPriceModel.
    sigma, alpha, drift_mu : float
        Model parameters.
    num_scenarios : int
        Number of MC scenarios per batch.
    num_batches : int
        Number of simulation batches (RiskFlow loops over these).
    use_antithetic : bool
        Whether to use antithetic sampling.
    random_seed : int
        For reproducibility.
    """
    np.random.seed(random_seed)

    print("=" * 70)
    print(f"CLEWLOW-STRICKLAND SIMULATION — {model_type.upper()} MODEL")
    print("=" * 70)

    # ---- 1. Define the initial forward curve ----
    # These are today's observed forward prices at different delivery dates.
    # In RiskFlow, these come from Price Factors -> ForwardPrice.COMMODITY -> Curve
    # The tenors are stored as excel day numbers (absolute dates).

    base_date_excel = 44927  # e.g., 2023-01-02 as excel serial number
    # Delivery dates: 3m, 6m, 9m, 12m, 18m, 24m from now
    delivery_offsets_days = np.array([91, 182, 274, 365, 548, 730])
    tenors_in_days = base_date_excel + delivery_offsets_days
    delivery_years = delivery_offsets_days / DAYS_IN_YEAR

    # Forward prices (typical backwardated commodity curve)
    initial_curve = np.array([85.0, 83.0, 81.5, 80.0, 78.0, 76.0])

    print(f"\n  Initial forward curve:")
    for dy, fp in zip(delivery_years, initial_curve):
        print(f"    T = {dy:.2f}y:  F(0,T) = ${fp:.2f}")

    # ---- 2. Define the simulation time grid ----
    # In RiskFlow, this comes from the 'Time_grid' parameter, e.g.
    # '0d 2d 1w(1w) 1m(1m) 3m(3m)'
    # For simplicity, we use monthly steps for 2 years.

    time_grid_days = np.array([0] + list(range(30, 731, 30)))  # 0, 30, 60, ..., 730 days
    num_timesteps = len(time_grid_days)
    print(f"\n  Simulation time grid: {num_timesteps} points, 0 to {time_grid_days[-1]} days")

    # ---- 3. Set model parameters ----
    if model_type == 'implied':
        drift_mu = 0.0  # risk-neutral: no drift
        use_implied_tensors = True
    else:
        use_implied_tensors = False

    print(f"\n  Model parameters:")
    print(f"    Sigma = {sigma}")
    print(f"    Alpha = {alpha}")
    print(f"    Drift = {drift_mu}")

    # ---- 4. Precalculate (done ONCE before any simulation batch) ----
    print(f"\n  Running precalculate()...")
    precalc = precalculate(
        initial_curve, tenors_in_days, time_grid_days,
        sigma, alpha, drift_mu, base_date_excel,
        use_implied=False  # use numpy for simplicity in validation
    )

    print(f"    vol shape:   {precalc['vol'].shape}   = [timesteps, tenors, 1]")
    print(f"    drift shape: {precalc['drift'].shape} = [timesteps, tenors, 1]")

    # ---- 5. Build correlation matrix + Cholesky ----
    # For a single-factor simulation, this is just [[1.0]] → L = [[1.0]]
    # But we show the full machinery for when you have multiple commodities.

    factor_names = ['BRENT']
    correlations = {}  # no cross-correlations for single factor
    L = build_cholesky(correlations, factor_names)
    print(f"\n  Cholesky matrix L (single factor): {L}")

    # ---- 6. Simulation loop (over batches) ----
    all_results = []

    for batch in range(num_batches):
        # Generate correlated random numbers
        random_numbers = generate_random_numbers(
            L, num_timesteps, num_scenarios, use_antithetic=use_antithetic)

        # Generate forward price paths
        simulated = generate_paths(precalc, random_numbers, factor_index=0)
        all_results.append(simulated)

        if batch == 0:
            print(f"\n  Batch {batch}: simulated shape = {simulated.shape}")
            print(f"    = [{num_timesteps} timesteps, {len(initial_curve)} tenors, {num_scenarios} scenarios]")

    # ---- 7. Aggregate results ----
    # In RiskFlow, results from multiple batches are concatenated along the scenario axis
    all_simulated = np.concatenate(all_results, axis=2)
    total_scenarios = all_simulated.shape[2]

    print(f"\n  Total simulated scenarios: {total_scenarios}")

    # ---- 8. Analyse results ----
    print(f"\n  VALIDATION: Checking simulated forward prices at final timestep")
    print(f"  {'Tenor':>8s}  {'F(0,T)':>8s}  {'E[F(t,T)]':>10s}  {'Std[F]':>10s}  "
          f"{'Theo Mean':>10s}  {'Theo Std':>10s}")
    print("  " + "-" * 68)

    final_t_idx = -1  # last timestep
    t_final = time_grid_days[-1] / DAYS_IN_YEAR

    for tenor_idx in range(len(initial_curve)):
        F0 = initial_curve[tenor_idx]
        sim_F = all_simulated[final_t_idx, tenor_idx, :]

        sim_mean = np.mean(sim_F)
        sim_std = np.std(sim_F)

        # Theoretical moments
        Tmt = max(delivery_years[tenor_idx] - t_final, 0.0)
        ln_var = sigma ** 2 * np.exp(-2.0 * alpha * Tmt) * \
                 (1.0 - np.exp(-2.0 * alpha * t_final)) / (2.0 * alpha)
        theo_mean = F0 * np.exp(drift_mu * t_final + 0.5 * ln_var)
        # This is E[F] for lognormal: E[F] * sqrt(exp(ln_var) - 1)
        theo_std = theo_mean * np.sqrt(np.exp(ln_var) - 1.0)

        print(f"  {delivery_years[tenor_idx]:8.2f}  {F0:8.2f}  {sim_mean:10.4f}  "
              f"{sim_std:10.4f}  {theo_mean:10.4f}  {theo_std:10.4f}")

    # ---- 9. Plot sample paths ----
    plot_results(all_simulated, time_grid_days, delivery_years, initial_curve, model_type)

    return all_simulated, precalc


def plot_results(simulated, time_grid_days, delivery_years, initial_curve, model_type):
    """Plot a selection of simulated paths for visual inspection."""
    num_paths_to_show = 50
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'CS {model_type.title()} Model — Simulated Forward Prices', fontsize=14)

    for idx, ax in enumerate(axes.flat):
        if idx >= len(delivery_years):
            ax.set_visible(False)
            continue

        time_years = time_grid_days / DAYS_IN_YEAR

        # Plot sample paths
        for s in range(min(num_paths_to_show, simulated.shape[2])):
            ax.plot(time_years, simulated[:, idx, s], alpha=0.15, color='steelblue', linewidth=0.5)

        # Plot mean
        mean_path = simulated[:, idx, :].mean(axis=1)
        ax.plot(time_years, mean_path, color='red', linewidth=2, label='Mean')

        # Plot percentiles
        p5 = np.percentile(simulated[:, idx, :], 5, axis=1)
        p95 = np.percentile(simulated[:, idx, :], 95, axis=1)
        ax.fill_between(time_years, p5, p95, alpha=0.15, color='red', label='5-95%')

        ax.axhline(y=initial_curve[idx], color='black', linestyle='--', alpha=0.5, label='F(0,T)')
        ax.set_title(f'T = {delivery_years[idx]:.2f}y')
        ax.set_xlabel('Simulation time (years)')
        ax.set_ylabel('Forward price')
        if idx == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f'cs_{model_type}_simulation.png', dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved: cs_{model_type}_simulation.png")
    plt.close()


# ============================================================================
# MULTI-FACTOR EXAMPLE (multiple correlated commodities)
# ============================================================================

def run_multi_factor_simulation():
    """
    Demonstrates how RiskFlow simulates MULTIPLE correlated commodities.

    In RiskFlow, the correlation matrix is built from the 'Correlations'
    section of the market data file. Each CS process registers its
    correlation_name as ('ClewlowStricklandProcess', factor_name).
    """
    np.random.seed(123)

    print("\n" + "=" * 70)
    print("MULTI-FACTOR CORRELATED SIMULATION (BRENT + WTI)")
    print("=" * 70)

    base_date_excel = 44927
    delivery_offsets = np.array([91, 182, 365])
    tenors_brent = base_date_excel + delivery_offsets
    tenors_wti = base_date_excel + delivery_offsets
    delivery_years = delivery_offsets / DAYS_IN_YEAR

    curve_brent = np.array([85.0, 83.0, 80.0])
    curve_wti = np.array([78.0, 76.0, 73.0])

    time_grid = np.array([0] + list(range(30, 366, 30)))

    # Different parameters for each commodity
    params_brent = {'sigma': 0.35, 'alpha': 1.2, 'drift': 0.0}
    params_wti = {'sigma': 0.40, 'alpha': 0.8, 'drift': 0.0}

    # Precalculate for each
    precalc_brent = precalculate(
        curve_brent, tenors_brent, time_grid,
        params_brent['sigma'], params_brent['alpha'], params_brent['drift'],
        base_date_excel)

    precalc_wti = precalculate(
        curve_wti, tenors_wti, time_grid,
        params_wti['sigma'], params_wti['alpha'], params_wti['drift'],
        base_date_excel)

    # Build correlation matrix with rho = 0.85 between BRENT and WTI
    factor_names = ['BRENT', 'WTI']
    correlations = {('BRENT', 'WTI'): 0.85}
    L = build_cholesky(correlations, factor_names)

    print(f"\n  Correlation matrix:")
    print(f"    BRENT-WTI: 0.85")
    print(f"\n  Cholesky L:")
    print(f"    {L}")

    # Generate correlated random numbers
    num_scenarios = 4096
    random_numbers = generate_random_numbers(
        L, len(time_grid), num_scenarios, use_antithetic=True)

    # Simulate both commodities using the SAME random number block
    sim_brent = generate_paths(precalc_brent, random_numbers, factor_index=0)
    sim_wti = generate_paths(precalc_wti, random_numbers, factor_index=1)

    # Verify correlation is preserved
    # Pick the 3-month tenor at the 6-month horizon
    t_idx = 6  # ~6 months
    tenor_idx = 0  # 3-month delivery
    brent_samples = sim_brent[t_idx, tenor_idx, :]
    wti_samples = sim_wti[t_idx, tenor_idx, :]
    realized_corr = np.corrcoef(brent_samples, wti_samples)[0, 1]

    print(f"\n  Realised correlation at t=6m, 3m-tenor: {realized_corr:.4f}")
    print(f"  Target correlation:                      0.85")
    print(f"  Correlation error:                       {abs(realized_corr - 0.85):.4f}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Run historical (P-measure) simulation
    sim_hist, precalc_hist = run_simulation(
        model_type='historical',
        sigma=0.35, alpha=1.0, drift_mu=0.02,
        num_scenarios=4096, num_batches=1,
        use_antithetic=True, random_seed=42
    )

    print("\n\n")

    # Run implied (Q-measure / risk-neutral) simulation
    sim_impl, precalc_impl = run_simulation(
        model_type='implied',
        sigma=0.30, alpha=1.2, drift_mu=0.0,  # drift ignored for implied
        num_scenarios=4096, num_batches=1,
        use_antithetic=True, random_seed=42
    )

    print("\n\n")

    # Run multi-factor correlated simulation
    run_multi_factor_simulation()
