"""
==============================================================================
CS FORWARD PRICE SIMULATION — USING RISKFLOW'S EXACT MECHANICS
==============================================================================

PURPOSE:
    Replicates the Monte Carlo simulation mechanics of RiskFlow for the
    Clewlow-Strickland commodity forward price model.

    This script covers BOTH model variants:
        - CSForwardPriceModel (historical/P-measure with drift)
        - CSImpliedForwardPriceModel (implied/Q-measure, drift = 0)

    This corresponds to:
        - riskflow/config.py             Context.parse_grid()     — time grid parsing
        - riskflow/utils.py              TimeGrid.set_base_date() — date conversion
        - riskflow/riskfactors.py        ForwardPrice             — curve loading
        - riskflow/stochasticprocess.py  CSForwardPriceModel.precalculate() + generate()
        - riskflow/calculation.py        CMC_State.reset() + get_cholesky_decomp()

HOW TO USE:
    1. Point `json_path` at your CVAMarketData JSON file
    2. Set the `factor_name` to the ForwardPrice factor you want to simulate
       e.g. 'ForwardPrice.BRENT.OIL'
    3. Run the script — it will:
       a) Load the forward curve and model params from the JSON
       b) Parse the time grid string exactly as RiskFlow does
       c) Convert dates to excel serial numbers exactly as RiskFlow does
       d) Run precalculate() and generate() to produce simulated paths

THE MODEL (final form):
    F(t,T) = F(0,T) * exp( drift(t,T) + cumsum(vol(t,T) * Z(t)) )

    where:
        drift(t,T)  = mu * t - 0.5 * sigma^2 * exp(-2*alpha*(T-t)) * v(t)
        vol(t,T)    = incremental standard deviation at each timestep
        v(t)        = (1 - exp(-2*alpha*t)) / (2*alpha)    [OU variance]
        Z(t)        = correlated standard normal draws
==============================================================================
"""

import json
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# CONSTANTS — must match riskflow/utils.py exactly
# =============================================================================

# riskflow/utils.py line 40
DAYS_IN_YEAR = 365.25

# riskflow/utils.py line 33 — Excel epoch used for all date conversions
EXCEL_OFFSET = pd.Timestamp('1899-12-30 00:00:00')

# Unit mapping for grid parser — riskflow/config.py line 184
OFFSET_LOOKUP = {'M': 'months', 'D': 'days', 'Y': 'years', 'W': 'weeks'}


# =============================================================================
# DATE UTILITIES — replicating riskflow/utils.py and riskflow/config.py
# =============================================================================

def timestamp_to_excel_days(ts):
    """
    Convert a pandas Timestamp to an Excel serial day number.

    Replicates the pattern used throughout RiskFlow:
        excel_offset = (ref_date - utils.excel_offset).days

    See: riskflow/stochasticprocess.py line 916
         riskflow/riskfactors.py line 751
         riskflow/utils.py line 33

    Parameters
    ----------
    ts : pd.Timestamp
        The date to convert.

    Returns
    -------
    int : Excel serial day number (days since 1899-12-30).
    """
    return (ts - EXCEL_OFFSET).days


def excel_days_to_timestamp(excel_days):
    """
    Convert an Excel serial day number back to a pandas Timestamp.

    This is the inverse of timestamp_to_excel_days().
    """
    return EXCEL_OFFSET + pd.Timedelta(days=int(excel_days))


# =============================================================================
# TIME GRID PARSER — replicating riskflow/config.py parse_grid()
# =============================================================================

def parse_time_grid(run_date, max_date, grid_string):
    """
    Parse a RiskFlow time grid string and return sorted day offsets from run_date.

    Replicates riskflow/config.py Context.parse_grid() (line 218) and
    riskflow/utils.py TimeGrid.set_base_date() (line 564).

    HOW RISKFLOW GRID STRINGS WORK:
        A grid string like '0d 2d 1w(1w) 1m(1m) 3m(3m)' defines simulation dates:

        '0d'        → A single date: run_date + 0 days (i.e. today)
        '2d'        → A single date: run_date + 2 days
        '1w(1w)'    → Starting at run_date + 1 week, repeat every 1 week
                       until the NEXT segment's start date
        '1m(1m)'    → Starting at run_date + 1 month, repeat every 1 month
                       until the next segment's start
        '3m(3m)'    → Starting at run_date + 3 months, repeat every 3 months
                       until max_date

        Each segment runs until the next segment takes over, or until max_date.

    WHAT THIS FUNCTION RETURNS:
        A 1D numpy array of integers = day offsets from run_date.
        This is exactly what RiskFlow stores as TimeGrid.scen_time_grid.

    Parameters
    ----------
    run_date : pd.Timestamp
        The valuation / base date.
    max_date : pd.Timestamp
        The latest date the grid should reach (typically the last deal maturity).
    grid_string : str
        RiskFlow grid string, e.g. '0d 2d 1w(1w) 1m(1m) 3m(3m)'.

    Returns
    -------
    np.ndarray of int : Sorted day offsets from run_date (scen_time_grid).
    """
    # ---- Step 1: Parse the grid string into segments ----
    # Each segment is either:
    #   'Xd'       → single offset
    #   'Xm(Ym)'   → start offset with repeating increment
    segments = grid_string.strip().split()
    parsed_segments = []

    for seg in segments:
        if '(' in seg:
            # Repeating segment: e.g. '1m(1m)' or '3m(3m)'
            start_str, repeat_str = seg.split('(')
            repeat_str = repeat_str.rstrip(')')
            start_offset = _parse_offset(start_str)
            repeat_offset = _parse_offset(repeat_str)
            parsed_segments.append((start_offset, repeat_offset))
        else:
            # Single date segment: e.g. '0d' or '2d'
            parsed_segments.append((_parse_offset(seg), None))

    # ---- Step 2: Generate dates exactly as parse_grid() does ----
    # Add a sentinel at the end (Timestamp.max)
    fixed_dates = [(run_date + seg[0], seg[1]) for seg in parsed_segments]
    fixed_dates.append((pd.Timestamp.max, None))

    dates = set()
    finish = False

    for (date_rule, repeat), (next_start, _) in zip(fixed_dates[:-1], fixed_dates[1:]):
        next_date = date_rule
        if next_date > max_date:
            break
        else:
            dates.add(next_date)

        if repeat:
            while True:
                next_date = next_date + repeat
                if next_date > max_date:
                    finish = True
                    break
                if next_date > next_start:
                    break
                dates.add(next_date)

        if finish:
            break

    # ---- Step 3: Convert to day offsets from run_date ----
    # This is what TimeGrid.set_base_date() does:
    #   self.scen_time_grid = np.array([(x - base_date).days for x in sorted(self.scenario_dates)])
    scen_time_grid = np.array(sorted([(d - run_date).days for d in dates]))

    return scen_time_grid


def _parse_offset(s):
    """
    Parse a single offset string like '2d', '1m', '3m', '1w', '2y' into a pd.DateOffset.

    Replicates the grammar actions in riskflow/config.py get_grid_grammar() (line 47).
    Handles compound offsets like '1y3m' by accumulating multiple unit-value pairs.
    """
    import re
    # Find all (number, unit) pairs in the string
    pairs = re.findall(r'(\d+)([dDmMwWyY])', s)
    if not pairs:
        raise ValueError(f"Cannot parse offset: '{s}'")

    kwargs = {}
    for value, unit in pairs:
        key = OFFSET_LOOKUP[unit.upper()]
        kwargs[key] = kwargs.get(key, 0) + int(value)

    return pd.DateOffset(**kwargs)


# =============================================================================
# JSON MARKET DATA LOADER — replicating riskflow/config.py parse_json()
# =============================================================================

def load_market_data(json_path):
    """
    Load a RiskFlow CVAMarketData JSON file and extract the relevant sections.

    Replicates riskflow/config.py Context.parse_json() (line 702).

    The JSON structure for forward prices is:
        {
            "MarketData": {
                "Price Factors": {
                    "ForwardPrice.BRENT.OIL": {
                        "Currency": "USD",
                        "Curve": {".Curve": {"meta": [], "data": [[excel_date, price], ...]}}
                    }
                },
                "Price Models": {
                    "CSForwardPriceModel.BRENT.OIL": {
                        "Sigma": 0.35,
                        "Alpha": 1.0,
                        "Drift": 0.02
                    }
                },
                "Model Configuration": "..."
            }
        }

    The .Curve object stores tenors as EXCEL SERIAL DAY NUMBERS and values as
    forward prices. See riskflow/utils.py Curve class (line 288).

    Parameters
    ----------
    json_path : str
        Path to the CVAMarketData JSON file.

    Returns
    -------
    dict with keys: 'Price Factors', 'Price Models', 'Model Configuration',
                    'Correlations', 'Valuation Configuration'
    """

    def as_internal(dct):
        """
        JSON object_hook that converts RiskFlow's custom types.

        Replicates config.py line 704-729.
        We only handle .Curve here since that's what ForwardPrice uses.
        The .Curve data is a list of [excel_date, value] pairs.
        """
        if '.Curve' in dct:
            meta = dct['.Curve']['meta']
            data = dct['.Curve']['data']
            # Sort by tenor (first column) and convert to numpy array
            # This is what utils.Curve.__init__() does
            return {'_type': 'Curve', 'meta': meta, 'array': np.array(sorted(data))}
        elif '.Percent' in dct:
            return dct['.Percent'] / 100.0
        elif '.Timestamp' in dct:
            return pd.Timestamp(dct['.Timestamp'])
        elif '.DateOffset' in dct:
            return pd.DateOffset(**dct['.DateOffset'])
        return dct

    with open(json_path, 'rt') as f:
        data = json.load(f, object_hook=as_internal)

    if 'MarketData' in data:
        return data['MarketData']
    return data


def extract_forward_curve(market_data, factor_name):
    """
    Extract the forward price curve for a given factor from the market data.

    Replicates how riskflow/riskfactors.py ForwardPrice (line 734) loads its data:
        - self.tenors = Curve.array[:, 0]    — excel serial day numbers
        - current_value() = Curve.array[:, 1] — forward prices

    Also replicates Factor1D.get_tenor() (line 83) which deduplicates tenors.

    Parameters
    ----------
    market_data : dict
        Output from load_market_data().
    factor_name : str
        Full factor name, e.g. 'ForwardPrice.BRENT.OIL'.

    Returns
    -------
    tenors_excel : np.ndarray
        Delivery dates as excel serial day numbers (column 0 of Curve).
    prices : np.ndarray
        Forward prices at each delivery date (column 1 of Curve).
    currency : str
        The currency of this factor.
    """
    factor_data = market_data['Price Factors'][factor_name]
    curve = factor_data['Curve']

    if isinstance(curve, dict) and '_type' in curve and curve['_type'] == 'Curve':
        arr = curve['array']
    else:
        # If loaded without our object_hook (raw data)
        arr = np.array(sorted(curve))

    # Deduplicate tenors — replicates Factor1D.get_tenor() line 88-91
    tenors = np.unique(arr[:, 0])
    prices = np.interp(tenors, arr[:, 0], arr[:, 1])

    currency = factor_data.get('Currency', 'USD')
    return tenors, prices, currency


def extract_model_params(market_data, factor_name):
    """
    Extract CS model parameters from Price Models or Price Factors (for implied).

    For historical models, looks in:
        market_data['Price Models']['CSForwardPriceModel.<name>']

    For implied models, looks in:
        market_data['Price Factors']['CSForwardPriceModelParameters.<name>']

    Replicates:
        - riskflow/calculation.py update_factors() line 627-629 (historical)
        - riskflow/stochasticprocess.py CSImpliedForwardPriceModel.__init__() line 967 (implied)

    Parameters
    ----------
    market_data : dict
        Output from load_market_data().
    factor_name : str
        The ForwardPrice factor name, e.g. 'ForwardPrice.BRENT.OIL'.
        The function strips 'ForwardPrice.' to find the model.

    Returns
    -------
    params : dict with keys 'Sigma', 'Alpha', 'Drift'
    model_type : str, either 'historical' or 'implied'
    """
    # Strip 'ForwardPrice.' prefix to get the commodity name
    # e.g. 'ForwardPrice.BRENT.OIL' → 'BRENT.OIL'
    commodity_name = factor_name.replace('ForwardPrice.', '')

    # Check Model Configuration to determine which model is used
    model_config = market_data.get('Model Configuration', {})

    # Check if this factor uses an implied model
    # In RiskFlow, Model Configuration maps factor types to model classes:
    #   'ForwardPrice' → 'CSImpliedForwardPriceModel' or 'CSForwardPriceModel'
    forward_price_model = None
    if isinstance(model_config, dict):
        forward_price_model = model_config.get('ForwardPrice')

    # Try implied first
    implied_key = f'CSForwardPriceModelParameters.{commodity_name}'
    historical_key = f'CSForwardPriceModel.{commodity_name}'

    if forward_price_model == 'CSImpliedForwardPriceModel' or \
       implied_key in market_data.get('Price Factors', {}):
        # IMPLIED MODEL — params from Price Factors
        # CSImpliedForwardPriceModel.__init__() line 967:
        #   self.param = {'Drift': 0.0, 'Sigma': implied_factor.param['Sigma'],
        #                 'Alpha': implied_factor.param['Alpha']}
        implied_data = market_data['Price Factors'].get(implied_key, {})
        return {
            'Sigma': implied_data.get('Sigma', 0.3),
            'Alpha': implied_data.get('Alpha', 1.0),
            'Drift': 0.0,  # Always 0 for implied (line 967)
        }, 'implied'

    # HISTORICAL MODEL — params from Price Models
    if historical_key in market_data.get('Price Models', {}):
        hist_data = market_data['Price Models'][historical_key]
        return {
            'Sigma': hist_data.get('Sigma', 0.3),
            'Alpha': hist_data.get('Alpha', 1.0),
            'Drift': hist_data.get('Drift', 0.0),
        }, 'historical'

    raise KeyError(f"No model parameters found for '{commodity_name}' "
                   f"in Price Models or Price Factors")


def extract_correlations(market_data):
    """
    Extract correlation data from the market data JSON.

    In RiskFlow, correlations are stored as:
        market_data['Correlations'] = {
            'factor1': {'factor2': rho, ...},
            ...
        }

    For CS models, the correlation key is:
        ('ClewlowStricklandProcess', factor_name)

    See: riskflow/config.py line 739

    Parameters
    ----------
    market_data : dict
        Output from load_market_data().

    Returns
    -------
    dict : Maps (name1, name2) → correlation value.
    """
    corr_section = market_data.get('Correlations', {})
    correlations = {}

    for rate1, rate_list in corr_section.items():
        if isinstance(rate_list, dict):
            for rate2, rho in rate_list.items():
                correlations[(rate1, rate2)] = rho

    return correlations


# =============================================================================
# PRECALCULATE — replicating stochasticprocess.py CSForwardPriceModel (line 910)
# =============================================================================

def precalculate(initial_curve, tenors_in_days, scen_time_grid_days,
                 sigma, alpha, drift, base_date_excel, use_implied=False):
    """
    Replicates riskflow/stochasticprocess.py CSForwardPriceModel.precalculate() (line 910).

    Pre-computes the volatility and drift tensors that will be used in generate().

    STEP-BY-STEP — exactly matching the RiskFlow source:

    1. REBASE DATES (line 916-918):
       excel_offset = (ref_date - utils.excel_offset).days
       excel_date_time_grid = time_grid.scen_time_grid + excel_offset

       This converts the scenario day-offsets to absolute excel day numbers
       so they can be subtracted from the tenor dates (also excel day numbers).

    2. COMPUTE TIME-TO-DELIVERY (line 919-920):
       tenors = (factor.get_tenor() - excel_date_time_grid).clip(0) / DAYS_IN_YEAR

       For each simulation time t and each forward delivery date T:
           tenors[t, T] = max(T_excel - t_excel, 0) / 365.25
       When the simulation time passes the delivery date, it clips to 0.

    3. COMPUTE dt — TIME INCREMENTS (line 921-926):
       tenor_rel = factor.get_tenor() - excel_offset
       delta = tenor_rel.clip(scen_time_grid[:-1], scen_time_grid[1:]) - scen_time_grid[:-1]
       dt = np.insert(delta, 0, 0, axis=0) / DAYS_IN_YEAR

       This is the EFFECTIVE time step for each tenor at each timestep.
       Once a tenor's delivery date passes, its dt becomes 0 — no more
       variance accumulation. The clip ensures each tenor only accumulates
       time within the window [scen_time_grid[i], scen_time_grid[i+1]].

    4. COMPUTE VARIANCE (line 930-931):
       var_adj = (1 - exp(-2*alpha * cumsum(dt))) / (2*alpha)
       var = sigma^2 * exp(-2*alpha * tenors) * var_adj

    5. COMPUTE INCREMENTAL VOL (line 933):
       vol = sqrt(diff(insert(var, 0, 0)))

    6. COMPUTE DRIFT (line 935):
       drift = mu * cumsum(dt) - 0.5 * var

    Parameters
    ----------
    initial_curve : np.ndarray, shape [num_tenors]
        F(0, T_i) — today's forward prices at each delivery date.
    tenors_in_days : np.ndarray, shape [num_tenors]
        Delivery dates as excel day numbers (absolute).
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
    # ---- Step 1: Rebase dates to excel format (line 916-918) ----
    excel_date_time_grid = scen_time_grid_days + base_date_excel

    # ---- Step 2: Time-to-delivery at each simulation time (line 919-920) ----
    tenors = (tenors_in_days.reshape(1, -1) -
              excel_date_time_grid.reshape(-1, 1)).clip(0.0, np.inf) / DAYS_IN_YEAR

    # ---- Step 3: Time increments with tenor clipping (line 921-926) ----
    tenor_rel = tenors_in_days - base_date_excel
    delta = tenor_rel.reshape(1, -1).clip(
        scen_time_grid_days[:-1].reshape(-1, 1),
        scen_time_grid_days[1:].reshape(-1, 1)
    ) - scen_time_grid_days[:-1].reshape(-1, 1)
    dt = np.insert(delta, 0, 0, axis=0) / DAYS_IN_YEAR

    # ---- Step 4-6: Variance, vol, drift ----
    cumulative_dt = dt.cumsum(axis=0)

    if not use_implied:
        # HISTORICAL (numpy) branch — line 928-935
        var_adj = (1.0 - np.exp(-2.0 * alpha * cumulative_dt)) / (2.0 * alpha)
        var = np.square(sigma) * np.exp(-2.0 * alpha * tenors) * var_adj

        vol = np.sqrt(np.diff(np.insert(var, 0, 0, axis=0), axis=0))

        drift_tensor = drift * cumulative_dt - 0.5 * var

        return {
            'initial_curve': initial_curve.reshape(1, -1, 1),
            'vol': np.expand_dims(vol, axis=2),
            'drift': np.expand_dims(drift_tensor, axis=2),
        }
    else:
        # IMPLIED (torch) branch — line 937-946
        t_cumulative_dt = torch.tensor(cumulative_dt, dtype=torch.float64)
        t_tenors = torch.tensor(tenors, dtype=torch.float64)
        t_sigma = torch.tensor(sigma, dtype=torch.float64, requires_grad=True)
        t_alpha = torch.tensor(alpha, dtype=torch.float64, requires_grad=True)

        var_adj = (1.0 - torch.exp(-2.0 * t_alpha * t_cumulative_dt)) / (2.0 * t_alpha)
        var = torch.square(t_sigma) * torch.exp(-2.0 * t_alpha * t_tenors) * var_adj

        delta_var = torch.diff(torch.nn.functional.pad(var, [0, 0, 1, 0]), dim=0)
        safe_delta = torch.where(delta_var > 0.0, delta_var, torch.ones_like(delta_var))
        vol = torch.where(delta_var > 0.0, torch.sqrt(safe_delta), torch.zeros_like(delta_var))

        drift_tensor = -0.5 * var

        return {
            'initial_curve': torch.tensor(initial_curve.reshape(1, -1, 1), dtype=torch.float64),
            'vol': torch.unsqueeze(vol, dim=2),
            'drift': torch.unsqueeze(drift_tensor, dim=2),
            '_sigma_tensor': t_sigma,
            '_alpha_tensor': t_alpha,
        }


# =============================================================================
# CHOLESKY — replicating calculation.py get_cholesky_decomp() (line 820)
# =============================================================================

def build_cholesky(correlation_dict, factor_names):
    """
    Replicates riskflow/calculation.py get_cholesky_decomp() (line 820).

    Builds the Cholesky decomposition of the correlation matrix.

    If we have N independent standard normals Z, then X = L @ Z gives
    us N correlated normals where L @ L^T = Sigma (correlation matrix).

    Includes eigenvalue healing for non-positive-definite matrices.
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

    eigval, eigvec = np.linalg.eig(corr_matrix)
    eigval, eigvec = np.real(eigval), np.real(eigvec)

    if (eigval < 1e-8).any():
        print("  WARNING: Correlation matrix not positive definite - raising eigenvalues")
        P_plus_B = eigvec @ np.diag(np.maximum(eigval, 1e-4)) @ eigvec.T
        diag_norm = np.diag(1.0 / np.sqrt(P_plus_B.diagonal()))
        corr_matrix = diag_norm @ P_plus_B @ diag_norm

    L = np.linalg.cholesky(corr_matrix)
    return L


# =============================================================================
# RANDOM NUMBER GENERATION — replicating calculation.py CMC_State.reset() (line 405)
# =============================================================================

def generate_random_numbers(cholesky_L, num_timesteps, batch_size, use_antithetic=False):
    """
    Replicates riskflow/calculation.py CMC_State.reset() (line 405).

    1. Draw independent standard normals Z ~ N(0, I)
    2. Correlate them: X = L @ Z
    3. Reshape to [num_factors, num_timesteps, batch_size]

    If antithetic sampling is enabled, generates batch_size/2 draws
    and mirrors them: X_full = [X, -X].
    """
    num_factors = cholesky_L.shape[0]
    sample_size = batch_size // 2 if use_antithetic else batch_size

    Z = np.random.randn(num_factors, num_timesteps * sample_size)
    X = cholesky_L @ Z
    X = X.reshape(num_factors, num_timesteps, sample_size)

    if use_antithetic:
        X = np.concatenate([X, -X], axis=2)

    return X


# =============================================================================
# PATH GENERATION — replicating stochasticprocess.py CSForwardPriceModel.generate() (line 952)
# =============================================================================

def generate_paths(precalc, random_numbers, factor_index=0):
    """
    Replicates riskflow/stochasticprocess.py CSForwardPriceModel.generate() (line 952).

    The RiskFlow source is:
        z_portion = unsqueeze(shared_mem.t_random_numbers[self.z_offset, :self.scenario_horizon], dim=1) * self.vol
        return self.initial_curve * torch.exp(self.drift + torch.cumsum(z_portion, dim=0))

    Step by step:
        1. Extract this factor's random draws Z[t, scenario]
        2. Add a tenor dimension: Z → [timesteps, 1, scenarios]
        3. Scale by volatility: vol[t, tenor, 1] * Z[t, 1, scenario] → [t, tenor, scenario]
        4. Cumulative sum over time → stochastic integral
        5. F(t,T) = F(0,T) * exp(drift + cumsum(vol * Z))
    """
    initial_curve = precalc['initial_curve']
    vol = precalc['vol']
    drift = precalc['drift']

    num_timesteps = vol.shape[0]

    Z = random_numbers[factor_index, :num_timesteps, :]
    Z = np.expand_dims(Z, axis=1)

    z_portion = vol * Z
    stoch_integral = np.cumsum(z_portion, axis=0)

    simulated = initial_curve * np.exp(drift + stoch_integral)
    return simulated


# =============================================================================
# FULL SIMULATION PIPELINE — from JSON to simulated paths
# =============================================================================

def run_simulation_from_json(json_path, factor_name,
                             time_grid_string=None, max_date=None,
                             num_scenarios=4096, num_batches=1,
                             use_antithetic=True, random_seed=42):
    """
    Run a complete CS simulation using data loaded from a RiskFlow JSON file.

    This is the main entry point that ties everything together, replicating:
        1. config.parse_json()                    → load market data
        2. riskfactors.ForwardPrice.__init__()     → extract curve
        3. config.parse_grid()                     → build time grid
        4. TimeGrid.set_base_date()                → convert to day offsets
        5. CSForwardPriceModel.precalculate()       → pre-compute vol/drift
        6. CMC_State.reset()                        → generate random numbers
        7. CSForwardPriceModel.generate()            → simulate paths

    Parameters
    ----------
    json_path : str
        Path to CVAMarketData JSON file.
    factor_name : str
        ForwardPrice factor name, e.g. 'ForwardPrice.BRENT.OIL'.
    time_grid_string : str, optional
        RiskFlow time grid string. If None, uses the one from
        Valuation Configuration in the JSON.
    max_date : pd.Timestamp, optional
        Last simulation date. If None, uses last tenor delivery date.
    num_scenarios : int
        Number of MC scenarios per batch.
    num_batches : int
        Number of simulation batches.
    use_antithetic : bool
        Whether to use antithetic sampling.
    random_seed : int
        For reproducibility.

    Returns
    -------
    simulated : np.ndarray, shape [timesteps, tenors, total_scenarios]
    precalc : dict
    metadata : dict with run info
    """
    np.random.seed(random_seed)

    print("=" * 70)
    print("CS SIMULATION — LOADING FROM RISKFLOW JSON")
    print("=" * 70)

    # ---- 1. Load market data from JSON ----
    print(f"\n  Loading: {json_path}")
    market_data = load_market_data(json_path)

    # ---- 2. Extract forward curve ----
    tenors_excel, prices, currency = extract_forward_curve(market_data, factor_name)
    print(f"\n  Factor: {factor_name}")
    print(f"  Currency: {currency}")
    print(f"  Curve points: {len(tenors_excel)}")

    # ---- 3. Extract model parameters ----
    params, model_type = extract_model_params(market_data, factor_name)
    print(f"\n  Model type: {model_type}")
    print(f"  Sigma = {params['Sigma']}")
    print(f"  Alpha = {params['Alpha']}")
    print(f"  Drift = {params['Drift']}")

    # ---- 4. Determine base date ----
    # In RiskFlow this comes from Valuation Configuration -> Run_Date
    val_config = market_data.get('Valuation Configuration', {})
    if isinstance(val_config, dict):
        base_date = val_config.get('Run_Date')
        if base_date is None:
            base_date = val_config.get('Base_Date')
    else:
        base_date = None

    if base_date is None:
        # Fall back: infer from the first tenor (assume it's in the future)
        base_date = excel_days_to_timestamp(tenors_excel[0] - 90)
        print(f"\n  WARNING: No Run_Date found in JSON, inferring base_date = {base_date.date()}")
    elif isinstance(base_date, str):
        base_date = pd.Timestamp(base_date)

    base_date_excel = timestamp_to_excel_days(base_date)
    print(f"\n  Base date: {base_date.date()} (excel: {base_date_excel})")

    # ---- 5. Parse the time grid ----
    if time_grid_string is None:
        # Try to get from Valuation Configuration
        if isinstance(val_config, dict):
            time_grid_string = val_config.get('Time_grid', val_config.get('Tenor'))
        if time_grid_string is None:
            time_grid_string = '0d 2d 1w(1w) 1m(1m) 3m(3m)'
            print(f"  WARNING: No Time_grid in JSON, using default: '{time_grid_string}'")

    if max_date is None:
        max_date = excel_days_to_timestamp(tenors_excel[-1])

    scen_time_grid = parse_time_grid(base_date, max_date, time_grid_string)
    print(f"\n  Time grid string: '{time_grid_string}'")
    print(f"  Max date: {max_date.date()}")
    print(f"  Scenario time grid: {len(scen_time_grid)} points")
    print(f"    First 10 day offsets: {scen_time_grid[:10]}")
    print(f"    Last 5 day offsets:   {scen_time_grid[-5:]}")

    # Show the forward curve with delivery dates
    print(f"\n  Initial forward curve:")
    print(f"    {'Excel Date':>12s}  {'Calendar Date':>14s}  {'Days Fwd':>10s}  {'Years Fwd':>10s}  {'Price':>10s}")
    print("    " + "-" * 64)
    for t_excel, p in zip(tenors_excel, prices):
        t_date = excel_days_to_timestamp(t_excel)
        days_fwd = t_excel - base_date_excel
        years_fwd = days_fwd / DAYS_IN_YEAR
        print(f"    {t_excel:12.0f}  {str(t_date.date()):>14s}  {days_fwd:10.0f}  {years_fwd:10.3f}  {p:10.4f}")

    # ---- 6. Precalculate ----
    print(f"\n  Running precalculate()...")
    precalc = precalculate(
        prices, tenors_excel, scen_time_grid,
        params['Sigma'], params['Alpha'], params['Drift'],
        base_date_excel, use_implied=(model_type == 'implied')
    )
    print(f"    vol shape:   {precalc['vol'].shape}")
    print(f"    drift shape: {precalc['drift'].shape}")

    # ---- 7. Build Cholesky (single factor = identity) ----
    factor_names = [factor_name]
    L = build_cholesky({}, factor_names)

    # ---- 8. Simulation loop ----
    num_timesteps = len(scen_time_grid)
    all_results = []

    for batch in range(num_batches):
        random_numbers = generate_random_numbers(
            L, num_timesteps, num_scenarios, use_antithetic=use_antithetic)
        simulated = generate_paths(precalc, random_numbers, factor_index=0)
        all_results.append(simulated)

        if batch == 0:
            print(f"\n  Batch 0: shape = {simulated.shape} "
                  f"[{num_timesteps} times, {len(prices)} tenors, {num_scenarios} scenarios]")

    all_simulated = np.concatenate(all_results, axis=2)
    total_scenarios = all_simulated.shape[2]
    print(f"\n  Total scenarios: {total_scenarios}")

    # ---- 9. Validation output ----
    print(f"\n  VALIDATION: Simulated vs Theoretical moments at final timestep")
    t_final = scen_time_grid[-1] / DAYS_IN_YEAR
    print(f"  (t_final = {t_final:.3f} years = day {scen_time_grid[-1]})")
    print(f"\n  {'Delivery':>10s}  {'F(0,T)':>10s}  {'E[F] sim':>10s}  {'Std sim':>10s}  "
          f"{'E[F] theo':>10s}  {'Std theo':>10s}")
    print("  " + "-" * 68)

    for tenor_idx in range(len(prices)):
        F0 = prices[tenor_idx]
        sim_F = all_simulated[-1, tenor_idx, :]
        sim_mean = np.mean(sim_F)
        sim_std = np.std(sim_F)

        # Theoretical: Tmt = remaining time-to-delivery at t_final
        Tmt = max((tenors_excel[tenor_idx] - base_date_excel) / DAYS_IN_YEAR - t_final, 0.0)
        ln_var = params['Sigma'] ** 2 * np.exp(-2.0 * params['Alpha'] * Tmt) * \
                 (1.0 - np.exp(-2.0 * params['Alpha'] * t_final)) / (2.0 * params['Alpha'])
        theo_mean = F0 * np.exp(params['Drift'] * t_final + 0.5 * ln_var)
        theo_std = theo_mean * np.sqrt(max(np.exp(ln_var) - 1.0, 0.0))

        delivery_years = (tenors_excel[tenor_idx] - base_date_excel) / DAYS_IN_YEAR
        print(f"  {delivery_years:10.3f}  {F0:10.4f}  {sim_mean:10.4f}  {sim_std:10.4f}  "
              f"{theo_mean:10.4f}  {theo_std:10.4f}")

    print("=" * 70)

    metadata = {
        'factor_name': factor_name,
        'model_type': model_type,
        'params': params,
        'base_date': base_date,
        'base_date_excel': base_date_excel,
        'time_grid_string': time_grid_string,
        'scen_time_grid': scen_time_grid,
        'tenors_excel': tenors_excel,
        'prices': prices,
        'currency': currency,
    }

    return all_simulated, precalc, metadata


# =============================================================================
# STANDALONE SIMULATION (no JSON needed)
# =============================================================================

def run_simulation_standalone(base_date_str, delivery_dates_str, forward_prices,
                              sigma, alpha, drift_mu,
                              time_grid_string='0d 2d 1w(1w) 1m(1m) 3m(3m)',
                              max_date_str=None,
                              model_type='historical',
                              num_scenarios=4096, use_antithetic=True, random_seed=42):
    """
    Run a simulation with manually specified parameters, still using RiskFlow's
    exact time grid parsing and date handling.

    Parameters
    ----------
    base_date_str : str
        Valuation date, e.g. '2024-01-15'.
    delivery_dates_str : list of str
        Delivery dates, e.g. ['2024-04-15', '2024-07-15', '2025-01-15'].
    forward_prices : list of float
        Forward prices at each delivery date.
    sigma, alpha, drift_mu : float
        CS model parameters.
    time_grid_string : str
        RiskFlow grid string.
    max_date_str : str, optional
        Max simulation date. Defaults to last delivery date.
    model_type : str
        'historical' or 'implied'.
    """
    np.random.seed(random_seed)

    base_date = pd.Timestamp(base_date_str)
    base_date_excel = timestamp_to_excel_days(base_date)

    delivery_dates = [pd.Timestamp(d) for d in delivery_dates_str]
    tenors_excel = np.array([timestamp_to_excel_days(d) for d in delivery_dates], dtype=np.float64)
    prices = np.array(forward_prices, dtype=np.float64)

    if model_type == 'implied':
        drift_mu = 0.0

    max_date = pd.Timestamp(max_date_str) if max_date_str else delivery_dates[-1]

    print("=" * 70)
    print(f"CS SIMULATION (STANDALONE) - {model_type.upper()} MODEL")
    print("=" * 70)
    print(f"\n  Base date:  {base_date.date()} (excel: {base_date_excel})")
    print(f"  Sigma={sigma}, Alpha={alpha}, Drift={drift_mu}")

    # Parse time grid using RiskFlow's exact logic
    scen_time_grid = parse_time_grid(base_date, max_date, time_grid_string)
    print(f"\n  Time grid: '{time_grid_string}'")
    print(f"  Parsed {len(scen_time_grid)} scenario dates")
    print(f"  Day offsets: {scen_time_grid[:15]}{'...' if len(scen_time_grid) > 15 else ''}")

    # Show curve
    print(f"\n  Forward curve:")
    for t_excel, d, p in zip(tenors_excel, delivery_dates, prices):
        days_fwd = t_excel - base_date_excel
        print(f"    {d.date()}  (excel={t_excel:.0f}, +{days_fwd:.0f}d = "
              f"{days_fwd/DAYS_IN_YEAR:.3f}y)  F={p:.4f}")

    # Precalculate
    precalc = precalculate(
        prices, tenors_excel, scen_time_grid,
        sigma, alpha, drift_mu, base_date_excel, use_implied=False)

    # Simulate
    L = build_cholesky({}, ['factor'])
    num_timesteps = len(scen_time_grid)
    random_numbers = generate_random_numbers(L, num_timesteps, num_scenarios, use_antithetic)
    simulated = generate_paths(precalc, random_numbers, factor_index=0)

    print(f"\n  Simulated: {simulated.shape} [timesteps, tenors, scenarios]")

    # Validation
    t_final = scen_time_grid[-1] / DAYS_IN_YEAR
    print(f"\n  Moments at t_final = {t_final:.3f}y:")
    print(f"  {'Delivery':>12s}  {'F(0,T)':>8s}  {'E[F]':>10s}  {'Std[F]':>10s}  "
          f"{'Theo E[F]':>10s}  {'Theo Std':>10s}")
    print("  " + "-" * 68)

    for idx in range(len(prices)):
        F0 = prices[idx]
        sim_F = simulated[-1, idx, :]
        Tmt = max((tenors_excel[idx] - base_date_excel) / DAYS_IN_YEAR - t_final, 0.0)
        ln_var = sigma**2 * np.exp(-2.0 * alpha * Tmt) * \
                 (1.0 - np.exp(-2.0 * alpha * t_final)) / (2.0 * alpha)
        theo_mean = F0 * np.exp(drift_mu * t_final + 0.5 * ln_var)
        theo_std = theo_mean * np.sqrt(max(np.exp(ln_var) - 1.0, 0.0))

        print(f"  {str(delivery_dates[idx].date()):>12s}  {F0:8.2f}  {np.mean(sim_F):10.4f}  "
              f"{np.std(sim_F):10.4f}  {theo_mean:10.4f}  {theo_std:10.4f}")

    print("=" * 70)
    return simulated, precalc, scen_time_grid


# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(simulated, scen_time_grid, tenors_excel, prices, base_date_excel,
                 model_type='historical', num_paths=50, save_path=None):
    """Plot simulated forward price paths for visual inspection."""
    num_tenors = len(prices)
    cols = min(3, num_tenors)
    rows = (num_tenors + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if num_tenors == 1:
        axes = np.array([axes])
    axes = axes.flat

    fig.suptitle(f'CS {model_type.title()} Model - Simulated Forward Prices', fontsize=14)
    time_years = scen_time_grid / DAYS_IN_YEAR

    for idx in range(num_tenors):
        ax = axes[idx]
        delivery_years = (tenors_excel[idx] - base_date_excel) / DAYS_IN_YEAR

        for s in range(min(num_paths, simulated.shape[2])):
            ax.plot(time_years, simulated[:, idx, s], alpha=0.15, color='steelblue', linewidth=0.5)

        mean_path = simulated[:, idx, :].mean(axis=1)
        ax.plot(time_years, mean_path, color='red', linewidth=2, label='Mean')

        p5 = np.percentile(simulated[:, idx, :], 5, axis=1)
        p95 = np.percentile(simulated[:, idx, :], 95, axis=1)
        ax.fill_between(time_years, p5, p95, alpha=0.15, color='red', label='5-95%')
        ax.axhline(y=prices[idx], color='black', linestyle='--', alpha=0.5, label='F(0,T)')
        ax.set_title(f'Delivery = {delivery_years:.2f}y')
        ax.set_xlabel('Simulation time (years)')
        ax.set_ylabel('Forward price')
        if idx == 0:
            ax.legend(fontsize=8)

    # Hide unused axes
    for idx in range(num_tenors, len(list(axes))):
        axes[idx].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  Plot saved: {save_path}")
    plt.show()
    plt.close()


# =============================================================================
# MULTI-FACTOR SIMULATION (multiple correlated commodities from JSON)
# =============================================================================

def run_multi_factor_simulation_from_json(json_path, factor_names,
                                          time_grid_string=None, max_date=None,
                                          num_scenarios=4096, use_antithetic=True,
                                          random_seed=42):
    """
    Simulate multiple correlated commodity forward prices from a JSON file.

    Replicates the full RiskFlow pipeline for multiple stochastic factors:
        - Each factor gets its own precalculate() call
        - They share the same correlated random number block
        - Correlations come from the JSON's 'Correlations' section

    Parameters
    ----------
    json_path : str
        Path to CVAMarketData JSON file.
    factor_names : list of str
        ForwardPrice factor names, e.g. ['ForwardPrice.BRENT.OIL', 'ForwardPrice.WTI.OIL'].
    """
    np.random.seed(random_seed)

    print("=" * 70)
    print("MULTI-FACTOR CS SIMULATION FROM JSON")
    print("=" * 70)

    market_data = load_market_data(json_path)

    # Extract curves and params for each factor
    factor_data = {}
    for fname in factor_names:
        tenors, prices, currency = extract_forward_curve(market_data, fname)
        params, model_type = extract_model_params(market_data, fname)
        factor_data[fname] = {
            'tenors': tenors, 'prices': prices, 'currency': currency,
            'params': params, 'model_type': model_type,
        }
        print(f"\n  {fname}: {len(tenors)} tenors, {model_type}, "
              f"Sigma={params['Sigma']}, Alpha={params['Alpha']}")

    # Base date
    val_config = market_data.get('Valuation Configuration', {})
    base_date = val_config.get('Run_Date') if isinstance(val_config, dict) else None
    if base_date is None:
        base_date = excel_days_to_timestamp(
            min(fd['tenors'][0] for fd in factor_data.values()) - 90)
    elif isinstance(base_date, str):
        base_date = pd.Timestamp(base_date)

    base_date_excel = timestamp_to_excel_days(base_date)
    print(f"\n  Base date: {base_date.date()}")

    # Time grid
    if time_grid_string is None:
        if isinstance(val_config, dict):
            time_grid_string = val_config.get('Time_grid', '0d 2d 1w(1w) 1m(1m) 3m(3m)')
        else:
            time_grid_string = '0d 2d 1w(1w) 1m(1m) 3m(3m)'

    if max_date is None:
        max_date = excel_days_to_timestamp(
            max(fd['tenors'][-1] for fd in factor_data.values()))

    scen_time_grid = parse_time_grid(base_date, max_date, time_grid_string)
    num_timesteps = len(scen_time_grid)
    print(f"  Time grid: {num_timesteps} points, max day = {scen_time_grid[-1]}")

    # Precalculate for each factor
    precalcs = {}
    for fname, fd in factor_data.items():
        precalcs[fname] = precalculate(
            fd['prices'], fd['tenors'], scen_time_grid,
            fd['params']['Sigma'], fd['params']['Alpha'], fd['params']['Drift'],
            base_date_excel, use_implied=False)

    # Build correlation matrix
    correlations = extract_correlations(market_data)
    L = build_cholesky(correlations, factor_names)
    print(f"\n  Cholesky L:\n{L}")

    # Simulate
    random_numbers = generate_random_numbers(L, num_timesteps, num_scenarios, use_antithetic)

    results = {}
    for idx, fname in enumerate(factor_names):
        results[fname] = generate_paths(precalcs[fname], random_numbers, factor_index=idx)
        print(f"  {fname}: simulated shape = {results[fname].shape}")

    # Check cross-correlations
    if len(factor_names) >= 2:
        t_idx = min(6, num_timesteps - 1)
        f1 = results[factor_names[0]][t_idx, 0, :]
        f2 = results[factor_names[1]][t_idx, 0, :]
        rho_realized = np.corrcoef(f1, f2)[0, 1]
        rho_target = correlations.get(
            (factor_names[0], factor_names[1]),
            correlations.get((factor_names[1], factor_names[0]), 0.0))
        print(f"\n  Cross-correlation check at t_idx={t_idx}:")
        print(f"    Target:   {rho_target:.4f}")
        print(f"    Realized: {rho_realized:.4f}")

    print("=" * 70)
    return results, precalcs


# =============================================================================
# MAIN — EXAMPLE USAGE
# =============================================================================

if __name__ == '__main__':
    # ------------------------------------------------------------------
    # OPTION A: Run from a RiskFlow JSON file
    # ------------------------------------------------------------------
    # Uncomment and modify the path/factor to match your data:
    #
    # simulated, precalc, meta = run_simulation_from_json(
    #     json_path=r'path\to\CVAMarketData.json',
    #     factor_name='ForwardPrice.BRENT.OIL',
    #     time_grid_string=None,   # reads from JSON, or override here
    #     num_scenarios=4096,
    #     use_antithetic=True,
    # )
    #
    # plot_results(simulated, meta['scen_time_grid'], meta['tenors_excel'],
    #              meta['prices'], meta['base_date_excel'], meta['model_type'],
    #              save_path='cs_simulation_from_json.png')

    # ------------------------------------------------------------------
    # OPTION B: Run standalone with manual parameters
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXAMPLE: Standalone simulation with RiskFlow time grid parsing")
    print("=" * 70)

    simulated, precalc, scen_grid = run_simulation_standalone(
        base_date_str='2024-01-15',
        delivery_dates_str=[
            '2024-04-15',   # ~3 months
            '2024-07-15',   # ~6 months
            '2024-10-15',   # ~9 months
            '2025-01-15',   # ~12 months
            '2025-07-15',   # ~18 months
            '2026-01-15',   # ~24 months
        ],
        forward_prices=[85.0, 83.0, 81.5, 80.0, 78.0, 76.0],
        sigma=0.35,
        alpha=1.0,
        drift_mu=0.02,
        time_grid_string='0d 2d 1w(1w) 1m(1m) 3m(3m)',
        model_type='historical',
        num_scenarios=4096,
        use_antithetic=True,
        random_seed=42,
    )

    # Plot
    base_date = pd.Timestamp('2024-01-15')
    base_excel = timestamp_to_excel_days(base_date)
    delivery_dates = ['2024-04-15', '2024-07-15', '2024-10-15',
                      '2025-01-15', '2025-07-15', '2026-01-15']
    tenors_excel = np.array([timestamp_to_excel_days(pd.Timestamp(d)) for d in delivery_dates])
    prices = np.array([85.0, 83.0, 81.5, 80.0, 78.0, 76.0])

    plot_results(simulated, scen_grid, tenors_excel, prices, base_excel,
                 model_type='historical', save_path='cs_standalone_simulation.png')

    # ------------------------------------------------------------------
    # Show how the time grid parser works
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TIME GRID PARSING EXAMPLES")
    print("=" * 70)

    base = pd.Timestamp('2024-01-15')

    grids = [
        ('0d 2d 1w(1w) 1m(1m) 3m(3m)', pd.Timestamp('2026-01-15')),
        ('0d 1m(1m)', pd.Timestamp('2025-01-15')),
        ('0d 1w(1w)', pd.Timestamp('2024-04-15')),
        ('0d 3m(3m)', pd.Timestamp('2026-01-15')),
    ]

    for grid_str, max_dt in grids:
        days = parse_time_grid(base, max_dt, grid_str)
        print(f"\n  '{grid_str}' (max={max_dt.date()}):")
        print(f"    {len(days)} points, range: day {days[0]} to day {days[-1]}")
        # Show as calendar dates
        cal_dates = [str((base + pd.Timedelta(days=int(d))).date()) for d in days[:8]]
        print(f"    First dates: {', '.join(cal_dates)}{'...' if len(days) > 8 else ''}")
