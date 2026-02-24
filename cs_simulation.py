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

def _as_internal(dct):
    """
    JSON object_hook that converts RiskFlow's custom types.

    Replicates config.py line 704-729 — handles ALL custom types so that
    the JSON dict hierarchy is preserved correctly. The object_hook is
    called bottom-up on every dict in the JSON. If we don't handle a
    custom type (like .ModelParams), its dict passes through raw and can
    break the parent structure.
    """
    if '.Curve' in dct:
        # riskflow/utils.py Curve class (line 288)
        meta = dct['.Curve']['meta']
        data = dct['.Curve']['data']
        return {'_type': 'Curve', 'meta': meta, 'array': np.array(sorted(data))}
    elif '.Percent' in dct:
        return dct['.Percent'] / 100.0
    elif '.Basis' in dct:
        return dct['.Basis']
    elif '.Descriptor' in dct:
        return dct['.Descriptor']
    elif '.DateList' in dct:
        from collections import OrderedDict
        return OrderedDict([(pd.Timestamp(date), val) for date, val in dct['.DateList']])
    elif '.DateEqualList' in dct:
        return [[pd.Timestamp(values[0])] + values[1:] for values in dct['.DateEqualList']]
    elif '.CreditSupportList' in dct:
        return dct['.CreditSupportList']
    elif '.DateOffset' in dct:
        return pd.DateOffset(**dct['.DateOffset'])
    elif '.Offsets' in dct:
        return dct['.Offsets']
    elif '.Timestamp' in dct:
        return pd.Timestamp(dct['.Timestamp'])
    elif '.ModelParams' in dct:
        mp = dct['.ModelParams']
        return {'_type': 'ModelParams',
                'modeldefaults': mp.get('modeldefaults', {}),
                'modelfilters': mp.get('modelfilters', {})}
    elif '.Deal' in dct:
        return dct['.Deal']
    return dct


def _process_correlations(market_data):
    """Convert correlations from nested dict to (name1, name2) → rho format."""
    if 'Correlations' in market_data and isinstance(market_data['Correlations'], dict):
        correlations = {}
        for rate1, rate_list in market_data['Correlations'].items():
            if isinstance(rate_list, dict):
                for rate2, rho in rate_list.items():
                    correlations[(rate1, rate2)] = rho
        market_data['Correlations'] = correlations


def load_market_data(json_path):
    """
    Load market data from a RiskFlow JSON file.

    Handles TWO different JSON formats:

    FORMAT 1 — Standalone market data file (CVAMarketData.json):
        {
            "MarketData": {
                "Price Factors": { ... },
                "Price Models": { ... },
                "Model Configuration": { ... },
                "Correlations": { ... }
            }
        }

    FORMAT 2 — Deal/job file with embedded or referenced market data:
        {
            "Calc": {
                "MergeMarketData": {
                    "MarketDataFile": "path/to/base_market_data.json",
                    "ExplicitMarketData": {
                        "Price Factors": { ... },
                        "Price Models": { ... },
                        ...
                    }
                }
            }
        }

        In this format, ExplicitMarketData contains overrides that RiskFlow
        merges on top of a base market data file (MarketDataFile).
        See riskflow/__init__.py Context.load_json() line 302-331.

        This function loads the base file first, then applies the overrides,
        exactly as RiskFlow does.

    Parameters
    ----------
    json_path : str
        Path to either a standalone market data JSON or a deal/job JSON.

    Returns
    -------
    dict with keys: 'Price Factors', 'Price Models', 'Model Configuration',
                    'Correlations', 'Valuation Configuration', etc.
    """
    import os

    with open(json_path, 'rt') as f:
        data = json.load(f, object_hook=_as_internal)

    # ---- FORMAT 1: Standalone market data file ----
    if 'MarketData' in data:
        market_data = data['MarketData']
        _process_correlations(market_data)
        return market_data

    # ---- FORMAT 2: Deal/job file with Calc → MergeMarketData ----
    if 'Calc' in data and 'MergeMarketData' in data.get('Calc', {}):
        merge_section = data['Calc']['MergeMarketData']

        # Step 1: Load the base market data file (if referenced)
        # Replicates __init__.py line 315-316
        base_market_data_file = merge_section.get('MarketDataFile')
        base_params = {
            'Price Factors': {},
            'Price Models': {},
            'Model Configuration': {},
            'Correlations': {},
            'Valuation Configuration': {},
            'System Parameters': {},
            'Price Factor Interpolation': {},
        }

        if base_market_data_file:
            # Resolve relative path from the deal file's directory
            base_dir = os.path.dirname(os.path.abspath(json_path))
            base_path = os.path.join(base_dir, base_market_data_file)

            if os.path.exists(base_path):
                print(f"  Loading base market data: {base_path}")
                with open(base_path, 'rt') as f:
                    base_data = json.load(f, object_hook=_as_internal)
                if 'MarketData' in base_data:
                    base_params = base_data['MarketData']
                    _process_correlations(base_params)
            else:
                print(f"  WARNING: Base market data file not found: {base_path}")
                print(f"           Using only ExplicitMarketData overrides")

        # Step 2: Apply ExplicitMarketData overrides on top of the base
        # Replicates __init__.py line 330-331:
        #   for section, section_data in market_data['ExplicitMarketData'].items():
        #       cfg.params[section].update(section_data)
        explicit = merge_section.get('ExplicitMarketData', {})
        for section, section_data in explicit.items():
            if isinstance(section_data, dict):
                if section not in base_params:
                    base_params[section] = {}
                if isinstance(base_params[section], dict):
                    base_params[section].update(section_data)
                else:
                    base_params[section] = section_data
            else:
                base_params[section] = section_data

        # Step 3: Also grab valuation params from the Calc level if present
        for key in ['Valuation Configuration', 'System Parameters']:
            if key in data['Calc'] and isinstance(data['Calc'][key], dict):
                if key not in base_params:
                    base_params[key] = {}
                if isinstance(base_params[key], dict):
                    base_params[key].update(data['Calc'][key])

        return base_params

    # ---- Fallback: Price Factors at top level ----
    if 'Price Factors' in data:
        return data

    raise KeyError(
        f"Cannot find market data in JSON. "
        f"Top-level keys are: {list(data.keys())}"
    )


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

def generate_random_numbers(cholesky_L, num_timesteps, batch_size,
                            use_antithetic=False, dtype=torch.float64):
    """
    Replicates riskflow/calculation.py CMC_State.reset() (line 405).

    IMPORTANT: Uses torch.randn() (not np.random.randn) to match RiskFlow's
    exact random number sequence. RiskFlow generates all random numbers via
    PyTorch's RNG engine, so we must do the same for scenario-level reproducibility.

    Steps (matching reset() exactly):
        1. Draw independent standard normals: torch.randn(num_factors, N, dtype=dtype)
        2. Correlate: torch.matmul(cholesky, Z)
        3. Reshape to [num_factors, num_timesteps, batch_size]
        4. If antithetic: concat [X, -X] on scenario axis

    The caller must have set torch.manual_seed() before calling this function,
    exactly as RiskFlow does in CMC_State.__init__().

    Parameters
    ----------
    dtype : torch.dtype
        Precision for random number generation. RiskFlow defaults to torch.float32
        (see calculation.py Calculation.__init__ line 202: prec=torch.float32).
        Use torch.float32 for exact scenario-level matching with default RiskFlow.
        Use torch.float64 for higher-precision standalone validation.
    """
    num_factors = cholesky_L.shape[0]
    sample_size = batch_size // 2 if use_antithetic else batch_size

    # Convert Cholesky to torch tensor (matches RiskFlow's self.t_cholesky)
    t_cholesky = torch.tensor(cholesky_L, dtype=dtype)

    # Draw and correlate — exactly as CMC_State.reset() lines 408-412
    Z = torch.randn(num_factors, sample_size * num_timesteps, dtype=dtype)
    correlated = torch.matmul(t_cholesky, Z).reshape(num_factors, num_timesteps, -1)

    if use_antithetic:
        correlated = torch.concat([correlated, -correlated], dim=-1)

    # Return as numpy (float64) for downstream consumption
    return correlated.numpy().astype(np.float64)


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

    Handles both numpy (historical) and torch (implied) precalc tensors.
    When torch tensors are present (implied model), uses torch operations
    exactly as RiskFlow does, then detaches to numpy for the final output.
    """
    initial_curve = precalc['initial_curve']
    vol = precalc['vol']
    drift = precalc['drift']

    num_timesteps = vol.shape[0]

    use_torch = isinstance(vol, torch.Tensor)

    if use_torch:
        # Implied branch: vol/drift are torch tensors (possibly with requires_grad)
        # Replicate RiskFlow's generate() using all-torch operations
        Z = torch.tensor(random_numbers[factor_index, :num_timesteps, :],
                         dtype=torch.float64)
        Z = torch.unsqueeze(Z, dim=1)

        z_portion = vol * Z
        stoch_integral = torch.cumsum(z_portion, dim=0)

        simulated = initial_curve * torch.exp(drift + stoch_integral)
        return simulated.detach().numpy()
    else:
        # Historical branch: everything is numpy
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
                             batch_size=1024, simulation_batches=4,
                             use_antithetic=True, random_seed=42,
                             precision=torch.float64):
    """
    Run a complete CS simulation using data loaded from a RiskFlow JSON file.

    Replicates the EXACT RiskFlow Credit_Monte_Carlo batch loop:
        1. config.parse_json()                          → load market data
        2. riskfactors.ForwardPrice.__init__()           → extract curve
        3. config.parse_grid()                           → build time grid
        4. TimeGrid.set_base_date()                      → convert to day offsets
        5. CSForwardPriceModel.precalculate()             → pre-compute vol/drift
        6. FOR each batch in Simulation_Batches:          (calculation.py line 1053)
             CMC_State.reset()                            → fresh random numbers
             CSForwardPriceModel.generate()                → simulate batch_size paths
             append to scenario buffer
        7. np.concatenate(batches, axis=-1)               → join on scenario axis
        8. Build DataFrame with MultiIndex (tenor, scenario)

    Parameters
    ----------
    json_path : str
        Path to CVAMarketData JSON file.
    factor_name : str
        ForwardPrice factor name, e.g. 'ForwardPrice.GOLD'.
    time_grid_string : str, optional
        RiskFlow time grid string. If None, uses the one from the JSON.
    max_date : pd.Timestamp, optional
        Last simulation date. If None, uses last tenor delivery date.
    batch_size : int
        Number of MC scenarios per batch (RiskFlow: params['Batch_Size']).
    simulation_batches : int
        Number of batches (RiskFlow: params['Simulation_Batches']).
        Total scenarios = batch_size * simulation_batches.
    use_antithetic : bool
        Whether to use antithetic sampling.
    random_seed : int
        For reproducibility. Corresponds to params['Random_Seed'].

    Returns
    -------
    all_simulated : np.ndarray, shape [timesteps, tenors, total_scenarios]
        Raw simulation array. total_scenarios = batch_size * simulation_batches.
    scenario_df : pd.DataFrame
        RiskFlow-format output:
            - rows = MultiIndex (tenor, scenario)
              where tenor = excel day numbers of delivery dates,
              scenario = 0..total_scenarios-1
            - columns = scenario dates (pd.DatetimeIndex)
            - Transposed so .T gives dates-as-rows (CSV export format)
    metadata : dict with run info
    """
    # Seed PyTorch's RNG exactly as RiskFlow does in CMC_State.__init__() (line 352):
    #   torch.manual_seed(seed + job_id)
    # For single-job runs, job_id = 0, so this matches directly.
    torch.manual_seed(random_seed)
    total_scenarios = batch_size * simulation_batches

    print("=" * 70)
    print("CS SIMULATION — LOADING FROM RISKFLOW JSON")
    print("=" * 70)
    print(f"  Batch_Size: {batch_size}")
    print(f"  Simulation_Batches: {simulation_batches}")
    print(f"  Total scenarios: {total_scenarios}")

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
    val_config = market_data.get('Valuation Configuration', {})
    if isinstance(val_config, dict):
        base_date = val_config.get('Run_Date')
        if base_date is None:
            base_date = val_config.get('Base_Date')
    else:
        base_date = None

    if base_date is None:
        base_date = excel_days_to_timestamp(tenors_excel[0] - 90)
        print(f"\n  WARNING: No Run_Date found in JSON, inferring base_date = {base_date.date()}")
    elif isinstance(base_date, str):
        base_date = pd.Timestamp(base_date)

    base_date_excel = timestamp_to_excel_days(base_date)
    print(f"\n  Base date: {base_date.date()} (excel: {base_date_excel})")

    # ---- 5. Parse the time grid ----
    if time_grid_string is None:
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

    # Show the forward curve
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

    # ---- 8. Batch simulation loop ----
    # Replicates calculation.py line 1053-1061:
    #   for run in range(params['Simulation_Batches']):
    #       shared_mem.reset(...)        → fresh random numbers per batch
    #       for key, value in stoch_factors.items():
    #           shared_mem.t_Scenario_Buffer[key] = value.generate(shared_mem)
    #       output['scenarios'][key].append(buffer.numpy())
    #   values = np.concatenate(factor_values, axis=-1)  → join on scenario axis

    num_timesteps = len(scen_time_grid)
    batch_results = []

    for batch in range(simulation_batches):
        # Each batch gets FRESH random numbers (like CMC_State.reset())
        random_numbers = generate_random_numbers(
            L, num_timesteps, batch_size, use_antithetic=use_antithetic,
            dtype=precision)
        simulated = generate_paths(precalc, random_numbers, factor_index=0)
        batch_results.append(simulated)

        if batch == 0:
            print(f"\n  Batch 0: shape = {simulated.shape} "
                  f"[{num_timesteps} times, {len(prices)} tenors, {batch_size} scenarios]")

    # Concatenate batches on the scenario axis (axis=-1)
    # Replicates: values = np.concatenate(v, axis=-1)  (calculation.py line 951)
    all_simulated = np.concatenate(batch_results, axis=-1)
    print(f"\n  Concatenated: {all_simulated.shape} "
          f"[{num_timesteps} times, {len(prices)} tenors, {total_scenarios} scenarios]")

    # ---- 9. Build RiskFlow-format DataFrame ----
    # Replicates calculation.py report() lines 957-963:
    #   tenors = self.all_tenors[k][0].tenor               → excel day numbers
    #   columns = MultiIndex.from_product([tenors, range(N)], names=['tenor','scenario'])
    #   df = DataFrame(values.reshape(T, -1), index=scenario_date_index, columns=columns).T
    scenario_dates = pd.DatetimeIndex(
        sorted([base_date + pd.Timedelta(days=int(d)) for d in scen_time_grid]))

    columns = pd.MultiIndex.from_product(
        [tenors_excel, np.arange(total_scenarios)],
        names=['tenor', 'scenario'])

    # Reshape from [timesteps, tenors, scenarios] to [timesteps, tenors*scenarios]
    flat_values = all_simulated.reshape(num_timesteps, -1)

    scenario_df = pd.DataFrame(
        flat_values,
        index=scenario_dates[:num_timesteps],
        columns=columns
    ).T  # Transpose: rows = (tenor, scenario), columns = dates

    print(f"\n  DataFrame shape: {scenario_df.shape}")
    print(f"    Rows (tenor x scenario): {scenario_df.shape[0]}  "
          f"({len(tenors_excel)} tenors x {total_scenarios} scenarios)")
    print(f"    Columns (dates): {scenario_df.shape[1]}")

    # ---- 10. Quick validation ----
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

        Tmt = max((tenors_excel[tenor_idx] - base_date_excel) / DAYS_IN_YEAR - t_final, 0.0)
        ln_var = params['Sigma'] ** 2 * np.exp(-2.0 * params['Alpha'] * Tmt) * \
                 (1.0 - np.exp(-2.0 * params['Alpha'] * t_final)) / (2.0 * params['Alpha'])
        # E[F(t,T)] = F(0,T) * exp(mu*t) — the -0.5V in the drift cancels with +0.5V
        # from the MGF of the lognormal (eq 1.116 in FIS Theory Guide)
        theo_mean = F0 * np.exp(params['Drift'] * t_final)
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
        'batch_size': batch_size,
        'simulation_batches': simulation_batches,
        'total_scenarios': total_scenarios,
        'scenario_dates': scenario_dates,
    }

    return all_simulated, scenario_df, metadata


# =============================================================================
# RISKFLOW-FORMAT DATAFRAME UTILITIES
# =============================================================================

def to_riskflow_dataframe(simulated, metadata):
    """
    Convert a raw simulation array to a RiskFlow-format DataFrame.

    Replicates calculation.py report() lines 957-963:
        tenors = self.all_tenors[k][0].tenor
        columns = MultiIndex.from_product([tenors, range(N)], names=['tenor','scenario'])
        scen[factor_name] = DataFrame(
            values.reshape(T, -1), index=scenario_date_index, columns=columns).T

    The result has:
        - rows = MultiIndex (tenor, scenario)
            tenor = excel day number of each delivery date
            scenario = 0, 1, ..., total_scenarios-1
        - columns = scenario dates (DatetimeIndex)

    Parameters
    ----------
    simulated : np.ndarray, shape [timesteps, tenors, scenarios]
    metadata : dict — must contain tenors_excel, base_date, scen_time_grid

    Returns
    -------
    pd.DataFrame in RiskFlow scenario format
    """
    tenors_excel = metadata['tenors_excel']
    base_date = metadata['base_date']
    scen_time_grid = metadata['scen_time_grid']
    n_timesteps, _, n_scenarios = simulated.shape

    scenario_dates = pd.DatetimeIndex(
        sorted([base_date + pd.Timedelta(days=int(d)) for d in scen_time_grid]))

    columns = pd.MultiIndex.from_product(
        [tenors_excel, np.arange(n_scenarios)],
        names=['tenor', 'scenario'])

    flat_values = simulated.reshape(n_timesteps, -1)

    return pd.DataFrame(
        flat_values,
        index=scenario_dates[:n_timesteps],
        columns=columns
    ).T


def from_riskflow_dataframe(scenario_df, metadata=None):
    """
    Convert a RiskFlow-format scenario DataFrame back to a numpy array.

    Parameters
    ----------
    scenario_df : pd.DataFrame
        Rows = MultiIndex (tenor, scenario), columns = dates.
    metadata : dict, optional
        If provided, updates it with extracted info.

    Returns
    -------
    simulated : np.ndarray, shape [timesteps, tenors, scenarios]
    tenors : np.ndarray of tenor values
    scenario_dates : pd.DatetimeIndex
    """
    # Extract structure from MultiIndex
    tenors = scenario_df.index.get_level_values('tenor').unique().values
    scenarios = scenario_df.index.get_level_values('scenario').unique().values
    scenario_dates = scenario_df.columns

    n_tenors = len(tenors)
    n_scenarios = len(scenarios)
    n_timesteps = len(scenario_dates)

    # Transpose back: dates as rows, (tenor, scenario) as columns
    df_T = scenario_df.T  # shape: [timesteps, tenors*scenarios]

    # Reshape to [timesteps, tenors, scenarios]
    simulated = df_T.values.reshape(n_timesteps, n_tenors, n_scenarios)

    if metadata is not None:
        metadata['tenors_excel'] = tenors
        metadata['total_scenarios'] = n_scenarios
        metadata['scenario_dates'] = scenario_dates

    return simulated, tenors, scenario_dates


def export_scenarios_csv(scenario_df, filepath, factor_name=None):
    """
    Export scenario DataFrame to CSV in the same format RiskFlow produces.

    The CSV has:
        - First two columns: 'tenor' and 'scenario' (from the MultiIndex)
        - Remaining columns: dates
        - Each row: the forward price path for one (tenor, scenario) pair

    Parameters
    ----------
    scenario_df : pd.DataFrame — RiskFlow-format
    filepath : str
    factor_name : str, optional — added as header comment
    """
    df_out = scenario_df.copy()
    df_out.columns = [str(d.date()) for d in df_out.columns]

    if factor_name:
        print(f"  Exporting {factor_name}: {df_out.shape} to {filepath}")

    df_out.to_csv(filepath)
    print(f"  Saved: {filepath} ({df_out.shape[0]} rows x {df_out.shape[1]} columns)")


# =============================================================================
# STANDALONE SIMULATION (no JSON needed)
# =============================================================================

def run_simulation_standalone(base_date_str, delivery_dates_str, forward_prices,
                              sigma, alpha, drift_mu,
                              time_grid_string='0d 2d 1w(1w) 1m(1m) 3m(3m)',
                              max_date_str=None,
                              model_type='historical',
                              batch_size=1024, simulation_batches=4,
                              use_antithetic=True, random_seed=42,
                              precision=torch.float64):
    """
    Run a simulation with manually specified parameters, still using RiskFlow's
    exact time grid parsing, date handling, and batch loop.

    Replicates the same batch structure as run_simulation_from_json():
        - batch_size scenarios per batch
        - simulation_batches batches with fresh random numbers each
        - Concatenates on scenario axis (axis=-1)
        - Returns RiskFlow-format DataFrame

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
    batch_size : int
        Number of MC scenarios per batch (RiskFlow: params['Batch_Size']).
    simulation_batches : int
        Number of batches (RiskFlow: params['Simulation_Batches']).
    use_antithetic : bool
        Whether to use antithetic sampling.
    random_seed : int
        For reproducibility.

    Returns
    -------
    all_simulated : np.ndarray, shape [timesteps, tenors, total_scenarios]
    scenario_df : pd.DataFrame — RiskFlow-format
    metadata : dict
    """
    # Seed PyTorch's RNG exactly as RiskFlow does (CMC_State.__init__ line 352)
    torch.manual_seed(random_seed)
    total_scenarios = batch_size * simulation_batches

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
    print(f"  Batch_Size: {batch_size}")
    print(f"  Simulation_Batches: {simulation_batches}")
    print(f"  Total scenarios: {total_scenarios}")

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
        sigma, alpha, drift_mu, base_date_excel, use_implied=(model_type == 'implied'))

    # Batch simulation loop (matching RiskFlow's CMC_State pattern)
    L = build_cholesky({}, ['factor'])
    num_timesteps = len(scen_time_grid)
    batch_results = []

    for batch in range(simulation_batches):
        random_numbers = generate_random_numbers(
            L, num_timesteps, batch_size, use_antithetic=use_antithetic,
            dtype=precision)
        simulated = generate_paths(precalc, random_numbers, factor_index=0)
        batch_results.append(simulated)

        if batch == 0:
            print(f"\n  Batch 0: shape = {simulated.shape} "
                  f"[{num_timesteps} times, {len(prices)} tenors, {batch_size} scenarios]")

    # Concatenate batches on scenario axis
    all_simulated = np.concatenate(batch_results, axis=-1)
    print(f"\n  Concatenated: {all_simulated.shape} "
          f"[{num_timesteps} times, {len(prices)} tenors, {total_scenarios} scenarios]")

    # Build RiskFlow-format DataFrame
    scenario_dates = pd.DatetimeIndex(
        sorted([base_date + pd.Timedelta(days=int(d)) for d in scen_time_grid]))

    columns = pd.MultiIndex.from_product(
        [tenors_excel, np.arange(total_scenarios)],
        names=['tenor', 'scenario'])

    flat_values = all_simulated.reshape(num_timesteps, -1)
    scenario_df = pd.DataFrame(
        flat_values,
        index=scenario_dates[:num_timesteps],
        columns=columns
    ).T

    print(f"\n  DataFrame shape: {scenario_df.shape}")

    # Validation
    t_final = scen_time_grid[-1] / DAYS_IN_YEAR
    print(f"\n  Moments at t_final = {t_final:.3f}y:")
    print(f"  {'Delivery':>12s}  {'F(0,T)':>8s}  {'E[F]':>10s}  {'Std[F]':>10s}  "
          f"{'Theo E[F]':>10s}  {'Theo Std':>10s}")
    print("  " + "-" * 68)

    for idx in range(len(prices)):
        F0 = prices[idx]
        sim_F = all_simulated[-1, idx, :]
        Tmt = max((tenors_excel[idx] - base_date_excel) / DAYS_IN_YEAR - t_final, 0.0)
        ln_var = sigma**2 * np.exp(-2.0 * alpha * Tmt) * \
                 (1.0 - np.exp(-2.0 * alpha * t_final)) / (2.0 * alpha)
        # E[F(t,T)] = F(0,T) * exp(mu*t) — Jensen correction cancels (eq 1.116)
        theo_mean = F0 * np.exp(drift_mu * t_final)
        theo_std = theo_mean * np.sqrt(max(np.exp(ln_var) - 1.0, 0.0))

        print(f"  {str(delivery_dates[idx].date()):>12s}  {F0:8.2f}  {np.mean(sim_F):10.4f}  "
              f"{np.std(sim_F):10.4f}  {theo_mean:10.4f}  {theo_std:10.4f}")

    print("=" * 70)

    metadata = {
        'factor_name': f'Standalone.{model_type}',
        'model_type': model_type,
        'params': {'Sigma': sigma, 'Alpha': alpha, 'Drift': drift_mu},
        'base_date': base_date,
        'base_date_excel': base_date_excel,
        'time_grid_string': time_grid_string,
        'scen_time_grid': scen_time_grid,
        'tenors_excel': tenors_excel,
        'prices': prices,
        'currency': 'USD',
        'batch_size': batch_size,
        'simulation_batches': simulation_batches,
        'total_scenarios': total_scenarios,
        'scenario_dates': scenario_dates,
    }

    return all_simulated, scenario_df, metadata


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
                                          batch_size=1024, simulation_batches=4,
                                          use_antithetic=True, random_seed=42,
                                          precision=torch.float64):
    """
    Simulate multiple correlated commodity forward prices from a JSON file.

    Replicates the full RiskFlow pipeline for multiple stochastic factors:
        - Each factor gets its own precalculate() call
        - They share the same correlated random number block
        - Correlations come from the JSON's 'Correlations' section
        - Uses the batch loop pattern (CMC_State.reset() per batch)

    Parameters
    ----------
    json_path : str
        Path to CVAMarketData JSON file.
    factor_names : list of str
        ForwardPrice factor names, e.g. ['ForwardPrice.BRENT.OIL', 'ForwardPrice.WTI.OIL'].
    batch_size : int
        Number of MC scenarios per batch.
    simulation_batches : int
        Number of batches. Total scenarios = batch_size * simulation_batches.

    Returns
    -------
    results : dict of {factor_name: np.ndarray} — shape [timesteps, tenors, total_scenarios]
    scenario_dfs : dict of {factor_name: pd.DataFrame} — RiskFlow-format DataFrames
    metadata_dict : dict of {factor_name: metadata}
    """
    # Seed PyTorch's RNG exactly as RiskFlow does (CMC_State.__init__ line 352)
    torch.manual_seed(random_seed)
    total_scenarios = batch_size * simulation_batches

    print("=" * 70)
    print("MULTI-FACTOR CS SIMULATION FROM JSON")
    print("=" * 70)
    print(f"  Batch_Size: {batch_size}")
    print(f"  Simulation_Batches: {simulation_batches}")
    print(f"  Total scenarios: {total_scenarios}")

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
            base_date_excel, use_implied=(fd['model_type'] == 'implied'))

    # Build correlation matrix
    correlations = extract_correlations(market_data)
    L = build_cholesky(correlations, factor_names)
    print(f"\n  Cholesky L:\n{L}")

    # Batch simulation loop
    batch_results = {fname: [] for fname in factor_names}

    for _ in range(simulation_batches):
        random_numbers = generate_random_numbers(
            L, num_timesteps, batch_size, use_antithetic=use_antithetic,
            dtype=precision)

        for idx, fname in enumerate(factor_names):
            simulated = generate_paths(precalcs[fname], random_numbers, factor_index=idx)
            batch_results[fname].append(simulated)

    # Concatenate batches on scenario axis
    results = {}
    for fname in factor_names:
        results[fname] = np.concatenate(batch_results[fname], axis=-1)
        print(f"  {fname}: shape = {results[fname].shape}")

    # Build RiskFlow-format DataFrames and metadata for each factor
    scenario_dates = pd.DatetimeIndex(
        sorted([base_date + pd.Timedelta(days=int(d)) for d in scen_time_grid]))

    scenario_dfs = {}
    metadata_dict = {}
    for fname, fd in factor_data.items():
        sim = results[fname]
        columns = pd.MultiIndex.from_product(
            [fd['tenors'], np.arange(total_scenarios)],
            names=['tenor', 'scenario'])
        flat_values = sim.reshape(num_timesteps, -1)
        scenario_dfs[fname] = pd.DataFrame(
            flat_values,
            index=scenario_dates[:num_timesteps],
            columns=columns
        ).T

        metadata_dict[fname] = {
            'factor_name': fname,
            'model_type': fd['model_type'],
            'params': fd['params'],
            'base_date': base_date,
            'base_date_excel': base_date_excel,
            'time_grid_string': time_grid_string,
            'scen_time_grid': scen_time_grid,
            'tenors_excel': fd['tenors'],
            'prices': fd['prices'],
            'currency': fd['currency'],
            'batch_size': batch_size,
            'simulation_batches': simulation_batches,
            'total_scenarios': total_scenarios,
            'scenario_dates': scenario_dates,
        }

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
    return results, scenario_dfs, metadata_dict


# =============================================================================
# MAIN — EXAMPLE USAGE
# =============================================================================

if __name__ == '__main__':
    # ------------------------------------------------------------------
    # OPTION A: Run from a RiskFlow JSON file
    # ------------------------------------------------------------------
    # Uncomment and modify the path/factor to match your data:
    #
    # simulated, scenario_df, meta = run_simulation_from_json(
    #     json_path=r'path\to\CVAMarketData.json',
    #     factor_name='ForwardPrice.BRENT.OIL',
    #     time_grid_string=None,   # reads from JSON, or override here
    #     batch_size=1024,
    #     simulation_batches=4,
    #     use_antithetic=True,
    # )
    #
    # plot_results(simulated, meta['scen_time_grid'], meta['tenors_excel'],
    #              meta['prices'], meta['base_date_excel'], meta['model_type'],
    #              save_path='cs_simulation_from_json.png')
    #
    # # Export to CSV (matches RiskFlow's format)
    # export_scenarios_csv(scenario_df, 'cs_scenarios.csv', meta['factor_name'])

    # ------------------------------------------------------------------
    # OPTION B: Run standalone with manual parameters
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXAMPLE: Standalone simulation with RiskFlow time grid parsing")
    print("=" * 70)

    simulated, scenario_df, meta = run_simulation_standalone(
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
        batch_size=1024,
        simulation_batches=4,
        use_antithetic=True,
        random_seed=42,
    )

    # Plot using metadata
    plot_results(simulated, meta['scen_time_grid'], meta['tenors_excel'],
                 meta['prices'], meta['base_date_excel'], meta['model_type'],
                 save_path='cs_standalone_simulation.png')

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
