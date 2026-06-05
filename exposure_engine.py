from __future__ import annotations

from bisect import bisect_right
from datetime import date, timedelta

import numpy as np

from portfolio.csa import CloseOutMethod, InitialMarginMethod
from portfolio.netting_set import NettingSet
from market_data.risk_factor import CurveSlice, RiskFactorSlice, ScalarSlice, SurfaceSlice
from market_data.scenario_cube import ScenarioCube
from market_data.static_market_data import StaticMarketData
from pricing.exposure_profile import ExposureProfile


def _interp_scenario_state(
    all_states: list[dict],
    scenario_dates: list[date],
    prev_idx: int,
    target_date: date,
) -> dict:
    """Return a linearly interpolated market state at target_date.

    prev_idx is bisect_right(scenario_dates, target_date) - 1 (clamped to 0).
    Returns all_states[prev_idx] unchanged when target_date exactly matches
    scenario_dates[prev_idx] or prev_idx is the last available index.
    Otherwise linearly interpolates each RiskFactorSlice between prev_idx
    and prev_idx+1 by the day-fraction elapsed in the interval, mirroring
    RiskFlow's gather_scenario_interp / TimeGrid.get_scenario_offset logic.
    """
    next_idx = prev_idx + 1
    if next_idx >= len(all_states) or scenario_dates[prev_idx] == target_date:
        return all_states[prev_idx]

    span = (scenario_dates[next_idx] - scenario_dates[prev_idx]).days
    if span == 0:
        return all_states[prev_idx]

    alpha = (target_date - scenario_dates[prev_idx]).days / span
    if alpha <= 0.0:
        return all_states[prev_idx]
    if alpha >= 1.0:
        return all_states[next_idx]

    state_a = all_states[prev_idx]
    state_b = all_states[next_idx]
    result: dict = {}
    for name, sa in state_a.items():
        sb = state_b.get(name)
        if sb is None or type(sa) is not type(sb):
            result[name] = sa
            continue
        v = (1.0 - alpha) * sa.values + alpha * sb.values
        if isinstance(sa, SurfaceSlice):
            result[name] = SurfaceSlice(values=v, tenors=sa.tenors, strikes=sa.strikes)
        elif isinstance(sa, CurveSlice):
            result[name] = CurveSlice(values=v, tenors=sa.tenors)
        else:
            result[name] = ScalarSlice(values=v)
    return result


class ExposureEngine:
    """Computes an ExposureProfile for a NettingSet against a ScenarioCube.

    Iterates over all simulation dates, prices each trade in the netting
    set, aggregates NPVs (netting), simulates the collateral account
    pathwise with MPOR lookback, and returns the full ExposureProfile.

    Parameters
    ----------
    cube : ScenarioCube
        Simulated scenario cube containing all market risk factors.
    static_data : StaticMarketData or None
        Path-invariant market data (e.g. deterministic hazard curves,
        basis spreads) merged into every time-slice before pricing.
        Stochastic cube factors take precedence on name collision.
    """

    def __init__(
        self,
        cube: ScenarioCube,
        static_data: StaticMarketData | None = None,
    ) -> None:
        self.cube = cube
        self.static_data = static_data or StaticMarketData()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(self, netting_set: NettingSet) -> ExposureProfile:
        """Compute the exposure profile for a netting set.

        Parameters
        ----------
        netting_set : NettingSet
            The netting set to evaluate. Contains trades and an optional CSA.

        Returns
        -------
        ExposureProfile
        """
        n_paths = self.cube.n_paths
        n_times = self.cube.n_times
        scenario_dates = list(self.cube.dates)
        cube_end = scenario_dates[-1]

        # Validate that the cube's time grid covers every trade's effective
        # maturity.  The ScenarioCubeBuilder intersects factor date sets, so
        # if the simulation was run too short the longer-dated trade's exposure
        # beyond the last cube date is silently zero — a silent wrong answer.
        for trade in netting_set.trades:
            trade_end = trade.instrument.effective_maturity
            if isinstance(trade_end, date) and trade_end > cube_end:
                raise ValueError(
                    f"Trade '{trade.trade_id}' effective maturity {trade_end} "
                    f"extends beyond the last cube date {cube_end}. "
                    f"Re-run the simulation with a grid that covers at least {trade_end}."
                )

        # Validate FX factors for cross-currency trades up front.
        for trade in netting_set.trades:
            if trade.currency != netting_set.reporting_currency:
                if trade.fx_rate_factor is None:
                    raise ValueError(
                        f"Trade '{trade.trade_id}' currency '{trade.currency}' differs "
                        f"from netting set reporting currency "
                        f"'{netting_set.reporting_currency}' but fx_rate_factor is not set."
                    )

        # fixing_cache keyed by (instrument_id, curve_name, period_start)
        # persists across time steps so each reset is computed at most once
        fixing_cache: dict[tuple, np.ndarray] = {}

        # cpi_fixings_cache keyed by instrument_id; per-instrument accumulator
        # of locked-in per-path CPI levels for bracket dates that have been
        # published during the simulation (mirrors Riskflow's old_resets).
        cpi_fixings_cache: dict[int, dict] = {}

        # commodity_fixings_cache keyed by instrument_id; per-instrument
        # accumulator of realized forward prices for past delivery /
        # averaging dates (CommodityForward and CommodityAverageForward).
        # Each date is stamped once at the first crossing step and never
        # re-evaluated, eliminating the growing re-evaluation error.
        commodity_fixings_cache: dict[int, dict] = {}

        # equity_fixings_cache keyed by instrument_id; per-instrument
        # accumulator of realized per-path equity spot prices for return-leg
        # reset dates (EquityTRS "Price" scaling).  Each reset is stamped
        # once from the scenario state most contemporary with that date,
        # mirroring the commodity and CPI patterns.
        equity_fixings_cache: dict[int, dict] = {}

        mtm_paths = np.zeros((n_paths, n_times))

        # Allow instruments to pre-compute anything required before the loop
        # (e.g. PDE surfaces, LSM exercise boundaries, surrogate models).
        all_states = [
            {**self.static_data.factors, **self.cube.get_time_slice(t)}
            for t in range(n_times)
        ]
        for trade in netting_set.trades:
            trade.instrument.precompute(all_states, scenario_dates)

        for t_idx in range(n_times):
            sim_date = scenario_dates[t_idx]
            base_market_state = all_states[t_idx]

            for trade in netting_set.trades:
                instrument = trade.instrument
                fixings = self._build_fixings(
                    instrument, sim_date, scenario_dates, fixing_cache, all_states,
                )
                commodity_fixings = self._build_commodity_fixings(
                    instrument, sim_date, commodity_fixings_cache,
                    scenario_dates, all_states,
                )
                if commodity_fixings:
                    fixings = {**fixings, **commodity_fixings}
                equity_fixings = self._build_equity_fixings(
                    instrument, sim_date, equity_fixings_cache,
                    scenario_dates, all_states,
                )
                if equity_fixings:
                    fixings = {**fixings, **equity_fixings}
                cpi_kwargs = self._build_cpi_fixings(
                    instrument, base_market_state, sim_date, cpi_fixings_cache,
                    scenario_dates, all_states,
                )
                pricing_state = self._pricing_market_state(
                    base_market_state, instrument, netting_set, trade.currency
                )
                npv = instrument.scenario_npvs(
                    sim_date, pricing_state, fixings=fixings or None,
                    **cpi_kwargs,
                )
                if trade.currency != netting_set.reporting_currency:
                    fx_slice = base_market_state[trade.fx_rate_factor]
                    npv = npv * fx_slice.values
                mtm_paths[:, t_idx] += trade.notional_scale * npv

        # Collateral simulation
        if netting_set.csa is not None:
            collateral = self._simulate_collateral(mtm_paths, scenario_dates, netting_set.csa)
        else:
            collateral = np.zeros((n_paths, n_times))

        net = mtm_paths - collateral
        exposure = np.maximum(net, 0.0)
        neg_exposure = np.minimum(net, 0.0)

        return ExposureProfile(
            netting_set_id=netting_set.netting_set_id,
            dates=tuple(scenario_dates),
            mtm=mtm_paths,
            collateral=collateral,
            exposure=exposure,
            neg_exposure=neg_exposure,
            currency=netting_set.reporting_currency,
        )

    # ------------------------------------------------------------------
    # Fixing cache (Category B resets)
    # ------------------------------------------------------------------

    def _build_fixings(
        self,
        instrument,
        sim_date: date,
        scenario_dates: list[date],
        fixing_cache: dict,
        all_states: list[dict] | None = None,
    ) -> dict[tuple, np.ndarray]:
        """Return fixings for all floating resets before sim_date.

        LIBOR-style resets
        ------------------
        Computed once at the reset date via ``compute_fixings`` and cached
        for all later simulation dates (the rate never changes).

        OIS-style resets (is_overnight=True)
        -------------------------------------
        CF_realized must grow at each simulation step because overnight
        compounding is path-dependent and accumulates across time.  The
        engine calls ``compute_cf_increment`` to advance the running
        product by one time step using the scenario curve at the START of
        that step (the forward-looking convention: use the curve you see
        at t_j to compound from t_j to t_{j+1}).

        ``fixing_cache`` stores two keys per OIS period:
          ``(inst_id, curve_name, p_start, '_ois_cf')``   – running CF
          ``(inst_id, curve_name, p_start, '_ois_last')``  – last processed date
        """
        if not hasattr(instrument, "get_reset_dates"):
            return {}

        has_libor = hasattr(instrument, "compute_fixings")
        has_ois = hasattr(instrument, "compute_cf_increment")
        if not has_libor and not has_ois:
            return {}

        fixings: dict[tuple, np.ndarray] = {}
        inst_id = id(instrument)
        n_paths = self.cube.n_paths

        for reset_tuple in instrument.get_reset_dates():
            reset_date, curve_name, p_start, p_end = reset_tuple[:4]
            is_overnight = reset_tuple[4] if len(reset_tuple) > 4 else False

            # OIS: only accumulate after period start (nothing to compound on day 0).
            # LIBOR: also process on the reset date itself — the rate is observable
            # today and must be cached so apply_fixings uses it consistently.
            if is_overnight and reset_date >= sim_date:
                continue
            if not is_overnight and reset_date > sim_date:
                continue

            if is_overnight and has_ois:
                # ----------------------------------------------------------
                # OIS: accumulate CF_realized incrementally.
                # At each call we advance the running product by the steps
                # from the last processed date up to sim_date.
                # ----------------------------------------------------------
                cf_key = (inst_id, curve_name, p_start, "_ois_cf")
                last_key = (inst_id, curve_name, p_start, "_ois_last")

                prev_date = fixing_cache.get(last_key)
                cf_realized = fixing_cache.get(cf_key)

                # Determine which scenario dates still need to be processed.
                if prev_date is None:
                    # First time entering this period.  Seed CF_realized from
                    # the instrument's historical compound factor when the period
                    # started before the simulation horizon (mirrors RiskFlow's
                    # old_resets initialization).  Falls back to 1.0 when no
                    # seed is available (periods that start within the grid).
                    initial_cf = instrument.get_ois_initial_cf(curve_name, p_start)
                    if initial_cf is not None:
                        cf_realized = np.full(n_paths, float(initial_cf))
                    else:
                        cf_realized = np.ones(n_paths)
                    step_starts = [
                        t for t in scenario_dates
                        if p_start <= t < sim_date
                    ]
                else:
                    # Incremental: advance from prev_date to sim_date.
                    step_starts = [
                        t for t in scenario_dates
                        if prev_date <= t < sim_date
                    ]

                for j, t_j in enumerate(step_starts):
                    t_j1 = (
                        step_starts[j + 1]
                        if j + 1 < len(step_starts)
                        else sim_date
                    )
                    t_j_idx = max(
                        0, bisect_right(scenario_dates, t_j) - 1
                    )
                    fix_slice = {
                        **self.static_data.factors,
                        **self.cube.get_time_slice(t_j_idx),
                    }
                    cf_realized = (
                        cf_realized
                        * instrument.compute_cf_increment(
                            curve_name, t_j, t_j1, fix_slice
                        )
                    )

                fixing_cache[cf_key] = cf_realized
                fixing_cache[last_key] = sim_date
                fixings[(curve_name, p_start)] = cf_realized

            elif has_libor:
                # ----------------------------------------------------------
                # LIBOR: compute rate once at reset_date, cache forever.
                # ----------------------------------------------------------
                cache_key = (inst_id, curve_name, p_start)
                if cache_key not in fixing_cache:
                    fix_t_idx = max(
                        0, bisect_right(scenario_dates, reset_date) - 1
                    )
                    fix_slice = (
                        all_states[fix_t_idx]
                        if all_states is not None
                        else {**self.static_data.factors,
                              **self.cube.get_time_slice(fix_t_idx)}
                    )
                    computed = instrument.compute_fixings(
                        [(reset_date, curve_name, p_start, p_end)],
                        fix_slice,
                        reset_date,
                    )
                    fixing_cache.update(
                        {(inst_id, k[0], k[1]): v
                         for k, v in computed.items()}
                    )
                fixings[(curve_name, p_start)] = fixing_cache[cache_key]

        return fixings

    # ------------------------------------------------------------------
    # CPI fixings accumulator (inflation instruments)
    # ------------------------------------------------------------------

    def _build_cpi_fixings(
        self,
        instrument,
        base_market_state: dict[str, RiskFactorSlice],
        sim_date: date,
        cpi_fixings_cache: dict[int, dict],
        scenario_dates: list[date],
        all_states: list[dict],
    ) -> dict:
        """Return cpi_fixings kwargs for scenario_npvs, or {} for non-inflation instruments.

        Detects inflation instruments via duck-typing on ``get_cpi_reference_dates``.
        Maintains a per-instrument accumulator so each bracket date is stamped
        exactly once from the scenario state most contemporary with that date.

        For each bracket date ``ref_date <= sim_date`` not yet accumulated, the
        engine bisects at ``ref_date`` to locate the nearest prior scenario state,
        linearly interpolates to the exact bracket date, and calls
        ``_compute_cpi_fixing_for_date``.  This stamps the per-path CPI level
        from the simulated curve at the crossing step.

        T_last_pub pre-seeding (via ``_compute_t_last_pub_fixing``) runs before
        the loop so the anchor used for forward projection in ``_get_cpi_level``
        is populated from the first simulation step onward.
        """
        if not hasattr(instrument, "get_cpi_reference_dates"):
            return {}

        inst_id = id(instrument)
        if inst_id not in cpi_fixings_cache:
            cpi_fixings_cache[inst_id] = {}

        accumulated = cpi_fixings_cache[inst_id]

        if hasattr(instrument, "_compute_t_last_pub_fixing"):
            accumulated.update(
                instrument._compute_t_last_pub_fixing(
                    base_market_state, sim_date, accumulated,
                )
            )

        cpi_last_pub_date = (
            instrument.get_cpi_last_pub_date(sim_date)
            if hasattr(instrument, "get_cpi_last_pub_date")
            else None
        )

        for ref_date, _cpi_name in instrument.get_cpi_reference_dates():
            if ref_date > sim_date:
                break
            if ref_date in accumulated:
                continue
            fix_t_idx = max(0, bisect_right(scenario_dates, ref_date) - 1)
            fix_state = _interp_scenario_state(
                all_states, scenario_dates, fix_t_idx, ref_date,
            )
            accumulated.update(
                instrument._compute_cpi_fixing_for_date(ref_date, fix_state)
            )

        return {
            "cpi_fixings": accumulated,
            "cpi_last_pub_date": cpi_last_pub_date,
        }

    # ------------------------------------------------------------------
    # Commodity fixing accumulator
    # ------------------------------------------------------------------

    def _build_commodity_fixings(
        self,
        instrument,
        sim_date: date,
        commodity_fixings_cache: dict[int, dict],
        scenario_dates: list[date],
        all_states: list[dict],
    ) -> dict:
        """Accumulate realized commodity prices using historical market states.

        Duck-types on ``get_commodity_fixing_schedule`` (present on both
        CommodityForward and CommodityAverageForward).  For each averaging
        or delivery date newly crossed at this simulation step, the method
        linearly interpolates the scenario state at the exact pricing_date
        (mirroring RiskFlow's gather_scenario_interp) and calls
        ``_compute_fixing_for_date`` with that interpolated state, passing
        pricing_date as scenario_date so forward-curve year fractions are
        measured correctly.

        Returns the full accumulated dict for this instrument, which the caller
        merges with standard fixings before calling scenario_npvs.
        """
        if not hasattr(instrument, "get_commodity_fixing_schedule"):
            return {}

        inst_id = id(instrument)
        if inst_id not in commodity_fixings_cache:
            commodity_fixings_cache[inst_id] = {}

        accumulated = commodity_fixings_cache[inst_id]

        for avg_date, pricing_date, fx_settle_date in (
            instrument.get_commodity_fixing_schedule()
        ):
            if pricing_date > sim_date:
                break

            key_fwd = (instrument.forward_curve_name, avg_date)
            if key_fwd in accumulated:
                continue

            # Interpolate the scenario state at the exact pricing_date so
            # the forward curve is read as of the settlement observation date
            # (mirroring RiskFlow's gather_scenario_interp).
            fix_t_idx = max(0, bisect_right(scenario_dates, pricing_date) - 1)
            fix_state = _interp_scenario_state(
                all_states, scenario_dates, fix_t_idx, pricing_date,
            )
            new_fixings = instrument._compute_fixing_for_date(
                avg_date, pricing_date, fx_settle_date,
                fix_state, pricing_date,
            )
            accumulated.update(new_fixings)

        return accumulated

    # ------------------------------------------------------------------
    # Equity spot fixing accumulator
    # ------------------------------------------------------------------

    def _build_equity_fixings(
        self,
        instrument,
        sim_date: date,
        equity_fixings_cache: dict[int, dict],
        scenario_dates: list[date],
        all_states: list[dict],
    ) -> dict:
        """Accumulate realized per-path equity spot prices for return-leg reset dates.

        Duck-types on ``get_equity_reset_schedule`` /
        ``_compute_equity_fixing_for_date`` (present on EquityTRS).

        For each return-leg period start date newly crossed at this simulation
        step, the market state at the nearest scenario date ≤ reset_date is
        used to stamp the simulated spot (bisect pattern, same as
        ``_build_commodity_fixings``).  Each date is locked in exactly once;
        later steps serve the cached value unchanged.

        This makes the per-path F_start for in-progress "Price"-scaling
        periods consistent with RiskFlow's old_resets convention: the notional
        base for each period is the spot observed when the reset boundary was
        first crossed, not a recomputed forward from a later scenario.
        """
        if not hasattr(instrument, "get_equity_reset_schedule"):
            return {}

        inst_id = id(instrument)
        if inst_id not in equity_fixings_cache:
            equity_fixings_cache[inst_id] = {}

        accumulated = equity_fixings_cache[inst_id]

        for reset_date in instrument.get_equity_reset_schedule():
            if reset_date > sim_date:
                break
            key = (instrument.spot_name, reset_date)
            if key in accumulated:
                continue
            fix_t_idx = max(0, bisect_right(scenario_dates, reset_date) - 1)
            fix_state = _interp_scenario_state(
                all_states, scenario_dates, fix_t_idx, reset_date,
            )
            accumulated.update(
                instrument._compute_equity_fixing_for_date(reset_date, fix_state)
            )

        return accumulated

    # ------------------------------------------------------------------
    # Close-out market state
    # ------------------------------------------------------------------

    def _pricing_market_state(
        self,
        market_state: dict[str, RiskFactorSlice],
        instrument,
        netting_set: NettingSet,
        trade_currency: str = "",
    ) -> dict[str, RiskFactorSlice]:
        """Return the market state to use for pricing under the close-out method.

        For STANDARD close-out this is the unmodified market state.
        For FORWARD close-out the instrument's discount curve is replaced
        with the risky (spread-adjusted) curve from the CSA, approximating
        the CVA-adjusted replacement value.

        When ``csa.risky_curve_name`` is a dict, the curve is resolved by
        ``trade_currency`` so each currency leg uses its own spread-adjusted
        curve (e.g. ZAR-OIS-SPREAD for ZAR trades, USD-OIS-SPREAD for USD).
        If the currency is not present in the dict the market state is returned
        unchanged.
        """
        csa = netting_set.csa
        if csa is None or csa.close_out_method is CloseOutMethod.STANDARD:
            return market_state

        risky_name = csa.risky_curve_name
        if isinstance(risky_name, dict):
            risky_name = risky_name.get(trade_currency or netting_set.reporting_currency)
        if risky_name is None or risky_name not in market_state:
            return market_state

        disc_name = getattr(instrument, "discount_curve_name", None)
        if disc_name is None or disc_name == risky_name:
            return market_state

        # Replace the discount curve slice with the risky curve slice
        return {**market_state, disc_name: market_state[risky_name]}

    # ------------------------------------------------------------------
    # Collateral simulation
    # ------------------------------------------------------------------

    def _simulate_collateral(
        self,
        mtm_paths: np.ndarray,
        dates: list[date],
        csa,
    ) -> np.ndarray:
        """Simulate the pathwise collateral account with MPOR lookback.

        At each simulation date ``t`` the collateral available at default
        is based on the last margin call, which used the MTM at
        ``t - mpor_days``. The latest simulation date on or before that
        lookback date is used.

        VM is computed as the net of received and posted margin:
            VM(t) = max(mtm(t_lag) - threshold, 0)
                  - max(-mtm(t_lag) - threshold_post, 0)

        Note: MTA (minimum transfer amount) is not applied here; it is a
        secondary refinement that can be added in a future extension.
        """
        n_paths, n_times = mtm_paths.shape
        collateral = np.zeros((n_paths, n_times))
        mpor = timedelta(days=csa.mpor_days)

        for t_idx, sim_date in enumerate(dates):
            lookback_date = sim_date - mpor
            lag_idx = bisect_right(dates, lookback_date) - 1
            if lag_idx < 0:
                # Lookback falls before first sim date — no collateral yet
                continue

            lagged_mtm = mtm_paths[:, lag_idx]

            vm_recv = np.maximum(lagged_mtm - csa.vm_threshold, 0.0)
            vm_post = np.maximum(-lagged_mtm - csa.vm_threshold_post, 0.0)
            vm = vm_recv - vm_post

            im = self._compute_im(n_paths, csa)
            collateral[:, t_idx] = vm + im

        return collateral

    def _compute_im(self, n_paths: int, csa) -> np.ndarray:
        """Return the pathwise IM array for a single time step."""
        if csa.im_method is InitialMarginMethod.NONE:
            return np.zeros(n_paths)
        if csa.im_method is InitialMarginMethod.FIXED:
            return np.full(n_paths, csa.im_amount)
        if csa.im_method is InitialMarginMethod.SCHEDULE:
            raise NotImplementedError(
                "Schedule-based IM requires notional and regulatory factor data "
                "from the instrument — not yet implemented."
            )
        if csa.im_method is InitialMarginMethod.SIMM:
            raise NotImplementedError("SIMM is not yet implemented.")
        raise ValueError(f"Unknown IM method: {csa.im_method}")
