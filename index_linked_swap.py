from __future__ import annotations

from datetime import date
from typing import Callable, Optional

import numpy as np

from instruments.components.cashflow_leg import CashflowLeg, LegType
from instruments.components.inflation_leg import InflationLeg
from instruments.components.schedule_config import ScheduleConfig
from instruments.instrument import Instrument
from market_data.risk_factor import CurveSlice, RiskFactorSlice
from market_data.yield_curve import YieldCurve
from models.cashflow_pv import leg_pv
from models.inflation_pv import _first_of_month, _shift_months, _shift_months_any, besa_bracket, inflation_leg_pv
from utils.ql_helpers import to_ql_date, generate_sub_periods, BUSINESS_CONVENTIONS


class IndexLinkedSwap(Instrument):
    """
    Index-linked (inflation) swap priced against a ScenarioCube.

    Structure
    ---------
    One leg pays a real fixed coupon on a CPI-indexed notional (the inflation
    leg).  The counterparty leg pays a conventional nominal fixed or floating
    rate on the base notional (the nominal leg).

    The inflation leg cash flows are:

        CF(t_i) = base_notional * (CPI(ref_i) / base_cpi) * accrual_i * real_rate

    At maturity (last period) an additional notional exchange is paid:

        CF_notional = base_notional * (CPI(ref_T) / base_cpi)

    where ``ref_i`` is the BESA-lagged reference date for ``end_i`` (first of
    month, lag_months before the payment end date).

    CPI values at future reference dates come from a stochastic ``CurveSlice``
    stored in the scenario cube under ``inflation_leg.cpi_curve_name``.  For
    bracket dates that fall on or before the current simulation date, the
    pre-computed historical CPI map is used instead.

    Parameters
    ----------
    name : str
        Trade identifier.
    effective_date : date
        Swap start date.
    maturity_date : date
        Swap end date / final exchange date.
    notional : float
        Base notional amount.
    inflation_leg : InflationLeg
        Parameters of the CPI-indexed leg (real rate, base CPI, cpi_curve_name,
        lag, etc.).
    nominal_leg : CashflowLeg
        Parameters of the nominal leg (fixed or floating).
    discount_curve_name : str
        Scenario-cube factor name for the nominal discount curve.
    discount_interpolator : callable
        Interpolator factory for the nominal discount curve.
    cpi_interpolator : callable
        Interpolator factory for the CPI spot curve (legacy mode: also used
        to interpolate the full CPI level term structure when
        ``inflation_leg.inflation_rate_curve_name`` is empty).
    forward_interpolator : callable
        Interpolator factory for the nominal floating-leg projection curve.
        Only used when ``nominal_leg.leg_type == FLOATING``.
    inflation_rate_interpolator : callable or None
        Interpolator factory for the real inflation rate curve
        (``inflation_leg.inflation_rate_curve_name``).  When provided
        together with a non-empty ``inflation_rate_curve_name`` on the leg,
        activates the two-curve Riskflow mode: forward CPI is derived as
        ``spot_CPI / DF_infl(T)`` rather than interpolated from a CPI level
        surface.  Pass ``None`` (default) to use the legacy approach.
    inflation_index : InflationIndex or None
        Historical CPI data object.  When supplied, its internal monthly CPI
        map is used to seed the ``historical_cpi_map`` used during pricing for
        bracket dates that fall in the past.
    inflation_receiver : bool
        If True the holder *receives* the inflation leg and *pays* the nominal
        leg (NPV = inflation_pv − nominal_pv).
        If False the sign is reversed (payer of inflation).
    schedule_config : ScheduleConfig or None
        Shared QuantLib schedule configuration.  When None a default is
        built from the individual ``calendar`` / ``day_count`` / … params.
    include_sim_date_cashflows : bool
        Whether to include cashflows settling exactly on the simulation date.

    Legacy flat params (used only when schedule_config is None)
    -----------------------------------------------------------
    calendar, business_convention, termination_business_convention,
    date_generation, day_count, curve_day_count, end_of_month
    """

    def __init__(
        self,
        name: str,
        effective_date: date,
        maturity_date: date,
        notional: float,
        inflation_leg: InflationLeg,
        nominal_leg: CashflowLeg,
        discount_curve_name: str,
        discount_interpolator: Callable,
        cpi_interpolator: Callable,
        forward_interpolator: Callable,
        inflation_rate_interpolator: Optional[Callable] = None,
        inflation_index=None,
        inflation_receiver: bool = True,
        schedule_config: Optional[ScheduleConfig] = None,
        # legacy flat params
        calendar: str = "ZAR",
        business_convention: str = "ModifiedFollowing",
        termination_business_convention: str = "ModifiedFollowing",
        date_generation: str = "Backward",
        day_count: str = "ACT/365",
        curve_day_count: str = "ACT/365",
        end_of_month: bool = False,
        include_sim_date_cashflows: bool = False,
        ois_initial_cfs: dict[tuple[str, date], float] | None = None,
    ):
        super().__init__(name)
        self._ois_initial_cfs: dict[tuple[str, date], float] = ois_initial_cfs or {}
        self.effective_date = effective_date
        self.maturity_date = maturity_date
        self.notional = notional
        self.inflation_leg = inflation_leg
        self.nominal_leg = nominal_leg
        self.discount_curve_name = discount_curve_name
        self.discount_interpolator = discount_interpolator
        self.cpi_interpolator = cpi_interpolator
        self.forward_interpolator = forward_interpolator
        self.inflation_rate_interpolator = inflation_rate_interpolator
        self.inflation_index = inflation_index
        self.inflation_receiver = inflation_receiver
        self.include_sim_date_cashflows = include_sim_date_cashflows

        self.schedule_config = schedule_config or ScheduleConfig(
            calendar=calendar,
            business_convention=business_convention,
            termination_business_convention=termination_business_convention,
            date_generation=date_generation,
            day_count=day_count,
            curve_day_count=curve_day_count,
            end_of_month=end_of_month,
        )

        self._generate_schedules()
        self._build_historical_cpi_map()

    # ------------------------------------------------------------------
    # Schedule generation
    # ------------------------------------------------------------------

    def _generate_schedules(self):
        """Build QuantLib schedules for both legs."""
        self.inflation_schedule = self.schedule_config.build(
            self.effective_date, self.maturity_date,
            self.inflation_leg.frequency,
        )
        self.nominal_schedule = self.schedule_config.build(
            self.effective_date, self.maturity_date,
            self.nominal_leg.frequency,
        )
        # Cache the QuantLib-adjusted last payment date across both legs.
        # This may differ from self.maturity_date when QuantLib's business-day
        # convention shifts the terminal date (e.g. Sunday 2028-01-02 → Friday
        # 2028-01-28 under end-of-month rules).  Using the raw maturity_date for
        # the expiry check would incorrectly zero NPV before the last cashflow.
        self._effective_maturity: date = max(
            max(p for _, _, p, _ in self.inflation_schedule),
            max(p for _, _, p, _ in self.nominal_schedule),
        )

    # ------------------------------------------------------------------
    # Historical CPI map
    # ------------------------------------------------------------------

    def _build_historical_cpi_map(self):
        """
        Seed ``self._historical_cpi_map`` from the ``InflationIndex`` object.

        The map is ``dict[date, float]`` keyed by first-of-month CPI reference
        dates.  During pricing, entries whose date ≤ the simulation date are
        used as known historical fixings; future entries are overridden by the
        stochastic CPI curve.
        """
        self._historical_cpi_map: dict[date, float] = {}
        if self.inflation_index is None:
            return
        # Use the full monthly fixing map (historical + any pre-extended projection)
        if hasattr(self.inflation_index, "_monthly_cpi"):
            self._historical_cpi_map = dict(self.inflation_index._monthly_cpi)

    # ------------------------------------------------------------------
    # Reset / fixing interface (nominal floating leg)
    # ------------------------------------------------------------------

    def get_reset_dates(self) -> list[tuple[date, str, date, date, bool]]:
        """
        Return ``(reset_date, curve_name, period_start, period_end, is_overnight)``
        for all floating resets on the nominal leg.

        The 5th element ``is_overnight`` mirrors :meth:`IRSwap.get_reset_dates`
        and is required by the ExposureEngine so it can distinguish LIBOR-style
        resets (cache a rate once at reset_date) from OIS resets (accumulate the
        compound factor ``∏ 1/DF(t_j → t_j+1)`` incrementally across every sim
        step via :meth:`compute_cf_increment`).

        For sub-period reset legs (``reset_frequency_months > 0``) the flag is
        always ``False`` — sub-period rates are LIBOR-style by definition.

        For plain ZC OIS legs (``frequency=0``, ``overnight_compounding=True``)
        the single reset tuple carries ``is_overnight=True`` so the engine
        correctly accumulates the daily compound factor.

        The inflation leg does not have floating resets — its CPI reference
        values come directly from the scenario cube at pricing time.
        """
        leg = self.nominal_leg
        if leg.leg_type != LegType.FLOATING:
            return []
        sc = self.schedule_config
        resets: list[tuple[date, str, date, date, bool]] = []
        if leg.reset_frequency_months > 0:
            for pay_start, pay_end, _, _ in self.nominal_schedule:
                for sub_start, sub_end, _ in generate_sub_periods(
                    pay_start, pay_end, leg.reset_frequency_months,
                    sc.ql_calendar, sc.ql_convention, sc.day_counter,
                    direction="Backward",
                ):
                    # Sub-period resets are always LIBOR-style (False).
                    resets.append((sub_start, leg.curve_name, sub_start, sub_end, False))
        else:
            for start, end, _, _ in self.nominal_schedule:
                resets.append((start, leg.curve_name, start, end, leg.overnight_compounding))
        return resets

    def compute_cf_increment(
        self,
        curve_name: str,
        t_from: date,
        t_to: date,
        time_slice: dict[str, RiskFactorSlice],
    ) -> np.ndarray:
        """One-step OIS compound factor ``1 / DF(t_from → t_to)``.

        Called by the ExposureEngine for each simulation step while an OIS
        floating nominal leg is in progress.  Uses the forward projection curve
        from *time_slice* (the scenario state at *t_from*) to estimate the
        overnight compounding over the sub-interval [t_from, t_to].

        The accumulated product across all steps gives ``CF_realized``, which
        :func:`leg_pv` uses in the in-progress OIS formula:

            PV = N × (CF_realized − DF(val → end))

        Mirrors :meth:`IRSwap.compute_cf_increment` exactly; uses
        ``self.forward_interpolator`` for the nominal forward curve.

        Parameters
        ----------
        curve_name : str
            OIS projection curve key in *time_slice*.
        t_from, t_to : date
            Endpoints of the one-step increment (consecutive simulation dates
            or the step boundary at the current valuation date).
        time_slice : dict
            Market state as of *t_from*.

        Returns
        -------
        np.ndarray, shape (n_paths,)
            ``1 / DF(τ)``  where ``τ = curve_day_counter.yearFraction(t_from, t_to)``.
        """
        sc = self.schedule_config
        fwd_slice: CurveSlice = time_slice[curve_name]
        fwd_curve = YieldCurve(
            year_fracs=fwd_slice.tenors,
            rates=fwd_slice.values,
            interpolator=self.forward_interpolator,
        )
        tau = sc.curve_day_counter.yearFraction(
            to_ql_date(t_from), to_ql_date(t_to),
        )
        return 1.0 / fwd_curve.discount_factor(tau)  # (n_paths,)

    def get_cpi_last_pub_date(self, val_date: date) -> date:
        """
        Return the exact T_last_pub for ``val_date``.

        When ``inflation_leg.next_publication_date`` is set, mirrors RiskFlow's
        ``get_last_publication_dates`` exactly: counts how many publication
        events have been crossed up to ``val_date`` and advances from the last
        historical CPI reference date by that many periods.  This is correct to
        the day for any simulation grid, including daily grids where the simple
        approximation diverges during the 1–14 day window before each
        mid-month publication.

        When ``next_publication_date`` is ``None`` (default), or when
        ``_historical_cpi_map`` is empty, falls back to the legacy approximation:
        ``first_of_month(val_date) − 1 month``.  This is exact for month-end
        simulation grids and approximates within ~±15 days on daily grids.

        Pass the return value as ``cpi_last_pub_date`` to ``scenario_npvs()``
        to eliminate the T_last_pub growth bias.
        """
        npd = self.inflation_leg.next_publication_date
        freq = self.inflation_leg.publication_frequency_months

        if npd is None or not self._historical_cpi_map:
            return _shift_months(_first_of_month(val_date), -1)

        last_period_start = max(self._historical_cpi_map)
        # Count how many publication events have been crossed up to val_date.
        # Mirrors RiskFlow: publication[searchsorted(next_pub_dates, val_date, 'right')]
        # where next_pub_dates = [npd, npd+freq, npd+2*freq, ...].
        n = 0
        while _shift_months_any(npd, n * freq) <= val_date:
            n += 1
        return _shift_months(last_period_start, n * freq)

    def get_cpi_reference_dates(self) -> list[tuple[date, str]]:
        """
        Return ``(bracket_date, cpi_curve_name)`` for every unique BESA bracket
        date in the full inflation schedule, sorted chronologically.

        Analogous to ``get_reset_dates()`` for the nominal floating leg.  The
        XVA engine calls this once at trade setup to know which CPI publication
        dates to track across the simulation; as each simulation step completes,
        ``compute_cpi_fixings()`` captures the per-path CPI level for bracket
        dates that have become past, and these are forwarded to subsequent
        ``scenario_npvs()`` calls via the ``cpi_fixings`` argument.

        This mirrors Riskflow's separation between the ``PriceIndex`` (spot)
        factor — which the engine reads back for past reference dates — and the
        forward ``InflationRate`` curve used for future CPI levels.
        """
        seen: set[date] = set()
        refs: list[tuple[date, str]] = []
        for _, end_date, _, _ in self.inflation_schedule:
            j, j1 = besa_bracket(end_date, self.inflation_leg.lag_months)
            for ref_date in sorted({j, j1}):
                if ref_date not in seen:
                    refs.append((ref_date, self.inflation_leg.cpi_curve_name))
                    seen.add(ref_date)
        return sorted(refs, key=lambda x: x[0])

    def _compute_cpi_fixing_for_date(
        self,
        ref_date: date,
        fix_state: dict[str, RiskFactorSlice],
    ) -> "dict[date, np.ndarray]":
        """Return {ref_date: spot_cpi} from fix_state, or {} for historical dates.

        Called by ExposureEngine._build_cpi_fixings with the historical market
        state nearest to ref_date (bisect pattern).  Called by
        compute_cpi_fixings with the current step's state for standalone use.
        """
        if ref_date in self._historical_cpi_map:
            return {}
        cpi_slice: CurveSlice = fix_state[self.inflation_leg.cpi_curve_name]
        return {ref_date: cpi_slice.values[:, 0].copy()}

    def _compute_t_last_pub_fixing(
        self,
        time_slice: dict[str, RiskFactorSlice],
        sim_date: date,
        existing_fixings: "dict[date, np.ndarray]",
    ) -> "dict[date, np.ndarray]":
        """Return {T_last_pub: spot_cpi} for the current step, or {} if already stamped.

        T_last_pub is inherently defined relative to sim_date (publication lag),
        so it always uses the current step's market state, never a historical state.

        Uses ``get_cpi_last_pub_date(sim_date)`` so that when
        ``inflation_leg.next_publication_date`` is set the stamp key matches the
        exact publication schedule — the same value that the ExposureEngine will
        pass as ``cpi_last_pub_date`` (pub_cutoff) in the pricing call.  Without
        this, the stamped key could be one month ahead of pub_cutoff during the
        1–14 day pre-publication window, landing in the "unpublished past" zone
        and being filtered out of the forward-projection anchor selection even
        though pre-seeding was the purpose of the stamp.
        """
        t_pub = self.get_cpi_last_pub_date(sim_date)
        if t_pub in self._historical_cpi_map or t_pub in existing_fixings:
            return {}
        cpi_slice: CurveSlice = time_slice[self.inflation_leg.cpi_curve_name]
        return {t_pub: cpi_slice.values[:, 0].copy()}

    def compute_cpi_fixings(
        self,
        time_slice: dict[str, RiskFactorSlice],
        scenario_date: date,
        existing_fixings: "dict[date, np.ndarray] | None" = None,
    ) -> dict[date, np.ndarray]:
        """
        Return per-path CPI levels for BESA bracket dates **newly** crossed at
        ``scenario_date``, using the spot CPI from ``time_slice``.

        Standalone convenience method: uses ``scenario_date``'s ``time_slice``
        for all bracket dates.  The ExposureEngine's ``_build_cpi_fixings``
        provides better accuracy by calling ``_compute_cpi_fixing_for_date``
        with the market state at the nearest scenario date ≤ each bracket date
        (bisect pattern), avoiding the growing re-evaluation bias on coarser
        simulation grids.

        Parameters
        ----------
        time_slice : dict[str, RiskFactorSlice]
            Market state at ``scenario_date`` from the scenario cube.
        scenario_date : date
            Simulation date at which the CPI path is being recorded.
        existing_fixings : dict[date, np.ndarray] or None
            Already-accumulated per-path fixings from earlier simulation steps.
            Bracket dates present here are skipped so the first-crossing value
            is preserved.  Pass ``None`` to return all past bracket dates
            (legacy standalone behaviour).

        Returns
        -------
        dict[date, np.ndarray]
            Newly-crossed bracket dates mapped to per-path CPI levels,
            shape ``(n_paths,)`` each.
        """
        fixings: dict[date, np.ndarray] = {}

        if existing_fixings is not None:
            fixings.update(
                self._compute_t_last_pub_fixing(time_slice, scenario_date, existing_fixings)
            )

        for _, end_date, _, _ in self.inflation_schedule:
            j, j1 = besa_bracket(end_date, self.inflation_leg.lag_months)
            for ref_date in sorted({j, j1}):
                if ref_date > scenario_date:
                    continue
                if ref_date in fixings:
                    continue
                if existing_fixings is not None and ref_date in existing_fixings:
                    continue
                fixings.update(self._compute_cpi_fixing_for_date(ref_date, time_slice))
        return fixings

    def compute_fixings(
        self,
        resets: list[tuple[date, str, date, date]],
        time_slice: dict[str, RiskFactorSlice],
        scenario_date: date,
    ) -> dict[tuple[str, date], np.ndarray]:
        """Compute forward rates for nominal-leg floating resets from an earlier scenario.

        Handles ``fixing_tenor_months`` on the nominal leg: when set, the
        forward rate tenor is advanced from ``p_start`` by that many months
        (matching the LIBOR index natural tenor) rather than using ``p_end``
        directly.  This is consistent with :meth:`IRSwap.compute_fixings`.
        """
        fixings: dict[tuple[str, date], np.ndarray] = {}
        sc = self.schedule_config
        ql_scenario = to_ql_date(scenario_date)
        leg = self.nominal_leg

        fwd_conv = (
            BUSINESS_CONVENTIONS[leg.forward_business_convention]
            if leg.forward_business_convention is not None
            else None
        )
        import QuantLib as ql  # local import — ql already pulled in by schedule_config

        for _reset_date, curve_name, p_start, p_end in resets:
            fwd_slice: CurveSlice = time_slice[curve_name]
            fwd_curve = YieldCurve(
                year_fracs=fwd_slice.tenors,
                rates=fwd_slice.values,
                interpolator=self.forward_interpolator,
            )
            t_start = sc.curve_day_counter.yearFraction(ql_scenario, to_ql_date(p_start))

            if leg.fixing_tenor_months is not None:
                conv = fwd_conv if fwd_conv is not None else ql.ModifiedFollowing
                ql_fix_end = sc.ql_calendar.advance(
                    to_ql_date(p_start),
                    ql.Period(leg.fixing_tenor_months, ql.Months),
                    conv,
                )
                t_end = sc.curve_day_counter.yearFraction(ql_scenario, ql_fix_end)
                fwd_tau = sc.day_counter.yearFraction(to_ql_date(p_start), ql_fix_end)
                rate = fwd_curve.forward_rate(t_start, t_end, tau=fwd_tau)
            else:
                t_end = sc.curve_day_counter.yearFraction(ql_scenario, to_ql_date(p_end))
                rate = fwd_curve.forward_rate(t_start, t_end)

            fixings[(curve_name, p_start)] = rate

        return fixings

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def scenario_npvs(
        self,
        val_date: date,
        market_state: dict[str, RiskFactorSlice],
        fixings: dict[tuple[str, date], np.ndarray] | None = None,
        rng: "np.random.Generator | None" = None,
        cpi_fixings: "dict[date, np.ndarray] | None" = None,
        cpi_last_pub_date: "date | None" = None,
    ) -> np.ndarray:
        """
        Compute NPV for all Monte Carlo paths at a single simulation date.

        Returns ``(n_paths,)`` NPV array.

        The sign convention follows ``inflation_receiver``:
          - ``True``  → NPV = inflation_leg_pv − nominal_leg_pv
          - ``False`` → NPV = nominal_leg_pv − inflation_leg_pv

        Both legs are priced to the nominal discount curve.
        CPI index values are sourced from the stochastic CurveSlice at
        ``market_state[inflation_leg.cpi_curve_name]`` for future bracket
        dates.  For bracket dates that have become past during the simulation,
        pass the per-path levels accumulated by ``compute_cpi_fixings()`` via
        ``cpi_fixings``; they take priority over the static historical map.

        In Riskflow two-curve mode, ``cpi_last_pub_date`` is the last CPI
        publication date (``T_last_pub``) for this simulation step.  When
        provided, forward CPI is grown from ``T_last_pub`` rather than
        ``val_date``, matching Riskflow's ``calc_index`` formula and
        eliminating a systematic bias of ``exp(r × (val_date − T_last_pub))``.
        When None the function derives ``T_last_pub`` automatically from
        ``historical_cpi_map`` and ``cpi_fixings``.
        """
        disc_slice: CurveSlice = market_state[self.discount_curve_name]
        n_paths = disc_slice.values.shape[0]

        if val_date > self._effective_maturity:
            return np.zeros(n_paths)

        discount_curve = YieldCurve(
            year_fracs=disc_slice.tenors,
            rates=disc_slice.values,
            interpolator=self.discount_interpolator,
        )

        sc = self.schedule_config

        # --- Inflation leg ---
        infl_pv = inflation_leg_pv(
            schedule=self.inflation_schedule,
            leg=self.inflation_leg,
            base_notional=self.notional,
            val_date=val_date,
            market_state=market_state,
            discount_curve=discount_curve,
            n_paths=n_paths,
            cpi_interpolator=self.cpi_interpolator,
            curve_day_counter=sc.curve_day_counter,
            historical_cpi_map=self._historical_cpi_map,
            include_on_val_date=self.include_sim_date_cashflows,
            cpi_fixings=cpi_fixings,
            inflation_rate_interpolator=self.inflation_rate_interpolator,
            cpi_last_pub_date=cpi_last_pub_date,
            calendar=sc.ql_calendar,
            day_counter=sc.day_counter,
            convention=sc.ql_convention,
        )

        # --- Nominal leg ---
        nom_pv = leg_pv(
            schedule=self.nominal_schedule,
            leg=self.nominal_leg,
            notional=self.notional,
            val_date=val_date,
            market_state=market_state,
            discount_curve=discount_curve,
            n_paths=n_paths,
            interpolator=self.forward_interpolator,
            day_counter=sc.day_counter,
            curve_day_counter=sc.curve_day_counter,
            calendar=sc.ql_calendar,
            fixings=fixings,
            include_on_val_date=self.include_sim_date_cashflows,
        )

        if self.inflation_receiver:
            return infl_pv - nom_pv
        return nom_pv - infl_pv
