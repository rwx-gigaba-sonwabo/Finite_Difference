from __future__ import annotations

from datetime import date
from typing import Callable

import numpy as np

from instruments.components.cashflow_leg import CashflowLeg
from instruments.components.schedule_config import ScheduleConfig
from instruments.instrument import Instrument
import QuantLib as ql
from utils.ql_helpers import to_ql_date, advance_business_days, BUSINESS_CONVENTIONS
from market_data.yield_curve import YieldCurve
from market_data.risk_factor import CurveSlice, ScalarSlice, RiskFactorSlice
from models.cashflow_pv import leg_pv, filter_future_periods, compute_period_year_fractions
from models.equity_pv import trs_return_leg_pv, equity_forward_price


class EquityTRS(Instrument):
    """Equity Total Return Swap.

    The total return receiver gets the total return on the equity
    (capital gains + dividends) and pays a floating (or fixed) interest
    rate leg.

    Valued as:
        NPV = PV_return - PV_interest   (from the receiver's perspective)

    Both legs support a ``nominal_scaling`` flag (``"Price"`` or
    ``"Initial Price"``) that controls how the notional evolves:

    * ``"Price"`` — notional resets each period to ``F(T_{i-1}) × quantity``
      (riskflow ``PrincipleNotShares=0``).  For the return leg this yields
      an absolute price-increment payoff; for the interest leg each coupon
      is scaled by the equity forward at the period's reset date.
    * ``"Initial Price"`` — notional stays fixed at
      ``initial_price × quantity`` (riskflow ``PrincipleNotShares=1``).
      Return leg payoffs are fractional returns on that fixed notional;
      interest leg uses ``self.notional`` unchanged.

    Parameters
    ----------
    name : str
        Trade identifier.
    effective_date : date
        Start date / first reset date.
    maturity_date : date
        Final reset / maturity date.
    quantity : float
        Number of shares / units.  The return leg payoff per period is
        ``quantity × (F(T_i) - F(T_{i-1}) + d_fv)`` — the absolute
        equity price change scaled by this fixed share count.
    notional : float
        Fixed notional for the interest leg (typically
        ``initial_price × quantity``).  Stays constant across all
        interest periods regardless of equity price moves.
    interest_leg : CashflowLeg
        The interest rate leg (typically FLOATING with a spread).
    spot_name : str
        Risk factor name for the equity spot price (scalar).
    carry_curve_name : str
        Risk factor name for the equity carry / funding curve.
    dividend_curve_name : str
        Risk factor name for the continuous dividend yield curve.
    discount_curve_name : str
        Risk factor name for the TRS currency discount curve.
    carry_interpolator : callable
        Interpolator factory for the carry curve.
    dividend_interpolator : callable
        Interpolator factory for the dividend yield curve.
    discount_interpolator : callable
        Interpolator factory for the discount curve.
    schedule_config : ScheduleConfig or None
        Schedule generation configuration.
    return_frequency : int
        Payment frequency for the return leg in months (default matches
        the interest leg frequency).
    initial_price : float or None
        Fixed equity reference price for the in-progress period.  This
        is the spot price observed at the most recent reset date.
    return_nominal_scaling : str
        ``"Price"`` (default) — return leg notional resets to
        ``F(T_{i-1}) × quantity`` each period.
        ``"Initial Price"`` — fixed notional ``initial_price × quantity``
        with fractional period returns.
    interest_nominal_scaling : str
        ``"Price"`` — interest leg notional resets to
        ``F(T_{i-1}) × quantity`` each period.
        ``"Initial Price"`` (default) — fixed notional ``self.notional``.
    is_receiver : bool
        True if pricing from the total return receiver's perspective
        (default True).
    return_first_date : date or None
        Front-stub anchor for the return leg schedule.
    return_next_to_last_date : date or None
        Back-stub anchor for the return leg schedule.  Backwards generation
        anchors from here, with a short final period to maturity_date.
    interest_first_date : date or None
        Front-stub anchor for the interest leg schedule.
    interest_next_to_last_date : date or None
        Back-stub anchor for the interest leg schedule.
    """

    def __init__(
        self,
        name: str,
        effective_date: date,
        maturity_date: date,
        quantity: float,
        notional: float,
        interest_leg: CashflowLeg,
        spot_name: str,
        carry_curve_name: str,
        dividend_curve_name: str,
        discount_curve_name: str,
        carry_interpolator: Callable,
        dividend_interpolator: Callable,
        discount_interpolator: Callable,
        schedule_config: ScheduleConfig | None = None,
        return_frequency: int | None = None,
        initial_price: float | None = None,
        return_nominal_scaling: str = "Price",
        interest_nominal_scaling: str = "Initial Price",
        is_receiver: bool = True,
        include_sim_date_cashflows: bool = True,
        spot_lag: int = 0,
        settlement_calendar: str | None = None,
        return_first_date: date | None = None,
        return_next_to_last_date: date | None = None,
        interest_first_date: date | None = None,
        interest_next_to_last_date: date | None = None,
        ois_initial_cfs: dict[tuple[str, date], float] | None = None,
    ):
        super().__init__(name)
        self._ois_initial_cfs: dict[tuple[str, date], float] = ois_initial_cfs or {}
        self.effective_date = effective_date
        self.maturity_date = maturity_date
        self.quantity = quantity
        self.notional = notional
        self.interest_leg = interest_leg
        self.spot_name = spot_name
        self.carry_curve_name = carry_curve_name
        self.dividend_curve_name = dividend_curve_name
        self.discount_curve_name = discount_curve_name
        self.carry_interpolator = carry_interpolator
        self.dividend_interpolator = dividend_interpolator
        self.discount_interpolator = discount_interpolator
        self.initial_price = initial_price
        self.return_nominal_scaling = return_nominal_scaling
        self.interest_nominal_scaling = interest_nominal_scaling
        self.is_receiver = is_receiver
        self.include_sim_date_cashflows = include_sim_date_cashflows
        self.spot_lag = spot_lag
        self._settlement_cal = settlement_calendar  # resolved after schedule_config is set

        self.schedule_config = schedule_config or ScheduleConfig()
        self.return_frequency = return_frequency or interest_leg.frequency
        self._return_first_date = return_first_date
        self._return_next_to_last_date = return_next_to_last_date
        self._interest_first_date = interest_first_date
        self._interest_next_to_last_date = interest_next_to_last_date

        self._generate_schedules()
        self._build_settle_map()

    # ------------------------------------------------------------------
    # schedule generation
    # ------------------------------------------------------------------

    def _generate_schedules(self):
        self.return_schedule = self.schedule_config.build(
            self.effective_date, self.maturity_date, self.return_frequency,
            first_date=self._return_first_date,
            next_to_last_date=self._return_next_to_last_date,
        )
        self.interest_schedule = self.schedule_config.build(
            self.effective_date, self.maturity_date, self.interest_leg.frequency,
            first_date=self._interest_first_date,
            next_to_last_date=self._interest_next_to_last_date,
        )

    def _build_settle_map(self):
        if self._settlement_cal is None:
            self._settlement_cal = self.schedule_config.calendar

        if self.spot_lag > 0:
            all_dates: set[date] = set()
            for sched in (self.return_schedule, self.interest_schedule):
                for s, e, _, _ in sched:
                    all_dates.add(s)
                    all_dates.add(e)

            self._settle_map = {
                d: advance_business_days(d, self.spot_lag, self._settlement_cal)
                for d in all_dates
            }
        else:
            self._settle_map = {}

        self._final_settle_date = self.maturity_date

    def _build_equity_forward_tenors(
        self,
        future_periods: list[tuple[date, date, date, float]],
        val_date: date,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        float,
    ]:
        sc = self.schedule_config
        t_starts, t_ends, _, _ = compute_period_year_fractions(
            future_periods, val_date, sc.curve_day_counter,
        )

        if self.spot_lag <= 0:
            return t_starts, t_ends, None, None, 0.0

        val_settle = advance_business_days(
            val_date, self.spot_lag, self._settlement_cal,
        )
        ql_val_date = to_ql_date(val_date)
        ql_val_settle = to_ql_date(val_settle)
        dc = sc.curve_day_counter

        t_settle = dc.yearFraction(ql_val_date, ql_val_settle)

        t_starts_fwd = np.array([
            dc.yearFraction(
                ql_val_settle, to_ql_date(self._settle_map[s]),
            )
            for s, _, _, _ in future_periods
        ])
        t_ends_fwd = np.array([
            dc.yearFraction(
                ql_val_settle, to_ql_date(self._settle_map[e]),
            )
            for _, e, _, _ in future_periods
        ])

        return t_starts, t_ends, t_starts_fwd, t_ends_fwd, t_settle

    def _resolve_return_leg_initial_price(
        self,
        future_return: list[tuple[date, date, date, float]],
        t_starts: np.ndarray,
        t_starts_fwd: np.ndarray | None,
        fixings: dict[tuple[str, date], np.ndarray] | None,
    ) -> float | np.ndarray | None:
        """Resolve the start reference for the first return period when needed.

        For a truly in-progress first period, use the path-specific fixing
        stored at the period start when available; otherwise fall back to the
        scalar ``self.initial_price`` supplied at construction.
        """
        if not future_return:
            return self.initial_price

        t_starts_check = t_starts_fwd if t_starts_fwd is not None else t_starts
        initial_price = self.initial_price

        if t_starts_check[0] <= 0 and fixings is not None:
            period_start_date = future_return[0][0]
            stored = fixings.get((self.spot_name, period_start_date))
            if stored is not None:
                initial_price = stored

        return initial_price

    # ------------------------------------------------------------------
    # reset / fixing interface
    # ------------------------------------------------------------------

    def get_reset_dates(self) -> list[tuple[date, str, date, date, bool]]:
        """Return interest-leg floating reset dates only.

        Equity return-leg and interest-leg spot resets are handled by the
        dedicated get_equity_reset_schedule / _compute_equity_fixing_for_date
        path, which uses interpolated scenario states (more accurate) and
        mirrors RiskFlow's equity swaplet reset structure.  Keeping them here
        as well caused double-stamping — the equity fixing path always won via
        the merge-last rule in ExposureEngine.compute, but the wasted work and
        split responsibility created confusion.
        """
        from instruments.components.cashflow_leg import LegType

        resets: list[tuple[date, str, date, date, bool]] = []

        if self.interest_leg.leg_type == LegType.FLOATING:
            is_ois = self.interest_leg.overnight_compounding
            resets.extend(
                (start, self.interest_leg.curve_name, start, end, is_ois)
                for start, end, _, _accrual in self.interest_schedule
            )

        return resets

    def compute_fixings(
        self,
        resets: list[tuple[date, str, date, date]],
        time_slice: dict[str, RiskFactorSlice],
        scenario_date: date,
    ) -> dict[tuple[str, date], np.ndarray]:
        """Compute LIBOR forward rates from an earlier scenario's curve.

        Called by the exposure engine for non-overnight floating periods
        whose reset date precedes the current simulation date.
        """
        fixings: dict[tuple[str, date], np.ndarray] = {}
        sc = self.schedule_config
        ql_scenario = to_ql_date(scenario_date)
        leg = self.interest_leg

        for _reset_date, curve_name, p_start, p_end in resets:
            if curve_name == self.spot_name:
                spot_slice = time_slice[curve_name]
                fixings[(curve_name, p_start)] = np.asarray(
                    spot_slice.values, dtype=np.float64
                ).copy()
                continue

            fwd_slice: CurveSlice = time_slice[curve_name]
            fwd_curve = YieldCurve(
                year_fracs=fwd_slice.tenors,
                rates=fwd_slice.values,
                interpolator=self.discount_interpolator,
            )

            ql_p_start = to_ql_date(p_start)
            ql_p_end = to_ql_date(p_end)

            t_start = sc.curve_day_counter.yearFraction(ql_scenario, ql_p_start)

            if leg.fixing_tenor_months is not None:
                fwd_conv = (
                    BUSINESS_CONVENTIONS[leg.forward_business_convention]
                    if leg.forward_business_convention is not None
                    else ql.ModifiedFollowing
                )
                ql_fix_end = sc.ql_calendar.advance(
                    ql_p_start,
                    ql.Period(leg.fixing_tenor_months, ql.Months),
                    fwd_conv,
                )
                # t_end for DF lookup uses curve_day_counter (ACT/365).
                # fwd_tau (rate accrual denominator) uses day_counter so the
                # stored fixing is quoted in the coupon convention (e.g. ACT/360
                # for USD), consistent with the accruals used in leg_pv.
                t_end = sc.curve_day_counter.yearFraction(ql_scenario, ql_fix_end)
                fwd_tau = sc.day_counter.yearFraction(ql_p_start, ql_fix_end)
                fixings[(curve_name, p_start)] = fwd_curve.forward_rate(
                    t_start, t_end, tau=fwd_tau,
                )
            else:
                t_end = sc.curve_day_counter.yearFraction(ql_scenario, ql_p_end)
                fwd_tau = sc.day_counter.yearFraction(ql_p_start, ql_p_end)
                fixings[(curve_name, p_start)] = fwd_curve.forward_rate(
                    t_start, t_end, tau=fwd_tau,
                )

        return fixings

    def compute_cf_increment(
        self,
        curve_name: str,
        t_from: date,
        t_to: date,
        time_slice: dict[str, RiskFactorSlice],
    ) -> np.ndarray:
        """One-step OIS compound factor 1/DF(t_from → t_to).

        Used by the exposure engine to accumulate CF_realized for an
        in-progress OIS period across simulation time steps.
        """
        sc = self.schedule_config
        fwd_slice: CurveSlice = time_slice[curve_name]
        fwd_curve = YieldCurve(
            year_fracs=fwd_slice.tenors,
            rates=fwd_slice.values,
            interpolator=self.discount_interpolator,
        )
        tau = sc.curve_day_counter.yearFraction(
            to_ql_date(t_from), to_ql_date(t_to),
        )
        return 1.0 / fwd_curve.discount_factor(tau)  # (n_paths,)

    # ------------------------------------------------------------------
    # equity spot fixing interface (return-leg "Price" scaling)
    # ------------------------------------------------------------------

    def get_equity_reset_schedule(self) -> list[date]:
        """Return all equity spot dates that need a per-path fixing stamped.

        RiskFlow creates both start and end resets for each equity swaplet:
        - Start resets: needed for in-progress and future-period F_start.
        - End resets: needed for completed-but-unpaid periods where T_end has
          passed but cash has not yet settled (T_end < val_date <= T_end_settle).

        When interest_nominal_scaling is "Price", interest period starts are
        also included because the interest leg notional resets to F(T_{i-1}).
        """
        reset_dates: set[date] = set()
        for start, end, _, _accrual in self.return_schedule:
            reset_dates.add(start)
            reset_dates.add(end)
        if self.interest_nominal_scaling == "Price":
            for start, _end, _, _accrual in self.interest_schedule:
                reset_dates.add(start)
        return sorted(reset_dates)

    def _compute_equity_fixing_for_date(
        self,
        reset_date: date,
        fix_state: dict[str, RiskFactorSlice],
    ) -> dict[tuple, np.ndarray]:
        """Return the per-path equity spot at reset_date from fix_state.

        Called by ExposureEngine._build_equity_fixings with the market state
        most contemporary with reset_date (nearest scenario date ≤ reset_date).
        Returns a single-entry dict keyed by (spot_name, reset_date) so the
        engine can accumulate across steps.
        """
        spot_slice = fix_state[self.spot_name]
        return {
            (self.spot_name, reset_date): np.asarray(
                spot_slice.values, dtype=np.float64
            ).copy()
        }

    # ------------------------------------------------------------------
    # pricing
    # ------------------------------------------------------------------

    def scenario_npvs(
        self,
        val_date: date,
        market_state: dict[str, RiskFactorSlice],
        fixings: dict[tuple[str, date], np.ndarray] | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        spot_slice: ScalarSlice = market_state[self.spot_name]
        spot = spot_slice.values  # (n_paths,)
        n_paths = spot.shape[0]

        if val_date >= self.maturity_date:
            return np.zeros(n_paths)

        sc = self.schedule_config

        # -- Build curves --
        carry_slice: CurveSlice = market_state[self.carry_curve_name]
        carry_curve = YieldCurve(
            carry_slice.tenors, carry_slice.values, self.carry_interpolator,
        )

        div_slice: CurveSlice = market_state[self.dividend_curve_name]
        div_curve = YieldCurve(
            div_slice.tenors, div_slice.values, self.dividend_interpolator,
        )

        disc_slice: CurveSlice = market_state[self.discount_curve_name]
        disc_curve = YieldCurve(
            disc_slice.tenors, disc_slice.values, self.discount_interpolator,
        )

        # ------------------------------------------------------------------
        # Return leg — three cases mirroring RiskFlow's pv_equity_cashflows:
        #   1. future periods (val_date < T_start): both prices forward
        #   2. in-progress (T_start <= val_date < T_end): S_start locked
        #   3. completed-but-unpaid (T_end < val_date <= T_end_settle):
        #      both S_start and S_end locked; only arises when spot_lag > 0
        # ------------------------------------------------------------------
        future_return = filter_future_periods(
            self.return_schedule,
            val_date,
            include_on_val_date=self.include_sim_date_cashflows,
        )

        if future_return:
            (
                t_starts, t_ends, t_starts_fwd, t_ends_fwd, t_settle,
            ) = self._build_equity_forward_tenors(future_return, val_date)

            initial_price = self._resolve_return_leg_initial_price(
                future_return=future_return,
                t_starts=t_starts,
                t_starts_fwd=t_starts_fwd,
                fixings=fixings,
            )

            return_pv = trs_return_leg_pv(
                spot=spot,
                carry_curve=carry_curve,
                dividend_curve=div_curve,
                discount_curve=disc_curve,
                t_starts=t_starts,
                t_ends=t_ends,
                quantity=self.quantity,
                initial_price=initial_price,
                nominal_scaling=self.return_nominal_scaling,
                notional_fixed=self.notional,
                t_starts_fwd=t_starts_fwd,
                t_ends_fwd=t_ends_fwd,
                t_settle=t_settle,
            )
        else:
            return_pv = np.zeros(n_paths)

        # -- Interest leg --
        # "Price": per-period notional = F(T_{i-1}) × quantity (equity-forward scaled).
        # "Initial Price": fixed notional = self.notional (default leg_pv behaviour).
        notional_sched: np.ndarray | None = None
        if self.interest_nominal_scaling == "Price":
            future_int = filter_future_periods(
                self.interest_schedule,
                val_date,
                self.include_sim_date_cashflows,
            )

            if future_int:
                (
                    t_starts_int, _,
                    t_starts_int_fwd, _,
                    t_settle_int,
                ) = self._build_equity_forward_tenors(future_int, val_date)
                t_starts_check = (
                    t_starts_int_fwd
                    if t_starts_int_fwd is not None
                    else t_starts_int
                )

                notional_sched = np.empty((n_paths, len(future_int)))

                for idx, (p_start, _p_end, _, _accrual) in enumerate(future_int):
                    t_s_check = float(t_starts_check[idx])

                    stored = (
                        fixings.get((self.spot_name, p_start))
                        if fixings else None
                    )

                    if t_s_check < 0:
                        # Reset already happened: use realized start fixing.
                        if stored is not None:
                            ref = np.asarray(stored, dtype=np.float64)
                            notional_sched[:, idx] = (
                                ref
                                if ref.ndim == 1
                                else np.full(n_paths, float(ref))
                            ) * self.quantity
                        else:
                            notional_sched[:, idx] = spot * self.quantity
                    else:
                        # Future reset: forward to period start using forward
                        # rates anchored at val_settle when lag > 0.
                        t_s_full = (
                            t_settle_int + t_s_check
                            if t_settle_int > 0.0
                            else t_s_check
                        )
                        F_s, _, _ = equity_forward_price(
                            spot, carry_curve, div_curve,
                            t_s_full, t0=t_settle_int,
                        )
                        notional_sched[:, idx] = F_s * self.quantity

        interest_pv = leg_pv(
            self.interest_schedule, self.interest_leg,
            notional=self.notional,
            val_date=val_date,
            market_state=market_state,
            discount_curve=disc_curve,
            n_paths=n_paths,
            interpolator=self.discount_interpolator,
            day_counter=sc.day_counter,
            curve_day_counter=sc.curve_day_counter,
            calendar=sc.ql_calendar,
            fixings=fixings,
            include_on_val_date=self.include_sim_date_cashflows,
            notional_schedule=notional_sched,
        )

        direction = 1.0 if self.is_receiver else -1.0
        return direction * (return_pv - interest_pv)
