from __future__ import annotations

from datetime import date
from typing import Callable

import numpy as np

from instruments.components.cashflow_leg import CashflowLeg, LegType, SwapLeg  # noqa: F401 — re-export
from instruments.components.schedule_config import ScheduleConfig
from instruments.instrument import Instrument
import QuantLib as ql

from utils.ql_helpers import to_ql_date, BUSINESS_CONVENTIONS, generate_sub_periods
from market_data.yield_curve import YieldCurve
from market_data.risk_factor import CurveSlice, RiskFactorSlice
from models.cashflow_pv import leg_pv, _build_overnight_tenors


# ---------------------------------------------------------------------------
# IRSwap
# ---------------------------------------------------------------------------

class IRSwap(Instrument):
    """Interest-rate swap priced against a ScenarioCube.

    Schedules are generated once at construction via QuantLib.
    On each call to :meth:`price`, a yield curve is built per path
    from the simulated curve factor in *market_state*, forward rates
    are computed for each accrual period, and cashflows are discounted.
    """

    def __init__(
        self,
        name: str,
        effective_date: date,
        maturity_date: date,
        notional: float,
        receive_leg: SwapLeg,
        pay_leg: SwapLeg,
        discount_curve_name: str,
        interpolator: Callable,
        schedule_config: ScheduleConfig | None = None,
        # legacy flat params — used only when schedule_config is not supplied
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
        self.receive_leg = receive_leg
        self.pay_leg = pay_leg
        self.discount_curve_name = discount_curve_name
        self.interpolator = interpolator
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

    # ------------------------------------------------------------------
    # schedule generation
    # ------------------------------------------------------------------

    def _generate_schedules(self):
        """Build QuantLib schedules and store accrual info for both legs."""
        self.receive_schedule = self.schedule_config.build(
            self.effective_date, self.maturity_date, self.receive_leg.frequency,
        )
        self.pay_schedule = self.schedule_config.build(
            self.effective_date, self.maturity_date, self.pay_leg.frequency,
        )
        self._effective_maturity: date = max(
            max(p for _, _, p, _ in self.receive_schedule),
            max(p for _, _, p, _ in self.pay_schedule),
        )

    # ------------------------------------------------------------------
    # reset / fixing interface
    # ------------------------------------------------------------------

    def get_reset_dates(self) -> list[tuple[date, str, date, date, bool]]:
        """Return (reset_date, curve_name, p_start, p_end, is_overnight).

        When a leg has ``reset_frequency_months > 0`` each payment period is
        split into sub-periods and one reset tuple is emitted per sub-period.
        The ExposureEngine caches each sub-period rate at its reset date so
        that ``leg_pv`` can compound them at pricing time.
        """
        resets: list[tuple[date, str, date, date, bool]] = []
        sc = self.schedule_config
        for schedule, leg in [
            (self.receive_schedule, self.receive_leg),
            (self.pay_schedule, self.pay_leg),
        ]:
            if leg.leg_type != LegType.FLOATING:
                continue
            if leg.reset_frequency_months > 0:
                for pay_start, pay_end, _, _ in schedule:
                    for sub_start, sub_end, _ in generate_sub_periods(
                        pay_start, pay_end, leg.reset_frequency_months,
                        sc.ql_calendar, sc.ql_convention, sc.day_counter,
                        direction="Backward",
                    ):
                        resets.append(
                            (sub_start, leg.curve_name, sub_start, sub_end, False)
                        )
            else:
                for start, end, _, _accrual in schedule:
                    resets.append(
                        (start, leg.curve_name, start, end, leg.overnight_compounding)
                    )
        return resets

    def compute_cf_increment(
        self,
        curve_name: str,
        t_from: date,
        t_to: date,
        time_slice: dict[str, RiskFactorSlice],
    ) -> np.ndarray:
        """One-step OIS compound factor ∏ DF(dᵢ)/DF(dᵢ₊₁) over [t_from, t_to].

        Builds the business-day tenor grid between t_from and t_to (tenors
        measured from t_from, the scenario origin) and returns the product of
        consecutive discount-factor ratios.  By telescoping this equals
        DF(0)/DF(τ) = 1/DF(τ), matching the scalar shortcut exactly.  The
        daily grid is kept explicit so the accumulation mirrors the overnight
        compounding structure used in _calc_forward_rates for future periods,
        and extends naturally to lockout / lookback variants.

        Parameters
        ----------
        curve_name : str
            OIS projection curve key in *time_slice*.
        t_from, t_to : date
            Endpoints of the step (consecutive simulation dates).
        time_slice : dict
            Market state at *t_from*.

        Returns
        -------
        np.ndarray, shape (n_paths,)
            Compound factor for [t_from, t_to].
        """
        sc = self.schedule_config
        fwd_slice: CurveSlice = time_slice[curve_name]
        fwd_curve = YieldCurve(
            year_fracs=fwd_slice.tenors,
            rates=fwd_slice.values,
            interpolator=self.interpolator,
        )
        t_sched = _build_overnight_tenors(
            t_from, t_to,
            val_date=t_from,
            calendar=sc.ql_calendar,
            curve_day_counter=sc.curve_day_counter,
        )
        dfs = fwd_curve.discount_factor(t_sched)          # (n_paths, n_bdays+1)
        return np.prod(dfs[:, :-1] / dfs[:, 1:], axis=1)  # (n_paths,)

    def compute_fixings(
        self,
        resets: list[tuple[date, str, date, date]],
        time_slice: dict[str, RiskFactorSlice],
        scenario_date: date,
    ) -> dict[tuple[str, date], np.ndarray]:
        """Compute forward rates for the given resets from an earlier scenario's curve."""
        fixings: dict[tuple[str, date], np.ndarray] = {}
        sc = self.schedule_config
        ql_scenario = to_ql_date(scenario_date)

        # Map curve_name → leg so we can look up fixing_tenor_months
        leg_by_curve = {
            leg.curve_name: leg
            for leg in (self.receive_leg, self.pay_leg)
            if leg.leg_type == LegType.FLOATING and leg.curve_name
        }

        for _reset_date, curve_name, p_start, p_end in resets:
            fwd_slice: CurveSlice = time_slice[curve_name]
            fwd_curve = YieldCurve(
                year_fracs=fwd_slice.tenors,
                rates=fwd_slice.values,
                interpolator=self.interpolator,
            )
            t_start = sc.curve_day_counter.yearFraction(
                ql_scenario, to_ql_date(p_start),
            )

            leg = leg_by_curve.get(curve_name)
            if leg is not None and leg.fixing_tenor_months is not None:
                fwd_conv = (
                    BUSINESS_CONVENTIONS[leg.forward_business_convention]
                    if leg.forward_business_convention is not None
                    else ql.ModifiedFollowing
                )
                ql_fix_end = sc.ql_calendar.advance(
                    to_ql_date(p_start),
                    ql.Period(leg.fixing_tenor_months, ql.Months),
                    fwd_conv,
                )
                t_end = sc.curve_day_counter.yearFraction(ql_scenario, ql_fix_end)
                fwd_tau = sc.day_counter.yearFraction(to_ql_date(p_start), ql_fix_end)
                rate = fwd_curve.forward_rate(t_start, t_end, tau=fwd_tau)
            else:
                t_end = sc.curve_day_counter.yearFraction(
                    ql_scenario, to_ql_date(p_end),
                )
                rate = fwd_curve.forward_rate(t_start, t_end)

            fixings[(curve_name, p_start)] = rate

        return fixings

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
        disc_slice: CurveSlice = market_state[self.discount_curve_name]
        n_paths = disc_slice.values.shape[0]

        if val_date > self._effective_maturity:
            return np.zeros(n_paths)

        discount_curve = YieldCurve(
            year_fracs=disc_slice.tenors,
            rates=disc_slice.values,
            interpolator=self.interpolator,
        )

        sc = self.schedule_config
        # On the effective maturity date include cashflows due today so
        # the final coupon is not silently dropped (matches RiskFlow behaviour).
        include_on_date = (
            self.include_sim_date_cashflows
            or val_date == self._effective_maturity
        )
        common_kwargs = dict(
            notional=self.notional,
            val_date=val_date,
            market_state=market_state,
            discount_curve=discount_curve,
            n_paths=n_paths,
            interpolator=self.interpolator,
            day_counter=sc.day_counter,
            curve_day_counter=sc.curve_day_counter,
            calendar=sc.ql_calendar,
