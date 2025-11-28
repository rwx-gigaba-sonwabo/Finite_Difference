import math
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Optional

import QuantLib as ql


@dataclass
class MarketParams:
    spot: float
    strike: float
    barrier: float              # single barrier for up/down
    rebate: float               # cash rebate
    r: float                    # risk-free (ccy) rate (flat, cont.comp.)
    q: float                    # dividend / repo yield (flat, cont.comp.)
    sigma: float                # flat Black vol
    maturity: dt.date           # calendar date of maturity
    valuation_date: dt.date     # valuation date
    day_count: ql.DayCounter = ql.Actual365Fixed()
    calendar: ql.Calendar = ql.Target()
    business_convention: ql.BusinessDayConvention = ql.Following()


class QLDiscreteBarrierPricer:
    """
    Discrete(-ish) barrier pricer using QuantLib FD engines.

    - Crank–Nicolson scheme (with Rannacher handled by QuantLib).
    - Space grid controlled via grid_points.
    - Time grid chosen so that barrier monitoring dates are well-resolved.
    - Knock-out directly via FdBlackScholesBarrierEngine.
    - Knock-in via standard knock-in/knock-out parity with the vanilla price.
    """

    def __init__(
        self,
        market: MarketParams,
        is_call: bool,
        barrier_type: str,            # "up-and-out", "down-and-out", "up-and-in", "down-and-in"
        monitoring_dates: List[dt.date],  # actual calendar dates when barrier is observed
        grid_points: int = 200,       # spatial grid points for FD
        min_time_steps: int = 200,    # minimum time steps
        steps_per_monitor: int = 4    # refine time grid relative to monitoring frequency
    ):
        self.market = market
        self.is_call = is_call
        self.barrier_type_str = barrier_type.lower()
        self.monitoring_dates = sorted(monitoring_dates)
        self.grid_points = grid_points
        self.min_time_steps = min_time_steps
        self.steps_per_monitor = steps_per_monitor

        # set up QuantLib environment
        self._build_term_structures()
        self._build_process()

    # ---------- setup ---------- #

    def _build_term_structures(self) -> None:
        m = self.market
        cal = m.calendar

        self.valuation_qldate = ql.Date(m.valuation_date.day,
                                        m.valuation_date.month,
                                        m.valuation_date.year)
        ql.Settings.instance().evaluationDate = self.valuation_qldate

        self.maturity_qldate = ql.Date(m.maturity.day,
                                       m.maturity.month,
                                       m.maturity.year)

        # flat yield curves
        r_curve = ql.FlatForward(self.valuation_qldate,
                                 ql.QuoteHandle(ql.SimpleQuote(m.r)),
                                 m.day_count)
        q_curve = ql.FlatForward(self.valuation_qldate,
                                 ql.QuoteHandle(ql.SimpleQuote(m.q)),
                                 m.day_count)

        self.r_ts = ql.YieldTermStructureHandle(r_curve)
        self.q_ts = ql.YieldTermStructureHandle(q_curve)

        # flat vol
        vol_ts = ql.BlackConstantVol(self.valuation_qldate,
                                     cal,
                                     m.sigma,
                                     m.day_count)
        self.vol_ts = ql.BlackVolTermStructureHandle(vol_ts)

    def _build_process(self) -> None:
        m = self.market
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(m.spot))
        self.process = ql.BlackScholesMertonProcess(spot_handle,
                                                    self.q_ts,
                                                    self.r_ts,
                                                    self.vol_ts)

    # ---------- helper: mapping barrier type ---------- #

    def _ql_barrier_type_KO(self) -> ql.Barrier.Type:
        if self.barrier_type_str.startswith("up"):
            return ql.Barrier.UpOut
        elif self.barrier_type_str.startswith("down"):
            return ql.Barrier.DownOut
        else:
            raise ValueError("Barrier type must be up-*/down-* for KO")

    def _ql_barrier_type_KI(self) -> ql.Barrier.Type:
        if self.barrier_type_str.startswith("up"):
            return ql.Barrier.UpIn
        elif self.barrier_type_str.startswith("down"):
            return ql.Barrier.DownIn
        else:
            raise ValueError("Barrier type must be up-*/down-* for KI")

    # ---------- grid / engine construction ---------- #

    def _time_steps_for_discrete_barrier(self) -> int:
        """
        Choose N_t so that each monitoring date aligns well with a time layer.

        We take:
            N_t = max(min_time_steps, steps_per_monitor * n_monitors)
        and also ensure at least ~sqrt(GridPoints)*T*something if you want.
        """
        n_mon = max(1, len(self.monitoring_dates))
        return max(self.min_time_steps, self.steps_per_monitor * n_mon)

    def _build_fd_barrier_engine(self) -> ql.PricingEngine:
        time_steps = self._time_steps_for_discrete_barrier()
        grid_points = self.grid_points

        scheme = ql.FdmSchemeDesc.CrankNicolson()  # theta=0.5, Rannacher done in FdmBackwardSolver

        engine = ql.FdBlackScholesBarrierEngine(
            self.process,
            time_steps,
            grid_points,
            scheme
        )
        return engine

    def _build_fd_vanilla_engine(self) -> ql.PricingEngine:
        time_steps = self._time_steps_for_discrete_barrier()
        grid_points = self.grid_points

        scheme = ql.FdmSchemeDesc.CrankNicolson()
        engine = ql.FdBlackScholesVanillaEngine(
            self.process,
            time_steps,
            grid_points,
            scheme
        )
        return engine

    # ---------- option objects ---------- #

    def _plain_vanilla_option(self) -> ql.VanillaOption:
        if self.is_call:
            payoff = ql.PlainVanillaPayoff(ql.Option.Call, self.market.strike)
        else:
            payoff = ql.PlainVanillaPayoff(ql.Option.Put, self.market.strike)
        exercise = ql.EuropeanExercise(self.maturity_qldate)
        return ql.VanillaOption(payoff, exercise)

    def _barrier_option_KO(self) -> ql.BarrierOption:
        """Knock-OUT option with same payoff as the underlying vanilla."""
        if self.is_call:
            payoff = ql.PlainVanillaPayoff(ql.Option.Call, self.market.strike)
        else:
            payoff = ql.PlainVanillaPayoff(ql.Option.Put, self.market.strike)

        barrier_type = self._ql_barrier_type_KO()
        barrier = self.market.barrier
        rebate = self.market.rebate

        exercise = ql.EuropeanExercise(self.maturity_qldate)
        return ql.BarrierOption(barrier_type, barrier, rebate, payoff, exercise)

    def _barrier_option_KI(self) -> ql.BarrierOption:
        """Knock-IN version (useful if you ever want a direct FD KI price)."""
        if self.is_call:
            payoff = ql.PlainVanillaPayoff(ql.Option.Call, self.market.strike)
        else:
            payoff = ql.PlainVanillaPayoff(ql.Option.Put, self.market.strike)

        barrier_type = self._ql_barrier_type_KI()
        barrier = self.market.barrier
        rebate = self.market.rebate

        exercise = ql.EuropeanExercise(self.maturity_qldate)
        return ql.BarrierOption(barrier_type, barrier, rebate, payoff, exercise)

    # ---------- pricing & Greeks ---------- #

    def price_vanilla_FD(self) -> Dict[str, float]:
        opt = self._plain_vanilla_option()
        opt.setPricingEngine(self._build_fd_vanilla_engine())

        return {
            "price": opt.NPV(),
            "delta": opt.delta(),
            "gamma": opt.gamma(),
            "theta": opt.theta(),   # calendar theta (per year)
            "vega": opt.vega(),
        }

    def price_KO_FD(self) -> Dict[str, float]:
        opt = self._barrier_option_KO()
        opt.setPricingEngine(self._build_fd_barrier_engine())

        return {
            "price": opt.NPV(),
            "delta": opt.delta(),
            "gamma": opt.gamma(),
            "theta": opt.theta(),
            "vega": opt.vega(),
        }

    def price_KI_from_parity(self) -> Dict[str, float]:
        """
        Use KI = Vanilla - KO for price; Greeks from the same parity identity.

        For a single barrier (no double) and same vanilla underlier,
        in a standard Black–Scholes setting:

            V_vanilla = V_KI + V_KO

        ⇒  V_KI = V_vanilla − V_KO
        and similarly for the sensitivities under the same model.
        """
        v = self.price_vanilla_FD()
        ko = self.price_KO_FD()

        ki = {g: v[g] - ko[g] for g in v.keys()}
        return ki

    # convenience: choose KO vs KI based on barrier_type string

    def price_and_greeks(self) -> Dict[str, float]:
        if "out" in self.barrier_type_str:
            return self.price_KO_FD()
        elif "in" in self.barrier_type_str:
            return self.price_KI_from_parity()
        else:
            raise ValueError("barrier_type must contain 'in' or 'out'.")


# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Example: up-and-out call, daily monitoring approximated via FD time grid
    val_date = dt.date(2025, 1, 2)
    maturity = dt.date(2025, 7, 2)

    market = MarketParams(
        spot=229.74,
        strike=200.0,
        barrier=270.0,
        rebate=0.0,
        r=0.08,       # 8% flat
        q=0.0,
        sigma=0.2784,
        maturity=maturity,
        valuation_date=val_date,
        day_count=ql.Actual365Fixed(),
        calendar=ql.TARGET()
    )

    # Daily monitoring dates (for illustration, here we do monthly grid, but in
    # your real code you would pass actual daily date list)
    calendar = market.calendar
    mon_dates = []
    d = calendar.adjust(ql.Date(val_date.day, val_date.month, val_date.year))
    mat = ql.Date(maturity.day, maturity.month, maturity.year)
    while d <= mat:
        mon_dates.append(dt.date(d.year(), d.month(), d.dayOfMonth()))
        d = calendar.advance(d, 1, ql.Days)

    pricer = QLDiscreteBarrierPricer(
        market=market,
        is_call=True,
        barrier_type="up-and-out",
        monitoring_dates=mon_dates,
        grid_points=400,
        min_time_steps=600,
        steps_per_monitor=2,
    )

    res = pricer.price_and_greeks()
    print("KO price + Greeks (FD / CN / QuantLib):")
    for k, v in res.items():
        print(f"{k:6s}: {v:.8f}")
