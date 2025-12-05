import os
from math import ceil, sqrt
from datetime import datetime

import matplotlib.pyplot as plt
import QuantLib as ql
import numpy as np
import pandas as pd


class VanillaOptionPricerTest:
    """
    FDM (Crank–Nicolson) pricer for vanilla European / American options,
    using FIS-style mesher, time grid and your yield-curve / day-count setup.

    - American early exercise is handled via QuantLib FdmStepConditionComposite
      with an AmericanExercise (discount_start -> discount_end).
    - European exercise uses the same PDE but without early-exercise projection.
    """

    _K_DOMAIN       = 3    # log-space half-width multiplier; domain ≈ ± K_DOMAIN * σ * √T
    _XGRID_MIN      = 30   # minimum space nodes (will be made odd)
    _TGRID_MIN      = 30   # kept for API compat (used as min x-grid)
    _USE_RICHARDSON = True # enable Richardson (fine vs half x-grid)
    _DAMPING_STEPS  = 2    # Rannacher start

    def __init__(self,
                 spot_price: float,
                 strike_price: float,
                 discount_curve: pd.DataFrame,
                 forward_curve: pd.DataFrame,
                 volatility: float,
                 dividend_schedule: list[tuple[ql.Date, float]],
                 valuation_date: ql.Date,
                 maturity_date: ql.Date,
                 contracts: int,
                 contract_multiplier: float,
                 side: str,
                 option_type: str,
                 exercise_type: str,
                 option_spot_days: int,
                 option_settlement_days: int,
                 underlying_spot_days: int,
                 settlement_type: str,
                 calendar,
                 day_counter,
                 trade_number: int = None):

        ql.Settings.instance().evaluationDate = valuation_date

        # --- store inputs ---
        self.spot_price = float(spot_price)
        self.strike_price = float(strike_price)
        self.discount_curve = discount_curve.copy()
        self.forward_curve = forward_curve.copy()
        self.volatility = float(volatility)
        self.dividend_schedule = dividend_schedule
        self.valuation_date = valuation_date
        self.maturity_date = maturity_date
        self.contracts = int(contracts)
        self.contract_multiplier = float(contract_multiplier)
        self.side = side.lower().strip()
        self._side_sign = +1 if self.side in ("buy", "long", "+", "b") else -1
        self.option_type_str = option_type.lower()
        self.exercise_type_str = exercise_type.lower()
        self.option_spot_days = int(option_spot_days)
        self.option_settlement_days = int(option_settlement_days)
        self.underlying_spot_days = int(underlying_spot_days)
        self.settlement_type = settlement_type.lower()
        self.calendar = calendar
        self.day_counter = day_counter
        self.trade_number = trade_number

        # --- dividends to QuantLib schedule (amount, date) ---
        if not self.dividend_schedule:
            self.dividend_schedule_ql = []
        else:
            # incoming tuples are (pay_date, amount)
            self.dividend_schedule_ql = [
                ql.FixedDividend(amount, pay_date)
                for pay_date, amount in self.dividend_schedule
            ]
        self._ql_div_schedule = ql.DividendSchedule(self.dividend_schedule_ql)

        # --- key dates / times ---
        self.time_to_expiry = self.day_counter.yearFraction(self.valuation_date,
                                                            self.maturity_date)

        self.carry_start = self.calendar.advance(self.valuation_date,
                                                 self.underlying_spot_days, ql.Days)
        if self.settlement_type == "physical":
            self.carry_end = self.calendar.advance(self.maturity_date,
                                                   self.option_settlement_days, ql.Days)
        else:
            self.carry_end = self.calendar.advance(self.maturity_date,
                                                   self.underlying_spot_days, ql.Days)
        self.time_to_carry = self.day_counter.yearFraction(self.carry_start,
                                                           self.carry_end)

        self.discount_start = self.calendar.advance(self.valuation_date,
                                                    self.option_spot_days, ql.Days)
        self.discount_end = self.calendar.advance(self.maturity_date,
                                                  self.option_settlement_days, ql.Days)
        self.time_to_discount = self.day_counter.yearFraction(self.valuation_date,
                                                              self.discount_end)

        # --- curves / rates / PV(divs) ---
        # NOTE: this assumes your input curves already contain the "NACA" column etc.
        self.discount_curve = self._add_nacc_and_dfs(self.discount_curve)
        self.forward_curve  = self._add_nacc_and_dfs(self.forward_curve)

        self.discount_rate  = self.get_nacc_rate(self.discount_end)
        self.carry_rate     = self.get_forward_nacc_rate(self.carry_start, self.carry_end)
        self.underlying     = ql.SimpleQuote(self.spot_price)
        self.pv_dividends   = self.pv_dividend()

        # dividend TS is zero (we model discrete cash dividends explicitly)
        self.dividend_curve = ql.FlatForward(
            0, self.calendar, 0.0, self.day_counter,
            ql.Continuous, ql.Annual
        )

        # risk-free/carry curve used inside PDE
        self.carry_curve = ql.FlatForward(
            0, self.calendar, self.carry_rate,
            self.day_counter, ql.Continuous, ql.Annual
        )

        self.volatility_curve = ql.BlackConstantVol(
            0, self.calendar, self.volatility, self.day_counter
        )

        # Optional: a handle to your discount zero curve (not used by the PDE directly)
        self.yield_curve = self.build_quantlib_yield_curve_ts(self.discount_curve)

        # --- effective S used by process (cash vs physical) ---
        tau_value_to_cs = self.day_counter.yearFraction(self.valuation_date,
                                                        self.carry_start)
        tau_carry_start_to_carry_end = self.day_counter.yearFraction(self.carry_start,
                                                                     self.carry_end)

        # flat dividend yield that reproduces PV(divs) over [carry_start, carry_end]
        self.dividend_yield = self.dividend_yield_nacc()

        self.s_physical = (self.spot_price
                           * np.exp(-self.dividend_yield * tau_carry_start_to_carry_end)
                           * np.exp(-self.discount_rate * tau_value_to_cs))

        if self.option_type_str == "call":
            if self.spot_price <= self.strike_price:
                self.s_cash = self.spot_price - self.pv_dividends
            else:
                self.s_cash = self.spot_price
        else:
            self.s_cash = self.spot_price - self.pv_dividends

        if self.settlement_type == "physical":
            self.underlying.setValue(self.s_physical)
        else:
            self.underlying.setValue(self.s_cash)

        # --- process / payoff / exercise / vanilla option shell ---
        self.process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(self.underlying),
            ql.YieldTermStructureHandle(self.dividend_curve),  # =0 to avoid double-counting
            ql.YieldTermStructureHandle(self.carry_curve),
            ql.BlackVolTermStructureHandle(self.volatility_curve)
        )

        self.option_type = getattr(ql.Option, self.option_type_str.capitalize())
        self.payoff = ql.PlainVanillaPayoff(self.option_type, self.strike_price)
        self.exercise = self._get_exercise()  # American or European depending on input
        self.option = ql.VanillaOption(self.payoff, self.exercise)  # only used for payoff meta

    # -------------------------------------------------------------------------
    # Utilities (unchanged in spirit)
    # -------------------------------------------------------------------------

    def get_nacc_rate(self, lookup_date: ql.Date) -> float:
        lookup_date_str = lookup_date.ISO()
        row = self.discount_curve[self.discount_curve["Date"] == lookup_date_str]
        if row.empty:
            print(f"Warning: NACC rate not found for date: {lookup_date_str}. Returning 0.")
            return 0.0
        naca_rate = row["NACA"].values[0]
        return np.log(1 + naca_rate)

    def get_forward_nacc_rate(self, start_date: ql.Date, end_date: ql.Date) -> float:
        DF_far  = self.get_discount_factor(end_date)
        DF_near = self.get_discount_factor(start_date)
        forward_tau = self.day_counter.yearFraction(start_date, end_date)
        return -np.log(DF_far / DF_near) / forward_tau

    def get_discount_factor(self, lookup_date: ql.Date) -> float:
        lookup_date_str = lookup_date.ISO()
        row = self.discount_curve[self.discount_curve["Date"] == lookup_date_str]
        if row.empty:
            raise ValueError(f"Discount factor not found for date: {lookup_date_str}")
        naca_rate = row["NACA"].values[0]
        tau = self.day_counter.yearFraction(self.valuation_date, lookup_date)
        return (1 + naca_rate) ** (-tau)

    def _add_nacc_and_dfs(self, data: pd.DataFrame) -> pd.DataFrame:
        # Assumes incoming `data` already has a "Date" column and a NACA input column.
        data = data.copy()
        data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
        data['QuantLib_Date'] = data['Date'].apply(lambda d: ql.Date(d.day, d.month, d.year))
        # Here we assume "NACA" is already present and consistent with your inputs.
        # We recompute Discount_Factor for completeness.
        data["Discount_Factor"] = data.apply(
            lambda row: self.get_discount_factor(row["QuantLib_Date"]), axis=1
        )
        return data

    def pv_dividend(self) -> float:
        if not self.dividend_schedule_ql:
            return 0.0
        pv = 0.0
        for div in self.dividend_schedule_ql:
            pay_date = div.date()
            if pay_date <= self.valuation_date or pay_date > self.maturity_date:
                continue
            tau = self.day_counter.yearFraction(self.carry_start, pay_date)
            fwd = self.get_forward_nacc_rate(self.carry_start, pay_date)
            pv += div.amount() * np.exp(-fwd * tau)
        return pv

    def dividend_yield_nacc(self) -> float:
        pv_divs = self.pv_dividend()
        S = self.spot_price
        tau = max(1e-12, self.time_to_carry)
        if pv_divs >= S:
            raise ValueError("PV(dividends) >= spot.")
        return -np.log((S - pv_divs) / S) / tau

    def build_quantlib_yield_curve_ts(self, data: pd.DataFrame) -> ql.YieldTermStructureHandle:
        dates = [ql.Date(int(d.day), int(d.month), int(d.year)) for d in data['Date']]
        naccs = data['NACC'].tolist()
        curve = ql.ZeroCurve(dates, naccs, self.day_counter, self.calendar)
        curve.enableExtrapolation()
        return ql.YieldTermStructureHandle(curve)

    def _scale(self, x: float) -> float:
        return self._side_sign * self.contracts * self.contract_multiplier * x

    def _get_exercise(self):
        # IMPORTANT: this is the hook that makes the solver American vs European.
        if self.exercise_type_str.lower() == "european":
            # Exercise only at carry_end (consistent with physical vs cash logic)
            return ql.EuropeanExercise(self.carry_end)
        elif self.exercise_type_str.lower() == "american":
            # Early exercise possible from discount_start to discount_end
            return ql.AmericanExercise(self.discount_start, self.discount_end)
        else:
            raise ValueError(f"Unsupported exercise type: {self.exercise_type_str}")

    # -------------------------------------------------------------------------
    # FIS-style mesher + time grid + FDM backward solver
    # -------------------------------------------------------------------------

    def _div_times_from_eval(self) -> list[float]:
        """Dividend pay times measured from valuation date (t=0)."""
        ts = []
        for div in self.dividend_schedule_ql:
            d = div.date()
            if self.valuation_date < d < self.discount_end:
                ts.append(self.day_counter.yearFraction(self.valuation_date, d))
        ts = sorted(list(set(ts)))
        return ts

    def _build_mesher(self, x_grid: int) -> tuple[ql.FdmMesherComposite, float]:
        """Black–Scholes mesher; returns mesher and approx ΔlogS."""
        T = max(1e-12, self.time_to_discount)
        width = self._K_DOMAIN * self.volatility * sqrt(T)
        logS0 = np.log(max(1e-12, self.underlying.value()))
        x_min = logS0 - width
        x_max = logS0 + width
        if x_grid % 2 == 0:
            x_grid += 1  # enforce symmetry around logS0
        bs_mesher = ql.FdmBlackScholesMesher(
            x_grid, self.process, T, self.strike_price, x_min, x_max
        )
        dx = (x_max - x_min) / max(1, (x_grid - 1))
        return ql.FdmMesherComposite(bs_mesher), dx

    def _segment_dt(self, dlogS: float, T_ref: float) -> float:
        # FIS: Δt ≈ ΔlogS / (2 σ) * sqrt(T_ref)
        return max(1e-12, dlogS / (2.0 * self.volatility) * sqrt(max(1e-12, T_ref)))

    def _build_time_grid(self, dx: float) -> ql.TimeGrid:
        """
        Piecewise grid with mandatory nodes at dividends and expiry;
        applies FIS Δt relation per segment.
        """
        pivots = [0.0] + self._div_times_from_eval() + [max(1e-12, self.time_to_discount)]
        times = [0.0]
        for i in range(1, len(pivots)):
            t0, t1 = pivots[i - 1], pivots[i]
            seg_len = t1 - t0
            # reference: to next dividend for pre-div segments; to expiry for last
            T_ref = pivots[i] if i < len(pivots) - 1 else pivots[-1]
            dt_star = self._segment_dt(dx, T_ref)
            n = max(1, int(ceil(seg_len / dt_star)))
            for k in range(1, n + 1):
                tk = t0 + seg_len * (k / n)
                if tk > times[-1] + 1e-12:
                    times.append(min(tk, self.time_to_discount))
        if abs(times[-1] - self.time_to_discount) > 1e-12:
            times[-1] = self.time_to_discount
        return ql.TimeGrid(times)

    def _build_solver(self, x_grid: int,
                      scheme: ql.FdmSchemeDesc) -> ql.FdmBackwardSolver:
        """
        Build the Crank–Nicolson backward solver.

        - For American exercise, the FdmStepConditionComposite includes
          both dividend jumps and early-exercise projection.
        - For European, only dividends matter (no early exercise).
        """
        mesher, dx = self._build_mesher(max(self._XGRID_MIN, int(x_grid)))
        tgrid = self._build_time_grid(dx)

        op = ql.FdmBlackScholesOp(mesher, self.process)
        bc = ql.FdmBoundaryConditionSet()
        inner = ql.FdmLogInnerValue(self.payoff, mesher, 0)

        # This is where "barrier logic" from your discrete pricer is
        # conceptually swapped for "early exercise" logic:
        # - American: apply max(V, payoff) at each time level (free boundary)
        # - European: only impose payoff at maturity, no early exercise
        step = ql.FdmStepConditionComposite.vanillaComposite(
            self._ql_div_schedule, self.exercise, mesher, inner, tgrid
        )

        return ql.FdmBackwardSolver(op, bc, step, scheme, tgrid)

    # -------------------------------------------------------------------------
    # Pricing & Greeks via custom FD solver
    # -------------------------------------------------------------------------

    def _price_once(self, t_steps: int,
                    scheme: ql.FdmSchemeDesc = ql.FdmSchemeDesc.CrankNicolson()) -> float:
        """
        One FDM price run on a given xGrid/tGrid, with Rannacher damping
        and settlement correction.
        """
        x_grid = max(self._XGRID_MIN, int(t_steps))
        solver = self._build_solver(x_grid, scheme)

        # Rannacher: first few steps fully implicit, then CN
        solver.rollback(self.time_to_discount, self._DAMPING_STEPS)
        pv_at_t0 = float(solver.valueAt(self.underlying.value()))

        # settlement window correction (cash-settled vs physical)
        tau_maturity_to_discEnd = self.day_counter.yearFraction(
            self.maturity_date, self.carry_end
        )
        corr_cash_nacc = self.get_forward_nacc_rate(self.maturity_date, self.carry_end)
        corr_cash = np.exp(-corr_cash_nacc * tau_maturity_to_discEnd)

        if self.settlement_type == "physical":
            adjusted_pv = pv_at_t0
        else:
            adjusted_pv = pv_at_t0 * corr_cash

        return adjusted_pv

    def price_fd(self, time_steps: int,
                 scheme: ql.FdmSchemeDesc = ql.FdmSchemeDesc.CrankNicolson()) -> float:
        """
        Public FDM price (with optional Richardson extrapolation).
        """
        pN = self._price_once(time_steps, scheme)
        if not self._USE_RICHARDSON:
            return self._scale(pN)
        half = max(self._XGRID_MIN, int(max(1, time_steps // 2)))
        pH = self._price_once(half, scheme)
        return self._scale((4.0 * pN - pH) / 3.0)

    # Backwards-compatible name (if you were already calling .price)
    def price(self, time_steps: int,
              scheme: ql.FdmSchemeDesc = ql.FdmSchemeDesc.CrankNicolson()) -> float:
        return self.price_fd(time_steps, scheme)

    def batch_price(self, time_steps_list):
        return {int(steps): self.price(int(steps)) for steps in time_steps_list}

    def plot_price_convergence(self, time_steps_list,
                               style: str = 'seaborn-v0_8-darkgrid'):
        prices = self._clone().batch_price(time_steps_list)
        if style in plt.style.available:
            plt.style.use(style)
        plt.figure(figsize=(10, 6))
        plt.plot(list(prices.keys()), list(prices.values()),
                 marker='o', linestyle='-')
        plt.title(f"{self.option_type_str} Option Price vs Time Steps "
                  f"({self.exercise_type_str} Exercise)")
        plt.xlabel("Space grid size (xGrid)")
        plt.ylabel("Option Price (scaled)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def _clone(self, *, spot_price=None,
               volatility=None, valuation_date=None) -> "VanillaOptionPricerTest":
        return VanillaOptionPricerTest(
            spot_price=self.spot_price if spot_price is None else spot_price,
            strike_price=self.strike_price,
            discount_curve=self.discount_curve,
            forward_curve=self.forward_curve,
            volatility=self.volatility if volatility is None else volatility,
            dividend_schedule=self.dividend_schedule,
            valuation_date=self.valuation_date if valuation_date is None else valuation_date,
            maturity_date=self.maturity_date,
            contracts=self.contracts,
            contract_multiplier=self.contract_multiplier,
            side=self.side,
            option_type=self.option_type_str.capitalize(),
            exercise_type=self.exercise_type_str,
            option_spot_days=self.option_spot_days,
            option_settlement_days=self.option_settlement_days,
            underlying_spot_days=self.underlying_spot_days,
            settlement_type=self.settlement_type,
            calendar=self.calendar,
            day_counter=self.day_counter,
            trade_number=self.trade_number
        )

    def calculate_greeks(self, time_steps):
        """
        Greeks via bump-and-revalue on the FD engine (same grid), to keep
        behaviour transparent and aligned with your barrier pricer.
        """
        N = (int(time_steps[-1]) if isinstance(time_steps, (list, tuple))
             else int(time_steps))
        p0 = self._price_once(N)

        # --- Delta / Gamma: bump underlying (cash vs physical) ---
        bump_rel = 1e-4  # 1bp relative bump in S
        if self.settlement_type == "cash":
            base = self.s_cash
            up = self._clone()
            up.s_cash = (1 + bump_rel) * base
            up.underlying.setValue(up.s_cash)

            dn = self._clone()
            dn.s_cash = (1 - bump_rel) * base
            dn.underlying.setValue(dn.s_cash)
        else:
            base = self.s_physical
            up = self._clone()
            up.s_physical = (1 + bump_rel) * base
            up.underlying.setValue(up.s_physical)

            dn = self._clone()
            dn.s_physical = (1 - bump_rel) * base
            dn.underlying.setValue(dn.s_physical)

        p_up = up._price_once(N)
        p_dn = dn._price_once(N)
        dS = base * bump_rel
        delta = (p_up - p_dn) / (2.0 * dS)
        gamma = (p_up - 2.0 * p0 + p_dn) / (dS ** 2)

        # --- Vega: bump vol ---
        bump_v = 1e-4
        p_vu = self._clone(volatility=self.volatility + bump_v)._price_once(N)
        vega = (p_vu - p0) / (bump_v * 100.0)  # per 1% vol

        # --- Theta: shift valuation date by one calendar day ---
        vd1 = self.calendar.advance(self.valuation_date, 1, ql.Days)
        p1 = self._clone(valuation_date=vd1).price(N)
        dt = self.day_counter.yearFraction(self.valuation_date, vd1)
        theta_annual = (p1 - p0) / dt
        theta_daily = theta_annual / 365.0

        greeks = {
            "Delta": self._scale(delta),
            "Gamma": self._scale(gamma),
            "Vega": self._scale(vega),
            "Theta (Annual)": self._scale(theta_annual),
            "Theta (Daily)": self._scale(theta_daily)
        }
        self.greeks = greeks
        self.greeks_time_step = N
        return greeks
