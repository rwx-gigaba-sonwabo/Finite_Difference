import os
from math import ceil, sqrt
from datetime import datetime

import matplotlib.pyplot as plt
import QuantLib as ql
import numpy as np
import pandas as pd

class VanillaOptionPricerTest:
    """
    A class to price vanilla European or American options using the finite difference method.
    """

    _K_DOMAIN     = 3    # log-space half-width multiplier; domain ≈ ± K_DOMAIN * σ * √T
    _XGRID_MIN    = 30     # minimum space nodes
    _TGRID_MIN    = 30     # minimum time steps
    _USE_RICHARDSON = True # enable Richardson (N vs N/2)

    def __init__(self,
                 spot_price: float,
                 strike_price: float,
                 discount_curve,
                 forward_curve,
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
        """
        Initialise market data, option parameters, and setup QuantLib objects.
        """

        # Set global valuation date
        ql.Settings.instance().evaluationDate = valuation_date

        # Store parameters
        self.spot_price = spot_price
        self.strike_price = strike_price
        self.discount_curve = discount_curve
        self.forward_curve = forward_curve
        self.volatility = volatility
        self.dividend_schedule = dividend_schedule
        self.valuation_date = valuation_date
        self.maturity_date = maturity_date
        self.contracts = int(contracts)
        self.contract_multiplier = float(contract_multiplier)
        self.side = side.lower().strip()
        self._side_sign = +1 if self.side in ("buy", "long", "+", "b") else -1
        self.option_type_str = option_type.lower()
        self.exercise_type_str = exercise_type.lower()
        self.option_spot_days = option_spot_days
        self.option_settlement_days = option_settlement_days
        self.underlying_spot_days = underlying_spot_days
        self.settlement_type = settlement_type.lower()
        self.calendar = calendar
        self.day_counter = day_counter
        self.trade_number = trade_number

        # Check if the dividend schedule is empty
        if not self.dividend_schedule:
            self.dividend_schedule_ql = []  # No dividends expected
        else:
            self.dividend_schedule_ql = [
                ql.FixedDividend(amount, pay_date) for pay_date, amount in self.dividend_schedule
            ]

        # Set up time-period adjustments
        # Time to expiry from valuation to maturity
        self.time_to_expiry = self.day_counter.yearFraction(
            self.valuation_date, self.maturity_date
        )

        self.carry_start = self.calendar.advance(
            self.valuation_date, self.underlying_spot_days, ql.Days
        )
        if self.settlement_type == "physical":
            self.carry_end = self.calendar.advance(
                self.maturity_date, self.option_settlement_days, ql.Days
            )
        else:
            self.carry_end = self.calendar.advance(
                self.maturity_date, self.underlying_spot_days, ql.Days
            )
        # Time to carry, used for growing underlying asset to forward value
        self.time_to_carry = self.day_counter.yearFraction(self.carry_start, self.carry_end)

        self.discount_start = self.calendar.advance(
            self.valuation_date, self.option_spot_days, ql.Days)
        self.discount_end = self.calendar.advance(
            self.maturity_date, self.option_settlement_days, ql.Days
        )
        # Time to discount, used for present valuing the payoff
        self.time_to_discount = self.day_counter.yearFraction(
            self.valuation_date, self.discount_end
        )

        # Set up market data curves (continuous compounding)
        self.discount_curve = self._add_nacc_and_dfs(self.discount_curve)
        self.forward_curve = self._add_nacc_and_dfs(self.forward_curve)
        self.discount_rate = self.get_nacc_rate(self.discount_end)
        self.carry_rate = self.get_forward_nacc_rate(self.carry_start, self.carry_end)
        self.dividend_yield = self.dividend_yield_nacc()
        self.underlying = ql.SimpleQuote(self.spot_price)
        self.pv_dividends = self.pv_dividend()

        # curves anchored at t
        self.dividend_curve = ql.FlatForward(
            0,
            self.calendar,
            0,
            self.day_counter,
            ql.Continuous,
            ql.Annual
        )

        self.carry_curve = ql.FlatForward(
            0, self.calendar,
            self.carry_rate, self.day_counter,
            ql.Continuous, ql.Annual
        )

        self.volatility_curve = ql.BlackConstantVol(
            0, self.calendar,
            volatility, self.day_counter
        )

        self.yield_curve = self.build_quantlib_yield_curve_ts(self.discount_curve)

        tau_value_to_cs = self.day_counter.yearFraction(self.valuation_date, self.carry_start)
        nacc_rate_value_to_cs = self.get_nacc_rate(self.carry_start)
        tau_carry_start_to_carry_end = self.day_counter.yearFraction(self.carry_start, self.carry_end)
        self.s_physical = self.spot_price * np.exp(- self.dividend_yield * tau_carry_start_to_carry_end) * np.exp(-self.discount_rate * tau_value_to_cs)

        if self.option_type_str == "call":
            if self.spot_price <= self.strike_price:
                self.s_cash = (self.spot_price - self.pv_dividends)
            elif self.spot_price > self.strike_price:
                self.s_cash = self.spot_price
        else:
            self.s_cash = (self.spot_price- self.pv_dividends)

        if self.settlement_type == "physical":
            self.underlying.setValue(self.s_physical)
        else:
            self.underlying.setValue(self.s_cash)

        # Build pricing process
        self.process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(self.underlying),
            ql.YieldTermStructureHandle(self.dividend_curve),
            ql.YieldTermStructureHandle(self.carry_curve),
            ql.BlackVolTermStructureHandle(self.volatility_curve)
        )

        # Define option payoff and exercise
        self.option_type = getattr(ql.Option, self.option_type_str.capitalize())
        self.payoff = ql.PlainVanillaPayoff(self.option_type, strike_price)
        self.exercise = self._get_exercise()

        # Construct QuantLib vanilla option
        self.option = ql.VanillaOption(self.payoff, self.exercise)

    def get_nacc_rate(self, lookup_date: ql.Date) -> float:
        lookup_date_str = lookup_date.ISO()
        row = self.discount_curve[self.discount_curve["Date"] == lookup_date_str]

        # Check if the row is empty
        if row.empty:
            print(f"Warning: NACC rate not found for date: {lookup_date_str}. Returning default value.")
            return 0.0  # or some other default value

        naca_rate = row["NACA"].values[0]
        nacc_rate = np.log(1 + naca_rate)
        return nacc_rate

    def get_forward_nacc_rate(self, start_date: ql.Date, end_date: ql.Date) -> float:
        DF_far = self.get_discount_factor(end_date)
        DF_near = self.get_discount_factor(start_date)
        forward_tau = self.day_counter.yearFraction(start_date, end_date)

        forward_nacc_rate = -np.log(DF_far / DF_near) * (1 / forward_tau)
        return forward_nacc_rate

    def get_discount_factor(self, lookup_date: ql.Date) -> float:
        lookup_date_str = lookup_date.ISO()
        row = self.discount_curve[self.discount_curve["Date"] == lookup_date_str]

        # Check
        if row.empty:
            raise ValueError(f"Discount factor not found for date: {lookup_date_str}")

        naca_rate = row["NACA"].values[0]
        tau = self.day_counter.yearFraction(self.valuation_date, lookup_date)
        DF = (1 + naca_rate)**(-tau)
        return DF

    def _add_nacc_and_dfs(self, data: pd.DataFrame) -> pd.DataFrame:
        # Convert the 'Date' column to datetime format
        data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')

        # Create a QuantLib date from the 'Date' column
        data['QuantLib_Date'] = data['Date'].apply(lambda date: ql.Date(date.day, date.month, date.year))

        # Calculate NACC
        data["NACC"] = data["QuantLib_Date"].apply(lambda date: self.get_nacc_rate(date))

        # Calculate Discount Factor
        data["Discount_Factor"] = data.apply(
            lambda row: self.get_discount_factor(row["QuantLib_Date"]), axis=1)

        return data

    def pv_dividend(self) -> float:
        """PV Dividends to carry_start date"""
        # If no dividends are expected, return 0
        if not self.dividend_schedule_ql:
            return 0.0

        pv = 0.0
        for dividend in self.dividend_schedule_ql:
            pay_date = dividend.date()
            amount = dividend.amount()
            tau = self.day_counter.yearFraction(self.carry_start, pay_date)
            forward_nacc = self.get_forward_nacc_rate(self.carry_start, pay_date)
            df = np.exp(-forward_nacc * tau)

            pv += amount * df
        return pv


    def dividend_yield_nacc(self):
        """Back-out a flat q (NACC) that reproduces the PV of discrete dividends over [carry_start, carry_end]"""
        pv_divs = self.pv_dividend()
        spot_price = self.spot_price
        tau = self.time_to_carry

        if spot_price <= pv_divs:
            raise ValueError("Present value of dividends cannot be greater than or equal to the spot price.")

        dividend_yield_nacc = -np.log((spot_price - pv_divs) / spot_price) * (1 / tau)
        return dividend_yield_nacc

    def build_quantlib_yield_curve_ts(self, data: pd.DataFrame) -> ql.YieldTermStructureHandle:
        """
        Build a QuantLib yield curve from the DataFrame.
        """
        # Use the attributes of the Timestamp object directly
        dates = [ql.Date(int(d.day), int(d.month), int(d.year)) for d in data['Date']]
        naccs = data['NACC'].tolist()

        curve = ql.ZeroCurve(dates, naccs, self.day_counter, self.calendar)
        curve.enableExtrapolation()
        return ql.YieldTermStructureHandle(curve)

    def _scale(self, x: float) -> float:
        return self._side_sign * self.contracts * self.contract_multiplier * x

    def _get_exercise(self):
        if self.exercise_type_str == "european":
            return ql.EuropeanExercise(self.carry_end)
        elif self.exercise_type_str == "american":
            return ql.AmericanExercise(self.discount_start, self.discount_end)
        else:
            raise ValueError(f"Unsupported exercise type: {self.exercise_type_str}")

    # -------------------- NEW: Mesher + Backward solver --------------------
    def _xgrid_from_time(self, t_steps: int) -> int:
        """
        Keep a light rule for xGrid size; the mesher owns the actual spacing/clustering.
        You can keep your previous N->M mapping if you like; we just avoid hand-computing
        domain bounds now, since the mesher handles it.
        """
        N = max(self._TGRID_MIN, int(t_steps))
        # Simple, stable mapping: proportional to sqrt(N), but at least _XGRID_MIN and odd.
        M = max(self._XGRID_MIN, int(5.0 * sqrt(N)))
        if M % 2 == 0:
            M += 1
        return M

    def _build_solver(self, t_steps: int, damping_steps: int = 2,
                      scheme: ql.FdmSchemeDesc = ql.FdmSchemeDesc.CrankNicolson()):
        """
        Try to build a full FDM solver stack with an explicit FdmBlackScholesMesher.
        If your QuantLib build lacks the solver constructor, return None and the caller
        will fall back to the classic engine.
        """
        N = max(self._TGRID_MIN, int(t_steps))
        M = self._xgrid_from_time(N)
        T = max(1e-12, self.day_counter.yearFraction(self.valuation_date, self.discount_end))

        # Mesher: prefer the signature with (size, process, T, strike); concentrate at K automatically.
        try:
            mesher = ql.FdmBlackScholesMesher(M, self.process, T, self.strike_price)
        except Exception:
            # Fallback signature: without strike hint.
            mesher = ql.FdmBlackScholesMesher(M, self.process, T)

        mesh = ql.FdmMesherComposite(mesher)

        # Boundary conditions: none for vanilla (natural BCs)
        bc_set = []

        # Step conditions: American projection + dividends (if any)
        try:
            # If you pass a discrete schedule (list of ql.FixedDividend), use it; else empty
            div_sched = self.dividend_schedule_ql if self.dividend_schedule_ql else []
            step_cond = ql.FdmStepConditionComposite.vanillaComposite(div_sched, self.exercise, mesh, self.payoff)
        except Exception:
            # Older bindings: build empty composite (American only)
            step_cond = ql.FdmStepConditionComposite.vanillaComposite([], self.exercise, mesh, self.payoff)

        # Inner value calculator (payoff on log-mesh)
        inner = ql.FdmLogInnerValue(self.payoff, mesh, 0)

        # Prefer the convenience solver that can give value, delta, gamma at S directly
        try:
            solver = ql.FdmBlackScholesSolver(self.process,
                                              self.strike_price,
                                              scheme,
                                              mesh,
                                              bc_set,
                                              step_cond,
                                              inner,
                                              T,
                                              N,
                                              damping_steps)
            return solver
        except Exception:
            return None  # caller will fall back to engine

    # -------------------- pricing paths --------------------
    def _price_once_via_solver(self, t_steps: int,
                               scheme: ql.FdmSchemeDesc = ql.FdmSchemeDesc.CrankNicolson()):
        solver = self._build_solver(t_steps, damping_steps=2, scheme=scheme)
        if solver is None:
            raise RuntimeError("FdmBlackScholesSolver not available in this QuantLib build.")

        s_query = float(self.underlying.value())
        pv = float(solver.valueAt(s_query))

        tau_maturity_to_discEnd = self.day_counter.yearFraction(self.maturity_date, self.carry_end)
        tau_value_to_discStart = self.day_counter.yearFraction(self.valuation_date, self.carry_start)
        corr_physical = np.exp(self.discount_rate * tau_value_to_discStart + tau_maturity_to_discEnd)
        corr_cash_nacc = self.get_forward_nacc_rate(self.maturity_date, self.carry_end)
        corr_cash = np.exp(-corr_cash_nacc * tau_maturity_to_discEnd)
        if self.settlement_type == "physical":
            adjusted_pv = pv #*corr_physical
        else:
            adjusted_pv = pv * corr_cash
        return solver, adjusted_pv  # return solver so greeks can reuse its grid

    def _price_once_fallback_engine(self, t_steps: int,
                                    scheme: ql.FdmSchemeDesc = ql.FdmSchemeDesc.CrankNicolson()):
        """
        If the advanced FDM solver path is unavailable, revert to the classic engine.
        (This preserves old behavior; mesher won't be explicit in this path.)
        """
        N = max(self._TGRID_MIN, int(t_steps))
        M = self._xgrid_from_time(N)
        engine = ql.FdBlackScholesVanillaEngine(
            self.process,
            self.dividend_schedule_ql,  # empty list is OK
            N, M,
            2,
            scheme
        )
        self.option.setPricingEngine(engine)
        pv_engine = float(self.option.NPV())

        
        tau_maturity_to_discEnd = self.day_counter.yearFraction(self.maturity_date, self.carry_end)
        tau_value_to_discStart = self.day_counter.yearFraction(self.valuation_date, self.carry_start)
        corr_physical = np.exp(self.discount_rate * tau_value_to_discStart + tau_maturity_to_discEnd)
        corr_cash_nacc = self.get_forward_nacc_rate(self.maturity_date, self.carry_end)
        corr_cash = np.exp(-corr_cash_nacc * tau_maturity_to_discEnd)
        if self.settlement_type == "physical":
            adjusted_pv = pv_engine #*corr_physical
        else:
            adjusted_pv = pv_engine * corr_cash
        
        return adjusted_pv
    
    def _price_once(self, t_steps: int, scheme=ql.FdmSchemeDesc.CrankNicolson()):
        try:
            pv, _ = self._price_once_via_solver(t_steps, scheme)
            return pv
        except Exception:
            return self._price_once_fallback_engine(t_steps, scheme)

    # -------------------- public API (unchanged signatures) --------------------
    def price(self, time_steps: int, scheme=ql.FdmSchemeDesc.CrankNicolson()):
        pN = self._price_once(time_steps, scheme)
        if not self._USE_RICHARDSON:
            return self._scale(pN)

        half = max(self._TGRID_MIN, int(time_steps // 2))
        pH = self._price_once(half, scheme)
        p_star = (4.0 * pN - pH) / 3.0
        return self._scale(p_star)

    def batch_price(self, time_steps_list):
        return {int(steps): self.price(int(steps)) for steps in time_steps_list}

    def calculate_greeks(self, time_steps):
        """
        Prefer analytic grid-based greeks from FDM solver if available; else bump-and-revalue.
        """
        N = int(time_steps[-1]) if isinstance(time_steps, (list, tuple)) else int(time_steps)

        # Try solver greeks
        try:
            pv, solver = self._price_once_via_solver(N)
            s_query = float(self.underlying.value())
            delta = float(solver.deltaAt(s_query))
            gamma = float(solver.gammaAt(s_query))

            # Vega: FDM solver class often lacks vega; keep small-forward bump
            dv = 1e-4
            self.volatility_curve = ql.BlackConstantVol(self.valuation_date, self.calendar,
                                                        self.volatility + dv, self.day_counter)
            self.process = ql.BlackScholesMertonProcess(
                ql.QuoteHandle(self.underlying),
                ql.YieldTermStructureHandle(self.dividend_curve),
                ql.YieldTermStructureHandle(self.risk_free_curve),
                ql.BlackVolTermStructureHandle(self.volatility_curve)
            )
            pv_up = self._price_once(N)
            vega = (pv_up - pv) / dv

            # Theta (per-year) by bumping valuation date one business day
            vd1 = self.calendar.advance(self.valuation_date, 1, ql.Days)
            old_eval = ql.Settings.instance().evaluationDate
            ql.Settings.instance().evaluationDate = vd1
            try:
                pv_t1 = self._price_once(N)
            finally:
                ql.Settings.instance().evaluationDate = old_eval

            dt = self.day_counter.yearFraction(self.valuation_date, vd1)
            theta_annual = (pv_t1 - pv) / dt
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

        except Exception:
            # Fallback: classic bump-and-revalue on price path (no Richardson for greeks)
            p0 = self._price_once(N)
            bump_rel = 1e-4
            dS = bump_rel * self.spot_price
            self.underlying.setValue(self.spot_price + dS)
            p_up = self._price_once(N)
            self.underlying.setValue(self.spot_price - dS)
            p_dn = self._price_once(N)
            self.underlying.setValue(self.spot_price)
            delta = (p_up - p_dn) / (2.0 * dS)
            gamma = (p_up - 2.0 * p0 + p_dn) / (dS ** 2)

            dv = 1e-4
            self.volatility_curve = ql.BlackConstantVol(self.valuation_date, self.calendar,
                                                        self.volatility + dv, self.day_counter)
            self.process = ql.BlackScholesMertonProcess(
                ql.QuoteHandle(self.underlying),
                ql.YieldTermStructureHandle(self.dividend_curve),
                ql.YieldTermStructureHandle(self.carry_curve),
                ql.BlackVolTermStructureHandle(self.volatility_curve)
            )
            p_vu = self._price_once(N)
            vega = (p_vu - p0) / dv

            vd1 = self.calendar.advance(self.valuation_date, 1, ql.Days)
            old_eval = ql.Settings.instance().evaluationDate
            ql.Settings.instance().evaluationDate = vd1
            try:
                p1 = self._price_once(N)
            finally:
                ql.Settings.instance().evaluationDate = old_eval
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

    # -------------------- plotting & reports (unchanged) --------------------
    def plot_price_convergence(self, time_steps_list, style='seaborn-v0_8-darkgrid'):
        prices = self.batch_price(time_steps_list)
        if style in plt.style.available:
            plt.style.use(style)
        plt.figure(figsize=(10, 6))
        plt.plot(list(prices.keys()), list(prices.values()), marker='o', linestyle='-')
        plt.title(f"{self.option_type_str} Option Price vs Time Steps ({self.exercise_type_str} Exercise, Richardson)")
        plt.xlabel("Time Steps (N)")
        plt.ylabel("Option Price (scaled)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def export_report(self, time_steps_list):
        results = self.batch_price(time_steps_list)
        timestamp = datetime.now().strftime("%Y-%m-%d %H%M%S")
        filename = f"vanilla_option_report {timestamp}.txt"
        report_lines = [
            "Vanilla Option Pricing Report",
            f"Generated: {timestamp}",
            "-" * 40,
            "Market        : ZAR (example)",
            "Calendar      : South Africa",
            f"Day Count     : {self.day_counter.name()}",
            f"Trade number  : {self.trade_number}",
            "-" * 40,
            f"Option Type     : {self.option_type_str}",
            f"Exercise Type   : {self.exercise_type_str}",
            f"Spot Price      : {self.spot_price}",
            f"Strike Price    : {self.strike_price}",
            f"Risk-Free Rate  : {self.risk_free_rate}",
            f"Volatility      : {self.volatility}",
            f"Dividend Yield  : {self.dividend_yield}",
            f"Valuation Date  : {self.valuation_date.ISO()}",
            f"Maturity Date   : {self.maturity_date.ISO()}",
            f"Position        : {self.side}",
            f"Contracts       : {self.contracts}",
            f"Contract Mult   : {self.contract_multiplier}",
            "",
            "Pricing Results (Time Steps -> Option Price):"
        ]
        for steps, price in results.items():
            report_lines.append(f"  {steps:5} steps : {price:.6f}")
        if hasattr(self, "greeks"):
            report_lines.append(f"\nGreeks (calculated with {self.greeks_time_step} time steps):")
            for greek, value in self.greeks.items():
                report_lines.append(f"  {greek:<14}: {value:.6f}")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filename)
        with open(filepath, "w") as f:
            f.write("\n".join(report_lines))
        print(f"Report saved to: {filepath}")
