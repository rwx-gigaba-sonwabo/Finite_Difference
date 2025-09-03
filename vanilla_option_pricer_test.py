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

    _K_DOMAIN     = 1    # log-space half-width multiplier; domain ≈ ± K_DOMAIN * σ * √T
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
            ql.Continuous,  # Compounding type
            ql.Annual  # Frequency
        )

        self.carry_curve = ql.FlatForward(
            0, self.calendar,
            self.discount_rate, self.day_counter,
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
        self.s_cash = self.spot_price #- self.pv_dividends) #* np.exp(-nacc_rate_value_to_cs * tau_value_to_cs)

        if self.settlement_type == "physical":
            self.underlying.setValue(self.s_physical)
        else:
            self.underlying.setValue(self.s_cash)

        # Build pricing process
        self.process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(self.underlying),
            ql.YieldTermStructureHandle(self.dividend_curve),
            self.yield_curve,
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
            tau = self.day_counter.yearFraction(self.discount_start, pay_date)
            forward_nacc = self.get_forward_nacc_rate(self.discount_start, pay_date)
            df = np.exp(-forward_nacc * tau)

            pv += amount * df
        return pv


    def dividend_yield_nacc(self):
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
        """Scale a per-contract PV/greek by contracts, multiplier, and side (+/-)."""
        return self._side_sign * self.contracts * self.contract_multiplier * x

    def _get_exercise(self):
        """
        Return QuantLib exercise object based on user input.
        """
        if self.exercise_type_str.lower() == "european":
            return ql.EuropeanExercise(self.discount_end)
        elif self.exercise_type_str.lower() == "american":
            return ql.AmericanExercise(self.discount_start, self.discount_end)
        else:
            raise ValueError(f"Unsupported exercise type: {self.exercise_type_str}")

    def _domain_width_L(self) -> float:
        # Width in log S used by QL’s mesher internally; pick ±K_DOMAIN σ √T on each side
        T = max(1e-12, self.time_to_discount)
        return 2.0 * self._K_DOMAIN * self.volatility * sqrt(T)

    def _xgrid_for(self, t_steps: int) -> int:
        N = max(self._TGRID_MIN, int(t_steps))
        T = max(1e-12, self.time_to_discount)
        L = self._domain_width_L()
        M = int(np.ceil((N * L) / (2.0 * self.volatility * (T**1.5))))
        return max(self._XGRID_MIN, M)

    def _engine(self, t_steps: int, scheme=ql.FdmSchemeDesc.CrankNicolson()):
        """Build engine with optimal x_grid for the given N and Rannacher start-up (dampingSteps=2)."""
        N = max(self._TGRID_MIN, int(t_steps))
        M = self._xgrid_for(N)
        # Rannacher: two implicit-Euler start steps; improves stability near kinks
        damping_steps = 2
        return ql.FdBlackScholesVanillaEngine(self.process, self.dividend_schedule_ql,N, M, damping_steps, scheme)

    def _price_once(self, t_steps: int, scheme=ql.FdmSchemeDesc.CrankNicolson()):
        engine = self._engine(t_steps, scheme)
        self.option.setPricingEngine(engine)
        pv_engine_at_optSpot = float(self.option.NPV())


        tau_maturity_to_discEnd = self.day_counter.yearFraction(self.maturity_date, self.carry_end)
        tau_value_to_discStart = self.day_counter.yearFraction(self.valuation_date, self.carry_start)
        corr_physical = np.exp(self.discount_rate * tau_value_to_discStart + tau_maturity_to_discEnd)
        corr_cash_nacc = self.get_forward_nacc_rate(self.maturity_date, self.carry_end)
        corr_cash = np.exp(-corr_cash_nacc * tau_maturity_to_discEnd)
        if self.settlement_type == "physical":
            adjusted_pv = pv_engine_at_optSpot #*corr_physical
        else:
            adjusted_pv = pv_engine_at_optSpot * corr_cash
        return adjusted_pv

    def price(self, time_steps: int, scheme=ql.FdmSchemeDesc.CrankNicolson()):
        """
        Price the option using finite difference method with:
          • Richardson extrapolation on time grid (default on)
        """
        pN = self._price_once(time_steps, scheme)
        if not self._USE_RICHARDSON:
            return pN

        half = max(self._TGRID_MIN, int(time_steps // 2))
        pH = self._price_once(half, scheme)
        # Richardson (second-order in time): P* ≈ (4 P_N - P_{N/2}) / 3
        price_per_contract = (4.0 * pN - pH) / 3.0
        return self._scale(price_per_contract)

    def batch_price(self, time_steps_list):
        """
        Price the option over a range of time steps ( Richardson on).
        """
        return {int(steps): self.price(int(steps)) for steps in time_steps_list}

    def plot_price_convergence(self, time_steps_list, style='seaborn-v0_8-darkgrid'):
        """
        Plot option price as a function of time step granularity.
        """
        prices = self._clone().batch_price(time_steps_list)

        if style in plt.style.available:
            plt.style.use(style)
        else:
            print(f"Style '{style}' not found. Using default.")

        plt.figure(figsize=(10, 6))
        plt.plot(list(prices.keys()), list(prices.values()), marker='o', linestyle='-')
        plt.title(
            f"{self.option_type_str} Option Price vs Time Steps "
            f"({self.exercise_type_str} Exercise, Richardson)"
        )
        plt.xlabel("Time Steps (N)")
        plt.ylabel("Option Price")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def _clone(self, *,
               spot_price=None,
               volatility=None,
               valuation_date=None) -> "VanillaOptionPricerTest":
        """Minimal internal cloner for greeks"""
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
            option_type=self.option_type_str.capitalize() if hasattr(str, "capitalize") else self.option_type_str,
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
        Calculate greeks via bump-and-revalue with Richardson-disabled pricing.
        Returns Delta, Gamma, Vega, and Theta (per year and per day).
        """
        N = int(time_steps[-1]) if isinstance(time_steps, (list, tuple)) else int(time_steps)

        # Base price
        #engine = self._engine(N)
        #self.option.setPricingEngine(engine)
        p0 = self._price_once(N)
        bump = 1e-4 * 100

        #if hasattr(self.option, "delta") and hasattr(self.option, "gamma"):
         #   delta = self.option.delta()
          #  gamma = self.option.gamma()

        if self.settlement_type == "cash":
            #bump up S prime
            option_up = self._clone()
            option_up.s_cash = (1 + bump) * self.s_cash
            option_up.underlying.setValue(option_up.s_cash)
            p_up = option_up._price_once(N)

            #bump down S prime
            option_down = self._clone()
            option_down.s_cash = (1 - bump) * self.s_cash
            option_down.underlying.setValue(option_down.s_cash)
            p_dn = option_down._price_once(N)

            d_s_prime = self.s_cash * bump
            delta =  (p_up - p_dn) / (2.0 * d_s_prime)
            gamma =  (p_up - 2.0 * p0 + p_dn) / (d_s_prime ** 2)
        else:
            # bump up S prime
            option_up = self._clone()
            option_up.s_physical = (1 + bump) * self.s_physical
            option_up.underlying.setValue(option_up.s_physical)
            p_up = option_up._price_once(N)

            # bump down S prime
            option_down = self._clone()
            option_down.s_physical = (1 - bump) * self.s_physical
            option_down.underlying.setValue(option_down.s_physical)
            p_dn = option_down._price_once(N)

            d_s_prime = self.s_physical * bump
            delta = (p_up - p_dn) / (2.0 * d_s_prime)
            gamma = (p_up - 2.0 * p0 + p_dn) / (d_s_prime ** 2)

        # Vega (vol bumps, absolute 0.10% by default)
        bump_v = 1e-4
        p_vu = self._clone(volatility=self.volatility + bump_v)._price_once(N)
        vega = (p_vu - p0) / (bump_v*100)

        # Theta (date bump by 1 business day → per year; daily also returned)
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

    def export_report(self, time_steps_list):
        """
        Export a report containing pricing results and greeks (if calculated).
        """
        results = self._clone().batch_price(time_steps_list)
        timestamp = datetime.now().strftime("%Y-%m-%d %H%M%S")
        filename = f"vanilla_option_report {timestamp}.txt"

        report_lines = [
            "Vanilla Option Pricing Report",
            f"Generated: {timestamp}",
            "-" * 40,
            "Market        : South Africa (ZAR-denominated option)",
            "Calendar      : South Africa",
            "Day Count     : Actual/365 (Fixed)",
            f"Trade number : {self.trade_number}",
            "-" * 40,
            f"Option Type     : {self.option_type_str}",
            f"Exercise Type   : {self.exercise_type_str}",
            f"Spot Price      : {self.spot_price}",
            f"Strike Price    : {self.strike_price}",
            f"Carry_rate      : {self.carry_rate}",
            f"Discount_rate   : {self.discount_rate}",
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

    def export_report_fx(self, time_steps_list):
        """
        Export a version of the report tailored for FX options (domestic/foreign rates).
        """
        results = self._clone().batch_price(time_steps_list)
        timestamp = datetime.now().strftime("%Y-%m-%d %H%M%S")
        filename = f"vanilla_option_report {timestamp}.txt"

        report_lines = [
            "Vanilla Option Pricing Report (FX)",
            f"Generated: {timestamp}",
            "-" * 40,
            "Market        : South Africa (ZAR-denominated option)",
            "Calendar      : South Africa",
            "Day Count     : Actual/365 (Fixed)",
            f"Trade number : {self.trade_number}",
            "-" * 40,
            f"Option Type              : {self.option_type_str}",
            f"Exercise Type            : {self.exercise_type_str}",
            f"Spot Price               : {self.spot_price}",
            f"Strike Price             : {self.strike_price}",
            f"Carry_rate               : {self.carry_rate}",
            f"Discount_rate            : {self.discount_rate}",
            f"Dividend_yield           : {self.dividend_yield}",
            f"Volatility               : {self.volatility}",
            f"Valuation Date           : {self.valuation_date.ISO()}",
            f"Maturity Date            : {self.maturity_date.ISO()}",
            f"Position                 : {self.side}",
            f"Contracts                : {self.contracts}",
            f"Contract Mult            : {self.contract_multiplier}",
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
