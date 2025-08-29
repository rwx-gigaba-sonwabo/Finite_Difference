import os
from math import ceil, sqrt
from datetime import datetime

import matplotlib.pyplot as plt
import QuantLib as ql
import numpy as np

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
                 risk_free_rate: float,
                 volatility: float,
                 dividend_yield: float,
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
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.dividend_yield = dividend_yield
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
        self.underlying = ql.SimpleQuote(self.spot_price)

        # curves anchored at t + option_spot_days
        self.dividend_curve = ql.FlatForward(
            self.option_spot_days, self.calendar,
            0, self.day_counter,
            ql.Continuous, ql.Annual
        )

        self.risk_free_curve = ql.FlatForward(
            self.underlying_spot_days, self.calendar,
            risk_free_rate, self.day_counter,
            ql.Continuous, ql.Annual
        )

        self.volatility_curve = ql.BlackConstantVol(
            self.option_spot_days, self.calendar,
            volatility, self.day_counter
        )

        # -------- Spot alignment: make drift run over [t+optSpot → t+U] before PDE --------
        # a drift with (r - q) reproduces your intended forward window.
        tau_value_to_cs = self.day_counter.yearFraction(self.discount_start, self.carry_start)
        tau_carry_start_to_carry_end = self.day_counter.yearFraction(self.carry_start, self.carry_end)
        self.s_physical = self.spot_price * np.exp(-(self.risk_free_rate - self.dividend_yield) * tau_value_to_cs)
        self.s_cash = self.spot_price * np.exp(-self.dividend_yield * tau_carry_start_to_carry_end)

        if self.settlement_type == "physical":
            self.underlying.setValue(self.s_physical)
        else:
            self.underlying.setValue(self.s_cash)

        # Build pricing process
        self.process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(self.underlying),
            ql.YieldTermStructureHandle(self.dividend_curve),
            ql.YieldTermStructureHandle(self.risk_free_curve),
            ql.BlackVolTermStructureHandle(self.volatility_curve)
        )

        # Define option payoff and exercise
        self.option_type = getattr(ql.Option, self.option_type_str.capitalize())
        self.payoff = ql.PlainVanillaPayoff(self.option_type, strike_price)
        self.exercise = self._get_exercise()

        # Construct QuantLib vanilla option
        self.option = ql.VanillaOption(self.payoff, self.exercise)

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
            return ql.AmericanExercise(self.carry_start, self.carry_end)
        else:
            raise ValueError(f"Unsupported exercise type: {self.exercise_type_str}")

    def _domain_width_L(self) -> float:
        # Width in log S used by QL’s mesher internally; pick ±K_DOMAIN σ √T on each side
        T = max(1e-12, self.time_to_expiry)
        return 2.0 * self._K_DOMAIN * self.volatility * sqrt(T)

    def _xgrid_for(self, t_steps: int) -> int:
        N = max(self._TGRID_MIN, int(t_steps))
        T = max(1e-12, self.time_to_expiry)
        L = self._domain_width_L()
        M = int(np.ceil((N * L) / (2.0 * self.volatility * (T**1.5))))
        return max(self._XGRID_MIN, M)

    def _engine(self, t_steps: int, scheme=ql.FdmSchemeDesc.CrankNicolson()):
        """Build engine with optimal x_grid for the given N and Rannacher start-up (dampingSteps=2)."""
        N = max(self._TGRID_MIN, int(t_steps))
        M = self._xgrid_for(N)
        # Rannacher: two implicit-Euler start steps; improves stability near kinks
        damping_steps = 2
        return ql.FdBlackScholesVanillaEngine(self.process, N, M, damping_steps, scheme)

    def _price_once(self, t_steps: int, scheme=ql.FdmSchemeDesc.CrankNicolson()):
        engine = self._engine(t_steps, scheme)
        self.option.setPricingEngine(engine)
        pv_engine_at_optSpot = float(self.option.NPV())

        # correction so discounting ends at discount_end, not carry_end.
        # If carry_end > discount_end (cash case with U>0), undo the extra discount.
        tau_discEnd_to_carryEnd = self.day_counter.yearFraction(self.discount_end, self.carry_end)
        tau_discStart_to_carryStart = self.day_counter.yearFraction(self.discount_start, self.carry_start)
        corr_physical = np.exp((self.risk_free_rate - self.dividend_yield) * tau_discEnd_to_carryEnd)
        corr_cash = np.exp(-(self.risk_free_rate) * tau_discStart_to_carryStart)
        if self.settlement_type == "physical":
            adjusted_pv = pv_engine_at_optSpot *corr_physical
        else:
            adjusted_pv = pv_engine_at_optSpot * corr_cash
        return adjusted_pv

    def price(self, time_steps: int, scheme=ql.FdmSchemeDesc.CrankNicolson()):
        """
        Price the option using finite difference method with:
          • optimal Δt-Δx relation (μ ≈ μ*)
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
        Price the option over a range of time steps (each with μ ≈ μ*; Richardson on).
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
        """Minimal internal cloner to reuse your constructor exactly as-is."""
        return VanillaOptionPricerTest(
            spot_price=self.spot_price if spot_price is None else spot_price,
            strike_price=self.strike_price,
            risk_free_rate=self.risk_free_rate,
            volatility=self.volatility if volatility is None else volatility,
            dividend_yield=self.dividend_yield,
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
        engine = self._engine(N)
        self.option.setPricingEngine(engine)
        p0 = self._price_once(N)

        # Delta & Gamma (spot bumps)
        bump =1e-4 * 100
        dS = bump * self.spot_price
        p_up = self._clone(spot_price=self.spot_price + dS)._price_once(N)
        p_dn = self._clone(spot_price=self.spot_price - dS)._price_once(N)
        delta = self.option.delta() if hasattr(self.option, "delta") else (p_up - p_dn) / (2.0 * dS)
        gamma = self.option.gamma() if hasattr(self.option, "gamma") else (p_up - 2.0 * p0 + p_dn) / (dS ** 2)

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
            f"Domestic Risk-Free Rate  : {self.risk_free_rate}",
            f"Foreign Risk-Free Rate   : {self.dividend_yield}",
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
