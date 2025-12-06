import datetime as _dt
from typing import List, Optional, Tuple, Literal, Dict
import math

import numpy as np
import pandas as pd  # type: ignore
from workalendar.africa import SouthAfrica

OptionType = Literal["call", "put"]


class AmericanFDMPricer:
    """
    American vanilla option pricer using a Crank–Nicolson finite difference scheme
    in log-space (with Rannacher smoothing).

    - Dates: datetime.date for valuation/maturity and dividend dates.
    - Curves: discount_curve is a DataFrame with columns ["Date", "NACA"] where
      Date is isoformat "YYYY-MM-DD" and NACA is an annual nominal rate.
    - DF convention: ACT/365F-style simple year fraction, DF = (1 + NACA)^(-tau).
    - Dividends: discrete dividend_schedule is converted into an equivalent flat
      dividend yield q, by matching PV(divs); that q feeds the PDE drift term.

    This class focuses on:
    - American early exercise via projection V = max(V, intrinsic) at each time step.
    - Robust Greeks from the PDE grid: price, delta, gamma, vega.
    """

    def __init__(
        self,
        spot: float,
        strike: float,
        valuation_date: _dt.date,
        maturity_date: _dt.date,
        sigma: float,
        option_type: OptionType,
        discount_curve: pd.DataFrame,
        dividend_schedule: Optional[List[Tuple[_dt.date, float]]] = None,
        # Spot / settlement conventions
        underlying_spot_days: int = 3,
        option_spot_days: int = 0,
        option_settlement_days: int = 0,
        # Grid / numerical controls
        num_space_nodes: int = 400,
        num_time_steps: int = 400,
        rannacher_steps: int = 2,
        s_max_mult: float = 4.5,
        day_count: str = "ACT/365",
    ) -> None:
        # Basic validation
        if any(x <= 0.0 for x in (spot, strike, sigma)):
            raise ValueError("spot, strike, sigma must be positive.")
        if maturity_date <= valuation_date:
            raise ValueError("maturity_date must be after valuation_date.")

        # Core inputs
        self.spot = float(spot)
        self.strike = float(strike)
        self.valuation_date = valuation_date
        self.maturity_date = maturity_date
        self.sigma = float(sigma)
        self.option_type = option_type.lower()
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")

        # Curves / dividends
        self.discount_curve_df = discount_curve.copy()
        self.dividend_schedule = sorted(dividend_schedule or [], key=lambda x: x[0])

        # Calendar and day count
        self.calendar = SouthAfrica()
        self.day_count = day_count.upper().replace("F", "")
        self._year_denominator = self._infer_denominator(self.day_count)

        # Spot / settlement conventions
        self.underlying_spot_days = int(underlying_spot_days)
        self.option_spot_days = int(option_spot_days)
        self.option_settlement_days = int(option_settlement_days)

        # Derived dates
        self.carry_start_date = self.calendar.add_working_days(
            self.valuation_date, self.underlying_spot_days
        )
        self.carry_end_date = self.calendar.add_working_days(
            self.maturity_date, self.underlying_spot_days
        )

        self.discount_start_date = self.calendar.add_working_days(
            self.valuation_date, self.option_spot_days
        )
        self.discount_end_date = self.calendar.add_working_days(
            self.maturity_date, self.option_settlement_days
        )

        # Year fractions
        self.time_to_expiry = self._year_fraction(
            self.valuation_date, self.maturity_date
        )
        self.time_to_carry = self._year_fraction(
            self.carry_start_date, self.carry_end_date
        )
        self.time_to_discount = self._year_fraction(
            self.discount_start_date, self.discount_end_date
        )

        if self.time_to_expiry <= 0.0:
            raise ValueError("time_to_expiry must be positive.")

        # Flat rates (continuously compounded equivalents)
        self.discount_rate_nacc = self.get_forward_nacc_rate(
            self.discount_start_date, self.discount_end_date
        )
        self.carry_rate_nacc = self.get_forward_nacc_rate(
            self.carry_start_date, self.carry_end_date
        )

        # Equivalent flat dividend yield
        self.div_yield_nacc = self.dividend_yield_nacc()
        self.pv_divs = self.pv_dividends()

        # Forward price (used mostly for sanity / diagnostics)
        if self.time_to_carry > 0.0:
            self.forward_price = self.spot * math.exp(
                (self.carry_rate_nacc - self.div_yield_nacc) * self.time_to_carry
            )
        else:
            self.forward_price = self.spot

        # Grid parameters
        self.num_space_nodes = int(num_space_nodes)
        self.num_time_steps = int(num_time_steps)
        self.rannacher_steps = int(rannacher_steps)
        self.s_max_mult = float(s_max_mult)

        # These will be populated by grid builder
        self._S_min: float = 0.0
        self._S_max: float = 0.0
        self.s_nodes: List[float] = []
        self.x_nodes: List[float] = []

    # ---------------------------------------------------------------------
    #  Basic date / curve utilities
    # ---------------------------------------------------------------------
    def _infer_denominator(self, day_count: str) -> int:
        """Map day count to denominator used for simple year fractions."""
        if day_count in ("ACT/365", "ACT/365F"):
            return 365
        if day_count in ("ACT/360", "ACT/364"):
            return 360 if day_count == "ACT/360" else 364
        if day_count in ("30/360", "BOND", "US30/360"):
            return 360
        return 365

    def _year_fraction(self, start_date: _dt.date, end_date: _dt.date) -> float:
        """Compute year fraction according to configured day count (simple versions)."""
        if end_date <= start_date:
            return 0.0
        if self.day_count in ("ACT/365", "ACT/365F", "ACT/360", "ACT/364"):
            return (end_date - start_date).days / float(self._year_denominator)
        if self.day_count in ("30/360", "BOND", "US30/360"):
            y1, m1, d1 = start_date.year, start_date.month, start_date.day
            y2, m2, d2 = end_date.year, end_date.month, end_date.day
            d1 = min(d1, 30)
            if d1 == 30:
                d2 = min(d2, 30)
            days = (y2 - y1) * 360 + (m2 - m1) * 30 + (d2 - d1)
            return days / 360.0
        # fallback
        return (end_date - start_date).days / 365.0

    def get_discount_factor(self, lookup_date: _dt.date) -> float:
        """Discount factor DF(valuation_date, lookup_date) from the NACA curve."""
        if self.discount_curve_df is None:
            raise ValueError("No discount curve attached.")
        iso = lookup_date.isoformat()
        row = self.discount_curve_df[self.discount_curve_df["Date"] == iso]
        if row.empty:
            raise ValueError(f"Discount factor not found for date: {iso}")
        naca = float(row["NACA"].values[0])
        tau = self._year_fraction(self.valuation_date, lookup_date)
        return (1.0 + naca) ** (-tau)

    def get_forward_nacc_rate(
        self, start_date: _dt.date, end_date: _dt.date
    ) -> float:
        """Flat continuously-compounded forward rate between two dates from NACA curve."""
        df_far = self.get_discount_factor(end_date)
        df_near = self.get_discount_factor(start_date)
        tau = self._year_fraction(start_date, end_date)
        return -math.log(df_far / df_near) / max(1e-12, tau)

    def pv_dividends(self) -> float:
        """PV of discrete dividend_schedule at valuation_date using discount curve."""
        if not self.dividend_schedule:
            return 0.0

        pv = 0.0
        for pay_date, amount in self.dividend_schedule:
            if self.valuation_date < pay_date <= self.maturity_date:
                df = self.get_discount_factor(pay_date)
                pv += float(amount) * df
        return pv

    def dividend_yield_nacc(self) -> float:
        """
        Back-out a flat continuous dividend yield q reproducing PV(dividend_schedule)
        over [valuation, maturity], i.e. S0_eff = S0 * e^{-q T} where S0_eff = S0 - PV(divs).
        """
        pv_divs = self.pv_dividends()
        S = self.spot
        tau = max(1e-12, self.time_to_carry)

        if pv_divs <= 0.0:
            return 0.0

        if pv_divs >= S:
            raise ValueError("PV(dividend_schedule) >= spot.")
        return -math.log((S - pv_divs) / S) / tau

    # ---------------------------------------------------------------------
    #  Payoff, boundaries, grid setup
    # ---------------------------------------------------------------------
    def _intrinsic_payoff(self, S: float) -> float:
        """Intrinsic payoff for the American constraint."""
        if self.option_type == "call":
            return max(S - self.strike, 0.0)
        return max(self.strike - S, 0.0)

    def _terminal_payoff(self) -> List[float]:
        """Terminal payoff at maturity on the S-grid."""
        return [self._intrinsic_payoff(s) for s in self.s_nodes]

    def _boundary_values(self, tau: float) -> Tuple[float, float]:
        """
        Dirichlet boundaries at time-to-maturity tau (European continuation).
        For American, the early-exercise projection will enforce V >= intrinsic
        at the boundaries as well.
        """
        S_min = self.s_nodes[0]
        S_max = self.s_nodes[-1]

        r = self.discount_rate_nacc
        b = self.carry_rate_nacc
        k = self.strike
        is_call = self.option_type == "call"

        if is_call:
            # Deep OTM call near S=0; upper behaves like discounted forward minus PV(K)
            V_min = 0.0
            V_max = S_max * math.exp((b - r) * tau) - k * math.exp(-r * tau)
        else:
            # Deep ITM put near S≈0 ≈ PV(K); deep OTM at large S
            V_min = k * math.exp(-r * tau)
            V_max = 0.0

        return V_min, V_max

    def _configure_grid(self) -> None:
        """
        Choose [S_min, S_max] and N_space in a simple, robust way for vanilla options.
        Domain is centred in log-space around spot and strike and widened by s_max_mult
        volatility bands.
        """
        T = self.time_to_expiry
        sig = self.sigma

        # Center around the geometric mean of spot and strike
        s_low = min(self.spot, self.strike)
        s_high = max(self.spot, self.strike)
        s_c = math.sqrt(s_low * s_high)

        # Rough volatility band in log-space
        band = self.s_max_mult * sig * math.sqrt(max(T, 1e-12))
        x_c = math.log(s_c)
        x_min = x_c - 0.5 * band
        x_max = x_c + 0.5 * band

        S_min = math.exp(x_min)
        S_max = math.exp(x_max)

        # Ensure domain fully contains [0.5 * s_low, 2 * s_high]
        S_min = min(S_min, 0.5 * s_low)
        S_max = max(S_max, 2.0 * s_high)

        self._S_min = max(S_min, 1e-8)
        self._S_max = S_max

        # Keep num_space_nodes as provided; you can adjust externally
        self.num_space_nodes = int(self.num_space_nodes)

    def _build_log_grid(self) -> float:
        """
        Build log-S grid and store S-grid in self.s_nodes.
        Returns the log-space grid step dx.
        """
        self._configure_grid()
        S_min = self._S_min
        S_max = self._S_max

        x_min = math.log(S_min)
        x_max = math.log(S_max)

        n = self.num_space_nodes
        dx = (x_max - x_min) / n
        self.x_nodes = [x_min + i * dx for i in range(n + 1)]
        self.s_nodes = [math.exp(x) for x in self.x_nodes]

        return dx

    # ---------------------------------------------------------------------
    #  Core CN + Rannacher solver with American projection
    # ---------------------------------------------------------------------
    def _solve_grid(self, N_time: Optional[int] = None) -> List[float]:
        """
        Solve the Black–Scholes PDE in log-space using a theta-scheme
        (Rannacher: backward Euler for first steps, then Crank–Nicolson),
        with an American early-exercise projection V = max(V, intrinsic).
        """
        dx = self._build_log_grid()

        N = self.num_space_nodes - 1
        N_time = int(N_time) if N_time is not None else int(self.num_time_steps)
        assert N_time >= 1
        dt = self.time_to_expiry / float(N_time)

        # Rates and drift
        sig = self.sigma
        sig2 = sig * sig

        r = self.discount_rate_nacc
        b = self.carry_rate_nacc
        q = self.div_yield_nacc

        mu_x = (b - q) - 0.5 * sig2

        # Operator coefficients in log space
        alpha = 0.5 * sig2 / (dx * dx)
        beta_adv = mu_x / (2.0 * dx)

        a = alpha - beta_adv          # coeff for V_{j-1}
        c = alpha + beta_adv          # coeff for V_{j+1}
        bcoef = -2.0 * alpha - r      # coeff for V_j

        # Matrices for generic theta (we'll use theta=1 for BE, 0.5 for CN)
        def build_matrices(theta: float) -> Tuple[float, float, float, float, float, float]:
            A_L = -theta * dt * a
            A_C = 1.0 - theta * dt * bcoef
            A_U = -theta * dt * c

            B_L = (1.0 - theta) * dt * a
            B_C = 1.0 + (1.0 - theta) * dt * bcoef
            B_U = (1.0 - theta) * dt * c
            return A_L, A_C, A_U, B_L, B_C, B_U

        # Thomas solver for constant tridiagonal system
        def solve_tridiag(A_L, A_C, A_U, rhs):
            n = len(rhs)
            c_prime = [0.0] * n
            d_prime = [0.0] * n

            # first row
            denom = A_C
            c_prime[0] = A_U / denom
            d_prime[0] = rhs[0] / denom

            # forward sweep
            for i in range(1, n):
                denom = A_C - A_L * c_prime[i - 1]
                if i < n - 1:
                    c_prime[i] = A_U / denom
                d_prime[i] = (rhs[i] - A_L * d_prime[i - 1]) / denom

            # back substitution
            x = [0.0] * n
            x[-1] = d_prime[-1]
            for i in range(n - 2, -1, -1):
                x[i] = d_prime[i] - c_prime[i] * x[i + 1]
            return x

        # Initial condition at tau = 0 (maturity)
        V = self._terminal_payoff()  # length N+1: j=0..N

        theta = 1.0
        rannacher_left = self.rannacher_steps

        for m in range(N_time):
            # Switch from BE (theta=1) to CN (theta=0.5)
            if rannacher_left > 0:
                theta = 1.0
                rannacher_left -= 1
            else:
                theta = 0.5

            A_L, A_C, A_U, B_L, B_C, B_U = build_matrices(theta)
            tau_next = (m + 1) * dt

            # Boundary values at tau_next (Dirichlet)
            V_min_next, V_max_next = self._boundary_values(tau_next)

            # RHS for interior points j=1..N-1
            rhs = [0.0] * (N - 1)
            for j in range(1, N):
                Vjm1, Vj, Vjp1 = V[j - 1], V[j], V[j + 1]
                rhs[j - 1] = B_L * Vjm1 + B_C * Vj + B_U * Vjp1

            # Impose boundaries
            rhs[0] -= A_L * V_min_next
            rhs[-1] -= A_U * V_max_next

            # Solve interior tridiagonal system
            V_int = solve_tridiag(A_L, A_C, A_U, rhs)

            # Assemble full grid at tau_next
            V[0] = V_min_next
            V[-1] = V_max_next
            V[1:-1] = V_int

            # American early-exercise projection
            for j, S in enumerate(self.s_nodes):
                intrinsic = self._intrinsic_payoff(S)
                if V[j] < intrinsic:
                    V[j] = intrinsic

        return V

    # ---------------------------------------------------------------------
    #  Price extraction and Greeks
    # ---------------------------------------------------------------------
    def _value_at_spot(self, V: List[float]) -> float:
        """
        Extract price at S = spot from the grid via linear interpolation
        in S-space.
        """
        s_nodes = self.s_nodes
        S0 = self.spot
        N = len(s_nodes) - 1

        # Find bracketing interval
        if S0 <= s_nodes[0]:
            return float(V[0])
        if S0 >= s_nodes[N]:
            return float(V[N])

        for i in range(N):
            sL, sR = s_nodes[i], s_nodes[i + 1]
            if sL <= S0 <= sR:
                VL, VR = V[i], V[i + 1]
                w = (S0 - sL) / (sR - sL)
                return float((1.0 - w) * VL + w * VR)

        # Fallback: nearest node
        idx = min(range(N + 1), key=lambda k: abs(s_nodes[k] - S0))
        return float(V[idx])

    def _delta_gamma_from_grid(self, V: List[float]) -> Tuple[float, float]:
        """
        Non-uniform central finite-difference formulas for delta and gamma
        on the S-grid, centered at the node closest to S0.
        """
        s_nodes = self.s_nodes
        S0 = self.spot
        N = len(s_nodes) - 1

        i = min(range(N + 1), key=lambda k: abs(s_nodes[k] - S0))
        # Ensure interior
        i = max(1, min(N - 1, i))

        h1 = s_nodes[i] - s_nodes[i - 1]
        h2 = s_nodes[i + 1] - s_nodes[i]

        delta = (
            - (h2 / (h1 * (h1 + h2))) * V[i - 1]
            + ((h2 - h1) / (h1 * h2)) * V[i]
            + (h1 / (h2 * (h1 + h2))) * V[i + 1]
        )

        gamma = 2.0 * (
            V[i - 1] / (h1 * (h1 + h2))
            - V[i] / (h1 * h2)
            + V[i + 1] / (h2 * (h1 + h2))
        )

        return float(delta), float(gamma)

    def price(self) -> float:
        """American option price per unit notional."""
        V = self._solve_grid()
        return self._value_at_spot(V)

    def greeks(self, dv_sigma: float = 0.0001) -> Dict[str, float]:
        """
        Compute price, delta, gamma, vega (w.r.t. sigma) via:
        - price, delta, gamma: PDE grid
        - vega: symmetric bump-and-revalue on sigma
        """
        # Base solve
        V_base = self._solve_grid()
        price_base = self._value_at_spot(V_base)
        delta, gamma = self._delta_gamma_from_grid(V_base)

        # Vega via sigma bump
        orig_sigma = self.sigma

        # Up bump
        self.sigma = orig_sigma + dv_sigma
        V_up = self._solve_grid()
        price_up = self._value_at_spot(V_up)

        # Down bump
        self.sigma = max(1e-8, orig_sigma - dv_sigma)
        V_dn = self._solve_grid()
        price_dn = self._value_at_spot(V_dn)

        # Restore
        self.sigma = orig_sigma

        vega = (price_up - price_dn) / (2.0 * dv_sigma)

        return {
            "price": float(price_base),
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
        }
