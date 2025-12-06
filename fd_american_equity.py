import datetime as _dt
from typing import List, Optional, Tuple, Literal, Dict, Any
import math

import pandas as pd  # type: ignore
from workalendar.africa import SouthAfrica

OptionType = Literal["call", "put"]


class AmericanFDMPricer:
    """
    American vanilla option pricer using a Crank–Nicolson finite-difference scheme
    in log-space (with Rannacher smoothing).

    Aligned with:
    - Date / curve conventions of DiscreteBarrierFDMPricer (NACA curve, ACT/365F-style),
    - Batch runner interface in run_american_scenarios:
      * price_log2()
      * greeks_log2()

    No barrier logic – just American early exercise enforced via projection:
      V_j <- max(V_j, intrinsic(S_j)) at every time step.
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
        forward_curve: Optional[pd.DataFrame] = None,
        dividend_schedule: Optional[List[Tuple[_dt.date, float]]] = None,
        trade_id: Optional[int] = None,
        direction: str = "long",
        quantity: int = 1,
        contract_multiplier: float = 1.0,
        underlying_spot_days: int = 0,
        option_days: int = 0,
        option_settlement_days: int = 0,
        day_count: str = "ACT/365",
        grid_type: str = "uniform",
        num_space_nodes: int = 400,
        num_time_steps: int = 400,
        rannacher_steps: int = 2,
        s_max_mult: float = 4.5,
    ) -> None:
        # Basic validation
        if spot <= 0.0 or strike <= 0.0 or sigma <= 0.0:
            raise ValueError("spot, strike and sigma must be positive.")
        if maturity_date <= valuation_date:
            raise ValueError("maturity_date must be after valuation_date.")

        # Core economic inputs
        self.spot = float(spot)
        self.strike = float(strike)
        self.valuation_date = valuation_date
        self.maturity_date = maturity_date
        self.sigma = float(sigma)
        self.option_type = option_type.lower()
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")

        # Curves and dividends
        self.discount_curve_df = discount_curve.copy()
        self.forward_curve_df = forward_curve.copy() if forward_curve is not None else None
        self.dividend_schedule = sorted(dividend_schedule or [], key=lambda x: x[0])

        # Trade/meta info (not used in PDE math but kept for alignment)
        self.trade_id = trade_id
        self.direction = direction
        self.quantity = int(quantity)
        self.contract_multiplier = float(contract_multiplier)

        # Day count / calendar and spot-settlement conventions
        self.calendar = SouthAfrica()
        self.underlying_spot_days = int(underlying_spot_days)
        self.option_days = int(option_days)
        self.option_settlement_days = int(option_settlement_days)

        self.day_count = day_count.upper().replace("F", "")
        self._year_denominator = self._infer_denominator(self.day_count)

        # Carry (underlying) leg dates
        self.carry_start_date = self.calendar.add_working_days(
            self.valuation_date, self.underlying_spot_days
        )
        self.carry_end_date = self.calendar.add_working_days(
            self.maturity_date, self.underlying_spot_days
        )

        # Option discount leg dates
        self.discount_start_date = self.calendar.add_working_days(
            self.valuation_date, self.option_days
        )
        self.discount_end_date = self.calendar.add_working_days(
            self.maturity_date, self.option_settlement_days
        )

        # Year fractions
        self.time_to_expiry = self._year_fraction(self.valuation_date, self.maturity_date)
        self.time_to_carry = self._year_fraction(self.carry_start_date, self.carry_end_date)
        self.time_to_discount = self._year_fraction(self.discount_start_date, self.discount_end_date)

        if self.time_to_expiry <= 0.0:
            raise ValueError("time_to_expiry must be positive.")

        # Flat NACC rates from NACA curve
        self.discount_rate_nacc = self.get_forward_nacc_rate(
            self.discount_start_date, self.discount_end_date
        )
        self.carry_rate_nacc = self.get_forward_nacc_rate(
            self.carry_start_date, self.carry_end_date
        )

        # Dividend handling (same style as barrier engine)
        self.div_yield_nacc = self.dividend_yield_nacc()
        self.pv_divs = self.pv_dividends()

        # Forward price and effective carry (diagnostics)
        if self.time_to_carry > 0.0:
            self.forward_price = self.spot * math.exp(
                (self.carry_rate_nacc - self.div_yield_nacc) * self.time_to_carry
            )
        else:
            self.forward_price = self.spot
        self.b = math.log(self.forward_price / self.spot) / max(self.time_to_carry, 1e-12)

        # Grid / CN controls
        self.grid_type = grid_type
        self.s_max_mult = float(s_max_mult)
        self.num_space_nodes = int(num_space_nodes)
        self.num_time_steps = int(num_time_steps)
        self.rannacher_steps = int(rannacher_steps)

        # Grid containers
        self._S_min: float = 0.0
        self._S_max: float = 0.0
        self.s_nodes: List[float] = []
        self.x_nodes: List[float] = []

    # ------------------------------------------------------------------
    #  Date / curve helpers
    # ------------------------------------------------------------------
    def _infer_denominator(self, day_count: str) -> int:
        if day_count in ("ACT/365", "ACT/365F"):
            return 365
        if day_count in ("ACT/360", "ACT/364"):
            return 360 if day_count == "ACT/360" else 364
        if day_count in ("30/360", "BOND", "US30/360"):
            return 360
        return 365

    def _year_fraction(self, start_date: _dt.date, end_date: _dt.date) -> float:
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
        return (end_date - start_date).days / 365.0

    def get_discount_factor(self, lookup_date: _dt.date) -> float:
        """
        DF(valuation_date, lookup_date) from NACA discount_curve_df.
        """
        iso = lookup_date.isoformat()
        row = self.discount_curve_df[self.discount_curve_df["Date"] == iso]
        if row.empty:
            raise ValueError(f"Discount factor not found for date: {iso}")
        naca = float(row["NACA"].values[0])
        tau = self._year_fraction(self.valuation_date, lookup_date)
        return (1.0 + naca) ** (-tau)

    def get_forward_nacc_rate(self, start_date: _dt.date, end_date: _dt.date) -> float:
        df_far = self.get_discount_factor(end_date)
        df_near = self.get_discount_factor(start_date)
        tau = self._year_fraction(start_date, end_date)
        return -math.log(df_far / df_near) / max(1e-12, tau)

    def pv_dividends(self) -> float:
        """
        PV of discrete dividends between valuation and maturity,
        discounted using the discount curve, scaled relative to carry_start_date.
        """
        if not self.dividend_schedule:
            return 0.0

        pv = 0.0
        for pay_date, amount in self.dividend_schedule:
            if self.valuation_date < pay_date <= self.maturity_date:
                df = self.get_discount_factor(pay_date) / self.get_discount_factor(self.carry_start_date)
                pv += float(amount) * df
        return pv

    def dividend_yield_nacc(self) -> float:
        """
        Flat continuous dividend yield q such that PV(divs) matches the
        escrowed-dividend interpretation on [valuation, maturity].
        """
        pv_divs = self.pv_dividends()
        S = self.spot
        tau = max(1e-12, self.time_to_carry)

        if pv_divs <= 0.0:
            return 0.0
        if pv_divs >= S:
            raise ValueError("PV(dividend_schedule) >= spot.")
        return -math.log((S - pv_divs) / S) / tau

    # ------------------------------------------------------------------
    #  Payoff, boundaries, grid
    # ------------------------------------------------------------------
    def _intrinsic_payoff(self, S: float) -> float:
        if self.option_type == "call":
            return max(S - self.strike, 0.0)
        return max(self.strike - S, 0.0)

    def _terminal_payoff(self) -> List[float]:
        return [self._intrinsic_payoff(s) for s in self.s_nodes]

    def _boundary_values(self, tau: float) -> Tuple[float, float]:
        """
        Dirichlet boundaries at time-to-maturity tau (European continuation).
        Early exercise is enforced separately via V = max(V, intrinsic).
        """
        S_min = self.s_nodes[0]
        S_max = self.s_nodes[-1]

        r = self.discount_rate_nacc
        b = self.carry_rate_nacc
        k = self.strike
        is_call = self.option_type == "call"

        if is_call:
            V_min = 0.0
            V_max = S_max * math.exp((b - r) * tau) - k * math.exp(-r * tau)
        else:
            V_min = k * math.exp(-r * tau)
            V_max = 0.0

        return V_min, V_max

    def _configure_grid(self) -> None:
        """
        Simple, robust log-space domain for vanilla options.
        Center roughly around the geometric mean of S0 and K and widen
        with volatility bands scaled by s_max_mult.
        """
        T = self.time_to_expiry
        sig = self.sigma
        S0_eff = self.spot - self.pv_divs
        if S0_eff <= 0.0:
            S0_eff = self.spot

        s_low = min(S0_eff, self.strike)
        s_high = max(S0_eff, self.strike)

        s_c = math.sqrt(max(s_low * s_high, 1e-12))

        band = self.s_max_mult * sig * math.sqrt(max(T, 1e-12))
        x_c = math.log(s_c)
        x_min = x_c - 0.5 * band
        x_max = x_c + 0.5 * band

        S_min = math.exp(x_min)
        S_max = math.exp(x_max)

        # Ensure reasonable coverage
        S_min = min(S_min, 0.5 * s_low)
        S_max = max(S_max, 2.0 * s_high)

        self._S_min = max(S_min, 1e-8)
        self._S_max = S_max
        # num_space_nodes / num_time_steps are taken as given

    def _build_log_grid(self) -> float:
        """
        Build log-S grid and store in self.s_nodes; returns dx.
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

    # ------------------------------------------------------------------
    #  CN + Rannacher solver with American projection
    # ------------------------------------------------------------------
    def _solve_grid(self, N_time: Optional[int] = None) -> List[float]:
        """
        Solve the Black–Scholes PDE in log S with a theta-scheme:
        - first rannacher_steps: backward Euler (theta=1),
        - rest: Crank–Nicolson (theta=0.5),
        and at each time step enforce American early exercise
        via V_j = max(V_j, intrinsic(S_j)).
        """
        dx = self._build_log_grid()

        N = self.num_space_nodes - 1
        N_time = int(N_time) if N_time is not None else int(self.num_time_steps)
        assert N_time >= 1
        dt = self.time_to_expiry / float(N_time)

        sig = self.sigma
        sig2 = sig * sig

        r = self.discount_rate_nacc
        b = self.carry_rate_nacc
        q = self.div_yield_nacc

        mu_x = (b - q) - 0.5 * sig2

        alpha = 0.5 * sig2 / (dx * dx)
        beta_adv = mu_x / (2.0 * dx)

        a = alpha - beta_adv           # coeff for V_{j-1}
        c = alpha + beta_adv           # coeff for V_{j+1}
        bcoef = -2.0 * alpha - r       # coeff for V_j

        def build_matrices(theta: float) -> Tuple[float, float, float, float, float, float]:
            A_L = -theta * dt * a
            A_C = 1.0 - theta * dt * bcoef
            A_U = -theta * dt * c

            B_L = (1.0 - theta) * dt * a
            B_C = 1.0 + (1.0 - theta) * dt * bcoef
            B_U = (1.0 - theta) * dt * c
            return A_L, A_C, A_U, B_L, B_C, B_U

        def solve_tridiag(A_L, A_C, A_U, rhs):
            n = len(rhs)
            c_prime = [0.0] * n
            d_prime = [0.0] * n

            denom = A_C
            c_prime[0] = A_U / denom
            d_prime[0] = rhs[0] / denom

            for i in range(1, n):
                denom = A_C - A_L * c_prime[i - 1]
                if i < n - 1:
                    c_prime[i] = A_U / denom
                d_prime[i] = (rhs[i] - A_L * d_prime[i - 1]) / denom

            x = [0.0] * n
            x[-1] = d_prime[-1]
            for i in range(n - 2, -1, -1):
                x[i] = d_prime[i] - c_prime[i] * x[i + 1]
            return x

        V = self._terminal_payoff()
        theta = 1.0
        rannacher_left = self.rannacher_steps

        for m in range(N_time):
            if rannacher_left > 0:
                theta = 1.0
                rannacher_left -= 1
            else:
                theta = 0.5

            A_L, A_C, A_U, B_L, B_C, B_U = build_matrices(theta)
            tau_next = (m + 1) * dt

            V_min_next, V_max_next = self._boundary_values(tau_next)

            rhs = [0.0] * (N - 1)
            for j in range(1, N):
                Vjm1, Vj, Vjp1 = V[j - 1], V[j], V[j + 1]
                rhs[j - 1] = B_L * Vjm1 + B_C * Vj + B_U * Vjp1

            rhs[0] -= A_L * V_min_next
            rhs[-1] -= A_U * V_max_next

            V_int = solve_tridiag(A_L, A_C, A_U, rhs)

            V[0] = V_min_next
            V[-1] = V_max_next
            V[1:-1] = V_int

            # American early exercise: project onto intrinsic
            for j, S in enumerate(self.s_nodes):
                intrinsic = self._intrinsic_payoff(S)
                if V[j] < intrinsic:
                    V[j] = intrinsic

        return V

    # ------------------------------------------------------------------
    #  Interpolation and Greeks
    # ------------------------------------------------------------------
    def _interp_price(self, V: List[float]) -> float:
        """
        Price at S0 (escrowed-dividend effective spot) by linear interpolation
        in S-space on the computed grid.
        """
        s_nodes = self.s_nodes
        S0_eff = self.spot - self.pv_divs
        if S0_eff <= 0.0:
            S0_eff = self.spot

        if S0_eff <= s_nodes[0]:
            return float(V[0])
        if S0_eff >= s_nodes[-1]:
            return float(V[-1])

        lo, hi = 0, len(s_nodes) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if S0_eff < s_nodes[mid]:
                hi = mid
            else:
                lo = mid

        S_lo, S_hi = s_nodes[lo], s_nodes[hi]
        V_lo, V_hi = V[lo], V[hi]
        w = (S0_eff - S_lo) / (S_hi - S_lo)
        return float((1.0 - w) * V_lo + w * V_hi)

    def _delta_gamma_from_grid(self, V: List[float]) -> Tuple[float, float]:
        """
        Non-uniform central finite-difference formulas for delta and gamma
        on the S-grid, centered at the node closest to S0_eff.
        """
        s = self.s_nodes
        S0_eff = self.spot - self.pv_divs
        if S0_eff <= 0.0:
            S0_eff = self.spot

        N = len(s) - 1
        if N < 2:
            raise RuntimeError("Need at least 3 spatial nodes for Greeks.")

        idx = min(range(N + 1), key=lambda k: abs(s[k] - S0_eff))
        idx = max(1, min(N - 1, idx))

        S_im1, S_i, S_ip1 = s[idx - 1], s[idx], s[idx + 1]
        V_im1, V_i, V_ip1 = V[idx - 1], V[idx], V[idx + 1]

        h1 = S_i - S_im1
        h2 = S_ip1 - S_i

        # First derivative (delta)
        delta = (
            - (h2 / (h1 * (h1 + h2))) * V_im1
            + ((h2 - h1) / (h1 * h2)) * V_i
            + (h1 / (h2 * (h1 + h2))) * V_ip1
        )

        # Second derivative (gamma)
        gamma = 2.0 * (
            V_im1 / (h1 * (h1 + h2))
            - V_i / (h1 * h2)
            + V_ip1 / (h2 * (h1 + h2))
        )

        # Optional clamp to avoid numerical explosions
        gamma = max(min(gamma, 1e5), -1e5)

        return float(delta), float(gamma)

    # ------------------------------------------------------------------
    #  Public API: price_log2 / greeks_log2
    # ------------------------------------------------------------------
    def price_log2(self, use_richardson: bool = False) -> float:
        """
        American option price (per unit notional) from CN + Rannacher engine.
        The use_richardson flag is present for interface alignment but ignored.
        """
        V = self._solve_grid()
        return self._interp_price(V)

    def greeks_log2(self, dv_sigma: float = 0.0001, use_richardson: bool = False) -> Dict[str, float]:
        """
        American option price and Greeks (delta, gamma, vega, theta):
        - price, delta, gamma from base PDE solution;
        - vega from symmetric sigma bump (per 1% vol);
        - theta from PDE identity (approximate).
        The use_richardson flag is included for interface alignment and ignored.
        """
        # Base solution
        V_base = self._solve_grid()
        price_base = self._interp_price(V_base)
        delta, gamma = self._delta_gamma_from_grid(V_base)

        # Vega via sigma bump (per 1% volatility)
        orig_sigma = self.sigma

        self.sigma = orig_sigma + dv_sigma
        V_up = self._solve_grid()
        price_up = self._interp_price(V_up)

        self.sigma = max(1e-8, orig_sigma - dv_sigma)
        V_dn = self._solve_grid()
        price_dn = self._interp_price(V_dn)

        self.sigma = orig_sigma

        vega = (price_up - price_dn) / (2.0 * dv_sigma * 100.0)

        # Theta using the BS PDE identity at S0_eff
        theta = -(
            0.5 * orig_sigma * orig_sigma * (self.spot ** 2) * gamma
            + (self.carry_rate_nacc - self.div_yield_nacc) * self.spot * delta
            - self.discount_rate_nacc * price_base
        )

        return {
            "price": float(price_base),
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "theta": float(theta),
        }
