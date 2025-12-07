import datetime as _dt
from typing import List, Optional, Tuple, Literal, Dict
import math

import pandas as pd  # type: ignore
from workalendar.africa import SouthAfrica

OptionType = Literal["call", "put"]


class AmericanFDMPricer:
    """
    American vanilla option pricer using a Crank–Nicolson finite-difference
    scheme in log S with Rannacher smoothing, Ikonen–Toivanen compensation,
    and local cubic interpolation for price and Greeks.

    Designed to mirror the FIS finite-difference solver layout for American
    options (Black–Scholes PDE in log S, IT splitting, 3rd-degree polynomial
    interpolation, Richardson extrapolation).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
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
        if spot <= 0.0 or strike <= 0.0 or sigma <= 0.0:
            raise ValueError("spot, strike and sigma must be positive.")
        if maturity_date <= valuation_date:
            raise ValueError("maturity_date must be after valuation_date.")

        # Core economics
        self.spot = float(spot)
        self.strike = float(strike)
        self.valuation_date = valuation_date
        self.maturity_date = maturity_date
        self.sigma = float(sigma)

        self.option_type: str = option_type.lower()
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")

        # Curves / dividends
        self.discount_curve_df = discount_curve.copy()
        self.forward_curve_df = forward_curve.copy() if forward_curve is not None else None
        self.dividend_schedule = sorted(dividend_schedule or [], key=lambda x: x[0])

        # Trade/meta
        self.trade_id = trade_id
        self.direction = direction
        self.quantity = int(quantity)
        self.contract_multiplier = float(contract_multiplier)

        # Calendar / conventions
        self.calendar = SouthAfrica()
        self.underlying_spot_days = int(underlying_spot_days)
        self.option_days = int(option_days)
        self.option_settlement_days = int(option_settlement_days)

        self.day_count = day_count.upper().replace("F", "")
        self._year_denominator = self._infer_denominator(self.day_count)

        # Carry / discount dates
        self.carry_start_date = self.calendar.add_working_days(
            self.valuation_date, self.underlying_spot_days
        )
        self.carry_end_date = self.calendar.add_working_days(
            self.maturity_date, self.underlying_spot_days
        )

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

        # NACC rates from curves
        self.discount_rate_nacc = self.get_forward_nacc_rate(
            self.discount_start_date, self.discount_end_date
        )
        self.carry_rate_nacc = self.get_forward_nacc_rate(
            self.carry_start_date, self.carry_end_date
        )

        # Dividends -> PV and equivalent continuous yield
        self.pv_divs = self.pv_dividends()
        self.div_yield_nacc = self.dividend_yield_nacc()

        # Forward (diagnostic only)
        if self.time_to_carry > 0.0:
            self.forward_price = self.spot * math.exp(
                (self.carry_rate_nacc - self.div_yield_nacc) * self.time_to_carry
            )
        else:
            self.forward_price = self.spot

        # Grid controls
        self.grid_type = grid_type
        self.s_max_mult = float(s_max_mult)
        self.num_space_nodes = max(int(num_space_nodes), 3)
        self.num_time_steps = max(int(num_time_steps), 3)
        self.rannacher_steps = max(int(rannacher_steps), 0)

        # Grid state
        self._S_min: float = 0.0
        self._S_max: float = 0.0
        self.s_nodes: List[float] = []
        self.x_nodes: List[float] = []

    # ------------------------------------------------------------------
    # Day-count / curve helpers
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

    # ------------------------------------------------------------------
    # Dividends: PV and equivalent continuous yield
    # ------------------------------------------------------------------
    def pv_dividends(self) -> float:
        """
        PV of discrete dividends between valuation and maturity, discounted
        using the curve and normalised by DF(carry_start).
        """
        if not self.dividend_schedule:
            return 0.0

        pv = 0.0
        for pay_date, amount in self.dividend_schedule:
            if self.valuation_date < pay_date <= self.maturity_date:
                df = self.get_discount_factor(pay_date) / self.get_discount_factor(
                    self.carry_start_date
                )
                pv += float(amount) * df
        return pv

    def dividend_yield_nacc(self) -> float:
        """
        Flat continuous dividend yield q such that
            S0 - PV(divs) = S0 * exp(-q * T_carry)
        (escrowed-dividend interpretation).
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
    # Grid construction
    # ------------------------------------------------------------------
    def _configure_grid(self) -> None:
        """
        Configure S_min and S_max for the log-S grid.

        Domain is chosen around an effective spot S_eff = S0 - PV(divs)
        and the strike, with width proportional to sigma * sqrt(T).
        """
        T = self.time_to_expiry
        sig = self.sigma

        S_eff = self.spot - self.pv_divs
        if S_eff <= 0.0:
            S_eff = self.spot

        s_low = min(S_eff, self.strike)
        s_high = max(S_eff, self.strike)
        s_c = math.sqrt(max(s_low * s_high, 1e-12))

        band = self.s_max_mult * sig * math.sqrt(max(T, 1e-12))
        x_c = math.log(s_c)
        x_min = x_c - 0.5 * band
        x_max = x_c + 0.5 * band

        S_min = math.exp(x_min)
        S_max = math.exp(x_max)

        S_min = min(S_min, 0.5 * s_low)
        S_max = max(S_max, 2.0 * s_high)

        self._S_min = max(S_min, 1e-8)
        self._S_max = S_max

    def _build_log_grid(self) -> float:
        """
        Build the log-S grid.
        """
        self._configure_grid()
        S_min = self._S_min
        S_max = self._S_max

        x_min = math.log(S_min)
        x_max = math.log(S_max)

        n = self.num_space_nodes
        dx = (x_max - x_min) / float(n)

        self.x_nodes = [x_min + i * dx for i in range(n + 1)]
        self.s_nodes = [math.exp(x) for x in self.x_nodes]

        return dx

    # ------------------------------------------------------------------
    # Payoff and boundaries
    # ------------------------------------------------------------------
    def _intrinsic_payoff(self, S: float) -> float:
        if self.option_type == "call":
            return max(S - self.strike, 0.0)
        return max(self.strike - S, 0.0)

    def _terminal_payoff(self) -> List[float]:
        return [self._intrinsic_payoff(s) for s in self.s_nodes]

    def _boundary_values(self, tau: float) -> Tuple[float, float]:
        """
        European-style boundary conditions at time-to-maturity tau.
        Early exercise is handled via Ikonen–Toivanen on the interior nodes.
        """
        S_min = self.s_nodes[0]
        S_max = self.s_nodes[-1]

        r = self.discount_rate_nacc
        b = self.carry_rate_nacc
        k = self.strike

        if self.option_type == "call":
            V_min = 0.0
            V_max = S_max * math.exp((b - r) * tau) - k * math.exp(-r * tau)
        else:
            V_min = k * math.exp(-r * tau)
            V_max = 0.0
        return V_min, V_max

    # ------------------------------------------------------------------
    # Core solver: CN + Rannacher + Ikonen–Toivanen
    # ------------------------------------------------------------------
    def _solve_grid(self, N_time: int, rannacher_steps: Optional[int] = None) -> List[float]:
        """
        Solve the BS PDE on the log-S grid using a theta-scheme with
        Rannacher smoothing and Ikonen–Toivanen splitting for the
        American constraint.
        """
        dx = self._build_log_grid()

        N = len(self.s_nodes) - 1
        if N < 2:
            raise RuntimeError("Spatial grid too coarse; need at least 3 nodes.")

        if N_time < 1:
            raise RuntimeError("Need at least 1 time step.")
        dt = self.time_to_expiry / float(N_time)

        sig = self.sigma
        sig2 = sig * sig

        r = self.discount_rate_nacc
        b = self.carry_rate_nacc
        q = self.div_yield_nacc

        # Drift in log S under risk-neutral measure with dividend yield q
        mu_x = (b - q) - 0.5 * sig2

        alpha = 0.5 * sig2 / (dx * dx)
        beta_adv = mu_x / (2.0 * dx)

        # Semi-discrete operator coefficients: dV/dt = a V_{j-1} + bcoef V_j + c V_{j+1}
        a = alpha - beta_adv
        c = alpha + beta_adv
        bcoef = -2.0 * alpha - r

        def build_matrices(theta: float):
            # (I - theta dt A) V^{n-1} = (I + (1-theta) dt A) V^n + dt lambda
            A_L = -theta * dt * a
            A_C = 1.0 - theta * dt * bcoef
            A_U = -theta * dt * c

            B_L = (1.0 - theta) * dt * a
            B_C = 1.0 + (1.0 - theta) * dt * bcoef
            B_U = (1.0 - theta) * dt * c
            return A_L, A_C, A_U, B_L, B_C, B_U

        def solve_tridiag(A_L: float, A_C: float, A_U: float, rhs: List[float]) -> List[float]:
            n = len(rhs)
            if n == 0:
                raise RuntimeError("Tridiagonal system has zero size.")
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

        # Terminal condition at maturity
        V = self._terminal_payoff()
        payoff_full = V.copy()

        # Multiplier lambda for American constraint (interior nodes only)
        lambda_int = [0.0] * (N - 1)
        payoff_int = payoff_full[1:-1]

        ran_steps = self.rannacher_steps if rannacher_steps is None else int(rannacher_steps)

        for m in range(N_time):
            theta = 1.0 if m < ran_steps else 0.5

            A_L, A_C, A_U, B_L, B_C, B_U = build_matrices(theta)
            tau_next = (m + 1) * dt

            V_min_next, V_max_next = self._boundary_values(tau_next)

            # Predictor: solve linear system for tilde V^{n-1}
            rhs = [0.0] * (N - 1)
            for j in range(1, N):
                Vjm1, Vj, Vjp1 = V[j - 1], V[j], V[j + 1]
                rhs[j - 1] = (
                    B_L * Vjm1 + B_C * Vj + B_U * Vjp1
                    + dt * lambda_int[j - 1]
                )

            # Boundary contributions
            rhs[0] -= A_L * V_min_next
            rhs[-1] -= A_U * V_max_next

            tilde_int = solve_tridiag(A_L, A_C, A_U, rhs)

            # Compensation: update V and lambda so that V >= payoff
            new_lambda_int = [0.0] * (N - 1)
            new_V_int = [0.0] * (N - 1)

            for k in range(N - 1):
                tilde_vk = tilde_int[k]
                phi_k = payoff_int[k]
                lam_old = lambda_int[k]

                # Candidate value: tilde V - dt * lambda^{n-1}
                v_candidate = tilde_vk - dt * lam_old

                # Enforce inequality
                v_new = phi_k if phi_k > v_candidate else v_candidate

                # New lambda as in Ikonen–Toivanen
                lam_new = lam_old + (phi_k - tilde_vk) / dt
                if lam_new < 0.0:
                    lam_new = 0.0

                new_lambda_int[k] = lam_new
                new_V_int[k] = v_new

            # Assemble full grid including boundaries
            V[0] = V_min_next
            V[-1] = V_max_next
            V[1:-1] = new_V_int

            lambda_int = new_lambda_int

        return V

    # ------------------------------------------------------------------
    # Cubic interpolation for price, delta, gamma
    # ------------------------------------------------------------------
    def _price_delta_gamma_cubic(self, V: List[float]) -> Tuple[float, float, float]:
        """
        Local 3rd-degree polynomial interpolation around the spot S0.

        We select 4 consecutive nodes surrounding S0, fit
            p(S) = a S^3 + b S^2 + c S + d
        and compute price, delta and gamma analytically as
            p(S0), p'(S0), p''(S0).
        """
        s = self.s_nodes
        S0 = self.spot
        n = len(s)
        if n < 4:
            # Fallback: linear interpolation + FD Greeks
            price = self._interp_price_linear(V)
            delta, gamma = self._delta_gamma_fd(V)
            return price, delta, gamma

        # Find interval containing S0
        lo, hi = 0, n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if S0 < s[mid]:
                hi = mid
            else:
                lo = mid

        # Choose 4 consecutive points around S0
        start = max(0, min(lo - 1, n - 4))
        xs = [s[start + i] for i in range(4)]
        ys = [V[start + i] for i in range(4)]

        # Solve for cubic coefficients via small Gaussian elimination
        M = [[xs[i] ** 3, xs[i] ** 2, xs[i], 1.0] for i in range(4)]
        A = [M[i] + [ys[i]] for i in range(4)]  # augmented matrix 4x5

        # Forward elimination
        for i in range(4):
            pivot = A[i][i]
            if abs(pivot) < 1e-18:
                pivot = 1e-18
            inv_p = 1.0 / pivot
            for j in range(i, 5):
                A[i][j] *= inv_p
            for k in range(i + 1, 4):
                factor = A[k][i]
                for j in range(i, 5):
                    A[k][j] -= factor * A[i][j]

        # Back substitution
        coeffs = [0.0] * 4
        for i in range(3, -1, -1):
            val = A[i][4]
            for j in range(i + 1, 4):
                val -= A[i][j] * coeffs[j]
            coeffs[i] = val

        a, b, c, d = coeffs

        price = ((a * S0 + b) * S0 + c) * S0 + d
        delta = (3.0 * a * S0 + 2.0 * b) * S0 + c
        gamma = 6.0 * a * S0 + 2.0 * b

        return float(price), float(delta), float(gamma)

    def _interp_price_linear(self, V: List[float]) -> float:
        s = self.s_nodes
        S0 = self.spot
        if S0 <= s[0]:
            return float(V[0])
        if S0 >= s[-1]:
            return float(V[-1])
        lo, hi = 0, len(s) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if S0 < s[mid]:
                hi = mid
            else:
                lo = mid
        S_lo, S_hi = s[lo], s[hi]
        V_lo, V_hi = V[lo], V[hi]
        w = (S0 - S_lo) / (S_hi - S_lo)
        return float((1.0 - w) * V_lo + w * V_hi)

    def _delta_gamma_fd(self, V: List[float]) -> Tuple[float, float]:
        """
        Backup finite-difference stencil for delta/gamma if the grid
        is too coarse for a stable cubic fit.
        """
        s = self.s_nodes
        N = len(s) - 1
        if N < 2:
            return 0.0, 0.0
        S0 = self.spot
        idx = min(range(N + 1), key=lambda k: abs(s[k] - S0))
        idx = max(1, min(N - 1, idx))
        S_im1, S_i, S_ip1 = s[idx - 1], s[idx], s[idx + 1]
        V_im1, V_i, V_ip1 = V[idx - 1], V[idx], V[idx + 1]
        h1 = S_i - S_im1
        h2 = S_ip1 - S_i
        delta = (
            -h2 / (h1 * (h1 + h2)) * V_im1
            + (h2 - h1) / (h1 * h2) * V_i
            + h1 / (h2 * (h1 + h2)) * V_ip1
        )
        gamma = 2.0 * (
            V_im1 / (h1 * (h1 + h2))
            - V_i / (h1 * h2)
            + V_ip1 / (h2 * (h1 + h2))
        )
        return float(delta), float(gamma)

    def _solve_and_cubic(self, N_time: int, ran_steps: int) -> Tuple[float, float, float]:
        V = self._solve_grid(N_time, rannacher_steps=ran_steps)
        price, delta, gamma = self._price_delta_gamma_cubic(V)
        return price, delta, gamma

    # ------------------------------------------------------------------
    # Public: price with Richardson extrapolation
    # ------------------------------------------------------------------
    def price_log2(self, use_richardson: bool = True) -> float:
        """
        American price from the CN+Rannacher+IT PDE solver.

        With Richardson, we run with N and 2N time steps (and Rannacher
        windows r and 2r) and combine via

            V ≈ (4 V_{2N} - V_N) / 3.
        """
        N = self.num_time_steps
        r = self.rannacher_steps

        if not use_richardson:
            price, _, _ = self._solve_and_cubic(N, r)
            return price

        pN, _, _ = self._solve_and_cubic(N, r)
        p2N, _, _ = self._solve_and_cubic(2 * N, 2 * r)

        return (4.0 * p2N - pN) / 3.0

    # ------------------------------------------------------------------
    # Helper: bumped price for vega/theta
    # ------------------------------------------------------------------
    def _price_fd(
        self,
        spot: Optional[float] = None,
        sigma: Optional[float] = None,
        maturity_date: Optional[_dt.date] = None,
        use_richardson: bool = True,
    ) -> float:
        new_spot = self.spot if spot is None else spot
        new_sigma = self.sigma if sigma is None else sigma
        new_maturity = self.maturity_date if maturity_date is None else maturity_date

        bumped = AmericanFDMPricer(
            spot=new_spot,
            strike=self.strike,
            valuation_date=self.valuation_date,
            maturity_date=new_maturity,
            sigma=new_sigma,
            option_type=self.option_type,  # type: ignore[arg-type]
            discount_curve=self.discount_curve_df,
            forward_curve=self.forward_curve_df,
            dividend_schedule=self.dividend_schedule,
            trade_id=self.trade_id,
            direction=self.direction,
            quantity=self.quantity,
            contract_multiplier=self.contract_multiplier,
            underlying_spot_days=self.underlying_spot_days,
            option_days=self.option_days,
            option_settlement_days=self.option_settlement_days,
            day_count=self.day_count,
            grid_type=self.grid_type,
            num_space_nodes=self.num_space_nodes,
            num_time_steps=self.num_time_steps,
            rannacher_steps=self.rannacher_steps,
            s_max_mult=self.s_max_mult,
        )
        return bumped.price_log2(use_richardson=use_richardson)

    # ------------------------------------------------------------------
    # Public: Greeks with Richardson extrapolation
    # ------------------------------------------------------------------
    def greeks_log2(
        self,
        dv_sigma: float = 0.0001,
        use_richardson: bool = True,
    ) -> Dict[str, float]:
        """
        Greeks for the American option:

        - price : PDE price (with Richardson if enabled)
        - delta : from cubic interpolation, with Richardson
        - gamma : from cubic interpolation, with Richardson
        - vega  : bump-and-revalue in sigma (per 1% vol)
        - theta : bump in maturity date (calendar time)
        """
        N = self.num_time_steps
        r = self.rannacher_steps

        if use_richardson:
            pN, dN, gN = self._solve_and_cubic(N, r)
            p2N, d2N, g2N = self._solve_and_cubic(2 * N, 2 * r)
            price = (4.0 * p2N - pN) / 3.0
            delta = (4.0 * d2N - dN) / 3.0
            gamma = (4.0 * g2N - gN) / 3.0
        else:
            price, delta, gamma = self._solve_and_cubic(N, r)

        # Vega: bump vol (per 1% absolute vol)
        S0 = self.spot
        sigma0 = self.sigma
        dSigma = dv_sigma

        p_up_v = self._price_fd(
            spot=S0,
            sigma=sigma0 + dSigma,
            maturity_date=self.maturity_date,
            use_richardson=use_richardson,
        )
        vega = (p_up_v - price) / (100.0 * dSigma)

        # Theta: symmetric bump in maturity where possible
        T0 = self.time_to_expiry
        dT = 0.0001  # ~0.0365 days
        dt_days = max(1, int(round(dT * 365.0)))

        if T0 > 2.0 * dT:
            mat_up = self.maturity_date + _dt.timedelta(days=dt_days)
            mat_dn = self.maturity_date - _dt.timedelta(days=dt_days)
            p_up_T = self._price_fd(
                spot=S0,
                sigma=sigma0,
                maturity_date=mat_up,
                use_richardson=use_richardson,
            )
            p_dn_T = self._price_fd(
                spot=S0,
                sigma=sigma0,
                maturity_date=mat_dn,
                use_richardson=use_richardson,
            )
            dV_dT = (p_up_T - p_dn_T) / (2.0 * dT)
            theta = -dV_dT
        else:
            mat_dn = self.maturity_date - _dt.timedelta(days=dt_days)
            p_dn_T = self._price_fd(
                spot=S0,
                sigma=sigma0,
                maturity_date=mat_dn,
                use_richardson=use_richardson,
            )
            dV_dT = (price - p_dn_T) / dT
            theta = -dV_dT

        return {
            "price": float(price),
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "theta": float(theta),
        }
