import datetime as _dt
from typing import List, Optional, Tuple, Literal, Dict
import math

import pandas as pd  # type: ignore
from workalendar.africa import SouthAfrica  # same calendar as barrier pricer
from scipy.stats import norm

OptionType = Literal["call", "put"]


class AmericanFDMPricer:
    """
    American vanilla option pricer using a Crank–Nicolson finite-difference
    scheme in log S with Rannacher smoothing.

    Design is aligned with the FIS-style finite-difference solver in your
    discrete barrier engine:

    - Log-space grid with domain chosen from a lognormal band around an
      effective spot S_eff = S0 - PV(divs).
    - Critical levels (spot, strike) are snapped to the nearest S-grid node
      for PDE/Greek computations.
    - Discrete dividends are converted to a flat continuous dividend yield q
      such that PV(divs) is preserved over the carry tenor.
    - PDE drift uses (b - q), with b the carry rate implied by the curves;
      discounting at rate r from the discount curve.
    - American early exercise is enforced via projection:
          V_j <- max(V_j, intrinsic(S_j)) at every time step.
    """

    # ------------------------------------------------------------------
    #  Construction
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
        snap_strike_to_grid: bool = True,
        snap_spot_to_grid: bool = True,
    ) -> None:
        if spot <= 0.0 or strike <= 0.0 or sigma <= 0.0:
            raise ValueError("spot, strike, sigma must be positive.")
        if maturity_date <= valuation_date:
            raise ValueError("maturity_date must be after valuation_date.")

        # Economic inputs
        self.spot = float(spot)
        self.strike = float(strike)
        self.valuation_date = valuation_date
        self.maturity_date = maturity_date
        self.sigma = float(sigma)

        self.option_type: str = option_type.lower()
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")

        self.discount_curve_df = discount_curve.copy()
        self.forward_curve_df = forward_curve.copy() if forward_curve is not None else None
        self.dividend_schedule = sorted(dividend_schedule or [], key=lambda x: x[0])

        self.trade_id = trade_id
        self.direction = direction
        self.quantity = int(quantity)
        self.contract_multiplier = float(contract_multiplier)

        # Calendars / conventions
        self.calendar = SouthAfrica()
        self.underlying_spot_days = int(underlying_spot_days)
        self.option_days = int(option_days)
        self.option_settlement_days = int(option_settlement_days)

        self.day_count = day_count.upper().replace("F", "")
        self._year_denominator = self._infer_denominator(self.day_count)

        # Dates for carry and discount legs
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

        # Rates (NACC) from curves
        self.discount_rate_nacc = self.get_forward_nacc_rate(
            self.discount_start_date, self.discount_end_date
        )
        self.carry_rate_nacc = self.get_forward_nacc_rate(
            self.carry_start_date, self.carry_end_date
        )

        # Dividends → flat continuous dividend yield q matching PV(divs)
        self.pv_divs = self.pv_dividends()
        self.div_yield_nacc = self.dividend_yield_nacc()

        # Forward (diagnostics)
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

        # Snap controls (FIS-style)
        self.snap_strike_to_grid = bool(snap_strike_to_grid)
        self.snap_spot_to_grid = bool(snap_spot_to_grid)

        # Grid / snapping state
        self._S_min: float = 0.0
        self._S_max: float = 0.0
        self.s_nodes: List[float] = []
        self.x_nodes: List[float] = []

        self.spot_grid_index: Optional[int] = None
        self.spot_snapped: Optional[float] = None
        self.strike_grid_index: Optional[int] = None
        self.strike_snapped: Optional[float] = None

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
    # Dividend handling (aligned with discrete barrier FDM)
    # ------------------------------------------------------------------
    def pv_dividends(self) -> float:
        """
        PV of discrete dividend_schedule over [valuation, maturity],
        discounted by the discount curve and normalised by DF(carry_start).
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
        Flat continuous dividend yield q such that PV(divs) matches
        the escrowed-dividend argument:
           S0_eff = S0 - PV(divs) = S0 * e^{-q T_carry}.
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
    # Grid selection & snapping (FIS-style)
    # ------------------------------------------------------------------
    def choose_grid_parameters(
        self,
        S0_eff: float,
        K: float,
        T: float,
        sigma: float,
    ) -> Tuple[int, int, float, float]:
        """
        Return (N_space, N_time, S_min, S_max) for a log-space CN scheme.

        Mirrors the logic from the discrete barrier FDM solver,
        but without barrier levels.
        """
        if T <= 0.0:
            raise ValueError("Maturity T must be positive.")
        if sigma <= 0.0:
            raise ValueError("Volatility sigma must be positive.")
        if S0_eff <= 0.0:
            raise ValueError("Effective spot S0_eff must be positive.")

        # Critical levels to be well-resolved
        candidates = [S0_eff, K]
        s_low = min(candidates)
        s_high = max(candidates)

        # Wide log-normal band (e.g. 99.999% quantile) around geometric mean
        k = norm.ppf(0.99999)
        domain_width = 2.0 * k * sigma * math.sqrt(T)

        s_c = math.sqrt(s_low * s_high)
        x_c = math.log(s_c)

        x_min = x_c - 0.5 * domain_width
        x_max = x_c + 0.5 * domain_width

        S_min = math.exp(x_min)
        S_max = math.exp(x_max)

        # Ensure we still cover at least [0.5 s_low, 2 s_high]
        S_min = min(S_min, 0.5 * s_low)
        S_max = max(S_max, 2.0 * s_high)

        # Time steps from user, space refined from CFL-like relation
        N_time = self.num_time_steps
        N_space = math.ceil((domain_width * N_time) / (2.0 * sigma * math.sqrt(T)))
        N_space = max(N_space, 3)

        return N_space, N_time, S_min, S_max

    def configure_grid(self) -> None:
        """
        Choose N_space, N_time, and S-domain [S_min,S_max] based on current inputs.
        """
        S0_eff = self.spot - self.pv_divs
        if S0_eff <= 0.0:
            S0_eff = self.spot

        N_space, N_time, S_min, S_max = self.choose_grid_parameters(
            S0_eff=S0_eff,
            K=self.strike,
            T=self.time_to_expiry,
            sigma=self.sigma,
        )
        self.num_space_nodes = N_space
        self.num_time_steps = N_time
        self._S_min = S_min
        self._S_max = S_max

    def _snap_critical_levels_to_grid(self) -> None:
        """
        Snap spot and strike to the closest S-grid nodes and record
        the indices and snapped values for PDE / Greeks.
        """
        s = self.s_nodes
        N = len(s)
        if N == 0:
            return

        # Effective spot used for snapping (escrowed-div interpretation)
        S_eff = self.spot - self.pv_divs if self.pv_divs else self.spot

        # Spot
        if self.snap_spot_to_grid:
            i = min(range(N), key=lambda k: abs(s[k] - S_eff))
            self.spot_grid_index = i
            self.spot_snapped = s[i]
        else:
            self.spot_grid_index = None
            self.spot_snapped = None

        # Strike
        if self.snap_strike_to_grid:
            iK = min(range(N), key=lambda k: abs(s[k] - self.strike))
            self.strike_grid_index = iK
            self.strike_snapped = s[iK]
        else:
            self.strike_grid_index = None
            self.strike_snapped = None

    def _build_log_grid(self) -> float:
        """
        Configure grid, build log-S nodes, and snap critical levels.
        """
        self.configure_grid()
        S_min = self._S_min
        S_max = self._S_max

        x_min = math.log(S_min)
        x_max = math.log(S_max)

        n = self.num_space_nodes
        dx = (x_max - x_min) / float(n)

        self.x_nodes = [x_min + i * dx for i in range(n + 1)]
        self.s_nodes = [math.exp(x) for x in self.x_nodes]

        self._snap_critical_levels_to_grid()
        return dx

    # ------------------------------------------------------------------
    # Payoff, boundaries, intrinsic
    # ------------------------------------------------------------------
    def _current_strike_for_pde(self) -> float:
        """Strike used inside the PDE / payoff (snapped if enabled)."""
        if self.snap_strike_to_grid and self.strike_snapped is not None:
            return float(self.strike_snapped)
        return self.strike

    def _intrinsic_payoff(self, S: float) -> float:
        k = self._current_strike_for_pde()
        if self.option_type == "call":
            return max(S - k, 0.0)
        return max(k - S, 0.0)

    def _terminal_payoff(self) -> List[float]:
        return [self._intrinsic_payoff(s) for s in self.s_nodes]

    def _boundary_values(self, tau: float) -> Tuple[float, float]:
        """
        Dirichlet boundaries at time-to-maturity tau for the continuation
        (European) problem. Early exercise is enforced separately.
        """
        S_min = self.s_nodes[0]
        S_max = self.s_nodes[-1]

        r = self.discount_rate_nacc
        b = self.carry_rate_nacc
        k = self._current_strike_for_pde()

        if self.option_type == "call":
            V_min = 0.0
            V_max = S_max * math.exp((b - r) * tau) - k * math.exp(-r * tau)
        else:
            V_min = k * math.exp(-r * tau)
            V_max = 0.0

        return V_min, V_max

    # ------------------------------------------------------------------
    # CN + Rannacher solver with American projection
    # ------------------------------------------------------------------
    def _solve_grid(self, N_time: Optional[int] = None) -> List[float]:
        dx = self._build_log_grid()

        N = len(self.s_nodes) - 1
        if N < 2:
            raise RuntimeError("Spatial grid too coarse; need at least 3 nodes.")

        N_time = int(N_time) if N_time is not None else int(self.num_time_steps)
        if N_time < 1:
            raise RuntimeError("Need at least 1 time step.")
        dt = self.time_to_expiry / float(N_time)

        sig = self.sigma
        sig2 = sig * sig

        r = self.discount_rate_nacc
        b = self.carry_rate_nacc
        q = self.div_yield_nacc

        # Drift in log-space under risk-neutral measure with carry b and yield q
        mu_x = (b - q) - 0.5 * sig2

        alpha = 0.5 * sig2 / (dx * dx)
        beta_adv = mu_x / (2.0 * dx)

        a = alpha - beta_adv
        c = alpha + beta_adv
        bcoef = -2.0 * alpha - r

        def build_matrices(theta: float):
            A_L = -theta * dt * a
            A_C = 1.0 - theta * dt * bcoef
            A_U = -theta * dt * c

            B_L = (1.0 - theta) * dt * a
            B_C = 1.0 + (1.0 - theta) * dt * bcoef
            B_U = (1.0 - theta) * dt * c
            return A_L, A_C, A_U, B_L, B_C, B_U

        def solve_tridiag(A_L, A_C, A_U, rhs):
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

            # American early exercise projection
            for j, S in enumerate(self.s_nodes):
                intrinsic = self._intrinsic_payoff(S)
                if V[j] < intrinsic:
                    V[j] = intrinsic

        return V

    # ------------------------------------------------------------------
    # Interpolation and PDE-based delta/gamma (with snapping)
    # ------------------------------------------------------------------
    def _interp_price(self, V: List[float]) -> float:
        """Price at the true spot S0 via linear interpolation in S-space."""
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

    def _delta_gamma_from_grid(self, V: List[float]) -> Tuple[float, float]:
        """
        Delta and gamma from the PDE grid, using a non-uniform
        3-point central stencil around the (snapped) spot index.
        """
        s = self.s_nodes
        N = len(s) - 1
        if N < 2:
            raise RuntimeError("Need at least 3 spatial nodes for Greeks.")

        # Use snapped index if available, else nearest
        if self.spot_grid_index is not None:
            i = self.spot_grid_index
        else:
            S_eff = self.spot - self.pv_divs if self.pv_divs else self.spot
            i = min(range(N + 1), key=lambda k: abs(s[k] - S_eff))

        # Require interior node
        i = max(1, min(N - 1, i))

        S_im1, S_i, S_ip1 = s[i - 1], s[i], s[i + 1]
        V_im1, V_i, V_ip1 = V[i - 1], V[i], V[i + 1]

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

        gamma = max(min(gamma, 1e5), -1e5)
        return float(delta), float(gamma)

    # ------------------------------------------------------------------
    # Public API: price and Greeks
    # ------------------------------------------------------------------
    def price_log2(self, use_richardson: bool = False) -> float:
        V = self._solve_grid()
        return self._interp_price(V)

    def _price_fd(
        self,
        spot: Optional[float] = None,
        sigma: Optional[float] = None,
        maturity_date: Optional[_dt.date] = None,
    ) -> float:
        """
        Helper for bump-and-revalue Greeks: reprice with bumped parameters
        via a fresh PDE instance, keeping all curve/dividend logic intact.
        """
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
            snap_strike_to_grid=self.snap_strike_to_grid,
            snap_spot_to_grid=self.snap_spot_to_grid,
        )
        return bumped.price_log2()

    def greeks_log2(
        self,
        dv_sigma: float = 0.0001,
        use_richardson: bool = False,
    ) -> Dict[str, float]:
        """
        Greeks from the American CN PDE, aligned with FIS-style methodology:

        - price : base PDE price
        - delta : PDE-based from grid using snapped spot index
        - gamma : PDE-based from grid
        - vega  : bump-and-revalue in volatility (per 1% vol)
        - theta : symmetric bump in maturity (time value of money)
        """
        # Base solution
        V_base = self._solve_grid()
        price_base = self._interp_price(V_base)
        delta, gamma = self._delta_gamma_from_grid(V_base)

        S0 = self.spot
        sigma0 = self.sigma
        T0 = self.time_to_expiry

        # Vega: bump σ, per 1% vol
        dSigma = dv_sigma
        p_up_v = self._price_fd(spot=S0, sigma=sigma0 + dSigma, maturity_date=self.maturity_date)
        vega = (p_up_v - price_base) / (100.0 * dSigma)

        # Theta: bump maturity in calendar time
        dT = 0.0001  # 1e-4 years ~ 0.0365 days
        dt_days = max(1, int(round(dT * 365.0)))

        if T0 > 2.0 * dT:
            mat_up = self.maturity_date + _dt.timedelta(days=dt_days)
            mat_dn = self.maturity_date - _dt.timedelta(days=dt_days)
            p_up_T = self._price_fd(spot=S0, sigma=sigma0, maturity_date=mat_up)
            p_dn_T = self._price_fd(spot=S0, sigma=sigma0, maturity_date=mat_dn)
            dV_dT = (p_up_T - p_dn_T) / (2.0 * dT)
            theta = -dV_dT
        else:
            mat_dn = self.maturity_date - _dt.timedelta(days=dt_days)
            p_dn_T = self._price_fd(spot=S0, sigma=sigma0, maturity_date=mat_dn)
            dV_dT = (price_base - p_dn_T) / dT
            theta = -dV_dT

        return {
            "price": float(price_base),
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "theta": float(theta),
        }
