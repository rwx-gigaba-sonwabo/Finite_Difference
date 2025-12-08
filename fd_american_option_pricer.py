import datetime as _dt
from typing import List, Optional, Tuple, Literal, Dict
import math

import numpy as np
import pandas as pd  # type: ignore
from workalendar.africa import SouthAfrica

OptionType = Literal["call", "put"]


class AmericanFDMPricer:
    """
    American vanilla option pricer using a Crank–Nicolson finite-difference
    scheme in log S, with discrete dividends handled via explicit jumps,
    and Ikonen–Toivanen operator splitting for early exercise.

    Key features aligned with FIS documentation and the referenced dividend paper:

      - Underlying follows BS dynamics with *no continuous dividend yield* q=0.
      - Discrete cash dividends: PDE is split between ex-div dates; at each ex-div:
            V(t_d-, S) = V(t_d+, S - D)
        interpolated by a natural cubic spline in S.
      - For American CALLS only:
            V(t_d-, S) = max( V(t_d+, S - D), payoff(S) )
        (immediate exercise possibility before ex-div).
      - Rannacher time-stepping (EB+CN) restarted at each dividend for CALLs,
        and at expiry for both calls and puts.
      - American PUTs: no special dividend smoothing (per FIS note).

    Greeks:
      - Price interpolated at S0 from the final grid.
      - Delta & gamma from a local cubic fit in spot.
      - Vega by symmetric vol bump on sigma with optional Richardson
        extrapolation in time.
      - Theta from the BS PDE relation at S0.
    """

    # ------------------------------------------------------------------ #
    #  Constructor / setup
    # ------------------------------------------------------------------ #

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

        self.calendar = SouthAfrica()
        self.underlying_spot_days = int(underlying_spot_days)
        self.option_days = int(option_days)
        self.option_settlement_days = int(option_settlement_days)

        self.day_count = day_count.upper().replace("F", "")
        self._year_denominator = self._infer_denominator(self.day_count)

        self.grid_type = grid_type.lower()

        # Carry/discount dates
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

        self.time_to_expiry = self._year_fraction(self.valuation_date, self.maturity_date)
        self.time_to_carry = self._year_fraction(self.carry_start_date, self.carry_end_date)
        self.time_to_discount = self._year_fraction(
            self.discount_start_date, self.discount_end_date
        )

        if self.time_to_expiry <= 0.0:
            raise ValueError("time_to_expiry must be positive.")

        # Rates (NACC)
        self.discount_rate_nacc = self.get_forward_nacc_rate(
            self.discount_start_date, self.discount_end_date
        )

        if self.forward_curve_df is not None:
            self.carry_rate_nacc = self.get_forward_nacc_rate(
                self.carry_start_date, self.carry_end_date
            )
        else:
            self.carry_rate_nacc = self.discount_rate_nacc

        # Discrete-div model: PDE uses q = 0, dividends handled by jumps.
        self.div_yield_nacc = 0.0

        # Grid controls
        self.num_space_nodes = max(int(num_space_nodes), 3)
        self.num_time_steps = max(int(num_time_steps), 4)
        self.rannacher_steps = max(int(rannacher_steps), 0)
        self.s_max_mult = float(s_max_mult)

        # Snapping
        self.snap_spot_to_grid: bool = True
        self.snap_strike_to_grid: bool = True
        self.spot_grid_index: Optional[int] = None
        self.spot_snapped: Optional[float] = None
        self.strike_grid_index: Optional[int] = None
        self.strike_snapped: Optional[float] = None

        # Grid state
        self.s_nodes: List[float] = []
        self.x_nodes: List[float] = []
        self._S_min: float = 0.0
        self._S_max: float = 0.0
        self._dx: float = 0.0

    # ------------------------------------------------------------------ #
    #  Day count / curves
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    #  Grid construction in log S
    # ------------------------------------------------------------------ #

    def _configure_grid(self) -> None:
        T = self.time_to_expiry
        sig = self.sigma

        s_low = min(self.spot, self.strike)
        s_high = max(self.spot, self.strike)
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
        self._configure_grid()
        x_min = math.log(self._S_min)
        x_max = math.log(self._S_max)

        n = self.num_space_nodes
        dx = (x_max - x_min) / float(n)

        self.x_nodes = [x_min + i * dx for i in range(n + 1)]
        self.s_nodes = [math.exp(x) for x in self.x_nodes]
        self._dx = dx

        self._snap_critical_levels_to_grid()
        return dx

    def _snap_critical_levels_to_grid(self) -> None:
        s = self.s_nodes
        N = len(s)
        if N == 0:
            return

        if self.snap_spot_to_grid:
            idx = min(range(N), key=lambda i: abs(s[i] - self.spot))
            self.spot_grid_index = idx
            self.spot_snapped = s[idx]
        else:
            self.spot_grid_index = None
            self.spot_snapped = None

        if self.snap_strike_to_grid:
            idx = min(range(N), key=lambda i: abs(s[i] - self.strike))
            self.strike_grid_index = idx
            self.strike_snapped = s[idx]
        else:
            self.strike_grid_index = None
            self.strike_snapped = None

    # ------------------------------------------------------------------ #
    #  Payoff & boundaries
    # ------------------------------------------------------------------ #

    def _strike_for_pde(self) -> float:
        if self.snap_strike_to_grid and self.strike_snapped is not None:
            return self.strike_snapped
        return self.strike

    def _intrinsic_payoff(self, S: float) -> float:
        k = self._strike_for_pde()
        if self.option_type == "call":
            return max(S - k, 0.0)
        return max(k - S, 0.0)

    def _terminal_payoff(self) -> List[float]:
        return [self._intrinsic_payoff(s) for s in self.s_nodes]

    def _boundary_values(self, tau: float) -> Tuple[float, float]:
        S_max = self.s_nodes[-1]
        r = self.discount_rate_nacc
        b = self.carry_rate_nacc
        k = self._strike_for_pde()

        if self.option_type == "call":
            V_min = 0.0
            V_max = S_max * math.exp((b - r) * tau) - k * math.exp(-r * tau)
        else:
            V_min = k * math.exp(-r * tau)
            V_max = 0.0
        return V_min, V_max

    # ------------------------------------------------------------------ #
    #  Dividend times (tau = time-to-maturity)
    # ------------------------------------------------------------------ #

    def _div_times_tau(self) -> List[Tuple[float, float]]:
        if not self.dividend_schedule:
            return []
        res: List[Tuple[float, float]] = []
        for pay_date, amount in self.dividend_schedule:
            if self.valuation_date < pay_date < self.maturity_date:
                t_rel = self._year_fraction(self.valuation_date, pay_date)
                if 0.0 < t_rel < self.time_to_expiry:
                    tau_div = self.time_to_expiry - t_rel
                    res.append((tau_div, float(amount)))
        res.sort(key=lambda x: x[0])
        return res

    # ------------------------------------------------------------------ #
    #  Natural cubic spline builder used at dividend jumps
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_natural_cubic_spline(x: List[float], y: List[float]):
        """
        Build a natural cubic spline interpolant S(x) through knot points
        (x[i], y[i]). Returns a callable f(xq).
        """
        n = len(x)
        if n < 2:
            raise ValueError("Need at least two points for spline.")

        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        h = np.diff(x_arr)
        if np.any(h <= 0.0):
            raise ValueError("x must be strictly increasing.")

        alpha = np.zeros(n)
        for i in range(1, n - 1):
            alpha[i] = (
                3.0 / h[i] * (y_arr[i + 1] - y_arr[i])
                - 3.0 / h[i - 1] * (y_arr[i] - y_arr[i - 1])
            )

        l = np.ones(n)
        mu = np.zeros(n)
        z = np.zeros(n)

        for i in range(1, n - 1):
            l[i] = 2.0 * (x_arr[i + 1] - x_arr[i - 1]) - h[i - 1] * mu[i - 1]
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

        c = np.zeros(n)
        b = np.zeros(n - 1)
        d = np.zeros(n - 1)

        for j in range(n - 2, -1, -1):
            c[j] = z[j] - mu[j] * c[j + 1]
            b[j] = (
                (y_arr[j + 1] - y_arr[j]) / h[j]
                - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
            )
            d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])

        a = y_arr[:-1]

        def spline_eval(xq: float) -> float:
            # Find segment index j such that x[j] <= xq <= x[j+1]
            if xq <= x_arr[0]:
                j = 0
            elif xq >= x_arr[-1]:
                j = n - 2
            else:
                lo, hi = 0, n - 1
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if xq < x_arr[mid]:
                        hi = mid
                    else:
                        lo = mid
                j = lo
            dx = xq - x_arr[j]
            return float(a[j] + b[j] * dx + c[j] * dx * dx + d[j] * dx * dx * dx)

        return spline_eval

    # ------------------------------------------------------------------ #
    #  Segment solver: CN + Rannacher + Ikonen–Toivanen
    # ------------------------------------------------------------------ #

    def _solve_segment(
        self,
        V_init: List[float],
        tau_start: float,
        tau_end: float,
        n_steps: int,
        restart_rannacher: bool,
    ) -> List[float]:
        """
        Solve backwards in tau from tau_start to tau_end with n_steps steps.
        tau=0 is expiry; tau increases as we go back to valuation.
        """
        if n_steps < 1:
            return V_init

        dx = self._dx
        N = len(self.s_nodes) - 1
        if N < 2:
            raise RuntimeError("Spatial grid too coarse.")

        dt = (tau_end - tau_start) / float(n_steps)
        sig = self.sigma
        sig2 = sig * sig
        r = self.discount_rate_nacc
        b = self.carry_rate_nacc
        q = 0.0  # discrete-div model: no continuous yield

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

        def solve_tridiag(A_L: float, A_C: float, A_U: float, rhs: List[float]) -> List[float]:
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

        V = V_init[:]
        payoff_full = [self._intrinsic_payoff(s) for s in self.s_nodes]
        payoff_int = payoff_full[1:-1]

        lambda_int = [0.0] * (N - 1)
        base_ran = self.rannacher_steps if restart_rannacher else 0

        tau = tau_start
        for m in range(n_steps):
            tau_next = tau + dt
            theta = 1.0 if m < base_ran else 0.5

            A_L, A_C, A_U, B_L, B_C, B_U = build_matrices(theta)
            V_min, V_max = self._boundary_values(tau_next)

            rhs = [0.0] * (N - 1)
            for j in range(1, N):
                Vjm1, Vj, Vjp1 = V[j - 1], V[j], V[j + 1]
                rhs[j - 1] = (
                    B_L * Vjm1 + B_C * Vj + B_U * Vjp1
                    + dt * lambda_int[j - 1]
                )

            rhs[0] -= A_L * V_min
            rhs[-1] -= A_U * V_max

            tilde_int = solve_tridiag(A_L, A_C, A_U, rhs)

            new_lambda_int = [0.0] * (N - 1)
            new_V_int = [0.0] * (N - 1)

            for k in range(N - 1):
                tilde_vk = tilde_int[k]
                phi_k = payoff_int[k]
                lam_old = lambda_int[k]

                v_candidate = tilde_vk - dt * lam_old
                v_new = phi_k if phi_k > v_candidate else v_candidate

                lam_new = lam_old + (phi_k - tilde_vk) / dt
                if lam_new < 0.0:
                    lam_new = 0.0

                new_lambda_int[k] = lam_new
                new_V_int[k] = v_new

            V[0] = V_min
            V[-1] = V_max
            V[1:-1] = new_V_int
            lambda_int = new_lambda_int
            tau = tau_next

        return V

    # ------------------------------------------------------------------ #
    #  Dividend jump mapping
    # ------------------------------------------------------------------ #

    def _apply_dividend_jump(self, V_after: List[float], cash_div: float) -> List[float]:
        """
        Apply ex-dividend jump mapping at a dividend of cash_div:

            V(t_d-, S) = V(t_d+, S - D)

        using natural cubic spline interpolation in S. For American CALLS:

            V(t_d-, S) = max( V(t_d+, S - D), payoff(S) )

        which captures potential early exercise just before ex-div.
        """
        s = self.s_nodes
        spline = self._build_natural_cubic_spline(s, V_after)

        V_new: List[float] = []
        for Sj in s:
            Sj_minus = Sj - cash_div
            if Sj_minus <= s[0]:
                cont_val = V_after[0]
            elif Sj_minus >= s[-1]:
                cont_val = V_after[-1]
            else:
                cont_val = spline(Sj_minus)

            if self.option_type == "call":
                ex_val = self._intrinsic_payoff(Sj)
                V_new.append(max(cont_val, ex_val))
            else:
                V_new.append(cont_val)

        return V_new

    # ------------------------------------------------------------------ #
    #  Global solver with time segments and dividend jumps
    # ------------------------------------------------------------------ #

    def _solve_grid(self, N_time: Optional[int] = None) -> List[float]:
        # Build grid once
        self._build_log_grid()

        V = self._terminal_payoff()
        T = self.time_to_expiry

        div_times = self._div_times_tau()  # sorted by tau increasing
        base_N = self.num_time_steps if N_time is None else int(N_time)
        base_dt = T / float(base_N)

        tau_points = [0.0] + [tau for tau, _ in div_times] + [T]
        n_segments = len(tau_points) - 1

        seg_lengths = [tau_points[i + 1] - tau_points[i] for i in range(n_segments)]
        seg_steps: List[int] = []
        remaining_steps = base_N
        for L in seg_lengths[:-1]:
            n = max(1, int(round(L / base_dt)))
            seg_steps.append(n)
            remaining_steps -= n
        seg_steps.append(max(1, remaining_steps))

        tau = 0.0
        for seg_idx in range(n_segments):
            tau_start = tau_points[seg_idx]
            tau_end = tau_points[seg_idx + 1]
            n_steps = seg_steps[seg_idx]

            restart_ran = (
                seg_idx == 0
                or (seg_idx > 0 and self.option_type == "call")
            )

            V = self._solve_segment(
                V_init=V,
                tau_start=tau_start,
                tau_end=tau_end,
                n_steps=n_steps,
                restart_rannacher=restart_ran,
            )
            tau = tau_end

            # Apply dividend jump after segment if there is a dividend here
            if seg_idx < len(div_times):
                _, D_cash = div_times[seg_idx]
                V = self._apply_dividend_jump(V, D_cash)

        return V

    # ------------------------------------------------------------------ #
    #  Interpolation at spot & local cubic for Δ / Γ
    # ------------------------------------------------------------------ #

    def _spot_for_interp(self) -> float:
        if self.snap_spot_to_grid and self.spot_snapped is not None:
            return self.spot_snapped
        return self.spot

    def _interp_price(self, V: List[float]) -> float:
        s = self.s_nodes
        S0 = self._spot_for_interp()

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
        w = (S0 - s[lo]) / (s[hi] - s[lo])
        return float((1.0 - w) * V[lo] + w * V[hi])

    def _local_cubic_delta_gamma(self, V: List[float]) -> Tuple[float, float]:
        s = self.s_nodes
        Sx = self._spot_for_interp()
        n = len(s) - 1

        i_near = min(range(n + 1), key=lambda k: abs(s[k] - Sx))
        if i_near < 1:
            i_near = 1
        elif i_near > n - 2:
            i_near = n - 2

        idx = [i_near - 1, i_near, i_near + 1, i_near + 2]
        x = np.array([s[j] for j in idx], dtype=float)
        y = np.array([V[j] for j in idx], dtype=float)

        z = x - Sx
        M = np.vstack([z ** 3, z ** 2, z, np.ones_like(z)]).T
        a, b, c, d = np.linalg.solve(M, y)

        delta = float(c)
        gamma = float(2.0 * b)
        return delta, gamma

    # ------------------------------------------------------------------ #
    #  Public price & Greeks (API aligned with your scenario runner)
    # ------------------------------------------------------------------ #

    def price_log(self, N_time: Optional[int] = None) -> float:
        V = self._solve_grid(N_time=N_time)
        return self._interp_price(V)

    def price_log2(self, apply_KO: bool = True, use_richardson: bool = True) -> float:
        """
        Price with optional Richardson extrapolation in time (N vs 2N).
        apply_KO is ignored (kept for API symmetry with barrier pricer).
        """
        if not use_richardson:
            return self.price_log(N_time=self.num_time_steps)
        pN = self.price_log(N_time=self.num_time_steps)
        p2N = self.price_log(N_time=2 * self.num_space_nodes)
        return (4.0 * p2N - pN) / 3.0

    def _price_for_sigma(self, sigma: float, N_time: Optional[int] = None) -> float:
        orig_sigma = self.sigma
        try:
            self.sigma = sigma
            return self.price_log(N_time=N_time)
        finally:
            self.sigma = orig_sigma

    def greeks_log2(
        self,
        dv_sigma: float = 0.01,
        use_richardson: bool = True,
    ) -> Dict[str, float]:
        """
        Price and Greeks:
          - price: PDE + CN/IT with time segmentation (Richardson in time)
          - delta, gamma: local cubic at spot
          - vega: symmetric sigma bump (per 1 vol point) with Richardson
          - theta: BS PDE identity at spot
        """
        # Base grid (N steps)
        V_N = self._solve_grid(N_time=self.num_time_steps)
        price_N = self._interp_price(V_N)
        delta_N, gamma_N = self._local_cubic_delta_gamma(V_N)

        if use_richardson:
            V_2N = self._solve_grid(N_time=2 * self.num_time_steps)
            price_2N = self._interp_price(V_2N)
            delta_2N, gamma_2N = self._local_cubic_delta_gamma(V_2N)

            price = (4.0 * price_2N - price_N) / 3.0
            delta = (4.0 * delta_2N - delta_N) / 3.0
            gamma = (4.0 * gamma_2N - gamma_N) / 3.0
        else:
            price = price_N
            delta = delta_N
            gamma = gamma_N

        # Vega: symmetric bump in sigma
        sigma0 = self.sigma
        h = dv_sigma

        if use_richardson:
            p_up_h = self._price_for_sigma(sigma0 + h, N_time=self.num_time_steps)
            p_dn_h = self._price_for_sigma(sigma0 - h, N_time=self.num_time_steps)
            D_h = (p_up_h - p_dn_h) / (2.0 * h)

            p_up_2h = self._price_for_sigma(sigma0 + 2.0 * h, N_time=self.num_time_steps)
            p_dn_2h = self._price_for_sigma(sigma0 - 2.0 * h, N_time=self.num_time_steps)
            D_2h = (p_up_2h - p_dn_2h) / (4.0 * h)

            dV_dsigma = (4.0 * D_h - D_2h) / 3.0
        else:
            p_up = self._price_for_sigma(sigma0 + h, N_time=self.num_time_steps)
            p_dn = self._price_for_sigma(sigma0 - h, N_time=self.num_time_steps)
            dV_dsigma = (p_up - p_dn) / (2.0 * h)

        vega = dV_dsigma / 100.0  # per 1% vol

        # Theta from BS PDE at S0 (q = 0 in discrete-div model)
        r = self.discount_rate_nacc
        b = self.carry_rate_nacc
        q = 0.0
        S0 = self.spot

        theta = -(
            0.5 * sigma0 * sigma0 * S0 * S0 * gamma
            + (b - q) * S0 * delta
            - r * price
        )

        return {
            "price": float(price),
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "theta": float(theta),
        }
