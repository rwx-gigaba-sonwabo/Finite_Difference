import datetime as _dt
from typing import List, Optional, Tuple, Literal, Dict, Any
import math
import pandas as pd  # type: ignore
from workalendar.africa import SouthAfrica
from scipy.stats import norm
import numpy as np

"""
Discrete Barrier Option Pricer (Crank-Nicolson + Rannacher), datetime + daily NACA curves.

- Dates: use datetime.date for valuation_date, maturity_date, dividend dates, monitoring dates.
- Curves: pandas DataFrames with columns ["Date", "NACA"] where Date is ISO YYYY-MM-DD string.
- DF math: ACT/365F day count, DF = (1 + NACA)^(-tau).
- Escrowed dividend_schedule: PV at valuation using DF(0, t_div); PDE uses q=0; S0_eff = spot_price - PV(divs).
- Knock-outs handled by projection at monitoring dates; knock-ins via in-out parity.

"""

BarrierType = Literal[
    "down-and-out",
    "up-and-out",
    "double-out",
    "down-and-in",
    "up-and-in",
    "double-in",
    "none",
]

OptionType = Literal["call", "put"]


class DiscreteBarrierFDMPricer:
    """
    CN FDM pricer for discretely monitored European barrier options with daily curves.

    Curve DataFrames: columns "Date" (YYYY-MM-DD), "NACA" (decimal).
    Dividends: list[(datetime.date, cash_amount)].
    Monitoring: list[datetime.date] for KO projection.
    """

    def __init__(
        self,
        spot: float,
        strike: float,
        valuation_date: _dt.date,
        maturity_date: _dt.date,
        sigma: float,
        option_type: OptionType,
        barrier_type: BarrierType = "none",
        lower_barrier: Optional[float] = None,
        upper_barrier: Optional[float] = None,
        monitor_dates: Optional[List[_dt.date]] = None,
        rebate_amount: float = 0.0,
        rebate_at_hit: bool = False,
        already_hit: bool = False,
        already_in: bool = False,
        underlying_spot_days: float = 3,
        option_days: float = 0,
        option_settlement_days: float = 0,
        discount_curve: Optional[Any] = None,
        forward_curve: Optional[Any] = None,
        dividend_schedule: Optional[List[Tuple[_dt.date, float]]] = None,
        trade_id: float = None,
        direction: Literal["long", "short"] = "long",
        quantity: int = 1,
        contract_multiplier: float = 1.0,
        min_substeps_between_monitors: int = 1,
        grid_type: Literal["uniform", "sinh"] = "uniform",
        sinh_alpha: float = 1.5,
        lambda_diff_target: float = 0.5,
        num_space_nodes: int = 400,
        num_time_steps: int = 400,
        rannacher_steps: int = 2,
        s_max_mult: float = 4.5,
        restart_on_monitoring: bool = False,
        use_one_sided_greeks_near_barrier: bool = True,
        mollify_final: bool = True,
        mollify_band_nodes: int = 2,
        price_extrapolation: bool = False,
        day_count: str = "ACT/365",
        calculate_greeks_in_pde: bool = True,
    ) -> None:
        # Basic validation
        if any(x <= 0 for x in (spot, strike, sigma)):
            raise ValueError("spot, strike, sigma must be positive.")
        if maturity_date <= valuation_date:
            raise ValueError("maturity_date must be after valuation_date.")

        # Core inputs
        self.spot = spot
        self.strike = strike
        self.valuation_date = valuation_date
        self.maturity_date = maturity_date
        self.sigma = sigma
        self.option_type = option_type

        # Barrier description
        self.barrier_type = barrier_type
        self.lower_barrier = lower_barrier
        self.upper_barrier = upper_barrier
        self.monitor_dates = sorted(monitor_dates or [])
        self.rebate_amount = rebate_amount
        self.rebate_at_hit = rebate_at_hit

        self.already_hit = already_hit
        self.already_in = already_in

        # Spot considerations
        self.underlying_spot_days = underlying_spot_days
        self.option_days = option_days
        self.option_settlement_days = option_settlement_days
        self.calendar = SouthAfrica()

        # Yield curve and dividend handling
        self.discount_curve_df = discount_curve.copy() if discount_curve is not None else None
        self.forward_curve_df = forward_curve.copy() if forward_curve is not None else None
        self.dividend_schedule = sorted(dividend_schedule or [], key=lambda x: x[0])

        # Trade details
        self.trade_id = trade_id
        self.direction = direction
        self.quantity = int(quantity)
        self.contract_multiplier = float(contract_multiplier)

        # Numerical controls
        self.num_space_nodes = int(num_space_nodes)
        self.num_time_steps = int(num_time_steps)
        self.rannacher_steps = int(rannacher_steps)
        self.min_substeps = max(1, int(min_substeps_between_monitors))
        self.lambda_diff_target = float(lambda_diff_target)

        self.s_max_mult = s_max_mult
        self.restart_on_monitoring = restart_on_monitoring
        self.mollify_final = mollify_final
        self.mollify_band_nodes = int(mollify_band_nodes)
        self.price_extrapolation = price_extrapolation
        self.use_one_sided_greeks_near_barrier = use_one_sided_greeks_near_barrier
        self.calculate_greeks_in_pde = calculate_greeks_in_pde

        # Day count , spot adjustments
        self.day_count = day_count.upper().replace("F", "")
        self._year_denominator = self._infer_denominator(self.day_count)

        self.carry_start_date = self.calendar.add_working_days(self.valuation_date, self.underlying_spot_days)
        self.carry_end_date = self.calendar.add_working_days(self.maturity_date, self.underlying_spot_days)

        self.discount_start_date = self.calendar.add_working_days(self.valuation_date, self.option_days)
        self.discount_end_date = self.calendar.add_working_days(self.maturity_date, self.option_settlement_days)

        # Year fraction for option tenor
        self.time_to_expiry = self._year_fraction(self.valuation_date, self.maturity_date)
        self.time_to_carry = self._year_fraction(self.carry_start_date, self.carry_end_date)
        self.time_to_discount = self._year_fraction(self.discount_start_date, self.discount_end_date)

        # Flat rates from curves
        self.discount_rate_nacc = self.get_forward_nacc_rate(self.discount_start_date, self.discount_end_date)
        self.carry_rate_nacc = self.get_forward_nacc_rate(self.carry_start_date, self.carry_end_date)
        self.div_yield_nacc = self.dividend_yield_nacc()
        self.pv_divs = self.pv_dividends()
        self.forward_price = self.spot * math.exp((self.carry_rate_nacc-self.div_yield_nacc) * self.time_to_carry)
        self.b = math.log(self.forward_price / self.spot) / self.time_to_carry

        # Grid
        self.grid_type = grid_type
        self.sinh_alpha = sinh_alpha
        self.stock_grid = self._build_stock_price_grid()
        self.grid_spacing = self.stock_grid[1] - self.stock_grid[0]
        self.time_spacing = self.time_to_expiry / self.num_time_steps
        self.time_grid = [i * self.time_to_expiry / self.num_time_steps for i in range(self.num_time_steps + 1)]
        self.monitor_times = self._build_monitor_times_exact()


    def _infer_denominator(self, day_count: str) -> int:
        """
        Map day count to denominator used for simple year fractions and to convert
        continuous-time fractions to calendar days for mid-point sampling.
        """
        if day_count in ("ACT/365", "ACT/365F"):
            return 365
        if day_count in ("ACT/360", "ACT/364"):
            return 360 if day_count == "ACT/360" else 364
        if day_count in ("30/360", "BOND", "US30/360"):
            return 360
        # default
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
        if self.discount_curve_df is None:
            raise ValueError("No discount curve attached.")
        iso = lookup_date.isoformat()
        row = self.discount_curve_df[self.discount_curve_df["Date"] == iso]
        if row.empty:
            raise ValueError(f"Discount factor not found for date: {iso}")
        naca = float(row["NACA"].values[0])
        tau = self._year_fraction(self.valuation_date, lookup_date)
        return (1.0 + naca) ** (-tau)

    def get_nacc_rate(self, lookup_date: _dt.date) -> float:
        if self.discount_curve_df is None:
            return 0.0
        iso = lookup_date.isoformat()
        row = self.discount_curve_df[self.discount_curve_df["Date"] == iso]
        if row.empty:
            return 0.0
        naca = float(row["NACA"].values[0])
        return math.log(1.0 + naca)

    def get_forward_nacc_rate(self, start_date: _dt.date, end_date: _dt.date) -> float:
        df_far = self.get_discount_factor(end_date)
        df_near = self.get_discount_factor(start_date)
        tau = self._year_fraction(start_date, end_date)
        return -math.log(df_far / df_near) / max(1e-12, tau)

    def pv_dividends(self) -> float:
        """PV of discrete dividend_schedule at valuation_date using DF from discount curve."""
        if self.dividend_schedule is None:
            return 0.0

        pv = 0.0
        for pay_date, amount in self.dividend_schedule:
            if self.valuation_date < pay_date <= self.maturity_date:
                df = self.get_discount_factor(pay_date) / self.get_discount_factor(self.carry_start_date)
                pv += amount * df
        return pv

    def dividend_yield_nacc(self) -> float:
        """Back-out a flat q (NACC) reproducing PV(dividend_schedule) on [valuation, maturity]."""
        pv_divs = self.pv_dividends()
        S = self.spot
        tau = max(1e-12, self.time_to_carry)

        if pv_divs <= 0.0:
            return 0.0

        if pv_divs >= S:
            raise ValueError("PV(dividend_schedule) >= spot.")
        return -math.log((S - pv_divs) / S) / tau

    def _build_monitor_times_exact(self) -> List[float]:
        times = []
        for d in self.monitor_dates:
            if self.valuation_date <= d <= self.maturity_date:
                t = self._year_fraction(self.valuation_date, d)
                if 0.0 <= t <= self.time_to_expiry:
                    times.append(t)
        if times[-1] < self.time_to_expiry - 1e-14:
            times.append(self.time_to_expiry)

        times = sorted(set(times))
        return times

    def choose_grid_parameters(
        self,
        S0: float,
        K: float,
        lower_barrier: Optional[float],
        upper_barrier: Optional[float],
        T: float,
        sigma: float,
    ) -> Tuple[int, int, float, float]:
        """
        Return (N_space, N_time, S_min, S_max) for a log-space CN scheme,
        given the main trade parameters and barrier/monitor structure.
        """

        if T <= 0.0:
            raise ValueError("Maturity T must be positive.")
        if sigma <= 0.0:
            raise ValueError("Volatility sigma must be positive.")
        if S0 <= 0.0:
            raise ValueError("Spot S0 must be positive.")

        candidates = [S0, K]
        if lower_barrier is not None and lower_barrier > 0.0:
            candidates.append(lower_barrier)
        if upper_barrier is not None and upper_barrier > 0.0:
            candidates.append(upper_barrier)

        s_low = min(candidates)
        s_high = max(candidates)

        k = norm.ppf(0.99999)
        domain_width = 2.0 * k * sigma * math.sqrt(self.time_to_expiry)

        s_c = math.sqrt(s_low * s_high)
        x_c = math.log(s_c)

        x_min = x_c - 0.5 * domain_width
        x_max = x_c + 0.5 * domain_width

        S_min = math.exp(x_min)
        S_max = math.exp(x_max)

        S_min = min(S_min, 0.5 * s_low)
        S_max = max(S_max, 2 * s_high)


        N_time = self.num_time_steps
        N_space = math.ceil((domain_width * N_time) / (2 * sigma * math.sqrt(self.time_to_expiry)))


        return N_space, N_time, S_min, S_max

    def configure_grid(self) -> None:
        """
        Choose N_space, N_time, and price domain [S_min, S_max]
        in a principled way based on the current trade parameters.
        Call this once before pricing/greeks.
        """
        N_space, N_time, S_min, S_max = self.choose_grid_parameters(
            S0=self.spot - self.pv_divs,
            K=self.strike,
            lower_barrier=self.lower_barrier,
            upper_barrier=self.upper_barrier,
            T=self.time_to_expiry,
            sigma=self.sigma,
        )

        self.num_space_nodes = N_space
        self.num_time_steps = N_time
        self._S_min = S_min   # store if you want deterministic domain
        self._S_max = S_max

    def _build_log_grid(self) -> float:


        self.configure_grid()
        S_min = self._S_min        # avoid log(0)
        S_max = self._S_max

        x_min = math.log(S_min)
        x_max = math.log(S_max)


        n = self.num_space_nodes
        dx = (x_max - x_min) / n
        x_nodes = [x_min + i * dx for i in range(n + 1)]

        # precompute S-grid for convenience
        self.s_nodes = [math.exp(x) for x in x_nodes]

        # store log-barriers (for comparisons) if needed
        self.lower_barrier_log = math.log(self.lower_barrier) if self.lower_barrier else None
        self.upper_barrier_log = math.log(self.upper_barrier) if self.upper_barrier else None

        return dx

    def _terminal_payoff(self) -> List[float]:
        s_nodes = self.s_nodes
        if self.option_type=="call":
            return [max(s-self.strike,0.0) for s in s_nodes]
        return [max(self.strike-s,0.0) for s in s_nodes]

    def _boundary_values(self, tau: float) -> Tuple[float, float]:
        """
        Dirichlet boundaries at time-to-maturity tau, consistent with
        Black–Scholes with carry b and discount r.
        """
        S_min = self.s_nodes[0]
        S_max = self.s_nodes[-1]

        r = self.discount_rate_nacc
        b = self.carry_rate_nacc
        k = self.strike

        is_call = self.option_type.lower() == "call"

        if is_call:
            V_min = 0.0
            V_max = S_max * math.exp((b - r) * tau) - k * math.exp(-r * tau)
        else:
            V_max = 0.0
            V_min = k * math.exp(-r * tau) * S_min * math.exp((b - r) * tau)

        return V_min, V_max

    def _monitor_indices_tau(self, dt: float) -> set:
        """
        Map discrete monitoring times t_mon in [0,T] to indices in tau = T - t.
        We project at tau_k = (k * dt) that corresponds to t = T - tau_k.
        """
        idx_set = set()
        if not self.monitor_times:
            return idx_set

        for t_mon in self.monitor_times:
            if t_mon <= 0.0 or t_mon > self.time_to_expiry:
                continue
            tau_mon = self.time_to_expiry - t_mon
            k = int(math.floor(tau_mon / dt + 1e-9))
            k = max(1, min(self.num_time_steps, k))
            idx_set.add(k)
        return idx_set

    def _apply_KO_projection(self, V: List[float], s_nodes: List[float],tau_left: float) -> None:

        if self.barrier_type in ("none","down-and-in","up-and-in","double-in"):
            return

        lo = self.lower_barrier
        up = self.upper_barrier

        if self.rebate_at_hit:
            rebate = self.rebate_amount
        else:
            rebate = self.rebate_amount*math.exp(-self.carry_rate_nacc*tau_left)  # PV to expiry
        n = min(len(V), len(s_nodes))

        for i in range(n):
            s = s_nodes[i]
            out=False

            if self.barrier_type=="down-and-out" and lo is not None and s<=lo:
                out=True
            elif self.barrier_type=="up-and-out" and up is not None and s>=up:
                out=True
            elif self.barrier_type=="double-out":
                if (lo is not None and s<=lo) or (up is not None and s>=up):
                    out=True

            if out:
                V[i]=rebate

    def _solve_grid(self, apply_KO: bool, N_time: int = None) -> List[float]:
        """
        Solve dV/dτ = L V with L from Black–Scholes in log S via CN,
        marching τ from 0 to T (backwards in calendar time).
        """
        dx = self._build_log_grid()

        N = self.num_space_nodes - 1
        N_time = int(N_time) if N_time is not None else int(self.num_time_steps)
        assert N_time >= 1
        dt = (self.time_to_expiry)/ self.num_time_steps

        # 2) PDE coefficients in log space
        sig = self.sigma
        sig2 = sig * sig

        # Discount & dividend yields in *continuous* compounding
        r = self.discount_rate_nacc
        b = self.carry_rate_nacc 
        q = self.div_yield_nacc

        sig2 = self.sigma * self.sigma
        mu_x = (b - q) - 0.5 * sig2

        # Operator coefficients in log space
        alpha = 0.5 * sig2 / (dx * dx)
        beta_adv = mu_x / (2.0 * dx)

        a = alpha - beta_adv                                # coeff for V_{j-1}
        c = alpha + beta_adv                                # coeff for V_{j+1}
        bcoef = -2.0 * alpha - r      # coeff for V_j

        # matrices for theta = 0.5 (CN); we will override theta=1 on the fly
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

        V = self._terminal_payoff()  # length N+1: j=0..N
        monitor_idx = self._monitor_indices_tau(dt)

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


            # boundary values at tau_next (Dirichlet)
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

            if apply_KO and (m + 1) in monitor_idx:
                self._apply_KO_projection(V,self.s_nodes, tau_next)
        return V

    def _delta_gamma_nonuniform(self, s_nodes: List[float], V: List[float]) -> Tuple[float, float]:
        """
        N=len(s_nodes)-1
        i = min(range(N+1), key=lambda k: abs(s_nodes[k]-Sx))
        i = max(1, min(N-1, i))
        h1 = s_nodes[i] - s_nodes[i-1]
        h2 = s_nodes[i+1] - s_nodes[i]
        delta_c = ( -(h2/(h1*(h1+h2)))*V[i-1] + ((h2-h1)/(h1*h2))*V[i] + (h1/(h2*(h1+h2)))*V[i+1] )
        gamma_c = 2.0*( V[i-1]/(h1*(h1+h2)) - V[i]/(h1*h2) + V[i+1]/(h2*(h1+h2)) )
        """
        N=len(s_nodes)-1
        i = min(range(N+1), key=lambda k: abs(s_nodes[k]-self.spot))
        i = max(1, min(N-1, i))
        h1 = s_nodes[i] - s_nodes[i-1]
        h2 = s_nodes[i+1] - s_nodes[i]
        delta_c = (
                -(h2 / (h1 * (h1+h2))) * V[i-1]
                + ((h2 - h1) / (h1*h2)) *V[i]
                + (h1 / (h2 * (h1+h2))) * V[i+1]
        )
        gamma_c = 2.0* (
                V[i-1] / (h1 * (h1+h2))
                - V[i] / (h1 * h2)
                + V[i+1] / (h2 * (h1+h2))
        )

        if not self.use_one_sided_greeks_near_barrier:
            return float(delta_c), float(gamma_c)

        # One-sided near KO barrier
        H=None
        side=None
        if self.barrier_type in ("down-and-out","double-out") and self.lower_barrier is not None:
            H=self.lower_barrier
            side="down"
        if self.barrier_type in ("up-and-out","double-out") and self.upper_barrier is not None and H is None:
            H=self.upper_barrier
            side="up"
        if H is None:
            return float(delta_c), float(gamma_c)

        j = max(0, min(N-1, min(range(N), key=lambda k: abs(H - s_nodes[k]))))
        near = (abs(i-j) <= self.mollify_band_nodes)
        if not near:
            return float(delta_c), float(gamma_c)

        if side=="down":
            i2 = max(1, min(N-1, j+1))
            h = s_nodes[i2+1]-s_nodes[i2]
            delta = (V[i2+1]-V[i2])/h
            gamma = 2.0*( V[i2-1]/((s_nodes[i2]-s_nodes[i2-1])*(s_nodes[i2+1]-s_nodes[i2-1]))
                          - V[i2]/((s_nodes[i2]-s_nodes[i2-1])*(s_nodes[i2+1]-s_nodes[i2]))
                          + V[i2+1]/((s_nodes[i2+1]-s_nodes[i2])*(s_nodes[i2+1]-s_nodes[i2-1])) )
        else:
            i2 = max(1, min(N-1, j))
            h = s_nodes[i2]-s_nodes[i2-1]
            delta = (V[i2]-V[i2-1])/h
            gamma = 2.0*( V[i2-1]/((s_nodes[i2]-s_nodes[i2-1])*(s_nodes[i2+1]-s_nodes[i2-1]))
                          - V[i2]/((s_nodes[i2]-s_nodes[i2-1])*(s_nodes[i2+1]-s_nodes[i2]))
                          + V[i2+1]/((s_nodes[i2+1]-s_nodes[i2])*(s_nodes[i2+1]-s_nodes[i2-1])) )

        gamma = max(min(gamma, 1e5), -1e5)

        return float(delta), float(gamma)


    def _map_KI_to_KO(self) -> Optional[str]:
        """Return the KO barrier type corresponding to KI type"""
        if self.barrier_type == "down-and-in":
            return "down-and-out"

        if self.barrier_type == "up-and-in":
            return "up-and-out"

        if self.barrier_type == "double-in":
            return "double-out"

        return None


    def _interp_price(self, V: List[float]) -> float:
        nodes = self.s_nodes
        s= nodes
        S0 = self.spot - self.pv_divs

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

    def _vanilla_black76_price(self,
                               S: Optional[float] = None,
                               sigma: Optional[float] = None,
                               T: Optional[float] = None) -> float:
        """
        Standard Black-Scholes (with continuous r and q).
        """
        S = self.spot - self.pv_divs if S is None else S - self.pv_divs
        K = self.strike
        time_to_discount = self.time_to_discount
        time_to_carry = self.time_to_carry
        time_to_expiry = self.time_to_expiry if T is None else T
        carry_rate = self.carry_rate_nacc
        discount_rate = self.discount_rate_nacc
        sigma = self.sigma if sigma is None else sigma
        q = self.div_yield_nacc
        disc_q = math.exp(-q * self.time_to_carry)
        b = self.b

        if time_to_discount <= 0 or sigma <= 0:
            # intrinsic
            if self.option_type == "call":
                price = max(S - K, 0.0)
            else:
                price = max(K - S, 0.0)

            return price

        sqrtT = math.sqrt(time_to_expiry)
        F = S * math.exp(carry_rate * time_to_carry)

        d1 = (math.log(F / K) + ( 0.5 * sigma * sigma) * time_to_expiry) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT

        Nd1 = norm.cdf(d1)
        Nd2 = norm.cdf(d2)

        if self.option_type == "call":
            price = math.exp(-discount_rate * time_to_discount) * (F * Nd1 - K * Nd2)
        else:
            Nmd1 = 1.0 - Nd1
            Nmd2 = 1.0 - Nd2
            price = math.exp(-discount_rate * time_to_discount) * (K * Nmd2 - F * Nmd1)

        return price

    def _vanilla_black76_greeks_fd(self,
                                   dS: float = 0.0001,
                                   dSigma: float = 0.0001,
                                   dT: float = 0.0001) -> Dict[str, float]:
        """
        Greeks (price, delta, gamma, theta, vega) for the vanilla option
        computed by finite differences on the Black-76 price.

        - dS is an absolute spot bump.
        - dSigma is an absolute vol bump (e.g. 0.001 = 0.1 vol point).
        - dT is an absolute time bump in YEARS.
        """

        S0 = self.spot
        sigma0 = self.sigma
        T0 = self.time_to_expiry

        dS_pert = S0 * (dS)

        # Base price
        p0 = self._vanilla_black76_price(S=S0, sigma=sigma0, T=T0)

        p_up_S = self._vanilla_black76_price(S=S0 + dS_pert, sigma=sigma0, T=T0)
        p_dn_S = self._vanilla_black76_price(S=S0 - dS_pert, sigma=sigma0, T=T0)

        delta = (p_up_S - p_dn_S) / (2.0 * dS_pert)
        gamma = (p_up_S - 2.0 * p0 + p_dn_S) / (dS_pert**2)

        p_up_v = self._vanilla_black76_price(S=S0, sigma=sigma0 + dSigma, T=T0)

        vega = (p_up_v - p0) / (100 * dSigma)


        if T0 > 2.0 * dT:
            p_up_T = self._vanilla_black76_price(S=S0, sigma=sigma0, T=T0 + dT)
            p_dn_T = self._vanilla_black76_price(S=S0, sigma=sigma0, T=T0 - dT)

            dV_dT = (p_up_T - p_dn_T) / (2.0 * dT)
            theta = -dV_dT
        else:

            p_dn_T = self._vanilla_black76_price(S=S0, sigma=sigma0, T=max(T0 - dT, 1e-8))
            dV_dT = (p0 - p_dn_T) / dT
            theta = -dV_dT

        return {
            "price": p0,
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
    }
    """
    def _delta_gamma_from_grid(self, V: List[float]) -> Tuple[float, float]:
        """
        Compute delta and gamma at S0 from the PDE grid in *spot* space.

        - Uses a non-uniform 3-point stencil.
        - Away from barriers: central difference.
        - Within `mollify_band_nodes` of a barrier and
          `use_one_sided_greeks_near_barrier=True`, switches to a
          one-sided stencil pointing *away* from the barrier.

        This is the "model delta" and "model gamma" that the FIS-style
        risk function will later use in the Taylor approximation.
        """
        s = self.s_nodes
        V_arr = V
        S0 = self.spot
        N = len(s)

        if N < 3:
            raise RuntimeError("Need at least 3 spatial nodes for Greeks.")

        # --- find interior node closest to S0 ---
        idx = min(range(1, N - 1), key=lambda i: abs(s[i] - S0))

        # --- locate barrier indices (if any) ---
        barrier_indices: List[int] = []
        if self.lower_barrier is not None:
            i_low = min(range(N), key=lambda i: abs(s[i] - self.lower_barrier))
            barrier_indices.append(i_low)
        if self.upper_barrier is not None:
            i_up = min(range(N), key=lambda i: abs(s[i] - self.upper_barrier))
            barrier_indices.append(i_up)

        # nearest barrier in index-space
        near_barrier = False
        is_lower_side = False
        if barrier_indices and self.use_one_sided_greeks_near_barrier:
            i_bar = min(barrier_indices, key=lambda j: abs(j - idx))
            dist = abs(idx - i_bar)
            if dist <= self.mollify_band_nodes:
                near_barrier = True
                # if barrier index is below idx, it's a lower barrier; otherwise upper
                is_lower_side = i_bar < idx

        # Helper functions for non-uniform finite differences
        def central_stencil(i: int) -> Tuple[float, float]:
            S_im1, S_i, S_ip1 = s[i - 1], s[i], s[i + 1]
            V_im1, V_i, V_ip1 = V_arr[i - 1], V_arr[i], V_arr[i + 1]
            h1 = S_i - S_im1
            h2 = S_ip1 - S_i
            delta_c = (
                -h2 / (h1 * (h1 + h2)) * V_im1
                + (h2 - h1) / (h1 * h2) * V_i
                + h1 / (h2 * (h1 + h2)) * V_ip1
            )
            gamma_c = 2.0 * (
                V_im1 / (h1 * (h1 + h2))
                - V_i / (h1 * h2)
                + V_ip1 / (h2 * (h1 + h2))
            )
            return float(delta_c), float(gamma_c)

        def forward_stencil(i: int) -> Tuple[float, float]:
            """
            One-sided forward stencil at node i using i, i+1, i+2.
            Appropriate when barrier is below and we want information
            from the interior (higher S).
            """
            if i + 2 >= N:
                return central_stencil(i)

            S0_, S1, S2 = s[i], s[i + 1], s[i + 2]
            V0, V1, V2 = V_arr[i], V_arr[i + 1], V_arr[i + 2]

            h1 = S1 - S0_
            h2 = S2 - S1

            # first derivative coefficients (non-uniform forward)
            a0 = (-2.0 * h1 - h2) / (h1 * h1 + h1 * h2)
            a1 = (h1 + h2) / (h1 * h2)
            a2 = -h1 / (h1 * h2 + h2 * h2)

            # second derivative coefficients (non-uniform forward)
            b0 = 2.0 / (h1 * h1 + h1 * h2)
            b1 = -2.0 / (h1 * h2)
            b2 = 2.0 / (h1 * h2 + h2 * h2)

            delta_f = a0 * V0 + a1 * V1 + a2 * V2
            gamma_f = b0 * V0 + b1 * V1 + b2 * V2
            return float(delta_f), float(gamma_f)

        def backward_stencil(i: int) -> Tuple[float, float]:
            """
            One-sided backward stencil at node i using i, i-1, i-2.
            Appropriate when barrier is above and we want information
            from the interior (lower S).
            """
            if i - 2 <= 0:
                return central_stencil(i)

            S0_, S_1, S_2 = s[i], s[i - 1], s[i - 2]
            V0, V1, V2 = V_arr[i], V_arr[i - 1], V_arr[i - 2]

            h1 = S0_ - S_1
            h2 = S_1 - S_2

            # first derivative coefficients (non-uniform backward)
            c0 = (2.0 * h1 + h2) / (h1 * h1 + h1 * h2)
            c1 = -(h1 + h2) / (h1 * h2)
            c2 = h1 / (h1 * h2 + h2 * h2)

            # second derivative coefficients (same structure as forward)
            d0 = 2.0 / (h1 * h1 + h1 * h2)
            d1 = -2.0 / (h1 * h2)
            d2 = 2.0 / (h1 * h2 + h2 * h2)

            delta_b = c0 * V0 + c1 * V1 + c2 * V2
            gamma_b = d0 * V0 + d1 * V1 + d2 * V2
            return float(delta_b), float(gamma_b)

        if near_barrier:
            # If the nearest barrier is below idx, we are near a lower barrier
            # and want a forward stencil (towards higher S). Otherwise backward.
            if is_lower_side:
                delta, gamma = forward_stencil(idx)
            else:
                delta, gamma = backward_stencil(idx)
        else:
            delta, gamma = central_stencil(idx)

        # Small safety clamp on gamma to avoid exploding values
        gamma = max(min(gamma, 1e5), -1e5)

        return float(delta), float(gamma)
    """
    
    def _pde_price_and_greeks3(self, apply_KO: bool, dv_sigma: float = 0.0001, use_richardson: bool = False) -> Dict[str, float]:
        # Simplified: use_richardson is never True in practice, so removed that code path
        V_grid = self._solve_grid(apply_KO=apply_KO)
        price_base = self._interp_price(V_grid)
        delta, gamma = self._delta_gamma_from_grid(V_grid)

        # Vega via bump
        orig_sigma = self.sigma
        self.sigma = self.sigma + dv_sigma
        V_up = self._solve_grid(apply_KO=apply_KO)
        price_up = self._interp_price(V_up)
        self.sigma = orig_sigma

        vega = (price_up - price_base) / (dv_sigma * 100)

        theta = -(
            0.5 * self.sigma * self.sigma * self.spot * self.spot * gamma
            + (self.carry_rate_nacc - self.div_yield_nacc) * self.spot * delta
            - self.discount_rate_nacc * price_base
        )

        return {"price": price_base, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


    def price_log2(self, apply_KO: bool = True, use_richardson: bool = False) -> float:
        """
        Price dispatcher:
        - vanilla (no barrier): closed-form Black–Scholes,
        - KO: CN PDE with KO projection,
        - KI: parity V_KI = V_vanilla - V_KO.
        """
        bt = self.barrier_type.lower()


        # Vanilla: Black–Scholes closed form
        if bt == "none":
            return self._vanilla_black76_price()

        # Knock-out: PDE with KO projection
        if bt in ("down-and-out", "up-and-out"):
            if self.already_hit:
                df = self.get_discount_factor(self.discount_end_date) #/ self.get_discount_factor(self.carry_start_date)
                out = self.rebate_amount * df
                return out
            else:
                return self._pde_price_and_greeks3(apply_KO=True, dv_sigma=0.0001, use_richardson=use_richardson)["price"]

        # Knock-in: parity vs vanilla and KO
        if bt in ("down-and-in", "up-and-in"):
            if self.already_in:
                return self._vanilla_black76_price()
            else:
                original_bt = bt

                P_van = self._vanilla_black76_price()

                # matching knock-out
                self.barrier_type = "down-and-out" if bt == "down-and-in" else "up-and-out"
                g_ko = self._pde_price_and_greeks3(apply_KO=True, dv_sigma=0.0001, use_richardson=use_richardson)

                self.barrier_type = original_bt
                return P_van - g_ko["price"]

        raise ValueError(f"Unsupported barrier_type: {self.barrier_type}")


    def _delta_gamma_from_grid(self, V: List[float]) -> Tuple[float, float]:
        """Non-uniform central differences for delta & gamma at S0."""
        s = self.s_nodes
        S0 = self.spot

        # choose interior node closest to S0
        idx = min(range(1, len(s) - 1), key=lambda i: abs(s[i] - S0))

        S_im1 = s[idx - 1]
        S_i =  s[idx]
        S_ip1 = s[idx + 1]

        V_im1 = V[idx - 1]
        V_i = V[idx]
        V_ip1 = V[idx + 1]

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
        return delta, gamma

    def greeks_log2(self, dv_sigma: float = 0.0001, use_richardson: bool = False) -> Dict[str, float]:
        """
        Greeks consistent with:
        - vanilla (no barrier): Black-76 price with FD Greeks,
        - KO: CN PDE engine,
        - KI: parity (Greek_KI = Greek_vanilla - Greek_KO).
        """
        bt = self.barrier_type.lower()

        # 1) Vanilla: Black-76 + FD Greeks
        if bt == "none":
            return self._vanilla_black76_greeks_fd()

        # 2) Knock-out: PDE CN engine (same as before)
        if bt in ("down-and-out", "up-and-out"):
            if self.already_hit:
                return {
            "price": 0.00,
            "delta": 0.00,
            "gamma": 0.00,
            "vega": 0.00,
            "theta": 0.00,
        }
            else:
                return self._pde_price_and_greeks3(apply_KO=True, dv_sigma=dv_sigma, use_richardson=use_richardson)

        # 3) Knock-in: parity Greek_KI = Greek_vanilla - Greek_KO
        if bt in ("down-and-in", "up-and-in"):
            if self.already_in:
                return self._vanilla_black76_greeks_fd()
            else:
                original_bt = bt

                # Vanilla Greeks (Black-76 FD)
                self.barrier_type = "none"
                g_van = self._vanilla_black76_greeks_fd()

                # Matching KO Greeks (PDE)
                self.barrier_type = "down-and-out" if bt == "down-and-in" else "up-and-out"
                g_ko = self._pde_price_and_greeks3(apply_KO=True, dv_sigma=dv_sigma, use_richardson=use_richardson)

                # Restore original barrier type
                self.barrier_type = original_bt

                return {k: g_van[k] - g_ko[k] for k in g_van.keys()}

        raise ValueError(f"Unsupported barrier_type: {self.barrier_type}")

    def print_details(self) -> None:
        p=self.price_log2(); g= self.greeks_log2() ;dx = self._build_log_grid()
        print("==== Discrete Barrier Option (CN + Rannacher) — Discrete monitors, no BGK ====")
        print(f"T (years)         : {self.time_to_expiry:.9f}   [{self.day_count}]")
        print(f"sigma / r / q     : {self.sigma:.9f} / {self.carry_rate_nacc:.9f} / {self.div_yield_nacc:.9f}")
        print(f"Barrier type      : {self.barrier_type}  (lo={self.lower_barrier}, up={self.upper_barrier})")
        print(f"Rebate (amt/hit)  : {self.rebate_amount} / {self.rebate_at_hit}")
        print(f"Status (hit/in)   : {self.already_hit} / {self.already_in}")
        print(f"Grid(S,N)         : {len(self.s_nodes)}, {self.num_time_steps}  | grid_type={self.grid_type}")
        print(f"Monitors (count)  : {len(self.monitor_times)} @ {self.monitor_times}")
        print(f"Spot/Strike       : {self.spot:.6f} / {self.strike:.6f}")
        print(f"Price             : {p:.9f}")
        print(f"Greeks            : Δ={g['delta']:.9f}, Γ={g['gamma']:.9f}, ν={g['vega']:.9f}, Θ={g['theta']:.9f}")

    # ----------------------- convergence validator ---------------------
    def validate_convergence(self,
                             N_list: List[int],
                             M_list: List[int]) -> List[Dict[str,float]]:
        """Run multiple (N,M) grids and report price/Greeks for comparison."""
        out = []
        for N in N_list:
            for M in M_list:
                # clone settings with new grid/time
                clone = DiscreteBarrierFDMPricer(
                    spot=self.spot, strike=self.strike,
                    valuation_date=self.valuation_date, maturity_date=self.maturity_date,
                    sigma=self.sigma, discount_curve=self.discount_curve_df, dividend_schedule=self.dividend_schedule,
                    option_type=self.option_type,
                    barrier_type=self.barrier_type, lower_barrier=self.lower_barrier, upper_barrier=self.upper_barrier,
                    monitor_dates=self.monitor_dates, rebate_amount=self.rebate_amount, rebate_at_hit=self.rebate_at_hit,
                    already_hit=self.already_hit, already_in=self.already_in,
                    num_space_nodes=N, num_time_steps=M, rannacher_steps=self.rannacher_steps,
                    day_count=self.day_count,
                    min_substeps_between_monitors=self.min_substeps,
                    grid_type=self.grid_type, sinh_alpha=self.sinh_alpha,
                    use_one_sided_greeks_near_barrier=self.use_one_sided_greeks_near_barrier,
                    mollify_band_nodes=self.mollify_band_nodes
                )
                price = clone.price_log()
                greeks = clone.greeks_log()
                out.append({
                    "N": N, "M": M,
                    "price": price,
                    "delta": greeks["delta"],
                    "gamma": greeks["gamma"],
                    "vega":  greeks["vega"],
                    "theta": greeks["theta"],
                })
        # Sort by size
        out.sort(key=lambda r: (r["N"], r["M"]))
        # Print a compact table
        print("\n=== Grid Convergence (sorted by N,M) ===")
        print("   N     M        Price          Delta           Gamma            Vega            Theta")
        for r in out:
            print(f"{r['N']:5d} {r['M']:5d}  {r['price']:12.8f}  {r['delta']:13.8f}  {r['gamma']:13.8f}  {r['vega']:13.8f}  {r['theta']:13.8f}")
        return out

        """
            def _pde_price_and_greeks3(
        self,
        apply_KO: bool,
        dv_sigma: float = 0.0001,
        use_richardson: bool = False,
    ) -> Dict[str, float]:
        """
        Core CN + Rannacher PDE engine.

        Returns at least {"price": price}. If
        `self.calculate_greeks_in_pde` is True, also returns
        delta, gamma, vega, theta (the "model Greeks" used
        by the FIS-style risk function).

        - Price: interpolated at S0 from the grid.
        - Delta, gamma: from the barrier-aware non-uniform stencil.
        - Vega: bump-and-revalue on sigma.
        - Theta: from the BS PDE relation at S0.
        """
        # Solve PDE once for base sigma
        V_grid = self._solve_grid(apply_KO=apply_KO)
        price_base = self._interp_price(V_grid)

        # If Greeks disabled (e.g. risk-function full reval), stop here
        if not self.calculate_greeks_in_pde:
            return {"price": float(price_base)}

        # Barrier-aware delta and gamma from grid
        delta, gamma = self._delta_gamma_from_grid(V_grid)

        # Vega via bump
        orig_sigma = self.sigma
        self.sigma = orig_sigma + dv_sigma
        V_up = self._solve_grid(apply_KO=apply_KO)
        price_up = self._interp_price(V_up)
        self.sigma = orig_sigma

        vega = (price_up - price_base) / (dv_sigma * 100.0)

        # Theta from PDE relation at S0
        theta = -(
            0.5 * orig_sigma * orig_sigma * self.spot * self.spot * gamma
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
        """
        
        """
            def risk_reprice_spot(
        self,
        shifted_spot: float,
        *,
        rel_price_shift_model: float = 0.01,
        price_domain_scale_factor: float = 1.1,
        force_full_revaluation: bool = False,
        base_price: Optional[float] = None,
        base_greeks: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        FIS-style risk function for underlying price shifts using this PDE model.

        - If only the underlying price is shifted and the shift magnitude is
          within the price domain, approximate the shifted price via:

              f̂(S*) = f(S0) + Δ_model * h + ½ Γ_model * h²

          where h = S* - S0 and (Δ_model, Γ_model) are the model Greeks from
          the base PDE run.

        - If the shift is outside the price domain or force_full_revaluation=True,
          perform a full PDE revaluation with Greeks turned off.

        Parameters
        ----------
        shifted_spot : float
            New underlying price S* for the shifted scenario.
        rel_price_shift_model : float, optional
            Relative price shift used in the model's own Greek calculation
            (e.g. 0.01 for 1%). This defines the baseline "radius" in which
            the Taylor approximation is trusted.
        price_domain_scale_factor : float, optional
            FIS priceDomainScaleFactor. Default 1.1.
        force_full_revaluation : bool, optional
            If True, always perform full PDE revaluation.
        base_price : float, optional
            Base price f(S0). If None, computed via price_log2().
        base_greeks : dict, optional
            Base Greeks, containing at least 'delta' and 'gamma'.
            If None, computed via greeks_log2().

        Returns
        -------
        dict
            {
                "result": shifted_price,
                "used_taylor_approx": bool,
                "shift_magnitude": float,
                "price_domain": float,
            }
        """
        from copy import deepcopy

        # --- Base values ---
        S0 = self.spot
        if base_price is None:
            base_price = self.price_log2()
        if base_greeks is None:
            base_greeks = self.greeks_log2()

        price_shift = shifted_spot - S0
        shift_magnitude = abs(price_shift)

        # priceDomain = scaleFactor * relPriceShiftModel * S0
        price_domain = price_domain_scale_factor * rel_price_shift_model * S0

        # Decision: use full revaluation or Taylor approx?
        outside_domain = shift_magnitude > price_domain

        if force_full_revaluation or outside_domain:
            # ---- Full revaluation branch ----
            shifted_pricer = deepcopy(self)
            shifted_pricer.spot = shifted_spot
            shifted_pricer.calculate_greeks_in_pde = False  # price-only PDE

            shifted_price = shifted_pricer.price_log2()

            return {
                "result": shifted_price,
                "used_taylor_approx": False,
                "shift_magnitude": shift_magnitude,
                "price_domain": price_domain,
            }

        # ---- Taylor approximation branch ----
        delta = base_greeks.get("delta", 0.0)
        gamma = base_greeks.get("gamma", 0.0)

        recalced_price = (
            base_price
            + delta * price_shift
            + 0.5 * gamma * price_shift * price_shift
        )

        return {
            "result": recalced_price,
            "used_taylor_approx": True,
            "shift_magnitude": shift_magnitude,
            "price_domain": price_domain,
        }
        """

    def __init__(..., 
             spot_shift_rel_for_greeks: float = 0.0,
             ...):
    ...
    self.spot_shift_rel_for_greeks = float(spot_shift_rel_for_greeks)
    
        def _interp_price_at_S(self, V: List[float], S_target: float) -> float:
        """
        Interpolate the PDE solution V(s) at an arbitrary spot S_target
        using linear interpolation in spot-space.
        """
        s = self.s_nodes
        N = len(s)
        if S_target <= s[0]:
            return float(V[0])
        if S_target >= s[-1]:
            return float(V[-1])

        # find bracket i such that s[i] <= S_target <= s[i+1]
        # (you can binary-search this later if you want)
        for i in range(N - 1):
            if s[i] <= S_target <= s[i + 1]:
                w = (S_target - s[i]) / (s[i + 1] - s[i])
                return float((1.0 - w) * V[i] + w * V[i + 1])

        return float(V[-1])  # fallback

        def _interp_price(self, V: List[float]) -> float:
        return self._interp_price_at_S(V, self.spot)

        def _delta_gamma_from_grid(
        self,
        V: List[float],
        spot_shift_rel: float | None = None,
    ) -> Tuple[float, float]:

        def _delta_gamma_from_grid(
        self,
        V: List[float],
        spot_shift_rel: float | None = None,
    ) -> Tuple[float, float]:

            s = self.s_nodes
        V_arr = V
        S0 = self.spot
        N = len(s)

        if N < 3:
            raise RuntimeError("Need at least 3 spatial nodes for Greeks.")

        # --- If a user-specified perturbation is provided, use bump-based Greeks ---
        if spot_shift_rel is not None and spot_shift_rel > 0.0:
            h = spot_shift_rel * S0
            S_down = S0 - h
            S_up = S0 + h

            V0 = self._interp_price_at_S(V_arr, S0)
            V_down = self._interp_price_at_S(V_arr, S_down)
            V_up = self._interp_price_at_S(V_arr, S_up)

            # central finite differences in *spot*
            delta = (V_up - V_down) / (S_up - S_down)
            h_eff = 0.5 * (S_up - S_down)
            gamma = (V_up - 2.0 * V0 + V_down) / (h_eff * h_eff)

            # small safety clamp on gamma
            gamma = max(min(gamma, 1e5), -1e5)
            return float(delta), float(gamma)

    from typing import List, Tuple, Optional

    def _interp_price_at_S(self, V: List[float], S_target: float) -> float:
        """
        Interpolate the PDE solution V(s) at an arbitrary spot S_target
        using linear interpolation in spot space.
        """
        s = self.s_nodes
        N = len(s)
        if S_target <= s[0]:
            return float(V[0])
        if S_target >= s[-1]:
            return float(V[-1])

        for i in range(N - 1):
            if s[i] <= S_target <= s[i + 1]:
                w = (S_target - s[i]) / (s[i + 1] - s[i])
                return float((1.0 - w) * V[i] + w * V[i + 1])

        # Fallback (should not really be hit)
        return float(V[-1])

    def _delta_gamma_from_grid(
        self,
        V: List[float],
        spot_shift_rel: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Compute delta and gamma at S0 from the PDE grid in *spot* space.

        Two modes:

        1) If spot_shift_rel > 0:
           - Use a bump-based central difference in spot:
                 h = spot_shift_rel * S0
                 Delta ~ [V(S0+h) - V(S0-h)] / (2h)
                 Gamma ~ [V(S0+h) - 2V(S0) + V(S0-h)] / h^2
           - All prices are obtained by interpolation on the existing grid.

        2) If spot_shift_rel is None or <= 0:
           - Use a non-uniform 3-point central stencil in spot.
           - Near a barrier (within mollify_band_nodes in index-space) and
             use_one_sided_greeks_near_barrier=True, shift the stencil one
             node away from the barrier to avoid straddling it.
        """
        s = self.s_nodes
        V_arr = V
        S0 = self.spot
        N = len(s)

        if N < 3:
            raise RuntimeError("Need at least 3 spatial nodes for Greeks.")

        # ------------------------------------------------------------------
        # 1) Bump-based Greeks (user-specified perturbation)
        # ------------------------------------------------------------------
        if spot_shift_rel is not None and spot_shift_rel > 0.0:
            h = spot_shift_rel * S0
            S_down = S0 - h
            S_up = S0 + h

            V0 = self._interp_price_at_S(V_arr, S0)
            V_down = self._interp_price_at_S(V_arr, S_down)
            V_up = self._interp_price_at_S(V_arr, S_up)

            delta = (V_up - V_down) / (S_up - S_down)
            h_eff = 0.5 * (S_up - S_down)
            gamma = (V_up - 2.0 * V0 + V_down) / (h_eff * h_eff)

            # Safety clamp on gamma
            gamma = max(min(gamma, 1e5), -1e5)
            return float(delta), float(gamma)

        # ------------------------------------------------------------------
        # 2) Grid-based non-uniform central stencil (with barrier awareness)
        # ------------------------------------------------------------------

        # Find interior node closest to S0
        idx = min(range(1, N - 1), key=lambda i: abs(s[i] - S0))

        # Identify barrier indices (if any)
        barrier_indices: List[int] = []
        if self.lower_barrier is not None:
            i_low = min(range(N), key=lambda i: abs(s[i] - self.lower_barrier))
            barrier_indices.append(i_low)
        if self.upper_barrier is not None:
            i_up = min(range(N), key=lambda i: abs(s[i] - self.upper_barrier))
            barrier_indices.append(i_up)

        near_barrier = False
        is_lower_side = False  # True if nearest barrier lies below idx

        if barrier_indices and self.use_one_sided_greeks_near_barrier:
            i_bar = min(barrier_indices, key=lambda j: abs(j - idx))
            dist = abs(idx - i_bar)
            if dist <= self.mollify_band_nodes:
                near_barrier = True
                is_lower_side = i_bar < idx

        # Choose stencil index:
        #  - Default: idx (central around S0)
        #  - Near lower barrier: move one node up (towards larger S)
        #  - Near upper barrier: move one node down (towards smaller S)
        i = idx
        if near_barrier:
            if is_lower_side:
                i = min(idx + 1, N - 2)
            else:
                i = max(idx - 1, 1)

        S_im1, S_i, S_ip1 = s[i - 1], s[i], s[i + 1]
        V_im1, V_i, V_ip1 = V_arr[i - 1], V_arr[i], V_arr[i + 1]

        h1 = S_i - S_im1
        h2 = S_ip1 - S_i

        # Non-uniform central finite-difference coefficients:
        #   f'(x0) ≈ a_-1 f_{-1} + a_0 f_0 + a_+1 f_{+1}
        #   f''(x0) ≈ b_-1 f_{-1} + b_0 f_0 + b_+1 f_{+1}
        # with x0 = S_i, x_{-1} = S_i - h1, x_{+1} = S_i + h2.
        a_m1 = -h2 / (h1 * (h1 + h2))
        a_0 = (h2 - h1) / (h1 * h2)
        a_p1 = h1 / (h2 * (h1 + h2))

        b_m1 = 2.0 / (h1 * (h1 + h2))
        b_0 = -2.0 / (h1 * h2)
        b_p1 = 2.0 / (h2 * (h1 + h2))

        delta = a_m1 * V_im1 + a_0 * V_i + a_p1 * V_ip1
        gamma = b_m1 * V_im1 + b_0 * V_i + b_p1 * V_ip1

        # Safety clamp on gamma
        gamma = max(min(gamma, 1e5), -1e5)

        return float(delta), float(gamma)

    def _pde_price_and_greeks3(
        self,
        apply_KO: bool,
        dv_sigma: float = 0.0001,
        use_richardson: bool = False,
    ) -> Dict[str, float]:
        """
        Core CN + Rannacher PDE engine.

        Returns at least {"price": price}. If
        self.calculate_greeks_in_pde is True, also returns
        delta, gamma, vega, theta (the "model Greeks" used
        by the risk function).
        """
        V_grid = self._solve_grid(apply_KO=apply_KO)
        price_base = self._interp_price(V_grid)

        # If Greeks disabled (e.g. risk-function full reval), stop here
        if not self.calculate_greeks_in_pde:
            return {"price": float(price_base)}

        # Choose how to compute model delta/gamma
        if self.use_local_quad_greeks:
            delta, gamma = self._delta_gamma_local_quad(V_grid, self.spot)
        elif self.spot_shift_rel_for_greeks > 0.0:
            delta, gamma = self._delta_gamma_from_grid(
                V_grid, spot_shift_rel=self.spot_shift_rel_for_greeks
            )
        else:
            delta, gamma = self._delta_gamma_from_grid(V_grid, spot_shift_rel=None)

        # Vega via bump
        orig_sigma = self.sigma
        self.sigma = orig_sigma + dv_sigma
        V_up = self._solve_grid(apply_KO=apply_KO)
        price_up = self._interp_price(V_up)
        self.sigma = orig_sigma

        vega = (price_up - price_base) / (dv_sigma * 100.0)

        # Theta from PDE relation at S0
        theta = -(
            0.5 * orig_sigma * orig_sigma * self.spot * self.spot * gamma
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

    def _delta_gamma_from_grid(
        self,
        V: List[float],
        spot_shift_rel: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Compute delta and gamma at S0 from the PDE grid in *spot* space.

        Two modes:

        1) Bump-based (spot_shift_rel > 0):
           - Use perturbations of size h = spot_shift_rel * S0.
           - Prefer central differences (S0-h, S0, S0+h).
           - If that would cross a barrier, switch to a one-sided forward
             (S0, S0+h, S0+2h) or backward (S0-2h, S0-h, S0) stencil,
             chosen so all evaluation points lie on the alive side.

        2) Grid-based (spot_shift_rel <= 0 or None):
           - Use non-uniform central 3-point stencil in spot.
           - Near a barrier (within mollify_band_nodes in index space) and
             use_one_sided_greeks_near_barrier=True, shift the stencil one
             node away from the barrier (effectively one-sided).
        """
        s = self.s_nodes
        V_arr = V
        S0 = self.spot
        N = len(s)

        if N < 3:
            raise RuntimeError("Need at least 3 spatial nodes for Greeks.")

        # ------------------------------------------------------------------
        # 1) Bump-based Greeks (user-specified perturbation size)
        # ------------------------------------------------------------------
        if spot_shift_rel is not None and spot_shift_rel > 0.0:
            h = spot_shift_rel * S0
            S_down = S0 - h
            S_up = S0 + h

            # Collect barriers that are alive-side separators
            barriers: List[float] = []
            if self.lower_barrier is not None:
                barriers.append(self.lower_barrier)
            if self.upper_barrier is not None:
                barriers.append(self.upper_barrier)

            def crosses_barrier(S_lo: float, S_hi: float) -> bool:
                for b in barriers:
                    if S_lo < b < S_hi:
                        return True
                return False

            central_crosses = crosses_barrier(min(S_down, S_up), max(S_down, S_up))

            use_forward = False
            use_backward = False

            if central_crosses:
                # If lower barrier below S0 and S_down passes it, avoid S_down → forward
                if self.lower_barrier is not None and self.lower_barrier < S0:
                    if S_down <= self.lower_barrier:
                        use_forward = True
                # If upper barrier above S0 and S_up passes it, avoid S_up → backward
                if self.upper_barrier is not None and self.upper_barrier > S0:
                    if S_up >= self.upper_barrier:
                        use_backward = True

                # If both triggered (very tight double-barrier), give up and
                # fall back to central (but you should probably reduce h)
                if use_forward and use_backward:
                    use_forward = False
                    use_backward = False
                    central_crosses = False

            # ---- One-sided forward stencil: (S0, S0+h, S0+2h) ----
            if use_forward:
                S1 = S0 + h
                S2 = S0 + 2.0 * h
                if S2 <= s[-1]:
                    V0 = self._interp_price_at_S(V_arr, S0)
                    V1 = self._interp_price_at_S(V_arr, S1)
                    V2 = self._interp_price_at_S(V_arr, S2)
                    delta = (-3.0 * V0 + 4.0 * V1 - V2) / (2.0 * h)
                    gamma = (V0 - 2.0 * V1 + V2) / (h * h)
                    gamma = max(min(gamma, 1e5), -1e5)
                    return float(delta), float(gamma)
                else:
                    # Out of grid; fall back to central
                    central_crosses = False

            # ---- One-sided backward stencil: (S0-2h, S0-h, S0) ----
            if use_backward:
                S_1 = S0 - h
                S_2 = S0 - 2.0 * h
                if S_2 >= s[0]:
                    V_2 = self._interp_price_at_S(V_arr, S_2)
                    V_1 = self._interp_price_at_S(V_arr, S_1)
                    V0 = self._interp_price_at_S(V_arr, S0)
                    delta = (3.0 * V0 - 4.0 * V_1 + V_2) / (2.0 * h)
                    gamma = (V_2 - 2.0 * V_1 + V0) / (h * h)
                    gamma = max(min(gamma, 1e5), -1e5)
                    return float(delta), float(gamma)
                else:
                    central_crosses = False

            # ---- Central stencil (S0-h, S0, S0+h) if safe ----
            V0 = self._interp_price_at_S(V_arr, S0)
            V_down = self._interp_price_at_S(V_arr, S_down)
            V_up = self._interp_price_at_S(V_arr, S_up)
            delta = (V_up - V_down) / (2.0 * h)
            gamma = (V_up - 2.0 * V0 + V_down) / (h * h)
            gamma = max(min(gamma, 1e5), -1e5)
            return float(delta), float(gamma)

        # ------------------------------------------------------------------
        # 2) Grid-based non-uniform central stencil (with barrier awareness)
        # ------------------------------------------------------------------
        idx = min(range(1, N - 1), key=lambda i: abs(s[i] - S0))

        # Identify barrier indices (if any)
        barrier_indices: List[int] = []
        if self.lower_barrier is not None:
            i_low = min(range(N), key=lambda i: abs(s[i] - self.lower_barrier))
            barrier_indices.append(i_low)
        if self.upper_barrier is not None:
            i_up = min(range(N), key=lambda i: abs(s[i] - self.upper_barrier))
            barrier_indices.append(i_up)

        near_barrier = False
        is_lower_side = False  # True if nearest barrier lies below idx

        if barrier_indices and self.use_one_sided_greeks_near_barrier:
            i_bar = min(barrier_indices, key=lambda j: abs(j - idx))
            dist = abs(idx - i_bar)
            if dist <= self.mollify_band_nodes:
                near_barrier = True
                is_lower_side = i_bar < idx

        # Choose stencil index
        i = idx
        if near_barrier:
            if is_lower_side:
                i = min(idx + 1, N - 2)
            else:
                i = max(idx - 1, 1)

        S_im1, S_i, S_ip1 = s[i - 1], s[i], s[i + 1]
        V_im1, V_i, V_ip1 = V_arr[i - 1], V_arr[i], V_arr[i + 1]

        h1 = S_i - S_im1
        h2 = S_ip1 - S_i

        # Non-uniform central finite-difference coefficients
        a_m1 = -h2 / (h1 * (h1 + h2))
        a_0 = (h2 - h1) / (h1 * h2)
        a_p1 = h1 / (h2 * (h1 + h2))

        b_m1 = 2.0 / (h1 * (h1 + h2))
        b_0 = -2.0 / (h1 * h2)
        b_p1 = 2.0 / (h2 * (h1 + h2))

        delta = a_m1 * V_im1 + a_0 * V_i + a_p1 * V_ip1
        gamma = b_m1 * V_im1 + b_0 * V_i + b_p1 * V_ip1
        gamma = max(min(gamma, 1e5), -1e5)
        return float(delta), float(gamma)

    def _delta_gamma_local_quad(self, V: List[float], S0: float) -> Tuple[float, float]:
        """
        Fit a local quadratic V(S) ≈ a0 + a1 S + a2 S^2 around S0 using
        nearby interior nodes on the alive side of the barrier, then
        return delta = dV/dS and gamma = d²V/dS² at S0.

        This is a smoothing operator on top of the raw PDE solution.
        """
        s_arr = np.array(self.s_nodes, dtype=float)
        V_arr = np.array(V, dtype=float)

        # Alive side of the barrier for up/out or down/out
        alive_mask = np.ones_like(s_arr, dtype=bool)
        if self.upper_barrier is not None:
            alive_mask &= (s_arr < self.upper_barrier - 1e-10)
        if self.lower_barrier is not None:
            alive_mask &= (s_arr > self.lower_barrier + 1e-10)

        idx_alive = np.where(alive_mask)[0]
        if len(idx_alive) < 5:
            # Fallback to raw FD if not enough points
            return self._delta_gamma_from_grid(V, spot_shift_rel=None)

        # Sort alive indices by distance to S0
        idx_sorted = idx_alive[np.argsort(np.abs(s_arr[idx_alive] - S0))]
        use_idx = idx_sorted[:5]  # use 5 nearest nodes

        S_loc = s_arr[use_idx]
        V_loc = V_arr[use_idx]

        # Design matrix for quadratic in S: [1, S, S^2]
        A = np.column_stack([np.ones_like(S_loc), S_loc, S_loc**2])
        coeffs, *_ = np.linalg.lstsq(A, V_loc, rcond=None)
        a0, a1, a2 = coeffs

        delta = a1 + 2.0 * a2 * S0
        gamma = 2.0 * a2
        gamma = max(min(gamma, 1e5), -1e5)
        return float(delta), float(gamma)

        # Numerical controls
        self.num_space_nodes = int(num_space_nodes)
        self.num_time_steps = int(num_time_steps)
        self.rannacher_steps = int(rannacher_steps)
        self.min_substeps = max(1, int(min_substeps_between_monitors))
        self.lambda_diff_target = float(lambda_diff_target)
        self.grid_type = grid_type
        self.sinh_alpha = float(sinh_alpha)
        self.s_max_mult = float(s_max_mult)
        self.restart_on_monitoring = bool(restart_on_monitoring)
        self.use_one_sided_greeks_near_barrier = bool(use_one_sided_greeks_near_barrier)
        self.mollify_final = bool(mollify_final)
        self.mollify_band_nodes = int(mollify_band_nodes)
        self.price_extrapolation = bool(price_extrapolation)
        self.day_count = day_count

        # Controls for how model Greeks are computed / smoothed
        # Relative perturbation size for spot when computing delta/gamma.
        # If > 0, uses bump-based Greeks in spot; if 0, uses pure grid stencils.
        self.spot_shift_rel_for_greeks: float = 0.01  # e.g. 1% default

        # If True, override delta/gamma with a local quadratic fit in spot
        # (Savitzky–Golay style smoothing) on the alive side of the barrier.
        self.use_local_quad_greeks: bool = False

        # Risk-function integration: base PDE runs compute Greeks, but
        # risk-function full revaluations can set this to False so that
        # only the price is computed.
        self.calculate_greeks_in_pde: bool = True

    def price_delta_gamma_with_risk(
        self,
        shifted_spot: float,
        *,
        rel_price_shift_model: float = 0.01,
        price_domain_scale_factor: float = 1.1,
    ) -> Dict[str, float]:
        """
        Convenience wrapper: for this shifted_spot, either
        - use Taylor approximation around current self.spot, or
        - do a full PDE revaluation if outside price domain.

        Returns price, delta, gamma for the scenario.
        """
        S0 = self.spot
        base_price = self.price_log2()
        base_greeks = self.greeks_log2()
        delta0 = base_greeks["delta"]
        gamma0 = base_greeks["gamma"]

        rf_out = self.risk_reprice_spot(
            shifted_spot=shifted_spot,
            rel_price_shift_model=rel_price_shift_model,
            price_domain_scale_factor=price_domain_scale_factor,
            base_price=base_price,
            base_greeks=base_greeks,
        )

        h = shifted_spot - S0

        if rf_out["used_taylor_approx"]:
            price = rf_out["result"]
            delta = delta0 + gamma0 * h
            gamma = gamma0
        else:
            # Full model run at shifted_spot
            clone = deepcopy(self)
            clone.spot = shifted_spot
            g = clone.greeks_log2()
            price = clone.price_log2()
            delta = g["delta"]
            gamma = g["gamma"]

        return {"price": price, "delta": delta, "gamma": gamma}
    
    from copy import deepcopy
from typing import Sequence, Dict, Any
from discrete_barrier_fdm_pricer import DiscreteBarrierFDMPricer


def front_arena_style_spot_curve(
    base_pricer: DiscreteBarrierFDMPricer,
    spot_grid: Sequence[float],
    *,
    rel_price_shift_model: float = 0.01,
    price_domain_scale_factor: float = 1.1,
) -> Dict[str, Any]:
    """
    Build a smooth Front-Arena-style curve of price, delta and gamma
    as the underlying approaches a barrier, using the FIS-style
    Taylor risk function to avoid noisy revaluations.

    Parameters
    ----------
    base_pricer : DiscreteBarrierFDMPricer
        Pricer set up at some central spot S0 (typically mid of spot_grid).
    spot_grid : list/array of floats
        Spots at which you want price/delta/gamma.
    rel_price_shift_model : float
        Relative spot shift used to define the price domain and (ideally)
        also used in the PDE for model Greeks (e.g. 0.01 = 1%).
    price_domain_scale_factor : float
        Extension-like factor from the FIS doc (default 1.1).

    Returns
    -------
    dict with keys "spots", "price", "delta", "gamma", "used_taylor".
    """
    # Ensure base_pricer's Greek bump matches risk-function notion
    base_pricer.spot_shift_rel_for_greeks = rel_price_shift_model

    # Base model run at S0
    S0 = base_pricer.spot
    base_price = base_pricer.price_log2()
    base_greeks = base_pricer.greeks_log2()
    delta0 = base_greeks["delta"]
    gamma0 = base_greeks["gamma"]

    prices = []
    deltas = []
    gammas = []
    used_taylor_flags = []

    for S in spot_grid:
        h = S - S0

        # Decide whether to use Taylor or full reval
        rf_out = base_pricer.risk_reprice_spot(
            shifted_spot=S,
            rel_price_shift_model=rel_price_shift_model,
            price_domain_scale_factor=price_domain_scale_factor,
            base_price=base_price,
            base_greeks=base_greeks,
        )

        prices.append(rf_out["result"])
        used_taylor_flags.append(rf_out["used_taylor_approx"])

        if rf_out["used_taylor_approx"]:
            # Within price domain: Greeks come directly from Taylor
            deltas.append(delta0 + gamma0 * h)
            gammas.append(gamma0)
        else:
            # Outside domain: do a proper model run at S
            pr = deepcopy(base_pricer)
            pr.spot = S
            # Here we want "true" model Greeks again
            pr.calculate_greeks_in_pde = True
            g = pr.greeks_log2()
            deltas.append(g["delta"])
            gammas.append(g["gamma"])

    return {
        "spots": list(spot_grid),
        "price": prices,
        "delta": deltas,
        "gamma": gammas,
        "used_taylor": used_taylor_flags,
        "S0": S0,
        "base_price": base_price,
        "base_delta": delta0,
        "base_gamma": gamma0,
    }

    def _delta_gamma_nonuniform(
        self, s_nodes: List[float], V: List[float]
    ) -> Tuple[float, float]:
        """
        FIS-style approximation of delta and gamma from a *single* grid.

        - Base: non-uniform central differences at the node closest to S0.
        - If a KO barrier is present and S0 is within a "barrier band", we
          blend central and one-sided approximations:

              Δ ≈ q * Δ_one_sided + (1 - q) * Δ_central
              Γ ≈ q * Γ_one_sided + (1 - q) * Γ_central

          where q is 1 at the grid point closest to the barrier and decays
          linearly to 0 at the edge of the band (cf. FIS "q D_s V + (1-q) D_c V").
        """
        N = len(s_nodes) - 1
        if N < 2:
            raise ValueError("Need at least 3 space nodes to compute Greeks.")

        # --- 1) Base central non-uniform stencil at S0 --------------------
        S0 = self.spot
        i0 = min(range(N + 1), key=lambda k: abs(s_nodes[k] - S0))
        # keep away from boundaries
        i0 = max(1, min(N - 1, i0))

        h1 = s_nodes[i0] - s_nodes[i0 - 1]
        h2 = s_nodes[i0 + 1] - s_nodes[i0]

        V_im1 = V[i0 - 1]
        V_i = V[i0]
        V_ip1 = V[i0 + 1]

        # central first derivative (non-uniform three-point stencil)
        delta_c = (
            -h2 / (h1 * (h1 + h2)) * V_im1
            + (h2 - h1) / (h1 * h2) * V_i
            + h1 / (h2 * (h1 + h2)) * V_ip1
        )

        # central second derivative (non-uniform three-point stencil)
        gamma_c = 2.0 * (
            V_im1 / (h1 * (h1 + h2))
            - V_i / (h1 * h2)
            + V_ip1 / (h2 * (h1 + h2))
        )

        # If we are *not* doing FIS-style barrier smoothing, just return.
        if not self.use_one_sided_greeks_near_barrier:
            return float(delta_c), float(gamma_c)

        # --- 2) Identify relevant knock-out barrier and "barrier band" ----
        H: Optional[float] = None
        side: Optional[str] = None

        # we only care about KO behaviour for smoothing
        if self.barrier_type in ("down-and-out", "double-out") and self.lower_barrier is not None:
            H = float(self.lower_barrier)
            side = "down"
        elif self.barrier_type in ("up-and-out", "double-out") and self.upper_barrier is not None:
            H = float(self.upper_barrier)
            side = "up"

        if H is None or side is None:
            # no active KO => central is fine
            return float(delta_c), float(gamma_c)

        # We want the *last* node inside the domain before an up-barrier,
        # or the *first* node inside the domain above a down-barrier.
        if side == "up":
            # nodes strictly below the barrier
            inside_indices = [k for k in range(N + 1) if s_nodes[k] < H]
            if not inside_indices:
                return float(delta_c), float(gamma_c)
            j = max(inside_indices)  # closest from below
        else:  # side == "down"
            inside_indices = [k for k in range(N + 1) if s_nodes[k] > H]
            if not inside_indices:
                return float(delta_c), float(gamma_c)
            j = min(inside_indices)  # closest from above

        # How far is our evaluation index i0 from the barrier index j?
        band = max(1, int(self.mollify_band_nodes))
        dist = abs(i0 - j)

        # Outside the "barrier band": use pure central.
        if dist > band:
            return float(delta_c), float(gamma_c)

        # Linear weight q in [0,1], as in FIS: q = 1 at closest-to-barrier
        # node, decays to 0 at the edge of the band.
        q = 1.0 - (dist / float(band))
        q = max(0.0, min(1.0, q))

        # --- 3) One-sided approximations that do *not* cross the barrier ---

        if side == "up":
            # Domain is S < H. Use backward stencil at node j (closest below).
            i1 = max(1, min(N - 1, j))

            # First-order backward for delta (safe, monotone close to barrier)
            h_b = s_nodes[i1] - s_nodes[i1 - 1]
            if h_b <= 0:
                return float(delta_c), float(gamma_c)
            delta_os = (V[i1] - V[i1 - 1]) / h_b

            # Gamma based on three *inside* points (i1-2, i1-1, i1) if possible.
            if i1 >= 2:
                S_im2 = s_nodes[i1 - 2]
                S_im1 = s_nodes[i1 - 1]
                S_i   = s_nodes[i1]
                V_im2 = V[i1 - 2]
                V_im1 = V[i1 - 1]
                V_i   = V[i1]

                h1_b = S_im1 - S_im2
                h2_b = S_i - S_im1

                gamma_os = 2.0 * (
                    V_im2 / (h1_b * (h1_b + h2_b))
                    - V_im1 / (h1_b * h2_b)
                    + V_i / (h2_b * (h1_b + h2_b))
                )
            else:
                gamma_os = gamma_c  # fallback if we don't have enough points

        else:  # side == "down"
            # Domain is S > H. Use forward stencil at node j (closest above).
            i1 = max(1, min(N - 2, j))

            h_f = s_nodes[i1 + 1] - s_nodes[i1]
            if h_f <= 0:
                return float(delta_c), float(gamma_c)
            delta_os = (V[i1 + 1] - V[i1]) / h_f

            # Gamma based on three inside points (i1, i1+1, i1+2) if possible.
            if i1 + 2 <= N:
                S_i   = s_nodes[i1]
                S_ip1 = s_nodes[i1 + 1]
                S_ip2 = s_nodes[i1 + 2]
                V_i   = V[i1]
                V_ip1 = V[i1 + 1]
                V_ip2 = V[i1 + 2]

                h1_f = S_ip1 - S_i
                h2_f = S_ip2 - S_ip1

                gamma_os = 2.0 * (
                    V_i / (h1_f * (h1_f + h2_f))
                    - V_ip1 / (h1_f * h2_f)
                    + V_ip2 / (h2_f * (h1_f + h2_f))
                )
            else:
                gamma_os = gamma_c  # fallback if not enough points

        # --- 4) Blend one-sided and central, as in FIS (q D_s + (1-q) D_c) ---

        delta = q * delta_os + (1.0 - q) * delta_c
        gamma = q * gamma_os + (1.0 - q) * gamma_c

        # Light clipping to avoid ridiculous spikes
        gamma = max(min(gamma, 1e5), -1e5)

        return float(delta), float(gamma)

    def _delta_gamma_from_grid(self, V: List[float]) -> Tuple[float, float]:
        """
        FIS-style delta/gamma from the *PDE* grid at t=0.

        Delegates to `_delta_gamma_nonuniform`, which:
        - uses a non-uniform central stencil away from the barrier,
        - switches to a one-sided stencil in the interval closest to the barrier,
        - linearly blends one-sided and central in a band of size
          `self.mollify_band_nodes` around the barrier (q-weighting).
        """
        return self._delta_gamma_nonuniform(self.s_nodes, V)

    def _snap_critical_levels_to_grid(self) -> None:
        """
        Snap spot, barriers and strike to the *closest* S-grid nodes and
        record both indices and snapped values.

        We do NOT change self.spot / self.lower_barrier / self.strike
        (they remain the 'true' economic inputs). The snapped versions
        are only used inside the PDE / Greeks logic.
        """
        s = self.s_nodes
        N = len(s)
        if N == 0:
            return

        # Effective spot in S-domain (you’re already using spot - PV(divs) elsewhere)
        S_eff = self.spot - self.pv_divs

        # --- spot ---
        self.spot_grid_index = min(range(N), key=lambda i: abs(s[i] - S_eff))
        self.spot_snapped = s[self.spot_grid_index]

        # --- lower barrier ---
        if self.snap_barriers_to_grid and self.lower_barrier is not None:
            idx = min(range(N), key=lambda i: abs(s[i] - self.lower_barrier))
            self.lower_barrier_index = idx
            self.lower_barrier_snapped = s[idx]
        else:
            self.lower_barrier_index = None
            self.lower_barrier_snapped = self.lower_barrier

        # --- upper barrier ---
        if self.snap_barriers_to_grid and self.upper_barrier is not None:
            idx = min(range(N), key=lambda i: abs(s[i] - self.upper_barrier))
            self.upper_barrier_index = idx
            self.upper_barrier_snapped = s[idx]
        else:
            self.upper_barrier_index = None
            self.upper_barrier_snapped = self.upper_barrier

        # --- strike ---
        if self.snap_strike_to_grid:
            idx = min(range(N), key=lambda i: abs(s[i] - self.strike))
            self.strike_grid_index = idx
            self.strike_snapped = s[idx]
        else:
            self.strike_grid_index = None
            self.strike_snapped = self.strike

    def _build_log_grid(self) -> float:
        self.configure_grid()
        S_min = self._S_min
        S_max = self._S_max

        x_min = math.log(S_min)
        x_max = math.log(S_max)

        n = self.num_space_nodes
        dx = (x_max - x_min) / n
        x_nodes = [x_min + i * dx for i in range(n + 1)]

        # precompute S-grid
        self.s_nodes = [math.exp(x) for x in x_nodes]

        # (you already had the log-barrier lines here; keep them if you want)
        self.lower_barrier_log = math.log(self.lower_barrier) if self.lower_barrier else None
        self.upper_barrier_log = math.log(self.upper_barrier) if self.upper_barrier else None

        # --- NEW: snap crit levels to the grid ---
        self._snap_critical_levels_to_grid()

        return dx

    def _terminal_payoff(self) -> List[float]:
        s_nodes = self.s_nodes
        # use snapped strike if enabled
        k = (
            self.strike_snapped
            if (self.snap_strike_to_grid and self.strike_snapped is not None)
            else self.strike
        )
        if self.option_type == "call":
            return [max(s - k, 0.0) for s in s_nodes]
        else:
            return [max(k - s, 0.0) for s in s_nodes]

    def _apply_KO_projection(self, V: List[float], s_nodes: List[float], tau_left: float) -> None:
        """
        Project knock-outs at a monitoring time by zeroing / rebating values
        on the KO side of the barrier. Uses snapped barrier indices if present.
        """
        if self.barrier_type in ("none", "down-and-in", "up-and-in", "double-in"):
            return

        if self.rebate_at_hit:
            rebate = self.rebate_amount
        else:
            rebate = self.rebate_amount * math.exp(-self.carry_rate_nacc * tau_left)

        n = min(len(V), len(s_nodes))

        lo_idx = self.lower_barrier_index
        up_idx = self.upper_barrier_index

        for i in range(n):
            out = False

            if self.barrier_type == "down-and-out":
                if lo_idx is not None and i <= lo_idx:
                    out = True
            elif self.barrier_type == "up-and-out":
                if up_idx is not None and i >= up_idx:
                    out = True
            elif self.barrier_type == "double-out":
                if (lo_idx is not None and i <= lo_idx) or (up_idx is not None and i >= up_idx):
                    out = True

            if out:
                V[i] = rebate

        # One-sided near KO barrier
        H=None
        side=None

        if self.barrier_type in ("down-and-out","double-out") and self.lower_barrier is not None:
            H=float(self.lower_barrier)
            side="down"
        elif self.barrier_type in ("up-and-out","double-out") and self.upper_barrier is not None:
            H=float(self.upper_barrier)
            side="up"

        if H is None or side is None:
            return float(delta_c), float(gamma_c)

        if side=="up":
            inside_indices = [k for k in range(N+1) if s_nodes[k]<H]
            ...

        # One-sided near KO barrier, using snapped indices if available
        j: Optional[int] = None
        side: Optional[str] = None

        if self.barrier_type in ("down-and-out", "double-out") and self.lower_barrier_index is not None:
            j = self.lower_barrier_index
            side = "down"
        elif self.barrier_type in ("up-and-out", "double-out") and self.upper_barrier_index is not None:
            j = self.upper_barrier_index
            side = "up"

        if j is None or side is None:
            return float(delta_c), float(gamma_c)

        band = max(1, int(self.mollify_band_nodes))
        dist = abs(i - j)

        if dist > band:
            # outside barrier band ⇒ pure central
            return float(delta_c), float(gamma_c)

        # linear weight q ∈ [0,1], q=1 at node closest to barrier, 0 at edge
        q = 1.0 - dist / float(band)
        q = max(0.0, min(1.0, q))

        # One-sided approximations staying strictly inside the alive region
        if side == "up":
            # alive side is below j ⇒ use backward stencil at/near j
            i1 = max(1, min(N - 1, j))
            h_b = s_nodes[i1] - s_nodes[i1 - 1]
            delta_os = (V[i1] - V[i1 - 1]) / h_b if h_b > 0 else delta_c

            if i1 >= 2:
                S_im2, S_im1, S_i = s_nodes[i1 - 2], s_nodes[i1 - 1], s_nodes[i1]
                V_im2, V_im1, V_i = V[i1 - 2], V[i1 - 1], V[i1]
                h1_b = S_im1 - S_im2
                h2_b = S_i - S_im1
                gamma_os = 2.0 * (
                    V_im2 / (h1_b * (h1_b + h2_b))
                    - V_im1 / (h1_b * h2_b)
                    + V_i / (h2_b * (h1_b + h2_b))
                )
            else:
                gamma_os = gamma_c
        else:
            # side == "down": alive side is above j ⇒ forward stencil
            i1 = max(1, min(N - 2, j))
            h_f = s_nodes[i1 + 1] - s_nodes[i1]
            delta_os = (V[i1 + 1] - V[i1]) / h_f if h_f > 0 else delta_c

            if i1 + 2 <= N:
                S_i, S_ip1, S_ip2 = s_nodes[i1], s_nodes[i1 + 1], s_nodes[i1 + 2]
                V_i, V_ip1, V_ip2 = V[i1], V[i1 + 1], V[i1 + 2]
                h1_f = S_ip1 - S_i
                h2_f = S_ip2 - S_ip1
                gamma_os = 2.0 * (
                    V_i / (h1_f * (h1_f + h2_f))
                    - V_ip1 / (h1_f * h2_f)
                    + V_ip2 / (h2_f * (h1_f + h2_f))
                )
            else:
                gamma_os = gamma_c

        delta = q * delta_os + (1.0 - q) * delta_c
        gamma = q * gamma_os + (1.0 - q) * gamma_c
        gamma = max(min(gamma, 1e5), -1e5)
        return float(delta), float(gamma)

