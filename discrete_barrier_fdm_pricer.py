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

    def _time_subgrid_counts(self) -> List[int]:
        # Allocate substeps per interval proportionally to length, with a minimum
        lengths = [self.monitor_times[i+1]-self.monitor_times[i] for i in range(len(self.monitor_times)-1)]
        total = sum(lengths)

        if total <= 0:
            return [self.min_substeps]*(len(lengths))

        raw = [max(self.min_substeps, int(round(self.num_time_steps * (L/total)))) for L in lengths]
        # ensure at least 1 per interval and adjust sum
        s = sum(raw)
        if s == 0:
            raw = [self.min_substeps]*len(lengths)
            s = sum(raw)
        # adjust to exactly M_target by distributing diff
        diff = self.num_time_steps - s
        i = 0
        while diff != 0 and len(raw)>0:
            j = i % len(raw)
            if diff > 0:
                raw[j] += 1
                diff -= 1
            else:
                if raw[j] > self.min_substeps:
                    raw[j] -= 1
                    diff += 1
            i += 1
        return raw

    def _build_stock_price_grid(self) -> List[float]:
        """
        Uniform price grid [0, S_max]. Snap strike and barriers to nodes for accuracy.
        """
        reference_points = [self.spot, self.strike]
        if self.lower_barrier:
            reference_points.append(self.lower_barrier)
        if self.upper_barrier:
            reference_points.append(self.upper_barrier)

        s_star = max(reference_points) if reference_points else self.spot
        s_max = self.s_max_mult * s_star * math.exp(self.sigma * math.sqrt(self.time_to_expiry))
        s_min = 0.0
        n = self.num_space_nodes

        if self.grid_type == "uniform":
            dS = (s_max - s_min) / n
            grid = [s_min + i * dS for i in range(n + 1)]
        else:
            # sinh clustering around KO barrier (if any) or around Sref
            if self.barrier_type in ("down-and-out", "double-out") and self.lower_barrier:
                Sc = self.lower_barrier
            elif self.barrier_type in ("up-and-out", "double-out") and self.upper_barrier:
                Sc = self.upper_barrier
            else:
                Sc = s_star
            a = self.sinh_alpha
            # map x as an element between [-1,1] to S via sinh
            xs = [-1.0 + 2.0* i / n for i in range(n+1)]
            span = s_max-Sc
            scale = span / max(1e-12, math.sinh(a))
            grid = [Sc + scale * math.sinh(a * x) for x in xs]
            # ensure lower bound >=0
            shift = -min(0.0, min(grid))
            if shift > 0:
                grid = [s + shift for s in grid]

        # Snap key levels to nearest grid nodes
        def snap_to_grid(value: Optional[float]) -> None:
            if value is None:
                return
            idx = min(range(len(grid)), key=lambda i: abs(grid[i] - value))
            grid[idx] = float(value)

        snap_to_grid(self.strike)
        snap_to_grid(self.lower_barrier)
        snap_to_grid(self.upper_barrier)
        return grid

    def choose_grid_parameters(
        self,
        S0: float,
        K: float,
        lower_barrier: Optional[float],
        upper_barrier: Optional[float],
        T: float,
        sigma: float,
        monitor_times: Optional[List[float]] = None,
        # tuning knobs
        L_low: float = 4.0,          # how far down from min(ref) → S_min
        L_high: float = 4.0,         # how far up from max(ref) → S_max
        min_space: int = 300,        # minimum N_space for barrier work
        points_per_sigma: int = 12,  # grid points per 1σ√T in log-space
        lambda_target: float = 0.4,  # target diffusion number in log-space
        steps_per_interval: int = 10 # min CN steps per (monitor) time interval
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
            monitor_times=self.monitor_times,
            # you can override these knobs per product if you like:
            L_low=4.0,
            L_high=4.0,
            min_space=300,
            points_per_sigma=15,
            lambda_target=0.5,
            steps_per_interval=1,
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

    def _european_payoff(self, S: float) -> float:
        if self.option_type == "call":
            return max(S - self.strike, 0.0)
        else:
            return max(self.strike - S, 0.0)

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

    def _solve_tridiagonal_system(
        self,
        lower_diag: List[float],
        main_diag: List[float],
        upper_diag: List[float],
        right_hand_side: List[float],
    ) -> List[float]:
        """Thomas algorithm specialized for tridiagonal matrices.
            -Solve A x = b where A is tridiagonal with lower/main/upper diagonals.
        """
        n = len(right_hand_side)
        modified_upper = [0.0] * n
        modified_rhs = [0.0] * n
        solution = [0.0] * n

        # Forward sweep
        pivot = main_diag[0]
        if abs(pivot) < 1e-14:
            pivot = 1e-14

        modified_upper[0] = upper_diag[0] / pivot
        modified_rhs[0] = right_hand_side[0] / pivot

        for i in range(1, n):
            pivot = main_diag[i] - lower_diag[i] * modified_upper[i - 1]

            if abs(pivot) < 1e-14:
                pivot = 1e-14
            modified_upper[i] = upper_diag[i] / pivot if i < n - 1 else 0.0
            modified_rhs[i] = (right_hand_side[i] - lower_diag[i] * modified_rhs[i - 1]) / pivot

        # Back substitution
        solution[-1] = modified_rhs[-1]
        for i in range(n - 2, -1, -1):
            solution[i] = modified_rhs[i] - modified_upper[i] * solution[i + 1]

        return solution

    def _cn_subinterval(self,
                        s_nodes: List[float],
                        V: List[float],
                        t0: float,
                        t1: float,
                        theta: float,
                        rannacher_left_steps: int,
                        m_steps: int = 1,
                        ) -> Tuple[List[float], float]:
        """March from t1 to t0 (backwards). Return new V and dt used in the **last** step of the whole march when t0==0 (for theta)."""
        self.configure_grid()
        dx = self._build_log_grid()

        r = self.discount_rate_nacc
        b = self.carry_rate_nacc - self.div_yield_nacc
        sig = self.sigma
        q = self.div_yield_nacc

        N = len(s_nodes) - 1
        m = max(1, int(m_steps))
        # choose number of steps in this sub-interval
        L = t1 - t0
        dt = L / m
        last_dt_at_zero = None

        # Assuming uniform grid here for simplicity
        dS = s_nodes[1] - s_nodes[0]

        for step in range(m):
            use_theta = 1.0 if (rannacher_left_steps > 0) else theta
            if rannacher_left_steps > 0:
                rannacher_left_steps -= 1

            tau_after = t0 + (m - step - 1) * dt  # time to maturity after this step
            tau_left = tau_after

            # Build tri-diagonal using non‑uniform coefficients (Tavella–Randall)
            sub= [0.0] * (N + 1)
            main= [0.0] * (N + 1)
            sup= [0.0] * (N + 1)
            rhs= [0.0] * (N + 1)


            # Dirichlet boundaries at after-step time
            if self.option_type=="call":
                rhs[0] = 0.0
                rhs[N] = s_nodes[-1]*math.exp(-q*tau_left) - self.strike*math.exp(-r*tau_left)
            else:
                rhs[0] = self.strike*math.exp(-r*tau_left)
                rhs[N] = 0.0

            main[0]=1.0
            main[N]=1.0


            for i in range(1,N):
                Si = s_nodes[i]
                h1 = s_nodes[i] - s_nodes[i-1]
                h2 = s_nodes[i+1] - s_nodes[i]
                A = 0.5*sig*sig*Si*Si
                mu = b

                # Canonical non‑uniform coefficients:
                aI = use_theta*dt*( A*(2.0/(h1*(h1+h2))) - mu*Si*(1.0/(2.0*h1)) )
                bI = 1.0 + use_theta*dt*( A*(2.0/(h1*h2)) + r )
                cI = use_theta*dt*( A*(2.0/(h2*(h1+h2))) + mu*Si*(1.0/(2.0*h2)) )

                aE = (1.0-use_theta)*dt*( A*(2.0/(h1*(h1+h2))) - mu*Si*(1.0/(2.0*h1)) )
                bE = 1.0 - (1.0-use_theta)*dt*( A*(2.0/(h1*h2)) + r )
                cE = (1.0-use_theta)*dt*( A*(2.0/(h2*(h1+h2))) + mu*Si*(1.0/(2.0*h2)) )

                sub[i]  = -aI
                main[i] =  bI
                sup[i]  = -cI
                rhs[i]  =  aE*V[i-1] + bE*V[i] + cE*V[i+1]
            
            V = self._solve_tridiagonal_system(sub, main, sup, rhs)

            if abs(t0) < 1e-14 and step == m-1:
                last_dt_at_zero = dt

        return V, (last_dt_at_zero if last_dt_at_zero is not None else dt)

    def solve_grid2(self, apply_KO: bool) -> list[float]:
        """
        Solve dV/dt = L V with L from Black–Scholes in S-space using
        Rannacher + Crank–Nicolson between monitoring dates.

        This keeps the external interface identical to the original
        `solve_grid`, but internally delegates to the theoretically
        consistent `_run_backward` + `_cn_subinterval` engine.

        Parameters
        ----------
        apply_KO : bool
            If True, knock-out projection is applied at monitoring dates.
            If False, the barrier is ignored and a vanilla grid is returned.

        Returns
        -------
        V : list[float]
            Option values on the spatial grid `self.s_nodes` at valuation
            time t = 0 (time-to-expiry τ = T). This grid is then used by
            the interpolation routines to obtain the price at S0.
        """
        # Ensure grid in S is configured exactly once
        self.configure_grid()
        dx = self._build_log_grid()

        # Run the backward time-march between all monitoring dates.
        V, last_dt_at_zero, monitors_applied = self._run_backward(
            s_nodes=self.s_nodes,
            apply_barrier=apply_KO,
        )

        # Optionally store diagnostics if you want them later
        self._last_dt_at_zero = last_dt_at_zero
        self._monitors_applied = monitors_applied

        return V

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
        q = self.div_yield_nacc
        mu_x = self.carry_rate_nacc - q
        """
        alpha = 0.5 * sig2
        beta  = (mu_x - 0.5 * sig2)
        
        a = alpha / (dx * dx) - beta / (2.0 * dx)
        b = -2.0 * alpha / (dx * dx) - r
        c = alpha / (dx * dx) + beta / (2.0 * dx)
        """
        """"""
        sig2 = self.sigma * self.sigma
        mu_x = (self.carry_rate_nacc - self.div_yield_nacc) - 0.5 * sig2

        
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

    def _run_backward(self, s_nodes: List[float], apply_barrier: bool) -> Tuple[List[float], float,List[float]]:
        """Return (final V at t=0, dt_last, monitors_applied_times)."""
        N = len(s_nodes) - 1
        V = self._terminal_payoff()

        # Handle status flags at t=0 before marching
        if apply_barrier:
            if self.barrier_type in ("down-and-out","up-and-out","double-out") and self.already_hit:
                # immediate rebate
                instant = self.rebate_amount if self.rebate_at_hit else self.rebate_amount*math.exp(-self.carry_rate_nacc * self.time_to_carry)
                return [instant]* (N+1), 0.0,[]
            if self.barrier_type in ("down-and-in","up-and-in","double-in") and self.already_in:
                apply_barrier = False  # vanilla from the start

        theta = 0.5
        monitors_applied = []
        # Allocate substeps per interval (approximately M_target total)
        subcounts = self._time_subgrid_counts()

        rannacher_left = self.rannacher_steps
        dt_last_global = None

        # March interval by interval: [t_{k-1}, t_k], backward in time
        for k in range(len(self.monitor_times)-1, 0, -1):
            t0 = self.monitor_times[k-1]
            t1 = self.monitor_times[k]
            m_steps = subcounts[k-1] if k-1 < len(subcounts) else 1

            V, dt_last = self._cn_subinterval(s_nodes, V, t0, t1, theta, rannacher_left, m_steps)
            rannacher_left = max(0, rannacher_left - m_steps)

            # After completing the subinterval, apply KO projection if t0 is a monitor
            if apply_barrier and (abs(t0) > 1e-14):  # do not project at t=0
                self._apply_KO_projection(V, s_nodes, tau_left=t0)
                monitors_applied.append(t0)

            if abs(t0) < 1e-14:
                dt_last_global = dt_last

        return V, (dt_last_global if dt_last_global is not None else self.time_spacing), monitors_applied


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

    def _interp_linear(self, x: float, xs: List[float], ys: List[float]) -> float:
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]

        lo, hi = 0, len(xs)-1
        while hi - lo > 1:
            mid = (lo + hi)//2
            if x < xs[mid]:
                hi = mid
            else:
                lo = mid
        w = (x - xs[lo])/(xs[hi] - xs[lo])
        return (1-w)*ys[lo] + w* ys[hi]

    def price(self) -> float:
        s_nodes = self.s_nodes

        # Pure Vanilla
        if self.barrier_type == "none":
            V, _, _ = self._run_backward(s_nodes,apply_barrier=False)
            return self._interp_linear(self.spot, s_nodes, V)

        # Knock-out directly
        if self.barrier_type in ("down-and-out", "up-and-out", "double-out"):
            V, _, _ = self._run_backward(s_nodes,apply_barrier=True)
            return self._interp_linear(self.spot, s_nodes, V)

        # Knock-in via parity
        bt = self.barrier_type
        if bt in ("down-and-in", "up-and-in"):
            if self.already_in:
                return self._vanilla_black76_price()
            else:
                original_bt = bt

                P_van = self._vanilla_black76_price()

                # matching knock-out
                self.barrier_type = "down-and-out" if bt == "down-and-in" else "up-and-out"
                g_ko = self._pde_price_and_greeks(apply_KO=True, dv_sigma=0.0001)

                self.barrier_type = original_bt
                return P_van - g_ko["price"]
        
        # Double-in or unsupported barrier type
        if bt == "double-in":
            if self.already_in:
                return self._vanilla_black76_price()
            else:
                P_van = self._vanilla_black76_price()
                self.barrier_type = "double-out"
                g_ko = self._pde_price_and_greeks(apply_KO=True, dv_sigma=0.0001)
                self.barrier_type = bt
                return P_van - g_ko["price"]
        
        raise ValueError(f"Unsupported barrier_type: {self.barrier_type}")

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

    def _pde_price_and_greeks(self, apply_KO: bool, dv_sigma: float = 0.0001) -> Dict[str, float]:
        """Use CN grid to compute price + Greeks for current barrier_type."""

        V_grid = self._solve_grid(apply_KO=apply_KO)
        price = self._interp_price(V_grid)
        delta, gamma = self._delta_gamma_from_grid(V_grid)

        S0 = self.spot
        sigma = self.sigma
        r = self.discount_rate_nacc
        b = self.carry_rate_nacc

        theta = -(
            0.5 * sigma * sigma * S0 * S0 * gamma
            + b * S0 * delta
            - r * price
        )

        # Vega via central bump in sigma (same CN engine)
        sigma_orig = sigma

        self.sigma = sigma_orig + dv_sigma
        V_up = self._solve_grid(apply_KO=apply_KO)
        p_up = self._interp_price(V_up)

        self.sigma = sigma_orig
        vega = (p_up - price) / (100 * dv_sigma)

        return {
            "price": price,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
        }

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

    def price_log(self) -> float:
        dx = self._build_log_grid()

        S_eff = self.spot

        # Vanilla
        if self.barrier_type == "none":
            V = self._vanilla_black76_price()
            return V

        # Pure KO
        if self.barrier_type in ("down-and-out", "up-and-out", "double-out"):
            V, _,_ = self._run_backward(self.s_nodes, apply_barrier=True)
            return self._interp_price(V)

        # KI via parity: V_in = V_van - V_KO
        ko_type = self._map_KI_to_KO()
        if ko_type is None:
            raise ValueError(f"Unknown KI barrier_type {self.barrier_type}")

        original_type = self.barrier_type

        # vanilla
        self.barrier_type = "none"
        V_van = self._vanilla_black76_price()

        # KO with mapped type
        self.barrier_type = ko_type
        V_ko, _,_ = self._run_backward(self.s_nodes, apply_barrier=True)
        P_ko = self._interp_price(V_ko)

        self.barrier_type = original_type

        V_in = V_van - P_ko
        return V_in

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

