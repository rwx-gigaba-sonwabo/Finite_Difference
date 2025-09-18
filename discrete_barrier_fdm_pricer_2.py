
import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

BarrierFlavor = Literal[
    "none",
    "down-and-out", "up-and-out", "double-out",
    "down-and-in",  "up-and-in",  "double-in",
]
OptionFlavor = Literal["call", "put"]


@dataclass
class Dividend:
    """Cash dividend paid on a calendar date."""
    pay_date: date
    amount: float


class DiscreteBarrierFDMPricer2:
    """
    Discrete barrier option pricer using:
      - Black–Scholes PDE solved by Crank–Nicolson with Rannacher start,
      - flat **continuous** short rate (NACC) across the whole grid,
      - PV escrow of discrete cash dividends (spot is shifted by PV(divs)),
      - FIS-style barrier handling:
          * BGK continuity correction when monitoring is 'frequent enough'
            (continuous-monitoring window with adjusted barrier levels),
          * tridiagonal row nearest the barrier replaced by a non-symmetric
            stencil (based on h_minus, h_plus),
          * Greeks (Δ, Γ) taken from the solved grid with barrier-aware stencils
            (one-sided in the first interval next to the barrier; blended in
            the second interval; central elsewhere).
    """

    # ---- constants / toggles ----
    BGK_BETA = 0.5826        # Broadie–Glasserman–Kou continuity constant
    N_LIM    = 5             # FIS limiter for "frequent monitoring" decision
    DEFAULT_DAYCOUNT = "ACT/365"

    def __init__(
        self,
        # Trade economics
        spot: float,
        strike: float,
        valuation_date: date,
        maturity_date: date,
        volatility: float,
        option_type: OptionFlavor,
        barrier_type: BarrierFlavor = "none",
        lower_barrier: Optional[float] = None,
        upper_barrier: Optional[float] = None,
        monitoring_dates: Optional[List[date]] = None,

        # Rates / dividends (flat NACC; dividend list optional)
        flat_rate_nacc: float = 0.0,                # r in BS PDE (continuous compounding)
        dividends: Optional[List[Tuple[date, float]]] = None,

        # Grids / numerics
        num_space_nodes: int = 600,                 # S grid nodes
        num_time_steps: int = 600,                  # t steps
        rannacher_steps: int = 2,                   # EB steps at start
        day_count: str = DEFAULT_DAYCOUNT,

        # Optional smoothing of terminal payoff around K (mollification)
        smooth_payoff_around_strike: bool = True,
        payoff_smoothing_half_width_nodes: int = 2,
    ):
        # --- inputs ---
        self.spot_price = float(spot)
        self.strike_price = float(strike)
        self.valuation_date = valuation_date
        self.maturity_date = maturity_date
        self.option_type = option_type
        self.barrier_type = barrier_type
        self.barrier_lower = lower_barrier
        self.barrier_upper = upper_barrier
        self.monitoring_dates = sorted(monitoring_dates or [])

        self.volatility = float(volatility)
        self.r_flat = float(flat_rate_nacc)
        self.day_count = day_count.upper()

        self.dividends = [Dividend(d, a) for (d, a) in (dividends or [])]

        self.num_space_nodes = int(num_space_nodes)
        self.num_time_steps = int(num_time_steps)
        self.rannacher_steps = int(rannacher_steps)

        self.smooth_payoff_around_strike = bool(smooth_payoff_around_strike)
        self.payoff_smoothing_half_width_nodes = int(payoff_smoothing_half_width_nodes)

        # --- time to maturity ---
        self.year_fraction = self._year_fraction
        self.tenor_years = self.year_fraction(self.valuation_date, self.maturity_date)

        # --- space grid (uniform; snap K/H to nodes) ---
        self.S_nodes = self._build_space_grid()
        self.dS = self.S_nodes[1] - self.S_nodes[0]

        # --- time grid (uniform) ---
        self.dt = self.tenor_years / self.num_time_steps

        # --- BGK decision: frequent monitoring? adjusted barriers? ---
        self.use_bgk_correction, self.bgk_lower, self.bgk_upper, \
            self.k_first_cont, self.k_last_cont = self._decide_and_adjust_for_continuous_window()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _year_fraction(self, d0: date, d1: date) -> float:
        if self.day_count in ("ACT/365", "ACT/365F", "ACT/365 FIXED"):
            return max(0, (d1 - d0).days) / 365.0
        if self.day_count in ("ACT/360",):
            return max(0, (d1 - d0).days) / 360.0
        if self.day_count in ("30/360", "30E/360"):
            y0, m0, dd0 = d0.year, d0.month, min(d0.day, 30)
            y1, m1, dd1 = d1.year, d1.month, min(d1.day, 30)
            return ((y1 - y0) * 360 + (m1 - m0) * 30 + (dd1 - dd0)) / 360.0
        return max(0, (d1 - d0).days) / 365.0

    def _pv_dividends_escrow(self) -> float:
        """PV of cash dividends discounted at flat r over [valuation, pay_date]."""
        if not self.dividends:
            return 0.0
        pv = 0.0
        for div in self.dividends:
            tau = self.year_fraction(self.valuation_date, div.pay_date)
            if tau <= 0:  # ignore already paid
                continue
            df = math.exp(-self.r_flat * tau)
            pv += div.amount * df
        return pv

    def _build_space_grid(self) -> List[float]:
        """
        Uniform S grid, oversized relative to anchors (S0, K, barriers),
        and snap those anchors exactly onto grid nodes for smoother Greeks.
        """
        anchors = [self.spot_price, self.strike_price]
        if self.barrier_lower: anchors.append(self.barrier_lower)
        if self.barrier_upper: anchors.append(self.barrier_upper)
        s_ref = max(anchors)
        s_max = 4.0 * s_ref * math.exp(self.volatility * math.sqrt(max(self.tenor_years, 1e-12)))
        s_min = 0.0

        N = max(200, int(self.num_space_nodes))
        dS = (s_max - s_min) / N
        nodes = [s_min + i * dS for i in range(N + 1)]

        def snap(x: Optional[float]):
            if x is None: return
            j = min(range(len(nodes)), key=lambda i: abs(nodes[i] - x))
            nodes[j] = float(x)

        snap(self.strike_price)
        snap(self.barrier_lower)
        snap(self.barrier_upper)
        return nodes

    # ------------------------------------------------------------------
    # BGK decision (FIS n_lim) + adjusted barriers for continuous window
    # ------------------------------------------------------------------
    def _decide_and_adjust_for_continuous_window(self):
        """
        Decide if monitoring is 'frequent enough' to approximate as continuous
        (FIS n_lim rule-of-thumb). When true, we apply BGK adjustment to the
        barrier level and project KO at *every* time step inside the window
        spanning first→last monitoring date.
        """
        if self.barrier_type == "none" or len(self.monitoring_dates) == 0:
            return (False, self.barrier_lower, self.barrier_upper, None, None)

        # Monitoring span and cadence
        first_mon = min(self.monitoring_dates)
        last_mon  = max(self.monitoring_dates)
        if last_mon <= first_mon:
            return (False, self.barrier_lower, self.barrier_upper, None, None)

        num_mon_points = len(self.monitoring_dates)
        span_years = self.year_fraction(first_mon, last_mon)
        avg_dt = span_years / max(1, num_mon_points - 1)

        # FIS-style limiter: if (sum over sub-interval equivalent steps) > n_lim * n,
        # use continuous approx. Practical rule: daily/weekly monitoring ⇒ frequent.
        frequent_enough = (num_mon_points >= self.N_LIM)

        # BGK adjusted barrier for the average interval Δt
        adj = math.exp(self.BGK_BETA * self.volatility * math.sqrt(max(avg_dt, 1e-12)))
        lo_adj = self.barrier_lower
        up_adj = self.barrier_upper
        if self.barrier_lower is not None:
            lo_adj = self.barrier_lower / adj   # down barrier shifts DOWN (harder to hit)
        if self.barrier_upper is not None:
            up_adj = self.barrier_upper * adj   # up barrier shifts UP (harder to hit)

        # Map the continuous window into time-step indices
        k0 = int(round(self.year_fraction(self.valuation_date, first_mon) / self.dt))
        k1 = int(round(self.year_fraction(self.valuation_date, last_mon)  / self.dt))
        k0 = max(0, min(self.num_time_steps, k0))
        k1 = max(0, min(self.num_time_steps, k1))

        return (frequent_enough, lo_adj, up_adj, min(k0, k1), max(k0, k1))

    # ------------------------------------------------------------------
    # Payoff & mollification
    # ------------------------------------------------------------------
    def _terminal_payoff_scalar(self, S: float) -> float:
        if self.option_type == "call":
            return max(S - self.strike_price, 0.0)
        return max(self.strike_price - S, 0.0)

    def _terminal_payoff_array(self, s_nodes: List[float]) -> List[float]:
        V = [self._terminal_payoff_scalar(S) for S in s_nodes]
        if not self.smooth_payoff_around_strike or self.payoff_smoothing_half_width_nodes <= 0:
            return V

        m = self.payoff_smoothing_half_width_nodes
        k_star = min(range(len(s_nodes)), key=lambda i: abs(s_nodes[i] - self.strike_price))
        i0, i1 = max(0, k_star - m), min(len(s_nodes) - 1, k_star + m)
        S0, V0 = s_nodes[i0], V[i0]
        S1, V1 = s_nodes[i1], V[i1]
        a = (V1 - V0) / ((S1 - S0) ** 2) if S1 != S0 else 0.0
        for i in range(i0, i1 + 1):
            s = s_nodes[i]
            V[i] = a * (s - S0) ** 2 + V0
        return V

    # ------------------------------------------------------------------
    # Barrier projection (knock-out) at a given time layer
    # ------------------------------------------------------------------
    def _apply_knockout_projection(self, values: List[float],
                                   lo_bar: Optional[float], up_bar: Optional[float],
                                   s_nodes: List[float]) -> None:
        for i, s in enumerate(s_nodes):
            hit = False
            if self.barrier_type == "down-and-out" and lo_bar is not None and s <= lo_bar: hit = True
            elif self.barrier_type == "up-and-out" and up_bar is not None and s >= up_bar: hit = True
            elif self.barrier_type == "double-out":
                if (lo_bar is not None and s <= lo_bar) or (up_bar is not None and s >= up_bar):
                    hit = True
            if hit: values[i] = 0.0

    # ------------------------------------------------------------------
    # Tridiagonal solver (Thomas algorithm)
    # ------------------------------------------------------------------
    @staticmethod
    def _solve_tridiagonal(a_sub: List[float], a_diag: List[float],
                           a_sup: List[float], rhs: List[float]) -> List[float]:
        n = len(rhs)
        c_star = [0.0] * n
        d_star = [0.0] * n
        x = [0.0] * n

        beta = a_diag[0]
        if abs(beta) < 1e-14:
            raise ZeroDivisionError("Tridiagonal pivot ~ 0 at row 0")
        c_star[0] = a_sup[0] / beta
        d_star[0] = rhs[0] / beta

        for i in range(1, n):
            beta = a_diag[i] - a_sub[i] * c_star[i - 1]
            if abs(beta) < 1e-14:
                raise ZeroDivisionError(f"Tridiagonal pivot ~ 0 at row {i}")
            c_star[i] = (a_sup[i] / beta) if i < n - 1 else 0.0
            d_star[i] = (rhs[i] - a_sub[i] * d_star[i - 1]) / beta

        x[-1] = d_star[-1]
        for i in range(n - 2, -1, -1):
            x[i] = d_star[i] - c_star[i] * x[i + 1]
        return x

    # ------------------------------------------------------------------
    # Helpers for barrier-aware stencils
    # ------------------------------------------------------------------
    def _effective_barriers_for_pricing(self) -> Tuple[Optional[float], Optional[float]]:
        if self.use_bgk_correction:
            return self.bgk_lower, self.bgk_upper
        return self.barrier_lower, self.barrier_upper

    def _locate_barrier_interval(self, s_nodes: List[float],
                                 lo_bar: Optional[float], up_bar: Optional[float]):
        """
        Find the barrier interval [s_j, s_{j+1}] that contains the active KO barrier.
        Return (side, j, h_minus, h_plus, downwind_index)
        side in {"down","up",None}
        """
        N = len(s_nodes) - 1

        if self.barrier_type in ("down-and-out", "double-out") and lo_bar is not None:
            H = lo_bar
            if H <= s_nodes[0]:    return ("down", 0, 1e-12, s_nodes[1]-s_nodes[0], 1)
            if H >= s_nodes[-1]:   return ("down", N-1, s_nodes[N-1]-s_nodes[N-2], 1e-12, N-2)
            j = max(0, min(N-1, next(k for k in range(N) if s_nodes[k] <= H <= s_nodes[k+1])))
            return ("down", j, max(1e-12, H - s_nodes[j]), max(1e-12, s_nodes[j+1] - H), j+1)

        if self.barrier_type in ("up-and-out", "double-out") and up_bar is not None:
            H = up_bar
            if H <= s_nodes[0]:    return ("up", 0, 1e-12, s_nodes[1]-s_nodes[0], 1)
            if H >= s_nodes[-1]:   return ("up", N-1, s_nodes[N-1]-s_nodes[N-2], 1e-12, N-2)
            j = max(0, min(N-1, next(k for k in range(N) if s_nodes[k] <= H <= s_nodes[k+1])))
            return ("up", j, max(1e-12, H - s_nodes[j]), max(1e-12, s_nodes[j+1] - H), j-1)

        return (None, None, None, None, None)

    # ------------------------------------------------------------------
    # Backward time-stepping (CN + Rannacher) with non-symmetric row at barrier
    # ------------------------------------------------------------------
    def _solve_pde_backward(self,
                            lo_bar: Optional[float],
                            up_bar: Optional[float],
                            monitor_step_index: Dict[int, bool],
                            s_nodes: List[float]) -> List[float]:
        N = len(s_nodes) - 1
        M = self.num_time_steps
        dt = self.dt
        r = self.r_flat
        sig = self.volatility
        dS = s_nodes[1] - s_nodes[0]

        V = self._terminal_payoff_array(s_nodes)

        for m in range(M, 0, -1):
            # Rannacher parameter: θ=1 (EB) for first few steps, else CN(θ=1/2)
            theta = 1.0 if (M - m) < self.rannacher_steps else 0.5

            sub = [0.0] * (N + 1)
            main = [0.0] * (N + 1)
            sup = [0.0] * (N + 1)
            rhs =  [0.0] * (N + 1)

            # boundary nodes (Dirichlet)
            if self.option_type == "call":
                rhs[0] = 0.0
                rhs[N] = s_nodes[-1] - self.strike_price * math.exp(-r * (self.tenor_years - (m - 1) * dt))
            else:
                rhs[0] = self.strike_price * math.exp(-r * (self.tenor_years - (m - 1) * dt))
                rhs[N] = 0.0
            main[0] = 1.0
            main[N] = 1.0

            # Which row is nearest to the active barrier?
            side, j_bar, h_minus, h_plus, _ = self._locate_barrier_interval(s_nodes, lo_bar, up_bar)

            for i in range(1, N):
                S = s_nodes[i]
                sig2S2 = (sig * S) ** 2

                if side is None or i not in (j_bar, j_bar + 1):
                    # ---- standard interior row (central) ----
                    a_impl = 0.5 * dt * theta * (sig2S2 / (dS ** 2) - r * S / dS)
                    b_impl = 1.0 + dt * theta * (sig2S2 / (dS ** 2) + r)
                    c_impl = 0.5 * dt * theta * (sig2S2 / (dS ** 2) + r * S / dS)

                    a_expl = -0.5 * dt * (1 - theta) * (sig2S2 / (dS ** 2) - r * S / dS)
                    b_expl =  1.0 - dt * (1 - theta) * (sig2S2 / (dS ** 2) + r)
                    c_expl = -0.5 * dt * (1 - theta) * (sig2S2 / (dS ** 2) + r * S / dS)

                    sub[i]  = -a_impl
                    main[i] =  b_impl
                    sup[i]  = -c_impl
                    rhs[i]  =  a_expl * V[i - 1] + b_expl * V[i] + c_expl * V[i + 1]

                else:
                    # ---- non-symmetric row closest to barrier (FIS §4.2) ----
                    # coefficients for first derivative u_S ≈ a*u_{i+1} + b*u_i + c*u_{i-1}
                    # and second derivative u_SS ≈ d*u_{i+1} + e*u_i + f*u_{i-1}
                    # with step sizes h_minus (to the left) and h_plus (to the right).
                    hm = float(h_minus)
                    hp = float(h_plus)

                    a1 =  hp / (hm * (hm + hp))
                    b1 = (hp - hm) / (hm * hp)
                    c1 = -hm / (hp * (hm + hp))

                    d2 =  2.0 / (hm * (hm + hp))
                    e2 = -2.0 / (hm * hp)
                    f2 =  2.0 / (hp * (hm + hp))

                    # Build the CN/EB row using these derivative approximations.
                    L_left   = 0.5 * sig2S2 * f2   + r * S * c1
                    L_center = 0.5 * sig2S2 * e2   + r * S * b1 - r
                    L_right  = 0.5 * sig2S2 * d2   + r * S * a1

                    a_impl = -theta * dt * L_left
                    b_impl =  1.0 - theta * dt * L_center
                    c_impl = -theta * dt * L_right

                    a_expl = (1 - theta) * dt * L_left
                    b_expl =  1.0 + (1 - theta) * dt * L_center
                    c_expl = (1 - theta) * dt * L_right

                    sub[i]  = -a_impl
                    main[i] =  b_impl
                    sup[i]  = -c_impl
                    rhs[i]  =  a_expl * V[i - 1] + b_expl * V[i] + c_expl * V[i + 1]

            # Solve tridiagonal system
            V = self._solve_tridiagonal(sub, main, sup, rhs)

            # Apply KO projection at monitoring times
            step_index_after = (m - 1)
            if step_index_after in monitor_step_index:
                self._apply_knockout_projection(V, lo_bar, up_bar, s_nodes)
                # Per FIS note: do NOT restart Rannacher because of barrier projection

        return V

    # ------------------------------------------------------------------
    # Linear interpolation in S
    # ------------------------------------------------------------------
    @staticmethod
    def _interp_linear(x: float, xs: List[float], ys: List[float]) -> float:
        if x <= xs[0]:  return float(ys[0])
        if x >= xs[-1]: return float(ys[-1])
        lo, hi = 0, len(xs) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if x < xs[mid]: hi = mid
            else:           lo = mid
        x0, x1 = xs[lo], xs[hi]
        y0, y1 = ys[lo], ys[hi]
        w = (x - x0) / (x1 - x0)
        return float((1 - w) * y0 + w * y1)

    # ------------------------------------------------------------------
    # Public API: price and Greeks
    # ------------------------------------------------------------------
    def _monitoring_map(self) -> Dict[int, bool]:
        """Return {time_step_index: True} where KO projection must be applied."""
        mp: Dict[int, bool] = {}
        if self.use_bgk_correction:
            for k in range(self.k_first_cont, self.k_last_cont + 1):
                mp[k] = True
        else:
            for d in self.monitoring_dates:
                if self.valuation_date < d <= self.maturity_date:
                    k = int(round(self.year_fraction(self.valuation_date, d) / self.dt))
                    mp[k] = True
        return mp

    def _solve_grid_for_current_barrier(self) -> Tuple[List[float], List[float], float]:
        """Solve PDE once on the PV-dividend-shifted grid and return (S_grid_shifted, V_grid, S_eff)."""
        # PV escrow shift
        pv_divs = self._pv_dividends_escrow()
        S_eff = self.spot_price - pv_divs
        S_shifted = [max(s - pv_divs, 0.0) for s in self.S_nodes]

        lo_eff, up_eff = self._effective_barriers_for_pricing()
        monitor_map = self._monitoring_map()

        V_grid = self._solve_pde_backward(lo_eff, up_eff, monitor_map, S_shifted)

        # KO-in via parity: price(KI) = price(Vanilla) - price(KO)
        if self.barrier_type in ("down-and-in", "up-and-in", "double-in"):
            V_vanilla = self._solve_pde_backward(None, None, {}, S_shifted)
            V_grid = [V_vanilla[i] - V_grid[i] for i in range(len(V_grid))]

        return S_shifted, V_grid, S_eff

    def price(self) -> float:
        S_shifted, V_grid, S_eff = self._solve_grid_for_current_barrier()
        return self._interp_linear(S_eff, S_shifted, V_grid)

    # ----- Greeks (FIS §4.3): Δ/Γ from grid stencils; vega by bump -----
    def _delta_gamma_from_grid(self, s_nodes: List[float], V: List[float],
                               S_eff: float, lo_bar: Optional[float], up_bar: Optional[float]) -> Tuple[float, float]:
        N = len(s_nodes) - 1
        dS = s_nodes[1] - s_nodes[0]

        # spot index on shifted grid (keep inside interior)
        iS = max(1, min(N - 1, min(range(N), key=lambda k: abs(S_eff - s_nodes[k]))))

        # default central
        delta_c = (V[iS + 1] - V[iS - 1]) / (2.0 * dS)
        gamma_c = (V[iS + 1] - 2.0 * V[iS] + V[iS - 1]) / (dS * dS)

        # barrier proximity
        side, j_bar, h_minus, h_plus, j_down = self._locate_barrier_interval(s_nodes, lo_bar, up_bar)
        if side is None or j_bar is None:
            return float(delta_c), float(gamma_c)

        # first interval next to barrier
        in_first = (iS == j_bar or iS == j_bar + 1)
        in_second = (iS == j_bar - 1 or iS == j_bar + 2)

        if in_first:
            # choose interior node closest to barrier:
            if side == "down":
                i = j_bar + 1; i_left = i - 1; i_dw = min(N, i + 1)
            else:
                i = j_bar;     i_left = i + 1; i_dw = max(0, i - 1)

            delta_os = ((1.5 * V[i]) - (2.0 * V[i_left]) + (0.5 * V[i_dw])) / dS
            gamma_os = (V[i + 1] - 2.0 * V[i] + V[i - 1]) / (dS * dS)
            return float(delta_os), float(gamma_os)

        if in_second:
            if side == "down":
                V_i, V_im1, V_dw = V[iS], V[iS - 1], V[min(N, iS + 1)]
            else:
                V_i, V_im1, V_dw = V[iS], V[iS + 1], V[max(0, iS - 1)]
            delta_os = ((1.5 * V_i) - (2.0 * V_im1) + (0.5 * V_dw)) / dS
            gamma_os = (V[iS + 1] - 2.0 * V[iS] + V[iS - 1]) / (dS * dS)
            alpha = 0.5
            delta = alpha * delta_os + (1 - alpha) * delta_c
            gamma = alpha * gamma_os + (1 - alpha) * gamma_c
            return float(delta), float(gamma)

        return float(delta_c), float(gamma_c)

    def greeks(self, vega_bump: float = 0.01) -> Dict[str, float]:
        # Solve once and read Δ/Γ from grid (barrier-aware)
        lo_eff, up_eff = self._effective_barriers_for_pricing()
        S_shifted, V_grid, S_eff = self._solve_grid_for_current_barrier()
        delta, gamma = self._delta_gamma_from_grid(S_shifted, V_grid, S_eff, lo_eff, up_eff)

        # Vega: symmetric bump in σ (FIS also uses numerical difference for vega)
        sig0 = self.volatility
        self.volatility = sig0 + vega_bump; upv = self.price()
        self.volatility = sig0 - vega_bump; dnv = self.price()
        self.volatility = sig0
        vega = (upv - dnv) / (2.0 * vega_bump)

        return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega)}

    # ------------------------------------------------------------------
    # Pretty print (all key details)
    # ------------------------------------------------------------------
    def print_details(self) -> None:
        lo_eff, up_eff = self._effective_barriers_for_pricing()
        price = self.price()
        g = self.greeks()

        print("==== Discrete Barrier Option ====")
        print(f"Maturity Date         : {self.maturity_date.isoformat()}")
        print(f"T (years)             : {self.tenor_years:.9f} ({self.day_count})")
        print(f"Volatility (sigma)    : {self.volatility:.9f}")
        print(f"Flat r (cont)         : {self.r_flat:.9f}")
        print(f"PV(dividends)         : {self._pv_dividends_escrow():.9f}")
        print("")
        print(f"Barrier type          : {self.barrier_type}")
        print(f"KO lower / upper      : {self.barrier_lower} / {self.barrier_upper}")
        print(f"BGK lower / upper     : {self.bgk_lower} / {self.bgk_upper}")
        print(f"BGK window steps      : {self.k_first_cont} .. {self.k_last_cont} (use_bgk={self.use_bgk_correction})")
        print("")
        print(f"Grid (space,time)     : {self.num_space_nodes}, {self.num_time_steps} (Rannacher {self.rannacher_steps})")
        print("")
        print(f"Spot / Strike         : {self.spot_price:.6f} / {self.strike_price:.6f}")
        print(f"Effective barriers    : {lo_eff} / {up_eff}")
        print("")
        print(f"Price                 : {price:.9f}")
        print(f"Greeks                : Δ={g['delta']:.9f}, Γ={g['gamma']:.9f}, Vega={g['vega']:.9f}")
