# -*- coding: utf-8 -*-
import math
from datetime import date
from typing import Dict, List, Literal, Optional, Tuple

import pandas as pd

BarrierType = Literal[
    "none",
    "down-and-out", "up-and-out", "double-out",
    "down-and-in",  "up-and-in",  "double-in",
]
OptionType = Literal["call", "put"]


class DiscreteBarrierFDMPricer2:
    """
    Discrete barrier option (European) using a Black–Scholes PDE solved by
    Crank–Nicolson with a Rannacher start. The implementation follows the
    FIS barrier treatment:

      * Frequent-monitoring decision via the n_lim rule (FIS):
            build n_m per monitoring interval; if sum(n_m) > n_lim * N_t,
            use a continuous-monitoring approximation with BGK barrier shift.

      * BGK shift in 'continuous window':
            B_adj = B * exp( ± 0.5826 * sigma * a_b ),
            with a_b = t_b / n_mon (t_b = time from first to last monitoring date).

      * Non-symmetric stencil on the row closest to the barrier (h_-, h_+)
        to stabilize the PDE and smooth Greeks near the barrier.

      * Greeks:
            - Delta: one-sided in the first interval next to barrier,
              blended (α=0.5) in the second interval, central elsewhere.
            - Gamma: central except in the first interval; there, blend
              non-symmetric second derivative with PDE-limit gamma
              Γ_lim = 2 ( r V - g S Δ ) / (σ^2 S^2)  (θ=0 on the barrier).

    Rates/dividends:
      * Flat continuous-compounding short rate r across the grid (NACC).
      * Discrete cash dividends are PV-escrowed: the S-grid is shifted by PV(divs).
    """

    # ---- constants / toggles ----
    BGK_BETA = 0.5826          # Broadie–Glasserman–Kou constant used by FIS
    N_LIM = 5                  # FIS limiter for frequent monitoring decision
    MIN_INTERVAL_STEPS = 1     # n_min in FIS decision
    DEFAULT_DAYCOUNT = "ACT/365"

    def __init__(
        self,
        # economics
        spot: float,
        strike: float,
        valuation_date: date,
        maturity_date: date,
        volatility: float,
        option_type: OptionType,
        barrier_type: BarrierType = "none",
        lower_barrier: Optional[float] = None,
        upper_barrier: Optional[float] = None,
        monitoring_dates: Optional[List[date]] = None,

        # rates / dividends
        flat_rate_nacc: float = 0.0,                    # continuous compounding
        dividends: Optional[List[Tuple[date, float]]] = None,

        # numerics
        num_space_nodes: int = 600,
        num_time_steps: int = 600,
        rannacher_steps: int = 2,
        day_count: str = DEFAULT_DAYCOUNT,

        # payoff smoothing near K
        smooth_payoff_around_strike: bool = True,
        payoff_smoothing_half_width_nodes: int = 2,
    ):
        # inputs
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

        self.dividends = [(d, float(a)) for (d, a) in (dividends or [])]

        self.num_space_nodes = int(num_space_nodes)
        self.num_time_steps = int(num_time_steps)
        self.rannacher_steps = int(rannacher_steps)

        self.smooth_payoff_around_strike = bool(smooth_payoff_around_strike)
        self.payoff_smoothing_half_width_nodes = int(payoff_smoothing_half_width_nodes)

        # time
        self.year_fraction = self._year_fraction
        self.tenor_years = self.year_fraction(self.valuation_date, self.maturity_date)
        self.dt = self.tenor_years / max(1, self.num_time_steps)

        # space grid
        self.S_nodes = self._build_space_grid()
        self.dS = self.S_nodes[1] - self.S_nodes[0]

        # decision + BGK-adjusted barriers and continuous window indices
        d = self._decide_and_adjust_for_continuous_window()
        (self.use_bgk_correction,
         self.bgk_lower, self.bgk_upper,
         self.k_first_cont, self.k_last_cont) = d

    # ------------------------------------------------------------------
    # Time / cash utilities
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
        """PV of discrete cash dividends from valuation to each pay date at flat r."""
        if not self.dividends:
            return 0.0
        pv = 0.0
        for (pay_date, amount) in self.dividends:
            tau = self.year_fraction(self.valuation_date, pay_date)
            if tau > 0:
                pv += amount * math.exp(-self.r_flat * tau)
        return pv

    # ------------------------------------------------------------------
    # Space grid (uniform, with snapping of K and barriers to nodes)
    # ------------------------------------------------------------------
    def _build_space_grid(self) -> List[float]:
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
            if x is None:
                return
            j = min(range(len(nodes)), key=lambda i: abs(nodes[i] - x))
            nodes[j] = float(x)

        snap(self.strike_price)
        snap(self.barrier_lower)
        snap(self.barrier_upper)
        return nodes

    # ------------------------------------------------------------------
    # FIS n_lim decision + BGK adjustment
    # ------------------------------------------------------------------
    def _decide_and_adjust_for_continuous_window(self):
        """
        FIS decision:
          1) Build an equidistant Δt = T / N_t (N_t = num_time_steps).
          2) For each interval between monitoring points, compute
                n_m = max(n_min, t_interval / Δt).
          3) Sum N̂ = Σ n_m.
          4) If N̂ > n_lim * N_t  =>  'frequent enough' (use continuous approx).
             Replace the discrete barrier with a barrier monitored *each step*
             from first to last monitoring date and apply BGK shift.
        """
        if self.barrier_type == "none" or len(self.monitoring_dates) == 0:
            return (False, self.barrier_lower, self.barrier_upper, None, None)

        first_mon = min(self.monitoring_dates)
        last_mon = max(self.monitoring_dates)
        if last_mon <= first_mon:
            return (False, self.barrier_lower, self.barrier_upper, None, None)

        # build intervals (valuation->first, between dates, last->maturity) but
        # for the FIS rule we only use the intervals between monitoring dates
        sorted_mons = [d for d in self.monitoring_dates if self.valuation_date < d <= self.maturity_date]
        if len(sorted_mons) == 0:
            return (False, self.barrier_lower, self.barrier_upper, None, None)

        # equidistant Δt and interval steps n_m
        dt_uniform = self.tenor_years / max(1, self.num_time_steps)
        intervals = []
        for i in range(1, len(sorted_mons)):
            intervals.append(self.year_fraction(sorted_mons[i - 1], sorted_mons[i]))
        N_hat = sum(max(self.MIN_INTERVAL_STEPS, int(round(ti / dt_uniform))) for ti in intervals)

        frequent_enough = (N_hat > self.N_LIM * self.num_time_steps)

        # BGK shift (FIS form): φ = β σ a_b, where a_b = t_b / n_mon
        num_mon = len(sorted_mons)
        t_b = self.year_fraction(first_mon, last_mon)
        a_b = (t_b / max(1, num_mon))
        phi = self.BGK_BETA * self.volatility * a_b
        adj = math.exp(phi)

        lo_adj = self.barrier_lower
        up_adj = self.barrier_upper
        if self.barrier_lower is not None:
            lo_adj = self.barrier_lower / adj  # down barrier moves down
        if self.barrier_upper is not None:
            up_adj = self.barrier_upper * adj  # up barrier moves up

        # map the continuous window into step indices
        k0 = int(round(self.year_fraction(self.valuation_date, first_mon) / self.dt))
        k1 = int(round(self.year_fraction(self.valuation_date, last_mon) / self.dt))
        k0 = max(0, min(self.num_time_steps, k0))
        k1 = max(0, min(self.num_time_steps, k1))

        return (frequent_enough, lo_adj, up_adj, min(k0, k1), max(k0, k1))

    # ------------------------------------------------------------------
    # Terminal payoff (with optional local smoothing around strike)
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
            V[i] = a * (s_nodes[i] - S0) ** 2 + V0
        return V

    # ------------------------------------------------------------------
    # Barrier KO projection on a time layer
    # ------------------------------------------------------------------
    def _apply_knockout_projection(self, values: List[float],
                                   lo_bar: Optional[float],
                                   up_bar: Optional[float],
                                   s_nodes: List[float]) -> None:
        for i, s in enumerate(s_nodes):
            ko = False
            if self.barrier_type == "down-and-out" and lo_bar is not None and s <= lo_bar:
                ko = True
            elif self.barrier_type == "up-and-out" and up_bar is not None and s >= up_bar:
                ko = True
            elif self.barrier_type == "double-out":
                if (lo_bar is not None and s <= lo_bar) or (up_bar is not None and s >= up_bar):
                    ko = True
            if ko:
                values[i] = 0.0

    # ------------------------------------------------------------------
    # Tridiagonal solver
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
    # Barrier location & non-symmetric stencil helpers
    # ------------------------------------------------------------------
    def _effective_barriers_for_pricing(self) -> Tuple[Optional[float], Optional[float]]:
        if self.use_bgk_correction:
            return self.bgk_lower, self.bgk_upper
        return self.barrier_lower, self.barrier_upper

    def _locate_barrier_interval(self, s_nodes: List[float],
                                 lo_bar: Optional[float],
                                 up_bar: Optional[float]):
        """
        Return (side, j, h_minus, h_plus) for the active KO barrier:
          side ∈ {"down","up",None}, j is left index of [s_j, s_{j+1}] containing the barrier,
          h_minus = H - s_j, h_plus = s_{j+1} - H.
        """
        N = len(s_nodes) - 1

        if self.barrier_type in ("down-and-out", "double-out") and lo_bar is not None:
            H = lo_bar
            if H <= s_nodes[0]:  return ("down", 0, 1e-12, s_nodes[1]-s_nodes[0])
            if H >= s_nodes[-1]: return ("down", N-1, s_nodes[N-1]-s_nodes[N-2], 1e-12)
            j = max(0, min(N - 1, next(k for k in range(N) if s_nodes[k] <= H <= s_nodes[k + 1])))
            return ("down", j, max(1e-12, H - s_nodes[j]), max(1e-12, s_nodes[j + 1] - H))

        if self.barrier_type in ("up-and-out", "double-out") and up_bar is not None:
            H = up_bar
            if H <= s_nodes[0]:  return ("up", 0, 1e-12, s_nodes[1]-s_nodes[0])
            if H >= s_nodes[-1]: return ("up", N-1, s_nodes[N-1]-s_nodes[N-2], 1e-12)
            j = max(0, min(N - 1, next(k for k in range(N) if s_nodes[k] <= H <= s_nodes[k + 1])))
            return ("up", j, max(1e-12, H - s_nodes[j]), max(1e-12, s_nodes[j + 1] - H))

        return (None, None, None, None)

    # ------------------------------------------------------------------
    # Backward CN + Rannacher, with non-symmetric row at barrier
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
            # Rannacher: θ=1 (EB) for first few steps, then θ=1/2 (CN)
            theta = 1.0 if (M - m) < self.rannacher_steps else 0.5

            sub = [0.0] * (N + 1)
            main = [0.0] * (N + 1)
            sup = [0.0] * (N + 1)
            rhs = [0.0] * (N + 1)

            # Dirichlet boundaries
            tau_left = self.tenor_years - (m - 1) * dt
            if self.option_type == "call":
                rhs[0] = 0.0
                rhs[N] = s_nodes[-1] - self.strike_price * math.exp(-r * tau_left)
            else:
                rhs[0] = self.strike_price * math.exp(-r * tau_left)
                rhs[N] = 0.0
            main[0] = 1.0
            main[N] = 1.0

            # barrier row info (if any)
            side, j_bar, h_minus, h_plus = self._locate_barrier_interval(s_nodes, lo_bar, up_bar)

            for i in range(1, N):
                S = s_nodes[i]
                sig2S2 = (sig * S) ** 2

                if side is None or i not in (j_bar, j_bar + 1):
                    # standard central row
                    a_impl = 0.5 * dt * theta * (sig2S2 / (dS ** 2) - r * S / dS)
                    b_impl = 1.0 + dt * theta * (sig2S2 / (dS ** 2) + r)
                    c_impl = 0.5 * dt * theta * (sig2S2 / (dS ** 2) + r * S / dS)

                    a_expl = -0.5 * dt * (1 - theta) * (sig2S2 / (dS ** 2) - r * S / dS)
                    b_expl = 1.0 - dt * (1 - theta) * (sig2S2 / (dS ** 2) + r)
                    c_expl = -0.5 * dt * (1 - theta) * (sig2S2 / (dS ** 2) + r * S / dS)

                else:
                    # non-symmetric row at the barrier (h_-, h_+)
                    hm = float(h_minus)
                    hp = float(h_plus)

                    # first derivative u_S ≈ a1*u_{i+1} + b1*u_i + c1*u_{i-1}
                    a1 =  hp / (hm * (hm + hp))
                    b1 = (hp - hm) / (hm * hp)
                    c1 = -hm / (hp * (hm + hp))

                    # second derivative u_SS ≈ d2*u_{i+1} + e2*u_i + f2*u_{i-1}
                    d2 =  2.0 / (hm * (hm + hp))
                    e2 = -2.0 / (hm * hp)
                    f2 =  2.0 / (hp * (hm + hp))

                    L_left   = 0.5 * sig2S2 * f2 + r * S * c1
                    L_center = 0.5 * sig2S2 * e2 + r * S * b1 - r
                    L_right  = 0.5 * sig2S2 * d2 + r * S * a1

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

            # solve
            V = self._solve_tridiagonal(sub, main, sup, rhs)

            # barrier projection at monitoring times
            step_index_after = (m - 1)
            if step_index_after in monitor_step_index:
                self._apply_knockout_projection(V, lo_bar, up_bar, s_nodes)
                # per FIS: do not restart Rannacher because of this projection

        return V

    # ------------------------------------------------------------------
    # Interpolation in S
    # ------------------------------------------------------------------
    @staticmethod
    def _interp_linear(x: float, xs: List[float], ys: List[float]) -> float:
        if x <= xs[0]:
            return float(ys[0])
        if x >= xs[-1]:
            return float(ys[-1])
        lo, hi = 0, len(xs) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if x < xs[mid]:
                hi = mid
            else:
                lo = mid
        x0, x1 = xs[lo], xs[hi]
        y0, y1 = ys[lo], ys[hi]
        w = (x - x0) / (x1 - x0)
        return float((1 - w) * y0 + w * y1)

    # ------------------------------------------------------------------
    # Public API: price and Greeks
    # ------------------------------------------------------------------
    def _monitoring_step_map(self) -> Dict[int, bool]:
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

    def _solve_grid_once(self) -> Tuple[List[float], List[float], float]:
        """Return S_grid_shifted, V_grid, and effective spot S0' after PV-dividend escrow."""
        pv_divs = self._pv_dividends_escrow()
        S_eff = self.spot_price - pv_divs
        S_shifted = [max(s - pv_divs, 0.0) for s in self.S_nodes]

        lo_eff, up_eff = self._effective_barriers_for_pricing()
        mp = self._monitoring_step_map()
        V = self._solve_pde_backward(lo_eff, up_eff, mp, S_shifted)

        if self.barrier_type in ("down-and-in", "up-and-in", "double-in"):
            V_vanilla = self._solve_pde_backward(None, None, {}, S_shifted)
            V = [V_vanilla[i] - V[i] for i in range(len(V))]

        return S_shifted, V, S_eff

    def price(self) -> float:
        Sg, Vg, S_eff = self._solve_grid_once()
        return self._interp_linear(S_eff, Sg, Vg)

    # ---- Greeks (FIS barrier-aware stencils) ----
    def _delta_gamma_from_grid(self,
                               s_nodes: List[float],
                               V: List[float],
                               S_eff: float,
                               lo_bar: Optional[float],
                               up_bar: Optional[float]) -> Tuple[float, float]:
        N = len(s_nodes) - 1
        dS = s_nodes[1] - s_nodes[0]
        iS = max(1, min(N - 1, min(range(N), key=lambda k: abs(S_eff - s_nodes[k]))))

        # central defaults
        delta_c = (V[iS + 1] - V[iS - 1]) / (2.0 * dS)
        gamma_c = (V[iS + 1] - 2.0 * V[iS] + V[iS - 1]) / (dS * dS)

        side, j_bar, h_minus, h_plus = self._locate_barrier_interval(s_nodes, lo_bar, up_bar)
        if side is None or j_bar is None:
            return float(delta_c), float(gamma_c)

        # intervals relative to barrier
        in_first = (iS == j_bar or iS == j_bar + 1)
        in_second = (iS == j_bar - 1 or iS == j_bar + 2)

        if in_first:
            # one-sided Δ
            if side == "down":
                # interior node closest to barrier is j_bar+1
                i = j_bar + 1
                delta_os = (1.5 * V[i] - 2.0 * V[i - 1] + 0.5 * V[min(N, i + 1)]) / dS
            else:
                # interior node closest is j_bar
                i = j_bar
                delta_os = (2.0 * V[i + 1] - 1.5 * V[i] - 0.5 * V[max(0, i - 1)]) / dS  # mirrored

            # Γ: blend non-symmetric second derivative with PDE-limit gamma
            S_bar = s_nodes[i]
            sig = self.volatility
            r = self.r_flat
            g = 0.0  # carry = r - q; with PV escrow we use 'g=0' in Γ_lim (theta=0 on barrier)

            # non-symmetric second derivative at the closest interior node
            gamma_ns = (V[i + 1] - 2.0 * V[i] + V[i - 1]) / (dS * dS)

            # PDE-limit gamma at the barrier using local Δ (delta_os) and V[i]
            denom = max(1e-14, (sig * sig) * S_bar * S_bar)
            gamma_lim = 2.0 * (r * V[i] - g * S_bar * delta_os) / denom

            q = 0.5  # blend weight as in FIS description
            gamma = q * gamma_ns + (1.0 - q) * gamma_lim
            return float(delta_os), float(gamma)

        if in_second:
            # blend Δ and Γ between one-sided and central
            if side == "down":
                delta_os = (1.5 * V[iS] - 2.0 * V[iS - 1] + 0.5 * V[min(N, iS + 1)]) / dS
            else:
                delta_os = (2.0 * V[iS + 1] - 1.5 * V[iS] - 0.5 * V[max(0, iS - 1)]) / dS

            gamma_os = (V[iS + 1] - 2.0 * V[iS] + V[iS - 1]) / (dS * dS)
            alpha = 0.5
            return float(alpha * delta_os + (1 - alpha) * delta_c), \
                   float(alpha * gamma_os + (1 - alpha) * gamma_c)

        return float(delta_c), float(gamma_c)

    def greeks(self, vega_bump: float = 0.01) -> Dict[str, float]:
        lo_eff, up_eff = self._effective_barriers_for_pricing()
        Sg, Vg, S_eff = self._solve_grid_once()
        delta, gamma = self._delta_gamma_from_grid(Sg, Vg, S_eff, lo_eff, up_eff)

        # vega: symmetric bump (as in FIS)
        sig0 = self.volatility
        self.volatility = sig0 + vega_bump; upv = self.price()
        self.volatility = sig0 - vega_bump; dnv = self.price()
        self.volatility = sig0
        vega = (upv - dnv) / (2.0 * vega_bump)

        return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega)}

    # ------------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------------
    def print_details(self) -> None:
        lo_eff, up_eff = self._effective_barriers_for_pricing()
        price = self.price()
        greeks = self.greeks()

        print("==== Discrete Barrier Option (FD + CN, Rannacher) ====")
        print(f"Maturity Date           : {self.maturity_date.isoformat()}")
        print(f"T (years)               : {self.tenor_years:.9f}   [{self.day_count}]")
        print(f"Volatility (sigma)      : {self.volatility:.9f}")
        print(f"Flat r (NACC)           : {self.r_flat:.9f}")
        print(f"PV(dividends, escrow)   : {self._pv_dividends_escrow():.9f}")
        print("")
        print(f"Barrier type            : {self.barrier_type}")
        print(f"KO lower / upper        : {self.barrier_lower} / {self.barrier_upper}")
        print(f"BGK lower / upper       : {self.bgk_lower} / {self.bgk_upper}")
        print(f"BGK window steps        : {self.k_first_cont} .. {self.k_last_cont} (use_bgk={self.use_bgk_correction})")
        print("")
        print(f"Grid (space,time)       : {self.num_space_nodes}, {self.num_time_steps} (Rannacher {self.rannacher_steps})")
        print(f"Spot / Strike           : {self.spot_price:.6f} / {self.strike_price:.6f}")
        print(f"Effective barriers      : {lo_eff} / {up_eff}")
        print("")
        print(f"Price                   : {price:.9f}")
        print(f"Greeks                  : delta={greeks['delta']:.9f}, gamma={greeks['gamma']:.9f}, vega={greeks['vega']:.9f}")
