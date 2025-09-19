"""
Discrete barrier European option pricer using a Black–Scholes PDE solved by
Crank–Nicolson with a Rannacher start (no BGK adjustment).

Design goals (aligned to your existing layout and naming style):
- Single class with explicit __init__ arguments (no dataclasses).
- Clear, interpretable method and variable names for FD readers.
- Discrete monitoring via projection at specified monitoring dates only.
- Up/Down × In/Out (+ double) barriers; cash rebate either at hit or at expiry.
- Greeks: delta/gamma with prudent handling near the active KO barrier
  (one‑sided deltas in the first cell from the barrier; blended second cell).
- KI by in–out parity:  KI = Vanilla − KO
- PV‑escrow of cash dividends to shift S‑grid (robust for discrete cash divs).
- Simple, uniform S‑grid with snapping of K and barrier(s) to nearest node.
- Dirichlet boundaries consistent with vanilla asymptotics.

"""

import math
from typing import Dict, List, Literal, Optional, Tuple
from datetime import date

BarrierType = Literal[
    "none",
    "down-and-out", "up-and-out", "double-out",
    "down-and-in",  "up-and-in",  "double-in",
]
OptionType = Literal["call", "put"]


class DiscreteBarrierFDMPricerCN:
    # --------------------------- construction ---------------------------
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
        rebate_amount: float = 0.0,
        rebate_at_hit: bool = True,   # if False, rebate paid at expiry

        # rates / dividends
        flat_rate_nacc: float = 0.0,  # continuous compounding
        dividend_list: Optional[List[Tuple[date, float]]] = None,

        # numerics
        num_space_nodes: int = 600,
        num_time_steps: int = 600,
        rannacher_steps: int = 2,
        day_count: str = "ACT/365",

        # payoff smoothing near K (optional, for stability)
        smooth_payoff_around_strike: bool = True,
        payoff_smoothing_half_width_nodes: int = 2,

        # Greek stabilization near barrier
        use_one_sided_greeks_near_barrier: bool = True,
        barrier_safety_cells: int = 2,
    ):
        # inputs
        self.spot_price = float(spot)
        self.strike_price = float(strike)
        self.valuation_date = valuation_date
        self.maturity_date = maturity_date
        self.volatility = float(volatility)
        self.option_type = option_type

        self.barrier_type = barrier_type
        self.barrier_lower = lower_barrier
        self.barrier_upper = upper_barrier
        self.monitoring_dates = sorted(monitoring_dates or [])

        self.rebate_amount = float(rebate_amount)
        self.rebate_at_hit = bool(rebate_at_hit)

        self.r_flat = float(flat_rate_nacc)
        self.day_count = day_count.upper()
        self.dividends = [(d, float(a)) for (d, a) in (dividend_list or [])]

        self.num_space_nodes = int(num_space_nodes)
        self.num_time_steps = int(num_time_steps)
        self.rannacher_steps = int(rannacher_steps)

        self.smooth_payoff_around_strike = bool(smooth_payoff_around_strike)
        self.payoff_smoothing_half_width_nodes = int(payoff_smoothing_half_width_nodes)

        self.use_one_sided_greeks_near_barrier = bool(use_one_sided_greeks_near_barrier)
        self.barrier_safety_cells = int(barrier_safety_cells)

        # time
        self.year_fraction = self._year_fraction
        self.T_years = self.year_fraction(self.valuation_date, self.maturity_date)
        self.dt = self.T_years / max(1, self.num_time_steps)

        # S-grid
        self.S_nodes = self._build_space_grid()
        self.dS = self.S_nodes[1] - self.S_nodes[0]

    # -------------------- basic day-count / cash utils ------------------
    def _year_fraction(self, d0: date, d1: date) -> float:
        days = max(0, (d1 - d0).days)
        if self.day_count in ("ACT/360",):
            return days / 360.0
        if self.day_count in ("30/360", "30E/360"):
            y0, m0, dd0 = d0.year, d0.month, min(d0.day, 30)
            y1, m1, dd1 = d1.year, d1.month, min(d1.day, 30)
            return ((y1 - y0) * 360 + (m1 - m0) * 30 + (dd1 - dd0)) / 360.0
        return days / 365.0

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

    # --------------------------- space grid -----------------------------
    def _build_space_grid(self) -> List[float]:
        anchors = [self.spot_price, self.strike_price]
        if self.barrier_lower is not None: anchors.append(self.barrier_lower)
        if self.barrier_upper is not None: anchors.append(self.barrier_upper)

        s_ref = max(anchors) if anchors else max(1.0, self.spot_price)
        s_max = 4.0 * s_ref * math.exp(self.volatility * math.sqrt(max(self.T_years, 1e-12)))
        s_min = 0.0

        N = max(200, int(self.num_space_nodes))
        dS = (s_max - s_min) / N
        nodes = [s_min + i * dS for i in range(N + 1)]

        def snap_to_nearest(x: Optional[float]):
            if x is None: return
            j = min(range(len(nodes)), key=lambda i: abs(nodes[i] - x))
            nodes[j] = float(x)

        snap_to_nearest(self.strike_price)
        snap_to_nearest(self.barrier_lower)
        snap_to_nearest(self.barrier_upper)
        return nodes

    # -------------------- payoff / boundaries / KO proj -----------------
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

    def _apply_knockout_projection(self, values: List[float],
                                   lo_bar: Optional[float],
                                   up_bar: Optional[float],
                                   s_nodes: List[float],
                                   time_to_maturity_left: float) -> None:
        """At a monitoring time: if the barrier is breached, set value = rebate (or 0)."""
        # Rebate present value at this hit time (if rebate_at_hit=False, pay at expiry):
        if self.rebate_at_hit:
            rebate_val = self.rebate_amount
        else:
            # rebate paid at expiry: value at monitoring time = PV to expiry of rebate
            rebate_val = self.rebate_amount * math.exp(-self.r_flat * time_to_maturity_left)

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
                values[i] = rebate_val

    # ---------------------- monitoring map (discrete) -------------------
    def _monitoring_step_map(self) -> Dict[int, float]:
        """Return {time_step_index: tau_left} where KO projection must be applied."""
        mp: Dict[int, float] = {}
        for d in self.monitoring_dates:
            if self.valuation_date < d <= self.maturity_date:
                k = int(round(self.year_fraction(self.valuation_date, d) / self.dt))
                tau_left = max(0.0, self.T_years - k * self.dt)
                mp[k] = tau_left
        # Always include maturity to ensure final payout at T (projection will be harmless there).
        kT = self.num_time_steps
        mp.setdefault(kT, 0.0)
        return mp

    # ------------------- barrier location for Greek logic ---------------
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
            for k in range(N):
                if s_nodes[k] <= H <= s_nodes[k+1]:
                    j = max(0, min(N-1, k))
                    return ("down", j, max(1e-12, H - s_nodes[j]), max(1e-12, s_nodes[j + 1] - H))

        if self.barrier_type in ("up-and-out", "double-out") and up_bar is not None:
            H = up_bar
            if H <= s_nodes[0]:  return ("up", 0, 1e-12, s_nodes[1]-s_nodes[0])
            if H >= s_nodes[-1]: return ("up", N-1, s_nodes[N-1]-s_nodes[N-2], 1e-12)
            for k in range(N):
                if s_nodes[k] <= H <= s_nodes[k+1]:
                    j = max(0, min(N-1, k))
                    return ("up", j, max(1e-12, H - s_nodes[j]), max(1e-12, s_nodes[j + 1] - H))

        return (None, None, None, None)

    # ----------------------- tridiagonal solver -------------------------
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

    # ---------------------- core PDE time-march -------------------------
    def _solve_backward_once(self,
                             lo_bar: Optional[float],
                             up_bar: Optional[float],
                             monitor_map: Dict[int, float],
                             s_nodes_shifted: List[float]) -> List[float]:
        N = len(s_nodes_shifted) - 1
        M = self.num_time_steps
        dt = self.dt
        r = self.r_flat
        sig = self.volatility
        dS = s_nodes_shifted[1] - s_nodes_shifted[0]

        V = self._terminal_payoff_array(s_nodes_shifted)

        for m in range(M, 0, -1):
            theta = 1.0 if (M - m) < self.rannacher_steps else 0.5

            sub = [0.0] * (N + 1)
            main = [0.0] * (N + 1)
            sup = [0.0] * (N + 1)
            rhs = [0.0] * (N + 1)

            # Dirichlet boundaries (tau_left is time remaining after completing this step)
            tau_left = self.T_years - (m - 1) * dt
            if self.option_type == "call":
                rhs[0] = 0.0
                rhs[N] = s_nodes_shifted[-1] - self.strike_price * math.exp(-r * tau_left)
            else:
                rhs[0] = self.strike_price * math.exp(-r * tau_left)
                rhs[N] = 0.0
            main[0] = 1.0
            main[N] = 1.0

            # Standard central rows (we keep the stencil simple & robust)
            for i in range(1, N):
                S = s_nodes_shifted[i]
                sig2S2 = (sig * S) ** 2

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

            # solve
            V = self._solve_tridiagonal(sub, main, sup, rhs)

            # barrier projection at monitoring times (discrete monitoring only)
            k_after = (m - 1)  # time index after finishing the step
            if k_after in monitor_map:
                tau_left_after = monitor_map[k_after]
                self._apply_knockout_projection(
                    values=V,
                    lo_bar=lo_bar,
                    up_bar=up_bar,
                    s_nodes=s_nodes_shifted,
                    time_to_maturity_left=tau_left_after
                )

        return V

    # ---------------------- single full run helper ----------------------
    def _run_grid_once(self) -> Tuple[List[float], List[float], float]:
        """Return (S_grid_shifted, V_grid, effective_spot)."""
        pv_divs = self._pv_dividends_escrow()
        S_eff = self.spot_price - pv_divs
        S_shifted = [max(s - pv_divs, 0.0) for s in self.S_nodes]

        lo_eff, up_eff = self.barrier_lower, self.barrier_upper
        monitor_map = self._monitoring_step_map()

        V = self._solve_backward_once(lo_eff, up_eff, monitor_map, S_shifted)

        if self.barrier_type in ("down-and-in", "up-and-in", "double-in"):
            V_vanilla = self._solve_backward_once(None, None, {}, S_shifted)
            V = [V_vanilla[i] - V[i] for i in range(len(V))]

        return S_shifted, V, S_eff

    # ----------------------------- API ---------------------------------
    @staticmethod
    def _interp_linear(x: float, xs: List[float], ys: List[float]) -> float:
        if x <= xs[0]:   return float(ys[0])
        if x >= xs[-1]:  return float(ys[-1])
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

    def price(self) -> float:
        Sg, Vg, S_eff = self._run_grid_once()
        return self._interp_linear(S_eff, Sg, Vg)

    # Greeks (with prudent near-barrier handling)
    def _delta_gamma_from_grid(self,
                               s_nodes: List[float],
                               V: List[float],
                               S_eff: float,
                               lo_bar: Optional[float],
                               up_bar: Optional[float]) -> Tuple[float, float]:
        N = len(s_nodes) - 1
        dS = s_nodes[1] - s_nodes[0]
        iS = max(1, min(N - 1, min(range(N), key=lambda k: abs(S_eff - s_nodes[k]))))

        delta_c = (V[iS + 1] - V[iS - 1]) / (2.0 * dS)
        gamma_c = (V[iS + 1] - 2.0 * V[iS] + V[iS - 1]) / (dS * dS)

        if not self.use_one_sided_greeks_near_barrier:
            return float(delta_c), float(gamma_c)

        side, j_bar, _, _ = self._locate_barrier_interval(s_nodes, lo_bar, up_bar)
        if side is None or j_bar is None:
            return float(delta_c), float(gamma_c)

        # Distance in cells from the closest barrier interface
        in_first = (iS == j_bar or iS == j_bar + 1)
        in_second = (iS == j_bar - 1 or iS == j_bar + 2)

        if in_first:
            # one-sided delta pointing away from barrier; gamma damped
            if side == "down":
                i = max(1, j_bar + 1)
                delta_os = (V[i + 1] - V[i]) / dS
            else:
                i = max(1, j_bar)
                delta_os = (V[i] - V[i - 1]) / dS
            gamma_os = (V[i + 1] - 2.0 * V[i] + V[i - 1]) / (dS * dS)
            gamma_os = max(min(gamma_os, 1e4), -1e4)  # cap extreme spikes
            return float(delta_os), float(gamma_os)

        if in_second:
            # blend one-sided and central
            if side == "down":
                delta_os = (V[iS + 1] - V[iS]) / dS
            else:
                delta_os = (V[iS] - V[iS - 1]) / dS
            gamma_os = (V[iS + 1] - 2.0 * V[iS] + V[iS - 1]) / (dS * dS)
            alpha = 0.5
            return float(alpha * delta_os + (1 - alpha) * delta_c), \
                   float(alpha * gamma_os + (1 - alpha) * gamma_c)

        return float(delta_c), float(gamma_c)

    def greeks(self, vega_bump: float = 0.01) -> Dict[str, float]:
        Sg, Vg, S_eff = self._run_grid_once()
        delta, gamma = self._delta_gamma_from_grid(
            s_nodes=Sg, V=Vg, S_eff=S_eff,
            lo_bar=self.barrier_lower, up_bar=self.barrier_upper
        )

        # vega via symmetric sigma bump
        sig0 = self.volatility
        self.volatility = sig0 + vega_bump; upv = self.price()
        self.volatility = sig0 - vega_bump; dnv = self.price()
        self.volatility = sig0
        vega = (upv - dnv) / (2.0 * vega_bump)

        return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega)}

    # Pretty printer -----------------------------------------------------
    def print_details(self) -> None:
        price = self.price()
        greeks = self.greeks()
        print("==== Discrete Barrier Option (FD + CN, Rannacher) — No BGK ====")
        print(f"Maturity Date           : {self.maturity_date.isoformat()}")
        print(f"T (years)               : {self.T_years:.9f}   [{self.day_count}]")
        print(f"Volatility (sigma)      : {self.volatility:.9f}")
        print(f"Flat r (NACC)           : {self.r_flat:.9f}")
        print(f"PV(dividends, escrow)   : {self._pv_dividends_escrow():.9f}")
        print("")
        print(f"Barrier type            : {self.barrier_type}")
        print(f"KO lower / upper        : {self.barrier_lower} / {self.barrier_upper}")
        print(f"Rebate (amt/at_hit)     : {self.rebate_amount} / {self.rebate_at_hit}")
        print("")
        print(f"Grid (space,time)       : {self.num_space_nodes}, {self.num_time_steps} (Rannacher {self.rannacher_steps})")
        print(f"Spot / Strike           : {self.spot_price:.6f} / {self.strike_price:.6f}")
        print("")
        print(f"Price                   : {price:.9f}")
        print(f"Greeks                  : delta={greeks['delta']:.9f}, gamma={greeks['gamma']:.9f}, vega={greeks['vega']:.9f}")