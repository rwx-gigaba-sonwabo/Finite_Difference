from __future__ import annotations
from typing import List, Optional, Dict, Tuple, Literal
from datetime import date
import math
import pandas as pd

BarrierType = Literal["none","down-and-out","up-and-out","double-out","down-and-in","up-and-in","double-in"]

class DiscreteBarrierFDMPricer:
    """
    Discretely monitored barrier option (Crank–Nicolson + Rannacher) with
    optional BGK continuity correction when monitoring is frequent.

    Flat continuous-compounding short rate is used on the grid (NACC).
    Dividends still handled by escrow (PV from curve) outside this class.
    """

    # ---- configuration knobs ----
    BGK_BETA = 0.5826        # Broadie–Glasserman–Kou constant
    CONT_MONITORING_LIMIT = 5 # n_lim heuristic for “frequent enough” monitoring

    def __init__(
        self,
        spot: float,
        strike: float,
        valuation_date: date,
        maturity_date: date,
        sigma: float,
        option_type: Literal["call", "put"],
        barrier_type: BarrierType = "none",
        lower_barrier: Optional[float] = None,
        upper_barrier: Optional[float] = None,
        monitor_dates: Optional[List[date]] = None,
        # curves / dividends handled as in your surrounding app
        discount_curve: Optional[pd.DataFrame] = None,
        forward_curve: Optional[pd.DataFrame] = None,
        dividend_schedule: Optional[List[Tuple[date, float]]] = None,
        # grid controls
        grid_points: int = 400,
        time_steps: int = 400,
        rannacher_steps: int = 2,
        # rate control (flat NACC rate for PDE)
        flat_r_nacc: float = 0.0,
        # day count
        day_count: str = "ACT/365",
        # behavior flags
        restart_on_monitoring: bool = False,  # intentionally ignored for barrier events
        mollify_final: bool = True,
        mollify_band_nodes: int = 2,
        price_extrapolation: bool = False,
    ):
        # store inputs
        self.spot = float(spot)
        self.strike = float(strike)
        self.valuation_date = valuation_date
        self.maturity_date = maturity_date
        self.sigma = float(sigma)
        self.option_type = option_type
        self.barrier_type = barrier_type
        self.lower_barrier = lower_barrier
        self.upper_barrier = upper_barrier
        self.monitor_dates = sorted(monitor_dates or [])
        self.discount_curve_df = discount_curve.copy() if discount_curve is not None else None
        self.forward_curve_df  = forward_curve.copy()  if forward_curve  is not None else None
        self.dividend_schedule = sorted(dividend_schedule or [], key=lambda x: x[0])

        self.grid_points = int(grid_points)
        self.time_steps = int(time_steps)
        self.rannacher_steps = int(rannacher_steps)
        self.flat_r_nacc = float(flat_r_nacc)
        self.day_count = day_count.upper()
        self.mollify_final = bool(mollify_final)
        self.mollify_band_nodes = int(mollify_band_nodes)
        self.price_extrapolation = bool(price_extrapolation)

        # year fraction + tenor
        self._yf = self._year_fraction
        self.T = self._yf(self.valuation_date, self.maturity_date)

        # normalize curves if provided
        if self.discount_curve_df is not None:
            self.discount_curve_df = self._normalize_curve_df(self.discount_curve_df)
        if self.forward_curve_df is not None:
            self.forward_curve_df = self._normalize_curve_df(self.forward_curve_df)

        # grids
        self.S_nodes = self._build_space_grid()
        self.dS = self.S_nodes[1] - self.S_nodes[0]
        self.time_nodes = [i * self.T / self.time_steps for i in range(self.time_steps + 1)]

        # BGK continuity decision and adjusted barriers
        (self.use_continuous_barrier,
         self.bgk_lower, self.bgk_upper,
         self.continuous_k0, self.continuous_k1) = self._bgk_continuous_decision_and_adjustment()

    # ---------- day count ----------
    def _year_fraction(self, d0: date, d1: date) -> float:
        if self.day_count in ("ACT/365", "ACT/365F", "ACT/365 FIXED"):
            return max(0.0, (d1 - d0).days) / 365.0
        if self.day_count in ("ACT/360",):
            return max(0.0, (d1 - d0).days) / 360.0
        if self.day_count in ("30/360", "30E/360"):
            y0, m0, dd0 = d0.year, d0.month, min(d0.day, 30)
            y1, m1, dd1 = d1.year, d1.month, min(d1.day, 30)
            return ((y1 - y0) * 360 + (m1 - m0) * 30 + (dd1 - dd0)) / 360.0
        return max(0.0, (d1 - d0).days) / 365.0

    # ---------- curve helper ----------
    @staticmethod
    def _normalize_curve_df(df: pd.DataFrame) -> pd.DataFrame:
        if "Date" not in df.columns or "NACA" not in df.columns:
            raise ValueError("Curve DataFrame must have columns: 'Date', 'NACA'.")
        if not pd.api.types.is_string_dtype(df["Date"]):
            df = df.copy()
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        return df

    # ---------- BGK decision & barrier adjustment ----------
    def _bgk_continuous_decision_and_adjustment(self):
        """
        Decide if monitoring is 'frequent enough' per FIS and compute BGK-adjusted barriers.

        Frequent-enough test (FIS):
            Δt = t_e / n, where t_e is time to expiry and n is the chosen CN time-steps.
        For each interval between consecutive monitoring dates:
                n_m = max(n_min, ceil(t_mon / Δt))
        If sum(n_m) > n_lim * n  ->  use continuous approximation on [first,last].

        Barrier adjustment (FIS):
            B_adj(t) = B * exp( ± φ(t) ),    φ(t) = 0.5826 * σ(x_B(t), t) * t * a_b,
            a_b = t_b / n_mon, with t_b = time from first to last monitoring date.
            Use '+' for upper barriers (moves up), '−' for lower barriers (moves down).
        """
        # no barrier monitoring => nothing to adjust
        if self.barrier_type in ("none",) or len(self.monitor_dates) == 0:
            return (False, self.lower_barrier, self.upper_barrier, None, None)

        # ensure sorted monitoring dates
        mons = sorted(self.monitor_dates)
        first = mons[0]
        last  = mons[-1]
        if last <= first:
            return (False, self.lower_barrier, self.upper_barrier, None, None)

        # --- Frequent-monitoring decision (uses time-to-expiry t_e) ---
        t_e = self.T                             # time to expiry (valuation -> maturity)
        n    = max(1, int(self.time_steps))      # target CN steps
        dt_eq = t_e / n                          # equidistant step
        n_min = 1
        # sum n_m over *consecutive* monitoring intervals (inside [first,last])
        total_nm = 0
        for i in range(1, len(mons)):
            t_mon = self._yf(mons[i-1], mons[i])
            nm_i = max(n_min, math.ceil(t_mon / max(1e-12, dt_eq)))
            total_nm += nm_i

        frequent = (total_nm > self.CONT_MONITORING_LIMIT * n)

        # --- Barrier adjustment (uses t = time to expiry) ---
        if not frequent:
            # no switch -> keep original barriers, no continuous window
            return (False, self.lower_barrier, self.upper_barrier, None, None)

        n_mon = len(mons)
        t_b   = self._yf(first, last)            # time from first to last monitoring date
        a_b   = t_b / max(1, n_mon)              # FIS definition
        t     = t_e                              # time to expiry in φ(t)
        phi   = 0.5826 * self.sigma * t * a_b

        lo_adj = self.lower_barrier
        up_adj = self.upper_barrier
        if lo_adj is not None:
            lo_adj = lo_adj * math.exp(-phi)     # lower barrier moves DOWN
        if up_adj is not None:
            up_adj = up_adj * math.exp(+phi)     # upper barrier moves UP

        # map [first,last] to time-step indices (continuous-monitoring window)
        k0 = int(round(self._yf(self.valuation_date, first) / (self.T / self.time_steps)))
        k1 = int(round(self._yf(self.valuation_date, last)  / (self.T / self.time_steps)))
        k0 = max(0, min(self.time_steps, k0))
        k1 = max(0, min(self.time_steps, k1))
        return (True, lo_adj, up_adj, min(k0, k1), max(k0, k1))

    # ---------- space grid ----------
    def _build_space_grid(self) -> List[float]:
        anchors = [self.spot, self.strike]
        if self.lower_barrier: anchors.append(self.lower_barrier)
        if self.upper_barrier: anchors.append(self.upper_barrier)
        s_ref = max(anchors)
        s_max = 4.0 * s_ref * math.exp(self.sigma * math.sqrt(max(self.T, 1e-12)))
        s_min = 0.0
        N = max(200, int(self.grid_points))
        dS = (s_max - s_min) / N
        nodes = [s_min + i * dS for i in range(N + 1)]

        def snap(x: Optional[float]):
            if x is None: return
            j = min(range(len(nodes)), key=lambda i: abs(nodes[i] - x))
            nodes[j] = x

        snap(self.strike); snap(self.lower_barrier); snap(self.upper_barrier)
        return nodes

    # ---------- payoff and smoothing ----------
    def _terminal_payoff(self, S: float) -> float:
        return max(S - self.strike, 0.0) if self.option_type == "call" else max(self.strike - S, 0.0)

    def _terminal_array(self) -> List[float]:
        V = [self._terminal_payoff(S) for S in self.S_nodes]
        if not self.mollify_final or self.mollify_band_nodes <= 0:
            return V
        m = self.mollify_band_nodes
        k_idx = min(range(len(self.S_nodes)), key=lambda i: abs(self.S_nodes[i] - self.strike))
        i0, i1 = max(0, k_idx - m), min(len(self.S_nodes) - 1, k_idx + m)
        S0, V0 = self.S_nodes[i0], V[i0]
        S1, V1 = self.S_nodes[i1], V[i1]
        a = (V1 - V0) / ((S1 - S0) ** 2) if S1 != S0 else 0.0
        for i in range(i0, i1 + 1):
            s = self.S_nodes[i]
            V[i] = a * (s - S0) ** 2 + V0
        return V

    # ---------- KO projection ----------
    def _apply_knockout(self, values: List[float], lo: Optional[float], up: Optional[float]) -> None:
        for i, s in enumerate(self.S_nodes):
            ko = False
            if self.barrier_type == "down-and-out" and lo is not None and s <= lo: ko = True
            elif self.barrier_type == "up-and-out" and up is not None and s >= up: ko = True
            elif self.barrier_type == "double-out":
                if (lo is not None and s <= lo) or (up is not None and s >= up): ko = True
            if ko: values[i] = 0.0

    # ---------- Thomas tridiagonal solver ----------
    def _solve_tridiagonal(self, sub: List[float], main: List[float],
                           sup: List[float], rhs: List[float]) -> List[float]:
        n = len(rhs)
        alpha = [0.0]*n
        beta  = [0.0]*n
        x     = [0.0]*n

        piv = main[0]
        if abs(piv) < 1e-14:
            raise ZeroDivisionError("Tridiagonal pivot ~ 0 at row 0")
        alpha[0] = sup[0]/piv
        beta[0]  = rhs[0]/piv

        for i in range(1, n):
            piv = main[i] - sub[i]*alpha[i-1]
            if abs(piv) < 1e-14:
                raise ZeroDivisionError(f"Tridiagonal pivot ~ 0 at row {i}")
            alpha[i] = sup[i]/piv if i < n-1 else 0.0
            beta[i]  = (rhs[i] - sub[i]*beta[i-1]) / piv

        x[-1] = beta[-1]
        for i in range(n-2, -1, -1):
            x[i] = beta[i] - alpha[i]*x[i+1]
        return x

    # ---------- interpolation ----------
    def _interp_linear(self, x: float, xs: List[float], ys: List[float]) -> float:
        if x <= xs[0]:  return ys[0]
        if x >= xs[-1]: return ys[-1]
        lo, hi = 0, len(xs)-1
        while hi - lo > 1:
            mid = (lo + hi)//2
            if x < xs[mid]: hi = mid
            else: lo = mid
        w = (x - xs[lo])/(xs[hi] - xs[lo])
        return (1-w)*ys[lo] + w*ys[hi]

    # ---------- barrier-aware node selection & one-sided stencils ----------
    def _closest_barrier_index(self, lo: Optional[float], up: Optional[float]) -> Optional[int]:
        """Index of grid node closest to the currently effective KO barrier."""
        idx = None
        if self.barrier_type in ("down-and-out","double-out") and lo is not None:
            idx = min(range(len(self.S_nodes)), key=lambda i: abs(self.S_nodes[i] - lo))
        if self.barrier_type in ("up-and-out","double-out") and up is not None:
            j = min(range(len(self.S_nodes)), key=lambda i: abs(self.S_nodes[i] - up))
            if idx is None or abs(self.S_nodes[j] - (up or 0.0)) < abs(self.S_nodes[idx] - (lo or 0.0)):
                idx = j
        return idx

    def _one_sided_coeffs(self, i_bar: int) -> tuple[float,float,float,float,float,float]:
        """
        Non-symmetric coefficients at the node nearest the barrier.
        Uses step sizes h_minus = S_i - S_{i-1}, h_plus = S_{i+1} - S_i,
        with the φ safeguard (φ=0.05) as suggested by the FIS doc.
        """
        S = self.S_nodes
        i = i_bar
        h_minus = max(1e-14, S[i] - S[i-1])
        h_plus  = max(1e-14, S[i+1] - S[i])

        phi = 0.05
        if h_plus  < phi*h_minus: h_plus  = phi*h_minus
        if h_minus < phi*h_plus:  h_minus = phi*h_plus

        # First derivative: V_S ≈ a_i V_{i+1} + b_i V_i + c_i V_{i-1}
        a_i =  h_plus / (h_plus*(h_minus + h_plus))
        b_i = (h_plus - h_minus) / (h_minus*h_plus)
        c_i = -h_minus / (h_minus*(h_minus + h_plus))

        # Second derivative: V_SS ≈ d_i V_{i+1} + e_i V_i + f_i V_{i-1}
        d_i =  2.0 / (h_minus*(h_minus + h_plus))
        e_i = -2.0 / (h_minus*h_plus)
        f_i =  2.0 / (h_plus*(h_minus + h_plus))
        return a_i, b_i, c_i, d_i, e_i, f_i

    # ---------- PDE stepping ----------
    def _backward_CN(self, effective_lo: Optional[float], effective_up: Optional[float],
                     monitor_steps: Dict[int, bool]) -> List[float]:
        N = len(self.S_nodes) - 1
        M = self.time_steps
        dt = self.T / M
        r = self.flat_r_nacc
        sigma = self.sigma
        dS = self.dS

        V = self._terminal_array()

        # node close to barrier (only used inside the “continuous” window)
        i_barrier = None
        if self.use_continuous_barrier:
            i_barrier = self._closest_barrier_index(effective_lo, effective_up)

        for m in range(M, 0, -1):
            theta = 1.0 if (M - m) < self.rannacher_steps else 0.5

            sub = [0.0]*(N+1)
            main = [0.0]*(N+1)
            sup = [0.0]*(N+1)
            rhs =  [0.0]*(N+1)

            # boundary values
            if self.option_type == "call":
                rhs[0] = 0.0
                rhs[N] = self.S_nodes[-1] - self.strike * math.exp(-r*(self.T - (m-1)*dt))
            else:
                rhs[0] = self.strike * math.exp(-r*(self.T - (m-1)*dt))
                rhs[N] = 0.0
            main[0] = 1.0
            main[N] = 1.0

            for i in range(1, N):
            S_i = self.S[i]
            sig2S2 = (sig * S_i) ** 2

            # Flags: are we at the closest interior node to an active barrier?
            use_nonuniform = False
            barrier_on_left = False  # True for down barrier; False for up barrier
            h_minus = h_plus = None  # left/right distances for the non-uniform formulas

            if self.use_bgk_correction:
                if idx_lo is not None and i == idx_lo + 1:
                    # Down-and-out style barrier on the LEFT of node i (between B and i)
                    # left "point" is the barrier at B = self.S[idx_lo], right point is grid i+1
                    barrier_on_left = True
                    use_nonuniform = True
                    B = self.S[idx_lo]
                    # distances from node i
                    h_minus = S_i - B                         # to the barrier (left)
                    h_plus  = self.S[i+1] - S_i if i+1 <= N else self.S[i] - self.S[i-1]  # to right neighbor
                if idx_up is not None and i == idx_up - 1:
                    # Up-and-out style barrier on the RIGHT of node i (between i and B)
                    # right "point" is the barrier at B, left point is grid i-1
                    barrier_on_left = False
                    use_nonuniform = True
                    B = self.S[idx_up]
                    h_minus = S_i - self.S[i-1] if i-1 >= 0 else self.S[i+1] - self.S[i]  # to left neighbor
                    h_plus  = B - S_i                          # to the barrier (right)

            if use_nonuniform and h_minus > 0 and h_plus > 0:
                # --- Non-uniform coefficients (FIS formulas)
                a_i =  h_minus / (h_plus * (h_minus + h_plus))         # multiplies V_{i+1}
                b_i = (h_plus - h_minus) / (h_minus * h_plus)          # multiplies V_i
                c_i = -h_plus / (h_minus * (h_minus + h_plus))         # multiplies V_{i-1}

                d_i =  2.0 / (h_plus * (h_minus + h_plus))             # multiplies V_{i+1}
                e_i = -2.0 / (h_minus * h_plus)                        # multiplies V_i
                f_i =  2.0 / (h_minus * (h_minus + h_plus))            # multiplies V_{i-1}

                # Build Black–Scholes operator L = 0.5*σ^2 S^2 * V_SS + r S * V_S - r V
                L_left   = 0.5 * sig2S2 * f_i + r * S_i * c_i
                L_mid    = 0.5 * sig2S2 * e_i + r * S_i * b_i - r
                L_right  = 0.5 * sig2S2 * d_i + r * S_i * a_i

                # If the barrier is on the LEFT, V_{i-1} is the barrier value (0 for KO) → drop sub[i]
                # If the barrier is on the RIGHT, V_{i+1} is the barrier value (0 for KO) → drop sup[i]
                sub[i]  = 0.0 if barrier_on_left else -theta * dt * L_left
                sup[i]  = -theta * dt * L_right if barrier_on_left else 0.0
                diag[i] = 1.0 - theta * dt * L_mid

                # Explicit side (note: barrier value is zero → no extra RHS term)
                rhs[i]  = (1.0 + (1.0 - theta) * dt * L_mid) * V[i]
                if not barrier_on_left:
                    rhs[i] += ((1.0 - theta) * dt * L_left)  * V[i-1]
                if barrier_on_left:
                    rhs[i] += ((1.0 - theta) * dt * L_right) * V[i+1]

            else:
                # --- Default (uniform) row with barrier-aware upwind drift in a band near the barrier
                drift_left = -r * S_i / (2*dS)
                drift_mid  =  0.0
                drift_right=  r * S_i / (2*dS)

                if idx_lo is not None and (i <= idx_lo + self.NEAR_BARRIER_BW):
                    # Backward (down barrier on the left)
                    drift_left = -r * S_i / dS
                    drift_mid  =  r * S_i / dS
                    drift_right=  0.0
                if idx_up is not None and (i >= idx_up - self.NEAR_BARRIER_BW):
                    # Forward (up barrier on the right)
                    drift_left =  0.0
                    drift_mid  = -r * S_i / dS
                    drift_right=  r * S_i / dS

                diff_left  =  0.5 * sig2S2 / (dS**2)
                diff_mid   = -sig2S2 / (dS**2)
                diff_right =  0.5 * sig2S2 / (dS**2)

                L_left  = diff_left  + drift_left
                L_mid   = diff_mid   + drift_mid  - r
                L_right = diff_right + drift_right

                sub[i]  = -theta * dt * L_left
                diag[i] =  1.0 - theta * dt * L_mid
                sup[i]  = -theta * dt * L_right

                rhs[i]  = (1.0 + (1.0 - theta) * dt * L_mid) * V[i] \
                        + ((1.0 - theta) * dt * L_left)  * V[i-1] \
                        + ((1.0 - theta) * dt * L_right) * V[i+1]

            V = self._solve_tridiagonal(sub, main, sup, rhs)

            # barrier projection at monitoring instants
            step_index_after = (m-1)
            if step_index_after in monitor_steps:
                self._apply_knockout(V, effective_lo, effective_up)
                # IMPORTANT: no Rannacher restart on barrier events

        return V

    # ---------- price ----------
    def price(self) -> float:
        # escrow dividends outside this class -> pass effective spot
        div_pv = 0.0  # keep hook; your outer code sets effective S if needed
        S_eff = self.spot - div_pv
        shifted_nodes = [max(s - div_pv, 0.0) for s in self.S_nodes]
        backup_nodes = self.S_nodes
        self.S_nodes = shifted_nodes
        self.dS = self.S_nodes[1] - self.S_nodes[0]

        # build monitoring step map
        monitor_steps: Dict[int,bool] = {}
        if self.use_continuous_barrier:
            for k in range(self.continuous_k0, self.continuous_k1 + 1):
                monitor_steps[k] = True
            lo_eff, up_eff = self.bgk_lower, self.bgk_upper
        else:
            for d in self.monitor_dates:
                if self.valuation_date < d <= self.maturity_date:
                    k = int(round(self._yf(self.valuation_date, d) / (self.T/self.time_steps)))
                    monitor_steps[k] = True
            lo_eff, up_eff = self.lower_barrier, self.upper_barrier

        # KO directly or KI via parity
        if self.barrier_type in ("down-and-in","up-and-in","double-in"):
            V_plain = self._backward_CN(None, None, {})
            V_ko    = self._backward_CN(lo_eff, up_eff, monitor_steps)
            price_val = self._interp_linear(S_eff, self.S_nodes, [V_plain[i]-V_ko[i] for i in range(len(V_plain))])
        else:
            V = self._backward_CN(lo_eff, up_eff, monitor_steps)
            price_val = self._interp_linear(S_eff, self.S_nodes, V)

        # restore nodes
        self.S_nodes = backup_nodes
        self.dS = self.S_nodes[1] - self.S_nodes[0]
        return float(price_val)

    def greeks(self,
            spot_rel_bump: float = 0.01,
            vega_abs_bump: float = 0.01) -> dict:
        """
        Barrier-aware finite-difference Greeks, aligned with the FIS guidance.
        - Delta: one-sided near the active barrier; central elsewhere
        - Gamma: central; near barrier use a blended estimate for stability
        - Vega : symmetric absolute bump on sigma
        Returns a dict with {'delta','gamma','vega'}.
        """

        # --- helper: which barrier is effectively active right now? (BGK if continuous) ---
        eff_lo = self.bgk_lower if getattr(self, "use_continuous_barrier", False) else self.lower_barrier
        eff_up = self.bgk_upper  if getattr(self, "use_continuous_barrier", False) else self.upper_barrier

        def _near_effective_barrier(S: float, tol_mult: float = 2.0) -> bool:
            tol = tol_mult * max(1e-14, self.dS)
            close_lo = (eff_lo is not None) and (abs(S - eff_lo) <= tol)
            close_up = (eff_up is not None) and (abs(S - eff_up) <= tol)
            return close_lo or close_up

        # --- store base state ---
        base_spot = self.spot
        base_sigma = self.sigma

        # --- base price ---
        base_px = self.price()

        # --- delta & gamma via spot bumps (barrier-aware stencil) ---
        ds = max(1e-12, spot_rel_bump * base_spot)

        self.spot = base_spot + ds
        px_up = self.price()

        self.spot = base_spot - ds
        px_dn = self.price()

        # restore spot
        self.spot = base_spot

        if _near_effective_barrier(base_spot):
            # one-sided delta (downwind) at barrier per FIS figures
            delta = (base_px - px_dn) / ds
            # gamma: blend to reduce ringing near KO
            gamma_central = (px_up - 2.0 * base_px + px_dn) / (ds * ds)
            gamma_onesided = gamma_central  # same stencil here; keep structure for clarity
            blend_q = 0.5
            gamma = blend_q * gamma_central + (1.0 - blend_q) * gamma_onesided
        else:
            # standard central stencils away from barrier
            delta = (px_up - px_dn) / (2.0 * ds)
            gamma = (px_up - 2.0 * base_px + px_dn) / (ds * ds)

        # --- vega via symmetric sigma bump ---
        dv = max(1e-12, vega_abs_bump)

        self.sigma = base_sigma + dv
        px_vup = self.price()

        self.sigma = base_sigma - dv
        px_vdn = self.price()

        # restore sigma
        self.sigma = base_sigma

        vega = (px_vup - px_vdn) / (2.0 * dv)

        return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega)}


def _discount_factor_from_curve(self, lookup: date) -> float:
        if self.discount_curve_df is None:
            raise ValueError("discount_curve is required for DF lookups.")
        iso = lookup.isoformat()
        row = self.discount_curve_df[self.discount_curve_df["Date"] == iso]
        if row.empty:
            raise ValueError(f"Discount factor not found for date: {iso}")
        naca = float(row["NACA"].values[0])                       # nominal annual compounding
        tau  = self._yf(self.valuation_date, lookup)
        return (1.0 + naca) ** (-tau)

    def _nacc_from_curve(self, lookup: date) -> float:
        if self.discount_curve_df is None:
            return 0.0
        iso = lookup.isoformat()
        row = self.discount_curve_df[self.discount_curve_df["Date"] == iso]
        if row.empty:
            return 0.0
        naca = float(row["NACA"].values[0])
        return math.log(1.0 + naca)                               # NACC

    def _forward_nacc(self, d0: date, d1: date) -> float:
        df1 = self._discount_factor_from_curve(d1)
        df0 = self._discount_factor_from_curve(d0)
        tau = max(1e-12, self._yf(d0, d1))
        return -math.log(df1 / df0) / tau

    def pv_dividends(self) -> float:
        """PV of discrete dividends using discount curve."""
        if not self.dividend_schedule:
            return 0.0
        pv = 0.0
        for pay_date, amount in self.dividend_schedule:
            if self.valuation_date < pay_date <= self.maturity_date:
                fwd_r = self._forward_nacc(self.valuation_date, pay_date)
                tau   = self._yf(self.valuation_date, pay_date)
                df    = math.exp(-fwd_r * tau)
                pv += amount * df
        return pv

    def dividend_yield_nacc(self) -> float:
        """Flat q (NACC) such that PV(divs) is reproduced over [valuation, maturity]."""
        S0 = self.spot
        pv = self.pv_dividends()
        T  = max(1e-12, self.T)
        if pv >= S0:
            raise ValueError("PV(dividends) >= spot; cannot back out q.")
        return -math.log((S0 - pv)/S0) / T

    def flat_r_from_curve(self) -> float:
        """Convenience: forward NACC from valuation to maturity (flat r for PDE)."""
        return self._forward_nacc(self.valuation_date, self.maturity_date)

    def price(self) -> float:
        # escrow discrete dividends into the spot (escrowed-spot shift)
        div_pv = self.pv_dividends() if self.dividend_schedule else 0.0
        S_eff = self.spot - div_pv
        shifted_nodes = [max(s - div_pv, 0.0) for s in self.S_nodes]
        backup_nodes = self.S_nodes
        self.S_nodes = shifted_nodes
        self.dS = self.S_nodes[1] - self.S_nodes[0]

        monitor_steps: Dict[int,bool] = {}
        if self.use_continuous_barrier:
            for k in range(self.continuous_k0, self.continuous_k1 + 1):
                monitor_steps[k] = True
            lo_eff, up_eff = self.bgk_lower, self.bgk_upper
        else:
            for d in self.monitor_dates:
                if self.valuation_date < d <= self.maturity_date:
                    k = int(round(self._yf(self.valuation_date, d) / (self.T/self.time_steps)))
                    monitor_steps[k] = True
            lo_eff, up_eff = self.lower_barrier, self.upper_barrier

        if self.barrier_type in ("down-and-in","up-and-in","double-in"):
            V_plain = self._backward_CN(None, None, {})
            V_ko    = self._backward_CN(lo_eff, up_eff, monitor_steps)
            price_val = self._interp_linear(S_eff, self.S_nodes, [V_plain[i]-V_ko[i] for i in range(len(V_plain))])
        else:
            V = self._backward_CN(lo_eff, up_eff, monitor_steps)
            price_val = self._interp_linear(S_eff, self.S_nodes, V)

        self.S_nodes = backup_nodes
        self.dS = self.S_nodes[1] - self.S_nodes[0]
        return float(price_val)

    def print_details(self,
                      trade_id: str = "NA",
                      direction: Literal["long","short"] = "long",
                      quantity: int = 1,
                      contract_multiplier: float = 1.0) -> None:
        """
        Prints a self-contained report (trade + curves + dividends + result).
        """
        # rates (prefer curve if present; else the user-supplied flat_r_nacc)
        r_flat = None
        try:
            r_flat = self.flat_r_from_curve()
        except Exception:
            r_flat = self.flat_r_nacc

        # dividend yield (flat NACC from discrete schedule)
        try:
            q_flat = self.dividend_yield_nacc()
        except Exception:
            q_flat = 0.0

        # compute price & Greeks
        px = self.price()
        greeks = self.greeks()

        # barrier info
        lo_eff = self.bgk_lower if self.use_continuous_barrier else self.lower_barrier
        up_eff = self.bgk_upper  if self.use_continuous_barrier else self.upper_barrier

        # print
        print("===== Discrete Barrier Option — Run Summary =====")
        print(f"Trade ID            : {trade_id}")
        print(f"Direction           : {direction}")
        print(f"Quantity            : {quantity}")
        print(f"Contract Multiplier : {contract_multiplier}")
        print("---- Contract ----")
        print(f"Option Type         : {self.option_type}")
        print(f"Barrier Type        : {self.barrier_type}")
        print(f"Lower Barrier (H_d) : {lo_eff if lo_eff is not None else '-'}"
              + ("  (BGK-adj)" if self.use_continuous_barrier and self.lower_barrier is not None else ""))
        print(f"Upper Barrier (H_u) : {up_eff if up_eff is not None else '-'}"
              + ("  (BGK-adj)" if self.use_continuous_barrier and self.upper_barrier is not None else ""))
        print(f"Spot (S0)           : {self.spot:.6f}")
        print(f"Strike (K)          : {self.strike:.6f}")
        print(f"Valuation Date      : {self.valuation_date.isoformat()}")
        print(f"Maturity Date       : {self.maturity_date.isoformat()}")
        print(f"T (years, {self.day_count}) : {self.T:.6f}")
        print(f"Volatility (sigma)  : {self.sigma:.6f}")
        print("---- Rates & Dividends ----")
        print(f"Flat r (NACC)       : {r_flat:.8f}")
        print(f"Flat q (NACC)       : {q_flat:.8f}")
        print(f"PV(dividends)       : {self.pv_dividends():.6f}")
        print("---- Grid ----")
        print(f"Grid (time,space)   : {self.time_steps}, {self.grid_points}  (Rannacher {self.rannacher_steps})")
        print(f"BGK continuous?     : {self.use_continuous_barrier}  "
              f"[window steps: {self.continuous_k0 if self.use_continuous_barrier else '-'}"
              f"→{self.continuous_k1 if self.use_continuous_barrier else '-'}]")
        print(f"#Monitoring dates   : {len(self.monitor_dates)}")
        print("---- Results ----")
        print(f"Price               : {px:.8f}")
        print(f"Delta               : {greeks['delta']:.8f}")
        print(f"Gamma               : {greeks['gamma']:.8f}")
        print(f"Vega                : {greeks['vega']:.8f}")
        print("=================================================")