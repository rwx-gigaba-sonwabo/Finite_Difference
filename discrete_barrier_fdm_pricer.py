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
        if self.barrier_type in ("none",) or len(self.monitor_dates) == 0:
            return (False, self.lower_barrier, self.upper_barrier, None, None)

        first = min(self.monitor_dates)
        last  = max(self.monitor_dates)
        if last <= first:
            return (False, self.lower_barrier, self.upper_barrier, None, None)

        n_mon = len(self.monitor_dates)
        # Use the full time-to-expiry as the cadence scale (robust in practice)
        dt_eff = self.T / max(1, n_mon)   # "effective" average gap
        frequent = (n_mon >= self.CONT_MONITORING_LIMIT)

        adj = math.exp(self.BGK_BETA * self.sigma * math.sqrt(max(1e-16, dt_eff)))
        lo_adj = self.lower_barrier / adj if self.lower_barrier is not None else None
        up_adj = self.upper_barrier * adj if self.upper_barrier is not None else None

        # map the continuous window [first,last] to time-step indices
        k0 = int(round(self._yf(self.valuation_date, first) / (self.T / self.time_steps)))
        k1 = int(round(self._yf(self.valuation_date, last)  / (self.T / self.time_steps)))
        k0 = max(0, min(self.time_steps, k0))
        k1 = max(0, min(self.time_steps, k1))
        return (frequent, lo_adj, up_adj, min(k0, k1), max(k0, k1))

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

    # --- build the map of "nearest-to-barrier" indices once (for speed/clarity) ---
    def _nearest_barrier_index(self, barrier: float, is_up: bool) -> int | None:
        """
        For an up barrier, pick the interior node just below the barrier.
        For a down barrier, pick the interior node just above the barrier.
        """
        if barrier is None:
            return None
        # find j with S_j < H < S_{j+1}
        j = None
        for k in range(1, len(self.S_nodes)):
            if self.S_nodes[k-1] < barrier < self.S_nodes[k]:
                j = k-1
                break
        if j is None:
            return None
        return j if is_up else (j+1)


    def _one_sided_row_coeffs(self, i: int, barrier_level: float, rebate_value: float = 0.0):
        """
        Build the operator row L(V)_i at the interior node i that is closest to a barrier B,
        using the unequal three-point stencil with points (S_{i-1}, S_i, B).

        Returns:
            a_im1, a_i, const_term
        such that:
            L(V)_i = a_im1 * V_{i-1} + a_i * V_i + const_term,
        where const_term contains all pieces that multiply the known barrier value (rebate).
        """
        S_im1 = self.S_nodes[i-1]
        S_i   = self.S_nodes[i]

        # Distances (positive)
        h_minus = max(1e-14, S_i - S_im1)               # grid step to the left
        h_plus  = max(1e-14, barrier_level - S_i)       # distance from S_i to the barrier (not a grid node)

        # FIS safeguard when barrier is *very* close: h_plus >= phi * h_minus
        phi = 0.05
        if h_plus < phi * h_minus:
            # move the evaluation slightly towards the barrier by adding "alpha" to h_plus as per doc
            h_plus = phi * h_minus

        # Unequal three-point weights at x = S_i, for nodes (S_{i-1}, S_i, B)
        # First derivative dV/dS ≈ w0 * V_{i-1} + w1 * V_i + w2 * V_B
        w0  = -h_plus / (h_minus * (h_minus + h_plus))
        w1  =  (h_plus - h_minus) / (h_minus * h_plus)
        w2  =  h_minus / (h_plus * (h_minus + h_plus))

        # Second derivative d2V/dS2 ≈ W0 * V_{i-1} + W1 * V_i + W2 * V_B
        W0  =  2.0 / (h_minus * (h_minus + h_plus))
        W1  = -2.0 / (h_minus * h_plus)
        W2  =  2.0 / (h_plus  * (h_minus + h_plus))

        # Black–Scholes operator pieces at S_i
        S = S_i
        sig2S2 = (self.sigma * self.sigma) * (S * S)
        r  = float(self.flat_r_nacc)
        q  = 0.0  # dividend PV escrowed ⇒ PDE with q=0

        # L(V)_i = 0.5*σ^2 S^2 * (W0 V_{i-1} + W1 V_i + W2 V_B)
        #        + (r-q) S     * (w0 V_{i-1} + w1 V_i + w2 V_B)
        #        - r * V_i
        a_im1 = 0.5 * sig2S2 * W0 + (r - q) * S * w0
        a_i   = 0.5 * sig2S2 * W1 + (r - q) * S * w1 - r
        # Constant term from the (known) barrier value V_B = rebate_value
        const_term = (0.5 * sig2S2 * W2 + (r - q) * S * w2) * rebate_value

        return a_im1, a_i, const_term


    def _one_sided_coeffs(self, i_bar: int) -> tuple[float,float,float,float,float,float]:
        """
        Non-symmetric coefficients at the node nearest the barrier.
        Uses step sizes h_minus = S_i - S_{i-1}, h_plus = S_{i+1} - S_i,
        with the φ safeguard (φ=0.05) as suggested by the FIS doc.
        """
        S_im1 = self.S_nodes[i-1]
        S_i   = self.S_nodes[i]
        # distances (positive):
        h_minus = S_i - S_im1                      # grid step to the left
        h_plus  = max(1e-14, barrier - S_i)        # distance from S_i to barrier (non-grid)
        # First derivative:  a V_{i+1} + b V_i + c V_{i-1}, but at barrier there is no V_{i+1};
        # we approximate using the two interior nodes (i and i-1) plus the virtual point at the barrier.
        # The FIS doc gives the compact coefficients below (rearranged form):
        a =  h_minus**2 / (h_plus * (h_plus + h_minus))
        b = (h_plus**2 - h_minus**2) / (h_plus * h_minus)
        c = -h_plus**2 / (h_minus * (h_plus + h_minus))
        # Second derivative: d V_{i+1} + e V_i + f V_{i-1}
        d =  2.0 / (h_plus * (h_plus + h_minus))
        e = -2.0 / (h_plus * h_minus)
        f =  2.0 / (h_minus * (h_plus + h_minus))
        return (a, b, c), (d, e, f)

    def _backward_CN(self,
                    effective_lower_barrier: float | None,
                    effective_upper_barrier: float | None,
                    monitor_steps: dict[int, bool]) -> list[float]:

        N  = len(self.S_nodes) - 1
        M  = self.time_steps
        dt = self.T / M
        r  = float(self.flat_r_nacc)
        q  = 0.0  # escrowed dividends => PDE with q=0

        V = self._terminal_array()

        # Pre-compute which interior index needs a one-sided row (if any)
        idx_lo = self._nearest_barrier_index(effective_lower_barrier, is_up=False) if effective_lower_barrier is not None else None
        idx_up = self._nearest_barrier_index(effective_upper_barrier, is_up=True)  if effective_upper_barrier is not None else None
        rebate = 0.0

        for m in range(M, 0, -1):
            theta = 1.0 if (M - m) < self.rannacher_steps else 0.5

            sub = [0.0]*(N+1); dia = [0.0]*(N+1); sup = [0.0]*(N+1); rhs = [0.0]*(N+1)

            # boundaries (as you already had)
            tau_next = self.T - (m-1)*dt
            if self.option_type == "call":
                rhs[0] = 0.0
                rhs[N] = self.S_nodes[-1] - self.strike * math.exp(-self.flat_r_nacc * tau_next)
            else:
                rhs[0] = self.strike * math.exp(-self.flat_r_nacc * tau_next)
                rhs[N] = 0.0
            dia[0] = 1.0; dia[N] = 1.0

            for i in range(1, N):
                S = self.S_nodes[i]
                sig2S2 = (self.sigma * self.sigma) * (S * S)

                use_lo = (self.barrier_type in ("down-and-out","double-out")) and (idx_lo is not None and i == idx_lo)
                use_up = (self.barrier_type in ("up-and-out","double-out"))   and (idx_up is not None and i == idx_up)

                if use_lo:
                    a_im1, a_i, const_term = self._one_sided_row_coeffs(i, effective_lower_barrier, rebate)
                    # CN assembly with NO 'sup' (no V_{i+1} term) and constant goes to RHS with total +dt
                    sub[i] = -theta * dt * a_im1
                    dia[i] = 1.0 - theta * dt * a_i
                    sup[i] = 0.0
                    rhs[i] = (1.0 + (1.0 - theta) * dt * a_i) * V[i] \
                        + (1.0 - theta) * dt * a_im1 * V[i-1] \
                        + dt * const_term

                elif use_up:
                    a_im1, a_i, const_term = self._one_sided_row_coeffs(i, effective_upper_barrier, rebate)
                    # For an up barrier, the nearest interior node is *below* the barrier,
                    # still the same (i-1, i, B) stencil is used.
                    sub[i] = -theta * dt * a_im1
                    dia[i] = 1.0 - theta * dt * a_i
                    sup[i] = 0.0
                    rhs[i] = (1.0 + (1.0 - theta) * dt * a_i) * V[i] \
                        + (1.0 - theta) * dt * a_im1 * V[i-1] \
                        + dt * const_term

                else:
                    # standard central CN row
                    dS = self.dS
                    r  = float(self.flat_r_nacc); q = 0.0
                    a_minus = 0.5*sig2S2/(dS*dS) - 0.5*(r-q)*S/dS
                    a_zero  = -sig2S2/(dS*dS)    - r
                    a_plus  = 0.5*sig2S2/(dS*dS) + 0.5*(r-q)*S/dS

                    sub[i] = -theta * dt * a_minus
                    dia[i] =  1.0   - theta * dt * a_zero
                    sup[i] = -theta * dt * a_plus
                    rhs[i] = (1.0 + (1.0 - theta) * dt * a_zero) * V[i] \
                        + (1.0 - theta) * dt * a_minus * V[i-1] \
                        + (1.0 - theta) * dt * a_plus  * V[i+1]

                V = self._solve_tridiagonal(sub, dia, sup, rhs)

                # barrier projection at monitoring steps (unchanged)
                k_after = m - 1
                if k_after in monitor_steps:
                    self._apply_knockout(V, effective_lower_barrier, effective_upper_barrier)
                # per FIS: do NOT trigger a Rannacher restart because of the barrier projection

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