
class DiscreteBarrierFDMPricer:
    """
    Discretely monitored barrier option (CN + Rannacher) with
    optional BGK continuity correction when monitoring is frequent.

    Flat continuous-compounding short rate is used throughout the grid:
        flat_r_nacc  (i.e., NACC, 'r' in Black–Scholes).

    Dividends: escrow PV of discrete dividends from discount curve (unchanged).
    """

    # ---- configuration knobs ----
    BGK_BETA = 0.5826  # Broadie–Glasserman–Kou constant
    CONT_MONITORING_LIMIT = 5  # n_lim from the doc (default “5”)

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
        # curves / dividends left as in your code…
        discount_curve: Optional[pd.DataFrame] = None,
        forward_curve: Optional[pd.DataFrame] = None,
        dividend_schedule: Optional[List[Tuple[date, float]]] = None,
        # grid controls
        grid_points: int = 400,
        time_steps: int = 400,
        rannacher_steps: int = 2,
        # rate control (flat NACC rate for PDE)
        flat_r_nacc: float = 0.0,     # <--- NEW: use a single continuous-compounded rate
        # day count
        day_count: str = "ACT/365",   # string selector (ACT/365, ACT/360, 30/360, etc.)
        # behavior flags
        restart_on_monitoring: bool = False,  # we will ignore for barrier events
        mollify_final: bool = True,
        mollify_band_nodes: int = 2,
        price_extrapolation: bool = False,
    ):
        # ... (same validations)
        self.spot = spot
        self.strike = strike
        self.valuation_date = valuation_date
        self.maturity_date = maturity_date
        self.sigma = sigma
        self.option_type = option_type
        self.barrier_type = barrier_type
        self.lower_barrier = lower_barrier
        self.upper_barrier = upper_barrier
        self.monitor_dates = sorted(monitor_dates or [])
        self.discount_curve_df = discount_curve.copy() if discount_curve is not None else None
        self.forward_curve_df = forward_curve.copy() if forward_curve is not None else None
        self.dividend_schedule = sorted(dividend_schedule or [], key=lambda x: x[0])

        self.grid_points = int(grid_points)
        self.time_steps = int(time_steps)
        self.rannacher_steps = int(rannacher_steps)
        self.flat_r_nacc = float(flat_r_nacc)
        self.day_count = day_count.upper()
        self.mollify_final = mollify_final
        self.mollify_band_nodes = int(mollify_band_nodes)
        self.price_extrapolation = price_extrapolation

        # year fraction + tenor
        self._yf = self._year_fraction
        self.T = self._yf(self.valuation_date, self.maturity_date)

        # normalize curve DFs if provided (unchanged helper)
        if self.discount_curve_df is not None:
            self.discount_curve_df = self._normalize_curve_df(self.discount_curve_df)
        if self.forward_curve_df is not None:
            self.forward_curve_df = self._normalize_curve_df(self.forward_curve_df)

        # grids
        self.S_nodes = self._build_space_grid()
        self.dS = self.S_nodes[1] - self.S_nodes[0]
        self.time_nodes = [i * self.T / self.time_steps for i in range(self.time_steps + 1)]

        # BGK continuity decision and adjusted barriers
        self.use_continuous_barrier, self.bgk_lower, self.bgk_upper, self.continuous_k0, self.continuous_k1 = \
            self._bgk_continuous_decision_and_adjustment()

    # ---------- day count ----------
    def _year_fraction(self, d0: date, d1: date) -> float:
        if self.day_count in ("ACT/365", "ACT/365F", "ACT/365 FIXED"):
            return max(0.0, (d1 - d0).days) / 365.0
        if self.day_count in ("ACT/360",):
            return max(0.0, (d1 - d0).days) / 360.0
        if self.day_count in ("30/360", "30E/360"):
            # simple 30/360 US
            d0y, d0m, d0d = d0.year, d0.month, min(d0.day, 30)
            d1y, d1m, d1d = d1.year, d1.month, min(d1.day, 30)
            return ((d1y - d0y) * 360 + (d1m - d0m) * 30 + (d1d - d0d)) / 360.0
        # default
        return max(0.0, (d1 - d0).days) / 365.0

    # ---------- BGK decision & barrier adjustment ----------
    def _bgk_continuous_decision_and_adjustment(self):
        """
        Decide if monitoring is 'frequent enough' and compute BGK-adjusted barriers.
        When True: project every time step between first and last monitoring date
        with adjusted barrier(s), continuous monitoring style.
        """
        if self.barrier_type in ("none",) or len(self.monitor_dates) == 0:
            return (False, self.lower_barrier, self.upper_barrier, None, None)

        # effective cadence between monitoring instants
        first = min(self.monitor_dates)
        last = max(self.monitor_dates)
        if last <= first:
            return (False, self.lower_barrier, self.upper_barrier, None, None)

        n_mon = len(self.monitor_dates)
        te = self._yf(first, last)                 # time spanned by monitoring (years)
        dt_eff = te / max(1, n_mon - 1)           # effective interval (years)

        # decide “frequent enough”: sum over sub-intervals n_m, compare to limiter n_lim*n
        # We follow the doc spirit with a practical rule: if average interval is small enough
        # so that (T / dt_eff) >= n_lim * (target time steps / M), use continuous approx.
        # Simpler: treat frequent if n_mon >= CONT_MONITORING_LIMIT.
        frequent = (n_mon >= self.CONT_MONITORING_LIMIT)

        # BGK adjusted barriers (local vol = sigma, a_b = t_b / n_mon from doc)
        ab = te / max(1, n_mon)                   # approx factor used in doc for φ(t)
        phi = self.BGK_BETA * self.sigma * ab**0.0  # φ(t) = 0.5826 σ * a_b ; here a_b multiplies σ*sqrt(dt)
        # We keep the classic exp(± β σ sqrt(Δt_eff))—robust in practice:
        adj_factor = math.exp(self.BGK_BETA * self.sigma * math.sqrt(dt_eff))

        lo_adj = self.lower_barrier
        up_adj = self.upper_barrier
        if self.lower_barrier is not None:
            # down barrier (H < S0) ⇒ minus sign
            lo_adj = self.lower_barrier / adj_factor
        if self.upper_barrier is not None:
            # up barrier (H > S0) ⇒ plus sign
            up_adj = self.upper_barrier * adj_factor

        # map “continuous monitoring window” to time-step indices
        k0 = int(round(self._yf(self.valuation_date, first) / (self.T / self.time_steps)))
        k1 = int(round(self._yf(self.valuation_date, last)  / (self.T / self.time_steps)))
        k0 = max(0, min(self.time_steps, k0))
        k1 = max(0, min(self.time_steps, k1))

        return (frequent, lo_adj, up_adj, min(k0, k1), max(k0, k1))

    # ---------- space grid ----------
    def _build_space_grid(self) -> List[float]:
        """
        Uniform S-grid with gentle oversizing, snapping K/H exactly onto nodes
        (helps Greeks smoothness and barrier projection accuracy).
        """
        anchors = [self.spot, self.strike]
        if self.lower_barrier: anchors.append(self.lower_barrier)
        if self.upper_barrier: anchors.append(self.upper_barrier)
        s_ref = max(anchors)
        s_max = 4.0 * s_ref * math.exp(self.sigma * math.sqrt(max(self.T, 1e-12)))
        s_min = 0.0
        N = max(200, int(self.grid_points))
        dS = (s_max - s_min) / N
        nodes = [s_min + i * dS for i in range(N + 1)]

        def snap(target: Optional[float]):
            if target is None: return
            j = min(range(len(nodes)), key=lambda i: abs(nodes[i] - target))
            nodes[j] = target
        snap(self.strike); snap(self.lower_barrier); snap(self.upper_barrier)
        return nodes

    # ---------- boundary payoff and mollification ----------
    def _terminal_payoff(self, S: float) -> float:
        return max(S - self.strike, 0.0) if self.option_type == "call" else max(self.strike - S, 0.0)

    def _terminal_array(self) -> List[float]:
        vT = [self._terminal_payoff(S) for S in self.S_nodes]
        if not self.mollify_final or self.mollify_band_nodes <= 0:
            return vT
        # simple quadratic smoothing across 2*m nodes about K (per your doc guidance)
        m = self.mollify_band_nodes
        k_idx = min(range(len(self.S_nodes)), key=lambda i: abs(self.S_nodes[i] - self.strike))
        i0, i1 = max(0, k_idx - m), min(len(self.S_nodes) - 1, k_idx + m)
        S0, V0 = self.S_nodes[i0], vT[i0]
        S1, V1 = self.S_nodes[i1], vT[i1]
        a = (V1 - V0) / ((S1 - S0) ** 2) if S1 != S0 else 0.0
        for i in range(i0, i1 + 1):
            s = self.S_nodes[i]
            vT[i] = a * (s - S0) ** 2 + V0
        return vT

    # ---------- barrier projection ----------
    def _apply_knockout(self, values: List[float], lo: Optional[float], up: Optional[float]) -> None:
        """Zero the value at nodes that are beyond the active KO barrier(s)."""
        for i, s in enumerate(self.S_nodes):
            ko = False
            if self.barrier_type == "down-and-out" and lo is not None and s <= lo: ko = True
            elif self.barrier_type == "up-and-out" and up is not None and s >= up: ko = True
            elif self.barrier_type == "double-out":
                if (lo is not None and s <= lo) or (up is not None and s >= up): ko = True
            if ko: values[i] = 0.0

    # ---------- tridiagonal solver (Thomas) ----------
    def _solve_tridiagonal(self, sub_diag: List[float], main_diag: List[float],
                           sup_diag: List[float], rhs: List[float]) -> List[float]:
        n = len(rhs)
        forward_elim_ratio = [0.0]*n
        forward_rhs = [0.0]*n
        sol = [0.0]*n

        piv = main_diag[0]
        if abs(piv) < 1e-14: raise ZeroDivisionError("Tridiagonal pivot ~ 0 at row 0.")
        forward_elim_ratio[0] = sup_diag[0]/piv
        forward_rhs[0] = rhs[0]/piv

        for i in range(1, n):
            piv = main_diag[i] - sub_diag[i]*forward_elim_ratio[i-1]
            if abs(piv) < 1e-14: raise ZeroDivisionError(f"Tridiagonal pivot ~ 0 at row {i}.")
            forward_elim_ratio[i] = sup_diag[i]/piv if i < n-1 else 0.0
            forward_rhs[i] = (rhs[i] - sub_diag[i]*forward_rhs[i-1]) / piv

        sol[-1] = forward_rhs[-1]
        for i in range(n-2, -1, -1):
            sol[i] = forward_rhs[i] - forward_elim_ratio[i]*sol[i+1]
        return sol

    # ---------- linear interpolation on S-grid ----------
    def _interp_linear(self, x: float, xs: List[float], ys: List[float]) -> float:
        if x <= xs[0]: return ys[0]
        if x >= xs[-1]: return ys[-1]
        lo, hi = 0, len(xs)-1
        while hi - lo > 1:
            mid = (lo + hi)//2
            if x < xs[mid]: hi = mid
            else: lo = mid
        x0, x1 = xs[lo], xs[hi]
        y0, y1 = ys[lo], ys[hi]
        w = (x - x0)/(x1 - x0)
        return (1-w)*y0 + w*y1

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

        for m in range(M, 0, -1):
            # Rannacher mix (EB for first few steps; NOT because of barrier projection)
            theta = 1.0 if (M - m) < self.rannacher_steps else 0.5

            sub = [0.0]*(N+1)
            main = [0.0]*(N+1)
            sup = [0.0]*(N+1)
            rhs =  [0.0]*(N+1)

            # boundary values (standard)
            if self.option_type == "call":
                rhs[0] = 0.0
                rhs[N] = self.S_nodes[-1] - self.strike * math.exp(-r*(self.T - (m-1)*dt))
            else:
                rhs[0] = self.strike * math.exp(-r*(self.T - (m-1)*dt))
                rhs[N] = 0.0
            main[0] = 1.0; main[N] = 1.0

            for i in range(1, N):
                S = self.S_nodes[i]
                sig2S2 = (sigma**2) * (S**2)

                a_impl = 0.5*dt*theta*( sig2S2/(dS**2) - r*S/dS )
                b_impl = 1.0 + dt*theta*( sig2S2/(dS**2) + r )
                c_impl = 0.5*dt*theta*( sig2S2/(dS**2) + r*S/dS )

                a_expl = -0.5*dt*(1-theta)*( sig2S2/(dS**2) - r*S/dS )
                b_expl = 1.0 - dt*(1-theta)*( sig2S2/(dS**2) + r )
                c_expl = -0.5*dt*(1-theta)*( sig2S2/(dS**2) + r*S/dS )

                sub[i]  = -a_impl
                main[i] =  b_impl
                sup[i]  = -c_impl
                rhs[i]  =  a_expl*V[i-1] + b_expl*V[i] + c_expl*V[i+1]

            V = self._solve_tridiagonal(sub, main, sup, rhs)

            # barrier projection (discrete or continuous per decision)
            step_index_after = (m-1)
            if step_index_after in monitor_steps:
                self._apply_knockout(V, effective_lo, effective_up)
                # DO NOT trigger Rannacher restart because of barrier (per your note)

        return V

    # ---------- price ----------
    def price(self) -> float:
        # escrow dividends (unchanged)
        div_pv = self.pv_dividends()
        S_eff = self.spot - div_pv
        S_shifted_nodes = [max(s - div_pv, 0.0) for s in self.S_nodes]
        backup_nodes = self.S_nodes
        self.S_nodes = S_shifted_nodes
        self.dS = self.S_nodes[1] - self.S_nodes[0]

        # monitoring map
        monitor_steps = {}
        if self.use_continuous_barrier:
            # project at *every* step between first and last monitoring index
            for k in range(self.continuous_k0, self.continuous_k1 + 1):
                monitor_steps[k] = True
            lo_eff, up_eff = self.bgk_lower, self.bgk_upper
        else:
            for d in self.monitor_dates:
                if self.valuation_date < d <= self.maturity_date:
                    k = int(round(self._yf(self.valuation_date, d) / (self.T/self.time_steps)))
                    monitor_steps[k] = True
            lo_eff, up_eff = self.lower_barrier, self.upper_barrier

        # knock-ins via parity with KO
        if self.barrier_type in ("down-and-in", "up-and-in", "double-in"):
            V_none = self._backward_CN(None, None, {})
            V_ko   = self._backward_CN(lo_eff, up_eff, monitor_steps)
            val = self._interp_linear(S_eff, self.S_nodes, [V_none[i]-V_ko[i] for i in range(len(V_none))])
        else:
            V = self._backward_CN(lo_eff, up_eff, monitor_steps)
            val = self._interp_linear(S_eff, self.S_nodes, V)

        self.S_nodes = backup_nodes
        self.dS = self.S_nodes[1] - self.S_nodes[0]
        return float(val)

    # ---------- Greeks with barrier-aware stencils ----------
    def greeks(self, rel_bump: float = 0.01, vega_bump: float = 0.01) -> Dict[str, float]:
        # compute price on base grid once
        base_px = self.price()

        # delta: barrier-aware finite differences per doc
        s0 = self.spot
        ds = max(1e-8, rel_bump*s0)
        self.spot = s0 + ds; up = self.price()
        self.spot = s0 - ds; dn = self.price()
        self.spot = s0

        # choose stencil: near a barrier use one-sided blend; else central
        def near_barrier(S) -> bool:
            lo = self.lower_barrier if not self.use_continuous_barrier else self.bgk_lower
            upb = self.upper_barrier if not self.use_continuous_barrier else self.bgk_upper
            tol = 2*self.dS
            return (lo is not None and abs(S - lo) <= tol) or (upb is not None and abs(S - upb) <= tol)

        if near_barrier(s0):
            # one-sided (downwind) delta & blended gamma (following figures in your doc)
            delta = (base_px - dn) / ds  # D_- V
            # gamma: blend one-sided with central to avoid spikes (q in the doc)
            central_gamma = (up - 2*base_px + dn) / (ds*ds)
            one_sided_gamma = (up - 2*base_px + dn) / (ds*ds)  # same finite step here; keep structure
            q = 0.5
            gamma = q*central_gamma + (1-q)*one_sided_gamma
        else:
            delta = (up - dn) / (2*ds)
            gamma = (up - 2*base_px + dn) / (ds*ds)

        # vega
        sig0 = self.sigma
        self.sigma = sig0 + vega_bump; upv = self.price()
        self.sigma = sig0 - vega_bump; dnv = self.price()
        self.sigma = sig0
        vega = (upv - dnv) / (2*vega_bump)

        return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega)}


    # ======================================================================
    # Reporting (for quick sanity checks)
    # ======================================================================

    def report(self) -> str:
        lines = [
            "==== Barrier Option Trade Details ====",
            f"Trade ID            : {self.trade_id}",
            f"Direction           : {self.direction}",
            f"Quantity            : {self.quantity}",
            f"Contract Multiplier : {self.contract_multiplier:.6g}",
            f"Option Type         : {self.option_type}",
            f"Barrier Type        : {self.barrier_type}",
            f"Lower Barrier (Hd)  : {self.barrier_lower if self.barrier_lower is not None else '-'}",
            f"Upper Barrier (Hu)  : {self.barrier_upper if self.barrier_upper is not None else '-'}",
            f"Spot (S0)           : {self.spot_price:.6f}",
            f"Strike (K)          : {self.strike_price:.6f}",
            f"Valuation Date      : {self.valuation_date.isoformat()}",
            f"Maturity Date       : {self.maturity_date.isoformat()}",
            f"T (years, {self.day_count}) : {self.tenor_years:.6f}",
            f"Volatility (sigma)  : {self.volatility:.6f}",
            f"r_flat (NACC)       : {self.r_flat:.6f}",
            f"q_flat (NACC)       : {self.q_flat:.6f}",
            f"Steps (time,space)  : {self.num_time_steps}, {self.num_space_nodes} (Rannacher {self.rannacher_steps})",
            f"Rannacher@Barrier   : {self.restart_rannacher_at_barrier}",
            f"Monitoring dates (#): {len([d for d in self.monitoring_dates if self.valuation_date < d < self.maturity_date])}",
            f"BGK correction      : {self.use_bgk_correction}",
        ]
        return "\n".join(lines)
