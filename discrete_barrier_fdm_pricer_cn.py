# discrete_barrier_fdm_pricer_cn.py
# -*- coding: utf-8 -*-
"""
CN finite-difference pricer for European discrete-barrier options in S-space.

Key features
------------
- Price-space Black–Scholes PDE:
    V_t + 0.5 σ² S² V_SS + (r-q) S V_S - r V = 0
- Crank–Nicolson with Rannacher start (θ=1 for first steps), and **Rannacher restarts**
  after each discrete KO projection (per FIS guidance).
- **Discrete monitoring**: KO projection at monitor times only. KI via parity.
- **Barrier-aware PDE row** at the interior node adjacent to the KO barrier:
  Uses a **non-symmetric** (barrier-referenced) second/first derivative with distances
  h_- = |S_i - B| and h_+ = S_{i+1} - S_i (mirror for up-and-out).
  The barrier “ghost” value is taken from the **previous time level** (CN split), so
  the tri-diagonal remains intact (ghost term goes to RHS).
- **Greeks (Δ, Γ)** at t=0 use the same barrier-aware stencil; central elsewhere,
  with a one-sided Δ in the first cell off the barrier and blended Δ in the second.
- **Time–space coupling guard**: increases sub-steps in any monitor interval whose
  local diffusion number exceeds a target (keeps accuracy stable).
- Optional **numerical theta** from a one-step bump in time near 0 (robust).

This is built to match the spirit of FIS’ FCA3761-09 treatment but stays in S-space.
"""

from typing import List, Tuple, Dict, Optional, Literal
from datetime import date
import math

OptionType = Literal["call", "put"]
BarrierType = Literal[
    "none",
    "down-and-out", "up-and-out", "double-out",
    "down-and-in",  "up-and-in",  "double-in",
]

class DiscreteBarrierFDMPricer:
    # --------------------------- init ---------------------------
    def __init__(
        self,
        # instrument
        spot: float,
        strike: float,
        valuation_date: date,
        maturity_date: date,
        volatility: float,
        flat_rate_nacc: float,
        dividend_yield: float,
        option_type: OptionType,
        # barriers
        barrier_type: BarrierType = "none",
        lower_barrier: Optional[float] = None,
        upper_barrier: Optional[float] = None,
        monitoring_dates: Optional[List[date]] = None,
        rebate_amount: float = 0.0,
        rebate_at_hit: bool = True,
        # status flags
        already_hit: bool = False,
        already_in: bool = False,
        # numerics
        num_space_nodes: int = 600,       # S-grid nodes (N+1); kept fixed
        num_time_steps: int = 800,        # total time steps M (distributed over monitors)
        rannacher_steps: int = 2,
        min_substeps_between_monitors: int = 1,
        grid_type: Literal["uniform", "sinh"] = "uniform",
        sinh_alpha: float = 1.75,         # controls clustering when grid_type="sinh"
        # time-space coupling (guard)
        lambda_diff_target: float = 0.5,  # target diffusion number ~ [0.3..0.6]
        # misc
        day_count: str = "ACT/365",
        # greeks near barrier
        use_one_sided_greeks_near_barrier: bool = True,
        barrier_safety_cells: int = 2,
        compute_numerical_theta: bool = False,
    ):
        self.S0 = float(spot)
        self.K = float(strike)
        self.valuation_date = valuation_date
        self.maturity_date  = maturity_date
        self.sigma = float(volatility)
        self.r = float(flat_rate_nacc)
        self.q = float(dividend_yield)
        self.option_type = option_type

        self.barrier_type  = barrier_type
        self.lower_barrier = lower_barrier
        self.upper_barrier = upper_barrier
        self.monitoring_dates = sorted(monitoring_dates or [])
        self.rebate_amount = float(rebate_amount)
        self.rebate_at_hit = bool(rebate_at_hit)

        self.already_hit = bool(already_hit)
        self.already_in  = bool(already_in)

        self.N = max(200, int(num_space_nodes))        # ensure enough points
        self.M = max(1, int(num_time_steps))
        self.rannacher = int(rannacher_steps)
        self.min_substeps = max(1, int(min_substeps_between_monitors))
        self.grid_type = grid_type
        self.sinh_alpha = float(sinh_alpha)

        self.lambda_diff_target = float(lambda_diff_target)
        self.day_count = day_count.upper()

        self.use_one_sided = bool(use_one_sided_greeks_near_barrier)
        self.barrier_safety_cells = int(barrier_safety_cells)
        self.compute_numerical_theta = bool(compute_numerical_theta)

        self.T = self._year_fraction(self.valuation_date, self.maturity_date)
        self.S_nodes = self._build_space_grid()   # stores S[0..N]
        self.monitor_times = self._build_monitor_times_exact()  # includes 0 and T

    # --------------------------- utilities ---------------------------
    def _year_fraction(self, d0: date, d1: date) -> float:
        days = max(0, (d1 - d0).days)
        if self.day_count in ("ACT/360",):
            return days/360.0
        if self.day_count in ("30/360", "30E/360"):
            y0,m0,d0_ = d0.year, d0.month, min(d0.day,30)
            y1,m1,d1_ = d1.year, d1.month, min(d1.day,30)
            return ((y1-y0)*360 + (m1-m0)*30 + (d1_-d0_))/360.0
        return days/365.0

    def _build_monitor_times_exact(self) -> List[float]:
        times = [0.0]
        for d in self.monitoring_dates:
            if self.valuation_date <= d <= self.maturity_date:
                t = self._year_fraction(self.valuation_date, d)
                if 0.0 <= t <= self.T:
                    times.append(t)
        if times[-1] < self.T - 1e-14:
            times.append(self.T)
        return sorted(set(times))

    def _build_space_grid(self) -> List[float]:
        anchors = [self.S0, self.K]
        if self.lower_barrier is not None: anchors.append(self.lower_barrier)
        if self.upper_barrier is not None: anchors.append(self.upper_barrier)
        Sref = max(anchors)
        Smax = 4.5 * Sref * math.exp(self.sigma * math.sqrt(max(self.T, 1e-12)))
        N = self.N
        if self.grid_type == "uniform":
            dS = Smax / N
            nodes = [i*dS for i in range(N+1)]
        else:
            # sinh-stretch centered at KO barrier if present, else at Sref
            if self.barrier_type in ("down-and-out","double-out") and self.lower_barrier:
                Sc = self.lower_barrier
            elif self.barrier_type in ("up-and-out","double-out") and self.upper_barrier:
                Sc = self.upper_barrier
            else:
                Sc = Sref
            a = self.sinh_alpha
            xs = [-1.0 + 2.0*i/N for i in range(N+1)]
            span = Smax - Sc
            scale = span / max(1e-12, math.sinh(a))
            nodes = [Sc + scale*math.sinh(a*x) for x in xs]
            shift = -min(0.0, min(nodes))
            if shift>0: nodes = [s+shift for s in nodes]

        # snap K / barriers
        def snap(x: Optional[float]):
            if x is None: return
            j = min(range(len(nodes)), key=lambda i: abs(nodes[i]-x))
            nodes[j] = float(x)
        snap(self.K); snap(self.lower_barrier); snap(self.upper_barrier)
        return nodes

    # ---------------- payoff / KO projection ----------------
    def _terminal_payoff(self, s_nodes: List[float]) -> List[float]:
        if self.option_type == "call":
            return [max(s-self.K,0.0) for s in s_nodes]
        return [max(self.K-s,0.0) for s in s_nodes]

    def _apply_KO_projection(self, V: List[float], s_nodes: List[float], tau_left: float) -> None:
        if self.barrier_type in ("none","down-and-in","up-and-in","double-in"):
            return
        lo, up = self.lower_barrier, self.upper_barrier
        rebate = self.rebate_amount if self.rebate_at_hit else self.rebate_amount*math.exp(-self.r*tau_left)
        for i,s in enumerate(s_nodes):
            out=False
            if self.barrier_type=="down-and-out" and lo is not None and s<=lo: out=True
            elif self.barrier_type=="up-and-out" and up is not None and s>=up: out=True
            elif self.barrier_type=="double-out":
                if (lo is not None and s<=lo) or (up is not None and s>=up): out=True
            if out: V[i]=rebate

    # ---------------- tridiagonal solver ----------------
    @staticmethod
    def _solve_tridiagonal(a: List[float], b: List[float], c: List[float], d: List[float]) -> List[float]:
        n=len(d); cp=[0.0]*n; dp=[0.0]*n; x=[0.0]*n
        beta=b[0]; 
        if abs(beta)<1e-14: beta=1e-14
        cp[0]=c[0]/beta; dp[0]=d[0]/beta
        for i in range(1,n):
            beta=b[i]-a[i]*cp[i-1]
            if abs(beta)<1e-14: beta=1e-14
            cp[i]=(c[i]/beta) if i<n-1 else 0.0
            dp[i]=(d[i]-a[i]*dp[i-1])/beta
        x[-1]=dp[-1]
        for i in range(n-2,-1,-1):
            x[i]=dp[i]-cp[i]*x[i+1]
        return x

    # ---------------- CN step over [t0,t1] ----------------
    def _cn_subinterval(self, s_nodes: List[float], V: List[float],
                        t0: float, t1: float, theta: float,
                        rannacher_left_steps: int, m_steps: int) -> Tuple[List[float], float]:
        r, q, sig = self.r, self.q, self.sigma
        N = len(s_nodes)-1
        L = t1 - t0
        m = max(1, int(m_steps))
        dt = L / m
        last_dt_at_zero = None

        mu = r - q  # drift coefficient (in S-space)
        for step in range(m):
            use_theta = 1.0 if (rannacher_left_steps > 0) else theta
            if rannacher_left_steps > 0:
                rannacher_left_steps -= 1

            tau_left = t0 + (m-step-1)*dt  # time to maturity *after* this step

            sub=[0.0]*(N+1); main=[0.0]*(N+1); sup=[0.0]*(N+1); rhs=[0.0]*(N+1)

            # Dirichlet boundaries at after-step time
            if self.option_type=="call":
                rhs[0] = 0.0
                rhs[N] = s_nodes[-1]*math.exp(-q*tau_left) - self.K*math.exp(-r*tau_left)
            else:
                rhs[0] = self.K*math.exp(-r*tau_left)
                rhs[N] = 0.0
            main[0]=1.0; main[N]=1.0

            # Identify barrier-adjacent interior index (KO only)
            ko_type = self.barrier_type
            have_ko = ko_type in ("down-and-out","up-and-out","double-out")
            i_adj = None; B = None; side = None
            if have_ko:
                if ko_type in ("down-and-out","double-out") and self.lower_barrier is not None:
                    B = self.lower_barrier; side="down"
                    # i_adj: first node strictly above B
                    j = min(range(N+1), key=lambda k: abs(s_nodes[k]-B))
                    j = max(1, min(N-1, j))
                    if s_nodes[j] <= B and j < N: j += 1
                    i_adj = j
                elif ko_type in ("up-and-out","double-out") and self.upper_barrier is not None:
                    B = self.upper_barrier; side="up"
                    # i_adj: last node strictly below B
                    j = min(range(N+1), key=lambda k: abs(s_nodes[k]-B))
                    j = max(1, min(N-1, j))
                    if s_nodes[j] >= B and j > 0: j -= 1
                    i_adj = j

            # Precompute "barrier ghost" value from previous time level by linear interpolation
            def barrier_value_prev() -> Optional[float]:
                if i_adj is None or B is None: return None
                if side=="down":
                    j=i_adj; S1=s_nodes[j]; S2=s_nodes[j+1]
                else: # "up"
                    j=i_adj; S1=s_nodes[j-1]; S2=s_nodes[j]
                if S2==S1: return V[j]
                t = (B - S1) / (S2 - S1)
                if side=="down":
                    V1, V2 = V[j], V[j+1]
                else:
                    V1, V2 = V[j-1], V[j]
                return (1.0-t)*V1 + t*V2

            VB_prev = barrier_value_prev()

            # Assemble interior rows
            for i in range(1,N):
                Si = s_nodes[i]
                if i_adj is not None and i == i_adj and VB_prev is not None:
                    # ---- Barrier-aware, non-symmetric row at node adjacent to KO barrier ----
                    if side=="down":
                        h_minus = max(1e-14, Si - B)
                        h_plus  = max(1e-14, s_nodes[i+1] - Si)
                        # coefficients for D1 and D2 using (B, VB_prev), (Si,Vi), (Si+h_plus, V_{i+1})
                        a1 = -h_plus/(h_minus*(h_minus+h_plus))
                        a2 = (h_plus - h_minus)/(h_minus*h_plus)
                        a3 =  h_minus/(h_plus*(h_minus+h_plus))
                        d1 =  2.0/(h_minus*(h_minus+h_plus))
                        d2 = -2.0/(h_minus*h_plus)
                        d3 =  2.0/(h_plus*(h_minus+h_plus))
                        # Implicit coefficients for V_i, V_{i+1}
                        A = 0.5*sig*sig*Si*Si
                        c_impl = use_theta*dt*( A*d3 + mu*Si*a3 )
                        b_impl = 1.0 + use_theta*dt*( A*d2 + mu*Si*a2 + r )
                        # Explicit (previous-level) weights for V_i, V_{i+1}
                        c_expl = (1.0-use_theta)*dt*( A*d3 + mu*Si*a3 )
                        b_expl = 1.0 - (1.0-use_theta)*dt*( A*d2 + mu*Si*a2 + r )
                        # Barrier-ghost term goes entirely to RHS with total dt
                        ghost_term = dt * ( A*d1 + mu*Si*a1 ) * VB_prev
                        sub[i]=0.0; main[i]=b_impl; sup[i]=-c_impl
                        rhs[i]= b_expl*V[i] + c_expl*V[i+1] + ghost_term
                    else:
                        # side == "up" (up-and-out): use nodes (S_{i-1}, S_i) and barrier at B
                        h_plus  = max(1e-14, B - Si)            # to barrier on the right
                        h_minus = max(1e-14, Si - s_nodes[i-1]) # to left node
                        # Flip roles: use (S_{i-1}, Vi-1), (Si, Vi), (B, VB_prev)
                        a1 = -h_plus/(h_minus*(h_minus+h_plus))     # coeff for V_{i-1}
                        a2 = (h_plus - h_minus)/(h_minus*h_plus)    # coeff for V_i
                        a3 =  h_minus/(h_plus*(h_minus+h_plus))     # coeff for V_B
                        d1 =  2.0/(h_minus*(h_minus+h_plus))        # for V_{i-1}
                        d2 = -2.0/(h_minus*h_plus)                  # for V_i
                        d3 =  2.0/(h_plus*(h_minus+h_plus))         # for V_B
                        A = 0.5*sig*sig*Si*Si
                        a_impl = use_theta*dt*( A*d1 + mu*Si*a1 )
                        b_impl = 1.0 + use_theta*dt*( A*d2 + mu*Si*a2 + r )
                        a_expl = (1.0-use_theta)*dt*( A*d1 + mu*Si*a1 )
                        b_expl = 1.0 - (1.0-use_theta)*dt*( A*d2 + mu*Si*a2 + r )
                        ghost_term = dt * ( A*d3 + mu*Si*a3 ) * VB_prev
                        sub[i]=-a_impl; main[i]=b_impl; sup[i]=0.0
                        rhs[i]= a_expl*V[i-1] + b_expl*V[i] + ghost_term
                else:
                    # ---- Standard non-uniform CN (Tavella–Randall) ----
                    h1 = s_nodes[i]   - s_nodes[i-1]
                    h2 = s_nodes[i+1] - s_nodes[i]
                    A = 0.5*sig*sig*Si*Si
                    # implicit (θ part)
                    aI = use_theta*dt*( A*(2.0/(h1*(h1+h2))) - mu*Si*(1.0/(2.0*h1)) )
                    bI = 1.0 + use_theta*dt*( A*(2.0/(h1*h2)) + r )
                    cI = use_theta*dt*( A*(2.0/(h2*(h1+h2))) + mu*Si*(1.0/(2.0*h2)) )
                    # explicit (1-θ part)
                    aE = (1.0-use_theta)*dt*( A*(2.0/(h1*(h1+h2))) - mu*Si*(1.0/(2.0*h1)) )
                    bE = 1.0 - (1.0-use_theta)*dt*( A*(2.0/(h1*h2)) + r )
                    cE = (1.0-use_theta)*dt*( A*(2.0/(h2*(h1+h2))) + mu*Si*(1.0/(2.0*h2)) )
                    sub[i]  = -aI
                    main[i] =  bI
                    sup[i]  = -cI
                    rhs[i]  =  aE*V[i-1] + bE*V[i] + cE*V[i+1]

            V = self._solve_tridiagonal(sub, main, sup, rhs)
            if abs(t0) < 1e-14 and step == m-1:
                last_dt_at_zero = dt

        return V, (last_dt_at_zero if last_dt_at_zero is not None else dt)

    # ---------------- time partition (with coupling guard) ----------------
    def _time_subgrid_counts(self) -> List[int]:
        """Split M across monitor spans; then enforce diffusion-number target by increasing mk as needed."""
        spans = [self.monitor_times[i+1]-self.monitor_times[i] for i in range(len(self.monitor_times)-1)]
        total = sum(spans) or 1.0
        mk = [max(self.min_substeps, int(round(self.M*(L/total)))) for L in spans]
        # fix rounding to sum=M
        diff = self.M - sum(mk); j=0
        while diff!=0 and mk:
            k = j % len(mk)
            if diff>0: mk[k]+=1; diff-=1
            else:
                if mk[k] > self.min_substeps: mk[k]-=1; diff+=1
            j+=1

        # --- coupling guard: ensure a*dt / dS^2 <= lambda_diff_target near S0/barrier ---
        # pick reference location: closest of S0, K, active KO barrier
        refs = [self.S0, self.K]
        if self.barrier_type in ("down-and-out","double-out") and self.lower_barrier: refs.append(self.lower_barrier)
        if self.barrier_type in ("up-and-out","double-out") and self.upper_barrier: refs.append(self.upper_barrier)
        Sref = min(self.S_nodes, key=lambda s: abs(s - max(refs)))
        # local dS (min of the two adjacent spacings around Sref)
        idx = min(range(len(self.S_nodes)), key=lambda i: abs(self.S_nodes[i]-Sref))
        idx = max(1, min(len(self.S_nodes)-2, idx))
        dS_local = min(self.S_nodes[idx]-self.S_nodes[idx-1], self.S_nodes[idx+1]-self.S_nodes[idx])
        a_ref = 0.5*self.sigma*self.sigma*Sref*Sref

        for k in range(len(spans)):
            Lk = spans[k]; mk_k = max(1, mk[k]); dtk = Lk/mk_k
            while a_ref*dtk/(dS_local*dS_local) > self.lambda_diff_target:
                mk_k += 1
                dtk = Lk/mk_k
            mk[k] = mk_k
        return mk

    # ---------------- full backward run ----------------
    def _run_backward(self, s_nodes: List[float], apply_barrier: bool) -> Tuple[List[float], float]:
        N = len(s_nodes)-1
        V = self._terminal_payoff(s_nodes)

        # t=0 status flags
        if apply_barrier:
            if self.barrier_type in ("down-and-out","up-and-out","double-out") and self.already_hit:
                instant = self.rebate_amount if self.rebate_at_hit else self.rebate_amount*math.exp(-self.r*self.T)
                return [instant]*(N+1), 0.0
            if self.barrier_type in ("down-and-in","up-and-in","double-in") and self.already_in:
                apply_barrier = False

        theta = 0.5
        subcounts = self._time_subgrid_counts()
        rannacher_left = self.rannacher
        dt_last_global = None

        for k in range(len(self.monitor_times)-1, 0, -1):
            t0 = self.monitor_times[k-1]
            t1 = self.monitor_times[k]
            m_steps = subcounts[k-1] if k-1 < len(subcounts) else 1

            V, dt_last = self._cn_subinterval(s_nodes, V, t0, t1, theta, rannacher_left, m_steps)
            rannacher_left = max(0, rannacher_left - m_steps)

            if apply_barrier and (abs(t0) > 1e-14):
                self._apply_KO_projection(V, s_nodes, tau_left=t0)
                # ---- Rannacher restart after projection ----
                rannacher_left = max(rannacher_left, self.rannacher)

            if abs(t0) < 1e-14:
                dt_last_global = dt_last

        if apply_barrier and self.barrier_type in ("down-and-in","up-and-in","double-in"):
            V_van, _ = self._run_backward(s_nodes, apply_barrier=False)
            V = [V_van[i] - V[i] for i in range(len(V))]

        return V, (dt_last_global if dt_last_global is not None else (self.T/max(1,self.M)))

    # ---------------- interpolation & Greeks ----------------
    @staticmethod
    def _interp_linear(x: float, xs: List[float], ys: List[float]) -> float:
        if x<=xs[0]: return float(ys[0])
        if x>=xs[-1]: return float(ys[-1])
        lo,hi=0,len(xs)-1
        while hi-lo>1:
            mid=(lo+hi)//2
            if x<xs[mid]: hi=mid
            else: lo=mid
        x0,x1=xs[lo],xs[hi]; y0,y1=ys[lo],ys[hi]
        w=(x-x0)/(x1-x0)
        return float((1-w)*y0 + w*y1)

    def _delta_gamma_nonuniform(self, s_nodes: List[float], V: List[float], Sx: float) -> Tuple[float,float]:
        N=len(s_nodes)-1
        i = min(range(N+1), key=lambda k: abs(s_nodes[k]-Sx))
        i = max(1, min(N-1, i))
        h1 = s_nodes[i] - s_nodes[i-1]
        h2 = s_nodes[i+1] - s_nodes[i]
        # central (second-order) away from barrier
        delta_c = ( -(h2/(h1*(h1+h2)))*V[i-1] + ((h2-h1)/(h1*h2))*V[i] + (h1/(h2*(h1+h2)))*V[i+1] )
        gamma_c = 2.0*( V[i-1]/(h1*(h1+h2)) - V[i]/(h1*h2) + V[i+1]/(h2*(h1+h2)) )

        if not self.use_one_sided: 
            return float(delta_c), float(gamma_c)

        # Barrier-aware stencil near KO barrier
        H=None; side=None
        if self.barrier_type in ("down-and-out","double-out") and self.lower_barrier is not None:
            H=self.lower_barrier; side="down"
        if self.barrier_type in ("up-and-out","double-out") and self.upper_barrier is not None and H is None:
            H=self.upper_barrier; side="up"
        if H is None: 
            return float(delta_c), float(gamma_c)

        j = min(range(N+1), key=lambda k: abs(s_nodes[k]-H))
        j = max(1, min(N-1, j))
        near = (abs(i-j) <= self.barrier_safety_cells)
        if not near: 
            return float(delta_c), float(gamma_c)

        # Build barrier-aware Δ, Γ using quadratic through (B, VB), (Si,Vi), (Si+h+,Vi+1) or mirrored
        if side=="down":
            # Adjacent interior node is j >= 1 with s_nodes[j] > H
            jj = j if s_nodes[j] > H else min(j+1, N-1)
            S1, S2 = s_nodes[jj], s_nodes[jj+1]
            h_minus = max(1e-14, S1 - H)
            h_plus  = max(1e-14, S2 - S1)
            # Interpolate VB from V at (S1,S2) (no projection at t=0)
            t = (H - S1)/(S2 - S1)
            VB = (1.0-t)*V[jj] + t*V[jj+1]
            # Δ, Γ at evaluation cell (use closest interior jj)
            a1 = -h_plus/(h_minus*(h_minus+h_plus))
            a2 = (h_plus - h_minus)/(h_minus*h_plus)
            a3 =  h_minus/(h_plus*(h_minus+h_plus))
            d1 =  2.0/(h_minus*(h_minus+h_plus))
            d2 = -2.0/(h_minus*h_plus)
            d3 =  2.0/(h_plus*(h_minus+h_plus))
            Ux  = a1*VB + a2*V[jj] + a3*V[jj+1]
            Uxx = d1*VB + d2*V[jj] + d3*V[jj+1]
        else:
            # side == "up": interior node just below H
            jj = j if s_nodes[j] < H else max(j-1, 1)
            S0, S1 = s_nodes[jj-1], s_nodes[jj]
            h_minus = max(1e-14, S1 - S0)
            h_plus  = max(1e-14, H - S1)
            t = (H - S1)/(H - S0) if (H - S0)!=0 else 0.0
            VB = (1.0-t)*V[jj] + t*V[jj-1]
            a1 = -h_plus/(h_minus*(h_minus+h_plus))   # for V_{jj-1}
            a2 = (h_plus - h_minus)/(h_minus*h_plus)  # for V_{jj}
            a3 =  h_minus/(h_plus*(h_minus+h_plus))   # for V_B
            d1 =  2.0/(h_minus*(h_minus+h_plus))
            d2 = -2.0/(h_minus*h_plus)
            d3 =  2.0/(h_plus*(h_minus+h_plus))
            Ux  = a1*V[jj-1] + a2*V[jj] + a3*VB
            Uxx = d1*V[jj-1] + d2*V[jj] + d3*VB

        # Map to price Greeks directly (we’re in S-space already)
        delta = Ux
        gamma = Uxx
        # conservative clamp to avoid reporting tiny numerical spikes
        gamma = max(min(gamma, 1e6), -1e6)
        return float(delta), float(gamma)

    # ---------------- public API ----------------
    def price(self) -> float:
        V, _ = self._run_backward(self.S_nodes, apply_barrier=(self.barrier_type!="none"))
        return self._interp_linear(self.S0, self.S_nodes, V)

    def greeks(self, vega_bump: float = 1e-3) -> Dict[str,float]:
        V, dt0 = self._run_backward(self.S_nodes, apply_barrier=(self.barrier_type!="none"))
        P = self._interp_linear(self.S0, self.S_nodes, V)
        delta, gamma = self._delta_gamma_nonuniform(self.S_nodes, V, self.S0)

        # vega (bump σ; keep same grid for robustness)
        sig0=self.sigma
        self.sigma=sig0+vega_bump; V_up,_=self._run_backward(self.S_nodes, apply_barrier=(self.barrier_type!="none")); P_up=self._interp_linear(self.S0,self.S_nodes,V_up)
        self.sigma=sig0-vega_bump; V_dn,_=self._run_backward(self.S_nodes, apply_barrier=(self.barrier_type!="none")); P_dn=self._interp_linear(self.S0,self.S_nodes,V_dn)
        self.sigma=sig0
        vega=(P_up-P_dn)/(2.0*vega_bump)

        # theta: PDE form (cheap)
        theta_pde = -((self.r - self.q)*self.S0*delta + 0.5*self.sigma*self.sigma*self.S0*self.S0*gamma - self.r*P)

        if not self.compute_numerical_theta:
            theta = theta_pde
        else:
            # numerical theta using last dt near 0 (robust if available)
            dt_eps = max(1e-8, dt0)
            # shift valuation date forward by an epsilon dt (approximate)
            # We reuse the same V_up computed with σ bump to avoid another full run; recompute one tiny substep if needed.
            # Simpler: backward a very short extra step from t=0 to t=dt_eps using implicit θ=1 and read price:
            V_now = V[:]  # solution at t=0
            # do 1 implicit step forward in time (equivalently back toward maturity) of size dt_eps
            V_eps, _ = self._cn_subinterval(self.S_nodes, V_now, 0.0, dt_eps, theta=1.0, rannacher_left_steps=0, m_steps=1)
            P_eps = self._interp_linear(self.S0, self.S_nodes, V_eps)
            theta = -(P - P_eps)/dt_eps

        return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega), "theta": float(theta)}

    def print_details(self) -> None:
        p=self.price(); g=self.greeks()
        print("==== Discrete Barrier Option (CN + Rannacher) — Discrete monitors, S-space ====")
        print(f"T (years)         : {self.T:.9f}   [{self.day_count}]")
        print(f"sigma / r / q     : {self.sigma:.9f} / {self.r:.9f} / {self.q:.9f}")
        print(f"Barrier type      : {self.barrier_type}  (lo={self.lower_barrier}, up={self.upper_barrier})")
        print(f"Rebate (amt/hit)  : {self.rebate_amount} / {self.rebate_at_hit}")
        print(f"Status (hit/in)   : {self.already_hit} / {self.already_in}")
        print(f"Grid(S,N)         : {len(self.S_nodes)}, N={self.N}, grid_type={self.grid_type}")
        print(f"Monitors count    : {len(self.monitor_times)}")
        print(f"Spot/Strike       : {self.S0:.6f} / {self.K:.6f}")
        print(f"Price             : {p:.9f}")
        print(f"Greeks            : Δ={g['delta']:.9f}, Γ={g['gamma']:.9f}, ν={g['vega']:.9f}, Θ={g['theta']:.9f}")
