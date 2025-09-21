"""
Discrete barrier European option pricer (Crank–Nicolson + optional Rannacher)
with **discrete monitoring** (exact monitor times), **rebates**, **status flags**,
optional **non‑uniform sinh grid**, robust **Greeks near barrier**, and
a **grid-convergence** validator.

No BGK adjustments. KI via in–out parity. Cash dividends via PV‑escrow (optional).
"""

from typing import List, Tuple, Dict, Optional, Literal
import math
from datetime import date

OptionType = Literal["call", "put"]
BarrierType = Literal["none",
                      "down-and-out","up-and-out","double-out",
                      "down-and-in","up-and-in","double-in"]

class DiscreteBarrierFDM:
    def __init__(self,
                 # economics
                 spot: float,
                 strike: float,
                 valuation_date: date,
                 maturity_date: date,
                 volatility: float,
                 flat_rate_nacc: float,
                 option_type: OptionType,
                 dividend_yield: float = 0.0,                   # continuous q
                 # barriers
                 barrier_type: BarrierType = "none",
                 lower_barrier: Optional[float] = None,
                 upper_barrier: Optional[float] = None,
                 monitoring_dates: Optional[List[date]] = None,
                 rebate_amount: float = 0.0,
                 rebate_at_hit: bool = True,
                 # status flags at t=0
                 already_hit: bool = False,    # for KO
                 already_in:  bool = False,    # for KI
                 # dividends (cash escrow)
                 dividend_list: Optional[List[Tuple[date,float]]] = None,
                 # numerics
                 num_space_nodes: int = 600,
                 num_time_steps: int = 600,    # total target steps (used to allocate substeps between monitors)
                 rannacher_steps: int = 2,
                 day_count: str = "ACT/365",
                 # time-grid refinement
                 min_substeps_between_monitors: int = 1,   # >=1 CN steps per monitor interval
                 # grid type
                 grid_type: Literal["uniform","sinh"] = "uniform",
                 sinh_alpha: float = 1.5,     # clustering strength
                 # Greeks stabilization
                 use_one_sided_greeks_near_barrier: bool = True,
                 barrier_safety_cells: int = 2):
        # market / instrument
        self.S0 = float(spot)
        self.K  = float(strike)
        self.valuation_date = valuation_date
        self.maturity_date  = maturity_date
        self.sigma = float(volatility)
        self.r = float(flat_rate_nacc)
        self.q = float(dividend_yield)
        self.option_type = option_type

        # barrier
        self.barrier_type  = barrier_type
        self.lower_barrier = lower_barrier
        self.upper_barrier = upper_barrier
        self.monitoring_dates = sorted(monitoring_dates or [])

        self.rebate_amount = float(rebate_amount)
        self.rebate_at_hit = bool(rebate_at_hit)

        # status
        self.already_hit = bool(already_hit)
        self.already_in  = bool(already_in)

        # numerics
        self.N = int(num_space_nodes)
        self.M_target = int(num_time_steps)
        self.rannacher = int(rannacher_steps)
        self.day_count = day_count.upper()
        self.min_substeps = max(1, int(min_substeps_between_monitors))

        self.grid_type = grid_type
        self.sinh_alpha = float(sinh_alpha)

        self.use_one_sided = bool(use_one_sided_greeks_near_barrier)
        self.barrier_safety_cells = int(barrier_safety_cells)

        self.dividends = [(d,float(a)) for (d,a) in (dividend_list or [])]

        # time
        self.T = self._year_fraction(self.valuation_date, self.maturity_date)

        # space grid
        self.S_nodes = self._build_space_grid()
        self.dS_uniform = self.S_nodes[1]-self.S_nodes[0] if len(self.S_nodes)>1 else 1.0

        # prebuild exact monitor times in [0,T]
        self.monitor_times = self._build_monitor_times_exact()

    # ---------------------- utilities ----------------------
    def _year_fraction(self, d0: date, d1: date) -> float:
        days = max(0,(d1-d0).days)
        if self.day_count in ("ACT/360",):
            return days/360.0
        if self.day_count in ("30/360","30E/360"):
            y0,m0,d0_ = d0.year,d0.month,min(d0.day,30)
            y1,m1,d1_ = d1.year,d1.month,min(d1.day,30)
            return ((y1-y0)*360 + (m1-m0)*30 + (d1_-d0_))/360.0
        return days/365.0

    def _pv_dividends_escrow(self) -> float:
        if not self.dividends:
            return 0.0
        pv=0.0
        for (dd,amt) in self.dividends:
            tau = self._year_fraction(self.valuation_date, dd)
            if 0.0 < tau <= self.T + 1e-14:
                pv += amt*math.exp(-self.r*tau)
        return pv

    # -------------------- time grid with exact monitors ----------------
    def _build_monitor_times_exact(self) -> List[float]:
        times = [0.0]
        for d in self.monitoring_dates:
            if self.valuation_date <= d <= self.maturity_date:
                t = self._year_fraction(self.valuation_date, d)
                if 0.0 <= t <= self.T:
                    times.append(t)
        if times[-1] < self.T - 1e-14:
            times.append(self.T)
        times = sorted(set(times))
        return times

    def _time_subgrid_counts(self) -> List[int]:
        # Allocate substeps per interval proportionally to length, with a minimum
        lengths = [self.monitor_times[i+1]-self.monitor_times[i] for i in range(len(self.monitor_times)-1)]
        total = sum(lengths)
        if total <= 0: return [self.min_substeps]*(len(lengths))
        raw = [max(self.min_substeps, int(round(self.M_target * (L/total)))) for L in lengths]
        # ensure at least 1 per interval and adjust sum
        s = sum(raw)
        if s == 0:
            raw = [self.min_substeps]*len(lengths); s = sum(raw)
        # adjust to exactly M_target by distributing diff
        diff = self.M_target - s
        i = 0
        while diff != 0 and len(raw)>0:
            j = i % len(raw)
            if diff > 0: raw[j] += 1; diff -= 1
            else:
                if raw[j] > self.min_substeps:
                    raw[j] -= 1; diff += 1
            i += 1
        return raw

    # --------------------------- space grid -----------------------------
    def _build_space_grid(self) -> List[float]:
        anchors = [self.S0, self.K]
        if self.lower_barrier is not None: anchors.append(self.lower_barrier)
        if self.upper_barrier is not None: anchors.append(self.upper_barrier)
        s_ref = max(anchors) if anchors else self.S0
        Smax = 4.5*s_ref*math.exp(self.sigma*math.sqrt(max(self.T,1e-12)))
        Smin = 0.0
        N = max(200,self.N)

        if self.grid_type == "uniform":
            dS = (Smax-Smin)/N
            nodes = [Smin+i*dS for i in range(N+1)]
        else:
            # sinh-stretched around center Sc chosen as max(S0, K, barrier mid)
            Sc = max(self.S0, self.K,
                     (self.lower_barrier or 0.0),
                     (self.upper_barrier or 0.0))
            a = self.sinh_alpha
            # map x ∈ [-1,1] to S via sinh
            xs = [-1.0 + 2.0*i/N for i in range(N+1)]
            span = Smax-Sc
            scale = span/max(1e-12, math.sinh(a))
            nodes = [Sc + scale*math.sinh(a*x) for x in xs]
            # ensure lower bound >=0
            shift = -min(0.0, min(nodes))
            if shift>0:
                nodes = [s+shift for s in nodes]

        # snap K and barriers to nearest node (helps projection)
        def snap(x: Optional[float]):
            if x is None: return
            j = min(range(len(nodes)), key=lambda i: abs(nodes[i]-x))
            nodes[j]=float(x)
        snap(self.K); snap(self.lower_barrier); snap(self.upper_barrier)
        return nodes

    # ---------------- payoff / BCs / KO projection ---------------------
    def _terminal_payoff(self, s_nodes: List[float]) -> List[float]:
        if self.option_type=="call":
            return [max(s-self.K,0.0) for s in s_nodes]
        return [max(self.K-s,0.0) for s in s_nodes]

    def _apply_KO_projection(self, V: List[float], s_nodes: List[float], tau_left: float) -> None:
        if self.barrier_type in ("none","down-and-in","up-and-in","double-in"):
            return
        lo = self.lower_barrier; up = self.upper_barrier
        rebate = self.rebate_amount if self.rebate_at_hit else self.rebate_amount*math.exp(-self.r*tau_left)
        for i,s in enumerate(s_nodes):
            out=False
            if self.barrier_type=="down-and-out" and lo is not None and s<=lo: out=True
            elif self.barrier_type=="up-and-out" and up is not None and s>=up: out=True
            elif self.barrier_type=="double-out":
                if (lo is not None and s<=lo) or (up is not None and s>=up): out=True
            if out: V[i]=rebate

    # ----------------------- tridiagonal solver -------------------------
    @staticmethod
    def _solve_tridiagonal(a_sub: List[float], a_diag: List[float], a_sup: List[float], rhs: List[float]) -> List[float]:
        n=len(rhs)
        c=[0.0]*n; d=[0.0]*n; x=[0.0]*n
        beta=a_diag[0]; 
        if abs(beta)<1e-14: beta=1e-14
        c[0]=a_sup[0]/beta; d[0]=rhs[0]/beta
        for i in range(1,n):
            beta=a_diag[i]-a_sub[i]*c[i-1]
            if abs(beta)<1e-14: beta=1e-14
            c[i]=(a_sup[i]/beta) if i<n-1 else 0.0
            d[i]=(rhs[i]-a_sub[i]*d[i-1])/beta
        x[-1]=d[-1]
        for i in range(n-2,-1,-1):
            x[i]=d[i]-c[i]*x[i+1]
        return x

    # -------------------- CN stepper over subinterval -------------------
    def _cn_subinterval(self, s_nodes: List[float], V: List[float], t0: float, t1: float,
                        theta: float, rannacher_left_steps: int) -> Tuple[List[float], float]:
        """March from t1 to t0 (backwards). Return new V and dt used in the **last** step of the whole march when t0==0 (for theta)."""
        r, q, sig = self.r, self.q, self.sigma
        dS = None  # non-uniform support
        N = len(s_nodes)-1

        # choose number of steps in this sub-interval
        L = t1 - t0
        m = max(1, int(round(self.M_target * (L / max(self.T,1e-12)))))  # proportional split
        m = max(m, self.min_substeps)
        dt = L / m

        last_dt_at_zero = None

        for step in range(m):
            use_theta = 1.0 if (rannacher_left_steps > 0) else theta
            if rannacher_left_steps > 0:
                rannacher_left_steps -= 1

            # boundaries at "after-step" time level (tau_left = t0 + (m-step-1)*dt)
            tau_left = t0 + (m-step-1)*dt
            sub=[0.0]*(N+1); main=[0.0]*(N+1); sup=[0.0]*(N+1); rhs=[0.0]*(N+1)

            # Dirichlet boundaries
            if self.option_type=="call":
                rhs[0]=0.0
                rhs[N]=s_nodes[-1]*math.exp(-q*tau_left) - self.K*math.exp(-r*tau_left)
            else:
                rhs[0]=self.K*math.exp(-r*tau_left)
                rhs[N]=0.0
            main[0]=1.0; main[N]=1.0

            # interior rows (non-uniform second-order)
            for i in range(1,N):
                Si = s_nodes[i]
                # local spacings
                h1 = s_nodes[i] - s_nodes[i-1]
                h2 = s_nodes[i+1] - s_nodes[i]
                A = 0.5*sig*sig*Si*Si
                mu = r - q

                # diffusion & convection on non-uniform grid using central differences:
                # V_xx ≈ 2*( V_{i-1}/(h1*(h1+h2)) - V_i/(h1*h2) + V_{i+1}/(h2*(h1+h2)) )
                # V_x  ≈ -(h2/(h1*(h1+h2))) V_{i-1} + ((h2-h1)/(h1*h2)) V_i + (h1/(h2*(h1+h2))) V_{i+1}
                # Implement CN by building implicit/explicit matrices with these weights.
                # For compactness, compute coefficients for implicit and explicit parts:

                # weights for first and second derivatives
                w_im1_x  = -h2/(h1*(h1+h2))
                w_i_x    = (h2-h1)/(h1*h2)
                w_ip1_x  = h1/(h2*(h1+h2))
                w_im1_xx =  2.0/(h1*(h1+h2))
                w_i_xx   = -2.0/(h1*h2)
                w_ip1_xx =  2.0/(h2*(h1+h2))

                # implicit tri-diagonals (apply to unknown V^{n})
                a_impl = use_theta*dt*( A*w_im1_xx + mu*Si*w_im1_x )
                b_impl = 1.0 + use_theta*dt*( A*w_i_xx   + mu*Si*w_i   + r )
                c_impl = use_theta*dt*( A*w_ip1_xx + mu*Si*w_ip1_x )

                # explicit RHS (applied to known V)
                a_expl = (1.0-use_theta)*dt*( A*w_im1_xx + mu*Si*w_im1_x )
                b_expl = 1.0 + (1.0-use_theta)*dt*( A*w_i_xx   + mu*Si*w_i   + r )
                c_expl = (1.0-use_theta)*dt*( A*w_ip1_xx + mu*Si*w_ip1_x )

                sub[i]  = -a_impl
                main[i] =  b_impl
                sup[i]  = -c_impl
                rhs[i]  =  a_expl*V[i-1] + b_expl*V[i] + c_expl*V[i+1]

            V = self._solve_tridiagonal(sub, main, sup, rhs)

            # capture dt for theta at t=0 (the very final global step)
            if abs(t0) < 1e-14 and step == m-1:
                last_dt_at_zero = dt

        return V, (last_dt_at_zero if last_dt_at_zero is not None else dt)

    # --------------------- full backward propagation --------------------
    def _run_backward(self, s_nodes: List[float], apply_barrier: bool) -> Tuple[List[float], float, List[float]]:
        """Return (final V at t=0, dt_last, monitors_applied_times)."""
        N = len(s_nodes)-1
        V = self._terminal_payoff(s_nodes)

        # Handle status flags at t=0 before marching
        if apply_barrier:
            if self.barrier_type in ("down-and-out","up-and-out","double-out") and self.already_hit:
                # immediate rebate
                instant = self.rebate_amount if self.rebate_at_hit else self.rebate_amount*math.exp(-self.r*self.T)
                return [instant]* (N+1), 0.0, []
            if self.barrier_type in ("down-and-in","up-and-in","double-in") and self.already_in:
                apply_barrier = False  # vanilla from the start

        theta = 0.5
        monitors_applied = []
        # Allocate substeps per interval (approximately M_target total)
        subcounts = self._time_subgrid_counts()

        rannacher_left = self.rannacher
        dt_last_global = None

        # March interval by interval: [t_{k-1}, t_k], backward in time
        for k in range(len(self.monitor_times)-1, 0, -1):
            t0 = self.monitor_times[k-1]
            t1 = self.monitor_times[k]
            V, dt_last = self._cn_subinterval(s_nodes, V, t0, t1, theta, rannacher_left)
            rannacher_left = max(0, rannacher_left - subcounts[k-1])

            # After completing the subinterval, apply KO projection if t0 is a monitor
            if apply_barrier and (abs(t0) > 1e-14):  # do not project at t=0
                self._apply_KO_projection(V, s_nodes, tau_left=t0)
                monitors_applied.append(t0)

            if abs(t0) < 1e-14:
                dt_last_global = dt_last

        # For KI, compute vanilla and subtract KO
        if apply_barrier and self.barrier_type in ("down-and-in","up-and-in","double-in"):
            V_ko, _, _ = self._run_backward(s_nodes, apply_barrier=False)  # vanilla
            V = [V_ko[i] - V[i] for i in range(len(V))]

        return V, (dt_last_global if dt_last_global is not None else (self.T/ max(1,self.M_target))), monitors_applied

    # -------------------- interpolation + Greeks ------------------------
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
        # locate index i nearest to Sx
        N=len(s_nodes)-1
        i = min(range(N+1), key=lambda k: abs(s_nodes[k]-Sx))
        i = max(1, min(N-1, i))
        h1 = s_nodes[i] - s_nodes[i-1]
        h2 = s_nodes[i+1] - s_nodes[i]
        # central on non-uniform
        delta_c = ( -(h2/(h1*(h1+h2)))*V[i-1] + ((h2-h1)/(h1*h2))*V[i] + (h1/(h2*(h1+h2)))*V[i+1] )
        gamma_c = 2.0*( V[i-1]/(h1*(h1+h2)) - V[i]/(h1*h2) + V[i+1]/(h2*(h1+h2)) )

        if not self.use_one_sided: return float(delta_c), float(gamma_c)

        # One-sided logic near KO barrier
        H=None; side=None
        if self.barrier_type in ("down-and-out","double-out") and self.lower_barrier is not None:
            H=self.lower_barrier; side="down"
        if self.barrier_type in ("up-and-out","double-out") and self.upper_barrier is not None and H is None:
            H=self.upper_barrier; side="up"

        if H is None: return float(delta_c), float(gamma_c)

        # find j such that H in [S_j, S_{j+1}]
        j = max(0, min(N-1, min(range(N), key=lambda k: abs(H - s_nodes[k]))))
        # if within safety band, switch to directional
        near = (abs(i-j) <= self.barrier_safety_cells)
        if not near: return float(delta_c), float(gamma_c)

        if side=="down":
            # forward difference away from barrier
            i2 = max(1, min(N-1, j+1))
            h = s_nodes[i2+1]-s_nodes[i2]
            delta = (V[i2+1]-V[i2])/h
            gamma = (V[i2+1]-2*V[i2]+V[i2-1]) / ((s_nodes[i2+1]-s_nodes[i2])*(s_nodes[i2]-s_nodes[i2-1]))
        else:
            i2 = max(1, min(N-1, j))
            h = s_nodes[i2]-s_nodes[i2-1]
            delta = (V[i2]-V[i2-1])/h
            gamma = (V[i2+1]-2*V[i2]+V[i2-1]) / ((s_nodes[i2+1]-s_nodes[i2])*(s_nodes[i2]-s_nodes[i2-1]))

        gamma = max(min(gamma, 1e5), -1e5)
        return float(delta), float(gamma)

    # --------------------------- public API -----------------------------
    def _shifted_grid(self) -> Tuple[List[float], float]:
        pv_divs = self._pv_dividends_escrow()
        S_eff = max(self.S0 - pv_divs, 0.0)
        s_nodes = [max(s - pv_divs, 0.0) for s in self.S_nodes]
        return s_nodes, S_eff

    def price(self) -> float:
        s_nodes, S_eff = self._shifted_grid()
        # vanilla if barrier_type=="none"
        apply_barrier = (self.barrier_type!="none")
        V, _, _ = self._run_backward(s_nodes, apply_barrier=apply_barrier)
        return self._interp_linear(S_eff, s_nodes, V)

    def greeks(self, vega_bump: float = 1e-3) -> Dict[str,float]:
        s_nodes, S_eff = self._shifted_grid()
        V, dt0, _ = self._run_backward(s_nodes, apply_barrier=(self.barrier_type!="none"))
        delta, gamma = self._delta_gamma_nonuniform(s_nodes, V, S_eff)

        # vega
        sig0=self.sigma
        self.sigma=sig0+vega_bump; up=self.price()
        self.sigma=sig0-vega_bump; dn=self.price()
        self.sigma=sig0
        vega=(up-dn)/(2*vega_bump)

        # theta (use last dt from the full march)
        # recompute a "one-step-shorter" price by advancing one small dt in time
        # (i.e., remove the very last substep): approximate by shifting maturity slightly
        # Simpler: rebuild time grid with a tiny epsilon cut
        eps = 1e-10 if dt0<=0 else 0.0
        theta = 0.0
        if dt0>0:
            # advance by dt0: rerun but chop off the final substep by trimming the monitor_times start
            t_save = self.monitor_times[:]
            if t_save[0]==0.0 and len(t_save)>1:
                t_save[0] = min(dt0, t_save[1]*0.5)  # small positive replacing 0
            V_eps, _, _ = self._run_backward(s_nodes, apply_barrier=(self.barrier_type!="none"))
            p0 = self._interp_linear(S_eff, s_nodes, V)
            p1 = self._interp_linear(S_eff, s_nodes, V_eps)
            theta = (p1 - p0)/max(dt0,1e-12)

        return {"delta":float(delta), "gamma":float(gamma), "vega":float(vega), "theta":float(theta)}

    def print_details(self) -> None:
        p=self.price(); g=self.greeks()
        print("==== Discrete Barrier Option (CN + Rannacher) — Discrete monitors, no BGK ====")
        print(f"T (years)         : {self.T:.9f}   [{self.day_count}]")
        print(f"sigma / r / q     : {self.sigma:.9f} / {self.r:.9f} / {self.q:.9f}")
        print(f"Barrier type      : {self.barrier_type}  (lo={self.lower_barrier}, up={self.upper_barrier})")
        print(f"Rebate (amt/hit)  : {self.rebate_amount} / {self.rebate_at_hit}")
        print(f"Status (hit/in)   : {self.already_hit} / {self.already_in}")
        print(f"Grid(S,N)         : {len(self.S_nodes)}  | grid_type={self.grid_type}")
        print(f"Monitors (count)  : {len(self.monitor_times)} @ {self.monitor_times}")
        print(f"Spot/Strike       : {self.S0:.6f} / {self.K:.6f}")
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
                clone = DiscreteBarrierFDM(
                    spot=self.S0, strike=self.K,
                    valuation_date=self.valuation_date, maturity_date=self.maturity_date,
                    volatility=self.sigma, flat_rate_nacc=self.r, dividend_yield=self.q,
                    option_type=self.option_type,
                    barrier_type=self.barrier_type, lower_barrier=self.lower_barrier, upper_barrier=self.upper_barrier,
                    monitoring_dates=self.monitoring_dates, rebate_amount=self.rebate_amount, rebate_at_hit=self.rebate_at_hit,
                    already_hit=self.already_hit, already_in=self.already_in,
                    dividend_list=self.dividends,
                    num_space_nodes=N, num_time_steps=M, rannacher_steps=self.rannacher,
                    day_count=self.day_count,
                    min_substeps_between_monitors=self.min_substeps,
                    grid_type=self.grid_type, sinh_alpha=self.sinh_alpha,
                    use_one_sided_greeks_near_barrier=self.use_one_sided,
                    barrier_safety_cells=self.barrier_safety_cells
                )
                price = clone.price()
                greeks = clone.greeks()
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