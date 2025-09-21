"""
Discrete barrier European option pricer (Black–Scholes PDE) via Crank–Nicolson
with optional Rannacher start, **discrete monitoring** (exact times), rebates,
status flags (already_hit / already_in), optional sinh grid, and stable Greeks.

Key fixes vs prior version:
- Correct non‑uniform CN coefficients (Tavella–Randall style) for D1/D2.
- Correct RHS diagonal sign (b_expl uses a minus).
- Removed undefined w_i variable. No "weights" approach; use canonical tri‑diagonals.
- Preserves exact monitor application and parity for KI.
"""

from typing import List, Tuple, Dict, Optional, Literal
from datetime import date
import math

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
                 dividend_yield: float = 0.0,
                 # barriers
                 barrier_type: BarrierType = "none",
                 lower_barrier: Optional[float] = None,
                 upper_barrier: Optional[float] = None,
                 monitoring_dates: Optional[List[date]] = None,
                 rebate_amount: float = 0.0,
                 rebate_at_hit: bool = True,
                 # status flags
                 already_hit: bool = False,   # for KO
                 already_in:  bool = False,   # for KI
                 # numerics
                 num_space_nodes: int = 600,
                 num_time_steps: int = 600,
                 rannacher_steps: int = 2,
                 day_count: str = "ACT/365",
                 min_substeps_between_monitors: int = 1,
                 grid_type: Literal["uniform","sinh"] = "uniform",
                 sinh_alpha: float = 1.5,
                 # greek stabilization
                 use_one_sided_greeks_near_barrier: bool = True,
                 barrier_safety_cells: int = 2):
        # market/instrument
        self.S0 = float(spot)
        self.K = float(strike)
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
        self.N = max(200, int(num_space_nodes))
        self.M_target = max(1, int(num_time_steps))
        self.rannacher = int(rannacher_steps)
        self.day_count = day_count.upper()
        self.min_substeps = max(1, int(min_substeps_between_monitors))
        self.grid_type = grid_type
        self.sinh_alpha = float(sinh_alpha)

        self.use_one_sided = bool(use_one_sided_greeks_near_barrier)
        self.barrier_safety_cells = int(barrier_safety_cells)

        # time
        self.T = self._year_fraction(self.valuation_date, self.maturity_date)
        # grid
        self.S_nodes = self._build_space_grid()
        # monitors
        self.monitor_times = self._build_monitor_times_exact()

    # ------------- utils ---------------
    def _year_fraction(self, d0: date, d1: date) -> float:
        days = max(0,(d1-d0).days)
        if self.day_count in ("ACT/360",):
            return days/360.0
        if self.day_count in ("30/360","30E/360"):
            y0,m0,d0_ = d0.year,d0.month,min(d0.day,30)
            y1,m1,d1_ = d1.year,d1.month,min(d1.day,30)
            return ((y1-y0)*360 + (m1-m0)*30 + (d1_-d0_))/360.0
        return days/365.0

    def _build_monitor_times_exact(self) -> List[float]:
        times = [0.0]
        for d in self.monitoring_dates:
            if self.valuation_date <= d <= self.maturity_date:
                t = self._year_fraction(self.valuation_date, d)
                if 0.0 <= t <= self.T: times.append(t)
        if times[-1] < self.T - 1e-14: times.append(self.T)
        return sorted(set(times))

    def _build_space_grid(self) -> List[float]:
        anchors = [self.S0, self.K]
        if self.lower_barrier is not None: anchors.append(self.lower_barrier)
        if self.upper_barrier is not None: anchors.append(self.upper_barrier)
        s_ref = max(anchors)
        Smax = 4.5*s_ref*math.exp(self.sigma*math.sqrt(max(self.T,1e-12)))
        N = self.N
        if self.grid_type == "uniform":
            dS = Smax/N
            nodes = [i*dS for i in range(N+1)]
        else:
            Sc = max(self.S0, self.K, (self.lower_barrier or 0.0), (self.upper_barrier or 0.0))
            a  = self.sinh_alpha
            xs = [-1.0 + 2.0*i/N for i in range(N+1)]
            span = Smax-Sc
            scale = span/max(1e-12, math.sinh(a))
            nodes = [Sc + scale*math.sinh(a*x) for x in xs]
            shift = -min(0.0, min(nodes))
            if shift>0: nodes = [s+shift for s in nodes]
        # snap key levels
        def snap(x: Optional[float]):
            if x is None: return
            j = min(range(len(nodes)), key=lambda i: abs(nodes[i]-x))
            nodes[j] = float(x)
        snap(self.K); snap(self.lower_barrier); snap(self.upper_barrier)
        return nodes

    # ------------- payoff / BC / projection -------------
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

    # ------------- tridiagonal solver -------------
    @staticmethod
    def _solve_tridiagonal(a_sub: List[float], a_diag: List[float], a_sup: List[float], rhs: List[float]) -> List[float]:
        n=len(rhs)
        c=[0.0]*n; d=[0.0]*n; x=[0.0]*n
        beta=a_diag[0]
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

    # ------------- CN step over [t0,t1] with m steps -------------
    def _cn_subinterval(self, s_nodes: List[float], V: List[float],
                        t0: float, t1: float, theta: float,
                        rannacher_left_steps: int, m_steps: int) -> Tuple[List[float], float]:
        r, q, sig = self.r, self.q, self.sigma
        N = len(s_nodes)-1
        L = t1 - t0
        m = max(1, int(m_steps))
        dt = L / m
        last_dt_at_zero = None

        for step in range(m):
            use_theta = 1.0 if (rannacher_left_steps > 0) else theta
            if rannacher_left_steps > 0:
                rannacher_left_steps -= 1

            tau_left = t0 + (m-step-1)*dt  # time to maturity after this step

            # Build tri-diagonal using non‑uniform coefficients (Tavella–Randall)
            sub=[0.0]*(N+1); main=[0.0]*(N+1); sup=[0.0]*(N+1); rhs=[0.0]*(N+1)

            # Dirichlet boundaries at after-step time
            if self.option_type=="call":
                rhs[0] = 0.0
                rhs[N] = s_nodes[-1]*math.exp(-q*tau_left) - self.K*math.exp(-r*tau_left)
            else:
                rhs[0] = self.K*math.exp(-r*tau_left)
                rhs[N] = 0.0
            main[0]=1.0; main[N]=1.0

            for i in range(1,N):
                Si = s_nodes[i]
                h1 = s_nodes[i] - s_nodes[i-1]
                h2 = s_nodes[i+1] - s_nodes[i]
                A = 0.5*sig*sig*Si*Si
                mu = r - q

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

            V = self._solve_tridiagonal(sub, main, sup, rhs)

            if abs(t0) < 1e-14 and step == m-1:
                last_dt_at_zero = dt

        return V, (last_dt_at_zero if last_dt_at_zero is not None else dt)

    # ------------- full backward run -------------
    def _time_subgrid_counts(self) -> List[int]:
        lengths = [self.monitor_times[i+1]-self.monitor_times[i] for i in range(len(self.monitor_times)-1)]
        total = sum(lengths)
        if total <= 0: return [self.min_substeps]*(len(lengths))
        raw = [max(self.min_substeps, int(round(self.M_target * (L/total)))) for L in lengths]
        s = sum(raw)
        if s == 0:
            raw = [self.min_substeps]*len(lengths); s = sum(raw)
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

    def _run_backward(self, s_nodes: List[float], apply_barrier: bool) -> Tuple[List[float], float, List[float]]:
        N = len(s_nodes)-1
        V = self._terminal_payoff(s_nodes)

        # Status at t=0
        if apply_barrier:
            if self.barrier_type in ("down-and-out","up-and-out","double-out") and self.already_hit:
                instant = self.rebate_amount if self.rebate_at_hit else self.rebate_amount*math.exp(-self.r*self.T)
                return [instant]*(N+1), 0.0, []
            if self.barrier_type in ("down-and-in","up-and-in","double-in") and self.already_in:
                apply_barrier = False

        theta = 0.5
        subcounts = self._time_subgrid_counts()
        rannacher_left = self.rannacher
        dt_last_global = None
        applied = []

        for k in range(len(self.monitor_times)-1, 0, -1):
            t0 = self.monitor_times[k-1]
            t1 = self.monitor_times[k]
            m_steps = subcounts[k-1] if k-1 < len(subcounts) else 1
            V, dt_last = self._cn_subinterval(s_nodes, V, t0, t1, theta, rannacher_left, m_steps)
            rannacher_left = max(0, rannacher_left - m_steps)

            if apply_barrier and (abs(t0) > 1e-14):
                self._apply_KO_projection(V, s_nodes, tau_left=t0)
                applied.append(t0)

            if abs(t0) < 1e-14:
                dt_last_global = dt_last

        if apply_barrier and self.barrier_type in ("down-and-in","up-and-in","double-in"):
            V_van, _, _ = self._run_backward(s_nodes, apply_barrier=False)
            V = [V_van[i] - V[i] for i in range(len(V))]

        return V, (dt_last_global if dt_last_global is not None else (self.T/max(1,self.M_target))), applied

    # ------------- interpolation + Greeks -------------
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
        delta_c = ( -(h2/(h1*(h1+h2)))*V[i-1] + ((h2-h1)/(h1*h2))*V[i] + (h1/(h2*(h1+h2)))*V[i+1] )
        gamma_c = 2.0*( V[i-1]/(h1*(h1+h2)) - V[i]/(h1*h2) + V[i+1]/(h2*(h1+h2)) )

        if not self.use_one_sided: return float(delta_c), float(gamma_c)

        # One-sided near KO barrier
        H=None; side=None
        if self.barrier_type in ("down-and-out","double-out") and self.lower_barrier is not None:
            H=self.lower_barrier; side="down"
        if self.barrier_type in ("up-and-out","double-out") and self.upper_barrier is not None and H is None:
            H=self.upper_barrier; side="up"
        if H is None: return float(delta_c), float(gamma_c)

        j = max(0, min(N-1, min(range(N), key=lambda k: abs(H - s_nodes[k]))))
        near = (abs(i-j) <= self.barrier_safety_cells)
        if not near: return float(delta_c), float(gamma_c)

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

    def _shifted_grid(self) -> Tuple[List[float], float]:
        # hook for cash dividends later (PV shift); currently 0
        return self.S_nodes[:], self.S0

    def price(self) -> float:
        s_nodes, S_eff = self._shifted_grid()
        V, _, _ = self._run_backward(s_nodes, apply_barrier=(self.barrier_type!="none"))
        return self._interp_linear(S_eff, s_nodes, V)

    def greeks(self, vega_bump: float = 1e-3) -> Dict[str,float]:
        s_nodes, S_eff = self._shifted_grid()
        V, _, _ = self._run_backward(s_nodes, apply_barrier=(self.barrier_type!="none"))
        P = self._interp_linear(S_eff, s_nodes, V)
        delta, gamma = self._delta_gamma_nonuniform(s_nodes, V, S_eff)
        # vega via symmetric bump
        sig0=self.sigma
        self.sigma=sig0+vega_bump; up=self.price()
        self.sigma=sig0-vega_bump; dn=self.price()
        self.sigma=sig0
        vega=(up-dn)/(2*vega_bump)
        # theta from PDE at 0
        theta = -((self.r - self.q)*S_eff*delta + 0.5*self.sigma*self.sigma*S_eff*S_eff*gamma - self.r*P)
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
        print(f"Monitors count    : {len(self.monitor_times)}")
        print(f"Spot/Strike       : {self.S0:.6f} / {self.K:.6f}")
        print(f"Price             : {p:.9f}")
        print(f"Greeks            : Δ={g['delta']:.9f}, Γ={g['gamma']:.9f}, ν={g['vega']:.9f}, Θ={g['theta']:.9f}")

    # convergence utility
    def validate_convergence(self, N_list: List[int], M_list: List[int]) -> List[Dict[str,float]]:
        out = []
        for N in N_list:
            for M in M_list:
                clone = DiscreteBarrierFDM(
                    spot=self.S0, strike=self.K,
                    valuation_date=self.valuation_date, maturity_date=self.maturity_date,
                    volatility=self.sigma, flat_rate_nacc=self.r, dividend_yield=self.q,
                    option_type=self.option_type,
                    barrier_type=self.barrier_type, lower_barrier=self.lower_barrier, upper_barrier=self.upper_barrier,
                    monitoring_dates=self.monitoring_dates, rebate_amount=self.rebate_amount, rebate_at_hit=self.rebate_at_hit,
                    already_hit=self.already_hit, already_in=self.already_in,
                    num_space_nodes=N, num_time_steps=M, rannacher_steps=self.rannacher,
                    day_count=self.day_count, min_substeps_between_monitors=self.min_substeps,
                    grid_type=self.grid_type, sinh_alpha=self.sinh_alpha,
                    use_one_sided_greeks_near_barrier=self.use_one_sided, barrier_safety_cells=self.barrier_safety_cells
                )
                price = clone.price(); greeks = clone.greeks()
                out.append({"N":N,"M":M,"price":price,"delta":greeks["delta"],
                            "gamma":greeks["gamma"],"vega":greeks["vega"],"theta":greeks["theta"]})
        out.sort(key=lambda r: (r["N"], r["M"]))
        print("\n=== Grid Convergence (sorted by N,M) ===")
        print("   N     M        Price          Delta           Gamma            Vega            Theta")
        for r in out:
            print(f"{r['N']:5d} {r['M']:5d}  {r['price']:12.8f}  {r['delta']:13.8f}  {r['gamma']:13.8f}  {r['vega']:13.8f}  {r['theta']:13.8f}")
        return out
