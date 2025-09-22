"""
Log-price Crank–Nicolson FDM for discrete barrier European options.
- Grid coordinate x = log(S). Uniform spacing in x.
- You choose the number of time steps M; the space step Δx is chosen from the
  "optimal relation" Δt ≈ λ * (Δx)^2 / (0.5 σ^2)  (default λ≈0.45).
  This sets Δx from Δt, then N from the domain [xmin, xmax].
- Discrete monitoring: KO projection only at monitoring dates. KI via parity.
- Rebates (at-hit vs paid-at-expiry).
- Status flags: already_hit (KO), already_in (KI).
- Greeks (Δ, Γ, vega, theta) computed in log-space with chain rule; near KO barrier
  we use FIS-style one-sided Δ in the first cell and blended Δ in the second cell.
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
                 # instrument / market
                 spot: float,
                 strike: float,
                 valuation_date: date,
                 maturity_date: date,
                 volatility: float,
                 flat_rate_nacc: float,
                 dividend_yield: float,
                 option_type: OptionType,
                 # barrier
                 barrier_type: BarrierType = "none",
                 lower_barrier: Optional[float] = None,
                 upper_barrier: Optional[float] = None,
                 monitoring_dates: Optional[List[date]] = None,
                 rebate_amount: float = 0.0,
                 rebate_at_hit: bool = True,
                 # status flags
                 already_hit: bool = False,
                 already_in: bool = False,
                 # numerics (YOU SPECIFY ONLY M; N is derived)
                 num_time_steps: int = 800,
                 rannacher_steps: int = 2,
                 lambda_target: float = 0.45,   # targets Δt ≈ λ*(Δx)^2/(0.5σ^2)
                 day_count: str = "ACT/365"):
        self.S0 = float(spot)
        self.K  = float(strike)
        self.valuation_date = valuation_date
        self.maturity_date  = maturity_date
        self.sigma = float(volatility)
        self.r = float(flat_rate_nacc)
        self.q = float(dividend_yield)
        self.opt_type = option_type

        self.barrier_type  = barrier_type
        self.lower_barrier = lower_barrier
        self.upper_barrier = upper_barrier
        self.monitoring_dates = sorted(monitoring_dates or [])
        self.rebate_amount = float(rebate_amount)
        self.rebate_at_hit = bool(rebate_at_hit)

        self.already_hit = bool(already_hit)
        self.already_in  = bool(already_in)

        self.M = max(1, int(num_time_steps))
        self.rannacher = int(rannacher_steps)
        self.lambda_target = float(lambda_target)
        self.day_count = day_count.upper()

        # time to maturity
        self.T = self._year_fraction(self.valuation_date, self.maturity_date)
        self.dt = self.T / self.M

        # build log-space grid from Δt via optimal relation
        self._build_log_grid_from_time()

        # exact monitor times
        self.monitor_times = self._build_monitor_times_exact()

    # ---------------- utilities ----------------
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
        times=[0.0]
        for d in self.monitoring_dates:
            if self.valuation_date <= d <= self.maturity_date:
                t=self._year_fraction(self.valuation_date, d)
                if 0.0<=t<=self.T: times.append(t)
        if times[-1] < self.T - 1e-14:
            times.append(self.T)
        return sorted(set(times))

    # --------------- grid construction (log S) ----------------
    def _build_log_grid_from_time(self) -> None:
        # choose domain in S first, then x=log S
        Sref = max(self.S0, self.K, *(x for x in [self.lower_barrier, self.upper_barrier] if x is not None))
        Smax = 4.5*Sref*math.exp(self.sigma*math.sqrt(max(self.T,1e-12)))
        Smin = max(1e-12, Smax/4.5**2)  # loose lower
        self.xmin = math.log(Smin)
        self.xmax = math.log(Smax)

        # from optimal relation: dt ≈ λ*(dx)^2/(0.5σ^2) => dx = sqrt(0.5 σ^2 dt / λ)
        dx = math.sqrt(max(1e-18, 0.5*self.sigma*self.sigma*self.dt / max(1e-12, self.lambda_target)))
        N = int(math.ceil((self.xmax - self.xmin)/dx))
        N = max(200, N)  # ensure enough nodes
        self.N = N
        self.dx = (self.xmax - self.xmin)/N
        self.X_nodes = [self.xmin + i*self.dx for i in range(N+1)]
        self.S_nodes = [math.exp(x) for x in self.X_nodes]

        # snap key points
        def snap(value: Optional[float]):
            if value is None: return None
            xv = math.log(max(value, 1e-300))
            j = min(range(len(self.X_nodes)), key=lambda i: abs(self.X_nodes[i]-xv))
            self.X_nodes[j] = xv
            self.S_nodes[j] = math.exp(xv)
            return j
        snap(self.K)
        snap(self.lower_barrier)
        snap(self.upper_barrier)

    # --------------- payoff / BC / projection -----------------
    def _terminal_payoff(self) -> List[float]:
        if self.opt_type=="call":
            return [max(S-self.K,0.0) for S in self.S_nodes]
        return [max(self.K-S,0.0) for S in self.S_nodes]

    def _apply_KO_projection(self, V: List[float], tau_left: float) -> None:
        if self.barrier_type in ("none","down-and-in","up-and-in","double-in"):
            return
        lo, up = self.lower_barrier, self.upper_barrier
        rebate = self.rebate_amount if self.rebate_at_hit else self.rebate_amount*math.exp(-self.r*tau_left)
        for i,S in enumerate(self.S_nodes):
            out=False
            if self.barrier_type=="down-and-out" and lo is not None and S<=lo: out=True
            elif self.barrier_type=="up-and-out" and up is not None and S>=up: out=True
            elif self.barrier_type=="double-out":
                if (lo is not None and S<=lo) or (up is not None and S>=up): out=True
            if out: V[i]=rebate

    # --------------- Thomas solver -----------------------------
    @staticmethod
    def _solve_tridiagonal(a: List[float], b: List[float], c: List[float], d: List[float]) -> List[float]:
        n=len(d)
        cp=[0.0]*n; dp=[0.0]*n; x=[0.0]*n
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

    # --------------- CN step on [t0,t1] with m steps (log grid) --------
    def _cn_log_subinterval(self, V: List[float], t0: float, t1: float,
                            theta: float, rannacher_left: int, m_steps: int) -> Tuple[List[float], float]:
        N = self.N
        dx = self.dx
        a = 0.5*self.sigma*self.sigma
        mu_t = self.r - self.q - 0.5*self.sigma*self.sigma

        L = t1 - t0
        m = max(1, int(m_steps))
        dt = L / m
        last_dt_zero = None

        for step in range(m):
            use_theta = 1.0 if (rannacher_left > 0) else theta
            if rannacher_left > 0: rannacher_left -= 1

            tau_left = t0 + (m-step-1)*dt

            sub=[0.0]*(N+1); main=[0.0]*(N+1); sup=[0.0]*(N+1); rhs=[0.0]*(N+1)

            # boundary values at "after-step" time
            Smin = self.S_nodes[0]; Smax = self.S_nodes[-1]
            if self.opt_type=="call":
                rhs[0]=0.0
                rhs[N]=Smax*math.exp(-self.q*tau_left) - self.K*math.exp(-self.r*tau_left)
            else:
                rhs[0]=self.K*math.exp(-self.r*tau_left)
                rhs[N]=0.0
            main[0]=1.0; main[N]=1.0

            # tri-diagonals (uniform in x)
            Ai = a/(dx*dx); Bi = mu_t/(2*dx); Ci = self.r
            for i in range(1,N):
                aI = use_theta*dt*( Ai - Bi )
                bI = 1.0 + use_theta*dt*( -2.0*Ai - Ci ) * (-1.0)  # i.e., 1 + θΔt(2Ai + Ci)
                cI = use_theta*dt*( Ai + Bi )
                aE = (1.0-use_theta)*dt*( Ai - Bi )
                bE = 1.0 - (1.0-use_theta)*dt*( 2.0*Ai + Ci )
                cE = (1.0-use_theta)*dt*( Ai + Bi )

                sub[i]  = -aI
                main[i] =  bI
                sup[i]  = -cI
                rhs[i]  =  aE*V[i-1] + bE*V[i] + cE*V[i+1]

            V = self._solve_tridiagonal(sub, main, sup, rhs)

            if abs(t0) < 1e-14 and step == m-1:
                last_dt_zero = dt

        return V, (last_dt_zero if last_dt_zero is not None else dt)

    # --------------- time alloc & full run ------------------------------
    def _time_subgrid_counts(self) -> List[int]:
        lengths = [self.monitor_times[i+1]-self.monitor_times[i] for i in range(len(self.monitor_times)-1)]
        total = sum(lengths) or 1.0
        raw = [max(1, int(round(self.M * (L/total)))) for L in lengths]
        # force exact sum
        diff = self.M - sum(raw)
        i=0
        while diff!=0 and len(raw)>0:
            j=i%len(raw)
            if diff>0: raw[j]+=1; diff-=1
            else:
                if raw[j]>1: raw[j]-=1; diff+=1
            i+=1
        return raw

    def _run_backward(self, apply_barrier: bool) -> Tuple[List[float], float]:
        N = self.N
        V = self._terminal_payoff()

        # t=0 status flags
        if apply_barrier:
            if self.barrier_type in ("down-and-out","up-and-out","double-out") and self.already_hit:
                instant = self.rebate_amount if self.rebate_at_hit else self.rebate_amount*math.exp(-self.r*self.T)
                return [instant]*(N+1), 0.0
            if self.barrier_type in ("down-and-in","up-and-in","double-in") and self.already_in:
                apply_barrier = False

        theta=0.5
        subcounts = self._time_subgrid_counts()
        rannacher_left = self.rannacher
        dt_last_global = None

        for k in range(len(self.monitor_times)-1,0,-1):
            t0 = self.monitor_times[k-1]
            t1 = self.monitor_times[k]
            m_steps = subcounts[k-1]
            V, dt_last = self._cn_log_subinterval(V, t0, t1, theta, rannacher_left, m_steps)
            rannacher_left = max(0, rannacher_left - m_steps)

            if apply_barrier and (abs(t0)>1e-14):
                self._apply_KO_projection(V, tau_left=t0)

            if abs(t0) < 1e-14:
                dt_last_global = dt_last

        if apply_barrier and self.barrier_type in ("down-and-in","up-and-in","double-in"):
            V_van, _ = self._run_backward(apply_barrier=False)
            V = [V_van[i]-V[i] for i in range(len(V))]

        return V, (dt_last_global if dt_last_global is not None else self.dt)

    # --------------- interpolation & Greeks (log grid) ------------------
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

    def _delta_gamma_log(self, V: List[float], S_eff: float) -> Tuple[float,float]:
        # locate on x-grid
        x = math.log(max(S_eff,1e-300))
        xs = self.X_nodes; i = min(range(len(xs)), key=lambda k: abs(xs[k]-x))
        i = max(1, min(self.N-1, i))
        dx = self.dx
        # central
        Ux  = (V[i+1]-V[i-1])/(2*dx)
        Uxx = (V[i+1]-2*V[i]+V[i-1])/(dx*dx)

        # FIS-style barrier-aware Δ: one-sided in the first cell off KO barrier; blended in second
        def ko_side_and_index():
            if self.barrier_type in ("down-and-out","double-out") and self.lower_barrier is not None:
                xb = math.log(self.lower_barrier); side="down"
            elif self.barrier_type in ("up-and-out","double-out") and self.upper_barrier is not None:
                xb = math.log(self.upper_barrier); side="up"
            else:
                return None, None, None
            j = min(range(len(xs)), key=lambda k: abs(xs[k]-xb))
            return side, xb, j

        side, xb, j = ko_side_and_index()
        S = S_eff
        if side is not None:
            # distance in index
            kdist = abs(i-j)
            if kdist==1:
                if side=="down":
                    Ux = (V[j+2]-V[j+1])/dx  # forward (away from barrier)
                else:
                    Ux = (V[j]-V[j-1])/dx    # backward (away from barrier)
            elif kdist==2:
                # linear blend: q in [1..0] from closest to next
                q = 0.5
                if side=="down":
                    Ux_os = (V[j+2]-V[j+1])/dx
                    Ux_ce = (V[i+1]-V[i-1])/(2*dx)
                    Ux = q*Ux_os + (1-q)*Ux_ce
                else:
                    Ux_os = (V[j]-V[j-1])/dx
                    Ux_ce = (V[i+1]-V[i-1])/(2*dx)
                    Ux = q*Ux_os + (1-q)*Ux_ce

        delta = (1.0/S)*Ux
        gamma = (Uxx - Ux)/(S*S)
        gamma = max(min(gamma, 1e6), -1e6)
        return float(delta), float(gamma)

    # ---------------- public API -----------------
    def price(self) -> float:
        V, _ = self._run_backward(apply_barrier=(self.barrier_type!="none"))
        return self._interp_linear(self.S0, self.S_nodes, V)

    def greeks(self, vega_bump: float = 1e-3) -> Dict[str,float]:
        V, _ = self._run_backward(apply_barrier=(self.barrier_type!="none"))
        price_here = self._interp_linear(self.S0, self.S_nodes, V)
        delta, gamma = self._delta_gamma_log(V, self.S0)

        sig0=self.sigma
        self.sigma=sig0+vega_bump; self._build_log_grid_from_time(); up=self.price()
        self.sigma=sig0-vega_bump; self._build_log_grid_from_time(); dn=self.price()
        self.sigma=sig0;           self._build_log_grid_from_time()
        vega=(up-dn)/(2*vega_bump)

        theta = -((self.r - self.q)*self.S0*delta + 0.5*self.sigma*self.sigma*self.S0*self.S0*gamma - self.r*price_here)
        return {"delta":float(delta), "gamma":float(gamma), "vega":float(vega), "theta":float(theta)}

    def print_details(self) -> None:
        p=self.price(); g=self.greeks()
        print("==== Discrete Barrier Option (CN, log-space) — Discrete monitors, optimal Δt–Δx ====")
        print(f"T (years)         : {self.T:.9f}   [{self.day_count}]")
        print(f"σ / r / q         : {self.sigma:.9f} / {self.r:.9f} / {self.q:.9f}")
        print(f"Barrier type      : {self.barrier_type}  (lo={self.lower_barrier}, up={self.upper_barrier})  rebate={self.rebate_amount} @hit={self.rebate_at_hit}")
        print(f"Status (hit/in)   : {self.already_hit} / {self.already_in}")
        print(f"Grid (M,N,dx)     : {self.M}, {self.N}, {self.dx:.6g}")
        print(f"Spot/Strike       : {self.S0:.6f} / {self.K:.6f}")
        print(f"Price             : {p:.9f}")
        print(f"Greeks            : Δ={g['delta']:.9f}, Γ={g['gamma']:.9f}, ν={g['vega']:.9f}, Θ={g['theta']:.9f}")