# discrete_barrier_fdm_pricer.py
# -*- coding: utf-8 -*-
"""
Discrete barrier European option pricer via Black–Scholes PDE solved with
Crank–Nicolson (theta-scheme, theta=0.5) and optional Rannacher start.
- No BGK adjustment (pure FD).
- Discrete monitoring: apply KO projection only on monitoring dates.
- KI valued via in–out parity (Vanilla − KO).
- Rebate supported (at hit or paid at expiry).
- Greeks: Δ, Γ (one‑sided stabilization near KO barrier), vega via σ bump.
- Optional escrow of cash dividends (PV-shift).
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
                 spot: float,
                 strike: float,
                 valuation_date: date,
                 maturity_date: date,
                 volatility: float,
                 flat_rate_nacc: float,
                 option_type: OptionType,
                 barrier_type: BarrierType = "none",
                 lower_barrier: Optional[float] = None,
                 upper_barrier: Optional[float] = None,
                 monitoring_dates: Optional[List[date]] = None,
                 rebate_amount: float = 0.0,
                 rebate_at_hit: bool = True,
                 # dividends / carry
                 dividend_yield: float = 0.0,                      # continuous q (set 0 if unused)
                 dividend_list: Optional[List[Tuple[date,float]]] = None,  # cash dividends (PV-escrow)
                 # numerics
                 num_space_nodes: int = 600,
                 num_time_steps: int = 600,
                 rannacher_steps: int = 2,
                 day_count: str = "ACT/365",
                 # Greeks stabilization
                 use_one_sided_greeks_near_barrier: bool = True,
                 barrier_safety_cells: int = 2):
        # instrument / market
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
        # numerics
        self.N = int(num_space_nodes)
        self.M = int(num_time_steps)
        self.rannacher = int(rannacher_steps)
        self.day_count = day_count.upper()
        self.use_one_sided = bool(use_one_sided_greeks_near_barrier)
        self.barrier_safety_cells = int(barrier_safety_cells)
        # dividends
        self.dividends = [(d,float(a)) for (d,a) in (dividend_list or [])]

        # time
        self.T = self._year_fraction(self.valuation_date, self.maturity_date)
        self.dt = max(self.T / max(1,self.M), 1e-12)

        # space grid
        self.S_nodes = self._build_space_grid()
        self.dS = self.S_nodes[1] - self.S_nodes[0]

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
            if tau>0: pv += amt*math.exp(-self.r*tau)
        return pv

    def _build_space_grid(self) -> List[float]:
        anchors = [self.S0, self.K]
        if self.lower_barrier is not None: anchors.append(self.lower_barrier)
        if self.upper_barrier is not None: anchors.append(self.upper_barrier)
        s_ref = max(anchors) if anchors else self.S0
        # generous Smax; CN is stable; choose uniform grid for clarity
        Smax = 4.5*s_ref*math.exp(self.sigma*math.sqrt(max(self.T,1e-12)))
        Smin = 0.0
        N = max(200,self.N)
        dS = (Smax-Smin)/N
        nodes = [Smin+i*dS for i in range(N+1)]
        # snap K and barriers to nearest node (helps projection smoothness)
        def snap(x: Optional[float]):
            if x is None: return
            j = min(range(len(nodes)), key=lambda i: abs(nodes[i]-x))
            nodes[j]=float(x)
        snap(self.K); snap(self.lower_barrier); snap(self.upper_barrier)
        return nodes

    # ---------------- payoff / BCs / projection ----------------
    def _terminal_payoff(self, s_nodes: List[float]) -> List[float]:
        if self.option_type=="call":
            return [max(s-self.K,0.0) for s in s_nodes]
        return [max(self.K-s,0.0) for s in s_nodes]

    def _apply_KO_projection(self, V: List[float], s_nodes: List[float], tau_left: float) -> None:
        if self.barrier_type in ("none","down-and-in","up-and-in","double-in"):
            return
        lo = self.lower_barrier; up = self.upper_barrier
        if self.rebate_at_hit:
            rebate = self.rebate_amount
        else:
            rebate = self.rebate_amount*math.exp(-self.r*tau_left)  # PV to expiry
        for i,s in enumerate(s_nodes):
            out=False
            if self.barrier_type=="down-and-out" and lo is not None and s<=lo: out=True
            elif self.barrier_type=="up-and-out" and up is not None and s>=up: out=True
            elif self.barrier_type=="double-out":
                if (lo is not None and s<=lo) or (up is not None and s>=up): out=True
            if out: V[i]=rebate

    def _monitor_index_map(self) -> Dict[int,float]:
        """Map time index k to time-remaining tau where projection is applied AFTER stepping to k."""
        m: Dict[int,float]={}
        for d in self.monitoring_dates:
            if self.valuation_date < d <= self.maturity_date:
                t = self._year_fraction(self.valuation_date, d)
                k = int(round(t/self.dt))
                k = max(1, min(self.M, k))
                tau_left = max(0.0, self.T - k*self.dt)
                m[k]=tau_left
        # include maturity to ensure projection at T (harmless for vanilla)
        m.setdefault(self.M, 0.0)
        return m

    # ---------------- core solver (CN θ=0.5 with Rannacher) -------------
    def _backward_CN(self, s_nodes: List[float], apply_barrier: bool) -> List[float]:
        N=self.N; M=self.M; dt=self.dt; r=self.r; q=self.q; sig=self.sigma
        dS = s_nodes[1]-s_nodes[0]
        mu = r - q

        V = self._terminal_payoff(s_nodes)

        monitor = self._monitor_index_map() if apply_barrier else {}

        for m in range(M,0,-1):
            theta = 1.0 if (M - m) < self.rannacher else 0.5

            sub=[0.0]*(N+1); main=[0.0]*(N+1); sup=[0.0]*(N+1); rhs=[0.0]*(N+1)

            # boundary values at the NEW time layer (after step), with tau_left after step k=m-1
            tau_left = self.T - (m-1)*dt
            if self.option_type=="call":
                rhs[0]=0.0
                rhs[N]=s_nodes[-1]*math.exp(-q*tau_left) - self.K*math.exp(-r*tau_left)
            else:
                rhs[0]=self.K*math.exp(-r*tau_left)
                rhs[N]=0.0
            main[0]=1.0; main[N]=1.0

            # interior coefficients
            for i in range(1,N):
                S = s_nodes[i]
                A = 0.5*sig*sig*S*S/(dS*dS)        # diffusion
                B = 0.5*mu*S/dS                    # convection (NOTE the 1/2 factor)
                C = r                               # discount

                a_impl = theta*dt*(A - B)
                b_impl = 1.0 + theta*dt*(2*A + C)
                c_impl = theta*dt*(A + B)

                a_expl = (1-theta)*dt*(A - B)
                b_expl = 1.0 - (1-theta)*dt*(2*A + C)
                c_expl = (1-theta)*dt*(A + B)

                sub[i]  = -a_impl
                main[i] =  b_impl
                sup[i]  = -c_impl
                rhs[i]  =  a_expl*V[i-1] + b_expl*V[i] + c_expl*V[i+1]

            # solve tridiagonal
            V = self._solve_tridiagonal(sub, main, sup, rhs)

            # Apply KO projection at monitoring dates (discrete monitoring)
            if apply_barrier and (m-1) in monitor:
                self._apply_KO_projection(V, s_nodes, tau_left=monitor[m-1])

        # For KI types: return Vanilla - KO
        if apply_barrier and self.barrier_type in ("down-and-in","up-and-in","double-in"):
            V_van = self._backward_CN(s_nodes, apply_barrier=False)
            V = [V_van[i]-V[i] for i in range(len(V))]

        return V

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

    def _delta_gamma_from_grid(self, s_nodes: List[float], V: List[float], S_eff: float) -> Tuple[float,float]:
        N=len(s_nodes)-1; dS=s_nodes[1]-s_nodes[0]
        # nearest interior index
        i = max(1, min(N-1, min(range(N+1), key=lambda k: abs(s_nodes[k]-S_eff))))
        delta_c = (V[i+1]-V[i-1])/(2*dS)
        gamma_c = (V[i+1]-2*V[i]+V[i-1])/(dS*dS)

        if not self.use_one_sided: return float(delta_c), float(gamma_c)

        # determine proximity to KO barrier
        H=None; side=None
        if self.barrier_type in ("down-and-out","double-out") and self.lower_barrier is not None:
            H=self.lower_barrier; side="down"
        if self.barrier_type in ("up-and-out","double-out") and self.upper_barrier is not None:
            if H is None: H=self.upper_barrier; side="up"
        if H is None: return float(delta_c), float(gamma_c)
        # locate barrier interval
        j = max(0, min(N-1, int((H - s_nodes[0]) / dS)))  # coarse
        close = (abs(i-j)<=self.barrier_safety_cells)
        if not close: return float(delta_c), float(gamma_c)

        if side=="down":
            # forward derivative away from barrier
            i = max(1, min(N-1, j+1))
            delta = (V[i+1]-V[i])/dS
        else:
            i = max(1, min(N-1, j))
            delta = (V[i]-V[i-1])/dS
        gamma = (V[i+1]-2*V[i]+V[i-1])/(dS*dS)
        # cap extreme spikes
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
        V = self._backward_CN(s_nodes, apply_barrier=(self.barrier_type!="none"))
        return self._interp_linear(S_eff, s_nodes, V)

    def greeks(self, vega_bump: float = 1e-3) -> Dict[str,float]:
        s_nodes, S_eff = self._shifted_grid()
        V = self._backward_CN(s_nodes, apply_barrier=(self.barrier_type!="none"))
        delta, gamma = self._delta_gamma_from_grid(s_nodes, V, S_eff)

        sig0=self.sigma
        self.sigma=sig0+vega_bump; up=self.price()
        self.sigma=sig0-vega_bump; dn=self.price()
        self.sigma=sig0
        vega=(up-dn)/(2*vega_bump)

        return {"delta":float(delta), "gamma":float(gamma), "vega":float(vega)}

    # ---------------------- linear solver -------------------------------
    @staticmethod
    def _solve_tridiagonal(a_sub: List[float], a_diag: List[float],
                           a_sup: List[float], rhs: List[float]) -> List[float]:
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

    # ------------------------ pretty print ------------------------------
    def print_details(self) -> None:
        p=self.price(); g=self.greeks()
        print("==== Discrete Barrier Option (FD CN; Rannacher) — No BGK ====")
        print(f"Maturity Date     : {self.maturity_date.isoformat()}")
        print(f"T (years)         : {self.T:.9f}   [{self.day_count}]")
        print(f"Volatility (σ)    : {self.sigma:.9f}")
        print(f"r (NACC) / q      : {self.r:.9f} / {self.q:.9f}")
        print(f"PV(divs, escrow)  : {self._pv_dividends_escrow():.9f}")
        print("")
        print(f"Barrier type      : {self.barrier_type}")
        print(f"KO lower / upper  : {self.lower_barrier} / {self.upper_barrier}")
        print(f"Rebate (amt/hit)  : {self.rebate_amount} / {self.rebate_at_hit}")
        print("")
        print(f"Grid (S,N / t,M)  : {len(self.S_nodes)}, {self.M} (Rannacher {self.rannacher})")
        print(f"Spot / Strike     : {self.S0:.6f} / {self.K:.6f}")
        print("")
        print(f"Price             : {p:.9f}")
        print(f"Greeks            : delta={g['delta']:.9f}, gamma={g['gamma']:.9f}, vega={g['vega']:.9f}")
