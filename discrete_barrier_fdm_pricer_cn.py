"""
Log-price Crank–Nicolson FDM for discrete barrier European options.

- Grid coordinate: x = log(S), uniform in x.
- You choose time steps M; the space step Δx and number of nodes N are derived from:
      Δt ≈ λ * (Δx)^2 / (0.5 σ^2)    (λ ≈ 0.45 by default)
- Discrete monitoring: KO projection at monitoring dates only. KI via parity.
- Rebates: paid at hit (immediate) or at expiry (PV’d).
- Status flags: already_hit (KO), already_in (KI).
- Rannacher start: first few implicit steps to damp CN oscillations.
- Greeks (Δ, Γ, vega, theta): computed on the log-grid via chain rule.
- FIS-style barrier treatment:
    • non-symmetric (upwind) PDE stencil at the node adjacent to a KO barrier,
      to avoid mixing information across the barrier;
    • one-sided Δ in the first cell off the barrier and blended Δ in the second.
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


class DiscreteBarrierFDM:
    def __init__(self,
                 # instrument & market
                 spot: float,
                 strike: float,
                 valuation_date: date,
                 maturity_date: date,
                 volatility: float,
                 flat_rate_nacc: float,
                 dividend_yield: float,
                 option_type: OptionType,
                 # barrier package
                 barrier_type: BarrierType = "none",
                 lower_barrier: Optional[float] = None,
                 upper_barrier: Optional[float] = None,
                 monitoring_dates: Optional[List[date]] = None,
                 rebate_amount: float = 0.0,
                 rebate_at_hit: bool = True,
                 # status flags
                 already_hit: bool = False,   # relevant for KO
                 already_in: bool = False,    # relevant for KI
                 # numerics (YOU select only M; N is derived)
                 num_time_steps: int = 900,
                 rannacher_steps: int = 2,
                 lambda_target: float = 0.45,   # Δt ≈ λ (Δx)^2 / (0.5 σ^2)
                 day_count: str = "ACT/365"):
        # economics
        self.S0 = float(spot)
        self.K = float(strike)
        self.valuation_date = valuation_date
        self.maturity_date  = maturity_date
        self.sigma = float(volatility)
        self.r = float(flat_rate_nacc)
        self.q = float(dividend_yield)
        self.opt_type = option_type

        # barrier set
        self.barrier_type = barrier_type
        self.lower_barrier = lower_barrier
        self.upper_barrier = upper_barrier
        self.monitoring_dates = sorted(monitoring_dates or [])
        self.rebate_amount = float(rebate_amount)
        self.rebate_at_hit = bool(rebate_at_hit)

        # status
        self.already_hit = bool(already_hit)
        self.already_in  = bool(already_in)

        # numerics
        self.M = max(1, int(num_time_steps))
        self.rannacher = int(rannacher_steps)
        self.lambda_target = float(lambda_target)
        self.day_count = day_count.upper()

        # time
        self.T  = self._year_fraction(self.valuation_date, self.maturity_date)
        self.dt = self.T / self.M

        # build log-space grid from Δt–Δx optimal relation
        self._build_log_grid_from_time()

        # exact monitor times (includes 0 and T)
        self.monitor_times = self._build_monitor_times_exact()

    # -------------------- utilities --------------------

    def _year_fraction(self, d0: date, d1: date) -> float:
        days = max(0, (d1 - d0).days)
        if self.day_count in ("ACT/360",):
            return days / 360.0
        if self.day_count in ("30/360", "30E/360"):
            y0, m0, d0_ = d0.year, d0.month, min(d0.day, 30)
            y1, m1, d1_ = d1.year, d1.month, min(d1.day, 30)
            return ((y1 - y0) * 360 + (m1 - m0) * 30 + (d1_ - d0_)) / 360.0
        return days / 365.0

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

    # -------------------- grid (log S) --------------------

    def _build_log_grid_from_time(self) -> None:
        """Derive Δx (and N) from chosen M via Δt ≈ λ (Δx)^2 / (0.5 σ^2),
        then build a uniform log(S) grid [xmin, xmax]."""
        # domain in S, then map to x
        Sref = max(self.S0, self.K, *(x for x in [self.lower_barrier, self.upper_barrier] if x is not None))
        Smax = 4.5 * Sref * math.exp(self.sigma * math.sqrt(max(self.T, 1e-12)))
        Smin = max(1e-12, Smax / 4.5**2)
        self.xmin = math.log(Smin)
        self.xmax = math.log(Smax)

        # Δx from optimal relation
        dx = math.sqrt(max(1e-18, 0.5 * self.sigma * self.sigma * self.dt / max(1e-12, self.lambda_target)))
        N  = int(math.ceil((self.xmax - self.xmin) / dx))
        N  = max(200, N)
        self.N  = N
        self.dx = (self.xmax - self.xmin) / N

        self.X_nodes = [self.xmin + i * self.dx for i in range(N + 1)]
        self.S_nodes = [math.exp(x) for x in self.X_nodes]

        # snap critical levels (strike, barriers) to nearest node
        def snap(value: Optional[float]) -> None:
            if value is None: 
                return
            xv = math.log(max(value, 1e-300))
            j  = min(range(len(self.X_nodes)), key=lambda i: abs(self.X_nodes[i] - xv))
            self.X_nodes[j] = xv
            self.S_nodes[j] = math.exp(xv)

        snap(self.K)
        snap(self.lower_barrier)
        snap(self.upper_barrier)

    # ---------------- payoff / BC / KO projection ----------------

    def _terminal_payoff(self) -> List[float]:
        if self.opt_type == "call":
            return [max(S - self.K, 0.0) for S in self.S_nodes]
        return [max(self.K - S, 0.0) for S in self.S_nodes]

    def _apply_KO_projection(self, V: List[float], tau_left: float) -> None:
        """Apply knock-out + rebate at a monitoring date (discrete)."""
        if self.barrier_type in ("none", "down-and-in", "up-and-in", "double-in"):
            return

        lo, up = self.lower_barrier, self.upper_barrier
        if self.rebate_at_hit:
            rebate = self.rebate_amount
        else:
            rebate = self.rebate_amount * math.exp(-self.r * tau_left)

        for i, S in enumerate(self.S_nodes):
            knocked = False
            if self.barrier_type == "down-and-out" and lo is not None and S <= lo:
                knocked = True
            elif self.barrier_type == "up-and-out" and up is not None and S >= up:
                knocked = True
            elif self.barrier_type == "double-out":
                if (lo is not None and S <= lo) or (up is not None and S >= up):
                    knocked = True
            if knocked:
                V[i] = rebate

    # ------------------ tridiagonal solver ------------------

    @staticmethod
    def _solve_tridiagonal(a: List[float], b: List[float], c: List[float], d: List[float]) -> List[float]:
        """Thomas algorithm. a: subdiag, b: diag, c: superdiag, d: rhs."""
        n = len(d)
        cp = [0.0] * n
        dp = [0.0] * n
        x  = [0.0] * n

        beta = b[0]
        if abs(beta) < 1e-14:
            beta = 1e-14
        cp[0] = c[0] / beta
        dp[0] = d[0] / beta

        for i in range(1, n):
            beta = b[i] - a[i] * cp[i - 1]
            if abs(beta) < 1e-14:
                beta = 1e-14
            cp[i] = (c[i] / beta) if i < n - 1 else 0.0
            dp[i] = (d[i] - a[i] * dp[i - 1]) / beta

        x[-1] = dp[-1]
        for i in range(n - 2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i + 1]
        return x

    # ------------- CN step on [t0, t1] with m steps (log grid) -------------

    def _cn_log_subinterval(self, V: List[float], t0: float, t1: float,
                            theta: float, rannacher_left: int, m_steps: int) -> Tuple[List[float], float]:
        """
        Backward propagation from t1 → t0 in m_steps using CN (θ=0.5) or Rannacher (θ=1 on first steps).
        PDE in x=log(S):  U_t + μ~ U_x + (1/2)σ^2 U_xx - r U = 0.
        """
        N  = self.N
        dx = self.dx
        a  = 0.5 * self.sigma * self.sigma
        mu_t = self.r - self.q - 0.5 * self.sigma * self.sigma

        L_interval = t1 - t0
        m  = max(1, int(m_steps))
        dt = L_interval / m
        last_dt_at_zero = None

        # Precompute drift/diffusion constants
        Ai = a / (dx * dx)
        Bi = mu_t / (2.0 * dx)
        Ci = self.r

        # Identify node adjacent to KO barrier for non-symmetric PDE stencil (FIS-style)
        def ko_adjacent_index() -> Tuple[Optional[int], Optional[str]]:
            """
            Returns (i_adj, side) if a KO barrier is active:
              side == 'down': down-and-out (barrier below grid region) ⇒ use forward/upwind away from barrier
              side == 'up'  : up-and-out   (barrier above grid region) ⇒ use backward/upwind away from barrier
            """
            if self.barrier_type in ("down-and-out", "double-out") and self.lower_barrier is not None:
                xb = math.log(max(self.lower_barrier, 1e-300))
                j  = min(range(N + 1), key=lambda k: abs(self.X_nodes[k] - xb))
                j  = max(1, min(N - 1, j))
                # choose the interior node just above the barrier
                if self.X_nodes[j] < xb and j < N:
                    j += 1
                return j, "down"

            if self.barrier_type in ("up-and-out", "double-out") and self.upper_barrier is not None:
                xb = math.log(max(self.upper_barrier, 1e-300))
                j  = min(range(N + 1), key=lambda k: abs(self.X_nodes[k] - xb))
                j  = max(1, min(N - 1, j))
                # choose the interior node just below the barrier
                if self.X_nodes[j] > xb and j > 0:
                    j -= 1
                return j, "up"

            return None, None

        i_adj, side_adj = ko_adjacent_index()

        for step in range(m):
            use_theta = 1.0 if (rannacher_left > 0) else theta
            if rannacher_left > 0:
                rannacher_left -= 1

            tau_left = t0 + (m - step - 1) * dt  # time-to-maturity after this step

            # set up tri-diagonals
            sub = [0.0] * (N + 1)
            main = [0.0] * (N + 1)
            sup = [0.0] * (N + 1)
            rhs = [0.0] * (N + 1)

            # boundary values at "after-step" time
            Smin, Smax = self.S_nodes[0], self.S_nodes[-1]
            if self.opt_type == "call":
                rhs[0] = 0.0
                rhs[N] = Smax * math.exp(-self.q * tau_left) - self.K * math.exp(-self.r * tau_left)
            else:
                rhs[0] = self.K * math.exp(-self.r * tau_left)
                rhs[N] = 0.0
            main[0] = 1.0
            main[N] = 1.0

            for i in range(1, N):
                if i_adj is not None and i == i_adj and self.barrier_type in ("down-and-out", "up-and-out", "double-out"):
                    # --- FIS-style non-symmetric stencil at node adjacent to KO barrier ---
                    # Upwind drift AWAY from the barrier; keep diffusion centered.
                    if side_adj == "down":
                        # barrier below ⇒ use forward (right) upwind: reduce left-drift coupling at sup
                        aI = use_theta * dt * (Ai + 0.0)
                        bI = 1.0 - use_theta * dt * (2.0 * Ai + Ci)
                        cI = use_theta * dt * (Ai - (+Bi))
                        aE = (1.0 - use_theta) * dt * (Ai + 0.0)
                        bE = 1.0 + (1.0 - use_theta) * dt * (2.0 * Ai + Ci)
                        cE = (1.0 - use_theta) * dt * (Ai - (+Bi))
                    else:
                        # side_adj == "up": barrier above ⇒ use backward (left) upwind
                        aI = use_theta * dt * (Ai - (-Bi))
                        bI = 1.0 - use_theta * dt * (2.0 * Ai + Ci)
                        cI = use_theta * dt * (Ai + 0.0)
                        aE = (1.0 - use_theta) * dt * (Ai - (-Bi))
                        bE = 1.0 + (1.0 - use_theta) * dt * (2.0 * Ai + Ci)
                        cE = (1.0 - use_theta) * dt * (Ai + 0.0)
                else:
                    # --- standard symmetric operator elsewhere ---
                    aI = use_theta * dt * (Ai - Bi)
                    bI = 1.0 - use_theta * dt * (2.0 * Ai + Ci)
                    cI = use_theta * dt * (Ai + Bi)
                    aE = (1.0 - use_theta) * dt * (Ai - Bi)
                    bE = 1.0 + (1.0 - use_theta) * dt * (2.0 * Ai + Ci)
                    cE = (1.0 - use_theta) * dt * (Ai + Bi)

                sub[i] = -aI
                main[i] = bI
                sup[i] = -cI
                rhs[i] = aE * V[i - 1] + bE * V[i] + cE * V[i + 1]

            # implicit solve
            V = self._solve_tridiagonal(sub, main, sup, rhs)

            if abs(t0) < 1e-14 and step == m - 1:
                last_dt_at_zero = dt

        return V, (last_dt_at_zero if last_dt_at_zero is not None else dt)

    # ---------------- time allocation & full run ----------------

    def _time_subgrid_counts(self) -> List[int]:
        """Distribute M across monitor intervals proportionally to interval length; enforce sum == M."""
        lengths = [self.monitor_times[i + 1] - self.monitor_times[i] for i in range(len(self.monitor_times) - 1)]
        total = sum(lengths) or 1.0
        raw = [max(1, int(round(self.M * (L / total)))) for L in lengths]

        diff = self.M - sum(raw)
        k = 0
        while diff != 0 and len(raw) > 0:
            j = k % len(raw)
            if diff > 0:
                raw[j] += 1
                diff -= 1
            else:
                if raw[j] > 1:
                    raw[j] -= 1
                    diff += 1
            k += 1
        return raw

    def _run_backward(self, apply_barrier: bool) -> Tuple[List[float], float]:
        """Full backward induction with discrete KO projection and KI via parity."""
        N = self.N
        V = self._terminal_payoff()

        # status at t=0
        if apply_barrier:
            if self.barrier_type in ("down-and-out", "up-and-out", "double-out") and self.already_hit:
                instant = self.rebate_amount if self.rebate_at_hit else self.rebate_amount * math.exp(-self.r * self.T)
                return [instant] * (N + 1), 0.0
            if self.barrier_type in ("down-and-in", "up-and-in", "double-in") and self.already_in:
                apply_barrier = False  # treat as vanilla from start

        theta = 0.5
        subcounts = self._time_subgrid_counts()
        rannacher_left = self.rannacher
        dt_last_global = None

        for k in range(len(self.monitor_times) - 1, 0, -1):
            t0 = self.monitor_times[k - 1]
            t1 = self.monitor_times[k]
            m_steps = subcounts[k - 1]

            V, dt_last = self._cn_log_subinterval(V, t0, t1, theta, rannacher_left, m_steps)
            rannacher_left = max(0, rannacher_left - m_steps)

            # apply KO at monitor (not at t = 0)
            if apply_barrier and (abs(t0) > 1e-14):
                self._apply_KO_projection(V, tau_left=t0)

            if abs(t0) < 1e-14:
                dt_last_global = dt_last

        # KI parity: vanilla minus KO
        if apply_barrier and self.barrier_type in ("down-and-in", "up-and-in", "double-in"):
            V_van, _ = self._run_backward(apply_barrier=False)
            V = [V_van[i] - V[i] for i in range(len(V))]

        return V, (dt_last_global if dt_last_global is not None else self.dt)

    # ---------------- interpolation & Greeks ----------------

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

    def _delta_gamma_log(self, V: List[float], S_eval: float) -> Tuple[float, float]:
        """Greeks on log-grid: Δ = Ux/S, Γ = (Uxx - Ux)/S^2, with barrier-aware Δ near KO."""
        x = math.log(max(S_eval, 1e-300))
        xs = self.X_nodes
        i  = min(range(len(xs)), key=lambda k: abs(xs[k] - x))
        i  = max(1, min(self.N - 1, i))
        dx = self.dx

        # central derivatives
        Ux  = (V[i + 1] - V[i - 1]) / (2.0 * dx)
        Uxx = (V[i + 1] - 2.0 * V[i] + V[i - 1]) / (dx * dx)

        # barrier-aware Δ (FIS): one-sided in first cell off barrier, blended in second
        def ko_info():
            if self.barrier_type in ("down-and-out", "double-out") and self.lower_barrier is not None:
                xb = math.log(self.lower_barrier)
                side = "down"
            elif self.barrier_type in ("up-and-out", "double-out") and self.upper_barrier is not None:
                xb = math.log(self.upper_barrier)
                side = "up"
            else:
                return None, None, None
            j = min(range(len(xs)), key=lambda k: abs(xs[k] - xb))
            return side, xb, j

        side, xb, j = ko_info()
        if side is not None:
            kdist = abs(i - j)
            if kdist == 1:
                # first cell off barrier: one-sided away from barrier
                if side == "down":      # barrier below -> forward difference
                    Ux = (V[j + 2] - V[j + 1]) / dx
                else:                   # barrier above -> backward difference
                    Ux = (V[j] - V[j - 1]) / dx
            elif kdist == 2:
                # second cell: blend one-sided and central
                q = 0.5
                Uc = (V[i + 1] - V[i - 1]) / (2.0 * dx)
                if side == "down":
                    Uo = (V[j + 2] - V[j + 1]) / dx
                else:
                    Uo = (V[j] - V[j - 1]) / dx
                Ux = q * Uo + (1.0 - q) * Uc

        S = S_eval
        delta = (1.0 / S) * Ux
        gamma = (Uxx - Ux) / (S * S)
        # clamp Γ to a large but finite band to avoid reporting numerical spikes
        gamma = max(min(gamma, 1e6), -1e6)
        return float(delta), float(gamma)

    # ---------------- public API ----------------

    def price(self) -> float:
        V, _ = self._run_backward(apply_barrier=(self.barrier_type != "none"))
        return self._interp_linear(self.S0, self.S_nodes, V)

    def greeks(self, vega_bump: float = 1e-3) -> Dict[str, float]:
        V, _ = self._run_backward(apply_barrier=(self.barrier_type != "none"))
        P = self._interp_linear(self.S0, self.S_nodes, V)
        delta, gamma = self._delta_gamma_log(V, self.S0)

        # symmetric vega bump (rebuild grid since N depends on σ via λ coupling)
        sig0 = self.sigma
        self.sigma = sig0 + vega_bump
        self._build_log_grid_from_time()
        up = self.price()

        self.sigma = sig0 - vega_bump
        self._build_log_grid_from_time()
        dn = self.price()

        self.sigma = sig0
        self._build_log_grid_from_time()
        vega = (up - dn) / (2.0 * vega_bump)

        # theta from PDE at t=0 (consistent and cheap)
        theta = -((self.r - self.q) * self.S0 * delta
                  + 0.5 * self.sigma * self.sigma * self.S0 * self.S0 * gamma
                  - self.r * P)
        return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega), "theta": float(theta)}

    def print_details(self) -> None:
        p = self.price()
        g = self.greeks()
        print("==== Discrete Barrier Option (CN, log-space) — Discrete monitors, optimal Δt–Δx ====")
        print(f"T (years)         : {self.T:.9f}   [{self.day_count}]")
        print(f"σ / r / q         : {self.sigma:.9f} / {self.r:.9f} / {self.q:.9f}")
        print(f"Barrier type      : {self.barrier_type}  (lo={self.lower_barrier}, up={self.upper_barrier})  "
              f"rebate={self.rebate_amount} @hit={self.rebate_at_hit}")
        print(f"Status (hit/in)   : {self.already_hit} / {self.already_in}")
        print(f"Grid (M,N,dx)     : {self.M}, {self.N}, {self.dx:.6g}")
        print(f"Spot/Strike       : {self.S0:.6f} / {self.K:.6f}")
        print(f"Price             : {p:.9f}")
        print(f"Greeks            : Δ={g['delta']:.9f}, Γ={g['gamma']:.9f}, ν={g['vega']:.9f}, Θ={g['theta']:.9f}")

    # --------- optional: quick convergence table (M drives N here) ---------

    def validate_convergence(self, M_list: List[int], lambda_target: Optional[float] = None) -> List[Dict[str, float]]:
        """
        Sweep over M (time steps). For each M, N is derived via the Δt–Δx relation.
        Returns a list of rows with price and Greeks.
        """
        out: List[Dict[str, float]] = []
        keep_lambda = self.lambda_target
        for M in M_list:
            self.M = max(1, int(M))
            self.dt = self.T / self.M
            if lambda_target is not None:
                self.lambda_target = float(lambda_target)
            self._build_log_grid_from_time()

            price = self.price()
            greeks = self.greeks()

            out.append({
                "M": self.M, "N": self.N,
                "price": price,
                "delta": greeks["delta"],
                "gamma": greeks["gamma"],
                "vega":  greeks["vega"],
                "theta": greeks["theta"],
            })
        # restore λ
        self.lambda_target = keep_lambda

        # pretty print
        print("\n=== Convergence (M drives N via Δt–Δx) ===")
        print("    M        N        Price          Delta           Gamma            Vega            Theta")
        for r in out:
            print(f"{r['M']:6d} {r['N']:8d}  {r['price']:12.8f}  {r['delta']:13.8f}  "
                  f"{r['gamma']:13.8f}  {r['vega']:13.8f}  {r['theta']:13.8f}")
        return out