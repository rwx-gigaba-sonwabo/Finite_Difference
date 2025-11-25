import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class DiscreteBarrierCrankNicolsonLog:
    """
    Crank–Nicolson log-space pricer for single-barrier (discrete) options.

    - Underlying S follows risk-neutral dynamics with carry b_carry and
      discount rate r_disc.
    - PDE in tau = T - t:
        U_tau = 0.5 * sigma^2 * U_xx + (b - 0.5*sigma^2) * U_x - r * U
    - Discrete KO monitoring is imposed at given calendar monitoring times.
    - KI options are handled via vanilla-KO parity.
    """

    S0: float
    K: float
    T: float
    sigma: float
    r_disc: float
    b_carry: float              # e.g. r_disc - q_eff
    option_type: str            # "call" or "put"
    barrier_type: str           # "none", "down-and-out", "up-and-out",
                                # "down-and-in", "up-and-in"
    lower_barrier: Optional[float] = None
    upper_barrier: Optional[float] = None
    rebate: float = 0.0
    monitor_times: Optional[List[float]] = None  # in calendar time t

    # Optional grid overrides
    N_space: Optional[int] = None
    N_time: Optional[int] = None

    # Internal fields
    _S_min: float = field(init=False, default=0.0)
    _S_max: float = field(init=False, default=0.0)
    s_nodes: List[float] = field(init=False, default_factory=list)

    # ---------- grid configuration ----------

    def configure_grid(self) -> None:
        """
        Choose N_space, N_time and S-domain [S_min, S_max] in a principled way.
        """
        if self.T <= 0.0:
            raise ValueError("T must be positive")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if self.S0 <= 0.0:
            raise ValueError("S0 must be positive")

        # Domain in S: include spot, strike, barriers, then pad by factors
        candidates = [self.S0, self.K]
        if self.lower_barrier is not None and self.lower_barrier > 0:
            candidates.append(self.lower_barrier)
        if self.upper_barrier is not None and self.upper_barrier > 0:
            candidates.append(self.upper_barrier)

        s_low = min(candidates)
        s_high = max(candidates)

        L_low = 4.0
        L_high = 4.0
        S_min = max(1e-8, s_low / L_low)
        S_max = s_high * L_high
        if S_min >= S_max:
            S_min = self.S0 / 5.0
            S_max = self.S0 * 5.0

        self._S_min = S_min
        self._S_max = S_max

        # Space resolution: target points per 1σ√T in log-space
        x_min, x_max = math.log(S_min), math.log(S_max)
        x_range = x_max - x_min
        points_per_sigma = 12
        dx_target = self.sigma * math.sqrt(self.T) / points_per_sigma
        if dx_target <= 0.0:
            dx_target = x_range / 300.0

        if self.N_space is None:
            N_space = int(math.ceil(x_range / dx_target))
            self.N_space = max(N_space, 300)  # minimum for barrier stuff

        # Time resolution from diffusion number + monitoring structure
        if self.N_time is None:
            dx = x_range / self.N_space
            lambda_target = 0.4
            Ntime_opt = int(
                math.ceil(0.5 * self.sigma * self.sigma * self.T /
                          (lambda_target * dx * dx))
            )
            valid_mon = [
                t for t in (self.monitor_times or [])
                if 0.0 < t < self.T
            ]
            M_intervals = len(valid_mon) + 1
            self.N_time = max(Ntime_opt, self.N_space, 10 * M_intervals)

    def _build_log_grid(self) -> float:
        if self._S_min <= 0.0 or self._S_max <= 0.0 or self.N_space is None:
            self.configure_grid()

        x_min, x_max = math.log(self._S_min), math.log(self._S_max)
        N = self.N_space
        dx = (x_max - x_min) / N
        x_nodes = [x_min + i * dx for i in range(N + 1)]
        self.s_nodes = [math.exp(x) for x in x_nodes]
        return dx

    # ---------- payoff and boundaries ----------

    def _terminal_payoff(self) -> List[float]:
        payoff = []
        is_call = self.option_type.lower() == "call"
        for S in self.s_nodes:
            if is_call:
                payoff.append(max(S - self.K, 0.0))
            else:
                payoff.append(max(self.K - S, 0.0))
        return payoff

    def _boundary_values(self, tau: float) -> (float, float):
        """
        Dirichlet boundaries at time-to-maturity tau.
        """
        S_min = self.s_nodes[0]
        S_max = self.s_nodes[-1]
        r = self.r_disc
        b = self.b_carry
        is_call = self.option_type.lower() == "call"

        if is_call:
            V_min = 0.0
            V_max = S_max * math.exp((b - r) * tau) - self.K * math.exp(-r * tau)
        else:
            V_max = 0.0
            V_min = self.K * math.exp(-r * tau)

        return V_min, V_max

    # ---------- monitoring indices in tau ----------

    def _monitor_indices_tau(self, dt: float) -> set:
        """
        Map calendar monitoring times t_mon to tau indices on [0, T].
        """
        idx_set = set()
        if not self.monitor_times:
            return idx_set

        for t_mon in self.monitor_times:
            if t_mon <= 0.0 or t_mon >= self.T:
                continue
            tau_mon = self.T - t_mon
            k = int(round(tau_mon / dt))
            if 0 < k < self.N_time:
                idx_set.add(k)

        return idx_set

    # ---------- KO projection ----------

    def _apply_KO_projection(self, V: List[float]) -> None:
        bt = self.barrier_type.lower()
        if bt == "down-and-out" and self.lower_barrier is not None:
            B = self.lower_barrier
            for i, S in enumerate(self.s_nodes):
                if S <= B:
                    V[i] = self.rebate
        elif bt == "up-and-out" and self.upper_barrier is not None:
            B = self.upper_barrier
            for i, S in enumerate(self.s_nodes):
                if S >= B:
                    V[i] = self.rebate

    # ---------- CN solver core ----------

    def _solve_grid(self, apply_KO: bool) -> List[float]:
        """
        Solve the PDE on the full S-grid and return V(S, tau=T) at valuation.
        """
        self.configure_grid()
        dx = self._build_log_grid()
        N = self.N_space
        dt = self.T / self.N_time

        sig2 = self.sigma * self.sigma
        mu_x = self.b_carry - 0.5 * sig2
        alpha = 0.5 * sig2 / (dx * dx)
        beta_adv = mu_x / (2.0 * dx)

        a = alpha - beta_adv
        c = alpha + beta_adv
        bcoef = -2.0 * alpha - self.r_disc

        # CN matrices: (I - dt/2 L) V^{m+1} = (I + dt/2 L) V^m
        A_L = -0.5 * dt * a
        A_C = 1.0 - 0.5 * dt * bcoef
        A_U = -0.5 * dt * c

        B_L = 0.5 * dt * a
        B_C = 1.0 + 0.5 * dt * bcoef
        B_U = 0.5 * dt * c

        def solve_tridiag(rhs: List[float]) -> List[float]:
            n = len(rhs)
            c_prime = [0.0] * n
            d_prime = [0.0] * n

            c_prime[0] = A_U / A_C
            d_prime[0] = rhs[0] / A_C

            for i in range(1, n):
                denom = A_C - A_L * c_prime[i - 1]
                if i < n - 1:
                    c_prime[i] = A_U / denom
                d_prime[i] = (rhs[i] - A_L * d_prime[i - 1]) / denom

            x = [0.0] * n
            x[-1] = d_prime[-1]
            for i in range(n - 2, -1, -1):
                x[i] = d_prime[i] - c_prime[i] * x[i + 1]

            return x

        # Initial condition at tau = 0 (maturity)
        V = self._terminal_payoff()
        monitor_idx = self._monitor_indices_tau(dt)

        # March in tau: 0 -> T
        for m in range(self.N_time):
            tau_next = (m + 1) * dt
            V_min_next, V_max_next = self._boundary_values(tau_next)

            # Build RHS for interior nodes j=1..N-1
            rhs = [0.0] * (N - 1)
            for j in range(1, N):
                Vjm1, Vj, Vjp1 = V[j - 1], V[j], V[j + 1]
                rhs[j - 1] = (
                    B_L * Vjm1 +
                    B_C * Vj +
                    B_U * Vjp1
                )

            # Boundary contributions for next step (Dirichlet BC)
            rhs[0] -= A_L * V_min_next
            rhs[-1] -= A_U * V_max_next

            V_int = solve_tridiag(rhs)
            V[0] = V_min_next
            V[-1] = V_max_next
            V[1:-1] = V_int

            # Apply KO projection at monitoring times (in tau)
            if apply_KO and (m + 1) in monitor_idx:
                self._apply_KO_projection(V)

        return V

    # ---------- interpolation & spatial Greeks ----------

    def _interp_price_from_grid(self, V: List[float]) -> float:
        S0 = self.S0
        s = self.s_nodes
        if S0 <= s[0]:
            return V[0]
        if S0 >= s[-1]:
            return V[-1]

        lo, hi = 0, len(s) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if S0 < s[mid]:
                hi = mid
            else:
                lo = mid

        w = (S0 - s[lo]) / (s[hi] - s[lo])
        return (1.0 - w) * V[lo] + w * V[hi]

    def _delta_gamma_from_grid(self, V: List[float]) -> (float, float):
        """
        Compute delta and gamma at S0 from the S-grid using
        second-order finite differences on a non-uniform grid.
        """
        s = self.s_nodes
        S0 = self.S0

        # choose interior node closest to S0
        idx = min(range(1, len(s) - 1), key=lambda i: abs(s[i] - S0))

        S_im1, S_i, S_ip1 = s[idx - 1], s[idx], s[idx + 1]
        V_im1, V_i, V_ip1 = V[idx - 1], V[idx], V[idx + 1]

        h1 = S_i - S_im1
        h2 = S_ip1 - S_i

        # Delta: non-uniform central difference
        delta = (
            -h2 / (h1 * (h1 + h2)) * V_im1
            + (h2 - h1) / (h1 * h2) * V_i
            + h1 / (h2 * (h1 + h2)) * V_ip1
        )

        # Gamma: non-uniform central difference
        gamma = 2.0 * (
            V_im1 / (h1 * (h1 + h2))
            - V_i / (h1 * h2)
            + V_ip1 / (h2 * (h1 + h2))
        )

        return delta, gamma

    # ---------- public pricing ----------

    def price(self) -> float:
        bt = self.barrier_type.lower()

        # Vanilla
        if bt == "none":
            V = self._solve_grid(apply_KO=False)
            return self._interp_price_from_grid(V)

        # Pure KO
        if bt in ("down-and-out", "up-and-out"):
            V = self._solve_grid(apply_KO=True)
            return self._interp_price_from_grid(V)

        # KI via parity
        if bt not in ("down-and-in", "up-and-in"):
            raise ValueError(f"Unsupported barrier_type: {self.barrier_type}")

        original_bt = bt

        # Vanilla
        self.barrier_type = "none"
        V_van = self._solve_grid(apply_KO=False)
        P_van = self._interp_price_from_grid(V_van)

        # Matching KO
        self.barrier_type = "down-and-out" if bt == "down-and-in" else "up-and-out"
        V_ko = self._solve_grid(apply_KO=True)
        P_ko = self._interp_price_from_grid(V_ko)

        # Restore original barrier type
        self.barrier_type = original_bt

        return P_van - P_ko

    # ---------- public Greeks (price, delta, gamma, theta, vega) ----------

    def greeks(self, dv_sigma: float = 1e-3) -> Dict[str, float]:
        """
        Return price, delta, gamma, theta, vega for the current barrier_type.

        - For vanilla / KO: delta, gamma from grid; theta via PDE;
          vega via bump-and-revalue in sigma.
        - For KI: use parity (Greek_KI = Greek_vanilla - Greek_KO).
        """
        bt = self.barrier_type.lower()

        # Vanilla or KO: compute directly from PDE
        if bt in ("none", "down-and-out", "up-and-out"):
            apply_KO = (bt != "none")

            V_grid = self._solve_grid(apply_KO=apply_KO)
            price = self._interp_price_from_grid(V_grid)
            delta, gamma = self._delta_gamma_from_grid(V_grid)

            # Theta from PDE at t=0:
            # theta = -[ 0.5 * sigma^2 * S^2 * V_SS + b * S * V_S - r * V ]
            S0 = self.S0
            theta = -(
                0.5 * self.sigma * self.sigma * S0 * S0 * gamma
                + self.b_carry * S0 * delta
                - self.r_disc * price
            )

            # Vega via central bump in sigma
            sigma_orig = self.sigma

            self.sigma = sigma_orig + dv_sigma
            p_up = self.price()

            self.sigma = sigma_orig - dv_sigma
            p_dn = self.price()

            self.sigma = sigma_orig
            vega = (p_up - p_dn) / (2.0 * dv_sigma)

            return {
                "price": price,
                "delta": delta,
                "gamma": gamma,
                "theta": theta,
                "vega": vega,
            }

        # KI: Greeks via parity with vanilla and KO
        if bt not in ("down-and-in", "up-and-in"):
            raise ValueError(f"Unsupported barrier_type: {self.barrier_type}")

        original_bt = bt

        # Vanilla Greeks
        self.barrier_type = "none"
        g_van = self.greeks(dv_sigma=dv_sigma)

        # Matching KO Greeks
        self.barrier_type = "down-and-out" if bt == "down-and-in" else "up-and-out"
        g_ko = self.greeks(dv_sigma=dv_sigma)

        # Restore original barrier type
        self.barrier_type = original_bt

        return {
            key: g_van[key] - g_ko[key]
            for key in g_van.keys()
        }

    # --------------------------------------------------------------------
# 1) Single CN / BE subinterval with optional Rannacher
# --------------------------------------------------------------------
def _cn_subinterval_log(
    self,
    s_nodes: List[float],
    V: List[float],
    t_left: float,
    t_right: float,
    theta: float,
) -> Tuple[List[float], float]:
    """
    March one time subinterval [t_left, t_right] in log(S) using a
    theta-scheme (theta=0.5 => Crank–Nicolson, theta=1.0 => fully implicit BE).

    Returns:
        V_new : option values at t_left
        dt    : time step length (t_right - t_left) in YEARS
    """
    # Time step size in calendar years
    dt = t_right - t_left
    if dt <= 0.0:
        raise ValueError("Non-positive dt in _cn_subinterval_log")

    self.configure_grid()         # ensures self.num_space_nodes etc
    dx = self._build_log_grid()   # also stores self.s_nodes

    N  = self.num_space_nodes
    sig2 = self.sigma * self.sigma
    r   = self.discount_rate_nacc
    b   = self.carry_rate_nacc    # includes any effective dividend

    # Drift term in log-space
    mu_x = (b - 0.5 * sig2)

    # Operator coefficients in log-space:
    # V_t = L V  with
    # L V = 0.5 * sig2 * V_xx + mu_x * V_x - r * V
    alpha = 0.5 * sig2 / (dx * dx)
    beta  = 0.5 * mu_x / dx

    # Build tri-diagonal system matrices for theta-scheme:
    # (I + theta * dt * L) V^{n+1} = (I - (1 - theta) * dt * L) V^{n}
    a_co = alpha - beta      # coeff for V_{j-1}
    c_co = alpha + beta      # coeff for V_{j+1}
    b_co = -2.0 * alpha - r  # coeff for V_j

    # Left-hand and right-hand diagonals
    A_L = [0.0] * N
    A_C = [0.0] * N
    A_U = [0.0] * N

    B_L = [0.0] * N
    B_C = [0.0] * N
    B_U = [0.0] * N

    for j in range(1, N - 1):
        A_L[j] = -theta * dt * a_co
        A_C[j] = 1.0 - theta * dt * b_co
        A_U[j] = -theta * dt * c_co

        B_L[j] = (1.0 + (1.0 - theta) * dt * a_co)
        B_C[j] = (1.0 + (1.0 - theta) * dt * b_co)
        B_U[j] = (1.0 + (1.0 - theta) * dt * c_co)

    # Boundary rows will be replaced by Dirichlet boundary conditions below
    # so we don't bother to fill them fully here.

    # Apply theta-scheme for interior nodes j = 1..N-2
    rhs = [0.0] * N
    for j in range(1, N - 1):
        rhs[j] = (
            B_L[j] * V[j - 1]
            + B_C[j] * V[j]
            + B_U[j] * V[j + 1]
        )

    # Boundary values at time t_right (Dirichlet)
    V_min_next, V_max_next = self._boundary_values(t_right)

    # Impose boundary conditions in RHS
    rhs[1]   -= A_L[1] * V_min_next
    rhs[N-2] -= A_U[N-2] * V_max_next

    # Fill tri-diagonal diagonal vectors for solver
    diag  = [0.0] * N
    lower = [0.0] * N
    upper = [0.0] * N

    for j in range(1, N - 1):
        lower[j] = A_L[j]
        diag[j]  = A_C[j]
        upper[j] = A_U[j]

    # Boundary rows: fix value = boundary
    diag[0] = diag[N - 1] = 1.0
    rhs[0]  = V_min_next
    rhs[N-1] = V_max_next

    # Solve tri-diagonal system (Thomas algorithm)
    V_new = self._solve_tridiagonal_system(lower, diag, upper, rhs)

    return V_new, dt

# --------------------------------------------------------------------
# 2) Backward time-march with monitoring + Rannacher
# --------------------------------------------------------------------
def _run_backward(
    self,
    s_nodes: List[float],
    apply_barrier: bool,
) -> Tuple[List[float], float, List[float]]:
    """
    March option values backward in time from T to 0 in log(S).

    - Time is partitioned into intervals [t_{k-1}, t_k] that match the
      monitoring dates in self.monitor_times (in YEARS from carry start).
    - Within each interval we use a theta-scheme:
        * For the first `self.rannacher_steps` *global* steps we use
          BE (theta=1.0)  -> Rannacher smoothing.
        * Thereafter we use CN (theta=0.5).
    - At the end of any interval whose right end is a monitoring date,
      we apply KO projection if `apply_barrier=True`.

    Returns
    -------
    V_out            : option values at t=0
    dt_last_global   : last time step size used (for diagnostics)
    monitors_applied : list of monitoring times where KO was projected
    """
    # Start from terminal payoff at t = T
    V = self._terminal_payoff()
    monitors_applied: List[float] = []

    # Basic checks
    if not self.monitor_times:
        raise ValueError("monitor_times must contain at least the expiry time.")

    # Determine number of CN/BE steps per interval
    subcounts = self._time_subgrid_counts()   # len == len(monitor_times)
    if len(subcounts) != len(self.monitor_times):
        raise ValueError("subcounts and monitor_times length mismatch.")

    # Rannacher: number of *global* steps to do with BE
    rannacher_left = getattr(self, "rannacher_steps", 2)
    dt_last_global: Optional[float] = None

    # March intervals backwards: [t_{k-1}, t_k], k = n..1
    n_int = len(self.monitor_times)
    for k in range(n_int - 1, -1, -1):
        t_right = self.monitor_times[k]
        t_left  = self.monitor_times[k - 1] if k > 0 else 0.0

        # Subdivide this interval into subcounts[k] substeps
        m_steps = subcounts[k]
        if m_steps <= 0:
            continue

        dt_local = (t_right - t_left) / float(m_steps)
        # march subintervals: t_right -> t_left
        t_hi = t_right
        for m in range(m_steps):
            t_lo = t_hi - dt_local

            # Rannacher: first rannacher_left global steps use BE (theta=1.0)
            if rannacher_left > 0:
                theta = 1.0
                rannacher_left -= 1
            else:
                theta = 0.5  # standard Crank–Nicolson

            V, dt_used = self._cn_subinterval_log(
                s_nodes=s_nodes,
                V=V,
                t_left=t_lo,
                t_right=t_hi,
                theta=theta,
            )
            dt_last_global = dt_used
            t_hi = t_lo

        # After completing this interval, apply KO at t = t_left if it
        # is a monitoring date and KO is requested.
        if apply_barrier:
            # By construction t_left equals the monitoring date
            # (monitor_times[k-1] or 0.0 for k=0)
            tau_here = t_left
            V = self._apply_KO_projection_log(V, s_nodes, tau_here)
            monitors_applied.append(tau_here)

    # When done, V is at t = 0
    return V, (dt_last_global if dt_last_global is not None else dt_local), monitors_applied

# --------------------------------------------------------------------
# 3) Black-76 Greeks via finite differences (no barrier)
# --------------------------------------------------------------------
def _vanilla_black76_greeks_fd(
    self,
    dS: float = 0.0001,
    dSigma: float = 0.0001,
    dT: float = 0.0001,
) -> Dict[str, float]:
    """
    Finite-difference Greeks for the vanilla Black-76 price.

    All bumps are ABSOLUTE bumps:
        dS      : absolute spot bump
        dSigma  : absolute vol bump (0.0001 ~ 1bp of vol)
        dT      : absolute time bump in YEARS

    Returns a dict with 'delta', 'gamma', 'vega', 'theta'.
    """
    S0     = self.spot
    sigma0 = self.sigma
    T0     = self.time_to_expiry

    # Base price
    p0 = self._vanilla_black76_price(S=S0, sigma=sigma0, T=T0)

    # --- Delta & Gamma (bump in spot) ---
    S_up  = S0 + dS
    S_dn  = max(1e-12, S0 - dS)
    p_up  = self._vanilla_black76_price(S=S_up, sigma=sigma0, T=T0)
    p_dn  = self._vanilla_black76_price(S=S_dn, sigma=sigma0, T=T0)

    delta = (p_up - p_dn) / (2.0 * dS)
    gamma = (p_up - 2.0 * p0 + p_dn) / (dS * dS)

    # --- Vega (central bump in sigma) ---
    sig_up = sigma0 + dSigma
    sig_dn = max(1e-12, sigma0 - dSigma)
    p_vu   = self._vanilla_black76_price(S=S0, sigma=sig_up, T=T0)
    p_vd   = self._vanilla_black76_price(S=S0, sigma=sig_dn, T=T0)
    vega   = (p_vu - p_vd) / (2.0 * dSigma)

    # --- Theta (central bump in time-to-expiry) ---
    T_up = max(0.0, T0 - dT)      # d/dt with calendar time, so minus on T
    T_dn = T0 + dT
    p_tu = self._vanilla_black76_price(S=S0, sigma=sigma0, T=T_up)
    p_td = self._vanilla_black76_price(S=S0, sigma=sigma0, T=T_dn)
    theta = (p_tu - p_td) / (2.0 * dT)

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
    }

# --------------------------------------------------------------------
# 4) PDE price + Greeks with KO (CN + Rannacher)
# --------------------------------------------------------------------
def _pde_price_and_greeks(
    self,
    apply_KO: bool,
    dSigma: float = 0.0001,
) -> Dict[str, float]:
    """
    Price + Greeks from the CN PDE engine for the current barrier type.

    If apply_KO is True, KO projection is applied at each monitoring date.
    Greeks:
        - Delta, Gamma from the grid around S0 (non-uniform central diff).
        - Vega from CENTRAL bump in sigma (two PDE runs).
        - Theta from PDE time-derivative formula.
    """
    # Base grid solution
    V_grid, _, _ = self._run_backward(self.s_nodes, apply_barrier=apply_KO)
    price = self._interp_price(V_grid)

    # Delta & Gamma from grid
    delta, gamma = self._delta_gamma_from_grid(V_grid)

    # Theta from PDE identity: V_t = L V
    S0    = self.spot
    sigma = self.sigma
    r     = self.discount_rate_nacc
    b     = self.carry_rate_nacc

    theta = 0.5 * sigma * sigma * S0 * S0 * gamma \
            + (b - 0.0 * sigma * sigma) * S0 * delta \
            - r * price

    # Vega via central bump in sigma (CN engine)
    sig_up = sigma + dSigma
    sig_dn = max(1e-12, sigma - dSigma)

    # up
    sigma_saved = self.sigma
    self.sigma  = sig_up
    V_up, _, _  = self._run_backward(self.s_nodes, apply_barrier=apply_KO)
    p_up        = self._interp_price(V_up)

    # down
    self.sigma  = sig_dn
    V_dn, _, _  = self._run_backward(self.s_nodes, apply_barrier=apply_KO)
    p_dn        = self._interp_price(V_dn)

    # restore
    self.sigma  = sigma_saved

    vega = (p_up - p_dn) / (2.0 * dSigma)

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
    }

# --------------------------------------------------------------------
# 5) Public price() using parity for KI
# --------------------------------------------------------------------
def price_log2(self) -> float:
    """
    Barrier price in log-space:

    - If barrier_type is 'none': vanilla Black-76.
    - If KO ('down-and-out', 'up-and-out'): CN PDE with KO projection.
    - If KI ('down-and-in', 'up-and-in'): vanilla - KO price (parity).
    """
    bt = self.barrier_type.lower()

    # Vanilla (no barrier)
    if bt == "none":
        return self._vanilla_black76_price()

    # Knock-out
    if bt in ("down-and-out", "up-and-out"):
        out = self._pde_price_and_greeks(apply_KO=True)
        return out["price"]

    # Knock-in via parity
    if bt in ("down-and-in", "up-and-in"):
        # Save original barrier type
        original_bt = bt

        # Vanilla leg
        p_van = self._vanilla_black76_price()

        # Matching KO type
        self.barrier_type = "down-and-out" if bt == "down-and-in" else "up-and-out"
        p_ko = self._pde_price_and_greeks(apply_KO=True)["price"]

        # Restore
        self.barrier_type = original_bt

        return p_van - p_ko

    raise ValueError(f"Unsupported barrier type: {bt}")

# --------------------------------------------------------------------
# 6) Public Greeks with parity for KI
# --------------------------------------------------------------------
def greeks_log2(self, dSigma: float = 0.0001) -> Dict[str, float]:
    """
    Greeks consistent with pricing:

    - Vanilla: Black-76 FD Greeks.
    - KO: CN PDE Greeks with KO projection.
    - KI: vanilla Greeks - KO Greeks (parity).
    """
    bt = self.barrier_type.lower()

    # Vanilla
    if bt == "none":
        return self._vanilla_black76_greeks_fd()

    # KO
    if bt in ("down-and-out", "up-and-out"):
        return self._pde_price_and_greeks(apply_KO=True, dSigma=dSigma)

    # KI = vanilla - KO
    if bt in ("down-and-in", "up-and-in"):
        original_bt = bt

        g_van = self._vanilla_black76_greeks_fd()

        self.barrier_type = "down-and-out" if bt == "down-and-in" else "up-and-out"
        g_ko  = self._pde_price_and_greeks(apply_KO=True, dSigma=dSigma)

        self.barrier_type = original_bt

        return {
            k: g_van[k] - g_ko[k] for k in g_van.keys()
        }

    raise ValueError(f"Unsupported barrier type: {bt}")

# --------------------------------------------------------------------
# KO PROJECTION IN LOG-SPACE
# --------------------------------------------------------------------
def _apply_KO_projection_log(
    self,
    V: List[float],
    s_nodes: List[float],
    tau: float
) -> List[float]:
    """
    Apply knock-out (KO) projection to the grid values V at a monitoring time.

    Parameters
    ----------
    V : List[float]
        Option values on the grid BEFORE KO projection.
    s_nodes : List[float]
        Log-space grid mapped to spot prices.
    tau : float
        Calendar time (years from carry start) at which monitoring occurs.

    Behavior
    --------
    If the barrier is of type:
        - down-and-out:   KO if S <= barrier_level
        - up-and-out:     KO if S >= barrier_level
        - double-out:     KO if S <= lower or S >= upper
    Otherwise (vanilla, knock-in): return V unchanged.

    Notes
    -----
    - Projection occurs ONLY at monitoring dates.
    - KO sets the option value to rebate discounted from carry end,
      if rebate_at_hit=True OR rebate amount is defined.
    - If no rebate, KO → value = 0.
    """
    bt = self.barrier_type.lower()

    # Vanilla or knock-in = no KO projection
    if bt in ("none", "down-and-in", "up-and-in"):
        return V

    N = len(s_nodes)
    barrier = self.barrier_level
    rebate  = self.rebate_amount
    have_rebate = (rebate is not None and rebate > 0.0)

    # Discount rebate from carry end back to time tau
    if have_rebate:
        df = math.exp(-self.discount_rate_nacc * (self.time_to_carry - tau))
        rebate_value = rebate * df
    else:
        rebate_value = 0.0

    # Prepare new array
    V_new = V[:]

    # -----------------------------
    # DOWN-AND-OUT
    # -----------------------------
    if bt == "down-and-out":
        for j, S in enumerate(s_nodes):
            if S <= barrier:
                V_new[j] = rebate_value

    # -----------------------------
    # UP-AND-OUT
    # -----------------------------
    elif bt == "up-and-out":
        for j, S in enumerate(s_nodes):
            if S >= barrier:
                V_new[j] = rebate_value

    # -----------------------------
    # DOUBLE-OUT
    # -----------------------------
    elif bt == "double-out":
        lower = self.lower_barrier
        upper = self.upper_barrier
        for j, S in enumerate(s_nodes):
            if S <= lower or S >= upper:
                V_new[j] = rebate_value

    # Any other case → return unchanged
    else:
        return V

    # Track which monitoring times applied KO (for validation reporting)
    if not hasattr(self, "_ko_monitor_times"):
        self._ko_monitor_times = []
    self._ko_monitor_times.append(tau)

    return V_new

class DiscreteBarrierFDMPricer:
    ...

    def _solve_grid(self, apply_KO: bool) -> List[float]:
        """
        Solve dV/dt = L V with L from Black–Scholes in log S using a
        θ–scheme in time (Rannacher + Crank–Nicolson) and uniform grid in log S.

        - First 2 global steps: θ = 1.0 (backward Euler, Rannacher smoothing)
        - Remaining steps:      θ = 0.5 (Crank–Nicolson)

        Barrier projection is done at the scheduled monitoring times.
        """

        # --- grid / basic coefficients --------------------------------------
        self.configure_grid()
        dx = self._build_log_grid()
        N = self.num_space_nodes
        dt = self.time_to_expiry / self.num_time_steps

        sig2 = self.sigma * self.sigma
        # drift in log S under risk-neutral measure
        mu_x = (self.carry_rate_nacc - self.discount_rate_nacc) - 0.5 * sig2

        # spatial operator coefficients in log-space
        alpha = 0.5 * sig2 / (dx * dx)        # diffusion term
        beta_adv = mu_x / (2.0 * dx)         # advection term

        a = alpha + beta_adv                 # coeff for V_{j-1}
        c = alpha - beta_adv                 # coeff for V_{j+1}
        b0 = -2.0 * alpha + self.discount_rate_nacc  # coeff for V_j

        # which time steps are monitoring dates?
        monitor_idx = self._monitor_indices_tau(dt)   # e.g. set of step numbers

        # --- initial condition at maturity ----------------------------------
        V = self._terminal_payoff()
        last_dt_at_zero = None

        # --- march backwards in time ----------------------------------------
        for n in range(self.num_time_steps):
            # time level n -> n+1 in τ (or T -> 0 in calendar time, depending on your convention)
            t_n = n * dt
            t_np1 = (n + 1) * dt

            # 1) choose θ for this global time step
            #    (two Rannacher steps with BE, then CN)
            theta = 1.0 if n < 2 else 0.5

            # 2) build A,V^{n+1} = B,V^{n} system for this θ
            A_L = -theta * dt * a
            A_C = 1.0 - theta * dt * b0
            A_U = -theta * dt * c

            B_L = (1.0 - theta) * dt * a
            B_C = 1.0 + (1.0 - theta) * dt * b0
            B_U = (1.0 - theta) * dt * c

            # 3) boundary values for next time level (Dirichlet)
            V_min_next, V_max_next = self._boundary_values(t_np1)

            # 4) build RHS = B V^n  (interior nodes j = 1..N-2)
            rhs = [0.0] * N
            for j in range(1, N - 1):
                rhs[j] = (
                    B_L * V[j - 1] +
                    B_C * V[j] +
                    B_U * V[j + 1]
                )

            # add boundary contributions to RHS
            rhs[1]  += A_L * V_min_next   # left boundary affects j=1
            rhs[N-2] += A_U * V_max_next  # right boundary affects j=N-2

            # 5) solve tridiagonal system for interior points with Thomas
            #    we only pass interior coefficients to the solver
            V_int = self._solve_tridiagonal_system(
                n_unknowns=N - 2,
                A_L=A_L, A_C=A_C, A_U=A_U,
                rhs=[rhs[j] for j in range(1, N - 1)]
            )

            # 6) reconstruct full grid including boundaries
            V_new = [0.0] * N
            V_new[0] = V_min_next
            V_new[-1] = V_max_next
            V_new[1:-1] = V_int
            V = V_new

            # 7) apply KO projection if this step is a monitoring date
            if apply_KO and (n + 1) in monitor_idx:
                tau_here = t_np1
                V = self._apply_KO_projection_log(V, self.s_nodes, tau_here)

            # keep the last dt close to t=0, in case a caller wants it
            if abs(t_np1 - self.time_to_expiry) < 1e-14:
                last_dt_at_zero = dt

        return V

def _solve_grid(self, apply_KO: bool) -> List[float]:
    """
    Solve dV/dt = L V with L from Black–Scholes in log S using
    Crank–Nicolson in log-space. Marches from tau = T (terminal) to tau = 0.

    Uses:
      - self._solve_tridiagonal_system(...) for the linear system at each step
      - Dirichlet boundaries from self._boundary_values(tau)
      - optional KO projection at monitoring times in tau-space
      - Rannacher smoothing: first self.rannacher_steps global steps use BE (theta = 1.0),
        remaining steps use CN (theta = 0.5).

    Returns:
        V at tau = 0 on the log-S grid (len == self.num_space_nodes).
    """

    # --- 1) Grid + PDE coefficients -------------------------------------------------
    self.configure_grid()
    dx = self._build_log_grid()
    N = self.num_space_nodes              # total nodes, indices 0 .. N-1
    dt = self.time_to_expiry / self.num_time_steps

    sig2 = self.sigma * self.sigma

    # drift in log-space (you already have this in your code; keep your exact formula)
    mu_x = self.carry_rate_naec - self.discount_rate_naec - 0.5 * sig2

    # spatial operator coefficients in log S
    alpha = 0.5 * sig2 / (dx * dx)
    beta_adv = mu_x / (2.0 * dx)

    # L V_j = a V_{j-1} + b V_j + c V_{j+1}
    a = alpha - beta_adv               # coefficient for V_{j-1}
    c = alpha + beta_adv               # coefficient for V_{j+1}
    b = -2.0 * alpha - self.discount_rate_naec  # coefficient for V_j  (−r term)

    # monitoring indices (in tau = k * dt space)
    monitor_idx = self._monitor_indices_tau(dt)

    # Rannacher: how many global steps to do with theta=1.0
    rannacher_left = getattr(self, "rannacher_steps", 2)

    # --- 2) Terminal condition -------------------------------------------------------
    V = self._terminal_payoff()        # len N

    # --- 3) March backwards in time --------------------------------------------------
    # time-level index n: tau_n = (n) * dt, tau_{n+1} = (n+1)*dt
    for n in range(self.num_time_steps - 1, -1, -1):
        tau_next = (n + 1) * dt  # this is where V is currently known
        tau_curr = n * dt        # we are solving for this

        # choose theta (Rannacher)
        if rannacher_left > 0:
            theta = 1.0          # backward Euler for first few global steps
            rannacher_left -= 1
        else:
            theta = 0.5          # Crank–Nicolson afterwards

        # boundaries at tau_next (Dirichlet)
        V_min_next, V_max_next = self._boundary_values(tau_next)

        # --- Build tri-diagonal system A V_int^{curr} = rhs -------------------------
        n_int = N - 2                             # number of interior nodes
        lower_diag = [0.0] * n_int               # sub-diagonal (for rows 1..n_int-1)
        main_diag  = [0.0] * n_int               # main diagonal
        upper_diag = [0.0] * n_int               # super-diagonal (for rows 0..n_int-2)
        rhs        = [0.0] * n_int

        # coefficients for A and B matrices (theta-scheme)
        # A = I - theta * dt * L
        # B = I + (1-theta) * dt * L
        A_l = -theta * dt * a
        A_c = 1.0 - theta * dt * b
        A_u = -theta * dt * c

        B_l = (1.0 - theta) * dt * a
        B_c = 1.0 + (1.0 - theta) * dt * b
        B_u = (1.0 - theta) * dt * c

        # interior nodes j = 1 .. N-2  ->  row index i = j-1
        for j in range(1, N - 1):
            i = j - 1

            # tri-diagonal coefficients:
            main_diag[i] = A_c
            lower_diag[i] = A_l if i > 0 else 0.0
            upper_diag[i] = A_u if i < n_int - 1 else 0.0

            # RHS from explicit part: B * V^{next}
            rhs[i] = (
                B_l * V[j - 1] +
                B_c * V[j] +
                B_u * V[j + 1]
            )

        # add boundary contributions to RHS (Dirichlet)
        # first interior node (j=1) is affected by left boundary
        rhs[0]  += A_l * V_min_next
        # last interior node (j=N-2) is affected by right boundary
        rhs[-1] += A_u * V_max_next

        # --- Solve tri-diagonal system for interior values --------------------------
        V_int = self._solve_tridiagonal_system(
            lower_diag=lower_diag,
            main_diag=main_diag,
            upper_diag=upper_diag,
            right_hand_side=rhs,
        )

        # reconstruct full solution at tau_curr
        V_new = [0.0] * N
        V_new[0] = V_min_next       # left boundary in tau_curr
        V_new[-1] = V_max_next      # right boundary in tau_curr
        for j in range(1, N - 1):
            V_new[j] = V_int[j - 1]

        V = V_new

        # --- apply discrete KO at monitoring times (if requested) -------------------
        if apply_KO and n in monitor_idx:
            V = self._apply_KO_projection_log(V, self.s_nodes, tau_curr)

    return V

    def _solve_grid(self, apply_KO: bool) -> list[float]:
        """
        Solve dV/dt = L V on log-S grid using Rannacher-smoothed
        Crank–Nicolson and Dirichlet boundaries.

        Returns the value vector V(t=0, S_j) on the existing grid self.s_nodes.
        """
        # 1) Grid & time set-up
        self.configure_grid()
        dx = self._build_log_grid()             # uniform log spacing
        N = self.num_space_nodes                # number of spatial nodes
        M = self.num_time_steps                 # number of time steps
        dt = self.time_to_expiry / float(M)     # constant Δt

        if N < 3:
            raise ValueError("Need at least 3 space nodes for CN scheme.")

        # 2) PDE coefficients in log space
        sig = self.sigma
        sig2 = sig * sig

        # Discount & dividend yields in *continuous* compounding
        r = self.discount_rate_nacc
        q = getattr(self, "div_yield_nacc", 0.0)

        alpha = 0.5 * sig2
        beta  = (r - q - 0.5 * sig2)

        a = alpha / (dx * dx) - beta / (2.0 * dx)
        b = -2.0 * alpha / (dx * dx) - r
        c = alpha / (dx * dx) + beta / (2.0 * dx)

        # 3) Terminal condition at maturity (τ = 0)
        V = self._terminal_payoff()             # length N

        # 4) Monitoring structure in τ-space
        #    monitor step indices k such that τ_k = k*dt is a monitoring time
        monitor_indices = set(self._monitor_indices_tau(dt))

        # 5) Rannacher: number of *global* BE steps
        rannacher_steps = 2

        # --- time-stepping from τ=0 -> τ=T (backwards in calendar time) ---
        for n_step in range(M):
            tau_next = (n_step + 1) * dt

            # 5a) choose θ for this step
            theta = 1.0 if n_step < rannacher_steps else 0.5

            # 5b) precompute θ-dependent matrix coefficients
            AL = -theta * dt * a
            AC = 1.0   - theta * dt * b
            AU = -theta * dt * c

            BL = (1.0 - theta) * dt * a
            BC = 1.0   + (1.0 - theta) * dt * b
            BU = (1.0 - theta) * dt * c

            # 5c) Boundary values at τ_next
            V_min_next, V_max_next = self._boundary_values(tau_next)

            # 5d) Build RHS for interior nodes j = 1..N-2
            n_int = N - 2
            rhs = [0.0] * n_int

            for j in range(1, N - 1):
                rhs[j - 1] = (
                    BL * V[j - 1] +
                    BC * V[j]     +
                    BU * V[j + 1]
                )

            # Add boundary contributions moved to RHS:
            # left boundary enters at j=1 (index 0 in rhs)
            rhs[0]      -= AL * V_min_next
            # right boundary enters at j=N-2 (index n_int-1 in rhs)
            rhs[-1]     -= AU * V_max_next

            # 5e) Build tri-diagonal system for interior nodes
            lower = [0.0] + [AL] * (n_int - 1)      # length n_int
            main  = [AC]  * n_int
            upper = [AU] * (n_int - 1) + [0.0]

            V_int = self._solve_tridiagonal_system(
                lower_diag=lower,
                main_diag=main,
                upper_diag=upper,
                right_hand_side=rhs,
            )

            # 5f) Assemble full solution at τ_next
            V_new = [0.0] * N
            V_new[0]    = V_min_next
            V_new[-1]   = V_max_next
            V_new[1:-1] = V_int

            # 5g) Apply knock-out at this monitoring time (in τ-space)
            if apply_KO and (n_step + 1) in monitor_indices:
                V_new = self._apply_KO_projection_log(V_new, self.s_nodes, tau_next)

            V = V_new

        # Finished marching to τ = T (calendar t = 0)
        return V
    
    def _build_CN_coeffs(self, dt: float, dx: float, theta: float):
        sig2 = self.sigma * self.sigma
        mu_x = (self.carry_rate_nacc - 0.5 * sig2)

        alpha = 0.5 * sig2 / (dx * dx)
        beta  = mu_x / (2.0 * dx)

        # continuous operator L V = a V_{j-1} + b V_j + c V_{j+1}
        a = alpha - beta
        b = -2.0 * alpha - self.discount_rate_nacc
        c = alpha + beta

        # generic θ-scheme: (I - θ dt L) V^{n+1} = (I + (1-θ) dt L) V^n
        A_L = -theta    * dt * a
        B_C = 1.0 - theta    * dt * b
        C_U = -theta    * dt * c

        A_L_rhs = (1.0 - theta) * dt * a
        B_C_rhs = 1.0 + (1.0 - theta) * dt * b
        C_U_rhs = (1.0 - theta) * dt * c

        return (A_L, B_C, C_U, A_L_rhs, B_C_rhs, C_U_rhs)

    def _solve_grid(self, apply_KO: bool) -> List[float]:
    self.configure_grid()
    dx = self._build_log_grid()
    dt = self.time_to_expiry / self.num_time_steps

    N = self.num_space_nodes
    V = self._terminal_payoff()   # size N

    # Precompute barrier monitor indices in τ if needed
    monitor_idx = self._monitor_indices_tau(dt)

    # global step counter (from t = T down to 0)
    for n in range(self.num_time_steps):
        tau_next = (n + 1) * dt

        # ---------- Rannacher: θ = 1 for first 2 global steps ----------
        theta = 1.0 if n < 2 else 0.5

        (A_L, B_C, C_U,
         A_L_rhs, B_C_rhs, C_U_rhs) = self._build_CN_coeffs(dt, dx, theta)

        # boundaries at tau_next (Dirichlet)
        V_min_next, V_max_next = self._boundary_values(tau_next)

        # build RHS for interior nodes j = 1..N-2
        rhs = [0.0] * (N - 2)
        for j in range(1, N-1):
            jm = j - 1
            jp = j + 1
            i  = j - 1        # index in rhs

            rhs[i] = (
                A_L_rhs * V[jm] +
                B_C_rhs * V[j]  +
                C_U_rhs * V[jp]
            )

        # add boundaries to RHS (only interior nodes see boundaries)
        rhs[0]     += -A_L * V_min_next
        rhs[-1]    += -C_U * V_max_next

        # tri-diagonal coefficients for interior
        lower = [A_L] * (N - 3)
        main  = [B_C] * (N - 2)
        upper = [C_U] * (N - 3)

        V_int = self._solve_tridiagonal_system(lower, main, upper, rhs)

        # reconstruct full grid including boundaries
        V_new = [0.0] * N
        V_new[0]  = V_min_next
        V_new[-1] = V_max_next
        V_new[1:-1] = V_int

        V = V_new

        # Apply KO projection if this is a monitoring date
        if apply_KO and n in monitor_idx:
            V = self._apply_KO_projection_log(V, self.s_nodes, tau_next)

    return V

def pde_price_and_greeks(
    self,
    apply_KO: bool,
    dv_sigma: float = 0.0001,
    N_time_override: int | None = None,
) -> dict[str, float]:
    """
    Solve PDE grid and compute price & Greeks for the current barrier type.

    Parameters
    ----------
    apply_KO : bool
        Whether to project out knock-out at monitoring times.
    dv_sigma : float
        Vol bump for Vega (absolute, e.g. 0.0001 = 1bp vol).
    N_time_override : int, optional
        If provided, temporarily override self.num_time_steps.

    Returns
    -------
    dict with keys 'price', 'delta', 'gamma', 'vega', 'theta'
    """
    # Temporarily override number of time steps
    old_N = self.num_time_steps
    if N_time_override is not None:
        self.num_time_steps = int(N_time_override)

    try:
        # base grid
        V_grid = self._solve_grid(apply_KO=apply_KO)
        price = self._interp_price_from_grid(V_grid)
        delta, gamma = self._delta_gamma_from_grid(V_grid)

        # crude Theta from Black-76 parity (consistent with your existing code)
        S0 = self.spot
        r  = self.discount_rate_nacc
        sigma = self.sigma
        T = self.time_to_expiry

        theta = r * price - 0.5 * sigma * sigma * S0 * S0 * gamma - r * S0 * delta

        # Vega via central bump in sigma using CN engine
        sigma_orig = self.sigma
        self.sigma = sigma_orig + dv_sigma
        V_up = self._solve_grid(apply_KO=apply_KO)
        price_up = self._interp_price_from_grid(V_up)
        self.sigma = sigma_orig - dv_sigma
        V_dn = self._solve_grid(apply_KO=apply_KO)
        price_dn = self._interp_price_from_grid(V_dn)
        self.sigma = sigma_orig

        vega = (price_up - price_dn) / (2.0 * dv_sigma)

        return {
            "price": price,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
        }
    finally:
        # restore time steps
        self.num_time_steps = old_N

def pde_price_and_greeks_richardson(
    self,
    apply_KO: bool,
    dv_sigma: float = 0.0001,
    N_time_base: int | None = None,
) -> dict[str, float]:
    """
    Richardson-extrapolated price & Greeks.

    Computes with N and N/2 time steps and combines
    V* = (4 V_N - V_{N/2}) / 3, assuming error ~ O(N^{-2}).

    This should be used for 'smooth enough' cases.
    For barrier / very near barrier, you may want to switch it off.
    """
    # Choose base N: default to current num_time_steps
    if N_time_base is None:
        N_time_base = self.num_time_steps

    N_time_base = int(N_time_base)
    N_half = max(1, N_time_base // 2)

    # Coarse (N/2) and fine (N) runs
    res_half = self.pde_price_and_greeks(
        apply_KO=apply_KO,
        dv_sigma=dv_sigma,
        N_time_override=N_half,
    )
    res_full = self.pde_price_and_greeks(
        apply_KO=apply_KO,
        dv_sigma=dv_sigma,
        N_time_override=N_time_base,
    )

    res_extrap = {}
    for key in res_full.keys():
        vN = res_full[key]
        vH = res_half[key]
        res_extrap[key] = (4.0 * vN - vH) / 3.0

    return res_extrap

def price_log2(self, use_richardson: bool = True) -> float:
    bt = self.barrier_type.lower()

    # Vanilla: Black-76 closed form
    if bt == "none":
        return self._vanilla_black76_price()

    # Knock-out: PDE with KO projection
    if bt in ("down-and-out", "up-and-out"):
        if not use_richardson:
            price_greeks = self.pde_price_and_greeks(apply_KO=True)
        else:
            price_greeks = self.pde_price_and_greeks_richardson(apply_KO=True)
        return price_greeks["price"]

    # Knock-in via parity: V_KI = V_vanilla - V_KO
    if bt in ("down-and-in", "up-and-in"):
        # vanilla (no barrier), no need for Richardson – Black-76
        p_van = self._vanilla_black76_price()

        # KO with PDE, possibly Richardson
        if not use_richardson:
            g_ko = self.pde_price_and_greeks(apply_KO=True)
        else:
            g_ko = self.pde_price_and_greeks_richardson(apply_KO=True)
        p_ko = g_ko["price"]

        return p_van - p_ko

    raise ValueError(f"Unsupported barrier type: {self.barrier_type}")

    def solve_grid(self, apply_KO: bool) -> list[float]:
        """
        Solve dV/dt = L V with L from Black–Scholes in S-space using
        Rannacher + Crank–Nicolson between monitoring dates.

        This keeps the external interface identical to the original
        `solve_grid`, but internally delegates to the theoretically
        consistent `_run_backward` + `_cn_subinterval` engine.

        Parameters
        ----------
        apply_KO : bool
            If True, knock-out projection is applied at monitoring dates.
            If False, the barrier is ignored and a vanilla grid is returned.

        Returns
        -------
        V : list[float]
            Option values on the spatial grid `self.s_nodes` at valuation
            time t = 0 (time-to-expiry τ = T). This grid is then used by
            the interpolation routines to obtain the price at S0.
        """
        # Ensure grid in S is configured exactly once
        self.configure_grid()     # sets up self.s_nodes, self.num_time_steps, etc.

        # Run the backward time-march between all monitoring dates.
        V, last_dt_at_zero, monitors_applied = self._run_backward(
            s_nodes=self.s_nodes,
            apply_barrier=apply_KO,
        )

        # Optionally store diagnostics if you want them later
        self._last_dt_at_zero = last_dt_at_zero
        self._monitors_applied = monitors_applied

        return V
    
        def vanilla_black76_price(
        self,
        S: Optional[float] = None,
        sigma: Optional[float] = None,
        T: Optional[float] = None,
    ) -> float:
        """
        Standard Black–76 price using the same rates and dividend
        treatment that the PDE uses.

        We treat the option as written on a forward with:
            F = (S_eff) * exp(carry_rate * time_to_carry)
        discounted back with:
            df = exp(-discount_rate * time_to_discount)

        Parameters
        ----------
        S : optional float
            Spot. If None, self.spot is used.
        sigma : optional float
            Volatility. If None, self.sigma is used.
        T : optional float
            Time to expiry. If None, self.time_to_expiry is used.

        Returns
        -------
        price : float
            Vanilla option price (no barrier).
        """
        S0 = self.spot if S is None else S
        vol = self.sigma if sigma is None else sigma
        T_exp = self.time_to_expiry if T is None else T

        if T_exp <= 0.0 or vol <= 0.0:
            # Intrinsic value fallback
            if self.option_type.lower() == "call":
                return max(S0 - self.strike, 0.0)
            else:
                return max(self.strike - S0, 0.0)

        # Same carry & discount as PDE
        r = self.discount_rate_nacc
        b = self.carry_rate_nacc

        # If you pre-compute PV of discrete dividends and subtract on the PDE,
        # do the same here for consistency:
        if getattr(self, "pv_divs", 0.0) != 0.0:
            S_eff = S0 - self.pv_divs
        else:
            S_eff = S0

        # Forward (in whatever measure is consistent with your PDE)
        F = S_eff * math.exp(b * self.time_to_carry)
        df = math.exp(-r * self.time_to_discount)

        sqrtT = math.sqrt(T_exp)
        sig_sqrtT = vol * sqrtT

        if F <= 0.0 or sig_sqrtT <= 0.0:
            # Degenerate: behave like forward intrinsic
            if self.option_type.lower() == "call":
                return df * max(F - self.strike, 0.0)
            else:
                return df * max(self.strike - F, 0.0)

        d1 = (math.log(F / self.strike) + 0.5 * vol * vol * T_exp) / sig_sqrtT
        d2 = d1 - sig_sqrtT

        Nd1 = norm.cdf(d1)
        Nd2 = norm.cdf(d2)
        if self.option_type.lower() == "call":
            price = df * (F * Nd1 - self.strike * Nd2)
        else:
            Nmd1 = norm.cdf(-d1)
            Nmd2 = norm.cdf(-d2)
            price = df * (self.strike * Nmd2 - F * Nmd1)

        return float(price)
    
       def vanilla_black76_greeks_fd(
        self,
        dS: float = 0.001,
        dSigma: float = 0.0001,   # 1bp vol
        dT: float = 1.0 / 365.0,  # 1 calendar day in years
    ) -> dict[str, float]:
        """
        Greeks from finite differences on the vanilla Black–76 price.

        dS      = absolute spot bump
        dSigma  = absolute vol bump (e.g. 0.0001 = 1bp)
        dT      = absolute time bump in YEARS
        """
        S0 = self.spot
        sig0 = self.sigma
        T0 = self.time_to_expiry

        # Base price
        p0 = self.vanilla_black76_price(S=S0, sigma=sig0, T=T0)

        # Delta & Gamma (central in S)
        p_up = self.vanilla_black76_price(S=S0 + dS, sigma=sig0, T=T0)
        p_dn = self.vanilla_black76_price(S=S0 - dS, sigma=sig0, T=T0)

        delta = (p_up - p_dn) / (2.0 * dS)
        gamma = (p_up - 2.0 * p0 + p_dn) / (dS ** 2)

        # Vega (one-sided bump in vol, to match FA convention)
        p_vu = self.vanilla_black76_price(S=S0, sigma=sig0 + dSigma, T=T0)
        vega = (p_vu - p0) / (dSigma * 100.0)  # “per 1% vol” like FA

        # Theta (bump T *forward* in calendar time)
        T1 = max(T0 - dT, 0.0)
        p_T1 = self.vanilla_black76_price(S=S0, sigma=sig0, T=T1)
        theta_annual = (p_T1 - p0) / dT   # per year
        theta_daily = theta_annual / 365.0

        return {
            "price": p0,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta_annual,
            "theta_daily": theta_daily,
        }
        
            def _pde_price_only(
        self,
        n_time: int,
        apply_KO: bool,
        sigma: Optional[float] = None,
    ) -> float:
        """
        Helper: run the PDE with a given number of time steps and,
        optionally, a bumped volatility; return the price at S0.
        """
        # Save original state
        orig_sigma = self.sigma
        orig_N = self.num_time_steps

        try:
            if sigma is not None:
                self.sigma = sigma
            self.num_time_steps = n_time

            V_grid = self.solve_grid(apply_KO=apply_KO)
            price = self._interp_price_from_grid(V_grid)  # your existing interp
        finally:
            # Restore
            self.sigma = orig_sigma
            self.num_time_steps = orig_N

        return price

    def _pde_price_and_greeks(
        self,
        apply_KO: bool,
        dv_sigma: float = 0.0001,
        use_richardson: bool = True,
    ) -> dict[str, float]:
        """
        Use the CN+Rannacher PDE engine to compute price & Greeks.

        * Price, delta, gamma, theta come from the base grid.
        * Vega uses a one-sided bump in σ (to match FA) and can
          also benefit from Richardson extrapolation in time.

        If use_richardson is True, we assume the temporal error
        behaves like O((Δt)^2) and combine N and N/2 solutions via:

            V* ≈ (4 V_N  - V_{N/2}) / 3
        """
        N_full = self.num_time_steps
        N_half = max(N_full // 2, 1)

        # ---------- Base grid: N_full steps ----------
        V_full = self.solve_grid(apply_KO=apply_KO)
        price_full = self._interp_price_from_grid(V_full)
        delta_full, gamma_full = self._delta_gamma_from_grid(V_full)

        # A crude theta from the PDE grid: treat last global Δt as “time step”
        # You already store this in _last_dt_at_zero from _run_backward.
        dt_last = getattr(self, "_last_dt_at_zero", None)
        if dt_last is not None and dt_last > 0.0:
            # Re-use grid at t=0 and t=dt_last from diagnostics if you store it.
            # If not available, you can approximate theta from PDE vs vanilla:
            theta_full = (price_full - self.vanilla_black76_price()) / dt_last
        else:
            theta_full = float("nan")

        # ---------- Base grid at N_half for Richardson ----------
        if use_richardson and N_half != N_full:
            orig_N = self.num_time_steps
            try:
                self.num_time_steps = N_half
                V_half = self.solve_grid(apply_KO=apply_KO)
                price_half = self._interp_price_from_grid(V_half)
            finally:
                self.num_time_steps = orig_N

            # Richardson on price (p≈2)
            price_base = (4.0 * price_full - price_half) / 3.0
        else:
            price_base = price_full

        # ---------- Vega: one-sided bump in vol ----------
        sig0 = self.sigma
        sig_up = sig0 + dv_sigma

        # For vega we also want Richardson on EACH price run if enabled
        # so that the vega error is dominated by the bump, not Δt.
        if use_richardson and N_half != N_full:
            # Base price we already have (price_base).
            # Now bumped vol:
            p_up_full = self._pde_price_only(
                n_time=N_full, apply_KO=apply_KO, sigma=sig_up
            )
            p_up_half = self._pde_price_only(
                n_time=N_half, apply_KO=apply_KO, sigma=sig_up
            )
            p_up = (4.0 * p_up_full - p_up_half) / 3.0
        else:
            # No Richardson: just N_full
            p_up = self._pde_price_only(
                n_time=N_full, apply_KO=apply_KO, sigma=sig_up
            )

        vega = (p_up - price_base) / (dv_sigma * 100.0)  # “per 1% vol”

        return {
            "price": price_base,
            "delta": delta_full,
            "gamma": gamma_full,
            "vega": vega,
            "theta": theta_full,
        }
        
            def price_log2(self) -> float:
        """
        Barrier price using PDE + Rannacher+CN with optional KO projection,
        and KI/KO parity when needed. This is the main pricing interface.

        - bt == 'none': vanilla Black–76 (no barrier)
        - KO types: PDE price with KO projection
        - KI types: vanilla price minus matching KO price
        """
        bt = self.barrier_type.lower()

        if bt == "none":
            # Pure vanilla
            return self.vanilla_black76_price()

        # 1) Knock-out: directly from PDE
        if bt in ("down-and-out", "up-and-out", "double-out"):
            if getattr(self, "already_hit", False):
                # Immediate rebate if already knocked out (if you have this)
                if self.rebate_at_hit:
                    df = math.exp(-self.discount_rate_nacc * self.time_to_discount)
                    return df * self.rebate_amount
                return 0.0
            res = self._pde_price_and_greeks(apply_KO=True)
            return res["price"]

        # 2) Knock-in: parity with KO
        if bt in ("down-and-in", "up-and-in", "double-in"):
            # Save / restore barrier_type
            original_bt = bt
            try:
                # Vanilla price (no barrier)
                self.barrier_type = "none"
                p_van = self.vanilla_black76_price()

                # Matching KO type:
                if original_bt == "down-and-in":
                    self.barrier_type = "down-and-out"
                elif original_bt == "up-and-in":
                    self.barrier_type = "up-and-out"
                else:  # double-in
                    self.barrier_type = "double-out"

                p_KO = self._pde_price_and_greeks(apply_KO=True)["price"]

            finally:
                # Restore
                self.barrier_type = original_bt

            return p_van - p_KO

        raise ValueError(f"Unsupported barrier type: {self.barrier_type}")
    
       def greeks_log2(self, dv_sigma: float = 0.0001) -> dict[str, float]:
        """
        Barrier Greeks consistent with price_log2:

        - For KO: directly from PDE _pde_price_and_greeks.
        - For KI: vanilla Greeks minus matching KO Greeks (parity).
        - For 'none': vanilla Black–76 Greeks.

        Uses a one-sided vol bump for vega to be consistent with FA.
        """
        bt = self.barrier_type.lower()

        if bt == "none":
            return self.vanilla_black76_greeks_fd(dSigma=dv_sigma)

        # KO types: full PDE Greeks
        if bt in ("down-and-out", "up-and-out", "double-out"):
            return self._pde_price_and_greeks(apply_KO=True, dv_sigma=dv_sigma)

        # KI types: parity
        if bt in ("down-and-in", "up-and-in", "double-in"):
            original_bt = bt
            try:
                # Vanilla Greeks (no barrier)
                self.barrier_type = "none"
                g_van = self.vanilla_black76_greeks_fd(dSigma=dv_sigma)

                # Matching KO Greeks
                if original_bt == "down-and-in":
                    self.barrier_type = "down-and-out"
                elif original_bt == "up-and-in":
                    self.barrier_type = "up-and-out"
                else:  # double-in
                    self.barrier_type = "double-out"

                g_KO = self._pde_price_and_greeks(
                    apply_KO=True,
                    dv_sigma=dv_sigma,
                )

            finally:
                self.barrier_type = original_bt

            # Parity: KI = vanilla − KO, applied component-wise
            return {
                "price": g_van["price"] - g_KO["price"],
                "delta": g_van["delta"] - g_KO["delta"],
                "gamma": g_van["gamma"] - g_KO["gamma"],
                "vega": g_van["vega"] - g_KO["vega"],
                "theta": g_van["theta"] - g_KO["theta"],
                "theta_daily": g_van["theta_daily"] - g_KO.get("theta_daily", 0.0),
            }

        raise ValueError(f"Unsupported barrier type: {self.barrier_type}")
    
        def _solve_grid(self, apply_KO: bool, N_time: int | None = None) -> list[float]:
        """
        Solve the PDE on the log-S grid using a theta-scheme with
        Rannacher time-stepping (first few steps = BE, then CN).

        Parameters
        ----------
        apply_KO : bool
            If True, apply knock-out projection at monitoring times.
        N_time : Optional[int]
            Number of time steps for this solve. If None, uses self.N_time.

        Returns
        -------
        V : list[float]
            Option values on the S-grid at valuation time (tau = T).
        """
        # ----- grid & basic coefficients -----
        self.configure_grid()
        dx = self._build_log_grid()          # assumes uniform log grid
        N = self.N_space                     # number of space steps (V has N+1 nodes)

        # possibly override N_time for Richardson
        N_time = int(N_time) if N_time is not None else int(self.N_time)
        assert N_time >= 1

        dt = self.T / float(N_time)

        sig2 = self.sigma * self.sigma
        # drift in log-space under risk-neutral measure
        mu_x = self.b_carry - 0.5 * sig2

        alpha = 0.5 * sig2 / (dx * dx)
        beta_adv = mu_x / (2.0 * dx)

        # spatial operator coefficients L V_j = a V_{j-1} + b V_j + c V_{j+1}
        a = alpha - beta_adv
        c = alpha + beta_adv
        bcoef = -2.0 * alpha - self.r_disc

        # ----- local Thomas solver (for interior nodes only) -----
        def solve_tridiag(A_L: float, A_C: float, A_U: float,
                          rhs: list[float]) -> list[float]:
            """
            Solve tri-diagonal system with *constant* diagonals:
                (A_L, A_C, A_U) for interior nodes.
            Dimension is len(rhs).
            """
            n = len(rhs)
            c_prime = [0.0] * n
            d_prime = [0.0] * n

            # first row
            pivot = A_C
            if abs(pivot) < 1e-14:
                pivot = 1e-14
            c_prime[0] = A_U / pivot
            d_prime[0] = rhs[0] / pivot

            # forward sweep
            for i in range(1, n):
                pivot = A_C - A_L * c_prime[i - 1]
                if abs(pivot) < 1e-14:
                    pivot = 1e-14
                if i < n - 1:
                    c_prime[i] = A_U / pivot
                d_prime[i] = (rhs[i] - A_L * d_prime[i - 1]) / pivot

            # back substitution
            x = [0.0] * n
            x[-1] = d_prime[-1]
            for i in range(n - 2, -1, -1):
                x[i] = d_prime[i] - c_prime[i] * x[i + 1]

            return x

        # ----- initial condition at tau = 0 (maturity) -----
        V = self._terminal_payoff()      # V[j] at tau = 0
        monitor_idx = self._monitor_indices_tau(dt)

        # ----- march forward in tau : 0 -> T (i.e. backward in calendar time) -----
        for m in range(N_time):
            # Rannacher: first RANNACHER_STEPS global steps = backward Euler
            if m < self.RANNACHER_STEPS:
                theta = 1.0   # fully implicit (EB)
            else:
                theta = 0.5   # CN afterwards

            # theta–scheme matrices:
            # (I - theta dt L) V^{m+1} = (I + (1-theta) dt L) V^m
            A_L = -theta * dt * a
            A_C = 1.0 - theta * dt * bcoef
            A_U = -theta * dt * c

            B_L = (1.0 - theta) * dt * a
            B_C = 1.0 + (1.0 - theta) * dt * bcoef
            B_U = (1.0 - theta) * dt * c

            tau_next = (m + 1) * dt
            V_min_next, V_max_next = self._boundary_values(tau_next)

            # RHS for interior nodes j = 1..N-1
            rhs = [0.0] * (N - 1)
            for j in range(1, N):
                Vjm1, Vj, Vjp1 = V[j - 1], V[j], V[j + 1]
                rhs[j - 1] = (B_L * Vjm1 + B_C * Vj + B_U * Vjp1)

            # Dirichlet boundary contributions for next step
            rhs[0] -= A_L * V_min_next
            rhs[-1] -= A_U * V_max_next

            # solve for interior values at tau_next
            V_int = solve_tridiag(A_L, A_C, A_U, rhs)

            # inject boundaries and interior into full grid
            V[0] = V_min_next
            V[-1] = V_max_next
            V[1:-1] = V_int

            # knock-out projection at monitoring times
            if apply_KO and (m + 1) in monitor_idx:
                self._apply_KO_projection(V)

        return V
    
        def _price_from_grid(self, V: list[float]) -> float:
        """
        Interpolate V(S) at spot S0 to get a single scalar price.
        Assumes self._interp_price_from_grid exists.
        """
        return self._interp_price_from_grid(V)

    # ---------- core PDE price (no Richardson) ----------
    def _pde_price_single(self, N_time: int, apply_KO: bool) -> float:
        V = self._solve_grid(apply_KO=apply_KO, N_time=N_time)
        return self._price_from_grid(V)

    # ---------- public price with optional Richardson ----------
    def price_log2(self, apply_KO: bool = True, use_richardson: bool = True) -> float:
        """
        Discrete barrier price in log-space via FDM.

        - Uses Rannacher time-stepping inside _solve_grid.
        - Optionally applies Richardson extrapolation in *time*
          assuming O(dt^2) temporal error.

        Returns
        -------
        float : theoretical price per contract.
        """
        N = int(self.N_time)
        if N < 2 or not use_richardson:
            return self._pde_price_single(N_time=N, apply_KO=apply_KO)

        # high-resolution solution
        p_N = self._pde_price_single(N_time=N, apply_KO=apply_KO)

        # coarser solution with half the steps
        N_half = max(1, N // 2)
        p_half = self._pde_price_single(N_time=N_half, apply_KO=apply_KO)

        # Richardson (error ~ C dt^2):
        # p* = p_N + (p_N - p_half) / (2^2 - 1) = (4 p_N - p_half) / 3
        p_star = (4.0 * p_N - p_half) / 3.0
        return p_star
    
        def _pde_solution(self, N_time: int, apply_KO: bool,
                      sigma_override: float | None = None) -> list[float]:
        """
        Helper: run the PDE with optional volatility override and return V-grid.
        """
        old_sigma = self.sigma
        old_Ntime = self.N_time

        self.N_time = N_time
        if sigma_override is not None:
            self.sigma = sigma_override

        try:
            V = self._solve_grid(apply_KO=apply_KO, N_time=N_time)
        finally:
            self.sigma = old_sigma
            self.N_time = old_Ntime

        return V

    def _richardson_price(self, apply_KO: bool,
                          sigma_override: float | None = None) -> float:
        """
        Price with Richardson in time (N vs N/2), optionally with a given vol.
        """
        N = int(self.N_time)
        if N < 2:
            V = self._pde_solution(N_time=N, apply_KO=apply_KO,
                                   sigma_override=sigma_override)
            return self._price_from_grid(V)

        N_half = max(1, N // 2)

        V_N = self._pde_solution(N_time=N, apply_KO=apply_KO,
                                 sigma_override=sigma_override)
        p_N = self._price_from_grid(V_N)

        V_half = self._pde_solution(N_time=N_half, apply_KO=apply_KO,
                                    sigma_override=sigma_override)
        p_half = self._price_from_grid(V_half)

        return (4.0 * p_N - p_half) / 3.0, V_N  # return hi-res grid too

    # ---------- PDE price + Greeks ----------
    def pde_price_and_greeks(self,
                             apply_KO: bool,
                             dv_sigma: float = 0.0001,
                             use_richardson: bool = True) -> dict[str, float]:
        """
        Use the CN+Rannacher PDE to compute:
          - price
          - Delta, Gamma from the high-res grid (log-S)
          - Vega from vol bump, using same PDE and Richardson.

        Returns dictionary keyed by 'price', 'delta', 'gamma', 'vega'.
        """
        N = int(self.N_time)

        # Base price (with optional Richardson) + hi-res grid
        if use_richardson and N >= 2:
            base_price, V_grid = self._richardson_price(apply_KO=apply_KO)
        else:
            V_grid = self._pde_solution(N_time=N, apply_KO=apply_KO,
                                        sigma_override=None)
            base_price = self._price_from_grid(V_grid)

        # Delta & Gamma from grid (existing routine)
        delta, gamma = self._delta_gamma_from_grid(V_grid)

        # Vega – one-sided vol bump, consistent with FA
        sigma0 = self.sigma
        sigma_up = sigma0 + dv_sigma

        if use_richardson and N >= 2:
            price_up, _ = self._richardson_price(apply_KO=apply_KO,
                                                 sigma_override=sigma_up)
        else:
            V_up = self._pde_solution(N_time=N, apply_KO=apply_KO,
                                      sigma_override=sigma_up)
            price_up = self._price_from_grid(V_up)

        vega = (price_up - base_price) / dv_sigma   # per absolute vol unit

        return {
            "price": base_price,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
        }

    # ---------- external entry point used by your main script ----------
    def greeks_log2(self, dv_sigma: float = 0.0001,
                    use_richardson: bool = True) -> dict[str, float]:
        """
        Public interface: PDE-based price and Greeks in log-space.
        """
        return self.pde_price_and_greeks(
            apply_KO=True,
            dv_sigma=dv_sigma,
            use_richardson=use_richardson,
        )
        
    def _is_ko(self, barrier_type: str) -> bool:
        bt = barrier_type.lower()
        return bt in ("down-and-out", "up-and-out", "double-out")

    def _is_ki(self, barrier_type: str) -> bool:
        bt = barrier_type.lower()
        return bt in ("down-and-in", "up-and-in", "double-in")

    def _ko_type_from_ki(self, ki_type: str) -> str:
        bt = ki_type.lower()
        if bt == "down-and-in": return "down-and-out"
        if bt == "up-and-in":   return "up-and-out"
        if bt == "double-in":   return "double-out"
        raise ValueError(f"Unknown KI type: {ki_type}")
    
        def _pde_solution(self, N_time: int, apply_KO: bool, sigma_override: float | None = None):
        old_sig, old_N = self.sigma, self.N_time
        try:
            if sigma_override is not None:
                self.sigma = sigma_override
            self.N_time = N_time
            V = self._solve_grid(apply_KO=apply_KO, N_time=N_time)
        finally:
            self.sigma = old_sig
            self.N_time = old_N
        return V

    def _price_from_grid(self, V):
        return self._interp_price_from_grid(V)

    def _richardson_price(self, apply_KO: bool, sigma_override: float | None = None):
        N = int(self.N_time)
        if N < 2:
            V = self._pde_solution(N_time=N, apply_KO=apply_KO, sigma_override=sigma_override)
            return self._price_from_grid(V), V

        N_half = max(1, N // 2)
        V_N    = self._pde_solution(N_time=N,     apply_KO=apply_KO, sigma_override=sigma_override)
        V_H    = self._pde_solution(N_time=N_half,apply_KO=apply_KO, sigma_override=sigma_override)
        p_N    = self._price_from_grid(V_N)
        p_H    = self._price_from_grid(V_H)
        p_star = (4.0 * p_N - p_H) / 3.0
        return p_star, V_N  # use hi-res grid for greeks

    def pde_price_and_greeks(self, apply_KO: bool, dv_sigma: float = 1e-4, use_richardson: bool = True) -> dict:
        N = int(self.N_time)
        if use_richardson and N >= 2:
            price_base, V_grid = self._richardson_price(apply_KO=apply_KO, sigma_override=None)
        else:
            V_grid = self._pde_solution(N_time=N, apply_KO=apply_KO, sigma_override=None)
            price_base = self._price_from_grid(V_grid)

        delta, gamma = self._delta_gamma_from_grid(V_grid)  # your existing non-uniform central diff

        # one-sided vega (FA convention), *with* Richardson on each price if requested
        sig_up = self.sigma + dv_sigma
        if use_richardson and N >= 2:
            price_up, _ = self._richardson_price(apply_KO=apply_KO, sigma_override=sig_up)
        else:
            V_up = self._pde_solution(N_time=N, apply_KO=apply_KO, sigma_override=sig_up)
            price_up = self._price_from_grid(V_up)
        vega = (price_up - price_base) / dv_sigma

        return {"price": price_base, "delta": delta, "gamma": gamma, "vega": vega}
    
    
        def price_log2(self, use_richardson: bool = True) -> float:
        bt = self.barrier_type.lower()

        # no barrier → pure vanilla
        if bt == "none":
            return self.vanilla_black76_price()

        # KO → PDE
        if self._is_ko(bt):
            return self.pde_price_and_greeks(apply_KO=True, use_richardson=use_richardson)["price"]

        # KI → parity: vanilla − KO (same rebate convention)
        if self._is_ki(bt):
            ko_type = self._ko_type_from_ki(bt)
            # vanilla
            p_van = self.vanilla_black76_price()
            # KO price with PDE (temporarily switch type)
            save_bt = self.barrier_type
            try:
                self.barrier_type = ko_type
                p_ko = self.pde_price_and_greeks(apply_KO=True, use_richardson=use_richardson)["price"]
            finally:
                self.barrier_type = save_bt
            return p_van - p_ko

        raise ValueError(f"Unsupported barrier type: {self.barrier_type}")
    
        def greeks_log2(self, dv_sigma: float = 1e-4, use_richardson: bool = True) -> dict:
        bt = self.barrier_type.lower()

        if bt == "none":
            return self.vanilla_black76_greeks_fd(dSigma=dv_sigma)

        # KO → PDE Greeks directly
        if self._is_ko(bt):
            return self.pde_price_and_greeks(apply_KO=True, dv_sigma=dv_sigma, use_richardson=use_richardson)

        # KI → parity: (vanilla Greeks) − (KO Greeks)
        if self._is_ki(bt):
            ko_type = self._ko_type_from_ki(bt)

            save_bt = self.barrier_type
            try:
                # vanilla Greeks
                self.barrier_type = "none"
                g_van = self.vanilla_black76_greeks_fd(dSigma=dv_sigma)

                # KO Greeks via PDE
                self.barrier_type = ko_type
                g_ko  = self.pde_price_and_greeks(apply_KO=True, dv_sigma=dv_sigma, use_richardson=use_richardson)
            finally:
                self.barrier_type = save_bt

            return {
                "price": g_van["price"] - g_ko["price"],
                "delta": g_van["delta"] - g_ko["delta"],
                "gamma": g_van["gamma"] - g_ko["gamma"],
                "vega":  g_van["vega"]  - g_ko["vega"],
                "theta": g_van.get("theta", float("nan")) - g_ko.get("theta", 0.0),
            }

        raise ValueError(f"Unsupported barrier type: {self.barrier_type}")