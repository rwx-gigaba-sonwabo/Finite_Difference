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
