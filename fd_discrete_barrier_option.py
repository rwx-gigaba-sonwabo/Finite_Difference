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
