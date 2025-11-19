import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict

# -------------------------------------------------------------------
# Normal PDF / CDF helpers
# -------------------------------------------------------------------

SQRT_2PI = math.sqrt(2.0 * math.pi)


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def norm_cdf(x: float) -> float:
    # Good enough for risk-engine use
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# -------------------------------------------------------------------
# Discrete barrier Crank–Nicolson engine in log-spot
# -------------------------------------------------------------------

@dataclass
class DiscreteBarrierCrankNicolsonLog:
    # Core inputs
    S0: float                   # spot
    K: float                    # strike
    T: float                    # time to maturity (years)
    sigma: float                # volatility
    r_disc: float               # discount rate r
    b_carry: float              # carry (e.g. r - q_eff)
    option_type: str            # "call" or "put"
    barrier_type: str           # "none", "down-and-out", "up-and-out",
                                # "down-and-in", "up-and-in"

    # Barrier & rebate
    lower_barrier: Optional[float] = None
    upper_barrier: Optional[float] = None
    rebate: float = 0.0         # paid at hit / expiry (depending on KO logic)

    # Monitoring times in **calendar** years from valuation (discrete barrier)
    monitor_times: Optional[List[float]] = None

    # Grid controls (can be left as None to auto-configure)
    N_space: Optional[int] = None
    N_time: Optional[int] = None

    # Internal state (filled by configure_grid / _build_log_grid)
    _S_min: float = field(init=False, default=0.0)
    _S_max: float = field(init=False, default=0.0)
    s_nodes: List[float] = field(init=False, default_factory=list)

    # ------------------------------------------------------------------
    # Grid configuration
    # ------------------------------------------------------------------

    def configure_grid(self) -> None:
        """Choose sensible space / time steps given T, sigma, monitoring."""
        if self.T <= 0.0:
            raise ValueError("T must be positive")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if self.S0 <= 0.0:
            raise ValueError("S0 must be positive")

        # --- Space domain: cover spot, strike, barriers, with extra margin ---
        candidates = [self.S0, self.K]
        if self.lower_barrier is not None and self.lower_barrier > 0:
            candidates.append(self.lower_barrier)
        if self.upper_barrier is not None and self.upper_barrier > 0:
            candidates.append(self.upper_barrier)

        s_low = min(candidates)
        s_high = max(candidates)

        # A few sigmas out in price space (log grid will handle tails)
        L_low = 4.0
        L_high = 4.0
        S_min = max(1e-8, s_low / L_low)
        S_max = s_high * L_high
        if S_min >= S_max:
            S_min = self.S0 / 5.0
            S_max = self.S0 * 5.0

        self._S_min, self._S_max = S_min, S_max

        # --- Space step (log grid) ---
        x_min, x_max = math.log(S_min), math.log(S_max)
        x_range = x_max - x_min
        points_per_sigma = 12
        dx_target = self.sigma * math.sqrt(self.T) / points_per_sigma
        if dx_target <= 0.0:
            dx_target = x_range / 300.0

        if self.N_space is None:
            N_space = int(math.ceil(x_range / dx_target))
            self.N_space = max(N_space, 300)  # never too coarse

        # --- Time step: λ = 0.5 σ² Δt / (Δx)² ~ 0.4, and enough steps per monitor ---
        if self.N_time is None:
            dx = x_range / self.N_space
            lambda_target = 0.4
            Ntime_opt = int(
                math.ceil(
                    0.5 * self.sigma * self.sigma * self.T
                    / (lambda_target * dx * dx)
                )
            )

            valid_mon = [
                t for t in (self.monitor_times or [])
                if 0.0 < t < self.T
            ]
            M_intervals = len(valid_mon) + 1
            # At least as many time steps as space, and ~10 per interval
            self.N_time = max(Ntime_opt, self.N_space, 10 * M_intervals)

    # ------------------------------------------------------------------
    # Build log-spot grid
    # ------------------------------------------------------------------

    def _build_log_grid(self) -> float:
        """Construct uniform grid in log S; return dx."""
        if self._S_min <= 0.0 or self._S_max <= 0.0 or self.N_space is None:
            self.configure_grid()

        x_min, x_max = math.log(self._S_min), math.log(self._S_max)
        N = self.N_space
        dx = (x_max - x_min) / N

        x_nodes = [x_min + i * dx for i in range(N + 1)]
        self.s_nodes = [math.exp(x) for x in x_nodes]

        return dx

    # ------------------------------------------------------------------
    # Payoff and boundaries
    # ------------------------------------------------------------------

    def _terminal_payoff(self) -> List[float]:
        """Payoff at maturity for vanilla call/put."""
        is_call = self.option_type.lower() == "call"
        payoff = []
        for S in self.s_nodes:
            if is_call:
                payoff.append(max(S - self.K, 0.0))
            else:
                payoff.append(max(self.K - S, 0.0))
        return payoff

    def _boundary_values(self, tau: float) -> (float, float):
        """
        Dirichlet boundaries at time-to-maturity tau, consistent with
        Black–Scholes with carry b and discount r.
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

    # ------------------------------------------------------------------
    # Monitoring times mapping (calendar t -> tau index)
    # ------------------------------------------------------------------

    def _monitor_indices_tau(self, dt: float) -> set:
        """
        Map discrete monitoring times t_mon in [0,T] to indices in tau = T - t.
        We project at tau_k = (k * dt) that corresponds to t = T - tau_k.
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

    # ------------------------------------------------------------------
    # Knock-out projection
    # ------------------------------------------------------------------

    def _apply_KO_projection(self, V: List[float]) -> None:
        """Project values onto KO payoffs at a monitoring time."""
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

    # ------------------------------------------------------------------
    # Crank–Nicolson solve in log-spot
    # ------------------------------------------------------------------

    def _solve_grid(self, apply_KO: bool) -> List[float]:
        """
        Solve dV/dτ = L V with L from Black–Scholes in log S via CN,
        marching τ from 0 to T (backwards in calendar time).
        """
        self.configure_grid()
        dx = self._build_log_grid()

        N = self.N_space
        dt = self.T / self.N_time

        sig2 = self.sigma * self.sigma
        mu_x = self.b_carry - 0.5 * sig2

        # Operator coefficients in log space
        alpha = 0.5 * sig2 / (dx * dx)
        beta_adv = mu_x / (2.0 * dx)

        a = alpha - beta_adv                    # coeff for V_{j-1}
        c = alpha + beta_adv                    # coeff for V_{j+1}
        bcoef = -2.0 * alpha - self.r_disc      # coeff for V_j

        # CN matrices: (I - dt/2 L) V^{m+1} = (I + dt/2 L) V^m
        A_L = -0.5 * dt * a
        A_C = 1.0 - 0.5 * dt * bcoef
        A_U = -0.5 * dt * c

        B_L = 0.5 * dt * a
        B_C = 1.0 + 0.5 * dt * bcoef
        B_U = 0.5 * dt * c

        def solve_tridiag(rhs: List[float]) -> List[float]:
            """Thomas algorithm for Toeplitz tri-diagonal with A_L, A_C, A_U."""
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

        # Terminal condition
        V = self._terminal_payoff()

        # Monitoring indices in tau
        monitor_idx = self._monitor_indices_tau(dt)

        # March tau from 0 -> T (m = 0 is τ=dt)
        for m in range(self.N_time):
            tau_next = (m + 1) * dt
            V_min_next, V_max_next = self._boundary_values(tau_next)

            # RHS for interior nodes j = 1..N-1
            rhs = [0.0] * (N - 1)
            for j in range(1, N):
                Vjm1, Vj, Vjp1 = V[j - 1], V[j], V[j + 1]
                rhs[j - 1] = B_L * Vjm1 + B_C * Vj + B_U * Vjp1

            # Include boundaries (Dirichlet)
            rhs[0]   -= A_L * V_min_next
            rhs[-1]  -= A_U * V_max_next

            V_int = solve_tridiag(rhs)

            V[0] = V_min_next
            V[-1] = V_max_next
            V[1:-1] = V_int

            # Apply KO at this monitoring time in tau-space
            if apply_KO and (m + 1) in monitor_idx:
                self._apply_KO_projection(V)

        return V

    # ------------------------------------------------------------------
    # Interpolation & grid Greeks
    # ------------------------------------------------------------------

    def _interp_price_from_grid(self, V: List[float]) -> float:
        """Linear interpolation of grid value at S0."""
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
        """Non-uniform central differences for delta & gamma at S0."""
        s = self.s_nodes
        S0 = self.S0

        # choose interior node closest to S0
        idx = min(range(1, len(s) - 1), key=lambda i: abs(s[i] - S0))

        S_im1, S_i, S_ip1 = s[idx - 1], s[idx], s[idx + 1]
        V_im1, V_i, V_ip1 = V[idx - 1], V[idx], V[idx + 1]

        h1 = S_i - S_im1
        h2 = S_ip1 - S_i

        delta = (
            -h2 / (h1 * (h1 + h2)) * V_im1
            + (h2 - h1) / (h1 * h2) * V_i
            + h1 / (h2 * (h1 + h2)) * V_ip1
        )
        gamma = 2.0 * (
            V_im1 / (h1 * (h1 + h2))
            - V_i / (h1 * h2)
            + V_ip1 / (h2 * (h1 + h2))
        )
        return delta, gamma

    # ------------------------------------------------------------------
    # Closed-form vanilla Black–Scholes (with continuous carry b, discount r)
    # ------------------------------------------------------------------

    def _vanilla_bs_price_and_greeks(self) -> Dict[str, float]:
        """
        Closed-form European vanilla under Black–Scholes with continuous
        carry b and discount r. All Greeks are w.r.t SPOT S0.
        """
        S0 = self.S0
        K = self.K
        T = self.T
        sigma = self.sigma
        r = self.r_disc
        b = self.b_carry

        if T <= 0.0 or sigma <= 0.0:
            if self.option_type.lower() == "call":
                price = max(S0 - K, 0.0)
            else:
                price = max(K - S0, 0.0)
            return {
                "price": price,
                "delta": 0.0,
                "gamma": 0.0,
                "theta": 0.0,
                "vega": 0.0,
            }

        # implied continuous dividend yield q from carry b = r - q
        q = r - b
        sqrtT = math.sqrt(T)

        d1 = (math.log(S0 / K) + (b + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT

        Nd1 = norm_cdf(d1)
        Nd2 = norm_cdf(d2)
        nd1 = norm_pdf(d1)

        is_call = self.option_type.lower() == "call"

        disc_q = math.exp(-q * T)
        disc_r = math.exp(-r * T)

        if is_call:
            price = S0 * disc_q * Nd1 - K * disc_r * Nd2
            delta = disc_q * Nd1
        else:
            price = K * disc_r * norm_cdf(-d2) - S0 * disc_q * norm_cdf(-d1)
            delta = disc_q * (Nd1 - 1.0)

        gamma = disc_q * nd1 / (S0 * sigma * sqrtT)
        vega = S0 * disc_q * nd1 * sqrtT

        # PDE-consistent theta at t=0
        theta = -(
            0.5 * sigma * sigma * S0 * S0 * gamma
            + b * S0 * delta
            - r * price
        )

        return {
            "price": price,
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
        }

    # ------------------------------------------------------------------
    # PDE price + Greeks (for KO or vanilla via grid)
    # ------------------------------------------------------------------

    def _pde_price_and_greeks(self, apply_KO: bool, dv_sigma: float) -> Dict[str, float]:
        """Use CN grid to compute price + Greeks for current barrier_type."""
        V_grid = self._solve_grid(apply_KO=apply_KO)
        price = self._interp_price_from_grid(V_grid)
        delta, gamma = self._delta_gamma_from_grid(V_grid)

        S0 = self.S0
        sigma = self.sigma
        r = self.r_disc
        b = self.b_carry

        theta = -(
            0.5 * sigma * sigma * S0 * S0 * gamma
            + b * S0 * delta
            - r * price
        )

        # Vega via central bump in sigma (same CN engine)
        sigma_orig = sigma

        self.sigma = sigma_orig + dv_sigma
        V_up = self._solve_grid(apply_KO=apply_KO)
        p_up = self._interp_price_from_grid(V_up)

        self.sigma = sigma_orig - dv_sigma
        V_dn = self._solve_grid(apply_KO=apply_KO)
        p_dn = self._interp_price_from_grid(V_dn)

        self.sigma = sigma_orig
        vega = (p_up - p_dn) / (2.0 * dv_sigma)

        return {
            "price": price,
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
        }

    # ------------------------------------------------------------------
    # Public API: price and Greeks
    # ------------------------------------------------------------------

    def price(self) -> float:
        """
        Price dispatcher:
        - vanilla (no barrier): closed-form Black–Scholes,
        - KO: CN PDE with KO projection,
        - KI: parity V_KI = V_vanilla - V_KO.
        """
        bt = self.barrier_type.lower()

        # Vanilla: Black–Scholes closed form
        if bt == "none":
            return self._vanilla_bs_price_and_greeks()["price"]

        # Knock-out: PDE with KO projection
        if bt in ("down-and-out", "up-and-out"):
            return self._pde_price_and_greeks(apply_KO=True, dv_sigma=1e-3)["price"]

        # Knock-in: parity vs vanilla and KO
        if bt in ("down-and-in", "up-and-in"):
            original_bt = bt

            g_van = self._vanilla_bs_price_and_greeks()

            # matching knock-out
            self.barrier_type = "down-and-out" if bt == "down-and-in" else "up-and-out"
            g_ko = self._pde_price_and_greeks(apply_KO=True, dv_sigma=1e-3)

            self.barrier_type = original_bt
            return g_van["price"] - g_ko["price"]

        raise ValueError(f"Unsupported barrier_type: {self.barrier_type}")

    def greeks(self, dv_sigma: float = 1e-3) -> Dict[str, float]:
        """
        Greeks consistent with:
        - vanilla: closed-form Black–Scholes,
        - KO: CN PDE,
        - KI: parity (Greek_KI = Greek_vanilla − Greek_KO).
        """
        bt = self.barrier_type.lower()

        # Vanilla
        if bt == "none":
            return self._vanilla_bs_price_and_greeks()

        # KO
        if bt in ("down-and-out", "up-and-out"):
            return self._pde_price_and_greeks(apply_KO=True, dv_sigma=dv_sigma)

        # KI via parity
        if bt in ("down-and-in", "up-and-in"):
            original_bt = bt

            # Vanilla Greeks
            self.barrier_type = "none"
            g_van = self._vanilla_bs_price_and_greeks()

            # KO Greeks
            self.barrier_type = "down-and-out" if bt == "down-and-in" else "up-and-out"
            g_ko = self._pde_price_and_greeks(apply_KO=True, dv_sigma=dv_sigma)

            self.barrier_type = original_bt

            return {k: g_van[k] - g_ko[k] for k in g_van}

        raise ValueError(f"Unsupported barrier_type: {self.barrier_type}")

    def _vanilla_black76_greeks_fd(self,
                                   dS: float = 1e-4,
                                   dSigma: float = 1e-3,
                                   dT: float = 1e-4) -> Dict[str, float]:
        """
        Greeks (price, delta, gamma, theta, vega) for the vanilla option
        computed by finite differences on the Black-76 price.

        - dS is an absolute spot bump.
        - dSigma is an absolute vol bump (e.g. 0.001 = 0.1 vol point).
        - dT is an absolute time bump in YEARS.
        """

        S0 = self.S0
        sigma0 = self.sigma
        T0 = self.T

        # Base price
        p0 = self._vanilla_black76_price(S=S0, sigma=sigma0, T=T0)

        # ---- Delta & Gamma (bump spot) ----
        p_up_S = self._vanilla_black76_price(S=S0 + dS, sigma=sigma0, T=T0)
        p_dn_S = self._vanilla_black76_price(S=S0 - dS, sigma=sigma0, T=T0)

        delta = (p_up_S - p_dn_S) / (2.0 * dS)
        gamma = (p_up_S - 2.0 * p0 + p_dn_S) / (dS * dS)

        # ---- Vega (bump vol) ----
        p_up_v = self._vanilla_black76_price(S=S0, sigma=sigma0 + dSigma, T=T0)
        p_dn_v = self._vanilla_black76_price(S=S0, sigma=sigma0 - dSigma, T=T0)

        vega = (p_up_v - p_dn_v) / (2.0 * dSigma)

        # ---- Theta (bump time) ----
        # By convention, theta is dV/dt (calendar), so we want how price
        # changes as T decreases. Using central difference around T0:
        if T0 > 2.0 * dT:
            p_up_T = self._vanilla_black76_price(S=S0, sigma=sigma0, T=T0 + dT)
            p_dn_T = self._vanilla_black76_price(S=S0, sigma=sigma0, T=T0 - dT)
            # dV/dT with T as "time-to-expiry"; theta = -dV/dT
            dV_dT = (p_up_T - p_dn_T) / (2.0 * dT)
            theta = -dV_dT
        else:
            # One-sided near expiry
            p_dn_T = self._vanilla_black76_price(S=S0, sigma=sigma0, T=max(T0 - dT, 1e-8))
            dV_dT = (p0 - p_dn_T) / dT
            theta = -dV_dT

        return {
            "price": p0,
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
        }
        
    def greeks(self, dv_sigma: float = 1e-3) -> Dict[str, float]:
        """
        Greeks consistent with:
        - vanilla (no barrier): Black-76 price with FD Greeks,
        - KO: CN PDE engine,
        - KI: parity (Greek_KI = Greek_vanilla - Greek_KO).
        """
        bt = self.barrier_type.lower()

        # 1) Vanilla: Black-76 + FD Greeks
        if bt == "none":
            return self._vanilla_black76_greeks_fd(
                dS=max(1e-4, 1e-4 * self.S0),
                dSigma=dv_sigma,
                dT=min(1e-4, 0.5 * self.T),
            )

        # 2) Knock-out: PDE CN engine (same as before)
        if bt in ("down-and-out", "up-and-out"):
            return self._pde_price_and_greeks(apply_KO=True, dv_sigma=dv_sigma)

        # 3) Knock-in: parity Greek_KI = Greek_vanilla - Greek_KO
        if bt in ("down-and-in", "up-and-in"):
            original_bt = bt

            # Vanilla Greeks (Black-76 FD)
            self.barrier_type = "none"
            g_van = self._vanilla_black76_greeks_fd(
                dS=max(1e-4, 1e-4 * self.S0),
                dSigma=dv_sigma,
                dT=min(1e-4, 0.5 * self.T),
            )

            # Matching KO Greeks (PDE)
            self.barrier_type = "down-and-out" if bt == "down-and-in" else "up-and-out"
            g_ko = self._pde_price_and_greeks(apply_KO=True, dv_sigma=dv_sigma)

            # Restore original barrier type
            self.barrier_type = original_bt

            return {k: g_van[k] - g_ko[k] for k in g_van}

        raise ValueError(f"Unsupported barrier_type: {self.barrier_type}")
    
    def _grid_sizes_for(self, N_time: int) -> Tuple[float, float, int]:
        """
        Return (dt, dx, M_price) for a given number of time steps N_time.

        dt  : time step size in years
        dx  : log-price step size (Δlog S)
        M_price : number of price steps produced by _xgrid_for
        """
        # This calls your existing FIS-style grid chooser
        M_price = self._xgrid_for(N_time)

        # Horizon and log-domain width (already used in your code)
        T_disc = self.time_to_discount          # time to discount in years
        L = self._domain_width_L()              # total log-width of the domain

        if N_time <= 0:
            raise ValueError("N_time must be positive.")

        dt = T_disc / float(N_time)
        dx = L / float(M_price)

        return dt, dx, M_price
    
    def _quantity_on_grid(self, quantity: str, N_time: int) -> float:
        """
        Compute price or a Greek on a grid with N_time time steps.

        quantity:
            'price', 'delta', 'gamma', 'vega', 'theta_annual', 'theta_daily'
        """
        quantity = quantity.lower()

        if quantity == "price":
            # Use your existing one-shot pricing (no Richardson)
            return float(self._price_once(N_time))

        # Greeks: use existing bump-and-revalue routine
        greeks = self.calculate_greeks(N_time)

        if quantity == "delta":
            return float(greeks["Delta"])
        elif quantity == "gamma":
            return float(greeks["Gamma"])
        elif quantity == "vega":
            return float(greeks["Vega"])
        elif quantity in ("theta", "theta_annual"):
            return float(greeks["Theta (Annual)"])
        elif quantity in ("theta_daily", "theta_per_day"):
            return float(greeks["Theta (Daily)"])
        else:
            raise ValueError(f"Unknown quantity '{quantity}'.")

    def diagnose_order_of_accuracy(
        self,
        quantity: str = "vega",
        base_time_steps: int = 50,
        refinements: int = 3,
    ) -> Dict[str, object]:
        """
        Empirically estimate the convergence order for a given quantity
        (price or Greek) by refining the time grid.

        Parameters
        ----------
        quantity : str
            'price', 'delta', 'gamma', 'vega', 'theta_annual', 'theta_daily'.
        base_time_steps : int
            Coarsest number of time steps N0.
        refinements : int
            How many times to double N0, i.e. we use
            N = [N0, 2*N0, 4*N0, ..., 2^refinements * N0].

        Returns
        -------
        dict with keys:
            'quantity'         : name of the quantity
            'N_time_list'      : list of N_time values
            'dt_list'          : list of Δt for each N_time
            'dx_list'          : list of Δx = Δlog(S) for each N_time
            'values'           : approximations on each grid
            'errors_vs_finest' : |v(N) - v(N_max)| for each N
            'local_orders'     : estimated order between grid levels
            'global_order'     : overall order (median of local_orders)
        """
        if base_time_steps <= 0:
            raise ValueError("base_time_steps must be positive.")
        if refinements < 1:
            raise ValueError("refinements must be at least 1.")

        # Build refinement ladder
        N_time_list: List[int] = [base_time_steps * (2 ** k) for k in range(refinements + 1)]

        # Compute approximations and grid sizes
        values: List[float] = []
        dt_list: List[float] = []
        dx_list: List[float] = []

        for N in N_time_list:
            dt, dx, _ = self._grid_sizes_for(N)
            dt_list.append(dt)
            dx_list.append(dx)
            values.append(self._quantity_on_grid(quantity, N))

        # Treat the finest grid as "reference" solution
        v_ref = values[-1]

        # Errors for each N vs finest grid
        errors: List[float] = [abs(v - v_ref) for v in values]

        # Local order estimates between refinement levels
        local_orders: List[float] = []
        for k in range(len(N_time_list) - 2):
            e_k = errors[k]
            e_k1 = errors[k + 1]
            # Avoid log of zero – if errors are identical or zero, skip
            if e_k > 0.0 and e_k1 > 0.0 and e_k != e_k1:
                p_k = log(e_k / e_k1) / log(2.0)
                local_orders.append(p_k)

        global_order = float("nan")
        if local_orders:
            # Robust central tendency – median rather than mean
            local_orders_sorted = sorted(local_orders)
            mid = len(local_orders_sorted) // 2
            if len(local_orders_sorted) % 2 == 1:
                global_order = local_orders_sorted[mid]
            else:
                global_order = 0.5 * (local_orders_sorted[mid - 1] + local_orders_sorted[mid])

        result = {
            "quantity": quantity,
            "N_time_list": N_time_list,
            "dt_list": dt_list,
            "dx_list": dx_list,
            "values": values,
            "errors_vs_finest": errors,
            "local_orders": local_orders,
            "global_order": global_order,
        }

        return result
vega_diag = pricer.diagnose_order_of_accuracy(
    quantity="vega",
    base_time_steps=40,
    refinements=3,
)
print("Empirical order for vega:", vega_diag["global_order"])
print("N_time:", vega_diag["N_time_list"])
print("vega:", vega_diag["values"])
print("errors:", vega_diag["errors_vs_finest"])

# Price or delta if you want a baseline
price_diag = pricer.diagnose_order_of_accuracy("price", base_time_steps=40, refinements=3)
delta_diag = pricer.diagnose_order_of_accuracy("delta", base_time_steps=40, refinements=3)


    def compute_empirical_order(
        self,
        time_steps_list,
        greek: str = "vega",
        use_dt: bool = True,
    ):
        """
        Empirically estimate the convergence order in time for price or a Greek.

        Parameters
        ----------
        time_steps_list : iterable of int
            List of N_time values to test, e.g. [30, 60, 120, 240].
            Larger N means smaller time step.
        greek : {"price", "delta", "gamma", "vega", "theta_annual", "theta_daily"}
            Quantity to analyse. For Greeks we call `calculate_greeks(N)`,
            which uses the same one-sided Vega bump as Front Arena.
        use_dt : bool
            If True, use dt = T_disc / N as the step size on the x-axis.
            If False, use h = 1/N.

        Returns
        -------
        dict
            {
              "order": p,         # estimated convergence order
              "coeff": c,         # intercept in log(error) ≈ p log(h) + c
              "r2": r2,           # goodness-of-fit R^2
              "Ns": [...],        # N_time values used
              "h_vals": [...],    # step sizes (dt or 1/N)
              "errors": [...],    # absolute errors vs finest grid
              "ref_value": float, # value on finest grid
              "values": [...],    # raw values at each N (same order as Ns)
            }
        Notes
        -----
        * For Crank–Nicolson price/delta/gamma you expect p ≈ 2 (O(dt^2)).
        * For one-sided Vega you typically see p ≈ 1–1.5, because the
          Vega itself is a first-order FD in volatility and amplifies PDE noise.
        """

        # ----- 0. Prepare and sort N -----
        Ns = sorted({int(N) for N in time_steps_list if int(N) > 0})
        if len(Ns) < 3:
            raise ValueError("Provide at least three different time step counts.")

        # Time horizon for dt (same as your engine uses)
        T_disc = float(self.time_to_discount)
        if use_dt:
            h_vals = [T_disc / N for N in Ns]
        else:
            h_vals = [1.0 / N for N in Ns]

        # ----- 1. Reference on finest grid -----
        N_ref = max(Ns)

        greek_key_map = {
            "delta": "Delta",
            "gamma": "Gamma",
            "vega": "Vega",
            "theta_annual": "Theta (Annual)",
            "theta_daily": "Theta (Daily)",
        }

        if greek.lower() == "price":
            ref_value = float(self._price_once(N_ref))
        else:
            key = greek_key_map.get(greek.lower())
            if key is None:
                raise ValueError(f"Unsupported greek name: {greek}")
            ref_value = float(self.calculate_greeks(N_ref)[key])

        # ----- 2. Compute values and errors for each N -----
        values = []
        errors = []

        for N in Ns:
            if greek.lower() == "price":
                val = float(self._price_once(N))
            else:
                val = float(self.calculate_greeks(N)[key])

            values.append(val)
            errors.append(abs(val - ref_value))

        # Filter out any zero errors (cannot take log)
        pairs = [(h, e) for (h, e) in zip(h_vals, errors) if e > 0.0]
        if len(pairs) < 2:
            raise RuntimeError("Not enough non-zero errors to estimate order.")

        h_vals, errors = map(np.array, zip(*pairs))

        logh = np.log(h_vals)
        loge = np.log(errors)

        # ----- 3. Least-squares fit: log(error) = p log(h) + c -----
        x = logh
        y = loge
        x_mean = x.mean()
        y_mean = y.mean()

        Sxx = np.sum((x - x_mean) ** 2)
        Sxy = np.sum((x - x_mean) * (y - y_mean))

        p = Sxy / Sxx
        c = y_mean - p * x_mean

        # Goodness-of-fit R^2
        y_hat = p * x + c
        SS_res = np.sum((y - y_hat) ** 2)
        SS_tot = np.sum((y - y_mean) ** 2)
        r2 = 1.0 - SS_res / SS_tot if SS_tot > 0.0 else 1.0

        return {
            "order": float(p),
            "coeff": float(c),
            "r2": float(r2),
            "Ns": Ns,
            "h_vals": list(h_vals),
            "errors": list(errors),
            "ref_value": ref_value,
            "values": values,
        }

    def expected_fd_error_at_N(
        self,
        result_dict: dict,
        N_target: int,
        use_dt: bool = True,
    ) -> float:
        """
        Using the regression log(error) ≈ p log(h) + c returned by
        `compute_empirical_order`, predict the FD truncation error
        at a coarser grid with N_target time steps.
        """
        p = result_dict["order"]
        c = result_dict["coeff"]

        # We only care about h ∝ 1/N; absolute T_disc cancels in the slope.
        if use_dt:
            h = 1.0 / float(N_target)
        else:
            h = 1.0 / float(N_target)

        log_err = p * np.log(h) + c
        return float(np.exp(log_err))

    def fa_vs_validation_vega_diagnostic(
        self,
        time_steps_list,
        N_fa: int = 30,
        vega_fa: float | None = None,
        greek: str = "vega",
    ) -> dict:
        """
        End-to-end diagnostic for FD vs Front Arena:

        1. Compute empirical order for the chosen quantity (default: Vega).
        2. Predict FD truncation error at FA-like grid (N_fa, e.g. 30 steps).
        3. If vega_fa is provided, compare predicted error to observed
           FA-vs-validation difference and print a conclusion.

        Returns a dict with key stats so the main script can log / report.
        """

        print("\n========== FD ORDER-OF-ACCURACY DIAGNOSTIC ==========\n")

        # Step 1 – empirical order
        result = self.compute_empirical_order(
            time_steps_list=time_steps_list,
            greek=greek,
            use_dt=True,
        )
        p = result["order"]
        r2 = result["r2"]
        ref_val = result["ref_value"]
        N_ref = max(result["Ns"])

        print(f"Quantity      : {greek}")
        print(f"Empirical p   : {p:.3f}")
        print(f"R^2 (fit)     : {r2:.4f}")
        print(f"Reference N   : {N_ref}")
        print(f"Reference {greek}: {ref_val:.8f}\n")

        # Step 2 – predicted truncation error at FA grid
        predicted_error = self.expected_fd_error_at_N(
            result_dict=result,
            N_target=N_fa,
            use_dt=True,
        )
        print(f"Predicted FD truncation error at N={N_fa}: {predicted_error:.8f}")

        fa_val = vega_fa
        obs_diff = None

        # Step 3 – compare to FA if provided
        if fa_val is not None:
            obs_diff = abs(fa_val - ref_val)
            print(f"Observed FA vs validation difference    : {obs_diff:.8f}")

            # Simple consistency check with a 1.5× buffer
            if obs_diff <= 1.5 * predicted_error:
                print(
                    "\nConclusion: ✔ The observed discrepancy is CONSISTENT "
                    "with expected FD truncation error (one-sided bump).\n"
                )
            else:
                print(
                    "\nConclusion: ✘ The discrepancy EXCEEDS the expected FD "
                    "truncation error; investigate further (e.g. setup, inputs, "
                    "dividend modelling).\n"
                )

        return {
            "empirical_order": p,
            "r2": r2,
            "predicted_error_at_N_fa": predicted_error,
            "reference_value": ref_val,
            "N_ref": N_ref,
            "N_fa": N_fa,
            "fa_value": fa_val,
            "observed_diff": obs_diff,
        }