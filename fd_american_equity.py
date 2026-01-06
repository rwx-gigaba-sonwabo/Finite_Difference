"""
Finite-difference American option pricer.

This module implements an American vanilla option pricer using a
Crank–Nicolson finite-difference scheme in log-space, with discrete
dividends handled via explicit jumps and Ikonen–Toivanen operator
splitting for early exercise. It is designed to be used as a
production-quality validation/pricing component.

Key features (aligned with FIS documentation and the referenced
discrete-dividend literature):

- Underlying follows Black–Scholes dynamics with no continuous
  dividend yield (q = 0) inside the PDE. Discrete cash dividends are
  handled as explicit jumps in the underlying price at ex-div dates.
- PDE is solved in log S with a uniform grid and Crank–Nicolson
  time-stepping, combined with Rannacher smoothing for stability
  near payoff/discontinuity kinks.
- Ikonen–Toivanen operator splitting enforces the American
  early-exercise constraint efficiently.
- American calls can exercise optimally at ex-dividend dates by
  comparing continuation vs intrinsic values.
- Greeks (price, delta, gamma, vega, theta) are computed from the
  grid via interpolation, local cubic fits, and bump-and-revalue
  with optional Richardson extrapolation in time.
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd  # type: ignore
from workalendar.africa import SouthAfrica


OptionType = Literal["call", "put"]


class AmericanFDMPricer:
    """
    American vanilla option pricer using a Crank–Nicolson finite-difference
    scheme in log S, with discrete dividends handled via explicit jumps and
    Ikonen–Toivanen operator splitting for early exercise.

    Summary of modelling choices:

    * Underlying follows Black–Scholes dynamics under the risk-neutral
      measure, with no continuous dividend yield (q = 0) in the PDE.
    * Discrete cash dividends are treated by splitting the time domain
      between ex-div dates. At each ex-div date t_d with cash D:

          V(t_d-, S) = V(t_d+, S - D),

      where V(t_d+, ·) is interpolated via a natural cubic spline.

    * For American calls only, potential early exercise at ex-div is
      captured via:

          V(t_d-, S) = max(V(t_d+, S - D), payoff(S)).

    * Rannacher time-stepping (a few fully implicit steps) is applied
      at expiry and restarted at each dividend for calls, to damp
      oscillations near discontinuities in the payoff/boundary.
    * American puts do not apply additional dividend smoothing, in line
      with FIS’ comment that the boundary condition is not discontinuous
      in the same way.
    * Delta and gamma are obtained from a local cubic fit in spot.
    * Vega is computed via symmetric volatility bump with optional
      Richardson extrapolation in time.
    * Theta is obtained from the Black–Scholes PDE identity at spot.
    """

    # --------------------------------------------------------------------- #
    # Constructor / setup                                                   #
    # --------------------------------------------------------------------- #

    def __init__(
        self,
        spot: float,
        strike: float,
        valuation_date: _dt.date,
        maturity_date: _dt.date,
        sigma: float,
        option_type: OptionType,
        discount_curve: pd.DataFrame,
        forward_curve: Optional[pd.DataFrame] = None,
        dividend_schedule: Optional[List[Tuple[_dt.date, float]]] = None,
        trade_id: Optional[int] = None,
        direction: str = "long",
        quantity: int = 1,
        contract_multiplier: float = 1.0,
        underlying_spot_days: int = 0,
        option_days: int = 0,
        option_settlement_days: int = 0,
        day_count: str = "ACT/365",
        grid_type: str = "uniform",
        num_space_nodes: int = 400,
        num_time_steps: int = 400,
        rannacher_steps: int = 2,
        s_max_mult: float = 4.5,
    ) -> None:
        """
        Initialise the finite-difference pricer.

        Parameters
        ----------
        spot :
            Current spot price of the underlying.
        strike :
            Option strike price.
        valuation_date :
            Valuation date (t0) of the trade.
        maturity_date :
            Option maturity date.
        sigma :
            Black–Scholes volatility (per annum, in decimal).
        option_type :
            "call" or "put".
        discount_curve :
            DataFrame with at least columns ["Date", "NACA"], representing
            nominal annually compounded (NACA) discount rates.
        forward_curve :
            Forward curve for the underlying (same format as discount_curve);
            if None, discount curve is used for carry as well.
        dividend_schedule :
            List of (ex_div_date, cash_amount) pairs.
        trade_id :
            Optional identifier for logging / audit.
        direction :
            "long" or "short"; sign is not used in core pricing but may be
            useful for reporting.
        quantity :
            Number of contracts.
        contract_multiplier :
            Contract multiplier to scale unit price to monetary amount.
        underlying_spot_days :
            Spot lag (business days) for the underlying.
        option_days :
            Spot lag (business days) for the option.
        option_settlement_days :
            Settlement lag (business days) for the option cashflows.
        day_count :
            Day-count convention (e.g. "ACT/365").
        grid_type :
            Type of spatial grid ("uniform" currently).
        num_space_nodes :
            Number of spatial nodes (in S) minus one. The actual grid uses
            num_space_nodes + 1 points.
        num_time_steps :
            Number of time steps in the base grid.
        rannacher_steps :
            Number of fully implicit (Euler backward) steps at the start
            of each segment (for Rannacher smoothing).
        s_max_mult :
            Scaling factor controlling the width of the log S band.
        """
        if spot <= 0.0 or strike <= 0.0 or sigma <= 0.0:
            raise ValueError("spot, strike and sigma must be positive.")
        if maturity_date <= valuation_date:
            raise ValueError("maturity_date must be after valuation_date.")

        # Basic instrument attributes
        self.spot = float(spot)
        self.strike = float(strike)
        self.valuation_date = valuation_date
        self.maturity_date = maturity_date
        self.sigma = float(sigma)

        self.option_type: str = option_type.lower()
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")

        # Curves & dividends
        self.discount_curve_df = discount_curve.copy()
        self.forward_curve_df = (
            forward_curve.copy() if forward_curve is not None else None
        )
        self.dividend_schedule = sorted(dividend_schedule or [], key=lambda x: x[0])

        # Trade meta-data
        self.trade_id = trade_id
        self.direction = direction
        self.quantity = int(quantity)
        self.contract_multiplier = float(contract_multiplier)

        # Calendar and lag settings
        self.calendar = SouthAfrica()
        self.underlying_spot_days = int(underlying_spot_days)
        self.option_days = int(option_days)
        self.option_settlement_days = int(option_settlement_days)

        # Day-count handling
        self.day_count = day_count.upper().replace("F", "")
        self._year_denominator = self._infer_denominator(self.day_count)

        self.grid_type = grid_type.lower()

        # Carry / discount dates
        self.carry_start_date = self.calendar.add_working_days(
            self.valuation_date, self.underlying_spot_days
        )
        self.carry_end_date = self.calendar.add_working_days(
            self.maturity_date, self.underlying_spot_days
        )

        self.discount_start_date = self.calendar.add_working_days(
            self.valuation_date, self.option_days
        )
        self.discount_end_date = self.calendar.add_working_days(
            self.maturity_date, self.option_settlement_days
        )

        # Times to key dates
        self.time_to_expiry = self._year_fraction(
            self.valuation_date, self.maturity_date
        )
        self.time_to_carry = self._year_fraction(
            self.carry_start_date, self.carry_end_date
        )
        self.time_to_discount = self._year_fraction(
            self.discount_start_date, self.discount_end_date
        )

        if self.time_to_expiry <= 0.0:
            raise ValueError("time_to_expiry must be positive.")

        # Rates (nominal annually compounded, transformed to NACC for PDE)
        self.discount_rate_nacc = self.get_forward_nacc_rate(
            self.discount_start_date, self.discount_end_date
        )

        if self.forward_curve_df is not None:
            self.carry_rate_nacc = self.get_forward_nacc_rate(
                self.carry_start_date, self.carry_end_date
            )
        else:
            self.carry_rate_nacc = self.discount_rate_nacc

        # Discrete-dividend model: q = 0 inside PDE, dividends handled by jumps
        self.div_yield_nacc = 0.0

        # Grid controls
        self.num_space_nodes = max(int(num_space_nodes), 3)
        self.num_time_steps = max(int(num_time_steps), 4)
        self.rannacher_steps = max(int(rannacher_steps), 0)
        self.s_max_mult = float(s_max_mult)

        # Critical level snapping (spot, strike)
        self.snap_spot_to_grid: bool = True
        self.snap_strike_to_grid: bool = True
        self.spot_grid_index: Optional[int] = None
        self.spot_snapped: Optional[float] = None
        self.strike_grid_index: Optional[int] = None
        self.strike_snapped: Optional[float] = None

        # Grid state
        self.s_nodes: List[float] = []
        self.x_nodes: List[float] = []
        self._S_min: float = 0.0
        self._S_max: float = 0.0
        self._dx: float = 0.0

    # --------------------------------------------------------------------- #
    # Day count and curve utilities                                        #
    # --------------------------------------------------------------------- #

    def _infer_denominator(self, day_count: str) -> int:
        """Infer year denominator from a day-count convention string."""
        if day_count in ("ACT/365", "ACT/365F"):
            return 365
        if day_count in ("ACT/360", "ACT/364"):
            return 360 if day_count == "ACT/360" else 364
        if day_count in ("30/360", "BOND", "US30/360"):
            return 360
        # Fallback default
        return 365

    def _year_fraction(self, start_date: _dt.date, end_date: _dt.date) -> float:
        """Compute year fraction between two dates under self.day_count."""
        if end_date <= start_date:
            return 0.0

        if self.day_count in ("ACT/365", "ACT/365F", "ACT/360", "ACT/364"):
            days = (end_date - start_date).days
            return days / float(self._year_denominator)

        if self.day_count in ("30/360", "BOND", "US30/360"):
            y1, m1, d1 = start_date.year, start_date.month, start_date.day
            y2, m2, d2 = end_date.year, end_date.month, end_date.day
            d1 = min(d1, 30)
            if d1 == 30:
                d2 = min(d2, 30)
            days = (y2 - y1) * 360 + (m2 - m1) * 30 + (d2 - d1)
            return days / 360.0

        # Reasonable default: ACT/365
        days = (end_date - start_date).days
        return days / 365.0

    def get_discount_factor(self, lookup_date: _dt.date) -> float:
        """
        Return discount factor from valuation_date to lookup_date.

        The input discount_curve_df is assumed to hold NACA rates per
        calendar date; these are transformed into discount factors via
        the selected day-count convention.
        """
        iso = lookup_date.isoformat()
        row = self.discount_curve_df[self.discount_curve_df["Date"] == iso]
        if row.empty:
            raise ValueError(f"Discount factor not found for date: {iso}")

        naca = float(row["NACA"].values[0])
        tau = self._year_fraction(self.valuation_date, lookup_date)
        return (1.0 + naca) ** (-tau)

    def get_forward_nacc_rate(
        self,
        start_date: _dt.date,
        end_date: _dt.date,
    ) -> float:
        """
        Compute continuously compounded forward rate between two dates.

        This uses discount factors from the input curve and converts the
        forward rate into a NACC-equivalent rate for use in the PDE.
        """
        df_far = self.get_discount_factor(end_date)
        df_near = self.get_discount_factor(start_date)
        tau = self._year_fraction(start_date, end_date)
        return -math.log(df_far / df_near) / max(1e-12, tau)

    # --------------------------------------------------------------------- #
    # Grid construction in log S                                           #
    # --------------------------------------------------------------------- #

    def _configure_grid(self) -> None:
        """Determine S_min and S_max for the log-space grid."""
        T = self.time_to_expiry
        sig = self.sigma

        s_low = min(self.spot, self.strike)
        s_high = max(self.spot, self.strike)
        s_c = math.sqrt(max(s_low * s_high, 1e-12))

        band = self.s_max_mult * sig * math.sqrt(max(T, 1e-12))
        x_c = math.log(s_c)
        x_min = x_c - 0.5 * band
        x_max = x_c + 0.5 * band

        s_min = math.exp(x_min)
        s_max = math.exp(x_max)

        s_min = min(s_min, 0.5 * s_low)
        s_max = max(s_max, 2.0 * s_high)

        self._S_min = max(s_min, 1e-8)
        self._S_max = s_max

    def _build_log_grid(self) -> float:
        """
        Construct log S grid and snap critical levels (spot, strike).

        Returns
        -------
        float
            Log-space step size dx.
        """
        self._configure_grid()
        x_min = math.log(self._S_min)
        x_max = math.log(self._S_max)

        n = self.num_space_nodes
        dx = (x_max - x_min) / float(n)

        self.x_nodes = [x_min + i * dx for i in range(n + 1)]
        self.s_nodes = [math.exp(x) for x in self.x_nodes]
        self._dx = dx

        self._snap_critical_levels_to_grid()
        return dx

    def _snap_critical_levels_to_grid(self) -> None:
        """Snap spot and strike to nearest grid nodes (if enabled)."""
        s_nodes = self.s_nodes
        n = len(s_nodes)
        if n == 0:
            return

        if self.snap_spot_to_grid:
            idx_spot = min(range(n), key=lambda i: abs(s_nodes[i] - self.spot))
            self.spot_grid_index = idx_spot
            self.spot_snapped = s_nodes[idx_spot]
        else:
            self.spot_grid_index = None
            self.spot_snapped = None

        if self.snap_strike_to_grid:
            idx_strike = min(range(n), key=lambda i: abs(s_nodes[i] - self.strike))
            self.strike_grid_index = idx_strike
            self.strike_snapped = s_nodes[idx_strike]
        else:
            self.strike_grid_index = None
            self.strike_snapped = None

    # --------------------------------------------------------------------- #
    # Payoff and boundary conditions                                       #
    # --------------------------------------------------------------------- #

    def _strike_for_pde(self) -> float:
        """Return strike used inside PDE (snapped to grid if configured)."""
        if self.snap_strike_to_grid and self.strike_snapped is not None:
            return self.strike_snapped
        return self.strike

    def _intrinsic_payoff(self, spot: float) -> float:
        """Intrinsic payoff at a given spot level."""
        strike = self._strike_for_pde()
        if self.option_type == "call":
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)

    def _terminal_payoff(self) -> List[float]:
        """Payoff vector at expiry over all S grid nodes."""
        return [self._intrinsic_payoff(s) for s in self.s_nodes]

    def _boundary_values(self, tau: float) -> Tuple[float, float]:
        """
        Compute boundary values at S_min and S_max at time-to-maturity tau.

        The far-field behaviour is based on the standard Black–Scholes
        asymptotics for calls and puts.
        """
        s_max = self.s_nodes[-1]
        r = self.discount_rate_nacc
        b = self.carry_rate_nacc
        strike = self._strike_for_pde()

        if self.option_type == "call":
            v_min = 0.0
            v_max = s_max * math.exp((b - r) * tau) - strike * math.exp(-r * tau)
        else:
            v_min = strike * math.exp(-r * tau)
            v_max = 0.0
        return v_min, v_max

    # --------------------------------------------------------------------- #
    # Dividend times (tau = time-to-maturity from valuation)               #
    # --------------------------------------------------------------------- #

    def _div_times_tau(self) -> List[Tuple[float, float]]:
        """
        Return list of (tau_div, cash_amount) for relevant dividends.

        tau_div is the time-to-maturity measured from valuation_date
        (tau = time_to_expiry - t_rel), sorted in ascending tau.
        """
        if not self.dividend_schedule:
            return []

        result: List[Tuple[float, float]] = []
        for pay_date, amount in self.dividend_schedule:
            if self.valuation_date < pay_date < self.maturity_date:
                t_rel = self._year_fraction(self.valuation_date, pay_date)
                if 0.0 < t_rel < self.time_to_expiry:
                    tau_div = self.time_to_expiry - t_rel
                    result.append((tau_div, float(amount)))

        result.sort(key=lambda x: x[0])
        return result

    # --------------------------------------------------------------------- #
    # Natural cubic spline (used at dividend jumps)                        #
    # --------------------------------------------------------------------- #

    @staticmethod
    def _build_natural_cubic_spline(
        x: List[float],
        y: List[float],
    ):
        """
        Build a natural cubic spline interpolant S(x) through knot points
        (x[i], y[i]) and return a callable f(x_query).
        """
        n = len(x)
        if n < 2:
            raise ValueError("Need at least two points for spline.")

        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        h = np.diff(x_arr)
        if np.any(h <= 0.0):
            raise ValueError("x must be strictly increasing.")

        alpha = np.zeros(n)
        for i in range(1, n - 1):
            alpha[i] = (
                3.0 / h[i] * (y_arr[i + 1] - y_arr[i])
                - 3.0 / h[i - 1] * (y_arr[i] - y_arr[i - 1])
            )

        l = np.ones(n)
        mu = np.zeros(n)
        z = np.zeros(n)

        for i in range(1, n - 1):
            l[i] = 2.0 * (x_arr[i + 1] - x_arr[i - 1]) - h[i - 1] * mu[i - 1]
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

        c = np.zeros(n)
        b = np.zeros(n - 1)
        d = np.zeros(n - 1)

        for j in range(n - 2, -1, -1):
            c[j] = z[j] - mu[j] * c[j + 1]
            b[j] = (
                (y_arr[j + 1] - y_arr[j]) / h[j]
                - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
            )
            d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])

        a = y_arr[:-1]

        def spline_eval(x_query: float) -> float:
            """Evaluate spline at a single x_query."""
            if x_query <= x_arr[0]:
                j = 0
            elif x_query >= x_arr[-1]:
                j = n - 2
            else:
                lo, hi = 0, n - 1
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if x_query < x_arr[mid]:
                        hi = mid
                    else:
                        lo = mid
                j = lo

            dx_local = x_query - x_arr[j]
            return float(
                a[j]
                + b[j] * dx_local
                + c[j] * dx_local * dx_local
                + d[j] * dx_local * dx_local * dx_local
            )

        return spline_eval

    # --------------------------------------------------------------------- #
    # Segment solver: CN + Rannacher + Ikonen–Toivanen                     #
    # --------------------------------------------------------------------- #

    def _solve_segment(
        self,
        v_init: List[float],
        tau_start: float,
        tau_end: float,
        n_steps: int,
        restart_rannacher: bool,
    ) -> List[float]:
        """
        Solve the PDE backwards in tau over a single segment.

        Parameters
        ----------
        v_init :
            Initial value vector at tau_start (size = number of S nodes).
        tau_start :
            Starting time-to-maturity for the segment.
        tau_end :
            Ending time-to-maturity for the segment (tau_end > tau_start).
        n_steps :
            Number of time steps to use in this segment.
        restart_rannacher :
            Whether to apply Rannacher smoothing (fully implicit steps) at
            the beginning of this segment.

        Returns
        -------
        List[float]
            Value vector at tau_end.
        """
        if n_steps < 1:
            return v_init

        dx = self._dx
        n_space = len(self.s_nodes) - 1
        if n_space < 2:
            raise RuntimeError("Spatial grid too coarse.")

        dt = (tau_end - tau_start) / float(n_steps)

        sigma = self.sigma
        sigma_sq = sigma * sigma
        r = self.discount_rate_nacc
        b = self.carry_rate_nacc
        q = 0.0  # discrete-dividend model: no continuous yield

        mu_x = (b - q) - 0.5 * sigma_sq

        alpha = 0.5 * sigma_sq / (dx * dx)
        beta_adv = mu_x / (2.0 * dx)

        a_coef = alpha - beta_adv
        c_coef = alpha + beta_adv
        b_coef = -2.0 * alpha - r

        def build_matrices(theta: float) -> Tuple[float, float, float, float, float, float]:
            """Build scalar tri-diagonal coefficients for given theta."""
            a_l = -theta * dt * a_coef
            a_c = 1.0 - theta * dt * b_coef
            a_u = -theta * dt * c_coef

            b_l = (1.0 - theta) * dt * a_coef
            b_c = 1.0 + (1.0 - theta) * dt * b_coef
            b_u = (1.0 - theta) * dt * c_coef
            return a_l, a_c, a_u, b_l, b_c, b_u

        def solve_tridiagonal(
            a_l: float,
            a_c: float,
            a_u: float,
            rhs: List[float],
        ) -> List[float]:
            """
            Solve tri-diagonal system A x = rhs with constant diagonals
            (a_l, a_c, a_u) using Thomas algorithm.
            """
            n_local = len(rhs)
            c_prime = [0.0] * n_local
            d_prime = [0.0] * n_local

            denom = a_c
            c_prime[0] = a_u / denom
            d_prime[0] = rhs[0] / denom

            for i in range(1, n_local):
                denom = a_c - a_l * c_prime[i - 1]
                if i < n_local - 1:
                    c_prime[i] = a_u / denom
                d_prime[i] = (rhs[i] - a_l * d_prime[i - 1]) / denom

            x_sol = [0.0] * n_local
            x_sol[-1] = d_prime[-1]
            for i in range(n_local - 2, -1, -1):
                x_sol[i] = d_prime[i] - c_prime[i] * x_sol[i + 1]
            return x_sol

        # Initial value vector for this segment
        v_values = v_init[:]

        payoff_full = [self._intrinsic_payoff(s) for s in self.s_nodes]
        payoff_int = payoff_full[1:-1]  # interior points

        lambda_int = [0.0] * (n_space - 1)
        base_rannacher = self.rannacher_steps if restart_rannacher else 0

        tau = tau_start
        for step_index in range(n_steps):
            tau_next = tau + dt
            theta = 1.0 if step_index < base_rannacher else 0.5

            (
                a_l,
                a_c,
                a_u,
                b_l,
                b_c,
                b_u,
            ) = build_matrices(theta)

            v_min, v_max = self._boundary_values(tau_next)

            # Build RHS for interior nodes
            rhs = [0.0] * (n_space - 1)
            for j in range(1, n_space):
                v_jm1 = v_values[j - 1]
                v_j = v_values[j]
                v_jp1 = v_values[j + 1]
                rhs[j - 1] = (
                    b_l * v_jm1
                    + b_c * v_j
                    + b_u * v_jp1
                    + dt * lambda_int[j - 1]
                )

            # Incorporate boundary contributions
            rhs[0] -= a_l * v_min
            rhs[-1] -= a_u * v_max

            # Unconstrained PDE step (European)
            tilde_int = solve_tridiagonal(a_l, a_c, a_u, rhs)

            # Ikonen–Toivanen projection (early exercise)
            new_lambda_int = [0.0] * (n_space - 1)
            new_v_int = [0.0] * (n_space - 1)

            for k in range(n_space - 1):
                tilde_vk = tilde_int[k]
                payoff_k = payoff_int[k]
                lambda_old = lambda_int[k]

                v_candidate = tilde_vk - dt * lambda_old
                v_new = payoff_k if payoff_k > v_candidate else v_candidate

                lambda_new = lambda_old + (payoff_k - tilde_vk) / dt
                if lambda_new < 0.0:
                    lambda_new = 0.0

                new_lambda_int[k] = lambda_new
                new_v_int[k] = v_new

            # Update full grid with new interior and boundary values
            v_values[0] = v_min
            v_values[-1] = v_max
            v_values[1:-1] = new_v_int
            lambda_int = new_lambda_int
            tau = tau_next

        return v_values

    # --------------------------------------------------------------------- #
    # Dividend jump mapping                                                 #
    # --------------------------------------------------------------------- #

    def _apply_dividend_jump(
        self,
        v_after: List[float],
        cash_div: float,
    ) -> List[float]:
        """
        Apply ex-dividend jump mapping for a cash dividend.

        For ex-dividend date t_d with dividend D, forward in time:

            S(t_d+) = S(t_d-) - D.

        Backwards in time, this corresponds to:

            V(t_d-, S) = V(t_d+, S - D)   (continuation value).

        For American calls, we additionally consider the early exercise
        opportunity just before ex-div:

            V(t_d-, S) = max(V(t_d+, S - D), payoff(S)).
        """
        s_nodes = self.s_nodes
        spline = self._build_natural_cubic_spline(s_nodes, v_after)

        v_new: List[float] = []
        for s_val in s_nodes:
            s_minus = s_val - cash_div
            if s_minus <= s_nodes[0]:
                cont_val = v_after[0]
            elif s_minus >= s_nodes[-1]:
                cont_val = v_after[-1]
            else:
                cont_val = spline(s_minus)

            if self.option_type == "call":
                exercise_val = self._intrinsic_payoff(s_val)
                v_new.append(max(cont_val, exercise_val))
            else:
                v_new.append(cont_val)

        return v_new

    # --------------------------------------------------------------------- #
    # Global solver with time segments and dividends                        #
    # --------------------------------------------------------------------- #

    def _solve_grid(self, n_time: Optional[int] = None) -> List[float]:
        """
        Solve for the option value on the full grid at valuation date.

        This method:

        * Builds the log S grid.
        * Initialises payoff at expiry.
        * Splits the time interval into segments between dividend dates.
        * Solves each segment with CN + IT, optionally restarting
          Rannacher at dividend boundaries.
        * Applies dividend jumps at each ex-div date.
        """
        # Build grid once
        self._build_log_grid()

        v_values = self._terminal_payoff()
        total_tau = self.time_to_expiry

        div_times = self._div_times_tau()  # sorted by ascending tau
        base_n = self.num_time_steps if n_time is None else int(n_time)
        base_dt = total_tau / float(base_n)

        # Segment endpoints in tau (0 = expiry, total_tau = valuation)
        tau_points = [0.0] + [tau for tau, _ in div_times] + [total_tau]
        n_segments = len(tau_points) - 1

        seg_lengths = [
            tau_points[i + 1] - tau_points[i] for i in range(n_segments)
        ]
        seg_steps: List[int] = []
        remaining_steps = base_n

        # Allocate integer steps to each segment, preserving total step
        # count and approximate proportionality to segment length.
        for seg_length in seg_lengths[:-1]:
            n_seg = max(1, int(round(seg_length / base_dt)))
            seg_steps.append(n_seg)
            remaining_steps -= n_seg
        seg_steps.append(max(1, remaining_steps))

        tau = 0.0
        for seg_idx in range(n_segments):
            tau_start = tau_points[seg_idx]
            tau_end = tau_points[seg_idx + 1]
            n_steps_seg = seg_steps[seg_idx]

            restart_rannacher = (
                seg_idx == 0 or (seg_idx > 0 and self.option_type == "call")
            )

            v_values = self._solve_segment(
                v_init=v_values,
                tau_start=tau_start,
                tau_end=tau_end,
                n_steps=n_steps_seg,
                restart_rannacher=restart_rannacher,
            )
            tau = tau_end

            # Apply dividend jump after segment if a dividend occurs here
            if seg_idx < len(div_times):
                _, cash_div = div_times[seg_idx]
                v_values = self._apply_dividend_jump(v_values, cash_div)

        return v_values

    # --------------------------------------------------------------------- #
    # Interpolation at spot and local cubic fit for delta / gamma           #
    # --------------------------------------------------------------------- #

    def _spot_for_interp(self) -> float:
        """Spot level used for interpolation (snapped if configured)."""
        if self.snap_spot_to_grid and self.spot_snapped is not None:
            return self.spot_snapped
        return self.spot

    def _interp_price(self, v_values: List[float]) -> float:
        """Linearly interpolate price at the effective spot."""
        s_nodes = self.s_nodes
        s0 = self._spot_for_interp()

        if s0 <= s_nodes[0]:
            return float(v_values[0])
        if s0 >= s_nodes[-1]:
            return float(v_values[-1])

        lo, hi = 0, len(s_nodes) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if s0 < s_nodes[mid]:
                hi = mid
            else:
                lo = mid

        weight = (s0 - s_nodes[lo]) / (s_nodes[hi] - s_nodes[lo])
        return float((1.0 - weight) * v_values[lo] + weight * v_values[hi])

    def _local_cubic_delta_gamma(
        self,
        v_values: List[float],
    ) -> Tuple[float, float]:
        """
        Compute delta and gamma via a local cubic polynomial fit around spot.

        A four-point stencil around the effective spot is used to fit a
        cubic polynomial in S; the first and second derivatives at the
        spot give delta and gamma respectively.
        """
        s_nodes = self.s_nodes
        s0 = self._spot_for_interp()
        n = len(s_nodes) - 1

        idx_near = min(range(n + 1), key=lambda k: abs(s_nodes[k] - s0))
        if idx_near < 1:
            idx_near = 1
        elif idx_near > n - 2:
            idx_near = n - 2

        idx = [idx_near - 1, idx_near, idx_near + 1, idx_near + 2]
        x_vals = np.array([s_nodes[j] for j in idx], dtype=float)
        y_vals = np.array([v_values[j] for j in idx], dtype=float)

        z = x_vals - s0
        design = np.vstack([z ** 3, z ** 2, z, np.ones_like(z)]).T
        a_coef, b_coef, c_coef, d_coef = np.linalg.solve(design, y_vals)

        delta = float(c_coef)
        gamma = float(2.0 * b_coef)
        return delta, gamma

    # --------------------------------------------------------------------- #
    # Public pricing and Greeks                                             #
    # --------------------------------------------------------------------- #

    def price_log(self, n_time: Optional[int] = None) -> float:
        """
        Price the American option using the current grid settings.

        Parameters
        ----------
        n_time :
            Optional override for the number of time steps.
        """
        v_values = self._solve_grid(n_time=n_time)
        return self._interp_price(v_values)

    def price_log2(
        self,
        apply_ko: bool = True,
        use_richardson: bool = True,
    ) -> float:
        """
        Price with optional Richardson extrapolation in time (N vs 2N).

        Parameters
        ----------
        apply_ko :
            Kept for API symmetry with a barrier pricer (ignored here).
        use_richardson :
            If True, apply Richardson extrapolation.

        Notes
        -----
        Behaviour is preserved exactly as in the original implementation:
        the second run uses ``2 * self.num_space_nodes`` as the time-step
        count, even though this is not obviously intended.
        """
        if not use_richardson:
            return self.price_log(n_time=self.num_time_steps)

        p_n = self.price_log(n_time=self.num_time_steps)
        p_2n = self.price_log(n_time=2 * self.num_space_nodes)
        return (4.0 * p_2n - p_n) / 3.0

    def _price_for_sigma(
        self,
        sigma: float,
        n_time: Optional[int] = None,
    ) -> float:
        """
        Helper to re-price with a different volatility (for vega bumps).

        The original sigma is restored after pricing.
        """
        original_sigma = self.sigma
        try:
            self.sigma = sigma
            return self.price_log(n_time=n_time)
        finally:
            self.sigma = original_sigma

    def greeks_log2(
        self,
        dv_sigma: float = 0.01,
        use_richardson: bool = True,
    ) -> Dict[str, float]:
        """
        Compute price and Greeks using the finite-difference engine.

        Parameters
        ----------
        dv_sigma :
            Volatility bump size used for vega (in absolute volatility
            units, e.g. 0.01 for 1 vol point).
        use_richardson :
            If True, use Richardson extrapolation for price, delta,
            gamma, and vega.

        Returns
        -------
        Dict[str, float]
            Dictionary with keys "price", "delta", "gamma", "vega", "theta".
        """
        # Base grid (N steps)
        v_n = self._solve_grid(n_time=self.num_time_steps)
        price_n = self._interp_price(v_n)
        delta_n, gamma_n = self._local_cubic_delta_gamma(v_n)

        if use_richardson:
            v_2n = self._solve_grid(n_time=2 * self.num_time_steps)
            price_2n = self._interp_price(v_2n)
            delta_2n, gamma_2n = self._local_cubic_delta_gamma(v_2n)

            price = (4.0 * price_2n - price_n) / 3.0
            delta = (4.0 * delta_2n - delta_n) / 3.0
            gamma = (4.0 * gamma_2n - gamma_n) / 3.0
        else:
            price = price_n
            delta = delta_n
            gamma = gamma_n

        # Vega: symmetric bump in sigma
        sigma0 = self.sigma
        h = dv_sigma

        if use_richardson:
            p_up_h = self._price_for_sigma(
                sigma0 + h,
                n_time=self.num_time_steps,
            )
            p_dn_h = self._price_for_sigma(
                sigma0 - h,
                n_time=self.num_time_steps,
            )
            first_diff_h = (p_up_h - p_dn_h) / (2.0 * h)

            p_up_2h = self._price_for_sigma(
                sigma0 + 2.0 * h,
                n_time=self.num_time_steps,
            )
            p_dn_2h = self._price_for_sigma(
                sigma0 - 2.0 * h,
                n_time=self.num_time_steps,
            )
            first_diff_2h = (p_up_2h - p_dn_2h) / (4.0 * h)

            d_v_d_sigma = (4.0 * first_diff_h - first_diff_2h) / 3.0
        else:
            p_up = self._price_for_sigma(
                sigma0 + h,
                n_time=self.num_time_steps,
            )
            p_dn = self._price_for_sigma(
                sigma0 - h,
                n_time=self.num_time_steps,
            )
            d_v_d_sigma = (p_up - p_dn) / (2.0 * h)

        # Vega expressed per 1% vol
        vega = d_v_d_sigma / 100.0

        # Theta from BS PDE at S0 (q = 0 in discrete-dividend model)
        r = self.discount_rate_nacc
        b = self.carry_rate_nacc
        q = 0.0
        s0 = self.spot

        theta = -(
            0.5 * sigma0 * sigma0 * s0 * s0 * gamma
            + (b - q) * s0 * delta
            - r * price
        )

        return {
            "price": float(price),
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "theta": float(theta),
        }

def _delta_gamma_bump_from_surface(
    self,
    v_values: List[float],
    rel_bump: float = 0.01,
) -> Tuple[float, float]:
    """
    Compute delta and gamma by bump-and-revalue in spot using the
    existing FD surface.

    Parameters
    ----------
    v_values :
        FD solution at valuation (size = len(self.s_nodes)).
    rel_bump :
        Relative bump size in spot, e.g. 0.01 for ±1% bumps.

    Returns
    -------
    (delta, gamma)
    """
    s0 = self._spot_for_interp()
    h = max(s0 * rel_bump, 1e-8)

    v_down = self._interp_price_at_spot(v_values, s0 - h)
    v_0 = self._interp_price_at_spot(v_values, s0)
    v_up = self._interp_price_at_spot(v_values, s0 + h)

    delta = (v_up - v_down) / (2.0 * h)
    gamma = (v_up - 2.0 * v_0 + v_down) / (h * h)
    return float(delta), float(gamma)

def _interp_price_at_spot(
    self,
    v_values: List[float],
    spot: float,
) -> float:
    """Interpolate the FD solution at an arbitrary spot."""
    s_nodes = self.s_nodes

    if spot <= s_nodes[0]:
        return float(v_values[0])
    if spot >= s_nodes[-1]:
        return float(v_values[-1])

    lo, hi = 0, len(s_nodes) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if spot < s_nodes[mid]:
            hi = mid
        else:
            lo = mid

    w = (spot - s_nodes[lo]) / (s_nodes[hi] - s_nodes[lo])
    return float((1.0 - w) * v_values[lo] + w * v_values[hi])
