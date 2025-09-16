# -*- coding: utf-8 -*-
"""
Discrete Barrier Option Pricer (Crank–Nicolson + Rannacher) with intuitive names.

Highlights
- PDE uses flat continuously-compounded (NACC) rates: r_flat, q_flat.
- Correct CN coefficients (½ in the drift term).
- Rannacher smoothing only near payoff/discrete cash dividends (NOT at barriers).
- Optional BGK continuity correction for discretely monitored barriers.
- Barrier-aware Greeks (one-sided near barrier, blended in next interval).

Author’s intent: readability for reviewers / model validation.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Literal, Dict, Any
import math
import datetime as dt

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None


BarrierType = Literal[
    "down-and-out",
    "up-and-out",
    "double-out",
    "down-and-in",
    "up-and-in",
    "double-in",
    "none",
]
OptionType = Literal["call", "put"]


class DiscreteBarrierFDMPricer:
    """
    Intuitive interface (names and structure):

    Core inputs
    ----------
    spot_price, strike_price : floats
    valuation_date, maturity_date : datetime.date
    option_type : "call" | "put"
    barrier_type : e.g. "up-and-out", "down-and-in", "none"
    barrier_lower, barrier_upper : floats or None
    monitoring_dates : list[date] (exclude valuation and expiry)

    Market data
    -----------
    discount_curve : DataFrame with columns ["Date"(YYYY-MM-DD), "NACA"]
    dividend_schedule : list[(pay_date, cash_amount)]
    r_flat, q_flat : optional NACC rates; if omitted, r comes from curve, q from PV(divs)

    Numerics
    --------
    num_space_nodes, num_time_steps
    rannacher_steps : how many initial EB steps (typically 2)
    restart_rannacher_at_barrier : False per FIS note
    use_bgk_correction : apply BGK barrier shift for discrete monitoring
    s_max_multiplier : controls far-field boundary S_max
    day_count : "ACT/365", "ACT/360", etc.
    """

    def __init__(
        self,
        *,
        # Trade / contract
        spot_price: float,
        strike_price: float,
        valuation_date: dt.date,
        maturity_date: dt.date,
        option_type: OptionType,
        barrier_type: BarrierType = "none",
        barrier_lower: Optional[float] = None,
        barrier_upper: Optional[float] = None,
        monitoring_dates: Optional[List[dt.date]] = None,
        # Market data
        discount_curve: Optional[Any] = None,             # DataFrame: ["Date","NACA"]
        dividend_schedule: Optional[List[Tuple[dt.date, float]]] = None,
        r_flat: Optional[float] = None,                   # NACC (continuous)
        q_flat: Optional[float] = None,                   # NACC (continuous)
        volatility: float = 0.20,                         # annualized
        # Numerics
        num_space_nodes: int = 400,
        num_time_steps: int = 400,
        rannacher_steps: int = 2,
        restart_rannacher_at_barrier: bool = False,       # per FIS note: NO restart at barrier
        s_max_multiplier: float = 4.0,
        use_bgk_correction: bool = False,
        use_richardson_extrapolation: bool = False,
        mollify_payoff_near_strike: bool = True,
        mollify_nodes_half_width: int = 2,
        # Day count
        day_count: str = "ACT/365",
        # Position scaling / reporting
        trade_id: str = "T-0001",
        direction: Literal["long", "short"] = "long",
        quantity: int = 1,
        contract_multiplier: float = 1.0,
    ) -> None:

        # ---- Validate inputs
        if any(x <= 0 for x in (spot_price, strike_price, volatility)):
            raise ValueError("spot_price, strike_price, volatility must be positive.")
        if maturity_date <= valuation_date:
            raise ValueError("maturity_date must be after valuation_date.")
        if option_type not in ("call", "put"):
            raise ValueError(f"Invalid option_type: {option_type}")
        if barrier_type not in (
            "down-and-out", "up-and-out", "double-out",
            "down-and-in", "up-and-in", "double-in", "none"
        ):
            raise ValueError(f"Invalid barrier_type: {barrier_type}")

        # ---- Store trade / market
        self.spot_price = float(spot_price)
        self.strike_price = float(strike_price)
        self.valuation_date = valuation_date
        self.maturity_date = maturity_date
        self.option_type = option_type
        self.barrier_type = barrier_type
        self.barrier_lower = barrier_lower
        self.barrier_upper = barrier_upper
        self.monitoring_dates = sorted(monitoring_dates or [])
        self.discount_curve = discount_curve.copy() if discount_curve is not None else None
        self.dividend_schedule = sorted(dividend_schedule or [], key=lambda x: x[0])
        self.volatility = float(volatility)

        # ---- Position & reporting
        self.trade_id = trade_id
        self.direction = direction
        self.quantity = int(quantity)
        self.contract_multiplier = float(contract_multiplier)

        # ---- Numerics
        self.num_space_nodes = int(num_space_nodes)
        self.num_time_steps = int(num_time_steps)
        self.rannacher_steps = int(rannacher_steps)
        self.restart_rannacher_at_barrier = bool(restart_rannacher_at_barrier)
        self.s_max_multiplier = float(s_max_multiplier)
        self.use_bgk_correction = bool(use_bgk_correction)
        self.use_richardson_extrapolation = bool(use_richardson_extrapolation)
        self.mollify_payoff_near_strike = bool(mollify_payoff_near_strike)
        self.mollify_nodes_half_width = int(mollify_nodes_half_width)

        # ---- Day count & tenor
        self.day_count = day_count.upper().replace("F", "")
        self._year_denom = self._infer_year_denominator(self.day_count)
        self.tenor_years = self._year_fraction(self.valuation_date, self.maturity_date)

        # ---- Normalize curve df if provided
        if self.discount_curve is not None:
            if pd is None:
                raise ImportError("pandas is required for DataFrame curves.")
            self.discount_curve = self._normalize_curve_df(self.discount_curve)

        # ---- Flat NACC r and q
        self.r_flat = float(r_flat) if r_flat is not None else self._derive_flat_r_from_curve()
        self.q_flat = float(q_flat) if q_flat is not None else self._derive_flat_q_from_dividends()

        # ---- Grids
        self.stock_grid = self._build_stock_price_grid()
        self.grid_spacing = self.stock_grid[1] - self.stock_grid[0]
        self.time_grid = [i * self.tenor_years / self.num_time_steps for i in range(self.num_time_steps + 1)]

    # ======================================================================
    # Day count / utilities
    # ======================================================================

    def _infer_year_denominator(self, day_count: str) -> int:
        if day_count in ("ACT/365", "ACT/365F"):
            return 365
        if day_count in ("ACT/360", "ACT/364"):
            return 360 if day_count == "ACT/360" else 364
        if day_count in ("30/360", "BOND", "US30/360"):
            return 360
        return 365

    def _year_fraction(self, start: dt.date, end: dt.date) -> float:
        if end <= start:
            return 0.0
        if self.day_count in ("ACT/365", "ACT/365F", "ACT/360", "ACT/364"):
            return (end - start).days / float(self._year_denom)
        if self.day_count in ("30/360", "BOND", "US30/360"):
            y1, m1, d1 = start.year, start.month, start.day
            y2, m2, d2 = end.year, end.month, end.day
            d1 = min(d1, 30)
            if d1 == 30:
                d2 = min(d2, 30)
            days = (y2 - y1) * 360 + (m2 - m1) * 30 + (d2 - d1)
            return days / 360.0
        return (end - start).days / 365.0

    @staticmethod
    def _normalize_curve_df(df: Any) -> Any:
        if "Date" not in df.columns or "NACA" not in df.columns:
            raise ValueError("Curve DataFrame must have columns: 'Date', 'NACA'.")
        if not pd.api.types.is_string_dtype(df["Date"]):
            df = df.copy()
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        return df

    # ======================================================================
    # Flat r and q derivation (NACC)
    # ======================================================================

    def _discount_factor_from_curve(self, date: dt.date) -> float:
        if self.discount_curve is None:
            raise ValueError("discount_curve is None; pass r_flat or a curve.")
        iso = date.isoformat()
        row = self.discount_curve[self.discount_curve["Date"] == iso]
        if row.empty:
            raise ValueError(f"Discount input missing for date: {iso}")
        naca = float(row["NACA"].values[0])  # nominal compounded annually
        tau = self._year_fraction(self.valuation_date, date)
        return (1.0 + naca) ** (-tau)

    def _derive_flat_r_from_curve(self) -> float:
        if self.discount_curve is None or self.tenor_years <= 0.0:
            return 0.0
        df_T = self._discount_factor_from_curve(self.maturity_date)
        return -math.log(df_T) / self.tenor_years  # NACC

    def _present_value_of_dividends(self) -> float:
        pv = 0.0
        for pay_date, cash_amt in self.dividend_schedule:
            if self.valuation_date < pay_date <= self.maturity_date:
                df = self._discount_factor_from_curve(pay_date)
                pv += cash_amt * df
        return pv

    def _derive_flat_q_from_dividends(self) -> float:
        if not self.dividend_schedule or self.tenor_years <= 0.0:
            return 0.0
        pv_divs = self._present_value_of_dividends()
        if pv_divs <= 0.0:
            return 0.0
        if pv_divs >= self.spot_price:
            raise ValueError("PV(dividends) >= spot_price.")
        return -math.log((self.spot_price - pv_divs) / self.spot_price) / self.tenor_years

    # ======================================================================
    # Space grid and payoff
    # ======================================================================

    def _build_stock_price_grid(self) -> List[float]:
        """
        Uniform price grid [0, S_max]. Snap strike and barriers to nodes for accuracy.
        """
        reference_points = [self.spot_price, self.strike_price]
        if self.barrier_lower is not None:
            reference_points.append(self.barrier_lower)
        if self.barrier_upper is not None:
            reference_points.append(self.barrier_upper)

        s_star = max(reference_points)
        s_max = self.s_max_multiplier * s_star * math.exp(self.volatility * math.sqrt(max(self.tenor_years, 1e-12)))
        s_min = 0.0

        n = max(200, self.num_space_nodes)
        dS = (s_max - s_min) / n
        grid = [s_min + i * dS for i in range(n + 1)]

        # Snap key levels to nearest grid nodes
        def snap_to_grid(x: Optional[float]) -> None:
            if x is None:
                return
            j = min(range(len(grid)), key=lambda k: abs(grid[k] - x))
            grid[j] = x

        snap_to_grid(self.strike_price)
        snap_to_grid(self.barrier_lower)
        snap_to_grid(self.barrier_upper)
        return grid

    def _european_payoff(self, s: float) -> float:
        if self.option_type == "call":
            return max(s - self.strike_price, 0.0)
        return max(self.strike_price - s, 0.0)

    def _payoff_array_mollified(self) -> List[float]:
        """
        Quadratic smoothing around strike to reduce oscillations (optional).
        """
        payoff = [self._european_payoff(S) for S in self.stock_grid]
        if not self.mollify_payoff_near_strike or self.mollify_nodes_half_width <= 0:
            return payoff

        strike_idx = min(range(len(self.stock_grid)), key=lambda i: abs(self.stock_grid[i] - self.strike_price))
        m = self.mollify_nodes_half_width
        left = max(0, strike_idx - m)
        right = min(len(self.stock_grid) - 1, strike_idx + m)
        S_left, V_left = self.stock_grid[left], payoff[left]
        S_right, V_right = self.stock_grid[right], payoff[right]

        # Simple quadratic with zero left slope matching endpoints
        slope_left = 0.0
        a = (V_right - V_left - slope_left * (S_right - S_left)) / ((S_right - S_left) ** 2) if S_right != S_left else 0.0
        b = slope_left
        for i in range(left, right + 1):
            s = self.stock_grid[i]
            payoff[i] = a * (s - S_left) ** 2 + b * (s - S_left) + V_left
        return payoff

    # ======================================================================
    # Boundary conditions
    # ======================================================================

    def _boundary_value_at_s_min(self, time_to_maturity: float) -> float:
        if self.option_type == "call":
            return 0.0
        return self.strike_price * math.exp(-self.r_flat * time_to_maturity)

    def _boundary_value_at_s_max(self, s_max: float, time_to_maturity: float) -> float:
        if self.option_type == "call":
            intrinsic = s_max * math.exp(-self.q_flat * time_to_maturity) - self.strike_price * math.exp(
                -self.r_flat * time_to_maturity
            )
            return max(intrinsic, 0.0)
        return 0.0

    # ======================================================================
    # Barrier projection (discrete monitoring)
    # ======================================================================

    def _project_knockout_at_monitoring(self, option_values: List[float]) -> None:
        """
        Absorbing boundary at monitoring instants for KO types.
        """
        if self.barrier_type in ("none", "down-and-in", "up-and-in", "double-in"):
            return

        H_down, H_up = self.barrier_lower, self.barrier_upper
        for idx, S in enumerate(self.stock_grid):
            knocked_out = (
                (self.barrier_type == "down-and-out" and H_down is not None and S <= H_down)
                or (self.barrier_type == "up-and-out" and H_up is not None and S >= H_up)
                or (
                    self.barrier_type == "double-out"
                    and ((H_down is not None and S <= H_down) or (H_up is not None and S >= H_up))
                )
            )
            if knocked_out:
                option_values[idx] = 0.0

    # ======================================================================
    # Tridiagonal solver (Thomas algorithm)
    # ======================================================================

    @staticmethod
    def _solve_tridiagonal_system(
        lower_diag: List[float], main_diag: List[float], upper_diag: List[float], right_hand_side: List[float]
    ) -> List[float]:
        """
        Solve A x = b where A is tridiagonal with lower/main/upper diagonals.
        """
        n = len(right_hand_side)
        modified_upper = [0.0] * n
        modified_rhs = [0.0] * n
        solution = [0.0] * n

        pivot = main_diag[0]
        if abs(pivot) < 1e-14:
            raise ZeroDivisionError("Tridiagonal solver: near-singular pivot at index 0.")
        modified_upper[0] = upper_diag[0] / pivot
        modified_rhs[0] = right_hand_side[0] / pivot

        for i in range(1, n):
            pivot = main_diag[i] - lower_diag[i] * modified_upper[i - 1]
            if abs(pivot) < 1e-14:
                raise ZeroDivisionError(f"Tridiagonal solver: near-singular pivot at index {i}.")
            modified_upper[i] = upper_diag[i] / pivot if i < n - 1 else 0.0
            modified_rhs[i] = (right_hand_side[i] - lower_diag[i] * modified_rhs[i - 1]) / pivot

        solution[-1] = modified_rhs[-1]
        for i in range(n - 2, -1, -1):
            solution[i] = modified_rhs[i] - modified_upper[i] * solution[i + 1]
        return solution

    # ======================================================================
    # PDE time stepping (Crank–Nicolson with Rannacher)
    # ======================================================================

    def _solve_pde_surface(self, barrier_type: BarrierType) -> List[float]:
        """
        Returns the option value surface at t=0 on the stock grid.
        """
        # Optional BGK barrier shift (discrete monitoring continuity correction)
        effective_lower = self.barrier_lower
        effective_upper = self.barrier_upper
        if self.use_bgk_correction and self.num_time_steps > 0:
            beta = 0.5826
            dt_years = self.tenor_years / self.num_time_steps
            shift = math.exp(beta * self.volatility * math.sqrt(max(dt_years, 1e-12)))
            if barrier_type in ("up-and-out", "double-out") and effective_upper is not None:
                effective_upper = effective_upper * shift
            if barrier_type in ("down-and-out", "double-out") and effective_lower is not None:
                effective_lower = effective_lower / shift

        # Temporarily replace for projection
        original_lower, original_upper = self.barrier_lower, self.barrier_upper
        self.barrier_lower, self.barrier_upper = effective_lower, effective_upper

        last_index = len(self.stock_grid) - 1
        dt_years = self.tenor_years / self.num_time_steps

        option_values = self._payoff_array_mollified()

        # Map monitoring dates to interior time steps (exclude t=0 and t=T)
        monitoring_step = {}
        for d in self.monitoring_dates:
            if self.valuation_date < d < self.maturity_date:
                k = int(round(self._year_fraction(self.valuation_date, d) / dt_years))
                k = min(max(k, 1), self.num_time_steps - 1)
                monitoring_step[k] = True

        # CN with Rannacher (only at start or after discrete cash dividends; NOT at barrier)
        drift_carry = self.r_flat - self.q_flat
        restart_budget = 0

        for step in range(self.num_time_steps, 0, -1):
            time_prev = (step - 1) * dt_years
            time_to_maturity_prev = self.tenor_years - time_prev

            theta = 1.0 if ((self.num_time_steps - step) < self.rannacher_steps or restart_budget > 0) else 0.5

            lower = [0.0] * (last_index + 1)
            diag = [0.0] * (last_index + 1)
            upper = [0.0] * (last_index + 1)
            rhs = [0.0] * (last_index + 1)

            # Boundary rows
            rhs[0] = self._boundary_value_at_s_min(time_to_maturity_prev)
            rhs[last_index] = self._boundary_value_at_s_max(self.stock_grid[-1], time_to_maturity_prev)
            diag[0] = 1.0
            diag[last_index] = 1.0

            inv_dS2 = 1.0 / (self.grid_spacing * self.grid_spacing)
            inv_2dS = 1.0 / (2.0 * self.grid_spacing)

            for i in range(1, last_index):
                S = self.stock_grid[i]
                sigma2S2 = (self.volatility * self.volatility) * (S * S)

                # Correct CN coefficients (½ in drift term)
                a = 0.5 * dt_years * (sigma2S2 * inv_dS2 - drift_carry * S * inv_2dS)   # affects V_{i-1}
                b = dt_years * (sigma2S2 * inv_dS2 + self.r_flat)                      # diagonal magnitude
                c = 0.5 * dt_years * (sigma2S2 * inv_dS2 + drift_carry * S * inv_2dS)  # affects V_{i+1}

                lower[i] = -theta * a
                diag[i] = 1.0 + theta * b
                upper[i] = -theta * c

                rhs[i] = (1.0 - (1.0 - theta) * b) * option_values[i] \
                         + (1.0 - theta) * a * option_values[i - 1] \
                         + (1.0 - theta) * c * option_values[i + 1]

            option_values = self._solve_tridiagonal_system(lower, diag, upper, rhs)

            # Apply KO projection at monitoring dates (NO automatic Rannacher restart here)
            if (step - 1) in monitoring_step and barrier_type in ("down-and-out", "up-and-out", "double-out"):
                self._project_knockout_at_monitoring(option_values)
                if self.restart_rannacher_at_barrier:
                    restart_budget = max(restart_budget, 2)

            if restart_budget > 0:
                restart_budget -= 1

        # Restore original barriers
        self.barrier_lower, self.barrier_upper = original_lower, original_upper
        return option_values

    # ======================================================================
    # Pricing and interpolation
    # ======================================================================

    def _linear_interp_on_grid(self, values: List[float], s_query: float) -> float:
        if s_query <= self.stock_grid[0]:
            return values[0]
        if s_query >= self.stock_grid[-1]:
            return values[-1]
        lo, hi = 0, len(self.stock_grid) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if s_query < self.stock_grid[mid]:
                hi = mid
            else:
                lo = mid
        s0, s1 = self.stock_grid[lo], self.stock_grid[hi]
        v0, v1 = values[lo], values[hi]
        w = (s_query - s0) / (s1 - s0)
        return (1.0 - w) * v0 + w * v1

    def _core_price(self) -> float:
        if self.barrier_type in ("down-and-in", "up-and-in", "double-in"):
            vanilla_surface = self._solve_pde_surface("none")
            knockout_surface = self._solve_pde_surface(
                {"down-and-in": "down-and-out", "up-and-in": "up-and-out", "double-in": "double-out"}[self.barrier_type]
            )
            price = self._linear_interp_on_grid(vanilla_surface, self.spot_price) - \
                    self._linear_interp_on_grid(knockout_surface, self.spot_price)
        else:
            surface = self._solve_pde_surface(self.barrier_type)
            price = self._linear_interp_on_grid(surface, self.spot_price)
        return float(price)

    def price(self) -> float:
        if not self.use_richardson_extrapolation:
            base_price = self._core_price()
        else:
            m0, n0 = self.num_time_steps, self.num_space_nodes
            base0 = self._core_price()
            self.num_time_steps, self.num_space_nodes = 2 * m0, 2 * n0
            self.stock_grid = self._build_stock_price_grid()
            self.grid_spacing = self.stock_grid[1] - self.stock_grid[0]
            self.time_grid = [i * self.tenor_years / self.num_time_steps for i in range(self.num_time_steps + 1)]
            base1 = self._core_price()
            base_price = base1 + (base1 - base0) / 3.0
            # restore coarse grid
            self.num_time_steps, self.num_space_nodes = m0, n0
            self.stock_grid = self._build_stock_price_grid()
            self.grid_spacing = self.stock_grid[1] - self.stock_grid[0]
            self.time_grid = [i * self.tenor_years / self.num_time_steps for i in range(self.num_time_steps + 1)]

        sign = 1.0 if self.direction == "long" else -1.0
        return sign * self.quantity * self.contract_multiplier * base_price

    # ======================================================================
    # Barrier-aware Greeks (from surface) + vega by bump
    # ======================================================================

    def _index_of_left_node(self, s: float) -> int:
        lo, hi = 0, len(self.stock_grid) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if s < self.stock_grid[mid]:
                hi = mid
            else:
                lo = mid
        return lo

    def _central_delta(self, surface: List[float], i: int) -> float:
        return (surface[i + 1] - surface[i - 1]) / (2.0 * self.grid_spacing)

    def _central_gamma(self, surface: List[float], i: int) -> float:
        return (surface[i + 1] - 2.0 * surface[i] + surface[i - 1]) / (self.grid_spacing ** 2)

    def _forward_delta(self, surface: List[float], i: int) -> float:
        return (surface[i + 1] - surface[i]) / self.grid_spacing

    def _backward_delta(self, surface: List[float], i: int) -> float:
        return (surface[i] - surface[i - 1]) / self.grid_spacing

    def _forward_gamma(self, surface: List[float], i: int) -> float:
        return (surface[i + 2] - 2.0 * surface[i + 1] + surface[i]) / (self.grid_spacing ** 2)

    def _backward_gamma(self, surface: List[float], i: int) -> float:
        return (surface[i] - 2.0 * surface[i - 1] + surface[i - 2]) / (self.grid_spacing ** 2)

    def _barrier_aware_delta_gamma(self, surface: List[float]) -> Tuple[float, float]:
        """
        FIS-style: one-sided in closest interval to barrier; blended next interval; central elsewhere.
        Assumes barrier snapped to grid (we do).
        """
        s0 = self.spot_price
        N = len(self.stock_grid) - 1
        i = self._index_of_left_node(s0)
        i = max(1, min(N - 1, i))  # ensure interior for centrals

        # If no KO barrier active, use central
        if self.barrier_type not in ("down-and-out", "up-and-out", "double-out"):
            return self._central_delta(surface, i), self._central_gamma(surface, i)

        # Choose nearest barrier side (if both, pick closer)
        barrier_side = None
        if self.barrier_lower is not None:
            barrier_side = "lower"
            barrier_level = self.barrier_lower
        if self.barrier_upper is not None:
            if barrier_side is None or abs(s0 - self.barrier_upper) < abs(s0 - self.barrier_lower):
                barrier_side = "upper"
                barrier_level = self.barrier_upper  # type: ignore

        # find barrier index on grid
        j_bar = min(range(len(self.stock_grid)), key=lambda k: abs(self.stock_grid[k] - barrier_level))  # type: ignore

        # Determine whether spot lies in closest or next interval to barrier
        closest_interval = False
        next_interval = False
        if barrier_side == "lower":
            if i == j_bar - 1:
                closest_interval = True
            elif i == j_bar:
                next_interval = True
        else:  # upper
            if i == j_bar:
                closest_interval = True
            elif i == j_bar - 1:
                next_interval = True

        # Away from barrier: central
        if not closest_interval and not next_interval:
            return self._central_delta(surface, i), self._central_gamma(surface, i)

        # Compute weight q inside the local cell [i, i+1]
        sL, sR = self.stock_grid[i], self.stock_grid[i + 1]
        q_weight = (s0 - sL) / (sR - sL)

        if barrier_side == "lower":
            # away from lower barrier = right-biased (forward) stencil
            if closest_interval:
                if i <= N - 2:
                    delta = self._forward_delta(surface, i)
                    gamma = self._forward_gamma(surface, i)
                else:
                    delta, gamma = self._central_delta(surface, i), self._central_gamma(surface, i)
            else:
                # blend one-sided and central in the next interval
                d_os = self._forward_delta(surface, i)
                d_ce = self._central_delta(surface, i)
                delta = q_weight * d_ce + (1 - q_weight) * d_os
                if i <= N - 2:
                    g_os = self._forward_gamma(surface, i)
                    g_ce = self._central_gamma(surface, i)
                    gamma = q_weight * g_ce + (1 - q_weight) * g_os
                else:
                    gamma = self._central_gamma(surface, i)
        else:
            # away from upper barrier = left-biased (backward) stencil
            if closest_interval:
                if i >= 2:
                    delta = self._backward_delta(surface, i)
                    gamma = self._backward_gamma(surface, i)
                else:
                    delta, gamma = self._central_delta(surface, i), self._central_gamma(surface, i)
            else:
                d_os = self._backward_delta(surface, i)
                d_ce = self._central_delta(surface, i)
                delta = q_weight * d_ce + (1 - q_weight) * d_os
                if i >= 2:
                    g_os = self._backward_gamma(surface, i)
                    g_ce = self._central_gamma(surface, i)
                    gamma = q_weight * g_ce + (1 - q_weight) * g_os
                else:
                    gamma = self._central_gamma(surface, i)

        return delta, gamma

    def greeks(self, method: str = "barrier-aware", bump_rel: float = 0.01, vega_bump_abs: float = 0.01) -> Dict[str, float]:
        """
        method="barrier-aware": compute delta/gamma from one solved surface (fast, near-barrier safe).
        method="bump"        : symmetric spot bumps for delta/gamma (robust reference).
        vega: volatility bumps in both modes.
        """
        if method == "bump":
            s0 = self.spot_price
            ds = s0 * bump_rel
            self.spot_price = s0 + ds
            up = self.price()
            self.spot_price = s0 - ds
            down = self.price()
            self.spot_price = s0
            base = self.price()
            delta = (up - down) / (2.0 * ds)
            gamma = (up - 2.0 * base + down) / (ds * ds)
        else:
            # Solve a single surface consistent with current barrier setup
            if self.barrier_type in ("down-and-in", "up-and-in", "double-in"):
                vanilla_surface = self._solve_pde_surface("none")
                delta, gamma = self._barrier_aware_delta_gamma(vanilla_surface)
            else:
                surface = self._solve_pde_surface(self.barrier_type)
                delta, gamma = self._barrier_aware_delta_gamma(surface)

        # Vega by sigma bump (stable)
        sigma0 = self.volatility
        self.volatility = sigma0 + vega_bump_abs
        up_v = self.price()
        self.volatility = sigma0 - vega_bump_abs
        down_v = self.price()
        self.volatility = sigma0
        vega = (up_v - down_v) / (2.0 * vega_bump_abs)

        return {"delta": delta, "gamma": gamma, "vega": vega}

    # ======================================================================
    # Reporting (for quick sanity checks)
    # ======================================================================

    def report(self) -> str:
        lines = [
            "==== Barrier Option Trade Details ====",
            f"Trade ID            : {self.trade_id}",
            f"Direction           : {self.direction}",
            f"Quantity            : {self.quantity}",
            f"Contract Multiplier : {self.contract_multiplier:.6g}",
            f"Option Type         : {self.option_type}",
            f"Barrier Type        : {self.barrier_type}",
            f"Lower Barrier (Hd)  : {self.barrier_lower if self.barrier_lower is not None else '-'}",
            f"Upper Barrier (Hu)  : {self.barrier_upper if self.barrier_upper is not None else '-'}",
            f"Spot (S0)           : {self.spot_price:.6f}",
            f"Strike (K)          : {self.strike_price:.6f}",
            f"Valuation Date      : {self.valuation_date.isoformat()}",
            f"Maturity Date       : {self.maturity_date.isoformat()}",
            f"T (years, {self.day_count}) : {self.tenor_years:.6f}",
            f"Volatility (sigma)  : {self.volatility:.6f}",
            f"r_flat (NACC)       : {self.r_flat:.6f}",
            f"q_flat (NACC)       : {self.q_flat:.6f}",
            f"Steps (time,space)  : {self.num_time_steps}, {self.num_space_nodes} (Rannacher {self.rannacher_steps})",
            f"Rannacher@Barrier   : {self.restart_rannacher_at_barrier}",
            f"Monitoring dates (#): {len([d for d in self.monitoring_dates if self.valuation_date < d < self.maturity_date])}",
            f"BGK correction      : {self.use_bgk_correction}",
        ]
        return "\n".join(lines)
