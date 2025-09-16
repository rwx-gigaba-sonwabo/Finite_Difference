import datetime as dt
from typing import List, Optional, Tuple, Literal, Dict, Any
import math
import datetime as _dt
import pandas as pd  # type: ignore
from dataclasses import dataclass

"""
Discrete Barrier Option Pricer (Crank-Nicolson + Rannacher), datetime + daily NACA curves.

- Dates: use datetime.date for valuation_date, maturity_date, dividend dates, monitoring dates.
- Curves: pandas DataFrames with columns ["Date", "NACA"] where Date is ISO YYYY-MM-DD string.
- DF math: ACT/365F day count, DF = (1 + NACA)^(-tau).
- Escrowed dividends: PV at valuation using DF(0, t_div); PDE uses q=0; S0_eff = S0 - PV(divs).
- Knock-outs handled by projection at monitoring dates; knock-ins via in-out parity.

"""

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
    CN FDM pricer for discretely monitored European barrier options with daily curves.

    Curve DataFrames: columns "Date" (YYYY-MM-DD), "NACA" (decimal).
    Dividends: list[(datetime.date, cash_amount)].
    Monitoring: list[datetime.date] for KO projection.
    """

    def __init__(
        self,
        spot: float,
        strike: float,
        valuation_date: _dt.date,
        maturity_date: _dt.date,
        sigma: float,
        option_type: OptionType,
        barrier_type: BarrierType = "none",
        lower_barrier: Optional[float] = None,
        upper_barrier: Optional[float] = None,
        monitor_dates: Optional[List[_dt.date]] = None,
        discount_curve: Optional[Any] = None,
        forward_curve: Optional[Any] = None,
        dividend_schedule: Optional[List[Tuple[_dt.date, float]]] = None,
        trade_id: str = "T-0001",
        direction: Literal["long", "short"] = "long",
        quantity: int = 1,
        contract_multiplier: float = 1.0,
        grid_points: int = 400,
        time_steps: int = 400,
        rannacher_steps: int = 2,
        s_max_mult: float = 4.0,
        restart_on_monitoring: bool = False,
        mollify_final: bool = True,
        mollify_band_nodes: int = 2,
        price_extrapolation: bool = False,
        day_count: str = "ACT/365",
    ) -> None:
        # Basic validation
        if any(x <= 0 for x in (spot, strike, sigma)):
            raise ValueError("spot, strike, sigma must be positive.")
        if maturity_date <= valuation_date:
            raise ValueError("maturity_date must be after valuation_date.")

        # Core inputs
        self.spot = spot
        self.strike = strike
        self.valuation_date = valuation_date
        self.maturity_date = maturity_date
        self.sigma = sigma
        self.opt = option_type
        self.barrier_type = barrier_type
        self.Hd = lower_barrier
        self.Hu = upper_barrier
        self.monitor_dates = sorted(monitor_dates or [])
        self.discount_curve_df = discount_curve.copy() if discount_curve is not None else None
        self.forward_curve_df = forward_curve.copy() if forward_curve is not None else None
        self.dividend_schedule = sorted(dividend_schedule or [], key=lambda x: x[0])

        # Trade details
        self.trade_id = trade_id
        self.direction = direction
        self.quantity = int(quantity)
        self.contract_multiplier = float(contract_multiplier)

        # Numerical controls
        self.grid_points = int(grid_points)
        self.time_steps = int(time_steps)
        self.rannacher_steps = int(rannacher_steps)
        self.s_max_mult = s_max_mult
        self.restart_on_monitoring = restart_on_monitoring
        self.mollify_final = mollify_final
        self.mollify_band_nodes = int(mollify_band_nodes)
        self.price_extrapolation = price_extrapolation

        # Day count
        self.day_count = day_count.upper().replace("F", "")
        self._denom = self._infer_denominator(self.day_count)

        # Year fraction for option tenor
        self.T = self._year_fraction(self.valuation_date, self.maturity_date)

        # Normalize curves
        if self.discount_curve_df is not None:
            if pd is None:
                raise ImportError("pandas required for DataFrame curves.")
            self.discount_curve_df = self._normalize_curve_df(self.discount_curve_df)
        if self.forward_curve_df is not None:
            self.forward_curve_df = self._normalize_curve_df(self.forward_curve_df)

        # Space grid
        self.S_grid = self._build_space_grid()
        self.dS = self.S_grid[1] - self.S_grid[0]

        # Uniform time grid (events get mapped to nearest indices)
        self.t_grid = [i * self.T / self.time_steps for i in range(self.time_steps + 1)]

    def _infer_denominator(self, day_count: str) -> int:
        """
        Map day count to denominator used for simple year fractions and to convert
        continuous-time fractions to calendar days for mid-point sampling.
        """
        if day_count in ("ACT/365", "ACT/365F"):
            return 365
        if day_count in ("ACT/360", "ACT/364"):
            return 360 if day_count == "ACT/360" else 364
        if day_count in ("30/360", "BOND", "US30/360"):
            return 360
        # default
        return 365

    def _year_fraction(self, start_date: _dt.date, end_date: _dt.date) -> float:
        """Compute year fraction according to configured day count (simple versions)."""
        if end_date <= start_date:
            return 0.0
        if self.day_count in ("ACT/365", "ACT/365F", "ACT/360", "ACT/364"):
            return (end_date - start_date).days / float(self._denom)
        if self.day_count in ("30/360", "BOND", "US30/360"):
            y1, m1, d1 = start_date.year, start_date.month, start_date.day
            y2, m2, d2 = end_date.year, end_date.month, end_date.day
            d1 = min(d1, 30)
            if d1 == 30:
                d2 = min(d2, 30)
            days = (y2 - y1) * 360 + (m2 - m1) * 30 + (d2 - d1)
            return days / 360.0
        # fallback
        return (end_date - start_date).days / 365.0

    def _linear_interp(self, x: float, xs: List[float], ys: List[float]) -> float:
        """Piecewise-linear interpolation with clamped ends."""
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
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
        return (1 - w) * y0 + w * y1

    def _solve_tridiagonal(
        self,
        lower_diag: List[float],
        main_diag: List[float],
        upper_diag: List[float],
        rhs: List[float],
    ) -> List[float]:
        """Thomas algorithm specialized for tridiagonal matrices."""
        n = len(rhs)
        modified_upper = [0.0] * n
        modified_rhs = [0.0] * n
        solution = [0.0] * n

        # Forward sweep
        pivot = main_diag[0]
        if abs(pivot) < 1e-14:
            raise ZeroDivisionError("Singular matrix: pivot ~ 0 at index 0.")
        modified_upper[0] = upper_diag[0] / pivot
        modified_rhs[0] = rhs[0] / pivot

        for i in range(1, n):
            pivot = main_diag[i] - lower_diag[i] * modified_upper[i - 1]
            if abs(pivot) < 1e-14:
                raise ZeroDivisionError(f"Singular matrix: pivot ~ 0 at index {i}.")
            modified_upper[i] = upper_diag[i] / pivot if i < n - 1 else 0.0
            modified_rhs[i] = (rhs[i] - lower_diag[i] * modified_rhs[i - 1]) / pivot

        # Back substitution
        solution[-1] = modified_rhs[-1]
        for i in range(n - 2, -1, -1):
            solution[i] = modified_rhs[i] - modified_upper[i] * solution[i + 1]

        return solution

    @staticmethod
    def _normalize_curve_df(df: Any) -> Any:
        if "Date" not in df.columns or "NACA" not in df.columns:
            raise ValueError("Curve DataFrame must have columns: 'Date', 'NACA'.")
        if not pd.api.types.is_string_dtype(df["Date"]):
            df = df.copy()
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        return df

    def get_discount_factor(self, lookup_date: _dt.date) -> float:
        if self.discount_curve_df is None:
            raise ValueError("No discount curve attached.")
        iso = lookup_date.isoformat()
        row = self.discount_curve_df[self.discount_curve_df["Date"] == iso]
        if row.empty:
            raise ValueError(f"Discount factor not found for date: {iso}")
        naca = float(row["NACA"].values[0])
        tau = self._year_fraction(self.valuation_date, lookup_date)
        return (1.0 + naca) ** (-tau)

    def get_nacc_rate(self, lookup_date: _dt.date) -> float:
        if self.discount_curve_df is None:
            return 0.0
        iso = lookup_date.isoformat()
        row = self.discount_curve_df[self.discount_curve_df["Date"] == iso]
        if row.empty:
            return 0.0
        naca = float(row["NACA"].values[0])
        return math.log(1.0 + naca)

    def get_forward_nacc_rate(self, start_date: _dt.date, end_date: _dt.date) -> float:
        df_far = self.get_discount_factor(end_date)
        df_near = self.get_discount_factor(start_date)
        tau = self._year_fraction(start_date, end_date)
        return -math.log(df_far / df_near) / max(1e-12, tau)

    def pv_dividends(self) -> float:
        """PV of discrete dividends at valuation_date using DF from discount curve."""
        pv = 0.0
        for pay_date, amount in self.dividend_schedule:
            if self.valuation_date < pay_date <= self.maturity_date:
                df = self.get_discount_factor(pay_date)
                pv += amount * df
        return pv

    def dividend_yield_nacc(self) -> float:
        """Back-out a flat q (NACC) reproducing PV(dividends) on [valuation, maturity]."""
        pv_divs = self.pv_dividends()
        S = self.spot
        tau = max(1e-12, self.T)
        if pv_divs >= S:
            raise ValueError("PV(dividends) >= spot.")
        return -math.log((S - pv_divs) / S) / tau

    def _build_space_grid(self) -> List[float]:
        candidates = [self.spot, self.strike]
        if self.Hd:
            candidates.append(self.Hd)
        if self.Hu:
            candidates.append(self.Hu)
        s_star = max(candidates)
        s_max = self.s_max_mult * s_star * math.exp(self.sigma * math.sqrt(self.T))
        s_min = 0.0
        n = max(200, self.grid_points)
        dS = (s_max - s_min) / n
        grid = [s_min + i * dS for i in range(n + 1)]
        # align K/H
        def snap(value: Optional[float]) -> None:
            if value is None:
                return
            idx = min(range(len(grid)), key=lambda i: abs(grid[i] - value))
            grid[idx] = value
        snap(self.strike)
        snap(self.Hd)
        snap(self.Hu)
        return grid

    def _payoff(self, S: float) -> float:
        if self.opt == "call":
            return max(S - self.strike, 0.0)
        else:
            return max(self.strike - S, 0.0)

    def _mollified_payoff_array(self) -> List[float]:
        vT = [self._payoff(S) for S in self.S_grid]
        if not self.mollify_final or self.mollify_band_nodes <= 0:
            return vT
        k_idx = min(range(len(self.S_grid)), key=lambda i: abs(self.S_grid[i] - self.strike))
        m = self.mollify_band_nodes
        i0 = max(0, k_idx - m)
        i1 = min(len(self.S_grid) - 1, k_idx + m)
        S0, V0 = self.S_grid[i0], vT[i0]
        S1, V1 = self.S_grid[i1], vT[i1]
        slope_left = 0.0
        b = slope_left
        a = (V1 - V0 - b * (S1 - S0)) / ((S1 - S0) ** 2) if S1 != S0 else 0.0
        for i in range(i0, i1 + 1):
            s = self.S_grid[i]
            vT[i] = a * (s - S0) ** 2 + b * (s - S0) + V0
        return vT

    def _boundary_low(self, tau: float) -> float:
        if self.opt == "call":
            return 0.0
        else:
            K_df = self.get_discount_factor(self.maturity_date)
            return self.strike * K_df

    def _boundary_high(self, S_max: float, tau: float) -> float:
        if self.opt == "call":
            K_df = self.get_discount_factor(self.maturity_date)
            return S_max - self.strike * K_df
        else:
            return 0.0

    def _apply_barrier_projection(self, values: List[float]) -> None:
        if self.barrier_type in ("none", "down-and-in", "up-and-in", "double-in"):
            return
        Hd, Hu = self.Hd, self.Hu
        for i, S in enumerate(self.S_grid):
            ko = False
            if self.barrier_type == "down-and-out" and Hd is not None and S <= Hd:
                ko = True
            elif self.barrier_type == "up-and-out" and Hu is not None and S >= Hu:
                ko = True
            elif self.barrier_type == "double-out":
                if (Hd is not None and S <= Hd) or (Hu is not None and S >= Hu):
                    ko = True
            if ko:
                values[i] = 0.0

    def _solve_pde(self, barrier_type: BarrierType) -> List[float]:
        N = len(self.S_grid) - 1
        M = self.time_steps
        dt = self.T / M

        v = self._mollified_payoff_array()

        # map monitoring dates -> time indices
        mon_idx: Dict[int, bool] = {}
        for d in self.monitor_dates:
            if self.valuation_date < d <= self.maturity_date:
                k = int(round(self._year_fraction(self.valuation_date, d) / dt))
                k = min(max(k, 0), M)
                mon_idx[k] = True

        restart_budget = 0
        for m in range(M, 0, -1):
            t_prev = (m - 1) * dt
            tau_prev = self.T - t_prev

            theta = 1.0 if ((M - m) < self.rannacher_steps or restart_budget > 0) else 0.5

            # Sample rate at midpoint date via discount curve
            days = int(round((t_prev + theta * dt) * self._denom))
            mid_date = self.valuation_date + _dt.timedelta(days=days)
            r_now = self.get_nacc_rate(mid_date)

            lower = [0.0] * (N + 1)
            diag  = [0.0] * (N + 1)
            upper = [0.0] * (N + 1)
            rhs   = [0.0] * (N + 1)

            v_low = self._boundary_low(tau_prev)
            v_high = self._boundary_high(self.S_grid[-1], tau_prev)
            rhs[0] = v_low
            rhs[N] = v_high
            diag[0] = 1.0
            diag[N] = 1.0

            for i in range(1, N):
                S = self.S_grid[i]
                sigma2S2 = (self.sigma ** 2) * (S ** 2)

                A = 0.5 * dt * theta * (sigma2S2 / (self.dS ** 2) - r_now * S / self.dS)
                B = 1.0 + dt * theta * (sigma2S2 / (self.dS ** 2) + r_now)
                C = 0.5 * dt * theta * (sigma2S2 / (self.dS ** 2) + r_now * S / self.dS)

                A_ = -0.5 * dt * (1 - theta) * (sigma2S2 / (self.dS ** 2) - r_now * S / self.dS)
                B_ = 1.0 - dt * (1 - theta) * (sigma2S2 / (self.dS ** 2) + r_now)
                C_ = -0.5 * dt * (1 - theta) * (sigma2S2 / (self.dS ** 2) + r_now * S / self.dS)

                lower[i] = -A
                diag[i]  =  B
                upper[i] = -C
                rhs[i]   =  A_ * v[i - 1] + B_ * v[i] + C_ * v[i + 1]

            v = self._solve_tridiagonal(lower, diag, upper, rhs)

            if (m - 1) in mon_idx and barrier_type in ("down-and-out", "up-and-out", "double-out"):
                self._apply_barrier_projection(v)
                if self.restart_on_monitoring:
                    restart_budget = max(restart_budget, 2)
            if restart_budget > 0:
                restart_budget -= 1

        return v

    def _interp_value_at(self, values: List[float], S_eff: float) -> float:
        if S_eff <= self.S_grid[0]:
            return values[0]
        if S_eff >= self.S_grid[-1]:
            return values[-1]
        lo, hi = 0, len(self.S_grid) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if S_eff < self.S_grid[mid]:
                hi = mid
            else:
                lo = mid
        s0, s1 = self.S_grid[lo], self.S_grid[hi]
        v0, v1 = values[lo], values[hi]
        w = (S_eff - s0) / (s1 - s0)
        return (1 - w) * v0 + w * v1

    def _price_core(self) -> float:
        # Escrowed spot shift
        S_eff_shift = self.pv_dividends()
        S_grid_eff = [max(S - S_eff_shift, 0.0) for S in self.S_grid]
        backup_grid = self.S_grid
        self.S_grid = S_grid_eff

        S_eff = self.spot - S_eff_shift

        if self.barrier_type in ("down-and-in", "up-and-in", "double-in"):
            v_van = self._solve_pde("none")
            ko_map = {
                "down-and-in": "down-and-out",
                "up-and-in": "up-and-out",
                "double-in": "double-out",
            }
            v_ko = self._solve_pde(ko_map[self.barrier_type])
            px = self._interp_value_at(v_van, S_eff) - self._interp_value_at(v_ko, S_eff)
        else:
            v = self._solve_pde(self.barrier_type)
            px = self._interp_value_at(v, S_eff)

        self.S_grid = backup_grid
        return float(px)

    def price(self) -> float:
        if not self.price_extrapolation:
            raw = self._price_core()
        else:
            M0, N0 = self.time_steps, self.grid_points
            raw0 = self._price_core()
            self.time_steps = 2 * M0
            self.grid_points = 2 * N0
            self.S_grid = self._build_space_grid()
            self.dS = self.S_grid[1] - self.S_grid[0]
            self.t_grid = [i * self.T / self.time_steps for i in range(self.time_steps + 1)]
            raw1 = self._price_core()
            raw = raw1 + (raw1 - raw0) / 3.0
            # restore
            self.time_steps = M0
            self.grid_points = N0
            self.S_grid = self._build_space_grid()
            self.dS = self.S_grid[1] - self.S_grid[0]
            self.t_grid = [i * self.T / self.time_steps for i in range(self.time_steps + 1)]

        sign = 1.0 if self.direction == "long" else -1.0
        return sign * self.quantity * self.contract_multiplier * raw

    def greeks(self, bump_rel: float = 0.01, vega_bump_abs: float = 0.01) -> Dict[str, float]:
        # Greeks are returned in position terms (direction/size applied like price())
        s0 = self.spot
        ds = s0 * bump_rel

        self.spot = s0 + ds
        up = self.price()
        self.spot = s0 - ds
        down = self.price()
        self.spot = s0
        base = self.price()
        delta = (up - down) / (2 * ds)
        gamma = (up - 2 * base + down) / (ds ** 2)

        sig0 = self.sigma
        self.sigma = sig0 + vega_bump_abs
        up_v = self.price()
        self.sigma = sig0 - vega_bump_abs
        down_v = self.price()
        self.sigma = sig0
        vega = (up_v - down_v) / (2 * vega_bump_abs)

        return {"delta": delta, "gamma": gamma, "vega": vega}

    def report(self) -> str:
        df_T = self.get_discount_factor(self.maturity_date)
        fwd_rate = self.get_forward_nacc_rate(self.valuation_date, self.maturity_date)
        parts = [
            "==== Barrier Option Trade Details ====",
            f"Trade ID            : {self.trade_id}",
            f"Direction           : {self.direction}",
            f"Quantity            : {self.quantity}",
            f"Contract Multiplier : {self.contract_multiplier:.6g}",
            f"Option Type         : {self.opt}",
            f"Barrier Type        : {self.barrier_type}",
            f"Lower Barrier (Hd)  : {self.Hd if self.Hd is not None else '-'}",
            f"Upper Barrier (Hu)  : {self.Hu if self.Hu is not None else '-'}",
            f"Spot (S0)           : {self.spot:.6f}",
            f"Strike (K)          : {self.strike:.6f}",
            f"Valuation Date      : {self.valuation_date.isoformat()}",
            f"Maturity Date       : {self.maturity_date.isoformat()}",
            f"T (years, {self.day_count}) : {self.T:.6f}",
            f"Volatility (sigma)  : {self.sigma:.6f}",
            f"DF(0,T)             : {df_T:.8f}",
            f"Fwd NACC(0->T)      : {fwd_rate:.6f}",
            f"Div PV (escrow)     : {self.pv_dividends():.6f}",
            f"Steps (M,N)         : {self.time_steps}, {self.grid_points} (Rannacher {self.rannacher_steps})",
            f"Monitor dates (#)   : {len(self.monitor_dates)}",
        ]
        return "\n".join(parts)
