
"""
Discrete barrier option pricer (European):
- Decides discrete vs continuous monitoring using the exact FIS n_lim rule.
- If "frequent enough": uses analytic *continuous* barrier engines with BGK barrier shift.
- Else: uses a Crank-Nicolson PDE overlay and projects KO only on monitoring dates.

Rates & dividends:
- Flat r (continuous compounding) from daily NACA discount curve.
- Flat q (continuous) from PV of discrete dividends (escrow interpretation).

Greeks:
- Bump-and-reprice (stable).
- In continuous mode, if spot is close to barrier, use a one-sided delta stencil.

Rebates/flags:
- Passes rebate and timing flags to analytic single-barrier engine where relevant.
- Knock-ins via parity.

Author's note:
- The code fails safe: if the analytic engine import or call fails in the continuous branch,
  it falls back to the CN overlay with continuous projection and shifted barriers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal, Dict

import pandas as pd

try:
    from barrier_engine import BarrierEngine         # single-barrier, RR/Merton style
except Exception:  # pragma: no cover
    BarrierEngine = None

try:
    
    from double_barrier import DoubleBarrier 
except Exception:  # pragma: no cover
    DoubleBarrier = None


BarrierType = Literal[
    "none",
    "down-and-out", "up-and-out", "double-out",
    "down-and-in",  "up-and-in",  "double-in"
]
OptionType = Literal["call", "put"]


class DiscreteBarrierFDMPricerAnalytic:
    """See module docstring."""

    BGK_BETA = 0.5826  # Broadie–Glasserman–Kou continuity-correction constant

    def __init__(
        self,
        # Contract / trading inputs
        trade_id: str,
        direction: Literal["long", "short"],
        quantity: int,
        contract_multiplier: float,
        # Product
        option_type: OptionType,
        barrier_type: BarrierType,
        strike: float,
        lower_barrier: Optional[float],
        upper_barrier: Optional[float],
        rebate_amount: float = 0.0,
        rebate_timing_in: Optional[str] = None,   # 'expiry' or 'hit' or None
        rebate_timing_out: Optional[str] = None,  # 'hit' (typical) or 'expiry' or None
        barrier_status: Optional[str] = None,     # None / 'crossed' / 'not_crossed'
        # Market
        spot: float = 100.0,
        volatility: float = 0.20,                 # Black–Scholes sigma
        valuation_date: pd.Timestamp = pd.Timestamp.today().normalize(),
        maturity_date: pd.Timestamp   = pd.Timestamp.today().normalize() + pd.Timedelta(days=365),
        monitoring_dates: Optional[List[pd.Timestamp]] = None,
        # Curves & dividends
        discount_curve: Optional[pd.DataFrame] = None,    # columns: Date (YYYY-mm-dd), NACA
        forward_curve: Optional[pd.DataFrame] = None,     # optional (same format)
        dividend_schedule: Optional[List[Tuple[pd.Timestamp, float]]] = None,  # [(pay_date, cash), ...]
        day_count: str = "ACT/365",
        # CN overlay controls
        time_steps: int = 600,
        space_nodes: int = 600,
        rannacher_steps: int = 2,
        snap_strike_and_barrier: bool = True,
        # FIS discrete→continuous decision parameters
        n_desired_for_decision: int = 400,   # n in the doc
        n_min_steps_per_interval: int = 1,   # lower bound for n_m
        n_lim_multiplier: int = 5,           # n_lim in the doc (default 5)
    ) -> None:

        # --- Store trading + product inputs
        if spot <= 0 or strike <= 0 or volatility <= 0:
            raise ValueError("spot, strike, volatility must be positive.")
        if maturity_date <= valuation_date:
            raise ValueError("maturity_date must be after valuation_date.")

        self.trade_id = trade_id
        self.direction = direction
        self.quantity = int(quantity)
        self.contract_multiplier = float(contract_multiplier)

        self.option_type = option_type
        self.barrier_type = barrier_type
        self.strike = float(strike)
        self.lower_barrier = lower_barrier
        self.upper_barrier = upper_barrier

        self.rebate_amount = float(rebate_amount)
        self.rebate_timing_in = rebate_timing_in
        self.rebate_timing_out = rebate_timing_out
        self.barrier_status = barrier_status

        self.spot = float(spot)
        self.sigma = float(volatility)
        self.valuation_date = pd.Timestamp(valuation_date).normalize()
        self.maturity_date = pd.Timestamp(maturity_date).normalize()
        self.monitoring_dates = sorted([pd.Timestamp(d).normalize() for d in (monitoring_dates or [])])

        self.discount_curve = discount_curve.copy() if discount_curve is not None else None
        self.forward_curve = forward_curve.copy() if forward_curve is not None else None
        self.dividend_schedule = [(pd.Timestamp(d).normalize(), float(a)) for (d, a) in (dividend_schedule or [])]

        self.day_count = day_count.upper()

        self.time_steps = int(time_steps)
        self.space_nodes = int(space_nodes)
        self.rannacher_steps = int(rannacher_steps)
        self.snap_strike_and_barrier = bool(snap_strike_and_barrier)

        self.n_desired_for_decision = int(n_desired_for_decision)
        self.n_min_steps_per_interval = int(n_min_steps_per_interval)
        self.n_lim_multiplier = int(n_lim_multiplier)

        # --- Normalize curves
        def _norm(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            if not pd.api.types.is_string_dtype(out["Date"]):
                out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
            return out

        if self.discount_curve is not None:
            self.discount_curve = _norm(self.discount_curve)
        if self.forward_curve is not None:
            self.forward_curve = _norm(self.forward_curve)

        # --- Year fraction + tenor
        self._yf = self._year_fraction
        self.tenor_years = self._yf(self.valuation_date, self.maturity_date)

        # --- Flat r and q (continuous) from user curves & dividend schedule
        self.flat_rate_r = self._flat_r_from_curve()
        self.flat_dividend_q = self._flat_q_from_dividends()
        self.flat_carry_b = self.flat_rate_r - self.flat_dividend_q

        # --- Build space grid and snap anchors
        self.spot_grid = self._build_space_grid()
        self.grid_step_dS = self.spot_grid[1] - self.spot_grid[0]

        # --- Decide discrete vs continuous window per FIS n_lim rule; compute BGK shift if needed
        (self.use_continuous_window,
         self.window_k0, self.window_k1,
         self.bgk_lower_barrier, self.bgk_upper_barrier,
         self.monitor_steps_discrete, self.monitor_steps_continuous) = self._monitoring_decision_and_bgk_shift()

    # =============================================================================
    # Date / curve / dividend helpers
    # =============================================================================
    def _year_fraction(self, d0: pd.Timestamp, d1: pd.Timestamp) -> float:
        days = max(0, int((pd.Timestamp(d1) - pd.Timestamp(d0)).days))
        if self.day_count in ("ACT/365", "ACT/365F", "ACT/365 FIXED"):
            return days / 365.0
        if self.day_count in ("ACT/360",):
            return days / 360.0
        if self.day_count in ("30/360", "30E/360"):
            y0, m0, dd0 = d0.year, d0.month, min(d0.day, 30)
            y1, m1, dd1 = d1.year, d1.month, min(d1.day, 30)
            return ((y1 - y0) * 360 + (m1 - m0) * 30 + (dd1 - dd0)) / 360.0
        return days / 365.0

    def _df_from_naca(self, naca: float, tau_years: float) -> float:
        # daily nominal annually compounded -> DF(τ)
        return (1.0 + naca) ** (-tau_years)

    def _find_curve_naca_on(self, on_date: pd.Timestamp) -> float:
        if self.discount_curve is None:
            return 0.0
        key = pd.Timestamp(on_date).strftime("%Y-%m-%d")
        row = self.discount_curve[self.discount_curve["Date"] == key]
        if row.empty:
            # permissive default: return 0 NACA if missing
            return 0.0
        return float(row["NACA"].values[0])

    def _flat_r_from_curve(self) -> float:
        """Flat continuous r from the daily NACA curve over [0, T]."""
        tau = max(1e-12, self.tenor_years)
        naca_T = self._find_curve_naca_on(self.maturity_date)
        df_T = self._df_from_naca(naca_T, tau)
        return -math.log(max(df_T, 1e-16)) / tau

    def _pv_dividends(self) -> float:
        """Present value of cash dividends paid strictly after valuation and on/before maturity."""
        pv = 0.0
        for pay_date, amount in self.dividend_schedule:
            if self.valuation_date < pay_date <= self.maturity_date:
                tau = self._yf(self.valuation_date, pay_date)
                naca_d = self._find_curve_naca_on(pay_date)
                df_d = self._df_from_naca(naca_d, tau)
                pv += amount * df_d
        return pv

    def _flat_q_from_dividends(self) -> float:
        """Back out a flat continuous dividend yield q reproducing PV of discrete dividends."""
        pv = self._pv_dividends()
        if pv <= 0.0:
            return 0.0
        if pv >= self.spot:
            raise ValueError("PV(dividends) >= spot; cannot back out flat dividend yield.")
        return -math.log((self.spot - pv) / self.spot) / max(1e-12, self.tenor_years)

    # =============================================================================
    # Space grid & payoff/boundaries
    # =============================================================================
    def _build_space_grid(self) -> List[float]:
        anchors = [self.spot, self.strike]
        if self.lower_barrier: anchors.append(self.lower_barrier)
        if self.upper_barrier: anchors.append(self.upper_barrier)
        s_ref = max(anchors)
        s_max = 4.0 * s_ref * math.exp(self.sigma * math.sqrt(max(self.tenor_years, 1e-12)))
        s_min = 0.0
        dS = (s_max - s_min) / self.space_nodes
        grid = [s_min + i * dS for i in range(self.space_nodes + 1)]

        if self.snap_strike_and_barrier:
            def snap(x: Optional[float]):
                if x is None: return
                j = min(range(len(grid)), key=lambda i: abs(grid[i] - x))
                grid[j] = x
            snap(self.strike); snap(self.lower_barrier); snap(self.upper_barrier)
        return grid

    def _terminal_payoff(self, S: float) -> float:
        return max(S - self.strike, 0.0) if self.option_type == "call" else max(self.strike - S, 0.0)

    def _terminal_values(self) -> List[float]:
        values = [self._terminal_payoff(S) for S in self.spot_grid]
        # small mollifier around K
        m = 2
        jK = min(range(len(self.spot_grid)), key=lambda i: abs(self.spot_grid[i] - self.strike))
        i0, i1 = max(0, jK - m), min(len(self.spot_grid) - 1, jK + m)
        S0, V0 = self.spot_grid[i0], values[i0]
        S1, V1 = self.spot_grid[i1], values[i1]
        if S1 > S0:
            a = (V1 - V0) / ((S1 - S0) ** 2)
            for i in range(i0, i1 + 1):
                s = self.spot_grid[i]
                values[i] = a * (s - S0) ** 2 + V0
        return values

    def _upper_boundary(self, Smax: float, tau: float) -> float:
        if self.option_type == "call":
            return Smax * math.exp(-self.flat_dividend_q * tau) - self.strike * math.exp(-self.flat_rate_r * tau)
        return 0.0

    def _lower_boundary(self, tau: float) -> float:
        if self.option_type == "put":
            return self.strike * math.exp(-self.flat_rate_r * tau)
        return 0.0

    # =============================================================================
    # FIS n_lim decision + BGK barrier shift
    # =============================================================================
    def _monitoring_decision_and_bgk_shift(self):
        """
        Implements FIS n_lim rule:

        1) Choose equidistant Δt = T / n_desired_for_decision.
        2) For each interval between consecutive monitoring dates, set
             n_m = max(n_min_steps_per_interval, round(t_m / Δt)).
        3) Let N_total = sum(n_m). If N_total > n_lim_multiplier * n_desired_for_decision,
           then use a CONTINUOUS approximation between first and last monitoring dates,
           with BGK barrier shift H_adj = H * exp(± β σ sqrt(Δt_avg)),
           where Δt_avg is the *average* monitoring interval.
        """
        if self.barrier_type in ("none",) or len(self.monitoring_dates) == 0:
            return (False, None, None, self.lower_barrier, self.upper_barrier, {}, {})

        md = [d for d in self.monitoring_dates if self.valuation_date < d <= self.maturity_date]
        if len(md) == 0:
            return (False, None, None, self.lower_barrier, self.upper_barrier, {}, {})

        md = sorted(md)
        dt_eq = self.tenor_years / max(1, self.n_desired_for_decision)

        # Intervals between consecutive monitoring dates
        intervals = []
        prev = md[0]
        for d in md[1:]:
            intervals.append(self._yf(prev, d))
            prev = d
        if len(intervals) == 0:
            intervals = [self.tenor_years / len(md)]

        steps_per_interval = [
            max(self.n_min_steps_per_interval, int(round(ti / max(1e-12, dt_eq))))
            for ti in intervals
        ]
        N_total = sum(steps_per_interval)

        use_continuous = (N_total > self.n_lim_multiplier * self.n_desired_for_decision)

        # Discrete map: monitoring slice indices
        monitor_steps_discrete: Dict[int, bool] = {}
        for d in md:
            k = int(round(self._yf(self.valuation_date, d) / (self.tenor_years / self.time_steps)))
            k = max(0, min(self.time_steps, k))
            monitor_steps_discrete[k] = True

        monitor_steps_continuous: Dict[int, bool] = {}
        if use_continuous:
            first, last = md[0], md[-1]
            k0 = int(round(self._yf(self.valuation_date, first) / (self.tenor_years / self.time_steps)))
            k1 = int(round(self._yf(self.valuation_date, last)  / (self.tenor_years / self.time_steps)))
            k0, k1 = max(0, min(self.time_steps, k0)), max(0, min(self.time_steps, k1))
            for k in range(min(k0, k1), max(k0, k1) + 1):
                monitor_steps_continuous[k] = True

            # BGK shift using average interval
            avg_dt = sum(intervals) / len(intervals)
            adj = math.exp(self.BGK_BETA * self.sigma * math.sqrt(max(1e-12, avg_dt)))
            Hdn_adj = self.lower_barrier / adj if self.lower_barrier is not None else None
            Hup_adj = self.upper_barrier * adj if self.upper_barrier is not None else None
            return (True, min(k0, k1), max(k0, k1), Hdn_adj, Hup_adj,
                    monitor_steps_discrete, monitor_steps_continuous)

        return (False, None, None, self.lower_barrier, self.upper_barrier,
                monitor_steps_discrete, monitor_steps_continuous)

    # =============================================================================
    # CN engine (used for discrete monitoring; and as safety fallback)
    # =============================================================================
    def _apply_knockout_projection(self, values: List[float],
                                   eff_lower: Optional[float],
                                   eff_upper: Optional[float]) -> None:
        for i, s in enumerate(self.spot_grid):
            knock_out = False
            if self.barrier_type == "down-and-out" and eff_lower is not None and s <= eff_lower:
                knock_out = True
            elif self.barrier_type == "up-and-out" and eff_upper is not None and s >= eff_upper:
                knock_out = True
            elif self.barrier_type == "double-out":
                if (eff_lower is not None and s <= eff_lower) or (eff_upper is not None and s >= eff_upper):
                    knock_out = True
            if knock_out:
                values[i] = 0.0  # cash rebate-at-hit could be added here if needed

    @staticmethod
    def _solve_tridiagonal(a: List[float], b: List[float], c: List[float], d: List[float]) -> List[float]:
        n = len(d)
        cp = [0.0] * n
        dp = [0.0] * n
        x  = [0.0] * n
        beta = b[0]
        if abs(beta) < 1e-14:
            raise ZeroDivisionError("Tridiagonal pivot ~ 0 at row 0.")
        cp[0] = c[0] / beta
        dp[0] = d[0] / beta
        for i in range(1, n):
            beta = b[i] - a[i] * cp[i - 1]
            if abs(beta) < 1e-14:
                raise ZeroDivisionError(f"Tridiagonal pivot ~ 0 at row {i}.")
            cp[i] = c[i] / beta if i < n - 1 else 0.0
            dp[i] = (d[i] - a[i] * dp[i - 1]) / beta
        x[-1] = dp[-1]
        for i in range(n - 2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i + 1]
        return x

    def _cn_stepper(self, eff_lower: Optional[float], eff_upper: Optional[float],
                    monitor_steps_map: Dict[int, bool]) -> List[float]:
        N = len(self.spot_grid) - 1
        M = self.time_steps
        dt = self.tenor_years / M
        dS = self.grid_step_dS
        r, q, sig = self.flat_rate_r, self.flat_dividend_q, self.sigma

        values = self._terminal_values()

        for m in range(M, 0, -1):
            theta = 1.0 if (M - m) < self.rannacher_steps else 0.5

            sub = [0.0] * (N + 1)
            diag = [0.0] * (N + 1)
            sup = [0.0] * (N + 1)
            rhs =  [0.0] * (N + 1)

            tau_prev = self.tenor_years - (m - 1) * dt
            rhs[0]  = self._lower_boundary(tau_prev)
            rhs[N]  = self._upper_boundary(self.spot_grid[-1], tau_prev)
            diag[0] = 1.0
            diag[N] = 1.0

            for i in range(1, N):
                S = self.spot_grid[i]
                sig2S2 = (sig ** 2) * (S ** 2)

                A = 0.5 * dt * theta * (sig2S2 / (dS ** 2) - (r - q) * S / dS)
                B = 1.0 + dt * theta * (sig2S2 / (dS ** 2) + r)
                C = 0.5 * dt * theta * (sig2S2 / (dS ** 2) + (r - q) * S / dS)

                A_ = -0.5 * dt * (1 - theta) * (sig2S2 / (dS ** 2) - (r - q) * S / dS)
                B_ =  1.0 - dt * (1 - theta) * (sig2S2 / (dS ** 2) + r)
                C_ = -0.5 * dt * (1 - theta) * (sig2S2 / (dS ** 2) + (r - q) * S / dS)

                sub[i]  = -A
                diag[i] =  B
                sup[i]  = -C
                rhs[i]  =  A_ * values[i - 1] + B_ * values[i] + C_ * values[i + 1]

            values = self._solve_tridiagonal(sub, diag, sup, rhs)

            # Apply KO only on monitoring steps (discrete) OR at every step in the continuous window map
            if (m - 1) in monitor_steps_map:
                self._apply_knockout_projection(values, eff_lower, eff_upper)
                # Per FIS: do NOT trigger a Rannacher restart due to barrier projection.

        return values

    # =============================================================================
    # Pricing & Greeks
    # =============================================================================
    def _linear_interp(self, x: float, xs: List[float], ys: List[float]) -> float:
        if x <= xs[0]:  return ys[0]
        if x >= xs[-1]: return ys[-1]
        lo, hi = 0, len(xs) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if x < xs[mid]: hi = mid
            else: lo = mid
        x0, x1 = xs[lo], xs[hi]
        y0, y1 = ys[lo], ys[hi]
        w = (x - x0) / (x1 - x0)
        return (1 - w) * y0 + w * y1

    def _escrowed_spot(self) -> float:
        """Escrow interpretation: shift spot by PV of discrete dividends."""
        return self.spot - self._pv_dividends()

    def _can_use_single_barrier_analytic(self) -> bool:
        """Basic sanity gate for single-barrier analytic engine."""
        if self.barrier_type not in ("down-and-out", "up-and-out", "down-and-in", "up-and-in"):
            return False
        H = self.lower_barrier if "down" in self.barrier_type else self.upper_barrier
        if H is None or H <= 0.0:
            return False
        # If barrier status is constrained, prefer CN overlay
        if self.barrier_status not in (None, "crossed", "not_crossed"):
            return False
        if self.barrier_status is not None:
            return False
        # Rebate timing flags must be sane
        if self.rebate_timing_in not in (None, "hit", "expiry"):
            return False
        if self.rebate_timing_out not in (None, "hit", "expiry"):
            return False
        return True

    def _continuous_branch_analytic(self, S_eff: float) -> float:
        """
        Continuous monitoring approximation using analytic engines with BGK-shifted barriers.
        Fails safe to CN overlay if import or engine call fails.
        """
        # Double barrier path
        if self.barrier_type in ("double-out", "double-in"):
            if DoubleBarrier is None:
                return self._continuous_branch_cn(S_eff)  # fail safe
            if (self.bgk_lower_barrier is None) or (self.bgk_upper_barrier is None):
                return self._continuous_branch_cn(S_eff)  # need both
            inflag = 'in' if 'in' in self.barrier_type else 'out'
            callput = 'c' if self.option_type == "call" else 'p'
            try:
                engine = DoubleBarrier(
                    S=S_eff, X=self.strike,
                    L=self.bgk_lower_barrier, U=self.bgk_upper_barrier,
                    sigma=self.sigma, callflag=callput, inflag=inflag, m=6
                )
                return float(engine.price(b=self.flat_carry_b, r=self.flat_rate_r, T=self.tenor_years))
            except Exception:
                return self._continuous_branch_cn(S_eff)

        # Single barrier path
        if not self._can_use_single_barrier_analytic() or BarrierEngine is None:
            return self._continuous_branch_cn(S_eff)

        direction = 'd' if 'down' in self.barrier_type else 'u'
        inout     = 'i' if 'in'   in self.barrier_type else 'o'
        shifted_H = self.bgk_lower_barrier if 'down' in self.barrier_type else self.bgk_upper_barrier
        if shifted_H is None:
            return self._continuous_branch_cn(S_eff)

        try:
            engine = BarrierEngine(
                s=S_eff, b=self.flat_carry_b, r=self.flat_rate_r, t=self.tenor_years,
                x=self.strike, sigma=self.sigma, h=shifted_H,
                optionflag=('c' if self.option_type == "call" else 'p'),
                directionflag=direction, in_out_flag=inout,
                k=self.rebate_amount,
                barrier_status=self.barrier_status,
                rebate_timing_in=self.rebate_timing_in,
                rebate_timing_out=self.rebate_timing_out
            )
            return float(engine.price())
        except Exception:
            return self._continuous_branch_cn(S_eff)

    def _continuous_branch_cn(self, S_eff: float) -> float:
        """
        Safety fallback: CN with continuous projection *every step* in the monitoring window,
        using BGK-shifted barriers.
        """
        eff_lower = self.bgk_lower_barrier
        eff_upper = self.bgk_upper_barrier
        values = self._cn_stepper(eff_lower, eff_upper, self.monitor_steps_continuous)
        return float(self._linear_interp(S_eff, self.spot_grid, values))

    def _discrete_branch_cn(self, S_eff: float) -> float:
        """Discrete monitoring: CN with KO projection only at monitoring dates."""
        eff_lower = self.lower_barrier
        eff_upper = self.upper_barrier
        values = self._cn_stepper(eff_lower, eff_upper, self.monitor_steps_discrete)
        return float(self._linear_interp(S_eff, self.spot_grid, values))

    def price(self) -> float:
        # escrow the spot
        S_eff = self._escrowed_spot()

        # shift the *grid* by the same amount to keep interpolation tight
        backup_grid = self.spot_grid[:]
        self.spot_grid = [max(0.0, s - (self.spot - S_eff)) for s in self.spot_grid]
        self.grid_step_dS = self.spot_grid[1] - self.spot_grid[0]

        # knock-ins via vanilla − KO
        if self.barrier_type in ("down-and-in", "up-and-in", "double-in"):
            # vanilla = CN with no barriers (fast & stable)
            vanilla_vals = self._cn_stepper(None, None, {})  # continuous PDE without projection
            vanilla = self._linear_interp(S_eff, self.spot_grid, vanilla_vals)
            if self.use_continuous_window:
                ko_val = self._continuous_branch_analytic(S_eff)
            else:
                ko_val = self._discrete_branch_cn(S_eff)
            base_price = vanilla - ko_val
        else:
            # outs valued directly
            if self.use_continuous_window:
                base_price = self._continuous_branch_analytic(S_eff)
            else:
                base_price = self._discrete_branch_cn(S_eff)

        # restore grid
        self.spot_grid = backup_grid
        self.grid_step_dS = self.spot_grid[1] - self.spot_grid[0]

        # scale by trade direction / size
        sign = 1.0 if self.direction == "long" else -1.0
        scale = sign * self.quantity * self.contract_multiplier
        return float(scale * base_price)

    def greeks(self, rel_spot_bump: float = 1e-4, abs_vol_bump: float = 1e-4) -> Dict[str, float]:
        # Always compute on long 1 × 1 to get clean sensitivities, then rescale.
        save_dir = self.direction
        save_qty = self.quantity
        save_mult = self.contract_multiplier
        self.direction = "long"
        self.quantity = 1
        self.contract_multiplier = 1.0

        base_px = self.price()
        s0 = self.spot
        ds = max(1e-8, rel_spot_bump * s0)

        # choose stencil: if in continuous window and close to barrier, use one-sided delta
        def near_barrier(S) -> bool:
            tol = 2 * self.grid_step_dS
            Hdn = self.bgk_lower_barrier if self.use_continuous_window else self.lower_barrier
            Hup = self.bgk_upper_barrier if self.use_continuous_window else self.upper_barrier
            return (Hdn is not None and abs(S - Hdn) <= tol) or (Hup is not None and abs(S - Hup) <= tol)

        self.spot = s0 + ds; up = self.price()
        self.spot = s0 - ds; dn = self.price()
        self.spot = s0

        if self.use_continuous_window and near_barrier(s0):
            delta = (base_px - dn) / ds
        else:
            delta = (up - dn) / (2 * ds)
        gamma = (up - 2 * base_px + dn) / (ds * ds)

        sig0 = self.sigma
        self.sigma = sig0 + abs_vol_bump; upv = self.price()
        self.sigma = sig0 - abs_vol_bump; dnv = self.price()
        self.sigma = sig0
        vega = (upv - dnv) / (2 * abs_vol_bump)

        # restore trade scaling
        self.direction = save_dir
        self.quantity = save_qty
        self.contract_multiplier = save_mult

        sign = 1.0 if self.direction == "long" else -1.0
        scale = sign * self.quantity * self.contract_multiplier
        return {"delta": scale * float(delta), "gamma": scale * float(gamma), "vega": scale * float(vega)}

    # =============================================================================
    # Details printer
    # =============================================================================
    def print_details(self) -> None:
        info = {
            "Trade ID": self.trade_id,
            "Direction": self.direction,
            "Quantity": self.quantity,
            "Contract Multiplier": self.contract_multiplier,
            "Option Type": self.option_type,
            "Barrier Type": self.barrier_type,
            "Lower Barrier": self.lower_barrier if self.lower_barrier is not None else "-",
            "Upper Barrier": self.upper_barrier if self.upper_barrier is not None else "-",
            "Rebate Amount": self.rebate_amount,
            "Rebate Timing (in/out)": f"{self.rebate_timing_in} / {self.rebate_timing_out}",
            "Barrier Status": self.barrier_status,
            "Spot (S0)": self.spot,
            "Strike (K)": self.strike,
            "Valuation Date": self.valuation_date.date().isoformat(),
            "Maturity Date": self.maturity_date.date().isoformat(),
            "T (years)": self.tenor_years,
            "Volatility (sigma)": self.sigma,
            "Flat r (cont)": self.flat_rate_r,
            "Flat q (cont)": self.flat_dividend_q,
            "Carry (b=r-q)": self.flat_carry_b,
            "Time steps (M)": self.time_steps,
            "Space nodes (N)": self.space_nodes,
            "Rannacher steps": self.rannacher_steps,
            "Monitoring dates (#)": len([d for d in self.monitoring_dates if self.valuation_date < d <= self.maturity_date]),
            "Use continuous window?": self.use_continuous_window,
            "BGK Lower / Upper": f"{self.bgk_lower_barrier} / {self.bgk_upper_barrier}",
            "Decision (n_lim)": f"n={self.n_desired_for_decision}, n_min={self.n_min_steps_per_interval}, n_lim={self.n_lim_multiplier}",
        }
        print("==== Discrete Barrier Option (Hybrid Analytic + CN) ====")
        for k, v in info.items():
            if isinstance(v, float):
                print(f"{k:28s}: {v:.10g}")
            else:
                print(f"{k:28s}: {v}")
        px = self.price()
        greeks = self.greeks()
        print(f"\nPrice : {px:.10g}")
        print(f"Greeks: { {k: float(f'{v:.10g}') for k, v in greeks.items()} }")
