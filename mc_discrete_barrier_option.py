"""Monte Carlo valuation for discretely monitored single-barrier options.

This module is meant to mirror the *event-time* intuition in RiskFlow-style
barrier implementations:

- Build a time grid that is the union of all relevant event times:
  valuation (0), dividend times, barrier monitoring times, expiry.
- Simulate the underlying only across this non-uniform grid.
- Apply the barrier rule ONLY at the monitoring times.

Model
-----
Under risk-neutral pricing we assume a lognormal GBM with *carry* b(t)
(and discount rate r(t)):

    dS_t = b(t) S_t dt + sigma(t) S_t dW_t

Payoffs are discounted with the discount curve r(t):

    PV = E[ payoff * exp(-\int_0^T r(u) du) ]

Notes
-----
1) Separate carry and discount curves are supported. In the usual equity case
   b(t) = r(t) - q(t) (continuous dividend yield), but we keep them separate
   because RiskFlow frequently does.

2) Discrete cash dividends can be modelled as *jumps* at specified times.
   If a dividend time coincides with a monitoring time, we apply the dividend
   first (default) and then test the barrier.

3) Barrier "tolerance band" is implemented to match your requirement:
   a barrier is considered breached inside a small band around the barrier.
   For example, for a *down* barrier B, we treat it as breached when
   S <= B * (1 + tol) (i.e., you knock out slightly *above* the barrier).

4) This is a *discrete* barrier pricer. If you want *continuous* monitoring,
   you would add a Brownian-bridge crossing correction inside each interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import math

import numpy as np


# ---------------------------------------------------------------------------
# Curves (continuous compounding)
# ---------------------------------------------------------------------------


class RateCurve:
    """Minimal interface for a continuously-compounded rate curve."""

    def integral(self, t0: float, t1: float) -> float:
        """Return \int_{t0}^{t1} r(u) du (continuous-compounding rate integral)."""

        raise NotImplementedError

    def df(self, t: float) -> float:
        """Discount factor exp(-\int_0^t r(u) du)."""

        return math.exp(-self.integral(0.0, t))


@dataclass(frozen=True)
class FlatRateCurve(RateCurve):
    """Constant continuously-compounded rate curve."""

    rate: float

    def integral(self, t0: float, t1: float) -> float:
        if t1 < t0:
            raise ValueError("t1 must be >= t0")
        return self.rate * (t1 - t0)


@dataclass(frozen=True)
class PiecewiseFlatRateCurve(RateCurve):
    """Piecewise-flat continuously-compounded *instantaneous* rate curve.

    Parameters
    ----------
    times : increasing sequence of knot times (> 0)
    rates : instantaneous rate on [times[i-1], times[i]) for i>=1,
            and on [0, times[0]) for i=0.

    Example
    -------
    times=[0.5, 1.0, 2.0], rates=[0.06, 0.062, 0.065]
    means:
      r(t)=0.06 for t in [0,0.5)
      r(t)=0.062 for t in [0.5,1.0)
      r(t)=0.065 for t in [1.0,2.0) and for t>=2.0

    (We extend the last rate beyond the final knot.)
    """

    times: Tuple[float, ...]
    rates: Tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.times) == 0:
            raise ValueError("times must be non-empty")
        if len(self.rates) != len(self.times):
            raise ValueError("rates must have the same length as times")
        if any(t <= 0.0 for t in self.times):
            raise ValueError("all times must be > 0")
        if any(self.times[i] <= self.times[i - 1] for i in range(1, len(self.times))):
            raise ValueError("times must be strictly increasing")

    def _rate_at(self, t: float) -> float:
        for i, ti in enumerate(self.times):
            if t < ti:
                return self.rates[i]
        return self.rates[-1]

    def integral(self, t0: float, t1: float) -> float:
        if t1 < t0:
            raise ValueError("t1 must be >= t0")
        if t1 == t0:
            return 0.0

        # Integrate piecewise across knot boundaries
        total = 0.0
        a = t0
        for i, ti in enumerate(self.times):
            b = min(t1, ti)
            if b > a:
                total += self.rates[i] * (b - a)
                a = b
            if a >= t1:
                return total

        # Beyond last knot
        total += self.rates[-1] * (t1 - a)
        return total


def make_curve(curve: Union[float, RateCurve]) -> RateCurve:
    """Helper: accept either a float (flat curve) or a RateCurve."""

    if isinstance(curve, RateCurve):
        return curve
    return FlatRateCurve(rate=float(curve))


# ---------------------------------------------------------------------------
# Dividends
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CashDividend:
    time: float  # in years from valuation
    amount: float


# ---------------------------------------------------------------------------
# Option specification
# ---------------------------------------------------------------------------


BarrierType = str  # "none", "down-and-out", "up-and-out", "down-and-in", "up-and-in"
OptionType = str   # "call" or "put"


@dataclass(frozen=True)
class DiscreteBarrierOptionSpec:
    """Contract specification for a discretely monitored single-barrier option."""

    strike: float
    expiry: float
    option_type: OptionType

    barrier_type: BarrierType
    barrier_level: Optional[float] = None

    # Discrete monitoring times (years). If None, defaults to [expiry].
    monitoring_times: Optional[Sequence[float]] = None

    # Rebate settings
    rebate: float = 0.0
    rebate_at_hit: bool = False  # True: paid at hit time; False: paid at expiry

    # "Already hit" state at valuation time
    already_hit: bool = False

    # Barrier tolerance band: max( abs_tol, level * rel_tol_bp )
    barrier_tol_bp: float = 0.0      # basis points (1bp = 0.0001) of barrier level
    barrier_abs_tol: float = 0.0     # absolute tolerance in price units


# ---------------------------------------------------------------------------
# Monte Carlo pricer
# ---------------------------------------------------------------------------


class DiscreteBarrierMonteCarloPricer:
    """Monte Carlo pricer for discretely monitored single-barrier options."""

    def __init__(
        self,
        spot: float,
        vol: float,
        discount_curve: Union[float, RateCurve],
        carry_curve: Union[float, RateCurve],
        option: DiscreteBarrierOptionSpec,
        dividends: Optional[Sequence[CashDividend]] = None,
        dividend_before_monitor: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        if spot <= 0.0:
            raise ValueError("spot must be positive")
        if vol <= 0.0:
            raise ValueError("vol must be positive")
        if option.expiry <= 0.0:
            raise ValueError("option.expiry must be positive")
        if option.strike <= 0.0:
            raise ValueError("option.strike must be positive")

        self.spot = float(spot)
        self.vol = float(vol)
        self.discount_curve = make_curve(discount_curve)
        self.carry_curve = make_curve(carry_curve)
        self.option = option
        self.dividends = list(dividends) if dividends else []
        self.dividend_before_monitor = bool(dividend_before_monitor)

        self._rng = np.random.default_rng(seed)

        self._monitor_times = self._normalize_monitor_times(option)
        self._div_times = sorted({d.time for d in self.dividends if 0.0 < d.time <= option.expiry})

        # Pre-map dividend times to amounts (support multiple dividends at same time)
        self._div_by_time = {}
        for d in self.dividends:
            if 0.0 < d.time <= option.expiry and d.amount != 0.0:
                self._div_by_time[d.time] = self._div_by_time.get(d.time, 0.0) + float(d.amount)

    @staticmethod
    def _normalize_monitor_times(option: DiscreteBarrierOptionSpec) -> List[float]:
        if option.monitoring_times is None:
            return [float(option.expiry)]
        out = sorted({float(t) for t in option.monitoring_times if 0.0 < float(t) <= option.expiry})
        if not out or abs(out[-1] - option.expiry) > 1e-14:
            out.append(float(option.expiry))
        return out

    def build_time_grid(
        self,
        extra_times: Optional[Iterable[float]] = None,
        substeps_per_interval: int = 0,
    ) -> List[float]:
        """Build a non-uniform simulation grid.

        The grid is the union of:
          - 0
          - dividend times
          - monitoring times
          - expiry
          - (optional) extra_times

        Optionally split each interval into equal substeps (substeps_per_interval).
        Barrier is still checked ONLY at monitoring times.
        """

        times = {0.0, float(self.option.expiry)}
        times.update(self._monitor_times)
        times.update(self._div_times)
        if extra_times is not None:
            for t in extra_times:
                tt = float(t)
                if 0.0 < tt < self.option.expiry:
                    times.add(tt)

        grid = sorted(times)

        if substeps_per_interval <= 0:
            return grid

        refined: List[float] = [grid[0]]
        k = int(substeps_per_interval)
        for i in range(len(grid) - 1):
            t0, t1 = grid[i], grid[i + 1]
            dt = (t1 - t0) / float(k)
            for j in range(1, k + 1):
                refined.append(t0 + j * dt)
        return refined

    # -----------------------------
    # Barrier logic with tolerance
    # -----------------------------

    def _barrier_band(self) -> float:
        opt = self.option
        if opt.barrier_level is None:
            return 0.0
        rel = (opt.barrier_tol_bp / 10000.0) * abs(opt.barrier_level)
        return max(float(opt.barrier_abs_tol), float(rel))

    def _is_barrier_hit(self, s: np.ndarray) -> np.ndarray:
        """Vectorized hit test at a monitoring time."""

        bt = self.option.barrier_type.lower()
        if bt == "none":
            return np.zeros_like(s, dtype=bool)

        B = self.option.barrier_level
        if B is None:
            raise ValueError("barrier_level must be provided for barrier options")

        band = self._barrier_band()

        if bt in ("down-and-out", "down-and-in"):
            # Conservative band: treat as hit slightly ABOVE the barrier
            return s <= (B + band)

        if bt in ("up-and-out", "up-and-in"):
            # Conservative band: treat as hit slightly BELOW the barrier
            return s >= (B - band)

        raise ValueError(f"Unsupported barrier_type: {self.option.barrier_type}")

    # -----------------------------
    # Payoff
    # -----------------------------

    def _vanilla_payoff(self, s_T: np.ndarray) -> np.ndarray:
        k = self.option.strike
        if self.option.option_type.lower() == "call":
            return np.maximum(s_T - k, 0.0)
        if self.option.option_type.lower() == "put":
            return np.maximum(k - s_T, 0.0)
        raise ValueError(f"Unsupported option_type: {self.option.option_type}")

    # -----------------------------
    # Core MC
    # -----------------------------

    def price(
        self,
        n_paths: int = 200_000,
        batch_size: int = 50_000,
        antithetic: bool = True,
        substeps_per_interval: int = 0,
        return_std_error: bool = True,
        use_in_out_parity_for_in: bool = False,
    ) -> dict:
        """Price the option.

        Parameters
        ----------
        n_paths : number of Monte Carlo paths
        batch_size : paths simulated per batch (controls memory)
        antithetic : whether to use antithetic variates
        substeps_per_interval : optional refinement of the event-time grid
        return_std_error : include standard error and 95% CI
        use_in_out_parity_for_in : if True, compute KI as vanilla - KO
                                 (only correct under the same rebate convention
                                  that your FDM code assumes).

        Returns
        -------
        dict with at least: price
        optionally: std_error, ci95_low, ci95_high
        """

        bt = self.option.barrier_type.lower()

        # Handle "already hit" state
        if bt in ("down-and-out", "up-and-out") and self.option.already_hit:
            pv = self._rebate_pv(hit_time=0.0)
            return {"price": pv, "std_error": 0.0, "ci95_low": pv, "ci95_high": pv}
        if bt in ("down-and-in", "up-and-in") and self.option.already_hit:
            # already-in -> becomes vanilla
            vanilla = self._price_vanilla_mc(
                n_paths=n_paths,
                batch_size=batch_size,
                antithetic=antithetic,
                substeps_per_interval=substeps_per_interval,
                return_std_error=return_std_error,
            )
            return vanilla

        if bt in ("down-and-in", "up-and-in") and use_in_out_parity_for_in:
            # KI = Vanilla - KO, assuming same rebate convention.
            vanilla = self._price_vanilla_mc(
                n_paths=n_paths,
                batch_size=batch_size,
                antithetic=antithetic,
                substeps_per_interval=substeps_per_interval,
                return_std_error=True,
            )
            ko_spec = DiscreteBarrierOptionSpec(
                strike=self.option.strike,
                expiry=self.option.expiry,
                option_type=self.option.option_type,
                barrier_type=("down-and-out" if bt == "down-and-in" else "up-and-out"),
                barrier_level=self.option.barrier_level,
                monitoring_times=self.option.monitoring_times,
                rebate=self.option.rebate,
                rebate_at_hit=self.option.rebate_at_hit,
                already_hit=self.option.already_hit,
                barrier_tol_bp=self.option.barrier_tol_bp,
                barrier_abs_tol=self.option.barrier_abs_tol,
            )
            ko_pricer = DiscreteBarrierMonteCarloPricer(
                spot=self.spot,
                vol=self.vol,
                discount_curve=self.discount_curve,
                carry_curve=self.carry_curve,
                option=ko_spec,
                dividends=self.dividends,
                dividend_before_monitor=self.dividend_before_monitor,
                seed=None,
            )
            ko = ko_pricer.price(
                n_paths=n_paths,
                batch_size=batch_size,
                antithetic=antithetic,
                substeps_per_interval=substeps_per_interval,
                return_std_error=True,
            )

            price = vanilla["price"] - ko["price"]

            if not return_std_error:
                return {"price": price}

            # Conservative CI: add standard errors (upper bound). If you want
            # tighter bounds, use common random numbers and estimate covariance.
            se = math.sqrt(vanilla["std_error"] ** 2 + ko["std_error"] ** 2)
            ci_lo = price - 1.96 * se
            ci_hi = price + 1.96 * se
            return {"price": price, "std_error": se, "ci95_low": ci_lo, "ci95_high": ci_hi}

        # Otherwise: direct simulation for the actual barrier type.
        return self._price_barrier_mc(
            n_paths=n_paths,
            batch_size=batch_size,
            antithetic=antithetic,
            substeps_per_interval=substeps_per_interval,
            return_std_error=return_std_error,
        )

    def _rebate_pv(self, hit_time: float) -> float:
        if self.option.rebate <= 0.0:
            return 0.0
        if self.option.rebate_at_hit:
            return self.option.rebate * self.discount_curve.df(hit_time)
        return self.option.rebate * self.discount_curve.df(self.option.expiry)

    def _price_vanilla_mc(
        self,
        n_paths: int,
        batch_size: int,
        antithetic: bool,
        substeps_per_interval: int,
        return_std_error: bool,
    ) -> dict:
        # Vanilla = barrier_type "none" on the same dynamics.
        vanilla_spec = DiscreteBarrierOptionSpec(
            strike=self.option.strike,
            expiry=self.option.expiry,
            option_type=self.option.option_type,
            barrier_type="none",
            monitoring_times=[self.option.expiry],
        )
        pricer = DiscreteBarrierMonteCarloPricer(
            spot=self.spot,
            vol=self.vol,
            discount_curve=self.discount_curve,
            carry_curve=self.carry_curve,
            option=vanilla_spec,
            dividends=self.dividends,
            dividend_before_monitor=self.dividend_before_monitor,
            seed=None,
        )
        return pricer._price_barrier_mc(
            n_paths=n_paths,
            batch_size=batch_size,
            antithetic=antithetic,
            substeps_per_interval=substeps_per_interval,
            return_std_error=return_std_error,
        )

    def _price_barrier_mc(
        self,
        n_paths: int,
        batch_size: int,
        antithetic: bool,
        substeps_per_interval: int,
        return_std_error: bool,
    ) -> dict:
        grid = self.build_time_grid(substeps_per_interval=substeps_per_interval)
        monitors = set(self._monitor_times)
        divs = set(self._div_times)

        # Precompute integrals per time-step (deterministic)
        dt = np.diff(np.array(grid))
        if np.any(dt <= 0.0):
            raise ValueError("Time grid must be strictly increasing")

        r_int = np.array([self.discount_curve.integral(grid[i], grid[i + 1]) for i in range(len(grid) - 1)])
        b_int = np.array([self.carry_curve.integral(grid[i], grid[i + 1]) for i in range(len(grid) - 1)])

        var_int = (self.vol ** 2) * dt
        if np.any(var_int < 0.0):
            raise ValueError("Negative integrated variance")

        drift = b_int - 0.5 * var_int
        diff_scale = np.sqrt(var_int)

        # Monte Carlo
        n_paths = int(n_paths)
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")

        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        # Online aggregation for mean + variance (memory bounded)
        total_sum = 0.0
        total_sumsq = 0.0
        total_n = 0

        # We compute discounting using curve df at hit/expiry; note that with
        # time-dependent rates this is path-independent.
        df_T = self.discount_curve.df(self.option.expiry)

        remaining = n_paths
        while remaining > 0:
            m = min(batch_size, remaining)
            remaining -= m

            use_antithetic = antithetic
            if use_antithetic and m < 2:
                # If we're down to a tiny tail batch, fall back to plain MC.
                use_antithetic = False

            if use_antithetic:
                # enforce even batch for pairing
                if m % 2 == 1:
                    # keep the batch even; leave the final odd path for a later batch
                    m -= 1
                    remaining += 1
                if m == 0:
                    continue
                half = m // 2

            s = np.full(m, self.spot, dtype=float)

            bt = self.option.barrier_type.lower()
            is_ko = bt in ("down-and-out", "up-and-out")
            is_ki = bt in ("down-and-in", "up-and-in")

            knocked = np.zeros(m, dtype=bool)
            hit = np.zeros(m, dtype=bool)
            pv_if_hit = np.zeros(m, dtype=float)  # used when rebate_at_hit

            alive = np.ones(m, dtype=bool)

            for i in range(len(grid) - 1):
                if not np.any(alive):
                    break

                # Generate shocks
                if use_antithetic:
                    z_half = self._rng.standard_normal(half)
                    z = np.concatenate([z_half, -z_half])
                else:
                    z = self._rng.standard_normal(m)

                # Update only alive paths
                inc = drift[i] + diff_scale[i] * z
                s[alive] *= np.exp(inc[alive])

                t_next = grid[i + 1]
                is_div = t_next in divs
                is_mon = t_next in monitors

                # If dividend and monitor coincide, you may want the barrier to
                # be checked on the pre-div or post-div level. Default is
                # post-div (dividend_before_monitor=True).

                if is_mon and (not self.dividend_before_monitor):
                    hit_now = self._is_barrier_hit(s)
                    if is_ko:
                        newly_ko = alive & (~knocked) & hit_now
                        if np.any(newly_ko):
                            knocked[newly_ko] = True
                            if self.option.rebate_at_hit:
                                pv = self.option.rebate * self.discount_curve.df(t_next)
                                pv_if_hit[newly_ko] = pv
                            alive[newly_ko] = False
                    elif is_ki:
                        hit |= (alive & hit_now)

                if is_div:
                    div_amt = self._div_by_time.get(t_next, 0.0)
                    if div_amt != 0.0:
                        s[alive] = np.maximum(s[alive] - div_amt, 1e-12)

                if is_mon and self.dividend_before_monitor:
                    hit_now = self._is_barrier_hit(s)
                    if is_ko:
                        newly_ko = alive & (~knocked) & hit_now
                        if np.any(newly_ko):
                            knocked[newly_ko] = True
                            if self.option.rebate_at_hit:
                                pv = self.option.rebate * self.discount_curve.df(t_next)
                                pv_if_hit[newly_ko] = pv
                            alive[newly_ko] = False
                    elif is_ki:
                        hit |= (alive & hit_now)

                # vanilla: nothing

            # Terminal payoff on paths that survived to expiry
            if bt == "none":
                payoff = df_T * self._vanilla_payoff(s)
            elif is_ko:
                if self.option.rebate_at_hit:
                    payoff = pv_if_hit
                    # for paths never knocked out
                    alive_to_T = ~knocked
                    payoff[alive_to_T] = df_T * self._vanilla_payoff(s[alive_to_T])
                else:
                    payoff = np.zeros(m, dtype=float)
                    alive_to_T = ~knocked
                    payoff[alive_to_T] = df_T * self._vanilla_payoff(s[alive_to_T])
                    payoff[~alive_to_T] = self.option.rebate * df_T
            elif is_ki:
                payoff = np.zeros(m, dtype=float)
                payoff[hit] = df_T * self._vanilla_payoff(s[hit])
                # If you want a "no-hit" rebate, add it here.
            else:
                raise ValueError(f"Unsupported barrier_type: {self.option.barrier_type}")

            # Online aggregation for mean/variance
            total_sum += float(np.sum(payoff))
            total_sumsq += float(np.sum(payoff * payoff))
            total_n += int(payoff.size)

        if total_n <= 0:
            raise RuntimeError("No paths were simulated (check n_paths/batch_size)")

        est = total_sum / float(total_n)

        if not return_std_error:
            return {"price": est}

        # Unbiased sample variance and standard error
        if total_n <= 1:
            se = 0.0
        else:
            var = (total_sumsq - (total_sum * total_sum) / float(total_n)) / float(total_n - 1)
            var = max(var, 0.0)
            se = math.sqrt(var / float(total_n))

        ci_lo = est - 1.96 * se
        ci_hi = est + 1.96 * se
        return {"price": est, "std_error": se, "ci95_low": ci_lo, "ci95_high": ci_hi}


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Example: Down-and-out call with monthly monitoring, 1bp tolerance band.
    spot = 100.0
    vol = 0.25

    r = FlatRateCurve(0.08)
    b = FlatRateCurve(0.08 - 0.02)  # e.g. r - q

    T = 1.0
    monitors = [i / 12.0 for i in range(1, 13)]

    opt = DiscreteBarrierOptionSpec(
        strike=100.0,
        expiry=T,
        option_type="call",
        barrier_type="down-and-out",
        barrier_level=80.0,
        monitoring_times=monitors,
        rebate=0.0,
        rebate_at_hit=False,
        barrier_tol_bp=1.0,
    )

    divs = [CashDividend(time=0.5, amount=1.0)]

    pricer = DiscreteBarrierMonteCarloPricer(
        spot=spot,
        vol=vol,
        discount_curve=r,
        carry_curve=b,
        option=opt,
        dividends=divs,
        seed=123,
    )

    res = pricer.price(n_paths=200_000, batch_size=50_000, antithetic=True)
    print(res)
