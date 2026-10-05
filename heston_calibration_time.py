"""
Time-dependent (piecewise-constant) Heston calibration — bootstrap over maturity
buckets, per Elices (2008) "Models with time-dependent parameters using transform
methods: application to Heston's model" (arXiv:0708.2020), §II-IV, combined with
the calibration objective / weights / optimisers of Moodley (2005) already used in
heston_calibration.py.

THIS FILE DOES NOT MODIFY heston_calibration.py. It only imports from it (the
constant-parameter pricer, MarketData loader, numerical integration rules, implied
vol machinery). The constant-Heston codebase is untouched and keeps working exactly
as before; this module is additive.

WHAT'S NEW HERE
----------------
HestonBucket / HestonTDParams
    One shared v0 (today's variance) + N buckets, each carrying its OWN
    (kappa, theta, sigma, rho) over a maturity interval [T_start, T_end].
    5 constant-model parameters -> 1 + 4*N time-dependent parameters.

heston_f_td / heston_call_td / heston_vols_td
    The piecewise characteristic function and the pricers built on it. Same
    Fourier-inversion machinery as the constant model (heston_P / the same
    numerical integration rules), just fed a characteristic function that is
    built by RECURSION through the buckets instead of one closed form.

    The recursion solves Heston's Riccati ODE for D(tau) and C(tau) one bucket
    at a time, walking backwards from the longest maturity to today -- exactly
    Elices's bootstrap construction (his Fig. 2 / eq. 16), implemented here via
    a self-contained, closed-form generalisation of the SAME Riccati solution
    heston_calibration.py already uses for the constant model (see the long
    comment above heston_f_td for the derivation). Set every bucket to the same
    (kappa, theta, sigma, rho) and heston_f_td reduces EXACTLY to heston_f --
    that identity is exercised directly in test_constant_vs_time_dependent.py.

BootstrapCalibrator
    Sequential, per-bucket calibration (Elices Sec. IV): fit v0 + bucket 1 to
    the shortest maturity's smile, freeze it, fit bucket 2 to the next
    maturity's smile with bucket 1 frozen, and so on. Each bucket is a small
    4-parameter problem, solved with the same local/global optimisers as the
    constant Calibrator ("solver" = SLSQP, "asa" = adaptive simulated
    annealing, "asa+solver" = both) and the same weighting schemes
    ("spread" / "equal" / "downside"). The Feller condition
    2*kappa_i*theta_i > sigma_i^2 is enforced PER BUCKET, not just once
    globally -- see Elices Sec. V and Benhamou-Gobet-Miri (2010) Assumption (P),
    both of which require this pointwise, not only on average.

ONE SIMPLIFICATION vs heston_calibration.py: only the "stable" (Albrecher et al.
2007) characteristic-function branch is implemented here, not the "paper"
(original Heston 1993) branch -- "stable" is already the recommended production
choice in run_calibration.py's own comments, and supporting both would double
the algebra above with no benefit.

Usage
    python heston_calibration_time.py --source prices --method asa+solver
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize

from heston_calibration import (
    I, DEFAULT_BOUNDS, NAN_PENALTY, DOWNSIDE_WEIGHT,
    MarketData, HestonParams, heston_vols, heston_call,
    EXAMPLE_S0, EXAMPLE_PRICES, EXAMPLE_SVI,
)

_GL_X, _GL_W = np.polynomial.legendre.leggauss(128)
PHI_MIN, PHI_MAX = 1e-8, 100.0


# =============================================================================
# PARAMETERS
# =============================================================================
@dataclass
class HestonBucket:
    T_start: float
    T_end: float
    kappa: float
    theta: float
    sigma: float
    rho: float

    def feller(self) -> float:
        return 2 * self.kappa * self.theta - self.sigma ** 2

    def as_array(self) -> np.ndarray:
        return np.array([self.kappa, self.theta, self.sigma, self.rho])

    def __str__(self) -> str:
        return (f"[{self.T_start:.4f}-{self.T_end:.4f}] kappa={self.kappa:.4f} "
                f"theta={self.theta:.4f} sigma={self.sigma:.4f} rho={self.rho:.4f} "
                f"(Feller 2kt-s^2={self.feller():.4f})")


@dataclass
class HestonTDParams:
    v0: float
    buckets: list  # list[HestonBucket], sorted, contiguous, starting at T_start=0

    def bucket_index(self, T: float) -> int:
        """Index of the bucket containing maturity T (T must be <= last T_end)."""
        for idx, b in enumerate(self.buckets):
            if T <= b.T_end + 1e-12:
                return idx
        raise ValueError(f"T={T} is beyond the last bucket end "
                          f"({self.buckets[-1].T_end})")

    def feller_ok(self) -> bool:
        return all(b.feller() > 0 for b in self.buckets)

    def param_path_table(self) -> str:
        lines = [f"{'bucket':>6} {'T_start':>8} {'T_end':>8} {'kappa':>8} "
                 f"{'theta':>8} {'sigma':>8} {'rho':>8} {'Feller':>10}"]
        for i, b in enumerate(self.buckets, 1):
            lines.append(f"{i:>6} {b.T_start:>8.4f} {b.T_end:>8.4f} {b.kappa:>8.4f} "
                         f"{b.theta:>8.4f} {b.sigma:>8.4f} {b.rho:>8.4f} "
                         f"{b.feller():>10.4f}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"v0": self.v0, "buckets": [b.__dict__ for b in self.buckets]}

    @classmethod
    def from_dict(cls, d: dict) -> "HestonTDParams":
        return cls(d["v0"], [HestonBucket(**b) for b in d["buckets"]])

    def __str__(self) -> str:
        return f"v0={self.v0:.4f}\n" + self.param_path_table()


def save_td_params(td: HestonTDParams, path: str, S0=None, r=None, q=None, **extra):
    out = {**td.to_dict(), "S0": S0, "r": r, "q": q, **extra}
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=float)


def load_td_params(path: str) -> tuple:
    with open(path) as fh:
        d = json.load(fh)
    return HestonTDParams.from_dict(d), d


# =============================================================================
# PIECEWISE CHARACTERISTIC FUNCTION  (the one new piece of maths)
# =============================================================================
# Within a single bucket, Heston's D(tau) solves the Riccati ODE
#     dD/dtau = (sigma^2/2) * (D - r1) * (D - r2)
# where r1 = (bm-d)/s2, r2 = (bm+d)/s2 are the two equilibrium roots (bm, d, s2
# exactly as in heston_calibration.heston_f). heston_f's own closed form is the
# D(0)=0 special case of this ODE. Solving it for a GENERAL boundary D(0)=D0
# (separate variables, standard Riccati-with-constant-coefficients solution,
# verified to reduce to heston_f's own formula when D0=0) gives:
#
#     K      = (D0 - r1) / (D0 - r2)
#     D(tau) = (r1 - r2*K*exp(-d*tau)) / (1 - K*exp(-d*tau))
#     C(tau) = C0 + (r-q)*phi*i*tau + a*r1*tau
#              - a*(2/s2)*ln( (1 - K*exp(-d*tau)) / (1 - K) )
#
# where a = kappa*theta (this bucket's own). Chaining buckets backward from
# maturity to today -- feeding each bucket's own (D(tau), C(tau)) in as the next
# (earlier) bucket's (D0, C0) -- is exactly Elices's recursive construction
# (his eq. 16/22), done here in closed form bucket by bucket rather than by
# composing whole-horizon characteristic functions.
def heston_f_td(phi, td: HestonTDParams, S0, T, r, q, j):
    phi = np.asarray(phi, dtype=complex)
    u = 0.5 if j == 1 else -0.5

    relevant = [b for b in td.buckets if b.T_start < T - 1e-12]
    if not relevant:
        raise ValueError(f"No bucket covers maturity T={T}")

    D = np.zeros_like(phi)
    C = np.zeros_like(phi)
    for b in reversed(relevant):
        tau = min(b.T_end, T) - b.T_start
        if tau <= 0:
            continue
        bcoef = (b.kappa - b.rho * b.sigma) if u == 0.5 else b.kappa
        a = b.kappa * b.theta
        s2 = b.sigma ** 2
        bm = bcoef - b.rho * b.sigma * phi * I
        d = np.sqrt(bm ** 2 - s2 * (2 * u * phi * I - phi ** 2))
        r1 = (bm - d) / s2
        r2 = (bm + d) / s2
        with np.errstate(all="ignore"):
            K = (D - r1) / (D - r2)
        e = np.exp(-d * tau)
        D_new = (r1 - r2 * K * e) / (1 - K * e)
        logterm = np.log((1 - K * e) / (1 - K))
        C_new = C + (r - q) * phi * I * tau + a * r1 * tau - a * (2.0 / s2) * logterm
        D, C = D_new, C_new

    return np.exp(C + D * td.v0 + I * phi * np.log(S0))


def heston_P_td(td, S0, K, T, r, q, j, tol=1e-6):
    """Fourier inversion of heston_f_td, via 128-point Gauss-Legendre on
    [PHI_MIN, PHI_MAX] -- same rule and same bounds as heston_call's "legendre"
    integration option."""
    lnK = math.log(K)
    half = 0.5 * (PHI_MAX - PHI_MIN)
    phi = 0.5 * (PHI_MAX + PHI_MIN) + half * _GL_X
    w = half * _GL_W
    f = heston_f_td(phi, td, S0, T, r, q, j)
    integ = np.real(np.exp(-I * phi * lnK) * f / (I * phi))
    return 0.5 + float(np.sum(integ * w)) / math.pi


def heston_call_td(td: HestonTDParams, S0, K, T, r=0.0, q=0.0) -> np.ndarray:
    """European call(s). Loops per option (vectorised over the 128 phi nodes
    within each option) since different options touch different buckets."""
    S0, K, T, r, q = np.broadcast_arrays(*[np.atleast_1d(np.asarray(x, float))
                                           for x in (S0, K, T, r, q)])
    out = np.empty(T.shape)
    for idx in range(T.size):
        P1 = heston_P_td(td, S0[idx], K[idx], T[idx], r[idx], q[idx], 1)
        P2 = heston_P_td(td, S0[idx], K[idx], T[idx], r[idx], q[idx], 2)
        out[idx] = S0[idx] * math.exp(-q[idx] * T[idx]) * P1 - K[idx] * math.exp(-r[idx] * T[idx]) * P2
    return out


def heston_put_td(td, S0, K, T, r=0.0, q=0.0) -> np.ndarray:
    call = heston_call_td(td, S0, K, T, r, q)
    S0, K, T, r, q = (np.asarray(x, float) for x in (S0, K, T, r, q))
    return call - S0 * np.exp(-q * T) + K * np.exp(-r * T)


def heston_vols_td(td, S0, K, T, r=0.0, q=0.0) -> np.ndarray:
    """Black-Scholes implied vols of the time-dependent Heston prices."""
    from heston_calibration import implied_vol
    S0, K, T, r, q = np.broadcast_arrays(*[np.atleast_1d(np.asarray(x, float))
                                           for x in (S0, K, T, r, q)])
    with np.errstate(all="ignore"):
        call = heston_call_td(td, S0, K, T, r, q)
        F = S0 * np.exp((r - q) * T)
        cp = np.where(K >= F, 1.0, -1.0)
        price = np.where(cp > 0, call, call - S0 * np.exp(-q * T) + K * np.exp(-r * T))
        return implied_vol(price, S0, K, T, r, q, cp)


# =============================================================================
# BOOTSTRAP CALIBRATOR
# =============================================================================
BUCKET_BOUNDS = DEFAULT_BOUNDS[:4]      # kappa, theta, sigma, rho (v0 is separate)


@dataclass
class BucketFit:
    T: float
    n_evals: int
    seconds: float
    sse: float


@dataclass
class BootstrapResult:
    td: HestonTDParams
    market: MarketData
    method: str
    objective: str
    weights: str
    bucket_fits: list = field(default_factory=list)

    def model_vols(self) -> np.ndarray:
        m = self.market
        out = np.empty_like(m.vol)
        for Ti in np.unique(m.T):
            sel = m.T == Ti
            out[sel] = heston_vols_td(self.td, m.S0, m.K[sel], Ti,
                                      np.median(m.r[sel]), np.median(m.q[sel]))
        return out

    def in_band(self):
        """Per quote: is the model vol inside the bid-offer vol band? None without
        bid/offer quotes -- mirrors heston_calibration.CalibrationResult.in_band()."""
        m = self.market
        if m.vol_bid is None or m.vol_ask is None:
            return None
        mv = self.model_vols()
        lo = np.where(np.isfinite(m.vol_bid), m.vol_bid, 0.0)
        hi = np.where(np.isfinite(m.vol_ask), m.vol_ask, np.inf)
        return (mv >= lo) & (mv <= hi)

    def report(self):
        """One section per bucket: that bucket's own (kappa, theta, sigma, rho, Feller),
        then the per-quote fit table for every maturity the bucket was calibrated to --
        same granularity as heston_calibration.CalibrationResult.report()."""
        m = self.market
        mv = self.model_vols()
        inside = self.in_band()
        print(f"\nTime-dependent Heston ({self.method}, objective={self.objective}, "
              f"weights={self.weights})")
        print(f"v0 = {self.td.v0:.4f}   Feller satisfied in every bucket: {self.td.feller_ok()}")
        for Ti, b, bf in zip(np.unique(m.T), self.td.buckets, self.bucket_fits):
            sel = np.where(m.T == Ti)[0]
            sel = sel[np.argsort(m.K[sel])]
            rmse = float(np.sqrt(np.nanmean((mv[sel] - m.vol[sel]) ** 2)))
            print(f"\nBucket [{b.T_start:.4f}, {b.T_end:.4f}]  (T={Ti:.4f}, {len(sel)} quotes, "
                 f"{bf.n_evals} evals, {bf.seconds:.1f}s)")
            print(f"  {b}")
            print(f"  SSE = {bf.sse:.6e}   unweighted RMSE = {100 * rmse:.3f} vol pts")
            if inside is None:
                print(f"  {'K':>10} {'mkt vol':>9} {'model':>9} {'diff(pts)':>10}")
                for i in sel:
                    print(f"  {m.K[i]:10.2f} {100 * m.vol[i]:8.3f}% {100 * mv[i]:8.3f}% "
                         f"{100 * (mv[i] - m.vol[i]):10.3f}")
            else:
                print(f"  Model vol inside the bid-offer band: {inside[sel].sum()} of {len(sel)}")
                print(f"  {'K':>10} {'bid vol':>9} {'mid vol':>9} {'ask vol':>9} {'model':>9} "
                     f"{'diff(pts)':>10} {'in band':>8}")
                for i in sel:
                    print(f"  {m.K[i]:10.2f} {100 * m.vol_bid[i]:8.3f}% {100 * m.vol[i]:8.3f}% "
                         f"{100 * m.vol_ask[i]:8.3f}% {100 * mv[i]:8.3f}% "
                         f"{100 * (mv[i] - m.vol[i]):10.3f} {'yes' if inside[i] else 'NO':>8}")
        overall = float(np.sqrt(np.nanmean((mv - m.vol) ** 2)))
        print(f"\nOverall unweighted RMSE = {100 * overall:.3f} vol pts")

    def save(self, path="heston_td_params.json"):
        m = self.market
        save_td_params(self.td, path, S0=m.S0, r=float(np.median(m.r)),
                       q=float(np.median(m.q)), method=self.method,
                       objective=self.objective, weights=self.weights)
        print(f"Saved time-dependent parameters to {path}")

    def plot(self, save=None, show=True):
        """One skew panel per maturity (bid-offer vol band if available, market mid
        vols, the smooth 'Heston Skew' curve and 'Heston Quotes' -- the model's own
        vol at each traded strike) -- same layout as
        heston_calibration.CalibrationResult.plot() -- plus a smoothed parameter-path
        panel (PCHIP through the bucket midpoints, not a jagged point-to-point line)."""
        import matplotlib.pyplot as plt
        m = self.market
        Ts = np.unique(m.T)
        mv = self.model_vols()
        inside = self.in_band()
        n_pan = len(Ts) + 1
        ncols = 2 if n_pan <= 4 else 3
        nrows = math.ceil(n_pan / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.6 * nrows), squeeze=False)
        axes = axes.ravel()
        c_band, c_mkt, c_mod, c_out = "#9aa5b1", "#1f2933", "#1f77b4", "#d62728"
        for ax, T in zip(axes, Ts):
            sel = np.where(m.T == T)[0]
            sel = sel[np.argsort(m.K[sel])]
            K = m.K[sel]
            if inside is not None:
                ax.fill_between(K, 100 * np.nan_to_num(m.vol_bid[sel]), 100 * m.vol_ask[sel],
                                color=c_band, alpha=0.30, lw=0, label="bid-offer vol band")
                ax.plot(K, 100 * m.vol_bid[sel], color=c_band, lw=0.8)
                ax.plot(K, 100 * m.vol_ask[sel], color=c_band, lw=0.8)
                ax.vlines(K, 100 * np.nan_to_num(m.vol_bid[sel]), 100 * m.vol_ask[sel],
                          color=c_band, lw=1.0)
            ax.plot(K, 100 * m.vol[sel], "o", color=c_mkt, ms=5, label="market mid vol")
            Kg = np.linspace(K.min(), K.max(), 80)
            hv = heston_vols_td(self.td, m.S0, Kg, T, np.median(m.r[sel]), np.median(m.q[sel]))
            ax.plot(Kg, 100 * hv, color=c_mod, lw=2, label="Heston Skew")
            mvs = 100 * mv[sel]
            if inside is None:
                ax.plot(K, mvs, "D", color=c_mod, ms=5, label="Heston Quotes")
                title = f"T = {T:.4f}"
            else:
                ok = inside[sel]
                ax.plot(K[ok], mvs[ok], "D", color=c_mod, ms=5, label="Heston Quotes (in band)")
                ax.plot(K[~ok], mvs[~ok], "D", color=c_out, ms=6, label="Heston Quotes (outside)")
                title = f"T = {T:.4f}   {ok.sum()} of {len(sel)} inside the band"
            rm = 100 * math.sqrt(np.nanmean((mv[sel] - m.vol[sel]) ** 2))
            ax.axvline(m.S0, color="k", ls=":", lw=0.8)
            ax.set_title(f"{title}   RMSE {rm:.2f} pts", fontsize=10)
            ax.set_xlabel("Strike"); ax.set_ylabel("Implied vol (%)")
            ax.grid(alpha=0.25); ax.legend(fontsize=8)
        ax = axes[len(Ts)]
        mids = np.array([0.5 * (b.T_start + b.T_end) for b in self.td.buckets])
        colors = {"kappa": "#1f77b4", "theta": "#ff7f0e", "sigma": "#2ca02c", "rho": "#d62728"}
        order = np.argsort(mids)
        mids_sorted = mids[order]
        if len(mids_sorted) >= 3:
            from scipy.interpolate import PchipInterpolator
            grid = np.linspace(mids_sorted[0], mids_sorted[-1], 200)
        for name, color in colors.items():
            vals = np.array([getattr(b, name) for b in self.td.buckets])[order]
            if len(mids_sorted) >= 3:
                smooth = PchipInterpolator(mids_sorted, vals)(grid)
                ax.plot(grid, smooth, "-", color=color, lw=2, label=name)
                ax.plot(mids_sorted, vals, "o", color=color, ms=5)
            else:
                # not enough buckets for a spline (needs >=3 points) -- straight segments
                ax.plot(mids_sorted, vals, "o-", color=color, lw=2, label=name)
        ax.set_xlabel("Bucket midpoint (years)"); ax.set_ylabel("Parameter value")
        ax.set_title("Parameter path across buckets (PCHIP-smoothed)", fontsize=10)
        ax.grid(alpha=0.25); ax.legend(fontsize=8)
        for ax in axes[n_pan:]:
            ax.set_visible(False)
        fig.suptitle(f"Time-dependent Heston calibration   v0={self.td.v0:.4f}   |   "
                     f"unweighted RMSE {100 * math.sqrt(np.nanmean((mv - m.vol) ** 2)):.2f} vol pts",
                     fontsize=11)
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=130)
            print(f"Saved figure to {save}")
        if show:
            plt.show()
        return fig


class BootstrapCalibrator:
    def __init__(self, market: MarketData, feller=True, bounds=BUCKET_BOUNDS,
                 weights="spread", objective="vol"):
        if objective not in ("vol", "price"):
            raise ValueError("objective must be 'vol' or 'price'")
        if objective == "price" and market.price is None:
            raise ValueError("objective='price' needs market prices")
        self.m = market
        self.feller = feller
        self.lb, self.ub = np.asarray(bounds, float).T
        self.objective = objective
        self.weights_name = weights
        self.maturities = np.unique(market.T)

    # ---- per-maturity weights, same formulas as Calibrator._weights --------
    def _weights(self, sel: np.ndarray) -> np.ndarray:
        m = self.m
        n = int(sel.sum())
        if self.weights_name == "equal":
            w = np.ones(n)
        elif self.weights_name == "downside":
            w = np.where(m.K[sel] <= m.S0, DOWNSIDE_WEIGHT, 1.0)
        elif self.weights_name == "spread":
            if self.objective == "price":
                if m.bid is None or m.ask is None:
                    w = np.ones(n)
                else:
                    w = 1.0 / np.abs(m.ask[sel] - m.bid[sel])
            else:
                band = m.vol_band()
                w = np.ones(n) if band is None else 1.0 / band[sel] ** 2
        else:
            raise ValueError("weights must be 'spread', 'equal' or 'downside'")
        return w if self.objective == "price" else w / w.mean()

    def _bucket_feasible(self, x) -> bool:
        x = np.asarray(x, float)
        in_box = np.all(x >= self.lb) and np.all(x <= self.ub)
        return bool(in_box and (not self.feller or 2 * x[0] * x[1] > x[2] ** 2))

    def _bucket_residuals(self, x, td_frozen: HestonTDParams, T_start, T_end,
                          sel) -> np.ndarray:
        m = self.m
        bucket = HestonBucket(T_start, T_end, *x)
        td_try = HestonTDParams(td_frozen.v0, td_frozen.buckets + [bucket])
        w = self._weights(sel)
        if self.objective == "price":
            call = heston_call_td(td_try, m.S0, m.K[sel], T_end, m.r[sel], m.q[sel])
            put = call - m.S0 * np.exp(-m.q[sel] * T_end) + m.K[sel] * np.exp(-m.r[sel] * T_end)
            model = np.where(m.cp[sel] > 0, call, put)
            with np.errstate(all="ignore"):
                res = model - m.price[sel]
            return np.sqrt(w) * np.where(np.isfinite(res), res, m.S0)
        model = heston_vols_td(td_try, m.S0, m.K[sel], T_end, m.r[sel], m.q[sel])
        res = model - m.vol[sel]
        return np.sqrt(w) * np.where(np.isfinite(res), res, NAN_PENALTY)

    def _bucket_sse(self, x, td_frozen, T_start, T_end, sel) -> float:
        return float(np.sum(self._bucket_residuals(x, td_frozen, T_start, T_end, sel) ** 2))

    def _solver_4d(self, x0, td_frozen, T_start, T_end, sel):
        fun = lambda x: self._bucket_sse(x, td_frozen, T_start, T_end, sel)
        cons = ([{"type": "ineq", "fun": lambda x: 2 * x[0] * x[1] - x[2] ** 2 - 1e-10}]
                if self.feller else [])
        sol = optimize.minimize(fun, x0, method="SLSQP", bounds=list(zip(self.lb, self.ub)),
                                constraints=cons, options={"maxiter": 500, "ftol": 1e-14})
        return sol.x, 1

    def _asa_4d(self, x0, td_frozen, T_start, T_end, sel, max_evals=3000, seed=0):
        rng = np.random.default_rng(seed)
        lb, ub, D = self.lb, self.ub, len(x0)
        x = np.asarray(x0, float)
        f = self._bucket_sse(x, td_frozen, T_start, T_end, sel)
        best_x, best_f = x.copy(), f
        n_evals = 1

        sample = []
        while len(sample) < 20:
            cand = lb + rng.random(D) * (ub - lb)
            if self._bucket_feasible(cand):
                sample.append(self._bucket_sse(cand, td_frozen, T_start, T_end, sel))
                n_evals += 1
        T0_cost = float(np.std(sample)) or 1.0
        T0_gen, T_final_gen = 1.0, 1e-8
        c = -math.log(T_final_gen / T0_gen) / max_evals ** (1.0 / D)

        for k in range(1, max_evals + 1):
            decay = math.exp(-c * k ** (1.0 / D))
            T_gen, T_cost = T0_gen * decay, T0_cost * decay
            cand = x.copy()
            for i in range(D):
                for _ in range(100):
                    u = rng.random()
                    y = math.copysign(1.0, u - 0.5) * T_gen * ((1 + 1 / T_gen) ** abs(2 * u - 1) - 1)
                    xi = x[i] + y * (ub[i] - lb[i])
                    if lb[i] <= xi <= ub[i]:
                        cand[i] = xi
                        break
            if not self._bucket_feasible(cand):
                continue
            fc = self._bucket_sse(cand, td_frozen, T_start, T_end, sel)
            n_evals += 1
            delta = fc - f
            if delta < 0 or rng.random() < math.exp(-delta / max(T_cost, 1e-300)):
                x, f = cand, fc
                if f < best_f:
                    best_x, best_f = x.copy(), f
        return best_x, n_evals

    def _initial_guess_bucket(self, sel):
        m = self.m
        atm = m.vol[sel][np.argmin(np.abs(np.log(m.K[sel] / m.S0)))]
        theta_i, kappa_i = atm ** 2, 2.0
        sigma_i = min(1.0, 0.9 * math.sqrt(2 * kappa_i * theta_i))
        return np.clip([kappa_i, theta_i, sigma_i, -0.5], self.lb, self.ub)

    def calibrate(self, method="asa+solver", max_evals=3000, seed=0,
                 verbose=True) -> BootstrapResult:
        m = self.m
        atm0 = m.vol[m.T == self.maturities[0]]
        atm0 = atm0[np.argmin(np.abs(np.log(m.K[m.T == self.maturities[0]] / m.S0)))]
        v0 = float(atm0 ** 2)
        td = HestonTDParams(v0, [])
        fits = []
        T_prev = 0.0
        for Ti in self.maturities:
            sel = m.T == Ti
            x0 = self._initial_guess_bucket(sel)
            t0 = time.perf_counter()
            if method == "solver":
                x, n_evals = self._solver_4d(x0, td, T_prev, Ti, sel)
            elif method == "asa":
                x, n_evals = self._asa_4d(x0, td, T_prev, Ti, sel, max_evals, seed)
            elif method == "asa+solver":
                x_asa, n1 = self._asa_4d(x0, td, T_prev, Ti, sel, max_evals, seed)
                x, n2 = self._solver_4d(x_asa, td, T_prev, Ti, sel)
                n_evals = n1 + n2
            else:
                raise ValueError("method must be 'solver', 'asa' or 'asa+solver'")
            seconds = time.perf_counter() - t0
            sse = self._bucket_sse(x, td, T_prev, Ti, sel)
            td.buckets.append(HestonBucket(T_prev, Ti, *[float(v) for v in x]))
            fits.append(BucketFit(Ti, n_evals, seconds, sse))
            if verbose:
                print(f"  bucket [{T_prev:.4f}, {Ti:.4f}]: {td.buckets[-1]}   "
                     f"SSE={sse:.6e}  ({n_evals} evals, {seconds:.1f}s)")
            T_prev = Ti
        return BootstrapResult(td, m, method, self.objective, self.weights_name, fits)


# =============================================================================
# EXAMPLE DATA (re-exported for the driver script)
# =============================================================================
# EXAMPLE_S0, EXAMPLE_PRICES, EXAMPLE_SVI imported above from heston_calibration.


def main():
    ap = argparse.ArgumentParser(description="Bootstrap-calibrate a time-dependent Heston")
    ap.add_argument("--source", choices=["prices", "svi", "csv"], default="prices")
    ap.add_argument("--csv")
    ap.add_argument("--S0", type=float)
    ap.add_argument("--r", type=float, default=0.0)
    ap.add_argument("--q", type=float, default=0.0)
    ap.add_argument("--method", choices=["solver", "asa", "asa+solver"], default="asa+solver")
    ap.add_argument("--objective", choices=["vol", "price"], default="vol")
    ap.add_argument("--weights", choices=["spread", "equal", "downside"], default="spread")
    ap.add_argument("--max-evals", type=int, default=3000)
    ap.add_argument("--no-feller", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="heston_td_params.json")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    if args.source == "prices":
        r, T, K, px, bid, ask = EXAMPLE_PRICES.T
        market = MarketData.from_prices(EXAMPLE_S0, T, K, px, r=r, q=0.0,
                                        option_type="call", bid=bid, ask=ask)
    elif args.source == "svi":
        market = MarketData.from_svi(100.0, EXAMPLE_SVI, r=0.05, q=0.01,
                                     strikes=np.linspace(80, 120, 9))
    else:
        if not (args.csv and args.S0):
            ap.error("--source csv needs --csv and --S0")
        market = MarketData.from_csv(args.csv, args.S0, args.r, args.q)

    print(f"{len(market)} market vols from {market.source}, "
         f"{len(np.unique(market.T))} maturity buckets")
    cal = BootstrapCalibrator(market, feller=not args.no_feller, weights=args.weights,
                              objective=args.objective)
    res = cal.calibrate(method=args.method, max_evals=args.max_evals, seed=args.seed)
    res.report()
    res.save(args.save)
    if args.plot:
        res.plot()


if __name__ == "__main__":
    main()
