"""
Heston calibration to implied volatilities (closed-form pricer).

    Pricer       : Heston closed form  C = S e^{-qT} P1 - K e^{-rT} P2
    Integration  : "lobatto" | "simpson" | "legendre" | "scipy"
    Objective    : "vol" (below) or "price" = sum_i (C_model_i - C_market_i)^2 / |bid_i - offer_i|,
                   the objective of Moodley (2005)
                   SSE = sum_i w_i (vol_model_i - vol_market_i)^2
                   weights "spread" : w_i = 1 / (vol_ask_i - vol_bid_i)^2, scaled to mean 1
                                      (needs bid/offer prices; wide quotes count less)
                           "equal"  : w_i = 1
                           "downside" : w_i = 2 if K_i <= S0 (at or below ATM) else 1, scaled to mean 1
    Optimiser    : "solver"     - Excel-Solver style local gradient method (SLSQP)
                   "asa"        - adaptive simulated annealing (global)
                   "asa+solver" - ASA, then polished with the solver
    Market input : option prices (inverted to implied vols) or raw SVI slices

Usage
    python heston_calibration.py --source prices --method solver
    python heston_calibration.py --source svi --method asa --plot
    python heston_calibration.py --source prices --method asa+solver --save heston_params.json

Note: ASA makes thousands of objective calls, so use integration="legendre"
for it. The adaptive rules (lobatto/simpson/scipy) price one option at a time
and are best for the solver or for checking a result (--compare-integration).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict, field

import numpy as np
from scipy import optimize
from scipy.integrate import quad
from scipy.special import ndtr

I = 1j
PHI_MIN, PHI_MAX = 1e-8, 100.0          # integration range for P1, P2
_GL_X, _GL_W = np.polynomial.legendre.leggauss(128)
INTEGRATION_METHODS = ("lobatto", "simpson", "legendre", "scipy")


# =============================================================================
# PARAMETERS
# =============================================================================
@dataclass
class HestonParams:
    kappa: float    # mean-reversion speed of variance
    theta: float    # long-run variance
    sigma: float    # vol of vol
    rho: float      # spot/variance correlation
    v0: float       # initial variance

    def as_array(self) -> np.ndarray:
        return np.array([self.kappa, self.theta, self.sigma, self.rho, self.v0])

    @classmethod
    def from_array(cls, x) -> "HestonParams":
        return cls(*[float(v) for v in x])

    def feller(self) -> float:
        return 2 * self.kappa * self.theta - self.sigma ** 2

    def __str__(self) -> str:
        return ("kappa={kappa:.4f} theta={theta:.4f} sigma={sigma:.4f} "
                "rho={rho:.4f} v0={v0:.4f}".format(**asdict(self)))


def save_params(p: HestonParams, path: str, S0=None, r=None, q=None, **extra):
    out = {"params": asdict(p), "S0": S0, "r": r, "q": q, **extra}
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=float)


def load_params(path: str) -> tuple[HestonParams, dict]:
    with open(path) as fh:
        d = json.load(fh)
    return HestonParams(**d["params"]), d


# =============================================================================
# CLOSED-FORM HESTON PRICER
# =============================================================================
def heston_f(phi, p: HestonParams, S0, T, r, q, j, formulation="stable"):
    """Characteristic function f_j. All inputs broadcast (vectorised)."""
    phi = np.asarray(phi, dtype=complex)
    if j == 1:
        u, b = 0.5, p.kappa - p.rho * p.sigma
    else:
        u, b = -0.5, p.kappa
    a = p.kappa * p.theta
    s2 = p.sigma ** 2
    bm = b - p.rho * p.sigma * phi * I
    d = np.sqrt(bm ** 2 - s2 * (2 * u * phi * I - phi ** 2))

    if formulation == "stable":          # Albrecher et al. (2007), no branch-cut jumps
        g = (bm - d) / (bm + d)
        e = np.exp(-d * T)
        C = (r - q) * phi * I * T + a / s2 * ((bm - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
        D = (bm - d) / s2 * (1 - e) / (1 - g * e)
    elif formulation == "paper":         # Heston (1993) original form
        g = (bm + d) / (bm - d)
        e = np.exp(d * T)
        C = (r - q) * phi * I * T + a / s2 * ((bm + d) * T - 2 * np.log((1 - g * e) / (1 - g)))
        D = (bm + d) / s2 * (1 - e) / (1 - g * e)
    else:
        raise ValueError("formulation must be 'stable' or 'paper'")
    return np.exp(C + D * p.v0 + I * phi * np.log(S0))


def heston_P(p, S0, K, T, r, q, j, integration="lobatto", formulation="stable", tol=1e-6):
    """P_j = 1/2 + 1/pi * Int_0^inf Re[e^{-i phi ln K} f_j / (i phi)] d phi, one option."""
    lnK = math.log(K)

    def integrand(ph):
        ph = np.asarray(ph, dtype=float)
        f = heston_f(ph, p, S0, T, r, q, j, formulation)
        return np.real(np.exp(-I * ph * lnK) * f / (I * ph))

    if integration == "lobatto":
        val = adaptive_gauss_lobatto(integrand, PHI_MIN, PHI_MAX, tol)
    elif integration == "simpson":
        val = adaptive_simpson(integrand, PHI_MIN, PHI_MAX, tol)
    elif integration == "legendre":
        val = gauss_legendre(integrand, PHI_MIN, PHI_MAX)
    elif integration == "scipy":
        val = quad(lambda x: float(integrand(x)), PHI_MIN, PHI_MAX, limit=200, epsabs=tol)[0]
    else:
        raise ValueError(f"integration must be one of {INTEGRATION_METHODS}")
    return 0.5 + val / math.pi


def heston_call(p: HestonParams, S0, K, T, r=0.0, q=0.0, integration="legendre",
                formulation="stable", tol=1e-6) -> np.ndarray:
    """European call(s). S0, K, T, r, q may be scalars or arrays (broadcast)."""
    S0, K, T, r, q = np.broadcast_arrays(*[np.atleast_1d(np.asarray(x, float))
                                           for x in (S0, K, T, r, q)])
    if integration == "legendre":
        # every option on the same 128 nodes in one numpy call
        half = 0.5 * (PHI_MAX - PHI_MIN)
        phi = 0.5 * (PHI_MAX + PHI_MIN) + half * _GL_X[None, :]
        w = half * _GL_W[None, :]
        c = lambda x: x[:, None]
        P = []
        for j in (1, 2):
            f = heston_f(phi, p, c(S0), c(T), c(r), c(q), j, formulation)
            integ = np.real(np.exp(-I * phi * np.log(c(K))) * f / (I * phi))
            P.append(0.5 + (integ * w).sum(axis=1) / math.pi)
        P1, P2 = P
    else:
        args = list(zip(S0, K, T, r, q))
        P1 = np.array([heston_P(p, *a, 1, integration, formulation, tol) for a in args])
        P2 = np.array([heston_P(p, *a, 2, integration, formulation, tol) for a in args])
    return S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2


def heston_put(p, S0, K, T, r=0.0, q=0.0, **kw) -> np.ndarray:
    """Put by put-call parity."""
    call = heston_call(p, S0, K, T, r, q, **kw)
    S0, K, T, r, q = (np.asarray(x, float) for x in (S0, K, T, r, q))
    return call - S0 * np.exp(-q * T) + K * np.exp(-r * T)


def heston_vols(p, S0, K, T, r=0.0, q=0.0, integration="legendre", formulation="stable", tol=1e-6):
    """Black-Scholes implied vols of Heston prices (inverted on the OTM side)."""
    S0, K, T, r, q = np.broadcast_arrays(*[np.atleast_1d(np.asarray(x, float))
                                           for x in (S0, K, T, r, q)])
    with np.errstate(all="ignore"):
        call = heston_call(p, S0, K, T, r, q, integration, formulation, tol)
        F = S0 * np.exp((r - q) * T)
        cp = np.where(K >= F, 1.0, -1.0)
        price = np.where(cp > 0, call, call - S0 * np.exp(-q * T) + K * np.exp(-r * T))
        return implied_vol(price, S0, K, T, r, q, cp)


# =============================================================================
# NUMERICAL INTEGRATION
# =============================================================================
def adaptive_simpson(f, a, b, tol=1e-6, max_depth=50, panels=20):
    """Recursive adaptive Simpson with Richardson correction (MATLAB quad).
    Starts from `panels` sub-intervals so an oscillating integrand cannot fool
    the first coarse error estimate."""
    def _rec(a, b, fa, fm, fb, whole, tol, depth):
        m = 0.5 * (a + b)
        lm, rm = 0.5 * (a + m), 0.5 * (m + b)
        flm, frm = float(f(lm)), float(f(rm))
        left = (m - a) / 6.0 * (fa + 4 * flm + fm)
        right = (b - m) / 6.0 * (fm + 4 * frm + fb)
        delta = left + right - whole
        if not math.isfinite(delta):
            return float("nan")
        if depth <= 0 or abs(delta) <= 15.0 * tol:
            return left + right + delta / 15.0
        return (_rec(a, m, fa, flm, fm, left, tol / 2, depth - 1)
                + _rec(m, b, fm, frm, fb, right, tol / 2, depth - 1))

    total = 0.0
    edges = np.linspace(a, b, panels + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        fa, fb, fm = float(f(lo)), float(f(hi)), float(f(0.5 * (lo + hi)))
        total += _rec(lo, hi, fa, fm, fb, (hi - lo) / 6.0 * (fa + 4 * fm + fb),
                      tol / panels, max_depth)
    return total


def adaptive_gauss_lobatto(f, a, b, tol=1e-6):
    """Gander & Gautschi (1998) adaptlob (MATLAB quadl)."""
    eps = np.finfo(float).eps
    alpha, beta = math.sqrt(2.0 / 3.0), 1.0 / math.sqrt(5.0)
    x1, x2, x3 = 0.942882415695480, 0.641853342345781, 0.236383199662150

    m, h = 0.5 * (a + b), 0.5 * (b - a)
    x = np.array([a, m - x1 * h, m - alpha * h, m - x2 * h, m - beta * h,
                  m - x3 * h, m, m + x3 * h, m + beta * h, m + x2 * h,
                  m + alpha * h, m + x1 * h, b])
    y = np.asarray(f(x), float)
    fa, fb = y[0], y[12]
    i2 = (h / 6) * (y[0] + y[12] + 5 * (y[4] + y[8]))
    i1 = (h / 1470) * (77 * (y[0] + y[12]) + 432 * (y[2] + y[10])
                       + 625 * (y[4] + y[8]) + 672 * y[6])
    is_ = h * (0.0158271919734802 * (y[0] + y[12]) + 0.0942738402188500 * (y[1] + y[11])
               + 0.155071987336585 * (y[2] + y[10]) + 0.188821573960182 * (y[3] + y[9])
               + 0.199773405226859 * (y[4] + y[8]) + 0.224926465333340 * (y[5] + y[7])
               + 0.242611071901408 * y[6])
    s = 1.0 if is_ == 0 else math.copysign(1.0, is_)
    erri1, erri2 = abs(i1 - is_), abs(i2 - is_)
    R = erri1 / erri2 if erri2 != 0 else 1.0
    if 0 < R < 1:
        tol = tol / R
    is_ = s * abs(is_) * tol / eps
    if is_ == 0:
        is_ = b - a

    def _step(a, b, fa, fb, depth):
        h, m = 0.5 * (b - a), 0.5 * (a + b)
        mll, ml, mr, mrr = m - alpha * h, m - beta * h, m + beta * h, m + alpha * h
        fmll, fml, fm, fmr, fmrr = np.asarray(f(np.array([mll, ml, m, mr, mrr])), float)
        i2 = (h / 6) * (fa + fb + 5 * (fml + fmr))
        i1 = (h / 1470) * (77 * (fa + fb) + 432 * (fmll + fmrr) + 625 * (fml + fmr) + 672 * fm)
        if not (math.isfinite(i1) and math.isfinite(i2)):
            return float("nan")
        if (is_ + (i1 - i2) == is_) or (mll <= a) or (b <= mrr) or depth > 12:
            return i1
        return (_step(a, mll, fa, fmll, depth + 1) + _step(mll, ml, fmll, fml, depth + 1)
                + _step(ml, m, fml, fm, depth + 1) + _step(m, mr, fm, fmr, depth + 1)
                + _step(mr, mrr, fmr, fmrr, depth + 1) + _step(mrr, b, fmrr, fb, depth + 1))

    return _step(a, b, fa, fb, 0)


def gauss_legendre(f, a, b, n=128):
    x, w = (_GL_X, _GL_W) if n == 128 else np.polynomial.legendre.leggauss(n)
    xm, xr = 0.5 * (a + b), 0.5 * (b - a)
    return xr * np.sum(w * f(xm + xr * x))


# =============================================================================
# BLACK-SCHOLES AND IMPLIED VOL (vectorised)
# =============================================================================
def bs_price(S0, K, T, r, q, vol, cp=1.0):
    """cp = +1 call, -1 put."""
    F = S0 * np.exp((r - q) * T)
    sd = vol * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd ** 2) / sd
    d2 = d1 - sd
    return np.exp(-r * T) * cp * (F * ndtr(cp * d1) - K * ndtr(cp * d2))


def bs_vega(S0, K, T, r, q, vol):
    F = S0 * np.exp((r - q) * T)
    sd = vol * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd ** 2) / sd
    return np.exp(-r * T) * F * np.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi) * np.sqrt(T)


def implied_vol(price, S0, K, T, r=0.0, q=0.0, cp=1.0, lo=1e-4, hi=5.0,
                tol=1e-10, max_iter=100) -> np.ndarray:
    """Safeguarded Newton (bisection fallback). NaN where no vol in [lo, hi] fits."""
    price, S0, K, T, r, q, cp = np.broadcast_arrays(
        *[np.atleast_1d(np.asarray(x, float)) for x in (price, S0, K, T, r, q, cp)])
    with np.errstate(all="ignore"):
        valid = (np.isfinite(price) & (T > 0)
                 & (price > bs_price(S0, K, T, r, q, lo, cp))
                 & (price < bs_price(S0, K, T, r, q, hi, cp)))
        a, b = np.full(price.shape, lo), np.full(price.shape, hi)
        vol = np.full(price.shape, 0.3)
        for _ in range(max_iter):
            diff = bs_price(S0, K, T, r, q, vol, cp) - price
            if np.all(np.abs(diff[valid]) <= tol * (1.0 + np.abs(price[valid]))):
                break
            a = np.where(diff < 0, vol, a)
            b = np.where(diff > 0, vol, b)
            newton = vol - diff / bs_vega(S0, K, T, r, q, vol)
            ok = np.isfinite(newton) & (newton > a) & (newton < b)
            vol = np.where(ok, newton, 0.5 * (a + b))
    vol[~valid] = np.nan
    return vol


# =============================================================================
# SVI
# =============================================================================
def svi_total_variance(k, a, b, rho, m, sigma):
    """Raw SVI: w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2)), k = ln(K/F)."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def svi_vol(K, T, S0, r, q, svi_params):
    F = S0 * np.exp((r - q) * T)
    w = svi_total_variance(np.log(np.asarray(K, float) / F), *svi_params)
    return np.sqrt(w / T)


# =============================================================================
# MARKET DATA
# =============================================================================
def _per_T(x, T):
    return x[T] if isinstance(x, dict) else x


@dataclass
class MarketData:
    S0: float
    T: np.ndarray
    K: np.ndarray
    vol: np.ndarray            # market implied vols = calibration targets
    r: np.ndarray
    q: np.ndarray
    price: np.ndarray | None = None
    source: str = ""
    vol_bid: np.ndarray | None = None   # implied vols of the bid / offer prices
    vol_ask: np.ndarray | None = None   # (NaN where the bid is below intrinsic value)
    bid: np.ndarray | None = None       # bid / offer prices
    ask: np.ndarray | None = None
    cp: np.ndarray | None = None        # +1 call, -1 put (of the quoted prices)

    @classmethod
    def from_prices(cls, S0, T, K, price, r=0.0, q=0.0, option_type="call", bid=None, ask=None):
        """Market prices -> implied vols. option_type: 'call'/'put' (scalar or per option).
        bid, ask: optional bid / offer prices, inverted to the vol band used for weights."""
        T, K, price, r, q = np.broadcast_arrays(*[np.atleast_1d(np.asarray(x, float))
                                                  for x in (T, K, price, r, q)])
        types = np.broadcast_to(np.atleast_1d(np.asarray(option_type, dtype=str)), T.shape)
        cp = np.where(np.char.startswith(np.char.lower(types), "p"), -1.0, 1.0)
        vol = implied_vol(price, S0, K, T, r, q, cp)
        keep = np.isfinite(vol)
        if not keep.all():
            print(f"[MarketData] dropped {np.sum(~keep)} quote(s) with no implied vol "
                  f"(price outside no-arbitrage bounds): K={K[~keep]}, T={T[~keep]}")
        vol_bid = vol_ask = None
        if bid is not None and ask is not None:
            bid, ask = (np.broadcast_to(np.atleast_1d(np.asarray(x, float)), T.shape) for x in (bid, ask))
            vol_bid = implied_vol(bid, S0, K, T, r, q, cp)[keep]
            vol_ask = implied_vol(ask, S0, K, T, r, q, cp)[keep]
            bid, ask = bid[keep], ask[keep]
        return cls(float(S0), T[keep], K[keep], vol[keep], r[keep], q[keep],
                   price[keep], source="prices", vol_bid=vol_bid, vol_ask=vol_ask,
                   bid=bid, ask=ask, cp=cp[keep])

    def vol_band(self) -> np.ndarray | None:
        """Bid-offer width in vol, or None without bid/offer quotes. Where one side
        has no implied vol the width is twice the distance from the mid to the other."""
        if self.vol_bid is None or self.vol_ask is None:
            return None
        band = self.vol_ask - self.vol_bid
        band = np.where(np.isfinite(band), band, 2 * (self.vol_ask - self.vol))
        return np.where(np.isfinite(band), band, 2 * (self.vol - self.vol_bid))

    @classmethod
    def from_svi(cls, S0, slices: dict, r=0.0, q=0.0, strikes=None):
        """
        slices  : {T: (a, b, rho, m, sigma)} raw SVI per maturity
        strikes : array used for every T, or {T: array}; default S0 * [0.8 .. 1.2]
        r, q    : scalar or {T: value}
        """
        rows = []
        for T in sorted(slices):
            Ks = strikes[T] if isinstance(strikes, dict) else strikes
            Ks = S0 * np.linspace(0.8, 1.2, 9) if Ks is None else np.asarray(Ks, float)
            rT, qT = _per_T(r, T), _per_T(q, T)
            F = S0 * math.exp((rT - qT) * T)
            w = svi_total_variance(np.log(Ks / F), *slices[T])
            if np.any(w <= 0):
                raise ValueError(f"SVI slice T={T} gives non-positive total variance")
            rows += [(T, K, math.sqrt(wi / T), rT, qT) for K, wi in zip(Ks, w)]
        T, K, vol, r_, q_ = np.array(rows).T
        return cls(float(S0), T, K, vol, r_, q_, source="svi")

    @classmethod
    def from_csv(cls, path, S0, r=0.0, q=0.0):
        """CSV with header T,K,price[,bid,ask][,type][,r][,q]. Missing r/q columns use the arguments."""
        d = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        names = [n.lower() for n in d.dtype.names]
        col = lambda n: d[d.dtype.names[names.index(n)]]
        return cls.from_prices(S0, col("t"), col("k"), col("price"),
                               col("r") if "r" in names else r,
                               col("q") if "q" in names else q,
                               col("type") if "type" in names else "call",
                               bid=col("bid") if "bid" in names else None,
                               ask=col("ask") if "ask" in names else None)

    def __len__(self):
        return len(self.T)


# =============================================================================
# CALIBRATION
# =============================================================================
#                          kappa        theta        sigma        rho            v0
DEFAULT_BOUNDS = np.array([[1e-3, 20.0], [1e-4, 1.0], [1e-3, 5.0], [-0.999, 0.999], [1e-4, 1.0]])
NAN_PENALTY = 1.0        # vol error used when a model price has no implied vol
DOWNSIDE_WEIGHT = 2.0    # weights="downside": weight of quotes with K <= S0 relative to K > S0
VEGA_WEIGHT_CAP = 100.0  # weights="vega": cap 1/vega^2 at this multiple of the median weight,
                         # so one near-zero-vega (deep OTM/ITM, short-dated) quote can't
                         # numerically dominate the fit. Shared with heston_calibration_time.py's
                         # BootstrapCalibrator, which imports this rather than redefining it.


@dataclass
class CalibrationResult:
    params: HestonParams
    method: str
    integration: str
    sse: float
    n_evals: int
    seconds: float
    market: MarketData
    model_vol: np.ndarray
    history: list = field(default_factory=list)
    formulation: str = "stable"
    weights: str = "equal"
    objective: str = "vol"
    iter_history: list = field(default_factory=list)   # best objective per ASA iteration

    @property
    def rmse(self) -> float:
        """Unweighted vol RMSE over all quotes (comparable across weightings)."""
        return float(np.sqrt(np.nanmean((self.model_vol - self.market.vol) ** 2)))

    def in_band(self) -> np.ndarray | None:
        """Per quote: is the model vol inside the bid-offer vol band?"""
        m = self.market
        if m.vol_bid is None or m.vol_ask is None:
            return None
        lo = np.where(np.isfinite(m.vol_bid), m.vol_bid, 0.0)
        hi = np.where(np.isfinite(m.vol_ask), m.vol_ask, np.inf)
        return (self.model_vol >= lo) & (self.model_vol <= hi)

    def report(self):
        m = self.market
        inside = self.in_band()
        print(f"\nMethod: {self.method}  |  integration: {self.integration}  |  "
              f"formulation: {self.formulation}  |  objective: {self.objective}  |  "
              f"weights: {self.weights}  |  {self.n_evals} evals in {self.seconds:.1f}s")
        print(f"Params: {self.params}   Feller 2kt-s^2 = {self.params.feller():.4f}")
        print(f"Objective (weighted {self.objective} SSE) = {self.sse:.6e}   "
              f"unweighted RMSE = {100 * self.rmse:.3f} vol pts")
        if inside is None:
            print(f"{'T':>8} {'K':>10} {'mkt vol':>9} {'model':>9} {'diff(pts)':>10}")
            for T, K, mv, hv in zip(m.T, m.K, m.vol, self.model_vol):
                print(f"{T:8.4f} {K:10.2f} {100 * mv:8.3f}% {100 * hv:8.3f}% {100 * (hv - mv):10.3f}")
            return
        print(f"Model vol inside the bid-offer band: {inside.sum()} of {len(m)} quotes")
        print(f"{'T':>8} {'K':>10} {'bid vol':>9} {'mid vol':>9} {'ask vol':>9} {'model':>9} "
              f"{'diff(pts)':>10} {'in band':>8}")
        for i in np.lexsort((m.K, m.T)):
            print(f"{m.T[i]:8.4f} {m.K[i]:10.2f} {100 * m.vol_bid[i]:8.3f}% {100 * m.vol[i]:8.3f}% "
                  f"{100 * m.vol_ask[i]:8.3f}% {100 * self.model_vol[i]:8.3f}% "
                  f"{100 * (self.model_vol[i] - m.vol[i]):10.3f} {'yes' if inside[i] else 'NO':>8}")

    def save(self, path="heston_params.json"):
        m = self.market
        save_params(self.params, path, S0=m.S0, r=float(np.median(m.r)), q=float(np.median(m.q)),
                    method=self.method, integration=self.integration, formulation=self.formulation,
                    objective=self.objective, weights=self.weights, sse=self.sse,
                    rmse_vol=self.rmse)
        print(f"Saved parameters to {path}")

    def plot(self, save=None, show=True):
        """One skew panel per maturity (bid-offer vol band, mid vols, Heston skew and the
        Heston vol at every quoted strike) plus the optimiser convergence."""
        import matplotlib.pyplot as plt
        m = self.market
        Ts = np.unique(m.T)
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
            hv = heston_vols(self.params, m.S0, Kg, T, np.median(m.r[sel]), np.median(m.q[sel]),
                             integration="legendre" if self.formulation == "stable"
                             else self.integration, formulation=self.formulation)
            ax.plot(Kg, 100 * hv, color=c_mod, lw=2, label="Heston skew")
            mv = 100 * self.model_vol[sel]
            if inside is None:
                ax.plot(K, mv, "D", color=c_mod, ms=5, label="Heston at quotes")
                title = f"T = {T:.4f}"
            else:
                ok = inside[sel]
                ax.plot(K[ok], mv[ok], "D", color=c_mod, ms=5, label="Heston at quotes (in band)")
                ax.plot(K[~ok], mv[~ok], "D", color=c_out, ms=6, label="Heston at quotes (outside)")
                title = f"T = {T:.4f}   {ok.sum()} of {len(sel)} inside the band"
            rm = 100 * math.sqrt(np.nanmean((self.model_vol[sel] - m.vol[sel]) ** 2))
            ax.axvline(m.S0, color="k", ls=":", lw=0.8)
            ax.set_title(f"{title}   RMSE {rm:.2f} pts", fontsize=10)
            ax.set_xlabel("Strike"); ax.set_ylabel("Implied vol (%)")
            ax.grid(alpha=0.25); ax.legend(fontsize=8)
        ax = axes[len(Ts)]
        if self.iter_history:                       # ASA: one point per iteration, 1 .. MAX_EVALS
            n_it = len(self.iter_history)
            ax.semilogy(np.arange(1, n_it + 1), self.iter_history, color=c_mod)
            ax.set_xlim(0, n_it)
            ax.set_xlabel("ASA iteration")
        else:
            ax.semilogy(np.minimum.accumulate(self.history), color=c_mod)
            ax.set_xlabel("Objective evaluation")
        ax.set_ylabel(f"Best objective (weighted {self.objective} SSE)")
        ax.set_title(f"Convergence ({self.method}, objective: {self.objective}, "
                     f"weights: {self.weights})", fontsize=10)
        ax.grid(alpha=0.25, which="both")
        for ax in axes[n_pan:]:
            ax.set_visible(False)
        fig.suptitle(f"Heston calibration: {self.params}   |   unweighted RMSE "
                     f"{100 * self.rmse:.2f} vol pts", fontsize=11)
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=130)
            print(f"Saved figure to {save}")
        if show:
            plt.show()
        return fig


class Calibrator:
    def __init__(self, market: MarketData, integration="legendre", formulation="stable",
                 feller=True, bounds=DEFAULT_BOUNDS, tol=1e-6, weights="spread", objective="vol"):
        """objective: "vol"   - sum w_i (model vol - market vol)^2
                      "price" - sum w_i (model price - market price)^2, Moodley (2005) eq. 3.1
        weights: "spread" (vol: 1 / bid-offer vol band^2 scaled to mean 1; price:
        1 / |bid - offer| as in the paper; falls back to equal without bid/offer quotes),
        "equal", "downside" (DOWNSIDE_WEIGHT = 2 for strikes K <= S0, 1 above), "vega"
        (1 / BS-vega^2 off the market's own quoted mid vol -- downweights ATM, upweights
        the wings, capped at VEGA_WEIGHT_CAP x the median weight so one near-zero-vega
        quote can't dominate; opposite direction from "downside"), or an array with one
        weight per quote."""
        if integration not in INTEGRATION_METHODS:
            raise ValueError(f"integration must be one of {INTEGRATION_METHODS}")
        if objective not in ("vol", "price"):
            raise ValueError("objective must be 'vol' or 'price'")
        if objective == "price" and market.price is None:
            raise ValueError("objective='price' needs market prices (not available from SVI)")
        self.m = market
        self.integration, self.formulation, self.tol = integration, formulation, tol
        self.feller, self.objective = feller, objective
        self.lb, self.ub = np.asarray(bounds, float).T
        self.n_evals, self.history, self.iter_history = 0, [], []
        self.weights, self.w = self._weights(weights)

    def _weights(self, weights):
        """Returns (name, w). Vol weights are scaled to mean 1 so the SSE stays in vol^2
        units; price weights are left as they are so the SSE is the paper's S(Omega)."""
        n = len(self.m)
        if isinstance(weights, str):
            if weights == "equal":
                return "equal", np.ones(n)
            if weights == "downside":
                w = np.where(self.m.K <= self.m.S0, DOWNSIDE_WEIGHT, 1.0)
                return "downside", (w if self.objective == "price" else w / w.mean())
            if weights == "vega":
                vega = bs_vega(self.m.S0, self.m.K, self.m.T, self.m.r, self.m.q, self.m.vol)
                w = 1.0 / np.maximum(vega, 1e-12) ** 2
                w = np.minimum(w, VEGA_WEIGHT_CAP * np.median(w))
                return "vega", (w if self.objective == "price" else w / w.mean())
            if weights != "spread":
                raise ValueError("weights must be 'spread', 'equal', 'downside', 'vega' or an array")
            if self.objective == "price":
                if self.m.bid is None or self.m.ask is None:
                    print("[Calibrator] no bid/offer quotes: using equal weights")
                    return "equal", np.ones(n)
                spread = np.abs(self.m.ask - self.m.bid)
                if not np.all(spread > 0):
                    raise ValueError("bid-offer spread must be positive for every quote")
                return "spread", 1.0 / spread
            band = self.m.vol_band()
            if band is None:
                print("[Calibrator] no bid/offer quotes: using equal weights")
                return "equal", np.ones(n)
            if not np.all(np.isfinite(band) & (band > 0)):
                raise ValueError("bid-offer vol band must be positive for every quote")
            w, name = 1.0 / band ** 2, "spread"
        else:
            w, name = np.asarray(weights, float), "custom"
            if w.shape != (n,) or np.any(w < 0) or not w.sum() > 0:
                raise ValueError("weights array must have one non-negative weight per quote")
        return name, (w if self.objective == "price" else w / w.mean())

    # ---- objective ----------------------------------------------------------
    def model_vols(self, p: HestonParams) -> np.ndarray:
        m = self.m
        return heston_vols(p, m.S0, m.K, m.T, m.r, m.q, self.integration, self.formulation, self.tol)

    def model_prices(self, p: HestonParams) -> np.ndarray:
        """Model prices of the quoted options (calls, or puts by parity)."""
        m = self.m
        call = heston_call(p, m.S0, m.K, m.T, m.r, m.q, self.integration, self.formulation, self.tol)
        put = call - m.S0 * np.exp(-m.q * m.T) + m.K * np.exp(-m.r * m.T)
        return np.where(m.cp > 0, call, put)

    def residuals(self, p: HestonParams) -> np.ndarray:
        """sqrt(w_i) * (model - market) in vol or price, so sum(residuals^2) is the objective."""
        if self.objective == "price":
            with np.errstate(all="ignore"):
                res = self.model_prices(p) - self.m.price
            return np.sqrt(self.w) * np.where(np.isfinite(res), res, self.m.S0)
        res = self.model_vols(p) - self.m.vol
        return np.sqrt(self.w) * np.where(np.isfinite(res), res, NAN_PENALTY)

    def sse(self, x) -> float:
        p = x if isinstance(x, HestonParams) else HestonParams.from_array(x)
        val = float(np.sum(self.residuals(p) ** 2))
        self.n_evals += 1
        self.history.append(val)
        return val

    def feasible(self, x) -> bool:
        x = np.asarray(x, float)
        in_box = np.all(x >= self.lb) and np.all(x <= self.ub)
        return bool(in_box and (not self.feller or 2 * x[0] * x[1] > x[2] ** 2))

    def initial_guess(self) -> np.ndarray:
        """v0 from the shortest maturity, theta from the longest (ATM-ish vols)."""
        m = self.m
        atm = lambda T: m.vol[m.T == T][np.argmin(np.abs(np.log(m.K[m.T == T] / m.S0)))]
        v0, theta, kappa = atm(m.T.min()) ** 2, atm(m.T.max()) ** 2, 2.0
        sigma = min(0.5, 0.9 * math.sqrt(2 * kappa * theta))
        return np.clip([kappa, theta, sigma, -0.5, v0], self.lb, self.ub)

    # ---- optimisers ---------------------------------------------------------
    def calibrate(self, method="solver", x0=None, max_evals=20000, seed=0,
                  verbose=True) -> CalibrationResult:
        x0 = self.initial_guess() if x0 is None else np.asarray(
            x0.as_array() if isinstance(x0, HestonParams) else x0, float)
        if not self.feasible(x0):
            raise ValueError("x0 is outside the bounds or breaks the Feller condition")
        self.n_evals, self.history, self.iter_history = 0, [], []
        t0 = time.perf_counter()
        if method == "solver":
            x = self._solver(x0)
        elif method == "asa":
            x = self._asa(x0, max_evals, seed, verbose)
        elif method == "asa+solver":
            x = self._solver(self._asa(x0, max_evals, seed, verbose))
        else:
            raise ValueError("method must be 'solver', 'asa' or 'asa+solver'")
        p = HestonParams.from_array(x)
        return CalibrationResult(p, method, self.integration, self.sse(p), self.n_evals,
                                 time.perf_counter() - t0, self.m, self.model_vols(p),
                                 list(self.history), formulation=self.formulation,
                                 weights=self.weights, objective=self.objective,
                                 iter_history=list(self.iter_history))

    def _solver(self, x0) -> np.ndarray:
        """Excel-Solver style: local, gradient-based, bounds + Feller constraint (SLSQP)."""
        cons = ([{"type": "ineq", "fun": lambda x: 2 * x[0] * x[1] - x[2] ** 2 - 1e-10}]
                if self.feller else [])
        sol = optimize.minimize(self.sse, x0, method="SLSQP", bounds=list(zip(self.lb, self.ub)),
                                constraints=cons, options={"maxiter": 500, "ftol": 1e-14})
        return sol.x

    def _asa(self, x0, max_evals=20000, seed=0, verbose=True,
             T0_gen=1.0, T_final_gen=1e-8) -> np.ndarray:
        """Adaptive simulated annealing (Ingber): per-parameter generating
        temperatures T_i(k) = T0 exp(-c k^(1/D)) and heavy-tailed jumps."""
        rng = np.random.default_rng(seed)
        lb, ub, D = self.lb, self.ub, len(x0)
        x = np.asarray(x0, float)
        f = self.sse(x)
        best_x, best_f = x.copy(), f

        sample = []                                  # initial acceptance temperature
        while len(sample) < 20:
            cand = lb + rng.random(D) * (ub - lb)
            if self.feasible(cand):
                sample.append(self.sse(cand))
        T0_cost = float(np.std(sample)) or 1.0
        c = -math.log(T_final_gen / T0_gen) / max_evals ** (1.0 / D)

        for k in range(1, max_evals + 1):
            decay = math.exp(-c * k ** (1.0 / D))
            T_gen, T_cost = T0_gen * decay, T0_cost * decay
            cand = x.copy()
            for i in range(D):
                for _ in range(100):                 # resample until inside bounds
                    u = rng.random()
                    y = math.copysign(1.0, u - 0.5) * T_gen * ((1 + 1 / T_gen) ** abs(2 * u - 1) - 1)
                    xi = x[i] + y * (ub[i] - lb[i])
                    if lb[i] <= xi <= ub[i]:
                        cand[i] = xi
                        break
            if not self.feasible(cand):
                self.iter_history.append(best_f)
                continue
            fc = self.sse(cand)
            delta = fc - f
            if delta < 0 or rng.random() < math.exp(-delta / max(T_cost, 1e-300)):
                x, f = cand, fc
                if f < best_f:
                    best_x, best_f = x.copy(), f
                    if verbose:
                        print(f"   ASA k={k:6d}  SSE={best_f:.6e}  {HestonParams.from_array(best_x)}")
            self.iter_history.append(best_f)     # best objective after ASA iteration k
        return best_x


def compare_integration(p: HestonParams, market: MarketData, formulation="stable"):
    """Price the market options with every integration rule and compare."""
    m = market
    ref = None
    print(f"\n{'method':>10} {'time (s)':>9} {'max |price diff| vs lobatto':>28}")
    for meth in ("lobatto", "simpson", "legendre", "scipy"):
        t0 = time.perf_counter()
        px = heston_call(p, m.S0, m.K, m.T, m.r, m.q, integration=meth, formulation=formulation)
        dt = time.perf_counter() - t0
        ref = px if ref is None else ref
        print(f"{meth:>10} {dt:9.3f} {np.max(np.abs(px - ref)):28.2e}")


# =============================================================================
# EXAMPLE DATA
# =============================================================================
# Anglo American calls (Moodley 2005, A.4.1). Columns: r-q, T, K, mid price, bid, offer
EXAMPLE_S0 = 1544.50
EXAMPLE_PRICES = np.array([
    [0.022685, 0.126027, 1000, 559.00, 553.00, 565.00],
    [0.022685, 0.126027, 1050, 509.50, 503.50, 515.50],
    [0.022685, 0.126027, 1100, 460.00, 454.00, 466.00],
    [0.022685, 0.126027, 1150, 411.00, 405.00, 417.00],
    [0.022685, 0.126027, 1200, 362.50, 356.50, 368.50],
    [0.022342, 0.375342, 1200, 386.00, 378.50, 393.50],
    [0.022685, 0.126027, 1250, 315.00, 309.00, 321.00],
    [0.022342, 0.375342, 1250, 345.50, 338.00, 353.00],
    [0.022685, 0.126027, 1300, 269.50, 263.50, 275.50],
    [0.022342, 0.375342, 1300, 300.50, 293.00, 308.00],
    [0.022685, 0.126027, 1350, 223.00, 217.00, 229.00],
    [0.022342, 0.375342, 1350, 259.00, 251.50, 266.50],
    [0.021947, 0.627397, 1350, 281.00, 272.00, 290.00],
    [0.022685, 0.126027, 1400, 179.00, 176.00, 182.00],
    [0.022342, 0.375342, 1400, 221.00, 213.50, 228.50],
    [0.021947, 0.627397, 1400, 244.00, 235.00, 253.00],
    [0.022685, 0.126027, 1450, 140.00, 136.00, 144.00],
    [0.022342, 0.375342, 1450, 180.00, 174.00, 186.00],
    [0.021947, 0.627397, 1450, 207.50, 198.50, 216.50],
    [0.022685, 0.126027, 1500, 105.00, 102.00, 108.00],
    [0.022342, 0.375342, 1500, 149.50, 145.00, 154.00],
    [0.021947, 0.627397, 1500, 173.00, 166.00, 180.00],
    [0.022685, 0.126027, 1600, 56.50, 51.50, 61.50],
    [0.022342, 0.375342, 1600, 96.00, 92.00, 100.00],
    [0.021947, 0.627397, 1600, 121.00, 114.00, 128.00],
    [0.022685, 0.126027, 1700, 23.50, 20.50, 26.50],
    [0.022342, 0.375342, 1700, 57.25, 51.00, 63.50],
    [0.021947, 0.627397, 1700, 81.50, 77.00, 86.00],
    [0.022685, 0.126027, 1800, 10.00, 7.00, 13.00],
    [0.022342, 0.375342, 1800, 32.50, 28.00, 37.00],
    [0.021947, 0.627397, 1800, 50.50, 44.50, 56.50],
    [0.022685, 0.126027, 1900, 4.50, 2.00, 7.00],
    [0.022342, 0.375342, 1900, 18.25, 14.50, 22.00],
    [0.021947, 0.627397, 1900, 35.50, 29.50, 41.50],
])

# Raw SVI slices {T: (a, b, rho, m, sigma)} on S0 = 100 (illustrative)
EXAMPLE_SVI = {
    0.25: (0.008, 0.04, -0.6, 0.0, 0.10),
    0.50: (0.016, 0.06, -0.6, 0.0, 0.12),
    1.00: (0.032, 0.08, -0.6, 0.0, 0.15),
    2.00: (0.064, 0.11, -0.6, 0.0, 0.20),
}


def main():
    ap = argparse.ArgumentParser(description="Calibrate Heston to implied vols")
    ap.add_argument("--source", choices=["prices", "svi", "csv"], default="prices")
    ap.add_argument("--csv", help="CSV with T,K,price[,bid,ask][,type][,r][,q] (use with --source csv)")
    ap.add_argument("--S0", type=float, help="spot (required for --source csv)")
    ap.add_argument("--r", type=float, default=0.0)
    ap.add_argument("--q", type=float, default=0.0)
    ap.add_argument("--method", choices=["solver", "asa", "asa+solver"], default="solver")
    ap.add_argument("--integration", choices=INTEGRATION_METHODS, default="legendre")
    ap.add_argument("--formulation", choices=["stable", "paper"], default="stable")
    ap.add_argument("--max-evals", type=int, default=20000, help="ASA iterations")
    ap.add_argument("--no-feller", action="store_true", help="drop 2*kappa*theta > sigma^2")
    ap.add_argument("--weights", choices=["spread", "equal", "downside"], default="spread",
                    help="spread = 1 / bid-offer vol band^2 (needs bid/offer prices); "
                         "downside = 2x weight for strikes <= spot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="heston_params.json")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--compare-integration", action="store_true")
    args = ap.parse_args()

    if args.source == "prices":
        r, T, K, px, bid, ask = EXAMPLE_PRICES.T
        market = MarketData.from_prices(EXAMPLE_S0, T, K, px, r=r, q=0.0, option_type="call",
                                        bid=bid, ask=ask)
    elif args.source == "svi":
        market = MarketData.from_svi(100.0, EXAMPLE_SVI, r=0.05, q=0.01,
                                     strikes=np.linspace(80, 120, 9))
    else:
        if not (args.csv and args.S0):
            ap.error("--source csv needs --csv and --S0")
        market = MarketData.from_csv(args.csv, args.S0, args.r, args.q)

    print(f"{len(market)} market vols from {market.source}")
    cal = Calibrator(market, integration=args.integration, formulation=args.formulation,
                     feller=not args.no_feller, weights=args.weights)
    print(f"Initial guess: {HestonParams.from_array(cal.initial_guess())}  "
          f"SSE={cal.sse(cal.initial_guess()):.6e}")
    res = cal.calibrate(method=args.method, max_evals=args.max_evals, seed=args.seed)
    res.report()
    res.save(args.save)
    if args.compare_integration:
        compare_integration(res.params, market, args.formulation)
    if args.plot:
        res.plot()


if __name__ == "__main__":
    main()
