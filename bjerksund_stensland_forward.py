# bjerksund_stensland_forward.py
# Forward-based (Black-76) implementation of Bjerksund–Stensland approximation
# for American CALL and PUT on equities. Supports both continuous dividend yield
# (via q) and discrete cash dividends (via schedule). Greeks are computed with
# robust bump-and-revalue in the forward frame (holding curve-implied forward fixed).
#
# Usage:
#   python bjerksund_stensland_forward.py  (runs a quick demo)
#
# Author: ChatGPT

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union, List

import numpy as np

ArrayLike = Union[float, np.ndarray]


# ---------------------- Utilities ----------------------

def _to_array(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=float)


def norm_cdf(x: ArrayLike) -> np.ndarray:
    """Standard normal CDF using erf (accurate & fast)."""
    x = _to_array(x)
    return 0.5 * (1.0 + np.erf(x / np.sqrt(2.0)))


def black_call_from_forward(F: ArrayLike, K: ArrayLike, vol: ArrayLike, df: ArrayLike) -> np.ndarray:
    """
    European call price using Black-76 with a forward F and discount factor df = exp(-r*T).
    Here `vol` is sigma * sqrt(T).
    """
    F = _to_array(F)
    K = _to_array(K)
    vol = _to_array(vol)
    df = _to_array(df)

    eps = 1e-16
    F = np.maximum(F, eps)
    K = np.maximum(K, eps)
    vol = np.maximum(vol, eps)
    d1 = (np.log(F / K) + 0.5 * vol * vol) / vol
    d2 = d1 - vol
    return df * (F * norm_cdf(d1) - K * norm_cdf(d2))


def compute_forward_from_yield(spot: float, r: float, q: float, T: float) -> float:
    """Forward with continuous dividend yield q: F = S * exp((r - q) * T)."""
    return float(spot) * math.exp((float(r) - float(q)) * float(T))


def compute_forward_from_discrete_divs(spot: float, r: float, T: float, dividends: Iterable[Tuple[float, float]]) -> float:
    """
    Forward with discrete dividends: F = (S - PV(divs)) * exp(r*T),
    where PV(divs) = sum_i D_i * exp(-r * t_i) over dividends with 0 < t_i <= T.
    """
    r = float(r); T = float(T); S = float(spot)
    pv_divs = 0.0
    for (ti, Di) in dividends:
        ti = float(ti); Di = float(Di)
        if 0.0 < ti <= T and Di != 0.0:
            pv_divs += Di * math.exp(-r * ti)
    F = (S - pv_divs) * math.exp(r * T)
    return F


# ---------------------- Core φ kernel ----------------------

def _phi(gamma: ArrayLike, H: ArrayLike, I: ArrayLike,
         S_safe: ArrayLike, tau: ArrayLike, r_safe: ArrayLike, b_safe: ArrayLike,
         sigma2: ArrayLike, vol: ArrayLike) -> np.ndarray:
    """
    φ(γ; H, I) as used in the BS93/BS02 formulas; this mirrors the sign/order
    from the user's Torch reference to ensure a close match.
    Inputs are arrays broadcastable to the same shape.
    """
    gamma = _to_array(gamma)
    H = _to_array(H); I = _to_array(I); S_safe = _to_array(S_safe)
    tau = _to_array(tau); r_safe = _to_array(r_safe); b_safe = _to_array(b_safe)
    sigma2 = _to_array(sigma2); vol = _to_array(vol)

    # κ (kappa)
    kappa = (2.0 * b_safe) / np.maximum(sigma2, 1e-32) + 2.0 * gamma - 1.0
    # d using "safe" S (capped below by I outside)
    d = (np.log(np.maximum(H, 1e-32) / np.maximum(S_safe, 1e-32)) -
         (b_safe + (gamma - 0.5) * sigma2) * tau) / np.maximum(vol, 1e-32)
    # λ (lambda)
    lam = -r_safe + gamma * b_safe + 0.5 * gamma * (gamma - 1.0) * sigma2
    log_IS = np.log(np.maximum(I, 1e-32) / np.maximum(S_safe, 1e-32))
    safe_exp = np.minimum(kappa * log_IS, 25.0)   # guard overflow
    ret = norm_cdf(d) - np.exp(safe_exp) * norm_cdf(d - 2.0 * log_IS / np.maximum(vol, 1e-32))
    return np.exp(lam * tau) * ret


# ---------------------- American CALL via BS (forward frame) ----------------------

def pv_american_call_bjerksund_forward(
    T: ArrayLike, S: ArrayLike, F: ArrayLike, K: ArrayLike, r: ArrayLike, sigma: ArrayLike
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Price American CALL using the Bjerksund–Stensland (single-boundary) approximation
    in a forward/Black-76 setting. This mirrors the user's Torch structure.

    Returns:
        price: ndarray of option PVs
        early_exercise: boolean ndarray where exercise-now region is active
        I: trigger boundary
    """
    S = _to_array(S); F = _to_array(F); K = _to_array(K)
    T = _to_array(T); r = _to_array(r); sigma = _to_array(sigma)

    # Guards
    T = np.maximum(T, 1e-5)
    vol = sigma * np.sqrt(T)
    sigma2 = sigma * sigma

    # b from forward
    b = np.log(np.maximum(F, 1e-16) / np.maximum(S, 1e-16)) / T

    # European call in forward frame
    df = np.exp(-r * T)
    euro = black_call_from_forward(F, K, vol, df)

    # American flag
    american = b < (r - 1e-6)

    # Safe vars (mimic Torch logic)
    b_safe = np.where(american, b, 0.0)
    r_safe = np.where(american, r, 0.375 * sigma2)  # arbitrary pad to avoid NaNs when not American

    b_over_sigma2 = np.divide(b_safe, np.maximum(sigma2, 1e-32))
    sqrt_term = (b_over_sigma2 - 0.5) ** 2 + 2.0 * r_safe / np.maximum(sigma2, 1e-32)
    sqrt_term = np.maximum(sqrt_term, 1e-6)
    beta = (0.5 - b_over_sigma2) + np.sqrt(sqrt_term)

    r_minus_b = r_safe - b_safe
    ones = np.ones_like(K)
    B0 = K * np.maximum(np.divide(r_safe, np.maximum(r_minus_b, 1e-32)), ones)
    Binf = K * beta / (beta - 1.0 + 1e-32)

    h_tau = -(b * T + 2.0 * vol) * (B0 / (Binf - B0 + 1e-32))
    I = B0 + (Binf - B0) * (1.0 - np.exp(h_tau))

    # Safe S (below I)
    epsS = 1e-6
    S_safe = np.minimum(S - epsS, I)

    # φ terms
    phi_beta_II = _phi(beta, I, I, S_safe, T, r_safe, b_safe, sigma2, vol)
    phi_1_II    = _phi(1.0,  I, I, S_safe, T, r_safe, b_safe, sigma2, vol)
    phi_1_KI    = _phi(1.0,  K, I, S_safe, T, r_safe, b_safe, sigma2, vol)
    phi_0_KI    = _phi(0.0,  K, I, S_safe, T, r_safe, b_safe, sigma2, vol)
    phi_0_II    = _phi(0.0,  I, I, S_safe, T, r_safe, b_safe, sigma2, vol)

    # Core BS93 expression
    with np.errstate(divide='ignore', invalid='ignore'):
        log_S_over_I = np.log(np.maximum(S_safe, 1e-32) / np.maximum(I, 1e-32))
    core = (I - K) * np.exp(beta * log_S_over_I) * (1.0 - phi_beta_II)
    C_BS = core + S_safe * (phi_1_II - phi_1_KI) + K * (phi_0_KI - phi_0_II)

    # Fallback if K <= 0 (degenerate)
    C_BS = np.where(K > 0.0, C_BS, B0)

    # Take max with European
    C_BS = np.maximum(euro, C_BS)

    # Early-exercise region and final price
    early_ex = (b < r) & (S >= I)
    price = np.where(b >= r, euro, np.where(S < I, C_BS, S - K))

    return price, early_ex, I


# ---------------------- American PUT via call–put duality ----------------------

def pv_american_put_bjerksund_forward(
    T: ArrayLike, S: ArrayLike, F: ArrayLike, K: ArrayLike, r: ArrayLike, sigma: ArrayLike
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Price American PUT using call–put duality by transforming to a CALL
    on (S*, K*, r*, b*), keeping the forward-based structure exact.

    Transform:
        S* = K
        K* = S
        r* = r - b
        b* = -b
        F* = S* * exp(b* T) = K * S / F   (ensures ln(F*/S*)/T = b* = -b)
    """
    S = _to_array(S); F = _to_array(F); K = _to_array(K)
    T = _to_array(T); r = _to_array(r); sigma = _to_array(sigma)

    # b from forward
    T_eff = np.maximum(T, 1e-5)
    b = np.log(np.maximum(F, 1e-16) / np.maximum(S, 1e-16)) / T_eff

    # Transform
    S_star = K
    K_star = S
    r_star = r - b
    F_star = K * S / np.maximum(F, 1e-16)

    price_star, early_star, I_star = pv_american_call_bjerksund_forward(T_eff, S_star, F_star, K_star, r_star, sigma)
    # price_star equals put price in original variables
    return price_star, early_star, I_star


# ---------------------- High-level class & Greeks ----------------------

@dataclass
class AmericanBS:
    """
    Convenience wrapper to price American options (CALL/PUT) via Bjerksund–Stensland
    under a forward/Black-76 framing.
    You may provide either:
      - forward F directly, or
      - (q) continuous dividend yield, or
      - a discrete dividend schedule (list of (t_i, D_i)).

    If multiple are provided, priority is: F -> (q) -> (dividends).
    """
    spot: float
    strike: float
    expiry: float            # T in years
    rate: float              # r (continuous-compounded)
    vol: float               # sigma (per year)
    forward: Optional[float] = None
    div_yield: Optional[float] = None
    dividends: Optional[List[Tuple[float, float]]] = None

    # --------- plumbing ---------
    def _forward(self) -> float:
        if self.forward is not None:
            return float(self.forward)
        if self.div_yield is not None:
            return compute_forward_from_yield(self.spot, self.rate, self.div_yield, self.expiry)
        if self.dividends:
            return compute_forward_from_discrete_divs(self.spot, self.rate, self.expiry, self.dividends)
        # default: no dividends
        return self.spot * math.exp(self.rate * self.expiry)

    # --------- pricing ---------
    def price_call(self) -> float:
        F = self._forward()
        price, *_ = pv_american_call_bjerksund_forward(self.expiry, self.spot, F, self.strike, self.rate, self.vol)
        return float(np.asarray(price))

    def price_put(self) -> float:
        F = self._forward()
        price, *_ = pv_american_put_bjerksund_forward(self.expiry, self.spot, F, self.strike, self.rate, self.vol)
        return float(np.asarray(price))

    # --------- greeks (bump-and-revalue, forward held from curves) ---------
    def greeks_call(self, dS: float = 1e-4, dV: float = 1e-4, dT: float = 1/365.0) -> dict:
        F0 = self._forward()     # forward held fixed for delta/gamma bumps
        base, *_ = pv_american_call_bjerksund_forward(self.expiry, self.spot, F0, self.strike, self.rate, self.vol)

        # delta/gamma
        Su, Sd = self.spot*(1+dS), self.spot*(1-dS)
        up, *_ = pv_american_call_bjerksund_forward(self.expiry, Su, F0, self.strike, self.rate, self.vol)
        dn, *_ = pv_american_call_bjerksund_forward(self.expiry, Sd, F0, self.strike, self.rate, self.vol)
        delta = (up - dn) / (Su - Sd)
        gamma = (up - 2*base + dn) / ((0.5*(Su - Sd))**2)

        # vega
        vu, vd = self.vol*(1+dV), self.vol*(1-dV)
        upv, *_ = pv_american_call_bjerksund_forward(self.expiry, self.spot, F0, self.strike, self.rate, vu)
        dnv, *_ = pv_american_call_bjerksund_forward(self.expiry, self.spot, F0, self.strike, self.rate, vd)
        vega = (upv - dnv) / (2*self.vol*dV)

        # theta (per year; divide by 365 for per-day)
        Tu = max(1e-8, self.expiry - dT)
        upt, *_ = pv_american_call_bjerksund_forward(Tu, self.spot, F0, self.strike, self.rate, self.vol)
        theta = (upt - base)/(-dT)

        return {"price": float(base), "delta": float(delta), "gamma": float(gamma),
                "vega": float(vega), "theta": float(theta)}

    def greeks_put(self, dS: float = 1e-4, dV: float = 1e-4, dT: float = 1/365.0) -> dict:
        F0 = self._forward()
        base, *_ = pv_american_put_bjerksund_forward(self.expiry, self.spot, F0, self.strike, self.rate, self.vol)

        Su, Sd = self.spot*(1+dS), self.spot*(1-dS)
        up, *_ = pv_american_put_bjerksund_forward(self.expiry, Su, F0, self.strike, self.rate, self.vol)
        dn, *_ = pv_american_put_bjerksund_forward(self.expiry, Sd, F0, self.strike, self.rate, self.vol)
        delta = (up - dn) / (Su - Sd)
        gamma = (up - 2*base + dn) / ((0.5*(Su - Sd))**2)

        vu, vd = self.vol*(1+dV), self.vol*(1-dV)
        upv, *_ = pv_american_put_bjerksund_forward(self.expiry, self.spot, F0, self.strike, self.rate, vu)
        dnv, *_ = pv_american_put_bjerksund_forward(self.expiry, self.spot, F0, self.strike, self.rate, vd)
        vega = (upv - dnv) / (2*self.vol*dV)

        Tu = max(1e-8, self.expiry - dT)
        upt, *_ = pv_american_put_bjerksund_forward(Tu, self.spot, F0, self.strike, self.rate, self.vol)
        theta = (upt - base)/(-dT)

        return {"price": float(base), "delta": float(delta), "gamma": float(gamma),
                "vega": float(vega), "theta": float(theta)}


# ---------------------- Demo ----------------------

def _demo():
    # Parameters
    S = 100.0
    K = 100.0
    T = 0.75
    r = 0.06
    sig = 0.25

    # Example 1: continuous dividend yield q
    q = 0.02
    pricer_q = AmericanBS(spot=S, strike=K, expiry=T, rate=r, vol=sig, div_yield=q)
    print("CALL (q):", pricer_q.price_call(), pricer_q.greeks_call())
    print("PUT  (q):", pricer_q.price_put(),  pricer_q.greeks_put())

    # Example 2: discrete dividends (two payments)
    divs = [(0.25, 1.0), (0.5, 1.0)]
    pricer_d = AmericanBS(spot=S, strike=K, expiry=T, rate=r, vol=sig, dividends=divs)
    print("CALL (divs):", pricer_d.price_call(), pricer_d.greeks_call())
    print("PUT  (divs):", pricer_d.price_put(),  pricer_d.greeks_put())

if __name__ == "__main__":
    _demo()

with open('/mnt/data/bjerksund_stensland_forward.py', 'w') as f:
    f.write(content)
print("Saved to /mnt/data/bjerksund_stensland_forward.py")