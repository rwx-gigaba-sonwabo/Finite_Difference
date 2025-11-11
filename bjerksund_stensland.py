import math
from typing import Iterable, List, Tuple, Optional

class BjerksundStenslandOptionPricer:
    """
    A single-class implementation of Bjerksund–Stensland (BS93) for
    American calls/puts in a forward (Black-76) framing.

    You can specify one of:
      - forward   : explicit forward to expiry
      - div_yield : continuous dividend yield q
      - dividends : list of discrete cash dividends [(t_i, D_i)] up to expiry

    If multiple are provided, priority is forward -> div_yield -> dividends.
    """

    def __init__(self,
                 spot: float,
                 strike: float,
                 expiry: float,   # years
                 rate: float,     # continuous-compounded r
                 vol: float,      # annual sigma
                 forward: Optional[float] = None,
                 div_yield: Optional[float] = None,
                 dividends: Optional[List[Tuple[float, float]]] = None):
        self.spot = float(spot)
        self.strike = float(strike)
        self.expiry = float(expiry)
        self.rate = float(rate)
        self.vol = float(vol)
        self.forward = None if forward is None else float(forward)
        self.div_yield = None if div_yield is None else float(div_yield)
        self.dividends = dividends or []

    # ------------- Public API -------------

    def price_call(self) -> float:
        F = self._forward()
        price, _, _ = self._american_call_price_core(self.expiry, self.spot, F, self.strike, self.rate, self.vol)
        return price

    def price_put(self) -> float:
        F = self._forward()
        price, _, _ = self._american_put_price_core(self.expiry, self.spot, F, self.strike, self.rate, self.vol)
        return price

    def greeks_call(self, dS: float = 1e-4, dV: float = 1e-4, dT: float = 1/365.0) -> dict:
        """Bump-and-revalue; holds curve-implied forward fixed for delta/gamma."""
        F0 = self._forward()
        base, _, _ = self._american_call_price_core(self.expiry, self.spot, F0, self.strike, self.rate, self.vol)

        # Delta/Gamma: bump spot; keep forward fixed (matches forward-frame intuition)
        Su, Sd = self.spot * (1 + dS), self.spot * (1 - dS)
        up, _, _ = self._american_call_price_core(self.expiry, Su, F0, self.strike, self.rate, self.vol)
        dn, _, _ = self._american_call_price_core(self.expiry, Sd, F0, self.strike, self.rate, self.vol)
        delta = (up - dn) / (Su - Sd)
        gamma = (up - 2.0 * base + dn) / ((0.5 * (Su - Sd)) ** 2)

        # Vega: bump vol
        vu, vd = self.vol * (1 + dV), self.vol * (1 - dV)
        upv, _, _ = self._american_call_price_core(self.expiry, self.spot, F0, self.strike, self.rate, vu)
        dnv, _, _ = self._american_call_price_core(self.expiry, self.spot, F0, self.strike, self.rate, vd)
        vega = (upv - dnv) / (2.0 * self.vol * dV)

        # Theta: roll time down (per year; divide by 365 for per-day)
        Tu = max(1e-8, self.expiry - dT)
        upt, _, _ = self._american_call_price_core(Tu, self.spot, F0, self.strike, self.rate, self.vol)
        theta = (upt - base) / (-dT)

        return {"price": base, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta}

    def greeks_put(self, dS: float = 1e-4, dV: float = 1e-4, dT: float = 1/365.0) -> dict:
        F0 = self._forward()
        base, _, _ = self._american_put_price_core(self.expiry, self.spot, F0, self.strike, self.rate, self.vol)

        Su, Sd = self.spot * (1 + dS), self.spot * (1 - dS)
        up, _, _ = self._american_put_price_core(self.expiry, Su, F0, self.strike, self.rate, self.vol)
        dn, _, _ = self._american_put_price_core(self.expiry, Sd, F0, self.strike, self.rate, self.vol)
        delta = (up - dn) / (Su - Sd)
        gamma = (up - 2.0 * base + dn) / ((0.5 * (Su - Sd)) ** 2)

        vu, vd = self.vol * (1 + dV), self.vol * (1 - dV)
        upv, _, _ = self._american_put_price_core(self.expiry, self.spot, F0, self.strike, self.rate, vu)
        dnv, _, _ = self._american_put_price_core(self.expiry, self.spot, F0, self.strike, self.rate, vd)
        vega = (upv - dnv) / (2.0 * self.vol * dV)

        Tu = max(1e-8, self.expiry - dT)
        upt, _, _ = self._american_put_price_core(Tu, self.spot, F0, self.strike, self.rate, self.vol)
        theta = (upt - base) / (-dT)

        return {"price": base, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta}

    # ------------- Internals -------------

    def _forward(self) -> float:
        """Choose forward from (explicit forward) or (q) or (discrete dividends)."""
        if self.forward is not None:
            return self.forward
        if self.div_yield is not None:
            # F = S * exp((r - q) T)
            return self.spot * math.exp((self.rate - self.div_yield) * self.expiry)
        if self.dividends:
            # F = (S - sum(D_i e^{-r t_i})) e^{rT}
            pv_divs = 0.0
            for (ti, Di) in self.dividends:
                if 0.0 < ti <= self.expiry and Di != 0.0:
                    pv_divs += float(Di) * math.exp(-self.rate * float(ti))
            return (self.spot - pv_divs) * math.exp(self.rate * self.expiry)
        # default: no dividends
        return self.spot * math.exp(self.rate * self.expiry)

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _black_call_from_forward(self, F: float, K: float, vol: float, df: float) -> float:
        eps = 1e-16
        F = max(F, eps)
        K = max(K, eps)
        vol = max(vol, eps)           # vol = sigma * sqrt(T)
        d1 = (math.log(F / K) + 0.5 * vol * vol) / vol
        d2 = d1 - vol
        return df * (F * self._norm_cdf(d1) - K * self._norm_cdf(d2))

    def _phi(self,
             gamma: float, H: float, I: float,
             s_for_phi: float, T: float, r_safe: float, b_safe: float,
             sigma2: float, vol: float) -> float:
        """
        φ(γ; H, I) used in BS93; signs/order match the Torch reference you posted.
        Caller must pass s_for_phi strictly below I (we subtract a tiny epsilon and cap).
        """
        eps = 1e-32
        denom_sigma2 = max(sigma2, eps)
        kappa = (2.0 * b_safe) / denom_sigma2 + 2.0 * gamma - 1.0

        denom_vol = max(vol, eps)
        H_ = max(H, eps); I_ = max(I, eps); S_ = max(s_for_phi, eps)

        d = (math.log(H_ / S_) - (b_safe + (gamma - 0.5) * sigma2) * T) / denom_vol
        lam = -r_safe + gamma * b_safe + 0.5 * gamma * (gamma - 1.0) * sigma2

        log_IS = math.log(I_ / S_)
        safe_exp = min(kappa * log_IS, 25.0)  # overflow guard

        term1 = self._norm_cdf(d)
        term2 = math.exp(safe_exp) * self._norm_cdf(d - 2.0 * log_IS / denom_vol)
        return math.exp(lam * T) * (term1 - term2)

    def _american_call_price_core(self, T: float, S: float, F: float, K: float, r: float, sigma: float):
        """
        BS93 single-boundary call in forward frame.
        Returns (price, early_exercise_flag, I_boundary) as floats/bool.
        """
        eps = 1e-16
        T = max(float(T), 1e-5)
        sigma = float(sigma)
        vol = sigma * math.sqrt(T)
        sigma2 = sigma * sigma

        # Numerically safe spot & forward
        s_pos = max(float(S), eps)
        F = max(float(F), eps)

        # Carry from forward (exact by definition)
        b = math.log(F / s_pos) / T

        # European call in forward frame
        df = math.exp(-r * T)
        euro = self._black_call_from_forward(F, K, vol, df)

        # Early-exercise potentially relevant?
        american = (b < (r - 1e-6))

        # Safe variables (to avoid NaNs when not American)
        b_safe = b if american else 0.0
        r_safe = r if american else 0.375 * sigma2  # arbitrary pad (as in your Torch)

        denom_sigma2 = max(sigma2, eps)
        b_over_sigma2 = b_safe / denom_sigma2
        sqrt_term = (b_over_sigma2 - 0.5) ** 2 + 2.0 * r_safe / denom_sigma2
        sqrt_term = max(sqrt_term, 1e-6)
        beta = (0.5 - b_over_sigma2) + math.sqrt(sqrt_term)

        r_minus_b = r_safe - b_safe
        denom_rmb = max(r_minus_b, eps)
        B0 = K * max(r_safe / denom_rmb, 1.0)

        denom_beta = (beta - 1.0)
        if abs(denom_beta) < 1e-12:
            denom_beta = 1e-12 if denom_beta >= 0 else -1e-12
        Binf = K * beta / denom_beta

        denom_B = (Binf - B0)
        if abs(denom_B) < 1e-12:
            denom_B = 1e-12
        h_tau = -(b * T + 2.0 * vol) * (B0 / denom_B)
        I = B0 + (Binf - B0) * (1.0 - math.exp(h_tau))

        # s_for_phi: strictly below I
        s_for_phi = min(s_pos - 1e-6, I)

        # φ terms
        phi_beta_II = self._phi(beta, I, I, s_for_phi, T, r_safe, b_safe, sigma2, vol)
        phi_1_II    = self._phi(1.0,  I, I, s_for_phi, T, r_safe, b_safe, sigma2, vol)
        phi_1_KI    = self._phi(1.0,  K, I, s_for_phi, T, r_safe, b_safe, sigma2, vol)
        phi_0_KI    = self._phi(0.0,  K, I, s_for_phi, T, r_safe, b_safe, sigma2, vol)
        phi_0_II    = self._phi(0.0,  I, I, s_for_phi, T, r_safe, b_safe, sigma2, vol)

        # Core BS93 expression
        denom_I = max(I, eps)
        log_s_over_I = math.log(max(s_for_phi, eps) / denom_I)
        core = (I - K) * math.exp(beta * log_s_over_I) * (1.0 - phi_beta_II)
        c_bs = core + s_for_phi * (phi_1_II - phi_1_KI) + K * (phi_0_KI - phi_0_II)

        # Degenerate K <= 0 guard
        if K <= 0.0:
            c_bs = B0

        # Max with European
        c_bs = max(euro, c_bs)

        # Final early-exercise logic
        early_ex = (b < r) and (s_pos >= I)
        if b >= r:
            price = euro
        else:
            price = c_bs if (s_pos < I) else (s_pos - K)

        return float(price), bool(early_ex), float(I)

    def _american_put_price_core(self, T: float, S: float, F: float, K: float, r: float, sigma: float):
        """
        Put via call–put duality in forward frame:
          S* = K, K* = S, r* = r - b, b* = -b, F* = K*S/F  (so ln(F*/S*)/T = -b)
        Then price the transformed CALL with the same BS93 engine.
        """
        eps = 1e-16
        T_eff = max(float(T), 1e-5)
        s_pos = max(float(S), eps)
        F = max(float(F), eps)

        # original carry
        b = math.log(F / s_pos) / T_eff

        # transform to a call
        S_star = K
        K_star = s_pos
        r_star = r - b
        F_star = (K * s_pos) / F

        price_star, early_star, I_star = self._american_call_price_core(T_eff, S_star, F_star, K_star, r_star, float(sigma))
        return float(price_star), bool(early_star), float(I_star)