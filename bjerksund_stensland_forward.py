# Bjerksund–Stensland American option pricer (forward/Black-76 framing)
# One class, floats only, no NumPy/torch/dataclass.

import math
from typing import Optional, List, Tuple, Literal, Dict

OptionType = Literal["call", "put"]

class BjerksundStenslandOptionPricer:
    """Bjerksund–Stensland (single-boundary) American option pricer in forward (Black-76) form.

    Core intuition (aligned with RiskFlow):
      - Use FORWARD for the European piece (Black-76), discount by exp(-r*T).
      - Cost-of-carry b := ln(F/S)/T (so F = S * exp(b T)).
      - American exercise relevant if b < r; else price = European Black-76.
      - Call formula uses BS boundary I(T). Put priced by call–put transform.

    You can pass F directly, or derive F from:
      - continuous dividend yield q (F = S * exp((r - q)T)), or
      - discrete dividends [(t_i, D_i)] (F = (S - Σ D_i e^{-r t_i}) e^{rT}).
      If multiple are provided: F > q > dividends.
    """

    # ---------- Public API ----------

    def price(self,
              S: float,
              K: float,
              T: float,
              r: float,
              sigma: float,
              option_type: OptionType = "call",
              F: Optional[float] = None,
              q: Optional[float] = None,
              dividends: Optional[List[Tuple[float, float]]] = None) -> Dict[str, float]:
        """Return {"price":..., "I": boundary_or_0.0, "early_exercise": 0.0/1.0}."""
        if T <= 0.0:
            intrinsic = max(0.0, (S - K) if option_type == "call" else (K - S))
            return {"price": intrinsic, "I": 0.0, "early_exercise": 0.0}

        F_eff = self._resolve_forward(S, r, T, F, q, dividends)

        # cost-of-carry from forward (RiskFlow-style)
        b = math.log(max(F_eff, 1e-15) / max(S, 1e-15)) / T

        if option_type == "call":
            price, I, exercised = self._american_call_price(S, K, r, b, sigma, T)
            return {"price": price, "I": I, "early_exercise": 1.0 if exercised else 0.0}
        else:
            # Put via call–put transform in forward frame:
            #   S* = K, K* = S, r* = r - b, b* = -b (so F* = K*S/F)
            price_star, I_star, exercised = self._american_call_price(S=K, K=S, r=r - b, b=-b, sigma=sigma, T=T)
            return {"price": price_star, "I": I_star, "early_exercise": 1.0 if exercised else 0.0}

    def greeks(self,
               S: float,
               K: float,
               T: float,
               r: float,
               sigma: float,
               option_type: OptionType = "call",
               F: Optional[float] = None,
               q: Optional[float] = None,
               dividends: Optional[List[Tuple[float, float]]] = None,
               dS: float = 1e-4,
               dSigma: float = 1e-4,
               dR: float = 1e-6) -> Dict[str, float]:
        """Finite-difference greeks (robust). Spot-delta/gamma computed holding carry b fixed."""
        # Base
        F_eff = self._resolve_forward(S, r, T, F, q, dividends)
        base = self.price(S, K, T, r, sigma, option_type, F_eff)["price"]

        # Keep b fixed (so update F when bumping S): b = ln(F/S)/T
        b = math.log(max(F_eff, 1e-15) / max(S, 1e-15)) / max(T, 1e-12)
        S_up, S_dn = S * (1.0 + dS), S * (1.0 - dS)
        F_up = S_up * math.exp(b * T)
        F_dn = S_dn * math.exp(b * T)

        p_up = self.price(S_up, K, T, r, sigma, option_type, F_up)["price"]
        p_dn = self.price(S_dn, K, T, r, sigma, option_type, F_dn)["price"]
        delta = (p_up - p_dn) / (S_up - S_dn)
        gamma = (p_up - 2.0 * base + p_dn) / ((S_up - S) * (S - S_dn) + 1e-18)

        # Vega
        p_vs_up = self.price(S, K, T, r, sigma * (1.0 + dSigma), option_type, F_eff)["price"]
        p_vs_dn = self.price(S, K, T, r, sigma * (1.0 - dSigma), option_type, F_eff)["price"]
        vega = (p_vs_up - p_vs_dn) / (2.0 * sigma * dSigma + 1e-18)

        # Rho
        p_r_up = self.price(S, K, T, r + dR, sigma, option_type, F_eff)["price"]
        p_r_dn = self.price(S, K, T, r - dR, sigma, option_type, F_eff)["price"]
        rho = (p_r_up - p_r_dn) / (2.0 * dR)

        return {"delta": delta, "gamma": gamma, "vega": vega, "rho": rho}

    # ---------- Internals (helpers) ----------

    def _resolve_forward(self,
                         S: float,
                         r: float,
                         T: float,
                         F: Optional[float],
                         q: Optional[float],
                         dividends: Optional[List[Tuple[float, float]]]) -> float:
        """Pick forward by priority: F > q > dividends > no divs."""
        if F is not None:
            return float(F)
        if q is not None:
            return S * math.exp((r - float(q)) * T)
        if dividends:
            pv = 0.0
            for (ti, Di) in dividends:
                if 0.0 < ti <= T and Di != 0.0:
                    pv += float(Di) * math.exp(-r * float(ti))
            return (S - pv) * math.exp(r * T)
        return S * math.exp(r * T)

    @staticmethod
    def _ncdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def _black76(forward: float, strike: float, sigma: float, T: float, df: float, call: bool) -> float:
        if T <= 0.0 or sigma <= 0.0:
            intrinsic = max(0.0, (forward - strike) if call else (strike - forward))
            return df * intrinsic
        srt = math.sqrt(T)
        d1 = (math.log(forward / strike) + 0.5 * sigma * sigma * T) / (sigma * srt)
        d2 = d1 - sigma * srt
        N1, N2 = BjerksundStenslandOptionPricer._ncdf(d1), BjerksundStenslandOptionPricer._ncdf(d2)
        if call:
            return df * (forward * N1 - strike * N2)
        else:
            return df * (strike * (1.0 - N2) - forward * (1.0 - N1))

    @staticmethod
    def _phi(gamma: float, H: float, I: float, S_for_phi: float,
             T: float, volT: float, b_safe: float, sigma2: float, r_safe: float) -> float:
        # Mirrors the RiskFlow torch structure (clamps, ordering)
        # volT := sigma * sqrt(T)
        eps = 1e-32
        sigma2 = max(sigma2, eps)
        volT = max(volT, eps)
        H_ = max(H, eps)
        I_ = max(I, eps)
        S_ = max(S_for_phi, eps)

        kappa = (2.0 * b_safe) / sigma2 + 2.0 * gamma - 1.0
        d = (math.log(H_ / S_) - (b_safe + (gamma - 0.5) * sigma2) * T) / volT
        lam = -r_safe + gamma * b_safe + 0.5 * gamma * (gamma - 1.0) * sigma2
        log_IS = math.log(I_ / S_)
        safe_exp = min(kappa * log_IS, 25.0)

        term1 = BjerksundStenslandOptionPricer._ncdf(d)
        term2 = math.exp(safe_exp) * BjerksundStenslandOptionPricer._ncdf(d - 2.0 * log_IS / volT)
        return math.exp(lam * T) * (term1 - term2)

    def _american_call_price(self, S: float, K: float, r: float, b: float, sigma: float, T: float) -> Tuple[float, float, bool]:
        """RiskFlow-style BS American call: returns (price, I, exercised_now)."""
        eps = 1e-16
        T = max(T, 1e-5)
        sigma = float(sigma)
        sigma2 = sigma * sigma
        volT = sigma * math.sqrt(T)

        # European Black-76 on forward
        F = S * math.exp(b * T)
        df = math.exp(-r * T)
        euro = self._black76(F, K, sigma, T, df, call=True)

        # American trigger
        if not (b < r - 1e-6):
            return (euro, 0.0, False)

        # β, B0, Binf, I (with RF-style guards)
        b_over_sigma2 = b / max(sigma2, eps)
        safe_sqrt = (b_over_sigma2 - 0.5) ** 2 + 2.0 * r / max(sigma2, eps)
        safe_sqrt = max(safe_sqrt, 1e-6)
        beta = (0.5 - b_over_sigma2) + math.sqrt(safe_sqrt)

        r_b = r - b
        B0 = K * max(r / max(r_b, eps), 1.0)
        denom_beta = beta - 1.0
        if abs(denom_beta) < 1e-12:
            denom_beta = 1e-12 if denom_beta >= 0 else -1e-12
        Binf = K * beta / denom_beta

        denom_B = Binf - B0
        if abs(denom_B) < 1e-12:
            denom_B = 1e-12
        h_tau = -(b * T + 2.0 * volT) * (B0 / denom_B)
        I = B0 + (Binf - B0) * (1.0 - math.exp(h_tau))

        # Safe argument for φ must be strictly below I
        S_for_phi = min(max(S, eps) - 1e-6, I)

        # φ-terms
        phi_beta_II = self._phi(beta, I, I, S_for_phi, T, volT, b, sigma2, r)
        phi_1_II    = self._phi(1.0,  I, I, S_for_phi, T, volT, b, sigma2, r)
        phi_1_KI    = self._phi(1.0,  K, I, S_for_phi, T, volT, b, sigma2, r)
        phi_0_KI    = self._phi(0.0,  K, I, S_for_phi, T, volT, b, sigma2, r)
        phi_0_II    = self._phi(0.0,  I, I, S_for_phi, T, volT, b, sigma2, r)

        # Bjerksund–Stensland composition
        if K <= 0.0:
            c_bs = B0
        else:
            log_ratio = math.log(max(S_for_phi, eps) / max(I, eps))
            core = (I - K) * math.exp(beta * log_ratio) * (1.0 - phi_beta_II)
            c_bs = core + S_for_phi * (phi_1_II - phi_1_KI) + K * (phi_0_KI - phi_0_II)

        c_bs = max(euro, c_bs)

        # Exercise-now region
        if S >= I:
            return (S - K, I, True)
        else:
            return (c_bs, I, False)
