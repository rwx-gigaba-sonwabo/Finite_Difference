import math
from typing import Literal, Tuple, Optional, Dict

OptionType = Literal["call", "put"]

class BjerksundStenslandOptionPricer:
    """
    American option pricer (calls & puts) using the Bjerksund–Stensland 2002 approximation,
    aligned with RiskFlow’s forward/spot and cost-of-carry intuition.

    Inputs are floats. No NumPy/torch required.

    Key conventions (matching RiskFlow):
    - Use Black-76 for the European piece with FORWARD, not spot.
    - Cost of carry b := ln(F/S)/T   (so F = S * exp(b T))
    - For puts: use the RiskFlow swap trick (S,K) -> (K,S) and (r,b)->(r-b,-b), then call formula.
    """

    # ---------- Black-76 European ----------
    @staticmethod
    def _black76_price(forward: float, strike: float, vol: float, tau: float, df: float, is_call: bool) -> float:
        if tau <= 0.0 or vol <= 0.0:
            intrinsic = max(0.0, (forward - strike) if is_call else (strike - forward))
            return df * intrinsic
        sqrt_tau = math.sqrt(tau)
        sig = vol
        d1 = (math.log(forward / strike) + 0.5 * sig * sig * tau) / (sig * sqrt_tau)
        d2 = d1 - sig * sqrt_tau
        nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
        nd2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))
        if is_call:
            return df * (forward * nd1 - strike * nd2)
        else:
            # put via parity under forwards
            return df * (strike * (1.0 - nd2) - forward * (1.0 - nd1))

    # ---------- Normal CDF ----------
    @staticmethod
    def _ncdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    # ---------- Core φ(γ,H,I) helper from BS2002 (as coded in RiskFlow) ----------
    @staticmethod
    def _phi(gamma: float, H: float, I: float, S_safe: float, tau: float, vol: float, safe_b: float,
             sigma2: float, safe_r: float) -> float:
        # kappa, d, lambda, logs exactly as RiskFlow (torch code) but with math floats
        kappa = (2.0 * safe_b) / sigma2 + 2.0 * gamma - 1.0
        d = (math.log(H / S_safe) - (safe_b + (gamma - 0.5) * sigma2) * tau) / vol
        lamb = -safe_r + gamma * safe_b + 0.5 * gamma * (gamma - 1.0) * sigma2
        log_IS = math.log(I / S_safe)
        # clamp exp exponent like RiskFlow (max 25) to avoid overflow:
        safe_exp = min(kappa * log_IS, 25.0)
        ret = BjerksundStenslandOptionPricer._ncdf(d) - math.exp(safe_exp) * BjerksundStenslandOptionPricer._ncdf(d - 2.0 * log_IS / vol)
        return math.exp(lamb * tau) * ret

    # ---------- Main American price (BS2002) in RiskFlow style ----------
    @staticmethod
    def _american_call_price_riskflow_style(
        S: float, K: float, r: float, b: float, vol: float, tau: float
    ) -> Tuple[float, float, bool]:
        """
        Returns (price, I_boundary, exercised_now)
        """
        if tau <= 0.0:
            return (max(0.0, S - K), 0.0, False)

        sigma = vol
        sigma2 = sigma * sigma
        sqrt_tau = math.sqrt(tau)
        # European Black-76 with F = S*exp(bT)
        F = S * math.exp(b * tau)
        df = math.exp(-r * tau)
        black = BjerksundStenslandOptionPricer._black76_price(F, K, sigma, tau, df, is_call=True)

        # American trigger
        american = (b < r - 1e-6)
        if not american:
            # no early-exercise incentive -> European
            return (black, 0.0, False)

        # RiskFlow “safe” quantities / β, B0, Binf, I
        safe_b = b
        b_over_sigma2 = safe_b / sigma2
        # pad non-american r with ~ 0.375*sigma^2 in RF; here american==True already
        safe_r = r
        # sqrt term
        safe_sqrt = (b_over_sigma2 - 0.5) * (b_over_sigma2 - 0.5) + 2.0 * safe_r / sigma2
        safe_sqrt = max(safe_sqrt, 1e-6)
        beta = (0.5 - b_over_sigma2) + math.sqrt(safe_sqrt)
        r_b = safe_r - safe_b

        # boundaries
        B0 = K * max(safe_r / r_b, 1.0)
        Binf = K * beta / (beta - 1.0)

        if K > 0.0:
            volT = sigma * sqrt_tau
            h_tau = -(b * tau + 2.0 * volT) * (B0 / (Binf - B0))
            I = B0 + (Binf - B0) * (1.0 - math.exp(h_tau))
            # clamp S below I for stability (epsilon like RF)
            S_safe = min(S - 1e-6, I)

            # BS2002 Americans (same composition as RiskFlow)
            # C_BS = (I - K) * (S/I)^β * (1 - φ(β,I,I)) + S*(φ(1,I,I)-φ(1,K,I)) + K*(φ(0,K,I)-φ(0,I,I))
            #   implemented with the same φ() structure as RF
            # part 1
            pow_term = math.exp(math.log(S_safe / I) * beta)
            one_minus_phi_beta = 1.0 - BjerksundStenslandOptionPricer._phi(beta, I, I, S_safe, tau, volT, safe_b, sigma2, safe_r)
            C_BS = (I - K) * pow_term * one_minus_phi_beta
            # part 2
            x = BjerksundStenslandOptionPricer._phi(1.0, I, I, S_safe, tau, volT, safe_b, sigma2, safe_r)
            y = BjerksundStenslandOptionPricer._phi(1.0, K, I, S_safe, tau, volT, safe_b, sigma2, safe_r)
            C_BS += S_safe * (x - y)
            # part 3
            x0 = BjerksundStenslandOptionPricer._phi(0.0, K, I, S_safe, tau, volT, safe_b, sigma2, safe_r)
            y0 = BjerksundStenslandOptionPricer._phi(0.0, I, I, S_safe, tau, volT, safe_b, sigma2, safe_r)
            C_BS += K * (x0 - y0)
        else:
            I = B0
            C_BS = B0  # degenerate K

        # final composition as in RiskFlow
        C_BS = max(black, C_BS)
        if S >= I:
            # immediate exercise region
            return (S - K, I, True)
        else:
            return (C_BS, I, False)

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        F: float,
        sigma: float,
        option_type: OptionType = "call"
    ) -> Dict[str, float]:
        """
        Parameters
        ----------
        S : spot price
        K : strike
        T : time to expiry in years
        r : continuously compounded risk-free rate
        F : forward for maturity T (same currency as S)
        sigma : Black vol (annualized)
        option_type : "call" or "put"

        Returns dict with price, early_exercise flag (1.0/0.0), and I (exercise boundary).
        """
        if T <= 0.0:
            intrinsic = max(0.0, (S - K) if option_type == "call" else (K - S))
            return {"price": intrinsic, "I": 0.0, "early_exercise": float(False)}

        # cost of carry from forward/spot (RiskFlow style)
        # b = ln(F/S)/T  (so F = S * exp(bT))
        b = math.log(max(F, 1e-15) / max(S, 1e-15)) / T  # :contentReference[oaicite:5]{index=5}

        if option_type == "call":
            price, I, exercised = self._american_call_price_riskflow_style(S, K, r, b, sigma, T)
            return {"price": price, "I": I, "early_exercise": float(exercised)}
        else:
            # RiskFlow put logic: transform to a call with (S,K)->(K,S), (r,b)->(r-b,-b) then reuse call formula
            # :contentReference[oaicite:6]{index=6}
            price_call, I_star, exercised = self._american_call_price_riskflow_style(
                S=K, K=S, r=r - b, b=-b, vol=sigma, tau=T
            )
            # That call’s intrinsic in the “exercise now” region maps to put intrinsic K-S
            # so just reuse returned value (already intrinsic vs C_BS internally)
            return {"price": price_call, "I": I_star, "early_exercise": float(exercised)}

    # --------- Simple greeks via bump-and-revalue (finite differences) ---------
    def greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        F: float,
        sigma: float,
        option_type: OptionType = "call",
        dS: float = 1e-4,
        dSigma: float = 1e-4,
        dR: float = 1e-6
    ) -> Dict[str, float]:
        """
        Finite-difference greeks around the American price (practical and robust).
        Uses central differences where possible.
        """
        base = self.price(S, K, T, r, F, sigma, option_type)["price"]

        # Delta: hold F fixed or b fixed? RiskFlow is forward-based; practitioners usually report spot-delta.
        # Here we compute spot-delta holding b (hence F) fixed => re-compute F from (S,b,T).
        b = math.log(max(F, 1e-15) / max(S, 1e-15)) / max(T, 1e-12)
        S_up = S * (1.0 + dS)
        S_dn = S * (1.0 - dS)
        F_up = S_up * math.exp(b * T)
        F_dn = S_dn * math.exp(b * T)
        p_up = self.price(S_up, K, T, r, F_up, sigma, option_type)["price"]
        p_dn = self.price(S_dn, K, T, r, F_dn, sigma, option_type)["price"]
        delta = (p_up - p_dn) / (S_up - S_dn)

        # Vega (Black vol)
        p_vs_up = self.price(S, K, T, r, F, sigma * (1.0 + dSigma), option_type)["price"]
        p_vs_dn = self.price(S, K, T, r, F, sigma * (1.0 - dSigma), option_type)["price"]
        vega = (p_vs_up - p_vs_dn) / (2.0 * sigma * dSigma + 1e-18)

        # Rho
        p_r_up = self.price(S, K, T, r + dR, F, sigma, option_type)["price"]
        p_r_dn = self.price(S, K, T, r - dR, F, sigma, option_type)["price"]
        rho = (p_r_up - p_r_dn) / (2.0 * dR)

        # Gamma (second derivative w.r.t spot)
        gamma = (p_up - 2.0 * base + p_dn) / ((S_up - S) * (S - S_dn) + 1e-18)

        return {"delta": delta, "gamma": gamma, "vega": vega, "rho": rho}
