# Bjerksund–Stensland American option pricer (forward/Black-76 framing)
# Enhanced:
#   - Adds 2002 two-step exercise boundary method (Proposition 1) with A(·) evaluation function
#   - Adds optional proxy: 2*c_two_step - c_flat (as suggested in numerical results)
#   - Allows boundary variant selection: "riskflow_1993" vs "paper_2002_modified"
#
# One class, floats only, no NumPy/torch/dataclass.

import math
from typing import Optional, List, Tuple, Literal, Dict

OptionType = Literal["call", "put"]
Method = Literal["single", "two_step", "two_step_proxy"]
BoundaryVariant = Literal["riskflow_1993", "paper_2002_modified"]


class BjerksundStenslandOptionPricer:
    """Bjerksund–Stensland American option pricer in forward (Black-76) form.

    Supports:
      - single-step (flat boundary) approximation (RiskFlow-aligned)
      - two-step boundary approximation (Bjerksund–Stensland 2002, Proposition 1)
      - proxy = 2*(two-step) - (single-step), per paper's numerical discussion

    Dividend handling:
      - This pricer itself does NOT model discrete dividend jumps in the exercise policy.
      - Dividends affect the price via the effective forward F and carry b := ln(F/S)/T.

    Forward resolution priority:
      F > q > dividends > no dividends

    Notes on boundary formulas:
      - "riskflow_1993": uses the 1993 boundary form (paper footnote; RiskFlow-style)
      - "paper_2002_modified": uses Eq. (11) (the modified boundary in the 2002 paper)
    """

    # ---------- Public API ----------

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType = "call",
        F: Optional[float] = None,
        q: Optional[float] = None,
        dividends: Optional[List[Tuple[float, float]]] = None,
        method: Method = "single",
        boundary_variant: BoundaryVariant = "riskflow_1993",
    ) -> Dict[str, float]:
        """
        Returns a dict with:
          - price
          - early_exercise (0/1, based on boundary test at t=0)
          - boundary outputs:
              single: I
              two_step: X, x, t_split
        """
        if T <= 0.0:
            intrinsic = max(0.0, (S - K) if option_type == "call" else (K - S))
            return {
                "price": intrinsic,
                "early_exercise": 0.0,
                "I": 0.0,
                "X": 0.0,
                "x": 0.0,
                "t_split": 0.0,
            }

        F_eff = self._resolve_forward(S, r, T, F, q, dividends)
        b = math.log(max(F_eff, 1e-15) / max(S, 1e-15)) / max(T, 1e-12)

        if option_type == "call":
            return self._price_call(
                S=S, K=K, T=T, r=r, b=b, sigma=sigma,
                method=method, boundary_variant=boundary_variant
            )

        # Put via call–put transform (paper Eq. 19):
        #   P(S,K,T,r,b,σ) = C(K,S,T,r-b,-b,σ)
        res_star = self._price_call(
            S=K, K=S, T=T, r=r - b, b=-b, sigma=sigma,
            method=method, boundary_variant=boundary_variant
        )
        return {
            "price": res_star["price"],
            "early_exercise": res_star["early_exercise"],
            "I": res_star["I"],
            "X": res_star["X"],
            "x": res_star["x"],
            "t_split": res_star["t_split"],
        }

    def greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType = "call",
        F: Optional[float] = None,
        q: Optional[float] = None,
        dividends: Optional[List[Tuple[float, float]]] = None,
        method: Method = "single",
        boundary_variant: BoundaryVariant = "riskflow_1993",
        dS: float = 1e-4,
        dSigma: float = 1e-4,
        dR: float = 1e-6,
    ) -> Dict[str, float]:
        """Finite-difference greeks (robust). Spot-delta/gamma computed holding carry b fixed."""
        F_eff = self._resolve_forward(S, r, T, F, q, dividends)

        base = self.price(
            S, K, T, r, sigma, option_type,
            F=F_eff, method=method, boundary_variant=boundary_variant
        )["price"]

        # Keep b fixed when bumping S: b = ln(F/S)/T
        b = math.log(max(F_eff, 1e-15) / max(S, 1e-15)) / max(T, 1e-12)

        S_up, S_dn = S * (1.0 + dS), S * (1.0 - dS)
        F_up = S_up * math.exp(b * T)
        F_dn = S_dn * math.exp(b * T)

        p_up = self.price(
            S_up, K, T, r, sigma, option_type,
            F=F_up, method=method, boundary_variant=boundary_variant
        )["price"]
        p_dn = self.price(
            S_dn, K, T, r, sigma, option_type,
            F=F_dn, method=method, boundary_variant=boundary_variant
        )["price"]

        delta = (p_up - p_dn) / (S_up - S_dn)
        gamma = (p_up - 2.0 * base + p_dn) / (((S_up - S) * (S - S_dn)) + 1e-18)

        # Vega
        p_vs_up = self.price(
            S, K, T, r, sigma * (1.0 + dSigma), option_type,
            F=F_eff, method=method, boundary_variant=boundary_variant
        )["price"]
        p_vs_dn = self.price(
            S, K, T, r, sigma * (1.0 - dSigma), option_type,
            F=F_eff, method=method, boundary_variant=boundary_variant
        )["price"]
        vega = (p_vs_up - p_vs_dn) / (2.0 * sigma * dSigma + 1e-18)

        # Rho
        p_r_up = self.price(
            S, K, T, r + dR, sigma, option_type,
            F=F_eff, method=method, boundary_variant=boundary_variant
        )["price"]
        p_r_dn = self.price(
            S, K, T, r - dR, sigma, option_type,
            F=F_eff, method=method, boundary_variant=boundary_variant
        )["price"]
        rho = (p_r_up - p_r_dn) / (2.0 * dR)

        return {"delta": delta, "gamma": gamma, "vega": vega, "rho": rho}

    # ---------- Forward resolution ----------

    def _resolve_forward(
        self,
        S: float,
        r: float,
        T: float,
        F: Optional[float],
        q: Optional[float],
        dividends: Optional[List[Tuple[float, float]]],
    ) -> float:
        """Pick forward by priority: F > q > dividends > no divs."""
        if F is not None:
            return float(F)

        if q is not None:
            return S * math.exp((r - float(q)) * T)

        if dividends:
            pv = 0.0
            for (ti, Di) in dividends:
                if 0.0 < float(ti) <= T and float(Di) != 0.0:
                    pv += float(Di) * math.exp(-r * float(ti))
            return (S - pv) * math.exp(r * T)

        return S * math.exp(r * T)

    # ---------- Normal / Black-76 helpers ----------

    @staticmethod
    def _ncdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def _npdf(x: float) -> float:
        return 0.3989422804014327 * math.exp(-0.5 * x * x)

    @staticmethod
    def _black76(forward: float, strike: float, sigma: float, T: float, df: float, call: bool) -> float:
        if T <= 0.0 or sigma <= 0.0:
            intrinsic = max(0.0, (forward - strike) if call else (strike - forward))
            return df * intrinsic
        srt = math.sqrt(T)
        d1 = (math.log(forward / strike) + 0.5 * sigma * sigma * T) / (sigma * srt)
        d2 = d1 - sigma * srt
        N1 = BjerksundStenslandOptionPricer._ncdf(d1)
        N2 = BjerksundStenslandOptionPricer._ncdf(d2)
        if call:
            return df * (forward * N1 - strike * N2)
        return df * (strike * (1.0 - N2) - forward * (1.0 - N1))

    # ---------- Core pricing (call) ----------

    def _price_call(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        b: float,
        sigma: float,
        method: Method,
        boundary_variant: BoundaryVariant,
    ) -> Dict[str, float]:
        eps = 1e-16
        T = max(T, 1e-8)
        sigma = float(sigma)
        sigma2 = sigma * sigma

        # European Black-76 on forward
        F = S * math.exp(b * T)
        df = math.exp(-r * T)
        euro = self._black76(F, K, sigma, T, df, call=True)

        # No-early-exercise case (call): if b >= r, American = European
        if not (b < r - 1e-6):
            return {
                "price": euro,
                "early_exercise": 0.0,
                "I": 0.0,
                "X": 0.0,
                "x": 0.0,
                "t_split": 0.0,
            }

        if method == "single":
            price, I, exercised = self._american_call_single(S, K, r, b, sigma, T, boundary_variant)
            return {
                "price": price,
                "early_exercise": 1.0 if exercised else 0.0,
                "I": I,
                "X": 0.0,
                "x": 0.0,
                "t_split": 0.0,
            }

        if method in ("two_step", "two_step_proxy"):
            c_two, X, x, t_split, exercised_now = self._american_call_two_step(
                S=S, K=K, r=r, b=b, sigma=sigma, T=T, boundary_variant=boundary_variant
            )

            if method == "two_step":
                return {
                    "price": max(euro, c_two),  # safeguard
                    "early_exercise": 1.0 if exercised_now else 0.0,
                    "I": 0.0,
                    "X": X,
                    "x": x,
                    "t_split": t_split,
                }

            # proxy: 2*c_two - c_flat (paper suggests as a "reasonable proxy")
            c_flat, _, _ = self._american_call_single(S, K, r, b, sigma, T, boundary_variant)
            c_proxy = 2.0 * c_two - c_flat
            return {
                "price": max(euro, c_proxy),  # safeguard
                "early_exercise": 1.0 if exercised_now else 0.0,
                "I": 0.0,
                "X": X,
                "x": x,
                "t_split": t_split,
            }

        # fallback
        return {
            "price": euro,
            "early_exercise": 0.0,
            "I": 0.0,
            "X": 0.0,
            "x": 0.0,
            "t_split": 0.0,
        }

    # ---------- Boundary construction (X_T) ----------

    def _beta_B0_B1(
        self,
        K: float,
        r: float,
        b: float,
        sigma: float,
    ) -> Tuple[float, float, float]:
        eps = 1e-16
        sigma2 = max(sigma * sigma, eps)
        b_over_sigma2 = b / sigma2
        rad = (b_over_sigma2 - 0.5) ** 2 + 2.0 * r / sigma2
        rad = max(rad, 1e-12)
        beta = (0.5 - b_over_sigma2) + math.sqrt(rad)

        # B0, B1 per paper (Eq. 12-13); keep stable for r ~ b
        r_b = max(r - b, 1e-12)
        B0 = max(K, (r / r_b) * K)
        denom = max(beta - 1.0, 1e-12)
        B1 = (beta / denom) * K
        return beta, B0, B1

    def _boundary_XT(
        self,
        K: float,
        r: float,
        b: float,
        sigma: float,
        tau: float,
        boundary_variant: BoundaryVariant,
    ) -> float:
        """
        X_tau = B0 + (B1-B0) * (1 - exp(h(tau))).

        boundary_variant:
          - riskflow_1993: h(tau)=-(b tau + 2σ√tau) * (B0/(B1-B0))
          - paper_2002_modified: h(tau)=-(b tau + 2σ√tau) * (K^2/((B1-B0)B0))
        """
        tau = max(tau, 1e-8)
        beta, B0, B1 = self._beta_B0_B1(K, r, b, sigma)

        denom = max(B1 - B0, 1e-12)
        vol_sqrt = sigma * math.sqrt(tau)

        if boundary_variant == "paper_2002_modified":
            # Eq. (11) in 2002 paper
            scale = (K * K) / (denom * max(B0, 1e-12))
        else:
            # RiskFlow/1993 (footnote boundary form)
            scale = B0 / denom

        h = -(b * tau + 2.0 * vol_sqrt) * scale
        # keep exp stable
        h = max(min(h, 50.0), -50.0)

        X = B0 + (B1 - B0) * (1.0 - math.exp(h))
        return max(X, K)

    # ---------- Single-step (flat boundary) ----------

    @staticmethod
    def _phi(
        gamma: float,
        H: float,
        X: float,
        S_for_phi: float,
        T: float,
        sigma: float,
        r: float,
        b: float,
    ) -> float:
        """
        Paper's '(S,T | gamma; H; X) evaluation function for the flat boundary,
        implemented in a RiskFlow-like stable form.
        """
        eps = 1e-32
        T = max(T, 1e-12)
        sigma2 = max(sigma * sigma, eps)
        volT = max(sigma * math.sqrt(T), eps)

        H_ = max(H, eps)
        X_ = max(X, eps)
        S_ = max(S_for_phi, eps)

        kappa = (2.0 * b) / sigma2 + 2.0 * gamma - 1.0
        d = (math.log(H_ / S_) - (b + (gamma - 0.5) * sigma2) * T) / volT
        lam = -r + gamma * b + 0.5 * gamma * (gamma - 1.0) * sigma2

        log_XS = math.log(X_ / S_)
        safe_exp = min(kappa * log_XS, 25.0)  # stability (mirrors RiskFlow's clamp)

        term1 = BjerksundStenslandOptionPricer._ncdf(d)
        term2 = math.exp(safe_exp) * BjerksundStenslandOptionPricer._ncdf(d - 2.0 * log_XS / volT)
        return math.exp(lam * T) * (term1 - term2)

    def _american_call_single(
        self,
        S: float,
        K: float,
        r: float,
        b: float,
        sigma: float,
        T: float,
        boundary_variant: BoundaryVariant,
    ) -> Tuple[float, float, bool]:
        """
        Flat boundary approximation:
          - boundary I = X_T
          - value per paper Eq. (4) with X=I
          - immediate exercise if S >= I
        """
        eps = 1e-16
        T = max(T, 1e-8)

        # European
        F = S * math.exp(b * T)
        df = math.exp(-r * T)
        euro = self._black76(F, K, sigma, T, df, call=True)

        # boundary I = X_T
        I = self._boundary_XT(K=K, r=r, b=b, sigma=sigma, tau=T, boundary_variant=boundary_variant)

        if S >= I:
            return (max(S - K, 0.0), I, True)

        beta, _, _ = self._beta_B0_B1(K, r, b, sigma)

        alpha_I = (I - K) * (I ** (-beta))
        # RiskFlow-style: keep S_for_phi strictly below the boundary
        S_for_phi = min(max(S, eps) - 1e-10, I)

        phi_beta_II = self._phi(beta, I, I, S_for_phi, T, sigma, r, b)
        phi_1_II = self._phi(1.0, I, I, S_for_phi, T, sigma, r, b)
        phi_1_KI = self._phi(1.0, K, I, S_for_phi, T, sigma, r, b)
        phi_0_II = self._phi(0.0, I, I, S_for_phi, T, sigma, r, b)
        phi_0_KI = self._phi(0.0, K, I, S_for_phi, T, sigma, r, b)

        # Eq. (4) composition
        S_pow_beta = S_for_phi ** beta
        c_flat = (
            alpha_I * S_pow_beta
            - alpha_I * (I ** beta) * phi_beta_II
            + S_for_phi * (phi_1_II - phi_1_KI)
            + K * (phi_0_KI - phi_0_II)
        )

        c_flat = max(euro, c_flat)  # conservative safeguard
        return (c_flat, I, False)

    # ---------- Two-step method (2002 paper, Proposition 1) ----------

    # --- Bivariate normal CDF M(h,k,rho) via numerical integration ---
    def _bivnorm_cdf(self, h: float, k: float, rho: float) -> float:
        """
        Standard bivariate normal CDF:
          M(h,k,rho) = P(X<=h, Y<=k), corr(X,Y)=rho

        Computed as:
          ∫_{-∞}^{h} φ(x) Φ((k - rho x)/sqrt(1-rho^2)) dx

        Uses adaptive Simpson on a truncated domain for stdlib-only implementation.
        """
        # Handle edge cases
        rho = max(min(rho, 0.999999), -0.999999)

        # Very extreme limits
        if h <= -10.0 or k <= -10.0:
            return 0.0
        if h >= 10.0:
            return self._ncdf(k)
        if k >= 10.0:
            return self._ncdf(h)

        denom = math.sqrt(max(1.0 - rho * rho, 1e-16))

        def integrand(x: float) -> float:
            return self._npdf(x) * self._ncdf((k - rho * x) / denom)

        # Truncate lower tail
        a = -10.0
        b = h

        # Adaptive Simpson
        def simpson(f, lo, hi):
            mid = 0.5 * (lo + hi)
            return (hi - lo) * (f(lo) + 4.0 * f(mid) + f(hi)) / 6.0

        def adapt(f, lo, hi, whole, depth):
            mid = 0.5 * (lo + hi)
            left = simpson(f, lo, mid)
            right = simpson(f, mid, hi)
            if depth <= 0:
                return left + right
            # error estimate
            if abs(left + right - whole) <= 1e-10 * (1.0 + abs(whole)):
                return left + right
            return adapt(f, lo, mid, left, depth - 1) + adapt(f, mid, hi, right, depth - 1)

        whole = simpson(integrand, a, b)
        val = adapt(integrand, a, b, whole, depth=12)
        # clamp
        return max(0.0, min(1.0, val))

    def _A_eval(
        self,
        gamma: float,
        H: float,
        X: float,
        x: float,
        t: float,
        T: float,
        S: float,
        r: float,
        b: float,
        sigma: float,
    ) -> float:
        """
        Paper's A-function (ª) in Proposition 1 (Eq. definition),
        expressed via bivariate normal M(·,·,·).

        A = exp(λT) S^γ { M(d1,D1, ρ) - (X/S)^κ M(d2,D2, ρ)
                         - (x/S)^κ M(d3,D3, -ρ) + (x/X)^κ M(d4,D4, -ρ) }
        where ρ = sqrt(t/T), and d's / D's match the paper.
        """
        eps = 1e-16
        T = max(T, 1e-12)
        t = max(min(t, T - 1e-12), 1e-12)
        sigma2 = max(sigma * sigma, 1e-16)

        vol_t = sigma * math.sqrt(t)
        vol_T = sigma * math.sqrt(T)

        S_ = max(S, eps)
        H_ = max(H, eps)
        X_ = max(X, eps)
        x_ = max(x, eps)

        a = b + (gamma - 0.5) * sigma2

        # d's (at t)
        d1 = (-math.log(S_ / x_) + a * t) / vol_t
        d2 = (-math.log((X_ * X_) / (S_ * x_)) + a * t) / vol_t
        d3 = (-math.log(S_ / x_) - a * t) / vol_t
        d4 = (-math.log((X_ * X_) / (S_ * x_)) - a * t) / vol_t

        # D's (at T)
        D1 = (-math.log(S_ / H_) + a * T) / vol_T
        D2 = (-math.log((X_ * X_) / (S_ * H_)) + a * T) / vol_T
        D3 = (-math.log((x_ * x_) / (S_ * H_)) + a * T) / vol_T
        D4 = (-math.log((S_ * x_ * x_) / (H_ * X_ * X_)) + a * T) / vol_T

        # λ and κ (paper Eq. (8)-(9))
        lam = -r + gamma * b + 0.5 * gamma * (gamma - 1.0) * sigma2
        kappa = (2.0 * b) / sigma2 + (2.0 * gamma - 1.0)

        rho = math.sqrt(t / T)

        M1 = self._bivnorm_cdf(d1, D1, rho)
        M2 = self._bivnorm_cdf(d2, D2, rho)
        M3 = self._bivnorm_cdf(d3, D3, -rho)
        M4 = self._bivnorm_cdf(d4, D4, -rho)

        # power terms
        pow_XS = math.exp(kappa * math.log(X_ / S_))
        pow_xS = math.exp(kappa * math.log(x_ / S_))
        pow_xX = math.exp(kappa * math.log(x_ / X_))

        inner = (M1 - pow_XS * M2 - pow_xS * M3 + pow_xX * M4)
        return math.exp(lam * T) * (S_ ** gamma) * inner

    def _american_call_two_step(
        self,
        S: float,
        K: float,
        r: float,
        b: float,
        sigma: float,
        T: float,
        boundary_variant: BoundaryVariant,
    ) -> Tuple[float, float, float, float, bool]:
        """
        Two-step boundary approximation (paper Proposition 1).

        - Split time:
            t = 0.5*(sqrt(5)-1)*T
        - Boundaries:
            X = X_T
            x = X_{T-t}
          where X_tau is computed via Eq. (10)-(13) using chosen boundary variant.

        Returns:
          (c_two_step, X, x, t, exercised_now)
        """
        eps = 1e-16
        T = max(T, 1e-8)

        # European safeguard
        F = S * math.exp(b * T)
        df = math.exp(-r * T)
        euro = self._black76(F, K, sigma, T, df, call=True)

        beta, _, _ = self._beta_B0_B1(K, r, b, sigma)

        # Eq. (16)
        t_split = 0.5 * (math.sqrt(5.0) - 1.0) * T
        t_split = max(min(t_split, T - 1e-10), 1e-10)

        # Eq. (17)-(18)
        X = self._boundary_XT(K=K, r=r, b=b, sigma=sigma, tau=T, boundary_variant=boundary_variant)
        x = self._boundary_XT(K=K, r=r, b=b, sigma=sigma, tau=T - t_split, boundary_variant=boundary_variant)

        # enforce X > x > K (theoretical); keep robust in weird parameter corners
        x = max(min(x, X - 1e-12), K + 1e-12)

        if S >= X:
            return (max(S - K, 0.0), X, x, t_split, True)

        # α(·) (paper Eq. (5))
        alpha_X = (X - K) * (X ** (-beta))
        alpha_x = (x - K) * (x ** (-beta))

        # φ terms (paper's ' function) with maturity t_split and barrier X
        S_for_phi = min(max(S, eps) - 1e-10, X)

        phi_beta_X_X = self._phi(beta, X, X, S_for_phi, t_split, sigma, r, b)
        phi_1_X_X = self._phi(1.0, X, X, S_for_phi, t_split, sigma, r, b)
        phi_1_x_X = self._phi(1.0, x, X, S_for_phi, t_split, sigma, r, b)
        phi_0_X_X = self._phi(0.0, X, X, S_for_phi, t_split, sigma, r, b)
        phi_0_x_X = self._phi(0.0, x, X, S_for_phi, t_split, sigma, r, b)
        phi_beta_x_X = self._phi(beta, x, X, S_for_phi, t_split, sigma, r, b)

        # A terms (paper's ª function) at final maturity T
        A_beta_x = self._A_eval(beta, x, X, x, t_split, T, S_for_phi, r, b, sigma)
        A_1_x = self._A_eval(1.0, x, X, x, t_split, T, S_for_phi, r, b, sigma)
        A_1_K = self._A_eval(1.0, K, X, x, t_split, T, S_for_phi, r, b, sigma)
        A_0_x = self._A_eval(0.0, x, X, x, t_split, T, S_for_phi, r, b, sigma)
        A_0_K = self._A_eval(0.0, K, X, x, t_split, T, S_for_phi, r, b, sigma)

        # Proposition 1 composition
        c_two = (
            alpha_X * (S_for_phi ** beta)
            - alpha_X * ((X ** beta) * phi_beta_X_X)
            + (phi_1_X_X - phi_1_x_X) * S_for_phi
            - K * phi_0_X_X + K * phi_0_x_X
            + alpha_x * ((x ** beta) * phi_beta_x_X)
            - alpha_x * A_beta_x
            + A_1_x - A_1_K
            - K * A_0_x + K * A_0_K
        )

        c_two = max(euro, c_two)  # conservative safeguard
        return (c_two, X, x, t_split, False)
