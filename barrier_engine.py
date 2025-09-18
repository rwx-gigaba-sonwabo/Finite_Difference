# barrier_engine.py

import numpy as np
from scipy.stats import norm

def _norm_rebate_timing(s: str | None, default: str) -> str:
    if s is None:
        return default
    s = s.strip().lower()
    # accept common misspellings / variants
    if s in ("hit", "pay at hit", "at hit"):
        return "hit"
    if s in ("expiry", "exp", "maturity", "pay at expiry", "at expiry", "expiary", "pay at expiary"):
        return "expiry"
    raise ValueError("rebate timing must be 'hit' or 'expiry'")

class BarrierEngine:
    """
    Barrier option engine implementing Reiner & Rubinstein (1991a) / Merton (1973) A–F factors.

    Flags:
      optionflag    : 'c' (call) or 'p' (put)
      directionflag : 'u' (up) or 'd' (down)
      in_out_flag   : 'i' (knock-in) or 'o' (knock-out)

    Rebate timing (user-selectable):
      rebate_timing_in  : 'expiry' (default) or 'hit'
      rebate_timing_out : 'hit' (default) or 'expiry'

    barrier_status (optional conditioning):
      None or 'not_crossed' : standard unconditional valuation at t=0
      'crossed'             : condition on barrier already having been crossed
                              -> IN  value = vanilla (A)
                              -> OUT value = K   (if rebate_timing_out='hit')
                                            or K*e^{-rT} (if 'expiry')
    """

    def __init__(self,
                 s: float, b: float, r: float, t: float, x: float, sigma: float, h: float,
                 optionflag: str, directionflag: str, in_out_flag: str,
                 k: float,
                 barrier_status: str | None = None,
                 rebate_timing_in: str | None = None,
                 rebate_timing_out: str | None = None):

        # Store params
        self.s = float(s); self.b = float(b); self.r = float(r)
        self.t = float(t); self.x = float(x); self.sigma = float(sigma)
        self.h = float(h); self.k = float(k)

        # Basic checks
        if self.sigma <= 0 or self.t <= 0:
            raise ValueError("sigma and t must be positive.")
        if optionflag.lower() not in ("c", "p"):
            raise ValueError("optionflag must be 'c' or 'p'.")
        if directionflag.lower() not in ("u", "d"):
            raise ValueError("directionflag must be 'u' or 'd'.")
        if in_out_flag.lower() not in ("i", "o"):
            raise ValueError("in_out_flag must be 'i' or 'o'.")
        if barrier_status not in (None, "crossed", "not_crossed"):
            raise ValueError("barrier_status must be None, 'crossed', or 'not_crossed'.")

        # Map flags -> signs
        self.optionflag = optionflag.lower()
        self.directionflag = directionflag.lower()
        self.in_out_flag = in_out_flag.lower()
        self.barrier_status = barrier_status

        self.phi = +1 if self.optionflag == 'c' else -1      # call=+1, put=-1
        self.eta = -1 if self.directionflag == 'u' else +1    # up=+1, down=-1

        # Normalize rebate timing choices
        self.rebate_timing_in  = _norm_rebate_timing(rebate_timing_in,  default="expiry")
        self.rebate_timing_out = _norm_rebate_timing(rebate_timing_out, default="hit")

        # Shorthands
        N = norm.cdf
        s, x, h = self.s, self.x, self.h
        t, r, b, sigma = self.t, self.r, self.b, self.sigma

        sqrtT = np.sqrt(t)
        sigRT = sigma * sqrtT
        ebmt  = np.exp((b - r) * t)  # e^{(b-r)T}
        erT   = np.exp(-r * t)       # e^{-rT}

        # μ and λ
        mu  = (b - 0.5 * sigma**2) / sigma**2
        lam = np.sqrt(mu**2 + 2.0 * r / sigma**2)

        # x1, x2, y1, y2, z (exact per literature)
        x1 = (np.log(s / x) / (sigma * sqrtT)) + (1.0 + mu) * sigRT
        x2 = (np.log(s / h) / (sigma * sqrtT)) + (1.0 + mu) * sigRT
        y1 = (np.log(h**2 / (s * x)) / (sigma * sqrtT)) + (1.0 + mu) * sigRT
        y2 = (np.log(h / s) / (sigma * sqrtT)) + (1.0 + mu) * sigRT
        z  = (np.log(h / s) / (sigma * sqrtT)) + lam * sigRT

        # Convenience powers
        HS_pow_2mu1      = (h / s) ** (2.0 * (mu + 1.0))
        HS_pow_2mu       = (h / s) ** (2.0 * mu)
        HS_pow_mulam_pos = (h / s) ** (mu + lam)
        HS_pow_mulam_neg = (h / s) ** (mu - lam)

        phi, eta, K = self.phi, self.eta, self.k

        # ----- A–F factors (direct) -----
        A = (phi * s * ebmt * N(phi * x1)
             - phi * x * erT * N(phi * (x1 - sigRT)))

        B = (phi * s * ebmt * N(phi * x2)
             - phi * x * erT * N(phi * (x2 - sigRT)))

        C = (phi * s * ebmt * HS_pow_2mu1 * N(eta * y1)
             - phi * x * erT * HS_pow_2mu * N(eta * (y1 - sigRT)))

        D = (phi * s * ebmt * HS_pow_2mu1 * N(eta * y2)
             - phi * x * erT * HS_pow_2mu * N(eta * (y2 - sigRT)))

        # Rebate legs (both forms available)
        E = (K * erT *
             (N(eta * (x2 - sigRT)) - HS_pow_2mu * N(eta * (y2 - sigRT))))
        F = (K * (HS_pow_mulam_pos * N(eta * z)
                  + HS_pow_mulam_neg * N(eta * (z - 2.0 * lam * sigRT))))

        # Publishables
        self.elements = {
            "x1": x1, "x2": x2, "y1": y1, "y2": y2, "z": z,
            "mu": mu, "lambda": lam
        }
        self.factors = {"A": A, "B": B, "C": C, "D": D, "E": E, "F": F}

        # ----- Pricing with optional conditioning on barrier_status -----
        # Rebates chosen by timing (IN usually 'expiry', OUT usually 'hit', but both supported)
        rebate_in  = E if self.rebate_timing_in  == "expiry" else F
        rebate_out = F if self.rebate_timing_out == "hit"   else (K * erT - E)

        # If conditioning on barrier already crossed
        if self.barrier_status == "crossed":
            if self.in_out_flag == 'i':
                self.price_value = A                  # IN: becomes vanilla once crossed
            else:
                self.price_value = K if self.rebate_timing_out == "hit" else K * erT
            self.vanilla_value = A
            return

        # Unconditional valuation at t=0 (piecewise by flag and X vs H)
        eps = 1e-14
        x_gt_h = (self.x - self.h) > eps
        x_lt_h = (self.h - self.x) > eps  # equality falls into the "else" branch below

        if self.optionflag == 'c':   # CALL
            if self.directionflag == 'd':  # DOWN (typical S > H)
                if self.in_out_flag == 'i':   # Down-and-IN Call
                    base = (C if x_gt_h else (A - B + D))
                    self.price_value = base + (rebate_in)
                else:                          # Down-and-OUT Call
                    base = ((A - C) if x_gt_h else (B - D))
                    self.price_value = base + (rebate_out)

            else:  # 'u'  UP (typical S < H)
                if self.in_out_flag == 'i':   # Up-and-IN Call
                    base = (A if x_gt_h else (B - C + D))
                    self.price_value = base + (rebate_in)
                else:                          # Up-and-OUT Call
                    base = (0.0 if x_gt_h else (A - B + C - D))
                    self.price_value = base + (rebate_out)

        else:  # PUT
            if self.directionflag == 'd':  # DOWN (S > H)
                if self.in_out_flag == 'i':   # Down-and-IN Put
                    base = ((B - C + D) if x_gt_h else A)
                    self.price_value = base + (rebate_in)
                else:                          # Down-and-OUT Put
                    base = ((A - B + C - D) if x_gt_h else 0.0)
                    self.price_value = base + (rebate_out)

            else:  # 'u'  UP (S < H)
                if self.in_out_flag == 'i':   # Up-and-IN Put
                    base = ((A - B + D) if x_gt_h else C)
                    self.price_value = base + (rebate_in)
                else:                          # Up-and-OUT Put
                    base = ((B - D) if x_gt_h else (A - C))
                    self.price_value = base + (rebate_out)

        self.vanilla_value = A

    # Public API
    def get_factors(self):  return self.factors
    def get_elements(self): return self.elements
    def price(self) -> float:   return self.price_value
    def vanilla(self) -> float: return self.vanilla_value
