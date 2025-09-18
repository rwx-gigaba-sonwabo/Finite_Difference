# double_barrier_douady_style.py
import numpy as np
from scipy.stats import norm


class DoubleBarrier:

    def __init__(self, S, X, L, U, sigma,
                 callflag: str, inflag: str, m: int = 4):
        self.S = float(S)
        self.X = float(X)        # strike ()
        self.L = float(L)
        self.U = float(U)
        self.sigma = float(sigma)
        self.callflag = callflag.lower()  # 'c' or 'p'
        self.inflag = inflag.lower()      # 'in' or 'out'
        self.m = int(m)

    # ----- Black–Scholes with carry b -----------------------------
    @staticmethod
    def _bs_price(callput: str, S: float, K: float, r: float, b: float,
                  sigma: float, T: float) -> float:
        d1 = (np.log(S / K) + (b + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        erT = np.exp(-r * T)
        ebrT = np.exp((b - r) * T)
        if callput == 'c':
            return S * ebrT * norm.cdf(d1) - K * erT * norm.cdf(d2)
        else:
            return K * erT * norm.cdf(-d2) - S * ebrT * norm.cdf(-d1)

    # ------------------ Douady series price --------------------------
    def price(self, b: float, r: float, T: float) -> float:
        # BS price 
        BS_price = self._bs_price(self.callflag, self.S, self.X, r, b, self.sigma, T)

        # Dimensionless variables 
        u = (1.0 / self.sigma) * np.log(self.U / self.S)
        k = (1.0 / self.sigma) * np.log(self.X / self.S)
        l = (1.0 / self.sigma) * np.log(self.L / self.S)

        # lambda, lambda_prime, delta 
        lam = b / self.sigma - self.sigma / 2.0
        lam_prime = b / self.sigma + self.sigma / 2.0
        delta = u - l

        # ---------------------- CALL BRANCH ('c') ----------------------
        if self.callflag == 'c':

            if self.X < self.U:
                orig_lambda = lam
                alpha = max(k, l)
                beta = u

                P_1 = []  # collects (I1 - J1)
                P_2 = []  # collects (I2 - J2)

                for n in range(-self.m, self.m + 1):
                    
                    lam = lam_prime
                    I_1 = np.exp(-2 * n * lam * delta) * (norm.cdf((beta + 2 * n * delta) / np.sqrt(T) - lam * np.sqrt(T))-norm.cdf((alpha+2*n*delta)/np.sqrt(T) - lam*np.sqrt(T)))
                    J_1 = np.exp(2*lam*(n*delta+u))*(norm.cdf((2*u-alpha+2*n*delta)/np.sqrt(T) + lam*np.sqrt(T))-norm.cdf((2*u-beta+2*n*delta)/np.sqrt(T) + lam*np.sqrt(T)))

                    # THEN swap back to original lambda
                    lam = orig_lambda
                    I_2 = np.exp(-2 * n * lam * delta) * (norm.cdf((beta + 2 * n * delta) / np.sqrt(T) - lam * np.sqrt(T))-norm.cdf((alpha+2*n*delta)/np.sqrt(T) - lam*np.sqrt(T)))
                    J_2 = np.exp(2*lam*(n*delta+u))*(norm.cdf((2*u-alpha+2*n*delta)/np.sqrt(T) + lam*np.sqrt(T))-norm.cdf((2*u-beta+2*n*delta)/np.sqrt(T) + lam*np.sqrt(T)))

                    P_1.append(I_1 - J_1)
                    P_2.append(I_2 - J_2)

                if self.inflag == 'out':
                    return np.exp((b - r) * T) * self.S * np.sum(P_1) - np.exp(-r * T) * self.X * np.sum(P_2)
                elif self.inflag == 'in':
                    return BS_price - (np.exp((b - r) * T) * self.S * np.sum(P_1) - np.exp(-r * T) * self.X * np.sum(P_2))
                else:
                    raise ValueError("Incorrect inflag")

            elif self.X >= self.U:
                if self.inflag == 'out':
                    return 0.0
                elif self.inflag == 'in':
                    return BS_price
                else:
                    raise ValueError("Incorrect inflag")

            else:
                raise ValueError("Void inputs")

        # ---------------------- PUT BRANCH ('p') ----------------------
        elif self.callflag == 'p':

            if self.X > self.L:
                orig_lambda = lam
                alpha = 1
                beta = min(k, u)

                P_1 = []
                P_2 = []

                for n in range(-self.m, self.m + 1):
                    # FIRST use original lambda 
                    lam = orig_lambda
                    I_1 = np.exp(-2 * n * lam * delta) * (norm.cdf((beta + 2 * n * delta) / np.sqrt(T) - lam * np.sqrt(T))-norm.cdf((alpha+2*n*delta)/np.sqrt(T) - lam*np.sqrt(T)))
                    J_1 = np.exp(2*lam*(n*delta+u))*(norm.cdf((2*u-alpha+2*n*delta)/np.sqrt(T) + lam*np.sqrt(T))-norm.cdf((2*u-beta+2*n*delta)/np.sqrt(T) + lam*np.sqrt(T)))
    
                    # THEN switch to lambda_prime
                    lam = lam_prime
                    I_2 = np.exp(-2 * n * lam * delta) * (norm.cdf((beta + 2 * n * delta) / np.sqrt(T) - lam * np.sqrt(T))-norm.cdf((alpha+2*n*delta)/np.sqrt(T) - lam*np.sqrt(T)))
                    J_2 = np.exp(2*lam*(n*delta+u))*(norm.cdf((2*u-alpha+2*n*delta)/np.sqrt(T) + lam*np.sqrt(T))-norm.cdf((2*u-beta+2*n*delta)/np.sqrt(T) + lam*np.sqrt(T)))

                    P_1.append(I_1 - J_1)
                    P_2.append(I_2 - J_2)

                if self.inflag == 'out':
                    return np.exp((- r) * T) * self.X * np.sum(P_1) - np.exp((b-r) * T) * self.S * np.sum(P_2)
                elif self.inflag == 'in':
                    return BS_price - (np.exp((- r) * T) * self.X * np.sum(P_1) - np.exp((b-r) * T) * self.S * np.sum(P_2))
                else:
                    raise ValueError("Incorrect inflag")

            elif self.X <= self.L:
                if self.inflag == 'out':
                    return 0.0
                elif self.inflag == 'in':
                    return BS_price
                else:
                    raise ValueError("Incorrect inflag")

            else:
                raise ValueError("Void inputs")

        else:
            raise ValueError("Incorrect callflag (use 'c' or 'p')")



# Example parameters Double Barrier Calls
S     = 20.786     
X     = 21        
U     = 23        
L     = 19        
r     = 0.0709454892         
b     = 0.049493018       
T     = (49/365)        
sigma = 0.10994120968      

# Choose option style
optFlag = 'c'   # 'c' for call, 'p' for put
inOut   = 'in'   # 'o' for knock-out, 'i' for knock-in
mMax    = 4    # use [-4..4] terms in the series

# instantiate and price
pricer = DoubleBarrier(S, X, L, U, sigma, optFlag, inOut, mMax)

# double barrier option value (scaled by 10000 if you want)
val = pricer.price(b=b, r=r, T=T) * 10000
print("Double Knock In Call:", val)

# vanilla BSM value using the staticmethod (This will be the value of the option if the barrier is crossed)
bsm_val = DoubleBarrier._bs_price(optFlag, S, X, r, b, sigma, T)*10000
print("Vanilla Black–Scholes Call Value:", bsm_val)

#Double Knock Out: 
Double_Knock_out = bsm_val - val
print("Double Knock Out Call using symmetry", Double_Knock_out)


# Example parameters Double Barrier Puts
S     = 17.862     
X     = 19  
U     = 21        
L     = 15        
r     = 0.0709454892         
b     = 0.02526685
T     = (49/365)        
sigma = 0.143176220424

# Choose option style
optFlag = 'p'   # 'c' for call, 'p' for put
inOut   = 'out'   # 'o' for knock-out, 'i' for knock-in
mMax    = 4    # use [-4..4] terms in the series

# instantiate and price
pricer = DoubleBarrier(S, X, L, U, sigma, optFlag, inOut, mMax)

# double barrier option value (scaled by 10000 if you want)
val = pricer.price(b=b, r=r, T=T) * 10000
print("Double Knock Out Put:", val)

# vanilla BSM value using the staticmethod (This will be the value of the option if the barrier is crossed)
bsm_val = DoubleBarrier._bs_price(optFlag, S, X, r, b, sigma, T)*10000
print("Vanilla Black–Scholes value:", bsm_val)

#Double Knock Out: 
Double_Knock_in = bsm_val - val
print("Double Knock Out using symmetry", Double_Knock_in)

