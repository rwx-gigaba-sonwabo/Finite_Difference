"""
Run a Heston calibration. Edit the SETTINGS and MARKET DATA sections, then:

    python run_calibration.py

Output: a report in the terminal, the calibrated parameters in OUTPUT_FILE
(read by heston_monte_carlo.py) and, optionally, a plot of the fit.
"""
import numpy as np

from heston_calibration import (
    MarketData, Calibrator, HestonParams, compare_integration,
    EXAMPLE_S0, EXAMPLE_PRICES, DEFAULT_BOUNDS,
)

# =============================================================================
# SETTINGS
# =============================================================================
# The values below reproduce Moodley (2005), section 3.3, table for 20 October 2005:
# price objective weighted by 1/|bid - offer|, the paper's (Heston 1993) form of the
# characteristic function, adaptive Gauss-Lobatto (MATLAB quadl) on [0, 100], the
# paper's ASA bounds and its initial estimate.
# For a production calibration use instead:
#   OBJECTIVE = "vol", FORMULATION = "stable", INTEGRATION = "legendre", BOUNDS = DEFAULT_BOUNDS
# (the "paper" formulation misprices the T = 0.63 options; see PAPER_TABLE check below).
INPUT = "prices"            # "prices" | "svi" | "csv"
METHOD = "asa"              # "solver" | "asa" | "asa+solver"
OBJECTIVE = "price"         # "price" = paper's S(Omega) | "vol" = implied-vol errors
INTEGRATION = "lobatto"     # "legendre" | "lobatto" | "simpson" | "scipy"
FORMULATION = "paper"       # "stable" | "paper"
FELLER = True               # enforce 2*kappa*theta > sigma^2
WEIGHTS = "spread"          # "spread" = price: 1/|bid-offer|, vol: 1/(bid-offer vol band)^2 | "equal" | "downside" = 2x for K <= S0
MAX_EVALS = 3000            # ASA iterations = x-axis of the convergence plot (about 3 min with lobatto).
                            # The result depends on this value and on SEED: 3000 with seed 0 lands near the paper's ASA row.
SEED = 0                    # ASA random seed
X0 = [5, 0.057, 0.7, -0.75, 0.16]  # starting guess [kappa, theta, sigma, rho, v0], or None for automatic
#          kappa        theta        sigma        rho          v0       (paper A.4.2: lb = [0 0 0 -1 0], ub = [10 1 5 0 1])
BOUNDS = [[1e-3, 10.0], [1e-4, 1.0], [1e-3, 5.0], [-0.999, 0.0], [1e-4, 1.0]]   # or DEFAULT_BOUNDS

# Paper's results for this data set: {method: ([kappa, theta, sigma, rho, v0], reported S(Omega))}.
# With CHECK_PAPER_TABLE the objective is evaluated at each of them before calibrating.
CHECK_PAPER_TABLE = True
PAPER_TABLE = {
    "Initial estimate": ([5, 0.057, 0.7, -0.75, 0.16], 148.32),
    "lsqnonlin": ([15.096, 0.1604, 2.0859, -0.7416, 0.1469], 88.58),
    "ASA": ([10, 0.1072, 1.4189, -0.8236, 0.1829], 77.38),
    "Solver": ([7.3284, 0.0745, 1.0227, -0.7670, 0.1938], 89.88),
}

# Separate files for the paper reproduction, so heston_monte_carlo.py (which reads
# heston_params.json) does not pick up parameters fitted with the "paper" formulation.
OUTPUT_FILE = "heston_params_paper.json"
PLOT = True
PLOT_FILE = "heston_fit_paper.png"  # also save the figure here, or None
COMPARE_INTEGRATION = False  # reprice the result with all four integration rules

# Market
S0 = EXAMPLE_S0
R = 0.05                    # risk-free rate (continuous)
Q = 0.01                    # dividend yield (continuous)

# =============================================================================
# MARKET DATA
# =============================================================================
# Moodley (2005) Anglo American call data: columns are (r-q, T, K, mid price, bid, offer).
# The paper's convention passes r-q as r and uses q=0. The bid/offer prices give the
# vol band shown in the plot and used by WEIGHTS = "spread".
PRICES = EXAMPLE_PRICES

# INPUT = "svi": raw SVI per maturity {T: (a, b, rho, m, sigma)},
# w(k) = a + b*(rho*(k - m) + sqrt((k - m)^2 + sigma^2)),  k = ln(K/F)
SVI_SLICES = {
    0.25: (0.008, 0.04, -0.6, 0.0, 0.10),
    0.50: (0.016, 0.06, -0.6, 0.0, 0.12),
    1.00: (0.032, 0.08, -0.6, 0.0, 0.15),
    2.00: (0.064, 0.11, -0.6, 0.0, 0.20),
}
SVI_STRIKES = S0 * np.linspace(0.8, 1.2, 9)  # same strikes for every T, or {T: [strikes]}

# INPUT = "csv": header T,K,price[,bid,ask][,type][,r][,q]  (missing r/q columns use R, Q above)
CSV_FILE = "my_options.csv"


# =============================================================================
# RUN
# =============================================================================
def build_market() -> MarketData:
    if INPUT == "prices":
        rate, T, K, price, bid, ask = PRICES.T
        return MarketData.from_prices(S0, T, K, price, r=rate, q=0.0, option_type="call",
                                      bid=bid, ask=ask)
    if INPUT == "svi":
        return MarketData.from_svi(S0, SVI_SLICES, r=R, q=Q, strikes=SVI_STRIKES)
    if INPUT == "csv":
        return MarketData.from_csv(CSV_FILE, S0, r=R, q=Q)
    raise ValueError("INPUT must be 'prices', 'svi' or 'csv'")


def main():
    market = build_market()
    print(f"{len(market)} market implied vols from {market.source}")

    cal = Calibrator(market, integration=INTEGRATION, formulation=FORMULATION, feller=FELLER,
                     bounds=BOUNDS, weights=WEIGHTS, objective=OBJECTIVE)
    if CHECK_PAPER_TABLE and INPUT == "prices":
        print(f"\nObjective at the paper's parameter sets ({OBJECTIVE}, {FORMULATION}, {INTEGRATION})")
        print(f"{'':18s} {'kappa':>8} {'theta':>7} {'sigma':>7} {'rho':>8} {'v0':>7} {'here':>10} {'paper':>8}")
        for name, (x, reported) in PAPER_TABLE.items():
            print(f"{name:18s} {x[0]:8.4f} {x[1]:7.4f} {x[2]:7.4f} {x[3]:8.4f} {x[4]:7.4f} "
                  f"{cal.sse(x):10.2f} {reported:8.2f}")
        print()
    x0 = cal.initial_guess() if X0 is None else np.asarray(X0, float)
    print(f"Start: {HestonParams.from_array(x0)}   objective={cal.sse(x0):.6e}")

    res = cal.calibrate(method=METHOD, x0=x0, max_evals=MAX_EVALS, seed=SEED, verbose=True)
    res.report()
    res.save(OUTPUT_FILE)

    if COMPARE_INTEGRATION:
        compare_integration(res.params, market, FORMULATION)
    if PLOT:
        res.plot(save=PLOT_FILE)
    return res


if __name__ == "__main__":
    main()
