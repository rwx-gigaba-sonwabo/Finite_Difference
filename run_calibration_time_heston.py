"""
Run a TIME-DEPENDENT (piecewise-constant) Heston calibration -- the bootstrap
companion to run_calibration.py. Edit the SETTINGS and MARKET DATA sections,
then:

    python run_calibration_time_heston.py

Output: a report in the terminal (parameter path across buckets + per-maturity
fit), the calibrated buckets in OUTPUT_FILE, and optionally a plot.

This does NOT touch run_calibration.py or heston_calibration.py. Same market
data (EXAMPLE_PRICES, the Moodley 2005 Anglo American calls) is used by default
so the two calibrations are directly comparable -- see
test_constant_vs_time_dependent.py for the actual side-by-side comparison.
"""
import numpy as np

from heston_calibration_time import (
    BootstrapCalibrator, BUCKET_BOUNDS,
)
from heston_calibration import MarketData, EXAMPLE_S0, EXAMPLE_PRICES

# =============================================================================
# SETTINGS
# =============================================================================
INPUT = "prices"            # "prices" | "svi" | "csv"
METHOD = "asa+solver"       # "solver" | "asa" | "asa+solver" -- same choices as
                             # the constant model, applied once PER BUCKET
OBJECTIVE = "price"           # "vol" = implied-vol errors | "price" = Moodley's S(Omega)
FELLER = True                # enforce 2*kappa_i*theta_i > sigma_i^2 in EVERY bucket
WEIGHTS = "spread"           # "spread" | "equal" | "downside" -- same formulas as
                             # Calibrator._weights, applied within each maturity slice
MAX_EVALS = 3000             # ASA iterations PER BUCKET (5 buckets x 3000 ~ 15000 evals total)
SEED = 0
BOUNDS = BUCKET_BOUNDS       # kappa, theta, sigma, rho bounds (v0 fixed from the shortest maturity)

# {bucket_index: (lo_pct, hi_pct)}, 0-based in maturity order (0 = shortest). Restricts
# which quotes THAT bucket is calibrated against to K/S0 in [lo_pct, hi_pct] -- the rest
# of that bucket's own wings, and every OTHER bucket, are untouched. Use this when a
# short-dated bucket's deep OTM/ITM quotes (wide bid-ask, near-zero vega) are dragging
# kappa/sigma to extremes and producing NaN implied vols -- e.g. {0: (0.90, 1.10)} fits
# only 90-110% moneyness on the shortest bucket. {} (default) = every bucket uses its
# full smile, unchanged from before this option existed.
MONEYNESS_BOUNDS = {}

OUTPUT_FILE = "heston_td_params.json"
PLOT = True
PLOT_FILE = "heston_td_fit.png"

# Market (same example data as run_calibration.py, so the two outputs line up
# maturity-for-maturity in the comparison test)
S0 = EXAMPLE_S0
PRICES = EXAMPLE_PRICES

# INPUT = "csv": header T,K,price[,bid,ask][,type][,r][,q]
CSV_FILE = "my_options.csv"
R, Q = 0.05, 0.01


# =============================================================================
# RUN
# =============================================================================
def build_market() -> MarketData:
    if INPUT == "prices":
        rate, T, K, price, bid, ask = PRICES.T
        return MarketData.from_prices(S0, T, K, price, r=rate, q=0.0, option_type="call",
                                      bid=bid, ask=ask)
    if INPUT == "csv":
        return MarketData.from_csv(CSV_FILE, S0, r=R, q=Q)
    raise ValueError("INPUT must be 'prices' or 'csv' in this driver "
                     "(use --source svi on the command line for an SVI run)")


def main():
    market = build_market()
    n_buckets = len(np.unique(market.T))
    print(f"{len(market)} market implied vols from {market.source}, {n_buckets} maturity buckets")
    print(f"Free parameters: 1 (v0) + 4 x {n_buckets} buckets = {1 + 4 * n_buckets}  "
         f"(vs. 5 for the constant model)\n")

    cal = BootstrapCalibrator(market, feller=FELLER, bounds=BOUNDS,
                              weights=WEIGHTS, objective=OBJECTIVE,
                              moneyness_bounds=MONEYNESS_BOUNDS)
    res = cal.calibrate(method=METHOD, max_evals=MAX_EVALS, seed=SEED, verbose=True)
    res.report()
    res.save(OUTPUT_FILE)
    if PLOT:
        res.plot(save=PLOT_FILE)
    return res


if __name__ == "__main__":
    main()
