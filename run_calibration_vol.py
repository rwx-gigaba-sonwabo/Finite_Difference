"""
Same calibration as run_calibration.py (every setting is read from that file:
market data, method, integration, formulation, bounds, starting guess, seed, ...)
but with the objective changed to the equally weighted squared error in implied vol:

    SSE = sum_i (vol_model_i - vol_market_i)^2

    python run_calibration_vol.py

Output goes to its own files so the two runs can be compared side by side.
"""
import run_calibration as rc

rc.OBJECTIVE = "vol"                # squared implied-vol errors
rc.WEIGHTS = "downside"                # every quote has weight 1 | "downside" = 2x weight for K <= S0 (ATM)
rc.CHECK_PAPER_TABLE = False        # the paper's S(Omega) values are price errors, not comparable

rc.OUTPUT_FILE = "heston_params_vol.json"
rc.PLOT_FILE = "heston_fit_vol.png"


if __name__ == "__main__":
    rc.main()
