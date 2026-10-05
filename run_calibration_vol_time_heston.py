"""
Same bootstrap calibration as run_calibration_time_heston.py (every setting read
from that file) but with the objective changed to equally-weighted implied-vol
squared error per bucket, mirroring run_calibration_vol.py's relationship to
run_calibration.py.

    python run_calibration_vol_time_heston.py

Output goes to its own files so the two time-dependent runs (price- vs
vol-objective) can be compared side by side, same pattern as the constant model.
"""
import run_calibration_time_heston as rc

rc.OBJECTIVE = "vol"
rc.WEIGHTS = "downside"       # every quote weight 1, except 2x for K <= S0 (ATM)

rc.OUTPUT_FILE = "heston_td_params_vol.json"
rc.PLOT_FILE = "heston_td_fit_vol.png"


if __name__ == "__main__":
    rc.main()
