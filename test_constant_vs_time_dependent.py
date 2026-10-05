"""
TEST GATE: constant Heston vs. time-dependent (piecewise-constant) Heston.

Two separate things are checked here, and they are NOT the same question --
read both sections before reading the pass/fail verdicts.

GATE 1 -- SELF-CONSISTENCY (a correctness check on the new recursion maths)
    If every bucket of a HestonTDParams is given the SAME (kappa, theta, sigma,
    rho), heston_f_td's backward Riccati recursion must reduce EXACTLY to the
    constant model's own closed-form heston_f -- that's the identity the
    derivation in heston_calibration_time.py is built to satisfy. This is
    checked directly against heston_calibration.py's untouched pricer, with
    both a single-bucket and a multi-bucket (same params in every bucket) case.
    A failure here means the new piecewise characteristic function has a bug;
    nothing else in this file can be trusted until it passes.

GATE 2 -- FIT QUALITY, PER MATURITY (the economically meaningful comparison)
    The constant model and the time-dependent model are NOT expected to give
    the same prices once genuinely calibrated -- that would defeat the purpose
    of going time-dependent. What IS expected, and what this gate checks:

      (a) every bucket of the calibrated time-dependent model satisfies the
          Feller condition 2*kappa_i*theta_i > sigma_i^2 on its own (Elices
          2008 Sec. V; Benhamou-Gobet-Miri 2010 Assumption (P) -- both require
          this POINTWISE across the horizon, not just on average), and

      (b) at EVERY maturity, the time-dependent model's fit to that maturity's
          own smile is at least as good as the constant model's (within a
          small numerical tolerance) -- since the time-dependent model has
          strictly more degrees of freedom available at every maturity, it
          should never do materially WORSE there. Where it does only about as
          well as the constant model, that maturity's smile is one the single
          global theta was already fitting fine; where it does noticeably
          BETTER, that's the term-structure slack the constant model couldn't
          reach -- report, don't fail, on the SIZE of that gap, since a large
          gap is exactly the evidence motivating the time-dependent model in
          the first place, not a defect.

Usage
    python test_constant_vs_time_dependent.py                  # quick (SLSQP), recalibrates
    python test_constant_vs_time_dependent.py --use-saved       # read existing JSON files instead
    python test_constant_vs_time_dependent.py --method asa+solver --max-evals 3000
Exit code 0 = both gates passed, 1 = a gate failed (message explains which).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from heston_calibration import (
    HestonParams, MarketData, Calibrator, heston_call,
    EXAMPLE_S0, EXAMPLE_PRICES, load_params,
)
from heston_calibration_time import (
    HestonBucket, HestonTDParams, BootstrapCalibrator,
    heston_call_td, load_td_params,
)

TOL_SELF_CONSISTENCY = 1e-6    # abs price error, self-consistency gate
TOL_FIT_SLACK = 0.0005         # 0.05 vol pts slack before TD counts as "worse" at a maturity


# =============================================================================
# GATE 1 -- self-consistency
# =============================================================================
def test_self_consistency(verbose=True) -> bool:
    p = HestonParams(kappa=2.0, theta=0.05, sigma=0.6, rho=-0.6, v0=0.04)
    S0, r, q = 100.0, 0.03, 0.01
    Ks = np.array([70.0, 85.0, 95.0, 100.0, 105.0, 115.0, 130.0])
    Ts = np.array([0.25, 0.5, 1.0, 2.0, 3.0])

    # (a) single bucket spanning the whole horizon
    td_single = HestonTDParams(p.v0, [HestonBucket(0.0, 5.0, p.kappa, p.theta, p.sigma, p.rho)])
    # (b) three buckets, same params in every bucket -- exercises the backward
    #     chaining across bucket boundaries, not just a single Riccati solve
    bounds = [0.0, 0.4, 1.3, 5.0]
    td_multi = HestonTDParams(p.v0, [
        HestonBucket(bounds[i], bounds[i + 1], p.kappa, p.theta, p.sigma, p.rho)
        for i in range(len(bounds) - 1)
    ])

    max_err = 0.0
    for T in Ts:
        c_const = heston_call(p, S0, Ks, T, r=r, q=q)
        c_single = heston_call_td(td_single, S0, Ks, T, r=r, q=q)
        c_multi = heston_call_td(td_multi, S0, Ks, T, r=r, q=q)
        max_err = max(max_err, np.max(np.abs(c_const - c_single)),
                      np.max(np.abs(c_const - c_multi)))

    ok = max_err < TOL_SELF_CONSISTENCY
    if verbose:
        print("GATE 1 -- self-consistency (degenerate time-dependent == constant)")
        print(f"  max |price_td - price_constant| over {len(Ts)} maturities x "
             f"{len(Ks)} strikes x {{1-bucket, 3-bucket}} = {max_err:.3e}")
        print(f"  {'PASS' if ok else '*** FAIL ***'} (tolerance {TOL_SELF_CONSISTENCY:.0e})\n")
    return ok


# =============================================================================
# market data (shared by both calibrations, so the comparison is apples-to-apples)
# =============================================================================
def build_market() -> MarketData:
    r, T, K, px, bid, ask = EXAMPLE_PRICES.T
    return MarketData.from_prices(EXAMPLE_S0, T, K, px, r=r, q=0.0, option_type="call",
                                  bid=bid, ask=ask)


def get_constant(market, method, max_evals, seed, use_saved):
    if use_saved:
        try:
            p, _ = load_params("heston_params_vol.json")
            print("Loaded constant parameters from heston_params_vol.json")
            return p
        except FileNotFoundError:
            print("heston_params_vol.json not found, calibrating instead")
    cal = Calibrator(market, weights="spread", objective="vol")
    res = cal.calibrate(method=method, max_evals=max_evals, seed=seed, verbose=False)
    return res.params


def get_time_dependent(market, method, max_evals, seed, use_saved):
    if use_saved:
        try:
            td, _ = load_td_params("heston_td_params_vol.json")
            print("Loaded time-dependent parameters from heston_td_params_vol.json")
            return td
        except FileNotFoundError:
            print("heston_td_params_vol.json not found, calibrating instead")
    cal = BootstrapCalibrator(market, weights="spread", objective="vol")
    res = cal.calibrate(method=method, max_evals=max_evals, seed=seed, verbose=False)
    return res.td


# =============================================================================
# GATE 2 -- fit quality per maturity
# =============================================================================
def test_fit_by_maturity(market, const_params, td_params, verbose=True) -> bool:
    from heston_calibration import heston_vols
    from heston_calibration_time import heston_vols_td

    feller_ok = td_params.feller_ok()
    if verbose:
        print("GATE 2 -- fit quality per maturity, and per-bucket Feller")
        print(f"\nTime-dependent parameter path:\n{td_params.param_path_table()}")
        print(f"\nFeller (2*kappa_i*theta_i > sigma_i^2) satisfied in every bucket: "
             f"{'YES' if feller_ok else 'NO *** FAIL ***'}\n")
        print(f"{'T':>8} {'n':>4} {'const RMSE':>12} {'TD RMSE':>10} {'TD vs const':>12} "
             f"{'max|vol diff|':>14} {'verdict':>10}")

    all_ok = feller_ok
    rows = []
    for T in np.unique(market.T):
        sel = market.T == T
        K, r, q = market.K[sel], market.r[sel], market.q[sel]
        mkt_vol = market.vol[sel]
        const_vol = heston_vols(const_params, market.S0, K, T, r, q)
        td_vol = heston_vols_td(td_params, market.S0, K, T, np.median(r), np.median(q))

        const_rmse = float(np.sqrt(np.nanmean((const_vol - mkt_vol) ** 2)))
        td_rmse = float(np.sqrt(np.nanmean((td_vol - mkt_vol) ** 2)))
        max_diff = float(np.nanmax(np.abs(td_vol - const_vol)))

        ok_here = td_rmse <= const_rmse + TOL_FIT_SLACK
        all_ok = all_ok and ok_here
        verdict = "ok" if ok_here else "*** WORSE ***"
        rows.append((T, int(sel.sum()), const_rmse, td_rmse, max_diff, verdict))
        if verbose:
            print(f"{T:8.4f} {int(sel.sum()):4d} {100*const_rmse:11.3f}% {100*td_rmse:9.3f}% "
                 f"{100*(td_rmse-const_rmse):+11.3f}pp {100*max_diff:13.3f}pp {verdict:>10}")

    if verbose:
        print("\n(TD RMSE should be <= const RMSE everywhere -- that's the gate. A LARGE")
        print(" positive gap between the two models' prices at a maturity, while TD's own")
        print(" fit stays good, is the term-structure slack a single constant theta could")
        print(" not reach -- that is the expected, desired outcome of going time-dependent,")
        print(" not a failure.)\n")
    return all_ok


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Test gate: constant vs time-dependent Heston")
    ap.add_argument("--method", choices=["solver", "asa", "asa+solver"], default="solver",
                    help="recalibration method if --use-saved is not given or files are "
                         "missing; 'solver' (fast, local) is the default so this test runs "
                         "quickly -- pass asa+solver to check the production calibration")
    ap.add_argument("--max-evals", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-saved", action="store_true",
                    help="read heston_params_vol.json / heston_td_params_vol.json instead "
                         "of recalibrating (run run_calibration_vol.py and "
                         "run_calibration_vol_time_heston.py first)")
    args = ap.parse_args()

    gate1 = test_self_consistency()

    market = build_market()
    const_params = get_constant(market, args.method, args.max_evals, args.seed, args.use_saved)
    td_params = get_time_dependent(market, args.method, args.max_evals, args.seed, args.use_saved)
    print(f"\nConstant Heston:  {const_params}\n")

    gate2 = test_fit_by_maturity(market, const_params, td_params)

    print("=" * 70)
    print(f"GATE 1 (self-consistency): {'PASS' if gate1 else 'FAIL'}")
    print(f"GATE 2 (fit quality / Feller): {'PASS' if gate2 else 'FAIL'}")
    print("=" * 70)
    sys.exit(0 if (gate1 and gate2) else 1)


if __name__ == "__main__":
    main()
