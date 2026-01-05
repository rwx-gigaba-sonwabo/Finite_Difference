import os
import csv
import math
import QuantLib as ql  # type: ignore

from zero_coupon.instruments.zero_coupon_bond import (
    ZeroCouponBond,
)
from discount.discount import YieldCurve


class ZeroCouponBondPricer:
    """
    Pricer for ZeroCouponBond using a QuantLib yield curve.
    Computes PV = F * DF and PV01 via analytic modified duration, and
    exports valuation details to CSV in a `results/` folder.
    """
    def __init__(self, bond: ZeroCouponBond, yield_curve: YieldCurve):
        self.bond = bond
        self.yield_curve = yield_curve
        self.eval_date = yield_curve.py_value_date

    def present_value(self) -> float:
        """
        PV = face_value * discount_factor(maturity).
        """
        df = self.yield_curve.get_discount_factor(self.bond.maturity_date)
        return self.bond.face_value * df

    def pv01(self, bump: float = 1e-4) -> float:
        """
        PV01 = Modified Duration * PV * 1bp
        """
        zero_rate = self.yield_curve.get_zero_rate(self.bond.maturity_date)
        zero_rate_up = zero_rate + bump
        zero_rate_down = zero_rate - bump

        ql_value_date = ql.Date(
            self.eval_date.day,
            self.eval_date.month,
            self.eval_date.year
        )
        ql_maturity_date = ql.Date(
            self.bond.maturity_date.day,
            self.bond.maturity_date.month,
            self.bond.maturity_date.year
        )
        T = self.yield_curve.day_count.yearFraction(
            ql_value_date, ql_maturity_date
        )

        df_up = math.exp(-zero_rate_up * T)
        df_down = math.exp(-zero_rate_down * T)

        pv_up = self.bond.face_value * df_up
        pv_down = self.bond.face_value * df_down

        pv01 = (pv_down - pv_up) / (2 * bump)
        return pv01 * bump

    def print_details(self) -> None:
        """
        Prints bond specs and valuation results and writes them to CSV.
        """
        pv = self.present_value()
        pv01 = self.pv01()
        zero_rate = self.yield_curve.get_zero_rate(self.bond.maturity_date)

        # Print to console
        print("=" * 50)
        print("Zero-Coupon Bond Details")
        print(f"  Face value:      {self.bond.face_value}")
        print(f"  Maturity date:   {self.bond.maturity_date}")
        print(f"  Valuation date:  {self.eval_date}")
        print(f"  Zero rate (cc):  {zero_rate:.6%}")
        print("-" * 50)
        print(f"Present value:    {pv:.6f}")
        print(f"PV01:             {pv01:.6f}")
        print("=" * 50)

        # Prepare results directory
        project_root = os.getcwd()
        results_dir = os.path.join(project_root, "results")
        os.makedirs(results_dir, exist_ok=True)

        # Write to CSV
        # Filename includes bond maturity for uniqueness
        filename = f"zc_bond_{self.bond.maturity_date}.csv"
        file_path = os.path.join(results_dir, filename)
        with open(file_path, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "face_value",
                "maturity_date",
                "valuation_date",
                "zero_rate_cc",
                "present_value",
                "pv01"
            ])
            writer.writerow([
                self.bond.face_value,
                self.bond.maturity_date,
                self.eval_date,
                f"{zero_rate:.6%}",
                f"{pv:.6f}",
                f"{pv01:.6f}"
            ])

        print(f"Results exported to {file_path}")
