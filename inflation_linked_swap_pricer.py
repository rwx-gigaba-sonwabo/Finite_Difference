import os
import csv

from zero_coupon.instruments.inflation_linked_swap import InflationLinkedSwap
import QuantLib as ql  # type: ignore
from QuantLib import as_cpi_coupon


class InflationLinkedSwapPricer:
    def __init__(self, swap_def: InflationLinkedSwap):
        self.instrument = swap_def
        self.swap = self.instrument.to_quantlib()

    def npv(self) -> float:
        return self.swap.NPV()

    def fair_fixed_rate(self) -> float:
        """
        Solve for fixed rate r such that
           PV_float_leg = PV_inflation_coupons(r)
        i.e. r = (PV_float) / Annuity,
        where Annuity = sum_i tau_i * (I(t_i)/I_base) * DF(t_i) * Notional
        """
        # discount curve handle
        discount_curve = self.instrument.nominal_handle

        # 1) PV of floating leg (leg index 1)
        pv_float_leg = sum(
            cf.amount() * discount_curve.discount(cf.date())
            for cf in self.swap.leg(1)
        )

        # 2) Build annuity for inflation leg
        base_cpi = self.instrument.base_cpi
        annuity = 0.0
        for cf in self.swap.leg(0):
            cpi_coupon = as_cpi_coupon(cf)
            tau = cpi_coupon.accrualPeriod()
            cpi_t = cpi_coupon.indexFixing()
            index_ratio = cpi_t / base_cpi
            df = discount_curve.discount(cf.date())
            annuity += tau * index_ratio * df * self.instrument.notional

        # 3) Fair fixed rate
        fair_fixed_rate = pv_float_leg / annuity
        return fair_fixed_rate

    def print_details(self) -> None:
        issue_date = ql.Date(
            self.instrument.issue_date.day,
            self.instrument.issue_date.month,
            self.instrument.issue_date.year
        )
        maturity_date = self.swap.maturityDate()
        value_date = self.instrument.nominal_handle.referenceDate()
        side = "Payer" if self.instrument.pay_fixed_leg else "Receiver"
        npv = self.npv()
        fair_fixed_rate = self.fair_fixed_rate()

        print("=" * 60)
        print("Inflation-Linked Swap")
        print(f"  Issue Date     : {issue_date}")
        print(f"  Value Date     : {value_date}")
        print(f"  Maturity Date  : {maturity_date}")
        print(f"  Notional       : {self.instrument.notional:,.2f}")
        print(f"  Side           : {side}")
        print(f"  Fixed rate     : {self.instrument.fixed_rate:.4%}")
        print("-" * 60)
        print(f"NPV              : {npv:,.6f}")
        print(f"Fair fixed rate  : {fair_fixed_rate:.4%}")
        print("=" * 60)

        # export
        root = os.path.dirname(os.path.dirname(__file__))
        out = os.path.join(root, "results")
        os.makedirs(out, exist_ok=True)
        fn = f"ils_{self.swap.maturityDate()}.csv"
        path = os.path.join(out, fn)
        rows = [
            ("Issue Date",        f"{self.instrument.issue_date}"),
            ("Value Date",        f"{value_date}"),
            ("Maturity Date",     f"{maturity_date}"),
            ("Notional",          f"{self.instrument.notional:,.2f}"),
            ("Side",              f"{side}"),
            ("Fixed Rate",        f"{self.instrument.fixed_rate:.4%}"),
            ("NPV",               f"{npv:.6f}"),
            ("Fair Fixed Rate",   f"{fair_fixed_rate:.4%}"),
        ]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        print(f"Results exported to {path}")
