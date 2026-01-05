import os
import csv
import QuantLib as ql  # type: ignore

from zero_coupon.instruments.forward_rate_agreement import ForwardRateAgreement
from discount.discount import YieldCurve


class ForwardRateAgreementPricer:
    """
    Pricer for a QuantLib ForwardRateAgreement.
    """

    def __init__(
        self,
        fra_def: ForwardRateAgreement,
        yield_curve: YieldCurve,
        ibor_index: ql.IborIndex = None,
    ):
        self.fra_def = fra_def
        self.yield_curve = yield_curve
        self.fra = fra_def.to_quantlib_fra(self.yield_curve, ibor_index)

    def npv(self) -> float:
        """Present value (NPV) of the FRA payoff."""
        return self.fra.NPV()

    def forward_rate(self) -> float:
        """The forward rate implied by the FRA."""
        return self.fra.forwardRate().rate()

    def print_details(self) -> None:
        """Prints main metrics and writes them to results/fra.csv."""
        npv = self.npv()
        fr = self.forward_rate()
        settle_date = ql.Date(
            self.fra_def.settle_date.day,
            self.fra_def.settle_date.month,
            self.fra_def.settle_date.year,
        )
        maturity_date = ql.Date(
            self.fra_def.maturity_date.day,
            self.fra_def.maturity_date.month,
            self.fra_def.maturity_date.year,
        )
        val_date = self.yield_curve.value_date

        print("=" * 60)
        print("Forward Rate Agreement Pricing")
        print(f"  Value date      : {val_date}")
        print(f"  Settlement date : {settle_date}")
        print(f"  Maturity        : {maturity_date}")
        print(f"  Position        : {self.fra_def.position}")
        print(f"  Strike rate     : {self.fra_def.strike_rate:.4%}")
        print(f"  Notional        : {self.fra_def.notional:,.2f}")
        print("-" * 60)
        print(f"NPV               : {npv:,.6f}")
        print(f"Fair Forward rate : {fr:.4%}")
        print("=" * 60)

        # write CSV
        root = os.path.dirname(os.path.dirname(__file__))
        outdir = os.path.join(root, "results")
        os.makedirs(outdir, exist_ok=True)
        fname = f"fra_{self.fra_def.maturity_date}.csv"
        path = os.path.join(outdir, fname)

        rows = [
            ("Value date",        val_date),
            ("Settlement date",   settle_date),
            ("Maturity date",     maturity_date),
            ("Position",          self.fra_def.position),
            ("Strike rate",       f"{self.fra_def.strike_rate:.6%}"),
            ("Notional",          f"{self.fra_def.notional:,.2f}"),
            ("NPV",               f"{npv:.6f}"),
            ("Fair Forward rate", f"{fr:.6%}"),
        ]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        print(f"Results exported to {path}")
