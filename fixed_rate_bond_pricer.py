import os
import csv

import QuantLib as ql  # type: ignore

from zero_coupon.instruments.fixed_rate_bond import FixedRateBond
from discount.discount import YieldCurve


class FixedRateBondPricer:
    """
    Wraps a QuantLib FixedRateBond, exposes dirty/clean price, YTM
    """
    def __init__(
        self,
        bond_def: FixedRateBond,
        yield_curve: YieldCurve,
    ):
        self.bond_def = bond_def
        self.yield_curve = yield_curve
        self.ql_fix_bond = bond_def.to_quantlib_bond(self.yield_curve)

    def dirty_price(self) -> float:
        return self.ql_fix_bond.dirtyPrice()

    def clean_price(self) -> float:
        dirty_price = self.dirty_price()
        accrued_amount = self.accrued_amount()
        clean_price = dirty_price - accrued_amount
        return clean_price

    def accrued_amount(self) -> float:
        """Calculate Acrrued Interest based BESA Specifications
        CUMEX = 1 if S < BCD (Book close Date)
              = 0 if S ≥ BCD

        The number of days accrued interest, DAYSACC, as at Settle date (S):
        DAYSACC = S - LCD if CUMEX = 1 (Last Coupon Date)
                = S - NCD if CUMEX = 0 (Next Coupon Date)

        Acrrued Interest = (DAYSACC.Coupon_rate*100)/ 365
        """
        settle_date = self.ql_fix_bond.settlementDate()
        ncd = ql.Date(self.bond_def.ncd.day, self.bond_def.ncd.month, self.bond_def.ncd.year)
        lcd = ql.Date(self.bond_def.lcd.day, self.bond_def.lcd.month, self.bond_def.lcd.year)
        calendar = self.bond_def.calendar
        book_close_date = calendar.advance(ncd,
                                           -ql.Period(10, ql.Days))
        coupon_rate = self.bond_def.coupon_rate

        if settle_date < book_close_date:
            cumex = 1
        else:
            cumex = 0

        if cumex == 1:
            daysacc = float(settle_date - lcd)
        else:
            daysacc = float(settle_date - ncd)

        accrued_interest = (daysacc * coupon_rate * 100) / 365

        return accrued_interest

    def yield_to_maturity(self) -> float:
        # QuantLib’s YTM solver:
        """
        ytm = self.ql_fix_bond.bondYield(
            self.clean_price(),        # clean price per 100
            self.yield_curve.day_count,
            ql.Compounded,
            self.bond_def.ql_frequency
        )
        """
        solver = ql.NewtonSafe()
        ytm = ql.BondFunctions.yieldNewtonSafe(
            solver,
            self.ql_fix_bond,
            round(self.clean_price(), 1),
            self.yield_curve.day_count,
            ql.Compounded,
            self.bond_def.ql_frequency,
        )
        return ytm

    def val01(self, value_date, bump, yield_to_maturity=None) -> float:
        if yield_to_maturity is None:
            ytm = self.yield_to_maturity()
        else:
            ytm = yield_to_maturity

        eps = bump
        ytm_term_up = ql.FlatForward(
            value_date,
            ytm + eps,
            self.yield_curve.day_count,
            ql.Compounded,
            self.bond_def.ql_frequency
        )

        ytm_term_down = ql.FlatForward(
            value_date,
            ytm - eps,
            self.yield_curve.day_count,
            ql.Compounded,
            self.bond_def.ql_frequency
        )

        dirty_price_up = ql.BondFunctions.dirtyPrice(
            self.ql_fix_bond,
            ytm_term_up,
            value_date
            )

        dirty_price_down = ql.BondFunctions.dirtyPrice(
            self.ql_fix_bond,
            ytm_term_down,
            value_date
            )
        val01 = (dirty_price_up - dirty_price_down) / (2 * eps)
        return val01 * -eps

    def gamma(
        self,
        value_date: ql.Date,
        bump,
        yield_to_maturity=None
    ) -> float:
        if yield_to_maturity is None:
            ytm = self.yield_to_maturity()
        else:
            ytm = yield_to_maturity

        eps = bump
        ytm_term = ql.FlatForward(value_date,
                                  ytm,
                                  self.yield_curve.day_count,
                                  ql.Compounded,
                                  self.bond_def.ql_frequency)

        ytm_term_up = ql.FlatForward(
            value_date,
            ytm + eps,
            self.yield_curve.day_count,
            ql.Compounded,
            self.bond_def.ql_frequency
        )

        ytm_term_down = ql.FlatForward(
            value_date,
            ytm - eps,
            self.yield_curve.day_count,
            ql.Compounded,
            self.bond_def.ql_frequency
        )

        dirty_price = ql.BondFunctions.dirtyPrice(
            self.ql_fix_bond,
            ytm_term,
            value_date
            )

        dirty_price_up = ql.BondFunctions.dirtyPrice(
            self.ql_fix_bond,
            ytm_term_up,
            value_date
            )

        dirty_price_down = ql.BondFunctions.dirtyPrice(
            self.ql_fix_bond,
            ytm_term_down,
            value_date
            )

        gamma = (
            dirty_price_up - 2 * dirty_price + dirty_price_down
        ) / (eps ** 2)
        return gamma * eps

    def print_details(self) -> None:
        dirty_price = self.dirty_price()
        clean_price = self.clean_price()
        accrued_int = self.accrued_amount()
        ytm = self.yield_to_maturity() * 100
        val01 = self.val01(self.yield_curve.value_date,
                           0.0001)
        gamma = self.gamma(self.yield_curve.value_date,
                           0.0001)

        print("=" * 60)
        print("ZAR/R2030 Bond Validation")
        print(f"  Issue date:     {self.bond_def.issue_date}")
        print(f"  Value date:     {self.ql_fix_bond.settlementDate()}")
        print(f"  Maturity date:  {self.bond_def.maturity_date}")
        print(f"  Notional:       {self.bond_def.notional:,.0f}")
        print(f"  Coupon rate:    {self.bond_def.coupon_rate:.4%}")
        print(f"  Frequency:      {self.bond_def.ql_frequency}")
        print("-" * 60)
        print(f"Dirty price       : {dirty_price:,.6f}")
        print(f"Accrued amount    : {accrued_int:,.6f}")
        print(f"Clean price       : {clean_price:,.6f}")
        print(f"Val01             : {val01:,.6f}")
        print(f"Gamma             : {gamma:,.6f}")
        print(f"Yield to maturity : {ytm:.6f}%")
        print("-" * 60)
        print("Bond Cash flows:")
        for c in self.ql_fix_bond.cashflows():
            print('%20s %12f' % (c.date(), c.amount()))
        print("=" * 60)

        # export
        root = os.path.dirname(os.path.dirname(__file__))
        out = os.path.join(root, "results")
        os.makedirs(out, exist_ok=True)
        fn = f"zar_r2030_{self.bond_def.maturity_date}.csv"
        path = os.path.join(out, fn)
        rows = [
            ("Issue Date",        f"{self.bond_def.issue_date}"),
            ("Maturity Date",     f"{self.bond_def.maturity_date}"),
            ("Notional",          f"{self.bond_def.notional:,.0f}"),
            ("Frequency",         f"{self.bond_def.ql_frequency}"),
            ("Coupon Rate",       f"{self.bond_def.coupon_rate:.4%}"),
            ("Dirty price",       f"{dirty_price:.6f}"),
            ("Accrued amount",    f"{accrued_int:.6f}"),
            ("Clean price",       f"{clean_price:.6f}"),
            ("Yield to maturity", f"{ytm:.6f}%"),
            ("Val01",             f"{val01:.6f}"),
            ("Gamma",             f"{gamma:.6f}"),
        ]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        print(f"Results exported to {path}")
