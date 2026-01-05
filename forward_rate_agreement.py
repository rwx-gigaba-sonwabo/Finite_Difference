from datetime import date
from typing import Optional

import QuantLib as ql  # type: ignore

from discount.discount import YieldCurve


class ForwardRateAgreement:
    """
    Wrapper for QuantLib::ForwardRateAgreement.

    Parameters
    ----------
    settle_date
        settlement/fixing date of the FRA
    maturity_date
        end of the underlying period
    position
        "long" or "short"
    strike_rate
        the FRA contract rate (decimal, e.g. 0.018 for 1.8%)
    notional
        notional principal
    frequency
        "quarterly" or "semi-annual"
    """

    def __init__(
        self,
        settle_date: date,
        maturity_date: date,
        position: str,
        strike_rate: float,
        notional: float,
        frequency: str = "quarterly",
    ) -> None:
        self.settle_date = settle_date
        self.maturity_date = maturity_date
        self.position = position.lower()
        self.strike_rate = strike_rate
        self.notional = notional

        freq = frequency.lower()
        if freq in ("semi-annual", "semiannual", "6m"):
            self.index_tenor = ql.Period(6, ql.Months)
        else:
            # default to quarterly
            self.index_tenor = ql.Period(3, ql.Months)

    def to_quantlib_fra(
        self,
        yield_curve: YieldCurve,
        ibor_index: Optional[ql.IborIndex] = None,
    ) -> ql.ForwardRateAgreement:
        """
        Build a QuantLib ForwardRateAgreement using the curve's discount
        handle.
        """
        handle = yield_curve.discount_curve

        # create a 3m or 6m index if not provided
        if ibor_index is None:
            ibor_index = ql.IborIndex(
                "Jibar",                            # any name
                self.index_tenor,                   # 3M or 6M
                2,                                  # fixingDays (e.g. 2)
                ql.ZARCurrency(),                   # South African rand
                ql.SouthAfrica(),                   # holiday calendar
                ql.ModifiedFollowing,               # business‐day conv.
                False,                              # endOfMonth?
                yield_curve.day_count,              # YOUR day-count
                handle                              # forwarding handle
            )

        pos = (
            ql.Position.Long if self.position == "long" else ql.Position.Short
        )

        ql_settle = ql.Date(
            self.settle_date.day,
            self.settle_date.month,
            self.settle_date.year,
        )
        ql_mat = ql.Date(
            self.maturity_date.day,
            self.maturity_date.month,
            self.maturity_date.year,
        )

        fra = ql.ForwardRateAgreement(
            ibor_index,
            ql_settle,
            ql_mat,
            pos,
            self.strike_rate,
            self.notional,
            handle,
        )
        return fra
