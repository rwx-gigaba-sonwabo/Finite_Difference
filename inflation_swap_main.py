from datetime import date
import QuantLib as ql  # type: ignore
from QuantLib import as_cpi_coupon

from discount.discount import YieldCurve
from zero_coupon.markets.cpi_term_structure import CPITermStructure
from zero_coupon.instruments.inflation_linked_swap import InflationLinkedSwap
from zero_coupon.pricers.inflation_linked_swap_pricer import (
    InflationLinkedSwapPricer,
)


def main() -> None:
    # ────────────────────────────────
    # 1) Build zero discount curve
    # ────────────────────────────────
    value_date = date(2025, 6, 19)
    issue_date = date(2022, 6, 19)
    maturity_date = date(2026, 12, 21)  # QuantLib date format

    ql_value_date = ql.Date(value_date.day, value_date.month, value_date.year)
    ql_issue_date = ql.Date(issue_date.day, issue_date.month, issue_date.year)

    nom_zero_rates = [0.05,
                      0.051,
                      0.052,
                      0.054,
                      0.056,
                      0.055
                      ]
    nom_maturity_dates = [
        date(2025, 6, 23),
        date(2025, 9, 21),
        date(2025, 12, 22),
        date(2026, 3, 20),
        date(2026, 6, 22),
        date(2026, 12, 21)
    ]
    yield_curve = YieldCurve(nom_zero_rates, nom_maturity_dates, value_date)
    nominal_handle = yield_curve.discount_curve  # Handle<YieldTermStructure>

    # ────────────────────────────────
    # 2) Build inflation curve
    # ────────────────────────────────
    cpi_history = {
        # 2021
        date(2021, 1, 1):  81.7,
        date(2021, 2, 1):  82.2,
        date(2021, 3, 1):  82.8,
        date(2021, 4, 1):  83.3,
        date(2021, 5, 1):  83.4,
        date(2021, 6, 1):  83.5,
        date(2021, 7, 1):  84.5,
        date(2021, 8, 1):  84.8,
        date(2021, 9, 1):  85.0,
        date(2021, 10, 1):  85.3,
        date(2021, 11, 1):  85.6,
        date(2021, 12, 1):  86.1,

        # 2022
        date(2022, 1, 1):  86.3,
        date(2022, 2, 1):  86.8,
        date(2022, 3, 1):  87.7,
        date(2022, 4, 1):  88.2,
        date(2022, 5, 1):  88.8,
        date(2022, 6, 1):  89.5,
        date(2022, 7, 1):  91.1,
        date(2022, 8, 1):  91.3,
        date(2022, 9, 1):  91.4,
        date(2022, 10, 1):  91.7,
        date(2022, 11, 1):  92.0,
        date(2022, 12, 1):  92.3,

        # 2023
        date(2023, 1, 1):  92.2,
        date(2023, 2, 1):  92.9,
        date(2023, 3, 1):  93.9,
        date(2023, 4, 1):  94.2,
        date(2023, 5, 1):  94.4,
        date(2023, 6, 1):  94.6,
        date(2023, 7, 1):  95.7,
        date(2023, 8, 1):  96.3,
        date(2023, 9, 1):  97.2,
        date(2023, 10, 1):  97.1,
        date(2023, 11, 1):  97.1,
        date(2023, 12, 1):  98.3,

        # 2024
        date(2024, 1, 1):  97.2,
        date(2024, 2, 1):  98.1,
        date(2024, 3, 1):  98.9,
        date(2024, 4, 1):  99.1,
        date(2024, 5, 1):  99.3,
        date(2024, 6, 1):  99.4,
        date(2024, 7, 1):  99.8,
        date(2024, 8, 1):  99.9,
        date(2024, 9, 1):  100.0,
        date(2024, 10, 1):  99.9,
        date(2024, 11, 1):  99.9,
        date(2024, 12, 1):  100.0,

        # 2025 (so far)
        date(2025, 1, 1): 100.3,
        date(2025, 2, 1): 101.2,
        date(2025, 3, 1): 101.6,
        date(2025, 4, 1): 101.9,
        date(2025, 5, 1): 102.1,
    }

    # Real inflation swap quotes
    inflation_quotes = [
        (date(2025, 6, 23), 2.93),
        (date(2025, 12, 22), 2.95),
        (date(2026, 6, 22), 2.965),
        (date(2026, 12, 21), 2.98),
        (date(2027, 6, 22), 3.0),
    ]
    inflation_dates = [ql.Date(d.day, d.month, d.year)
                       for d, r in inflation_quotes]
    observation_lag = ql.Period(4, ql.Months)
    availability_lag = ql.Period(1, ql.Months)

    cpi_ts = CPITermStructure(
        historical_cpi=cpi_history,
        inflation_zero_rates=inflation_quotes,
        nominal_curve_handle=nominal_handle,
        observation_lag=observation_lag,
        calendar=ql.SouthAfrica(),
        frequency=ql.Monthly,
        day_counter=ql.Actual365Fixed(),
        availability_lag=availability_lag,
        currency=ql.ZARCurrency()
    )
    pw_inflation_curve = cpi_ts.build_handle()
    zero_index = cpi_ts.build_index()

    for T in inflation_dates:
        rate = pw_inflation_curve.zeroRate(
            T,
            ql.Period(0, ql.Days),
            False,
            True
        )
        print(f"{T.ISO()}: {rate:.2%}")
    # (2) ask the index, not the curve:
    I_interp = zero_index.fixing(ql_issue_date)
    print("lagged/interpolated CPI at issue date:", I_interp)
    I_interp_fr = ql.CPI.laggedFixing(
        zero_index,
        ql_issue_date,
        observation_lag,
        ql.CPI.Linear
    )
    print("lagged/interpolated CPI at issue date (fr):", I_interp_fr)

    # ────────────────────────────────
    # 3) Create a Jibar index using your nominal curve
    # ────────────────────────────────
    last_reset_rate = 0.05  # 5% last reset rate
    ibor_index = ql.IborIndex(
        "Jibar6M",
        ql.Period(6, ql.Months),  # tenor
        0,                        # fixing days
        ql.ZARCurrency(),         # currency
        ql.SouthAfrica(),         # calendar
        ql.ModifiedFollowing,     # convention
        False,                    # end-of-month?
        ql.Actual365Fixed(),      # day count
        nominal_handle            # forward curve handle
    )

    # override day count to match our curve (Act/365)
    ibor_index.dayCounter = yield_curve.day_count
    fix_date = ibor_index.fixingDate(ql_value_date)
    ibor_index.addFixing(fix_date, last_reset_rate)

    # ────────────────────────────────
    # 5) Define the Inflation-Linked Swap
    # ────────────────────────────────
    swap_def = InflationLinkedSwap(
        issue_date=issue_date,
        maturity_date=maturity_date,
        notional=1_000_000,
        fixed_rate=0.05,
        ibor_index=ibor_index,
        zero_index=zero_index,
        observation_lag=observation_lag,
        nominal_curve_handle=nominal_handle,
        pay_fixed_leg=True,              # Payer of fixed‐on‐indexed leg
        float_freq=ql.Semiannual,
        float_dc=ql.Actual365Fixed(),
    )

    # ────────────────────────────────
    # 6) Price & print
    # ────────────────────────────────
    pricer = InflationLinkedSwapPricer(swap_def)
    pricer.print_details()

    # ──────────────────────────────────────────────────────────────────────────
    # 8) Dump out each leg’s cash‐flows and PVs
    # ──────────────────────────────────────────────────────────────────────────
    infl_leg = pricer.swap.leg(0)
    float_leg = pricer.swap.leg(1)
    print("  ** Inflation leg **")
    for cf in infl_leg:
        d = cf.date()
        amt = cf.amount()
        ds = d.ISO()  # ISO format date
        df = nominal_handle.discount(d)
        print(
            f"    {ds:>12}  amt={amt:10,.2f}  df={df:.6f}  "
            f"PV={amt*df:10,.2f}"
        )

    print("  ** Floating leg **")
    for cf in float_leg:
        d = cf.date()
        amt = cf.amount()
        ds = d.ISO()
        df = nominal_handle.discount(d)
        print(
            f"    {ds:>12}  amt={amt:10,.2f}  df={df:.6f}  PV={amt*df:10,.2f}"
        )

    for cf in pricer.swap.leg(0):
        cpc = as_cpi_coupon(cf)
        if cpc is None:
            continue
        manual = ql.CPI.laggedFixing(
            zero_index,
            cf.date(),
            observation_lag,
            ql.CPI.Linear
        )
        print(
            f"  CPI Coupon: {cpc.date().ISO()} "
            f"Fixing={cpc.indexFixing():.6f} "
            f"Manual={manual:.6f} "
        )
        assert abs(cpc.indexFixing() - manual) < 1e-12


if __name__ == "__main__":
    main()
