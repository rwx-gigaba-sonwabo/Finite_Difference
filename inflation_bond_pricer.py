import pandas as pd
from datetime import date
from curves.discount_curve import DiscountYieldCurveHandler
from curves.historical_cpi import HistoricalCPI
from instruments.utils.swap_schedule import (
    generate_swap_leg_schedule,
    add_interpolated_cpi_notionals_to_schedule,
)


class InflationLinkedBondPricer:
    """
    Inflation-linked bond pricer.

    Handles schedule creation, CPI indexation, coupon payments, and present value
    calculation for inflation-linked bonds.

    The bond pays coupons indexed to CPI and returns an indexed principal at maturity.
    """

    def __init__(
        self,
        issue_date: date,
        maturity_date: date,
        notional: float,
        coupon_rate: float,
        value_date: date = None,
        discount_curve: DiscountYieldCurveHandler = None,
        historical_cpi: HistoricalCPI = None,
        base_cpi: float = None,
        payment_frequency: int = 6,
        calendar: str = "SouthAfrica",
        business_convention: str = "ModifiedFollowing",
        termination_business_convention: str = "ModifiedFollowing",
        date_generation: str = "Backward",
        day_count: str = "ACT/365",
        end_of_month: bool = False,
    ):
        """
        Parameters:
            issue_date: Bond issue date.
            maturity_date: Bond maturity date.
            notional: Notional amount.
            coupon_rate: Fixed real coupon rate (applied to indexed notional).
            value_date: Valuation date (defaults to issue_date).
            discount_curve: Curve for discounting cashflows.
            historical_cpi: Historical CPI data handler.
            base_cpi: Base CPI level at issue (for indexation).
            payment_frequency: Coupon payment frequency in months (default 6).
            calendar: Calendar (default 'SouthAfrica').
            business_convention: Business convention (default 'ModifiedFollowing').
            termination_business_convention: Termination convention.
            date_generation: Schedule rule (default 'Backward').
            day_count: Day count convention (default 'ACT/365').
            end_of_month: End of month convention (default False).
        """
        self.issue_date = issue_date
        self.maturity_date = maturity_date
        self.notional = notional
        self.coupon_rate = coupon_rate
        self.value_date = value_date if value_date else issue_date
        self.discount_curve = discount_curve
        self.historical_cpi = historical_cpi
        self.base_cpi = base_cpi
        self.payment_frequency = payment_frequency
        self.calendar = calendar
        self.business_convention = business_convention
        self.termination_business_convention = termination_business_convention
        self.date_generation = date_generation
        self.day_count = day_count
        self.end_of_month = end_of_month

        # Validation
        if discount_curve is None:
            raise ValueError("discount_curve must be provided")
        if historical_cpi is None:
            raise ValueError("historical_cpi must be provided")
        if base_cpi is None:
            raise ValueError("base_cpi must be provided")

        # Generate schedule and calculate cashflows
        self._generate_schedule()
        self._calculate_cashflows()
        self._calculate_present_values()

    def _generate_schedule(self):
        """Generates coupon payment schedule for the bond."""
        self.schedule = generate_swap_leg_schedule(
            effective_date=self.issue_date,
            maturity_date=self.maturity_date,
            valuation_date=None,  # Keep all dates for bond
            payment_frequency=self.payment_frequency,
            calendar=self.calendar,
            business_convention=self.business_convention,
            termination_business_convention=self.termination_business_convention,
            date_generation=self.date_generation,
            day_count=self.day_count,
            end_of_month=self.end_of_month,
        )

    def _calculate_cashflows(self):
        """Calculates CPI-indexed notionals and cashflows."""
        # Add CPI indexation to notionals
        self.schedule = add_interpolated_cpi_notionals_to_schedule(
            schedule=self.schedule,
            notional=self.notional,
            base_cpi=self.base_cpi,
            historical_cpi=self.historical_cpi,
            frequency=self.payment_frequency,
            use_schedule_end_date=True,
            calendar=self.calendar,
            business_convention=self.business_convention,
        )

        # Calculate coupon cashflows (indexed notional × coupon rate × year fraction)
        self.schedule["Coupon"] = (
            self.schedule["Notional"]
            * self.coupon_rate
            * self.schedule["YearFrac"]
        )

        # Add principal repayment at maturity (last cashflow)
        self.schedule["Principal"] = 0.0
        if not self.schedule.empty:
            last_index = len(self.schedule) - 1
            self.schedule.at[last_index, "Principal"] = self.schedule.at[last_index, "Notional"]

        # Total cashflow = coupon + principal
        self.schedule["Cashflow"] = self.schedule["Coupon"] + self.schedule["Principal"]

        # Filter cashflows after value date
        value_date_dt = pd.to_datetime(self.value_date)
        pay_dates = pd.to_datetime(self.schedule["PayDate"], errors="coerce")
        mask = pay_dates > value_date_dt

        # Zero out cashflows on or before value date
        self.schedule.loc[~mask, "Cashflow"] = 0.0
        self.schedule.loc[~mask, "Coupon"] = 0.0
        self.schedule.loc[~mask, "Principal"] = 0.0

    def _calculate_present_values(self):
        """Calculates present values using discount curve."""
        self.schedule["DiscountFactor"] = self.schedule["PayDate"].apply(
            lambda pay_date: self.discount_curve.get_discount_factor_for_date(pay_date)
        )
        self.schedule["PV"] = (
            self.schedule["Cashflow"] * self.schedule["DiscountFactor"]
        )

    def dirty_price(self) -> float:
        """
        Returns dirty price (present value including accrued interest).
        Price per unit of notional.
        """
        total_pv = self.schedule["PV"].sum()
        return (total_pv / self.notional) * 100.0

    def accrued_interest(self) -> float:
        """
        Calculate accrued interest from last coupon date to value date.
        Returns accrued amount per 100 of face value.
        """
        # Find the last coupon date before or on value date
        value_date_dt = pd.to_datetime(self.value_date)
        start_dates = pd.to_datetime(self.schedule["StartDate"], errors="coerce")

        # Find periods where value_date falls within the accrual period
        mask = (start_dates <= value_date_dt)

        if mask.any():
            # Get the last period before value date
            relevant_period = self.schedule[mask].iloc[-1]

            start_date = relevant_period["StartDate"]
            end_date = relevant_period["EndDate"]
            year_frac = relevant_period["YearFrac"]
            indexed_notional = relevant_period["Notional"]

            # Calculate days from start to value date
            days_accrued = (self.value_date - start_date).days
            total_days = (end_date - start_date).days

            if total_days > 0:
                accrual_fraction = days_accrued / total_days
                # Accrued = indexed_notional × coupon_rate × accrual_fraction × year_frac
                accrued = indexed_notional * self.coupon_rate * accrual_fraction * year_frac
                # Return per 100 of face value
                return (accrued / self.notional) * 100.0

        return 0.0

    def clean_price(self) -> float:
        """
        Returns clean price (dirty price minus accrued interest).
        Price per 100 of face value.
        """
        return self.dirty_price() - self.accrued_interest()

    def pv(self) -> float:
        """Returns present value in currency units."""
        return self.schedule["PV"].sum()

    def index_ratio(self, as_of_date: date = None) -> float:
        """
        Calculate the index ratio at a given date.
        Index ratio = CPI(as_of_date) / CPI(base)

        Parameters:
            as_of_date: Date for index ratio calculation (defaults to value_date)

        Returns:
            Index ratio (floored at 1.0)
        """
        if as_of_date is None:
            as_of_date = self.value_date

        cpi_current = self.historical_cpi.cpi_value(as_of_date)
        return max(cpi_current / self.base_cpi, 1.0)

    def get_schedule(self) -> pd.DataFrame:
        """Returns the bond cashflow schedule."""
        return self.schedule.copy()

    def summary(self) -> dict:
        """Returns summary with prices and bond details."""
        return {
            "pv": self.pv(),
            "dirty_price": self.dirty_price(),
            "clean_price": self.clean_price(),
            "accrued_interest": self.accrued_interest(),
            "index_ratio": self.index_ratio(),
            "notional": self.notional,
            "issue_date": self.issue_date,
            "maturity_date": self.maturity_date,
            "coupon_rate": self.coupon_rate,
        }

    def print_summary(self):
        """Print a formatted summary of the bond valuation."""
        print("=" * 70)
        print("Inflation-Linked Bond Summary")
        print("=" * 70)
        print(f"Issue Date:         {self.issue_date}")
        print(f"Maturity Date:      {self.maturity_date}")
        print(f"Value Date:         {self.value_date}")
        print(f"Notional:           {self.notional:,.2f}")
        print(f"Coupon Rate:        {self.coupon_rate:.4%}")
        print(f"Base CPI:           {self.base_cpi:.6f}")
        print(f"Payment Frequency:  {self.payment_frequency} months")
        print("-" * 70)
        print(f"Index Ratio:        {self.index_ratio():.6f}")
        print(f"Dirty Price:        {self.dirty_price():.6f}")
        print(f"Accrued Interest:   {self.accrued_interest():.6f}")
        print(f"Clean Price:        {self.clean_price():.6f}")
        print(f"Present Value:      {self.pv():,.2f}")
        print("=" * 70)
