import pandas as pd
from datetime import date
from typing import Literal
from instruments.bond.inflation_bond_pricer import InflationLinkedBondPricer
from curves.discount_curve import DiscountYieldCurveHandler
from curves.historical_cpi import HistoricalCPI


class ForwardInflationBondPricer:
    """
    Forward pricer for inflation-linked bonds.

    Calculates forward prices, NPV, and risk metrics for inflation-linked
    bond forward contracts, accounting for:
    - Carry (coupon payments between spot and forward date)
    - Index ratio adjustments at forward date
    - CPI-indexed cashflows
    - Discounting to present value

    Forward pricing formula:
        Forward Dirty Price = (Spot Dirty Price - Carry/Notional×100) / DF(spot→forward)
        Forward Clean Price = Forward Dirty Price - Accrued(forward)

    Where Carry = PV of all coupons paid between spot settlement and forward date.
    """

    def __init__(
        self,
        underlying_bond: InflationLinkedBondPricer,
        forward_date: date,
        settlement_date: date = None,
        strike_price: float = None,
        position: Literal["long", "short"] = "long",
        contract_notional: float = None,
    ):
        """
        Parameters:
            underlying_bond: The underlying inflation-linked bond pricer.
            forward_date: The forward contract date.
            settlement_date: Settlement date for the forward contract (defaults to forward_date).
            strike_price: Strike price for the forward (clean price per 100). If None, uses forward price.
            position: "long" or "short" position.
            contract_notional: Notional of the forward contract (defaults to bond notional).
        """
        self.underlying_bond = underlying_bond
        self.forward_date = forward_date
        self.settlement_date = settlement_date if settlement_date else forward_date
        self.position = position
        self.contract_notional = (
            contract_notional
            if contract_notional is not None
            else underlying_bond.notional
        )

        # If no strike provided, use theoretical forward price
        if strike_price is None:
            self.strike_price = self.forward_clean_price()
        else:
            self.strike_price = strike_price

        # Validation
        if self.forward_date <= underlying_bond.value_date:
            raise ValueError("Forward date must be after value date")

    def _calculate_carry(self) -> float:
        """
        Calculate the present value of all coupons paid between the spot value date
        and the forward date (not including the forward date itself).

        Returns:
            Present value of carry (in currency units)
        """
        schedule = self.underlying_bond.schedule
        spot_date = self.underlying_bond.value_date
        forward_date = self.forward_date

        # Convert to datetime for comparison
        spot_date_dt = pd.to_datetime(spot_date)
        forward_date_dt = pd.to_datetime(forward_date)
        pay_dates = pd.to_datetime(schedule["PayDate"], errors="coerce")

        # Find coupons paid between spot and forward (exclusive of forward date)
        mask = (pay_dates > spot_date_dt) & (pay_dates <= forward_date_dt)

        if mask.any():
            carry_cashflows = schedule.loc[mask, "Coupon"]
            discount_factors = schedule.loc[mask, "DiscountFactor"]
            carry_pv = (carry_cashflows * discount_factors).sum()
            return carry_pv

        return 0.0

    def _accrued_at_forward(self) -> float:
        """
        Calculate accrued interest at the forward date.
        Returns accrued interest per 100 of face value.
        """
        schedule = self.underlying_bond.schedule
        forward_date_dt = pd.to_datetime(self.forward_date)
        start_dates = pd.to_datetime(schedule["StartDate"], errors="coerce")

        # Find the period containing the forward date
        mask = start_dates <= forward_date_dt

        if mask.any():
            relevant_period = schedule[mask].iloc[-1]

            start_date = relevant_period["StartDate"]
            end_date = relevant_period["EndDate"]
            indexed_notional = relevant_period["Notional"]
            year_frac = relevant_period["YearFrac"]

            # Calculate accrual fraction
            days_accrued = (self.forward_date - start_date).days
            total_days = (end_date - start_date).days

            if total_days > 0:
                accrual_fraction = days_accrued / total_days
                accrued = (
                    indexed_notional
                    * self.underlying_bond.coupon_rate
                    * accrual_fraction
                    * year_frac
                )
                # Return per 100 of face value
                return (accrued / self.underlying_bond.notional) * 100.0

        return 0.0

    def forward_dirty_price(self) -> float:
        """
        Calculate the forward dirty price of the inflation-linked bond.

        Formula:
            Forward Dirty = (Spot Dirty - Carry/Notional×100) / DF(spot→forward)

        Returns:
            Forward dirty price per 100 face value
        """
        spot_dirty = self.underlying_bond.dirty_price()
        carry = self._calculate_carry()

        # Get discount factor from value date to forward date
        df_forward = self.underlying_bond.discount_curve.get_discount_factor_for_date(
            self.forward_date
        )

        # Adjust spot dirty for carry and discount to forward date
        # Carry is in currency units, so convert to per-100 basis
        carry_per_100 = (carry / self.underlying_bond.notional) * 100.0

        forward_dirty = (spot_dirty - carry_per_100) / df_forward

        return forward_dirty

    def forward_clean_price(self) -> float:
        """
        Calculate the forward clean price.

        Formula:
            Forward Clean = Forward Dirty - Forward Accrued

        Returns:
            Forward clean price per 100 face value
        """
        forward_dirty = self.forward_dirty_price()
        forward_accrued = self._accrued_at_forward()

        return forward_dirty - forward_accrued

    def npv(self) -> float:
        """
        Calculate the net present value of the forward contract.

        Formula:
            NPV = sign × (Forward Clean - Strike) × Notional × DF(settlement) / 100

        Where sign = +1 for long, -1 for short

        Returns:
            NPV of the forward contract in currency units
        """
        forward_clean = self.forward_clean_price()
        price_diff = forward_clean - self.strike_price

        # Discount to settlement date (typically same as forward date)
        df_settlement = self.underlying_bond.discount_curve.get_discount_factor_for_date(
            self.settlement_date
        )

        # Position sign
        sign = 1.0 if self.position == "long" else -1.0

        # NPV = sign × price_difference × notional × discount_factor / 100
        npv = sign * price_diff * self.contract_notional * df_settlement / 100.0

        return npv

    def forward_index_ratio(self) -> float:
        """
        Calculate the index ratio at the forward date.

        Returns:
            Index ratio at forward date (floored at 1.0)
        """
        cpi_forward = self.underlying_bond.historical_cpi.cpi_value(self.forward_date)
        return max(cpi_forward / self.underlying_bond.base_cpi, 1.0)

    def spot_index_ratio(self) -> float:
        """
        Calculate the index ratio at the spot value date.

        Returns:
            Index ratio at spot date
        """
        return self.underlying_bond.index_ratio()

    def summary(self) -> dict:
        """Returns summary with forward prices and contract details."""
        return {
            "forward_dirty_price": self.forward_dirty_price(),
            "forward_clean_price": self.forward_clean_price(),
            "forward_accrued": self._accrued_at_forward(),
            "strike_price": self.strike_price,
            "npv": self.npv(),
            "carry": self._calculate_carry(),
            "spot_index_ratio": self.spot_index_ratio(),
            "forward_index_ratio": self.forward_index_ratio(),
            "forward_date": self.forward_date,
            "settlement_date": self.settlement_date,
            "position": self.position,
            "contract_notional": self.contract_notional,
        }

    def print_summary(self):
        """Print a formatted summary of the forward contract valuation."""
        spot_dirty = self.underlying_bond.dirty_price()
        spot_clean = self.underlying_bond.clean_price()
        spot_accrued = self.underlying_bond.accrued_interest()

        forward_dirty = self.forward_dirty_price()
        forward_clean = self.forward_clean_price()
        forward_accrued = self._accrued_at_forward()

        carry = self._calculate_carry()
        npv = self.npv()

        spot_ir = self.spot_index_ratio()
        forward_ir = self.forward_index_ratio()

        print("=" * 70)
        print("Forward Inflation-Linked Bond Contract Summary")
        print("=" * 70)
        print("\nUnderlying Bond:")
        print(f"  Issue Date:        {self.underlying_bond.issue_date}")
        print(f"  Maturity Date:     {self.underlying_bond.maturity_date}")
        print(f"  Value Date:        {self.underlying_bond.value_date}")
        print(f"  Coupon Rate:       {self.underlying_bond.coupon_rate:.4%}")
        print(f"  Notional:          {self.underlying_bond.notional:,.2f}")
        print(f"  Base CPI:          {self.underlying_bond.base_cpi:.6f}")

        print("\nForward Contract:")
        print(f"  Position:          {self.position.upper()}")
        print(f"  Forward Date:      {self.forward_date}")
        print(f"  Settlement Date:   {self.settlement_date}")
        print(f"  Strike Price:      {self.strike_price:,.6f}")
        print(f"  Contract Notional: {self.contract_notional:,.2f}")

        print("\n" + "-" * 70)
        print("Spot Prices (Current):")
        print(f"  Spot Index Ratio:  {spot_ir:.6f}")
        print(f"  Spot Dirty Price:  {spot_dirty:,.6f}")
        print(f"  Spot Accrued:      {spot_accrued:,.6f}")
        print(f"  Spot Clean Price:  {spot_clean:,.6f}")

        print("\nForward Prices:")
        print(f"  Forward Index Ratio:  {forward_ir:.6f}")
        print(f"  Carry (PV):           {carry:,.2f}")
        print(f"  Forward Dirty Price:  {forward_dirty:,.6f}")
        print(f"  Forward Accrued:      {forward_accrued:,.6f}")
        print(f"  Forward Clean Price:  {forward_clean:,.6f}")

        print("\nContract Valuation:")
        print(f"  Strike Price:         {self.strike_price:,.6f}")
        print(f"  Forward Clean Price:  {forward_clean:,.6f}")
        print(f"  Price Difference:     {forward_clean - self.strike_price:,.6f}")
        print(f"  NPV:                  {npv:,.2f}")
        print("=" * 70)

    def get_forward_schedule(self) -> pd.DataFrame:
        """
        Returns the cashflow schedule from the forward date onwards.

        Returns:
            DataFrame with cashflows occurring after the forward date
        """
        schedule = self.underlying_bond.schedule.copy()
        forward_date_dt = pd.to_datetime(self.forward_date)
        pay_dates = pd.to_datetime(schedule["PayDate"], errors="coerce")

        # Filter for cashflows after forward date
        mask = pay_dates > forward_date_dt
        return schedule[mask].reset_index(drop=True)
