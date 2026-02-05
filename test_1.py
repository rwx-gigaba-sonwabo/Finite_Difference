import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from curves.discount_curve import DiscountYieldCurveHandler
from curves.discount_curve_flat import FlatDiscountCurve
from curves.discount_curve_no_interp import YieldCurveHandlerNoInterp
from curves.historical_cpi import HistoricalCPI
from data_handling.csv_handler import csv_to_dataframe as load_csv
from datetime import date
from utils.interpolation import hermite_rt_interp
from instruments.ir_swap.pricer import IRSwap


val_date = date(2025, 7, 28)
inflation_curve_anchor = date(2026,5,1)

inflation_data = load_csv(r'C:\Coding\derivatives-pricing\new\historical_cpi.csv')

rate_1_data = load_csv(r'C:\Coding\derivatives-pricing\new\zar-swap.csv')
rate_2_data = load_csv(r'C:\Coding\derivatives-pricing\new\jibar_3m.csv')
rate_3_data = load_csv(r'C:\Coding\derivatives-pricing\new\zar-cpi.csv')

receive_curve = DiscountYieldCurveHandler(val_date, rate_2_data, hermite_rt_interp)
inflation_curve = YieldCurveHandlerNoInterp(val_date,rate_3_data)
#pay_curve = DiscountYieldCurveHandler(val_date, rate_2_data, hermite_rt_interp)
discount_curve = DiscountYieldCurveHandler(val_date, rate_1_data, hermite_rt_interp)
fixed_curve = FlatDiscountCurve(val_date, simple_annual_rate=0.08)

historical_cpi = HistoricalCPI(
    value_date=val_date,
    curve_anchor_date=inflation_curve_anchor,
    monthly_cpi=inflation_data,
    curve=inflation_curve

)

base_cpi = 94.18465727543

# Create and price the IR swap using the new IRSwap class
swap = IRSwap(
    effective_date=date(2025, 7, 28),
    maturity_date=date(2030, 7, 28),
    notional=1_000_000,
    value_date=val_date,
    receive_curve=receive_curve,
    pay_curve=fixed_curve,
    discount_curve=discount_curve,
    historical_cpi=historical_cpi,
    base_cpi=base_cpi,
    pay_inflation_leg=1,
    receive_spread=0.02,
    receive_payment_frequency=3,
    pay_fixed_rate=True,
    pay_payment_frequency=3,
    calendar="SouthAfrica",
    business_convention="ModifiedFollowing",
    forward_business_convention="Following",
    termination_business_convention="Following",
    use_schedule_end_date=False,
)

#print(swap.pay_leg_pv())
FA_Pay_PV = 334439.05
rel_diff_pay = (swap.pay_leg_pv() - FA_Pay_PV)/FA_Pay_PV
print(f"Relative difference in Pay Leg PV: {rel_diff_pay:.6%}")

print(swap.receive_leg_pv())

FA_Receive_PV = -27800.25 + 334439.05
rel_diff_rec = (swap.receive_leg_pv() - FA_Receive_PV)/FA_Receive_PV
print(f"Relative difference in Pay Leg PV: {rel_diff_rec:.6%}")

Total_PV = swap.net_pv()
print(f"Total PV: {Total_PV:,.2f}")
FA_PV = -27800.25
print(f"FA Total PV: {FA_PV:,.2f}")
rel_diff_total = (Total_PV - FA_PV)/FA_PV
print(f"Relative difference in Total PV: {rel_diff_total:.6%}")

print("\n" + "="*70 + "\n")

# ============================================================================
# Inflation-Linked Bond Pricing Example
# ============================================================================
from instruments.bond.inflation_bond_pricer import InflationLinkedBondPricer
from instruments.forward.forward_inflation_bond_pricer import ForwardInflationBondPricer

print("INFLATION-LINKED BOND PRICING")
print("="*70)

# Create an inflation-linked bond
ilb = InflationLinkedBondPricer(
    issue_date=date(2020, 1, 15),
    maturity_date=date(2030, 1, 15),
    notional=1_000_000,
    coupon_rate=0.0250,  # 2.50% real coupon rate
    value_date=val_date,
    discount_curve=discount_curve,
    historical_cpi=historical_cpi,
    base_cpi=base_cpi,  # Using the same base_cpi from the swap example
    payment_frequency=6,  # Semi-annual coupons
    calendar="SouthAfrica",
    business_convention="ModifiedFollowing",
    termination_business_convention="ModifiedFollowing",
    date_generation="Backward",
    day_count="ACT/365",
)

# Print bond summary
ilb.print_summary()

# Print detailed schedule
print("\nInflation-Linked Bond Cashflow Schedule:")
print(ilb.get_schedule()[["PayDate", "CPIValue", "Notional", "Coupon", "Principal", "Cashflow", "DiscountFactor", "PV"]])

print("\n" + "="*70 + "\n")

# ============================================================================
# Forward on Inflation-Linked Bond Pricing Example
# ============================================================================
print("FORWARD ON INFLATION-LINKED BOND PRICING")
print("="*70)

# Create a forward contract on the inflation-linked bond
forward_date = date(2027, 1, 15)  # Forward date 2 years from now

forward_ilb = ForwardInflationBondPricer(
    underlying_bond=ilb,
    forward_date=forward_date,
    settlement_date=forward_date,  # Settlement same as forward date
    strike_price=None,  # Will use theoretical forward price
    position="long",
    contract_notional=1_000_000,
)

# Print forward contract summary
forward_ilb.print_summary()

# Print forward schedule (cashflows after forward date)
print("\nCashflows After Forward Date:")
forward_schedule = forward_ilb.get_forward_schedule()
if not forward_schedule.empty:
    print(forward_schedule[["PayDate", "CPIValue", "Notional", "Coupon", "Principal", "Cashflow"]])
else:
    print("No cashflows after forward date")

print("\n" + "="*70)

# Example with a specific strike price
print("\nFORWARD WITH SPECIFIC STRIKE PRICE")
print("="*70)

# Create forward with strike = 105.0
forward_ilb_strike = ForwardInflationBondPricer(
    underlying_bond=ilb,
    forward_date=forward_date,
    strike_price=105.0,  # Specific strike price
    position="long",
    contract_notional=1_000_000,
)

forward_summary = forward_ilb_strike.summary()
print(f"Forward Clean Price:  {forward_summary['forward_clean_price']:,.6f}")
print(f"Strike Price:         {forward_summary['strike_price']:,.6f}")
print(f"NPV (Long):           {forward_summary['npv']:,.2f}")

# Short position example
forward_ilb_short = ForwardInflationBondPricer(
    underlying_bond=ilb,
    forward_date=forward_date,
    strike_price=105.0,
    position="short",
    contract_notional=1_000_000,
)
print(f"NPV (Short):          {forward_ilb_short.npv():,.2f}")

print("="*70)