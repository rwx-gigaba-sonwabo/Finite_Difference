"""
run_commodity_digital_barrier_validation.py
==============================================
CSV-driven or JSON-driven base (t0) valuation of a commodity digital
barrier option (One-Touch / No-Touch, single or double barrier,
continuous or discrete monitoring), built for running quick validation
checks against an external pricing system (e.g. Front Arena).

Market data can come from either of two sources -- toggle with
``data_source``:

- ``"csv"`` (default): two-column (date, value) CSV files.
- ``"json"``: pulled directly from a RiskFlow-exported market data JSON
  via ``base_valuation.market_data_json``, by risk-factor name instead
  of a file path. See that module's docstring for what's verified and
  what to check before relying on a JSON-sourced run.

Curve roles
------------
Three distinct curve roles, following ``instruments/commodity_average_
forward.py``'s pattern:

- ``money_market_curve`` -- discounts the payoff. Used **directly** for
  a standard (non-composite) trade -- domestic/foreign/FX are not
  involved at all in that case. Always in the contract (domestic)
  currency.
- ``domestic_curve`` / ``foreign_curve`` -- used **only**, in a
  composite trade, to compute the FX forward via covered interest rate
  parity (see ``base_valuation.fx``). Each may optionally have a cross-
  currency basis spread added on top (``domestic_basis_curve`` /
  ``foreign_basis_curve``) via ``base_valuation.fx.combine_curve_with_
  basis`` -- the combined rate is the base curve's own rate,
  interpolated at the basis curve's tenors, plus the basis spread,
  built on the basis curve's tenor grid. Leave a basis field as ``None``
  to use the base curve unmodified. ``domestic_curve`` and
  ``money_market_curve`` are independent curves, even though usually the
  same currency -- e.g. ``domestic_curve`` might be a cross-currency
  basis-adjusted swap curve while ``money_market_curve`` is a plain
  OIS/deposit curve.

For a composite trade: set ``domestic_curve``, ``foreign_curve``, and
``fx_spot_rate`` (CSV) or ``domestic_curve_risk_key``,
``foreign_curve_risk_key``, and ``fx_spot_risk_key`` (JSON). Leave them
all as ``None`` for a standard single-currency trade -- only
``money_market_curve`` is used.

To run a validation check: edit the CONFIG block below directly, then
run this file. No command-line flags to remember.

Every run appends one row to
results/commodity_digital_barrier_validation_log.csv (created with a
header on first run). Set ``external_price`` to have the script compute
and report the diff against a price read off the system you're
validating.

Examples
--------
Quick sanity run with the bundled sample curves and default (continuous,
up-and-in, cash) trade terms: leave CONFIG as-is and run the file.

Discretely monitored down-and-out, own curves, diff against an external
system's price::

    CONFIG = Config(
        val_date=date(2025, 6, 2),
        forward_curve=Path("market_data_csv/brent_fwd_2025-06-02.csv"),
        money_market_curve=Path("market_data_csv/zar_swap_naca_2025-06-02.csv"),
        maturity=date(2026, 6, 2), lower_barrier=65.0, touch="out",
        monitoring="discrete", cost_of_carry=0.035,
        vol=0.28, external_price=12.40,
    )

With a skew adjustment (level-dependent local-vol Monte Carlo using a
(strike, vol) smile CSV -- see market_data_csv/vol_skew.csv). This
unconditionally routes through MC (a warning is printed if
``monitoring="continuous"`` was also requested), even for an otherwise
closed-form-eligible continuous single-barrier cash digital::

    CONFIG = Config(
        upper_barrier=95.0, vol_skew_curve=Path("market_data_csv/vol_skew.csv"),
        apply_skew_adjustment=True, external_price=45.00,
    )

Composite, CSV: USD Brent underlying, ZAR-settled barrier (barrier/payout
in ZAR), with a cross-currency basis spread on the domestic leg::

    CONFIG = Config(
        forward_curve=Path("market_data_csv/commodity_forward_curve.csv"),  # USD
        money_market_curve=Path("market_data_csv/zar_swap_naca.csv"),
        domestic_curve=Path("market_data_csv/zar_swap_naca.csv"),
        domestic_basis_curve=Path("market_data_csv/zar_usd_basis.csv"),
        foreign_curve=Path("market_data_csv/discount_curve_naca.csv"),  # USD
        upper_barrier=1800.0, payout=2000.0,
        fx_spot_rate=18.50, external_price=95.40,
    )

Same trade, sourced from a RiskFlow JSON instead::

    CONFIG = Config(
        data_source="json",
        json_path=Path(r"C:\\...\\309gold2_compo_asian.json"),
        forward_risk_key="ForwardPrice.GOLD",
        money_market_curve_risk_key="InterestRate.ZAR-SWAP",
        domestic_curve_risk_key="InterestRate.ZAR-SWAP",
        domestic_basis_curve_risk_key="InterestRate.ZAR-SWAP.ZAR-USD-BASIS",
        foreign_curve_risk_key="InterestRate.USD-SOFR",
        fx_spot_risk_key="FxRate.ZAR",
        upper_barrier=1800.0, payout=2000.0, external_price=95.40,
    )
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from base_valuation.market_data_io import (
    build_forward_curve_from_csv,
    build_yield_curve_from_csv,
    build_vol_skew_from_csv,
)
from base_valuation.market_data_json import (
    build_forward_curve_from_json,
    build_yield_curve_from_json,
    build_commodity_vol_skew_from_json,
    build_fx_spot_from_json,
)
from base_valuation.commodity_digital_barrier_option import (
    value_commodity_digital_barrier_option,
    bump_and_reprice_commodity_digital_barrier_option,
)
from base_valuation.fx import FXSpotRate, build_composite_forward_curve, combine_curve_with_basis
from base_valuation.yield_curve import INTERPOLATORS

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_FORWARD_CSV = REPO_ROOT / "market_data_csv" / "commodity_forward_curve.csv"
DEFAULT_DISCOUNT_CSV = REPO_ROOT / "market_data_csv" / "discount_curve_naca.csv"
DEFAULT_RESULTS_LOG = REPO_ROOT / "results" / "commodity_digital_barrier_validation_log.csv"


@dataclass
class Config:
    # --- Valuation date --------------------------------------------------
    val_date: date = date(2025, 1, 2)

    # --- Data source -------------------------------------------------------
    data_source: str = "csv"  # "csv" | "json"

    # --- Forward price curve (CSV, used when data_source == "csv") --------
    # -- underlying's own (foreign, if composite) currency -----------------
    forward_curve: Path = DEFAULT_FORWARD_CSV
    forward_date_col: str = "date"
    forward_value_col: str = "value"
    forward_interp: str = "linear"  # "linear" | "forward_price"

    # --- Money market curve (CSV, discounting -- always used) -------------
    money_market_curve: Path = DEFAULT_DISCOUNT_CSV
    money_market_date_col: str = "date"
    money_market_value_col: str = "value"
    money_market_rate_convention: str = "NACA"  # "NACA" | "NACC"
    money_market_compounding_freq: int = 1  # 1=NACA, 2=NACS, 4=NACQ, 12=NACM
    money_market_interp: str = "hermite_rt"
    # "linear" | "linear_rt" | "reciprocal_time" | "hermite_rt" | "stitched_linear_hermite_rt"
    money_market_settle_days: int = 2
    money_market_settlement_calendar: str = "USD"

    # --- Domestic curve (CSV composite, FX-forward only) -------------------
    domestic_curve: Path | None = None
    domestic_date_col: str = "date"
    domestic_value_col: str = "value"
    domestic_rate_convention: str = "NACA"
    domestic_compounding_freq: int = 1
    domestic_interp: str = "hermite_rt"
    domestic_settle_days: int = 2
    domestic_settlement_calendar: str = "USD"

    # Optional cross-currency basis spread added onto domestic_curve
    domestic_basis_curve: Path | None = None
    domestic_basis_date_col: str = "date"
    domestic_basis_value_col: str = "value"
    domestic_basis_rate_convention: str = "NACA"
    domestic_basis_compounding_freq: int = 1
    domestic_basis_settle_days: int = 2
    domestic_basis_settlement_calendar: str = "USD"

    # --- Foreign curve (CSV composite, FX-forward only) --------------------
    foreign_curve: Path | None = None
    foreign_date_col: str = "date"
    foreign_value_col: str = "value"
    foreign_rate_convention: str = "NACA"
    foreign_compounding_freq: int = 1
    foreign_interp: str = "hermite_rt"
    foreign_settle_days: int = 2
    foreign_settlement_calendar: str = "USD"

    # Optional cross-currency basis spread added onto foreign_curve
    foreign_basis_curve: Path | None = None
    foreign_basis_date_col: str = "date"
    foreign_basis_value_col: str = "value"
    foreign_basis_rate_convention: str = "NACA"
    foreign_basis_compounding_freq: int = 1
    foreign_basis_settle_days: int = 2
    foreign_basis_settlement_calendar: str = "USD"

    fx_spot_rate: float | None = None  # domestic currency units per 1 unit of foreign currency
    fx_spot_days: int = 2
    fx_spot_calendar: str = "USD"
    fx_spot_invert: bool = False
    # Set True if the raw fx_spot_rate / fx_spot_risk_key value is actually
    # quoted foreign-per-domestic (e.g. USD per ZAR, ~0.054) rather than the
    # domestic-per-foreign convention FXSpotRate needs (ZAR per USD, ~18.5)
    # -- the reciprocal is taken before use, in either data source.

    # --- JSON market data (used when data_source == "json") ----------------
    json_path: Path | None = None
    forward_risk_key: str | None = None  # e.g. "ForwardPrice.GOLD"

    money_market_curve_risk_key: str | None = None  # e.g. "InterestRate.ZAR-SWAP"
    money_market_rate_convention_json: str = "NACC"
    money_market_compounding_freq_json: int = 1
    money_market_interp_json: str = "hermite_rt"

    domestic_curve_risk_key: str | None = None
    domestic_basis_curve_risk_key: str | None = None
    domestic_rate_convention_json: str = "NACC"
    domestic_compounding_freq_json: int = 1
    domestic_interp_json: str = "hermite_rt"

    foreign_curve_risk_key: str | None = None
    foreign_basis_curve_risk_key: str | None = None
    foreign_rate_convention_json: str = "NACC"
    foreign_compounding_freq_json: int = 1
    foreign_interp_json: str = "hermite_rt"

    fx_spot_risk_key: str | None = None
    json_forward_interp: str = "forward_price"

    vol_risk_key: str | None = None  # alternative to vol_skew_curve, e.g. "CommodityPriceVol.GOLD"
    vol_surface_type: str = "malz"  # "malz" | "non_precious" -- see market_data_json docstring

    # --- Trade terms -------------------------------------------------------
    maturity: date = date(2026, 1, 2)
    monitoring_start: date | None = None  # None -> defaults to val_date
    upper_barrier: float | None = 95.0
    lower_barrier: float | None = None
    touch: str = "in"  # "in" | "out"
    digital_type: str = "cash"  # "cash" | "asset"
    payout: float = 200.0
    vol: float = 0.30  # flat scalar vol; ignored if vol_skew_curve/vol_risk_key is set
    monitoring: str = "continuous"  # "continuous" | "discrete"
    monitoring_calendar: str = "USD"
    cost_of_carry: float | None = None  # None -> derive from forward curve's own term structure
    spot_days: int = 0
    spot_calendar: str = "USD"
    day_count: str = "ACT/365"
    n_mc_paths: int = 20_000
    mc_seed: int = 0

    # --- Volatility skew (CSV, optional, used when data_source == "csv") ---
    vol_skew_curve: Path | None = None  # (strike, vol) CSV; used instead of `vol` when set
    vol_skew_strike_col: str = "strike"
    vol_skew_vol_col: str = "vol"
    apply_skew_adjustment: bool = False
    # Skew-consistent local-vol Monte Carlo instead of a plain vol-at-barrier
    # lookup. Forces MC even for an otherwise closed-form-eligible continuous
    # single-barrier cash digital (a warning is printed). Requires
    # vol_skew_curve or vol_risk_key.

    # --- Validation / output -------------------------------------------------
    trade_id: str = ""  # free-text label written to the results log
    external_price: float | None = None  # price from the system being validated, for a diff report
    results_log: Path = DEFAULT_RESULTS_LOG
    no_log: bool = False  # skip appending to the results log
    skip_bump: bool = False  # skip the bump-and-reprice cross-check (saves time on large MC runs)

    def __post_init__(self) -> None:
        if self.upper_barrier is None and self.lower_barrier is None:
            raise ValueError("at least one of upper_barrier / lower_barrier is required")
        if self.data_source not in ("csv", "json"):
            raise ValueError(f"data_source must be 'csv' or 'json', got {self.data_source!r}")
        if self.apply_skew_adjustment and self.vol_skew_curve is None and self.vol_risk_key is None:
            raise ValueError(
                "apply_skew_adjustment=True requires vol_skew_curve (csv) or "
                "vol_risk_key (json) to be set"
            )
        if self.monitoring_start is None:
            self.monitoring_start = self.val_date
        if self.data_source == "csv":
            composite_fields = (self.domestic_curve, self.foreign_curve, self.fx_spot_rate)
            composite = any(f is not None for f in composite_fields)
            if composite and not all(f is not None for f in composite_fields):
                raise ValueError(
                    "a composite trade requires domestic_curve, foreign_curve, and "
                    "fx_spot_rate all to be set (leave all three as None for a "
                    "single-currency trade)"
                )
        else:  # json
            if self.json_path is None or self.forward_risk_key is None or self.money_market_curve_risk_key is None:
                raise ValueError(
                    "data_source='json' requires json_path, forward_risk_key, and "
                    "money_market_curve_risk_key to be set"
                )
            composite_fields = (self.domestic_curve_risk_key, self.foreign_curve_risk_key, self.fx_spot_risk_key)
            composite = any(f is not None for f in composite_fields)
            if composite and not all(f is not None for f in composite_fields):
                raise ValueError(
                    "a composite trade in json mode requires domestic_curve_risk_key, "
                    "foreign_curve_risk_key, and fx_spot_risk_key all to be set"
                )


# ===========================================================================
# EDIT THIS BLOCK TO CONFIGURE A RUN
# ===========================================================================
CONFIG = Config()


def append_results_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _build_leg_csv(cfg: Config, path_attr: str, prefix: str, interp: str) -> "YieldCurve":  # noqa: F821
    """Build one CSV-sourced curve by reading ``path_attr`` (the CSV path
    field) and the ``f"{prefix}_*"`` date_col/value_col/rate_convention/
    compounding_freq fields."""
    path = getattr(cfg, path_attr)
    return build_yield_curve_from_csv(
        path, cfg.val_date,
        date_col=getattr(cfg, f"{prefix}_date_col"),
        rate_col=getattr(cfg, f"{prefix}_value_col"),
        rate_convention=getattr(cfg, f"{prefix}_rate_convention"),
        compounding_freq=getattr(cfg, f"{prefix}_compounding_freq"),
        day_count=cfg.day_count, interpolation=interp,
    )


def main(cfg: Config = CONFIG) -> None:
    # --- Build curves: CSV or JSON, per cfg.data_source ---------------
    if cfg.data_source == "csv":
        fwd_curve = build_forward_curve_from_csv(
            cfg.forward_curve, cfg.val_date,
            date_col=cfg.forward_date_col, price_col=cfg.forward_value_col,
            day_count=cfg.day_count, interpolation=cfg.forward_interp,
        )
        money_market_curve = _build_leg_csv(cfg, "money_market_curve", "money_market", cfg.money_market_interp)

        is_composite = cfg.domestic_curve is not None
        domestic_curve_obj = foreign_curve_obj = fx_spot = None
        if is_composite:
            domestic_curve_obj = _build_leg_csv(cfg, "domestic_curve", "domestic", cfg.domestic_interp)
            if cfg.domestic_basis_curve is not None:
                domestic_basis = _build_leg_csv(cfg, "domestic_basis_curve", "domestic_basis", "linear")
                domestic_curve_obj = combine_curve_with_basis(
                    domestic_curve_obj, domestic_basis, INTERPOLATORS[cfg.domestic_interp],
                )

            foreign_curve_obj = _build_leg_csv(cfg, "foreign_curve", "foreign", cfg.foreign_interp)
            if cfg.foreign_basis_curve is not None:
                foreign_basis = _build_leg_csv(cfg, "foreign_basis_curve", "foreign_basis", "linear")
                foreign_curve_obj = combine_curve_with_basis(
                    foreign_curve_obj, foreign_basis, INTERPOLATORS[cfg.foreign_interp],
                )

            fx_rate = cfg.fx_spot_rate if not cfg.fx_spot_invert else 1.0 / cfg.fx_spot_rate
            fx_spot = FXSpotRate(
                rate=fx_rate, spot_days=cfg.fx_spot_days, spot_calendar=cfg.fx_spot_calendar,
            )
    else:  # json
        fwd_curve = build_forward_curve_from_json(
            cfg.json_path, cfg.forward_risk_key, cfg.val_date,
            day_count=cfg.day_count, interpolation=cfg.json_forward_interp,
        )
        money_market_curve = build_yield_curve_from_json(
            cfg.json_path, cfg.money_market_curve_risk_key, cfg.val_date,
            rate_convention=cfg.money_market_rate_convention_json,
            compounding_freq=cfg.money_market_compounding_freq_json,
            interpolation=cfg.money_market_interp_json,
        )

        is_composite = cfg.domestic_curve_risk_key is not None
        domestic_curve_obj = foreign_curve_obj = fx_spot = None
        if is_composite:
            domestic_curve_obj = build_yield_curve_from_json(
                cfg.json_path, cfg.domestic_curve_risk_key, cfg.val_date,
                rate_convention=cfg.domestic_rate_convention_json,
                compounding_freq=cfg.domestic_compounding_freq_json,
                interpolation=cfg.domestic_interp_json,
            )
            if cfg.domestic_basis_curve_risk_key is not None:
                domestic_basis = build_yield_curve_from_json(
                    cfg.json_path, cfg.domestic_basis_curve_risk_key, cfg.val_date,
                    rate_convention=cfg.domestic_rate_convention_json,
                    compounding_freq=cfg.domestic_compounding_freq_json,
                    interpolation="linear",
                )
                domestic_curve_obj = combine_curve_with_basis(
                    domestic_curve_obj, domestic_basis, INTERPOLATORS[cfg.domestic_interp_json],
                )

            foreign_curve_obj = build_yield_curve_from_json(
                cfg.json_path, cfg.foreign_curve_risk_key, cfg.val_date,
                rate_convention=cfg.foreign_rate_convention_json,
                compounding_freq=cfg.foreign_compounding_freq_json,
                interpolation=cfg.foreign_interp_json,
            )
            if cfg.foreign_basis_curve_risk_key is not None:
                foreign_basis = build_yield_curve_from_json(
                    cfg.json_path, cfg.foreign_basis_curve_risk_key, cfg.val_date,
                    rate_convention=cfg.foreign_rate_convention_json,
                    compounding_freq=cfg.foreign_compounding_freq_json,
                    interpolation="linear",
                )
                foreign_curve_obj = combine_curve_with_basis(
                    foreign_curve_obj, foreign_basis, INTERPOLATORS[cfg.foreign_interp_json],
                )

            fx_rate_json = build_fx_spot_from_json(cfg.json_path, cfg.fx_spot_risk_key)
            if cfg.fx_spot_invert:
                fx_rate_json = 1.0 / fx_rate_json
            fx_spot = FXSpotRate(
                rate=fx_rate_json,
                spot_days=cfg.fx_spot_days, spot_calendar=cfg.fx_spot_calendar,
            )

    priced_forward_curve = fwd_curve
    if is_composite:
        priced_forward_curve = build_composite_forward_curve(
            foreign_forward_curve=fwd_curve, fx_spot=fx_spot,
            domestic_curve=domestic_curve_obj, foreign_curve=foreign_curve_obj,
            domestic_settle_days=cfg.domestic_settle_days,
            domestic_settlement_calendar=cfg.domestic_settlement_calendar,
            foreign_settle_days=cfg.foreign_settle_days,
            foreign_settlement_calendar=cfg.foreign_settlement_calendar,
            day_count=cfg.day_count,
        )

    # --- Volatility skew: CSV or JSON ----------------------------------
    vol_skew = None
    if cfg.vol_skew_curve is not None:
        vol_skew = build_vol_skew_from_csv(
            cfg.vol_skew_curve,
            strike_col=cfg.vol_skew_strike_col, vol_col=cfg.vol_skew_vol_col,
        )
    elif cfg.vol_risk_key is not None:
        vol_skew = build_commodity_vol_skew_from_json(
            cfg.json_path, cfg.vol_risk_key, priced_forward_curve,
            surface_type=cfg.vol_surface_type,
        )

    common_kwargs = dict(
        val_date=cfg.val_date, maturity_date=cfg.maturity,
        upper_barrier=cfg.upper_barrier, lower_barrier=cfg.lower_barrier,
        touch=cfg.touch, digital_type=cfg.digital_type, payout=cfg.payout,
        forward_curve=priced_forward_curve, discount_curve=money_market_curve,
        vol=cfg.vol, vol_skew=vol_skew, apply_skew_adjustment=cfg.apply_skew_adjustment,
        monitoring=cfg.monitoring, monitoring_calendar=cfg.monitoring_calendar,
        cost_of_carry=cfg.cost_of_carry,
        spot_days=cfg.spot_days, spot_calendar=cfg.spot_calendar,
        settle_days=cfg.money_market_settle_days, settlement_calendar=cfg.money_market_settlement_calendar,
        day_count=cfg.day_count, n_mc_paths=cfg.n_mc_paths, mc_seed=cfg.mc_seed,
    )

    result = value_commodity_digital_barrier_option(**common_kwargs)
    bump = None
    if not cfg.skip_bump:
        bump = bump_and_reprice_commodity_digital_barrier_option(**common_kwargs)

    barrier_desc = []
    if cfg.upper_barrier is not None:
        barrier_desc.append(f"upper={cfg.upper_barrier}")
    if cfg.lower_barrier is not None:
        barrier_desc.append(f"lower={cfg.lower_barrier}")

    print("=" * 72)
    print(f"Commodity Digital Barrier ({cfg.touch.title()}-Touch, "
          f"{', '.join(barrier_desc)}, {cfg.digital_type}, {cfg.monitoring})  "
          f"@ {cfg.val_date}  [source={cfg.data_source}]")
    print("=" * 72)
    if cfg.data_source == "csv":
        print(f"  Forward curve:      {cfg.forward_curve}  (interp={cfg.forward_interp})"
              f"{'  [foreign ccy]' if is_composite else ''}")
        print(f"  Money market curve: {cfg.money_market_curve}  "
              f"(convention={cfg.money_market_rate_convention}, interp={cfg.money_market_interp})")
        if is_composite:
            print(f"  Domestic curve:     {cfg.domestic_curve}  (interp={cfg.domestic_interp})"
                  f"{'  + basis ' + str(cfg.domestic_basis_curve) if cfg.domestic_basis_curve else ''}")
            print(f"  Foreign curve:      {cfg.foreign_curve}  (interp={cfg.foreign_interp})"
                  f"{'  + basis ' + str(cfg.foreign_basis_curve) if cfg.foreign_basis_curve else ''}")
            print(f"  FX spot:            {fx_spot.rate}  "
                  f"(spot_days={fx_spot.spot_days}, calendar={fx_spot.spot_calendar})")
    else:
        print(f"  JSON:               {cfg.json_path}")
        print(f"  Forward risk key:   {cfg.forward_risk_key}  (interp={cfg.json_forward_interp})"
              f"{'  [foreign ccy]' if is_composite else ''}")
        print(f"  Money market key:   {cfg.money_market_curve_risk_key}")
        if is_composite:
            print(f"  Domestic key:       {cfg.domestic_curve_risk_key}"
                  f"{'  + basis ' + cfg.domestic_basis_curve_risk_key if cfg.domestic_basis_curve_risk_key else ''}")
            print(f"  Foreign key:        {cfg.foreign_curve_risk_key}"
                  f"{'  + basis ' + cfg.foreign_basis_curve_risk_key if cfg.foreign_basis_curve_risk_key else ''}")
            print(f"  FX spot key:        {cfg.fx_spot_risk_key} = {fx_spot.rate}")
    if vol_skew is not None:
        skew_src = cfg.vol_skew_curve if cfg.vol_skew_curve is not None else cfg.vol_risk_key
        print(f"  Vol skew source: {skew_src}  (apply_skew_adjustment={cfg.apply_skew_adjustment})")
    print(f"  Maturity / Payout: {cfg.maturity} / {cfg.payout}")
    print(f"  Pricing method:              {result.pricing_method}")
    print(f"  Vol (at barrier):            {result.vol:.4%}")
    print(f"  Forward / Implied spot / b:  {result.forward:.6f} / "
          f"{result.implied_spot:.6f} / {result.cost_of_carry:.6%}")
    print(f"  T_opt / T_disc / DF:         {result.T_opt:.6f} / {result.T_disc:.6f} / {result.df:.8f}")
    if result.mc_stderr is not None:
        print(f"  MC standard error:           {result.mc_stderr:.6f}")
    print("-" * 72)
    print(f"  Price:              {result.price:>14.6f}")
    if result.analytic_delta is not None:
        print(f"  Analytic  Delta/Vega/Theta: "
              f"{result.analytic_delta:.6f} / {result.analytic_vega:.6f} / {result.analytic_theta:.6f}")
    else:
        print("  Analytic  Greeks:            n/a for this configuration (see docstring)")
    if bump is not None:
        print(f"  Bump      Delta/Gamma/Vega/Theta: "
              f"{bump['delta']:.6f} / {bump['gamma']:.8f} / "
              f"{bump['vega']:.6f} / {bump['theta']:.6f}")

    diff = diff_pct = None
    if cfg.external_price is not None:
        diff = result.price - cfg.external_price
        diff_pct = diff / cfg.external_price if cfg.external_price != 0.0 else float("nan")
        print("-" * 72)
        print(f"  External price:     {cfg.external_price:>14.6f}")
        print(f"  Diff (this - ext):  {diff:>14.6f}   ({diff_pct:+.4%})")
    print("=" * 72)

    if not cfg.no_log:
        row = {
            "trade_id": cfg.trade_id,
            "val_date": cfg.val_date.isoformat(),
            "maturity": cfg.maturity.isoformat(),
            "data_source": cfg.data_source,
            "upper_barrier": cfg.upper_barrier,
            "lower_barrier": cfg.lower_barrier,
            "touch": cfg.touch,
            "digital_type": cfg.digital_type,
            "monitoring": cfg.monitoring,
            "payout": cfg.payout,
            "vol_input": cfg.vol,
            "vol_at_barrier": result.vol,
            "vol_skew_source": (cfg.vol_skew_curve if cfg.vol_skew_curve is not None
                                 else cfg.vol_risk_key) if vol_skew is not None else "",
            "skew_adjusted": result.skew_adjusted,
            "forward_source": str(cfg.forward_curve) if cfg.data_source == "csv" else cfg.forward_risk_key,
            "money_market_source": (str(cfg.money_market_curve) if cfg.data_source == "csv"
                                     else cfg.money_market_curve_risk_key),
            "is_composite": is_composite,
            "domestic_source": (
                (str(cfg.domestic_curve) if cfg.data_source == "csv" else cfg.domestic_curve_risk_key)
                if is_composite else ""
            ),
            "domestic_basis_source": (
                (str(cfg.domestic_basis_curve) if cfg.data_source == "csv"
                 else cfg.domestic_basis_curve_risk_key) or ""
                if is_composite else ""
            ),
            "foreign_source": (
                (str(cfg.foreign_curve) if cfg.data_source == "csv" else cfg.foreign_curve_risk_key)
                if is_composite else ""
            ),
            "foreign_basis_source": (
                (str(cfg.foreign_basis_curve) if cfg.data_source == "csv"
                 else cfg.foreign_basis_curve_risk_key) or ""
                if is_composite else ""
            ),
            "fx_spot_rate": fx_spot.rate if is_composite else None,
            "pricing_method": result.pricing_method,
            "forward": result.forward,
            "implied_spot": result.implied_spot,
            "cost_of_carry": result.cost_of_carry,
            "T_opt": result.T_opt,
            "T_disc": result.T_disc,
            "df": result.df,
            "mc_stderr": result.mc_stderr,
            "price": result.price,
            "analytic_delta": result.analytic_delta,
            "analytic_vega": result.analytic_vega,
            "analytic_theta": result.analytic_theta,
            "bump_delta": bump["delta"] if bump else None,
            "bump_gamma": bump["gamma"] if bump else None,
            "bump_vega": bump["vega"] if bump else None,
            "bump_theta": bump["theta"] if bump else None,
            "external_price": cfg.external_price,
            "diff": diff,
            "diff_pct": diff_pct,
        }
        append_results_log(cfg.results_log, row)
        print(f"Logged to {cfg.results_log}")


if __name__ == "__main__":
    main()
