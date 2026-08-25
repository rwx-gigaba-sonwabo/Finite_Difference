"""
run_commodity_digital_validation.py
======================================
Base (t0) valuation of a European commodity digital option, built for
running quick validation checks against an external pricing system
(e.g. Front Arena).

Market data can come from either of two sources -- toggle with
``data_source``:

- ``"csv"`` (default): two-column (date, value) CSV files -- see
  market_data_csv/commodity_forward_curve.csv and
  market_data_csv/discount_curve_naca.csv.
- ``"json"``: pulled directly from a RiskFlow-exported market data JSON
  via ``base_valuation.market_data_json``, by risk-factor name (e.g.
  ``"ForwardPrice.GOLD"``) instead of a file path. See that module's
  docstring for exactly what's verified and what to check before relying
  on a JSON-sourced run -- unlike the CSV path, it wasn't built against
  a live system.

Whichever source is used, everything downstream -- trade terms, the
composite/cross-currency machinery, pricing, printing, logging -- is
identical; only the market-data-building step at the top of ``main()``
branches on ``data_source``.

To run a validation check: edit the CONFIG block below directly, then
run this file. No command-line flags to remember.

Every run appends one row to results/commodity_digital_validation_log.csv
(created with a header on first run) so a sequence of validation checks
builds up into a comparable log. Set ``external_price`` to have the
script compute and report the diff against a price read off the system
you're validating.

Composite (cross-currency) trades
----------------------------------
When the underlying's forward price currency differs from the contract
(settlement) currency: in CSV mode, set ``foreign_discount_curve`` (a
second CSV) and ``fx_spot_rate``; in JSON mode, set
``foreign_curve_risk_key`` and ``fx_spot_risk_key`` instead.
``forward_curve`` / ``forward_risk_key`` continues to describe the
underlying in its own natural currency; the strike, payout, and price
are all still in contract (domestic) currency terms. The domestic-
currency forward curve actually priced off is built via covered
interest rate parity -- see ``base_valuation.fx`` for the mechanics and
important caveats (no FX volatility / correlation is modelled; the
``vol`` you supply, or the JSON vol surface via ``vol_risk_key``, is
taken to already be the composite volatility of the domestic-currency
price). Leave the foreign-curve fields as ``None`` (the default) for an
ordinary single-currency trade -- nothing else changes.

Examples
--------
Quick sanity run with the bundled sample curves and default trade terms:
leave CONFIG as-is and run the file.

Point at your own curves, override trade terms, and diff against an
external system's price::

    CONFIG = Config(
        val_date=date(2025, 6, 2),
        forward_curve=Path("market_data_csv/brent_fwd_2025-06-02.csv"),
        domestic_discount_curve=Path("market_data_csv/zar_swap_naca_2025-06-02.csv"),
        maturity=date(2026, 6, 2), strike=82.5, is_call=True,
        digital_type="cash", payout=100.0, vol=0.28,
        spot_days=2, domestic_settle_days=2, external_price=41.90,
    )

With a skew adjustment (tight call/put-spread replication using a
(strike, vol) smile CSV -- see market_data_csv/vol_skew.csv), instead of
a flat scalar ``vol``::

    CONFIG = Config(
        strike=82.5, vol_skew_curve=Path("market_data_csv/vol_skew.csv"),
        apply_skew_adjustment=True, external_price=41.90,
    )

Composite, CSV: USD Brent underlying, ZAR-settled digital (strike/payout
in ZAR), diffed against an external ZAR price::

    CONFIG = Config(
        forward_curve=Path("market_data_csv/commodity_forward_curve.csv"),  # USD
        domestic_discount_curve=Path("market_data_csv/zar_swap_naca.csv"),
        foreign_discount_curve=Path("market_data_csv/discount_curve_naca.csv"),  # USD
        strike=1500.0, payout=2000.0,
        fx_spot_rate=18.50, external_price=780.00,
    )

Same trade, sourced from a RiskFlow JSON instead::

    CONFIG = Config(
        data_source="json",
        json_path=Path(r"C:\\...\\309gold2_compo_asian.json"),
        forward_risk_key="ForwardPrice.GOLD",
        domestic_curve_risk_key="InterestRate.ZAR-SWAP",
        foreign_curve_risk_key="InterestRate.USD-SOFR",
        fx_spot_risk_key="FxRate.ZAR",
        strike=1500.0, payout=2000.0, external_price=780.00,
    )

With a JSON-sourced vol skew instead of a flat ``vol``::

    CONFIG = Config(
        data_source="json", json_path=Path(r"C:\\...\\file.json"),
        forward_risk_key="ForwardPrice.GOLD",
        domestic_curve_risk_key="InterestRate.ZAR-SWAP",
        vol_risk_key="CommodityPriceVol.GOLD", strike=1500.0,
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
from base_valuation.commodity_digital_option import (
    value_commodity_digital_option,
    bump_and_reprice_commodity_digital_option,
)
from base_valuation.fx import FXSpotRate, build_composite_forward_curve

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_FORWARD_CSV = REPO_ROOT / "market_data_csv" / "commodity_forward_curve.csv"
DEFAULT_DISCOUNT_CSV = REPO_ROOT / "market_data_csv" / "discount_curve_naca.csv"
DEFAULT_RESULTS_LOG = REPO_ROOT / "results" / "commodity_digital_validation_log.csv"


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

    # --- Domestic discount curve (CSV, used when data_source == "csv") ----
    domestic_discount_curve: Path = DEFAULT_DISCOUNT_CSV
    domestic_discount_date_col: str = "date"
    domestic_discount_value_col: str = "value"
    domestic_rate_convention: str = "NACA"  # "NACA" | "NACC"
    domestic_compounding_freq: int = 1  # 1=NACA, 2=NACS, 4=NACQ, 12=NACM
    domestic_discount_interp: str = "hermite_rt"
    # "linear" | "linear_rt" | "reciprocal_time" | "hermite_rt" | "stitched_linear_hermite_rt"
    domestic_settle_days: int = 2
    domestic_settlement_calendar: str = "USD"

    # --- Foreign discount curve (CSV composite, used when data_source == "csv") -
    foreign_discount_curve: Path | None = None
    foreign_discount_date_col: str = "date"
    foreign_discount_value_col: str = "value"
    foreign_rate_convention: str = "NACA"  # "NACA" | "NACC"
    foreign_compounding_freq: int = 1
    foreign_discount_interp: str = "hermite_rt"
    foreign_settle_days: int = 2
    foreign_settlement_calendar: str = "USD"

    fx_spot_rate: float | None = None  # domestic currency units per 1 unit of foreign currency
    fx_spot_days: int = 2
    fx_spot_calendar: str = "USD"

    # --- JSON market data (used when data_source == "json") ----------------
    # Pulls directly from a RiskFlow-exported JSON via
    # base_valuation.market_data_json instead of the CSV fields above --
    # see that module's docstring before relying on a run.
    json_path: Path | None = None
    forward_risk_key: str | None = None  # e.g. "ForwardPrice.GOLD"
    domestic_curve_risk_key: str | None = None  # e.g. "InterestRate.ZAR-SWAP"
    foreign_curve_risk_key: str | None = None  # composite only, e.g. "InterestRate.USD-SOFR"
    fx_spot_risk_key: str | None = None  # composite only, e.g. "FxRate.ZAR"
    vol_risk_key: str | None = None  # alternative to vol_skew_curve, e.g. "CommodityPriceVol.GOLD"
    vol_surface_type: str = "malz"  # "malz" | "non_precious" -- see market_data_json docstring
    json_rate_convention: str = "NACC"  # "NACA" | "NACC" -- JSON curves default NACC, unlike CSV's NACA
    json_compounding_freq: int = 1
    json_discount_interp: str = "hermite_rt"
    json_forward_interp: str = "forward_price"

    # --- Trade terms -------------------------------------------------------
    maturity: date = date(2026, 1, 2)
    strike: float = 80.0
    is_call: bool = True
    digital_type: str = "cash"  # "cash" | "asset"
    payout: float = 100.0
    vol: float = 0.30  # flat scalar vol; ignored if vol_skew_curve/vol_risk_key is set
    spot_days: int = 0
    spot_calendar: str = "USD"
    day_count: str = "ACT/365"

    # --- Volatility skew (CSV, optional, used when data_source == "csv") ---
    vol_skew_curve: Path | None = None  # (strike, vol) CSV; used instead of `vol` when set
    vol_skew_strike_col: str = "strike"
    vol_skew_vol_col: str = "vol"
    apply_skew_adjustment: bool = False
    # Tight call/put-spread replication using the full local slope of the
    # skew at the strike, instead of a plain vol-at-strike lookup.
    # Requires vol_skew_curve or vol_risk_key.
    skew_strike_bump_frac: float = 1e-3
    # Relative strike bump for the replicating spread (only used with apply_skew_adjustment).

    # --- Validation / output -------------------------------------------------
    trade_id: str = ""  # free-text label written to the results log
    external_price: float | None = None  # price from the system being validated, for a diff report
    results_log: Path = DEFAULT_RESULTS_LOG
    no_log: bool = False  # skip appending to the results log

    def __post_init__(self) -> None:
        if self.data_source not in ("csv", "json"):
            raise ValueError(f"data_source must be 'csv' or 'json', got {self.data_source!r}")
        if self.apply_skew_adjustment and self.vol_skew_curve is None and self.vol_risk_key is None:
            raise ValueError(
                "apply_skew_adjustment=True requires vol_skew_curve (csv) or "
                "vol_risk_key (json) to be set"
            )
        if self.data_source == "csv":
            composite = self.foreign_discount_curve is not None or self.fx_spot_rate is not None
            if composite and (self.foreign_discount_curve is None or self.fx_spot_rate is None):
                raise ValueError(
                    "a composite trade requires both foreign_discount_curve and "
                    "fx_spot_rate to be set (leave both as None for a single-currency trade)"
                )
        else:  # json
            if self.json_path is None or self.forward_risk_key is None or self.domestic_curve_risk_key is None:
                raise ValueError(
                    "data_source='json' requires json_path, forward_risk_key, and "
                    "domestic_curve_risk_key to be set"
                )
            composite = self.foreign_curve_risk_key is not None or self.fx_spot_risk_key is not None
            if composite and (self.foreign_curve_risk_key is None or self.fx_spot_risk_key is None):
                raise ValueError(
                    "a composite trade in json mode requires both foreign_curve_risk_key "
                    "and fx_spot_risk_key to be set"
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


def main(cfg: Config = CONFIG) -> None:
    # --- Build curves: CSV or JSON, per cfg.data_source ---------------
    if cfg.data_source == "csv":
        fwd_curve = build_forward_curve_from_csv(
            cfg.forward_curve, cfg.val_date,
            date_col=cfg.forward_date_col, price_col=cfg.forward_value_col,
            day_count=cfg.day_count, interpolation=cfg.forward_interp,
        )
        domestic_curve = build_yield_curve_from_csv(
            cfg.domestic_discount_curve, cfg.val_date,
            date_col=cfg.domestic_discount_date_col, rate_col=cfg.domestic_discount_value_col,
            rate_convention=cfg.domestic_rate_convention, compounding_freq=cfg.domestic_compounding_freq,
            day_count=cfg.day_count, interpolation=cfg.domestic_discount_interp,
        )
        is_composite = cfg.foreign_discount_curve is not None
        foreign_curve = fx_spot = None
        if is_composite:
            foreign_curve = build_yield_curve_from_csv(
                cfg.foreign_discount_curve, cfg.val_date,
                date_col=cfg.foreign_discount_date_col, rate_col=cfg.foreign_discount_value_col,
                rate_convention=cfg.foreign_rate_convention, compounding_freq=cfg.foreign_compounding_freq,
                day_count=cfg.day_count, interpolation=cfg.foreign_discount_interp,
            )
            fx_spot = FXSpotRate(
                rate=cfg.fx_spot_rate, spot_days=cfg.fx_spot_days, spot_calendar=cfg.fx_spot_calendar,
            )
    else:  # json
        fwd_curve = build_forward_curve_from_json(
            cfg.json_path, cfg.forward_risk_key, cfg.val_date,
            day_count=cfg.day_count, interpolation=cfg.json_forward_interp,
        )
        domestic_curve = build_yield_curve_from_json(
            cfg.json_path, cfg.domestic_curve_risk_key,
            rate_convention=cfg.json_rate_convention, compounding_freq=cfg.json_compounding_freq,
            interpolation=cfg.json_discount_interp,
        )
        is_composite = cfg.foreign_curve_risk_key is not None
        foreign_curve = fx_spot = None
        if is_composite:
            foreign_curve = build_yield_curve_from_json(
                cfg.json_path, cfg.foreign_curve_risk_key,
                rate_convention=cfg.json_rate_convention, compounding_freq=cfg.json_compounding_freq,
                interpolation=cfg.json_discount_interp,
            )
            fx_spot = FXSpotRate(
                rate=build_fx_spot_from_json(cfg.json_path, cfg.fx_spot_risk_key),
                spot_days=cfg.fx_spot_days, spot_calendar=cfg.fx_spot_calendar,
            )

    priced_forward_curve = fwd_curve
    if is_composite:
        priced_forward_curve = build_composite_forward_curve(
            foreign_forward_curve=fwd_curve, fx_spot=fx_spot,
            domestic_discount_curve=domestic_curve, foreign_discount_curve=foreign_curve,
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
        # priced_forward_curve (not the raw foreign fwd_curve) so the vol
        # adapter's forward lookup matches the strike's currency when
        # composite -- see market_data_json.py's _MalzVolAdapter docstring.
        vol_skew = build_commodity_vol_skew_from_json(
            cfg.json_path, cfg.vol_risk_key, priced_forward_curve,
            surface_type=cfg.vol_surface_type,
        )

    common_kwargs = dict(
        val_date=cfg.val_date, maturity_date=cfg.maturity, strike=cfg.strike,
        is_call=cfg.is_call, digital_type=cfg.digital_type, payout=cfg.payout,
        forward_curve=priced_forward_curve, discount_curve=domestic_curve,
        vol=cfg.vol, vol_skew=vol_skew, apply_skew_adjustment=cfg.apply_skew_adjustment,
        skew_strike_bump_frac=cfg.skew_strike_bump_frac,
        spot_days=cfg.spot_days, spot_calendar=cfg.spot_calendar,
        settle_days=cfg.domestic_settle_days, settlement_calendar=cfg.domestic_settlement_calendar,
        day_count=cfg.day_count,
    )
    result = value_commodity_digital_option(**common_kwargs)
    bump = bump_and_reprice_commodity_digital_option(**common_kwargs)

    print("=" * 72)
    print(f"European Commodity Digital ({'Call' if cfg.is_call else 'Put'}, "
          f"{cfg.digital_type}-or-nothing)  @ {cfg.val_date}  [source={cfg.data_source}]")
    print("=" * 72)
    if cfg.data_source == "csv":
        print(f"  Forward curve:   {cfg.forward_curve}  (interp={cfg.forward_interp})"
              f"{'  [foreign ccy]' if is_composite else ''}")
        print(f"  Domestic curve:  {cfg.domestic_discount_curve}  "
              f"(convention={cfg.domestic_rate_convention}, interp={cfg.domestic_discount_interp})")
        if is_composite:
            print(f"  Foreign curve:   {cfg.foreign_discount_curve}  "
                  f"(convention={cfg.foreign_rate_convention}, interp={cfg.foreign_discount_interp})")
            print(f"  FX spot:         {fx_spot.rate}  "
                  f"(spot_days={fx_spot.spot_days}, calendar={fx_spot.spot_calendar})")
    else:
        print(f"  JSON:            {cfg.json_path}")
        print(f"  Forward risk key:  {cfg.forward_risk_key}  (interp={cfg.json_forward_interp})"
              f"{'  [foreign ccy]' if is_composite else ''}")
        print(f"  Domestic curve key: {cfg.domestic_curve_risk_key}  "
              f"(convention={cfg.json_rate_convention}, interp={cfg.json_discount_interp})")
        if is_composite:
            print(f"  Foreign curve key:  {cfg.foreign_curve_risk_key}")
            print(f"  FX spot key:        {cfg.fx_spot_risk_key} = {fx_spot.rate}  "
                  f"(spot_days={fx_spot.spot_days}, calendar={fx_spot.spot_calendar})")
    if vol_skew is not None:
        skew_src = cfg.vol_skew_curve if cfg.vol_skew_curve is not None else cfg.vol_risk_key
        print(f"  Vol skew source: {skew_src}  (apply_skew_adjustment={cfg.apply_skew_adjustment})")
    print(f"  Maturity / Strike / Payout: {cfg.maturity} / {cfg.strike} / {cfg.payout}")
    print(f"  Forward (spot-day adj.):    {result.forward:.6f}   Vol (at strike): {result.vol:.4%}")
    print(f"  T_opt / T_disc / DF:        {result.T_opt:.6f} / {result.T_disc:.6f} / {result.df:.8f}")
    print("-" * 72)
    print(f"  Price:              {result.price:>14.6f}")
    if result.analytic_delta is not None:
        print(f"  Analytic  Delta/Gamma/Vega/Theta: "
              f"{result.analytic_delta:.6f} / {result.analytic_gamma:.8f} / "
              f"{result.analytic_vega:.6f} / {result.analytic_theta:.6f}")
    else:
        print("  Analytic  Greeks:            n/a when skew-adjusted (see docstring)")
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
            "is_call": cfg.is_call,
            "digital_type": cfg.digital_type,
            "strike": cfg.strike,
            "payout": cfg.payout,
            "vol_input": cfg.vol,
            "vol_at_strike": result.vol,
            "vol_skew_source": (cfg.vol_skew_curve if cfg.vol_skew_curve is not None
                                 else cfg.vol_risk_key) if vol_skew is not None else "",
            "skew_adjusted": result.skew_adjusted,
            "forward_source": str(cfg.forward_curve) if cfg.data_source == "csv" else cfg.forward_risk_key,
            "domestic_curve_source": (str(cfg.domestic_discount_curve) if cfg.data_source == "csv"
                                       else cfg.domestic_curve_risk_key),
            "is_composite": is_composite,
            "foreign_curve_source": (
                (str(cfg.foreign_discount_curve) if cfg.data_source == "csv" else cfg.foreign_curve_risk_key)
                if is_composite else ""
            ),
            "fx_spot_rate": fx_spot.rate if is_composite else None,
            "forward": result.forward,
            "T_opt": result.T_opt,
            "T_disc": result.T_disc,
            "df": result.df,
            "price": result.price,
            "analytic_delta": result.analytic_delta,
            "analytic_gamma": result.analytic_gamma,
            "analytic_vega": result.analytic_vega,
            "analytic_theta": result.analytic_theta,
            "bump_delta": bump["delta"],
            "bump_gamma": bump["gamma"],
            "bump_vega": bump["vega"],
            "bump_theta": bump["theta"],
            "external_price": cfg.external_price,
            "diff": diff,
            "diff_pct": diff_pct,
        }
        append_results_log(cfg.results_log, row)
        print(f"Logged to {cfg.results_log}")


if __name__ == "__main__":
    main()
