"""
base_valuation/fx.py
=======================
FX spot/forward machinery for composite (cross-currency) instruments --
where an underlying's natural forward price curve is denominated in one
currency (the "foreign" currency) but the contract itself settles in a
different currency (the "domestic" / contract currency).

Three curves, three distinct roles
------------------------------------
Following the pattern in ``instruments/commodity_average_forward.py``:

- ``domestic_curve`` / ``foreign_curve`` -- used **only** to compute the
  FX forward via covered interest rate parity. Either can optionally have
  a cross-currency basis spread added on top via
  :func:`combine_curve_with_basis` before being passed in here -- this
  module doesn't know or care whether that happened; it just takes
  whatever ``YieldCurve`` it's given.
- ``money_market_curve`` -- used **only** to discount the (already
  domestic-currency) payoff to the valuation date. In the same currency
  as ``domestic_curve``, but not necessarily the same curve object --
  e.g. ``domestic_curve`` might be a cross-currency basis-adjusted swap
  curve while ``money_market_curve`` is a plain OIS/deposit curve. This
  module never touches ``money_market_curve`` directly -- it's passed
  straight through by the run scripts as ``discount_curve`` to
  ``value_commodity_digital_option`` / ``value_commodity_digital_barrier_option``.
  For a standard (non-composite) trade, ``money_market_curve`` is the
  *only* curve used at all -- no domestic/foreign/FX involved.

Composed from the same primitives used throughout base_valuation:

- ``base_valuation.commodity_curve.build_commodity_forward_curve`` /
  ``CommodityForwardCurve`` -- to build the domestic-currency-converted
  forward curve
- ``market_data.yield_curve.YieldCurve`` -- domestic and foreign
  risk-free discounting
- ``utils.ql_helpers.advance_business_days`` -- spot-day lag handling,
  consistent with the ``spot_days`` (forward curve) / ``settle_days``
  (discount curve) conventions already used elsewhere in this codebase

Modelling scope
----------------
No FX volatility or FX/underlying correlation is modelled here: the
underlying's forward price is converted to domestic-currency terms
**deterministically** via covered interest rate parity::

    F_domestic(T) = F_foreign(T) * fx_forward_rate(T)
    fx_forward_rate(T) = S * DF_domestic(spot) / DF_domestic(T)
                            / (DF_foreign(spot) / DF_foreign(T))

using ``domestic_curve`` and ``foreign_curve`` (each possibly basis-
adjusted). This is the standard, correlation-free way to express a
foreign-currency forward in domestic terms -- it is a **composite**
forward, not a **quanto** forward. A true quanto adjustment additionally
convexity-adjusts the drift for correlation between the underlying and
FX (requires an FX volatility and a correlation input); that is out of
scope here since it wasn't part of the requested feature, and can be
layered on separately if a genuine quanto payoff (fixed FX rate, not the
prevailing one) is ever needed. For a composite payoff -- which converts
at the *prevailing* FX rate at expiry -- the deterministic forward is
the correct risk-neutral drift; correlation only shows up in a
*composite vol* adjustment to the option's own volatility input, not
here. If the volatility you're feeding into the digital pricers already
reflects the composite volatility of the domestic-currency price, no
further adjustment is needed downstream of this module.

Once the composite forward curve is built, valuation proceeds exactly
as for a single-currency trade -- ``value_commodity_digital_option`` /
``value_commodity_digital_barrier_option`` need no changes at all: pass
the composite curve as their ``forward_curve`` and ``money_market_curve``
as their ``discount_curve``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from base_valuation.commodity_curve import (
    CommodityForwardCurve,
    build_commodity_forward_curve,
)
from market_data.yield_curve import YieldCurve
from utils.ql_helpers import DAY_COUNTERS, advance_business_days, to_ql_date


@dataclass
class FXSpotRate:
    """Today's FX spot rate, with its own settlement lag.

    Parameters
    ----------
    rate : float
        Domestic currency units per 1 unit of foreign currency (e.g. for
        a USD-denominated commodity underlying valued into a ZAR
        contract, ``rate`` is USDZAR -- how many ZAR buy 1 USD).
    spot_days, spot_calendar : int, str
        Standard FX spot-value lag for the currency pair (commonly T+2),
        mirroring the ``spot_days`` / ``spot_calendar`` convention
        already used for the commodity forward curve lookup elsewhere in
        this codebase.
    """
    rate: float
    spot_days: int = 2
    spot_calendar: str = "USD"


def combine_curve_with_basis(
    base_curve: YieldCurve,
    basis_curve: YieldCurve,
    interpolator,
) -> YieldCurve:
    """Add a cross-currency basis spread onto a base curve, following the
    additive convention in ``instruments/commodity_average_forward.py``::

        combined_rate(T) = base_curve.get_rate(T) + basis_curve.rates
        (evaluated at basis_curve's own tenor grid, T = basis_curve.year_fracs)

    The combined curve is built on *basis_curve*'s own pillar grid --
    ``base_curve`` is only ever queried (interpolated) at those tenors,
    never the other way round. Both curves must already be in the same
    (continuously compounded) rate convention -- if either was loaded
    from a NACA-quoted source, convert it (e.g. via
    ``base_valuation.market_data_io.effective_annual_to_continuous``)
    before building the ``YieldCurve`` passed in here; this function
    does no further conversion, just addition.

    Works identically for a domestic or a foreign curve -- which side (if
    either) gets a basis applied is entirely up to the caller.

    Parameters
    ----------
    base_curve : YieldCurve
        The curve the basis is layered onto (e.g. a domestic or foreign
        money-market / swap curve).
    basis_curve : YieldCurve
        The basis spread curve (e.g. "ZAR-USD-BASIS"). Only its
        ``year_fracs`` (tenor grid) and ``rates`` (spread values) are
        used -- its own interpolator, if any, is never invoked.
    interpolator : callable
        Factory with signature ``(year_fracs, rates) -> callable(t)``,
        used to build the combined curve -- may differ from either input
        curve's own interpolator.

    Returns
    -------
    YieldCurve
        A new curve on *basis_curve*'s tenor grid, rates = base + basis.
    """
    base_rates_at_basis_tenors = base_curve.get_rate(basis_curve.year_fracs)
    combined_rates = base_rates_at_basis_tenors + basis_curve.rates
    return YieldCurve(
        year_fracs=basis_curve.year_fracs,
        rates=combined_rates,
        interpolator=interpolator,
    )


def fx_forward_rate(
    fx_spot: FXSpotRate,
    val_date: date,
    delivery_date: date,
    domestic_curve: YieldCurve,
    foreign_curve: YieldCurve,
    domestic_settle_days: int = 2,
    domestic_settlement_calendar: str = "USD",
    foreign_settle_days: int = 2,
    foreign_settlement_calendar: str = "USD",
    day_count: str = "ACT/365",
) -> float:
    """Covered-interest-rate-parity FX forward for delivery on *delivery_date*.

        F = S * [DF_d(t_spot) / DF_d(t_domestic_settle)]
              / [DF_f(t_spot) / DF_f(t_foreign_settle)]

    ``t_spot`` is *fx_spot*'s own spot-value date, advanced from
    *val_date*. ``t_domestic_settle`` / ``t_foreign_settle`` are
    *delivery_date* advanced by each curve's own settle lag -- letting
    the domestic and foreign legs use different settlement conventions
    (currencies with differing T+n market practice), the same way the
    existing ``discount_curve`` / ``settle_days`` pattern already allows
    for a single-currency trade.

    When both curves share the same settle convention and *fx_spot* has
    no lag of its own (``spot_days=0``), this reduces to the textbook
    ``F = S * exp((r_d - r_f) * T)``.

    ``domestic_curve`` / ``foreign_curve`` here are used *only* to
    compute this forward -- they are not the discounting curve (see this
    module's docstring; discounting is ``money_market_curve``, handled
    entirely by the run scripts / pricers, not this module). Each may
    already have a basis spread layered on via
    :func:`combine_curve_with_basis` -- this function is indifferent to
    that, it just calls ``discount_factor`` on whatever it's given.

    Parameters
    ----------
    fx_spot : FXSpotRate
    val_date : date
        Valuation date (t0) -- reference date for both curves.
    delivery_date : date
        The date the domestic-currency-converted forward price is
        needed for (e.g. one of the commodity forward curve's own
        pillar dates).
    domestic_curve, foreign_curve : YieldCurve
        Both referenced to *val_date*.
    domestic_settle_days, domestic_settlement_calendar : as in
        ``value_commodity_digital_option``'s ``settle_days`` /
        ``settlement_calendar``.
    foreign_settle_days, foreign_settlement_calendar : the foreign
        curve's own settlement lag convention.
    day_count : str

    Returns
    -------
    float
    """
    dc = DAY_COUNTERS[day_count]
    ql_val = to_ql_date(val_date)

    fx_spot_date = advance_business_days(val_date, fx_spot.spot_days, fx_spot.spot_calendar)
    domestic_settle_date = advance_business_days(
        delivery_date, domestic_settle_days, domestic_settlement_calendar,
    )
    foreign_settle_date = advance_business_days(
        delivery_date, foreign_settle_days, foreign_settlement_calendar,
    )

    t_spot = max(float(dc.yearFraction(ql_val, to_ql_date(fx_spot_date))), 0.0)
    t_domestic = max(float(dc.yearFraction(ql_val, to_ql_date(domestic_settle_date))), 0.0)
    t_foreign = max(float(dc.yearFraction(ql_val, to_ql_date(foreign_settle_date))), 0.0)

    df_d_spot = float(domestic_curve.discount_factor(t_spot)[0])
    df_d_settle = float(domestic_curve.discount_factor(t_domestic)[0])
    df_f_spot = float(foreign_curve.discount_factor(t_spot)[0])
    df_f_settle = float(foreign_curve.discount_factor(t_foreign)[0])

    return fx_spot.rate * (df_d_spot / df_d_settle) / (df_f_spot / df_f_settle)


def build_composite_forward_curve(
    foreign_forward_curve: CommodityForwardCurve,
    fx_spot: FXSpotRate,
    domestic_curve: YieldCurve,
    foreign_curve: YieldCurve,
    domestic_settle_days: int = 2,
    domestic_settlement_calendar: str = "USD",
    foreign_settle_days: int = 2,
    foreign_settlement_calendar: str = "USD",
    day_count: str = "ACT/365",
) -> CommodityForwardCurve:
    """Convert a foreign-currency commodity forward curve into domestic
    (contract) currency terms, pillar by pillar, via covered interest
    rate parity.

    Each pillar's domestic-currency price is::

        foreign_price[i] * fx_forward_rate(..., delivery_date=curve_dates[i])

    The returned curve keeps the same pillar dates and interpolation
    scheme as *foreign_forward_curve*, so it is a drop-in replacement
    for a single-currency ``forward_curve`` everywhere else in this
    codebase -- the cost-of-carry curve-implied fallback, bump-and-
    reprice, etc. all continue to work completely unmodified. Pass the
    result straight into ``value_commodity_digital_option`` /
    ``value_commodity_digital_barrier_option`` as ``forward_curve``, and
    ``money_market_curve`` (**not** *domestic_curve* -- see this module's
    docstring) as their ``discount_curve``.

    Parameters
    ----------
    foreign_forward_curve : CommodityForwardCurve
        The underlying's own forward curve, in its natural (foreign)
        currency -- e.g. from ``build_forward_curve_from_csv``.
    fx_spot : FXSpotRate
    domestic_curve, foreign_curve : YieldCurve
        Used only for the FX forward calculation -- each already basis-
        combined via :func:`combine_curve_with_basis` if applicable.
        Both referenced to the same ``val_date`` as
        *foreign_forward_curve*.
    domestic_settle_days, domestic_settlement_calendar,
    foreign_settle_days, foreign_settlement_calendar : see
        :func:`fx_forward_rate`.
    day_count : str

    Returns
    -------
    CommodityForwardCurve
        Same pillar dates as *foreign_forward_curve*, prices converted
        to domestic currency.
    """
    val_date = foreign_forward_curve.val_date
    domestic_prices = np.array([
        price * fx_forward_rate(
            fx_spot, val_date, pillar_date,
            domestic_curve, foreign_curve,
            domestic_settle_days, domestic_settlement_calendar,
            foreign_settle_days, foreign_settlement_calendar,
            day_count,
        )
        for pillar_date, price in zip(
            foreign_forward_curve.curve_dates, foreign_forward_curve.prices,
        )
    ], dtype=np.float64)

    return build_commodity_forward_curve(
        val_date=val_date,
        curve_dates=foreign_forward_curve.curve_dates,
        prices=domestic_prices,
        day_count=day_count,
        interpolation=foreign_forward_curve.interpolation,
    )
