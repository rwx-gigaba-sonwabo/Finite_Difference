"""
base_valuation/market_data_json.py
=====================================
JSON-driven market data loaders for base_valuation runner / validation
scripts -- the JSON-file counterpart to ``base_valuation.market_data_io``'s
CSV loaders. Pulls directly from a RiskFlow-exported market data JSON via
``utils.json_handling`` instead of a CSV file, then builds the exact same
``CommodityForwardCurve`` / ``YieldCurve`` objects the CSV loaders
produce -- so everything downstream (the digital and digital-barrier
pricers, the composite/FX machinery in ``base_valuation.fx``) is
unaware of which loader built the curve.

Verified against this repo's actual ``utils/json_handling.py`` and
``market_data/vol_surface.py`` (not guessed from the reference script
alone, and tested against a synthetic JSON matching the real navigation
path in ``pull_curve`` -- see the test in this module's accompanying
notes). Two things worth knowing before using this:

1. ``pull_curve(filepath, curve_name)`` navigates
   ``Calc.MergeMarketData.ExplicitMarketData.["Price Factors"][curve_name]
   .Curve.Curve.data`` and returns whatever is stored at ``data`` --
   assumed (matching the reference script's own usage, ``np.array(pull_curve(...))
   [:, 0]`` / ``[:, 1]``) to be an ``(n_pillars, 2)`` array-like of
   ``(tenor_year_frac, value)`` pairs. If your actual JSON stores
   something else under that path this will surface as a shape error
   from ``_pulled_curve_to_dates_and_values`` below, not a silent
   wrong answer.
2. ``MalzVol`` and ``NonPreciousCommodityVol`` (both real classes in
   ``market_data.vol_surface``) expose ``get_vol(strike, forward, tenor)``
   -- a **different, 3-argument** calling convention from
   ``BenchmarkVolSkew`` / ``SimpleSkew``'s ``get_vol(strike=, tenor=)``
   used throughout the rest of ``base_valuation``. A JSON-pulled vol
   surface is therefore **not** a drop-in ``vol_skew`` -- plugging one in
   directly would fail at the first ``vol_skew.get_vol(strike=...,
   tenor=...)`` call inside ``commodity_digital_option.py`` /
   ``commodity_digital_barrier_option.py`` with a missing-argument
   ``TypeError``. ``_MalzVolAdapter`` below bridges this by looking the
   forward up from a ``CommodityForwardCurve`` at the query tenor --
   ``build_commodity_vol_skew_from_json`` returns the adapter, already
   wired up, so it *is* a drop-in ``vol_skew`` for the existing pricers.

``pull_3d_fx_vol_surface`` (delta-quoted, e.g. an FX smile) feeds
``MalzVol`` directly; ``pull_commodity_fx_vol_surface`` (moneyness- or
delta-quoted depending on how your commodity vol risk factor is stored)
can feed either ``MalzVol`` or ``NonPreciousCommodityVol`` -- pass
``surface_type`` to pick. Check which convention your specific commodity
risk factor actually uses; feeding delta-quoted data into
``NonPreciousCommodityVol`` (which expects moneyness) or vice versa will
run without error but silently misprice, since both just index a 3-column
array by position.
"""
from __future__ import annotations

from datetime import date

import numpy as np

from base_valuation.commodity_curve import CommodityForwardCurve, FORWARD_INTERPOLATORS
from base_valuation.market_data_io import effective_annual_to_continuous
from base_valuation.yield_curve import INTERPOLATORS
from market_data.yield_curve import YieldCurve
from market_data.vol_surface import MalzVol, NonPreciousCommodityVol
from utils.ql_helpers import DAY_COUNTERS, to_ql_date
from utils.interpolation import excel_serial_to_date
from utils.json_handling import pull_curve, pull_3d_fx_vol_surface, pull_commodity_fx_vol_surface

__all__ = [
    "build_forward_curve_from_json",
    "build_yield_curve_from_json",
    "build_commodity_vol_skew_from_json",
    "build_fx_spot_from_json",
]


def _pulled_curve_to_dates_and_values(json_path: str, risk_key: str) -> tuple[list[date], np.ndarray]:
    """``pull_curve`` -> sorted ``(dates, values)``.

    The tenor column stores **Excel serial dates** (e.g. ``45868``), not
    year fractions -- confirmed against a real export where
    ``ForwardPrice.GOLD``'s pillars looked like ``[45868, 3307.06]``,
    ``[45869, 3307.46...]``, one per calendar day. Each is converted via
    ``utils.interpolation.excel_serial_to_date`` before use; year
    fractions (needed by ``YieldCurve``, which doesn't work with dates
    directly) are derived downstream from these dates and a caller-
    supplied ``val_date`` + day-count convention, not computed here.

    Handles one extra level of nesting some RiskFlow exports use --
    ``{"meta": [...], "data": [[t, v], ...]}`` -- since ``pull_curve``'s
    own ``.get('data', {})`` step can land on that wrapper dict rather
    than the inner list, depending on how deep the JSON actually nests
    "meta"/"data" under Curve['.Curve'].
    """
    raw = pull_curve(json_path, risk_key)

    if isinstance(raw, dict):
        if "data" in raw:
            raw = raw["data"]
        else:
            raise ValueError(
                f"pull_curve(json_path, {risk_key!r}) returned a dict with no "
                f"'data' key -- keys found: {sorted(raw.keys())!r}. The JSON's "
                f"structure under Price Factors[{risk_key!r}].Curve['.Curve'] doesn't "
                "match what this loader expects; open the JSON and check what's "
                "actually stored there."
            )

    try:
        arr = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        sample = raw[0] if isinstance(raw, list) and raw else raw
        raise ValueError(
            f"pull_curve(json_path, {risk_key!r}) returned data that isn't a plain "
            f"list of (tenor, value) pairs -- got {type(raw).__name__}, first "
            f"element looks like {sample!r}. If pillars are dicts (e.g. "
            "{'date': ..., 'value': ...} or {'tenor': ..., 'rate': ...}), this "
            "loader needs updating to read those key names -- tell me what the "
            "actual keys are and I'll fix it."
        ) from exc

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f"pull_curve(json_path, {risk_key!r}) returned an array of shape "
            f"{arr.shape}; expected (n_pillars, 2) of (excel_serial_date, value). "
            "Check what's actually stored at Price Factors[risk_key].Curve['.Curve'].data "
            "in your JSON."
        )

    order = np.argsort(arr[:, 0])
    serials, values = arr[order, 0], arr[order, 1]
    dates = [excel_serial_to_date(s) for s in serials]
    return dates, values


def build_forward_curve_from_json(
    json_path: str,
    risk_key: str,
    val_date: date,
    day_count: str = "ACT/365",
    interpolation: str = "forward_price",
) -> CommodityForwardCurve:
    """JSON counterpart of
    :func:`base_valuation.market_data_io.build_forward_curve_from_csv`.

    Pulls ``(excel_serial_date, price)`` pillars for *risk_key* via
    ``pull_curve``, converts the serials to real calendar dates, and
    builds a ``CommodityForwardCurve`` on those exact dates -- no
    approximation needed here, unlike an earlier version of this
    function that reconstructed dates from a (wrongly assumed)
    year-fraction tenor column.

    Parameters
    ----------
    json_path : str
    risk_key : str
        E.g. ``"ForwardPrice.GOLD"``.
    val_date : date
        Must match the JSON's own valuation date -- pillar year
        fractions (needed only for the interpolator, not the dates
        themselves) are measured from here.
    day_count : str
    interpolation : {"forward_price", "linear"}

    Returns
    -------
    CommodityForwardCurve
    """
    curve_dates, prices = _pulled_curve_to_dates_and_values(json_path, risk_key)
    if interpolation not in FORWARD_INTERPOLATORS:
        raise ValueError(
            f"Unknown interpolation scheme {interpolation!r}; "
            f"choose from {sorted(FORWARD_INTERPOLATORS)}"
        )
    day_counter = DAY_COUNTERS[day_count]
    ql_val = to_ql_date(val_date)
    year_fracs = np.array(
        [float(day_counter.yearFraction(ql_val, to_ql_date(d))) for d in curve_dates],
        dtype=np.float64,
    )
    interp = FORWARD_INTERPOLATORS[interpolation](year_fracs, prices.reshape(1, -1))

    return CommodityForwardCurve(
        val_date=val_date,
        curve_dates=curve_dates,
        year_fracs=year_fracs,
        prices=prices,
        day_counter=day_counter,
        interpolation=interpolation,
        _interp=interp,
    )


def build_yield_curve_from_json(
    json_path: str,
    risk_key: str,
    val_date: date,
    rate_convention: str = "NACC",
    compounding_freq: int = 1,
    interpolation: str = "hermite_rt",
    day_count: str = "ACT/365",
) -> YieldCurve:
    """JSON counterpart of
    :func:`base_valuation.market_data_io.build_yield_curve_from_csv`.

    Pulls ``(excel_serial_date, rate)`` pillars for *risk_key* via
    ``pull_curve``, converts each serial to a real calendar date, and
    derives year fractions from *val_date* via *day_count* -- the tenor
    column is a calendar-date serial, not a pre-computed year fraction
    (confirmed against a real export; see
    :func:`_pulled_curve_to_dates_and_values`).

    Parameters
    ----------
    json_path : str
    risk_key : str
        E.g. ``"InterestRate.ZAR-SWAP"``.
    val_date : date
        Must match the JSON's own valuation date -- year fractions are
        measured from here.
    rate_convention : {"NACA", "NACC"}
        ``"NACC"`` default here (unlike the CSV loader's ``"NACA"``
        default) -- verify against your JSON's actual convention and
        pass ``"NACA"`` if the stored rates are annually compounded.
    compounding_freq : int
        Only used when ``rate_convention == "NACA"``.
    interpolation : str
        Any key in ``base_valuation.yield_curve.INTERPOLATORS``.
    day_count : str

    Returns
    -------
    YieldCurve
    """
    if rate_convention not in ("NACA", "NACC"):
        raise ValueError(f"rate_convention must be 'NACA' or 'NACC', got {rate_convention!r}")
    curve_dates, rates = _pulled_curve_to_dates_and_values(json_path, risk_key)
    day_counter = DAY_COUNTERS[day_count]
    ql_val = to_ql_date(val_date)
    year_fracs = np.array(
        [float(day_counter.yearFraction(ql_val, to_ql_date(d))) for d in curve_dates],
        dtype=np.float64,
    )
    if rate_convention == "NACA":
        rates = effective_annual_to_continuous(rates, compounding_freq)
    if interpolation not in INTERPOLATORS:
        raise ValueError(
            f"Unknown interpolation scheme {interpolation!r}; "
            f"choose from {sorted(INTERPOLATORS)}"
        )
    return YieldCurve(
        year_fracs=year_fracs,
        rates=rates.reshape(1, -1),
        interpolator=INTERPOLATORS[interpolation],
    )


class _MalzVolAdapter:
    """Bridges ``MalzVol`` / ``NonPreciousCommodityVol``'s
    ``get_vol(strike, forward, tenor)`` to the ``get_vol(strike=, tenor=)``
    convention ``BenchmarkVolSkew`` / ``SimpleSkew`` use, by looking the
    forward up from a ``CommodityForwardCurve`` at the query tenor. This
    is what makes a JSON-pulled vol surface a genuine drop-in ``vol_skew``
    for :func:`~base_valuation.commodity_digital_option.value_commodity_digital_option`
    / :func:`~base_valuation.commodity_digital_barrier_option.value_commodity_digital_barrier_option`
    with ``apply_skew_adjustment=True`` -- no changes needed on the
    pricer side.
    """

    def __init__(self, vol_surface, forward_curve: CommodityForwardCurve):
        self._vol_surface = vol_surface
        self._forward_curve = forward_curve

    def get_vol(self, strike, tenor):
        forward = self._forward_curve.price_at_year_frac(tenor)
        return self._vol_surface.get_vol(strike=strike, forward=forward, tenor=tenor)


def build_commodity_vol_skew_from_json(
    json_path: str,
    risk_key: str,
    forward_curve: CommodityForwardCurve,
    surface_type: str = "malz",
) -> _MalzVolAdapter:
    """Pulls the commodity vol surface for *risk_key* and wraps it as a
    drop-in ``vol_skew`` (``get_vol(strike=, tenor=)``) for the existing
    ``apply_skew_adjustment`` pricing path -- the JSON counterpart of
    :func:`base_valuation.market_data_io.build_vol_skew_from_csv`.

    Parameters
    ----------
    json_path : str
    risk_key : str
        E.g. ``"CommodityPriceVol.GOLD"``.
    forward_curve : CommodityForwardCurve
        Used by the returned adapter to look up the forward at each
        query tenor (``MalzVol`` / ``NonPreciousCommodityVol`` both need
        an explicit forward, not just a strike and tenor).
    surface_type : {"malz", "non_precious"}
        ``"malz"`` (default) -- delta-quoted surface via
        ``pull_3d_fx_vol_surface`` + ``MalzVol``, matching the only
        pattern actually exercised in the reference script (used there
        for both an FX smile and a commodity smile). ``"non_precious"``
        -- moneyness-quoted surface via ``pull_commodity_fx_vol_surface``
        + ``NonPreciousCommodityVol``. Check which convention your
        specific commodity vol risk factor is actually stored in --
        feeding the wrong shape into the wrong class runs without error
        but silently misprices, since both index a 3-column array by
        position rather than by label.

    Returns
    -------
    _MalzVolAdapter
    """
    if surface_type == "malz":
        raw = pull_3d_fx_vol_surface(json_path, risk_key)
        vol_surface = MalzVol(raw)
    elif surface_type == "non_precious":
        raw = pull_commodity_fx_vol_surface(json_path, risk_key)
        vol_surface = NonPreciousCommodityVol(raw)
    else:
        raise ValueError(f"surface_type must be 'malz' or 'non_precious', got {surface_type!r}")
    return _MalzVolAdapter(vol_surface, forward_curve)


def build_fx_spot_from_json(json_path: str, risk_key: str) -> float:
    """Pulls a scalar FX spot rate for *risk_key* via ``pull_curve``,
    reused here even though an FX spot has no term structure -- there is
    no dedicated scalar puller visible in ``utils.json_handling``.
    Raises if the pulled data has more than one distinct value (a real
    curve, not a scalar), since that would indicate *risk_key* points at
    the wrong risk factor.

    Parameters
    ----------
    json_path : str
    risk_key : str
        E.g. ``"FxRate.ZAR"``.

    Returns
    -------
    float
    """
    _, values = _pulled_curve_to_dates_and_values(json_path, risk_key)
    if not np.allclose(values, values[0]):
        raise ValueError(
            f"pull_curve(json_path, {risk_key!r}) returned multiple distinct values "
            f"{values.tolist()} for an FX spot rate, which should be scalar -- "
            f"{risk_key!r} may not be the right risk factor name."
        )
    return float(values[0])
