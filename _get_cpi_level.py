def _get_cpi_level(
    ref_date: date,
    val_date: date,
    cpi_interp: Optional[Callable],
    hist_map: Dict[date, float],
    curve_day_counter: ql.DayCounter,
    ql_val: ql.Date,
    n_paths: int,
    cpi_fixings: Optional[Dict[date, np.ndarray]] = None,
    spot_cpi: Optional[np.ndarray] = None,
    inflation_rate_curve: Optional[YieldCurve] = None,
    last_pub_date: Optional[date] = None,
) -> np.ndarray:
    """Return the CPI level at ``ref_date`` as an ``(n_paths,)`` array.

    Mirrors RiskFlow's pv_index_cashflows/calc_index exactly for the
    Riskflow (two-curve) mode: RiskFlow keeps no per-bracket "locked
    fixing" dictionary at all. Every valuation date re-reads the live
    simulated PriceIndex value AT THAT DATE and grows it forward using:

        CPI(ref_date) = spot_cpi(val_date) / DF_infl(T_last_pub(val_date) -> ref_date)

    -- unconditionally, for every ref_date that is not itself a confirmed
    published date. The only place a "known" value is used instead of this
    projection is when ref_date ITSELF is <= val_date and already resolved
    via cpi_fixings/hist_map (i.e. ref_date is the thing being priced, not
    an anchor for something else).
    """
    pub_cutoff = (
        last_pub_date
        if (last_pub_date is not None and inflation_rate_curve is not None)
        else val_date
    )

    if ref_date <= val_date:
        # Priority 1: per-path fixings for confirmed published bracket dates.
        if cpi_fixings is not None and ref_date in cpi_fixings and ref_date <= pub_cutoff:
            return np.asarray(cpi_fixings[ref_date], dtype=np.float64)

        # Priority 2: exact match in hist_map -- authoritative for any past date.
        if ref_date in hist_map:
            return np.full(n_paths, hist_map[ref_date], dtype=np.float64)

        if ref_date <= pub_cutoff:
            # Confirmed published, no exact hist_map match -- nearest before cutoff.
            known = [k for k in hist_map if k <= pub_cutoff]
            if known:
                return np.full(n_paths, hist_map[max(known)], dtype=np.float64)
            return np.zeros(n_paths, dtype=np.float64)

        # Unpublished past (pub_cutoff < ref_date <= val_date).
        if inflation_rate_curve is None:
            known = [k for k in hist_map if k <= val_date]
            if known:
                return np.full(n_paths, hist_map[max(known)], dtype=np.float64)
            return np.zeros(n_paths, dtype=np.float64)
        # Riskflow mode: fall through -- priced identically to a future
        # bracket date, exactly as RiskFlow's own calc_index does (it draws
        # no distinction between "unpublished past" and "future" -- both
        # get the same spot/DF projection).

    # --- Forward projection (Riskflow mode) ------------------------------
    if inflation_rate_curve is not None:
        origin = to_ql_date(last_pub_date) if last_pub_date is not None else ql_val
        t_ref = float(curve_day_counter.yearFraction(origin, to_ql_date(ref_date)))
        df_infl = np.asarray(
            inflation_rate_curve.discount_factor(t_ref), dtype=np.float64
        ).reshape(-1)
        anchor = np.asarray(spot_cpi, dtype=np.float64).reshape(-1)
        anchor = np.full(n_paths, float(anchor[0])) if anchor.size == 1 else anchor
        assert anchor.shape == df_infl.shape == (n_paths,), (anchor.shape, df_infl.shape)
        return anchor / df_infl

    # Legacy mode: interpolate CPI levels directly from the CurveSlice
    t_ref = float(curve_day_counter.yearFraction(ql_val, to_ql_date(ref_date)))
    return cpi_interp(t_ref)  # (n_paths,)
