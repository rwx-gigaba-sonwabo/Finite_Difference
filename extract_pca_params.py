def extract_pca_params(filepath, asset_names):
    """
    Extract PCA Interest Rate model parameters from MarketData.json.
    
    Args:
        filepath   : Path to MarketData.json
        asset_names: String or list of PCA model names
                     e.g. 'PCAInterestRateModel.ZAR-CORE-CPI'
                     or   ['PCAInterestRateModel.ZAR-CORE-CPI',
                            'PCAInterestRateModel.ZAR-BOND']
    Returns:
        Dictionary keyed by asset name, each containing:
        {
            'Reversion_Speed'  : float,
            'Historical_Yield' : list of [tenor, value] pairs,
            'Yield_Volatility' : list of [tenor, value] pairs,
            'Eigenvectors'     : [
                {'Eigenvalue': float, 'Eigenvector': list of [tenor, value]},
                ...   (one per PC)
            ],
            'Rate_Drift_Model' : str,
            'Princ_Comp_Source': str,
            'Distribution_Type': str,
        }
    """
    import json

    if isinstance(asset_names, str):
        asset_names = [asset_names]

    # Initialise result structure
    result = {name: {
        'Reversion_Speed'  : None,
        'Historical_Yield' : [],
        'Yield_Volatility' : [],
        'Eigenvectors'     : [],
        'Rate_Drift_Model' : None,
        'Princ_Comp_Source': None,
        'Distribution_Type': None,
    } for name in asset_names}

    with open(filepath, 'r') as f:
        market_data = json.load(f)

    price_models = (
        market_data
        .get('MarketData', {})
        .get('Price Models', {})
    )

    for asset_name in asset_names:
        if asset_name not in price_models:
            print(f"WARNING: '{asset_name}' not found in Price Models.")
            continue

        model = price_models[asset_name]
        r     = result[asset_name]

        # --- Scalar ---
        r['Reversion_Speed']   = model.get('Reversion_Speed')
        r['Rate_Drift_Model']  = model.get('Rate_Drift_Model')
        r['Princ_Comp_Source'] = model.get('Princ_Comp_Source')
        r['Distribution_Type'] = model.get('Distribution_Type')

        # --- Curve helper: extracts list of [tenor, value] from .Curve.data ---
        def extract_curve(key):
            return (
                model.get(key, {})
                     .get('.Curve', {})
                     .get('data', [])
            )

        r['Historical_Yield'] = extract_curve('Historical_Yield')
        r['Yield_Volatility'] = extract_curve('Yield_Volatility')

        # --- Eigenvectors: list of dicts, one per PC ---
        for ev_block in model.get('Eigenvectors', []):
            r['Eigenvectors'].append({
                'Eigenvalue' : ev_block.get('Eigenvalue'),
                'Eigenvector': (
                    ev_block.get('Eigenvector', {})
                             .get('.Curve', {})
                             .get('data', [])
                ),
            })

    return result


# ── Pretty-print helper ──────────────────────────────────────────────────────
def print_pca_params(result):
    for asset_name, r in result.items():
        print(f"\n{'='*60}")
        print(f"  {asset_name}")
        print(f"{'='*60}")
        print(f"  Reversion Speed : {r['Reversion_Speed']:.6f}")
        print(f"  Rate Drift Model: {r['Rate_Drift_Model']}")
        print(f"  Princ Comp Src  : {r['Princ_Comp_Source']}")
        print(f"  Distribution    : {r['Distribution_Type']}")

        print(f"\n  --- Historical Yield ---")
        for tenor, value in r['Historical_Yield']:
            print(f"    Tenor {tenor:.4f}y : {value:.6f}")

        print(f"\n  --- Yield Volatility ---")
        for tenor, value in r['Yield_Volatility']:
            print(f"    Tenor {tenor:.4f}y : {value:.6f}")

        print(f"\n  --- Eigenvectors ---")
        for i, ev in enumerate(r['Eigenvectors']):
            print(f"\n    PC{i+1}  Eigenvalue: {ev['Eigenvalue']:.6f}")
            for tenor, loading in ev['Eigenvector']:
                print(f"      Tenor {tenor:.4f}y : {loading:.8f}")


# ── Usage ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = extract_pca_params(
        filepath     = r"C:\XVA_engine\MarketData.json",
        asset_names  = "PCAInterestRateModel.ZAR-CORE-CPI"
    )
    print_pca_params(result)