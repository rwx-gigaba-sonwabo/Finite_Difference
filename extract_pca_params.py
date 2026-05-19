def extract_pca_params(filepath, asset_names):
    import json
    import os

    if isinstance(asset_names, str):
        asset_names = [asset_names]

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        market_data = json.load(f)

    price_models = (
        market_data
        .get('MarketData', {})
        .get('Price Models', {})
    )

    results = {}

    for asset_name in asset_names:
        if asset_name not in price_models:
            available = [k for k in price_models if 'PCA' in k]
            print(f"WARNING: '{asset_name}' not found.")
            print(f"  Available PCA models: {available}")
            continue

        model = price_models[asset_name]

        # ── Curve unpacker — handles both storage formats ────────────────────
        # Format A: value is already [[tenor, val], ...]
        # Format B: value is {'.Curve': {'meta':[], 'data':[[tenor,val],...]}}
        def unpack_curve(raw):
            if raw is None:
                return []
            if isinstance(raw, list):
                return raw                          # already clean pairs
            if isinstance(raw, dict):
                if '.Curve' in raw:
                    return raw['.Curve'].get('data', [])
                if 'data' in raw:
                    return raw['data']
            return []

        # ── Eigenvectors: list of {Eigenvalue, Eigenvector} dicts ────────────
        eigenvectors = []
        for ev_block in model.get('Eigenvectors', []):
            eigenvectors.append({
                'Eigenvalue' : ev_block.get('Eigenvalue'),
                'Eigenvector': unpack_curve(ev_block.get('Eigenvector')),
            })

        results[asset_name] = {
            'Reversion_Speed'  : model.get('Reversion_Speed'),
            'Historical_Yield' : unpack_curve(model.get('Historical_Yield')),
            'Yield_Volatility' : unpack_curve(model.get('Yield_Volatility')),
            'Eigenvectors'     : eigenvectors,
            'Rate_Drift_Model' : model.get('Rate_Drift_Model'),
            'Princ_Comp_Source': model.get('Princ_Comp_Source'),
            'Distribution_Type': model.get('Distribution_Type'),
        }
        
    return results


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
