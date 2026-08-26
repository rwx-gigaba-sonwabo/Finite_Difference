import json

def pull_curve(filepath, curve_name):
    #pulls data from trade json file
    with open(filepath) as f:
        data = json.load(f)

    # Navigate to Price Factors and pick the curve by its name
    factors = (data.get('Calc', {})
                   .get('MergeMarketData', {})
                   .get('ExplicitMarketData', {})
                   .get('Price Factors', {}))

    curve_item = factors.get(curve_name, {})
    # Extract the data list: Curve -> .Curve -> meta/data
    curve_data = (curve_item.get('Curve', {})
                   .get('.Curve', {})
                   .get('data', {}))

    return curve_data

def pull_3d_fx_vol_surface(filepath, surface_name):
    #pulls data from trade json file
    with open(filepath) as f:
        data = json.load(f)

    # Navigate to Price Factors and pick the curve by its name
    factors = (data.get('Calc', {})
                   .get('MergeMarketData', {})
                   .get('ExplicitMarketData', {})
                   .get('Price Factors', {}))

    curve_item = factors.get(surface_name, {})
    # Extract the data list: Curve -> .Curve -> meta/data
    surface_data = (curve_item.get('Delta_Surface', {})
                              .get('.Curve', {})
                              .get('data', []))
    return surface_data

def pull_commodity_fx_vol_surface(filepath, surface_name):
    #pulls data from trade json file
    with open(filepath) as f:
        data = json.load(f)

    # Navigate to Price Factors and pick the curve by its name
    factors = (data.get('Calc', {})
                   .get('MergeMarketData', {})
                   .get('ExplicitMarketData', {})
                   .get('Price Factors', {}))

    curve_item = factors.get(surface_name, {})
    # Extract the data list: Curve -> .Curve -> meta/data
    surface_data = (curve_item.get('Surface', {})
                              .get('.Curve', {})
                              .get('data', []))
    return surface_data

def pull_4d_IR_vol_surface(filepath, surface_name):
    #pulls data from trade json file
    with open(filepath) as f:
        data = json.load(f)

    # Navigate to Price Factors and pick the curve by its name
    factors = (data.get('Calc', {})
                   .get('MergeMarketData', {})
                   .get('ExplicitMarketData', {})
                   .get('Price Factors', {}))

    curve_item = factors.get(surface_name, {})
    # Extract the data list: Curve -> .Curve -> meta/data
    surface_data = (curve_item.get('Surface', {})
                              .get('.Curve', {})
                              .get('data', []))
    return surface_data

def pull_curve_cva_market_data(filepath, curve_name):
    # pulls data from CVAMarketData file
    with open(filepath) as f:
        data = json.load(f)
    
    # Navigate to Price Factors and pick the curve by its name
    factors = (data.get('MarketData', {})
                   .get('Price Factors', {}))
    
    curve_item = factors.get(curve_name, {})
    # Extract the data list: Curve -> .Curve -> meta/data
    curve_data = (curve_item.get('Curve', {})
                            .get('Curve', {})
                            .get('data', {}))
    
    return curve_data


def pull_div_cashflows(filepath):
    with open(filepath) as f:
        data = json.load(f)

    children = (data.get('Calc', {})
                    .get('Deals', {})
                    .get('Deals', {})
                    .get('Children', []))

    items = []
    for child in children:
        child_items = (child.get('Children', []))
        
        items.extend(child_items)
    
    for item in items:
        div_data = (item.get('Instrument', {})
                        .get('.Deal', {})
                        .get('Cashflows', {})
                        .get('Items', []))

    return div_data

def pull_eq_vol_skew(filepath, surface_name):
    with open(filepath) as f:
        data = json.load(f)
    
    surface = data.get('Calc', {})
    surface = surface.get('MergeMarketData', {})
    surface = surface.get('ExplicitMarketData', {})
                   
    surface = surface.get('Price Factors', {})
    surface = surface.get(surface_name, {})

    atm_vol = surface.get('ATM_Vol', {}).get('.Curve', {}).get('data', {})
    atm_ref = surface.get('ATM_Ref', {}).get('.Curve', {}).get('data', {})
    s = surface.get('s', {}).get('.Curve', {}).get('data', {})
    L = surface.get('L', {}).get('.Curve', {}).get('data', {})
    R = surface.get('R', {}).get('.Curve', {}).get('data', {})
    C = surface.get('C', {}).get('.Curve', {}).get('data', {})
    D = surface.get('D', {}).get('.Curve', {}).get('data', {})
    lam = surface.get('lam', {}).get('.Curve', {}).get('data', {})
    rho = surface.get('rho', {}).get('.Curve', {}).get('data', {})
    return atm_vol, atm_ref, s, L, R, C, D, lam, rho