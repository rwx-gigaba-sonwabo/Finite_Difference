# Digital options: adding asset-or-nothing settlement to `FXBinaryOption` and `EquityBinaryOption`

**Author:** Sonwabo (Model Validation)
**Date:** 2026-09-22
**Scope:** `derivus/instruments.py`, `derivus/pricing.py`, `derivus/utils.py`
**Purpose:** enable both cash-or-nothing and asset-or-nothing European digitals (call and put) with the put/call-spread skew approximation, within the current code structure.

---

## 1. Summary of current behaviour

Tracing `EquityBinaryOption.generate` / `FXBinaryOption.generate` → `pricing.pv_european_option` → `utils.black_european_option`, the engine already does more than we assumed:

| Payoff | Closed form | Call/put spread | Status |
|---|---|---|---|
| Cash-or-nothing **call** | ✅ | ✅ | Works today |
| Cash-or-nothing **put** | ✅ | ✅ | **Works today** (via `Option_Type`) |
| Asset-or-nothing call/put | ❌ | ❌ | **Not implemented** |

The put case is already covered because `factor_dep['Option_Type']` (+1 call / −1 put) flips the sign in **both** the closed form and the spread. `Option_Type` is populated in `Factor_dep` for the equity leg (`instruments.py:3500`) and in the FX deal's own field block, which is why the FX put digital prices correctly.

Relevant existing code:

```python
# utils.black_european_option — cash-or-nothing branch (L3941)
if cash_payoff:
    prem  = cash_payoff * norm_cdf(callorput * d2)         # call: N(d2), put: N(-d2)
    value = cash_payoff * (callorput * (forward - strike) > 0) * shared.one
```

```python
# pricing.pv_european_option — spread branch (L2669), central call/put spread
value = nominal * factor_dep['Buy_Sell'] * factor_dep['Option_Type'] * (
    legs[0] - legs[1]) / (2.0 * eps * strike)
```

**Conclusion:** the outstanding gap is *asset-or-nothing only*. No path in the library produces `forward · N(±d1)`; every `binary=True` path returns a cash payoff. The changes below add asset-or-nothing; they do not touch the (already working) cash-or-nothing put.

---

## 2. Intuition and the formula to implement

### 2.1 Cash-or-nothing vs asset-or-nothing

A European digital struck at `K`, expiry `T`, forward `F`, vol `σ(K)` from the surface:

```
d1 = ( ln(F/K) + 0.5*sigma^2*T ) / (sigma*sqrt(T))
d2 = d1 - sigma*sqrt(T)
```

- **Cash-or-nothing** pays a fixed cash amount if in-the-money:
  `CoN_call = DF * N(d2)`, `CoN_put = DF * N(-d2)`.
- **Asset-or-nothing** pays the underlying `S_T` if in-the-money:
  `AoN_call = DF * F * N(d1)`, `AoN_put = DF * F * N(-d1)`.

(`DF` is the settlement discount factor, applied once at the end of `pv_european_option`; the `forward` already carries carry/quanto/compo adjustments.)

### 2.2 Why asset-or-nothing is not just another spread

An asset-or-nothing digital is **not** a plain call/put spread. It decomposes into a vanilla plus the cash digital. With `w = Option_Type` (+1 call, −1 put):

```
S_T * 1{call ITM} = (S_T - K)^+          + K * 1{S_T > K}     ->  AoN_call = C(K) + K*CoN_call
S_T * 1{put  ITM} = K * 1{S_T < K}       - (K - S_T)^+        ->  AoN_put  = K*CoN_put - P(K)
```

Unified:

```
AoN = w * Vanilla(K; sigma(K)) + K * CoN_unit
```

where `Vanilla` and `CoN_unit` are the same option type as the deal and `CoN_unit` is the *unit* (pays 1) cash digital. Consistency check against the closed form:

- Call: `C + K*N(d2) = F*N(d1)` ✅
- Put: `-P + K*N(-d2) = F*N(-d1)` ✅

### 2.3 The skew (smile) point

The vanilla component `Vanilla(K)` is priced directly off the surface at its own strike `K`, so it carries no additional skew correction. **Only the cash digital piece needs the call/put-spread approximation** — that is where `dσ/dK` enters the price. A naive attempt to spread the `N(d1)` formula directly would give the vanilla piece a skew correction it should not have. So the correct implementation is:

```
AoN_spread = w * [ Vanilla(K; sigma(K)) + (leg_lo - leg_hi) / (2*eps) ]
```

where `leg_lo`, `leg_hi` are vanilla options (same type) evaluated at `K*(1-eps)` and `K*(1+eps)`, each reading the surface at its own strike — exactly the legs the existing cash spread already builds. This reuses the current central-difference machinery and keeps `Vanilla(K)` exact regardless of `eps`.

---

## 3. Code changes

Four edits. Change 1 and 2 are the pricing core; 3 and 4 expose the choice on each instrument.

### Change 1 — `utils.py`, `black_european_option` (~L3899, branch at L3941)

Add an `asset_payoff` argument mirroring `cash_payoff`, and a branch inside the computed (`strike != 0`) block:

```python
def black_european_option(F, X, vol, tenor, buyorsell, callorput, shared,
                          cash_payoff=0.0, asset_payoff=0.0, shift=0.0):
    ...
        if cash_payoff:
            prem  = cash_payoff * norm_cdf(callorput * d2)
            value = cash_payoff * (callorput * (forward - strike) > 0) * shared.one
        elif asset_payoff:
            # asset-or-nothing: pays asset_payoff * S_T if in-the-money
            prem  = asset_payoff * forward * norm_cdf(callorput * d1)
            value = asset_payoff * forward * (callorput * (forward - strike) > 0)
        else:
            prem  = callorput * (forward * norm_cdf(callorput * d1) - strike * norm_cdf(callorput * d2))
            value = torch.relu(callorput * (forward - strike))
    return buyorsell * torch.where(guard, prem, value)
```

> If these digitals can be priced under a **normal (Bachelier)** surface, add the analogous branch to `bachelier_european_option` (~L3885): `prem = asset_payoff * forward * norm_cdf(callorput * d)`.

### Change 2 — `pricing.py`, `pv_european_option` (signature L2630, binary block L2651–2679)

Add a `payoff_style` parameter and split the binary block into cash vs asset. The asset spread needs the vanilla at the deal strike (`vols`, already computed at `K`) plus the existing cash-spread numerator:

```python
def pv_european_option(shared, time_grid, deal_data, nominal, moneyness, forward,
                       binary=False, digital_spread=None, payoff_style='cash'):
    ...
    if binary:
        w      = factor_dep['Option_Type']
        strike = factor_dep['Strike_Price']
        if digital_spread is not None:
            eps, m_lo, m_hi = digital_spread
            legs = []
            for shift, m in ((-1.0, m_lo), (1.0, m_hi)):
                leg_vols = utils.VolSurface.rate(factor_dep['Volatility'], m, expiry, shared)
                if adj is not None and adj['fx_vol'] is not None:
                    leg_vols = compo_vol(leg_vols, adj['fx_vol'], adj['rho'])
                legs.append(utils.black_european_option(
                    forward, strike * (1.0 + shift * eps), leg_vols, expiry, 1.0,
                    w, shared))                                    # vanilla legs, own strike
            if payoff_style == 'asset':
                vanilla_K = utils.black_european_option(
                    forward, strike, vols, expiry, 1.0, w, shared)  # vanilla at K, vol(K)
                # AoN = w * [ Vanilla(K) + (leg_lo - leg_hi)/(2 eps) ]
                digital = w * (vanilla_K + (legs[0] - legs[1]) / (2.0 * eps))
            else:
                # cash-or-nothing (UNCHANGED behaviour)
                digital = w * (legs[0] - legs[1]) / (2.0 * eps * strike)
            value = nominal * factor_dep['Buy_Sell'] * digital
        else:
            if payoff_style == 'asset':
                value = utils.black_european_option(
                    forward, strike, vols, expiry,
                    factor_dep['Buy_Sell'], w, shared, asset_payoff=nominal)
            else:
                value = utils.black_european_option(
                    forward, strike, vols, expiry,
                    factor_dep['Buy_Sell'], w, shared, cash_payoff=nominal)
    else:
        ...
```

The cash path is byte-identical to today; only the `asset` path is new.

### Change 3 — `instruments.py`, `EquityBinaryOption` (fields L3943, generate call L3993)

```python
    fields = [ADMIN, EQUITYOPTIONBASE, own('EquityBinaryOption', [
        F('Cash_Payoff', 'Float', default=REQUIRED),
        F('Payoff_Style', 'Text', default='Cash', values=['Cash', 'Asset']),
        F('Settlement_Date', 'Date', default='')
    ])]
```

```python
        mtm = pricing.pv_european_option(
            shared, time_grid, deal_data, self.field['Cash_Payoff'], moneyness, forward,
            binary=True, digital_spread=spread,
            payoff_style=self.field['Payoff_Style'].lower()) * fx_rep
```

### Change 4 — `instruments.py`, `FXBinaryOption` (fields L6534, generate call L6584)

```python
        F('Cash_Payoff', 'Float', default=REQUIRED),
        F('Payoff_Style', 'Text', default='Cash', values=['Cash', 'Asset']),
```

```python
        mtm = pricing.pv_european_option(
            shared, time_grid, deal_data, self.field['Cash_Payoff'], moneyness, forward,
            binary=True, digital_spread=spread,
            payoff_style=self.field['Payoff_Style'].lower()) * fx_rep
```

---

## 4. Design decisions for the code owner

1. **Meaning of `nominal` under asset settlement.** Today `nominal = Cash_Payoff` is a cash amount. In the asset branch it becomes a *unit multiplier* on `forward · N(d1)` (number of underlying units). This is clean for equity (payoff = units × `S_T`). Rather than overloading `Cash_Payoff`, consider a dedicated `Units` / `Asset_Notional` field so the booking is self-documenting — the pricing code above is agnostic to which scalar is passed.

2. **FX asset-or-nothing is genuinely ambiguous** in a way the equity case is not. An FX digital can pay the **domestic** cash amount (cash-or-nothing, current behaviour) or **one unit of the foreign currency** worth `S_T` (asset-or-nothing), and the two settle in different currencies. The `forward · N(d1)` expression prices the foreign-paying version in domestic terms; the `fx_rep` / `SettleCurrency` handling in `cash_settle` (`pricing.py:2684`) should be confirmed to convert the correct leg before an FX asset-or-nothing number is trusted. The equity change is self-contained; the FX change needs this settlement-currency review.

3. **Spread convention.** The existing implementation is a *centred* call/put spread (unbiased to second order, `O(eps^2)` truncation error), not the one-sided over-replicating version. It is therefore a mid/unbiased price, not a conservative one — no built-in reserve. If a hedging spread or digital reserve is required for risk/reserving, that remains a separate consideration and is unchanged by this work.

---

## 5. Validation checks

After the changes, on a **flat** vol surface:

- `AoN_call - K * CoN_call` must reproduce the vanilla call from the same engine at strike `K`, **independent of `eps`**. If it drifts with `eps`, the spread is being applied to the `N(d1)` formula directly rather than to the decomposed cash piece.
- `AoN_call + AoN_put` must equal `forward · DF` (the discounted forward).
- `CoN_call + CoN_put` must equal `DF` (unchanged; regression check that cash behaviour is untouched).

With a **skewed** surface:

- `CoN_call` should exceed the flat-vol `DF · N(d2)` under negative equity skew, and `CoN_put` should be cheaper (equal and opposite `Vega · dσ/dK` term); `AoN` skew impact is `K ×` the cash impact.
- Sweep `K` across a quote pillar and confirm the digital price is continuous — discontinuities indicate surface interpolation artefacts rather than genuine skew, and are amplified by very small `eps`.

---

## 6. File / line reference

| File | Location | Edit |
|---|---|---|
| `derivus/utils.py` | `black_european_option` L3899, branch L3941 | add `asset_payoff` arg + branch |
| `derivus/utils.py` | `bachelier_european_option` L3885 | (optional) add asset branch if normal-vol pricing needed |
| `derivus/pricing.py` | `pv_european_option` L2630 / L2651–2679 | add `payoff_style`, split cash/asset |
| `derivus/instruments.py` | `EquityBinaryOption` L3943 / L3993 | add `Payoff_Style` field, thread through `generate` |
| `derivus/instruments.py` | `FXBinaryOption` L6534 / L6584 | add `Payoff_Style` field, thread through `generate` |
