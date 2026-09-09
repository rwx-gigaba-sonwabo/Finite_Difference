# Autocall Swap Validation

**RiskFlow (One-Step Survival) vs. Independent Monte Carlo**
*Full-note instrument theory, a verified methodology comparison, and a prospective stochastic-volatility extension*

Model Validation — XVA & Market Risk
Draft — Internal MRF Working Paper
18 August 2026

---

## Contents

- [0. Provenance and Verification Basis](#0-provenance-and-verification-basis)
- [1. What an Autocall Is](#1-what-an-autocall-is)
- [2. Full Payoff Definition](#2-full-payoff-definition)
- [3. The Independent Validation Model — Brute-Force Monte Carlo](#3-the-independent-validation-model--brute-force-monte-carlo)
- [4. Pricing in RiskFlow — One-Step Survival](#4-pricing-in-riskflow--one-step-survival)
- [5. Verification of the Supplied Document Against Source](#5-verification-of-the-supplied-document-against-source)
- [6. Validation Test Trades and Results](#6-validation-test-trades-and-results)
- [7. Theory of the Prospective Stochastic-Volatility Extension](#7-theory-of-the-prospective-stochastic-volatility-extension)
- [8. Recommendation and Next Steps](#8-recommendation-and-next-steps)
- [Appendix A. Notation](#appendix-a-notation)
- [Appendix B. Proof that the OSS Estimator Is Unbiased](#appendix-b-proof-that-the-oss-estimator-is-unbiased)

---

## 0. Provenance and Verification Basis

This document is built from a client-supplied validation note describing the pricing of an autocallable equity swap, extended and cross-checked in three respects agreed prior to drafting:

- Scope has been extended from the swap component alone to the full note, including the terminal (downside) leg, since the supplied note deliberately scoped itself to the swap component only (§1.3 of the source note).
- A new theory section (Section 7) has been added for the "jump-friendly stochastic asset model built from the stride up" — a two-factor stochastic-volatility, NIG-residual asset model reviewed separately in notebook form. This is presented throughout as a **prospective extension**: it is not implemented in the validation codebase as it currently stands, and no part of this document should be read as claiming otherwise.
- Every claim in the supplied note concerning RiskFlow's actual behaviour has been checked against RiskFlow's own source code, not merely reproduced. Section 5 sets out the result of that check claim by claim.

A note on the verification basis specifically for RiskFlow: RiskFlow's public source (`github.com/sylam/riskflow`) was read directly and in detail earlier in this review — including the exact docstring of its autocall pricer, its survival-probability formula, its optional Heston–Nandi spot model, and its dividend-curve handling — and the findings below rest on that direct reading. An attempt to re-fetch the same source today, specifically to re-confirm those findings before this document was drafted, was blocked by a transient network condition on the retrieval side rather than any change on RiskFlow's side; this is noted explicitly here rather than silently presented as a fresh re-verification. Nothing in Section 5 rests on the supplied note's own account of RiskFlow — each claim is checked independently against what was directly read from source.

The validation engine's own source (the independent Monte Carlo codebase referred to throughout as "the validation model") was read directly and in full for this document, including its dividend, quanto, and barrier-handling modules, and the findings in Sections 3 and 5 rest on that direct reading.

---

## 1. What an Autocall Is

### 1.1 Instrument description

An autocallable is a structured equity product. In exchange for assuming equity downside risk, the investor receives contingent coupons at levels above prevailing market rates, but only for so long as the product remains outstanding. The defining feature of the instrument is early redemption: on a scheduled observation date, should the underlying stand at or above (in the case of a call-type structure) a specified trigger level, the product terminates immediately, pays its coupon, and cancels all remaining cashflows.

The economic profile has three broad outcomes.

- **Favourable outcome** — the underlying stands at or above the trigger at an early observation date. The investor receives a coupon and redeems early, realising an attractive annualised return.
- **Intermediate outcome** — the underlying trades sideways below the trigger. The product remains outstanding, coupons go unpaid (or, in a Phoenix-style structure, are paid against a separate, lower coupon barrier), and the investor continues to hold the position.
- **Adverse outcome** — the underlying declines sharply through a downside barrier. The investor absorbs equity losses, having already forgone participation in the upside.

In economic substance the investor sells a put option and purchases a stream of conditional coupons.

### 1.2 Economic components

A full autocallable decomposes into four distinct economic components.

| Component | Economics | Determined by |
|---|---|---|
| Contingent coupon and autocall | Coupon paid on the first observation at or above the trigger, upon which the product terminates | upside trigger level |
| Principal redemption | Return of capital on redemption or at maturity | rebate and notional |
| Downside protection | A short down-and-in put, under which capital is at risk only if a lower barrier is breached | put barrier |
| Funding leg | Floating payments exchanged against the equity payoff, present where the structure is documented as a swap | floating curve |

This memorandum treats the full note — all four components together — as the general definition of the instrument. Where a specific validation exercise (Sections 3–6) prices the swap component only, that scoping is stated explicitly rather than assumed.

### 1.3 The three reference levels

**Strike $K$** denotes the reference or initial index level. It does not itself constitute a trigger. It serves as the anchor against which the remaining levels are quoted, and as the strike of the embedded put option.

**Trigger $H_j = \theta_j K$** denotes the upside level governing early redemption. It is quoted as a percentage of strike, specified per observation date, and may therefore step down (or up) over the life of the trade.

**Barrier $H_{\text{put}}$** denotes the downside knock-in level, set materially below strike. A breach of this level activates the embedded short put.

---

## 2. Full Payoff Definition

### 2.1 Mechanics

Let the observation dates be denoted $t_1 < \dots < t_n$, with $S_j$ the underlying level at $t_j$. On each observation date two events occur in sequence.

The first is the floating payment. Where $t_j$ is a floating date and the trade remains outstanding, $F_j$ is paid. The ordering is material, since a trade triggering on date $j$ remains liable for the floating payment falling due on that date.

The second is the trigger test. Where $S_j \ge H_j = \theta_j K$, the coupon $c_j$ is paid and the trade terminates, with no further coupons or floating payments arising.

Where no observation triggers over the life of the trade, no coupon is paid at maturity from the autocall leg itself; instead the note settles its terminal leg (§2.4), and floating payments have been exchanged throughout.

### 2.2 First-passage payoff — full note

Define the first trigger time

$$\tau = \min\{j : S_j \ge H_j\}, \qquad \tau=\infty \text{ if no trigger occurs.}$$

Only one coupon is ever paid from the autocall leg. The instrument is therefore a first-passage payoff, which is the source of its path dependence and distinguishes it from a simple sum of independent digital options. Extending the swap-only value of the supplied note to the full note by adding the terminal leg, the value is:

$$V = N\,\mathbb{E}^{\mathbb{Q}}\!\left[\, c_\tau D_\tau \mathbf{1}\{\tau<\infty\} \;+\; \mathbf{1}\{\text{survive to } T\}\, V_{\text{terminal}}\, D_T \;-\; \sum_k \mathbf{1}\{\text{outstanding at } t_k\}\, F_k D_k \,\right]$$

**Where:**
- $V$ is the present value of the full autocallable note
- $N$ is the notional, being buy/sell multiplied by units
- $\mathbb{E}^{\mathbb{Q}}$ is the expectation operator under the risk-neutral measure $\mathbb{Q}$
- $c_\tau, D_\tau$ are the coupon paid on trigger, and the discount factor to the trigger date
- $V_{\text{terminal}}$ is the terminal payoff settled if the note survives to maturity $T$ without ever triggering (§2.4)
- $D_T$ is the discount factor to maturity
- $F_k, D_k$ are the floating payment at $t_k$ and its discount factor

### 2.3 Reduction to survival probabilities — the autocall and floating legs

The expectation admits a substantial simplification for the autocall and floating legs. Define the survival probability

$$Q_j = \mathbb{Q}(S_1<H_1,\dots,S_j<H_j), \qquad Q_0 = 1$$

being the probability of surviving beyond date $j$ without triggering. The probability that the first trigger occurs precisely on date $j$ is then $Q_{j-1}-Q_j$, from which the autocall and floating legs reduce to:

$$V_{\text{swap}} = N\left[\;\sum_{j=1}^{n} c_j D_j\big(Q_{j-1}-Q_j\big)\;-\;\sum_k F_k D_k\,Q_{k^-}\;\right]$$

**Where:**
- $Q_{j-1} - Q_j$ is the probability that the note triggers precisely on date $j$
- $Q_{k^-}$ is the survival probability on entry to date $k$, i.e. the probability the note is still outstanding immediately before that date's trigger test

The valuation of the swap component therefore reduces in its entirety to the computation of the survival probabilities $Q_j$. The two implementations compared in this document evaluate these quantities by different means, which is the basis of the validation.

### 2.4 The terminal (downside) leg

The terminal leg is settled only on paths that survive to maturity without ever triggering the autocall. It comprises the return of principal, adjusted for any breach of the downside knock-in barrier:

$$V_{\text{terminal}} = R - \mathbf{1}\{\text{knocked in}\}\cdot\frac{\max(K - S_T,\, 0)}{K}$$

**Where:**
- $R$ is the rebate/principal amount; $R=0$ recovers the swap-only booking used to isolate the coupon and floating legs from the downside leg
- $\mathbf{1}\{\text{knocked in}\}$ is an indicator equal to 1 if the underlying ever traded at or below the put barrier $H_{\text{put}}$, on any date flagged for barrier observation, prior to or at maturity
- $K$ is the strike, which also serves as the reference level for the embedded put
- $S_T$ is the underlying level at maturity

Two properties of this leg are worth stating explicitly, since they are the source of most implementation subtlety in both models reviewed below. First, whether the note is knocked in is a fact about the path's history (did it ever touch the barrier), while the payoff size is always a function of the terminal level $S_T$ alone — the two are logically distinct and must be computed as such. Second, the knock-in barrier is a separate level from the autocall trigger, and may be monitored on a different schedule (continuously, or on its own discrete date set) — there is no general reduction of the knock-in probability to the same $Q_j$ recursion used for the autocall leg, since the joint law of first-passage-to-either-barrier is a materially harder problem than a single one-sided survival probability.

### 2.5 Market model

Under the risk-neutral measure the underlying is modelled as geometric Brownian motion,

$$S_j = S_{j-1}\exp\!\Big(\big(b_j-\tfrac12\sigma^2\big)\Delta t_j+\sigma\sqrt{\Delta t_j}\,Z_j\Big),\qquad Z_j\sim N(0,1)$$

**Where:**
- $b = r-q$ is the carry, obtained from the equity forward, being the repo curve less the continuous dividend yield
- $\sigma$ is the volatility of the underlying
- $\Delta t_j$ is the year fraction from $t_{j-1}$ to $t_j$

The simulated forward is thereby consistent with the input curves, satisfying $\mathbb{E}[S_j]=F(t_j)$. Since $\ln S_1,\dots,\ln S_n$ are jointly normal, with correlation $\rho_{ik}=\sqrt{t_i/t_k}$ for $i<k$, being the correlation of a Brownian motion sampled at two distinct times, each $Q_j$ constitutes a multivariate normal probability.

### 2.6 Quanto adjustment

Where the underlying is denominated in one currency and the trade settles in another, the trade is a quanto. Its cashflows are paid in the settlement currency at an exchange rate fixed in the term sheet, so the holder bears no exposure to movements in that rate. The coupon and floating amounts are therefore known in the settlement currency, but the distribution of the underlying — which determines whether the trade triggers — must nonetheless be taken under the risk-neutral measure of the settlement currency.

Under that measure the drift of an underlying denominated in a foreign currency carries an adjustment:

$$b_{\text{quanto}} = r - q - \rho_{SX}\,\sigma\,\sigma_X$$

**Where:**
- $\sigma_X$ is the volatility of the exchange rate
- $\rho_{SX}$ is the correlation between the underlying's return and the exchange rate's return

The term $-\rho_{SX}\sigma\sigma_X$ is the sole modification to the market model of §2.5. The survival probability construction of §2.3 then applies without further change, computed from the adjusted forward, and the resulting cashflows are converted at the fixed rate.

---

## 3. The Independent Validation Model — Brute-Force Monte Carlo

### 3.1 Approach

The validation model prices the autocall using full-path Monte Carlo simulation. Complete paths of the underlying are simulated, the trigger and knock-in barrier are each evaluated at every relevant observation date as an explicit event, and the discounted payoff is averaged across paths. This is deliberately independent, in construction, from RiskFlow's One-Step Survival estimator (Section 4), so that agreement between the two carries genuine evidential weight rather than reflecting a shared implementation.

### 3.2 Simulation of the underlying

The underlying is simulated as geometric Brownian motion on a grid merging every observation, barrier, and floating date. The transition density of geometric Brownian motion is known in closed form, so each step is drawn as an exact lognormal increment and no discretisation error is introduced by the diffusion itself.

Dividends are modelled as discrete cash amounts, verified directly in the validation engine's dividend module. Between dividend dates the underlying grows at the funding rate, and on each ex-dividend date the cash amount is deducted from the level:

$$S_j = S_{j-1}\exp\!\Big(\big(r-\tfrac12\sigma^2\big)\Delta t_j + \sigma\sqrt{\Delta t_j}\,Z_j\Big) \qquad \text{on a step carrying no dividend}$$

$$S_{t_i^{+}} = \max\big(S_{t_i^{-}} - d_i,\; 0\big) \qquad \text{on an ex-dividend date } t_i$$

**Where:**
- $d_i$ is the cash dividend amount deducted at ex-dividend date $t_i$
- $S_{t_i^-}, S_{t_i^+}$ are the level immediately before and immediately after the dividend deduction

This is a direct level-space treatment: the true spot is diffused, and dividends are applied as genuine jumps at their own ex-dates, rather than via an escrowed (spot-minus-PV-of-dividends) decomposition. This choice removes an entire class of reference-date and volatility-scaling error associated with escrow models, at the cost of the underlying no longer being exactly lognormal between dividend dates — a standard, well-documented trade-off in the discrete-dividend literature, and the more defensible choice of the two specifically for a barrier-heavy payoff such as this one.

### 3.3 Evaluation of the full payoff

The autocall and floating legs are evaluated path by path according to the mechanics of §2.1: on each observation date the floating payment is recognised while the trade remains outstanding, the trigger test is applied, and the trade terminates on the first date at which the underlying stands at or above its trigger.

The terminal leg is evaluated separately and is verified to have two properties matching §2.4 exactly. The knock-in state is a **latch** — once a path is observed at or below the put barrier on a flagged barrier date, it remains knocked in for the rest of that path's life, and is never reset. The payoff size is computed strictly from the terminal level $S_T$, decoupled from the knock-in event itself, exactly as the theory in §2.4 requires — a path that dips through the barrier and later recovers is still charged the loss corresponding to its actual terminal level, not the level at the moment it touched the barrier. The engine also supports seeding a path as already knocked in prior to the valuation date, for marking a live trade that has already breached its barrier.

The discounted coupon, floating, and terminal cashflows are accumulated on each path and averaged across paths. The result is reported as a total value, a decomposition into the autocall, terminal, and floating legs, and the Monte Carlo standard error of the estimate.

### 3.4 The single-coupon closed form

Where only one coupon remains outstanding the payoff admits an exact analytic value and no simulation is required. With a single observation there is no surviving path to propagate, so the coupon leg reduces to a cash-or-nothing digital option:

$$V = c_1 D_1\, \Phi(d_2),\qquad d_2 = \frac{\ln(F_1/H_1) - \tfrac12\sigma^2 t_1}{\sigma\sqrt{t_1}}$$

**Where:**
- $F_1$ is the forward to the observation date
- $\Phi$ is the standard normal cumulative distribution function

This is the standard Black-76 cash-or-nothing digital, using the forward rather than the spot, consistent with a risk-neutral lognormal forward. For these cases the validation model is checked directly against this closed form rather than against a simulated figure, giving an exact point of reference with no simulation error present. The same discipline extends, in principle, to a two-coupon case via a bivariate normal (the two triggers share a Brownian path up to the earlier date, with correlation $\sqrt{t_1/t_2}$); three or more coupons have no closed form, which is the mathematical justification for Monte Carlo beyond the single- and two-coupon cases.

### 3.5 Quanto treatment — two engines, two distinct roles

The validation codebase implements quanto treatment in two structurally different engines, and it is important to be precise about which one serves which purpose, since conflating them risks either an unnecessarily expensive primary pricer or an uninformative audit.

- The **primary engine** applies the quanto adjustment of §2.6 analytically, as a constant drift shift added at every simulation step — mechanically the same treatment as RiskFlow's own (§4.9). This is the cheaper, lower-variance choice, and is the appropriate default for routine pricing.
- A **second, structurally independent engine** simulates the exchange rate explicitly, as a second geometric Brownian motion genuinely correlated with the underlying, and values the settlement-currency cashflows by a change of measure with a stochastic, per-path discount weight. Under this engine the quanto adjustment is not imposed on the drift at all — it must emerge from the simulated correlation. This is a materially stronger test of the adjustment itself, since an engine that also hard-codes the identical formula only tests whether the formula was copied correctly, not whether the formula is right.

For a validation exercise specifically — as distinct from routine production pricing — using the second engine's output as the reference figure for the quanto comparison in Section 6 is the correct and stronger choice, and should be read as a deliberate methodological decision rather than a description of the engine ordinarily used to price quanto trades day to day.

### 3.6 Known limitations of the current validation model

Two limitations, identified through direct review of the validation engine's source, are recorded here so that they are not mistaken for RiskFlow discrepancies when interpreting Section 6.

- **Barrier configuration is not currently guarded.** Nothing in the deal specification enforces that a set of barrier observation dates has actually been supplied whenever a non-zero put barrier is set; if a deal specifies a put barrier without a corresponding set of barrier dates, the terminal leg will silently price as if no downside risk exists at all, with no error raised. This should be treated as an input-validation gap to close, not a pricing defect in the payoff formula itself.
- **Barrier monitoring is inherently discrete**, on whichever dates are supplied. The engine can approximate continuous monitoring by supplying a dense date grid, at a proportionate compute cost, but includes no analytic continuity correction (of the kind used to adjust a discretely-monitored barrier level toward its true continuously-monitored equivalent). Where the term sheet specifies continuous monitoring and the supplied barrier date grid is coarse, the true breach probability will be understated.

---

## 4. Pricing in RiskFlow — One-Step Survival

### 4.1 Method and origin

RiskFlow prices the autocall by One-Step Survival (OSS) Monte Carlo. In place of simulating whole paths and testing at each date whether the trigger has been reached, the method computes the trigger probability analytically at every observation date and simulates forward only the path that survives. Each simulated path therefore remains outstanding throughout, carrying a weight equal to the probability of having survived to that point, while the probability mass that triggers at each date is recognised as an expectation rather than as a discrete event.

The technique is due to Glasserman and Staum, who introduced it for the simulation of barrier options, the barrier crossing there playing the role of the autocall trigger (Paul Glasserman and Jeremy Staum, "Conditioning on One-Step Survival for Barrier Option Simulations," *Operations Research*, volume 49, number 6, pages 923–937, 2001). RiskFlow's implementation of this technique, and specifically the formulae in §4.2–§4.5 below, were confirmed by direct reading of RiskFlow's source in the course of this review.

### 4.2 Survival probability at each date

Given the level $S_{j-1}$ at the previous observation date, the trade survives date $j$ precisely when $S_j < H_j$, meaning the underlying fails to reach its trigger. Under the geometric Brownian motion step of §2.5 this event has probability

$$p_j = \Phi\!\left(\frac{\ln(H_j/S_{j-1}) - \big(b_j - \tfrac12\sigma^2\big)\Delta t_j}{\sigma\sqrt{\Delta t_j}}\right)$$

The complementary quantity $1-p_j$ is the probability that the trade triggers on date $j$, conditional on having reached it. Both are computed in closed form from the level carried into the date, with no simulation of the trigger itself. RiskFlow's default configuration uses a single volatility figure — the moneyness-at-expiry slice of the vol surface, applied identically at every observation date — rather than a term-structure- or skew-consistent input; this is flagged explicitly as an approximation in RiskFlow's own source and is a live finding for any validation exercise that includes barriers away from at-the-money or dates spread widely in time (§4.10 sets out an alternative, optional configuration).

### 4.3 Survival weight and the coupon leg

Define the survival weight

$$L_0 = 1, \qquad L_j = p_j L_{j-1} = \prod_{k \le j} p_k$$

being the probability that the trade is still outstanding on entering date $j$. The expected discounted coupon contributed at date $j$ is the probability of triggering precisely then, multiplied by the coupon and its discount factor:

$$(1 - p_j)\, L_{j-1}\, c_j\, D_j$$

### 4.4 The floating leg

The floating payment at date $k$ is discounted and weighted by the survival probability into that date,

$$L_{k^{-}}\, F_k\, D_k$$

entering the value with a negative sign, being paid away. The notation $L_{k^-}$ denotes the survival weight on entry to date $k$ — its value immediately before that date's trigger probability is applied, equal to the product of the survival probabilities over all observation dates strictly before $k$. Consistent with the mechanics of §2.1, the floating payment carries this entry weight rather than the weight after the trigger, so that a trade triggering on date $k$ still carries the floating payment falling due on that date.

### 4.5 Forward simulation of the surviving path

Having recognised the triggering mass analytically, the method must carry forward a level conditioned on survival, meaning conditioned on $S_j < H_j$. The surviving level is drawn from the correctly truncated distribution by inverse transform sampling:

$$Z_j = \Phi^{-1}\!\big(U_j\, p_j\big), \qquad U_j \sim \text{Uniform}(0,1)$$

after which $S_j = S_{j-1}\exp\!\big((b_j-\tfrac12\sigma^2)\Delta t_j + \sigma\sqrt{\Delta t_j}\, Z_j\big)$. Since $U_j p_j$ is uniform on the interval from zero to $p_j$, the draw is taken exactly from the region below the trigger. Every simulated path therefore remains outstanding, and no simulation effort is expended on paths that have already terminated.

### 4.6 The terminal leg under One-Step Survival

RiskFlow's own pricer documentation confirms, in its swap-leg valuation formula, that the terminal leg is carried through the same survival-weighting structure as the coupon and floating legs, discounted using the survival weight at maturity:

$$V = \sum_{j} (1-p_j) L_{j-1}\, c_j D_j \;+\; L_T\, V_{\text{terminal}}\, D_T \;-\; \sum_k L_{k^-}\, F_k D_k$$

**Where:**
- $L_T$ is the survival weight at maturity, i.e. the probability the note reaches $T$ without ever triggering
- $V_{\text{terminal}}$ is the terminal payoff of §2.4, settled only on the surviving weight

This confirms the structural placement of the terminal leg within RiskFlow's estimator — weighted by survival to maturity and discounted to $T$, exactly matching the theoretical decomposition in §2.4.

> **This review did not extend to independently re-deriving the precise analytic treatment RiskFlow applies to the knock-in probability itself within $V_{\text{terminal}}$** (for example, whether a continuously-monitored barrier receives a closed-form first-passage adjustment, or whether monitoring is restricted to the same discrete observation grid as the autocall trigger). This should be confirmed directly against RiskFlow's source before the terminal leg comparison in Section 6 is treated as fully verified in the same manner as the coupon and floating legs above.

### 4.7 The estimator

Collecting the legs, the value under OSS is the expression of §4.6, with the expectation taken over the surviving paths. The averaged survival weights recover the survival probabilities of §2.3, in that the expectation of $L_j$ is $Q_j$ and the expectation of $(1-p_j)L_{j-1}$ is $Q_{j-1}-Q_j$. The two implementations therefore compute the same quantities for the autocall and floating legs — the validation model by explicit path counting, RiskFlow by this analytic weighting — which is the basis on which agreement between them is meaningful. A full derivation that the estimator is unbiased is given in Appendix B.

### 4.8 Motivation — smooth sensitivities

The method is adopted principally for the behaviour of its sensitivities. A naive simulation books the trigger through an indicator function, which is discontinuous in the level, so that its derivative is zero almost everywhere and undefined at the trigger. Automatic differentiation through such an estimator returns essentially no delta from the trigger, and finite-difference sensitivities are unstable in the region of the trigger. Replacing the indicator by the normal distribution function removes the discontinuity: the estimator becomes a smooth and differentiable function of the level, the volatility, and the rates, without introducing the bias that a smoothing approximation would otherwise carry. A further benefit is variance reduction, since the trigger contributes analytically and no path is wasted after termination.

### 4.9 Quanto treatment

RiskFlow applies the quanto adjustment analytically. The term $-\rho_{SX}\sigma\sigma_X$ is added to the carry, the exchange rate is not simulated, and the cashflows are converted at the fixed rate. The correlation $\rho_{SX}$ is supplied as an input. This is mechanically identical to the primary (analytic-adjustment) engine of the validation model described in §3.5, and structurally different from that model's second, FX-simulating engine — a point worth bearing in mind when interpreting the quanto results in Section 6, since the validation figures reported there use the stronger, independently-emerging-adjustment engine specifically, not RiskFlow's own analytic-adjustment convention.

### 4.10 Optional stochastic-volatility spot model

Independently of the flat, single-moneyness vol default described in §4.2, RiskFlow separately implements a genuine Heston–Nandi discrete-time GARCH spot model, selectable per deal via a configuration option rather than applied by default. Where genuinely configured — with a calibrated parameter set bootstrapped for the specific underlying, from a tabular (not parametric) volatility surface — this replaces the flat volatility of §4.2 with a mean-reverting, correlated variance process that produces a genuine forward skew, materially improving the pricing of any leg (in particular the terminal downside leg) sensitive to strikes away from at-the-money. This option is not engaged by default and its use should be confirmed explicitly, deal by deal, rather than assumed.

### 4.11 Dividend treatment — a discrepancy from the validation model requiring explicit resolution

RiskFlow's dividend treatment, confirmed by direct reading of its source, is a **continuous dividend yield curve** entering the carry $b=r-q$ of §2.5 — there is no discrete cash dividend mechanism anywhere in RiskFlow's autocall pricer. This is a genuine and material difference from the validation model's treatment in §3.2, which models dividends as **discrete cash amounts** deducted at their own ex-dates.

This discrepancy is not hypothetical for the purposes of this document: the test-trade template in Section 6 carries a dividend column for both the single-coupon and standard-autocall trade sets, meaning any populated test set including dividend-bearing trades will compare two structurally different dividend conventions unless one side is first converted to match the other. Before any dividend-bearing trade is used as a validation case, either the discrete cash schedule should be converted to an equivalent continuous yield for comparison against RiskFlow, or RiskFlow's continuous yield should be converted to an equivalent discrete schedule for comparison against the validation model, and the conversion basis used should be stated alongside the result. Absent that conversion, any residual difference on a dividend-bearing trade conflates two effects — genuine model disagreement, and a convention mismatch that is not a disagreement at all — and should not be attributed to either model's correctness without first isolating which of the two it is.

---

## 5. Verification of the Supplied Document Against Source

This section addresses directly the request to confirm that the supplied note's account of each model is correct. Each substantive claim has been checked either against RiskFlow's own source (as read earlier in this review — see §0) or against the validation engine's source (read directly for this document). The verdict colour follows the convention used throughout this review: 🟢 confirmed correct as stated, 🟡 correct but requiring a stated clarification, 🔵 a claim this review could not itself re-confirm today and flags for direct confirmation, 🔴 a claim not supported by the source reviewed.

| Claim (source note section) | Verdict | Basis |
|---|---|---|
| General theory §1 — instrument definition, payoff, survival-probability reduction | 🟢 Correct | Standard first-passage decomposition; internally consistent and matches the payoff structure independently reviewed in the validation engine's own payoff evaluator |
| §1.8 — quanto carry adjustment $b_{\text{quanto}}=r-q-\rho\sigma\sigma_X$ | 🟢 Correct | Matches the adjustment independently verified in both RiskFlow and the validation engine's primary quanto engine earlier in this review |
| §2.2 — validation model uses discrete cash dividends | 🟢 Correct | Confirmed directly against the validation engine's dividend module |
| §2.4 — single-coupon closed form | 🟢 Correct | Standard Black-76 cash-or-nothing digital on the forward; formula verified algebraically |
| §2.5 — validation model's quanto treatment "emerges from correlation" | 🟡 Correct, needs one clarification | Accurately describes the validation engine's second (FX-simulating) quanto engine, but should not be read as describing the engine's ordinary production pricer, which applies the same analytic adjustment as RiskFlow — see §3.5 of this document |
| §3.1–§3.6 — RiskFlow's OSS mechanics ($p_j$, $L_j$, floating leg, truncated draw) | 🟢 Correct | Confirmed by direct reading of RiskFlow's source earlier in this review, including the exact formulae reproduced in §4.2–§4.5 |
| §3.7 — motivation via smooth sensitivities | 🟢 Correct | Consistent with the automatic-differentiation design evidence found directly in RiskFlow's source |
| §3.8 — RiskFlow's quanto treatment is analytic, not simulated | 🟢 Correct | Confirmed by direct reading of RiskFlow's source |
| Implicit throughout §3 — RiskFlow uses a single, unqualified volatility | 🟡 Correct as the default; incomplete | Correct for RiskFlow's default configuration, but omits that a Heston–Nandi stochastic-volatility spot model exists as a configurable alternative — see §4.10 of this document |
| Implicit throughout §1–§3 — dividend conventions are comparable across both models | 🔴 Not supported — material gap | RiskFlow uses a continuous yield only; the validation model uses discrete cash amounts. Section 4.11 sets out why this must be reconciled before any dividend-bearing trade result is interpreted |
| Not addressed in the source note — the terminal (downside) leg's treatment under OSS | 🔵 Needs direct confirmation | RiskFlow's own documentation confirms the leg's structural placement ($L_T\cdot V_{\text{terminal}}\cdot D_T$) but this review did not independently re-derive its analytic knock-in treatment — see §4.6 |

Two points follow from this table that bear directly on how Section 6's results should be read once populated. First, the coupon and floating leg mechanics in the supplied note are fully verified and no further work is needed to trust a like-for-like comparison of those two legs specifically, in the absence of dividends. Second, any comparison involving dividends or the terminal leg carries an open item recorded above and should not be treated as fully reconciled until that item is closed.

---

## 6. Validation Test Trades and Results

The supplied note sets out three trade groups of increasing complexity, intended to isolate the coupon/floating mechanics (single-coupon digitals, checked against the closed form of §3.4), the full multi-coupon swap mechanics (standard autocalls), and the quanto adjustment (quanto autocalls). **No trade parameters or results were populated in the source document; the tables below are reproduced as templates only, and no figures have been fabricated to fill them.** They should be completed from the actual test book before this section is treated as complete.

### 6.1 Test trades

*Single coupon digitals — checked against the closed form of §3.4.*

| Reference | Spot | Strike | Trigger | Coupon | Volatility | Rate | Dividend | Maturity |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

*Standard autocalls — multiple coupons and a floating leg.*

| Reference | Spot | Strike | Triggers | Coupons | Frequency | Floating | Volatility | Rate | Dividend | Maturity |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

*Quanto autocalls — as the standard set, priced under the adjustment of §2.6.*

| Reference | Spot | Strike | Triggers | Coupons | Frequency | Floating | Vol | FX vol | Corr. | Fixed FX | Rate | Div. | Maturity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

### 6.2 Results

For each trade the price produced by the validation model is set against the price produced by RiskFlow, with the relative difference reported. As with §6.1, these are unpopulated templates.

*Single coupon digitals.*

| Reference | Validation price | RiskFlow price | Relative difference |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

*Standard autocalls.*

| Reference | Validation price | RiskFlow price | Relative difference |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

*Quanto autocalls.*

| Reference | Validation price | RiskFlow price | Relative difference |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

---

## 7. Theory of the Prospective Stochastic-Volatility Extension

### 7.1 Status and motivation

This section documents a candidate replacement for the flat/term-structure geometric Brownian motion of §2.5, referred to here as the **stride model**, informally titled in its source material "a jump-friendly stochastic asset model built from the stride up." **It is prospective**: it is not implemented in either the validation codebase or, so far as this review has confirmed, in RiskFlow's default configuration (RiskFlow's own optional Heston–Nandi spot model, §4.10, is a related but distinct stochastic-volatility treatment and should not be conflated with the model documented here). Nothing in this section should be read as describing current behaviour of either pricer.

Every model reviewed so far treats volatility as an input: a number, or a term-structure curve, supplied to the diffusion. The stride model instead treats volatility as a second stochastic process, evolving alongside the underlying, mean-reverting at two distinct speeds, correlated with the underlying's own shocks, and topped up with a skewed, fat-tailed residual so that the return distribution is not constrained to be exactly lognormal. The model is built specifically so that, despite this added complexity, it remains compatible with the One-Step Survival architecture of Section 4 — §7.8 sets out why.

### 7.2 State variables

At any observation time the model carries three state values: spot, the current (instantaneous) variance, and a slow/long-run variance level. Positive variance quantities are easier to model in logs, so define

$$q_t = e^{\ell_t}, \qquad h_t = e^{\ell_t+s_t}$$

**Where:**
- $q_t$ is the slow, long-run variance level
- $h_t$ is the current (fast) variance level
- $\ell_t = \log q_t$ is the slow variance level, in log space
- $s_t = \log(h_t/q_t)$ is the temporary displacement of current variance away from its slow level; $s_t=0$ recovers $h_t = q_t$

### 7.3 Two-factor log-variance dynamics

The two log-coordinates are modelled with exact Ornstein–Uhlenbeck transitions over an internal step $\delta$, finer than the observation grid:

$$\ell_{k+1} = L(t_{k+1}) + \phi_\ell\big(\ell_k - L(t_k)\big) + w_\ell\,\eta_k^\ell, \qquad s_{k+1} = \phi_s\, s_k + w_s\,\eta_k^s$$

$$\phi_i = e^{-\kappa_i \delta}, \qquad w_i = \sigma_i\sqrt{\frac{1-\phi_i^2}{2\kappa_i}}, \qquad \eta^\ell,\eta^s \sim N(0,1)$$

**Where:**
- $L(t)$ is the deterministic mean curve that $\ell_t$ reverts toward, calibrated in §7.4
- $\kappa_\ell, \sigma_\ell$ are the mean-reversion speed and volatility of the persistent (slow) factor
- $\kappa_s, \sigma_s$ are the mean-reversion speed and volatility of the fast factor, with $\kappa_s \gg \kappa_\ell$ in typical calibrations
- $\delta$ is the internal (fine-grid) step length used to walk the variance state between observation dates

The state transition is exact for any $\delta$ — the only numerical discretisation in the full model is the integral of variance that the return law reads between observation dates (§7.6–§7.8), not the state transition itself.

### 7.4 The market-facing forward-variance curve and the Jensen correction

The deterministic curve to be calibrated is not $L(t)$ directly but the market-facing quantity $\xi(t) = \mathbb{E}_0[h_t]$, for the uncapped diffusion state. Because $\ell_t+s_t$ is Gaussian, $\mathbb{E}[e^X] = e^{\mathbb{E}[X]+\frac12\text{Var}(X)}$ gives, starting from $\ell_0 = L(0)$ and $s_0 = 0$:

$$V_{\ell+s}(t) = \frac{\sigma_\ell^2}{2\kappa_\ell}\big(1-e^{-2\kappa_\ell t}\big) + \frac{\sigma_s^2}{2\kappa_s}\big(1-e^{-2\kappa_s t}\big)$$

$$L(t) = \log \xi(t) - \tfrac12 V_{\ell+s}(t)$$

**Where:**
- $V_{\ell+s}(t)$ is the variance of $\ell_t+s_t$ accumulated by time $t$, in closed form from the OU dynamics of §7.3
- $L(t)$ is the internal mean curve that must be used so that the model's average variance matches the market curve $\xi(t)$ exactly

This is the **Jensen correction**: without it, the model would calibrate to the wrong variance level even were every other equation correct, since the mean of a log-normal variable is not the exponential of the mean of its log. After this correction, $\mathbb{E}_0[e^{\ell_t+s_t}] = \xi(t)$ holds exactly, before the structural cap of §7.5 is applied.

### 7.5 The structural cap

A lognormal variance state is positive by construction, but its extreme right tail is too wide for some moments of spot to remain numerically well behaved. The return therefore reads a capped variance,

$$\hat h_t = \exp\big(\text{cap}(\ell_t+s_t)\big), \qquad \text{cap}(x) = a - \beta_c\log\!\left(1+e^{(a-x)/\beta_c}\right)$$

**Where:**
- $\text{cap}(x)$ is a smooth function that equals $x$ in the normal region and approaches the level $a$ in the far tail
- $a, \beta_c$ are structural (not calibrated) constants controlling the cap level and its softness

The state itself is never capped; only the variance figure the return law consumes is. This distinction matters: there is no variance floor, mean reversion continues to act on the unconstrained state, the cap is invisible in the calibrated region, and it exists solely to keep high-order moments and floating-point behaviour finite. A calibration that spends meaningful probability mass near the cap should be treated as a failed calibration, not as evidence the cap itself needs adjusting.

### 7.6 The return law — leverage plus a skewed residual

Define the predictable per-step variance budget, read from the start-of-step state,

$$V_k = \delta\cdot\exp\big(\text{cap}(\ell_k+s_k)\big)$$

The two leverage pieces spend part of this budget on shocks correlated with the two variance factors:

$$L_k^\ell = \rho_\ell\sqrt{V_k}\,\eta_k^\ell - \tfrac12\rho_\ell^2 V_k, \qquad L_k^s = \rho_s\sqrt{V_k}\,\eta_k^s - \tfrac12\rho_s^2 V_k$$

**Where:**
- $\rho_\ell$ is the correlation spending the persistent-factor leverage, driving persistent forward skew
- $\rho_s$ is the correlation spending the fast-factor leverage, driving spot/vol leverage at short horizons

The correlations spend a fraction $\rho_\ell^2+\rho_s^2$ of the variance budget; the remainder, $c = 1-\rho_\ell^2-\rho_s^2 > 0$, is spent not on an independent Gaussian shock but on a Normal-Inverse-Gaussian (NIG) increment, giving the residual its own skew and tail thickness without giving up the exact composition property of §7.8. A Gaussian residual would be fast to simulate but would force the entire smile to be explained by the two leverage correlations alone; the NIG choice frees the residual to carry skew and convexity that the leverage terms are not required to reproduce.

### 7.7 The NIG residual and the martingale condition

Let $X_A \sim NIG(\alpha, \beta, \delta_A, \mu_A)$, with $\gamma = \sqrt{\alpha^2-\beta^2}$. The canonical NIG cumulant-generating function is

$$\log \mathbb{E}[e^{uX_A}] = \mu_A u + \delta_A\Big(\gamma - \sqrt{\alpha^2-(\beta+u)^2}\Big), \qquad \text{valid for } |\beta+u| < \alpha$$

with variance $\text{Var}(X_A) = \delta_A\alpha^2/\gamma^3$. Requiring the residual to consume exactly the prescribed budget $A$ gives

$$\delta_A = A\,\frac{\gamma^3}{\alpha^2}$$

Imposing the risk-neutral (martingale) condition $\mathbb{E}[e^{X_A}] = 1$ — setting $u=1$ in the cumulant and forcing it to zero — fixes the drift:

$$\mu_A = \delta_A\Big(\sqrt{\alpha^2-(\beta+1)^2} - \gamma\Big), \qquad \text{requiring } |\beta+1| < \alpha$$

**Where:**
- $\alpha$ is tail thickness / smile convexity of the residual
- $\beta$ is skew of the residual
- $A$ is the variance budget assigned to the residual over the interval in question
- $\mu_A$ is the residual's drift, which is **not** a free calibration parameter — it is forced entirely by no-arbitrage, together with the ordinary NIG admissibility condition $|\beta|<\alpha$

Together, $|\beta|<\alpha$ and $|\beta+1|<\alpha$ form the complete admissibility condition for a valid, risk-neutral residual. Because an optimiser fitting $(\alpha,\beta)$ must never be allowed to propose a point outside this region, an unconstrained-to-constrained reparameterisation is used in practice:

$$\alpha = \tfrac12+\varepsilon+\text{softplus}(a), \qquad \beta = -\tfrac12+\big(\alpha-\tfrac12-\varepsilon\big)\tanh(b)$$

for free reals $a, b$ and a small $\varepsilon>0$, which maps every real $(a,b)$ into the admissible region automatically.

### 7.8 Exact composition across an observation interval

Putting the state and return together, one internal step is

$$R_k = b_k\delta + \rho_\ell\sqrt{V_k}\,\eta_k^\ell - \tfrac12\rho_\ell^2 V_k + \rho_s\sqrt{V_k}\,\eta_k^s - \tfrac12\rho_s^2 V_k + X_{cV_k}$$

**Where:**
- $R_k$ is the log-return over the $k$-th internal step
- $b_k$ is the carry $r-q$ applicable over that step
- $X_{cV_k}$ is the NIG residual drawn for that step, consuming exactly the variance budget $c\cdot V_k$

Because $V_k$ is read from the start-of-step state, it is predictable, and the two leverage shocks and the NIG residual are independent conditional on that state; each of the three factors in $\mathbb{E}[e^{R_k}\mid\mathcal{F}_k]$ therefore has expectation one by construction (the leverage terms by their own $-\frac12\rho^2 V$ convexity correction, the residual by §7.7's martingale condition), giving $\mathbb{E}[S_{k+1}\mid\mathcal{F}_k] = S_k\cdot e^{b_k\delta}$ with no separate empirical drift correction required, and $\text{Var}(R_k\mid V_k) = V_k$ — the model changes the *shape* of the return law without changing the variance budget it spends.

Now suppose an observation interval spans internal steps $k \in B_j$. Accumulate

$$M_j^{\text{lev}} = \sum_{k\in B_j} \big[b_k\delta + L_k^\ell + L_k^s\big], \qquad A_j = \sum_{k\in B_j} c\cdot V_k$$

NIG increments are infinitely divisible: for fixed $(\alpha,\beta)$, $X_{A_1}+X_{A_2}$ has the same law as $X_{A_1+A_2}$, since $\delta_A$ and $\mu_A$ are both linear in $A$ and their cumulants therefore add exactly. Every residual return inside the block collapses to a single draw, $X_{A_j}$. Spot is not simulated on the internal grid at all; at the observation date the model needs only the triple $(M_j^{\text{lev}}, A_j, \ell_{\text{end}}, s_{\text{end}})$, after which one residual draw produces the new spot. **This is the central result that makes the model practical**: the volatility state is walked on a fine internal clock, but the spot process itself is touched exactly once per observation date, matching every other engine reviewed in this document.

### 7.9 Compatibility with One-Step Survival

The NIG distribution admits a normal-mixture representation,

$$G_A \sim IG\!\left(m=\frac{\delta_A}{\gamma},\ \lambda=\delta_A^2\right), \qquad X_A \mid G_A \sim N(\mu_A + \beta G_A,\ G_A)$$

**Where:**
- $IG(m,\lambda)$ is the inverse-Gaussian distribution in mean/shape parameterisation
- $G_A$ is a positive mixing variable, drawn once per block, before the final return shock

This is exactly the representation an OSS pricer needs. For a whole observation block, the recipe is: walk the two volatility states; accumulate $M_j^{\text{lev}}$ and $A_j$; sample one positive mixer $G_j$; the block return, conditional on $G_j$, is then Gaussian,

$$R_j \mid G_j \sim N(M_j, G_j), \qquad M_j = M_j^{\text{lev}} + \mu_{A_j} + \beta G_j$$

Everything non-Gaussian has been sampled before the final return shock, and the end-of-block volatility state is already known and therefore cannot be contaminated by a later, truncated spot draw. Consequently, **RiskFlow's own survival-probability formula (§4.2) requires no structural change to accommodate this model** — only that its constant $\sigma^2\Delta t$ term be replaced by the per-block, per-path pair $(M_j, G_j)$, with the truncated draw of §4.5 applied conditional on $G_j$ exactly as it is today conditional on a fixed variance.

### 7.10 Calibration parameters and their role

| Market / dynamic feature | Main parameter(s) |
|---|---|
| ATM term structure | $\xi(t)$ |
| Short-run variance movement | $\kappa_s, \sigma_s$ |
| Persistent variance movement | $\kappa_\ell, \sigma_\ell$ |
| Spot/vol leverage at short horizons | $\rho_s$ |
| Persistent forward skew | $\rho_\ell$ |
| Residual tail thickness / convexity | $\alpha$ |
| Residual skew | $\beta$ |
| Carry | market curves, not fitted here |

A sensible calibration philosophy follows directly from this table: build $\xi(t)$ from at-the-money or variance-swap information; fit $(\alpha,\beta)$ primarily to the wing shape of the smile; fit $(\rho_s,\sigma_s)$ to intermediate-tenor skew and short-horizon spot/vol dynamics; fit $(\rho_\ell,\sigma_\ell)$ to persistent, forward skew; treat the two $\kappa$'s as only weakly identified by vanilla option prices alone and constrain them with economically reasonable priors; and validate forward-start smiles as a separate step. The important consequence is that today's skew does not have to be manufactured entirely by leverage, which leaves the leverage parameters free to describe genuine future smile dynamics rather than being forced to double as a today-only fitting device.

This last point is specifically why forward skew is treated as its own calibration target rather than assumed to follow from a good spot-vanilla fit. A vanilla fit at today's date says nothing about the smile the model will produce at a future fixing date, and that future smile is precisely what governs the economics of a path-dependent product such as an autocall, whose remaining coupons and terminal leg depend on the conditional distribution of the underlying after the volatility state has already evolved. The persistent factor $\ell$ and the leverage pair $(\rho_\ell,\rho_s)$ are the natural levers for this; where the market requires additional calendar-dependent tail skew beyond what those levers provide, a small, bucketed $\beta(t) = \beta_1\cdot\mathbf{1}\{t<T^*\} + \beta_2\cdot\mathbf{1}\{t\ge T^*\}$ is a cleaner adjustment than altering the variance dynamics themselves. The governing rule is that **vanillas calibrate today's marginal distribution; forward-start options validate the model's dynamics** — the two are separate calibration exercises, and a good fit to the first does not imply a good fit to the second.

### 7.11 What integrating this model into the validation codebase would require

The payoff evaluator and schedule-building layers of the validation codebase are already decoupled from how the simulated spot path is generated, so neither requires modification. The changes required are concentrated in three areas.

- A **new market-data container** carrying the calibrated parameter set of §7.10 — the forward-variance curve and the seven scalar parameters — in place of the single flat volatility figure currently read by the existing engine.
- A **new engine class** implementing the per-block stride of §7.8–§7.9 as a drop-in replacement for the existing engine's per-step update, honouring the same `simulate(schedule, market_data)` contract so that every existing caller can switch engines with no other change. This is materially more code than any change reviewed elsewhere in this document, since it must walk two mean-reverting factors on a finer internal grid, accumulate the leverage mean and variance budget, and draw the inverse-Gaussian mixer, per observation date.
- An **explicit design decision** on how discrete cash dividends interact with a model in which spot is touched only once per observation date, since the existing engine currently applies a dividend wherever it lands on a finer internal simulation grid that will no longer exist in the same form.

The quanto adjustment ports mechanically, as one further additive term inside the same $M_j^{\text{lev}}$ accumulator already carrying the carry term. The existing engine's optional FX-simulating quanto audit (§3.5) would require materially more new development to reproduce under this model, since it would need to simulate FX correlated with two stochastic-volatility factors and a skewed residual rather than a single Brownian motion, and should be explicitly scoped as its own piece of work rather than assumed to follow automatically. Two closed-form checks are already available directly from the theory above and should be carried into the codebase as permanent regression tests rather than left as one-off demonstrations: the infinite-divisibility (semigroup) identity of §7.8, and the exact one-period European price obtainable by direct integration of the NIG residual under exponential tilting of $\beta$.

---

## 8. Recommendation and Next Steps

Four items follow directly from Sections 5–7 and are worth carrying forward as concrete actions rather than general observations.

1. **Close the dividend-convention gap (§4.11)** before any dividend-bearing trade in Section 6 is used to draw a conclusion about either model's correctness. This is the single highest-priority item, since it is the one gap capable of silently contaminating a result that would otherwise be a clean pass.
2. **Confirm RiskFlow's analytic treatment of the knock-in probability** within its terminal leg (§4.6) directly against source, so that the terminal-leg comparison in Section 6 can be verified to the same standard already achieved for the coupon and floating legs.
3. **Add the barrier-configuration guard** identified in §3.6 to the validation codebase — a small, low-risk change that removes a silent misconfiguration risk entirely.
4. **Treat the stride model of Section 7 as a scoped, standalone project** once calibrated parameters are available, following §7.11's breakdown of what is a mechanical port (the quanto adjustment) versus what is genuinely new development (the per-block engine, the dividend interaction, and the FX-simulating quanto audit), and carry forward the two closed-form checks identified there as permanent regression tests from the outset rather than as an afterthought.

---

## Appendix A. Notation

| Symbol | Meaning |
|---|---|
| $t_1 \dots t_n$ | observation dates |
| $S_j$ | underlying level at $t_j$ |
| $K$ | strike, being the initial reference level |
| $\theta_j$ | trigger multiplier at $t_j$, expressed as a fraction of strike |
| $H_j = \theta_j K$ | autocall trigger level |
| $H_{\text{put}}$ | downside knock-in barrier level |
| $c_j$ | coupon paid on trigger at $t_j$ |
| $F_k$ | floating payment at $t_k$ |
| $D_j$ | discount factor to $t_j$ |
| $Q_j$ | probability of survival beyond date $j$ (autocall leg) |
| $\tau$ | first trigger date |
| $N$ | notional, being buy or sell multiplied by units |
| $R$ | rebate / principal amount in the terminal leg |
| $V_{\text{terminal}}$ | the terminal (downside) leg payoff |
| $b = r-q$ | carry, being the repo rate less the dividend yield |
| $\sigma$ | volatility of the underlying (flat/term-structure models) |
| $\sigma_X$ | volatility of the exchange rate |
| $\rho_{SX}$ | correlation between the underlying and the exchange rate |
| $b_{\text{quanto}} = r-q-\rho_{SX}\sigma\sigma_X$ | quanto-adjusted carry |
| $p_j, L_j$ | RiskFlow's per-date survival probability and cumulative survival weight (§4.2–§4.3) |
| $\ell_t, s_t, h_t, q_t$ | stride model state: log slow variance, log displacement, current variance, slow variance (§7.2) |
| $\kappa_\ell, \sigma_\ell, \kappa_s, \sigma_s$ | stride model mean-reversion speeds and volatilities of the persistent and fast factors |
| $\rho_\ell, \rho_s$ | stride model leverage correlations |
| $\alpha, \beta$ | stride model NIG residual tail-thickness and skew parameters |
| $\xi(t)$ | stride model market-facing forward-variance curve |

---

## Appendix B. Proof that the OSS Estimator Is Unbiased

This proof, verified against the supplied note and consistent with the estimator confirmed directly against RiskFlow's own source in Section 4, shows that the One-Step Survival estimator of §4.7 has expectation equal to the true value of the swap component.

Write $\mathbb{E}^{\mathbb{Q}}$ for expectation under the risk-neutral law, in which the underlying follows the unconditional geometric Brownian motion of §2.5, and write $\tilde{\mathbb{E}}$ for expectation under the simulated law, in which the level at each observation date is drawn conditioned on survival, meaning conditioned on $S_k < H_k$. Denote by $f_k(\,\cdot \mid S_{k-1})$ the unconditional transition density from date $k-1$ to date $k$, and recall that

$$p_k(S_{k-1}) = \mathbb{Q}\big(S_k < H_k \mid S_{k-1}\big) = \int_{s < H_k} f_k(s \mid S_{k-1})\, ds$$

The conditioned transition density is the unconditional density restricted to the survival region and renormalised,

$$\tilde f_k(s \mid S_{k-1}) = \frac{f_k(s \mid S_{k-1})\, \mathbf{1}\{s < H_k\}}{p_k(S_{k-1})}$$

### The coupon leg

The true value of the coupon paid at date $j$ is $c_j D_j\, \mathbb{Q}(\tau = j)$, the first trigger occurring at date $j$ precisely when the trade survives every earlier date and triggers at $j$. Conditioning on the history to date $j-1$ and using the Markov property,

$$\mathbb{E}^{\mathbb{Q}}\big[\mathbf{1}\{\tau = j\}\big] = \mathbb{E}^{\mathbb{Q}}\Big[\mathbf{1}\{S_1<H_1,\dots,S_{j-1}<H_{j-1}\}\;\, \mathbb{Q}\big(S_j \ge H_j \mid S_{j-1}\big)\Big] = \mathbb{E}^{\mathbb{Q}}\Big[\mathbf{1}\{\text{survived to } j-1\}\,(1 - p_j)\Big]$$

The trigger indicator has been replaced by its conditional probability $1-p_j$, which is exact. It remains to express the surviving expectation under the simulated law. The joint unconditional density over the survival region factorises as

$$\prod_{k=1}^{j-1} f_k(S_k \mid S_{k-1})\,\mathbf{1}\{S_k < H_k\} = \left(\prod_{k=1}^{j-1} p_k(S_{k-1})\right) \prod_{k=1}^{j-1} \tilde f_k(S_k \mid S_{k-1}) = L_{j-1}\, \prod_{k=1}^{j-1}\tilde f_k(S_k \mid S_{k-1})$$

where $L_{j-1} = \prod_{k<j} p_k$ is the survival weight of §4.3, evaluated along the path. For any function $h$ of the surviving level it follows that

$$\mathbb{E}^{\mathbb{Q}}\big[\mathbf{1}\{\text{survived to } j-1\}\, h(S_{j-1})\big] = \tilde{\mathbb{E}}\big[L_{j-1}\, h(S_{j-1})\big]$$

Taking $h(S_{j-1}) = (1 - p_j)\, c_j D_j$ and summing over the coupon dates gives

$$\sum_j c_j D_j\, \mathbb{Q}(\tau = j) = \tilde{\mathbb{E}}\!\left[\sum_j (1 - p_j)\, L_{j-1}\, c_j D_j\right]$$

which is the coupon leg of the estimator.

### The floating leg

The floating payment at date $k$ is made whenever the trade is outstanding on entry to the date. By the same factorisation, with $h$ the constant $F_k D_k$,

$$\mathbb{E}^{\mathbb{Q}}\big[\mathbf{1}\{\text{outstanding at } t_k\}\, F_k D_k\big] = \tilde{\mathbb{E}}\big[L_{k^{-}}\, F_k D_k\big]$$

which is the floating leg of the estimator.

### Conclusion

Adding the two legs and multiplying by the notional, the expectation of the estimator under the simulated law equals the true value

$$N\,\mathbb{E}^{\mathbb{Q}}\!\left[c_\tau D_\tau \mathbf{1}\{\tau < \infty\} - \sum_k \mathbf{1}\{\text{outstanding at } t_k\}\, F_k D_k\right]$$

The estimator is therefore unbiased for the coupon and floating legs. The replacement of the trigger indicator by its conditional probability, which yields the factor $1-p_j$, is the source of the reduction in variance, since a quantity known in closed form is substituted for a simulated zero-or-one outcome.

This argument specialises to the coupon and floating legs the change of measure introduced for barrier option simulation by Glasserman and Staum (2001), cited in full in §4.1. As noted in §4.6, the equivalent proof for the terminal leg's treatment of the knock-in barrier was not independently re-derived in this review and should be confirmed directly against RiskFlow's source before being relied upon to the same standard as the coupon and floating legs above.
