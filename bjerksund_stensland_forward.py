# Bjerksund-Stensland American option pricer (forward/Black-76 framing)
# One class, floats only, no NumPy/torch/dataclass.

import math
import datetime as _dt
from typing import Optional, List, Tuple, Literal, Dict

import pandas as _pd
from workalendar.africa import SouthAfrica as _SouthAfrica

OptionType = Literal["call", "put"]


class BjerksundStenslandOptionPricer:
    """Bjerksund-Stensland (single-boundary) American option pricer in
    forward (Black-76) form.

    Core intuition (aligned with RiskFlow):
      - Use FORWARD for the European piece (Black-76), discount by exp(-r*T).
      - Cost-of-carry b := ln(F/S)/T (so F = S * exp(b T)).
      - American exercise relevant if b < r; else price = European Black-76.
      - Call formula uses BS boundary I(T). Put priced by call-put transform.

    Simple API (existing, unchanged)
    ---------------------------------
    You can pass F directly, or derive F from:
      - continuous dividend yield q (F = S * exp((r - q)T)), or
      - discrete dividends [(t_i, D_i)] (F = (S - D_i e^{-r t_i}) e^{rT}).
      If multiple are provided: F > q > dividends.

    Curve-based API (new)
    ----------------------
    price_from_curves() and greeks_from_curves() accept the same
    date/curve/spot-day inputs as AmericanFDMPricer:
      - discount_curve  : pd.DataFrame with columns ["Date" (ISO), "NACA"]
      - forward_curve   : same format; falls back to discount_curve if None
      - dividend_schedule : List[Tuple[dt.date, float]] - (ex-date, cash)
      - underlying_spot_days : int - business-day lag for equity settlement
      - option_days          : int - business-day lag for option spot start
      - option_settlement_days : int - business-day lag for option expiry
      - day_count            : str - e.g. "ACT/365"

    Time decomposition in the Black-76/B-S formula:
      - time_to_carry   - scales the carry rate to produce F
      - time_to_expiry  - scales sigma in d1/d2 (sigma * sqrt(T_exp))
      - time_to_discount - scales the discount rate to produce the df
    """

    # ------------------------------------------------------------------ #
    #  Simple public API (existing, unchanged)
    # ------------------------------------------------------------------ #

    def price(self,
              S: float,
              K: float,
              T: float,
              r: float,
              sigma: float,
              option_type: OptionType = "call",
              F: Optional[float] = None,
              q: Optional[float] = None,
              dividends: Optional[List[Tuple[float, float]]] = None,
              ) -> Dict[str, float]:
        """Return {"price":..., "I": boundary_or_0.0, "early_exercise":...}.

        Parameters mirror the original simple API (all floats, no curves).
        """
        if T <= 0.0:
            intrinsic = max(
                0.0, (S - K) if option_type == "call" else (K - S)
            )
            return {"price": intrinsic, "I": 0.0, "early_exercise": 0.0}

        F_eff = self._resolve_forward(S, r, T, F, q, dividends)

        # cost-of-carry from forward (RiskFlow-style)
        b = math.log(max(F_eff, 1e-15) / max(S, 1e-15)) / T

        if option_type == "call":
            price, I, exercised = self._american_call_price(
                S, K, r, b, sigma, T
            )
            return {
                "price": price,
                "I": I,
                "early_exercise": 1.0 if exercised else 0.0,
            }
        else:
            # Put via call-put transform in forward frame:
            #   S* = K, K* = S, r* = r - b, b* = -b
            price_star, I_star, exercised = self._american_call_price(
                S=K, K=S, r=r - b, b=-b, sigma=sigma, T=T
            )
            return {
                "price": price_star,
                "I": I_star,
                "early_exercise": 1.0 if exercised else 0.0,
            }

    def greeks(self,
               S: float,
               K: float,
               T: float,
               r: float,
               sigma: float,
               option_type: OptionType = "call",
               F: Optional[float] = None,
               q: Optional[float] = None,
               dividends: Optional[List[Tuple[float, float]]] = None,
               dS: float = 1e-4,
               dSigma: float = 1e-4,
               dR: float = 1e-6,
               ) -> Dict[str, float]:
        """Finite-difference greeks (robust). Spot-delta/gamma computed
        holding carry b fixed."""
        # Base
        F_eff = self._resolve_forward(S, r, T, F, q, dividends)
        base = self.price(S, K, T, r, sigma, option_type, F_eff)["price"]

        # Keep b fixed (so update F when bumping S): b = ln(F/S)/T
        b = math.log(max(F_eff, 1e-15) / max(S, 1e-15)) / max(T, 1e-12)
        S_up, S_dn = S * (1.0 + dS), S * (1.0 - dS)
        F_up = S_up * math.exp(b * T)
        F_dn = S_dn * math.exp(b * T)

        p_up = self.price(S_up, K, T, r, sigma, option_type, F_up)["price"]
        p_dn = self.price(S_dn, K, T, r, sigma, option_type, F_dn)["price"]
        delta = (p_up - p_dn) / (S_up - S_dn)
        gamma = (p_up - 2.0 * base + p_dn) / (
            (S_up - S) * (S - S_dn) + 1e-18
        )

        # Vega
        p_vs_up = self.price(
            S, K, T, r, sigma * (1.0 + dSigma), option_type, F_eff
        )["price"]
        p_vs_dn = self.price(
            S, K, T, r, sigma * (1.0 - dSigma), option_type, F_eff
        )["price"]
        vega = (p_vs_up - p_vs_dn) / (2.0 * sigma * dSigma + 1e-18)

        # Rho
        p_r_up = self.price(S, K, T, r + dR, sigma, option_type, F_eff)[
            "price"
        ]
        p_r_dn = self.price(S, K, T, r - dR, sigma, option_type, F_eff)[
            "price"
        ]
        rho = (p_r_up - p_r_dn) / (2.0 * dR)

        return {"delta": delta, "gamma": gamma, "vega": vega, "rho": rho}

    # ------------------------------------------------------------------ #
    #  Curve-based public API (new)
    # ------------------------------------------------------------------ #

    def price_from_curves(
        self,
        S: float,
        K: float,
        valuation_date: _dt.date,
        maturity_date: _dt.date,
        sigma: float,
        option_type: OptionType = "call",
        discount_curve: Optional[_pd.DataFrame] = None,
        forward_curve: Optional[_pd.DataFrame] = None,
        dividend_schedule: Optional[List[Tuple[_dt.date, float]]] = None,
        underlying_spot_days: int = 0,
        option_days: int = 0,
        option_settlement_days: int = 0,
        day_count: str = "ACT/365",
    ) -> Dict[str, float]:
        """Price using explicit yield curves, dividend schedule, and spot
        day conventions (mirrors AmericanFDMPricer input layout).

        Parameters
        ----------
        S  : spot price of the underlying.
        K  : strike price.
        valuation_date : pricing date.
        maturity_date  : option expiry date.
        sigma          : annualised volatility.
        option_type    : "call" or "put".
        discount_curve : DataFrame["Date" (ISO YYYY-MM-DD), "NACA"]. Used to
            compute the option discount factor.  Required.
        forward_curve  : same format; drives the equity carry rate.
            Falls back to discount_curve when None.
        dividend_schedule : List[(ex_date: dt.date, cash_amount: float)].
            Cash dividends between valuation and maturity are PV-ed and
            subtracted from spot before computing the forward.
        underlying_spot_days     : business-day lag for equity spot start.
        option_days              : business-day lag for option spot start.
        option_settlement_days   : business-day lag for option delivery.
        day_count  : e.g. "ACT/365".

        Returns
        -------
        Dict with keys: price, I, early_exercise, T_exp, T_carry, T_disc,
        carry_rate, disc_rate, F_eff, b.
        """
        if maturity_date <= valuation_date:
            intrinsic = max(
                0.0, (S - K) if option_type == "call" else (K - S)
            )
            return {
                "price": intrinsic, "I": 0.0, "early_exercise": 0.0,
                "T_exp": 0.0, "T_carry": 0.0, "T_disc": 0.0,
                "carry_rate": 0.0, "disc_rate": 0.0, "F_eff": S, "b": 0.0,
            }

        res = self._resolve_curve_inputs(
            S, valuation_date, maturity_date,
            discount_curve, forward_curve, dividend_schedule,
            underlying_spot_days, option_days, option_settlement_days,
            day_count,
        )

        T_exp = res["T_exp"]
        F_eff = res["F_eff"]
        df = res["df"]
        b = res["b"]
        r_disc = res["disc_rate"]
        T_disc = res["T_disc"]

        if option_type == "call":
            price, I, exercised = self._american_call_price(
                S, K, r_disc, b, sigma, T_exp,
                F_eff_in=F_eff, df_in=df,
            )
        else:
            # Put via exact call-put transform with curve-resolved F and df.
            # Transform: S*=K, K*=S, b*=-b, r*=r_disc-b
            #   F*  = K * S / F_eff  (= K * exp(-b * T_exp))
            #   df* = exp(-r* * T_disc)  [exact, uses T_disc not T_exp]
            b_put = -b
            r_put = r_disc - b
            F_put = K * S / max(F_eff, 1e-15)
            df_put = math.exp(-r_put * T_disc)
            price, I, exercised = self._american_call_price(
                K, S, r_put, b_put, sigma, T_exp,
                F_eff_in=F_put, df_in=df_put,
            )

        return {
            "price": price,
            "I": I,
            "early_exercise": 1.0 if exercised else 0.0,
            "T_exp": T_exp,
            "T_carry": res["T_carry"],
            "T_disc": T_disc,
            "carry_rate": res["carry_rate"],
            "disc_rate": r_disc,
            "F_eff": F_eff,
            "b": b,
        }

    def greeks_from_curves(
        self,
        S: float,
        K: float,
        valuation_date: _dt.date,
        maturity_date: _dt.date,
        sigma: float,
        option_type: OptionType = "call",
        discount_curve: Optional[_pd.DataFrame] = None,
        forward_curve: Optional[_pd.DataFrame] = None,
        dividend_schedule: Optional[List[Tuple[_dt.date, float]]] = None,
        underlying_spot_days: int = 0,
        option_days: int = 0,
        option_settlement_days: int = 0,
        day_count: str = "ACT/365",
        dS: float = 1e-4,
        dSigma: float = 1e-4,
    ) -> Dict[str, float]:
        """Finite-difference greeks using curve-resolved inputs.

        Delta / Gamma : symmetric spot bump holding carry_rate and T_carry
            fixed (F scales proportionally with S).
        Vega          : symmetric sigma bump holding F and df fixed.

        Parameters mirror price_from_curves(); dS and dSigma are relative
        bump sizes (defaults 1e-4).
        """
        res = self._resolve_curve_inputs(
            S, valuation_date, maturity_date,
            discount_curve, forward_curve, dividend_schedule,
            underlying_spot_days, option_days, option_settlement_days,
            day_count,
        )

        T_exp = res["T_exp"]
        T_carry = res["T_carry"]
        T_disc = res["T_disc"]
        carry_rate = res["carry_rate"]
        r_disc = res["disc_rate"]
        F_eff = res["F_eff"]
        df = res["df"]
        b = res["b"]

        # Base price (call it directly to avoid re-resolving curve)
        base = self.price_from_curves(
            S, K, valuation_date, maturity_date, sigma, option_type,
            discount_curve, forward_curve, dividend_schedule,
            underlying_spot_days, option_days, option_settlement_days,
            day_count,
        )["price"]

        # --- Delta / Gamma: bump S, keep carry_rate and T_carry fixed ---
        # b = carry_rate * T_carry / T_exp is constant when S changes,
        # so b is the same for up/dn bumps.
        S_up = S * (1.0 + dS)
        S_dn = S * (1.0 - dS)
        F_up = S_up * math.exp(carry_rate * T_carry)
        F_dn = S_dn * math.exp(carry_rate * T_carry)

        if option_type == "call":
            p_up = self._american_call_price(
                S_up, K, r_disc, b, sigma, T_exp,
                F_eff_in=F_up, df_in=df,
            )[0]
            p_dn = self._american_call_price(
                S_dn, K, r_disc, b, sigma, T_exp,
                F_eff_in=F_dn, df_in=df,
            )[0]
        else:
            b_put = -b
            r_put = r_disc - b
            df_put = math.exp(-r_put * T_disc)
            # F_put scales with S: F_put = K * S / F_eff => proportional
            F_put_up = K * S_up / max(F_up, 1e-15)
            F_put_dn = K * S_dn / max(F_dn, 1e-15)
            p_up = self._american_call_price(
                K, S_up, r_put, b_put, sigma, T_exp,
                F_eff_in=F_put_up, df_in=df_put,
            )[0]
            p_dn = self._american_call_price(
                K, S_dn, r_put, b_put, sigma, T_exp,
                F_eff_in=F_put_dn, df_in=df_put,
            )[0]

        delta = (p_up - p_dn) / (S_up - S_dn)
        gamma = (p_up - 2.0 * base + p_dn) / (
            (S_up - S) * (S - S_dn) + 1e-18
        )

        # --- Vega: bump sigma, keep F and df fixed ---
        sigma_up = sigma * (1.0 + dSigma)
        sigma_dn = sigma * (1.0 - dSigma)

        if option_type == "call":
            p_vs_up = self._american_call_price(
                S, K, r_disc, b, sigma_up, T_exp,
                F_eff_in=F_eff, df_in=df,
            )[0]
            p_vs_dn = self._american_call_price(
                S, K, r_disc, b, sigma_dn, T_exp,
                F_eff_in=F_eff, df_in=df,
            )[0]
        else:
            b_put = -b
            r_put = r_disc - b
            F_put = K * S / max(F_eff, 1e-15)
            df_put = math.exp(-r_put * T_disc)
            p_vs_up = self._american_call_price(
                K, S, r_put, b_put, sigma_up, T_exp,
                F_eff_in=F_put, df_in=df_put,
            )[0]
            p_vs_dn = self._american_call_price(
                K, S, r_put, b_put, sigma_dn, T_exp,
                F_eff_in=F_put, df_in=df_put,
            )[0]

        vega = (p_vs_up - p_vs_dn) / (2.0 * sigma * dSigma + 1e-18)

        return {"delta": delta, "gamma": gamma, "vega": vega}

    # ------------------------------------------------------------------ #
    #  Curve infrastructure (new)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _infer_denominator(day_count: str) -> int:
        """Return the year denominator for the given day-count convention."""
        if day_count in ("ACT/365", "ACT/365F"):
            return 365
        if day_count in ("ACT/360", "ACT/364"):
            return 360 if day_count == "ACT/360" else 364
        if day_count in ("30/360", "BOND", "US30/360"):
            return 360
        return 365

    @staticmethod
    def _year_fraction(
        start_date: _dt.date,
        end_date: _dt.date,
        day_count: str,
    ) -> float:
        """Return the year fraction between two dates for a given
        day-count convention."""
        if end_date <= start_date:
            return 0.0
        dc = day_count.upper()
        if dc in ("ACT/365", "ACT/365F", "ACT/360", "ACT/364"):
            denom = BjerksundStenslandOptionPricer._infer_denominator(dc)
            return (end_date - start_date).days / float(denom)
        if dc in ("30/360", "BOND", "US30/360"):
            y1, m1, d1 = start_date.year, start_date.month, start_date.day
            y2, m2, d2 = end_date.year, end_date.month, end_date.day
            d1 = min(d1, 30)
            if d1 == 30:
                d2 = min(d2, 30)
            days = (y2 - y1) * 360 + (m2 - m1) * 30 + (d2 - d1)
            return days / 360.0
        return (end_date - start_date).days / 365.0

    @staticmethod
    def _get_df_from_curve(
        curve_df: _pd.DataFrame,
        val_date: _dt.date,
        lookup_date: _dt.date,
        day_count: str,
    ) -> float:
        """Return the discount factor from curve_df for lookup_date,
        compounded from val_date.

        Mirrors AmericanFDMPricer.get_discount_factor():
          - Looks up the date by exact ISO match ("YYYY-MM-DD") in the
            "Date" column.
          - Reads the "NACA" rate and returns (1 + NACA)^(-tau).
          - Raises ValueError if the date is not found in the curve.
          - Returns 1.0 immediately if lookup_date <= val_date (tau = 0).
        """
        tau = BjerksundStenslandOptionPricer._year_fraction(
            val_date, lookup_date, day_count
        )
        if tau <= 0.0:
            return 1.0
        iso = lookup_date.isoformat()
        row = curve_df[curve_df["Date"] == iso]
        if row.empty:
            raise ValueError(
                f"Date '{iso}' not found in curve. "
                "Ensure the curve DataFrame covers all required dates."
            )
        naca = float(row["NACA"].values[0])
        return (1.0 + naca) ** (-tau)

    @staticmethod
    def _get_nacc_rate_from_curve(
        curve_df: _pd.DataFrame,
        val_date: _dt.date,
        start_date: _dt.date,
        end_date: _dt.date,
        day_count: str,
    ) -> float:
        """Return the continuously compounded (NACC) forward rate between
        start_date and end_date, implied from curve_df.

        Mirrors AmericanFDMPricer.get_forward_nacc_rate():
          r_nacc = -ln(df_far / df_near) / tau
        where df_near and df_far are discount factors from val_date to
        start_date and end_date respectively.
        """
        if start_date >= end_date:
            return 0.0
        df_far = BjerksundStenslandOptionPricer._get_df_from_curve(
            curve_df, val_date, end_date, day_count
        )
        df_near = BjerksundStenslandOptionPricer._get_df_from_curve(
            curve_df, val_date, start_date, day_count
        )
        tau = BjerksundStenslandOptionPricer._year_fraction(
            start_date, end_date, day_count
        )
        return -math.log(df_far / max(df_near, 1e-15)) / max(tau, 1e-12)

    def _resolve_curve_inputs(
        self,
        S: float,
        val_date: _dt.date,
        mat_date: _dt.date,
        discount_curve: Optional[_pd.DataFrame],
        forward_curve: Optional[_pd.DataFrame],
        div_schedule: Optional[List[Tuple[_dt.date, float]]],
        underlying_spot_days: int,
        option_days: int,
        option_settlement_days: int,
        day_count: str,
    ) -> Dict:
        """Resolve all time fractions, rates, forward price, and discount
        factor from curve-based inputs.

        Date rolling uses the South Africa business-day calendar
        (consistent with AmericanFDMPricer).

        Time decomposition
        ------------------
        carry_start  = val_date  + underlying_spot_days (biz days)
        carry_end    = mat_date  + underlying_spot_days (biz days)
        disc_start   = val_date  + option_days          (biz days)
        disc_end     = mat_date  + option_settlement_days (biz days)

        T_exp    = year_frac(val_date, mat_date)    -- sigma scaling
        T_carry  = year_frac(carry_start, carry_end) -- forward carry
        T_disc   = year_frac(disc_start, disc_end)   -- discount factor

        Forward:  F = (S - PV_divs) * exp(carry_rate * T_carry)
        DF:       df = exp(-disc_rate * T_disc)
        b:        ln(F / S) / T_exp   (cost-of-carry for B-S formulas)

        Returns a dict with all resolved quantities.
        """
        if discount_curve is None:
            raise ValueError(
                "discount_curve is required for price_from_curves() / "
                "greeks_from_curves()."
            )

        dc = day_count.upper()
        calendar = _SouthAfrica()

        # --- Business-day date rolling ---
        carry_start = calendar.add_working_days(val_date, underlying_spot_days)
        carry_end = calendar.add_working_days(mat_date, underlying_spot_days)
        disc_start = calendar.add_working_days(val_date, option_days)
        disc_end = calendar.add_working_days(mat_date, option_settlement_days)

        # --- Time fractions ---
        T_exp = self._year_fraction(val_date, mat_date, dc)
        T_carry = self._year_fraction(carry_start, carry_end, dc)
        T_disc = self._year_fraction(disc_start, disc_end, dc)

        if T_exp <= 0.0:
            raise ValueError("maturity_date must be strictly after valuation_date.")

        # --- Rates (NACC) from curves ---
        disc_rate = self._get_nacc_rate_from_curve(
            discount_curve, val_date, disc_start, disc_end, dc
        )

        carry_curve = forward_curve if forward_curve is not None else discount_curve
        carry_rate = self._get_nacc_rate_from_curve(
            carry_curve, val_date, carry_start, carry_end, dc
        )

        # --- Discrete dividends: PV subtracted from spot ---
        pv_divs = 0.0
        if div_schedule:
            for div_date, div_amount in div_schedule:
                if val_date < div_date <= mat_date and div_amount != 0.0:
                    df_div = self._get_df_from_curve(
                        discount_curve, val_date, div_date, dc
                    )
                    pv_divs += float(div_amount) * df_div

        S_pseudo = max(S - pv_divs, 1e-15)

        # --- Forward and discount factor ---
        F_eff = S_pseudo * math.exp(carry_rate * T_carry)
        df = math.exp(-disc_rate * T_disc)

        # --- Cost-of-carry: b = ln(F/S) / T_exp ---
        b = math.log(max(F_eff, 1e-15) / max(S, 1e-15)) / max(T_exp, 1e-12)

        return {
            "T_exp": T_exp,
            "T_carry": T_carry,
            "T_disc": T_disc,
            "carry_rate": carry_rate,
            "disc_rate": disc_rate,
            "F_eff": F_eff,
            "df": df,
            "b": b,
            "carry_start": carry_start,
            "carry_end": carry_end,
            "disc_start": disc_start,
            "disc_end": disc_end,
            "pv_divs": pv_divs,
        }

    # ------------------------------------------------------------------ #
    #  Internals (existing helpers, unchanged)
    # ------------------------------------------------------------------ #

    def _resolve_forward(
        self,
        S: float,
        r: float,
        T: float,
        F: Optional[float],
        q: Optional[float],
        dividends: Optional[List[Tuple[float, float]]],
    ) -> float:
        """Pick forward by priority: F > q > dividends > no divs."""
        if F is not None:
            return float(F)
        if q is not None:
            return S * math.exp((r - float(q)) * T)
        if dividends:
            pv = 0.0
            for (ti, Di) in dividends:
                if 0.0 < ti <= T and Di != 0.0:
                    pv += float(Di) * math.exp(-r * float(ti))
            return (S - pv) * math.exp(r * T)
        return S * math.exp(r * T)

    @staticmethod
    def _ncdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def _black76(
        forward: float,
        strike: float,
        sigma: float,
        T: float,
        df: float,
        call: bool,
    ) -> float:
        if T <= 0.0 or sigma <= 0.0:
            intrinsic = max(
                0.0,
                (forward - strike) if call else (strike - forward),
            )
            return df * intrinsic
        srt = math.sqrt(T)
        d1 = (math.log(forward / strike) + 0.5 * sigma * sigma * T) / (
            sigma * srt
        )
        d2 = d1 - sigma * srt
        N1 = BjerksundStenslandOptionPricer._ncdf(d1)
        N2 = BjerksundStenslandOptionPricer._ncdf(d2)
        if call:
            return df * (forward * N1 - strike * N2)
        else:
            return df * (strike * (1.0 - N2) - forward * (1.0 - N1))

    @staticmethod
    def _phi(
        gamma: float,
        H: float,
        I: float,
        S_for_phi: float,
        T: float,
        volT: float,
        b_safe: float,
        sigma2: float,
        r_safe: float,
    ) -> float:
        # Mirrors the RiskFlow torch structure (clamps, ordering)
        # volT := sigma * sqrt(T)
        eps = 1e-32
        sigma2 = max(sigma2, eps)
        volT = max(volT, eps)
        H_ = max(H, eps)
        I_ = max(I, eps)
        S_ = max(S_for_phi, eps)

        kappa = (2.0 * b_safe) / sigma2 + 2.0 * gamma - 1.0
        d = (
            math.log(H_ / S_) - (b_safe + (gamma - 0.5) * sigma2) * T
        ) / volT
        lam = -r_safe + gamma * b_safe + 0.5 * gamma * (gamma - 1.0) * sigma2
        log_IS = math.log(I_ / S_)
        safe_exp = min(kappa * log_IS, 25.0)

        term1 = BjerksundStenslandOptionPricer._ncdf(d)
        term2 = math.exp(safe_exp) * BjerksundStenslandOptionPricer._ncdf(
            d - 2.0 * log_IS / volT
        )
        return math.exp(lam * T) * (term1 - term2)

    def _american_call_price(
        self,
        S: float,
        K: float,
        r: float,
        b: float,
        sigma: float,
        T: float,
        F_eff_in: Optional[float] = None,
        df_in: Optional[float] = None,
    ) -> Tuple[float, float, bool]:
        """RiskFlow-style B-S American call: returns (price, I, exercised).

        F_eff_in and df_in are optional overrides for the forward price and
        discount factor respectively.  When provided they are used instead
        of the internally derived F = S*exp(b*T) and df = exp(-r*T).
        This allows price_from_curves() and greeks_from_curves() to pass in
        the curve-resolved values (with potentially different T_carry and
        T_disc from T_exp) without altering any other part of the formula.
        """
        eps = 1e-16
        T = max(T, 1e-5)
        sigma = float(sigma)
        sigma2 = sigma * sigma
        volT = sigma * math.sqrt(T)

        # European Black-76 on forward
        # F and df may be overridden by caller (curve-based API).
        F = F_eff_in if F_eff_in is not None else S * math.exp(b * T)
        df = df_in if df_in is not None else math.exp(-r * T)
        euro = self._black76(F, K, sigma, T, df, call=True)

        # American trigger
        if not (b < r - 1e-6):
            return (euro, 0.0, False)

        # beta, B0, Binf, I (with RF-style guards)
        b_over_sigma2 = b / max(sigma2, eps)
        safe_sqrt = (b_over_sigma2 - 0.5) ** 2 + 2.0 * r / max(sigma2, eps)
        safe_sqrt = max(safe_sqrt, 1e-6)
        beta = (0.5 - b_over_sigma2) + math.sqrt(safe_sqrt)

        r_b = r - b
        B0 = K * max(r / max(r_b, eps), 1.0)
        denom_beta = beta - 1.0
        if abs(denom_beta) < 1e-12:
            denom_beta = 1e-12 if denom_beta >= 0 else -1e-12
        Binf = K * beta / denom_beta

        denom_B = Binf - B0
        if abs(denom_B) < 1e-12:
            denom_B = 1e-12
        h_tau = -(b * T + 2.0 * volT) * (B0 / denom_B)
        I = B0 + (Binf - B0) * (1.0 - math.exp(h_tau))

        # Safe argument for phi must be strictly below I
        S_for_phi = min(max(S, eps) - 1e-6, I)

        # phi-terms
        phi_beta_II = self._phi(beta, I, I, S_for_phi, T, volT, b, sigma2, r)
        phi_1_II = self._phi(1.0, I, I, S_for_phi, T, volT, b, sigma2, r)
        phi_1_KI = self._phi(1.0, K, I, S_for_phi, T, volT, b, sigma2, r)
        phi_0_KI = self._phi(0.0, K, I, S_for_phi, T, volT, b, sigma2, r)
        phi_0_II = self._phi(0.0, I, I, S_for_phi, T, volT, b, sigma2, r)

        # Bjerksund-Stensland composition
        if K <= 0.0:
            c_bs = B0
        else:
            log_ratio = math.log(max(S_for_phi, eps) / max(I, eps))
            core = (I - K) * math.exp(beta * log_ratio) * (1.0 - phi_beta_II)
            c_bs = (
                core
                + S_for_phi * (phi_1_II - phi_1_KI)
                + K * (phi_0_KI - phi_0_II)
            )

        c_bs = max(euro, c_bs)

        # Exercise-now region
        if S >= I:
            return (S - K, I, True)
        else:
            return (c_bs, I, False)
