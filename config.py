
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SamplingConvention(str, Enum):
    DAILY = "daily"
    BULLET = "bullet"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass(frozen=True)
class SimulationConfig:
    """
    Core simulation controls.

    Notes on RiskFlow mimicry:
    - RiskFlow uses scrambled Sobol + inverse-normal (erfinv) with a seed.
    - Time grid is expressed in "days", then converted to year-fractions.
    - CS forward model uses per-step variance increments (OU-style) and exponential
      decay of vol with time-to-maturity (Samuelson effect).
    """
    num_sims: int = 50_000
    seed: int = 1
    fast_forward: int = 0  # RiskFlow fast-forwards Sobol streams for multi-job partitioning.

    dt_days: int = 1       # scenario step size in days (e.g., 1 or 3)
    horizon_days: int = 365

    days_in_year: float = 365.0  # RiskFlow uses utils.DAYS_IN_YEAR (often 365 in its codebase)


@dataclass(frozen=True)
class CounterpartyConfig:
    """
    Simple deterministic credit curve for CVA.

    - hazard_rate: flat hazard (per year)
    - recovery: R, so LGD = 1 - R
    """
    hazard_rate: float
    recovery: float = 0.4


@dataclass(frozen=True)
class DiscountingConfig:
    """
    Flat discount rate for simplicity; swap this out for a curve object if needed.
    """
    rate: float  # continuously-compounded
    collateral_rate: Optional[float] = None  # placeholder for CSA/OIS style setups
