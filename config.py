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

    Notes
    -----
    - RiskFlow-like: Sobol normals, scenario grid in days.
    - days_in_year used for year-fraction conversion.
    """
    num_sims: int = 50_000
    seed: int = 1
    fast_forward: int = 0

    dt_days: int = 1
    horizon_days: int = 365

    days_in_year: float = 365.0


@dataclass(frozen=True)
class CounterpartyConfig:
    """
    Deterministic credit curve for CVA:
    - hazard_rate: flat hazard (per year)
    - recovery: R, so LGD = 1 - R
    """
    hazard_rate: float
    recovery: float = 0.4


@dataclass(frozen=True)
class DiscountingConfig:
    """
    Flat discount rate for simplicity; replace with a curve object in production.
    """
    rate: float  # continuously-compounded
    collateral_rate: Optional[float] = None
