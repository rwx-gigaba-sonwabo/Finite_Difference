from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import TYPE_CHECKING, Sequence

import numpy as np

from market_data.risk_factor import RiskFactorSlice

if TYPE_CHECKING:
    pass


class Instrument(ABC):
    """Base class for all priceable instruments.

    Concrete subclasses hold all trade parameters (notional, rates,
    schedules, etc.) and implement :meth:`scenario_npvs` which values the
    instrument across simulation paths, and :meth:`precompute` for any
    offline preparation required before the simulation loop.
    """

    def __init__(self, name: str):
        self.name = name

    @property
    def effective_maturity(self) -> date | None:
        """Latest date on which this instrument can have a non-zero NPV.

        Used by ExposureEngine to validate that the scenario cube's time grid
        extends far enough to price this instrument.  Returns the first of:
        ``_effective_maturity`` (QL-adjusted, set by schedule-based instruments),
        ``maturity_date``, ``expiry_date``, ``delivery_date``, ``end_date``.
        Returns None for instruments that do not define any of these.
        """
        for attr in ("_effective_maturity", "maturity_date", "expiry_date",
                     "delivery_date", "end_date"):
            val = getattr(self, attr, None)
            if val is not None:
                return val
        return None

    def precompute(
        self,
        market_states: Sequence[dict[str, RiskFactorSlice]],
        dates: Sequence[date],
    ) -> None:
        """
        Called once per instrument before the simulation loop begins.

        Override this method for instruments whose pricing requires pre-computation
        against the full set of simulation scenarios — for example:

        - Finite difference / PDE pricers: solve the PDE offline on a representative
          curve and store a price surface for fast lookup at runtime.
        - Regression-based American MC (LSM): train exercise boundaries against the
          simulation paths before the exposure loop.
        - Surrogate / proxy models: fit a polynomial or Gaussian process to the
          scenario grid for cheap per-path evaluation.

        Parameters
        ----------
        market_states : sequence of dict[str, RiskFactorSlice]
            One market state per simulation date, in date order.
            Equivalent to [cube.get_time_slice(t) for t in range(cube.n_times)].
        dates : sequence of date
            The corresponding simulation dates.

        The default implementation is a no-op. Vanilla closed-form instruments
        (IRSwap, CDS) do not need to override this.
        """
        pass

    @abstractmethod
    def scenario_npvs(
        self,
        val_date: date,
        market_state: dict[str, RiskFactorSlice],
        fixings: dict[tuple[str, date], np.ndarray] | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Compute NPV for all paths at a single simulation date.

        Parameters
        ----------
        val_date : date
            The simulation date.
        market_state : dict[str, RiskFactorSlice]
            Factor values at this time step. Each key is a factor name,
            each value is a typed slice (ScalarSlice, CurveSlice, or
            SurfaceSlice). Comes from ScenarioCube.get_time_slice().
        fixings : dict or None
            Pre-computed Category B fixings. Keys are
            (curve_name, period_start_date), values are (n_paths,) arrays.
        rng : np.random.Generator or None
            Optional random number generator for instruments that use inner
            Monte Carlo simulation (e.g. Bermudan swaptions, exotics).
            Vanilla closed-form instruments ignore this parameter.

        Returns
        -------
        np.ndarray, shape (n_paths,)
            NPV per path.
        """
        pass

    def get_ois_initial_cf(self, curve_name: str, p_start: date) -> float | None:
        """Historical compound factor CF(period_start → sim_horizon_start) for OIS seeding.

        Called by ExposureEngine._build_fixings the first time it enters an OIS
        period whose start precedes the simulation horizon (i.e. the period was
        already in progress at t=0).  Mirrors RiskFlow's ``old_resets``
        initialization: instead of starting CF_realized at 1.0 and missing the
        historical overnight compounding, the engine primes the accumulator with
        the known realized compound factor up to the first scenario date.

        Instruments with OIS legs should populate ``self._ois_initial_cfs`` in
        their ``__init__``:

            ois_initial_cfs = {("SOFR_CURVE", date(2024, 10, 1)): 1.0212}

        The value is the scalar compound factor
        ``Index(sim_start) / Index(period_start)`` sourced from the official
        compounded OIS index (e.g. SOFR compound index, ESTR compound index).

        Returns None when no seed is available, leaving CF_realized = 1.0
        (the pre-existing behaviour for periods starting within the horizon).
        """
        cfs = getattr(self, "_ois_initial_cfs", None)
        if not cfs:
            return None
        return cfs.get((curve_name, p_start))

    def npv(
        self,
        val_date: date,
        market_state: dict[str, RiskFactorSlice],
        fixings: dict[tuple[str, date], np.ndarray] | None = None,
    ) -> float:
        """
        Compute NPV for a single market state (standalone / deterministic use).

        Wraps :meth:`scenario_npvs` for convenience when pricing outside a
        simulation context. Pass 1D curve values in market_state; CurveSlice
        normalises them to (1, n_tenors) automatically.

        Returns
        -------
        float
            Scalar NPV.
        """
        return float(self.scenario_npvs(val_date, market_state, fixings)[0])
