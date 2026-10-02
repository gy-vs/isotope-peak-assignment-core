"""The library entry point: theory cache + measurement-scoped assignment.

Typical use in a laboratory pipeline::

    from isoclaim import Engine, Calibration, Candidate, make_measurement

    engine = Engine()                       # reusable across all runs
    calibration = Calibration(version="batch-7")
    measurement = make_measurement(rows, calibration, serial="plate-A3")
    result = engine.analyze(candidates, measurement)

The same engine (hence the same cached theoretical envelopes) is reused for
many peak tables and calibration versions; observed peak identities are
created per :class:`~isoclaim.peaks.Measurement` and never survive across
runs.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .assignment import (
    AssignmentResult,
    MatchConfig,
    assign,
)
from .calibration import Calibration
from .errors import Budget, InvalidAdductError, InvalidFormulaError
from .formula import Candidate, default_adduct_registry
from .peaks import Measurement, make_measurement
from .theory import TheoryCache, TheoryEnvelope

__all__ = ["Engine"]


class _InvalidPlaceholder:
    """Minimal candidate-shaped object for an unparseable list entry."""

    __slots__ = ("key", "label", "formula", "adduct", "charge",
                 "adduct_obj", "composition", "ion_composition", "message")

    def __init__(self, *, label: str, message: str) -> None:
        self.key = f"invalid:{label}"
        self.label = label
        self.formula = ""
        self.adduct = ""
        self.charge = 0
        self.message = message
        self.adduct_obj = None
        self.composition = {}
        self.ion_composition = {}


class Engine:
    """Stateful but measurement-free front end.

    The only mutable state is the theory cache and a serial counter.  No
    observed peaks, assignments or calibrations are retained between calls to
    :meth:`analyze`, so rerunning with a new calibration version cannot
    inherit the previous sample's peak ownership.
    """

    def __init__(
        self,
        *,
        budget: Optional[Budget] = None,
        config: Optional[MatchConfig] = None,
        adducts: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.budget = budget or Budget()
        self.config = config or MatchConfig()
        self.adducts = adducts or default_adduct_registry()
        self.theory_cache = TheoryCache(budget=self.budget)
        self._serials = itertools.count(1)

    # ------------------------------------------------------------------
    # Candidate construction / theory
    # ------------------------------------------------------------------

    def make_candidate(
        self,
        formula: str,
        *,
        adduct: str = "[M+H]+",
        charge: Optional[int] = None,
        name: Optional[str] = None,
    ) -> Candidate:
        """Construct a candidate against this engine's adduct registry."""
        return Candidate(
            formula,
            adduct=adduct,
            charge=charge,
            name=name,
            adduct_registry=self.adducts,
        )

    def theory_for(self, candidate: Candidate) -> TheoryEnvelope:
        """Return the cached theoretical envelope or raise the cached failure."""
        envelope, failure = self.theory_cache.get_or_build(candidate)
        if failure is not None:
            from .errors import BudgetExceededError

            _code, message, kind = failure
            raise BudgetExceededError(message, kind=kind)
        assert envelope is not None
        return envelope

    # ------------------------------------------------------------------
    # Measurement construction
    # ------------------------------------------------------------------

    def make_measurement(
        self,
        peak_table: Sequence[Any],
        calibration: Optional[Calibration] = None,
        *,
        serial: Optional[str] = None,
        mz_range: Optional[Sequence[float]] = None,
    ) -> Measurement:
        """Build a measurement with a unique engine-issued serial by default."""
        if serial is None:
            serial = f"run-{next(self._serials):04d}"
        return make_measurement(
            peak_table,
            calibration or Calibration(),
            serial=serial,
            mz_range=mz_range,
        )

    # ------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------

    def analyze(
        self,
        candidates: Sequence[Candidate],
        measurement: Measurement,
        *,
        config: Optional[MatchConfig] = None,
    ) -> AssignmentResult:
        """Assign one measurement against all candidates, jointly.

        Adding/withdrawing candidates changes the full candidate list and the
        joint optimization is recomputed from scratch: peak ownership is
        globally re-coordinated, never appended as a local explanation.

        Invalid candidates are reported per-candidate (status ``INVALID``)
        without aborting the others.  Theoretical budget failures are reported
        likewise (status ``THEORY_BUDGET``).
        """
        config = config or self.config
        prepared: List[Candidate] = []
        envelopes: List[Optional[TheoryEnvelope]] = []
        failures: List[Optional[Tuple[str, str, str]]] = []

        for raw in candidates:
            candidate, failure = self._prepare(raw)
            if failure is not None or candidate is None:
                # Keep a placeholder so positional reporting stays aligned;
                # invalid candidates never enter the optimization.
                placeholder = candidate if candidate is not None else _InvalidPlaceholder(
                    label=getattr(raw, "label", repr(raw)[:40]),
                    message=failure[1] if failure else "invalid candidate",
                )
                prepared.append(placeholder)
                envelopes.append(None)
                failures.append(
                    failure or ("INVALID", "invalid candidate", "invalid")
                )
                continue
            envelope, theory_failure = self.theory_cache.get_or_build(candidate)
            prepared.append(candidate)
            envelopes.append(envelope)
            failures.append(theory_failure)

        return assign(
            prepared,
            envelopes,
            failures,
            measurement,
            config=config,
            budget=self.budget,
        )

    def _prepare(
        self, raw: Any
    ) -> Tuple[Optional[Candidate], Optional[Tuple[str, str, str]]]:
        if isinstance(raw, Candidate):
            candidate = Candidate(
                raw.formula,
                adduct=raw.adduct,
                charge=raw.charge,
                name=raw.name,
                adduct_registry=self.adducts,
            )
            return candidate, None
        if isinstance(raw, dict):
            try:
                candidate = self.make_candidate(
                    raw["formula"],
                    adduct=raw.get("adduct", "[M+H]+"),
                    charge=raw.get("charge"),
                    name=raw.get("name"),
                )
            except (
                KeyError,
                InvalidFormulaError,
                InvalidAdductError,
            ) as exc:
                return None, ("INVALID", str(exc), "invalid")
            return candidate, None
        if isinstance(raw, str):
            try:
                return self.make_candidate(raw), None
            except (InvalidFormulaError, InvalidAdductError) as exc:
                return None, ("INVALID", str(exc), "invalid")
        return None, (
            "INVALID",
            f"unsupported candidate specification of type {type(raw).__name__}",
            "invalid",
        )
