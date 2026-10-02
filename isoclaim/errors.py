"""Exception and diagnostic types.

Failure is part of the contract: an invalid formula or a computation that
cannot finish inside the resource budget must be reported explicitly, never
silently collapsed into an empty result.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "IsoClaimError",
    "InvalidFormulaError",
    "InvalidAdductError",
    "InvalidPeakTableError",
    "BudgetExceededError",
    "Budget",
    "Diagnostic",
]


class IsoClaimError(Exception):
    """Base class for isoclaim errors."""


# Re-exported here so callers only need ``isoclaim.errors``; the canonical
# definitions live in :mod:`isoclaim.formula` to avoid an import cycle.
from .formula import (  # noqa: F401
    InvalidAdductError,
    InvalidFormulaError,
)


class InvalidPeakTableError(ValueError):
    """The supplied peak table is structurally unusable."""


class BudgetExceededError(IsoClaimError):
    """A theoretical computation could not finish inside its resource budget."""

    def __init__(self, message: str, *, kind: str = "time", limit: float = 0.0,
                 used: float = 0.0) -> None:
        super().__init__(message)
        self.kind = kind
        self.limit = limit
        self.used = used


@dataclass(frozen=True)
class Budget:
    """Resource limits for theoretical enumeration.

    The isotope state space of heavy formulas is astronomical, but the
    abundance of exotic isotopologues collapses exponentially.  Enumeration is
    therefore truncated by *abundance*, capped by *state count* and guarded by
    wall-clock time so runtime/memory never grow with atom count alone.

    Attributes:
        min_abundance: Isotopologues below this absolute probability are not
            generated (passed to pyteomics as ``overall_threshold``).
        isotope_threshold: Per-isotope natural abundance cutoff passed to
            pyteomics (``isotope_threshold``).
        max_states: Hard cap on generated isotopologue states per candidate.
        max_peaks: Hard cap on aggregated envelope peaks per candidate.
        time_seconds: Wall-clock cap per theoretical computation.
        max_edges: Cap on candidate/observed edges fed to the optimizer.
        solver_time_seconds: Wall-clock cap for the global assignment solve.
    """

    min_abundance: float = 1e-4
    isotope_threshold: float = 1e-5
    max_states: int = 200_000
    max_peaks: int = 96
    time_seconds: float = 5.0
    max_edges: int = 50_000
    solver_time_seconds: float = 10.0

    def __post_init__(self) -> None:
        if not 0.0 < self.min_abundance < 1.0:
            raise ValueError("min_abundance must lie in (0, 1)")
        if not 0.0 <= self.isotope_threshold < 1.0:
            raise ValueError("isotope_threshold must lie in [0, 1)")
        if self.max_states <= 0 or self.max_peaks <= 0 or self.max_edges <= 0:
            raise ValueError("count budgets must be positive")
        if self.time_seconds <= 0 or self.solver_time_seconds <= 0:
            raise ValueError("time budgets must be positive")


@dataclass(frozen=True)
class Diagnostic:
    """A machine-readable failure/warning attached to a candidate or run."""

    code: str
    message: str
    details: tuple = ()

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "details": list(self.details),
        }
