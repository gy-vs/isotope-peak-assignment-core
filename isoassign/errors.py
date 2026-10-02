"""Exception hierarchy for :mod:`isoassign`.

Failures are explicit: an invalid formula, an isotope computation that cannot
finish inside the resource budget, or an assignment graph that is too large to
resolve never degrade into a silent empty result.  Per-candidate failures are
recorded on the candidate report instead of aborting the whole analysis; only
misuse of the API raises.
"""

from __future__ import annotations


class IsoAssignError(Exception):
    """Base class for all isoassign errors."""


class FormulaError(IsoAssignError):
    """Raised when a formula string or composition cannot form a valid molecule.

    The offending token and a human readable explanation are preserved.
    """

    def __init__(self, message: str, expression: str | None = None,
                 token: str | None = None):
        super().__init__(message)
        self.expression = expression
        self.token = token


class AdductError(IsoAssignError):
    """Raised when an adduct specification is unknown or inconsistent."""


class BudgetError(IsoAssignError):
    """Raised when a hard resource limit must be crossed to finish a result.

    Most budget breaches are *soft*: they are reported on the candidate
    (``status = "budget_exceeded"``) while other candidates are still resolved.
    This exception is reserved for hard limits where returning any number
    would be misleading.
    """

    def __init__(self, message: str, *, limit: int | None = None,
                 observed: int | None = None):
        super().__init__(message)
        self.limit = limit
        self.observed = observed


class CalibrationError(IsoAssignError):
    """Raised when a calibration cannot be constructed or applied."""
