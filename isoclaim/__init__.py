"""isoclaim: competitive isotope-cluster assignment for centroid peak tables.

Public API::

    Candidate, Adcept ...        chemical hypotheses
    Calibration, fit_calibration measurement calibration (versions, refit)
    make_measurement             immutable peak tables with run-local IDs
    Engine                       cached theory + per-measurement assignment
    MatchConfig, Budget          matching policy and resource limits
    Status                       candidate outcome constants
    AssignmentResult             auditable report (theory, raw peaks, decision)
"""

from .assignment import (
    AssignmentResult,
    CandidateOutcome,
    Edge,
    MatchConfig,
    Status,
)
from .calibration import Calibration, CalibrationError, fit_calibration
from .engine import Engine
from .errors import (
    Budget,
    BudgetExceededError,
    Diagnostic,
    InvalidAdductError,
    InvalidFormulaError,
    InvalidPeakTableError,
    IsoClaimError,
)
from .formula import (
    Adduct,
    Candidate,
    canonical_formula,
    default_adduct_registry,
    parse_formula,
)
from .peaks import Measurement, ObservedPeak, make_measurement
from .theory import (
    TheoryCache,
    TheoryEnvelope,
    TheoryPeak,
    build_envelope,
    ion_monoisotopic_mz,
    neutral_mass,
    pyteomics_exact_envelope,
)

__all__ = [
    # chemical identity
    "Candidate",
    "Adduct",
    "parse_formula",
    "canonical_formula",
    "default_adduct_registry",
    # masses / theory
    "neutral_mass",
    "ion_monoisotopic_mz",
    "build_envelope",
    "pyteomics_exact_envelope",
    "TheoryPeak",
    "TheoryEnvelope",
    "TheoryCache",
    # measurement
    "Calibration",
    "CalibrationError",
    "fit_calibration",
    "ObservedPeak",
    "Measurement",
    "make_measurement",
    # engine / assignment
    "Engine",
    "MatchConfig",
    "Budget",
    "Status",
    "Edge",
    "CandidateOutcome",
    "AssignmentResult",
    # errors / diagnostics
    "IsoClaimError",
    "InvalidFormulaError",
    "InvalidAdductError",
    "InvalidPeakTableError",
    "BudgetExceededError",
    "Diagnostic",
]

__version__ = "0.1.0"
