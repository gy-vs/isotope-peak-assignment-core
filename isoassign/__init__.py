"""isoassign -- competitive isotope-cluster assignment kernel.

Typical pipeline use::

    from isoassign import (TheoryLibrary, PeakTable, Calibration,
                           AssignmentSettings, assign)

    # 1) reusable theory (no measurement data enters here)
    lib = TheoryLibrary()
    lib.add("glucose", "C6H12O6", "[M+H]+")

    # 2) one measurement
    table = PeakTable([(181.0706, 1000.0), (182.0740, 66.0)])
    cal = Calibration.from_ppm(0.0, version="run42-v1")

    # 3) globally coordinated, mutually-exclusive assignment
    result = assign(lib.theories(), table, cal)

Theoretical masses, raw observations and the final decision are all present on
the result and independently checkable.
"""

from .adducts import Adduct, adduct
from .assignment import AssignmentSettings, assign
from .calibration import Calibration
from .errors import (AdductError, BudgetError, CalibrationError,
                     FormulaError, IsoAssignError)
from .formula import Formula
from .isotopes import IsotopeBudget, IsotopeCluster, IsotopePeak, isotope_distribution
from .peaks import ObservedPeak, PeakTable
from .reports import (AssignmentResult, CandidateReport, CompetingClaim,
                      OverlappingClaim, PeakMatch, UnassignedPeak,
                      UnmatchedTheoreticalPeak)
from .theory import (CandidateSpec, CandidateTheory, TheoryLibrary,
                     TheoreticalPeak, build_theory)

__version__ = "1.0.0"

__all__ = [
    "Adduct", "adduct",
    "AssignmentSettings", "assign",
    "Calibration",
    "IsoAssignError", "FormulaError", "AdductError", "BudgetError",
    "CalibrationError",
    "Formula",
    "IsotopeBudget", "IsotopeCluster", "IsotopePeak", "isotope_distribution",
    "ObservedPeak", "PeakTable",
    "AssignmentResult", "CandidateReport", "CompetingClaim",
    "OverlappingClaim", "PeakMatch", "UnassignedPeak",
    "UnmatchedTheoreticalPeak",
    "CandidateSpec", "CandidateTheory", "TheoryLibrary",
    "TheoreticalPeak", "build_theory",
]
