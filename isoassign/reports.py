"""Reviewable result objects.

Every number needed to audit a decision is kept in three distinct identities:

* :class:`PeakMatch` -- the theoretical peak, the *raw* observed m/z, the
  calibrated m/z, and the residual between them;
* :class:`CandidateReport` -- the candidate-level evidence for and against;
* :class:`UnassignedPeak` -- observed peaks that no candidate claimed.

Nothing is collapsed into a boolean: losing competitors survive as
``contenders`` and matched peaks that another candidate also could explain
appear in ``overlapping_claims``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class PeakMatch:
    candidate_id: str
    theoretical_index: int
    is_monoisotopic: bool
    theoretical_mz: float
    observed_label: str
    observed_mz_raw: float
    observed_mz_calibrated: float
    intensity: float
    error_ppm: float
    error_mda: float
    predicted_relative_abundance: float
    observed_to_expected_ratio: float
    mass_quality: float


@dataclass(frozen=True)
class UnmatchedTheoreticalPeak:
    theoretical_index: int
    mz: float
    relative_abundance: float
    is_monoisotopic: bool
    claimed_by: str | None = None       # rival candidate that took the obs
    observed_label: str | None = None


@dataclass(frozen=True)
class CompetingClaim:
    """An observed peak that fits this candidate's theory peak but was
    resolved to another candidate."""

    observed_label: str
    own_theoretical_index: int
    own_relative_abundance: float
    rival_candidate_id: str
    rival_theoretical_index: int
    rival_relative_abundance: float
    rival_error_ppm: float


@dataclass(frozen=True)
class OverlappingClaim:
    """An observed peak assigned to this candidate that another candidate
    could also explain within tolerance."""

    observed_label: str
    own_theoretical_index: int
    other_candidate_id: str
    other_theoretical_indices: tuple[int, ...]
    other_error_ppm: float


@dataclass(frozen=True)
class CandidateReport:
    candidate_id: str
    formula: str
    adduct: str
    charge: int
    neutral_mass: float
    monoisotopic_mz: float
    status: str
    n_predicted_peaks: int
    matches: tuple[PeakMatch, ...] = ()
    unmatched_theory: tuple[UnmatchedTheoreticalPeak, ...] = ()
    contenders: tuple[CompetingClaim, ...] = ()
    overlapping_claims: tuple[OverlappingClaim, ...] = ()
    n_matched_peaks: int = 0
    monoisotopic_matched: bool = False
    coverage: float = 0.0
    pattern_score: float = 0.0
    rms_error_ppm: float = float("nan")
    max_error_ppm: float = float("nan")
    anomalies: tuple[str, ...] = ()
    supporting_evidence: tuple[str, ...] = ()
    opposing_evidence: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    diagnostics: dict = field(default_factory=dict)

    @property
    def is_supported(self) -> bool:
        return self.status == "supported"


@dataclass(frozen=True)
class UnassignedPeak:
    observed_label: str
    observed_mz_raw: float
    observed_mz_calibrated: float
    intensity: float
    # candidate peaks within tolerance that did not win this peak
    nearby: tuple[tuple[str, int, float, float], ...] = ()


@dataclass(frozen=True)
class AssignmentResult:
    table_id: str
    calibration_version: str
    settings_fingerprint: tuple
    candidates: tuple[CandidateReport, ...]
    unassigned: tuple[UnassignedPeak, ...]
    fingerprint: tuple
    diagnostics: dict = field(default_factory=dict)

    # -- access ------------------------------------------------------------
    def report(self, candidate_id: str) -> CandidateReport:
        for c in self.candidates:
            if c.candidate_id == candidate_id:
                return c
        raise KeyError(f"unknown candidate {candidate_id!r}")

    def supported(self) -> tuple[CandidateReport, ...]:
        return tuple(c for c in self.candidates if c.status == "supported")

    def status_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.candidates:
            out[c.status] = out.get(c.status, 0) + 1
        return out

    def assignment_tuples(self) -> tuple[tuple[str, str, int], ...]:
        """``(candidate_id, observed_label, theory_index)`` per claimed peak."""
        return tuple(sorted(
            (m.candidate_id, m.observed_label, m.theoretical_index)
            for c in self.candidates for m in c.matches))

    # -- serialization -----------------------------------------------------
    def to_dict(self) -> dict:
        """Plain-dict form (JSON-safe floats/strings/tuples) for audits."""
        return asdict(self)
