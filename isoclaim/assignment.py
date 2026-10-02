"""Global peak-cluster assignment with mutual exclusion.

Assigning peaks to candidates one nearest-neighbour at a time lets the same
observed peak be "claimed" by many formula hypotheses and lets one big noise
peak masquerade as evidence.  Instead we build *all* candidate/observed edges
that fit the matching window and solve one joint optimization:

* an observed peak belongs to at most one explanation (hard exclusivity),
* each candidate is either active with its full coherent explanation or off,
* activation pays a fixed penalty, and every strong predicted peak that is
* absent* pays a missing-peak penalty,

so the chosen explanation is a globally coordinated one.  Candidates that lose
a shared peak are kept in the report as contested alternatives rather than
disappearing.

Scoring components are deliberately simple and fully exposed in each report:
a Gaussian mass-agreement term, a mild intensity-shape term and predicted
abundance scaling.  A single peak can never reach the acceptance threshold on
its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .calibration import Calibration
from .errors import Budget, Diagnostic
from .peaks import Measurement, ObservedPeak
from .theory import TheoryEnvelope

__all__ = [
    "MatchConfig",
    "Status",
    "Edge",
    "SolverOutcome",
    "CandidateOutcome",
    "assign",
]


class Status:
    """Candidate-level outcomes."""

    ACCEPTED = "ACCEPTED"            # coherent multi-peak evidence, won peaks
    WEAK_SUPPORT = "WEAK_SUPPORT"    # evidence present but below policy bar
    CONTESTED = "CONTESTED"          # competed for a shared peak and lost
    INSUFFICIENT = "INSUFFICIENT"    # no active explanation chosen
    OUT_OF_RANGE = "OUT_OF_RANGE"    # predicted base m/z outside scan window
    INVALID = "INVALID"              # candidate itself was not a valid formula
    THEORY_BUDGET = "THEORY_BUDGET"  # envelope could not be computed in budget
    UNRESOLVED = "UNRESOLVED"        # run-level failure (e.g. solver budget)

    _ALL = frozenset(
        {
            ACCEPTED, WEAK_SUPPORT, CONTESTED, INSUFFICIENT, OUT_OF_RANGE,
            INVALID, THEORY_BUDGET, UNRESOLVED,
        }
    )


class SolverOutcome:
    OPTIMAL = "OPTIMAL"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class MatchConfig:
    """Tuning for matching and acceptance.

    Attributes:
        tolerance_ppm: Mass-error half-window in ppm (combined additively with
            ``tolerance_da``).
        tolerance_da: Absolute mass-error half-window in Daltons.
        strong_relative: Predicted peaks at least this relative abundance are
            "strong": missing one is negative evidence.
        missing_penalty: Fraction of a peak's weight charged when a strong
            predicted peak has no observed counterpart.
        activation_penalty: Constant cost of activating any candidate, which
            suppresses matches explained by a single coincident peak.
        intensity_weight: Weight of the log-intensity shape term in the edge
            score (mass agreement always dominates).
        accept_score: Minimum net score required to accept a candidate.
        min_matched_peaks: Minimum number of distinct observed peaks required
            for acceptance (real isotope patterns contain several peaks).
        require_base_for_acceptance: If true, the base (most abundant)
            theoretical peak must be matched for acceptance.
    """

    tolerance_ppm: float = 10.0
    tolerance_da: float = 0.002
    strong_relative: float = 0.03
    missing_penalty: float = 0.25
    activation_penalty: float = 0.05
    intensity_weight: float = 0.25
    accept_score: float = 0.9
    min_matched_peaks: int = 2
    require_base_for_acceptance: bool = True

    def window(self, mz: float) -> float:
        return abs(mz) * self.tolerance_ppm * 1e-6 + self.tolerance_da

    def to_dict(self) -> dict:
        return {
            "tolerance_ppm": self.tolerance_ppm,
            "tolerance_da": self.tolerance_da,
            "strong_relative": self.strong_relative,
            "missing_penalty": self.missing_penalty,
            "activation_penalty": self.activation_penalty,
            "intensity_weight": self.intensity_weight,
            "accept_score": self.accept_score,
            "min_matched_peaks": self.min_matched_peaks,
            "require_base_for_acceptance": self.require_base_for_acceptance,
        }


@dataclass(frozen=True)
class Edge:
    """One possible theory/observation correspondence."""

    candidate_index: int
    peak_index: int
    theory_offset: int
    theoretical_mz: float
    calibrated_mz: float
    observed_mz: float
    error_da: float
    error_ppm: float
    predicted_relative: float
    observed_scaled: float
    mass_agreement: float
    intensity_agreement: float
    weight: float
    gain: float
    strong: bool

    def to_dict(self) -> dict:
        return {
            "candidate_index": self.candidate_index,
            "peak_id_index": self.peak_index,
            "theory_offset": self.theory_offset,
            "theoretical_mz": self.theoretical_mz,
            "calibrated_expected_mz": self.calibrated_mz,
            "observed_mz": self.observed_mz,
            "error_da": self.error_da,
            "error_ppm": self.error_ppm,
            "predicted_relative": self.predicted_relative,
            "observed_scaled": self.observed_scaled,
            "mass_agreement": self.mass_agreement,
            "intensity_agreement": self.intensity_agreement,
            "weight": self.weight,
            "gain": self.gain,
            "strong": self.strong,
        }


@dataclass(frozen=True)
class CandidateOutcome:
    candidate_index: int
    candidate_key: str
    label: str
    formula: str
    adduct: str
    charge: int
    status: str
    neutral_mass: Optional[float]
    monoisotopic_mz: Optional[float]
    envelope: Optional[TheoryEnvelope]
    edges: Tuple[Edge, ...]
    matched_edges: Tuple[Edge, ...]
    missing_strong_offsets: Tuple[int, ...]
    raw_gain: float
    penalty: float
    net_score: float
    active: bool
    diagnostics: Tuple[Diagnostic, ...]

    def to_dict(self) -> dict:
        return {
            "candidate_index": self.candidate_index,
            "candidate_key": self.candidate_key,
            "label": self.label,
            "formula": self.formula,
            "adduct": self.adduct,
            "charge": self.charge,
            "status": self.status,
            "neutral_mass": self.neutral_mass,
            "monoisotopic_mz": self.monoisotopic_mz,
            "envelope": self.envelope.to_dict() if self.envelope else None,
            "edges": [e.to_dict() for e in self.edges],
            "matched_peak_ids": [
                self._peak_ref(e) for e in self.matched_edges
            ],
            "missing_strong_offsets": list(self.missing_strong_offsets),
            "scores": {
                "raw_gain": self.raw_gain,
                "penalty": self.penalty,
                "net_score": self.net_score,
                "active": self.active,
            },
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }

    def _peak_ref(self, edge: Edge) -> dict:
        return {"measurement_peak_index": edge.peak_index}


@dataclass(frozen=True)
class AssignmentResult:
    """The full, auditable output of one measurement assignment."""

    measurement_serial: str
    measurement_fingerprint: str
    calibration: Calibration
    config: MatchConfig
    budget: Budget
    solver_status: str
    objective: Optional[float]
    outcomes: Tuple[CandidateOutcome, ...]
    peaks: Tuple[ObservedPeak, ...]
    assigned_peak_indices: Tuple[int, ...]
    unassigned_peak_indices: Tuple[int, ...]
    contested: Tuple[dict, ...]
    diagnostics: Tuple[Diagnostic, ...]

    def outcome_for(self, candidate_key: str) -> Optional[CandidateOutcome]:
        for outcome in self.outcomes:
            if outcome.candidate_key == candidate_key:
                return outcome
        return None

    def unassigned_peaks(self) -> List[ObservedPeak]:
        return [self.peaks[i] for i in self.unassigned_peak_indices]

    def to_dict(self) -> dict:
        return {
            "measurement": {
                "serial": self.measurement_serial,
                "fingerprint": self.measurement_fingerprint,
                "calibration": self.calibration.to_dict(),
            },
            "config": self.config.to_dict(),
            "budget": {
                "min_abundance": self.budget.min_abundance,
                "max_states": self.budget.max_states,
                "max_peaks": self.budget.max_peaks,
                "max_edges": self.budget.max_edges,
                "time_seconds": self.budget.time_seconds,
                "solver_time_seconds": self.budget.solver_time_seconds,
            },
            "solver": {
                "status": self.solver_status,
                "objective": self.objective,
            },
            "candidates": [o.to_dict() for o in self.outcomes],
            "unassigned_peak_ids": [
                self.peaks[i].peak_id for i in self.unassigned_peak_indices
            ],
            "unassigned_peaks": [
                self.peaks[i].to_dict() for i in self.unassigned_peak_indices
            ],
            "contested_peaks": list(self.contested),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


# ---------------------------------------------------------------------------
# Edge construction / scoring
# ---------------------------------------------------------------------------


def _error_ppm(expected: float, observed: float) -> float:
    return (observed - expected) / expected * 1e6


def _build_edges(
    candidate_index: int,
    envelope: TheoryEnvelope,
    measurement: Measurement,
    config: MatchConfig,
    obs_mz: np.ndarray,
    obs_intensity: np.ndarray,
) -> List[Edge]:
    calibration = measurement.calibration
    edges: List[Edge] = []
    base_abundance = max(p.abundance for p in envelope.peaks)

    # Intensity scale is anchored at the strongest *predicted* peak: the
    # envelope relative abundances are normalized so that peak is 1.0.
    for peak in envelope.peaks:
        expected_mz = calibration.apply(peak.mz)
        window = config.window(expected_mz)
        within = np.flatnonzero(np.abs(obs_mz - expected_mz) <= window)
        if within.size == 0:
            continue
        # At most one observed peak per theory peak: the closest m/z.
        j = int(within[np.argmin(np.abs(obs_mz[within] - expected_mz))])
        observed_mz = float(obs_mz[j])
        error_da = observed_mz - expected_mz
        error_ppm = _error_ppm(expected_mz, observed_mz)

        sigma = window / 2.0
        mass_agreement = math.exp(-0.5 * (error_da / sigma) ** 2)

        observed_scaled = float(
            obs_intensity[j] / obs_intensity.max()
            if obs_intensity.max() > 0
            else 0.0
        )
        predicted_relative = peak.relative_abundance
        # Log-ratio shape agreement mapped into (0, 1].
        if observed_scaled > 0.0:
            log_ratio = math.log10(
                max(observed_scaled, 1e-6)
                / max(predicted_relative, 1e-6)
            )
            intensity_agreement = math.exp(-0.5 * (log_ratio / 1.0) ** 2)
        else:
            intensity_agreement = 0.0

        weight = predicted_relative
        gain = weight * (
            mass_agreement
            + config.intensity_weight * intensity_agreement
        ) / (1.0 + config.intensity_weight)
        edges.append(
            Edge(
                candidate_index=candidate_index,
                peak_index=j,
                theory_offset=peak.nominal_offset,
                theoretical_mz=peak.mz,
                calibrated_mz=expected_mz,
                observed_mz=observed_mz,
                error_da=error_da,
                error_ppm=error_ppm,
                predicted_relative=predicted_relative,
                observed_scaled=observed_scaled,
                mass_agreement=mass_agreement,
                intensity_agreement=intensity_agreement,
                weight=weight,
                gain=gain,
                strong=peak.relative_abundance >= config.strong_relative,
            )
        )
    # Keep edge ordering deterministic.
    edges.sort(key=lambda e: (e.theory_offset, e.peak_index))
    return edges


# ---------------------------------------------------------------------------
# Joint MILP
# ---------------------------------------------------------------------------


def _solve(
    n_candidates: int,
    edges_by_candidate: Sequence[List[Edge]],
    n_peaks: int,
    missing_penalties: Sequence[float],
    config: MatchConfig,
    budget: Budget,
) -> Tuple[str, Optional[float], np.ndarray, np.ndarray]:
    """Solve the joint assignment.

    Variables: one activation ``a_c`` per candidate, one selection ``x_e`` per
    edge.  Objective (to maximize; scipy minimizes its negation)::

        sum_e gain_e * x_e
          - activation_penalty * sum_c a_c
          - sum_c (missing_penalty_c * (a_c - sum_{e in strong(c)} x_e))

    Constraints::

        x_e <= a_c                              (edge only if active)
        sum_{e touching peak p} x_e <= 1        (observed peak exclusivity)
        sum_{e at theory peak} x_e <= 1         (theory peak at most once)
    """
    from scipy.optimize import (
        Bounds,
        LinearConstraint,
        milp,
    )
    from scipy.sparse import lil_matrix

    flat_edges: List[Tuple[int, Edge]] = []
    for c, edges in enumerate(edges_by_candidate):
        for edge in edges:
            flat_edges.append((c, edge))

    total_edges = len(flat_edges)
    n_vars = n_candidates + total_edges
    # x layout: [a_0 .. a_{C-1}, x_0 .. x_{E-1}]

    # Objective coefficients for minimization.  For each candidate the
    # constant activation charge is the activation penalty plus the
    # missing-strong-peak penalties for strong theory peaks that have *no
    # candidate edge at all*; strong peaks that do have edges are charged via
    # a per-edge selector instead, so an edge "refunds" its own penalty when
    # chosen.  This keeps the objective linear without aux variables.
    cvec = np.zeros(n_vars)
    # Tie-breaking: with exactly equal utility, prefer earlier candidates and
    # edges at lower nominal offsets.  Coefficients are < 1e-9 so they cannot
    # change any scientifically meaningful comparison.
    edge_gain_by_id: Dict[Tuple[int, int, int], float] = {}
    strong_edges_present: Dict[int, set] = {}
    for ci, edges in enumerate(edges_by_candidate):
        offsets = {
            (e.theory_offset, e.peak_index) for e in edges if e.strong
        }
        strong_edges_present[ci] = {e.theory_offset for e in edges if e.strong}
        for edge in edges:
            edge_gain_by_id[
                (ci, edge.theory_offset, edge.peak_index)
            ] = edge.gain
    for ci in range(n_candidates):
        envelope_weight_missing = missing_penalties[ci]
        # Refund slots for strong theory peaks reachable by at least one edge:
        # those are charged on the edge columns instead.
        reachable_refund = 0.0
        seen_offsets = set()
        for edge in edges_by_candidate[ci]:
            if edge.strong and edge.theory_offset not in seen_offsets:
                seen_offsets.add(edge.theory_offset)
                reachable_refund += config.missing_penalty * edge.weight
        cvec[ci] = (
            config.activation_penalty
            + envelope_weight_missing
            - reachable_refund
        )
        cvec[ci] += 1e-10 * ci
    for ei, (cand_index, edge) in enumerate(flat_edges):
        column = n_candidates + ei
        cvec[column] = -edge.gain
        if edge.strong:
            # Selecting the edge pays the missing-peak charge that was kept
            # off the activation constant.
            cvec[column] += config.missing_penalty * edge.weight
        cvec[column] += 1e-12 * (
            cand_index * 1000 + edge.theory_offset * 10 + edge.peak_index
        )

    integrality = np.ones(n_vars)
    bounds = Bounds(lb=0.0, ub=1.0)

    rows: List[Tuple[List[int], List[float], float, float]] = []
    for ei, (cand_index, _edge) in enumerate(flat_edges):
        rows.append((
            [cand_index, n_candidates + ei],
            [-1.0, 1.0],
            -np.inf, 0.0,
        ))

    # Observed peak exclusivity: each observed peak is touched by at most one
    # selected edge across all candidates.
    by_peak: Dict[int, List[int]] = {}
    for ei, (_cand_index, edge) in enumerate(flat_edges):
        by_peak.setdefault(edge.peak_index, []).append(n_candidates + ei)
    for _peak_index, columns in by_peak.items():
        rows.append((columns, [1.0] * len(columns), -np.inf, 1.0))

    # Each theory peak matched by at most one edge per candidate.
    by_theory: Dict[Tuple[int, int], List[int]] = {}
    for ei, (cand_index, edge) in enumerate(flat_edges):
        by_theory.setdefault(
            (cand_index, edge.theory_offset), []
        ).append(n_candidates + ei)
    for _key, columns in by_theory.items():
        if len(columns) > 1:
            rows.append((columns, [1.0] * len(columns), -np.inf, 1.0))

    matrix = lil_matrix((len(rows), n_vars))
    lb = np.empty(len(rows))
    ub = np.empty(len(rows))
    for ri, (columns, values, lower, upper) in enumerate(rows):
        for column, value in zip(columns, values):
            matrix[ri, column] = value
        lb[ri] = lower
        ub[ri] = upper

    try:
        result = milp(
            c=cvec,
            integrality=integrality,
            bounds=bounds,
            constraints=LinearConstraint(matrix.tocsr(), lb, ub),
            options={
                "time_limit": float(budget.solver_time_seconds),
                "mip_rel_gap": 0.0,
                "disp": False,
            },
        )
    except Exception as exc:  # pragma: no cover - solver API failure
        return (
            SolverOutcome.FAILED,
            None,
            np.zeros(n_candidates, dtype=bool),
            np.zeros(total_edges, dtype=bool),
            str(exc),
        )

    if result.x is None or not getattr(result, "success", False):
        message = str(getattr(result, "message", "no solution"))
        if result.x is not None and "time" in message.lower():
            # HiGHS can return a feasible (but not proven optimal) incumbent.
            values = np.asarray(result.x)
            activations = values[:n_candidates] >= 0.5
            edge_values = values[n_candidates:] >= 0.5
            return (
                SolverOutcome.BUDGET_EXCEEDED,
                float(-result.fun),
                activations,
                edge_values,
                None,
            )
        return (
            SolverOutcome.FAILED,
            None,
            np.zeros(n_candidates, dtype=bool),
            np.zeros(total_edges, dtype=bool),
            message,
        )

    values = np.asarray(result.x)
    activations = values[:n_candidates] >= 0.5
    edge_values = values[n_candidates:] >= 0.5
    status = (
        SolverOutcome.OPTIMAL
        if getattr(result, "status", 0) == 0
        else SolverOutcome.BUDGET_EXCEEDED
    )
    return status, float(-result.fun), activations, edge_values, None


# ---------------------------------------------------------------------------
# Top-level assignment
# ---------------------------------------------------------------------------


def assign(
    candidates: Sequence[object],
    envelopes: Sequence[Optional[TheoryEnvelope]],
    failures: Sequence[Optional[Tuple[str, str, str]]],
    measurement: Measurement,
    *,
    config: Optional[MatchConfig] = None,
    budget: Optional[Budget] = None,
) -> AssignmentResult:
    """Coordinate every candidate against one measurement.

    ``envelopes[i]`` / ``failures[i]`` describe candidate i's theoretical
    computation (typically produced by :class:`isoclaim.theory.TheoryCache`).
    """
    config = config or MatchConfig()
    budget = budget or Budget()

    obs_mz = np.array([p.mz for p in measurement.peaks], dtype=float)
    obs_intensity = np.array(
        [p.intensity for p in measurement.peaks], dtype=float
    )

    n_candidates = len(candidates)
    edges_by_candidate: List[List[Edge]] = [[] for _ in candidates]
    missing_penalties = np.zeros(n_candidates)
    eligibility = np.ones(n_candidates, dtype=bool)

    for i, (candidate, envelope, failure) in enumerate(
        zip(candidates, envelopes, failures)
    ):
        if failure is not None or envelope is None:
            eligibility[i] = False
            continue
        base_peak = max(envelope.peaks, key=lambda p: p.abundance)
        base_expected = measurement.calibration.apply(base_peak.mz)
        # A candidate can only be supported if its base peak might land on a
        # recorded m/z given the matching window.  This is deliberately a
        # diagnostic category, not an exclusion of isotope matches beyond.
        base_window = config.window(base_expected)
        if not (
            base_expected - base_window <= measurement.mz_max
            and base_expected + base_window >= measurement.mz_min
        ):
            eligibility[i] = False
        edges_by_candidate[i] = _build_edges(
            i, envelope, measurement, config, obs_mz, obs_intensity
        )
        strong_weight = sum(
            p.relative_abundance
            for p in envelope.peaks
            if p.relative_abundance >= config.strong_relative
        )
        missing_penalties[i] = config.missing_penalty * strong_weight

    total_edges = sum(len(edges) for edges in edges_by_candidate)
    diagnostics: List[Diagnostic] = []
    if total_edges > budget.max_edges:
        diagnostics.append(
            Diagnostic(
                code="EDGE_BUDGET",
                message=(
                    f"{total_edges} candidate/observed edges exceed the "
                    f"{budget.max_edges} edge budget; results not solved"
                ),
                details=(total_edges, budget.max_edges),
            )
        )
        return _build_unsolved_result(
            candidates,
            envelopes,
            failures,
            measurement,
            config,
            budget,
            edges_by_candidate,
            diagnostics,
        )

    solved = _solve(
        n_candidates,
        edges_by_candidate,
        len(measurement.peaks),
        missing_penalties,
        config,
        budget,
    )
    (
        solver_status,
        objective,
        activations,
        edge_values,
        solver_exc,
    ) = solved
    if solver_exc is not None:
        diagnostics.append(
            Diagnostic(
                code="SOLVER_FAILED",
                message=f"assignment optimizer failed: {solver_exc}",
            )
        )
    if solver_status != SolverOutcome.OPTIMAL:
        diagnostics.append(
            Diagnostic(
                code="SOLVER_NONOPTIMAL",
                message=(
                    "assignment optimizer did not certify optimality inside "
                    "the resource budget"
                ),
            )
        )

    # Flatten selected edge flags per candidate.  Incumbents returned without
    # an optimality guarantee are not reported as assignments: the run is
    # explicitly failed instead of presenting unverifiable ownership.
    flat_index = 0
    matched_flags: List[np.ndarray] = []
    trusted = solver_status == SolverOutcome.OPTIMAL
    for edges in edges_by_candidate:
        flags = (
            edge_values[flat_index:flat_index + len(edges)]
            if trusted
            else np.zeros(len(edges), dtype=bool)
        )
        flat_index += len(edges)
        matched_flags.append(np.asarray(flags, dtype=bool))

    # Map observed peak -> list of candidates with a selected edge, and list
    # of all candidate edges (to expose un-chosen competition).
    selected_owners: Dict[int, List[int]] = {}
    competing: Dict[int, List[int]] = {}
    for i, edges in enumerate(edges_by_candidate):
        for edge in edges:
            competing.setdefault(edge.peak_index, []).append(i)
        if not activations[i]:
            continue
        for edge, chosen in zip(edges, matched_flags[i]):
            if chosen:
                selected_owners.setdefault(edge.peak_index, []).append(i)

    outcomes: List[CandidateOutcome] = []
    assigned_indices = set()

    for i, candidate in enumerate(candidates):
        envelope = envelopes[i]
        failure = failures[i]
        edges = edges_by_candidate[i]
        chosen = matched_flags[i]
        matched_edges = tuple(e for e, flag in zip(edges, chosen) if flag)
        active = bool(activations[i]) and trusted
        if active:
            assigned_indices.update(e.peak_index for e in matched_edges)

        candidate_diag: List[Diagnostic] = []
        if failure is not None:
            message = getattr(candidate, "message", failure[1])
            candidate_diag.append(
                Diagnostic(code=failure[0], message=message)
            )

        raw_gain = float(sum(e.gain for e in matched_edges))
        # Missing-strong penalty considers *predicted* strong peaks, including
        # ones with no edge at all (no observed peak anywhere in the window).
        strong_predicted = {
            p.nominal_offset: p.relative_abundance
            for p in (envelope.peaks if envelope else ())
            if p.relative_abundance >= config.strong_relative
        }
        matched_strong = {e.theory_offset for e in matched_edges}
        missing_strong: Tuple[int, ...] = tuple(
            sorted(set(strong_predicted) - matched_strong)
        )
        penalty = float(
            config.activation_penalty
            + sum(
                config.missing_penalty * weight
                for offset, weight in strong_predicted.items()
                if offset not in matched_strong
            )
        ) if active else 0.0
        net = raw_gain - penalty if active else 0.0

        status = _classify(
            candidate=i,
            envelope=envelope,
            failure=failure,
            eligible=bool(eligibility[i]),
            edges=edges,
            matched_edges=matched_edges,
            active=active,
            net=net,
            selected_owners=selected_owners,
            competing=competing,
            config=config,
            solver_status=solver_status,
            measurement=measurement,
        )

        diag_list = list(candidate_diag)
        diag_list.extend(_candidate_diagnostics(
            envelope=envelope,
            missing_strong=missing_strong,
            status=status,
            matched_edges=matched_edges,
            config=config,
        ))

        outcomes.append(
            CandidateOutcome(
                candidate_index=i,
                candidate_key=candidate.key,
                label=candidate.label,
                formula=candidate.formula,
                adduct=candidate.adduct,
                charge=candidate.charge
                if candidate.charge is not None
                else candidate.adduct_obj.charge,
                status=status,
                neutral_mass=envelope.neutral_mass if envelope else None,
                monoisotopic_mz=envelope.monoisotopic_mz if envelope else None,
                envelope=envelope,
                edges=tuple(edges),
                matched_edges=matched_edges,
                missing_strong_offsets=missing_strong,
                raw_gain=raw_gain,
                penalty=penalty,
                net_score=net,
                active=active,
                diagnostics=tuple(diag_list),
            )
        )

    contested_report = _contested_report(
        selected_owners, competing, edges_by_candidate, measurement
    )

    assigned_tuple = tuple(sorted(assigned_indices))
    unassigned_tuple = tuple(
        j for j in range(len(measurement.peaks)) if j not in assigned_indices
    )

    return AssignmentResult(
        measurement_serial=measurement.serial,
        measurement_fingerprint=measurement.fingerprint,
        calibration=measurement.calibration,
        config=config,
        budget=budget,
        solver_status=solver_status,
        objective=objective,
        outcomes=tuple(outcomes),
        peaks=measurement.peaks,
        assigned_peak_indices=assigned_tuple,
        unassigned_peak_indices=unassigned_tuple,
        contested=tuple(contested_report),
        diagnostics=tuple(diagnostics),
    )


def _status_is_invalid(failure: Tuple[str, str, str]) -> bool:
    return failure[0] in {"INVALID", "INVALID_FORMULA", "INVALID_ADDUCT"}


def _classify(
    *,
    candidate: int,
    envelope,
    failure,
    eligible: bool,
    edges,
    matched_edges,
    active: bool,
    net: float,
    selected_owners: Dict[int, List[int]],
    competing: Dict[int, List[int]],
    config: MatchConfig,
    solver_status: str,
    measurement: Measurement,
) -> str:
    if failure is not None:
        code = failure[0]
        return {
            "INVALID": Status.INVALID,
            "INVALID_FORMULA": Status.INVALID,
            "INVALID_ADDUCT": Status.INVALID,
            "THEORY_BUDGET": Status.THEORY_BUDGET,
        }.get(code, Status.INVALID)
    if not eligible:
        return Status.OUT_OF_RANGE
    if solver_status != SolverOutcome.OPTIMAL:
        return Status.UNRESOLVED
    if active:
        base_offset = max(envelope.peaks, key=lambda p: p.abundance).nominal_offset
        base_matched = any(e.theory_offset == base_offset for e in matched_edges)
        accepted = (
            net >= config.accept_score
            and len(matched_edges) >= config.min_matched_peaks
            and (
                not config.require_base_for_acceptance or base_matched
            )
        )
        if accepted:
            return Status.ACCEPTED
        return Status.WEAK_SUPPORT

    # Inactive candidate: did it lose specific peaks to a rival?
    lost_peaks = [
        e.peak_index
        for e in edges
        if e.peak_index in selected_owners
        and candidate not in selected_owners[e.peak_index]
    ]
    if lost_peaks:
        return Status.CONTESTED
    return Status.INSUFFICIENT


def _candidate_diagnostics(
    *, envelope, missing_strong, status, matched_edges, config
) -> List[Diagnostic]:
    diagnostics: List[Diagnostic] = []
    if envelope is not None and envelope.crosscheck_warning:
        diagnostics.append(
            Diagnostic(
                code="ISOTOPE_CROSSCHECK",
                message=envelope.crosscheck_warning,
                details=(envelope.crosscheck_max_abs_diff,),
            )
        )
    if status == Status.WEAK_SUPPORT and matched_edges:
        diagnostics.append(
            Diagnostic(
                code="BELOW_ACCEPTANCE_POLICY",
                message=(
                    f"only {len(matched_edges)} matched peak(s); acceptance "
                    f"requires score >= {config.accept_score} and "
                    f">= {config.min_matched_peaks} peaks"
                ),
                details=(len(matched_edges),),
            )
        )
    if missing_strong and status in (
        Status.ACCEPTED, Status.WEAK_SUPPORT, Status.CONTESTED
    ):
        diagnostics.append(
            Diagnostic(
                code="MISSING_STRONG_PEAK",
                message=(
                    "strong predicted isotope peak(s) absent from peak table "
                    "(treated as negative evidence, not as exclusion)"
                ),
                details=tuple(missing_strong),
            )
        )
    return diagnostics


def _contested_report(
    selected_owners: Dict[int, List[int]],
    competing: Dict[int, List[int]],
    edges_by_candidate: Sequence[List[Edge]],
    measurement: Measurement,
) -> List[dict]:
    report: List[dict] = []
    for peak_index, rivals in sorted(competing.items()):
        unique_rivals = sorted(set(rivals))
        if len(unique_rivals) < 2:
            continue
        owner = selected_owners.get(peak_index, [])
        peak = measurement.peaks[peak_index]
        detail = {
            "peak_id": peak.peak_id,
            "observed_mz": peak.mz,
            "observed_intensity": peak.intensity,
            "owner_candidate_index": owner[0] if owner else None,
            "rivals": [],
        }
        for cand_index in unique_rivals:
            edge = next(
                (e for e in edges_by_candidate[cand_index]
                 if e.peak_index == peak_index),
                None,
            )
            if edge is None:
                continue
            detail["rivals"].append({
                "candidate_index": cand_index,
                "theory_offset": edge.theory_offset,
                "theoretical_mz": edge.theoretical_mz,
                "error_ppm": edge.error_ppm,
                "gain": edge.gain,
                "selected": cand_index in owner,
            })
        detail["rivals"].sort(key=lambda r: (not r["selected"], r["candidate_index"]))
        report.append(detail)
    return report


def _build_unsolved_result(
    candidates,
    envelopes,
    failures,
    measurement,
    config,
    budget,
    edges_by_candidate,
    diagnostics,
) -> AssignmentResult:
    outcomes = []
    for i, candidate in enumerate(candidates):
        envelope = envelopes[i]
        failure = failures[i]
        if failure is not None:
            status = Status.THEORY_BUDGET if failure[0] == "THEORY_BUDGET" else Status.INVALID
        elif envelope is None:
            status = Status.THEORY_BUDGET
        else:
            status = Status.UNRESOLVED
        outcomes.append(
            CandidateOutcome(
                candidate_index=i,
                candidate_key=candidate.key,
                label=candidate.label,
                formula=candidate.formula,
                adduct=candidate.adduct,
                charge=candidate.charge
                if candidate.charge is not None
                else candidate.adduct_obj.charge,
                status=status,
                neutral_mass=envelope.neutral_mass if envelope else None,
                monoisotopic_mz=envelope.monoisotopic_mz if envelope else None,
                envelope=envelope,
                edges=tuple(edges_by_candidate[i]),
                matched_edges=(),
                missing_strong_offsets=(),
                raw_gain=0.0,
                penalty=0.0,
                net_score=0.0,
                active=False,
                diagnostics=(),
            )
        )
    return AssignmentResult(
        measurement_serial=measurement.serial,
        measurement_fingerprint=measurement.fingerprint,
        calibration=measurement.calibration,
        config=config,
        budget=budget,
        solver_status=SolverOutcome.BUDGET_EXCEEDED,
        objective=None,
        outcomes=tuple(outcomes),
        peaks=measurement.peaks,
        assigned_peak_indices=(),
        unassigned_peak_indices=tuple(range(len(measurement.peaks))),
        contested=(),
        diagnostics=tuple(diagnostics),
    )
