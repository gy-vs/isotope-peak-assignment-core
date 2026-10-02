"""Global, mutually-exclusive assignment of observed peaks to candidates.

The resolver does **not** walk candidates one by one (that is how shared peaks
get double-counted).  It builds one bipartite graph

    theoretical peak (candidate, peak)  --edge-->  observed peak

over *all* candidates and solves the resulting maximum-weight bipartite
matching.  Weights are driven by *predicted* isotope abundance and mass error
quality, so one huge noise peak can never outweigh a coherent envelope.
Matching is one-to-one: an observed peak is exclusively owned by at most one
theoretical peak and vice versa.

The graph is split into disconnected components (each is small in practice) and
each is solved with SciPy's Hungarian algorithm on a square padded matrix.  A
component above ``max_component_size`` is not silently approximated: every
candidate touching it is reported as ``resource_limit`` and the remaining
components are still resolved.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .calibration import Calibration
from .errors import IsoAssignError
from .peaks import PeakTable
from .reports import (AssignmentResult, CandidateReport, CompetingClaim,
                      OverlappingClaim, PeakMatch, UnassignedPeak,
                      UnmatchedTheoreticalPeak)
from .theory import CandidateTheory, TheoreticalPeak


@dataclass(frozen=True)
class AssignmentSettings:
    """Tolerances and evidence thresholds for one assignment judgement."""

    ppm_tolerance: float = 5.0
    mda_tolerance: float = 0.0          # 0 disables the absolute window
    min_matched_peaks: int = 2
    require_monoisotopic: bool = True
    coverage_threshold: float = 0.85    # fraction of *strong* predicted mass
    pattern_threshold: float = 0.70     # min intensity-pattern cosine
    reject_pattern: float = 0.30        # below this the pattern argues against
    strong_peak_threshold: float = 0.10 # rel. abundance counting as "strong"
    anomaly_ratio: float = 5.0          # observed/expected deviation factor
    max_component_size: int = 400       # nodes (theory + obs) per component
    min_edge_weight: float = 1.0e-6     # floor so boundary edges beat dummies

    def __post_init__(self):
        if self.ppm_tolerance <= 0 or self.mda_tolerance < 0:
            raise ValueError("tolerances must be positive (mda may be 0)")
        if self.min_matched_peaks < 1:
            raise ValueError("min_matched_peaks must be >= 1")
        if not 0.0 <= self.coverage_threshold <= 1.0:
            raise ValueError("coverage_threshold must lie in [0, 1]")
        if not 0.0 <= self.pattern_threshold <= 1.0:
            raise ValueError("pattern_threshold must lie in [0, 1]")

    def fingerprint(self) -> tuple:
        return (
            round(self.ppm_tolerance, 9), round(self.mda_tolerance, 9),
            self.min_matched_peaks, self.require_monoisotopic,
            round(self.coverage_threshold, 6), round(self.pattern_threshold, 6),
            round(self.reject_pattern, 6), round(self.strong_peak_threshold, 6),
            round(self.anomaly_ratio, 6), self.max_component_size,
        )


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def assign(theories,
           peak_table: PeakTable,
           calibration: Calibration | None = None,
           settings: AssignmentSettings | None = None) -> AssignmentResult:
    """Assign ``peak_table`` to candidate theories under one calibration.

    Pure function: the raw peak table is never modified and no observed-peak
    identity leaks across calls.  Re-running with changed calibration only
    changes the calibrated comparison; theories may be reused from a cache.
    """
    calibration = calibration or Calibration.identity()
    settings = settings or AssignmentSettings()
    theories = tuple(theories)
    _check_unique_ids(theories)

    mz_cal = peak_table.calibrated_mz(calibration)
    mz_raw = peak_table.mz
    intensity = peak_table.intensity

    # Canonical candidate order drives every deterministic tie-break.
    order = sorted(range(len(theories)),
                   key=lambda i: (theories[i].candidate_id, i))
    rank = {i: r for r, i in enumerate(order)}

    edges = _build_edges(theories, mz_cal, settings, rank)
    assignment, resource_limited = _resolve(edges, theories, settings)

    reports = []
    for ci, theory in enumerate(theories):
        if theory.status != "ok":
            reports.append(_failed_report(theory))
            continue
        if ci in resource_limited:
            reports.append(_resource_limit_report(theory, settings))
            continue
        reports.append(_evaluate(
            theory, ci, theories, mz_cal, mz_raw, intensity,
            peak_table, assignment, edges, settings))

    unassigned = _unassigned(theories, peak_table, mz_cal, assignment, edges,
                             resource_limited, settings)
    fp = _result_fingerprint(theories, peak_table, calibration, settings,
                             assignment)
    return AssignmentResult(
        table_id=peak_table.table_id,
        calibration_version=calibration.version,
        settings_fingerprint=settings.fingerprint(),
        candidates=tuple(reports),
        unassigned=tuple(unassigned),
        fingerprint=fp,
        diagnostics={"n_candidates": len(theories),
                     "n_observed": len(peak_table),
                     "n_edges": len(edges),
                     "n_resource_limited": len(resource_limited)},
    )


# ---------------------------------------------------------------------------
# edge construction
# ---------------------------------------------------------------------------

def _error_ppm(obs: float, theo: float) -> float:
    return (obs - theo) / theo * 1.0e6


def _error_mda(obs: float, theo: float) -> float:
    return (obs - theo) * 1000.0


def _in_tolerance(ppm: float, mda: float, s: AssignmentSettings) -> bool:
    if abs(ppm) <= s.ppm_tolerance:
        return True
    return s.mda_tolerance > 0 and abs(mda) <= s.mda_tolerance


def _build_edges(theories, mz_cal, settings, rank):
    edges = []
    order = np.argsort(mz_cal)
    sorted_mz = mz_cal[order]
    for ci, theory in enumerate(theories):
        if theory.status != "ok":
            continue
        for peak in theory.peaks:
            # The window is the union of the ppm and absolute tolerances.
            half = abs(peak.mz) * settings.ppm_tolerance * 1e-6
            if settings.mda_tolerance > 0:
                half = max(half, settings.mda_tolerance)
            lo = np.searchsorted(sorted_mz, peak.mz - half, side="left")
            hi = np.searchsorted(sorted_mz, peak.mz + half, side="right")
            for k in range(lo, hi):
                oi = int(order[k])
                obs = float(sorted_mz[k])
                ppm = _error_ppm(obs, peak.mz)
                mda = _error_mda(obs, peak.mz)
                if not _in_tolerance(ppm, mda, settings):
                    continue
                q = 1.0 - min(abs(ppm) / settings.ppm_tolerance, 1.0)
                mass_quality = max(0.0, min(1.0, q))
                base = max(peak.relative_abundance,
                           settings.min_edge_weight * 10) * mass_quality
                base = max(base, settings.min_edge_weight)
                # Deterministic tie-break bonus: candidate id rank first, then
                # theoretical peak index. < 1e-9 so it never changes a
                # genuinely unequal weight.
                bonus = 1e-9 / (rank[ci] + 1) + 1e-12 / (peak.index + 1)
                edges.append(_Edge(ci, peak.index, oi, base + bonus, base,
                                   ppm, mda, mass_quality))
    return edges


# ---------------------------------------------------------------------------
# global matching
# ---------------------------------------------------------------------------

class _Edge:
    __slots__ = ("ci", "ti", "oi", "weight", "raw_weight", "ppm", "mda",
                 "mass_quality")

    def __init__(self, ci, ti, oi, weight, raw_weight, ppm, mda,
                 mass_quality):
        self.ci = ci
        self.ti = ti
        self.oi = oi
        self.weight = weight
        self.raw_weight = raw_weight
        self.ppm = ppm
        self.mda = mda
        self.mass_quality = mass_quality


class _DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _resolve(edges, theories, settings):
    """Return (map (ci,ti)->oi, resource-limited candidate set).

    Two passes: candidates touching an oversized component are removed from
    contention everywhere (their partial support must not be reported as
    settled), then the remaining components are solved independently.
    """
    n_obs_total = 0
    # Observed count needed for node numbering; take max oi + 1.
    obs_count = max((e.oi for e in edges), default=-1) + 1

    def components(active_edges):
        n_theory_nodes = sum(
            len(t.peaks) for t in theories if t.status == "ok")
        # map (ci, ti) -> compact node id
        tnode = {}
        nid = 0
        for ci, t in enumerate(theories):
            if t.status != "ok":
                continue
            for peak in t.peaks:
                tnode[(ci, peak.index)] = nid
                nid += 1
        obs_base = nid
        dsu = _DSU(nid + obs_count)
        for e in active_edges:
            dsu.union(tnode[(e.ci, e.ti)], obs_base + e.oi)
        groups: dict[int, list[_Edge]] = {}
        for e in active_edges:
            r = dsu.find(tnode[(e.ci, e.ti)])
            groups.setdefault(r, []).append(e)
        return groups, tnode, obs_base

    groups, tnode, obs_base = components(edges)

    resource_limited: set[int] = set()
    for group in groups.values():
        t_set = {(e.ci, e.ti) for e in group}
        o_set = {e.oi for e in group}
        if len(t_set) + len(o_set) > settings.max_component_size:
            for ci, _ in t_set:
                resource_limited.add(ci)

    if resource_limited:
        active = [e for e in edges if e.ci not in resource_limited]
        groups, tnode, obs_base = components(active)
    else:
        active = edges

    assignment: dict[tuple[int, int], int] = {}
    for group in groups.values():
        t_keys = sorted({(e.ci, e.ti) for e in group},
                        key=lambda k: (theories[k[0]].candidate_id, k[1]))
        o_keys = sorted({e.oi for e in group})
        nT, nO = len(t_keys), len(o_keys)
        N = nT + nO
        t_pos = {k: i for i, k in enumerate(t_keys)}
        o_pos = {k: j for j, k in enumerate(o_keys)}
        edge_weight = np.empty((nT, nO))
        edge_weight.fill(np.nan)
        for e in group:
            # If several edges target the same cell, keep the strongest.
            i, j = t_pos[(e.ci, e.ti)], o_pos[e.oi]
            w = e.weight
            if math.isnan(edge_weight[i, j]) or w > edge_weight[i, j]:
                edge_weight[i, j] = w

        cost = np.ones((N, N))  # dummy edges cost 1
        real = np.zeros((nT, nO), dtype=bool)
        valid = ~np.isnan(edge_weight)
        real[valid] = True
        cost[:nT, :nO] = 2.0    # non-edges must lose to dummies
        cost[:nT, :nO][valid] = 1.0 - edge_weight[valid]
        # dummy diagonal blocks
        idx = np.arange(N)
        cost[idx[:nT], nO + idx[:nT]] = 1.0
        cost[nT + idx[:nO], idx[:nO]] = 1.0

        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if r < nT and c < nO and real[r, c]:
                assignment[t_keys[r]] = o_keys[c]
    return assignment, resource_limited


# ---------------------------------------------------------------------------
# per-candidate evaluation
# ---------------------------------------------------------------------------

def _failed_report(theory: CandidateTheory) -> CandidateReport:
    spec = theory.spec
    formula_label = str(spec.formula) if spec.valid else str(spec.formula_input)
    adduct_label = spec.adduct.name if spec.valid else str(spec.adduct_input)
    return CandidateReport(
        candidate_id=theory.candidate_id,
        formula=formula_label,
        adduct=adduct_label,
        charge=theory.charge,
        neutral_mass=theory.neutral_mass,
        monoisotopic_mz=theory.monoisotopic_mz,
        status=theory.status,
        n_predicted_peaks=len(theory.peaks),
        errors=theory.errors,
        diagnostics=dict(theory.diagnostics),
    )


def _resource_limit_report(theory, settings) -> CandidateReport:
    return CandidateReport(
        candidate_id=theory.candidate_id,
        formula=str(theory.spec.formula),
        adduct=theory.adduct_name,
        charge=theory.charge,
        neutral_mass=theory.neutral_mass,
        monoisotopic_mz=theory.monoisotopic_mz,
        status="resource_limit",
        n_predicted_peaks=len(theory.peaks),
        errors=("competition component exceeded max_component_size="
                f"{settings.max_component_size}; result not determined",),
        diagnostics={"max_component_size": settings.max_component_size},
    )


def _eligible(peaks, mz_span, settings) -> list[TheoreticalPeak]:
    lo, hi = mz_span
    out = []
    for p in peaks:
        half = abs(p.mz) * settings.ppm_tolerance * 1e-6
        if settings.mda_tolerance > 0:
            half = max(half, settings.mda_tolerance)
        if p.mz - half <= hi and p.mz + half >= lo:
            out.append(p)
    return out


def _evaluate(theory, ci, theories, mz_cal, mz_raw, intensity,
              peak_table, assignment, edges, settings):
    # assignment: (ci, ti) -> oi ; reverse: oi -> (ci, ti)
    owner_of: dict[int, tuple[int, int]] = {}
    for (cci, tti), ooi in assignment.items():
        owner_of[ooi] = (cci, tti)

    own_edges_by_ti: dict[int, list[_Edge]] = {}
    edges_by_obs: dict[int, list[_Edge]] = {}
    for e in edges:
        edges_by_obs.setdefault(e.oi, []).append(e)
        if e.ci == ci:
            own_edges_by_ti.setdefault(e.ti, []).append(e)

    matched_edges = [e for lst in own_edges_by_ti.values() for e in lst
                     if assignment.get((ci, e.ti)) == e.oi]

    span = (float(mz_cal.min()), float(mz_cal.max()))
    eligible = _eligible(theory.peaks, span, settings)
    eligible_idx = {p.index for p in eligible}
    matched_ti = {e.ti for e in matched_edges}

    # intensity pattern over matched peaks
    exp = np.array([theory.peaks[e.ti].relative_abundance
                    for e in matched_edges], dtype=float)
    obs_int = np.array([intensity[e.oi] for e in matched_edges], dtype=float)
    scale = _fit_scale(exp, obs_int)
    pattern, anomalies = _pattern(exp, obs_int, scale, settings)

    matches = []
    for e in matched_edges:
        peak = theory.peaks[e.ti]
        ratio = (intensity[e.oi] / (scale * peak.relative_abundance)
                 if scale > 0 and peak.relative_abundance > 0 else float("nan"))
        matches.append(PeakMatch(
            candidate_id=theory.candidate_id,
            theoretical_index=peak.index,
            is_monoisotopic=peak.is_monoisotopic,
            theoretical_mz=peak.mz,
            observed_label=peak_table[e.oi].label,
            observed_mz_raw=float(mz_raw[e.oi]),
            observed_mz_calibrated=float(mz_cal[e.oi]),
            intensity=float(intensity[e.oi]),
            error_ppm=e.ppm,
            error_mda=e.mda,
            predicted_relative_abundance=peak.relative_abundance,
            observed_to_expected_ratio=ratio,
            mass_quality=e.mass_quality,
        ))

    # unmatched eligible theory peaks; when a fitting observed peak was taken
    # by a rival, record who owns it
    unmatched, contenders = [], []
    for peak in eligible:
        if peak.index in matched_ti:
            continue
        cands = sorted(own_edges_by_ti.get(peak.index, []),
                       key=lambda x: abs(x.ppm))
        claimed_by, claimed_label = None, None
        rival_key = None
        for x in cands:
            owner = owner_of.get(x.oi)
            if owner is None or owner[0] == ci:
                # unassigned, or consumed by another theory peak of THIS
                # candidate (fine-structure peaks fitting one observation):
                # neither is a rival candidate.
                continue
            claimed_by = theories[owner[0]].candidate_id
            claimed_label = peak_table[x.oi].label
            rival_key = (owner, x)
            break
        unmatched.append(UnmatchedTheoreticalPeak(
            theoretical_index=peak.index, mz=peak.mz,
            relative_abundance=peak.relative_abundance,
            is_monoisotopic=peak.is_monoisotopic,
            claimed_by=claimed_by, observed_label=claimed_label))
        if rival_key is not None:
            (rci, rti), x = rival_key
            contenders.append(CompetingClaim(
                observed_label=peak_table[x.oi].label,
                own_theoretical_index=peak.index,
                own_relative_abundance=peak.relative_abundance,
                rival_candidate_id=theories[rci].candidate_id,
                rival_theoretical_index=rti,
                rival_relative_abundance=
                    theories[rci].peaks[rti].relative_abundance,
                rival_error_ppm=abs(
                    _error_ppm(float(mz_cal[x.oi]),
                               theories[rci].peaks[rti].mz))))

    # overlapping claims on peaks this candidate won (other candidates' theory
    # peaks also fit the same observed peak but did not get it). Aggregated per
    # (rival, observed peak); rival theory indices are all listed for review.
    overlap_map: dict[tuple[str, str], list] = {}
    for e in matched_edges:
        for x in edges_by_obs.get(e.oi, ()):
            if x.ci == ci:
                continue
            key = (theories[x.ci].candidate_id,
                   peak_table[e.oi].label)
            slot = overlap_map.get(key)
            if slot is None:
                overlap_map[key] = [e.ti, {x.ti}, abs(x.ppm)]
            else:
                slot[1].add(x.ti)
                slot[2] = min(slot[2], abs(x.ppm))
    overlapping = tuple(
        OverlappingClaim(
            observed_label=label,
            own_theoretical_index=own_ti,
            other_candidate_id=rival,
            other_theoretical_indices=tuple(sorted(rival_tis)),
            other_error_ppm=err)
        for (rival, label), (own_ti, rival_tis, err) in
        sorted(overlap_map.items()))

    # coverage: strong predicted peaks that are matched
    strong = [p for p in eligible
              if p.relative_abundance >= settings.strong_peak_threshold]
    strong_matched = sum(1 for p in strong if p.index in matched_ti)
    coverage = strong_matched / len(strong) if strong else 0.0
    mono = next((p for p in theory.peaks if p.is_monoisotopic), None)
    mono_eligible = mono is not None and mono.index in eligible_idx
    mono_matched = mono is not None and (ci, mono.index) in assignment

    ppms = [m.error_ppm for m in matches]
    rms = float(np.sqrt(np.mean(np.square(ppms)))) if ppms else float("nan")
    maxerr = float(max((abs(x) for x in ppms), default=float("nan")))

    # decision.  A candidate is "contested" when, despite passing thresholds,
    # a *strong* own peak lost to a rival, or a strong peak it won was also a
    # strong peak of a rival (resolved by tie-break) -- i.e. the evidence is
    # not exclusively its own.
    n = len(matches)
    strong_contenders = [
        c for c in contenders
        if c.own_relative_abundance >= settings.strong_peak_threshold]
    close_overlaps = []
    for o in overlapping:
        own_strength = theory.peaks[o.own_theoretical_index].relative_abundance
        if own_strength < settings.strong_peak_threshold:
            continue
        for other in theories:
            if other.candidate_id != o.other_candidate_id:
                continue
            if any(0 <= ti < len(other.peaks)
                   and other.peaks[ti].relative_abundance
                   >= settings.strong_peak_threshold
                   for ti in o.other_theoretical_indices):
                close_overlaps.append(o)
            break
    passes = (
        n >= settings.min_matched_peaks
        and coverage >= settings.coverage_threshold
        and pattern >= settings.pattern_threshold
        and (not settings.require_monoisotopic
             or not mono_eligible or mono_matched)
    )
    contested = passes and (bool(strong_contenders) or bool(close_overlaps))

    if passes:
        status = "contested" if contested else "supported"
    elif n == 0:
        status = "unsupported"
    elif pattern < settings.reject_pattern:
        status = "rejected"
    else:
        status = "weak"

    supporting, opposing = _evidence(
        n, coverage, pattern, mono_eligible, mono_matched, ppms, anomalies,
        unmatched, strong_contenders, settings, status)
    if contested and not any("shared peak" in x for x in opposing):
        for o in close_overlaps:
            opposing = tuple(opposing) + (
                f"strong peak {o.own_theoretical_index} is shared with "
                f"{o.other_candidate_id} (resolved by mass tie-break)",)

    return CandidateReport(
        candidate_id=theory.candidate_id,
        formula=str(theory.spec.formula),
        adduct=theory.adduct_name,
        charge=theory.charge,
        neutral_mass=theory.neutral_mass,
        monoisotopic_mz=theory.monoisotopic_mz,
        status=status,
        n_predicted_peaks=len(theory.peaks),
        matches=tuple(matches),
        unmatched_theory=tuple(unmatched),
        contenders=tuple(contenders),
        overlapping_claims=overlapping,
        n_matched_peaks=n,
        monoisotopic_matched=mono_matched,
        coverage=round(coverage, 6),
        pattern_score=round(pattern, 6),
        rms_error_ppm=rms,
        max_error_ppm=maxerr,
        anomalies=tuple(anomalies),
        supporting_evidence=tuple(supporting),
        opposing_evidence=tuple(opposing),
        diagnostics={"eligible_peak_count": len(eligible),
                     "strong_peak_count": len(strong)},
    )


def _fit_scale(exp, obs):
    """Best non-negative intensity scale s minimizing ||obs - s*exp||."""
    if len(exp) == 0 or np.sum(exp ** 2) == 0:
        return 0.0
    return max(0.0, float(np.dot(obs, exp) / np.dot(exp, exp)))


def _pattern(exp, obs, scale, settings):
    anomalies = []
    if len(exp) == 0 or scale <= 0:
        return 0.0, anomalies
    expected = scale * exp
    norm_o = np.linalg.norm(obs)
    norm_e = np.linalg.norm(expected)
    if norm_o == 0 or norm_e == 0:
        return 0.0, anomalies
    cos = float(np.dot(obs, expected) / (norm_o * norm_e))
    for i, (o, e) in enumerate(zip(obs, expected)):
        if e > 0 and (o / e > settings.anomaly_ratio
                      or e / o > settings.anomaly_ratio):
            anomalies.append(
                f"peak {i}: intensity/expected={o / e:.2g} "
                f"(outside 1/{settings.anomaly_ratio}..{settings.anomaly_ratio})")
    return max(0.0, cos), anomalies


def _evidence(n, coverage, pattern, mono_eligible, mono_matched, ppms,
              anomalies, unmatched, strong_contenders, settings, status):
    support = []
    oppose = []
    if n >= settings.min_matched_peaks:
        support.append(f"{n} peaks matched (>= {settings.min_matched_peaks})")
    else:
        oppose.append(f"only {n} peak(s) matched "
                      f"(need >= {settings.min_matched_peaks})")
    if coverage >= settings.coverage_threshold:
        support.append(f"strong-peak coverage {coverage:.0%}")
    else:
        oppose.append(f"strong-peak coverage only {coverage:.0%} "
                      f"(need {settings.coverage_threshold:.0%})")
    if pattern >= settings.pattern_threshold:
        support.append(f"isotope pattern agreement {pattern:.2f}")
    elif pattern < settings.reject_pattern:
        oppose.append(f"intensity pattern inconsistent (cosine {pattern:.2f})")
    else:
        oppose.append(f"weak pattern agreement {pattern:.2f}")
    if settings.require_monoisotopic and mono_eligible:
        if mono_matched:
            support.append("monoisotopic peak matched")
        else:
            oppose.append("monoisotopic peak not matched")
    if ppms:
        support.append(f"mass RMS {np.sqrt(np.mean(np.square(ppms))):.2f} ppm")
    for a in anomalies:
        oppose.append(a)
    for c in strong_contenders:
        oppose.append(f"strong peak {c.own_theoretical_index} also fits "
                      f"{c.rival_candidate_id} (lost shared peak)")
    return support, oppose


# ---------------------------------------------------------------------------
# unassigned observed peaks
# ---------------------------------------------------------------------------

def _unassigned(theories, peak_table, mz_cal, assignment, edges,
                resource_limited, settings):
    owners = {}
    for (ci, ti), oi in assignment.items():
        owners[oi] = (ci, ti)
    edges_by_obs: dict[int, list[_Edge]] = {}
    for e in edges:
        edges_by_obs.setdefault(e.oi, []).append(e)
    out = []
    for oi in range(len(peak_table)):
        if oi in owners:
            continue
        nearby = tuple(sorted(
            ((theories[e.ci].candidate_id, e.ti,
              theories[e.ci].peaks[e.ti].mz, e.ppm)
             for e in edges_by_obs.get(oi, ())
             if e.ci not in resource_limited),
            key=lambda t: abs(t[3])))
        out.append(UnassignedPeak(
            observed_label=peak_table[oi].label,
            observed_mz_raw=float(peak_table.mz[oi]),
            observed_mz_calibrated=float(mz_cal[oi]),
            intensity=float(peak_table.intensity[oi]),
            nearby=nearby))
    return out


# ---------------------------------------------------------------------------
# fingerprints / guards
# ---------------------------------------------------------------------------

def _check_unique_ids(theories):
    seen = set()
    for t in theories:
        if t.candidate_id in seen:
            raise IsoAssignError(
                f"duplicate candidate_id {t.candidate_id!r}; candidate ids "
                "must be unique within one assignment")
        seen.add(t.candidate_id)


def _array_fingerprint(name, arr) -> str:
    h = hashlib.sha256()
    h.update(name.encode())
    h.update(np.asarray(arr).tobytes())
    return h.hexdigest()


def _result_fingerprint(theories, peak_table, calibration, settings,
                        assignment) -> tuple:
    theory_fp = tuple(
        (t.candidate_id,
         tuple(sorted(t.spec.formula.as_dict().items())) if t.spec.valid
         else repr(t.spec.formula_input),
         t.adduct_name, t.charge,
         tuple(round(p.mz, 9) for p in t.peaks),
         tuple(round(p.relative_abundance, 9) for p in t.peaks))
        for t in sorted(theories, key=lambda x: x.candidate_id))
    cal_fp = (calibration.kind, calibration.intercept, calibration.slope,
              calibration.version)
    pairs = tuple(sorted(
        ((theories[ci].candidate_id, ti),
         peak_table[oi].label)
        for (ci, ti), oi in assignment.items()))
    return (theory_fp, cal_fp, settings.fingerprint(),
            _array_fingerprint("raw_mz", peak_table.mz),
            _array_fingerprint("intensity", peak_table.intensity),
            pairs)
