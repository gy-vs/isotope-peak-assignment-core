"""Global competitive assignment: exclusivity, calibration, coordination."""

import numpy as np
import pytest

from isoassign import (AssignmentSettings, Calibration, CandidateSpec,
                       IsoAssignError, PeakTable, assign, build_theory)
from isoassign.theory import TheoryLibrary

from conftest import synthetic_table


def _peak(theory, composition):
    return next(p for p in theory.peaks if p.composition == composition)


def test_support_clean_candidate(glucose_theory):
    table = synthetic_table(glucose_theory, n_peaks=3)
    result = assign([glucose_theory], table)
    rep = result.report("glc")
    assert rep.status == "supported"
    assert rep.n_matched_peaks == 3
    assert rep.monoisotopic_matched
    assert rep.coverage == pytest.approx(1.0)
    assert rep.pattern_score > 0.99
    assert not rep.overlapping_claims
    assert not rep.contenders
    assert len(result.unassigned) == 0


def test_matches_carry_raw_and_calibrated_identity(glucose_theory):
    p1 = _peak(glucose_theory, (("C", 13, 1),))
    # observed 20 ppm high
    table = PeakTable([(glucose_theory.monoisotopic_mz * 1.000020, 1000.0),
                       (p1.mz * 1.000020, 65.0)])
    cal = Calibration.from_ppm(-20.0, version="cal-A")
    result = assign([glucose_theory], table, cal)
    match = result.report("glc").matches[0]
    assert match.observed_mz_raw != match.observed_mz_calibrated
    assert match.observed_mz_raw == table.mz[0]  # raw table untouched
    assert abs(match.error_ppm) < 0.01
    # calibration never rewrites source data
    assert table.mz[0] != match.observed_mz_calibrated


def test_calibration_changes_judgement_but_not_identity(glucose_theory):
    p1 = _peak(glucose_theory, (("C", 13, 1),))
    table = PeakTable([(glucose_theory.monoisotopic_mz * 1.000020, 1000.0),
                       (p1.mz * 1.000020, 65.0)])
    none_ = assign([glucose_theory], table, Calibration.identity("v0"))
    assert none_.report("glc").status in ("weak", "unsupported")
    fixed = assign([glucose_theory], table,
                   Calibration.from_ppm(-20.0, version="v1"))
    assert fixed.report("glc").status == "supported"
    # same raw peak labels appear in both (one table identity)
    labels_a = {m.observed_label for m in none_.report("glc").matches}
    labels_b = {m.observed_label for m in fixed.report("glc").matches}
    assert all(lab.startswith(table.table_id + ":")
               for lab in labels_a | labels_b)


def test_lockmass_calibration(glucose_theory):
    p1 = _peak(glucose_theory, (("C", 13, 1),))
    m0, m1 = glucose_theory.monoisotopic_mz, p1.mz
    table = PeakTable([(m0 * 1.000020, 1000.0), (m1 * 1.000020, 65.0)])
    cal = Calibration.lock_mass(m0 * 1.000020, m0, m1 * 1.000020, m1,
                                version="lock1")
    rep = assign([glucose_theory], table, cal).report("glc")
    assert rep.status == "supported"
    assert max(abs(m.error_ppm) for m in rep.matches) < 0.01


def test_shared_peak_owned_once():
    # Two isomeric candidates (identical envelope, different ids), one table
    # whose central peak can serve both.  Globally: exclusive ownership.
    g = build_theory(CandidateSpec("glc", "C6H12O6", "[M+H]+"))
    f = build_theory(CandidateSpec("fru", "C6H12O6", "[M+H]+"))
    m0 = g.monoisotopic_mz
    m1 = _peak(g, (("C", 13, 1),)).mz
    table = PeakTable([(m0, 1000.0),
                       (m1 - 0.0008, 60.0),
                       (m1 + 0.0008, 60.0)])
    settings = AssignmentSettings(ppm_tolerance=10.0)
    result = assign([g, f], table, settings=settings)

    claimed = [m.observed_label for c in result.candidates
               for m in c.matches]
    assert len(claimed) == len(set(claimed))  # no double counting
    assert len(claimed) == 3

    statuses = {c.candidate_id: c.status for c in result.candidates}
    assert set(statuses.values()) <= {
        "supported", "contested", "weak", "unsupported", "rejected"}
    # the candidate that took the shared mono/M+1 peaks is marked contested
    winner = result.report("fru")  # lexicographically fru wins exact ties
    loser = result.report("glc")
    assert winner.n_matched_peaks + loser.n_matched_peaks == 3
    assert winner.status == "contested"
    assert loser.status == "weak"
    assert winner.overlapping_claims
    assert any(x.rival_candidate_id == "fru" for x in loser.contenders)


def test_adding_candidate_recoordinates_globally():
    g = build_theory(CandidateSpec("glc", "C6H12O6", "[M+H]+"))
    f = build_theory(CandidateSpec("fru", "C6H12O6", "[M+H]+"))
    m0 = g.monoisotopic_mz
    m1 = _peak(g, (("C", 13, 1),)).mz
    table = PeakTable([(m0, 1000.0),
                       (m1 - 0.0008, 60.0),
                       (m1 + 0.0008, 60.0)])
    settings = AssignmentSettings(ppm_tolerance=10.0)
    alone = assign([g], table, settings=settings).report("glc")
    together = assign([g, f], table, settings=settings).report("glc")
    assert alone.n_matched_peaks == 3
    assert together.n_matched_peaks < 3  # peaks were reallocated, not appended


def test_determinism_under_input_permutation():
    g = build_theory(CandidateSpec("glc", "C6H12O6", "[M+H]+"))
    f = build_theory(CandidateSpec("fru", "C6H12O6", "[M+H]+"))
    m0 = g.monoisotopic_mz
    m1 = _peak(g, (("C", 13, 1),)).mz
    rows = [(m0, 1000.0), (m1 - 0.0008, 60.0), (m1 + 0.0008, 60.0)]
    settings = AssignmentSettings(ppm_tolerance=10.0)

    t1 = PeakTable(rows, table_id="same")
    t2 = PeakTable(list(reversed(rows)), table_id="same")
    r1 = assign([g, f], t1, settings=settings)
    r2 = assign([f, g], t2, settings=settings)

    def pairs(res):
        return sorted((m.candidate_id, round(m.theoretical_mz, 5))
                      for c in res.candidates for m in c.matches)
    assert pairs(r1) == pairs(r2)
    # same inputs -> identical fingerprint and tuples
    r3 = assign([g, f], t1, settings=settings)
    assert r1.fingerprint == r3.fingerprint
    assert r1.assignment_tuples() == r3.assignment_tuples()


def test_single_big_noise_peak_is_not_support(glucose_theory):
    table = PeakTable([(glucose_theory.monoisotopic_mz, 1.0e7)])
    rep = assign([glucose_theory], table).report("glc")
    assert rep.status in ("weak", "unsupported")
    assert not rep.is_supported
    assert any("peak(s) matched" in x for x in rep.opposing_evidence)


def test_missing_weak_isotope_does_not_reject(glucose_theory):
    p1 = _peak(glucose_theory, (("C", 13, 1),))
    table = PeakTable([(glucose_theory.monoisotopic_mz, 1000.0),
                       (p1.mz, 65.0)])
    rep = assign([glucose_theory], table).report("glc")
    assert rep.status == "supported"  # M+2/weak fine structure may be absent
    assert any(u.relative_abundance < 0.05
               for u in rep.unmatched_theory)


def test_unrelated_huge_peaks_stay_unassigned(glucose_theory):
    p1 = _peak(glucose_theory, (("C", 13, 1),))
    table = PeakTable([(120.0, 9.0e6),
                       (glucose_theory.monoisotopic_mz, 1000.0),
                       (p1.mz, 65.0),
                       (250.0, 7.0e6)])
    result = assign([glucose_theory], table)
    assert result.report("glc").is_supported
    unassigned_mz = sorted(round(u.observed_mz_raw, 1)
                           for u in result.unassigned)
    assert unassigned_mz == [120.0, 250.0]


def test_duplicate_candidate_ids_rejected(glucose_theory):
    dup = build_theory(CandidateSpec("glc", "C6H12O6", "[M+H]+"))
    table = synthetic_table(glucose_theory)
    with pytest.raises(IsoAssignError):
        assign([glucose_theory, dup], table)


def test_resource_limit_component_marked_not_guessed(glucose_theory):
    # Force an impossibly small component budget: candidate is reported as
    # resource_limit rather than receiving a partial, misleading assignment.
    table = synthetic_table(glucose_theory)
    settings = AssignmentSettings(max_component_size=1)
    result = assign([glucose_theory], table, settings=settings)
    rep = result.report("glc")
    assert rep.status == "resource_limit"
    assert rep.errors and "max_component_size" in rep.errors[0]
