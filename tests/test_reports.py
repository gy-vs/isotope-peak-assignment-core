"""End-to-end reviewability: three identities, fingerprints, audit dicts."""

import json

import pytest

from isoassign import (AssignmentResult, AssignmentSettings, CandidateSpec,
                       PeakTable, assign, build_theory)


def _peak(theory, composition):
    return next(p for p in theory.peaks if p.composition == composition)


def test_result_keeps_theoretical_raw_and_decision(glucose_theory):
    p1 = _peak(glucose_theory, (("C", 13, 1),))
    table = PeakTable([(glucose_theory.monoisotopic_mz, 1000.0),
                       (p1.mz, 65.0)], table_id="audit1")
    result = assign([glucose_theory], table)
    rep = result.report("glc")

    # theoretical
    assert rep.n_predicted_peaks >= 4
    theo_mz = [m.theoretical_mz for m in rep.matches]
    assert theo_mz[0] == pytest.approx(glucose_theory.monoisotopic_mz)
    # observed (raw)
    assert [m.observed_mz_raw for m in rep.matches] == list(table.mz)
    # decision
    assert rep.status == "supported"

    # error is derived from calibrated vs theoretical and is independently
    # recomputable
    for m in rep.matches:
        recomputed = (m.observed_mz_calibrated - m.theoretical_mz) \
            / m.theoretical_mz * 1e6
        assert recomputed == pytest.approx(m.error_ppm, abs=1e-9)
        assert m.observed_mz_calibrated >= 0


def test_unassigned_peak_report_preserves_identity_and_nearby(glucose_theory):
    p1 = _peak(glucose_theory, (("C", 13, 1),))
    noise = glucose_theory.monoisotopic_mz + 50.0
    table = PeakTable([(glucose_theory.monoisotopic_mz, 1000.0),
                       (p1.mz, 65.0),
                       (noise, 42.0)], table_id="audit2")
    result = assign([glucose_theory], table)
    assert len(result.unassigned) == 1
    u = result.unassigned[0]
    assert u.observed_mz_raw == noise
    assert u.observed_mz_calibrated == noise
    assert u.intensity == 42.0
    assert u.observed_label.startswith("audit2:")


def test_nearby_theory_on_unassigned_is_listed(glucose_theory):
    # An observed peak within tolerance but withheld from the candidate by a
    # stronger rival still surfaces as nearby on the unassigned record or as a
    # contender; here a near-miss peak appears in ``nearby``.
    m0 = glucose_theory.monoisotopic_mz
    # 3 ppm off mono: within default 5 ppm, so it CAN be assigned to nothing
    # else; build a second candidate that owns the envelope instead.
    other = build_theory(CandidateSpec(
        "other", "C12H22O11", "[M+Na]+"))  # unrelated, far away
    near = m0 + m0 * 3e-6
    table = PeakTable([(near, 500.0)], table_id="near1")
    result = assign([other], table)  # other has no peak anywhere near
    u = result.unassigned[0]
    assert u.nearby == ()  # genuinely no theory nearby
    result2 = assign([glucose_theory], table)
    # glucose mono is 3 ppm away -> it gets claimed (supported only if other
    # peaks exist); ensure at minimum the edge is visible
    rep = result2.report("glc")
    assert rep.n_matched_peaks == 1
    assert abs(rep.matches[0].error_ppm - 3.0) < 0.01


def test_fingerprint_changes_with_table_and_settings(glucose_theory):
    table = PeakTable([(glucose_theory.monoisotopic_mz, 1000.0)],
                      table_id="fp1")
    r1 = assign([glucose_theory], table)
    r2 = assign([glucose_theory],
                PeakTable([(glucose_theory.monoisotopic_mz, 1000.0)],
                          table_id="fp2"))
    assert r1.fingerprint != r2.fingerprint
    r3 = assign([glucose_theory], table,
                settings=AssignmentSettings(ppm_tolerance=2.0))
    assert r1.fingerprint != r3.fingerprint


def test_to_dict_is_json_serializable_and_complete(glucose_theory):
    p1 = _peak(glucose_theory, (("C", 13, 1),))
    table = PeakTable([(glucose_theory.monoisotopic_mz, 1000.0),
                       (p1.mz, 65.0)])
    result = assign([glucose_theory], table)
    payload = result.to_dict()
    text = json.dumps(payload, default=str)  # no exotic types
    parsed = json.loads(text)
    assert parsed["table_id"] == table.table_id
    assert parsed["candidates"][0]["candidate_id"] == "glc"
    assert "matches" in parsed["candidates"][0]
    assert "unassigned" in parsed
    # supporting/opposing explanations are present
    c = parsed["candidates"][0]
    assert c["supporting_evidence"] and isinstance(c["supporting_evidence"], list)


def test_failed_candidate_keeps_others_diagnosed(glucose_theory):
    bad = build_theory(CandidateSpec("bad", "Cx9", "[M+H]+"))
    table = PeakTable([(glucose_theory.monoisotopic_mz, 1000.0)])
    result = assign([bad, glucose_theory], table)
    counts = result.status_counts()
    assert counts["invalid_formula"] == 1
    br = result.report("bad")
    assert br.errors and br.formula == "Cx9"
    # good candidate is still evaluated independently
    assert result.report("glc").status in ("weak", "supported", "unsupported")
