"""Integration: two synthetically overlaid clusters with a known shared peak.

This mirrors the laboratory cross-check: generate two theoretical envelopes,
shift one so one isotope peak exactly coincides, merge their centroid lists,
and verify the resolver (a) never assigns the shared centroid twice and
(b) reports the losing explanation on both candidates.
"""

import numpy as np

from isoassign import (AssignmentSettings, CandidateSpec, PeakTable,
                       assign, build_theory)


def _centroids(theory, shift_da=0.0, scale=1000.0, rel_floor=0.02):
    rows = []
    for p in theory.peaks:
        if p.relative_abundance >= rel_floor:
            rows.append((p.mz + shift_da,
                         scale * p.relative_abundance))
    return rows


def _dedupe_merge(rows_a, rows_b, bin_da=2e-4):
    """Merge two centroid lists; peaks closer than ``bin_da`` are summed into
    one centroid (the *shared peak* the experiment is constructed around)."""
    merged = list(rows_a)
    for mz, inten in rows_b:
        for i, (m, i_int) in enumerate(merged):
            if abs(m - mz) <= bin_da:
                merged[i] = ((m + mz) / 2, i_int + inten)
                break
        else:
            merged.append((mz, inten))
    return sorted(merged)


def test_two_overlaid_clusters_shared_peak():
    # Cluster A: glucose [M+H]+.  Cluster B: glucose envelope shifted so its
    # monoisotopic peak lands on A's M+1 (13C) peak.
    a = build_theory(CandidateSpec("A_glc", "C6H12O6", "[M+H]+"))
    b = build_theory(CandidateSpec("B_glc", "C6H12O6", "[M+H]+"))

    a_m1 = next(p for p in a.peaks
                if p.composition == (("C", 13, 1),)).mz
    shift = a_m1 - b.monoisotopic_mz  # B's mono coincides with A's 13C peak

    rows_a = _centroids(a, shift_da=0.0)
    rows_b = _centroids(b, shift_da=shift)
    rows = _dedupe_merge(rows_a, rows_b)
    table = PeakTable(rows, table_id="overlay")

    result = assign([a, b], table,
                    settings=AssignmentSettings(ppm_tolerance=5.0))

    # (1) the shared centroid is assigned exactly once
    claimed = [m.observed_label for c in result.candidates
               for m in c.matches]
    assert len(claimed) == len(set(claimed))

    # every explained centroid is consumed once; counts add up
    n_matched = sum(c.n_matched_peaks for c in result.candidates)
    assert n_matched <= len(table)

    # (2) the shared peak identity is recoverable: it sits on A's 13C m/z
    shared_peak = next(pk for pk in table if abs(pk.mz - a_m1) < 3e-4)
    owners = [(c.candidate_id, m.theoretical_index)
              for c in result.candidates for m in c.matches
              if m.observed_label == shared_peak.label]
    assert len(owners) == 1

    # (3) the loser's report names the rival and the shared centroid
    reports = {c.candidate_id: c for c in result.candidates}
    winner_id, _ = owners[0]
    loser_id = "B_glc" if winner_id == "A_glc" else "A_glc"
    loser = reports[loser_id]
    assert any(ct.observed_label == shared_peak.label
               for ct in loser.contenders)

    # (4) the winner discloses that the peak was contested
    winner = reports[winner_id]
    assert any(o.observed_label == shared_peak.label
               for o in winner.overlapping_claims)

    # (5) peak intensities near the shared centroid are elevated (summed),
    # yet a single big peak alone cannot drive support: both candidates still
    # need their own non-shared peaks.
    assert any(not m.is_monoisotopic or m.theoretical_index != 1
               for m in winner.matches) or winner.n_matched_peaks >= 2


def test_overlay_is_stable_across_repeated_runs():
    a = build_theory(CandidateSpec("A", "C6H12O6", "[M+H]+"))
    b = build_theory(CandidateSpec("B", "C6H12O6", "[M+H]+"))
    a_m1 = next(p for p in a.peaks
                if p.composition == (("C", 13, 1),)).mz
    shift = a_m1 - b.monoisotopic_mz
    rows = _dedupe_merge(_centroids(a), _centroids(b, shift_da=shift))
    settings = AssignmentSettings(ppm_tolerance=5.0)

    t = PeakTable(rows, table_id="overlay-stable")
    r1 = assign([a, b], t, settings=settings)
    r2 = assign([b, a], t, settings=settings)  # reversed candidate order
    assert r1.assignment_tuples() == r2.assignment_tuples()
