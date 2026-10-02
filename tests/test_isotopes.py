"""Isotope envelope: reference values, pyteomics cross-check, budget."""

import collections
import time

import pytest

from isoassign import (BudgetError, Formula, IsotopeBudget,
                       isotope_distribution)


def _strongest_by_nominal_bin(cluster):
    bins = collections.defaultdict(float)
    for p in cluster.peaks:
        bins[int(round(p.mass))] += p.relative_abundance  # nominal mass bin
    return bins


def test_glucose_envelope_reference(glucose_theory):
    # Neutral envelope explicitly; the candidate theory below is the [M+H]+ ion.
    cl = isotope_distribution(Formula("C6H12O6"))
    mono = cl.peaks[0]
    assert mono.is_monoisotopic
    assert mono.mass == pytest.approx(180.0633881022, abs=2e-7)

    fine = {p.composition: p.relative_abundance for p in cl.peaks}
    c13 = (("C", 13, 1),)
    o17 = (("O", 17, 1),)
    h2 = (("H", 2, 1),)
    o18 = (("O", 18, 1),)
    assert fine[c13] == pytest.approx(0.06489, rel=2e-3)
    assert fine[o17] == pytest.approx(0.002286, rel=2e-3)
    assert fine[h2] == pytest.approx(0.001380, rel=5e-3)
    assert fine[o18] == pytest.approx(0.01233, rel=2e-3)

    bins = _strongest_by_nominal_bin(cl)
    # nominal M+1 = 13C + 2H + 17O ~ 6.86%
    assert bins[181] == pytest.approx(0.06856, rel=3e-2)
    assert bins[182] == pytest.approx(0.01432, rel=3e-2)


def test_pyteomics_independent_cross_check():
    """Same fine-structure M+1 components computed by pyteomics' own engine.

    Note pyteomics' default ``isotope_threshold`` (5e-4) drops 17O/2H; it is
    lowered here for the comparison.
    """
    from pyteomics import mass
    from pyteomics.mass.mass import nist_mass

    agg = collections.defaultdict(float)
    mono_prob = 0.0
    for comp, a in mass.isotopologues(
            {"C": 6, "H": 12, "O": 6}, report_abundance=True,
            isotope_threshold=1e-6, overall_threshold=1e-7):
        m = sum(nist_mass[tok.split("[")[0]]
                [int(tok.split("[")[1][:-1])][0] * cnt
                for tok, cnt in comp.items())
        if 180.06 < m < 180.07:
            mono_prob = max(mono_prob, a)
        if 181.06 < m < 181.071:
            agg[round(m, 6)] += a

    cl = isotope_distribution(Formula("C6H12O6"))
    ours = {round(p.mass, 6): p.relative_abundance
            for p in cl.peaks if 181.06 < p.mass < 181.071}
    for mz, rel in agg.items():
        assert ours[mz] == pytest.approx(rel / mono_prob, rel=2e-3)


def test_monoisotopic_always_force_kept():
    # Tight pruning on a light-element molecule where mono is tiny.
    cl = isotope_distribution(
        Formula("B50"),
        IsotopeBudget(min_rel_abundance=0.5, max_peaks=300,
                      accepted_dropped_fraction=0.99))
    assert cl.peaks[0].is_monoisotopic
    assert cl.peaks[0].relative_abundance <= 1.0


def test_runtime_memory_do_not_explode_with_atom_count():
    # 50k atoms must be no worse than a few hundred peaks and sub-second.
    budget = IsotopeBudget()
    start = time.perf_counter()
    cl = isotope_distribution(Formula("C50000H100000"), budget)
    elapsed = time.perf_counter() - start
    assert len(cl.peaks) <= budget.max_peaks
    assert elapsed < 5.0
    assert cl.max_intermediate < 1e5


def test_budget_exceeded_is_flagged_not_silent():
    budget = IsotopeBudget(max_peaks=30, min_rel_abundance=1e-6,
                           accepted_dropped_fraction=1e-9)
    cl = isotope_distribution(Formula("C5000H10000O1000"), budget)
    assert len(cl.peaks) <= 30
    assert cl.budget_exceeded is True
    assert cl.dropped_fraction > 1e-9
    assert cl.truncated in (True, False)


def test_hard_pair_limit_is_a_safety_net():
    from isoassign.isotopes import _convolve, _Diag
    budget = IsotopeBudget(max_peaks=10, hard_pair_limit=10)
    d1 = {float(i): [1e-9 * (i + 1), ()] for i in range(5)}
    d2 = {float(i) + 100: [1e-9 * (i + 1), ()] for i in range(5)}
    with pytest.raises(BudgetError) as exc:
        _convolve(d1, d2, budget, _Diag())
    assert exc.value.observed == 25 and exc.value.limit == 10


def test_probability_accounting_is_consistent():
    cl = isotope_distribution(Formula("C6H12O6"))
    retained = sum(p.probability for p in cl.peaks)
    assert retained + cl.dropped_fraction == pytest.approx(1.0, abs=1e-9)


def test_adduct_ion_mz_values(glucose_theory):
    # [M+H]+ = neutral + proton mass
    assert glucose_theory.monoisotopic_mz == pytest.approx(
        180.0633881022 + 1.007276466621, abs=1e-6)
    from isoassign import Adduct, CandidateSpec, build_theory
    neg = build_theory(CandidateSpec("gneg", "C6H12O6", "[M-H]-"))
    assert neg.monoisotopic_mz == pytest.approx(
        180.0633881022 - 1.007276466621, abs=1e-6)
    z2 = build_theory(CandidateSpec("g2", "C6H12O6", "[M+2H]2+"))
    assert z2.monoisotopic_mz == pytest.approx(
        (180.0633881022 + 2 * 1.007276466621) / 2, abs=1e-6)
