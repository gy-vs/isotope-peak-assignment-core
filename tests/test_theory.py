"""Tests for theoretical masses, bounded isotope DP and resource budget."""

import time

import pytest

from isoclaim import (
    Budget,
    BudgetExceededError,
    Candidate,
    Engine,
    build_envelope,
    ion_monoisotopic_mz,
    neutral_mass,
    pyteomics_exact_envelope,
)

# Reference values from standard mass tables / textbook isotope abundances.
GLUCOSE_MH_MINUS = 179.056112
GLUCOSE_NEUTRAL_MONOISOTOPIC = 180.063388


class TestReferenceMasses:
    def test_glucose_neutral_mass(self, glucose):
        assert neutral_mass(glucose.composition) == pytest.approx(
            GLUCOSE_NEUTRAL_MONOISOTOPIC, abs=1e-6
        )

    def test_glucose_deprotonated_mz(self, glucose):
        assert ion_monoisotopic_mz(glucose) == pytest.approx(
            GLUCOSE_MH_MINUS, abs=1e-6
        )

    def test_sodiated_glucose(self):
        candidate = Candidate("C6H12O6", adduct="[M+Na]+")
        assert ion_monoisotopic_mz(candidate) == pytest.approx(
            203.05261, abs=1e-5
        )

    def test_mz_uses_electron_correction_not_proton_convention(self):
        candidate = Candidate("C6H12O6", adduct="[M+H]+")
        assert ion_monoisotopic_mz(candidate) == pytest.approx(
            GLUCOSE_NEUTRAL_MONOISOTOPIC + 1.007276, abs=1e-6
        )


class TestGlucoseEnvelope:
    def test_major_isotope_positions_and_abundances(self, glucose_envelope):
        peaks = {p.nominal_offset: p for p in glucose_envelope.peaks}
        assert peaks[0].relative_abundance == pytest.approx(1.0)
        assert 0.055 <= peaks[1].relative_abundance <= 0.080
        assert 0.008 <= peaks[2].relative_abundance <= 0.020

    def test_textbook_absolute_probabilities(self, glucose_envelope):
        # 6 carbons at 1.07% 13C: M+1 carbon contribution alone ~6.4%.
        peaks = {p.nominal_offset: p for p in glucose_envelope.peaks}
        assert 0.057 <= peaks[1].abundance <= 0.075
        # Total retained probability mass stays close to one.
        assert glucose_envelope.retained_probability > 0.99

    def test_base_peak_is_monoisotopic(self, glucose_envelope):
        base = max(glucose_envelope.peaks, key=lambda p: p.abundance)
        assert base.nominal_offset == 0
        assert base.mz == pytest.approx(glucose_envelope.monoisotopic_mz,
                                        abs=1e-9)

    def test_sorted_and_bounded(self, glucose_envelope):
        offsets = [p.nominal_offset for p in glucose_envelope.peaks]
        assert offsets == sorted(offsets)
        assert all(p.abundance > 0 for p in glucose_envelope.peaks)

    def test_n_states_counts_label_assignments(self, glucose_envelope):
        # [M-H]- glucose: M+1 has 6 (13C) + 6 (17O) + 11 (2H) assignments.
        peak_1 = glucose_envelope.peak_at(1)
        assert peak_1 is not None
        assert peak_1.n_states == 23
        assert glucose_envelope.peaks[0].n_states == 1


class TestIndependentCrossCheck:
    def test_dp_agrees_with_pyteomics_exact_enumeration(self, glucose_envelope):
        # The production DP and pyteomics' independent exact enumeration agree
        # on both abundances and bin centroids.
        assert glucose_envelope.crosscheck_warning is None
        assert glucose_envelope.crosscheck_max_abs_diff is not None
        assert glucose_envelope.crosscheck_max_abs_diff < 0.02

    def test_exact_enumeration_matches_glucose_reference_directly(self):
        exact = pyteomics_exact_envelope(
            {"C": 6, "H": 11, "O": 6}, overall_threshold=1e-5
        )
        relative_1 = exact[1][0] / exact[0][0]
        assert 0.055 <= relative_1 <= 0.080

    def test_crosscheck_skipped_for_large_formula(self):
        envelope = build_envelope(Candidate("C500H1000", adduct="[M+H]+"))
        assert envelope.crosscheck_warning is not None
        assert "cross-check skipped" in envelope.crosscheck_warning


class TestChargeStateSpacing:
    @pytest.mark.parametrize("adduct,charge,spacing", [
        ("[M+2H]2+", 2, 0.5),
        ("[M+3Na]3+", 3, 1.0 / 3.0),
    ])
    def test_isotope_spacing_is_one_over_charge(self, adduct, charge, spacing):
        candidate = Candidate("C12H22O11", adduct=adduct)
        envelope = build_envelope(candidate)
        peak_1 = envelope.peak_at(1)
        assert peak_1 is not None
        assert peak_1.mz - envelope.monoisotopic_mz == pytest.approx(
            spacing, abs=5e-3
        )
        assert envelope.charge == charge


class TestResourceBudget:
    def test_states_budget_fails_loudly(self):
        engine = Engine(budget=Budget(max_states=5))
        candidate = engine.make_candidate("C200H400O200")
        with pytest.raises(BudgetExceededError) as info:
            engine.theory_for(candidate)
        assert info.value.kind == "states"

    def test_time_budget_guarantees_return(self):
        # Even a very generous isotope threshold cannot make the computation
        # run away: the deadline is checked inside the binary power step.
        engine = Engine(budget=Budget(
            time_seconds=0.01, max_states=10_000_000,
            isotope_threshold=0.0, max_peaks=1_000_000,
        ))
        candidate = engine.make_candidate("C100000H200000")
        start = time.perf_counter()
        with pytest.raises(BudgetExceededError) as info:
            engine.theory_for(candidate)
        assert info.value.kind == "time"
        assert time.perf_counter() - start < 5.0

    def test_peaks_cap(self):
        engine = Engine(budget=Budget(max_peaks=2, min_abundance=1e-6))
        candidate = engine.make_candidate("C6H12O6", adduct="[M-H]-")
        with pytest.raises(BudgetExceededError):
            engine.theory_for(candidate)

    def test_runtime_is_sublinear_in_atom_count(self):
        small = build_envelope(Candidate("C60H120", adduct="[M-H]-"))
        t = time.perf_counter()
        large = build_envelope(Candidate("C6000H12000", adduct="[M-H]-"))
        elapsed = time.perf_counter() - t
        assert elapsed < 1.0
        assert len(large.peaks) < 100
        assert len(small.peaks) <= len(large.peaks)

    def test_heavy_molecule_remains_quantitatively_correct(self):
        envelope = build_envelope(Candidate("C1000H2000", adduct="[M+H]+"))
        # With ~1000 carbons the monoisotopic bin becomes negligible and the
        # envelope behaves as a Poisson centred near n*0.0107 ~ 10.7.
        base = max(envelope.peaks, key=lambda p: p.abundance)
        assert 8 <= base.nominal_offset <= 13
        assert base.relative_abundance == pytest.approx(1.0)
        # M+1 absolute probability ~ n*p13 * P(M0), a small but nonzero bin.
        peak_1 = envelope.peak_at(1)
        assert peak_1 is not None
        assert peak_1.abundance > 0.0
        assert peak_1.relative_abundance < 0.01
        # Combinatorial counts overflow for a huge molecule; it is flagged
        # rather than crashing or producing a nonsense integer.
        assert base.n_states == -1
        assert envelope.peaks[0].n_states == -1
