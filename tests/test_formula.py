"""Tests for formula parsing, canonicalization and adduct handling."""

import pytest

from isoclaim import (
    Adduct,
    Candidate,
    InvalidAdductError,
    InvalidFormulaError,
    canonical_formula,
    default_adduct_registry,
    parse_formula,
)


class TestParseFormula:
    def test_glucose(self):
        assert parse_formula("C6H12O6") == {"C": 6, "H": 12, "O": 6}

    def test_implicit_one(self):
        assert parse_formula("H2O") == {"H": 2, "O": 1}

    def test_repeated_elements_are_summed(self):
        assert parse_formula("CH3C") == {"C": 2, "H": 3}

    def test_hill_canonicalization(self):
        assert canonical_formula("O6C6H12") == "C6H12O6"
        assert canonical_formula("H2O") == "H2O"
        # Non-carbon formulas sort alphabetically.
        assert canonical_formula("Na2Cl") == "ClNa2"

    @pytest.mark.parametrize(
        "text",
        ["", " ", "C H", "6C", "C 6H12", "(CH3)2", "Qq9", "C0", "H-1O",
         "c6h12", "C6.5H", "CH₄", "C+H"],
    )
    def test_invalid(self, text):
        with pytest.raises(InvalidFormulaError):
            parse_formula(text)


class TestAdducts:
    def test_registry_labels(self):
        registry = default_adduct_registry()
        assert "[M+H]+" in registry
        assert "[M-H]-" in registry
        assert "[M+Na]+" in registry
        # Oligomer adducts are explicitly out of scope.
        assert "[2M+H]+" not in registry

    def test_signed_charges(self):
        assert default_adduct_registry()["[M+H]+"].charge == 1
        assert default_adduct_registry()["[M-H]-"].charge == -1

    def test_multicharge_ad_hoc(self):
        dbl = Candidate("C12H22O11", adduct="[M+2H]2+")
        assert dbl.adduct_obj.charge == 2
        assert dbl.adduct_obj.delta == {"H": 2}

    def test_charge_adduct_disagreement_rejected(self):
        with pytest.raises(InvalidAdductError):
            Candidate("C6H12O6", adduct="[M+H]+", charge=-1)

    def test_deprotonation_too_small_molecule_rejected(self):
        with pytest.raises(InvalidAdductError):
            Candidate("O", adduct="[M-H]-")

    def test_invalid_adduct_literal(self):
        with pytest.raises(InvalidAdductError):
            Candidate("C6H12O6", adduct="protonated")

    def test_candidate_identity_key(self):
        a = Candidate("C6H12O6", adduct="[M-H]-")
        b = Candidate("C6H12O6", adduct="[M+H]+")
        assert a.key != b.key
        # Candidate objects are immutable value identities.
        with pytest.raises(Exception):
            a.formula = "H2O"  # type: ignore[misc]
