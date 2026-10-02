"""Formula parsing, validation and exact masses."""

import pytest

from isoassign import Formula, FormulaError
from isoassign.elements import ELEMENTS


def test_basic_composition_and_hill_string():
    f = Formula("C6H12O6")
    assert f.as_dict() == {"C": 6, "H": 12, "O": 6}
    assert str(f) == "C6H12O6"
    assert f.atom_count == 24


def test_hill_ordering_no_carbon():
    # No carbon: Hill system is purely alphabetical across all elements.
    assert str(Formula("H2SO4")) == "H2O4S"
    assert str(Formula({"S": 1, "O": 4, "H": 2})) == "H2O4S"
    assert str(Formula("NaCl")) == "ClNa"
    assert str(Formula({"O": 2, "H": 1})) == "HO2"


def test_parentheses_nested():
    assert Formula("Ca(OH)2").as_dict() == {"Ca": 1, "O": 2, "H": 2}
    assert Formula("Mg3(Si4O10)(OH)2").as_dict() == \
        {"Mg": 3, "Si": 4, "O": 12, "H": 2}
    assert Formula("[Cu(NH3)4]SO4".replace("[", "(").replace("]", ")")).as_dict() \
        == {"Cu": 1, "N": 4, "H": 12, "S": 1, "O": 4}


def test_dict_constructor_normalizes_and_sums():
    f = Formula({"C": 6, "H": 12, "O": 6})
    assert f == Formula("C6H12O6")


def test_invalid_formulas_raise_with_diagnostics():
    bad = ["", "   ", "C6Hx12", "2C", "C2)H", "(C2H6", "C0", "Qx3",
           "C6H12O6("]
    for text in bad:
        with pytest.raises(FormulaError) as exc:
            Formula(text)
        assert str(exc.value)
        assert exc.value.expression == text


def test_unclosed_bracket_is_invalid():
    with pytest.raises(FormulaError) as exc:
        Formula("C(H2")
    assert "parenthesis" in str(exc.value).lower()


def test_zero_and_negative_counts_rejected():
    with pytest.raises(FormulaError):
        Formula({"C": 0, "H": -1})
    with pytest.raises(FormulaError):
        Formula({"C": 1.5})
    with pytest.raises(FormulaError):
        Formula({"X": 1})


def test_glucose_monoisotopic_mass_reference(glucose_theory):
    f = Formula("C6H12O6")
    assert f.monoisotopic_mass == pytest.approx(180.0633881022, abs=2e-7)
    # and it is built from per-element most-abundant isotopes
    expected = 6 * ELEMENTS["C"].monoisotopic_mass \
        + 12 * ELEMENTS["H"].monoisotopic_mass \
        + 6 * ELEMENTS["O"].monoisotopic_mass
    assert f.monoisotopic_mass == pytest.approx(expected, abs=1e-12)


def test_equality_hash_and_immutability():
    assert Formula("C6H12O6") == Formula({"O": 6, "H": 12, "C": 6})
    assert hash(Formula("C6H12O6")) == hash(Formula("C6H12O6"))
    with pytest.raises(Exception):
        Formula("C6H12O6").composition = None
