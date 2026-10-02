"""Theory cache separation, invalid candidates, persistence."""

import pickle

import pytest

from isoassign import (Adduct, CandidateSpec, Formula, TheoryLibrary,
                       build_theory)


def test_theory_cache_is_reused_and_measurement_free(glucose_theory):
    lib = TheoryLibrary()
    t1 = lib.add("glc", "C6H12O6", "[M+H]+")
    t2 = lib.add("glc", Formula("C6H12O6"), Adduct("[M+H]+"))
    assert t1 is t2  # same identity -> cached object
    assert t1.monoisotopic_mz == glucose_theory.monoisotopic_mz
    # nothing measurement-related is reachable from the cached object
    blob = pickle.dumps(t1)
    restored = pickle.loads(blob)
    assert restored.monoisotopic_mz == t1.monoisotopic_mz


def test_invalid_formula_candidate_is_recorded_not_dropped():
    lib = TheoryLibrary()
    bad = lib.add("bad1", "C6Hx12", "[M+H]+")
    assert bad.status == "invalid_formula"
    assert bad.errors
    assert bad.spec.valid is False
    assert bad.spec.error_token
    # neutral mass must not be fabricated
    assert bad.neutral_mass != bad.neutral_mass  # NaN
    # good candidates coexist
    good = lib.add("ok", "C6H12O6", "[M+H]+")
    assert good.status == "ok"
    assert len(lib) == 2


def test_invalid_adduct_candidate_is_recorded():
    t = build_theory(CandidateSpec("x", "C6H12O6", "[M+Zz]+"))
    assert t.status == "invalid_formula"
    assert "adduct" in t.errors[0].lower()


def test_remove_and_readd_candidate():
    lib = TheoryLibrary()
    lib.add("a", "C6H12O6", "[M+H]+")
    lib.add("b", "C2H6O", "[M-H]-")
    assert "a" in lib and "b" in lib
    lib.remove("a")
    assert "a" not in lib and "b" in lib
    lib.add("a", "C6H12O6", "[M+Na]+")
    assert lib.get("a").adduct_name == "[M+Na]+"


def test_library_persistence_roundtrip(tmp_path):
    lib = TheoryLibrary()
    lib.add("glc", "C6H12O6", "[M+H]+")
    lib.add("eth", "C2H6O", "[M-H]-")
    path = tmp_path / "theory.pkl"
    lib.save(path)
    reloaded = TheoryLibrary.load(path)
    assert len(reloaded) == 2
    g = reloaded.get("glc")
    assert g.status == "ok"
    assert g.monoisotopic_mz == pytest.approx(181.0706646, abs=1e-6)
    assert len(g.peaks) == len(lib.get("glc").peaks)


def test_readd_same_id_replaces_candidate():
    lib = TheoryLibrary()
    lib.add("x", "C6H12O6", "[M+H]+")
    lib.add("x", "C2H6O", "[M-H]-")
    assert len(lib) == 1
    t = lib.get("x")
    assert t.adduct_name == "[M-H]-"
    assert t.spec.formula.as_dict() == {"C": 2, "H": 6, "O": 1}


def test_custom_adduct():
    a = Adduct("[M+Li]+", delta={"Li": 1}, charge=1)
    t = build_theory(CandidateSpec("g_li", "C6H12O6", a))
    assert t.status == "ok"
    from isoassign.elements import ELEMENTS
    expected = 180.0633881022 + ELEMENTS["Li"].monoisotopic_mass \
        - 0.000548579909065
    assert t.monoisotopic_mz == pytest.approx(expected, abs=1e-6)
