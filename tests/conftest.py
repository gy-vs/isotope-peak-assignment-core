"""Shared pytest helpers."""

import pytest

import isoclaim as ic


@pytest.fixture
def engine():
    return ic.Engine()


@pytest.fixture
def glucose(engine):
    return engine.make_candidate(
        "C6H12O6", adduct="[M-H]-", name="glucose"
    )


@pytest.fixture
def glucose_envelope(engine, glucose):
    return engine.theory_for(glucose)


# Reference values from standard mass tables / textbook isotope abundances.
GLUCOSE_NEUTRAL_MONOISOTOPIC = 180.063388
GLUCOSE_MH_MINUS = 179.056112
