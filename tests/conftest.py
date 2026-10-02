"""Shared test fixtures and reference values."""

import pytest

from isoassign import (AssignmentSettings, Calibration, CandidateSpec,
                       PeakTable, build_theory)

GLUCOSE = "C6H12O6"
GLUCOSE_NEUTRAL_MONO = 180.06338810  # Da
GLUCOSE_MH_MONO = 181.0706646        # m/z [M+H]+


@pytest.fixture
def glucose_theory():
    return build_theory(CandidateSpec("glc", GLUCOSE, "[M+H]+"))


@pytest.fixture
def glucose_settings():
    return AssignmentSettings()


def synthetic_table(theory, n_peaks=3, jitter_ppm=1.0, intensity_scale=1000.0,
                    table_id=None):
    """Build a centroid table from a theory's own strongest peaks."""
    peaks = sorted(theory.peaks, key=lambda p: p.probability,
                   reverse=True)[:n_peaks]
    rows = []
    for i, p in enumerate(sorted(peaks, key=lambda p: p.mz)):
        shift = p.mz * jitter_ppm * 1e-6 * (1 if i % 2 else -1)
        rows.append((p.mz + shift,
                     max(1.0, intensity_scale * p.relative_abundance)))
    return PeakTable(rows, table_id=table_id)
