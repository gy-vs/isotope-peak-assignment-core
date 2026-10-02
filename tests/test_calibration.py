"""Tests for calibration semantics: comparisons only, raw peaks untouched."""

import pytest

from isoclaim import (
    Calibration,
    CalibrationError,
    Engine,
    fit_calibration,
    make_measurement,
)


def test_identity_calibration_is_default():
    calibration = Calibration()
    assert calibration.apply(123.456) == pytest.approx(123.456)
    assert calibration.fingerprint == Calibration().fingerprint


def test_offset_and_slope():
    calibration = Calibration(offset=0.01, slope_ppm=10.0, version="v9")
    assert calibration.apply(200.0) == pytest.approx(200.01 + 0.002)


def test_fingerprint_depends_on_coefficients_not_version():
    a = Calibration(offset=0.01, version="v1")
    b = Calibration(offset=0.01, version="v2-renamed")
    assert a.fingerprint == b.fingerprint
    assert a.version != b.version
    assert Calibration(offset=0.02).fingerprint != a.fingerprint


def test_robust_fit_resists_outlier():
    anchors = [
        (100.0, 100.002),
        (200.0, 200.006),
        (300.0, 300.010),
        (150.0, 150.9),  # gross bad anchor
    ]
    fit = fit_calibration(anchors, version="fit")
    # Inlier truth: offset ~0.002 + 40 ppm slope; the outlier must not win.
    assert fit.apply(100.0) == pytest.approx(100.006, abs=0.01)
    assert fit.apply(300.0) == pytest.approx(300.014, abs=0.01)


def test_single_anchor_estimates_offset_only():
    fit = fit_calibration([(200.0, 200.007)])
    assert fit.offset == pytest.approx(0.007)
    assert fit.slope_ppm == 0.0


def test_fit_requires_anchors():
    with pytest.raises(CalibrationError):
        fit_calibration([])


def test_raw_mz_are_never_modified_by_calibration():
    rows = [(100.0, 10.0), (101.0, 20.0)]
    shifted = Calibration(offset=0.5)
    measurement = make_measurement(rows, shifted, serial="s")
    assert [p.mz for p in measurement.peaks] == [100.0, 101.0]


def test_duplicate_exact_mz_rejected():
    with pytest.raises(Exception):
        make_measurement([(100.0, 1), (100.0, 2)], Calibration())


def test_peak_ids_are_run_local():
    rows_a = [(100.0, 1.0)]
    rows_b = [(100.0, 1.0)]  # identical content, different run
    engine = Engine()
    ma = engine.make_measurement(rows_a, Calibration(version="a"))
    mb = engine.make_measurement(rows_b, Calibration(version="b"))
    assert ma.peaks[0].peak_id != mb.peaks[0].peak_id
    assert ma.serial != mb.serial
