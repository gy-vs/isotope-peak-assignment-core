"""Mass calibration.

A :class:`Calibration` maps theoretical m/z to the m/z scale of one
measurement.  Calibration enters the *comparison* only: raw observed m/z
values are never rewritten, and every reported comparison keeps both numbers
side by side.

Changing calibration (e.g. fitting a refined offset, or loading a new
calibration version) must never change theoretical calculations; it only
triggers a re-judgement of an existing measurement against cached theory.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

__all__ = [
    "Calibration",
    "CalibrationError",
    "fit_calibration",
]


class CalibrationError(ValueError):
    """Raised when calibration data is insufficient or inconsistent."""


@dataclass(frozen=True)
class Calibration:
    """Polynomial calibration ``observed = expected + a + b*expected``.

    Attributes:
        offset: Constant m/z shift.
        slope_ppm: Linear correction in parts-per-million of expected m/z.
        version: Caller-supplied version tag used to distinguish calibration
            releases in reports and cache keys.
    """

    offset: float = 0.0
    slope_ppm: float = 0.0
    version: str = "uncalibrated"

    def apply(self, expected_mz: float) -> float:
        """Return the m/z expected on the instrument's scale."""
        return expected_mz * (1.0 + self.slope_ppm * 1e-6) + self.offset

    def delta(self, expected_mz: float) -> float:
        """Return ``calibrated(expected) - expected``."""
        return expected_mz * self.slope_ppm * 1e-6 + self.offset

    @property
    def fingerprint(self) -> str:
        """Content hash covering coefficients (the version is metadata)."""
        payload = json.dumps(
            {"a": self.offset, "b_ppm": self.slope_ppm},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "offset": self.offset,
            "slope_ppm": self.slope_ppm,
            "version": self.version,
            "fingerprint": self.fingerprint,
        }


def fit_calibration(
    anchors: Iterable[Tuple[float, float]],
    *,
    version: str = "refit",
    prior: Optional[Calibration] = None,
) -> Calibration:
    """Fit an affine calibration from ``(theoretical_mz, observed_mz)`` pairs.

    A robust fit (:func:`scipy.optimize.least_squares` with the soft_l1 loss)
    is used so one bad anchor does not pull the whole calibration.  A single
    anchor estimates the offset only; two or more anchors estimate slope as
    well.

    This is intentionally a *helper*: callers own when to refit and which
    anchors are trustworthy.
    """
    pairs = np.asarray(list(anchors), dtype=float)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.shape[0] == 0:
        raise CalibrationError(
            "at least one (theoretical_mz, observed_mz) anchor is required"
        )
    expected, observed = pairs[:, 0], pairs[:, 1]
    if np.any(expected <= 0) or np.any(observed <= 0):
        raise CalibrationError("m/z anchors must be positive")

    residual = observed - expected
    init_offset = float(np.median(residual))
    init_slope_ppm = float(
        prior.slope_ppm if prior is not None else 0.0
    )
    if expected.size < 2:
        return Calibration(
            offset=init_offset, slope_ppm=init_slope_ppm, version=version
        )

    # Robust nonlinear least squares (soft_l1 loss): one bad anchor does not
    # pull the affine calibration.  Scale residuals to ppm so offset and slope
    # coefficients live on comparable numerical scales.
    from scipy.optimize import least_squares

    def residuals(params: np.ndarray) -> np.ndarray:
        offset, slope_ppm = params
        return (
            residual - (offset + slope_ppm * 1e-6 * expected)
        ) / expected * 1e6

    result = least_squares(
        residuals,
        x0=np.array([init_offset, init_slope_ppm]),
        loss="soft_l1",
        f_scale=2.0,  # anchors deviating by >>2 ppm count as outliers
        max_nfev=1000,
    )
    offset, slope_ppm = result.x
    return Calibration(
        offset=float(offset), slope_ppm=float(slope_ppm), version=version
    )
