"""Per-measurement mass calibration.

A calibration is an immutable, versioned mapping from the *raw* m/z recorded by
the instrument to the calibrated m/z used for theoretical comparison::

    mz_cal = mz_raw + correction(mz_raw)

The correction never rewrites the peak table: :class:`~isoassign.peaks.PeakTable`
always keeps raw m/z, calibration is applied inside the assignment and every
report carries both values.  Calibrations are value-equal by their
coefficients and ``version``, so re-running with the same calibration is
cache-stable while changing the version forces a fresh judgement.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import CalibrationError


@dataclass(frozen=True)
class Calibration:
    """A versioned affine/lock-mass calibration model.

    Supported ``kind`` values:

    * ``"identity"`` -- no correction,
    * ``"linear"`` -- ``correction(x) = intercept + slope * x`` (ppm drift is
      ``slope * 1e6``),
    * ``"lockmass"`` -- fitted line through two ``(raw, reference)`` points.
    """

    kind: str = "identity"
    intercept: float = 0.0   # Da
    slope: float = 0.0       # fraction of m/z (slope * 1e6 = ppm)
    version: str = "v0"
    reference: tuple[float, float, float, float] | None = None
    """Lock masses (raw1, ref1, raw2, ref2); None for other kinds."""

    def __post_init__(self):
        if self.kind not in ("identity", "linear", "lockmass"):
            raise CalibrationError(f"unknown calibration kind {self.kind!r}")
        if self.kind == "lockmass":
            ref = self.reference
            if ref is None or len(ref) != 4:
                raise CalibrationError(
                    "lockmass calibration needs four values: "
                    "(raw1, ref1, raw2, ref2)")
            x1, y1, x2, y2 = ref
            if x2 == x1:
                raise CalibrationError("lock masses must have distinct raw m/z")
            object.__setattr__(self, "slope", (y2 - y1) / (x2 - x1) - 1.0)
            object.__setattr__(self, "intercept",
                               (y1 - (1.0 + self.slope) * x1))

    # -- constructors ------------------------------------------------------
    @classmethod
    def identity(cls, version: str = "v0") -> "Calibration":
        return cls(kind="identity", version=version)

    @classmethod
    def linear(cls, intercept: float = 0.0, slope: float = 0.0,
               version: str = "linear-v1") -> "Calibration":
        return cls(kind="linear", intercept=intercept, slope=slope,
                   version=version)

    @classmethod
    def from_ppm(cls, ppm: float, intercept: float = 0.0,
                 version: str = "ppm-v1") -> "Calibration":
        """Constant ppm offset: ``mz_cal = mz_raw * (1 + ppm/1e6) + intercept``."""
        return cls(kind="linear", intercept=intercept, slope=ppm * 1e-6,
                   version=version)

    @classmethod
    def lock_mass(cls, raw1: float, ref1: float,
                  raw2: float, ref2: float,
                  version: str = "lock-v1") -> "Calibration":
        """Two-point lock-mass calibration."""
        return cls(kind="lockmass",
                   reference=(raw1, ref1, raw2, ref2), version=version)

    # -- application -------------------------------------------------------
    def correction(self, mz):
        if self.kind == "identity":
            return 0.0 * mz if hasattr(mz, "__array__") else 0.0
        return self.intercept + self.slope * mz

    def apply(self, mz):
        """Return calibrated m/z for a scalar or array (raw m/z untouched)."""
        return mz + self.correction(mz)

    @property
    def ppm_drift(self) -> float:
        return self.slope * 1e6
