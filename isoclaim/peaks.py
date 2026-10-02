"""Observed peak tables and measurement identity.

A :class:`Measurement` binds one immutable centroid peak table to one
calibration version.  Observed peak identities are *run-local*: their IDs are
derived from the measurement serial and peak content, so assignments from one
sample can never leak into the report for another sample even if the raw m/z
values happen to coincide.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from .calibration import Calibration
from .errors import InvalidPeakTableError

__all__ = [
    "ObservedPeak",
    "Measurement",
]


@dataclass(frozen=True)
class ObservedPeak:
    """One centroid row exactly as recorded by the instrument."""

    peak_id: str
    mz: float
    intensity: float
    index: int

    def to_dict(self) -> dict:
        return {
            "peak_id": self.peak_id,
            "mz": self.mz,
            "intensity": self.intensity,
            "index": self.index,
        }


@dataclass(frozen=True)
class Measurement:
    """An immutable peak table measured under one named calibration."""

    peaks: Tuple[ObservedPeak, ...]
    calibration: Calibration
    serial: str
    mz_min: float
    mz_max: float
    fingerprint: str

    @property
    def mz_values(self) -> List[float]:
        return [p.mz for p in self.peaks]

    def to_dict(self) -> dict:
        return {
            "serial": self.serial,
            "fingerprint": self.fingerprint,
            "calibration": self.calibration.to_dict(),
            "mz_range": [self.mz_min, self.mz_max],
            "peaks": [p.to_dict() for p in self.peaks],
        }


def _row_mz_intensity(row: Any) -> Tuple[float, float]:
    if isinstance(row, dict):
        try:
            mz = row["mz"]
            intensity = row.get("intensity", row.get("i", 1.0))
        except KeyError as exc:
            raise InvalidPeakTableError(
                f"peak row {row!r} missing required key {exc.args[0]!r}"
            ) from exc
    else:
        try:
            mz = row[0]
            intensity = row[1] if len(row) > 1 else 1.0
        except (TypeError, IndexError) as exc:
            raise InvalidPeakTableError(
                f"peak row {row!r} must be (mz, intensity) or a dict"
            ) from exc
    try:
        mz = float(mz)
        intensity = float(intensity)
    except (TypeError, ValueError) as exc:
        raise InvalidPeakTableError(
            f"non-numeric m/z or intensity in row {row!r}"
        ) from exc
    return mz, intensity


def make_measurement(
    peak_table: Iterable[Any],
    calibration: Calibration,
    *,
    serial: str = "run-0001",
    mz_range: Optional[Sequence[float]] = None,
) -> Measurement:
    """Validate and freeze a peak table.

    Rows may be ``(mz, intensity)`` pairs/tuples or dicts with ``mz`` and
    ``intensity`` (alias ``i``).  Rows are *not* reordered: the input order is
    part of the deterministic assignment tie-break.
    """
    rows = list(peak_table)
    if not rows:
        raise InvalidPeakTableError("peak table must contain at least one row")
    peaks: List[ObservedPeak] = []
    seen = set()
    for index, row in enumerate(rows):
        mz, intensity = _row_mz_intensity(row)
        if not (mz > 0.0):
            raise InvalidPeakTableError(f"m/z must be positive, got {mz}")
        if not (intensity >= 0.0):
            raise InvalidPeakTableError(
                f"intensity must be non-negative, got {intensity}"
            )
        if mz in seen:
            raise InvalidPeakTableError(
                f"duplicate exact m/z {mz} at row {index}; centroids must be "
                f"unique within one measurement"
            )
        seen.add(mz)
        peaks.append(
            ObservedPeak(
                peak_id=f"{serial}:p{index:04d}",
                mz=mz,
                intensity=intensity,
                index=index,
            )
        )

    if mz_range is not None:
        mz_min, mz_max = float(mz_range[0]), float(mz_range[1])
        if not (0.0 < mz_min < mz_max):
            raise InvalidPeakTableError("mz_range must satisfy 0 < min < max")
    else:
        mz_min = min(p.mz for p in peaks)
        mz_max = max(p.mz for p in peaks)

    digest_payload = json.dumps(
        [(p.mz, p.intensity, p.index) for p in peaks], sort_keys=False
    )
    fingerprint = hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()[:16]
    return Measurement(
        peaks=tuple(peaks),
        calibration=calibration,
        serial=serial,
        mz_min=mz_min,
        mz_max=mz_max,
        fingerprint=fingerprint,
    )
