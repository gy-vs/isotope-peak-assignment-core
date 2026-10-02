"""Centroid peak tables (the observed world).

A :class:`PeakTable` is the identity of *one measurement's* centroid list.
Peak ids (``P000001`` ...) are unique across tables (a global counter), so two
different samples can never silently share an observed-peak identity.  The raw
m/z is immutable; calibration is layered on at assignment time.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .errors import IsoAssignError

_PEAK_COUNTER = itertools.count(1)
_TABLE_COUNTER = itertools.count(1)


@dataclass(frozen=True)
class ObservedPeak:
    id: str
    table_id: str
    mz: float                 # raw instrument m/z (never rewritten)
    intensity: float
    order: int                # index after m/z sorting within the table

    @property
    def label(self) -> str:
        return f"{self.table_id}:{self.id}"


class PeakTable:
    """An immutable, sorted view of one centroid peak list."""

    def __init__(self, peaks, *, table_id: str | None = None,
                 metadata: dict | None = None):
        """
        Args:
            peaks: sequence of ``(mz, intensity)`` pairs or dicts with
                ``mz``/``intensity`` keys.
            table_id: explicit table identity; auto-generated if omitted.
            metadata: arbitrary sample metadata (not used by the engine).
        """
        raw: list[tuple[float, float]] = []
        for i, p in enumerate(peaks):
            try:
                if isinstance(p, dict):
                    mz, inten = float(p["mz"]), float(p["intensity"])
                else:
                    mz, inten = float(p[0]), float(p[1])
            except (KeyError, TypeError, ValueError) as exc:
                raise IsoAssignError(
                    f"peak {i}: expected (mz, intensity) pair") from exc
            if not (mz > 0) or not np.isfinite(mz):
                raise IsoAssignError(f"peak {i}: mz must be positive finite")
            if inten < 0 or not np.isfinite(inten):
                raise IsoAssignError(f"peak {i}: intensity must be >= 0")
            raw.append((mz, inten))
        if not raw:
            raise IsoAssignError("peak table is empty")
        raw.sort(key=lambda t: (t[0], t[1]))
        self._table_id = table_id or f"T{next(_TABLE_COUNTER):04d}"
        self._mz = np.array([p[0] for p in raw], dtype=float)
        self._intensity = np.array([p[1] for p in raw], dtype=float)
        self._peaks = tuple(
            ObservedPeak(id=f"P{next(_PEAK_COUNTER):06d}",
                         table_id=self._table_id, mz=mz, intensity=inten,
                         order=i)
            for i, (mz, inten) in enumerate(raw)
        )
        self._metadata = dict(metadata or {})

    @property
    def table_id(self) -> str:
        return self._table_id

    @property
    def peaks(self) -> tuple[ObservedPeak, ...]:
        return self._peaks

    @property
    def mz(self) -> np.ndarray:
        """Raw m/z array."""
        return self._mz

    @property
    def intensity(self) -> np.ndarray:
        return self._intensity

    @property
    def metadata(self) -> dict:
        return dict(self._metadata)

    def __len__(self) -> int:
        return len(self._peaks)

    def __iter__(self):
        return iter(self._peaks)

    def __getitem__(self, idx: int) -> ObservedPeak:
        return self._peaks[idx]

    def calibrated_mz(self, calibration) -> np.ndarray:
        """Raw m/z mapped through ``calibration`` (raw array is untouched)."""
        return np.asarray(calibration.apply(self._mz), dtype=float)

    def __repr__(self) -> str:
        return f"PeakTable({self._table_id!r}, n={len(self)})"
