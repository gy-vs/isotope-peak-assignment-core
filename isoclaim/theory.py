"""Theoretical isotope envelopes with bounded resource use.

Production envelopes are computed by a *dynamic program in nominal mass-offset
space* rather than by enumerating isotopologues:

* two arrays per processed element hold bin probability ``P[k]`` and the
  probability-weighted exact mass shift ``W[k]``;
* convolutions add an atom or an element distribution in O(offsets * isotopes),
  so cost grows linearly with atom count and never with the combinatorial
  isotopologue count;
* bin centroids come from ``W / P`` and keep sub-millidalton mass accuracy
  (all isotope mass spreads enter the moment update);
* bins whose aggregate probability already lies below the overall threshold
  are discarded once an element distribution is complete, because combining
  with further elements only multiplies probabilities by <= 1.

pyteomics supplies the isotope constants and exact masses, and its *exact*
isotopologue enumeration is retained as an independent cross-check on small
molecules (where its combinatorial expansion is safe).  On large molecules its
enumeration explodes / overflows, which is precisely why it is not on the
production path.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from pyteomics import mass as _pm

from .errors import Budget, BudgetExceededError
from .formula import ELECTRON_MASS_U, Candidate

__all__ = [
    "TheoryPeak",
    "TheoryEnvelope",
    "TheoryCache",
    "neutral_mass",
    "ion_monoisotopic_mz",
    "build_envelope",
    "pyteomics_exact_envelope",
]


def neutral_mass(composition: Dict[str, int]) -> float:
    """Monoisotopic neutral mass of an element-count mapping."""
    return float(_pm.calculate_mass(_pm.Composition(dict(composition))))


def ion_monoisotopic_mz(candidate: Candidate) -> float:
    """Monoisotopic m/z of a candidate ion.

    Computed explicitly as ``(neutral-atom ion mass - z * m_e) / |z|``.
    pyteomics' ``charge=`` argument implements a different convention
    (adding/removing whole protons) and is deliberately not used.
    """
    z = candidate.charge if candidate.charge is not None else candidate.adduct_obj.charge
    atom_mass = neutral_mass(candidate.ion_composition)
    return (atom_mass - z * ELECTRON_MASS_U) / abs(z)


@dataclass(frozen=True)
class TheoryPeak:
    """One peak in a theoretical isotope envelope."""

    nominal_offset: int
    mz: float
    abundance: float           # absolute probability in this envelope
    relative_abundance: float  # relative to the base (most abundant) peak
    n_states: int              # multinomial terms aggregated into this bin

    def to_dict(self) -> dict:
        return {
            "nominal_offset": self.nominal_offset,
            "mz": self.mz,
            "abundance": self.abundance,
            "relative_abundance": self.relative_abundance,
            "n_states": self.n_states,
        }


@dataclass(frozen=True)
class TheoryEnvelope:
    """The theoretical cluster for one candidate."""

    candidate_key: str
    formula: str
    adduct: str
    charge: int
    neutral_mass: float
    monoisotopic_mz: float
    peaks: Tuple[TheoryPeak, ...]
    retained_probability: float
    truncated: bool
    crosscheck_max_abs_diff: Optional[float]
    crosscheck_warning: Optional[str]
    budget: Dict[str, float]

    def peak_at(self, nominal_offset: int) -> Optional[TheoryPeak]:
        for peak in self.peaks:
            if peak.nominal_offset == nominal_offset:
                return peak
        return None

    def to_dict(self) -> dict:
        return {
            "candidate_key": self.candidate_key,
            "formula": self.formula,
            "adduct": self.adduct,
            "charge": self.charge,
            "neutral_mass": self.neutral_mass,
            "monoisotopic_mz": self.monoisotopic_mz,
            "peaks": [p.to_dict() for p in self.peaks],
            "retained_probability": self.retained_probability,
            "truncated": self.truncated,
            "crosscheck": {
                "max_abs_diff": self.crosscheck_max_abs_diff,
                "warning": self.crosscheck_warning,
            },
            "budget": dict(self.budget),
        }


# ---------------------------------------------------------------------------
# Isotope constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Isotope:
    number: int
    mass: float
    abundance: float
    nominal_shift: int
    exact_shift: float


def _element_isotopes(element: str, isotope_threshold: float) -> List[_Isotope]:
    """Isotopes of one element, referenced to the monoisotopic (lightest).

    Mirrors pyteomics' convention: the plain element symbol stands for the
    monoisotopic isotope; every other isotope with natural abundance at or
    above ``isotope_threshold`` is retained explicitly.
    """
    table = _pm.nist_mass.get(element)
    if not table:
        raise KeyError(f"no isotope table for element {element!r}")
    mono_mass, _ = table[0]
    heavy = [
        _Isotope(
            number=number,
            mass=mass,
            abundance=abundance,
            # Standard nominal mass defect rounding (floor(x + 0.5)): the
            # 17O shift is 0.49995 u, which banker's rounding would collapse
            # onto the monoisotopic bin.
            nominal_shift=int(math.floor(mass - mono_mass + 0.5)),
            exact_shift=mass - mono_mass,
        )
        for number, (mass, abundance) in table.items()
        if number and abundance > 0.0 and abundance >= isotope_threshold
        and mass > mono_mass
    ]
    heavy.sort(key=lambda iso: iso.nominal_shift)
    return heavy


def _monoisotope_abundance(element: str) -> float:
    table = _pm.nist_mass[element]
    mono_mass, _placeholder = table[0]
    # The real natural abundance of the monoisotopic isotope is stored under
    # its isotope number; find the isotope with the monoisotopic mass.
    for number, (mass, abundance) in table.items():
        if number and abs(mass - mono_mass) < 1e-12:
            return abundance
    return 1.0


# ---------------------------------------------------------------------------
# Bounded dynamic program
# ---------------------------------------------------------------------------


_FFT_CONVOLVE_THRESHOLD = 256


def _convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Direct convolution for small arrays, FFT for large ones."""
    if min(a.size, b.size) >= _FFT_CONVOLVE_THRESHOLD:
        from scipy.signal import fftconvolve

        return fftconvolve(a, b)
    return np.convolve(a, b)


def _combine_moments(
    p1: np.ndarray,
    w1: np.ndarray,
    p2: np.ndarray,
    w2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convolve two independent offset distributions with mass moments."""
    return (
        _convolve(p1, p2),
        _convolve(w1, p2) + _convolve(p1, w2),
    )


def _power_distribution(
    single_p: np.ndarray,
    single_w: np.ndarray,
    count: int,
    *,
    deadline: Optional[float] = None,
    max_states: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Distribution of ``count`` independent, identical atoms.

    Returns ``(P, W)`` indexed by nominal offset: ``P[k]`` is aggregate
    probability at offset ``k``, ``W[k]`` its probability-weighted exact mass
    shift.  The power is taken by binary exponentiation (O(log count)
    convolutions of bounded arrays) rather than O(count) multiplications, so
    cost never approaches the combinatorial isotopologue count and a deadline
    is checked only O(log count) times.
    """
    result_p = np.ones(1)
    result_w = np.zeros(1)
    base_p, base_w = single_p, single_w
    remaining = count
    while remaining:
        if max_states is not None and base_p.size > max_states:
            raise BudgetExceededError(
                f"isotope DP needs a {base_p.size}-bin array, above the "
                f"{max_states} state budget",
                kind="states",
                limit=max_states,
                used=base_p.size,
            )
        if deadline is not None and time.perf_counter() > deadline:
            raise BudgetExceededError(
                "isotope DP exceeded its time budget while taking an "
                "element power",
                kind="time",
            )
        if remaining & 1:
            result_p, result_w = _combine_moments(
                result_p, result_w, base_p, base_w
            )
        remaining >>= 1
        if remaining:
            base_p, base_w = _combine_moments(
                base_p, base_w, base_p, base_w
            )
    return result_p, result_w


def _element_distribution(
    element: str,
    count: int,
    budget: Budget,
    *,
    deadline: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    heavies = _element_isotopes(element, budget.isotope_threshold)
    p0 = _monoisotope_abundance(element)
    max_shift = max((iso.nominal_shift for iso in heavies), default=0)
    single_p = np.zeros(max_shift + 1)
    single_w = np.zeros(max_shift + 1)
    single_p[0] = p0
    for iso in heavies:
        # Several isotopes may share a nominal bin; probability and weighted
        # mass add.
        single_p[iso.nominal_shift] += iso.abundance
        single_w[iso.nominal_shift] += iso.abundance * iso.exact_shift
    return _power_distribution(
        single_p, single_w, count,
        deadline=deadline, max_states=budget.max_states,
    )


# Binomial counts are diagnostic only and explode combinatorially; exact
# integer labels are only tracked for small nominal supports.
_COUNTS_MAX_BINS = 256


def _power_counts(single_counts: np.ndarray, count: int) -> np.ndarray:
    """Number of distinct isotope-label assignments per nominal offset.

    Combinatorial counterpart of :func:`_power_distribution`.  Counts are
    held in float64 for speed and only computed while the support stays within
    :data:`_COUNTS_MAX_BINS`; beyond it an all-``inf`` array signals
    "uncounted" (reported as ``n_states = -1``).
    """
    result = np.ones(1)
    base = np.asarray(single_counts, dtype=float)
    remaining = count
    while remaining:
        if max(result.size, base.size) > _COUNTS_MAX_BINS:
            # Further convolution would only widen the uncounted tail.
            width = result.size + base.size - 1
            result = np.full(width, np.inf)
            break
        if remaining & 1:
            result = np.convolve(result, base)
        remaining >>= 1
        if remaining:
            base = np.convolve(base, base)
    return result


def _element_isotope_counts(element: str, count: int, budget: Budget) -> np.ndarray:
    heavies = _element_isotopes(element, budget.isotope_threshold)
    max_shift = max((iso.nominal_shift for iso in heavies), default=0)
    support = max_shift * count + 1
    if support > _COUNTS_MAX_BINS:
        return np.full(support, np.inf)
    single = np.zeros(max_shift + 1)
    single[0] = 1
    for iso in heavies:
        single[iso.nominal_shift] = 1
    return _power_counts(single, count)


def build_envelope(candidate: Candidate, budget: Optional[Budget] = None) -> TheoryEnvelope:
    """Generate the probability-truncated isotope envelope for a candidate.

    Resource use is bounded by :class:`Budget`; failure to complete inside
    the budget raises :class:`BudgetExceededError` rather than returning a
    silently truncated envelope.
    """
    budget = budget or Budget()
    charge = candidate.charge if candidate.charge is not None else candidate.adduct_obj.charge
    ion_comp = candidate.ion_composition
    m_mono = ion_monoisotopic_mz(candidate)

    started = time.perf_counter()
    total_p = np.ones(1)
    total_w = np.zeros(1)
    total_c = np.ones(1)
    origin = 0  # nominal offset represented by array index 0

    for element, count in sorted(ion_comp.items()):
        deadline = started + budget.time_seconds
        p, w = _element_distribution(
            element, count, budget, deadline=deadline
        )
        if p.size > budget.max_states:
            raise BudgetExceededError(
                f"element {element!r} x{count} needs {p.size} nominal bins, "
                f"above the {budget.max_states} state budget",
                kind="states",
                limit=budget.max_states,
                used=p.size,
            )
        c_arr = _element_isotope_counts(element, count, budget)

        new_p = _convolve(total_p, p)
        new_w = _convolve(total_w, p) + _convolve(total_p, w)
        if np.isinf(total_c).any() or np.isinf(c_arr).any():
            # Counts are diagnostic and have overflowed: propagate the flag
            # without doing arithmetic with infinities.
            new_c = np.full(new_p.size, np.inf)
        else:
            new_c = np.convolve(total_c, c_arr)
        # FFT convolution can leave negligible negative tails; probabilities
        # and counts are non-negative quantities.
        np.maximum(new_p, 0.0, out=new_p)
        np.maximum(new_c, 0.0, out=new_c)

        # Element distributions are complete: a finished term can only be
        # multiplied by probabilities <= 1 from remaining elements, and
        # contributions from distinct discarded bins into one future bin sum
        # to at most the threshold (the remaining-element slice sums <= 1).
        alive = new_p >= budget.min_abundance
        if not alive.all():
            indices = np.flatnonzero(alive)
            if indices.size == 0:
                raise BudgetExceededError(
                    f"all isotope bins for {candidate.key} fell below "
                    f"{budget.min_abundance}",
                    kind="empty",
                )
            lo, hi = int(indices[0]), int(indices[-1]) + 1
            new_p = new_p[lo:hi]
            new_w = new_w[lo:hi]
            new_c = new_c[lo:hi]
            origin += lo
        total_p, total_w, total_c = new_p, new_w, new_c

        if total_p.size > budget.max_states:
            raise BudgetExceededError(
                f"envelope DP for {candidate.key} holds {total_p.size} bins, "
                f"above the {budget.max_states} state budget",
                kind="states",
                limit=budget.max_states,
                used=total_p.size,
            )
        if time.perf_counter() - started > budget.time_seconds:
            raise BudgetExceededError(
                f"isotope DP for {candidate.key} exceeded "
                f"{budget.time_seconds}s",
                kind="time",
                limit=budget.time_seconds,
                used=time.perf_counter() - started,
            )

    base_abundance = float(total_p.max())
    peaks: List[TheoryPeak] = []
    for k in range(total_p.size):
        probability = float(total_p[k])
        if probability < budget.min_abundance:
            continue
        exact_mass_shift = float(total_w[k] / total_p[k])
        state_count = total_c[k]
        n_states = int(round(state_count)) if np.isfinite(state_count) else -1
        peaks.append(
            TheoryPeak(
                nominal_offset=origin + k,
                mz=m_mono + exact_mass_shift / abs(charge),
                abundance=probability,
                relative_abundance=probability / base_abundance,
                n_states=n_states,
            )
        )
        if len(peaks) > budget.max_peaks:
            raise BudgetExceededError(
                f"aggregated envelope for {candidate.key} exceeded "
                f"{budget.max_peaks} peaks",
                kind="peaks",
                limit=budget.max_peaks,
                used=len(peaks),
            )

    if not peaks:
        raise BudgetExceededError(
            f"no isotope peaks survived for {candidate.key}; check thresholds",
            kind="empty",
        )

    retained = float(total_p.sum())
    crosscheck_diff, crosscheck_warning = _crosscheck_small(
        ion_comp,
        charge=abs(charge),
        monoisotopic_mz=m_mono,
        reference={p.nominal_offset: p for p in peaks},
        budget=budget,
    )
    elapsed = time.perf_counter() - started
    return TheoryEnvelope(
        candidate_key=candidate.key,
        formula=candidate.formula,
        adduct=candidate.adduct,
        charge=charge,
        neutral_mass=neutral_mass(candidate.composition),
        monoisotopic_mz=m_mono,
        peaks=tuple(peaks),
        retained_probability=retained,
        truncated=False,
        crosscheck_max_abs_diff=crosscheck_diff,
        crosscheck_warning=crosscheck_warning,
        budget={
            "bins": len(peaks),
            "seconds": elapsed,
            "max_array": float(total_p.size),
        },
    )


# ---------------------------------------------------------------------------
# Independent cross-checks
# ---------------------------------------------------------------------------


def pyteomics_exact_envelope(
    composition: Dict[str, int],
    *,
    overall_threshold: float = 1e-5,
    isotope_threshold: float = 1e-5,
) -> Dict[int, Tuple[float, float]]:
    """Aggregate pyteomics' exact isotopologues into nominal-offset bins.

    Returns ``offset -> (probability, probability-weighted exact mass shift
    in u relative to the monoisotopic ion)``.  Only safe for modest atom
    counts; callers guard molecule size.
    """
    mono_atom_mass = float(
        _pm.calculate_mass(_pm.Composition(dict(composition)))
    )
    bins: Dict[int, List[Tuple[float, float]]] = {}
    for isotopologue, probability in _pm.isotopologues(
        _pm.Composition(dict(composition)),
        report_abundance=True,
        overall_threshold=overall_threshold,
        isotope_threshold=isotope_threshold,
    ):
        atom_mass = float(_pm.calculate_mass(isotopologue))
        offset = int(round(atom_mass - mono_atom_mass))
        bins.setdefault(offset, []).append(
            (float(probability), atom_mass - mono_atom_mass)
        )
    result: Dict[int, Tuple[float, float]] = {}
    for offset, members in bins.items():
        weight = sum(p for p, _ in members)
        shift = sum(p * s for p, s in members) / weight
        result[offset] = (weight, shift)
    return result


def _crosscheck_small(
    composition: Dict[str, int],
    *,
    charge: int,
    monoisotopic_mz: float,
    reference: Dict[int, TheoryPeak],
    budget: Budget,
    max_atoms: int = 80,
    abundance_tolerance: float = 0.02,
    mass_tolerance_da: float = 2e-4,
) -> Tuple[Optional[float], Optional[str]]:
    """Validate the DP against pyteomics' exact enumeration on small molecules.

    Both bin *abundances* and bin *mass centroids* are compared.  Large
    formulas are skipped (exact enumeration is not resource-bounded); that is
    a property of the reference method, not of the envelope.
    """
    n_atoms = sum(composition.values())
    if n_atoms > max_atoms:
        return None, (
            f"exact-enumeration cross-check skipped: {n_atoms} atoms exceed "
            f"the {max_atoms}-atom safety limit"
        )
    try:
        exact = pyteomics_exact_envelope(
            composition,
            overall_threshold=max(budget.min_abundance, 1e-5),
            isotope_threshold=budget.isotope_threshold,
        )
    except (OverflowError, ValueError, ArithmeticError) as exc:
        return None, f"exact-enumeration cross-check failed: {exc}"

    base_weight = exact.get(0, (None, None))[0]
    if not base_weight:
        return None, "exact cross-check produced no base peak"
    abundance_diffs = []
    mass_diffs = []
    for offset, peak in reference.items():
        if offset not in exact:
            abundance_diffs.append(peak.relative_abundance)
            continue
        exact_weight, exact_shift = exact[offset]
        abundance_diffs.append(
            abs(exact_weight / base_weight - peak.relative_abundance)
        )
        # DP peak m/z is reported in ion space; map back to neutral mass shift.
        dp_shift = (peak.mz - monoisotopic_mz) * charge
        mass_diffs.append(abs(exact_shift - dp_shift))
    max_diff = max(abundance_diffs) if abundance_diffs else None
    max_mass_diff = max(mass_diffs) if mass_diffs else 0.0
    if max_diff is not None and max_diff > abundance_tolerance:
        return max_diff, (
            f"DP envelope differs from pyteomics exact enumeration by up "
            f"to {max_diff:.4f} relative abundance "
            f"(tolerance {abundance_tolerance})"
        )
    if max_mass_diff > mass_tolerance_da:
        return max_diff, (
            f"DP bin centroids differ from exact enumeration by up "
            f"{max_mass_diff:.2e} u (tolerance {mass_tolerance_da:.0e} u)"
        )
    return max_diff, None


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


@dataclass
class TheoryCache:
    """Cache of theoretical envelopes, keyed by chemistry + mass-table + budget.

    The cache contains *no* measurement, calibration or observed peak
    identity, so it is safe to reuse across runs, samples and calibration
    versions.  Adding or withdrawing a candidate never mutates cached
    envelopes for the others.
    """

    budget: Budget = field(default_factory=Budget)
    _entries: Dict[str, TheoryEnvelope] = field(default_factory=dict, repr=False)
    _failures: Dict[str, Tuple[str, str, str]] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @staticmethod
    def _mass_table_token() -> str:
        payload = {}
        for element in sorted(_pm.nist_mass):
            if not isinstance(element, str):
                continue
            table = _pm.nist_mass[element]
            payload[element] = [
                [number, data[0], data[1]]
                for number, data in sorted(table.items())
                if data[1] > 0.0
            ]
        blob = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def cache_key(self, candidate: Candidate) -> str:
        fingerprint = json.dumps(
            {
                "ion": candidate.ion_composition,
                "charge": candidate.charge
                if candidate.charge is not None
                else candidate.adduct_obj.charge,
                "min_abundance": self.budget.min_abundance,
                "isotope_threshold": self.budget.isotope_threshold,
                "mass_table": self._mass_table_token(),
            },
            sort_keys=True,
        )
        return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()

    def get_or_build(
        self, candidate: Candidate
    ) -> Tuple[Optional[TheoryEnvelope], Optional[Tuple[str, str, str]]]:
        """Return ``(envelope, failure)``; either outcome is cached.

        ``failure`` is ``None`` or ``(code, message, kind)`` where ``kind`` is
        the specific budget that was exceeded.
        """
        key = self.cache_key(candidate)
        with self._lock:
            if key in self._entries:
                return self._entries[key], None
            if key in self._failures:
                return None, self._failures[key]
        try:
            envelope = build_envelope(candidate, self.budget)
        except BudgetExceededError as exc:
            failure = ("THEORY_BUDGET", str(exc), exc.kind)
            with self._lock:
                self._failures[key] = failure
            return None, failure
        with self._lock:
            self._entries[key] = envelope
        return envelope, None

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "envelopes": len(self._entries),
                "failures": len(self._failures),
            }

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._failures.clear()
