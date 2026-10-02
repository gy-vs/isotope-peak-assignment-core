"""Theoretical isotope distributions with a hard resource budget.

The natural isotope envelope of a composition is the product (convolution) of
per-element multinomial distributions.  Enumerating every isotopologue explodes
combinatorially (glucose already has 49 raw isotopologues; a 30-carbon molecule
has thousands) even though nearly all of them carry negligible abundance.

This engine convolves distributions with *abundance pruning*:

* per-element powers are built by exponentiation-by-squaring,
* after every multiplication, peaks below ``min_rel_abundance`` of the current
  strongest peak are discarded,
* the monoisotopic peak is always force-kept,
* the output is capped at ``max_peaks``,
* the total natural probability lost to pruning is accounted exactly enough to
  decide whether the result is still reliable.

Runtime and memory are therefore bounded by the *shape* of the envelope, not by
the atom count: adding a thousand identical carbons is no worse than adding six.

Masses are kept as exact isotope masses (fine structure like 13C vs 2H is
resolved); only numerical duplicates are merged at a 1e-9 Da bin width.
"""

from __future__ import annotations

from dataclasses import dataclass

from .elements import ELEMENTS
from .errors import BudgetError
from .formula import Formula

_MERGE_DECIMALS = 9  # bin width 1e-9 Da: merges only numerical duplicates


@dataclass(frozen=True)
class IsotopePeak:
    """One peak of a theoretical isotope envelope (neutral masses, Da)."""

    mass: float
    probability: float          # raw natural probability (sum <~ 1)
    relative_abundance: float   # normalized to strongest peak = 1.0
    composition: tuple[tuple[str, int, int], ...]  # (element, A, count)
    is_monoisotopic: bool
    index: int


@dataclass(frozen=True)
class IsotopeBudget:
    """Resource limits for an isotope calculation.

    Attributes:
        min_rel_abundance: prune peaks below this fraction of the strongest
            current peak after every convolution.
        max_peaks: hard cap on emitted peaks (the strongest survive).
        hard_pair_limit: hard cap on convolution pair count at any one step;
            crossing it raises :class:`BudgetError` rather than allocating.
        accepted_dropped_fraction: if pruning/capping loses more than this
            fraction of total natural probability, the result is flagged
            ``budget_exceeded`` instead of being trusted.
    """

    min_rel_abundance: float = 1.0e-5
    max_peaks: int = 300
    hard_pair_limit: int = 2_000_000
    accepted_dropped_fraction: float = 1.0e-2

    def __post_init__(self):
        if not 0.0 < self.min_rel_abundance < 1.0:
            raise ValueError("min_rel_abundance must lie in (0, 1)")
        if self.max_peaks < 2:
            raise ValueError("max_peaks must be >= 2")
        if self.hard_pair_limit < self.max_peaks:
            raise ValueError("hard_pair_limit must be >= max_peaks")
        if not 0.0 < self.accepted_dropped_fraction < 1.0:
            raise ValueError("accepted_dropped_fraction must lie in (0, 1)")


@dataclass(frozen=True)
class IsotopeCluster:
    """Result of an envelope calculation."""

    peaks: tuple[IsotopePeak, ...]
    dropped_fraction: float
    truncated: bool                 # soft cap was applied
    budget: IsotopeBudget
    max_intermediate: int = 0
    operations: int = 0

    @property
    def budget_exceeded(self) -> bool:
        """True when too much natural probability was lost to be reliable."""
        return self.dropped_fraction > self.budget.accepted_dropped_fraction


@dataclass
class _Diag:
    max_intermediate: int = 0
    operations: int = 0


# A live distribution maps binned mass -> [probability, label tuple].
_Label = tuple[tuple[int, int], ...]  # sparse (channel -> count)


def isotope_distribution(formula: Formula,
                         budget: IsotopeBudget | None = None) -> IsotopeCluster:
    """Compute the pruned isotope envelope of a neutral ``formula``."""

    budget = budget or IsotopeBudget()
    diag = _Diag()
    composition = formula.as_dict()

    # One channel per non-monoisotopic isotope that actually occurs.
    channels: list[tuple[str, int]] = []
    for sym in sorted(composition):
        el = ELEMENTS[sym]
        for iso in el.isotopes:
            if iso.mass_number != el.monoisotope.mass_number:
                channels.append((sym, iso.mass_number))
    channel_index = {key: i for i, key in enumerate(channels)}

    retained_product = 1.0
    combined: dict[float, list] = {0.0: [1.0, ()]}

    for sym in sorted(composition):
        el = ELEMENTS[sym]
        count = composition[sym]
        single = _single_atom(el.symbol, el, channel_index)
        # Element powers are pruned against an *absolute* abundance floor
        # (the budget threshold, which can only be loosened by later
        # multiplication), never against a local relative maximum -- doing the
        # latter during exponentiation would discard isotope channels that are
        # weak at low atom counts but become major peaks of the final envelope
        # (e.g. M+1 of a C5000 molecule).
        powered, ret = _power(single, count, budget, diag)
        retained_product *= ret
        combined, ret = _convolve_and_prune(combined, powered, budget, diag)
        retained_product *= ret

    # Finalize: cap, attach labels, sort by mass.
    # Surviving natural probability is measured directly from the final
    # distribution (the per-step retained factors telescope to this value);
    # ``cap_dropped`` is the absolute probability removed by the hard cap.
    total = sum(row[0] for row in combined.values())
    capped, cap_dropped = _cap(combined, budget)
    dropped_fraction = _clamp01(1.0 - total + cap_dropped)

    max_prob = max(row[0] for row in capped.values()) or 1.0
    labels_lookup = channels
    raw = []
    for mass_key, (prob, label) in capped.items():
        comp = tuple((labels_lookup[ch][0], labels_lookup[ch][1], n)
                     for ch, n in label)
        raw.append((mass_key, prob, prob / max_prob, comp, len(label) == 0))
    raw.sort(key=lambda r: r[0])
    peaks = tuple(
        IsotopePeak(mass=mass, probability=prob, relative_abundance=rel,
                    composition=comp, is_monoisotopic=mono, index=i)
        for i, (mass, prob, rel, comp, mono) in enumerate(raw)
    )
    return IsotopeCluster(
        peaks=peaks,
        dropped_fraction=dropped_fraction,
        truncated=cap_dropped > 0.0,
        budget=budget,
        max_intermediate=diag.max_intermediate,
        operations=diag.operations,
    )


# --- internals --------------------------------------------------------------

def _single_atom(sym: str, el, channel_index) -> dict[float, list]:
    out: dict[float, list] = {}
    mono_a = el.monoisotope.mass_number
    for iso in el.isotopes:
        key = round(iso.mass, _MERGE_DECIMALS)
        if iso.mass_number == mono_a:
            label: _Label = ()
        else:
            label = ((channel_index[(sym, iso.mass_number)], 1),)
        if key in out:  # numerically identical isotope masses
            out[key][0] += iso.abundance
        else:
            out[key] = [iso.abundance, label]
    return out


def _power(single: dict[float, list], exponent: int,
           budget: IsotopeBudget, diag: _Diag):
    """Distribution of ``exponent`` independent atoms, via squaring.

    Uses an absolute probability floor (the configured relative threshold,
    which is a *lower* bound for any post-combination relative value).  No
    peak above that floor is ever dropped here, and the monoisotopic peak is
    force-kept regardless.
    """
    floor = budget.min_rel_abundance
    result = {0.0: [1.0, ()]}
    base = single
    retained = 1.0
    k = exponent
    while k:
        if k & 1:
            result, ret = _convolve_floor(result, base, floor, budget, diag)
            retained *= ret
        k >>= 1
        if k:
            base, ret = _convolve_floor(base, base, floor, budget, diag)
            retained *= ret
    return result, retained


def _convolve_and_prune(a: dict[float, list], b: dict[float, list],
                        budget: IsotopeBudget, diag: _Diag):
    """Cross-element convolution followed by relative-abundance pruning."""
    out = _convolve(a, b, budget, diag)
    return _prune_relative(out, budget)


def _convolve_floor(a: dict[float, list], b: dict[float, list], floor: float,
                    budget: IsotopeBudget, diag: _Diag):
    """Convolution with an *absolute* probability floor (element powers)."""
    out = _convolve(a, b, budget, diag)
    total = sum(row[0] for row in out.values())
    kept = {m: row for m, row in out.items()
            if row[0] >= floor or not row[1]}
    if len(kept) > budget.max_peaks:
        # Should be extremely rare; keep strongest + monoisotopic.
        ordered = sorted(out.items(), key=lambda kv: kv[1][0], reverse=True)
        kept = dict(ordered[:budget.max_peaks - 1])
        mono = next((kv for kv in ordered if not kv[1][1]), None)
        if mono is not None and mono[0] not in kept:
            kept[mono[0]] = mono[1]
    kept_mass = sum(row[0] for row in kept.values())
    return kept, kept_mass / total if total else 1.0


def _convolve(a: dict[float, list], b: dict[float, list],
              budget: IsotopeBudget, diag: _Diag) -> dict[float, list]:
    pairs = len(a) * len(b)
    diag.operations += 1
    diag.max_intermediate = max(diag.max_intermediate, pairs)
    if pairs > budget.hard_pair_limit:
        raise BudgetError(
            "isotope convolution would exceed the hard pair budget; raise "
            "min_rel_abundance / lower max_peaks or split the candidate",
            limit=budget.hard_pair_limit, observed=pairs)
    out: dict[float, list] = {}
    for ma, (pa, la) in a.items():
        for mb, (pb, lb) in b.items():
            key = round(ma + mb, _MERGE_DECIMALS)
            prob = pa * pb
            existing = out.get(key)
            if existing is None:
                out[key] = [prob, _merge_labels(la, lb)]
            else:
                existing[0] += prob
    return out


def _merge_labels(la: _Label, lb: _Label) -> _Label:
    if not la:
        return lb
    if not lb:
        return la
    merged: dict[int, int] = dict(la)
    for ch, n in lb:
        merged[ch] = merged.get(ch, 0) + n
    return tuple(sorted(merged.items()))


def _prune_relative(dist: dict[float, list], budget: IsotopeBudget):
    """Drop peaks below threshold relative to the strongest (mono force-kept)."""
    total = sum(row[0] for row in dist.values())
    floor = max(row[0] for row in dist.values()) * budget.min_rel_abundance
    items = [(m, row) for m, row in dist.items()
             if row[0] >= floor or not row[1]]
    items.sort(key=lambda item: item[1][0], reverse=True)
    if len(items) > budget.max_peaks:
        # monoisotopic (empty label) is guaranteed a slot
        items = items[:budget.max_peaks - 1]
        mono = next(((m, row) for m, row in dist.items() if not row[1]), None)
        if mono is not None and mono not in items:
            items.append(mono)
    kept = dict(items)
    kept_mass = sum(row[0] for row in kept.values())
    return kept, kept_mass / total if total else 1.0


def _cap(dist: dict[float, list], budget: IsotopeBudget):
    """Final hard cap on peak count. Returns (kept, dropped probability)."""
    if len(dist) <= budget.max_peaks:
        return dist, 0.0
    ordered = sorted(dist.items(), key=lambda item: item[1][0], reverse=True)
    kept_items = ordered[:budget.max_peaks - 1]
    mono_item = next((item for item in ordered if not item[1][1]), None)
    if mono_item is not None and mono_item not in kept_items:
        kept_items.append(mono_item)
    kept = dict(kept_items)
    kept_p = sum(row[0] for row in kept.values())
    total_p = sum(row[0] for row in dist.values())
    return kept, max(0.0, total_p - kept_p)


def _clamp01(x: float) -> float:
    return min(1.0, max(0.0, x))
