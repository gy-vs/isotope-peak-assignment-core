"""Chemical formulas: parsing, validation and exact masses.

A formula is either a dict-like composition (``{"C": 6, "H": 12, "O": 6}``) or
a standard condensed formula string supporting parentheses/brackets, e.g.
``"C6H12O6"``, ``"Ca(OH)2"``, ``"Mg3(Si4O10)(OH)2"``.

Invalid input raises :class:`~isoassign.errors.FormulaError` carrying the
offending expression/token -- it never silently becomes an empty composition.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .elements import ELEMENTS
from .errors import FormulaError


@dataclass(frozen=True)
class Formula:
    """An immutable elemental composition with non-negative integer counts."""

    composition: frozenset[tuple[str, int]]

    def __init__(self, spec: "str | Mapping[str, int] | Formula"):
        if isinstance(spec, Formula):
            counts = dict(spec.as_dict())
        elif isinstance(spec, str):
            counts = _parse(spec)
        elif isinstance(spec, Mapping):
            counts = _validate_mapping(spec)
        else:
            raise FormulaError(
                f"cannot build Formula from {type(spec).__name__!r}",
                expression=str(spec),
            )
        counts = {e: n for e, n in counts.items() if n != 0}
        if not counts:
            raise FormulaError("formula contains no atoms",
                               expression=str(spec))
        object.__setattr__(self, "composition", frozenset(counts.items()))

    # -- basic protocol ----------------------------------------------------
    def as_dict(self) -> dict[str, int]:
        return dict(self.composition)

    def __getitem__(self, element: str) -> int:
        for sym, n in self.composition:
            if sym == element:
                return n
        return 0

    def __contains__(self, element: object) -> bool:
        return any(sym == element for sym, _ in self.composition)

    def __iter__(self):
        return iter(self.as_dict())

    @property
    def atom_count(self) -> int:
        return sum(n for _, n in self.composition)

    def __add__(self, other: "Mapping[str, int] | Formula") -> "Formula":
        merged = self.as_dict()
        for sym, n in (other.composition if isinstance(other, Formula)
                       else other.items()):
            merged[sym] = merged.get(sym, 0) + n
        return _from_clean(merged)

    def __sub__(self, other: "Mapping[str, int] | Formula") -> "Formula":
        merged = self.as_dict()
        for sym, n in (other.composition if isinstance(other, Formula)
                       else other.items()):
            merged[sym] = merged.get(sym, 0) - n
            if merged[sym] < 0:
                raise FormulaError(
                    f"cannot remove {n} {sym}: composition would be negative")
        return _from_clean(merged)

    # -- masses ------------------------------------------------------------
    @property
    def monoisotopic_mass(self) -> float:
        """Neutral monoisotopic exact mass (Da), atoms only -- no electron."""
        total = 0.0
        for sym, n in self.composition:
            total += ELEMENTS[sym].monoisotopic_mass * n
        return total

    # -- representation ----------------------------------------------------
    def hill_formula(self) -> str:
        """Canonical Hill-system string.

        Carbon present: C, then H, then everything else alphabetical.
        No carbon: every element alphabetical (H included).
        """
        counts = self.as_dict()
        parts: list[str] = []
        if "C" in counts:
            parts.append("C" + _sub(counts["C"]))
            if "H" in counts:
                parts.append("H" + _sub(counts["H"]))
            rest = (s for s in sorted(counts) if s not in ("C", "H"))
        else:
            rest = iter(sorted(counts))
        for sym in rest:
            parts.append(sym + _sub(counts[sym]))
        return "".join(parts)

    def __str__(self) -> str:
        return self.hill_formula()

    def __repr__(self) -> str:
        return f"Formula({self.hill_formula()!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Formula) and other.composition == self.composition

    def __hash__(self) -> int:
        return hash(self.composition)


def _sub(n: int) -> str:
    return "" if n == 1 else str(n)


def _from_clean(counts: dict[str, int]) -> Formula:
    """Internal constructor that skips parsing/validation of known-good data."""
    f = object.__new__(Formula)
    object.__setattr__(f, "composition",
                       frozenset((e, n) for e, n in counts.items() if n != 0))
    return f


def _validate_mapping(mapping: Mapping) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sym, n in mapping.items():
        if not isinstance(sym, str) or sym not in ELEMENTS:
            raise FormulaError(f"unknown element symbol: {sym!r}", token=str(sym))
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise FormulaError(
                f"atom count for {sym} must be a non-negative integer, got {n!r}")
        counts[sym] = counts.get(sym, 0) + n
    return counts


# --- string parser ----------------------------------------------------------

_OPEN = {"(", "["}
_CLOSE = {")", "]"}


def _parse(text: str) -> dict[str, int]:
    if not text or not text.strip():
        raise FormulaError("empty formula", expression=text)
    stack: list[dict[str, int]] = [{}]
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch in _OPEN:
            stack.append({})
            i += 1
            continue
        if ch in _CLOSE:
            if len(stack) == 1:
                raise FormulaError(f"unmatched {ch!r}", expression=text, token=ch)
            group = stack.pop()
            mult, i = _read_count(text, i + 1)
            for sym, cnt in group.items():
                stack[-1][sym] = stack[-1].get(sym, 0) + cnt * mult
            continue
        if ch.isupper():
            sym = ch
            i += 1
            if i < n and text[i].islower():
                sym += text[i]
                i += 1
            if sym not in ELEMENTS:
                raise FormulaError(f"unknown element symbol: {sym!r}",
                                   expression=text, token=sym)
            mult, i = _read_count(text, i)
            stack[-1][sym] = stack[-1].get(sym, 0) + mult
            continue
        raise FormulaError(f"unexpected character {ch!r}",
                           expression=text, token=ch)
    if len(stack) != 1:
        raise FormulaError("unclosed parenthesis", expression=text)
    counts = stack[0]
    if not counts:
        raise FormulaError("formula contains no atoms", expression=text)
    return counts


def _read_count(text: str, i: int) -> tuple[int, int]:
    start = i
    while i < len(text) and text[i].isdigit():
        i += 1
    if i == start:
        return 1, i
    value = int(text[start:i])
    if value == 0:
        raise FormulaError("zero atom multiplier", expression=text,
                           token=text[start:i])
    return value, i
