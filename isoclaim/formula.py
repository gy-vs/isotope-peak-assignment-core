"""Chemical formulas, adducts and candidate definitions.

A :class:`Candidate` is the immutable identity a user submits: a neutral
molecular formula, an adduct and a charge state.  Everything that depends only
on the *theoretical* chemistry of a candidate (neutral mass, isotope envelope)
is derived from these objects and can be cached; nothing here knows anything
about an instrument run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional

from pyteomics import mass as _pm

__all__ = [
    "InvalidFormulaError",
    "InvalidAdductError",
    "Adduct",
    "Candidate",
    "KNOWN_ELEMENTS",
    "parse_formula",
    "formula_string",
    "canonical_formula",
    "default_adduct_registry",
]


# Elements present in the bundled pyteomics NIST mass table.
KNOWN_ELEMENTS: FrozenSet[str] = frozenset(
    symbol for symbol in _pm.nist_mass.keys() if isinstance(symbol, str)
)

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(-?\d*)")
# Moiety inside an adduct: leading stoichiometry (``2Na``, ``2H``, ``NH4``).
_ADDUCT_MOIETY_TOKEN = re.compile(r"(\d*)([A-Z][a-z]?)(\d*)")
# Adduct grammar: [nM+X]z+  /  [nM-X]z-  (charge 1 may omit the number)
_ADDUCT = re.compile(
    r"^\s*\[?\s*(\d*)\s*M\s*([+\-])\s*([0-9A-Za-z]+?)\s*\]?\s*"
    r"(\d*)\s*([+\-])\s*$"
)

# Electron mass in u (CODATA), used to turn neutral-atom ion mass into m/z.
ELECTRON_MASS_U = 5.48579909065e-4


class InvalidFormulaError(ValueError):
    """Raised when a molecular formula string cannot be parsed."""


class InvalidAdductError(ValueError):
    """Raised when an adduct specification is inconsistent or unknown."""


def parse_formula(text: str) -> Dict[str, int]:
    """Parse a plain chemical formula (no parentheses) into element counts.

    Examples: ``"C6H12O6"`` -> ``{"C": 6, "H": 12, "O": 6}``.

    Element symbols must exist in the mass table and every atom count must be
    a positive integer.  Whitespace is rejected; isotope brackets, dots and
    parentheses are not part of the supported grammar.
    """
    if not isinstance(text, str) or not text:
        raise InvalidFormulaError("formula must be a non-empty string")
    counts: Dict[str, int] = {}
    pos = 0
    for match in _FORMULA_TOKEN.finditer(text):
        if match.start() != pos:
            raise InvalidFormulaError(
                f"could not parse formula {text!r} at column {pos + 1}"
            )
        element, number_text = match.group(1), match.group(2)
        if element not in KNOWN_ELEMENTS:
            raise InvalidFormulaError(
                f"unknown element {element!r} in formula {text!r}"
            )
        number = int(number_text) if number_text else 1
        if number <= 0:
            raise InvalidFormulaError(
                f"non-positive count {number} for {element} in {text!r}"
            )
        counts[element] = counts.get(element, 0) + number
        pos = match.end()
    if pos != len(text):
        raise InvalidFormulaError(f"could not fully parse formula {text!r}")
    if not counts:
        raise InvalidFormulaError(f"no element counts found in {text!r}")
    return counts


def formula_string(counts: Dict[str, int]) -> str:
    """Render an element-count mapping as a formula string."""
    return "".join(
        f"{element}{count}" if count != 1 else element
        for element, count in counts.items()
    )


def canonical_formula(text: str) -> str:
    """Parse and re-render a formula in Hill order (C, H, then alphabetic)."""
    counts = parse_formula(text)
    order = sorted(counts, key=lambda e: (e != "C", e != "H", e))
    return formula_string({e: counts[e] for e in order})


@dataclass(frozen=True)
class Adduct:
    """An ionizing adduct, e.g. ``[M+H]+`` or ``[M+2Na]2+``.

    Attributes:
        label: Human-readable label such as ``"[M+H]+"``.
        delta: Element-count composition added (positive) or removed
            (negative) from the neutral molecule, e.g. ``{"H": 1}``.
        charge: Signed charge, non-zero for an observable ion.
        n: Number of neutral monomers (only ``1`` is constructed by the
            built-in registry; arbitrary oligomer adducts are out of scope).
    """

    label: str
    delta: Dict[str, int] = field(default_factory=dict)
    charge: int = 1
    n: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.charge, int) or self.charge == 0:
            raise InvalidAdductError("adduct charge must be a non-zero int")
        if self.n != 1:
            raise InvalidAdductError("only single-monomer adducts (n=1) supported")
        for element, count in self.delta.items():
            if element not in KNOWN_ELEMENTS:
                raise InvalidAdductError(f"unknown adduct element {element!r}")
            if not isinstance(count, int) or count == 0:
                raise InvalidAdductError(
                    f"adduct count for {element!r} must be a non-zero int"
                )

    def __str__(self) -> str:
        return self.label

    def apply(self, neutral: Dict[str, int]) -> Dict[str, int]:
        """Return the elemental composition of the ionized molecule."""
        ion = dict(neutral)
        for element, delta in self.delta.items():
            new_count = ion.get(element, 0) + delta
            if new_count < 0:
                raise InvalidAdductError(
                    f"adduct {self.label} would remove more {element} than the "
                    f"molecule contains ({ion.get(element, 0)} available)"
                )
            ion[element] = new_count
        return ion


def _parse_adduct_literal(text: str) -> Adduct:
    match = _ADDUCT.match(text)
    if not match:
        raise InvalidAdductError(f"cannot parse adduct notation {text!r}")
    n_text, sign, moiety, z_text, z_sign = match.groups()
    n = int(n_text) if n_text else 1
    # Parse the adduct moiety: a stoichiometry factor, an element and an
    # optional isotope/subscript count, e.g. "2Na", "H", "NH4".
    delta_formula: Dict[str, int] = {}
    cursor = 0
    for token in _ADDUCT_MOIETY_TOKEN.finditer(moiety):
        if token.start() != cursor:
            raise InvalidAdductError(f"cannot parse adduct moiety {moiety!r}")
        factor_text, element, count_text = token.groups()
        if not element:
            raise InvalidAdductError(f"cannot parse adduct moiety {moiety!r}")
        factor = int(factor_text) if factor_text else 1
        count = int(count_text) if count_text else 1
        delta_formula[element] = delta_formula.get(element, 0) + factor * count
        cursor = token.end()
    if cursor != len(moiety) or not delta_formula:
        raise InvalidAdductError(f"cannot parse adduct moiety {moiety!r}")
    magnitude = int(z_text) if z_text else 1
    charge = magnitude if z_sign == "+" else -magnitude
    if sign == "-":
        delta_formula = {e: -c for e, c in delta_formula.items()}
    return Adduct(label=text.strip().replace(" ", ""), delta=delta_formula,
                  charge=charge, n=n)


def default_adduct_registry() -> Dict[str, Adduct]:
    """Return a fresh copy of the built-in adduct catalog."""
    labels = [
        "[M+H]+",
        "[M+Na]+",
        "[M+K]+",
        "[M+NH4]+",
        "[M-H]-",
        "[M+Cl]-",
        "[2M+H]+",
        "[2M-H]-",
    ]
    registry: Dict[str, Adduct] = {}
    for label in labels:
        if label.startswith("[2M"):
            # Oligomer adducts are intentionally unsupported; keep the
            # registry honest rather than silently mis-handling them.
            continue
        registry[label] = _parse_adduct_literal(label)
    return registry


@dataclass(frozen=True)
class Candidate:
    """A molecular-formula hypothesis in one specific ion form.

    Two candidates with the same formula but different adduct/charge are
    distinct hypotheses and may compete for the same observed peaks.
    """

    formula: str
    adduct: str = "[M+H]+"
    charge: Optional[int] = None
    name: Optional[str] = None
    adduct_registry: Optional[Dict[str, Adduct]] = field(
        default=None, repr=False, hash=False, compare=False
    )

    def __post_init__(self) -> None:
        # Validate eagerly: an invalid candidate can never be a valid result.
        counts = parse_formula(self.formula)
        object.__setattr__(self, "formula", canonical_formula(self.formula))
        registry = self.adduct_registry or default_adduct_registry()
        adduct = registry.get(self.adduct)
        if adduct is None:
            # Allow ad-hoc adduct literals such as "[M+Li]+".
            adduct = _parse_adduct_literal(self.adduct)
        charge = self.charge if self.charge is not None else adduct.charge
        if charge != adduct.charge:
            raise InvalidAdductError(
                f"explicit charge {charge} disagrees with adduct "
                f"{self.adduct!r} charge {adduct.charge}"
            )
        # Raises InvalidAdductError if the molecule is too small for the loss.
        adduct.apply(counts)
        object.__setattr__(self, "_adduct", adduct)
        object.__setattr__(self, "_counts", counts)
        object.__setattr__(self, "adduct_registry", None)

    @property
    def composition(self) -> Dict[str, int]:
        """Element counts of the neutral formula."""
        return dict(self._counts)  # type: ignore[attr-defined]

    @property
    def adduct_obj(self) -> Adduct:
        return self._adduct  # type: ignore[attr-defined]

    @property
    def ion_composition(self) -> Dict[str, int]:
        """Element counts of the charged ion."""
        return self._adduct.apply(self._counts)  # type: ignore[attr-defined]

    @property
    def key(self) -> str:
        """Stable identity key, independent of the display name."""
        return f"{self.formula}|{self.adduct}|z{self.charge or self._adduct.charge}"

    @property
    def label(self) -> str:
        base = self.name or self.formula
        return f"{base} {self.adduct}"
