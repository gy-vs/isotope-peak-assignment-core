"""Element and isotope data.

Atomic isotope masses/abundances are taken from the NIST table shipped with
``pyteomics`` (``pyteomics.mass.mass.nist_mass``); pyteomics is therefore the
data source, while the cluster engine in :mod:`isoassign.isotopes` is our own.
Having one well-defined upstream table is what makes every reported theoretical
mass reproducible and independently checkable.

For every element the *monoisotopic* isotope is the most abundant stable
isotope (pyteomics uses the same convention: ``nist_mass[elem][0]`` mirrors the
most abundant entry).  This is the convention used by proteomics/MS tools and
matches ``pyteomics.mass.calculate_mass``.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyteomics.mass.mass import nist_mass as _nist

# CODATA 2018 constants (Da). Proton mass is also expressible as
# H_atom - electron + binding; the 1e-8 Da difference is negligible here.
ELECTRON_MASS = 0.000548579909065
PROTON_MASS = 1.007276466621

DATA_VERSION = "pyteomics-nist"


@dataclass(frozen=True)
class Isotope:
    """A single isotope of an element."""

    element: str
    mass_number: int
    mass: float
    abundance: float  # natural fractional abundance, [0, 1]

    @property
    def is_monoisotopic(self) -> bool:
        return _MONO_NUMBERS.get(self.element) == self.mass_number


@dataclass(frozen=True)
class Element:
    symbol: str
    isotopes: tuple[Isotope, ...]  # only abundance > 0, sorted by mass number
    monoisotope: Isotope
    average_mass: float

    @property
    def monoisotopic_mass(self) -> float:
        return self.monoisotope.mass


def _build_table() -> dict[str, Element]:
    table: dict[str, Element] = {}
    for symbol, entries in _nist.items():
        rows = [(n, m, a) for n, (m, a) in entries.items()
                if isinstance(n, int) and n != 0 and a > 0.0]
        if not rows:
            continue
        rows.sort(key=lambda r: r[0])
        isotopes = tuple(Isotope(symbol, n, m, a) for n, m, a in rows)
        # Most abundant isotope = monoisotope; ties resolved to lowest A.
        mono = min(isotopes, key=lambda iso: (-iso.abundance, iso.mass_number))
        avg = sum(iso.mass * iso.abundance for iso in isotopes)
        table[symbol] = Element(symbol, isotopes, mono, avg)
    return table


ELEMENTS: dict[str, Element] = _build_table()
_MONO_NUMBERS: dict[str, int] = {
    sym: el.monoisotope.mass_number for sym, el in ELEMENTS.items()
}


def element(symbol: str) -> Element:
    """Return the :class:`Element` for ``symbol`` or raise ``KeyError``."""
    try:
        return ELEMENTS[symbol]
    except KeyError:
        raise KeyError(f"unknown element symbol: {symbol!r}") from None
