"""Adducts and charge states.

An :class:`Adduct` describes how a neutral molecule becomes a detected ion:

* ``delta`` -- atoms added (positive counts) or removed (negative counts)
  relative to the neutral composition,
* ``charge`` -- signed integer charge ``z``.

Ion masses are computed from exact isotope masses of the full ion composition,
then converted to m/z with the electron-mass correction::

    m_ion(peak) = m_neutral_peak + sum(delta atoms, exact isotope masses)
                  - z * electron_mass
    m/z         = m_ion / |z|

The electron correction makes the [M+H]+ ion come out at
``M_neutral + proton_mass`` (H atom minus electron) rather than plus a full
hydrogen atom.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import AdductError
from .formula import Formula

# name -> (delta composition, signed charge)
_BUILTIN_ADDUCTS: dict[str, tuple[dict[str, int], int]] = {
    "[M+H]+":   ({"H": 1}, 1),
    "[M-H]-":   ({"H": -1}, -1),
    "[M]+":     ({}, 1),
    "[M]-":     ({}, -1),
    "[M+Na]+":  ({"Na": 1}, 1),
    "[M+K]+":   ({"K": 1}, 1),
    "[M+NH4]+": ({"N": 1, "H": 4}, 1),
    "[M+Cl]-":  ({"Cl": 1}, -1),
    "[M+2H]2+": ({"H": 2}, 2),
    "[M-2H]2-": ({"H": -2}, -2),
    "[M+H-H2O]+": ({"H": -1, "O": -1}, 1),
}


@dataclass(frozen=True)
class Adduct:
    name: str
    delta: frozenset[tuple[str, int]]
    charge: int

    def __init__(self, name: str,
                 delta: dict[str, int] | None = None,
                 charge: int | None = None):
        if delta is None or charge is None:
            key = name.strip()
            if key not in _BUILTIN_ADDUCTS:
                known = ", ".join(sorted(_BUILTIN_ADDUCTS))
                raise AdductError(
                    f"unknown adduct {name!r}; known: {known}. "
                    "Pass delta={...} and charge=... to define a custom adduct.")
            d, z = _BUILTIN_ADDUCTS[key]
            name, delta, charge = key, dict(d), z
        else:
            if not isinstance(charge, int) or charge == 0:
                raise AdductError("charge must be a non-zero signed integer")
            delta = dict(delta)
        for sym, n in delta.items():
            from .elements import ELEMENTS
            if sym not in ELEMENTS:
                raise AdductError(f"unknown element in adduct: {sym!r}")
            if not isinstance(n, int) or n == 0:
                raise AdductError("adduct atom counts must be non-zero ints")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "delta",
                           frozenset(delta.items()))
        object.__setattr__(self, "charge", charge)

    @property
    def delta_dict(self) -> dict[str, int]:
        return dict(self.delta)

    @property
    def mz_multiplier(self) -> float:
        return 1.0 / abs(self.charge)

    def ion_composition(self, neutral: Formula) -> Formula:
        """Full elemental composition of the ion (atoms only)."""
        return neutral + self.delta_dict

    def __repr__(self) -> str:
        return f"Adduct({self.name!r})"


def adduct(name: str) -> Adduct:
    """Look up a built-in adduct by name, e.g. ``adduct('[M+H]+')``."""
    return Adduct(name)
