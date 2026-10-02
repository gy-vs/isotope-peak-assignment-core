"""Candidate identities and cached theoretical envelopes.

This layer is deliberately **calibration-free**.  A
:class:`CandidateTheory` depends only on (formula, adduct, charge, isotope
budget, constants data), so it can be computed once and reused across every
peak table and calibration version.  Per-measurement assignment state lives in
:mod:`isoassign.assignment` and never enters the cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .adducts import Adduct, adduct as _adduct
from .elements import ELECTRON_MASS
from .errors import AdductError, FormulaError
from .formula import Formula
from .isotopes import IsotopeBudget, IsotopeCluster, isotope_distribution


@dataclass(frozen=True)
class TheoreticalPeak:
    """One predicted peak for a candidate, already converted to m/z."""

    index: int
    mz: float                      # theoretical m/z
    neutral_mass: float            # envelope peak neutral mass (Da)
    ion_mass: float                # ion mass before /|z| (Da)
    relative_abundance: float
    probability: float
    is_monoisotopic: bool
    composition: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True, eq=False)
class CandidateSpec:
    """A requested candidate: id + formula + ionization.

    Construction with a raw formula/adduct never raises: an unparseable
    request is retained with ``valid = False`` and an error message, so the
    request itself can be reported back rather than dropped.
    """

    candidate_id: str
    formula: Formula | None
    adduct: Adduct | None
    formula_input: object = None
    adduct_input: object = None
    valid: bool = True
    error: str = ""
    error_token: str | None = None

    def __init__(self, candidate_id: str, formula, adduct_spec):
        object.__setattr__(self, "candidate_id", str(candidate_id))
        if not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")
        object.__setattr__(self, "formula_input", formula)
        object.__setattr__(self, "adduct_input", adduct_spec)
        try:
            f = formula if isinstance(formula, Formula) else Formula(formula)
        except FormulaError as exc:
            object.__setattr__(self, "formula", None)
            object.__setattr__(self, "adduct", None)
            object.__setattr__(self, "valid", False)
            object.__setattr__(self, "error", str(exc))
            object.__setattr__(self, "error_token", exc.token)
            return
        try:
            a = adduct_spec if isinstance(adduct_spec, Adduct) \
                else _adduct(adduct_spec)
        except AdductError as exc:
            object.__setattr__(self, "formula", None)
            object.__setattr__(self, "adduct", None)
            object.__setattr__(self, "valid", False)
            object.__setattr__(self, "error", str(exc))
            return
        object.__setattr__(self, "formula", f)
        object.__setattr__(self, "adduct", a)
        object.__setattr__(self, "valid", True)

    @property
    def charge(self) -> int:
        return self.adduct.charge if self.valid else 0

    @property
    def neutral_monoisotopic_mass(self) -> float:
        return self.formula.monoisotopic_mass if self.valid else float("nan")

    def _identity(self):
        if self.valid:
            return (self.candidate_id,
                    tuple(sorted(self.formula.as_dict().items())),
                    self.adduct.name, self.adduct.charge)
        return (self.candidate_id, repr(self.formula_input),
                repr(self.adduct_input), False)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CandidateSpec) and \
            self._identity() == other._identity()

    def __hash__(self) -> int:
        return hash(self._identity())


@dataclass(frozen=True)
class CandidateTheory:
    """The complete, reusable theoretical prediction for one candidate."""

    spec: CandidateSpec
    status: str                     # "ok" | "invalid_formula" | "budget_exceeded"
    neutral_mass: float
    monoisotopic_mz: float
    charge: int
    adduct_name: str
    peaks: tuple[TheoreticalPeak, ...] = ()
    cluster: IsotopeCluster | None = None
    errors: tuple[str, ...] = ()
    diagnostics: dict = field(default_factory=dict)

    @property
    def candidate_id(self) -> str:
        return self.spec.candidate_id

    @property
    def is_usable(self) -> bool:
        return self.status == "ok"

    def fingerprint(self) -> tuple:
        """Hashable identity of the theoretical inputs (for cache keys)."""
        if not self.spec.valid:
            return (repr(self.spec.formula_input),
                    repr(self.spec.adduct_input), "invalid")
        return (
            tuple(sorted(self.spec.formula.as_dict().items())),
            self.adduct_name,
            self.charge,
            self.neutral_mass,
        )


def build_theory(spec: CandidateSpec,
                budget: IsotopeBudget | None = None) -> CandidateTheory:
    """Compute the full theoretical envelope for one candidate.

    Never raises for invalid formulas or budget failure: those are returned as
    failed :class:`CandidateTheory` objects with diagnostics, so a single bad
    candidate cannot blank out the rest of an analysis.
    """
    budget = budget or IsotopeBudget()
    if not spec.valid:
        return CandidateTheory(
            spec=spec, status="invalid_formula",
            neutral_mass=float("nan"), monoisotopic_mz=float("nan"),
            charge=0, adduct_name=str(spec.adduct_input),
            errors=(spec.error,),
            diagnostics={"expression": str(spec.formula_input),
                         "token": spec.error_token,
                         "error_type": "FormulaError"})

    neutral_mass = spec.formula.monoisotopic_mass
    try:
        ion_formula = spec.adduct.ion_composition(spec.formula)
        cluster = isotope_distribution(ion_formula, budget)
    except Exception as exc:  # budget / numerical failure -> explicit
        return CandidateTheory(
            spec=spec, status="budget_exceeded",
            neutral_mass=neutral_mass,
            monoisotopic_mz=float("nan"), charge=spec.charge,
            adduct_name=spec.adduct.name, errors=(str(exc),),
            diagnostics={"error_type": type(exc).__name__})

    z = spec.charge
    # The cluster is computed on the *ion composition* (neutral + delta), so
    # each peak already contains the adduct atoms; only the electron mass
    # correction and division by |z| remain.
    mono_delta_mass = sum(
        _monoisotope_mass(sym) * n for sym, n in spec.adduct.delta)
    mono_ion_mass = neutral_mass + mono_delta_mass - z * ELECTRON_MASS
    mono_mz = mono_ion_mass / abs(z)

    peaks = tuple(
        TheoreticalPeak(
            index=p.index,
            mz=(p.mass - z * ELECTRON_MASS) / abs(z),
            neutral_mass=p.mass - mono_delta_mass,
            ion_mass=p.mass - z * ELECTRON_MASS,
            relative_abundance=p.relative_abundance,
            probability=p.probability,
            is_monoisotopic=p.is_monoisotopic,
            composition=p.composition,
        )
        for p in cluster.peaks
    )
    status = "budget_exceeded" if cluster.budget_exceeded else "ok"
    return CandidateTheory(
        spec=spec,
        status=status,
        neutral_mass=neutral_mass,
        monoisotopic_mz=mono_mz,
        charge=z,
        adduct_name=spec.adduct.name,
        peaks=peaks,
        cluster=cluster,
        errors=() if status == "ok" else
        (f"dropped {cluster.dropped_fraction:.3e} of natural probability "
         f"(accepted {budget.accepted_dropped_fraction:.0e})",),
        diagnostics={
            "dropped_fraction": cluster.dropped_fraction,
            "accepted_dropped_fraction": budget.accepted_dropped_fraction,
            "truncated": cluster.truncated,
            "max_intermediate": cluster.max_intermediate,
        },
    )


def _monoisotope_mass(symbol: str) -> float:
    from .elements import ELEMENTS
    return ELEMENTS[symbol].monoisotopic_mass


class TheoryLibrary:
    """Cache of candidate theories, independent of measurement data.

    Add/remove candidates freely; repeated identical requests return the cached
    object.  Persist with :meth:`save` / :meth:`load` (pickle).  Cached values
    never reference a peak table or calibration.
    """

    def __init__(self, budget: IsotopeBudget | None = None):
        self._budget = budget or IsotopeBudget()
        self._cache: dict[CandidateSpec, CandidateTheory] = {}

    @property
    def budget(self) -> IsotopeBudget:
        return self._budget

    def key(self, candidate_id: str, formula, adduct_spec) -> CandidateSpec:
        return CandidateSpec(candidate_id, formula, adduct_spec)

    def add(self, candidate_id: str, formula, adduct_spec) -> CandidateTheory:
        """Add/replace a candidate by id.

        Re-adding an existing id with a different formula/adduct replaces the
        old entry (a candidate id uniquely identifies one candidate); the new
        theory then participates in a fresh global coordination on the next
        ``assign`` call.
        """
        spec = self.key(candidate_id, formula, adduct_spec)
        existing = self._cache.get(spec)
        if existing is None:
            if spec.candidate_id in self:
                self.remove(spec.candidate_id)
            existing = build_theory(spec, self._budget)
            self._cache[spec] = existing
        return existing

    def add_spec(self, spec: CandidateSpec) -> CandidateTheory:
        existing = self._cache.get(spec)
        if existing is None:
            existing = build_theory(spec, self._budget)
            self._cache[spec] = existing
        return existing

    def remove(self, candidate_id: str) -> None:
        for spec in list(self._cache):
            if spec.candidate_id == candidate_id:
                del self._cache[spec]

    def get(self, candidate_id: str) -> CandidateTheory | None:
        for spec, theory in self._cache.items():
            if spec.candidate_id == candidate_id:
                return theory
        return None

    def theories(self) -> tuple[CandidateTheory, ...]:
        return tuple(self._cache.values())

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, candidate_id: object) -> bool:
        return any(s.candidate_id == candidate_id for s in self._cache)

    def save(self, path) -> None:
        import pickle
        with open(path, "wb") as fh:
            pickle.dump({"budget": self._budget,
                         "theories": tuple(self._cache.values())}, fh)

    @classmethod
    def load(cls, path) -> "TheoryLibrary":
        import pickle
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        lib = cls(payload["budget"])
        for theory in payload["theories"]:
            lib._cache[theory.spec] = theory
        return lib
