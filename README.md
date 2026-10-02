# Isotope Peak Assignment — `isoclaim`

A small, importable Python kernel for **competitive isotope-cluster
assignment** against centroid mass-spectrometry peak tables.

It is built for laboratory analysis pipelines (not as a command-line tool):
you submit molecular-formula candidates, one immutable peak table and one
named mass calibration, and receive a report in which theoretical peaks, raw
observed peaks, unassigned peaks and the final assignment can all be audited
separately.

```python
from isoclaim import Engine, Calibration

engine = Engine()                          # theory cache lives here
calibration = Calibration(version="batch-2026-10-02")
measurement = engine.make_measurement(
    [(179.0561, 1000.0), (180.0596, 70.0), (181.0607, 14.0), (512.3, 800.0)],
    calibration,
)
candidates = [
    engine.make_candidate("C6H12O6", adduct="[M-H]-", name="glucose"),
    engine.make_candidate("C9H11NO3", adduct="[M-H]-", name="tyrosine"),
]
result = engine.analyze(candidates, measurement)
```

## Design guarantees

* **Global competition, not nearest-neighbour per-candidate greed.**  Every
  candidate/observed peak pair inside the matching window becomes an edge in
  one joint mixed-integer program (solved with HiGHS via SciPy).  An observed
  peak is owned by *at most one* explanation; activating a candidate charges a
  fixed penalty plus penalties for strong predicted isotope peaks that are
  absent.  Candidates that lose a shared peak remain in the report as
  `CONTESTED` alternatives with their own mass error and gain, instead of
  silently disappearing.
* **Adding or withdrawing candidates re-coordinates the whole decision.**
  `analyze()` always solves the complete candidate set from scratch; it never
  appends a local explanation to a previous one.
* **Theory cache and measurement state are separate objects.**  The
  `TheoryCache` contains only chemistry (keyed by ion composition, charge,
  thresholds and a hash of the pyteomics isotope table).  It contains no peak
  identities, no assignments and no calibration, so it is reused across
  samples and calibration versions.  Observed peak IDs are issued per
  `Measurement` serial and can never leak from one sample into another.
* **Calibration changes the comparison, never the raw data.**  Expected m/z is
  passed through the calibration polynomial; observed m/z values stay exactly
  as recorded.  Every edge reports theoretical m/z, calibrated expected m/z
  and raw observed m/z side by side.
* **Explicit failure, never an empty result.**  Invalid formulas/adducts are
  reported per candidate (`INVALID`) without aborting the batch; theory
  computations that exceed their resource budget are reported as
  `THEORY_BUDGET`; an optimizer that cannot certify optimality inside its time
  budget marks the run unresolved instead of presenting an unverifiable
  assignment.
* **Resource use is bounded independently of atom count.**  Isotope
  envelopes are computed by a dynamic program over nominal-mass bins with
  probability and exact-mass-moment arrays (direct convolution for small
  arrays, FFT for large ones, binary exponentiation for repeated elements),
  truncated by absolute isotopologue probability.  Runtime and memory do not
  grow with the combinatorial isotopologue count; states/time/peak/edge/solver
  caps are all explicit `Budget` fields that raise on breach.
* **Independent isotope cross-check.**  For small molecules the DP envelope
  is compared against pyteomics' independent *exact* isotopologue enumeration
  (abundance and bin centroids); disagreement is a diagnostic, not a silent
  replacement.  The check is skipped with an explicit note for molecules too
  large for the combinatorial reference.
* **Real data has noise and missing weak peaks.**  Acceptance needs a
  coherent multi-peak cluster (score plus minimum matched peaks), so one huge
  noise peak is at most `WEAK_SUPPORT`, while absent *weak* isotope peaks do
  not exclude the candidate.  Absent *strong* peaks count as negative
  evidence.
* **Deterministic reporting.**  Equal inputs and tolerances produce equal
  JSON reports; ties are broken by candidate order and nominal offset with
  coefficients too small to affect any scientific comparison.

## Report contents

`AssignmentResult.to_dict()` is JSON-serializable and keeps the three layers
independently checkable:

* `measurement` – run serial, peak-table fingerprint, calibration version and
  fingerprint;
* `candidates[*].envelope` – neutral mass, ion m/z and every predicted isotope
  peak with absolute/relative abundance plus the independent cross-check;
* `candidates[*].edges` / `matched_peak_ids` / `missing_strong_offsets` /
  `scores` – which raw peaks support the candidate and which predicted peaks
  argue against it;
* `contested_peaks` – every raw peak claimed by more than one candidate, with
  the winning and losing explanations;
* `unassigned_peaks` – raw peaks no selected explanation owns;
* `solver` and `diagnostics` – optimizer status and all warnings/failures.

## Layout

| module | responsibility |
| --- | --- |
| `isoclaim/formula.py` | formula grammar, adduct grammar, `Candidate` identity |
| `isoclaim/theory.py` | bounded isotope DP, exact cross-check, `TheoryCache` |
| `isoclaim/calibration.py` | versioned calibration polynomial, robust refit helper |
| `isoclaim/peaks.py` | validated immutable peak tables with run-local peak IDs |
| `isoclaim/assignment.py` | edge construction, joint MILP, scoring, classification |
| `isoclaim/engine.py` | the reusable entry point (`Engine`) |
| `tests/` | pytest suite (`pytest -q`) |

Dependencies: `pyteomics`, `scipy`, `numpy` (and `pytest` for the tests).
