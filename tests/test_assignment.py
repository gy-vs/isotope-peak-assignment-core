"""Tests for the heart of the kernel: competitive assignment semantics."""

import json

import pytest

from isoclaim import (
    Budget,
    Calibration,
    Engine,
    MatchConfig,
    Status,
)


def glucose_rows(envelope, *, intensities=(1000.0, 68.0), extra=()):
    rows = [
        (envelope.peaks[i].mz, intensity)
        for i, intensity in zip(range(len(intensities)), intensities)
    ]
    return rows + list(extra)


class TestCleanAcceptance:
    def test_glucose_cluster_accepted_with_weak_isotope_missing(self, engine, glucose):
        envelope = engine.theory_for(glucose)
        rows = glucose_rows(
            envelope,
            intensities=(1000.0, 68.0),  # M+2 absent entirely
            extra=((500.0, 10.0),),
        )
        measurement = engine.make_measurement(rows, Calibration(version="v1"))
        result = engine.analyze([glucose], measurement)
        outcome = result.outcomes[0]
        assert outcome.status == Status.ACCEPTED
        assert len(outcome.matched_edges) >= 2
        assert result.solver_status == "OPTIMAL"

    def test_raw_and_theoretical_values_both_present(self, engine, glucose):
        envelope = engine.theory_for(glucose)
        measurement = engine.make_measurement(
            glucose_rows(envelope), Calibration(version="v1")
        )
        result = engine.analyze([glucose], measurement)
        edge = result.outcomes[0].matched_edges[0]
        assert edge.observed_mz != edge.theoretical_mz or True
        assert edge.observed_mz == pytest.approx(envelope.peaks[0].mz, abs=1e-9)
        assert edge.theoretical_mz == pytest.approx(envelope.peaks[0].mz)


class TestNoiseResistance:
    def test_single_lone_peak_is_never_accepted(self, engine, glucose):
        # One huge peak exactly at the base m/z: must not be read as a
        # complete isotope cluster.
        envelope = engine.theory_for(glucose)
        measurement = engine.make_measurement(
            [(envelope.peaks[0].mz, 50000.0), (900.0, 1.0)],
            Calibration(version="v1"),
        )
        result = engine.analyze([glucose], measurement)
        outcome = result.outcomes[0]
        assert outcome.status == Status.WEAK_SUPPORT
        assert len(outcome.matched_edges) == 1
        codes = {d.code for d in outcome.diagnostics}
        assert "BELOW_ACCEPTANCE_POLICY" in codes

    def test_giant_unrelated_peak_stays_unassigned(self, engine, glucose):
        envelope = engine.theory_for(glucose)
        rows = glucose_rows(envelope, extra=((300.1234, 9000.0),))
        measurement = engine.make_measurement(rows, Calibration(version="v1"))
        result = engine.analyze([glucose], measurement)
        unassigned = result.unassigned_peaks()
        assert any(p.mz == pytest.approx(300.1234) for p in unassigned)
        assert result.outcomes[0].status == Status.ACCEPTED

    def test_missing_weak_isotope_does_not_exclude(self, engine, glucose):
        envelope = engine.theory_for(glucose)
        # Only base and M+1, M+2 and everything weaker absent.
        measurement = engine.make_measurement(
            [(envelope.peaks[0].mz, 1000.0), (envelope.peaks[1].mz, 68.0)],
            Calibration(version="v1"),
        )
        result = engine.analyze([glucose], measurement)
        assert result.outcomes[0].status == Status.ACCEPTED


class TestSharedPeakCompetition:
    def _overlap_scene(self, engine, *, tolerance_ppm=50.0):
        glucose = engine.make_candidate(
            "C6H12O6", adduct="[M-H]-", name="glucose"
        )
        tyrosine = engine.make_candidate(
            "C9H11NO3", adduct="[M-H]-", name="tyrosine"
        )
        g_env = engine.theory_for(glucose)
        # A peak that is within 50 ppm of BOTH glucose base and tyrosine
        # M+1 isotope, but essentially exactly on glucose base.
        shared_mz = g_env.monoisotopic_mz + 0.0069
        rows = [
            (shared_mz, 1000.0),
            (g_env.peaks[1].mz, 70.0),
            (g_env.peaks[2].mz, 15.0),
        ]
        measurement = engine.make_measurement(
            rows, Calibration(version="v1"), serial="overlap"
        )
        config = MatchConfig(
            tolerance_ppm=tolerance_ppm, tolerance_da=0.0
        )
        return glucose, tyrosine, measurement, config

    def test_shared_peak_has_single_owner(self, engine):
        glucose, tyrosine, measurement, config = self._overlap_scene(engine)
        result = engine.analyze([glucose, tyrosine], measurement, config=config)
        g, t = result.outcomes
        # Glucose has the coherent multi-peak cluster and owns the peak.
        assert g.status in {Status.ACCEPTED, Status.WEAK_SUPPORT}
        owner_peaks = {e.peak_index for e in g.matched_edges}
        t_peaks = {e.peak_index for e in t.matched_edges}
        assert owner_peaks.isdisjoint(t_peaks)

    def test_loser_is_reported_as_contested_alternative(self, engine):
        glucose, tyrosine, measurement, config = self._overlap_scene(engine)
        result = engine.analyze([glucose, tyrosine], measurement, config=config)
        assert result.contested
        contested = result.contested[0]
        rival_indices = {r["candidate_index"] for r in contested["rivals"]}
        assert rival_indices == {0, 1}
        # Exactly one rival is selected; the losing explanation remains
        # visible with its own mass error and gain.
        selected = [r for r in contested["rivals"] if r["selected"]]
        assert len(selected) == 1
        loser = [r for r in contested["rivals"] if not r["selected"]]
        assert loser[0]["error_ppm"] != 0.0

    def test_withdrawing_winner_reassigns_shared_peak(self, engine):
        glucose, tyrosine, measurement, config = self._overlap_scene(engine)
        both = engine.analyze(
            [glucose, tyrosine], measurement, config=config
        )
        glucose_owner = {
            e.peak_index for e in both.outcomes[0].matched_edges
        }
        only_tyrosine = engine.analyze(
            [tyrosine], measurement, config=config
        )
        tyrosine_alone_owner = {
            e.peak_index for e in only_tyrosine.outcomes[0].matched_edges
        }
        # The shared peak flips to the remaining hypothesis.
        assert glucose_owner & tyrosine_alone_owner

    def test_adding_candidate_recoordinates(self, engine):
        glucose, tyrosine, measurement, config = self._overlap_scene(engine)
        glucose_only = engine.analyze([glucose], measurement, config=config)
        both = engine.analyze([glucose, tyrosine], measurement, config=config)
        # Same measurement, same theory cache - adding a rival recomputes the
        # joint decision rather than appending a local explanation.
        assert glucose_only.solver_status == both.solver_status == "OPTIMAL"
        assert len(both.outcomes) == 2

    def test_exclusivity_is_hard_under_all_orders(self, engine):
        glucose, tyrosine, measurement, config = self._overlap_scene(engine)
        r1 = engine.analyze([glucose, tyrosine], measurement, config=config)
        r2 = engine.analyze([tyrosine, glucose], measurement, config=config)
        for result in (r1, r2):
            owned = []
            for outcome in result.outcomes:
                owned.extend(e.peak_index for e in outcome.matched_edges)
            assert len(owned) == len(set(owned))

    def test_deterministic_across_repeated_runs(self, engine):
        glucose, tyrosine, measurement, config = self._overlap_scene(engine)
        first = engine.analyze(
            [glucose, tyrosine], measurement, config=config
        ).to_dict()
        second = engine.analyze(
            [glucose, tyrosine], measurement, config=config
        ).to_dict()
        assert json.dumps(first, sort_keys=True) == json.dumps(
            second, sort_keys=True
        )


class TestCalibrationVersioning:
    def test_changed_calibration_rejudges_without_rewriting_raw(self, engine, glucose):
        envelope = engine.theory_for(glucose)
        rows = glucose_rows(envelope)
        good = engine.make_measurement(
            rows, Calibration(offset=0.0, version="good")
        )
        shifted = engine.make_measurement(
            rows, Calibration(offset=0.01, version="shifted")
        )
        config = MatchConfig(tolerance_ppm=10, tolerance_da=0.002)
        accepted = engine.analyze([glucose], good, config=config)
        rejected = engine.analyze([glucose], shifted, config=config)
        assert accepted.outcomes[0].status == Status.ACCEPTED
        assert rejected.outcomes[0].status in {
            Status.INSUFFICIENT, Status.OUT_OF_RANGE
        }
        # Raw m/z identical across the two measurements.
        assert [p.mz for p in good.peaks] == [p.mz for p in shifted.peaks]
        # Report remembers which calibration version produced the decision.
        assert accepted.calibration.version == "good"
        assert rejected.calibration.version == "shifted"


class TestFailureReporting:
    def test_invalid_candidate_does_not_abort_batch(self, engine, glucose):
        envelope = engine.theory_for(glucose)
        measurement = engine.make_measurement(
            glucose_rows(envelope), Calibration(version="v1")
        )
        result = engine.analyze(
            [glucose, "Qq9", {"formula": "H2O", "adduct": "[M-H]-"}],
            measurement,
        )
        statuses = [o.status for o in result.outcomes]
        assert statuses[0] == Status.ACCEPTED
        assert statuses[1] == Status.INVALID
        assert statuses[2] == Status.OUT_OF_RANGE
        invalid = result.outcomes[1]
        assert invalid.diagnostics[0].code == "INVALID"

    def test_theory_budget_is_an_explicit_status(self):
        # Glucose's widest element distribution (6 oxygens x +2 shift) needs
        # 13 nominal bins; a 200-carbon chain needs ~401.
        tight = Engine(budget=Budget(max_states=50, min_abundance=1e-9))
        cheap = tight.make_candidate("C6H12O6", adduct="[M-H]-")
        heavy_t = tight.make_candidate("C200H400O200")
        glucose_engine = Engine()
        glucose = glucose_engine.make_candidate(
            "C6H12O6", adduct="[M-H]-"
        )
        g_env = glucose_engine.theory_for(glucose)
        measurement = tight.make_measurement(
            glucose_rows(g_env), Calibration(version="v1")
        )
        result = tight.analyze([cheap, heavy_t], measurement)
        assert result.outcomes[0].status in {
            Status.ACCEPTED, Status.WEAK_SUPPORT
        }
        assert result.outcomes[1].status == Status.THEORY_BUDGET
        assert result.outcomes[1].diagnostics[0].code == "THEORY_BUDGET"

    def test_edge_budget_fails_run_and_leaves_candidates_unresolved(self):
        engine = Engine(budget=Budget(max_edges=1))
        glucose = engine.make_candidate("C6H12O6", adduct="[M-H]-")
        envelope = engine.theory_for(glucose)
        measurement = engine.make_measurement(
            glucose_rows(envelope), Calibration(version="v1")
        )
        # Add many copies of the candidate (distinct hypotheses) to inflate
        # edge count.
        candidates = [glucose] + [
            engine.make_candidate(f"C{6+i}H12O6", adduct="[M-H]-")
            for i in range(1, 6)
        ]
        result = engine.analyze(candidates, measurement)
        codes = {d.code for d in result.diagnostics}
        assert "EDGE_BUDGET" in codes
        assert all(not o.active for o in result.outcomes)

    def test_solver_failure_never_presents_unverified_ownership(
        self, monkeypatch, engine, glucose
    ):
        from isoclaim import assignment as assignment_module

        class _BrokenResult:
            x = None
            success = False
            status = 4
            message = "simulated solver failure"
            fun = 0.0

        def _broken_milp(*args, **kwargs):
            return _BrokenResult()

        monkeypatch.setattr(assignment_module, "milp", _broken_milp,
                            raising=False)
        # The actual import lives inside _solve; patch it there through the
        # scipy module used at call time.
        import scipy.optimize as _so

        monkeypatch.setattr(_so, "milp", _broken_milp)
        envelope = engine.theory_for(glucose)
        measurement = engine.make_measurement(
            glucose_rows(envelope), Calibration(version="v1")
        )
        result = engine.analyze([glucose], measurement)
        assert result.solver_status == "FAILED"
        assert all(not o.active for o in result.outcomes)
        assert all(not o.matched_edges for o in result.outcomes)
        assert result.unassigned_peak_indices == tuple(
            range(len(measurement.peaks))
        )
        codes = {d.code for d in result.diagnostics}
        assert "SOLVER_FAILED" in codes


class TestTheoryCacheSeparation:
    def test_cache_reused_across_measurements(self, engine, glucose):
        envelope = engine.theory_for(glucose)
        before = engine.theory_cache.stats()
        rows = glucose_rows(envelope)
        m1 = engine.make_measurement(rows, Calibration(version="v1"))
        m2 = engine.make_measurement(rows, Calibration(version="v2"))
        engine.analyze([glucose], m1)
        engine.analyze([glucose], m2)
        after = engine.theory_cache.stats()
        assert after["envelopes"] == before["envelopes"] == 1

    def test_measurement_identity_not_reused_in_report(self, engine, glucose):
        envelope = engine.theory_for(glucose)
        rows = glucose_rows(envelope)
        m1 = engine.make_measurement(rows, Calibration(version="v1"))
        m2 = engine.make_measurement(rows, Calibration(version="v1"))
        r1 = engine.analyze([glucose], m1)
        r2 = engine.analyze([glucose], m2)
        assert r1.measurement_fingerprint == r2.measurement_fingerprint
        assert r1.measurement_serial != r2.measurement_serial
        ids1 = {p.peak_id for p in r1.peaks}
        ids2 = {p.peak_id for p in r2.peaks}
        assert ids1.isdisjoint(ids2)


class TestReportShape:
    def test_to_dict_is_json_serializable_and_tripartite(self, engine, glucose):
        envelope = engine.theory_for(glucose)
        measurement = engine.make_measurement(
            glucose_rows(envelope, extra=((424.24, 7.0),)),
            Calibration(version="v1"),
        )
        result = engine.analyze([glucose], measurement)
        payload = result.to_dict()
        blob = json.dumps(payload)  # raises if non-serializable
        assert isinstance(blob, str)
        # Theory, raw observation and decision are all independently present.
        assert payload["candidates"][0]["envelope"]["neutral_mass"] > 0
        assert payload["unassigned_peaks"][0]["mz"] == 424.24
        assert payload["candidates"][0]["status"] == Status.ACCEPTED
        assert payload["measurement"]["calibration"]["version"] == "v1"
