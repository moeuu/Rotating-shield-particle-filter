"""Tests for the live PF runtime-ingestion boundary."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from runtime.measurement_log import load_measurement_log

from pf.live_session import (
    PFLiveSession,
    PFLiveSessionError,
    bind_published_measurement_log,
    build_live_estimator,
    measurement_record_to_station_input,
)
from pf.estimator_types import JointPlanningParticles
from pf.pure_estimator import RotatingShieldPFConfig
from tests.pure_pf_test_support import make_measurement_log


def test_live_builder_uses_run_context_without_synthetic_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Live construction resolves its forward model directly from context."""
    from pf import live_session

    context = SimpleNamespace(
        schema_version=2,
        runtime_config_sha256="a" * 64,
    )
    forward = object()
    estimator = object()
    calls: dict[str, object] = {}

    class _Resolver:
        """Capture direct shared-runtime context resolution."""

        @classmethod
        def from_run_context(
            cls,
            actual_context: object,
            *,
            run_root: object,
        ) -> object:
            """Return the sentinel authenticated forward context."""
            del cls
            calls["context"] = actual_context
            calls["run_root"] = run_root
            return forward

    def build_from_forward(
        actual_forward: object,
        config: object,
        **kwargs: object,
    ) -> object:
        """Capture the thin PF-specific construction adapter."""
        calls["forward"] = actual_forward
        calls["config"] = config
        calls["kwargs"] = kwargs
        return estimator

    monkeypatch.setattr(live_session, "ResolvedForwardContext", _Resolver)
    monkeypatch.setattr(
        live_session,
        "_build_live_estimator_from_forward_context",
        build_from_forward,
    )

    actual = build_live_estimator(
        context,  # type: ignore[arg-type]
        {"pure_pf_schema_version": 1},
        profile="pf_strict",
        seed=9,
        runtime_root=tmp_path,
    )

    assert actual is estimator
    assert calls["context"] is context
    assert calls["run_root"] == tmp_path.resolve()
    assert calls["forward"] is forward
    assert calls["kwargs"] == {
        "profile": "pf_strict",
        "seed": 9,
        "measurement_log_schema_version": 2,
        "measurement_runtime_config_sha256": "a" * 64,
        "config_hash": None,
        "inference_isotopes": None,
    }


def test_live_record_forwards_only_raw_spectrum_and_action_geometry(
    tmp_path: Path,
) -> None:
    """Live ingestion forwards the raw spectrum and selected shield action."""
    log = load_measurement_log(
        make_measurement_log(tmp_path / "measurement-log", record_count=1)
    )

    station_input = measurement_record_to_station_input(log.records[0])

    assert len(station_input) == 4
    spectrum, fe_index, pb_index, live_time_s = station_input
    assert isinstance(spectrum, np.ndarray)
    assert spectrum.dtype == np.int64
    np.testing.assert_array_equal(spectrum, log.records[0].spectrum_counts)
    assert (fe_index, pb_index, live_time_s) == (0, 0, 1.0)


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    (
        ("spectrum_counts", np.asarray([1.0, 0.5], dtype=np.float64)),
        ("spectrum_counts", np.asarray([1, 2], dtype=np.int32)),
        ("fe_orientation_index", True),
        ("pb_orientation_index", "0"),
        ("live_time_s", True),
        ("live_time_s", "1.0"),
    ),
)
def test_live_record_rejects_coercion(
    field_name: str,
    invalid: object,
) -> None:
    """The live adapter preserves raw observation types without coercion."""
    values: dict[str, Any] = {
        "spectrum_counts": np.asarray([1, 2], dtype=np.int64),
        "fe_orientation_index": 0,
        "pb_orientation_index": 0,
        "live_time_s": 1.0,
    }
    values[field_name] = invalid

    with pytest.raises(PFLiveSessionError):
        measurement_record_to_station_input(SimpleNamespace(**values))


def test_published_log_binding_requires_the_live_record_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Final provenance binding accepts only the assimilated live session."""
    from pf import live_session

    published = load_measurement_log(
        make_measurement_log(tmp_path / "measurement-log", record_count=1)
    )
    isotopes = tuple(published.run_manifest["isotopes"])
    estimator = SimpleNamespace(
        candidate_isotopes=isotopes,
        joint_isotope_order=lambda: isotopes,
        measurements=[object()],
        pf_config=RotatingShieldPFConfig(),
        random_seed=7,
        measurement_log_sha256="unavailable",
        resolved_config_hash="unavailable",
    )
    validated: list[object] = []
    monkeypatch.setattr(
        live_session,
        "_validate_published_forward_context",
        validated.append,
    )

    bind_published_measurement_log(
        estimator,
        published,
        live_records=published.records,
    )

    assert validated == [published]
    assert estimator.measurement_log_sha256 == published.log_sha256
    assert len(estimator.resolved_config_hash) == 64

    changed_record = replace(published.records[0], live_time_s=2.0)
    changed_published = replace(published, records=(changed_record,))
    with pytest.raises(PFLiveSessionError, match="ordered live records"):
        bind_published_measurement_log(
            estimator,
            changed_published,
            live_records=published.records,
        )


class _SpyPosterior:
    """Serialize the final identities supplied by a spy PF estimator."""

    def __init__(self, estimator: "_SpyEstimator") -> None:
        """Retain the estimator whose bound identities are reported."""
        self.estimator = estimator

    def to_dict(self) -> dict[str, object]:
        """Return the minimal identities checked by the live facade."""
        return {
            "schema_version": 1,
            "measurement_log_sha256": self.estimator.measurement_log_sha256,
            "record_count": len(self.estimator.measurements),
        }


class _SpyEstimator:
    """Record station updates without running PF science in facade tests."""

    def __init__(self, isotopes: tuple[str, ...], contract_hash: str) -> None:
        """Initialize mutable fields required by the existing live helpers."""
        self.candidate_isotopes = isotopes
        self.isotopes = isotopes
        self.measurements: list[object] = []
        self.poses = [np.asarray([0.25, 0.25, 0.4], dtype=np.float64)]
        self.kernel_cache: object | None = object()
        self.pf_config = RotatingShieldPFConfig()
        self.random_seed = 17
        self.measurement_log_sha256 = "unavailable"
        self.resolved_config_hash = "unavailable"
        self.full_spectrum_generative_model = SimpleNamespace(
            contract_hash_sha256=contract_hash
        )
        self.update_calls: list[dict[str, object]] = []
        self.planning_calls = 0
        self.posterior_summary_calls = 0
        self.particles = JointPlanningParticles(
            isotope_order=isotopes,
            weights_n=np.asarray([0.75, 0.25], dtype=np.float64),
            positions_nk3_by_isotope={
                isotope: np.asarray(
                    [[[0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6]]],
                    dtype=np.float64,
                )
                for isotope in isotopes
            },
            surface_chart_ids_nk_by_isotope={
                isotope: np.asarray([[1], [2]], dtype=np.int64)
                for isotope in isotopes
            },
            surface_uv_nk2_by_isotope={
                isotope: np.asarray(
                    [[[0.1, 0.2]], [[0.3, 0.4]]],
                    dtype=np.float64,
                )
                for isotope in isotopes
            },
            strengths_nk_by_isotope={
                isotope: np.asarray([[2.0], [3.0]], dtype=np.float64)
                for isotope in isotopes
            },
            source_mask_nk_by_isotope={
                isotope: np.asarray([[True], [True]], dtype=np.bool_)
                for isotope in isotopes
            },
            original_particle_indices=np.asarray([4, 7], dtype=np.int64),
        )

    def joint_isotope_order(self) -> tuple[str, ...]:
        """Return the spy's active isotope order."""
        return self.isotopes

    def add_measurement_pose(
        self,
        pose: np.ndarray,
        *,
        reset_filters: bool,
    ) -> None:
        """Append a copied detector pose without resetting state."""
        assert reset_filters is False
        self.poses.append(np.asarray(pose, dtype=np.float64).copy())

    def update_spectrum_station(
        self,
        records: tuple[tuple[object, ...], ...],
        *,
        pose_idx: int,
        generative_contract_hash_sha256: str,
    ) -> None:
        """Record one station update and expose its assimilated row count."""
        self.update_calls.append(
            {
                "records": records,
                "pose_idx": pose_idx,
                "contract": generative_contract_hash_sha256,
            }
        )
        self.measurements.extend(object() for _ in records)

    def planning_joint_particles(
        self,
        *,
        max_particles: int | None,
        method: str | None,
        rng: np.random.Generator | None,
    ) -> JointPlanningParticles:
        """Return mutable arrays so the facade must copy them."""
        del max_particles, method, rng
        self.planning_calls += 1
        return self.particles

    def posterior_point_estimate(self) -> dict[str, SimpleNamespace]:
        """Return one existing PF summary contract for the planning DTO."""
        self.posterior_summary_calls += 1
        return {
            isotope: SimpleNamespace(
                to_dict=lambda isotope=isotope: {
                    "map_cardinality": 1,
                    "isotope": isotope,
                }
            )
            for isotope in self.isotopes
        }

    def serialized_state(self) -> bytes:
        """Return state bytes depending only on completed station updates."""
        return json.dumps(
            {
                "measurement_count": len(self.measurements),
                "station_update_count": len(self.update_calls),
            },
            sort_keys=True,
        ).encode("utf-8")

    def posterior_snapshot(self) -> _SpyPosterior:
        """Return a result view using identities set by log binding."""
        return _SpyPosterior(self)


def _facade_with_spy(
    monkeypatch: pytest.MonkeyPatch,
    log: object,
) -> tuple[PFLiveSession, _SpyEstimator]:
    """Construct a facade while replacing only the expensive PF estimator."""
    from pf import live_session

    context = log.context
    contract_hash = str(
        context.runtime_config["full_spectrum_contract_hash_sha256"]
    )
    estimator = _SpyEstimator(tuple(context.isotopes), contract_hash)

    def build_spy(
        actual_context: object,
        config: object,
        **kwargs: object,
    ) -> _SpyEstimator:
        """Verify authenticated construction inputs and return the spy."""
        assert actual_context is context
        assert config == {"pure_pf_schema_version": 1}
        assert kwargs["profile"] == "pf_strict"
        assert kwargs["seed"] == 17
        return estimator

    monkeypatch.setattr(live_session, "build_live_estimator", build_spy)
    session = PFLiveSession(
        context,
        {"pure_pf_schema_version": 1},
        profile="pf_strict",
        seed=17,
        runtime_root=log.path,
    )
    return session, estimator


def test_facade_assimilates_only_complete_persisted_stations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Record delivery must use one canonical station update at its marker."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=2,
            station_complete_markers=True,
        )
    )
    session, estimator = _facade_with_spy(monkeypatch, log)

    assert session.receive_persisted(log.records[0]) is False
    assert estimator.update_calls == []
    with pytest.raises(PFLiveSessionError, match="station boundary"):
        session.planning_particle_snapshot()

    assert session.receive_persisted(log.records[1]) is True
    assert len(estimator.update_calls) == 1
    assert estimator.update_calls[0]["contract"] == (
        log.runtime_config["full_spectrum_contract_hash_sha256"]
    )
    assert session.records == log.records
    assert session.station_count == 1


def test_facade_particle_snapshot_is_an_immutable_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Planning DTO arrays and isotope mappings must not alias the estimator."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=2,
            station_complete_markers=True,
        )
    )
    session, estimator = _facade_with_spy(monkeypatch, log)
    session.receive_persisted_station(log.records)

    snapshot = session.planning_particle_snapshot()
    estimator.particles.weights_n[0] = 0.5
    estimator.particles.positions_nk3_by_isotope[log.context.isotopes[0]][0, 0, 0] = 9.0

    np.testing.assert_array_equal(snapshot.weights_n, [0.75, 0.25])
    assert snapshot.positions_nk3_by_isotope[log.context.isotopes[0]][0, 0, 0] == 0.1
    with pytest.raises(ValueError):
        snapshot.weights_n[0] = 1.0
    with pytest.raises(ValueError):
        snapshot.weights_n.setflags(write=True)
    with pytest.raises(TypeError):
        snapshot.positions_nk3_by_isotope["new"] = np.zeros((2, 1, 3))
    summary = snapshot.posterior_summary()
    assert summary["publishable"] is False
    assert set(summary["isotopes"]) == set(log.context.isotopes)
    summary["isotopes"] = {}
    assert snapshot.posterior_summary()["isotopes"]
    assert estimator.posterior_summary_calls == 1


def test_bind_and_publication_never_advance_completed_pf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Final binding and publication input must not assimilate observations."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=2,
            station_complete_markers=True,
        )
    )
    session, estimator = _facade_with_spy(monkeypatch, log)
    session.receive_persisted_station(log.records)
    completed = session.complete_live_state()
    update_count = len(estimator.update_calls)
    planning_count = estimator.planning_calls

    with pytest.raises(PFLiveSessionError, match="cannot receive"):
        session.receive_persisted(log.records[-1])
    with pytest.raises(PFLiveSessionError, match="cannot plan"):
        session.planning_particle_snapshot()
    bound = session.bind_published_log(log)
    publication = session.publication_input()

    assert len(estimator.update_calls) == update_count == 1
    assert estimator.planning_calls == planning_count
    assert publication is bound
    assert bound.completed is completed
    assert bound.checkpoint_state == completed.checkpoint_state
    assert json.loads(bound.posterior_json)["measurement_log_sha256"] == log.log_sha256
    assert session.phase == "bound"


def test_facade_rejects_out_of_order_and_mismatched_final_logs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The facade must bind only its exact run, context, and ordered records."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=2,
            station_complete_markers=True,
        )
    )
    out_of_order, _ = _facade_with_spy(monkeypatch, log)
    with pytest.raises(PFLiveSessionError, match="causal record order"):
        out_of_order.receive_persisted(log.records[1])

    incomplete, _ = _facade_with_spy(monkeypatch, log)
    incomplete.receive_persisted(log.records[0])
    with pytest.raises(PFLiveSessionError, match="station marker"):
        incomplete.complete_live_state()

    session, _ = _facade_with_spy(monkeypatch, log)
    session.receive_persisted_station(log.records)
    session.complete_live_state()
    changed_record = replace(log.records[-1], live_time_s=2.0)
    changed_log = replace(log, records=(log.records[0], changed_record))
    with pytest.raises(PFLiveSessionError, match="records differ"):
        session.bind_published_log(changed_log)

    changed_context = replace(
        log,
        environment={**log.environment, "size_x": 3.0},
    )
    with pytest.raises(PFLiveSessionError, match="context differs"):
        session.bind_published_log(changed_context)
