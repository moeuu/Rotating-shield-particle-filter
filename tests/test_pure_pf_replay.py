"""Causal replay tests for the full-spectrum PF boundary."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pf.replay import (
    PFReplayError,
    build_live_estimator,
    measurement_record_to_spectrum_input,
    build_replay_estimator,
    replay_measurement_log,
    replay_records,
    validate_local_full_spectrum_contract,
)
from runtime.measurement_log import (
    MeasurementLog,
    MeasurementLogArrayView,
    MeasurementLogStationView,
    build_forward_model_manifest,
    load_measurement_log,
)
from runtime.provenance import sha256_json as runtime_sha256_json
from tests.pure_pf_test_support import make_measurement_log


class _Posterior:
    """Provide the minimal immutable posterior interface used by traces."""

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic test posterior."""
        return {"schema_version": 2, "cardinality": {"Cs-137": {"0": 1.0}}}


class _RecordingEstimator:
    """Record station updates without implementing inference."""

    estimator_variant = "test_pure_particle_filter"

    def __init__(self) -> None:
        """Initialize empty poses and update records."""
        self.poses: list[np.ndarray] = []
        self.kernel_cache: object | None = object()
        self.updates: list[tuple[tuple[tuple[object, ...], ...], int, str]] = []
        self.serialization_calls = 0
        self.posterior_calls = 0

    def add_measurement_pose(
        self,
        pose: np.ndarray,
        *,
        reset_filters: bool,
    ) -> None:
        """Append one detector pose."""
        assert reset_filters is False
        self.poses.append(np.asarray(pose, dtype=np.float64).copy())

    def update_spectrum_station(
        self,
        station_records: tuple[tuple[object, ...], ...],
        *,
        pose_idx: int,
        generative_contract_hash_sha256: str,
    ) -> None:
        """Record one atomic station update."""
        self.updates.append(
            (
                station_records,
                int(pose_idx),
                str(generative_contract_hash_sha256),
            )
        )

    def serialized_state(self) -> bytes:
        """Return stable bytes for trace hashing."""
        self.serialization_calls += 1
        return f"updates={len(self.updates)}".encode("ascii")

    def posterior_snapshot(self) -> _Posterior:
        """Return the deterministic test posterior."""
        self.posterior_calls += 1
        return _Posterior()


def test_live_builder_uses_run_context_without_synthetic_log(
    monkeypatch,
    tmp_path,
) -> None:
    """Live construction must resolve its forward model directly from context."""
    from pf import replay as replay_module

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

    monkeypatch.setattr(replay_module, "ResolvedForwardContext", _Resolver)
    monkeypatch.setattr(
        replay_module,
        "_build_estimator_from_forward_context",
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
        "measurement_log_digest": "unavailable",
        "config_hash": None,
        "inference_isotopes": None,
    }


def test_replay_output_uses_shared_atomic_bundle_without_changing_bytes(
    monkeypatch,
    tmp_path,
) -> None:
    """Shared bundle publication must preserve the established result bytes."""
    from pf import replay as replay_module

    writes: dict[str, bytes] = {}
    publication: dict[str, object] = {}

    class _Publisher:
        """Capture shared atomic bundle operations without touching disk."""

        def __init__(self, target: object, *, policy: str) -> None:
            """Record the target and publication policy."""
            publication["target"] = target
            publication["policy"] = policy

        def __enter__(self) -> "_Publisher":
            """Enter the fake publisher lifetime."""
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: object,
        ) -> None:
            """Leave the fake publisher lifetime."""
            del exc_type, exc, traceback

        def write_bytes(self, name: str, payload: bytes) -> None:
            """Capture one exact bundle member."""
            writes[name] = payload

        def publish(self) -> None:
            """Record successful atomic publication."""
            publication["published"] = True

    class _OutputEstimator:
        """Provide the result metadata required by replay publication."""

        estimator_variant = "test"
        config_hash = "b" * 64
        resolved_config_hash = "c" * 64
        candidate_isotopes = ("Cs-137",)
        isotopes = ("Cs-137",)

        def posterior_snapshot(self) -> SimpleNamespace:
            """Return one deterministic posterior payload."""
            return SimpleNamespace(
                to_dict=lambda: {"structural_model_manifest": {"kind": "test"}}
            )

        def serialized_state(self) -> bytes:
            """Return stable final-state bytes."""
            return b"pf-state"

        def structural_transition_diagnostics(self) -> dict[str, object]:
            """Return the structural diagnostics required by publication."""
            return {
                "posterior_semantics": "test",
                "structural_kernel_family": "test",
                "structural_kernel_target_preserving": True,
                "structural_kernel_exact_rj": True,
                "reversible_jump_mcmc_used": True,
            }

        def posterior_predictive_check(self) -> dict[str, object]:
            """Return one deterministic predictive diagnostic."""
            return {"status": "test"}

        def joint_isotope_order(self) -> tuple[str, ...]:
            """Return the active isotope ordering."""
            return ("Cs-137",)

    log = SimpleNamespace(
        schema_version=2,
        log_sha256="d" * 64,
        resolved_config_sha256="e" * 64,
        run_manifest={"full_spectrum_contract_hash_sha256": "f" * 64},
    )
    monkeypatch.setattr(replay_module, "AtomicBundlePublisher", _Publisher)

    result = replay_module._write_replay_outputs(
        tmp_path / "result",
        estimator=_OutputEstimator(),  # type: ignore[arg-type]
        trace=({"b": 2, "a": 1},),
        log=log,  # type: ignore[arg-type]
    )

    assert result == tmp_path / "result"
    assert publication == {
        "target": tmp_path / "result",
        "policy": "create",
        "published": True,
    }
    assert writes["pf_trace.jsonl"] == b'{"a":1,"b":2}\n'
    assert writes["pf_posterior.json"] == replay_module.canonical_json_bytes(
        {"structural_model_manifest": {"kind": "test"}}
    )
    assert set(writes) == {
        "pf_posterior.json",
        "pf_trace.jsonl",
        "pf_diagnostics.json",
    }


def _with_records(
    log: MeasurementLog,
    records: tuple[object, ...],
) -> MeasurementLog:
    """Return an in-memory log with a deliberately altered record sequence."""
    return MeasurementLog(
        run_manifest=log.run_manifest,
        runtime_config=log.runtime_config,
        environment=log.environment,
        forward_model_manifest=log.forward_model_manifest,
        records=records,  # type: ignore[arg-type]
        path=log.path,
    )


def _as_single_completed_station(log: MeasurementLog) -> MeasurementLog:
    """Return a log whose records form one completed detector station."""
    first = log.records[0]
    station_records = []
    for index, record in enumerate(log.records):
        metadata = dict(record.metadata)
        metadata.pop("station_complete", None)
        if index == len(log.records) - 1:
            metadata["station_complete"] = True
        station_records.append(
            replace(
                record,
                station_id=0,
                detector_pose_xyz=first.detector_pose_xyz,
                detector_quat_wxyz=first.detector_quat_wxyz,
                metadata=metadata,
            )
        )
    return _with_records(log, tuple(station_records))


def _with_authenticated_candidate_isotopes(
    log: MeasurementLog,
    isotopes: tuple[str, ...],
    *,
    rebuild_forward_manifest: bool,
) -> MeasurementLog:
    """Return a synthetic log whose changed isotope selection remains hash-bound."""
    runtime_config = dict(log.runtime_config)
    runtime_config["candidate_isotopes"] = list(isotopes)
    resolved_hash = runtime_sha256_json(runtime_config)
    run_manifest = dict(log.run_manifest)
    run_manifest["isotopes"] = list(isotopes)
    run_manifest["resolved_config_sha256"] = resolved_hash
    forward_manifest = dict(log.forward_model_manifest)
    if rebuild_forward_manifest:
        assert log.path is not None
        forward_manifest = build_forward_model_manifest(
            runtime_config=runtime_config,
            environment=log.environment,
            obstacle_layout_path=run_manifest.get("obstacle_layout_path"),
            isotopes=isotopes,
            repository_commit=run_manifest["repository_commit"],
            resolved_config_sha256=resolved_hash,
            run_root=log.path,
        )
    return MeasurementLog(
        run_manifest=run_manifest,
        runtime_config=runtime_config,
        environment=log.environment,
        forward_model_manifest=forward_manifest,
        records=log.records,
        path=log.path,
    )


def test_full_spectrum_model_may_cover_a_candidate_isotope_superset(
    tmp_path,
) -> None:
    """A logged Cs/Co run may use an authenticated Cs/Co/Eu model basis."""
    log = load_measurement_log(
        make_measurement_log(tmp_path / "measurement-log", record_count=1)
    )
    subset_log = _with_authenticated_candidate_isotopes(
        log,
        ("Co-60", "Cs-137"),
        rebuild_forward_manifest=True,
    )

    model = validate_local_full_spectrum_contract(subset_log)

    assert {"Co-60", "Cs-137", "Eu-154"} == {
        str(row["isotope"]) for row in model.line_identity
    }


def test_full_spectrum_context_rejects_unauthenticated_candidate_changes(
    tmp_path,
) -> None:
    """A candidate change without a matching runtime manifest must fail closed."""
    log = load_measurement_log(
        make_measurement_log(tmp_path / "measurement-log", record_count=1)
    )
    invalid_log = _with_authenticated_candidate_isotopes(
        log,
        ("Co-60", "Cs-137", "Eu-154", "Xe-133"),
        rebuild_forward_manifest=False,
    )

    with pytest.raises(PFReplayError, match="forward context"):
        validate_local_full_spectrum_contract(invalid_log)


def test_spectrum_record_forwards_only_raw_spectrum_and_action_geometry(
    tmp_path,
) -> None:
    """Replay must not reconstruct isotope counts or auxiliary likelihoods."""
    log = load_measurement_log(
        make_measurement_log(tmp_path / "measurement-log", record_count=1)
    )

    replay_row = measurement_record_to_spectrum_input(log.records[0])

    assert len(replay_row) == 4
    spectrum, fe_index, pb_index, live_time_s = replay_row
    assert isinstance(spectrum, np.ndarray)
    assert spectrum.dtype == np.int64
    np.testing.assert_array_equal(spectrum, log.records[0].spectrum_counts)
    assert (fe_index, pb_index, live_time_s) == (0, 0, 1.0)


def test_spectrum_record_rejects_fractional_event_weights() -> None:
    """The replay adapter fails closed before a weighted spectrum reaches PF."""
    record = SimpleNamespace(
        spectrum_counts=np.asarray([1.0, 0.5], dtype=np.float64),
        fe_orientation_index=0,
        pb_orientation_index=0,
        live_time_s=1.0,
    )

    with pytest.raises(PFReplayError, match="raw nonnegative int64"):
        measurement_record_to_spectrum_input(record)


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    (
        ("spectrum_counts", np.asarray([1, 2], dtype=np.int32)),
        ("fe_orientation_index", True),
        ("pb_orientation_index", "0"),
        ("live_time_s", True),
        ("live_time_s", "1.0"),
    ),
)
def test_spectrum_record_rejects_scalar_and_dtype_coercion(
    field_name: str,
    invalid: object,
) -> None:
    """The final PF adapter independently preserves raw observation types."""
    values: dict[str, object] = {
        "spectrum_counts": np.asarray([1, 2], dtype=np.int64),
        "fe_orientation_index": 0,
        "pb_orientation_index": 0,
        "live_time_s": 1.0,
    }
    values[field_name] = invalid

    with pytest.raises(PFReplayError):
        measurement_record_to_spectrum_input(SimpleNamespace(**values))


def test_replay_groups_records_at_durable_station_boundaries(tmp_path) -> None:
    """One completed station becomes exactly one joint spectrum update."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=4,
            station_complete_markers=True,
        )
    )
    estimator = _RecordingEstimator()

    trace = replay_records(log, estimator)

    assert len(trace) == 4
    assert len(estimator.updates) == 2
    assert [len(update[0]) for update in estimator.updates] == [2, 2]
    assert [update[1] for update in estimator.updates] == [0, 1]
    assert all(
        update[2]
        == log.run_manifest["full_spectrum_contract_hash_sha256"]
        for update in estimator.updates
    )


def test_replay_delegates_prefix_grouping_and_alignment_to_runtime_views(
    tmp_path,
    monkeypatch,
) -> None:
    """Replay must consume the shared prefix, station, and array view APIs."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=4,
            station_complete_markers=True,
        )
    )
    calls: list[str] = []
    original_prefix = MeasurementLog.prefix
    original_station_view = MeasurementLog.station_view
    original_array_view = MeasurementLogStationView.array_view

    def prefix(self: MeasurementLog, record_count: int) -> MeasurementLog:
        """Record shared prefix selection before delegating."""
        calls.append("prefix")
        return original_prefix(self, record_count)

    def station_view(self: MeasurementLog) -> MeasurementLogStationView:
        """Record shared station grouping before delegating."""
        calls.append("station_view")
        return original_station_view(self)

    def array_view(self: MeasurementLogStationView) -> MeasurementLogArrayView:
        """Record shared array alignment before delegating."""
        calls.append("array_view")
        return original_array_view(self)

    monkeypatch.setattr(MeasurementLog, "prefix", prefix)
    monkeypatch.setattr(MeasurementLog, "station_view", station_view)
    monkeypatch.setattr(MeasurementLogStationView, "array_view", array_view)

    replay_records(log, _RecordingEstimator())

    assert calls == ["prefix", "station_view", "array_view"]


def test_replay_reuses_trace_payload_while_estimator_state_is_unchanged(
    tmp_path,
) -> None:
    """Intermediate shield views must not reserialize an unchanged PF state."""
    source_log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=4,
            station_complete_markers=True,
        )
    )
    log = _as_single_completed_station(source_log)
    estimator = _RecordingEstimator()

    trace = replay_records(log, estimator)

    assert len(trace) == 4
    assert estimator.serialization_calls == 2
    assert estimator.posterior_calls == 2
    assert len({row["state_sha256"] for row in trace[:3]}) == 1
    assert trace[2]["state_sha256"] != trace[3]["state_sha256"]
    trace[0]["posterior"]["cardinality"]["Cs-137"]["0"] = 0.0
    assert trace[1]["posterior"]["cardinality"]["Cs-137"]["0"] == 1.0


def test_replay_trace_cache_is_invalidated_for_callbacks(tmp_path) -> None:
    """An external callback must force a fresh trace state at every record."""
    source_log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=4,
            station_complete_markers=True,
        )
    )
    log = _as_single_completed_station(source_log)
    estimator = _RecordingEstimator()

    def callback(*_args: object) -> None:
        """Exercise the conservative callback invalidation boundary."""

    replay_records(
        log,
        estimator,
        pre_record_callback=callback,
    )

    assert estimator.serialization_calls == len(log.records)
    assert estimator.posterior_calls == len(log.records)


def test_replay_rejects_an_uncommitted_final_station(tmp_path) -> None:
    """A crash tail cannot be treated as a complete likelihood batch."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=2,
            station_complete_markers=False,
        )
    )

    with pytest.raises(PFReplayError, match="lacks station_complete"):
        replay_records(log, _RecordingEstimator())


@pytest.mark.parametrize("stop_after", (True, "1", 1.5, -1, 5))
def test_replay_rejects_prefix_coercion_and_clamping(
    tmp_path,
    stop_after: object,
) -> None:
    """Replay cannot truncate floats or clamp an out-of-range prefix."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=4,
            station_complete_markers=True,
        )
    )

    with pytest.raises(PFReplayError):
        replay_records(
            log,
            _RecordingEstimator(),
            stop_after=stop_after,  # type: ignore[arg-type]
        )


def test_replay_requires_a_completed_selected_prefix(tmp_path) -> None:
    """A prefix ending inside a station is not a valid PF observation batch."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=4,
            station_complete_markers=True,
        )
    )

    with pytest.raises(PFReplayError, match="selected replay boundary"):
        replay_records(log, _RecordingEstimator(), stop_after=1)


def test_replay_rejects_station_id_gaps(tmp_path) -> None:
    """A skipped station identifier cannot create a new valid PF pose."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=4,
            station_complete_markers=True,
        )
    )
    malformed = _with_records(
        log,
        (
            log.records[0],
            log.records[1],
            replace(log.records[2], station_id=2),
            replace(log.records[3], station_id=2),
        ),
    )

    with pytest.raises(PFReplayError, match="contiguous"):
        replay_records(malformed, _RecordingEstimator())


def test_replay_rejects_quaternion_drift_within_station(tmp_path) -> None:
    """A detector rotation cannot be ignored while station XYZ stays fixed."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=2,
            station_complete_markers=True,
        )
    )
    malformed = _with_records(
        log,
        (
            log.records[0],
            replace(
                log.records[1],
                detector_quat_wxyz=(0.0, 1.0, 0.0, 0.0),
            ),
        ),
    )

    with pytest.raises(PFReplayError, match="pose and quaternion"):
        replay_records(malformed, _RecordingEstimator())


def test_replay_rejects_noncanonical_record_ids(tmp_path) -> None:
    """Trace identifiers must remain bound to exact causal row order."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=2,
            station_complete_markers=True,
        )
    )
    malformed = _with_records(
        log,
        (
            log.records[0],
            replace(log.records[1], action_id=4),
        ),
    )

    with pytest.raises(PFReplayError, match="measurement-action order"):
        replay_records(malformed, _RecordingEstimator())


def test_replay_rejects_detector_pose_outside_logged_environment(tmp_path) -> None:
    """Logged runtime bounds override any conflicting PF solver support."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=1,
            station_complete_markers=True,
        )
    )
    malformed = _with_records(
        log,
        (
            replace(
                log.records[0],
                detector_pose_xyz=(2.5, 0.25, 0.4),
            ),
        ),
    )

    estimator = _RecordingEstimator()
    estimator.pf_config = SimpleNamespace(position_max=(100.0, 100.0, 100.0))

    with pytest.raises(PFReplayError, match="outside"):
        replay_records(malformed, estimator)


def test_replay_rejects_non_string_contract_hash(tmp_path) -> None:
    """A numeric hash-like value cannot be stringified at PF ingestion."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=1,
            station_complete_markers=True,
        )
    )
    malformed = MeasurementLog(
        run_manifest={
            **dict(log.run_manifest),
            "full_spectrum_contract_hash_sha256": 1,
        },
        runtime_config=log.runtime_config,
        environment=log.environment,
        forward_model_manifest=log.forward_model_manifest,
        records=log.records,
        path=log.path,
    )

    with pytest.raises(PFReplayError, match="SHA-256"):
        replay_records(malformed, _RecordingEstimator())


@pytest.mark.parametrize("seed", (True, "0", 0.5, -1))
def test_build_replay_rejects_seed_coercion(tmp_path, seed: object) -> None:
    """The replay root seed must be an exact nonnegative integer."""
    log = load_measurement_log(
        make_measurement_log(tmp_path / "measurement-log", record_count=1)
    )

    with pytest.raises(PFReplayError, match="seed"):
        build_replay_estimator(
            log,
            {},
            profile="pf_strict",
            seed=seed,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "config_text",
    (
        (
            '{"pure_pf_schema_version":1,'
            '"pure_pf_schema_version":1,'
            '"estimator_profile":"pf_strict"}'
        ),
        (
            '{"pure_pf_schema_version":1,'
            '"estimator_profile":"pf_strict",'
            '"num_particles":NaN}'
        ),
    ),
)
def test_replay_config_file_uses_strict_json(
    tmp_path,
    config_text: str,
) -> None:
    """Duplicate keys and non-finite constants fail before estimator creation."""
    log_path = make_measurement_log(
        tmp_path / "measurement-log",
        record_count=1,
    )
    config_path = tmp_path / "replay.json"
    config_path.write_text(config_text, encoding="utf-8")

    with pytest.raises(PFReplayError):
        replay_measurement_log(log_path, config_path)
