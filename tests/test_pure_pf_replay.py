"""Causal replay tests for the schema-v2 full-spectrum PF boundary."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pf.replay import (
    PFReplayError,
    _spectrum_record,
    build_replay_estimator,
    replay_measurement_log,
    replay_records,
)
from runtime.measurement_log import MeasurementLog, load_measurement_log
from tests.pure_pf_test_support import make_measurement_log


class _Posterior:
    """Provide the minimal immutable posterior interface used by traces."""

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic schema-v2 test posterior."""
        return {"schema_version": 2, "cardinality": {"Cs-137": {"0": 1.0}}}


class _RecordingEstimator:
    """Record station updates without implementing inference."""

    estimator_variant = "test_pure_particle_filter"

    def __init__(self) -> None:
        """Initialize empty poses and update records."""
        self.poses: list[np.ndarray] = []
        self.kernel_cache: object | None = object()
        self.updates: list[tuple[tuple[tuple[object, ...], ...], int, str]] = []

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
        return f"updates={len(self.updates)}".encode("ascii")

    def posterior_snapshot(self) -> _Posterior:
        """Return the deterministic test posterior."""
        return _Posterior()


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


def test_spectrum_record_forwards_only_raw_spectrum_and_action_geometry(
    tmp_path,
) -> None:
    """Replay must not reconstruct isotope counts or auxiliary likelihoods."""
    log = load_measurement_log(
        make_measurement_log(tmp_path / "measurement-log", record_count=1)
    )

    replay_row = _spectrum_record(log.records[0])

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
        _spectrum_record(record)


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
        _spectrum_record(SimpleNamespace(**values))


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

    with pytest.raises(PFReplayError, match="quaternions"):
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

    with pytest.raises(PFReplayError, match="record order"):
        replay_records(malformed, _RecordingEstimator())


def test_replay_rejects_detector_pose_outside_logged_environment(tmp_path) -> None:
    """A finite pose cannot escape the environment bound to the PF support."""
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

    with pytest.raises(PFReplayError, match="outside"):
        replay_records(malformed, _RecordingEstimator())


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
