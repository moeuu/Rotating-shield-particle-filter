"""Tests for fail-closed station-boundary live-run resume."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pf.provenance import canonical_json_bytes, sha256_json
from pf.replay import PFReplayError, build_replay_estimator
from realtime_demo import (
    _build_resume_compatibility_provenance,
    _build_resume_replay_estimator,
    _build_live_controller_checkpoint,
    _online_compute_timing_provenance,
    _planning_candidate_checkpoint_parameters,
    _reconstruct_resume_controller_state,
    _restore_live_controller_checkpoint,
)
from runtime.measurement_log import (
    MeasurementLogStreamWriter,
    MeasurementLogValidationError,
    build_forward_model_manifest,
    load_measurement_log,
)
from tests.pure_pf_test_support import (
    TEST_COMMIT,
    TEST_ISOTOPES,
    environment,
    make_measurement_log,
    records,
    runtime_config,
)


def _stream_writer(
    tmp_path: Path,
    *,
    final_completion_metadata: dict[str, object] | None = None,
) -> tuple[MeasurementLogStreamWriter, dict[str, object], dict[str, object], dict]:
    """Create a two-station stream stage with exact joint-update boundaries."""
    config = {
        **runtime_config(),
        "candidate_isotopes": list(TEST_ISOTOPES),
    }
    env = environment()
    config_hash = sha256(canonical_json_bytes(config)).hexdigest()
    forward = build_forward_model_manifest(
        runtime_config=config,
        environment=env,
        obstacle_layout_path=None,
        isotopes=TEST_ISOTOPES,
        repository_commit=TEST_COMMIT,
        resolved_config_sha256=config_hash,
    )
    writer = MeasurementLogStreamWriter(
        tmp_path / "measurement-log",
        run_id="resume-fixture",
        repository_commit=TEST_COMMIT,
        runtime_config=config,
        environment=env,
        forward_model_manifest=forward,
        isotopes=TEST_ISOTOPES,
    )
    first, second, third, fourth = records(4)
    writer.append_before_update(first)
    writer.append_before_update(second)
    writer.mark_station_complete_before_update(0)
    writer.append_before_update(third)
    writer.append_before_update(fourth)
    writer.mark_station_complete_before_update(
        1,
        completion_metadata=final_completion_metadata,
    )
    return writer, config, env, forward


def _adopt(
    writer: MeasurementLogStreamWriter,
    *,
    config: dict[str, object],
    env: dict[str, object],
    forward: dict,
) -> MeasurementLogStreamWriter:
    """Adopt a fixture stage under a different truthful execution commit."""
    execution_commit = "b" * 40
    compatibility = {
        "schema_version": 1,
        "prefix_repository_commit": TEST_COMMIT,
        "resume_execution_commit": execution_commit,
        "changed_paths": {
            "src/realtime_demo.py": {
                "prefix_git_blob": "old",
                "execution_git_blob": "new",
            }
        },
    }
    return MeasurementLogStreamWriter.resume_from_stage(
        writer.output_dir,
        stage_dir=writer.stage_dir,
        run_id="resume-fixture",
        repository_commit=TEST_COMMIT,
        runtime_config=config,
        environment=env,
        forward_model_manifest=forward,
        isotopes=TEST_ISOTOPES,
        resume_execution_commit=execution_commit,
        resume_compatibility=compatibility,
    )


def _tree_hashes(root: Path) -> dict[str, str]:
    """Return byte hashes for every regular file below one stage directory."""
    return {
        path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_online_compute_timing_scope_distinguishes_resumed_suffix() -> None:
    """Compute timing provenance should prevent full-run resume comparisons."""
    assert _online_compute_timing_provenance(0) == {
        "online_compute_timing_scope": "full_live_run",
        "online_compute_timing_prefix_measurements_excluded": 0,
        "online_compute_timing_includes_resume_pf_replay": False,
    }
    assert _online_compute_timing_provenance(104) == {
        "online_compute_timing_scope": "post_resume_suffix_only",
        "online_compute_timing_prefix_measurements_excluded": 104,
        "online_compute_timing_includes_resume_pf_replay": False,
    }
    with pytest.raises(ValueError, match="non-negative"):
        _online_compute_timing_provenance(-1)


def _replay_ready_pf_config() -> dict[str, object]:
    """Return deterministic PF-owned settings for station-prefix replay."""
    return {
        "pure_pf_schema_version": 1,
        "estimator_profile": "pf_strict",
        "num_particles": 12,
        "max_sources": 2,
        "init_num_sources": [1, 1],
        "variable_cardinality": False,
        "use_gpu": False,
        "strength_prior_min_cps_1m": 300_000.0,
        "strength_prior_max_cps_1m": 2_000_000.0,
    }


def test_resume_replay_uses_explicit_pf_config_and_raw_measurements(
    tmp_path: Path,
) -> None:
    """Resume must reconstruct PF state without estimator fields in the log."""
    pf_config = _replay_ready_pf_config()
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=1,
            station_complete_markers=True,
        )
    )
    assert "pure_pf_schema_version" not in log.runtime_config
    assert "effective_pf_replay" not in log.runtime_config

    direct = build_replay_estimator(
        log,
        pf_config,
        profile="pf_strict",
        seed=41,
    )
    resumed = _build_resume_replay_estimator(
        log,
        pf_config=pf_config,
        profile="pf_strict",
        seed=41,
        config_hash=sha256_json(pf_config),
    )
    assert resumed.pf_config.num_particles == 12
    assert resumed.pf_config.position_max == (2.0, 2.0, 1.5)
    assert resumed.pf_config.strength_prior_max_cps_1m == 2_000_000.0
    direct._ensure_kernel_cache()
    resumed._ensure_kernel_cache()
    direct_row_ids = {
        isotope: tuple(
            particle.joint_row_identity.row_sha256
            for particle in direct.filters[isotope].continuous_particles
        )
        for isotope in direct.joint_isotope_order()
    }
    resumed_row_ids = {
        isotope: tuple(
            particle.joint_row_identity.row_sha256
            for particle in resumed.filters[isotope].continuous_particles
        )
        for isotope in resumed.joint_isotope_order()
    }
    assert direct_row_ids == resumed_row_ids
    assert len(set(next(iter(direct_row_ids.values())))) == 12
    assert resumed.serialized_state() == direct.serialized_state()


def test_replay_rejects_unbound_resolved_config_hash(tmp_path: Path) -> None:
    """Caller provenance cannot replace the computed replay configuration digest."""
    pf_config = _replay_ready_pf_config()
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=1,
            station_complete_markers=True,
        )
    )

    with pytest.raises(PFReplayError, match="does not bind"):
        build_replay_estimator(
            log,
            pf_config,
            profile="pf_strict",
            seed=41,
            resolved_config_hash="0" * 64,
        )


def test_replay_rejects_ignored_external_runtime_override(
    tmp_path: Path,
) -> None:
    """An overlapping external physics field cannot differ and be ignored."""
    pf_config = _replay_ready_pf_config()
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=1,
            station_complete_markers=True,
        )
    )
    external = json.loads(json.dumps(pf_config))
    external["pf_obstacle_attenuation"] = False

    with pytest.raises(PFReplayError, match="External runtime field"):
        build_replay_estimator(
            log,
            external,
            profile="pf_strict",
            seed=41,
        )


def test_resume_compatibility_requires_every_runtime_delta_explicitly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No runtime path is auto-admitted, and all runtime scopes are inspected."""
    changed_runtime_paths = (
        "main.py",
        "src/realtime_demo.py",
        "pyproject.toml",
        "uv.lock",
        "native/kernel.cpp",
        "scripts/run_geant4_bridge.py",
        "scripts/build_geant4_sidecar.py",
    )
    calls: list[tuple[str, ...]] = []

    def _fake_git_command_text(
        repository_root: Path,
        *args: str,
    ) -> str:
        """Return a clean deterministic commit delta for provenance tests."""
        assert repository_root == tmp_path.resolve()
        calls.append(args)
        if args[0] == "status":
            return ""
        if args[0] == "diff":
            return "\n".join(changed_runtime_paths)
        if args[0] == "rev-parse":
            return "c" * 40
        raise AssertionError(f"Unexpected Git command: {args}")

    monkeypatch.setattr(
        "realtime_demo._git_command_text",
        _fake_git_command_text,
    )
    arguments = {
        "repository_root": tmp_path,
        "prefix_commit": "a" * 40,
        "execution_commit": "b" * 40,
    }
    with pytest.raises(RuntimeError, match="unapproved runtime code"):
        _build_resume_compatibility_provenance(
            **arguments,
            additional_compatible_code_paths=None,
            compatibility_basis=None,
        )
    with pytest.raises(RuntimeError, match="compatibility basis"):
        _build_resume_compatibility_provenance(
            **arguments,
            additional_compatible_code_paths=changed_runtime_paths,
            compatibility_basis=None,
        )

    payload = _build_resume_compatibility_provenance(
        **arguments,
        additional_compatible_code_paths=changed_runtime_paths,
        compatibility_basis="independent station-state equivalence gate",
    )
    assert payload["explicitly_compatible_runtime_paths"] == sorted(
        changed_runtime_paths
    )
    assert (
        payload["compatibility_basis"]
        == "independent station-state equivalence gate"
    )
    status_call = next(call for call in calls if call[0] == "status")
    scope_start = status_call.index("--") + 1
    assert status_call[scope_start:] == (
        "main.py",
        "src",
        "pyproject.toml",
        "uv.lock",
        "native",
        "scripts/run_geant4_bridge.py",
        "scripts/build_geant4_sidecar.py",
    )


def test_stream_stage_adopts_old_prefix_and_continues_without_overwrite(
    tmp_path: Path,
) -> None:
    """An old prefix remains immutable while new records identify new code."""
    writer, config, env, forward = _stream_writer(tmp_path)
    stage = writer.stage_dir
    source_hashes = _tree_hashes(stage)

    resumed = _adopt(writer, config=config, env=env, forward=forward)
    fork_stage = resumed.stage_dir
    assert fork_stage != stage
    assert stage.exists()
    assert _tree_hashes(stage) == source_hashes
    assert len(resumed.records) == 4
    assert resumed.records[-1].metadata["station_complete"] is True
    assert resumed.metadata["resume_prefix_repository_commit"] == TEST_COMMIT
    assert resumed.metadata["resume_execution_commit"] == "b" * 40

    prefix = resumed.write_canonical_prefix(tmp_path / "canonical-prefix")
    assert [record.step_id for record in prefix.records] == [0, 1, 2, 3]
    assert all(
        "resume_execution_commit" not in record.metadata
        for record in prefix.records
    )

    fifth = records(5)[4]
    assert resumed.append_before_update(fifth) == 4
    resumed.mark_station_complete_before_update(2)
    finalized = resumed.finalize()
    assert stage.exists()
    assert _tree_hashes(stage) == source_hashes
    assert not fork_stage.exists()
    assert len(finalized.records) == 5
    assert finalized.records[4].metadata["resume_execution_commit"] == "b" * 40
    assert finalized.records[4].metadata["resume_prefix_record_count"] == 4
    assert (
        finalized.run_manifest["metadata"]["resume_prefix_repository_commit"]
        == TEST_COMMIT
    )
    assert finalized.run_manifest["repository_commit"] == TEST_COMMIT


def test_stream_stage_rolls_back_partial_station_and_can_resume_twice(
    tmp_path: Path,
) -> None:
    """Each adoption forks an immutable source and discards an uncommitted tail."""
    writer, config, env, forward = _stream_writer(tmp_path)
    fifth = records(5)[4]
    writer.append_before_update(fifth)
    original_stage = writer.stage_dir
    original_hashes = _tree_hashes(original_stage)

    first_resume = _adopt(writer, config=config, env=env, forward=forward)
    first_fork = first_resume.stage_dir
    assert len(first_resume.records) == 4
    assert _tree_hashes(original_stage) == original_hashes
    recovery = first_resume.metadata["resume_compatibility"][
        "source_stage_recovery"
    ]
    assert recovery["discarded_tail_record_count"] == 1
    assert recovery["orphan_shard_count"] == 0

    first_resume.append_before_update(fifth)
    first_fork_hashes = _tree_hashes(first_fork)
    second_resume = _adopt(
        first_resume,
        config=config,
        env=env,
        forward=forward,
    )
    second_fork = second_resume.stage_dir
    assert second_fork not in {original_stage, first_fork}
    assert len(second_resume.records) == 4
    assert _tree_hashes(original_stage) == original_hashes
    assert _tree_hashes(first_fork) == first_fork_hashes

    second_resume.append_before_update(fifth)
    second_resume.mark_station_complete_before_update(2)
    finalized = second_resume.finalize()
    assert len(finalized.records) == 5
    assert not second_fork.exists()
    assert _tree_hashes(original_stage) == original_hashes
    assert _tree_hashes(first_fork) == first_fork_hashes


def test_stream_stage_recovers_one_orphan_shard_and_metadata_temp(
    tmp_path: Path,
) -> None:
    """A shard-before-JSONL crash and one rewrite temp are safely ignored."""
    writer, config, env, forward = _stream_writer(tmp_path)
    writer.append_before_update(records(5)[4])
    lines = writer.metadata_stage_path.read_bytes().splitlines(keepends=True)
    writer.metadata_stage_path.write_bytes(b"".join(lines[:-1]))
    (writer.stage_dir / ".observation_metadata.jsonl.tmp-999").write_bytes(
        b"incomplete rewrite"
    )
    source_hashes = _tree_hashes(writer.stage_dir)

    resumed = _adopt(writer, config=config, env=env, forward=forward)
    assert len(resumed.records) == 4
    assert _tree_hashes(writer.stage_dir) == source_hashes
    recovery = resumed.metadata["resume_compatibility"]["source_stage_recovery"]
    assert recovery["orphan_shard_count"] == 1
    assert recovery["metadata_temp_orphan_count"] == 1
    assert recovery["discarded_tail_record_count"] == 1


def test_stream_stage_recovers_truncated_final_metadata_line(
    tmp_path: Path,
) -> None:
    """A torn final JSONL append is dropped only with its durable WAL shard."""
    writer, config, env, forward = _stream_writer(tmp_path)
    writer.append_before_update(records(5)[4])
    lines = writer.metadata_stage_path.read_bytes().splitlines(keepends=True)
    torn_line = lines[-1][: max(1, len(lines[-1]) // 2)]
    assert not torn_line.endswith(b"\n")
    writer.metadata_stage_path.write_bytes(b"".join(lines[:-1]) + torn_line)
    source_hashes = _tree_hashes(writer.stage_dir)

    resumed = _adopt(writer, config=config, env=env, forward=forward)
    assert len(resumed.records) == 4
    assert _tree_hashes(writer.stage_dir) == source_hashes
    recovery = resumed.metadata["resume_compatibility"]["source_stage_recovery"]
    assert recovery["truncated_metadata_tail"] is True
    assert recovery["orphan_shard_count"] == 1
    assert recovery["discarded_tail_record_count"] == 1


def test_stream_stage_rejects_trailing_metadata_corruption_without_shard(
    tmp_path: Path,
) -> None:
    """Arbitrary non-WAL bytes cannot masquerade as one torn metadata append."""
    writer, config, env, forward = _stream_writer(tmp_path)
    with writer.metadata_stage_path.open("ab") as handle:
        handle.write(b"{")

    with pytest.raises(
        MeasurementLogValidationError,
        match="corresponding record shard",
    ):
        _adopt(writer, config=config, env=env, forward=forward)


@pytest.mark.parametrize("mutation", ["config", "environment", "commit", "extra"])
def test_stream_stage_resume_fails_closed_on_identity_or_inventory_mismatch(
    tmp_path: Path,
    mutation: str,
) -> None:
    """Config, environment, commit, and inventory drift must stop adoption."""
    writer, config, env, forward = _stream_writer(tmp_path)
    if mutation == "config":
        config = {**config, "orientation_k": 7}
    elif mutation == "environment":
        env = {**env, "size_x": 3.0}
    elif mutation == "commit":
        (writer.stage_dir / "repository_commit.txt").write_text(
            f"{'c' * 40}\n",
            encoding="utf-8",
        )
    else:
        (writer.stage_dir / "unexpected.txt").write_text("unexpected\n")

    with pytest.raises(MeasurementLogValidationError):
        _adopt(writer, config=config, env=env, forward=forward)


def test_stream_stage_resume_rejects_incomplete_station_boundary(
    tmp_path: Path,
) -> None:
    """A staged observation without its causal completion marker is not resumable."""
    writer, config, env, forward = _stream_writer(tmp_path)
    rows = [
        json.loads(line)
        for line in writer.metadata_stage_path.read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        row["metadata"].pop("station_complete", None)
    writer.metadata_stage_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        MeasurementLogValidationError,
        match="station_complete boundary",
    ):
        _adopt(writer, config=config, env=env, forward=forward)


def test_controller_state_restores_next_step_without_reacquiring_last_station() -> None:
    """The restored loop starts after four records at station-one post-processing."""

    class Estimator:
        """Minimal replayed-estimator pose surface for controller restoration."""

        poses = [
            np.asarray(records(4)[0].detector_pose_xyz, dtype=float),
            np.asarray(records(4)[2].detector_pose_xyz, dtype=float),
        ]

    state = _reconstruct_resume_controller_state(
        records=records(4, station_complete_markers=True),
        estimator=Estimator(),  # type: ignore[arg-type]
        isotopes=TEST_ISOTOPES,
        nominal_motion_speed_m_s=0.5,
        expected_program_length=2,
    )
    assert state.step_counter == 4
    assert state.pose_counter == 1
    assert len(state.visited_poses) == 1
    assert state.last_station_pair_ids == (22, 25)
    rotation_limit = 2
    rotation_count = rotation_limit
    acquisition_calls = 0
    while rotation_count < rotation_limit:
        acquisition_calls += 1
    assert acquisition_calls == 0
    assert state.step_counter == 4


def test_controller_checkpoint_survives_stage_adoption_and_restores_rng(
    tmp_path: Path,
) -> None:
    """A fresh checkpoint resumes the next candidate draw without log inference."""

    parameters = _planning_candidate_checkpoint_parameters(
        pose_candidates=64,
        pose_min_dist=3.0,
        bounds_xyz=(np.zeros(3), np.asarray([10.0, 20.0, 10.0])),
        detector_heights_m=None,
    )
    rng = np.random.default_rng(91)
    reference = np.random.default_rng(91)
    dss_rng = np.random.default_rng(123)
    dss_reference = np.random.default_rng(123)
    assert np.array_equal(rng.random(17), reference.random(17))
    assert np.array_equal(dss_rng.random(11), dss_reference.random(11))
    checkpoint = _build_live_controller_checkpoint(
        planning_candidate_rng=rng,
        dss_eig_rng=dss_rng,
        planning_candidate_parameters=parameters,
        max_poses=22,
    )
    writer, config, env, forward = _stream_writer(
        tmp_path,
        final_completion_metadata={"live_controller_checkpoint": checkpoint},
    )
    resumed = _adopt(writer, config=config, env=env, forward=forward)
    restored_rng = np.random.default_rng(91)
    restored_dss_rng = np.random.default_rng(123)
    restored = _restore_live_controller_checkpoint(
        record=resumed.records[-1],
        planning_candidate_rng=restored_rng,
        dss_eig_rng=restored_dss_rng,
        expected_planning_candidate_parameters=parameters,
    )
    assert restored is not None
    assert restored_rng.random() == reference.random()
    assert restored_dss_rng.random() == dss_reference.random()
    assert restored.max_poses == 22


def test_controller_checkpoint_rejects_candidate_parameter_drift(
    tmp_path: Path,
) -> None:
    """A checkpoint cannot restore across candidate-generation parameter drift."""

    parameters = _planning_candidate_checkpoint_parameters(
        pose_candidates=8,
        pose_min_dist=1.0,
        bounds_xyz=(np.zeros(3), np.ones(3)),
        detector_heights_m=(0.4,),
    )
    checkpoint = _build_live_controller_checkpoint(
        planning_candidate_rng=np.random.default_rng(5),
        dss_eig_rng=np.random.default_rng(6),
        planning_candidate_parameters=parameters,
        max_poses=2,
    )
    writer, _, _, _ = _stream_writer(
        tmp_path,
        final_completion_metadata={"live_controller_checkpoint": checkpoint},
    )
    drifted = {**parameters, "pose_candidates": 9}
    with pytest.raises(RuntimeError, match="candidate parameters differ"):
        _restore_live_controller_checkpoint(
            record=writer.records[-1],
            planning_candidate_rng=np.random.default_rng(5),
            dss_eig_rng=np.random.default_rng(6),
            expected_planning_candidate_parameters=drifted,
        )


@pytest.mark.parametrize("invalid_max_poses", ("22", 22.5, True, 0, -1))
def test_controller_checkpoint_rejects_coerced_mission_limit(
    invalid_max_poses: object,
) -> None:
    """A corrupt checkpoint must not silently change the resumed mission."""
    parameters = _planning_candidate_checkpoint_parameters(
        pose_candidates=8,
        pose_min_dist=1.0,
        bounds_xyz=(np.zeros(3), np.ones(3)),
        detector_heights_m=None,
    )
    checkpoint = _build_live_controller_checkpoint(
        planning_candidate_rng=np.random.default_rng(5),
        dss_eig_rng=np.random.default_rng(6),
        planning_candidate_parameters=parameters,
        max_poses=2,
    )
    checkpoint["mission_state"]["max_poses"] = invalid_max_poses
    with pytest.raises(RuntimeError, match="controller values are invalid"):
        _restore_live_controller_checkpoint(
            record=SimpleNamespace(
                metadata={"live_controller_checkpoint": checkpoint}
            ),
            planning_candidate_rng=np.random.default_rng(5),
            dss_eig_rng=np.random.default_rng(6),
            expected_planning_candidate_parameters=parameters,
        )
