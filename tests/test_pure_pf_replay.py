"""End-to-end tests for deterministic, causal, batch-free PF replay."""

from __future__ import annotations

import ast
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from measurement.observation_model import build_runtime_observation_model
from pf.estimator import RotatingShieldPFConfig
from pf.profiles import apply_profile_to_config, enforce_pure_runtime_settings
from pf.provenance import canonical_json_bytes, json_safe, sha256_json
from pf.pure_estimator import PurePFEstimator
from pf.replay import (
    PFReplayError,
    _resolved_physical_config,
    build_replay_estimator,
    replay_measurement_log,
    replay_records,
    validate_local_forward_model,
)
from runtime.measurement_log import (
    MeasurementLog,
    MeasurementLogValidationError,
    load_measurement_log,
)
from tests.pure_pf_test_support import (
    TEST_ISOTOPES,
    make_measurement_log,
    records,
    replay_config,
)


def _effective_runtime_overrides(
    *,
    seed: int = 41,
    joint: bool = True,
    delayed: bool = False,
) -> dict[str, object]:
    """Return a complete live-effective PF block for replay parity tests."""
    candidates = np.asarray(
        replay_config()["replay_candidate_sources_xyz"], dtype=float
    )
    config = RotatingShieldPFConfig(
        estimator_profile="pf_strict",
        num_particles=12,
        max_sources=2,
        init_num_sources=[1, 1],
        birth_enable=False,
        use_gpu=False,
        parallel_isotope_updates=False,
        position_min=(0.0, 0.0, 0.0),
        position_max=(2.0, 2.0, 1.5),
    )
    apply_profile_to_config(config)
    return {
        "effective_pf_replay": {
            "api_settings": {
                "pf_random_seed": int(seed),
                "joint_observation_update": bool(joint),
                "delayed_resample_update": bool(delayed),
            },
            "pf_config": json_safe(config),
            "candidate_grid": {
                "point_count": int(candidates.shape[0]),
                "xyz_sha256": sha256_json(candidates),
                "candidate_sources_xyz": candidates.tolist(),
                "position_min_xyz_m": [0.0, 0.0, 0.0],
                "position_max_xyz_m": [2.0, 2.0, 1.5],
            },
        }
    }


def test_replay_binds_transport_asset_to_validated_run_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CWD collision must not replace the run-root model used by replay."""
    relative_path = Path("assets/transport.json")
    run_root = tmp_path / "measurement-log"
    cwd_root = tmp_path / "unrelated-cwd"
    run_asset = run_root / relative_path
    cwd_asset = cwd_root / relative_path
    run_asset.parent.mkdir(parents=True)
    cwd_asset.parent.mkdir(parents=True)
    run_payload = {
        "pf_transport_response_model": {
            "enabled": True,
            "by_isotope": {"Cs-137": {"scale": 1.25}},
        }
    }
    cwd_payload = {
        "pf_transport_response_model": {
            "enabled": True,
            "by_isotope": {"Cs-137": {"scale": 9.0}},
        }
    }
    run_asset.write_text(json.dumps(run_payload), encoding="utf-8")
    cwd_asset.write_text(json.dumps(cwd_payload), encoding="utf-8")
    monkeypatch.chdir(cwd_root)

    log = MeasurementLog(
        run_manifest={},
        runtime_config={
            "pf_transport_response_model_path": relative_path.as_posix(),
        },
        environment={},
        forward_model_manifest={},
        records=(),
        path=run_root,
    )
    physical_config = _resolved_physical_config(log)
    assert Path(physical_config["pf_transport_response_model_path"]) == (
        run_asset.resolve()
    )
    observation_model = build_runtime_observation_model(
        physical_config,
        isotopes=("Cs-137",),
    )
    assert (
        observation_model.transport_response_model
        == run_payload["pf_transport_response_model"]
    )


def test_measurement_log_schema_rejects_invalid_energy_axis() -> None:
    """Raw spectrum bins stay log data and their axis is validated by the schema."""
    with pytest.raises(MeasurementLogValidationError, match="strictly increasing"):
        replace(
            records(1)[0],
            energy_bin_edges_keV=np.asarray(
                [0.0, 500.0, 500.0, 1500.0, 2000.0],
                dtype=np.float64,
            ),
        )


def test_active_pure_modules_do_not_import_batch_estimators() -> None:
    """The active runtime must not import deleted solvers at any scope."""
    root = Path(__file__).resolve().parents[1]
    active_modules = (
        root / "src/pf/pure_estimator.py",
        root / "src/pf/estimator.py",
        root / "src/pf/particle_filter.py",
        root / "src/pf/replay.py",
        root / "src/planning/dss_pp.py",
        root / "src/realtime_demo.py",
    )
    forbidden = {
        "pf.sparse_evidence",
        "pf.surface_map",
        "three_d_estimation",
    }
    imported: set[str] = set()
    for module in active_modules:
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)
    assert imported.isdisjoint(forbidden)


def test_replay_outputs_are_deterministic_and_provenance_hashes_bind_inputs(
    tmp_path: Path,
) -> None:
    """The same strict input must produce one reproducible, causally bound bundle."""
    log_path = make_measurement_log(tmp_path / "measurement-log", record_count=3)
    config_path = tmp_path / "pf-config.json"
    config_path.write_bytes(canonical_json_bytes(replay_config()))
    first_output = tmp_path / "first-result"
    second_output = tmp_path / "second-result"

    first_estimator, _ = replay_measurement_log(
        log_path,
        config_path,
        profile="pf_strict",
        seed=17,
        output_dir=first_output,
    )
    replay_measurement_log(
        log_path,
        config_path,
        profile="pf_strict",
        seed=17,
        output_dir=second_output,
    )

    expected_files = {
        "pf_posterior.json",
        "pf_trace.jsonl",
        "pf_diagnostics.json",
    }
    assert {path.name for path in first_output.iterdir()} == expected_files
    for filename in expected_files:
        assert (first_output / filename).read_bytes() == (
            second_output / filename
        ).read_bytes()
    posterior = json.loads((first_output / "pf_posterior.json").read_text())
    diagnostics = json.loads((first_output / "pf_diagnostics.json").read_text())
    provenance = posterior["provenance"]
    assert provenance["config_sha256"] == sha256(config_path.read_bytes()).hexdigest()
    assert provenance["resolved_config_sha256"] == sha256_json(
        enforce_pure_runtime_settings(replay_config(), profile="pf_strict")
    )
    assert provenance["resolved_config_sha256"] == provenance["config_sha256"]
    assert diagnostics["record_count"] == 3
    assert posterior["final_estimate_source"] == "pf_posterior"
    assert "forbidden_batch_methods_invoked" not in diagnostics
    assert "batch_methods_invoked" not in diagnostics
    assert "uses_all_history_batch_fit" not in posterior
    assert "mle_estimate" not in posterior
    assert "surface_map_estimate" not in posterior
    assert "strength_refit_applied" not in posterior
    assert posterior["posterior_semantics"] == (
        "fixed_cardinality_sequential_particle_filter"
    )
    assert posterior["structural_kernel_family"] == (
        "fixed_cardinality_no_structural_moves"
    )
    assert posterior["structural_kernel_target_preserving"] is True
    assert posterior["structural_kernel_exact_rj"] is False
    assert posterior["reversible_jump_mcmc_used"] is False
    assert (
        diagnostics["structural_transition_provenance"]
        == (posterior["structural_transition_provenance"])
    )


def test_cpu_batched_count_kernel_matches_uncompressed_oracle_and_equation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU views are one packed batch and retain the configured count equation."""
    log = load_measurement_log(
        make_measurement_log(tmp_path / "measurement-log", record_count=1)
    )
    config = {
        **replay_config(),
        "gpu_device": "cpu",
        "gpu_dtype": "float64",
        "use_tempering": False,
        "resample_threshold": 0.0,
        "min_particles": 12,
        "max_particles": 12,
    }
    estimator = build_replay_estimator(
        log,
        config,
        profile="pf_strict",
        seed=101,
    )
    estimator._ensure_kernel_cache()
    filt = next(iter(estimator.filters.values()))
    fe_indices = np.asarray([0, 3, 7, 1], dtype=np.int64)
    pb_indices = np.asarray([7, 2, 0, 5], dtype=np.int64)
    live_times = np.asarray([0.25, 0.5, 1.0, 1.75], dtype=np.float64)

    cpu_counts = filt._continuous_expected_counts_pair_sequence_cpu(
        pose_idx=0,
        fe_indices=fe_indices,
        pb_indices=pb_indices,
        live_times_s=live_times,
    )
    oracle_counts = (
        filt._continuous_expected_counts_pair_sequence_torch_uncompressed(
            pose_idx=0,
            fe_indices=fe_indices,
            pb_indices=pb_indices,
            live_times_s=live_times,
        )
        .detach()
        .cpu()
        .numpy()
    )
    np.testing.assert_allclose(cpu_counts, oracle_counts, rtol=1.0e-12, atol=1.0e-12)

    lam = cpu_counts[0]
    previous_log_weights = np.asarray(
        [particle.log_weight for particle in filt.continuous_particles],
        dtype=float,
    )
    observed = 20.0
    expected_log_weights = (
        previous_log_weights
        + observed * np.log(np.maximum(lam, 1.0e-12))
        - np.maximum(lam, 1.0e-12)
    )
    expected_weights = np.exp(expected_log_weights - np.max(expected_log_weights))
    expected_weights /= np.sum(expected_weights)

    def forbidden_gpu(*args: object, **kwargs: object) -> object:
        """Fail if the disabled GPU pair kernel is selected."""
        del args, kwargs
        raise AssertionError("use_gpu=false selected the GPU pair kernel")

    monkeypatch.setattr(filt, "_continuous_expected_counts_pair_torch", forbidden_gpu)
    filt.update_continuous_pair(
        z_obs=observed,
        pose_idx=0,
        fe_index=int(fe_indices[0]),
        pb_index=int(pb_indices[0]),
        live_time_s=float(live_times[0]),
    )
    np.testing.assert_allclose(
        filt.continuous_weights,
        expected_weights,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_resolved_hash_binds_replay_candidate_grid_inputs(tmp_path: Path) -> None:
    """Changing PF birth support must change resolved estimator provenance."""
    log = load_measurement_log(
        make_measurement_log(tmp_path / "measurement-log", record_count=1)
    )
    first_config = replay_config()
    second_config = {**first_config, "replay_candidate_spacing_m": 0.75}

    first = build_replay_estimator(log, first_config, profile="pf_strict", seed=3)
    second = build_replay_estimator(log, second_config, profile="pf_strict", seed=3)

    assert first.resolved_config_hash != second.resolved_config_hash


def test_prefix_causality_matches_full_log_stopped_at_same_record(
    tmp_path: Path,
) -> None:
    """A future suffix must not alter any state after the same causal prefix."""
    log = load_measurement_log(
        make_measurement_log(tmp_path / "measurement-log", record_count=4)
    )
    config = enforce_pure_runtime_settings(replay_config(), profile="pf_strict")
    raw_hash = sha256_json(replay_config())
    resolved_hash = sha256_json(config)

    full_estimator = build_replay_estimator(
        log,
        config,
        profile="pf_strict",
        seed=23,
        config_hash=raw_hash,
        resolved_config_hash=resolved_hash,
    )
    full_trace = replay_records(log, full_estimator, stop_after=2)

    prefix = log.prefix(2)
    prefix_estimator = build_replay_estimator(
        prefix,
        config,
        profile="pf_strict",
        seed=23,
        config_hash=raw_hash,
        resolved_config_hash=resolved_hash,
    )
    prefix_trace = replay_records(prefix, prefix_estimator)

    assert full_estimator.serialized_state() == prefix_estimator.serialized_state()
    assert full_trace == prefix_trace


@pytest.mark.parametrize(
    ("joint", "delayed"),
    ((True, False), (False, True)),
)
def test_station_prefix_never_infers_completion_from_eof_or_future_rows(
    tmp_path: Path,
    joint: bool,
    delayed: bool,
) -> None:
    """An exact-N log and the first N rows of a longer log have identical state."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=4,
            runtime_overrides=_effective_runtime_overrides(
                joint=joint,
                delayed=delayed,
            ),
            station_complete_markers=True,
        )
    )
    config = replay_config()
    for record_count in range(len(log.records) + 1):
        future_estimator = build_replay_estimator(
            log, config, profile="pf_strict", seed=41
        )
        future_trace = replay_records(
            log,
            future_estimator,
            stop_after=record_count,
        )

        exact_n = replace(
            log,
            run_manifest={**log.run_manifest, "record_count": record_count},
            records=log.records[:record_count],
        )
        exact_estimator = build_replay_estimator(
            exact_n, config, profile="pf_strict", seed=41
        )
        exact_trace = replay_records(exact_n, exact_estimator)

        assert future_estimator.serialized_state() == exact_estimator.serialized_state()
        assert future_trace == exact_trace
        expected_deferred = delayed and record_count % 2 == 1
        assert exact_estimator._defer_resample_birth is expected_deferred


def test_joint_replay_uses_only_explicit_station_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Joint updates execute once per marked station and reject bad transitions."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=5,
            runtime_overrides=_effective_runtime_overrides(),
            station_complete_markers=True,
        )
    )
    estimator = build_replay_estimator(
        log, replay_config(), profile="pf_strict", seed=41
    )
    original = estimator.update_pair_sequence
    group_sizes: list[int] = []

    def recording_update(sequence: object, *, pose_idx: int, **kwargs: object) -> None:
        records_sequence = tuple(sequence)  # type: ignore[arg-type]
        group_sizes.append(len(records_sequence))
        original(records_sequence, pose_idx=pose_idx, **kwargs)

    monkeypatch.setattr(estimator, "update_pair_sequence", recording_update)
    trace = replay_records(log, estimator)
    assert group_sizes == [2, 2, 1]
    assert len(trace) == 5
    assert len(estimator.poses) == 3
    np.testing.assert_array_equal(estimator.poses[0], log.records[0].detector_pose_xyz)

    missing_boundary = replace(
        log,
        records=(
            log.records[0],
            replace(
                log.records[1],
                metadata={"fixture_record": 1},
            ),
            *log.records[2:],
        ),
    )
    malformed = build_replay_estimator(
        missing_boundary, replay_config(), profile="pf_strict", seed=41
    )
    with pytest.raises(PFReplayError, match="before the next station"):
        replay_records(missing_boundary, malformed)

    conflicting_modes = replace(
        log,
        runtime_config={
            **log.runtime_config,
            "joint_observation_update": False,
            "delayed_resample_update": False,
        },
    )
    conflicting = build_replay_estimator(
        conflicting_modes, replay_config(), profile="pf_strict", seed=41
    )
    with pytest.raises(PFReplayError, match="conflicts with"):
        replay_records(conflicting_modes, conflicting)


@pytest.mark.parametrize("off_diagonal", (False, True))
def test_serialized_live_station_ingestion_matches_replay_covariance_path(
    tmp_path: Path,
    off_diagonal: bool,
) -> None:
    """Logged diagonal/off-diagonal covariance selects the same path as live."""
    loaded = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=4,
            runtime_overrides=_effective_runtime_overrides(),
            station_complete_markers=True,
        )
    )
    if off_diagonal:
        adjusted = []
        for record in loaded.records:
            covariance = np.asarray(record.isotope_count_covariance, dtype=float).copy()
            covariance[0, 1] = covariance[1, 0] = 0.25
            adjusted.append(replace(record, isotope_count_covariance=covariance))
        log = replace(loaded, records=tuple(adjusted))
    else:
        log = loaded

    replayed = build_replay_estimator(
        log, replay_config(), profile="pf_strict", seed=41
    )
    replay_records(log, replayed)

    live = build_replay_estimator(log, replay_config(), profile="pf_strict", seed=41)
    pending: list[tuple[object, ...]] = []
    active_station: int | None = None
    for record in log.records:
        pose = np.asarray(record.detector_pose_xyz, dtype=float)
        if active_station != int(record.station_id):
            if active_station is None:
                live.poses[-1] = pose.copy()
                live.kernel_cache = None
                pose_idx = len(live.poses) - 1
            else:
                live.add_measurement_pose(pose, reset_filters=False)
                pose_idx = len(live.poses) - 1
            active_station = int(record.station_id)
        dense = np.asarray(record.isotope_count_covariance, dtype=float)
        variances = {
            isotope: float(dense[index, index])
            for index, isotope in enumerate(TEST_ISOTOPES)
        }
        covariance = {
            row_isotope: {
                column_isotope: float(dense[row_index, column_index])
                for column_index, column_isotope in enumerate(TEST_ISOTOPES)
            }
            for row_index, row_isotope in enumerate(TEST_ISOTOPES)
        }
        pair: tuple[object, ...] = (
            dict(record.isotope_counts or {}),
            int(record.fe_orientation_index),
            int(record.pb_orientation_index),
            float(record.live_time_s),
            variances,
        )
        if off_diagonal:
            pair = (*pair, covariance)
        pending.append(pair)
        if bool(record.metadata.get("station_complete", False)):
            live.update_pair_sequence(tuple(pending), pose_idx=pose_idx)
            pending = []

    assert replayed.serialized_state() == live.serialized_state()


def test_same_effective_log_supports_only_strict_profile(
    tmp_path: Path,
) -> None:
    """A strict recorded run must reject the removed profiled estimator."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=2,
            runtime_overrides=_effective_runtime_overrides(),
            station_complete_markers=True,
        )
    )
    strict = build_replay_estimator(log, replay_config(), profile="pf_strict", seed=41)

    assert strict.resolved_config_hash == log.resolved_config_sha256
    assert strict.estimator_variant == "pf_strict"
    with pytest.raises(ValueError, match="only 'pf_strict' is available"):
        build_replay_estimator(
            log,
            replay_config(),
            profile="pf_profiled",
            seed=41,
        )


def test_tiny_live_run_finalized_log_replays_to_identical_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One production PF update round-trips through its finalized live log."""
    import realtime_demo

    def lightweight_ig(
        estimator: PurePFEstimator,
        rotations: object,
        **kwargs: object,
    ) -> np.ndarray:
        del kwargs
        estimator._ensure_kernel_cache()
        count = len(rotations)  # type: ignore[arg-type]
        return np.zeros((count, count), dtype=float)

    def lightweight_shield_grid(
        *args: object, **kwargs: object
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        del args
        scores = np.asarray(kwargs["ig_scores"], dtype=float)
        zeros = np.zeros_like(scores)
        return scores.copy(), {
            "eig": scores.copy(),
            "signature": zeros.copy(),
            "signature_utility": zeros.copy(),
            "low_count_penalty": zeros.copy(),
            "count_balance_penalty": zeros.copy(),
            "rotation_cost": zeros.copy(),
        }

    monkeypatch.setattr(realtime_demo, "_compute_ig_grid", lightweight_ig)
    monkeypatch.setattr(
        realtime_demo,
        "_compute_shield_selection_grid",
        lightweight_shield_grid,
    )
    log_path = tmp_path / "live-measurement-log"
    live_estimator = realtime_demo.run_live_pf(
        live=False,
        max_steps=1,
        max_poses=1,
        obstacle_layout_path=None,
        candidate_grid_spacing=(5.0, 10.0, 5.0),
        candidate_grid_margin=0.0,
        birth_enabled=False,
        num_particles=8,
        pf_config_overrides={
            "orientation_k": 1,
            "min_rotations_per_pose": 1,
            "max_sources": 1,
            "init_num_sources": [1, 1],
            "min_particles": 8,
            "max_particles": 8,
            "resample_threshold": 0.0,
            "use_tempering": False,
            "use_gpu": False,
        },
        measurement_time_s=1.0,
        pose_candidates=1,
        save_outputs=False,
        measurement_log_output=str(log_path),
        return_state=True,
    )
    assert isinstance(live_estimator, PurePFEstimator)

    replayed, trace = replay_measurement_log(
        log_path,
        {"use_gpu": False},
        profile="pf_strict",
        seed=0,
    )
    assert len(trace) == 1
    assert replayed.serialized_state() == live_estimator.serialized_state()


def test_full_strict_replay_exposes_no_removed_estimator_methods(
    tmp_path: Path,
) -> None:
    """A complete replay must finish without reintroducing deleted API stubs."""
    removed_methods = (
        "_refit_reported_strengths",
        "final_report_estimate",
        "fit_surface_map",
        "planning_surface_rescue_modes",
        "refresh_sparse_poisson_evidence",
        "report_model_order_diagnostics",
        "runtime_report_rescue_modes",
        "sparse_poisson_evidence_diagnostics",
    )
    for name in removed_methods:
        assert not hasattr(PurePFEstimator, name)

    estimator, trace = replay_measurement_log(
        make_measurement_log(tmp_path / "measurement-log", record_count=3),
        replay_config(),
        profile="pf_strict",
        seed=31,
    )
    assert len(trace) == 3
    for name in removed_methods:
        assert not hasattr(estimator, name)


def test_replay_rejects_forward_model_mismatch_and_existing_output(
    tmp_path: Path,
) -> None:
    """There is no fallback for a changed line table or overwrite target."""
    log_path = make_measurement_log(tmp_path / "measurement-log", record_count=1)
    log = load_measurement_log(log_path)
    mutated = json.loads(json.dumps(log.forward_model_manifest))
    mutated["model_identifiers"]["spectrum"]["id"] = "unknown-spectrum"
    with pytest.raises(PFReplayError, match="compatibility"):
        validate_local_forward_model(replace(log, forward_model_manifest=mutated))

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(FileExistsError):
        replay_measurement_log(
            log_path,
            replay_config(),
            profile="pf_strict",
            seed=7,
            output_dir=occupied,
        )
