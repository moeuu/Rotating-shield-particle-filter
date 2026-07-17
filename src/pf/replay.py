"""Sequentially replay a MeasurementLog through a pure particle filter."""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from measurement.observation_model import build_runtime_observation_model
from measurement.obstacles import ObstacleGrid
from measurement.model import EnvironmentConfig
from measurement.shielding import generate_octant_orientations
from measurement.source_surfaces import build_surface_candidate_sources
from pf.profiles import apply_profile_to_config, enforce_pure_runtime_settings
from pf.provenance import canonical_json_bytes, sha256_json
from pf.pure_estimator import PurePFEstimator, RotatingShieldPFConfig
from runtime.forward_model_manifest import resolve_file_backed_model_asset
from runtime.measurement_log import (
    MeasurementLog,
    MeasurementLogRecord,
    build_forward_model_manifest,
    load_measurement_log,
)
from sim.runtime import load_runtime_config


class PFReplayError(RuntimeError):
    """Report an incompatible log, configuration, or replay observation."""


def _sha256_bytes(payload: bytes) -> str:
    """Return a hexadecimal SHA-256 digest."""
    return hashlib.sha256(payload).hexdigest()


def validate_local_forward_model(log: MeasurementLog) -> None:
    """Fail closed unless the logged model identity matches the local PF model."""
    expected = build_forward_model_manifest(
        runtime_config=log.runtime_config,
        environment=log.environment,
        obstacle_layout_path=(
            None
            if log.run_manifest.get("obstacle_layout_path") is None
            else str(log.run_manifest["obstacle_layout_path"])
        ),
        isotopes=tuple(str(value) for value in log.run_manifest.get("isotopes", ())),
        repository_commit=str(log.run_manifest["repository_commit"]),
        resolved_config_sha256=str(log.run_manifest["resolved_config_sha256"]),
        run_root=log.path,
    )
    for field in (
        "source_rate_model",
        "source_rate_semantics",
        "units",
        "response_semantics",
        "model_identifiers",
    ):
        if log.forward_model_manifest.get(field) != expected[field]:
            raise PFReplayError(
                "Forward-model compatibility check failed for "
                f"{field}; replay refuses an unknown or mismatched model."
            )


def _resolved_physical_config(log: MeasurementLog) -> dict[str, Any]:
    """Bind replay file-backed physics to the asset validated for this log."""
    physical_config = dict(log.runtime_config)
    model_path = physical_config.get("pf_transport_response_model_path")
    if model_path is None:
        return physical_config
    if log.path is None:
        raise PFReplayError(
            "File-backed replay physics require a MeasurementLog source directory."
        )
    try:
        resolved_path = resolve_file_backed_model_asset(
            model_path,
            field_name="runtime_config.pf_transport_response_model_path",
            run_root=log.path,
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        raise PFReplayError(
            "Replay cannot resolve the validated PF transport-response asset."
        ) from exc
    physical_config["pf_transport_response_model_path"] = str(resolved_path)
    return physical_config


def _obstacle_grid_from_log(log: MeasurementLog) -> ObstacleGrid | None:
    """Build the exact logged obstacle grid without reading evaluation truth."""
    raw = log.environment.get("obstacle_grid")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise PFReplayError("environment.obstacle_grid must be an object or null.")
    return ObstacleGrid.from_dict(dict(raw))


def _environment_bounds(
    log: MeasurementLog,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return replay source-position bounds from the resolved environment."""
    try:
        upper = np.asarray(
            [
                float(log.environment["size_x"]),
                float(log.environment["size_y"]),
                float(log.environment["size_z"]),
            ],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PFReplayError(
            "MeasurementLog environment requires finite size_x/size_y/size_z."
        ) from exc
    if not np.all(np.isfinite(upper)) or np.any(upper <= 0.0):
        raise PFReplayError("Environment dimensions must be finite and positive.")
    return np.zeros(3, dtype=np.float64), upper


def _candidate_sources(
    config: Mapping[str, Any],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return deterministic replay birth candidates from config or room bounds."""
    explicit = config.get("replay_candidate_sources_xyz")
    if explicit is not None:
        candidates = np.asarray(explicit, dtype=np.float64)
        if candidates.ndim != 2 or candidates.shape[1] != 3:
            raise PFReplayError("replay_candidate_sources_xyz must have shape (N, 3).")
        if candidates.shape[0] == 0 or not np.all(np.isfinite(candidates)):
            raise PFReplayError("Replay candidates must be non-empty and finite.")
        return candidates
    spacing_raw = config.get("replay_candidate_spacing_m", 1.0)
    spacing = np.asarray(spacing_raw, dtype=np.float64)
    if spacing.ndim == 0:
        spacing = np.full(3, float(spacing), dtype=np.float64)
    if spacing.shape != (3,) or np.any(~np.isfinite(spacing)) or np.any(spacing <= 0.0):
        raise PFReplayError("replay_candidate_spacing_m must be positive scalar/XYZ.")
    axes: list[NDArray[np.float64]] = []
    for axis in range(3):
        values = np.arange(
            float(lower[axis]),
            float(upper[axis]) + 0.5 * float(spacing[axis]),
            float(spacing[axis]),
            dtype=np.float64,
        )
        values = values[values <= float(upper[axis]) + 1.0e-12]
        if values.size == 0:
            values = np.asarray([(lower[axis] + upper[axis]) / 2.0])
        axes.append(values)
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def _pf_config_values(
    config: Mapping[str, Any],
    *,
    profile: str,
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> dict[str, Any]:
    """Select only declared PF dataclass fields from a resolved runtime config."""
    allowed = {field.name for field in fields(RotatingShieldPFConfig)}
    values = {key: value for key, value in config.items() if key in allowed}
    values["estimator_profile"] = str(profile)
    values.setdefault("position_min", tuple(float(value) for value in lower))
    values.setdefault("position_max", tuple(float(value) for value in upper))
    return values


def _logged_candidate_sources(
    log: MeasurementLog,
    raw_grid: Mapping[str, Any],
) -> NDArray[np.float64]:
    """Load or deterministically regenerate the exact logged birth grid."""
    explicit = raw_grid.get("candidate_sources_xyz")
    if explicit is not None:
        return np.asarray(explicit, dtype=np.float64)
    if raw_grid.get("generator") != "realtime_source_candidate_grid.v1":
        raise PFReplayError(
            "Logged effective candidate grid lacks coordinates or a known generator."
        )
    spacing = np.asarray(raw_grid.get("spacing_xyz_m"), dtype=np.float64).reshape(-1)
    lower = np.asarray(raw_grid.get("position_min_xyz_m"), dtype=np.float64).reshape(-1)
    upper = np.asarray(raw_grid.get("position_max_xyz_m"), dtype=np.float64).reshape(-1)
    if (
        spacing.shape != (3,)
        or lower.shape != (3,)
        or upper.shape != (3,)
        or np.any(~np.isfinite(spacing))
        or np.any(spacing <= 0.0)
        or np.any(~np.isfinite(lower))
        or np.any(~np.isfinite(upper))
    ):
        raise PFReplayError("Logged candidate generator requires finite XYZ inputs.")
    try:
        env = EnvironmentConfig(
            size_x=float(log.environment["size_x"]),
            size_y=float(log.environment["size_y"]),
            size_z=float(log.environment["size_z"]),
            detector_position=tuple(
                float(value) for value in log.environment["detector_position"]
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PFReplayError(
            "Candidate-grid regeneration requires the logged room environment."
        ) from exc
    if bool(raw_grid.get("source_surface_prior", False)):
        return build_surface_candidate_sources(
            env,
            _obstacle_grid_from_log(log),
            tuple(float(value) for value in spacing),
            position_min=tuple(float(value) for value in lower),
            position_max=tuple(float(value) for value in upper),
            obstacle_height_m=float(raw_grid.get("obstacle_height_m", 2.0)),
        )
    margin = float(raw_grid.get("margin_m", 0.0))

    def _axis(start: float, stop: float, step: float) -> NDArray[np.float64]:
        if stop < start:
            return np.zeros(0, dtype=np.float64)
        count = int(np.floor((stop - start) / step)) + 1
        return start + step * np.arange(max(count, 0), dtype=np.float64)

    axes = [
        _axis(
            float(lower[index] + margin),
            float(upper[index] - margin),
            float(spacing[index]),
        )
        for index in range(3)
    ]
    if any(axis.size == 0 for axis in axes):
        raise PFReplayError("Logged candidate generator produced an empty grid.")
    return np.asarray(
        [[x, y, z] for x in axes[0] for y in axes[1] for z in axes[2]],
        dtype=np.float64,
    )


def _logged_effective_replay_config(
    log: MeasurementLog,
    external_config: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any] | None:
    """Resolve exact live-run PF inputs when the log provides them."""
    raw = log.runtime_config.get("effective_pf_replay")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise PFReplayError("effective_pf_replay must be an object.")
    raw_pf = raw.get("pf_config")
    raw_grid = raw.get("candidate_grid")
    raw_api = raw.get("api_settings")
    if (
        not isinstance(raw_pf, Mapping)
        or not isinstance(raw_grid, Mapping)
        or not isinstance(raw_api, Mapping)
    ):
        raise PFReplayError(
            "effective_pf_replay requires pf_config, candidate_grid, and "
            "api_settings objects."
        )
    for mode_name in ("joint_observation_update", "delayed_resample_update"):
        if not isinstance(raw_api.get(mode_name), bool):
            raise PFReplayError(
                f"effective_pf_replay.api_settings.{mode_name} must be a boolean."
            )
    candidates = _logged_candidate_sources(log, raw_grid)
    if candidates.ndim != 2 or candidates.shape[1] != 3 or candidates.shape[0] == 0:
        raise PFReplayError("Logged effective candidate grid must have shape (N, 3).")
    if int(raw_grid.get("point_count", -1)) != int(candidates.shape[0]):
        raise PFReplayError("Logged effective candidate-grid count is incompatible.")
    if str(raw_grid.get("xyz_sha256", "")) != sha256_json(candidates):
        raise PFReplayError("Logged effective candidate-grid SHA-256 is incompatible.")
    if isinstance(raw_api, Mapping) and "pf_random_seed" in raw_api:
        if int(raw_api["pf_random_seed"]) != int(seed):
            raise PFReplayError(
                "Replay seed differs from the logged effective PF random seed."
            )
    allowed_compute_overrides = {"use_gpu", "gpu_device", "gpu_dtype"}
    allowed_profile_overrides = {
        "estimator_profile",
        "conditional_strength_refit",
        "conditional_strength_profile_before_likelihood",
        "conditional_strength_refit_reweight",
        "refit_after_moves",
    }
    allowed_replay_overrides = allowed_compute_overrides | allowed_profile_overrides
    declared_pf_fields = {field.name for field in fields(RotatingShieldPFConfig)}
    for key, value in external_config.items():
        if (
            key in declared_pf_fields
            and key in raw_pf
            and key not in allowed_replay_overrides
            and sha256_json(value) != sha256_json(raw_pf[key])
        ):
            raise PFReplayError(
                f"External PF field {key!r} differs from the logged effective run."
            )
    merged = dict(external_config)
    merged.update({str(key): value for key, value in raw_pf.items()})
    for key in allowed_replay_overrides:
        if key in external_config:
            merged[key] = external_config[key]
    merged["replay_candidate_sources_xyz"] = candidates.tolist()
    return merged


def build_replay_estimator(
    log: MeasurementLog,
    config: Mapping[str, Any],
    *,
    profile: str,
    seed: int,
    config_hash: str | None = None,
    resolved_config_hash: str | None = None,
) -> PurePFEstimator:
    """Construct a locally validated pure estimator for one replay."""
    validate_local_forward_model(log)
    if str(log.runtime_config.get("spectrum_count_method", "")).strip().lower() != (
        "response_poisson"
    ):
        raise PFReplayError(
            "Pure PF replay requires spectrum_count_method='response_poisson'."
        )
    isotopes = tuple(str(value) for value in log.run_manifest.get("isotopes", ()))
    if not isotopes:
        raise PFReplayError("MeasurementLog must declare at least one isotope.")
    lower, upper = _environment_bounds(log)
    logged_config = _logged_effective_replay_config(
        log,
        config,
        seed=int(seed),
    )
    pure_config = enforce_pure_runtime_settings(
        dict(config) if logged_config is None else logged_config,
        profile=profile,
    )
    pf_config = RotatingShieldPFConfig(
        **_pf_config_values(
            pure_config,
            profile=profile,
            lower=lower,
            upper=upper,
        )
    )
    apply_profile_to_config(pf_config)
    candidates = _candidate_sources(pure_config, lower, upper)
    physical_config = _resolved_physical_config(log)
    observation_model = build_runtime_observation_model(
        physical_config,
        isotopes=isotopes,
    )
    obstacle_grid = _obstacle_grid_from_log(log)
    raw_obstacle_setting = physical_config.get("pf_obstacle_attenuation", True)
    obstacle_enabled = (
        raw_obstacle_setting
        if isinstance(raw_obstacle_setting, bool)
        else str(raw_obstacle_setting).strip().lower()
        not in {"0", "false", "no", "off", "disable", "disabled"}
    )
    if resolved_config_hash is None:
        if logged_config is None:
            replay_config_sha256 = sha256_json(pure_config)
        else:
            raw_effective = log.runtime_config["effective_pf_replay"]
            assert isinstance(raw_effective, Mapping)
            raw_pf = raw_effective["pf_config"]
            assert isinstance(raw_pf, Mapping)
            actual_pf = {
                field.name: getattr(pf_config, field.name)
                for field in fields(RotatingShieldPFConfig)
            }
            exact_live_pf = sha256_json(actual_pf) == sha256_json(raw_pf)
            replay_config_sha256 = (
                str(log.resolved_config_sha256)
                if exact_live_pf
                else sha256_json(
                    {
                        "base_live_config_sha256": log.resolved_config_sha256,
                        "pf_config": actual_pf,
                        "candidate_sources_xyz": candidates,
                    }
                )
            )
    else:
        replay_config_sha256 = str(resolved_config_hash)
    input_config_sha256 = (
        sha256_json(dict(config)) if config_hash is None else str(config_hash)
    )
    np.random.seed(int(seed))
    estimator = PurePFEstimator(
        isotopes=isotopes,
        candidate_sources=candidates,
        shield_normals=generate_octant_orientations(),
        mu_by_isotope=observation_model.mu_by_isotope,
        pf_config=pf_config,
        obstacle_grid=obstacle_grid if obstacle_enabled else None,
        obstacle_height_m=observation_model.obstacle_height_m,
        obstacle_mu_by_isotope=observation_model.obstacle_mu_by_isotope,
        obstacle_buildup_coeff=(
            observation_model.obstacle_buildup_coeff if obstacle_enabled else 0.0
        ),
        detector_radius_m=observation_model.detector_geometry.count_radius_m,
        detector_aperture_radius_m=(
            observation_model.detector_geometry.aperture_radius_m
        ),
        detector_aperture_samples=(
            observation_model.detector_geometry.aperture_samples
        ),
        detector_aperture_sampling=(
            observation_model.detector_geometry.aperture_sampling
        ),
        source_extent_radius_m=observation_model.source_extent_radius_m,
        source_extent_samples=observation_model.source_extent_samples,
        line_mu_by_isotope=observation_model.line_mu_by_isotope,
        transport_response_model=observation_model.transport_response_model,
        measurement_log_schema_version=log.schema_version,
        config_hash=input_config_sha256,
        resolved_config_hash=replay_config_sha256,
        measurement_log_sha256=log.log_sha256,
        random_seed=int(seed),
    )
    try:
        initial_pose = np.asarray(log.environment["detector_position"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise PFReplayError(
            "MeasurementLog environment requires an initial detector_position."
        ) from exc
    if initial_pose.shape != (3,) or not np.all(np.isfinite(initial_pose)):
        raise PFReplayError(
            "environment.detector_position must be a finite XYZ position."
        )
    estimator.add_measurement_pose(initial_pose, reset_filters=False)
    return estimator


def _covariance_mapping(
    record: MeasurementLogRecord,
    isotopes: Sequence[str],
) -> dict[str, dict[str, float]] | None:
    """Translate an optional dense log covariance into the PF update contract."""
    if record.isotope_count_covariance is None:
        return None
    covariance = np.asarray(record.isotope_count_covariance, dtype=float)
    expected = (len(isotopes), len(isotopes))
    if covariance.shape != expected:
        raise PFReplayError(f"Isotope covariance must have shape {expected}.")
    return {
        row_isotope: {
            column_isotope: float(covariance[row_index, column_index])
            for column_index, column_isotope in enumerate(isotopes)
        }
        for row_index, row_isotope in enumerate(isotopes)
    }


def _runtime_station_update_modes(log: MeasurementLog) -> tuple[bool, bool]:
    """Return explicitly logged modes, preserving legacy per-row compatibility."""
    effective = log.runtime_config.get("effective_pf_replay")
    if effective is not None:
        if not isinstance(effective, Mapping) or not isinstance(
            effective.get("api_settings"), Mapping
        ):
            raise PFReplayError("effective_pf_replay.api_settings must be an object.")
        api_settings = effective["api_settings"]
        assert isinstance(api_settings, Mapping)
        values: dict[str, bool] = {}
        for name in ("joint_observation_update", "delayed_resample_update"):
            raw = api_settings.get(name)
            if not isinstance(raw, bool):
                raise PFReplayError(
                    f"effective_pf_replay.api_settings.{name} must be a boolean."
                )
            top_level = log.runtime_config.get(name)
            if top_level is not None:
                if not isinstance(top_level, bool):
                    raise PFReplayError(f"runtime_config.{name} must be a boolean.")
                if top_level is not raw:
                    raise PFReplayError(
                        f"runtime_config.{name} conflicts with "
                        f"effective_pf_replay.api_settings.{name}."
                    )
            values[name] = raw
        joint = values["joint_observation_update"]
        return joint, values["delayed_resample_update"] and not joint

    resolved: dict[str, bool] = {}
    for name in ("joint_observation_update", "delayed_resample_update"):
        raw = log.runtime_config.get(name)
        if raw is None:
            resolved[name] = False
            continue
        if not isinstance(raw, bool):
            raise PFReplayError(f"runtime_config.{name} must be a boolean.")
        resolved[name] = raw
    joint = resolved["joint_observation_update"]
    delayed = resolved["delayed_resample_update"] and not joint
    return joint, delayed


def _station_complete(record: MeasurementLogRecord) -> bool:
    """Return the writer-owned causal station-boundary marker."""
    raw = record.metadata.get("station_complete", False)
    if not isinstance(raw, bool):
        raise PFReplayError("record.metadata.station_complete must be a boolean.")
    return raw


def _pair_record(
    record: MeasurementLogRecord,
    isotopes: Sequence[str],
) -> tuple[object, ...]:
    """Translate one count-domain log row into the PF pair-update contract."""
    if record.isotope_counts is None or any(
        isotope not in record.isotope_counts for isotope in isotopes
    ):
        raise PFReplayError(
            "Pure PF replay requires response_poisson isotope counts for every "
            "declared isotope."
        )
    covariance = _covariance_mapping(record, isotopes)
    variances = {
        isotope: float(
            covariance[isotope][isotope]
            if covariance is not None
            else max(float(record.isotope_counts[isotope]), 1.0)
        )
        for isotope in isotopes
    }
    payload: tuple[object, ...] = (
        {isotope: float(record.isotope_counts[isotope]) for isotope in isotopes},
        int(record.fe_orientation_index),
        int(record.pb_orientation_index),
        float(record.live_time_s),
        variances,
    )
    if covariance is None:
        return payload
    dense_covariance = np.asarray(record.isotope_count_covariance, dtype=float)
    off_diagonal = dense_covariance - np.diag(np.diag(dense_covariance))
    return payload if not np.any(np.abs(off_diagonal) > 0.0) else (*payload, covariance)


def _trace_row(
    estimator: PurePFEstimator,
    record: MeasurementLogRecord,
    *,
    record_index: int,
) -> dict[str, Any]:
    """Serialize one causal count-PF state after the record's update boundary."""
    serialized = estimator.serialized_state()
    return {
        "schema_version": 1,
        "estimator_family": "particle_filter",
        "estimator_variant": estimator.estimator_variant,
        "record_index": int(record_index),
        "step_id": int(record.step_id),
        "action_id": int(record.action_id),
        "station_id": int(record.station_id),
        "state_sha256": _sha256_bytes(serialized),
        "posterior": estimator.posterior_snapshot().to_dict(),
    }


def replay_records(
    log: MeasurementLog,
    estimator: PurePFEstimator,
    *,
    stop_after: int | None = None,
    pre_record_callback: Callable[
        [PurePFEstimator, MeasurementLogRecord, int, int], None
    ]
    | None = None,
    station_complete_callback: Callable[
        [PurePFEstimator, MeasurementLogRecord, int], None
    ]
    | None = None,
) -> tuple[dict[str, Any], ...]:
    """Replay a causal prefix using its explicitly logged station boundaries.

    Optional callbacks are reserved for estimator-neutral wrappers.  The
    standard replay command supplies neither callback, so its state, random
    stream, trace, and output bytes retain their existing semantics.
    """
    isotopes = tuple(str(value) for value in log.run_manifest.get("isotopes", ()))
    limit = len(log.records) if stop_after is None else max(0, int(stop_after))
    station_pose: dict[int, NDArray[np.float64]] = {}
    station_pose_index: dict[int, int] = {}
    trace: list[dict[str, Any]] = []
    joint_update, delayed_update = _runtime_station_update_modes(log)
    pending_station_id: int | None = None
    pending_pose_idx: int | None = None
    pending_pairs: list[tuple[object, ...]] = []
    completed_station_ids: set[int] = set()

    for record_index, record in enumerate(log.records[:limit]):
        pose = np.asarray(record.detector_pose_xyz, dtype=float)
        station_id = int(record.station_id)
        if station_id not in station_pose:
            if not station_pose and estimator.poses:
                estimator.poses[-1] = pose.copy()
                estimator.kernel_cache = None
                estimator._invalidate_report_cache()
                pose_index = len(estimator.poses) - 1
            else:
                estimator.add_measurement_pose(pose, reset_filters=False)
                pose_index = len(estimator.poses) - 1
            station_pose[station_id] = pose.copy()
            station_pose_index[station_id] = pose_index
        elif not np.array_equal(station_pose[station_id], pose):
            raise PFReplayError(
                f"station_id {station_id} contains multiple detector poses."
            )

        pair = _pair_record(record, isotopes)
        station_complete = _station_complete(record)
        pose_idx = station_pose_index[station_id]
        if pre_record_callback is not None:
            pre_record_callback(estimator, record, record_index, pose_idx)

        if not joint_update and not delayed_update:
            normalized = PurePFEstimator._normalize_pair_sequence_record(pair)
            z_k, fe, pb, live, variances, covariance, _spectrum = normalized
            kwargs: dict[str, object] = {}
            if covariance is not None:
                kwargs["z_covariance_k"] = covariance
            estimator.update_pair(
                z_k=z_k,
                pose_idx=pose_idx,
                fe_index=fe,
                pb_index=pb,
                live_time_s=live,
                z_variance_k=variances,
                **kwargs,
            )
            if station_complete and station_complete_callback is not None:
                station_complete_callback(estimator, record, record_index)
            trace.append(_trace_row(estimator, record, record_index=record_index))
            continue

        if station_id in completed_station_ids:
            raise PFReplayError(
                f"station_id {station_id} has observations after station_complete."
            )
        if pending_station_id is not None and station_id != pending_station_id:
            raise PFReplayError(
                f"station_id {pending_station_id} lacks a causal station_complete "
                "marker before the next station."
            )
        if pending_station_id is None:
            pending_station_id = station_id
            pending_pose_idx = pose_idx
            if delayed_update:
                estimator.begin_deferred_pose_update()
        pending_pairs.append(pair)

        if joint_update:
            if station_complete:
                assert pending_pose_idx is not None
                estimator.update_pair_sequence(
                    tuple(pending_pairs),
                    pose_idx=pending_pose_idx,
                )
                if station_complete_callback is not None:
                    station_complete_callback(estimator, record, record_index)
            trace.append(_trace_row(estimator, record, record_index=record_index))
        else:
            normalized = PurePFEstimator._normalize_pair_sequence_record(pair)
            z_k, fe, pb, live, variances, covariance, _spectrum = normalized
            kwargs = {}
            if covariance is not None:
                kwargs["z_covariance_k"] = covariance
            estimator.update_pair(
                z_k=z_k,
                pose_idx=pose_idx,
                fe_index=fe,
                pb_index=pb,
                live_time_s=live,
                z_variance_k=variances,
                **kwargs,
            )
            if station_complete:
                estimator.finalize_deferred_pose_update()
                if station_complete_callback is not None:
                    station_complete_callback(estimator, record, record_index)
            trace.append(_trace_row(estimator, record, record_index=record_index))

        if station_complete:
            completed_station_ids.add(station_id)
            pending_station_id = None
            pending_pose_idx = None
            pending_pairs = []

    # An unfinished joint/deferred station is a valid causal snapshot.  EOF is
    # deliberately not interpreted as a boundary: the same first N records
    # produce the same pending state whether or not a future suffix exists.
    return tuple(trace)


def _write_replay_outputs(
    output_dir: str | Path,
    *,
    estimator: PurePFEstimator,
    trace: Sequence[Mapping[str, Any]],
    log: MeasurementLog,
) -> Path:
    """Atomically publish the required pure-PF replay result contract."""
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"Refusing to replace replay output {target}.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Temporary replay output exists: {temporary}.")
    temporary.mkdir()
    try:
        posterior = estimator.posterior_snapshot().to_dict()
        (temporary / "pf_posterior.json").write_bytes(canonical_json_bytes(posterior))
        trace_bytes = b"".join(
            (
                json.dumps(
                    row,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            for row in trace
        )
        (temporary / "pf_trace.jsonl").write_bytes(trace_bytes)
        final_state = estimator.serialized_state()
        diagnostics = {
            "schema_version": 1,
            "estimator_family": "particle_filter",
            "estimator_variant": estimator.estimator_variant,
            "measurement_log_schema_version": log.schema_version,
            "measurement_log_sha256": log.log_sha256,
            "measurement_log_resolved_config_sha256": log.resolved_config_sha256,
            "config_sha256": estimator.config_hash,
            "resolved_config_sha256": estimator.resolved_config_hash,
            "record_count": len(trace),
            "records_processed": len(trace),
            "final_state_sha256": _sha256_bytes(final_state),
            "forward_model_compatibility": "local_manifest_exact_match",
            "forbidden_batch_methods_invoked": list(estimator.batch_methods_invoked),
        }
        (temporary / "pf_diagnostics.json").write_bytes(
            canonical_json_bytes(diagnostics)
        )
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def replay_measurement_log(
    measurement_log: str | Path,
    config: str | Path | Mapping[str, Any],
    *,
    profile: str = "pf_strict",
    seed: int = 0,
    stop_after: int | None = None,
    output_dir: str | Path | None = None,
) -> tuple[PurePFEstimator, tuple[dict[str, Any], ...]]:
    """Validate, replay, and optionally persist one pure-PF result."""
    log = load_measurement_log(measurement_log)
    if isinstance(config, (str, Path)):
        config_path = Path(config)
        config_hash = _sha256_bytes(config_path.read_bytes())
        resolved = load_runtime_config(config_path)
    else:
        raw_config = dict(config)
        config_hash = sha256_json(raw_config)
        resolved = raw_config
    resolved = enforce_pure_runtime_settings(resolved, profile=profile)
    estimator = build_replay_estimator(
        log,
        resolved,
        profile=profile,
        seed=int(seed),
        config_hash=config_hash,
        # Let the builder hash the complete resolved replay configuration,
        # including candidate-grid inputs that are intentionally outside the
        # PFConfig dataclass.
        resolved_config_hash=None,
    )
    trace = replay_records(log, estimator, stop_after=stop_after)
    if output_dir is not None:
        _write_replay_outputs(
            output_dir,
            estimator=estimator,
            trace=trace,
            log=log,
        )
    return estimator, trace


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the public sequential replay command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-log", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--profile",
        choices=("pf_strict", "pf_profiled"),
        default="pf_strict",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(None if argv is None else list(argv))
    replay_measurement_log(
        args.measurement_log,
        args.config,
        profile=args.profile,
        seed=args.seed,
        stop_after=args.stop_after,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
