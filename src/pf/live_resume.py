"""Station-boundary resume helpers for PF closed-loop acquisition."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pf.provenance import json_safe, sha256_json
from pf.pure_estimator import PurePFEstimator
from pf.replay import build_replay_estimator, detect_replay_isotopes
from runtime.measurement_log import MeasurementLog, MeasurementLogRecord

_RESUME_ORCHESTRATION_CODE_PATHS: frozenset[str] = frozenset()
_RESUME_RUNTIME_STATUS_PATHS = (
    "main.py",
    "src",
    "pyproject.toml",
    "uv.lock",
    "native",
    "scripts/run_geant4_bridge.py",
    "scripts/build_geant4_sidecar.py",
)
_RESUME_RUNTIME_EXACT_PATHS = frozenset(
    {
        "main.py",
        "pyproject.toml",
        "uv.lock",
        "scripts/run_geant4_bridge.py",
        "scripts/build_geant4_sidecar.py",
    }
)
_LIVE_CONTROLLER_CHECKPOINT_KEY = "live_controller_checkpoint"


def _strict_json_integer(
    value: object,
    *,
    name: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Return an exact JSON integer inside optional inclusive bounds."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a JSON integer.")
    resolved = int(value)
    if minimum is not None and resolved < int(minimum):
        raise ValueError(f"{name} must be at least {int(minimum)}.")
    if maximum is not None and resolved > int(maximum):
        raise ValueError(f"{name} must be at most {int(maximum)}.")
    return resolved


def _build_resume_replay_estimator(
    prefix_log: MeasurementLog,
    *,
    pf_config: Mapping[str, Any],
    profile: str,
    seed: int,
    config_hash: str,
) -> PurePFEstimator:
    """Build a resume estimator from explicit PF settings and a raw prefix."""
    detected_only = pf_config.get(
        "pf_detected_isotopes_only",
        pf_config.get("detected_isotopes_only", False),
    )
    if not isinstance(detected_only, bool):
        raise ValueError("pf_detected_isotopes_only must be a boolean.")
    inference_isotopes = None
    gate_diagnostics = None
    if detected_only:
        inference_isotopes, gate_diagnostics = detect_replay_isotopes(
            prefix_log,
            pf_config,
            profile=profile,
            seed=seed,
            config_hash=config_hash,
        )
    estimator = build_replay_estimator(
        prefix_log,
        pf_config,
        profile=profile,
        seed=seed,
        config_hash=config_hash,
        inference_isotopes=inference_isotopes,
    )
    estimator.detected_isotope_gate_diagnostics = gate_diagnostics
    return estimator


def _git_command_text(repository_root: Path, *args: str) -> str:
    """Run one read-only Git command and return stripped stdout."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"Cannot verify resume repository compatibility: git {' '.join(args)}"
        ) from exc
    return completed.stdout.strip()


def _full_git_commit(value: object) -> bool:
    """Return whether a value is one full lowercase hexadecimal Git commit."""
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _git_blob_at_commit(
    repository_root: Path,
    commit: str,
    relative_path: str,
) -> str | None:
    """Return a Git blob identifier, or None when the path did not exist."""
    try:
        return _git_command_text(
            repository_root,
            "rev-parse",
            "--verify",
            f"{commit}:{relative_path}",
        )
    except RuntimeError:
        return None


def _is_resume_runtime_path(path: str) -> bool:
    """Return whether a repository path can affect live runtime semantics."""
    return (
        path in _RESUME_RUNTIME_EXACT_PATHS
        or path.startswith("src/")
        or path.startswith("native/")
    )


def _build_resume_compatibility_provenance(
    *,
    repository_root: Path,
    prefix_commit: str,
    execution_commit: str,
    additional_compatible_code_paths: Sequence[str] | None,
    compatibility_basis: str | None,
) -> dict[str, Any]:
    """Verify clean tracked runtime code and describe the allowed commit delta."""
    root = repository_root.resolve()
    if not _full_git_commit(prefix_commit) or not _full_git_commit(execution_commit):
        raise RuntimeError("Resume requires full prefix and execution Git commits.")
    dirty_runtime = _git_command_text(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *_RESUME_RUNTIME_STATUS_PATHS,
    )
    if dirty_runtime:
        raise RuntimeError(
            "Resume refuses dirty or untracked live-runtime code; "
            "commit the verified implementation before continuing."
        )
    changed_text = _git_command_text(
        root,
        "diff",
        "--name-only",
        "--no-renames",
        prefix_commit,
        execution_commit,
    )
    changed_paths = tuple(
        sorted(path for path in changed_text.splitlines() if path.strip())
    )
    extra_allowed = {
        Path(str(path)).as_posix() for path in (additional_compatible_code_paths or ())
    }
    if any(
        path.startswith("/") or path == ".." or path.startswith("../") or "/../" in path
        for path in extra_allowed
    ):
        raise RuntimeError("Resume compatible code paths must be relative.")
    allowed_runtime_paths = set(_RESUME_ORCHESTRATION_CODE_PATHS) | extra_allowed
    changed_runtime_paths = {
        path for path in changed_paths if _is_resume_runtime_path(path)
    }
    incompatible = sorted(changed_runtime_paths - allowed_runtime_paths)
    if incompatible:
        raise RuntimeError(
            "Resume execution changes unapproved runtime code: "
            f"{incompatible}. Prove state equivalence and pass each path explicitly."
        )
    used_extra_paths = sorted(changed_runtime_paths & extra_allowed)
    basis = "" if compatibility_basis is None else str(compatibility_basis).strip()
    if changed_runtime_paths and not basis:
        raise RuntimeError(
            "An explicit compatibility basis is required for every admitted "
            "runtime change."
        )
    path_blobs = {
        path: {
            "prefix_git_blob": _git_blob_at_commit(root, prefix_commit, path),
            "execution_git_blob": _git_blob_at_commit(root, execution_commit, path),
        }
        for path in changed_paths
    }
    return {
        "schema_version": 1,
        "prefix_repository_commit": str(prefix_commit),
        "resume_execution_commit": str(execution_commit),
        "changed_paths": path_blobs,
        "allowed_runtime_paths": sorted(allowed_runtime_paths),
        "explicitly_compatible_runtime_paths": used_extra_paths,
        "compatibility_basis": (basis if basis else "no_live_runtime_path_delta"),
    }


@dataclass(frozen=True)
class _LiveResumeControllerState:
    """Store live-loop state reconstructed from a complete logged station."""

    step_counter: int
    pose_counter: int
    current_pose: NDArray[np.float64]
    current_pose_idx: int
    current_shield_pair_id: int
    visited_poses: tuple[NDArray[np.float64], ...]
    last_station_pair_ids: tuple[int, ...]
    elapsed_s: float
    total_motion_distance_m: float
    total_motion_time_s: float
    total_rotation_time_s: float
    measurement_live_times_s: tuple[float, ...]
    last_spectrum: NDArray[np.float64]
    last_observation_summary: dict[str, float]
    representative_spectrum: NDArray[np.float64]
    representative_step_index: int


def _online_compute_timing_provenance(
    resume_prefix_measurement_count: int,
) -> dict[str, object]:
    """Describe which live measurements are covered by online compute timings."""
    prefix_count = int(resume_prefix_measurement_count)
    if prefix_count < 0:
        raise ValueError("Resume prefix measurement count must be non-negative.")
    resumed = prefix_count > 0
    return {
        "online_compute_timing_scope": (
            "post_resume_suffix_only" if resumed else "full_live_run"
        ),
        "online_compute_timing_prefix_measurements_excluded": (
            prefix_count if resumed else 0
        ),
        "online_compute_timing_includes_resume_pf_replay": False,
    }


@dataclass(frozen=True)
class _LiveControllerCheckpoint:
    """Store controller-only state restored at a durable station boundary."""

    max_poses: int | None


def _planning_candidate_checkpoint_parameters(
    *,
    pose_candidates: int,
    pose_min_dist: float,
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]],
    detector_heights_m: Sequence[float] | None,
) -> dict[str, Any]:
    """Return the exact candidate-generation parameters guarded by a checkpoint."""
    bounds_lo = np.asarray(bounds_xyz[0], dtype=float).reshape(3)
    bounds_hi = np.asarray(bounds_xyz[1], dtype=float).reshape(3)
    return {
        "pose_candidates": int(pose_candidates),
        "pose_min_dist_m": float(pose_min_dist),
        "bounds_lo_xyz_m": [float(value) for value in bounds_lo],
        "bounds_hi_xyz_m": [float(value) for value in bounds_hi],
        "detector_heights_m": (
            None
            if detector_heights_m is None
            else [float(value) for value in detector_heights_m]
        ),
        "candidate_pool_contract": (
            "global_reachable_3d_sobol_with_physical_separation_v1"
        ),
    }


def _build_live_controller_checkpoint(
    *,
    planning_candidate_rng: np.random.Generator,
    dss_eig_rng: np.random.Generator,
    planning_candidate_parameters: Mapping[str, Any],
    max_poses: int | None,
) -> dict[str, Any]:
    """Build one truth-free controller checkpoint before post-station planning."""
    payload = {
        "schema_version": 4,
        "planning_candidate_rng_state": json_safe(
            planning_candidate_rng.bit_generator.state
        ),
        "dss_eig_rng_state": json_safe(dss_eig_rng.bit_generator.state),
        "planning_candidate_parameters": json_safe(dict(planning_candidate_parameters)),
        "mission_state": {
            "max_poses": None if max_poses is None else int(max_poses),
        },
    }
    try:
        json.dumps(payload, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Live controller checkpoint contains a non-finite or non-JSON value."
        ) from exc
    return payload


def _restore_live_controller_checkpoint(
    *,
    record: MeasurementLogRecord,
    planning_candidate_rng: np.random.Generator,
    dss_eig_rng: np.random.Generator,
    expected_planning_candidate_parameters: Mapping[str, Any],
) -> _LiveControllerCheckpoint | None:
    """Restore and validate a durable station-boundary controller checkpoint."""
    raw = record.metadata.get(_LIVE_CONTROLLER_CHECKPOINT_KEY)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise RuntimeError("Live controller checkpoint must be a JSON object.")
    checkpoint = dict(raw)
    expected_keys = {
        "schema_version",
        "planning_candidate_rng_state",
        "dss_eig_rng_state",
        "planning_candidate_parameters",
        "mission_state",
    }
    if set(checkpoint) != expected_keys or checkpoint["schema_version"] != 4:
        raise RuntimeError("Unsupported or malformed live controller checkpoint.")
    actual_parameters = checkpoint["planning_candidate_parameters"]
    if not isinstance(actual_parameters, Mapping) or sha256_json(
        dict(actual_parameters)
    ) != sha256_json(dict(expected_planning_candidate_parameters)):
        raise RuntimeError(
            "Live controller checkpoint candidate parameters differ from the "
            "current runtime."
        )
    rng_state = checkpoint["planning_candidate_rng_state"]
    if not isinstance(rng_state, Mapping):
        raise RuntimeError("Checkpoint planning RNG state must be an object.")
    try:
        planning_candidate_rng.bit_generator.state = dict(rng_state)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Checkpoint planning RNG state is invalid.") from exc
    if sha256_json(
        json_safe(planning_candidate_rng.bit_generator.state)
    ) != sha256_json(dict(rng_state)):
        raise RuntimeError("Checkpoint planning RNG state did not restore exactly.")
    dss_rng_state = checkpoint["dss_eig_rng_state"]
    if not isinstance(dss_rng_state, Mapping):
        raise RuntimeError("Checkpoint DSS/EIG RNG state must be an object.")
    try:
        dss_eig_rng.bit_generator.state = dict(dss_rng_state)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Checkpoint DSS/EIG RNG state is invalid.") from exc
    if sha256_json(json_safe(dss_eig_rng.bit_generator.state)) != sha256_json(
        dict(dss_rng_state)
    ):
        raise RuntimeError("Checkpoint DSS/EIG RNG state did not restore exactly.")

    mission_raw = checkpoint["mission_state"]
    if not isinstance(mission_raw, Mapping):
        raise RuntimeError("Checkpoint mission state has invalid structure.")
    mission = dict(mission_raw)
    if set(mission) != {"max_poses"}:
        raise RuntimeError("Checkpoint mission state has invalid fields.")
    try:
        max_poses_raw = mission["max_poses"]
        max_poses = (
            None
            if max_poses_raw is None
            else _strict_json_integer(
                max_poses_raw,
                name="checkpoint.mission_state.max_poses",
                minimum=1,
            )
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Checkpoint controller values are invalid.") from exc
    return _LiveControllerCheckpoint(max_poses=max_poses)


def _records_by_station(
    records: Sequence[MeasurementLogRecord],
) -> tuple[tuple[MeasurementLogRecord, ...], ...]:
    """Group a validated causal record prefix by contiguous station id."""
    if not records:
        raise ValueError("Resume requires at least one MeasurementLog record.")
    grouped: list[list[MeasurementLogRecord]] = []
    for record in records:
        station_id = int(record.station_id)
        if station_id == len(grouped):
            grouped.append([record])
        elif station_id == len(grouped) - 1:
            grouped[-1].append(record)
        else:
            raise ValueError(
                "Resume records require contiguous zero-based station identifiers."
            )
    return tuple(tuple(station) for station in grouped)


def _reconstruct_resume_controller_state(
    *,
    records: Sequence[MeasurementLogRecord],
    estimator: PurePFEstimator,
    isotopes: Sequence[str],
    nominal_motion_speed_m_s: float,
    expected_program_length: int,
) -> _LiveResumeControllerState:
    """Reconstruct counters, trajectory, timing, and display inputs from a prefix."""
    stations = _records_by_station(records)
    if len(estimator.poses) != len(stations):
        raise RuntimeError(
            "Pure replay pose count does not match the staged station count."
        )
    final_station = stations[-1]
    if len(final_station) != int(expected_program_length):
        raise RuntimeError(
            "Resume currently requires the completed station to contain the full "
            f"{int(expected_program_length)}-posture program."
        )
    final_pose = np.asarray(final_station[0].detector_pose_xyz, dtype=float)
    current_pose_idx = len(estimator.poses) - 1
    if not np.array_equal(
        np.asarray(estimator.poses[current_pose_idx], dtype=float),
        final_pose,
    ):
        raise RuntimeError(
            "Pure replay final pose does not match the staged station boundary."
        )
    station_poses = tuple(
        np.asarray(station[0].detector_pose_xyz, dtype=float).copy()
        for station in stations
    )
    pair_ids = tuple(
        int(record.fe_orientation_index) * 8 + int(record.pb_orientation_index)
        for record in final_station
    )
    representative = max(
        records,
        key=lambda record: float(
            np.sum(np.asarray(record.spectrum_counts, dtype=float))
        ),
    )
    last = records[-1]
    del isotopes
    last_observation_summary = {
        "raw_spectrum_total": float(
            np.sum(np.asarray(last.spectrum_counts, dtype=np.float64))
        )
    }
    motion_time = float(sum(float(record.travel_time_s) for record in records))
    rotation_time = float(
        sum(float(record.shield_actuation_time_s) for record in records)
    )
    live_times = tuple(float(record.live_time_s) for record in records)
    elapsed = motion_time + rotation_time + float(sum(live_times))
    return _LiveResumeControllerState(
        step_counter=len(records),
        pose_counter=len(stations) - 1,
        current_pose=final_pose.copy(),
        current_pose_idx=current_pose_idx,
        current_shield_pair_id=int(pair_ids[-1]),
        visited_poses=tuple(pose.copy() for pose in station_poses[:-1]),
        last_station_pair_ids=pair_ids,
        elapsed_s=elapsed,
        total_motion_distance_m=motion_time * max(float(nominal_motion_speed_m_s), 0.0),
        total_motion_time_s=motion_time,
        total_rotation_time_s=rotation_time,
        measurement_live_times_s=live_times,
        last_spectrum=np.asarray(last.spectrum_counts, dtype=float).copy(),
        last_observation_summary=last_observation_summary,
        representative_spectrum=np.asarray(
            representative.spectrum_counts,
            dtype=float,
        ).copy(),
        representative_step_index=int(representative.step_id),
    )


__all__ = [
    "_build_resume_compatibility_provenance",
    "_build_resume_replay_estimator",
    "_build_live_controller_checkpoint",
    "_online_compute_timing_provenance",
    "_planning_candidate_checkpoint_parameters",
    "_reconstruct_resume_controller_state",
    "_restore_live_controller_checkpoint",
]
