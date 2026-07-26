"""Real-time demo for the rotating-shield particle filter with visualization."""
# ruff: noqa: E402

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

import matplotlib

from measurement.detector_geometry import (
    DEFAULT_CRYSTAL_RADIUS_M,
    detector_active_radius_m,
)
from measurement.observation_model import build_runtime_observation_model

def _has_display() -> bool:
    """Return True when a GUI display is likely available."""
    if sys.platform.startswith("linux"):
        return bool(
            os.environ.get("DISPLAY")
            or os.environ.get("WAYLAND_DISPLAY")
            or os.environ.get("MIR_SOCKET")
        )
    return True


def _argv_requests_cui(argv: list[str] | None = None) -> bool:
    """Return True when command-line arguments request non-interactive CUI mode."""
    args = sys.argv[1:] if argv is None else argv
    if "--matplotlib-live" in args:
        return False

    def _is_run_mode_value(value: str) -> bool:
        """Return True when a CLI value names a non-Matplotlib run mode."""
        mode = value.strip().lower()
        return mode in {"gui", "cui"} or mode.endswith(("-gui", "-cui"))

    for index, arg in enumerate(args):
        if arg in {
            "--headless",
            "--no-live",
            "--gui",
            "--cui",
            "--python-gui",
            "--geant4-isaacsim-gui",
            "--python-cui",
            "--geant4-cui",
            "--full-simulation",
            "--standard-geant4-full",
        }:
            return True
        if arg in {"--mode", "--ui-mode"}:
            if index + 1 >= len(args):
                continue
            if _is_run_mode_value(args[index + 1]):
                return True
        if arg.startswith("--mode=") and _is_run_mode_value(arg.split("=", 1)[1]):
            return True
        if arg.startswith("--ui-mode=") and _is_run_mode_value(arg.split("=", 1)[1]):
            return True
    return False


def _resolve_station_update_modes(
    runtime_config: Mapping[str, Any],
) -> tuple[bool, bool]:
    """Return joint-observation and delayed-resample PF update switches."""
    joint_observation_update = bool(
        runtime_config.get("joint_observation_update", False)
    )
    delayed_default = not joint_observation_update
    delayed_resample_update = bool(
        runtime_config.get("delayed_resample_update", delayed_default)
    )
    if joint_observation_update:
        delayed_resample_update = False
    return joint_observation_update, delayed_resample_update


def _resolve_required_measurement_log_target(
    explicit_output: str | None,
    runtime_config: Mapping[str, Any],
    *,
    repository_root: Path,
) -> Path:
    """Resolve a mandatory pure-run log target before estimator construction."""
    raw = (
        explicit_output
        if explicit_output not in (None, "")
        else runtime_config.get("measurement_log_output_dir")
    )
    if raw in (None, ""):
        raise ValueError(
            "Pure PF live runs require measurement_log_output or "
            "runtime_config.measurement_log_output_dir before estimation."
        )
    target = Path(str(raw)).expanduser()
    if not target.is_absolute():
        target = Path(repository_root) / target
    if target.exists():
        raise FileExistsError(
            f"Refusing to replace required pure MeasurementLog {target}."
        )
    return target


def _measurement_log_obstacle_layout_path(
    obstacle_environment: RuntimeObstacleEnvironment,
    *,
    repository_root: Path,
) -> str | None:
    """Return the portable fixed-layout asset referenced by a live log."""
    if obstacle_environment.mode != "fixed":
        return None
    if obstacle_environment.layout_path is None:
        return None
    resolved_root = Path(repository_root).resolve()
    resolved_layout = Path(obstacle_environment.layout_path).resolve()
    try:
        relative = resolved_layout.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            "Fixed obstacle layouts recorded in MeasurementLog must be inside "
            "the repository."
        ) from exc
    return relative.as_posix()


def _truth_free_live_runtime_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove source-realization inputs before publishing PF provenance."""

    def _is_realization_key(key: object) -> bool:
        normalized = "".join(
            character for character in str(key).lower() if character.isalnum()
        )
        if normalized.startswith(("sourcerate", "sourceextent")):
            return any(
                marker in normalized
                for marker in (
                    "groundtruth",
                    "layout",
                    "generation",
                    "rng",
                    "seed",
                )
            )
        return (
            normalized.startswith("randomsource")
            or normalized.startswith("sourcegeneration")
            or normalized.startswith("sourcerng")
            or normalized.startswith("sourcelayout")
            or normalized
            in {
                "sourcecount",
                "sourceintensity",
                "sourceseed",
                "sources",
                "pointsources",
                "truesources",
            }
        )

    def _sanitize(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {
                str(key): _sanitize(nested)
                for key, nested in item.items()
                if not _is_realization_key(key)
            }
        if isinstance(item, list):
            return [_sanitize(nested) for nested in item]
        if isinstance(item, tuple):
            return tuple(_sanitize(nested) for nested in item)
        return item

    return dict(_sanitize(value))


def _build_effective_live_runtime_config(
    runtime_config: Mapping[str, Any],
    *,
    pf_config: object,
    candidate_sources_xyz: NDArray[np.float64],
    source_position_bounds: tuple[NDArray[np.float64], NDArray[np.float64]],
    api_settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one canonical config binding every resolved live-PF input."""
    candidates = np.asarray(candidate_sources_xyz, dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidate_sources_xyz must have shape (N, 3).")
    lower = np.asarray(source_position_bounds[0], dtype=np.float64).reshape(-1)
    upper = np.asarray(source_position_bounds[1], dtype=np.float64).reshape(-1)
    if lower.shape != (3,) or upper.shape != (3,):
        raise ValueError("source_position_bounds must contain two XYZ vectors.")
    spacing = np.asarray(
        api_settings.get("candidate_grid_spacing_m"), dtype=np.float64
    ).reshape(-1)
    if spacing.shape != (3,) or np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(
            "api_settings.candidate_grid_spacing_m must be a positive XYZ vector."
        )
    payload = _truth_free_live_runtime_config(runtime_config)
    payload["effective_pf_replay"] = {
        "api_settings": json_safe(dict(api_settings)),
        "pf_config": json_safe(pf_config),
        "candidate_grid": {
            "generator": "realtime_source_candidate_grid.v1",
            "point_count": int(candidates.shape[0]),
            "xyz_sha256": sha256_json(candidates),
            "spacing_xyz_m": json_safe(spacing),
            "margin_m": float(api_settings.get("candidate_grid_margin_m", 0.0)),
            "source_surface_prior": bool(
                api_settings.get("source_surface_prior", True)
            ),
            "obstacle_height_m": float(api_settings.get("obstacle_height_m", 2.0)),
            "position_min_xyz_m": json_safe(lower),
            "position_max_xyz_m": json_safe(upper),
        },
    }
    return dict(json_safe(payload))


def _resolve_random_source_isotopes(
    requested: str | Sequence[str] | None,
    runtime_config: Mapping[str, Any],
    library_isotopes: Sequence[str],
) -> tuple[str, ...]:
    """Return isotope names used by surface-random source generation."""
    config_requested = runtime_config.get("random_source_isotopes")
    raw_requested: object = requested if requested is not None else config_requested
    if raw_requested is None:
        names = [str(name) for name in library_isotopes]
    elif isinstance(raw_requested, str):
        names = [name.strip() for name in raw_requested.split(",") if name.strip()]
    elif isinstance(raw_requested, Sequence):
        names = [str(name).strip() for name in raw_requested if str(name).strip()]
    else:
        raise TypeError("random_source_isotopes must be a string or sequence.")
    if not names:
        raise ValueError("random_source_isotopes must contain at least one isotope.")
    library_set = {str(name) for name in library_isotopes}
    unknown = sorted(set(names).difference(library_set))
    if unknown:
        raise ValueError(
            "random_source_isotopes contains isotopes not in the spectrum library: "
            f"{unknown}"
        )
    return tuple(names)


def _runtime_float(
    runtime_config: Mapping[str, Any],
    key: str,
    default: float,
) -> float:
    """Return a runtime float while treating explicit JSON null as the default."""
    value = runtime_config.get(key)
    if value is None:
        return float(default)
    return float(value)


def _planning_primary_history_weight(
    runtime_config: Mapping[str, Any],
) -> float:
    """Return the minimum DSS history weight allowed by transport sampling."""
    sampling_fraction = _runtime_float(
        runtime_config,
        "primary_sampling_fraction",
        1.0,
    )
    if (
        not np.isfinite(sampling_fraction)
        or sampling_fraction <= 0.0
        or sampling_fraction > 1.0
    ):
        raise ValueError("primary_sampling_fraction must be finite and in (0, 1].")
    return 1.0 / sampling_fraction


def _target_sampled_primaries(
    runtime_config: Mapping[str, Any],
) -> int | None:
    """Return a validated per-transport-invocation primary budget when enabled."""
    raw_target = runtime_config.get("target_sampled_primaries")
    if raw_target in (None, ""):
        return None
    if isinstance(raw_target, bool) or not isinstance(raw_target, int):
        raise ValueError("target_sampled_primaries must be a positive integer.")
    if raw_target <= 0:
        raise ValueError("target_sampled_primaries must be a positive integer.")
    return int(raw_target)


def _transport_detector_budget_radius_m(
    runtime_config: Mapping[str, Any],
) -> float:
    """Return the physical crystal radius used by native history budgeting."""
    detector_model = runtime_config.get("detector_model", {})
    if not isinstance(detector_model, Mapping):
        detector_model = {}
    return detector_active_radius_m(
        detector_model,
        default_radius_m=DEFAULT_CRYSTAL_RADIUS_M,
    )


def _validate_adaptive_primary_budget_contract(
    runtime_config: Mapping[str, Any],
    *,
    adaptive_dwell: bool,
) -> None:
    """Reject adaptive dwell when a per-invocation primary budget is configured."""
    if not bool(adaptive_dwell):
        return
    if _target_sampled_primaries(runtime_config) is None:
        return
    raise ValueError(
        "target_sampled_primaries is a target budget per Geant4 transport "
        "invocation; adaptive planning with an unknown number of transport "
        "chunks is unsupported. Use fixed dwell or a fixed "
        "primary_sampling_fraction."
    )


def _validate_weighted_pf_runtime_contract(
    runtime_config: Mapping[str, Any],
    *,
    count_likelihood_model: str,
    observation_variance_semantics: str,
    direct_spectrum_likelihood_enable: bool,
    shield_contrast_likelihood_enable: bool,
    shield_view_ratio_likelihood_enable: bool,
    planning_primary_history_weight: float,
) -> None:
    """Fail closed when weighted transport would double-count PF evidence."""
    fraction = _runtime_float(runtime_config, "primary_sampling_fraction", 1.0)
    target_sampled_primaries = _target_sampled_primaries(runtime_config)
    weighted_requested = (
        fraction < 1.0 - 1.0e-12 or target_sampled_primaries is not None
    )
    if not weighted_requested:
        return
    if runtime_config.get("accelerated_weighted_transport_enable") is not True:
        raise ValueError(
            "Weighted PF runtime requires accelerated_weighted_transport_enable=true."
        )
    if str(observation_variance_semantics) != "complete_statistical":
        raise ValueError(
            "Weighted PF runtime requires complete_statistical observation variance."
        )
    if str(count_likelihood_model).strip().lower() not in {"gaussian", "student_t"}:
        raise ValueError(
            "Weighted PF runtime requires a gaussian or student_t count likelihood."
        )
    if bool(direct_spectrum_likelihood_enable):
        raise ValueError(
            "Weighted PF runtime cannot reuse counts in the direct spectrum likelihood."
        )
    if bool(shield_contrast_likelihood_enable) or bool(
        shield_view_ratio_likelihood_enable
    ):
        raise ValueError(
            "Weighted PF runtime requires shield contrast and view-ratio "
            "auxiliary likelihoods to be disabled."
        )
    minimum_weight = 1.0 / fraction
    if not np.isclose(
        float(planning_primary_history_weight),
        minimum_weight,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError(
            "DSS minimum primary history weight must be reciprocal to the "
            "maximum transport sampling fraction."
        )


def _seed_pf_random_generators(seed: int) -> None:
    """Seed NumPy and Torch PF/planning draws from one declared run seed."""
    resolved_seed = int(seed)
    np.random.seed(resolved_seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(resolved_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved_seed)


def _resolve_pf_initial_strength_prior(
    runtime_config: Mapping[str, Any],
) -> tuple[str, float, float | None]:
    """Resolve the explicit PF strength prior without using realized truth."""
    source_rate_model = str(runtime_config.get("source_rate_model", "")).strip().lower()
    generator_min = runtime_config.get("random_source_intensity_min_cps_1m")
    generator_max = runtime_config.get("random_source_intensity_max_cps_1m")
    generator_prior_available = bool(
        source_rate_model == "detector_cps_1m"
        and generator_min is not None
        and generator_max is not None
    )
    configured_prior = runtime_config.get("pf_init_strength_prior")
    prior = (
        "uniform"
        if configured_prior is None and generator_prior_available
        else str(configured_prior or "lognormal").strip().lower().replace("-", "_")
    )
    default_min = float(generator_min) if generator_prior_available else 0.0
    default_max = float(generator_max) if generator_prior_available else None
    minimum = max(
        _runtime_float(
            runtime_config,
            "pf_init_strength_min_cps_1m",
            default_min if prior in {"uniform", "log_uniform"} else 0.0,
        ),
        0.0,
    )
    maximum_raw = runtime_config.get("pf_init_strength_max_cps_1m")
    if maximum_raw is None and prior in {"uniform", "log_uniform"}:
        maximum_raw = default_max
    maximum = None if maximum_raw is None else float(maximum_raw)
    if prior not in {"lognormal", "uniform", "log_uniform"}:
        raise ValueError(
            "pf_init_strength_prior must be lognormal, uniform, or log_uniform."
        )
    if maximum is not None and maximum < minimum:
        raise ValueError(
            "pf_init_strength_max_cps_1m must be >= pf_init_strength_min_cps_1m."
        )
    if prior in {"uniform", "log_uniform"}:
        if maximum is None or not np.isfinite(maximum):
            raise ValueError("bounded PF strength priors require a finite maximum.")
    if prior == "log_uniform" and minimum <= 0.0:
        raise ValueError("log_uniform PF strength prior requires a positive minimum.")
    return prior, float(minimum), maximum


def _resolve_candidate_isotopes(
    runtime_config: Mapping[str, Any],
    library_isotopes: Sequence[str],
) -> tuple[str, ...]:
    """Return isotope names that the online PF should estimate."""
    raw_requested = runtime_config.get(
        "candidate_isotopes",
        runtime_config.get("pf_candidate_isotopes"),
    )
    if raw_requested is None:
        names = [str(name) for name in library_isotopes]
    elif isinstance(raw_requested, str):
        names = [name.strip() for name in raw_requested.split(",") if name.strip()]
    elif isinstance(raw_requested, Sequence):
        names = [str(name).strip() for name in raw_requested if str(name).strip()]
    else:
        raise TypeError("candidate_isotopes must be a string or sequence.")
    if not names:
        raise ValueError("candidate_isotopes must contain at least one isotope.")
    library_set = {str(name) for name in library_isotopes}
    unknown = sorted(set(names).difference(library_set))
    if unknown:
        raise ValueError(
            "candidate_isotopes contains isotopes not in the spectrum library: "
            f"{unknown}"
        )
    return tuple(names)


def _format_random_source_intensity_spec(
    intensity_spec: float | tuple[float, float],
) -> str:
    """Format fixed or random source-strength settings for runtime logs."""
    if isinstance(intensity_spec, tuple):
        return f"uniform[{intensity_spec[0]:.6g}, {intensity_spec[1]:.6g}]"
    return f"{float(intensity_spec):.6g}"


def _configure_matplotlib() -> None:
    """Configure matplotlib backend for interactive or headless use."""
    headless = _argv_requests_cui()
    if headless or not _has_display():
        matplotlib.use("Agg")
        return
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")


_configure_matplotlib()

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from measurement.model import EnvironmentConfig, PointSource
from measurement.obstacle_assets import obstacle_instances_to_dicts
from measurement.obstacles import ObstacleGrid
from measurement.source_surfaces import (
    build_surface_candidate_sources,
    generate_surface_sources,
    source_surface_kind_counts,
    source_surface_kinds,
    surface_observable_fractions,
    surface_response_observability_diagnostics,
)
from measurement.shielding import (
    generate_octant_orientations,
    generate_octant_rotation_matrices,
)
from spectrum.library import get_detection_lines_keV
from spectrum.peak_detection import detect_peaks
from spectrum.pipeline import (
    ResponsePoissonCovarianceChunk,
    SpectralDecomposer,
    SpectrumConfig,
)
from spectrum.runtime_config import (
    spectrum_config_from_runtime_config as build_spectrum_config_from_runtime_config,
)
from spectrum.runtime_counts import RuntimeCountExtractor, RuntimeCountResult
from spectrum.baseline import baseline_als
from spectrum.smoothing import gaussian_smooth
from pf.likelihood import (
    DEFAULT_GEANT4_COUNT_LIKELIHOOD_DF,
    DEFAULT_GEANT4_COUNT_LIKELIHOOD_MODEL,
    DEFAULT_GEANT4_LOW_COUNT_ABS_SIGMA,
    DEFAULT_GEANT4_LOW_COUNT_TRANSITION_COUNTS,
    DEFAULT_GEANT4_SPECTRUM_COUNT_ABS_SIGMA,
    DEFAULT_GEANT4_SPECTRUM_COUNT_REL_SIGMA,
    DEFAULT_GEANT4_TRANSPORT_MODEL_ABS_SIGMA,
    DEFAULT_GEANT4_TRANSPORT_MODEL_REL_SIGMA,
    expected_counts_per_source,
)
from pf.parallel import Measurement
from pf.pure_estimator import RotatingShieldPFEstimator, RotatingShieldPFConfig
from pf.profiles import apply_profile_to_config, enforce_pure_runtime_settings
from pf.provenance import json_safe, repository_commit, sha256_json
from pf.replay import build_replay_estimator, replay_records
from planning.candidate_generation import (
    generate_candidate_poses,
    resolve_detector_height_actions,
)
from planning.pose_selection import (
    DEFAULT_PLANNING_ROLLOUTS,
    minimum_observation_shortfall,
    select_next_pose_from_candidates,
)
from planning.dss_pp import DSSPPConfig, select_dss_pp_next_station
from planning.measurement_workspace import (
    AxisAlignedRoomBounds,
    DetectorAssemblyGeometry,
    MeasurementWorkspace,
)
from planning.remaining_measurements import (
    RemainingMeasurementConfig,
    estimate_remaining_measurement_budget,
    format_remaining_measurement_estimate,
)
from planning.traversability import (
    TraversabilityMap,
    build_traversability_map_from_obstacle_grid,
    render_traversability_map,
    shortest_grid_path_points,
)
from visualization.realtime_viz import (
    AsyncCUISplitPFVisualizer,
    CUISplitPFVisualizer,
    DEFAULT_ISOTOPE_COLORS,
    PFFrame,
    RealTimePFVisualizer,
    build_frame_from_pf,
    frame_to_isaac_pf_payload,
)
from visualization.ig_shield_geometry import render_octant_grid
from evaluation_diagnostics import (
    finish_gpu_memory_tracking,
    start_gpu_memory_tracking,
    summarize_cluster_stability,
    summarize_count_bias,
)
from evaluation_metrics import compute_metrics, print_metrics_report
from piplup_notify import PiplupNotificationConfig, PiplupNotifier
from cui_runtime import (
    ensure_cui_view_server as _ensure_cui_view_server,
    resolve_cui_split_view_enabled as _resolve_cui_split_view_enabled,
)
from mission_control import (
    has_birth_residual_evidence as _has_birth_residual_evidence,
    remaining_measurement_payload as _remaining_measurement_payload,
    remaining_measurement_ready_for_stop as _remaining_measurement_ready_for_stop,
    resolve_mission_max_poses as _resolve_mission_max_poses,
    resolve_mission_max_steps as _resolve_mission_max_steps,
)
from runtime_defaults import (
    DEFAULT_CUI_SPLIT_VIEW_DIR,
    DEFAULT_ENVIRONMENT_MODE,
    DEFAULT_FIXED_OBSTACLE_CONFIG,
    DEFAULT_MAX_SOURCES_PER_ISOTOPE,
    DEFAULT_MEASUREMENT_TIME_S,
    DEFAULT_RANDOM_SOURCE_COUNT,
    DEFAULT_RANDOM_SOURCE_INTENSITY_CPS_1M,
    DEFAULT_ROBOT_SPEED_M_S,
    DEFAULT_ROTATION_OVERHEAD_S,
)
from runtime_environment import (
    RuntimeObstacleEnvironment,
    build_runtime_obstacle_environment,
)
from runtime.measurement_log import (
    MeasurementLog,
    MeasurementLogRecord,
    MeasurementLogStreamWriter,
    build_forward_model_manifest,
)


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


def _build_resume_replay_estimator(
    prefix_log: MeasurementLog,
    *,
    profile: str,
    seed: int,
    config_hash: str,
    resolved_config_hash: str,
) -> RotatingShieldPFEstimator:
    """Build a resume estimator from the PF settings stored in the prefix."""
    external_config: Mapping[str, Any]
    if prefix_log.runtime_config.get("effective_pf_replay") is None:
        external_config = prefix_log.runtime_config
    else:
        external_config = {}
    return build_replay_estimator(
        prefix_log,
        external_config,
        profile=profile,
        seed=seed,
        config_hash=config_hash,
        resolved_config_hash=resolved_config_hash,
    )


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
    text = str(value)
    return len(text) == 40 and all(character in "0123456789abcdef" for character in text)


def _sha256_path(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for one regular file."""
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"Expected one regular file for SHA-256: {candidate}.")
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resume_stage_repository_commit(stage_dir: str | Path) -> str:
    """Read the staged prefix commit before rebuilding its forward identity."""
    stage = Path(stage_dir)
    if stage.is_symlink() or not stage.is_dir():
        raise ValueError("The resume MeasurementLog stage must be a real directory.")
    commit_path = stage / "repository_commit.txt"
    if commit_path.is_symlink() or not commit_path.is_file():
        raise ValueError("The resume stage lacks a regular repository_commit.txt.")
    try:
        raw = commit_path.read_bytes()
    except OSError as exc:
        raise ValueError("Cannot read the resume stage repository commit.") from exc
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise ValueError("The resume stage repository commit is not canonical.")
    try:
        commit = raw[:-1].decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("The resume stage repository commit is not ASCII.") from exc
    if not _full_git_commit(commit):
        raise ValueError("The resume stage requires one full Git commit.")
    return commit


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
        Path(str(path)).as_posix()
        for path in (additional_compatible_code_paths or ())
    }
    if any(
        path.startswith("/")
        or path == ".."
        or path.startswith("../")
        or "/../" in path
        for path in extra_allowed
    ):
        raise RuntimeError("Resume compatible code paths must be repository-relative.")
    allowed_runtime_paths = set(_RESUME_ORCHESTRATION_CODE_PATHS) | extra_allowed
    changed_runtime_paths = {
        path
        for path in changed_paths
        if _is_resume_runtime_path(path)
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
        "compatibility_basis": (
            basis if basis else "no_live_runtime_path_delta"
        ),
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
    last_counts: dict[str, float]
    representative_spectrum: NDArray[np.float64]
    representative_counts: dict[str, float]
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

    remaining_measurement_estimates: tuple[dict[str, Any], ...]
    remaining_measurement_budget_history: tuple[dict[str, float], ...]
    remaining_measurement_eta_ratios: tuple[float, ...]
    max_poses: int | None
    soft_pose_extension_used: int
    ig_max_global: float
    ig_max_pose: float
    ig_threshold_current: float
    last_max_ig: float | None


def _planning_candidate_checkpoint_parameters(
    *,
    pose_candidates: int,
    pose_min_dist: float,
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]],
    detector_heights_m: Sequence[float] | None,
    continuous_height_anchor_count: int,
    height_partner_xy_tolerance_m: float,
    height_partner_z_tolerance_m: float,
    height_partner_min_z_separation_m: float,
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
        "continuous_height_anchor_count": int(continuous_height_anchor_count),
        "height_partner_xy_tolerance_m": float(height_partner_xy_tolerance_m),
        "height_partner_z_tolerance_m": float(height_partner_z_tolerance_m),
        "height_partner_min_z_separation_m": float(
            height_partner_min_z_separation_m
        ),
    }


def _build_live_controller_checkpoint(
    *,
    planning_candidate_rng: np.random.Generator,
    planning_candidate_parameters: Mapping[str, Any],
    remaining_measurement_estimates: Sequence[Mapping[str, Any]],
    estimator: RotatingShieldPFEstimator,
    max_poses: int | None,
    soft_pose_extension_used: int,
    ig_max_global: float,
    ig_max_pose: float,
    ig_threshold_current: float,
    last_max_ig: float | None,
) -> dict[str, Any]:
    """Build one truth-free controller checkpoint before post-station planning."""
    budget_history_raw = getattr(
        estimator,
        "_remaining_measurement_budget_history",
        (),
    )
    eta_ratios_raw = getattr(
        estimator,
        "_remaining_measurement_eta_ratios",
        (),
    )
    payload = {
        "schema_version": 1,
        "planning_candidate_rng_state": json_safe(
            planning_candidate_rng.bit_generator.state
        ),
        "planning_candidate_parameters": json_safe(
            dict(planning_candidate_parameters)
        ),
        "remaining_measurement_estimates": json_safe(
            [dict(value) for value in remaining_measurement_estimates]
        ),
        "remaining_measurement_budget_history": json_safe(
            [dict(value) for value in budget_history_raw]
        ),
        "remaining_measurement_eta_ratios": json_safe(list(eta_ratios_raw)),
        "mission_state": {
            "max_poses": None if max_poses is None else int(max_poses),
            "soft_pose_extension_used": int(soft_pose_extension_used),
            "ig_max_global": float(ig_max_global),
            "ig_max_pose": float(ig_max_pose),
            "ig_threshold_current": float(ig_threshold_current),
            "last_max_ig": (
                None if last_max_ig is None else float(last_max_ig)
            ),
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
        "planning_candidate_parameters",
        "remaining_measurement_estimates",
        "remaining_measurement_budget_history",
        "remaining_measurement_eta_ratios",
        "mission_state",
    }
    if set(checkpoint) != expected_keys or checkpoint["schema_version"] != 1:
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

    estimates_raw = checkpoint["remaining_measurement_estimates"]
    history_raw = checkpoint["remaining_measurement_budget_history"]
    eta_raw = checkpoint["remaining_measurement_eta_ratios"]
    mission_raw = checkpoint["mission_state"]
    if (
        not isinstance(estimates_raw, list)
        or not all(isinstance(value, Mapping) for value in estimates_raw)
        or not isinstance(history_raw, list)
        or not all(isinstance(value, Mapping) for value in history_raw)
        or not isinstance(eta_raw, list)
        or not isinstance(mission_raw, Mapping)
    ):
        raise RuntimeError("Checkpoint controller history has invalid structure.")
    mission = dict(mission_raw)
    if set(mission) != {
        "max_poses",
        "soft_pose_extension_used",
        "ig_max_global",
        "ig_max_pose",
        "ig_threshold_current",
        "last_max_ig",
    }:
        raise RuntimeError("Checkpoint mission state has invalid fields.")
    try:
        history = tuple(
            {
                "budget": float(value["budget"]),
                "predicted_gain": float(value["predicted_gain"]),
            }
            for value in history_raw
        )
        eta_ratios = tuple(float(value) for value in eta_raw)
        max_poses_raw = mission["max_poses"]
        max_poses = None if max_poses_raw is None else int(max_poses_raw)
        soft_pose_extension_used = int(mission["soft_pose_extension_used"])
        ig_max_global = float(mission["ig_max_global"])
        ig_max_pose = float(mission["ig_max_pose"])
        ig_threshold_current = float(mission["ig_threshold_current"])
        last_max_ig_raw = mission["last_max_ig"]
        last_max_ig = (
            None if last_max_ig_raw is None else float(last_max_ig_raw)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Checkpoint controller values are invalid.") from exc
    finite_values = [
        *(value for entry in history for value in entry.values()),
        *eta_ratios,
        ig_max_global,
        ig_max_pose,
        ig_threshold_current,
    ]
    if last_max_ig is not None:
        finite_values.append(last_max_ig)
    if (
        any(not np.isfinite(value) for value in finite_values)
        or soft_pose_extension_used < 0
        or (max_poses is not None and max_poses < 1)
    ):
        raise RuntimeError("Checkpoint controller values are out of range.")
    return _LiveControllerCheckpoint(
        remaining_measurement_estimates=tuple(
            dict(value) for value in estimates_raw
        ),
        remaining_measurement_budget_history=history,
        remaining_measurement_eta_ratios=eta_ratios,
        max_poses=max_poses,
        soft_pose_extension_used=soft_pose_extension_used,
        ig_max_global=ig_max_global,
        ig_max_pose=ig_max_pose,
        ig_threshold_current=ig_threshold_current,
        last_max_ig=last_max_ig,
    )


def _pure_pf_profile_active(estimator: object) -> bool:
    """Return whether the estimator exposes the strict sequential PF contract."""
    capabilities = getattr(estimator, "profile_capabilities", None)
    profile = getattr(
        getattr(estimator, "pf_config", None),
        "estimator_profile",
        "",
    )
    return capabilities is not None and str(profile) == "pf_strict"


def _pure_pf_primary_estimates(
    estimator: object,
    isotopes: Sequence[str],
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] | None:
    """Return an unfiltered PF-posterior projection for a pure profile."""
    if not _pure_pf_profile_active(estimator):
        return None
    getter = getattr(estimator, "estimates", None)
    if not callable(getter):
        raise RuntimeError("A pure PF must expose its posterior estimates projection.")
    raw = getter()
    return {
        str(isotope): (
            np.asarray(
                raw.get(str(isotope), (np.zeros((0, 3)), np.zeros(0)))[0],
                dtype=float,
            ).reshape(-1, 3),
            np.asarray(
                raw.get(str(isotope), (np.zeros((0, 3)), np.zeros(0)))[1],
                dtype=float,
            ).reshape(-1),
        )
        for isotope in isotopes
    }


def _validate_surface_constrained_estimates(
    estimates: Mapping[
        str,
        tuple[NDArray[np.float64], NDArray[np.float64]],
    ],
    environment: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    *,
    obstacle_height_m: float,
    tolerance_m: float,
    surface_prior_active: bool,
) -> None:
    """Fail closed when a surface-prior report contains an off-surface point."""
    if not surface_prior_active:
        return
    position_groups = [
        np.asarray(estimate[0], dtype=float).reshape(-1, 3)
        for estimate in estimates.values()
        if np.asarray(estimate[0]).size
    ]
    if not position_groups:
        return
    positions = np.concatenate(position_groups, axis=0)
    surface_kinds = source_surface_kinds(
        positions,
        environment,
        obstacle_grid,
        obstacle_height_m=obstacle_height_m,
        tolerance_m=tolerance_m,
    )
    off_surface_count = int(np.count_nonzero(np.equal(surface_kinds, None)))
    if off_surface_count:
        raise RuntimeError(
            "Surface-constrained PF report contains "
            f"{off_surface_count}/{positions.shape[0]} off-surface positions."
        )


def _validate_surface_constrained_sources(
    sources: Sequence[PointSource],
    environment: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    *,
    obstacle_height_m: float,
    tolerance_m: float,
) -> None:
    """Reject simulation truth outside the strict PF position support."""
    if not sources:
        return
    positions = np.asarray([source.position for source in sources], dtype=float)
    surface_kinds = source_surface_kinds(
        positions,
        environment,
        obstacle_grid,
        obstacle_height_m=obstacle_height_m,
        tolerance_m=tolerance_m,
    )
    off_surface_count = int(np.count_nonzero(np.equal(surface_kinds, None)))
    if off_surface_count:
        raise ValueError(
            "Surface-constrained PF simulation contains "
            f"{off_surface_count}/{positions.shape[0]} off-surface true sources."
        )


_PURE_PF_SUMMARY_PROVENANCE_KEYS = (
    "schema_version",
    "estimator_family",
    "estimator_variant",
    "estimator_profile",
    "final_estimate_source",
    "posterior_semantics",
    "structural_kernel_family",
    "structural_kernel_target_preserving",
    "structural_kernel_exact_rj",
    "reversible_jump_mcmc_used",
    "structural_transition_provenance",
    "planner_belief_sources",
    "repository_commit",
    "measurement_log_schema_version",
    "measurement_log_sha256",
    "config_sha256",
    "resolved_config_sha256",
    "random_seed",
    "profile_capability_map",
)


def _pure_pf_summary_provenance(estimator: object) -> dict[str, Any]:
    """Embed mandatory PF provenance in every runtime summary result."""
    if not _pure_pf_profile_active(estimator):
        return {}
    snapshot_getter = getattr(estimator, "posterior_snapshot", None)
    if not callable(snapshot_getter):
        raise RuntimeError("A pure PF result requires posterior_snapshot provenance.")
    snapshot = snapshot_getter()
    serializer = getattr(snapshot, "to_dict", None)
    if not callable(serializer):
        raise RuntimeError("A pure PF posterior snapshot must be serializable.")
    payload = dict(serializer())
    missing = [key for key in _PURE_PF_SUMMARY_PROVENANCE_KEYS if key not in payload]
    if missing:
        raise RuntimeError(
            "Pure PF posterior provenance is incomplete: " + ", ".join(missing)
        )
    return {
        **{key: payload[key] for key in _PURE_PF_SUMMARY_PROVENANCE_KEYS},
        "estimator_provenance": dict(payload.get("provenance", {})),
        "pf_posterior": payload,
    }


from sim import (
    SimulationCommand,
    SimulationObservation,
    SimulationRuntime,
    create_simulation_runtime,
    load_runtime_config,
)
from sim.blender_environment import generate_blender_environment_usd
from sim.shield_geometry import resolve_shield_thickness_config
from baselines.ral_ablation.path_policies import select_baseline_next_pose
from baselines.ral_ablation.shield_policies import (
    BaselineShieldProgram,
    select_baseline_shield_program,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
SPECTRUM_DIR = RESULTS_DIR / "spectrum"
PF_DIR = RESULTS_DIR / "pf"
IG_DIR = RESULTS_DIR / "IG"
BLENDER_ENV_DIR = RESULTS_DIR / "blender_environments"
SAVE_IG_GRIDS = False
OBSTACLE_LAYOUT_DIR = ROOT / "obstacle_layouts"
DETECT_MIN_PEAKS_BY_ISOTOPE = {"Eu-154": 2, "Co-60": 2}
DETECT_REL_THRESH_BY_ISOTOPE = {"Co-60": 0.1}
DETECT_CONSECUTIVE_BY_ISOTOPE = {"Cs-137": 3, "Co-60": 3, "Eu-154": 5}
DETECT_MISS_AFTER_LOCK = 30
DEFAULT_SOURCE_CONFIG = ROOT / "source_layouts" / "demo_sources.json"
DEFAULT_OBSTACLE_CONFIG = ROOT / DEFAULT_FIXED_OBSTACLE_CONFIG
CANDIDATE_GRID_SPACING = (0.5, 0.5, 0.5)
CANDIDATE_GRID_MARGIN = 0.5
HEALTH_LOG_TOP_K = 0
ADAPTIVE_STEP_ID_STRIDE = 100000


class DeferredPFVisualizer:
    """Delay expensive Matplotlib rendering until a figure is explicitly saved."""

    def __init__(
        self,
        visualizer_factory: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Store construction arguments without creating the figure."""
        self._visualizer_factory = visualizer_factory
        self._args = args
        self._kwargs = kwargs
        self._inner: object | None = None
        self._last_frame: object | None = None

    def update(self, frame: object) -> None:
        """Keep only the latest frame for deferred rendering."""
        self._last_frame = frame

    def _materialize(self) -> object:
        """Create the underlying Matplotlib visualizer on first save."""
        if self._inner is None:
            factory = self._visualizer_factory
            if not callable(factory):
                raise TypeError("visualizer_factory must be callable")
            self._inner = factory(*self._args, **self._kwargs)
        return self._inner

    def _sync_latest_frame(self) -> object:
        """Update the underlying visualizer with the latest stored frame."""
        inner = self._materialize()
        if self._last_frame is not None:
            update = getattr(inner, "update")
            update(self._last_frame)
        return inner

    def save_final(self, path: str) -> None:
        """Render and save the latest full PF visualization."""
        inner = self._sync_latest_frame()
        save = getattr(inner, "save_final")
        save(path)

    def save_estimates_only(self, path: str) -> None:
        """Render and save the latest estimates-only visualization."""
        inner = self._sync_latest_frame()
        save = getattr(inner, "save_estimates_only")
        save(path)


def _has_environment_obstacles(obstacle_grid: ObstacleGrid | None) -> bool:
    """Return whether an obstacle grid contains authored physical obstacles."""
    return obstacle_grid is not None and bool(
        obstacle_grid.blocked_cells or obstacle_grid.collision_boxes_m
    )


def _pf_obstacle_attenuation_enabled(runtime_config: dict[str, object]) -> bool:
    """Return whether PF expected-count kernels should include obstacles."""
    raw = runtime_config.get("pf_obstacle_attenuation", True)
    if raw is None:
        return True
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    return str(raw).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "disable",
        "disabled",
    }


def _optional_runtime_bool(
    runtime_config: Mapping[str, object],
    key: str,
) -> bool | None:
    """Return an optional boolean runtime setting with string coercion."""
    if key not in runtime_config or runtime_config[key] is None:
        return None
    raw = runtime_config[key]
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    lowered = str(raw).strip().lower()
    if lowered in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if lowered in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    raise ValueError(f"Runtime config key {key!r} must be boolean-like.")


def _apply_baseline_shield_program_to_dss_config(
    dss_config: DSSPPConfig,
    baseline_shield_policy: Mapping[str, Any] | str | None,
    *,
    total_pairs: int,
    pose_index: int,
    current_pair_id: int | None,
) -> tuple[DSSPPConfig, BaselineShieldProgram | None]:
    """
    Force a baseline shield program while preserving the active DSS program length.

    The program length must be read from the already-adapted DSS-PP config so
    shield ablations compare posture selection rather than measurement budget.
    """
    baseline_program = select_baseline_shield_program(
        baseline_shield_policy,
        total_pairs=total_pairs,
        program_length=int(dss_config.program_length),
        pose_index=pose_index,
        current_pair_id=current_pair_id,
    )
    if baseline_program is None:
        return dss_config, None
    forced_pairs = tuple(int(pair_id) for pair_id in baseline_program.pair_ids)
    return (
        replace(dss_config, forced_program_pair_ids=forced_pairs),
        baseline_program,
    )


def _adapt_dss_config_from_pf_evidence(
    config: DSSPPConfig,
    *,
    adaptive_program_length_enabled: bool,
    residual_unresolved: bool,
    ambiguity_unresolved: bool,
    require_cardinality_gap: bool,
    cardinality_ready: bool,
    remaining_ready: bool,
    residual_program_length: int,
    simple_program_length: int,
    priority_isotopes: Sequence[str],
    planner_mode: str,
) -> tuple[DSSPPConfig, str]:
    """
    Return a DSS config adapted from already-computed PF evidence.

    A forced shield program disables every adaptive change. Otherwise the
    program-length switch controls only the number of views; PF isotope
    priorities and the explicit planner mode remain independent controls.
    """
    if config.forced_program_pair_ids is not None:
        return config, "forced_program"

    adapted = config
    reason = "disabled" if not adaptive_program_length_enabled else "full"
    base_length = max(1, int(config.program_length))
    if adaptive_program_length_enabled:
        residual_length = max(base_length, int(residual_program_length))
        resolved_simple_length = max(
            1,
            min(base_length, int(simple_program_length)),
        )
        if residual_unresolved and (
            not require_cardinality_gap or not cardinality_ready
        ):
            adapted = replace(adapted, program_length=residual_length)
            reason = "pf_residual"
        elif ambiguity_unresolved:
            reason = "pf_ambiguity"
        elif remaining_ready and cardinality_ready and (
            resolved_simple_length < base_length
        ):
            adapted = replace(adapted, program_length=resolved_simple_length)
            reason = "pf_ready"

    priorities = tuple(str(isotope) for isotope in priority_isotopes)
    if priorities:
        adapted = replace(adapted, recovery_isotopes=priorities)

    if bool(adapted.explicit_mode_switch):
        if planner_mode == "global_recovery":
            adapted = replace(
                adapted,
                planner_mode=planner_mode,
                lambda_frontier=max(float(adapted.lambda_frontier), 2.0),
                lambda_coverage=max(float(adapted.lambda_coverage), 2.0),
            )
        elif planner_mode == "local_disambiguation":
            adapted = replace(
                adapted,
                planner_mode=planner_mode,
                lambda_signature=max(float(adapted.lambda_signature), 3.0),
                lambda_temporal_separation=max(
                    float(adapted.lambda_temporal_separation),
                    10.0,
                ),
                lambda_correlation_reduction=max(
                    float(adapted.lambda_correlation_reduction),
                    6.0,
                ),
                lambda_station_condition=max(
                    float(adapted.lambda_station_condition),
                    8.0,
                ),
                lambda_elevation_condition=max(
                    float(adapted.lambda_elevation_condition),
                    5.0,
                ),
            )
        elif planner_mode == "verification":
            adapted = replace(
                adapted,
                planner_mode=planner_mode,
                lambda_correlation_reduction=max(
                    float(adapted.lambda_correlation_reduction),
                    6.0,
                ),
                lambda_temporal_separation=max(
                    float(adapted.lambda_temporal_separation),
                    8.0,
                ),
            )
        else:
            adapted = replace(adapted, planner_mode=planner_mode)
    return adapted, reason


def _resolve_rotation_limit_for_active_program(
    *,
    base_rotation_limit: int,
    active_shield_program: Sequence[int] | None,
    strict_planned_shield_program: bool,
    baseline_shield_policy: Mapping[str, Any] | str | None,
    force_strict_program: bool = False,
) -> int:
    """Return the rotation limit for a station with an explicit shield program."""
    base_limit = max(1, int(base_rotation_limit))
    if not active_shield_program:
        return base_limit
    program_limit = max(1, len(active_shield_program))
    if (
        strict_planned_shield_program
        or baseline_shield_policy is not None
        or force_strict_program
    ):
        return program_limit
    return max(base_limit, program_limit)


def _pf_obstacle_grid_for_runtime(
    obstacle_grid: ObstacleGrid | None,
    runtime_config: dict[str, object],
) -> ObstacleGrid | None:
    """Return the obstacle grid used by the PF observation model."""
    if _pf_obstacle_attenuation_enabled(runtime_config):
        return obstacle_grid
    return None


def _surface_count_payload(
    positions: NDArray[np.float64],
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    *,
    obstacle_height_m: float,
    tolerance_m: float = 1.0e-5,
) -> dict[str, object]:
    """Return serializable surface-kind counts for source positions."""
    pos_arr = np.asarray(positions, dtype=float).reshape(-1, 3)
    counts = source_surface_kind_counts(
        pos_arr,
        env,
        obstacle_grid,
        obstacle_height_m=obstacle_height_m,
        tolerance_m=max(float(tolerance_m), 0.0),
    )
    return {
        "total_sources": int(pos_arr.shape[0]),
        "surface_counts": counts,
        "off_surface_count": int(counts.get("off_surface", 0)),
    }


def _ground_visibility_reference_points(
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    *,
    robot_radius_m: float,
    detector_height_m: float,
) -> NDArray[np.float64]:
    """Return reachable ground detector positions for source-placement screening."""
    if obstacle_grid is None or not obstacle_grid.blocked_cells:
        return np.zeros((0, 3), dtype=float)
    traversable = build_traversability_map_from_obstacle_grid(
        obstacle_grid,
        robot_radius_m=float(robot_radius_m),
        reachable_from=env.detector_position,
    )
    points = [
        (
            float(traversable.cell_center(cell)[0]),
            float(traversable.cell_center(cell)[1]),
            float(detector_height_m),
        )
        for cell in traversable.traversable_cells
    ]
    return np.asarray(points, dtype=float).reshape(-1, 3)


def _source_ground_visibility_payload(
    sources: Sequence[PointSource],
    obstacle_grid: ObstacleGrid | None,
    measurement_points: NDArray[np.float64] | None,
    *,
    obstacle_height_m: float,
    detector_height_m: float,
    clear_path_max_m: float,
    min_visible_fraction: float,
) -> dict[str, object]:
    """Return serializable source ground-visibility diagnostics."""
    points = (
        np.zeros((0, 3), dtype=float)
        if measurement_points is None
        else np.asarray(measurement_points, dtype=float).reshape(-1, 3)
    )
    if not sources or points.size == 0:
        return {
            "enabled": False,
            "reference_point_count": int(points.shape[0]),
            "min_visible_fraction": float(min_visible_fraction),
            "source_visible_fractions": [],
        }
    positions = np.asarray([source.position for source in sources], dtype=float)
    fractions = surface_observable_fractions(
        positions,
        obstacle_grid,
        points,
        obstacle_height_m=obstacle_height_m,
        detector_height_m=detector_height_m,
        clear_path_max_m=clear_path_max_m,
    )
    response_diag = surface_response_observability_diagnostics(
        positions,
        obstacle_grid,
        points,
        isotopes=[source.isotope for source in sources],
        obstacle_height_m=obstacle_height_m,
        detector_height_m=detector_height_m,
        clear_path_max_m=clear_path_max_m,
    )
    return {
        "enabled": True,
        "reference_point_count": int(points.shape[0]),
        "min_visible_fraction": float(min_visible_fraction),
        "clear_path_max_m": float(clear_path_max_m),
        "min_source_visible_fraction": float(np.min(fractions)),
        "mean_source_visible_fraction": float(np.mean(fractions)),
        "source_visible_fractions": [float(value) for value in fractions],
        "response_observability": response_diag,
    }


def _estimate_surface_diagnostics(
    estimates: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    *,
    obstacle_height_m: float,
    tolerance_m: float = 1.0e-5,
) -> dict[str, dict[str, object]]:
    """Return per-isotope surface diagnostics for final reported estimates."""
    return {
        isotope: _surface_count_payload(
            np.asarray(positions, dtype=float),
            env,
            obstacle_grid,
            obstacle_height_m=obstacle_height_m,
            tolerance_m=tolerance_m,
        )
        for isotope, (positions, _strengths) in estimates.items()
    }


def _final_estimate_source_status(
    estimator: RotatingShieldPFEstimator,
    estimates: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
) -> dict[str, list[dict[str, object]]]:
    """Return confirmed/tentative status metadata for final reported sources."""
    pf_config = getattr(estimator, "pf_config", None)
    match_radius = max(float(getattr(pf_config, "cluster_eps_m", 1.0)), 0.5)
    status_by_iso: dict[str, list[dict[str, object]]] = {}
    for isotope, estimate in sorted(estimates.items()):
        positions = np.asarray(estimate[0], dtype=float).reshape(-1, 3)
        strengths = np.asarray(estimate[1], dtype=float).reshape(-1)
        filt = estimator.filters.get(isotope)
        state_positions = np.zeros((0, 3), dtype=float)
        tentative = np.zeros(0, dtype=bool)
        fail_streak = np.zeros(0, dtype=int)
        support = np.zeros(0, dtype=float)
        if filt is not None and getattr(filt, "continuous_particles", None):
            try:
                state = filt.best_particle().state
                count = max(0, int(getattr(state, "num_sources", 0)))
                raw_positions = np.asarray(
                    getattr(state, "positions", np.zeros((0, 3))),
                    dtype=float,
                ).reshape(-1, 3)
                state_positions = raw_positions[:count]
                tentative = _metadata_value_array(
                    state,
                    "tentative_sources",
                    count,
                    fill=False,
                    dtype=bool,
                )
                fail_streak = _metadata_value_array(
                    state,
                    "verification_fail_streaks",
                    count,
                    fill=0,
                    dtype=int,
                )
                support = _metadata_value_array(
                    state,
                    "support_scores",
                    count,
                    fill=0.0,
                    dtype=float,
                )
            except (AttributeError, RuntimeError, ValueError):
                state_positions = np.zeros((0, 3), dtype=float)
        entries: list[dict[str, object]] = []
        for idx, pos in enumerate(positions):
            nearest_idx: int | None = None
            nearest_distance: float | None = None
            matched_metadata = False
            source_tentative = False
            source_fail_streak = 0
            source_support: float | None = None
            if state_positions.size:
                distances = np.linalg.norm(state_positions - pos[None, :], axis=1)
                nearest_idx = int(np.argmin(distances))
                nearest_distance = float(distances[nearest_idx])
                matched_metadata = nearest_distance <= match_radius
            if matched_metadata and nearest_idx is not None:
                source_tentative = bool(
                    tentative[nearest_idx] if nearest_idx < tentative.size else False
                )
                source_fail_streak = int(
                    fail_streak[nearest_idx] if nearest_idx < fail_streak.size else 0
                )
                source_support = float(
                    support[nearest_idx] if nearest_idx < support.size else 0.0
                )
            if source_tentative or source_fail_streak > 0:
                status = "tentative"
                reason = "verification_unconfirmed"
            else:
                status = "confirmed"
                reason = "pf_posterior"
            entries.append(
                {
                    "estimate_index": int(idx),
                    "status": status,
                    "confirmed": bool(status == "confirmed"),
                    "reason": reason,
                    "pos": [
                        float(value)
                        for value in np.asarray(pos, dtype=float).reshape(3)
                    ],
                    "strength": float(strengths[idx]) if idx < strengths.size else 0.0,
                    "nearest_state_slot": nearest_idx,
                    "nearest_state_distance_m": nearest_distance,
                    "metadata_match_radius_m": float(match_radius),
                    "tentative": bool(source_tentative),
                    "verification_fail_streak": int(source_fail_streak),
                    "support_score": source_support,
                }
            )
        status_by_iso[isotope] = entries
    return status_by_iso


def _filter_serialized_sources_by_status(
    estimates_by_iso: dict[str, list[dict[str, float | list[float]]]],
    status_by_iso: dict[str, list[dict[str, object]]],
    *,
    status: str,
) -> dict[str, list[dict[str, float | list[float]]]]:
    """Return serialized final sources whose status matches the requested label."""
    filtered: dict[str, list[dict[str, float | list[float]]]] = {}
    for isotope, entries in sorted(estimates_by_iso.items()):
        source_statuses = status_by_iso.get(isotope, [])
        kept: list[dict[str, float | list[float]]] = []
        for idx, entry in enumerate(entries):
            if idx >= len(source_statuses):
                continue
            if str(source_statuses[idx].get("status", "")) == status:
                kept.append(entry)
        filtered[isotope] = kept
    return filtered


def _particle_surface_diagnostics(
    estimator: RotatingShieldPFEstimator,
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    *,
    obstacle_height_m: float,
    tolerance_m: float = 1.0e-5,
) -> dict[str, dict[str, object]]:
    """Return per-isotope surface diagnostics for the final PF particles."""
    diagnostics: dict[str, dict[str, object]] = {}
    for isotope, filt in estimator.filters.items():
        positions: list[NDArray[np.float64]] = []
        weights: list[NDArray[np.float64]] = []
        posterior_slots = 0
        particle_weights = np.asarray(
            getattr(filt, "continuous_weights", []),
            dtype=float,
        )
        particles = list(getattr(filt, "continuous_particles", []))
        if particle_weights.size != len(particles):
            particle_weights = np.ones(len(particles), dtype=float)
        weight_total = float(np.sum(particle_weights))
        if weight_total <= 0.0 and particle_weights.size:
            particle_weights = np.ones_like(particle_weights) / float(
                particle_weights.size
            )
        elif particle_weights.size:
            particle_weights = particle_weights / weight_total
        for particle, weight in zip(particles, particle_weights):
            state = particle.state
            count = max(0, int(getattr(state, "num_sources", 0)))
            posterior_slots += count
            if count <= 0:
                continue
            positions.append(np.asarray(state.positions[:count], dtype=float))
            weights.append(np.full(count, float(weight), dtype=float))
        if positions:
            pos_arr = np.vstack(positions)
            weight_arr = np.concatenate(weights)
            kinds = source_surface_kinds(
                pos_arr,
                env,
                obstacle_grid,
                obstacle_height_m=obstacle_height_m,
                tolerance_m=max(float(tolerance_m), 0.0),
            )
        else:
            pos_arr = np.zeros((0, 3), dtype=float)
            weight_arr = np.zeros(0, dtype=float)
            kinds = np.zeros(0, dtype=object)
        counts = source_surface_kind_counts(
            pos_arr,
            env,
            obstacle_grid,
            obstacle_height_m=obstacle_height_m,
            tolerance_m=max(float(tolerance_m), 0.0),
        )
        weighted = {
            label: float(np.sum(weight_arr[kinds == label]))
            for label in (
                "floor",
                "ceiling",
                "wall",
                "obstacle_side",
                "obstacle_top",
            )
        }
        weighted["off_surface"] = float(np.sum(weight_arr[np.equal(kinds, None)]))
        diagnostics[isotope] = {
            "particles": int(len(particles)),
            "posterior_source_slots": int(posterior_slots),
            "surface_counts": counts,
            "weighted_surface_mass": weighted,
            "off_surface_count": int(counts.get("off_surface", 0)),
            "weighted_off_surface_mass": float(weighted["off_surface"]),
        }
    return diagnostics


def _final_particle_cloud_payload(
    estimator: RotatingShieldPFEstimator,
    *,
    max_points_per_isotope: int = 1200,
) -> dict[str, dict[str, object]]:
    """Return deterministic final PF source-slot samples for paper figures."""
    output: dict[str, dict[str, object]] = {}
    max_points = max(0, int(max_points_per_isotope))
    for isotope, filt in estimator.filters.items():
        particle_weights = np.asarray(
            getattr(filt, "continuous_weights", []),
            dtype=float,
        )
        particles = list(getattr(filt, "continuous_particles", []))
        if particle_weights.size != len(particles):
            particle_weights = np.ones(len(particles), dtype=float)
        if particle_weights.size:
            weight_total = float(np.sum(particle_weights))
            if weight_total <= 0.0:
                particle_weights = np.ones_like(particle_weights) / float(
                    particle_weights.size
                )
            else:
                particle_weights = particle_weights / weight_total
        positions: list[NDArray[np.float64]] = []
        weights: list[NDArray[np.float64]] = []
        for particle, weight in zip(particles, particle_weights):
            state = particle.state
            count = max(0, int(getattr(state, "num_sources", 0)))
            if count <= 0:
                continue
            active_positions = np.asarray(state.positions[:count], dtype=float)
            positions.append(active_positions)
            weights.append(
                np.full(active_positions.shape[0], float(weight), dtype=float)
            )
        if positions:
            position_arr = np.vstack(positions)
            weight_arr = np.concatenate(weights)
            if max_points and position_arr.shape[0] > max_points:
                order = np.lexsort(
                    (
                        position_arr[:, 2],
                        position_arr[:, 1],
                        position_arr[:, 0],
                        -weight_arr,
                    )
                )
                order = order[:max_points]
                position_arr = position_arr[order]
                weight_arr = weight_arr[order]
        else:
            position_arr = np.zeros((0, 3), dtype=float)
            weight_arr = np.zeros(0, dtype=float)
        output[isotope] = {
            "positions": position_arr.tolist(),
            "weights": weight_arr.tolist(),
            "total_source_slots": int(sum(arr.shape[0] for arr in positions)),
            "stored_source_slots": int(position_arr.shape[0]),
        }
    return output


def _compact_path_segments(
    path_segments: list[dict[str, object]],
    *,
    max_waypoints_per_segment: int = 160,
) -> list[dict[str, object]]:
    """Return path segments without large planner diagnostics for summaries."""
    compact: list[dict[str, object]] = []
    waypoint_limit = max(2, int(max_waypoints_per_segment))
    for segment in path_segments:
        waypoints = np.asarray(segment.get("waypoints_xyz", []), dtype=float)
        if waypoints.ndim != 2 or waypoints.shape[1] < 3 or waypoints.shape[0] == 0:
            waypoints = np.zeros((0, 3), dtype=float)
        elif waypoints.shape[0] > waypoint_limit:
            indices = np.unique(
                np.linspace(0, waypoints.shape[0] - 1, waypoint_limit).astype(int)
            )
            waypoints = waypoints[indices]
        compact.append(
            {
                "from_pose_xyz": segment.get("from_pose_xyz"),
                "to_pose_xyz": segment.get("to_pose_xyz"),
                "waypoints_xyz": waypoints.tolist(),
                "distance_m": float(segment.get("distance_m", 0.0) or 0.0),
                "euclidean_distance_m": float(
                    segment.get("euclidean_distance_m", 0.0) or 0.0
                ),
                "travel_time_s": float(segment.get("travel_time_s", 0.0) or 0.0),
                "obstacle_aware": bool(segment.get("obstacle_aware", False)),
                "path_planner": str(segment.get("path_planner", "")),
                "planned_shield_program": segment.get("planned_shield_program"),
            }
        )
    return compact


def _build_demo_sources() -> list[PointSource]:
    """Define a small synthetic source set on known room surfaces."""
    return [
        PointSource("Cs-137", position=(5.0, 10.0, 0.0), intensity_cps_1m=50000.0),
        PointSource("Co-60", position=(0.0, 15.0, 7.0), intensity_cps_1m=20000.0),
        PointSource("Eu-154", position=(7.0, 5.0, 0.0), intensity_cps_1m=30000.0),
    ]


def _candidate_axis_points(
    start: float, stop: float, step: float
) -> NDArray[np.float64]:
    """Return evenly spaced axis points within [start, stop] using the given step."""
    if step <= 0:
        raise ValueError("step must be positive.")
    if stop < start:
        return np.zeros(0, dtype=float)
    count = int(np.floor((stop - start) / step)) + 1
    if count <= 0:
        return np.zeros(0, dtype=float)
    return start + step * np.arange(count, dtype=float)


def _build_candidate_sources(
    env: EnvironmentConfig,
    spacing: tuple[float, float, float],
    margin: float,
    position_min: tuple[float, float, float] | None = None,
    position_max: tuple[float, float, float] | None = None,
) -> NDArray[np.float64]:
    """Create a dense 3D grid of candidate sources inside the environment bounds."""
    lo = (
        np.array([0.0, 0.0, 0.0], dtype=float)
        if position_min is None
        else np.asarray(position_min, dtype=float)
    )
    hi = (
        np.array([env.size_x, env.size_y, env.size_z], dtype=float)
        if position_max is None
        else np.asarray(position_max, dtype=float)
    )
    if lo.shape != (3,) or hi.shape != (3,):
        raise ValueError("Candidate source bounds must be 3D vectors.")
    room_lo = np.array([0.0, 0.0, 0.0], dtype=float)
    room_hi = np.array([env.size_x, env.size_y, env.size_z], dtype=float)
    lo = np.maximum(lo, room_lo)
    hi = np.minimum(hi, room_hi)
    if bool(np.any(hi <= lo)):
        raise ValueError("Candidate source bounds are empty.")
    xs = _candidate_axis_points(lo[0] + margin, hi[0] - margin, spacing[0])
    ys = _candidate_axis_points(lo[1] + margin, hi[1] - margin, spacing[1])
    zs = _candidate_axis_points(lo[2] + margin, hi[2] - margin, spacing[2])
    if xs.size == 0 or ys.size == 0 or zs.size == 0:
        raise ValueError("Candidate grid is empty; check spacing and margin values.")
    return np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=float)


def _source_surface_prior_enabled(runtime_config: dict[str, object]) -> bool:
    """Return True when PF source positions should use known surface support."""
    raw = runtime_config.get("source_surface_prior", True)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    return str(raw).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
        "surface",
        "surfaces",
        "surface_constrained",
        "surface-constrained",
    }


def _build_source_candidate_grid(
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    spacing: tuple[float, float, float],
    margin: float,
    position_min: tuple[float, float, float],
    position_max: tuple[float, float, float],
    *,
    source_surface_prior: bool,
    obstacle_height_m: float,
) -> NDArray[np.float64]:
    """Create birth candidates using either volume or known-surface support."""
    if source_surface_prior:
        return build_surface_candidate_sources(
            env,
            obstacle_grid,
            spacing,
            position_min=position_min,
            position_max=position_max,
            obstacle_height_m=obstacle_height_m,
        )
    return _build_candidate_sources(
        env,
        spacing=spacing,
        margin=margin,
        position_min=position_min,
        position_max=position_max,
    )


def _resolve_source_position_bounds(
    env: EnvironmentConfig,
    runtime_config: dict[str, object],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Resolve the PF source-position support from runtime config and room bounds."""
    lo = np.array([0.0, 0.0, 0.0], dtype=float)
    hi = np.array([env.size_x, env.size_y, env.size_z], dtype=float)
    raw_min = runtime_config.get("source_position_min")
    raw_max = runtime_config.get("source_position_max")
    if raw_min is not None:
        arr = np.asarray(raw_min, dtype=float)
        if arr.shape != (3,):
            raise ValueError("source_position_min must be a 3-element vector.")
        lo = arr
    if raw_max is not None:
        arr = np.asarray(raw_max, dtype=float)
        if arr.shape != (3,):
            raise ValueError("source_position_max must be a 3-element vector.")
        hi = arr
    if "source_z_min_m" in runtime_config:
        lo[2] = float(runtime_config["source_z_min_m"])
    if "source_z_max_m" in runtime_config:
        hi[2] = float(runtime_config["source_z_max_m"])
    room_lo = np.array([0.0, 0.0, 0.0], dtype=float)
    room_hi = np.array([env.size_x, env.size_y, env.size_z], dtype=float)
    lo = np.maximum(lo, room_lo)
    hi = np.minimum(hi, room_hi)
    if bool(np.any(hi <= lo)):
        raise ValueError("Resolved source-position support is empty.")
    return tuple(float(v) for v in lo), tuple(float(v) for v in hi)


def _initial_particle_nearby_probability(
    num_particles: int,
    position_min: tuple[float, float, float],
    position_max: tuple[float, float, float],
    radius_m: float,
    init_num_sources: tuple[int, int],
) -> float:
    """
    Return the probability that at least one initial source lies within radius_m of a target.

    Assumes each source position is uniformly sampled within the bounding box.
    """
    if radius_m <= 0.0 or num_particles <= 0:
        return 0.0
    bounds_lo = np.asarray(position_min, dtype=float)
    bounds_hi = np.asarray(position_max, dtype=float)
    span = bounds_hi - bounds_lo
    volume = float(np.prod(span)) if np.all(span > 0.0) else 0.0
    if volume <= 0.0:
        return 0.0
    p = (4.0 / 3.0) * np.pi * (radius_m**3) / volume
    p = float(np.clip(p, 0.0, 1.0))
    r_min, r_max = init_num_sources
    r_min, r_max = (
        (int(r_min), int(r_max)) if r_min <= r_max else (int(r_max), int(r_min))
    )
    per_particle = sum((1.0 - p) ** r for r in range(r_min, r_max + 1)) / (
        r_max - r_min + 1
    )
    return float(1.0 - per_particle**num_particles)


def load_sources_from_json(path: Path) -> list[PointSource]:
    """Load point sources from a JSON configuration file."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        entries = data.get("sources", [])
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError("Source config must be a list or include a 'sources' list.")
    if not isinstance(entries, list):
        raise ValueError("Source config 'sources' must be a list.")
    sources: list[PointSource] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Source entry {idx} must be an object.")
        isotope = entry.get("isotope")
        position = entry.get("position")
        intensity = entry.get("intensity_cps_1m")
        if intensity is None:
            intensity = entry.get("strength_cps_1m")
        if intensity is None:
            intensity = entry.get("intensity")
        if isotope is None or position is None or intensity is None:
            raise ValueError(
                "Each source must include 'isotope', 'position', and 'intensity_cps_1m'."
            )
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            raise ValueError(f"Source entry {idx} position must be a 3-element list.")
        sources.append(
            PointSource(
                isotope=str(isotope),
                position=(float(position[0]), float(position[1]), float(position[2])),
                intensity_cps_1m=float(intensity),
            )
        )
    return sources


def _resolve_config_relative_path(
    path_value: object,
    config_path: str | None,
) -> Path | None:
    """Resolve a config path value relative to its JSON file."""
    if path_value in (None, ""):
        return None
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    if config_path:
        base_path = Path(config_path).expanduser()
        if not base_path.is_absolute():
            base_path = (ROOT / base_path).resolve()
        return (base_path.parent / path).resolve()
    return (ROOT / path).resolve()


def _update_detection_hysteresis(
    candidates: set[str],
    detect_counts: dict[str, int],
    miss_counts: dict[str, int],
    active_isotopes: set[str],
    consecutive: int,
    miss_consecutive: int | None = None,
    consecutive_by_isotope: dict[str, int] | None = None,
) -> set[str]:
    """
    Update detection state with consecutive hit/miss hysteresis.

    Isotopes are activated after `consecutive` hits and deactivated after
    `miss_consecutive` misses (defaults to `consecutive`).
    """
    updated = set(active_isotopes)
    miss_required = consecutive if miss_consecutive is None else miss_consecutive
    for iso in detect_counts:
        hit_required = consecutive
        if consecutive_by_isotope and iso in consecutive_by_isotope:
            hit_required = int(consecutive_by_isotope[iso])
        if iso in candidates:
            detect_counts[iso] += 1
            miss_counts[iso] = 0
        else:
            miss_counts[iso] += 1
            detect_counts[iso] = 0
        if detect_counts[iso] >= hit_required:
            updated.add(iso)
        if miss_counts[iso] >= miss_required:
            updated.discard(iso)
    return updated


def _detect_isotopes_from_counts(
    counts: dict[str, float],
    detect_threshold_abs: float,
    detect_threshold_rel: float,
    detect_threshold_rel_by_isotope: dict[str, float] | None,
) -> set[str]:
    """Return isotopes detected from counts using absolute/relative thresholds."""
    max_c = max(counts.values()) if counts else 0.0
    detected: set[str] = set()
    rel_by_iso = detect_threshold_rel_by_isotope or {}
    for iso, val in counts.items():
        rel_thresh = float(rel_by_iso.get(iso, detect_threshold_rel))
        if val >= detect_threshold_abs and (max_c <= 0.0 or val / max_c >= rel_thresh):
            detected.add(iso)
    return detected


def _pf_posterior_background_counts(
    estimator: RotatingShieldPFEstimator,
    isotope: str,
    live_times_s: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return background counts from the current PF state without refitting."""
    background_rate: float | None = None
    filters = getattr(estimator, "filters", {})
    filt = filters.get(str(isotope)) if isinstance(filters, dict) else None
    if filt is not None and getattr(filt, "continuous_particles", None):
        try:
            background_rate = float(filt.best_particle().state.background)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            background_rate = None
    if background_rate is None:
        pf_config = getattr(estimator, "pf_config", None)
        configured = getattr(pf_config, "background_level", 0.0)
        if isinstance(configured, Mapping):
            background_rate = float(configured.get(str(isotope), 0.0))
        else:
            background_rate = float(configured)
    live_times = np.maximum(
        np.asarray(live_times_s, dtype=float).reshape(-1),
        0.0,
    )
    return live_times * max(float(background_rate), 0.0)


def _final_isotope_count_residual_diagnostics(
    estimator: RotatingShieldPFEstimator,
    estimates: Mapping[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
) -> dict[str, dict[str, float | int | bool | str]]:
    """Return isotope-wise final count prediction and residual diagnostics."""
    diagnostics: dict[str, dict[str, float | int | bool | str]] = {}
    configured_isotopes = estimator.configured_isotope_order()
    active_isotopes = set(str(isotope) for isotope in estimator.filters)
    for isotope in configured_isotopes:
        scored_records = [
            record for record in estimator.measurements if isotope in record.z_k
        ]
        data = estimator._measurement_data_for_iso(
            isotope,
            window=None,
            records=scored_records,
        )
        common_metadata: dict[str, float | int | bool | str] = {
            "configured_isotope": True,
            "active_pf_isotope_at_evaluation": isotope in active_isotopes,
            "recorded_measurement_count": int(len(scored_records)),
            "missing_measurement_count": int(
                len(estimator.measurements) - len(scored_records)
            ),
            "prediction_scope": "current_pf_posterior_projection",
        }
        if data is None or data.z_k.size == 0:
            estimate = estimates.get(isotope, (np.zeros((0, 3)), np.zeros(0)))
            diagnostics[str(isotope)] = {
                **common_metadata,
                "measurement_count": 0,
                "observed_total_counts": 0.0,
                "predicted_total_counts": 0.0,
                "positive_residual_total_counts": 0.0,
                "negative_residual_total_counts": 0.0,
                "diagonal_unfolding_variance_chi2": 0.0,
                "residual_chi2": 0.0,
                "reported_source_count": int(
                    np.asarray(estimate[0], dtype=float).reshape(-1, 3).shape[0]
                ),
            }
            continue
        positions, strengths = estimates.get(
            isotope,
            (np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)),
        )
        pos_arr = np.asarray(positions, dtype=float).reshape(-1, 3)
        q_arr = np.asarray(strengths, dtype=float).reshape(-1)
        source_count = min(pos_arr.shape[0], q_arr.size)
        pos_arr = pos_arr[:source_count]
        q_arr = np.maximum(q_arr[:source_count], 0.0)
        background = _pf_posterior_background_counts(
            estimator,
            isotope,
            data.live_times,
        )
        if source_count:
            source_counts = estimator.configured_isotope_response_counts(
                isotope=isotope,
                data=data,
                source_positions=pos_arr,
                strengths=q_arr,
            )
            predicted = np.asarray(background, dtype=float) + np.sum(
                np.maximum(np.asarray(source_counts, dtype=float), 0.0),
                axis=1,
            )
        else:
            predicted = np.asarray(background, dtype=float)
        observed = np.maximum(np.asarray(data.z_k, dtype=float).reshape(-1), 0.0)
        residual = observed - np.asarray(predicted, dtype=float).reshape(-1)
        variances = np.maximum(
            np.asarray(data.observation_variances, dtype=float).reshape(-1),
            1.0,
        )
        diagonal_chi2 = float(np.sum((residual * residual) / variances))
        diagnostics[str(isotope)] = {
            **common_metadata,
            "measurement_count": int(observed.size),
            "reported_source_count": int(source_count),
            "observed_total_counts": float(np.sum(observed)),
            "observed_max_counts": float(np.max(observed)) if observed.size else 0.0,
            "predicted_total_counts": float(np.sum(predicted)),
            "background_total_counts": float(np.sum(background)),
            "positive_residual_total_counts": float(np.sum(np.maximum(residual, 0.0))),
            "negative_residual_total_counts": float(np.sum(np.maximum(-residual, 0.0))),
            "diagonal_unfolding_variance_chi2": diagonal_chi2,
            "residual_chi2": diagonal_chi2,
            "residual_chi2_definition": (
                "backward_compatible_alias_of_diagonal_unfolding_variance_chi2"
            ),
            "residual_snr": float(
                np.sum(residual) / np.sqrt(max(float(np.sum(variances)), 1.0e-12))
            ),
            "source_strength_total": float(np.sum(q_arr)),
            "source_strength_max": float(np.max(q_arr)) if q_arr.size else 0.0,
        }
    return diagnostics


def _final_count_bias_diagnostics(
    estimator: RotatingShieldPFEstimator,
    estimates: Mapping[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    *,
    count_regime_lower_edges: Sequence[float],
) -> dict[str, Any]:
    """Return count-model bias grouped by isotope, shield pair, and count regime."""
    observed_blocks: list[NDArray[np.float64]] = []
    predicted_blocks: list[NDArray[np.float64]] = []
    isotope_blocks: list[NDArray[np.str_]] = []
    fe_blocks: list[NDArray[np.int64]] = []
    pb_blocks: list[NDArray[np.int64]] = []
    configured_isotopes = estimator.configured_isotope_order()
    active_isotopes = set(str(isotope) for isotope in estimator.filters)
    missing_measurements_by_isotope: dict[str, int] = {}
    scored_measurements_by_isotope: dict[str, int] = {}
    for isotope in configured_isotopes:
        scored_records = [
            record for record in estimator.measurements if isotope in record.z_k
        ]
        missing_measurements_by_isotope[isotope] = int(
            len(estimator.measurements) - len(scored_records)
        )
        scored_measurements_by_isotope[isotope] = int(len(scored_records))
        data = estimator._measurement_data_for_iso(
            isotope,
            window=None,
            records=scored_records,
        )
        if data is None or data.z_k.size == 0:
            continue
        positions, strengths = estimates.get(
            isotope,
            (np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)),
        )
        pos_arr = np.asarray(positions, dtype=float).reshape(-1, 3)
        q_arr = np.maximum(np.asarray(strengths, dtype=float).reshape(-1), 0.0)
        source_count = min(pos_arr.shape[0], q_arr.size)
        pos_arr = pos_arr[:source_count]
        q_arr = q_arr[:source_count]
        background = _pf_posterior_background_counts(
            estimator,
            isotope,
            data.live_times,
        )
        if source_count:
            source_counts = estimator.configured_isotope_response_counts(
                isotope=isotope,
                data=data,
                source_positions=pos_arr,
                strengths=q_arr,
            )
            predicted = np.asarray(background, dtype=float) + np.sum(
                np.maximum(np.asarray(source_counts, dtype=float), 0.0),
                axis=1,
            )
        else:
            predicted = np.asarray(background, dtype=float)
        observed = np.maximum(np.asarray(data.z_k, dtype=float).reshape(-1), 0.0)
        observed_blocks.append(observed)
        predicted_blocks.append(np.asarray(predicted, dtype=float).reshape(-1))
        isotope_blocks.append(
            np.repeat(np.asarray([str(isotope)], dtype=str), observed.size)
        )
        fe_blocks.append(np.asarray(data.fe_indices, dtype=np.int64).reshape(-1))
        pb_blocks.append(np.asarray(data.pb_indices, dtype=np.int64).reshape(-1))
    observed_all = (
        np.concatenate(observed_blocks) if observed_blocks else np.zeros(0, dtype=float)
    )
    predicted_all = (
        np.concatenate(predicted_blocks)
        if predicted_blocks
        else np.zeros(0, dtype=float)
    )
    isotope_all = (
        np.concatenate(isotope_blocks) if isotope_blocks else np.zeros(0, dtype=str)
    )
    fe_all = np.concatenate(fe_blocks) if fe_blocks else np.zeros(0, dtype=np.int64)
    pb_all = np.concatenate(pb_blocks) if pb_blocks else np.zeros(0, dtype=np.int64)
    summary = summarize_count_bias(
        observed_all,
        predicted_all,
        isotope_all,
        fe_all,
        pb_all,
        num_orientations=int(estimator.num_orientations),
        count_regime_lower_edges=count_regime_lower_edges,
    )
    summary.update(
        {
            "configured_isotopes": list(configured_isotopes),
            "active_pf_isotopes_at_evaluation": sorted(active_isotopes),
            "inactive_configured_isotopes_scored": [
                isotope
                for isotope in configured_isotopes
                if isotope not in active_isotopes
                and scored_measurements_by_isotope[isotope] > 0
            ],
            "scored_measurements_by_isotope": scored_measurements_by_isotope,
            "missing_measurements_by_isotope": missing_measurements_by_isotope,
            "all_configured_isotopes_fully_recorded": all(
                count == 0 for count in missing_measurements_by_isotope.values()
            ),
        }
    )
    return summary


def _estimate_map_to_metric_sources(
    estimates: Mapping[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
) -> dict[str, list[dict[str, float | list[float]]]]:
    """Convert an estimator estimate map into compute_metrics source records."""
    payload: dict[str, list[dict[str, float | list[float]]]] = {}
    for isotope, (positions, strengths) in estimates.items():
        pos_arr = np.asarray(positions, dtype=float).reshape(-1, 3)
        q_arr = np.asarray(strengths, dtype=float).reshape(-1)
        count = min(pos_arr.shape[0], q_arr.size)
        payload[str(isotope)] = [
            {
                "pos": [float(value) for value in pos_arr[idx]],
                "strength": float(q_arr[idx]),
            }
            for idx in range(count)
        ]
    return payload


def _online_estimate_metric_summary(
    history_estimates: Sequence[
        Mapping[str, tuple[NDArray[np.float64], NDArray[np.float64]]]
    ],
    gt_by_iso: Mapping[str, list[dict[str, float | list[float]]]],
    *,
    match_radius_m: float,
) -> dict[str, dict[str, float | int | None]]:
    """Summarize online source-count and localization stability over PF history."""
    summaries: dict[str, dict[str, float | int | None]] = {}
    isotopes = sorted(
        set(str(name) for name in gt_by_iso)
        | {
            str(iso)
            for estimate_map in history_estimates
            for iso in estimate_map.keys()
        }
    )
    if not history_estimates:
        return {
            isotope: {
                "history_length": 0,
                "first_correct_cardinality_step": None,
                "last_correct_cardinality_step": None,
                "max_consecutive_wrong_cardinality": 0,
                "final_online_source_count_error": None,
            }
            for isotope in isotopes
        }
    cardinality_errors: dict[str, list[int]] = {isotope: [] for isotope in isotopes}
    mean_errors: dict[str, list[float | None]] = {isotope: [] for isotope in isotopes}
    for estimate_map in history_estimates:
        metrics = compute_metrics(
            dict(gt_by_iso),
            _estimate_map_to_metric_sources(estimate_map),
            match_radius_m=float(match_radius_m),
        )
        isotope_metrics = metrics.get("isotopes", {})
        for isotope in isotopes:
            data = isotope_metrics.get(isotope, {})
            counts = data.get("counts", {}) if isinstance(data, dict) else {}
            pos_summary = (
                data.get("position_error", {}) if isinstance(data, dict) else {}
            )
            cardinality_errors[isotope].append(int(counts.get("source_count_error", 0)))
            mean_errors[isotope].append(pos_summary.get("mean"))
    for isotope in isotopes:
        errors = cardinality_errors[isotope]
        correct_steps = [idx for idx, value in enumerate(errors) if int(value) == 0]
        max_wrong = 0
        current_wrong = 0
        for value in errors:
            if int(value) == 0:
                current_wrong = 0
            else:
                current_wrong += 1
                max_wrong = max(max_wrong, current_wrong)
        finite_mean_errors = [
            float(value)
            for value in mean_errors[isotope]
            if value is not None and np.isfinite(float(value))
        ]
        summaries[isotope] = {
            "history_length": int(len(errors)),
            "first_correct_cardinality_step": (
                int(correct_steps[0]) if correct_steps else None
            ),
            "last_correct_cardinality_step": (
                int(correct_steps[-1]) if correct_steps else None
            ),
            "max_consecutive_wrong_cardinality": int(max_wrong),
            "final_online_source_count_error": int(errors[-1]) if errors else None,
            "mean_online_position_error_m": (
                float(np.mean(finite_mean_errors)) if finite_mean_errors else None
            ),
            "final_online_position_error_m": (
                float(finite_mean_errors[-1]) if finite_mean_errors else None
            ),
        }
    return summaries


def _remaining_measurement_trace_summary(
    estimates: Sequence[Mapping[str, Any]],
) -> dict[str, float | int | list[str]]:
    """Summarize remaining-measurement trace health for final JSON output."""
    if not estimates:
        return {
            "trace_length": 0,
            "residual_budget_auc": 0.0,
            "max_estimated_remaining_stations": 0,
            "last_unresolved_factors": [],
        }
    residual_auc = 0.0
    max_remaining = 0
    for payload_raw in estimates:
        payload = dict(payload_raw)
        components = payload.get("components", {})
        if isinstance(components, Mapping):
            residual_auc += float(components.get("residual", 0.0))
        max_remaining = max(
            max_remaining,
            int(payload.get("estimated_remaining_stations", 0) or 0),
        )
    last = dict(estimates[-1])
    unresolved = last.get("unresolved_factors", [])
    if not isinstance(unresolved, Sequence) or isinstance(unresolved, (str, bytes)):
        unresolved_values: list[str] = []
    else:
        unresolved_values = [str(value) for value in unresolved]
    return {
        "trace_length": int(len(estimates)),
        "residual_budget_auc": float(residual_auc),
        "max_estimated_remaining_stations": int(max_remaining),
        "last_unresolved_factors": unresolved_values,
    }


def _build_isotope_colors(isotopes: list[str]) -> dict[str, str]:
    """Return a consistent color mapping for isotope-specific plots."""
    cmap = plt.get_cmap("tab10")
    colors: dict[str, str] = {}
    for i, iso in enumerate(isotopes):
        if iso in DEFAULT_ISOTOPE_COLORS:
            colors[iso] = DEFAULT_ISOTOPE_COLORS[iso]
        else:
            colors[iso] = cmap(i % 10)
    return colors


def _fmt_pos(pos: NDArray[np.float64]) -> str:
    """Format a position vector for logging."""
    return np.array2string(
        np.asarray(pos, dtype=float), precision=2, floatmode="fixed", separator=", "
    )


def _fmt_counts(counts: dict[str, float] | None) -> str:
    """Format a count dict for logging."""
    if counts is None:
        return "{}"
    items = ", ".join(f"{iso}: {float(val):.1f}" for iso, val in sorted(counts.items()))
    return "{" + items + "}"


def _thin_spectrum_for_notification(
    energy_keV: NDArray[np.float64],
    counts: NDArray[np.float64],
    max_bins: int,
) -> tuple[list[float], list[float]]:
    """Return spectrum arrays thinned to a notification-friendly size."""
    energy = np.asarray(energy_keV, dtype=float).reshape(-1)
    values = np.asarray(counts, dtype=float).reshape(-1)
    if energy.size != values.size:
        size = min(energy.size, values.size)
        energy = energy[:size]
        values = values[:size]
    limit = int(max_bins)
    if limit > 0 and values.size > limit:
        nonzero = np.flatnonzero(values > 0.0)
        if nonzero.size >= limit:
            ranked = nonzero[np.argsort(values[nonzero])[-limit:]]
            indices = np.sort(ranked)
        else:
            base = np.linspace(0, values.size - 1, limit, dtype=int)
            indices = np.unique(np.concatenate([base, nonzero]))
            if indices.size > limit:
                ranked = indices[np.argsort(values[indices])[-limit:]]
                indices = np.sort(ranked)
        energy = energy[indices]
        values = values[indices]
    return (
        [round(float(value), 3) for value in energy],
        [round(float(value), 6) for value in values],
    )


def _build_spectrum_notification_payload(
    *,
    decomposer: SpectralDecomposer,
    spectrum: NDArray[np.float64],
    step_index: int,
    pose_xyz: NDArray[np.float64],
    fe_index: int,
    pb_index: int,
    live_time_s: float,
    counts_by_isotope: dict[str, float],
    detected_isotopes: set[str],
    count_method: str,
    max_bins: int,
) -> dict[str, object]:
    """Build a compact spectrum payload for piplup/Railway display."""
    spectrum_values = np.asarray(spectrum, dtype=float)
    energy_keV, spectrum_counts = _thin_spectrum_for_notification(
        np.asarray(decomposer.energy_axis, dtype=float),
        spectrum_values,
        max_bins,
    )
    return {
        "step_index": int(step_index),
        "pose_xyz": [float(v) for v in np.asarray(pose_xyz, dtype=float)],
        "fe_index": int(fe_index),
        "pb_index": int(pb_index),
        "live_time_s": float(live_time_s),
        "count_method": str(count_method),
        "counts_by_isotope": {
            iso: float(value) for iso, value in sorted(counts_by_isotope.items())
        },
        "count_variance_by_isotope": {
            iso: float(value)
            for iso, value in sorted(decomposer.last_count_variances.items())
        },
        "detected_isotopes": sorted(detected_isotopes),
        "total_spectrum_counts": float(np.sum(spectrum_values)),
        "max_bin_count": float(np.max(spectrum_values))
        if spectrum_values.size
        else 0.0,
        "energy_keV": energy_keV,
        "spectrum_counts": spectrum_counts,
    }


def _fmt_sources(positions: NDArray[np.float64], strengths: NDArray[np.float64]) -> str:
    """Format a list of source positions/strengths for logging."""
    positions = np.asarray(positions, dtype=float)
    strengths = np.asarray(strengths, dtype=float)
    if positions.size == 0 or strengths.size == 0:
        return "[]"
    count = min(int(positions.shape[0]), int(strengths.size), 8)
    positions = positions[:count]
    strengths = strengths[:count]
    chunks = []
    for pos, strength in zip(positions, strengths):
        pos_str = np.array2string(pos, precision=2, floatmode="fixed", separator=", ")
        chunks.append(f"{pos_str}|{float(strength):.2f}")
    return "[" + ", ".join(chunks) + "]"


def _true_strength_array(
    true_strengths: dict[str, float | Sequence[float]],
    isotope: str,
    count: int,
) -> NDArray[np.float64]:
    """Return a true-strength array for one isotope."""
    values = true_strengths.get(isotope, [])
    if isinstance(values, (int, float, np.floating)):
        return np.full(count, float(values), dtype=float)
    arr = np.asarray(list(values), dtype=float).reshape(-1)
    if arr.size == count:
        return arr
    if arr.size == 1 and count > 1:
        return np.full(count, float(arr[0]), dtype=float)
    if arr.size < count:
        padded = np.full(count, np.nan, dtype=float)
        padded[: arr.size] = arr
        return padded
    return arr[:count]


def _estimate_accuracy_records(
    isotope: str,
    est_positions: NDArray[np.float64],
    est_strengths: NDArray[np.float64],
    true_positions: NDArray[np.float64],
    true_strengths: NDArray[np.float64],
    surface_kinds: NDArray[np.object_],
    *,
    match_radius_m: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Return per-estimate accuracy records and an isotope-level summary."""
    est_pos = np.asarray(est_positions, dtype=float).reshape(-1, 3)
    est_q = np.asarray(est_strengths, dtype=float).reshape(-1)
    true_pos = np.asarray(true_positions, dtype=float).reshape(-1, 3)
    true_q = np.asarray(true_strengths, dtype=float).reshape(-1)
    source_count = min(est_pos.shape[0], est_q.size)
    est_pos = est_pos[:source_count]
    est_q = est_q[:source_count]
    kinds = np.asarray(surface_kinds, dtype=object).reshape(-1)
    if kinds.size < source_count:
        padded = np.full(source_count, "unknown", dtype=object)
        padded[: kinds.size] = kinds
        kinds = padded
    else:
        kinds = kinds[:source_count]

    nearest_indices = np.full(source_count, -1, dtype=int)
    nearest_distances = np.full(source_count, np.nan, dtype=float)
    if true_pos.size and source_count:
        deltas = est_pos[:, None, :] - true_pos[None, :, :]
        distances = np.linalg.norm(deltas, axis=2)
        nearest_indices = np.argmin(distances, axis=1).astype(int, copy=False)
        nearest_distances = distances[np.arange(source_count), nearest_indices]

    records: list[dict[str, object]] = []
    rel_errors: list[float] = []
    matched_truth_indices: list[int] = []
    for idx in range(source_count):
        truth_idx = int(nearest_indices[idx])
        truth_strength = (
            float(true_q[truth_idx])
            if truth_idx >= 0 and truth_idx < true_q.size
            else None
        )
        strength_error = (
            float(est_q[idx] - truth_strength)
            if truth_strength is not None and np.isfinite(truth_strength)
            else None
        )
        strength_rel_error = None
        if (
            strength_error is not None
            and truth_strength is not None
            and abs(float(truth_strength)) > 1.0e-12
        ):
            strength_rel_error = float(strength_error / float(truth_strength))
            rel_errors.append(abs(strength_rel_error))
        position_error = float(nearest_distances[idx])
        within_radius = bool(
            np.isfinite(position_error) and position_error <= float(match_radius_m)
        )
        if truth_idx >= 0 and within_radius:
            matched_truth_indices.append(truth_idx)
        record: dict[str, object] = {
            "isotope": isotope,
            "estimate_index": int(idx),
            "pos": [float(v) for v in est_pos[idx]],
            "strength": float(est_q[idx]),
            "surface_kind": str(kinds[idx])
            if kinds[idx] is not None
            else "off_surface",
            "nearest_truth_index": truth_idx if truth_idx >= 0 else None,
            "position_error_m": position_error if np.isfinite(position_error) else None,
            "within_match_radius": within_radius,
            "nearest_truth_strength": truth_strength,
            "strength_error": strength_error,
            "strength_rel_error": strength_rel_error,
        }
        if truth_idx >= 0 and truth_idx < true_pos.shape[0]:
            record["nearest_truth_pos"] = [float(v) for v in true_pos[truth_idx]]
        else:
            record["nearest_truth_pos"] = None
        records.append(record)

    match_counts = Counter(matched_truth_indices)
    duplicate_truth_matches = sum(max(0, count - 1) for count in match_counts.values())
    unmatched_truth_count = max(0, int(true_pos.shape[0]) - len(match_counts))
    valid_distances = nearest_distances[np.isfinite(nearest_distances)]
    total_truth_strength = float(np.nansum(true_q)) if true_q.size else 0.0
    total_est_strength = float(np.sum(est_q)) if est_q.size else 0.0
    total_strength_rel_error = (
        float((total_est_strength - total_truth_strength) / total_truth_strength)
        if abs(total_truth_strength) > 1.0e-12
        else None
    )
    surface_counts = Counter(
        str(kind) if kind is not None else "off_surface" for kind in kinds
    )
    summary: dict[str, object] = {
        "estimate_count": int(source_count),
        "truth_count": int(true_pos.shape[0]),
        "source_count_error": int(source_count - true_pos.shape[0]),
        "matched_truth_count": int(len(match_counts)),
        "unmatched_truth_count": int(unmatched_truth_count),
        "duplicate_truth_matches": int(duplicate_truth_matches),
        "mean_position_error_m": (
            float(np.mean(valid_distances)) if valid_distances.size else None
        ),
        "max_position_error_m": (
            float(np.max(valid_distances)) if valid_distances.size else None
        ),
        "mean_abs_strength_rel_error": (
            float(np.mean(rel_errors)) if rel_errors else None
        ),
        "total_est_strength": total_est_strength,
        "total_truth_strength": total_truth_strength,
        "total_strength_rel_error": total_strength_rel_error,
        "surface_counts": dict(sorted(surface_counts.items())),
        "off_surface_count": int(surface_counts.get("off_surface", 0)),
    }
    return records, summary


def _truth_coverage_records(
    isotope: str,
    est_positions: NDArray[np.float64],
    est_strengths: NDArray[np.float64],
    true_positions: NDArray[np.float64],
    true_strengths: NDArray[np.float64],
    *,
    match_radius_m: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Return truth-centric nearest-estimate coverage diagnostics."""
    est_pos = np.asarray(est_positions, dtype=float).reshape(-1, 3)
    est_q = np.asarray(est_strengths, dtype=float).reshape(-1)
    true_pos = np.asarray(true_positions, dtype=float).reshape(-1, 3)
    true_q = np.asarray(true_strengths, dtype=float).reshape(-1)
    est_count = min(est_pos.shape[0], est_q.size)
    truth_count = min(true_pos.shape[0], true_q.size)
    est_pos = est_pos[:est_count]
    est_q = est_q[:est_count]
    true_pos = true_pos[:truth_count]
    true_q = true_q[:truth_count]

    nearest_indices = np.full(truth_count, -1, dtype=int)
    nearest_distances = np.full(truth_count, np.nan, dtype=float)
    if truth_count and est_count:
        deltas = true_pos[:, None, :] - est_pos[None, :, :]
        distances = np.linalg.norm(deltas, axis=2)
        nearest_indices = np.argmin(distances, axis=1).astype(int, copy=False)
        nearest_distances = distances[np.arange(truth_count), nearest_indices]

    records: list[dict[str, object]] = []
    covered = 0
    rel_errors: list[float] = []
    for truth_idx in range(truth_count):
        est_idx = int(nearest_indices[truth_idx])
        distance = float(nearest_distances[truth_idx])
        covered_here = bool(np.isfinite(distance) and distance <= float(match_radius_m))
        if covered_here:
            covered += 1
        nearest_strength = (
            float(est_q[est_idx]) if est_idx >= 0 and est_idx < est_q.size else None
        )
        strength_error = (
            float(nearest_strength - true_q[truth_idx])
            if nearest_strength is not None and np.isfinite(true_q[truth_idx])
            else None
        )
        strength_rel_error = None
        if strength_error is not None and abs(float(true_q[truth_idx])) > 1.0e-12:
            strength_rel_error = float(strength_error / float(true_q[truth_idx]))
            rel_errors.append(abs(strength_rel_error))
        records.append(
            {
                "isotope": isotope,
                "truth_index": int(truth_idx),
                "truth_pos": [float(v) for v in true_pos[truth_idx]],
                "truth_strength": float(true_q[truth_idx]),
                "nearest_estimate_index": est_idx if est_idx >= 0 else None,
                "nearest_estimate_pos": (
                    [float(v) for v in est_pos[est_idx]]
                    if est_idx >= 0 and est_idx < est_pos.shape[0]
                    else None
                ),
                "nearest_estimate_strength": nearest_strength,
                "nearest_estimate_distance_m": (
                    distance if np.isfinite(distance) else None
                ),
                "covered": covered_here,
                "nearest_strength_error": strength_error,
                "nearest_strength_rel_error": strength_rel_error,
            }
        )

    valid_distances = nearest_distances[np.isfinite(nearest_distances)]
    summary: dict[str, object] = {
        "truth_covered_count": int(covered),
        "truth_uncovered_count": int(max(0, truth_count - covered)),
        "mean_truth_nearest_estimate_error_m": (
            float(np.mean(valid_distances)) if valid_distances.size else None
        ),
        "max_truth_nearest_estimate_error_m": (
            float(np.max(valid_distances)) if valid_distances.size else None
        ),
        "mean_truth_nearest_strength_rel_error": (
            float(np.mean(rel_errors)) if rel_errors else None
        ),
    }
    return records, summary


def _frame_field(
    frame: PFFrame | dict[str, object], name: str, default: object
) -> object:
    """Return a PFFrame field from either a dataclass frame or test stub dict."""
    if isinstance(frame, dict):
        return frame.get(name, default)
    return getattr(frame, name, default)


def _current_map_estimate_trace_frame(
    estimator: RotatingShieldPFEstimator,
    isotopes: Sequence[str],
    frame: PFFrame | dict[str, object],
    *,
    step_index: int,
    elapsed_s: float,
    counts_by_isotope: dict[str, float],
    estimate_source: str = "current_map",
) -> dict[str, object]:
    """Return a lightweight trace frame from current MAP particles."""
    estimated_sources: dict[str, NDArray[np.float64]] = {}
    estimated_strengths: dict[str, NDArray[np.float64]] = {}
    estimated_metadata: dict[str, list[dict[str, object]]] = {}
    for isotope in sorted(set(str(name) for name in isotopes) | set(estimator.filters)):
        empty_pos = np.zeros((0, 3), dtype=float)
        empty_q = np.zeros(0, dtype=float)
        filt = estimator.filters.get(isotope)
        if filt is None or not getattr(filt, "continuous_particles", None):
            estimated_sources[isotope] = empty_pos
            estimated_strengths[isotope] = empty_q
            estimated_metadata[isotope] = []
            continue
        try:
            state = filt.best_particle().state
            source_count = max(0, int(getattr(state, "num_sources", 0)))
            positions = np.asarray(getattr(state, "positions"), dtype=float).reshape(
                -1, 3
            )
            strengths = np.asarray(getattr(state, "strengths"), dtype=float).reshape(-1)
            source_count = min(source_count, positions.shape[0], strengths.size)
            estimated_sources[isotope] = positions[:source_count].copy()
            estimated_strengths[isotope] = strengths[:source_count].copy()
            ages_raw = getattr(state, "ages", None)
            support_raw = getattr(state, "support_scores", None)
            tentative_raw = getattr(state, "tentative_sources", None)
            fail_raw = getattr(state, "verification_fail_streaks", None)
            ages = np.asarray(
                ages_raw if ages_raw is not None else np.zeros(0, dtype=int),
                dtype=int,
            ).reshape(-1)
            support = np.asarray(
                support_raw if support_raw is not None else np.zeros(0, dtype=float),
                dtype=float,
            ).reshape(-1)
            tentative = np.asarray(
                tentative_raw if tentative_raw is not None else np.zeros(0, dtype=bool),
                dtype=bool,
            ).reshape(-1)
            fail_streak = np.asarray(
                fail_raw if fail_raw is not None else np.zeros(0, dtype=int),
                dtype=int,
            ).reshape(-1)
            metadata: list[dict[str, object]] = []
            for idx in range(source_count):
                metadata.append(
                    {
                        "age": int(ages[idx]) if idx < ages.size else None,
                        "support_score": (
                            float(support[idx]) if idx < support.size else None
                        ),
                        "tentative": bool(tentative[idx])
                        if idx < tentative.size
                        else None,
                        "verification_fail_streak": int(fail_streak[idx])
                        if idx < fail_streak.size
                        else None,
                    }
                )
            estimated_metadata[isotope] = metadata
        except (AttributeError, IndexError, RuntimeError, ValueError):
            estimated_sources[isotope] = empty_pos
            estimated_strengths[isotope] = empty_q
            estimated_metadata[isotope] = []
    robot_position = np.asarray(
        _frame_field(frame, "robot_position", np.zeros(3)),
        dtype=float,
    ).reshape(-1)
    if robot_position.size < 3:
        robot_position = np.pad(robot_position, (0, 3 - robot_position.size))
    return {
        "estimate_source": str(estimate_source),
        "step_index": int(step_index),
        "time": float(elapsed_s),
        "robot_position": robot_position[:3].copy(),
        "counts_by_isotope": dict(counts_by_isotope),
        "estimated_sources": estimated_sources,
        "estimated_strengths": estimated_strengths,
        "estimated_metadata": estimated_metadata,
    }


def _build_intermediate_estimate_trace_payload(
    frame: PFFrame | dict[str, object],
    true_sources: dict[str, NDArray[np.float64]],
    true_strengths: dict[str, float | Sequence[float]],
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    *,
    obstacle_height_m: float,
    match_radius_m: float,
) -> dict[str, object]:
    """Build a JSON-serializable intermediate estimate accuracy payload."""
    payload_records: list[dict[str, object]] = []
    truth_records_payload: list[dict[str, object]] = []
    summaries: dict[str, dict[str, object]] = {}
    estimated_sources_raw = _frame_field(frame, "estimated_sources", {})
    estimated_strengths_raw = _frame_field(frame, "estimated_strengths", {})
    estimated_metadata_raw = _frame_field(frame, "estimated_metadata", {})
    estimated_sources = (
        dict(estimated_sources_raw) if isinstance(estimated_sources_raw, dict) else {}
    )
    estimated_strengths = (
        dict(estimated_strengths_raw)
        if isinstance(estimated_strengths_raw, dict)
        else {}
    )
    estimated_metadata = (
        dict(estimated_metadata_raw) if isinstance(estimated_metadata_raw, dict) else {}
    )
    isotope_names = sorted(
        set(estimated_sources) | set(estimated_strengths) | set(true_sources)
    )
    for isotope in isotope_names:
        est_positions = np.asarray(
            estimated_sources.get(isotope, np.zeros((0, 3))),
            dtype=float,
        ).reshape(-1, 3)
        est_strengths = np.asarray(
            estimated_strengths.get(isotope, np.zeros(0)),
            dtype=float,
        ).reshape(-1)
        true_positions = np.asarray(
            true_sources.get(isotope, np.zeros((0, 3))),
            dtype=float,
        ).reshape(-1, 3)
        true_strength_arr = _true_strength_array(
            true_strengths,
            isotope,
            true_positions.shape[0],
        )
        surface_kinds = source_surface_kinds(
            est_positions,
            env,
            obstacle_grid,
            obstacle_height_m=obstacle_height_m,
            tolerance_m=1.0e-5,
        )
        records, summary = _estimate_accuracy_records(
            isotope,
            est_positions,
            est_strengths,
            true_positions,
            true_strength_arr,
            surface_kinds,
            match_radius_m=match_radius_m,
        )
        metadata_records = estimated_metadata.get(isotope, [])
        if isinstance(metadata_records, Sequence):
            for record in records:
                idx = int(record.get("estimate_index", -1))
                if idx < 0 or idx >= len(metadata_records):
                    continue
                metadata = metadata_records[idx]
                if not isinstance(metadata, dict):
                    continue
                for key in (
                    "age",
                    "support_score",
                    "tentative",
                    "verification_fail_streak",
                ):
                    if key in metadata:
                        record[key] = metadata[key]
        truth_records, truth_summary = _truth_coverage_records(
            isotope,
            est_positions,
            est_strengths,
            true_positions,
            true_strength_arr,
            match_radius_m=match_radius_m,
        )
        summary.update(truth_summary)
        payload_records.extend(records)
        truth_records_payload.extend(truth_records)
        summaries[isotope] = summary
    counts_raw = _frame_field(frame, "counts_by_isotope", {})
    counts_by_isotope = dict(counts_raw) if isinstance(counts_raw, dict) else {}
    robot_position = np.asarray(
        _frame_field(frame, "robot_position", np.zeros(3)),
        dtype=float,
    ).reshape(-1)
    if robot_position.size < 3:
        robot_position = np.pad(robot_position, (0, 3 - robot_position.size))
    return {
        "estimate_source": str(_frame_field(frame, "estimate_source", "frame")),
        "step_index": int(_frame_field(frame, "step_index", -1)),
        "time_s": float(_frame_field(frame, "time", 0.0)),
        "robot_position": [float(v) for v in robot_position[:3]],
        "counts_by_isotope": {
            str(key): float(value) for key, value in sorted(counts_by_isotope.items())
        },
        "isotopes": summaries,
        "estimates": payload_records,
        "truth_sources": truth_records_payload,
    }


def _format_estimate_trace_log_line(
    step_index: int,
    isotope: str,
    summary: dict[str, object],
    records: list[dict[str, object]],
    *,
    max_records: int = 6,
) -> str:
    """Format one intermediate estimate trace summary line for the console log."""
    mean_err = summary.get("mean_position_error_m")
    max_err = summary.get("max_position_error_m")
    total_rel = summary.get("total_strength_rel_error")
    mean_q_rel = summary.get("mean_abs_strength_rel_error")
    estimate_source = str(summary.get("estimate_source", "frame"))
    chunks: list[str] = []
    for record in records[: max(0, int(max_records))]:
        pos = np.asarray(record.get("pos", np.zeros(3)), dtype=float)
        truth_idx = record.get("nearest_truth_index")
        dist = record.get("position_error_m")
        q_rel = record.get("strength_rel_error")
        distance_text = f"{float(dist):.2f}m" if dist is not None else "NA"
        chunks.append(
            "#"
            f"{int(record.get('estimate_index', 0))}"
            f" pos={_fmt_pos(pos)}"
            f" q={float(record.get('strength', 0.0)):.1f}"
            f" surface={record.get('surface_kind')}"
            f" nn={truth_idx if truth_idx is not None else 'NA'}"
            f" d={distance_text}"
        )
        if q_rel is not None:
            chunks[-1] += f" dq_rel={float(q_rel):+.2f}"
        if record.get("age") is not None:
            chunks[-1] += f" age={int(record.get('age', 0))}"
        if record.get("tentative") is not None:
            chunks[-1] += f" tent={bool(record.get('tentative'))}"
        if record.get("verification_fail_streak") is not None:
            chunks[-1] += f" fail={int(record.get('verification_fail_streak', 0))}"
        if record.get("support_score") is not None:
            chunks[-1] += f" support={float(record.get('support_score', 0.0)):.2f}"
    return (
        f"[step {step_index}] pf_estimates[{isotope}] "
        f"mode={estimate_source} "
        f"n={int(summary.get('estimate_count', 0))}/{int(summary.get('truth_count', 0))} "
        f"source_count_error={int(summary.get('source_count_error', 0))} "
        f"matched_truth={int(summary.get('matched_truth_count', 0))} "
        f"unmatched_truth={int(summary.get('unmatched_truth_count', 0))} "
        f"duplicate_truth={int(summary.get('duplicate_truth_matches', 0))} "
        f"mean_d={_fmt_optional_float(mean_err)}m "
        f"max_d={_fmt_optional_float(max_err)}m "
        f"total_q={float(summary.get('total_est_strength', 0.0)):.1f}/"
        f"{float(summary.get('total_truth_strength', 0.0)):.1f} "
        f"total_q_rel={_fmt_optional_float(total_rel)} "
        f"mean_abs_q_rel={_fmt_optional_float(mean_q_rel)} "
        f"off_surface={int(summary.get('off_surface_count', 0))} "
        f"entries=[{'; '.join(chunks)}]"
    )


def _format_truth_coverage_log_line(
    step_index: int,
    isotope: str,
    summary: dict[str, object],
    truth_records: list[dict[str, object]],
    *,
    max_records: int = 6,
) -> str:
    """Format truth-centric coverage diagnostics for the console log."""
    estimate_source = str(summary.get("estimate_source", "frame"))
    mean_err = summary.get("mean_truth_nearest_estimate_error_m")
    max_err = summary.get("max_truth_nearest_estimate_error_m")
    mean_q_rel = summary.get("mean_truth_nearest_strength_rel_error")
    chunks: list[str] = []
    for record in truth_records[: max(0, int(max_records))]:
        pos = np.asarray(record.get("truth_pos", np.zeros(3)), dtype=float)
        est_idx = record.get("nearest_estimate_index")
        dist = record.get("nearest_estimate_distance_m")
        q_est = record.get("nearest_estimate_strength")
        q_rel = record.get("nearest_strength_rel_error")
        distance_text = f"{float(dist):.2f}m" if dist is not None else "NA"
        q_est_text = f"{float(q_est):.1f}" if q_est is not None else "NA"
        chunk = (
            "#"
            f"{int(record.get('truth_index', 0))}"
            f" pos={_fmt_pos(pos)}"
            f" q_true={float(record.get('truth_strength', 0.0)):.1f}"
            f" nearest_est={est_idx if est_idx is not None else 'NA'}"
            f" d={distance_text}"
            f" q_est={q_est_text}"
            f" covered={bool(record.get('covered', False))}"
        )
        if q_rel is not None:
            chunk += f" dq_rel={float(q_rel):+.2f}"
        chunks.append(chunk)
    return (
        f"[step {step_index}] pf_truth_coverage[{isotope}] "
        f"mode={estimate_source} "
        f"covered={int(summary.get('truth_covered_count', 0))}/"
        f"{int(summary.get('truth_count', 0))} "
        f"uncovered={int(summary.get('truth_uncovered_count', 0))} "
        f"mean_truth_nn_d={_fmt_optional_float(mean_err)}m "
        f"max_truth_nn_d={_fmt_optional_float(max_err)}m "
        f"mean_truth_nn_abs_q_rel={_fmt_optional_float(mean_q_rel)} "
        f"entries=[{'; '.join(chunks)}]"
    )


def _append_estimate_trace_jsonl(
    trace_path: Path,
    payload: dict[str, object],
) -> None:
    """Append one intermediate estimate trace payload as JSON Lines."""
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _emit_intermediate_estimate_trace(
    estimator: RotatingShieldPFEstimator,
    isotopes: Sequence[str],
    frame: PFFrame | dict[str, object],
    true_sources: dict[str, NDArray[np.float64]],
    true_strengths: dict[str, float | Sequence[float]],
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    *,
    step_index: int,
    elapsed_s: float,
    counts_by_isotope: dict[str, float],
    obstacle_height_m: float,
    match_radius_m: float,
    trace_path: Path | None,
    log_enabled: bool,
    log_every: int,
    max_log_records: int,
    estimate_source: str,
) -> None:
    """Write and optionally print current MAP estimate trace diagnostics."""
    estimate_trace_frame = _current_map_estimate_trace_frame(
        estimator,
        isotopes,
        frame,
        step_index=step_index,
        elapsed_s=elapsed_s,
        counts_by_isotope={
            str(key): float(value) for key, value in counts_by_isotope.items()
        },
        estimate_source=estimate_source,
    )
    estimate_trace_payload = _build_intermediate_estimate_trace_payload(
        estimate_trace_frame,
        true_sources,
        true_strengths,
        env,
        obstacle_grid,
        obstacle_height_m=obstacle_height_m,
        match_radius_m=match_radius_m,
    )
    if trace_path is not None:
        _append_estimate_trace_jsonl(trace_path, estimate_trace_payload)
    if not log_enabled or step_index % max(1, int(log_every)) != 0:
        return
    estimate_records = list(estimate_trace_payload.get("estimates", []))
    truth_records = list(estimate_trace_payload.get("truth_sources", []))
    for iso, summary in sorted(
        dict(estimate_trace_payload.get("isotopes", {})).items()
    ):
        summary_with_source = dict(summary)
        summary_with_source["estimate_source"] = estimate_trace_payload.get(
            "estimate_source"
        )
        iso_records = [
            dict(record)
            for record in estimate_records
            if dict(record).get("isotope") == iso
        ]
        iso_truth_records = [
            dict(record)
            for record in truth_records
            if dict(record).get("isotope") == iso
        ]
        print(
            _format_estimate_trace_log_line(
                step_index,
                iso,
                summary_with_source,
                iso_records,
                max_records=max_log_records,
            ),
            flush=True,
        )
        print(
            _format_truth_coverage_log_line(
                step_index,
                iso,
                summary_with_source,
                iso_truth_records,
                max_records=max_log_records,
            ),
            flush=True,
        )


def _fmt_top_k(entries: list[dict[str, object]]) -> str:
    """Format top-k particle summaries for logging."""
    chunks = []
    for entry in entries:
        weight = float(entry.get("weight", 0.0))
        num_sources = int(entry.get("num_sources", 0))
        positions = np.asarray(
            entry.get("positions", np.zeros((0, 3))),
            dtype=float,
        )
        strengths = np.asarray(entry.get("strengths", np.zeros(0)), dtype=float)
        sources = _fmt_sources(positions, strengths)
        chunks.append(f"(w={weight:.3f}, r={num_sources}, sources={sources})")
    return "[" + "; ".join(chunks) + "]"


def _fmt_optional_float(value: float | None, precision: int = 2) -> str:
    """Format an optional float for logging."""
    if value is None:
        return "NA"
    return f"{float(value):.{precision}f}"


def _metadata_float(metadata: dict[str, object], key: str) -> float | None:
    """Read one numeric metadata field when present."""
    value = metadata.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metadata_bool(metadata: Mapping[str, object], key: str) -> bool | None:
    """Read one strict boolean-like metadata field when present."""
    value = metadata.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return None


def _measurement_transport_provenance(
    metadata: dict[str, object],
) -> dict[str, object]:
    """Return Geant4 fidelity fields that must survive in measurement logs."""
    keys = (
        "accelerated_weighted_transport_enable",
        "adaptive_dwell",
        "adaptive_dwell_child_step_ids",
        "adaptive_dwell_chunk_live_times_s",
        "adaptive_dwell_chunk_primary_history_weights",
        "adaptive_dwell_chunk_primary_sampling_fractions",
        "adaptive_dwell_chunks",
        "adaptive_dwell_effective_primary_sampling_fraction",
        "adaptive_dwell_live_time_s",
        "adaptive_dwell_primary_history_weight_semantics",
        "adaptive_dwell_primary_sampling_fraction_semantics",
        "adaptive_dwell_target_sampled_primaries_semantics",
        "adaptive_dwell_transport_chunk_provenance",
        "background_cps",
        "dead_time_observed_scale",
        "dead_time_tau_s",
        "detector_response_applied_in_native",
        "detector_scoring_mode",
        "dwell_time_s",
        "emission_model",
        "engine_mode",
        "expected_detector_equivalent_primaries",
        "expected_physical_primaries",
        "expected_primary_semantics",
        "expected_sampled_primaries",
        "expected_unthinned_primaries",
        "gamma_only_secondary_transport",
        "history_thinning_enabled",
        "intensity_cps_1m_definition",
        "line_intensities_normalized",
        "multithreaded_run_manager",
        "num_primaries",
        "pre_dead_time_total_spectrum_counts",
        "pre_dead_time_weighted_spectrum_sumw2",
        "primary_history_weight",
        "primary_sampling_budget_enabled",
        "primary_sampling_fraction",
        "primary_sampling_fraction_resolution",
        "poisson_background",
        "physics_profile",
        "requested_threads",
        "requested_primary_sampling_fraction",
        "secondary_transport_mode",
        "source_bias_weighted_transport",
        "source_bias_mode",
        "source_rate_model",
        "spectrum_variance_dead_time_propagation",
        "spectrum_variance_semantics",
        "target_sampled_primaries",
        "theory_tvl_attenuation",
        "transport_history_mode",
        "transport_tally_weighted",
        "weighted_spectrum_effective_entries",
        "weighted_spectrum_sumw2",
        "weighted_transport",
    )
    return {key: metadata[key] for key in keys if key in metadata}


def _fmt_count_map(counts: dict[str, float], precision: int = 2) -> str:
    """Format isotope count maps for compact step diagnostics."""
    if not counts:
        return "{}"
    chunks = [
        f"{iso}:{float(value):.{precision}f}" for iso, value in sorted(counts.items())
    ]
    return "{" + ", ".join(chunks) + "}"


def _count_ratio_map(
    numerator: dict[str, float],
    denominator: dict[str, float],
) -> dict[str, float]:
    """Return isotope-wise count ratios with zero denominators omitted."""
    ratios: dict[str, float] = {}
    for isotope, denom in sorted(denominator.items()):
        denom_value = float(denom)
        if denom_value <= 0.0:
            continue
        ratios[str(isotope)] = float(numerator.get(isotope, 0.0)) / denom_value
    return ratios


def _fmt_probability_map(probabilities: dict[str, float], precision: int = 3) -> str:
    """Format discrete probability maps for compact PF diagnostics."""
    if not probabilities:
        return "{}"
    chunks: list[str] = []
    for key, value in sorted(
        probabilities.items(),
        key=lambda item: (
            (
                0,
                int(item[0]),
            )
            if str(item[0]).lstrip("-").isdigit()
            else (1, str(item[0]))
        ),
    ):
        chunks.append(f"{key}:{float(value):.{precision}f}")
    return "{" + ", ".join(chunks) + "}"


def _source_equivalent_counts_from_metadata(
    metadata: dict[str, object],
    isotopes: list[str] | tuple[str, ...],
) -> dict[str, float]:
    """Extract Geant4 source-equivalent counts from observation metadata."""
    counts: dict[str, float] = {}
    for isotope in isotopes:
        value = _metadata_float(metadata, f"source_equivalent_counts_{isotope}")
        if value is not None:
            counts[str(isotope)] = value
    return counts


def _transport_detected_counts_from_metadata(
    metadata: dict[str, object],
    isotopes: list[str] | tuple[str, ...],
) -> dict[str, float]:
    """Extract source-labeled Geant4 detector-entry counts for diagnostics."""
    counts: dict[str, float] = {}
    for isotope in isotopes:
        value = _metadata_float(metadata, f"transport_detected_counts_{isotope}")
        if value is not None:
            counts[str(isotope)] = value
    return counts


def _response_poisson_counts_for_diagnostics(
    diagnostic_decomposer: SpectralDecomposer,
    spectrum: NDArray[np.float64],
    isotopes: list[str] | tuple[str, ...],
) -> dict[str, float]:
    """Compute response-Poisson counts without mutating the main decomposer state."""
    return {
        iso: float(value)
        for iso, value in diagnostic_decomposer.compute_response_poisson_counts(
            spectrum,
            isotopes=list(isotopes),
            include_background=True,
        ).items()
    }


def _log_geant4_transport_decomposition_diagnostics(
    *,
    step_index: int,
    metadata: dict[str, object],
    spectrum_total_counts: float,
    selected_count_method: str,
    selected_counts: dict[str, float],
    response_poisson_counts: dict[str, float],
    source_equivalent_counts: dict[str, float],
    transport_detected_counts: dict[str, float],
) -> None:
    """Log Geant4 transport and spectrum-decomposition diagnostics for one step."""
    if str(metadata.get("backend", "")).lower() != "geant4":
        return
    response_suffix = ""
    if str(selected_count_method) != "response_poisson":
        response_suffix = f" response_poisson={_fmt_count_map(response_poisson_counts)}"
    weighted_effective = _metadata_float(
        metadata, "weighted_spectrum_effective_entries"
    )
    primaries = _metadata_float(metadata, "num_primaries")
    run_time_s = _metadata_float(metadata, "run_time_s")
    primaries_per_sec = _metadata_float(metadata, "primaries_per_sec")
    effective_per_sec = _metadata_float(metadata, "effective_entries_per_sec")
    total_steps = _metadata_float(metadata, "total_track_steps")
    detector_hit_events = _metadata_float(metadata, "detector_hit_events")
    detector_hit_steps = _metadata_float(metadata, "detector_hit_steps")
    secondaries = _metadata_float(metadata, "secondary_count")
    killed_non_gamma = _metadata_float(metadata, "killed_non_gamma_secondary_count")
    compton = _metadata_float(metadata, "process_count_compton")
    rayleigh = _metadata_float(metadata, "process_count_rayleigh")
    photoelectric = _metadata_float(metadata, "process_count_photoelectric")
    print(
        f"[step {step_index}] geant4_transport "
        f"primaries={_fmt_optional_float(primaries, 0)} "
        f"run={_fmt_optional_float(run_time_s, 3)}s "
        f"primaries_per_sec={_fmt_optional_float(primaries_per_sec, 1)} "
        f"track_steps={_fmt_optional_float(total_steps, 0)} "
        f"detector_hit_events={_fmt_optional_float(detector_hit_events, 0)} "
        f"detector_hit_steps={_fmt_optional_float(detector_hit_steps, 0)} "
        f"secondaries={_fmt_optional_float(secondaries, 0)} "
        f"killed_non_gamma={_fmt_optional_float(killed_non_gamma, 0)} "
        f"compton={_fmt_optional_float(compton, 0)} "
        f"rayleigh={_fmt_optional_float(rayleigh, 0)} "
        f"photoelectric={_fmt_optional_float(photoelectric, 0)} "
        f"effective_entries={_fmt_optional_float(weighted_effective, 1)} "
        f"effective_entries_per_sec={_fmt_optional_float(effective_per_sec, 1)}"
    )
    print(
        f"[step {step_index}] geant4_decomposition "
        f"source_equivalent_unattenuated={_fmt_count_map(source_equivalent_counts)} "
        f"transport_detected={_fmt_count_map(transport_detected_counts)} "
        f"total_spectrum_counts={float(spectrum_total_counts):.2f} "
        f"{selected_count_method}={_fmt_count_map(selected_counts)}"
        f"{response_suffix}"
    )
    selected_to_transport = _count_ratio_map(
        selected_counts,
        transport_detected_counts,
    )
    response_to_transport = _count_ratio_map(
        response_poisson_counts,
        transport_detected_counts,
    )
    transport_to_source = _count_ratio_map(
        transport_detected_counts,
        source_equivalent_counts,
    )
    print(
        f"[step {step_index}] geant4_decomposition_ratios "
        f"selected_over_transport={_fmt_count_map(selected_to_transport, 3)} "
        f"response_poisson_over_transport={_fmt_count_map(response_to_transport, 3)} "
        f"transport_over_source_equivalent={_fmt_count_map(transport_to_source, 3)}"
    )
    volume_top = str(metadata.get("volume_step_counts_top", "")).strip()
    if volume_top and volume_top != "-":
        print(f"[step {step_index}] geant4_volume_steps_top {volume_top}")


def _log_pf_diagnostics(
    estimator: RotatingShieldPFEstimator,
    step_index: int,
    top_k: int = HEALTH_LOG_TOP_K,
    include_estimates: bool = False,
) -> None:
    """Log per-step PF diagnostics for each isotope."""
    diagnostics = estimator.step_diagnostics(
        top_k=top_k,
        include_estimates=include_estimates,
    )
    if not diagnostics:
        print(f"[step {step_index}] pf_diagnostics: no active filters")
        return
    for iso, stats in diagnostics.items():
        filt = estimator.filters.get(iso)
        ess_pre = float(stats["ess_pre"])
        resampled = bool(stats["resampled"])
        ess_post = stats["ess_post"]
        n_after_adapt = int(stats["n_after_adapt"])
        resamples = int(stats["resample_count"])
        mode_preserved = int(stats.get("mode_preserved_count", 0))
        births = int(stats["birth_count"])
        kills = int(stats["kill_count"])
        birth_gate_passed = bool(stats.get("birth_residual_gate_passed", False))
        birth_gate_support = int(stats.get("birth_residual_support", 0))
        birth_gate_distinct = int(stats.get("birth_residual_distinct_poses", 0))
        birth_gate_stations = int(stats.get("birth_residual_distinct_stations", 0))
        birth_gate_chi2 = float(stats.get("birth_residual_chi2", 0.0))
        birth_gate_p = float(stats.get("birth_residual_p_value", 1.0))
        birth_layer = str(stats.get("birth_residual_layer", "none"))
        birth_structural_eligible = int(stats.get("birth_structural_eligible", 0))
        pseudo_verified = int(stats.get("pseudo_source_verified", 0))
        pseudo_failed = int(stats.get("pseudo_source_failed", 0))
        pseudo_pruned = int(stats.get("pseudo_source_pruned", 0))
        pseudo_quarantined = int(stats.get("pseudo_source_quarantined", 0))
        pseudo_quarantine_active = int(stats.get("pseudo_source_quarantine_active", 0))
        pseudo_reasons = stats.get("pseudo_source_fail_reasons", {})
        structural_timing = stats.get("structural_timing_s", {})
        temper_steps = stats.get("temper_steps", [])
        temper_resamples = int(stats.get("temper_resamples", 0))
        mode_strata_summary = dict(stats.get("mode_preserving_strata_summary", {}))
        mode_selected_strata = list(stats.get("mode_preserving_selected_strata", []))
        mode_cardinality_summary = dict(
            stats.get("mode_preserving_cardinality_summary", {})
        )
        mode_selected_cardinalities = list(
            stats.get("mode_preserving_selected_cardinalities", [])
        )
        mode_dynamic_spatial = list(
            stats.get("mode_preserving_dynamic_spatial_summary", [])
        )
        r_mean = float(stats["r_mean"])
        r_var = float(stats["r_var"])
        r_weighted_mean = float(stats.get("r_weighted_mean", r_mean))
        r_weighted_var = float(stats.get("r_weighted_var", r_var))
        r_probabilities = dict(stats.get("r_probability_by_count", {}))
        r_particle_counts = dict(stats.get("r_particle_count_by_count", {}))
        map_pos, map_str = stats["map"]
        mmse_pos, mmse_str = stats["mmse"]
        top_entries = stats["top_k"]
        converged = bool(stats.get("converged", False))
        updates_skipped = int(stats.get("updates_skipped", 0))
        birth_enabled = bool(
            getattr(getattr(filt, "config", None), "birth_enable", False)
        )
        max_sources = getattr(getattr(filt, "config", None), "max_sources", None)
        p_birth = float(getattr(getattr(filt, "config", None), "p_birth", 0.0))
        structural_workers = int(
            getattr(getattr(filt, "config", None), "structural_trial_workers", 1)
        )
        structural_min_trials = int(
            getattr(
                getattr(filt, "config", None),
                "structural_trial_parallel_min_trials",
                1,
            )
        )
        print(
            f"[step {step_index}] pf[{iso}] ess_pre={ess_pre:.2f} resampled={resampled} "
            f"ess_post={_fmt_optional_float(ess_post)} n_after={n_after_adapt} "
            f"resamples={resamples} mode_preserved={mode_preserved} "
            f"births={births} kills={kills} "
            f"r_mean={r_mean:.2f} r_var={r_var:.2f} "
            f"r_weighted_mean={r_weighted_mean:.2f} "
            f"r_weighted_var={r_weighted_var:.2f} "
            f"r_posterior={_fmt_probability_map(r_probabilities)} "
            f"r_particles={r_particle_counts} "
            f"converged={converged} skipped={updates_skipped} "
            f"birth_enabled={birth_enabled} max_sources={max_sources} p_birth={p_birth:.3f} "
            f"structural_trial_workers={structural_workers} "
            f"structural_trial_parallel_min_trials={structural_min_trials} "
            f"birth_gate={birth_gate_passed} "
            f"birth_residual_support={birth_gate_support} "
            f"birth_residual_distinct_poses={birth_gate_distinct} "
            f"birth_residual_distinct_stations={birth_gate_stations} "
            f"birth_residual_chi2={birth_gate_chi2:.2f} "
            f"birth_residual_p={birth_gate_p:.3g} "
            f"birth_layer={birth_layer} "
            f"birth_structural_eligible={birth_structural_eligible} "
            f"pseudo_verified={pseudo_verified} "
            f"pseudo_failed={pseudo_failed} "
            f"pseudo_pruned={pseudo_pruned} "
            f"pseudo_quarantined={pseudo_quarantined} "
            f"pseudo_quarantine_active={pseudo_quarantine_active}"
        )
        if pseudo_reasons:
            reason_str = ", ".join(
                f"{key}={int(value)}"
                for key, value in sorted(dict(pseudo_reasons).items())
            )
            print(f"[step {step_index}] pf[{iso}] pseudo_fail_reasons={reason_str}")
        if (
            mode_strata_summary
            or mode_selected_strata
            or mode_cardinality_summary
            or mode_selected_cardinalities
            or mode_dynamic_spatial
        ):
            print(
                f"[step {step_index}] pf[{iso}] mode_preservation "
                f"strata_mass={_safe_json_dumps(mode_strata_summary)} "
                f"selected={_safe_json_dumps(mode_selected_strata)} "
                f"cardinality_mass={_safe_json_dumps(mode_cardinality_summary)} "
                f"selected_cardinality={_safe_json_dumps(mode_selected_cardinalities)} "
                f"dynamic_spatial={_safe_json_dumps(mode_dynamic_spatial)}"
            )
        if structural_timing:
            timing_items = {
                key: float(value)
                for key, value in dict(structural_timing).items()
                if float(value) > 0.0 or key == "total"
            }
            timing_str = " ".join(
                _format_pf_timing_item(key, value)
                for key, value in sorted(timing_items.items())
            )
            print(f"[step {step_index}] pf_timing[{iso}] {timing_str}")
        if not include_estimates:
            continue
        if temper_steps:
            temper_str = ", ".join(
                f"(beta={s['beta_total']:.3f},db={s['delta_beta']:.3f},ess={s['ess']:.1f})"
                for s in temper_steps
            )
            print(
                f"[step {step_index}] pf[{iso}] temper={temper_str} "
                f"temper_resamples={temper_resamples}"
            )
        print(
            f"[step {step_index}] pf[{iso}] map={_fmt_sources(map_pos, map_str)} "
            f"mmse={_fmt_sources(mmse_pos, mmse_str)}"
        )
        if top_entries:
            print(f"[step {step_index}] pf[{iso}] top_k={_fmt_top_k(top_entries)}")




def _log_surface_candidate_observability_diagnostics(
    estimator: RotatingShieldPFEstimator,
    step_index: int,
    *,
    label: str,
    window: int | None = None,
    max_candidates: int = 256,
) -> None:
    """Log truth-independent surface-candidate observability diagnostics."""
    if int(max_candidates) <= 0:
        return
    if not hasattr(estimator, "surface_candidate_observability_diagnostics"):
        return
    diagnostics = estimator.surface_candidate_observability_diagnostics(
        window=window,
        max_candidates=max_candidates,
    )
    if not diagnostics:
        print(f"[step {step_index}] surface_observability[{label}]: no diagnostics")
        return
    for iso, stats_raw in sorted(diagnostics.items()):
        stats = dict(stats_raw)
        print(
            f"[step {step_index}] surface_observability[{iso}] label={label} "
            f"measurements={int(stats.get('measurement_count', 0))} "
            f"candidates={int(stats.get('candidate_count', 0))} "
            f"sampled={int(stats.get('sampled_candidate_count', 0))} "
            f"active={int(stats.get('active_candidate_count', 0))} "
            f"weak={int(stats.get('weak_column_count', 0))} "
            f"condition={_fmt_optional_float(stats.get('condition_number'))} "
            f"max_corr={_fmt_optional_float(stats.get('max_abs_correlation'), 3)} "
            "ambiguous99="
            f"{int(stats.get('ambiguous_pair_count_corr_ge_0p99', 0))} "
            "ambiguous995="
            f"{int(stats.get('ambiguous_pair_count_corr_ge_0p995', 0))} "
            f"surface_counts={_safe_json_dumps(stats.get('surface_counts', {}))}"
        )


def _log_dss_ranked_node_diagnostics(
    diagnostics: dict[str, Any],
    *,
    label: str,
) -> None:
    """Log ranked DSS-PP station/program candidates with score components."""
    ranked_raw = diagnostics.get("ranked_nodes", [])
    if not isinstance(ranked_raw, Sequence):
        return
    ranked = [dict(node) for node in ranked_raw if isinstance(node, dict)]
    if not ranked:
        print(f"DSS-PP ranked candidates[{label}]: none")
        return
    limit = int(diagnostics.get("diagnostic_ranked_node_limit", len(ranked)))
    if limit <= 0:
        print(
            f"DSS-PP ranked candidates[{label}]: "
            f"logged=0 limit={limit} "
            f"nodes={int(diagnostics.get('node_count', len(ranked)))} "
            f"candidates={int(diagnostics.get('candidate_count', 0))} "
            f"programs={int(diagnostics.get('program_count', 0))}"
        )
        return
    print(
        f"DSS-PP ranked candidates[{label}]: "
        f"logged={len(ranked)} limit={limit} "
        f"nodes={int(diagnostics.get('node_count', len(ranked)))} "
        f"candidates={int(diagnostics.get('candidate_count', 0))} "
        f"programs={int(diagnostics.get('program_count', 0))}"
    )
    for entry in ranked:
        pose = np.asarray(entry.get("pose_xyz", np.zeros(3)), dtype=float)
        pairs = entry.get("pair_ids", [])
        print(
            "DSS-PP candidate "
            f"[{label}] rank={int(entry.get('rank', 0))} "
            f"pose_idx={int(entry.get('pose_index', -1))} "
            f"pose={_fmt_pos(pose)} "
            f"program={entry.get('program_name')} "
            f"kind={entry.get('program_kind')} "
            f"pairs={list(pairs) if isinstance(pairs, Sequence) else pairs} "
            f"score={float(entry.get('score', 0.0)):.6g} "
            f"static={float(entry.get('static_score', 0.0)):.6g} "
            f"ig={float(entry.get('information_gain', 0.0)):.6g} "
            f"signature={float(entry.get('signature_score', 0.0)):.6g} "
            f"temporal_sep={float(entry.get('temporal_separation_score', 0.0)):.6g} "
            f"elevation_sep={float(entry.get('elevation_signature_score', 0.0)):.6g} "
            f"obs_penalty={float(entry.get('observation_penalty', 0.0)):.6g} "
            f"count_balance={float(entry.get('count_balance_penalty', 0.0)):.6g} "
            f"diff_penalty={float(entry.get('differential_penalty', 0.0)):.6g} "
            f"dose={float(entry.get('dose_score', 0.0)):.6g} "
            f"count_util={float(entry.get('count_utility', 0.0)):.6g} "
            f"coverage={float(entry.get('coverage_gain', 0.0)):.6g} "
            f"revisit={float(entry.get('revisit_penalty', 0.0)):.6g} "
            f"bearing={float(entry.get('bearing_diversity_gain', 0.0)):.6g} "
            f"frontier={float(entry.get('frontier_gain', 0.0)):.6g} "
            f"turn={float(entry.get('turn_penalty', 0.0)):.6g} "
            f"local_orbit={float(entry.get('local_orbit_gain', 0.0)):.6g} "
            f"station_cond={float(entry.get('station_condition_gain', 0.0)):.6g} "
            f"corr_reduction={float(entry.get('correlation_reduction_gain', 0.0)):.6g} "
            f"isotope_balance={float(entry.get('isotope_balance_gain', 0.0)):.6g} "
            f"elev_cond={float(entry.get('elevation_condition_gain', 0.0)):.6g} "
            f"env_sig={float(entry.get('environment_signature_score', 0.0)):.6g} "
            f"vertical_env={float(entry.get('vertical_environment_signature_score', 0.0)):.6g} "
            f"occ_boundary={float(entry.get('occlusion_boundary_gain', 0.0)):.6g} "
            f"route_pressure={float(entry.get('remaining_route_pressure', 0.0)):.6g} "
            f"route_penalty={float(entry.get('remaining_route_penalty', 0.0)):.6g} "
            f"route_gain={float(entry.get('remaining_route_gain', 0.0)):.6g}"
        )


def _best_dss_first_step_guard_candidate(
    diagnostics: dict[str, Any],
    *,
    candidate_poses_xyz: NDArray[np.float64],
) -> tuple[int, float, NDArray[np.float64]] | None:
    """Return the best greedy first-step DSS-PP candidate from diagnostics."""
    ranked_raw = diagnostics.get("ranked_nodes", [])
    if not isinstance(ranked_raw, Sequence):
        return None
    candidates = np.asarray(candidate_poses_xyz, dtype=float).reshape(-1, 3)
    for entry_raw in ranked_raw:
        if not isinstance(entry_raw, dict):
            continue
        try:
            pose_index = int(entry_raw.get("pose_index", -1))
            score = float(entry_raw.get("score", -np.inf))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(score):
            continue
        pose_xyz = np.asarray(entry_raw.get("pose_xyz", []), dtype=float).reshape(-1)
        if pose_xyz.size != 3 or not np.all(np.isfinite(pose_xyz)):
            if pose_index < 0 or pose_index >= int(candidates.shape[0]):
                continue
            pose_xyz = candidates[pose_index]
        return pose_index, score, np.asarray(pose_xyz, dtype=float).copy()
    return None


def _log_dss_pairwise_ambiguity_diagnostics(
    diagnostics: dict[str, Any],
    *,
    label: str,
) -> None:
    """Log selected DSS-PP program diagnostics for ambiguous mode pairs."""
    payload_raw = diagnostics.get("selected_pairwise_ambiguity", {})
    if not isinstance(payload_raw, dict) or not payload_raw:
        print(f"DSS-PP pairwise ambiguity[{label}]: none")
        return
    for isotope, stats_raw in sorted(payload_raw.items()):
        stats = dict(stats_raw) if isinstance(stats_raw, dict) else {}
        print(
            f"DSS-PP pairwise ambiguity[{label}][{isotope}] "
            f"modes={int(stats.get('mode_count', 0))} "
            f"pairs={list(stats.get('program_pair_ids', []))} "
            f"before_meas={int(stats.get('before_measurements', 0))} "
            f"program_meas={int(stats.get('program_measurements', 0))} "
            f"before_min_sep={_fmt_optional_float(stats.get('before_min_separation'), 3)} "
            f"program_min_sep={_fmt_optional_float(stats.get('program_min_separation'), 3)} "
            f"combined_min_sep={_fmt_optional_float(stats.get('combined_min_separation'), 3)} "
            f"before_max_corr={_fmt_optional_float(stats.get('before_max_correlation'), 3)} "
            f"program_max_corr={_fmt_optional_float(stats.get('program_max_correlation'), 3)} "
            f"combined_max_corr={_fmt_optional_float(stats.get('combined_max_correlation'), 3)} "
            f"unresolved={int(stats.get('before_unresolved_pairs', 0))}->"
            f"{int(stats.get('combined_unresolved_pairs', 0))}"
        )
        bottlenecks = stats.get("bottleneck_pairs", [])
        if not isinstance(bottlenecks, Sequence):
            continue
        for pair_raw in bottlenecks:
            if not isinstance(pair_raw, dict):
                continue
            left = dict(pair_raw.get("left_mode", {}))
            right = dict(pair_raw.get("right_mode", {}))
            print(
                f"DSS-PP bottleneck_pair[{label}][{isotope}] "
                f"rank={int(pair_raw.get('rank', 0))} "
                f"left={int(left.get('index', -1))}@{left.get('pos')} "
                f"right={int(right.get('index', -1))}@{right.get('pos')} "
                f"before_sep={_fmt_optional_float(pair_raw.get('before_separation'), 3)} "
                f"program_sep={_fmt_optional_float(pair_raw.get('program_separation'), 3)} "
                f"combined_sep={_fmt_optional_float(pair_raw.get('combined_separation'), 3)} "
                f"combined_corr={_fmt_optional_float(pair_raw.get('combined_correlation'), 3)} "
                f"bearing_delta_deg={_fmt_optional_float(pair_raw.get('bearing_delta_deg'), 2)} "
                f"elevation_delta_deg={_fmt_optional_float(pair_raw.get('elevation_delta_deg'), 2)} "
                f"program_left={pair_raw.get('program_left_response')} "
                f"program_right={pair_raw.get('program_right_response')}"
            )


def _log_dss_component_leader_diagnostics(
    diagnostics: dict[str, Any],
    *,
    label: str,
) -> None:
    """Log per-component DSS-PP leaders for counterfactual selection analysis."""
    leaders_raw = diagnostics.get("component_leaders", {})
    if not isinstance(leaders_raw, dict) or not leaders_raw:
        print(f"DSS-PP component leaders[{label}]: none")
        return
    for component, entry_raw in sorted(leaders_raw.items()):
        if not isinstance(entry_raw, dict):
            continue
        pose = np.asarray(entry_raw.get("pose_xyz", np.zeros(3)), dtype=float)
        pairs = entry_raw.get("pair_ids", [])
        print(
            f"DSS-PP component leader[{label}] component={component} "
            f"value={_fmt_optional_float(entry_raw.get('component_value'), 6)} "
            f"pose_idx={int(entry_raw.get('pose_index', -1))} "
            f"pose={_fmt_pos(pose)} "
            f"program={entry_raw.get('program_name')} "
            f"kind={entry_raw.get('program_kind')} "
            f"pairs={list(pairs) if isinstance(pairs, Sequence) else pairs} "
            f"score={float(entry_raw.get('score', 0.0)):.6g}"
        )


def _log_remaining_measurement_detail(
    estimate: object,
    *,
    label: str,
) -> None:
    """Log remaining-measurement component and gain breakdowns."""
    components = getattr(estimate, "components", {})
    gains = getattr(estimate, "gains", {})
    isotope_details = getattr(estimate, "isotope_details", {})
    print(
        f"Remaining measurement detail[{label}]: "
        f"components={_safe_json_dumps(components)} "
        f"gains={_safe_json_dumps(gains)} "
        f"isotopes={_safe_json_dumps(isotope_details)}"
    )


def _sanitize_json_payload(payload: object) -> object:
    """Return recursively plain, strict-JSON-compatible data.

    NumPy containers and scalars are converted to their Python equivalents,
    mappings receive string keys, and non-finite floating-point values become
    ``None``. Unsupported objects raise instead of being silently stringified.
    """
    if payload is None or isinstance(payload, (str, bool)):
        return payload
    if isinstance(payload, np.bool_):
        return bool(payload)
    if isinstance(payload, (int, np.integer)):
        return int(payload)
    if isinstance(payload, (float, np.floating)):
        value = float(payload)
        return value if np.isfinite(value) else None
    if isinstance(payload, Path):
        return payload.as_posix()
    if isinstance(payload, np.ndarray):
        return _sanitize_json_payload(payload.tolist())
    if isinstance(payload, np.generic):
        return _sanitize_json_payload(payload.item())
    if isinstance(payload, Mapping):
        return {
            str(key): _sanitize_json_payload(value) for key, value in payload.items()
        }
    if isinstance(payload, (list, tuple)):
        return [_sanitize_json_payload(value) for value in payload]
    if isinstance(payload, (set, frozenset)):
        return [
            _sanitize_json_payload(value)
            for value in sorted(payload, key=lambda value: repr(value))
        ]
    raise TypeError(
        "Unsupported value in JSON payload: "
        f"{type(payload).__module__}.{type(payload).__qualname__}"
    )


def _safe_json_dumps(payload: object) -> str:
    """Return a compact JSON string for best-effort diagnostic logging."""

    def _default(value: object) -> object:
        """Convert common NumPy values and stringify unknown log-only values."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        return str(value)

    return json.dumps(payload, sort_keys=True, default=_default)


def _remaining_measurement_progress(
    estimates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return progress diagnostics from consecutive remaining-budget estimates."""
    if len(estimates) < 2:
        return {
            "available": False,
            "has_progress": False,
            "residual_improved": False,
            "budget_improved": False,
            "remaining_stations_improved": False,
        }
    current = estimates[-1]

    def _component(payload: Mapping[str, Any], key: str) -> float:
        """Return one component value from a remaining-budget payload."""
        components = payload.get("components", {})
        if not isinstance(components, Mapping):
            return 0.0
        try:
            return float(components.get(key, 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _float_value(payload: Mapping[str, Any], key: str) -> float:
        """Return one scalar value from a remaining-budget payload."""
        try:
            return float(payload.get(key, 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _int_value(payload: Mapping[str, Any], key: str) -> int:
        """Return one integer value from a remaining-budget payload."""
        try:
            return int(payload.get(key, 0))
        except (TypeError, ValueError):
            return 0

    previous_estimates = list(estimates[:-1])
    prev_residual = _component(previous_estimates[-1], "residual")
    cur_residual = _component(current, "residual")
    prev_budget = _float_value(previous_estimates[-1], "current_budget")
    cur_budget = _float_value(current, "current_budget")
    prev_remaining = _int_value(
        previous_estimates[-1],
        "estimated_remaining_stations",
    )
    cur_remaining = _int_value(current, "estimated_remaining_stations")
    best_prev_residual = min(
        _component(item, "residual") for item in previous_estimates
    )
    best_prev_budget = min(
        _float_value(item, "current_budget") for item in previous_estimates
    )
    best_prev_remaining = min(
        _int_value(item, "estimated_remaining_stations") for item in previous_estimates
    )
    residual_recent_improved = cur_residual < prev_residual - 1.0e-9
    budget_recent_improved = cur_budget < prev_budget - 1.0e-9
    remaining_recent_improved = cur_remaining < prev_remaining
    residual_improved = cur_residual < best_prev_residual - 1.0e-9
    budget_improved = cur_budget < best_prev_budget - 1.0e-9
    remaining_improved = cur_remaining < best_prev_remaining
    return {
        "available": True,
        "has_progress": bool(
            residual_improved or budget_improved or remaining_improved
        ),
        "residual_improved": bool(residual_improved),
        "budget_improved": bool(budget_improved),
        "remaining_stations_improved": bool(remaining_improved),
        "residual_recent_improved": bool(residual_recent_improved),
        "budget_recent_improved": bool(budget_recent_improved),
        "remaining_stations_recent_improved": bool(remaining_recent_improved),
        "previous_residual_budget": float(prev_residual),
        "current_residual_budget": float(cur_residual),
        "best_previous_residual_budget": float(best_prev_residual),
        "previous_current_budget": float(prev_budget),
        "current_current_budget": float(cur_budget),
        "best_previous_current_budget": float(best_prev_budget),
        "previous_estimated_remaining_stations": int(prev_remaining),
        "current_estimated_remaining_stations": int(cur_remaining),
        "best_previous_estimated_remaining_stations": int(best_prev_remaining),
    }


def _nearest_truth_diagnostic(
    isotope: str,
    position_xyz: NDArray[np.float64],
    true_sources: dict[str, NDArray[np.float64]],
    true_strengths: dict[str, float | Sequence[float]],
) -> dict[str, object]:
    """Return nearest truth-source diagnostics for one estimated position."""
    truths = np.asarray(true_sources.get(isotope, np.zeros((0, 3))), dtype=float)
    truths = truths.reshape(-1, 3)
    if truths.size == 0:
        return {
            "nearest_truth_index": None,
            "nearest_truth_distance_m": None,
            "nearest_truth_position": None,
            "nearest_truth_strength": None,
        }
    pos = np.asarray(position_xyz, dtype=float).reshape(3)
    distances = np.linalg.norm(truths - pos[None, :], axis=1)
    truth_idx = int(np.argmin(distances))
    strengths = _true_strength_array(true_strengths, isotope, truths.shape[0])
    truth_strength = float(strengths[truth_idx]) if truth_idx < strengths.size else None
    return {
        "nearest_truth_index": truth_idx,
        "nearest_truth_distance_m": float(distances[truth_idx]),
        "nearest_truth_position": [float(value) for value in truths[truth_idx]],
        "nearest_truth_strength": truth_strength,
    }


def _active_state_for_diagnostics(filt: object) -> object | None:
    """Return the current MAP state for PF diagnostics."""
    try:
        return filt.best_particle().state
    except (AttributeError, IndexError, RuntimeError, ValueError):
        return None


def _state_source_arrays(
    state: object | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Return source positions, strengths, and background from a PF state."""
    if state is None:
        return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float), 0.0
    count = max(0, int(getattr(state, "num_sources", 0)))
    positions = np.asarray(
        getattr(state, "positions", np.zeros((0, 3))),
        dtype=float,
    ).reshape(-1, 3)
    strengths = np.asarray(
        getattr(state, "strengths", np.zeros(0)),
        dtype=float,
    ).reshape(-1)
    count = min(count, positions.shape[0], strengths.size)
    background = float(getattr(state, "background", 0.0))
    return positions[:count], strengths[:count], background


def _measurement_geometry_arrays(
    measurement: Measurement,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]
]:
    """Return single-measurement detector, live-time, Fe-index, and Pb-index arrays."""
    detector = np.asarray(
        measurement.detector_position
        if measurement.detector_position is not None
        else np.zeros(3),
        dtype=float,
    ).reshape(1, 3)
    live_times = np.asarray([float(measurement.live_time_s)], dtype=float)
    fe_index = (
        int(measurement.fe_index)
        if measurement.fe_index is not None
        else int(measurement.orient_idx)
    )
    pb_index = (
        int(measurement.pb_index)
        if measurement.pb_index is not None
        else int(measurement.orient_idx)
    )
    return (
        detector,
        live_times,
        np.asarray([fe_index], dtype=int),
        np.asarray([pb_index], dtype=int),
    )


def _log_spectrum_response_poisson_diagnostics(
    decomposer: SpectralDecomposer,
    *,
    step_index: int,
) -> None:
    """Log the full-spectrum response-regression diagnostics from spectrum processing."""
    diagnostics = dict(getattr(decomposer, "last_response_poisson_diagnostics", {}))
    if not diagnostics:
        print(f"[step {step_index}] spectrum_response_poisson: no diagnostics")
        return
    print(
        f"[step {step_index}] spectrum_response_poisson {_safe_json_dumps(diagnostics)}"
    )


def _log_spectrum_isotope_channel_diagnostics(
    decomposer: SpectralDecomposer,
    *,
    step_index: int,
    selected_counts: dict[str, float],
    selected_variances: dict[str, float],
) -> None:
    """Log compact isotope-channel spectrum diagnostics for online PF debugging."""
    diagnostics = dict(getattr(decomposer, "last_response_poisson_diagnostics", {}))
    if not diagnostics:
        return
    response_counts = diagnostics.get("counts", {})
    response_variances = diagnostics.get("variances", {})
    photopeak_counts = diagnostics.get("photopeak_counts", {})
    photopeak_variances = diagnostics.get("photopeak_variances", {})
    snr = diagnostics.get("snr", {})
    methods = diagnostics.get("methods", {})
    correlations = diagnostics.get("coefficient_correlation_by_isotope", {})
    channel_payload: dict[str, dict[str, object]] = {}
    isotopes = sorted(
        set(str(value) for value in selected_counts)
        | (
            set(str(value) for value in response_counts)
            if isinstance(response_counts, dict)
            else set()
        )
        | (
            set(str(value) for value in photopeak_counts)
            if isinstance(photopeak_counts, dict)
            else set()
        )
    )
    for isotope in isotopes:
        selected = float(selected_counts.get(isotope, 0.0))
        response = (
            float(response_counts.get(isotope, selected))
            if isinstance(response_counts, dict)
            else selected
        )
        photopeak = (
            float(photopeak_counts.get(isotope, 0.0))
            if isinstance(photopeak_counts, dict)
            else 0.0
        )
        response_variance = (
            float(response_variances.get(isotope, max(response, 1.0)))
            if isinstance(response_variances, dict)
            else max(response, 1.0)
        )
        photopeak_variance = (
            float(photopeak_variances.get(isotope, 1.0))
            if isinstance(photopeak_variances, dict)
            else 1.0
        )
        channel_payload[isotope] = {
            "selected": selected,
            "selected_variance": float(selected_variances.get(isotope, 1.0)),
            "response_poisson": response,
            "response_variance": response_variance,
            "photopeak": photopeak,
            "photopeak_variance": photopeak_variance,
            "photopeak_over_response": (
                photopeak / response if response > 1.0e-12 else None
            ),
            "snr": (float(snr.get(isotope, 0.0)) if isinstance(snr, dict) else 0.0),
            "corr": (
                float(correlations.get(isotope, 0.0))
                if isinstance(correlations, dict)
                else 0.0
            ),
            "method": (
                str(methods.get(isotope, "")) if isinstance(methods, dict) else ""
            ),
        }
    print(
        f"[step {step_index}] spectrum_isotope_channels "
        f"{_safe_json_dumps(channel_payload)}"
    )


def _log_current_map_prediction_residuals(
    estimator: RotatingShieldPFEstimator,
    measurement: Measurement,
    true_sources: dict[str, NDArray[np.float64]],
    true_strengths: dict[str, float | Sequence[float]],
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    *,
    obstacle_height_m: float,
    step_index: int,
) -> None:
    """Log current MAP expected counts and residuals for the latest measurement."""
    detector, live_times, fe_indices, pb_indices = _measurement_geometry_arrays(
        measurement
    )
    for isotope, filt in sorted(estimator.filters.items()):
        state = _active_state_for_diagnostics(filt)
        positions, strengths, background = _state_source_arrays(state)
        observed = float(measurement.counts_by_isotope.get(isotope, 0.0))
        if measurement.count_variance_by_isotope is None:
            variance = max(observed, 1.0)
        else:
            variance = max(
                float(measurement.count_variance_by_isotope.get(isotope, observed)),
                1.0,
            )
        if positions.size:
            source_counts = expected_counts_per_source(
                kernel=filt.continuous_kernel,
                isotope=isotope,
                detector_positions=detector,
                sources=positions,
                strengths=strengths,
                live_times=live_times,
                fe_indices=fe_indices,
                pb_indices=pb_indices,
                source_scale=filt._measurement_source_scale(
                    fe_index=int(fe_indices[0]),
                    pb_index=int(pb_indices[0]),
                ),
            )[0]
        else:
            source_counts = np.zeros(0, dtype=float)
        background_counts = float(background) * float(measurement.live_time_s)
        predicted = float(np.sum(source_counts) + background_counts)
        residual = float(observed - predicted)
        whitened = residual / float(np.sqrt(max(variance, 1.0e-12)))
        rel_residual = residual / max(abs(predicted), 1.0e-12)
        surface_kinds = source_surface_kinds(
            positions,
            env,
            obstacle_grid,
            obstacle_height_m=obstacle_height_m,
            tolerance_m=1.0e-5,
        )
        print(
            f"[step {step_index}] map_prediction_residual[{isotope}] "
            f"observed={observed:.3f} predicted={predicted:.3f} "
            f"residual={residual:.3f} whitened={whitened:.3f} "
            f"rel={rel_residual:+.3f} variance={variance:.3f} "
            f"background_counts={background_counts:.3f} "
            f"source_count={positions.shape[0]} "
            f"detector={_fmt_pos(detector[0])} "
            f"fe={int(fe_indices[0])} pb={int(pb_indices[0])} "
            f"live={float(live_times[0]):.3f}"
        )
        for idx, (pos, strength, expected) in enumerate(
            zip(positions, strengths, source_counts)
        ):
            truth_diag = _nearest_truth_diagnostic(
                isotope,
                pos,
                true_sources,
                true_strengths,
            )
            fraction = float(expected) / max(predicted, 1.0e-12)
            surface_kind = (
                str(surface_kinds[idx]) if idx < len(surface_kinds) else "unknown"
            )
            print(
                f"[step {step_index}] map_source_contribution[{isotope}] "
                f"idx={idx} pos={_fmt_pos(pos)} q={float(strength):.3f} "
                f"expected_counts={float(expected):.3f} "
                f"fraction={fraction:.3f} surface={surface_kind} "
                f"nearest_truth={truth_diag.get('nearest_truth_index')} "
                f"truth_d={_fmt_optional_float(truth_diag.get('nearest_truth_distance_m'))}m "
                f"truth_q={_fmt_optional_float(truth_diag.get('nearest_truth_strength'))}"
            )


def _log_truth_observability_diagnostics(
    estimator: RotatingShieldPFEstimator,
    measurement: Measurement,
    true_sources: dict[str, NDArray[np.float64]],
    true_strengths: dict[str, float | Sequence[float]],
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    *,
    obstacle_height_m: float,
    step_index: int,
) -> None:
    """Log how observable each true source is from the latest detector pose."""
    detector, live_times, fe_indices, pb_indices = _measurement_geometry_arrays(
        measurement
    )
    detector_pos = detector[0]
    for isotope, truth_positions_raw in sorted(true_sources.items()):
        filt = estimator.filters.get(isotope)
        if filt is None:
            continue
        truth_positions = np.asarray(truth_positions_raw, dtype=float).reshape(-1, 3)
        truth_strength_arr = _true_strength_array(
            true_strengths,
            isotope,
            truth_positions.shape[0],
        )
        if truth_positions.size == 0:
            continue
        expected = expected_counts_per_source(
            kernel=filt.continuous_kernel,
            isotope=isotope,
            detector_positions=detector,
            sources=truth_positions,
            strengths=truth_strength_arr,
            live_times=live_times,
            fe_indices=fe_indices,
            pb_indices=pb_indices,
            source_scale=filt._measurement_source_scale(
                fe_index=int(fe_indices[0]),
                pb_index=int(pb_indices[0]),
            ),
        )[0]
        surface_kinds = source_surface_kinds(
            truth_positions,
            env,
            obstacle_grid,
            obstacle_height_m=obstacle_height_m,
            tolerance_m=1.0e-5,
        )
        for idx, pos in enumerate(truth_positions):
            q_true = (
                float(truth_strength_arr[idx]) if idx < truth_strength_arr.size else 0.0
            )
            distance = float(np.linalg.norm(detector_pos - pos))
            kernel = filt.continuous_kernel
            obstacle_path_cm = 0.0
            obstacle_tau = 0.0
            obstacle_att = 1.0
            try:
                obstacle_path_cm = float(
                    kernel.obstacle_path_length_cm(pos, detector_pos)
                )
                obstacle_tau = float(
                    kernel.obstacle_optical_depth_pair(
                        isotope,
                        pos,
                        detector_pos,
                    )
                )
                obstacle_att = float(
                    kernel.obstacle_attenuation_factor_pair(
                        isotope,
                        pos,
                        detector_pos,
                    )
                )
            except (AttributeError, RuntimeError, ValueError):
                obstacle_path_cm = 0.0
                obstacle_tau = 0.0
                obstacle_att = 1.0
            per_cps_1m = float(expected[idx]) / max(
                float(live_times[0]) * max(q_true, 1.0e-12),
                1.0e-12,
            )
            surface_kind = (
                str(surface_kinds[idx]) if idx < len(surface_kinds) else "unknown"
            )
            print(
                f"[step {step_index}] truth_observability[{isotope}] "
                f"truth_idx={idx} pos={_fmt_pos(pos)} q_true={q_true:.3f} "
                f"surface={surface_kind} detector={_fmt_pos(detector_pos)} "
                f"distance_m={distance:.3f} expected_counts={float(expected[idx]):.3f} "
                f"response_per_cps_1m={per_cps_1m:.6g} "
                f"obstacle_path_cm={obstacle_path_cm:.3f} "
                f"obstacle_tau={obstacle_tau:.6g} "
                f"obstacle_att={obstacle_att:.6g} "
                f"fe={int(fe_indices[0])} pb={int(pb_indices[0])}"
            )


def _log_posterior_truth_mass_diagnostics(
    estimator: RotatingShieldPFEstimator,
    true_sources: dict[str, NDArray[np.float64]],
    true_strengths: dict[str, float | Sequence[float]],
    *,
    step_index: int,
) -> None:
    """Log posterior mass and source slots near each true source."""
    radii = (1.0, 2.0, 3.0)
    for isotope, truth_positions_raw in sorted(true_sources.items()):
        filt = estimator.filters.get(isotope)
        if filt is None or not getattr(filt, "continuous_particles", None):
            continue
        truth_positions = np.asarray(truth_positions_raw, dtype=float).reshape(-1, 3)
        if truth_positions.size == 0:
            continue
        truth_strength_arr = _true_strength_array(
            true_strengths,
            isotope,
            truth_positions.shape[0],
        )
        weights = np.asarray(filt.continuous_weights, dtype=float).reshape(-1)
        if (
            weights.size != len(filt.continuous_particles)
            or float(np.sum(weights)) <= 0.0
        ):
            weights = np.ones(len(filt.continuous_particles), dtype=float)
            weights /= max(float(weights.size), 1.0)
        map_positions, map_strengths, _ = _state_source_arrays(
            _active_state_for_diagnostics(filt)
        )
        for truth_idx, truth_pos in enumerate(truth_positions):
            mass_by_radius = {radius: 0.0 for radius in radii}
            weighted_nn_distance = 0.0
            weighted_nn_strength = 0.0
            weighted_source_count = 0.0
            no_source_mass = 0.0
            finite_nn_mass = 0.0
            for weight, particle in zip(weights, filt.continuous_particles):
                state = particle.state
                positions, strengths, _ = _state_source_arrays(state)
                if positions.size == 0:
                    no_source_mass += float(weight)
                    continue
                distances = np.linalg.norm(positions - truth_pos[None, :], axis=1)
                nearest_idx = int(np.argmin(distances))
                nearest_distance = float(distances[nearest_idx])
                for radius in radii:
                    if np.any(distances <= radius):
                        mass_by_radius[radius] += float(weight)
                weighted_nn_distance += float(weight) * nearest_distance
                weighted_nn_strength += float(weight) * float(strengths[nearest_idx])
                weighted_source_count += float(weight) * float(positions.shape[0])
                finite_nn_mass += float(weight)
            if finite_nn_mass > 0.0:
                weighted_nn_distance /= finite_nn_mass
                weighted_nn_strength /= finite_nn_mass
            else:
                weighted_nn_distance = float("nan")
                weighted_nn_strength = float("nan")
            map_nn_distance = None
            map_nn_strength = None
            if map_positions.size:
                map_distances = np.linalg.norm(
                    map_positions - truth_pos[None, :],
                    axis=1,
                )
                map_idx = int(np.argmin(map_distances))
                map_nn_distance = float(map_distances[map_idx])
                map_nn_strength = float(map_strengths[map_idx])
            q_true = (
                float(truth_strength_arr[truth_idx])
                if truth_idx < truth_strength_arr.size
                else 0.0
            )
            print(
                f"[step {step_index}] posterior_truth_mass[{isotope}] "
                f"truth_idx={truth_idx} pos={_fmt_pos(truth_pos)} q_true={q_true:.3f} "
                f"mass_1m={mass_by_radius[1.0]:.4f} "
                f"mass_2m={mass_by_radius[2.0]:.4f} "
                f"mass_3m={mass_by_radius[3.0]:.4f} "
                f"weighted_nn_d={_fmt_optional_float(weighted_nn_distance, 3)}m "
                f"weighted_nn_q={_fmt_optional_float(weighted_nn_strength, 3)} "
                f"weighted_source_count={weighted_source_count:.3f} "
                f"no_source_mass={no_source_mass:.4f} "
                f"map_nn_d={_fmt_optional_float(map_nn_distance, 3)}m "
                f"map_nn_q={_fmt_optional_float(map_nn_strength, 3)}"
            )


def _metadata_value_array(
    state: object,
    name: str,
    count: int,
    *,
    fill: float | int | bool,
    dtype: object,
) -> NDArray[Any]:
    """Return a fixed-length per-source metadata array for a particle state."""
    values = np.asarray(getattr(state, name, np.zeros(0)), dtype=dtype).reshape(-1)
    out = np.full(max(count, 0), fill, dtype=dtype)
    if count > 0 and values.size > 0:
        out[: min(count, values.size)] = values[:count]
    return out


def _particle_source_payload(
    isotope: str,
    source_idx: int,
    pos: NDArray[np.float64],
    strength: float,
    surface: str,
    state: object,
    true_sources: dict[str, NDArray[np.float64]],
    true_strengths: dict[str, float | Sequence[float]],
) -> dict[str, object]:
    """Return JSON-serializable diagnostics for one particle source slot."""
    count = max(0, int(getattr(state, "num_sources", 0)))
    ages = _metadata_value_array(state, "ages", count, fill=0, dtype=int)
    support = _metadata_value_array(
        state,
        "support_scores",
        count,
        fill=0.0,
        dtype=float,
    )
    tentative = _metadata_value_array(
        state,
        "tentative_sources",
        count,
        fill=False,
        dtype=bool,
    )
    fail_streak = _metadata_value_array(
        state,
        "verification_fail_streaks",
        count,
        fill=0,
        dtype=int,
    )
    truth_diag = _nearest_truth_diagnostic(
        isotope,
        np.asarray(pos, dtype=float).reshape(3),
        true_sources,
        true_strengths,
    )
    payload: dict[str, object] = {
        "slot": int(source_idx),
        "pos": [round(float(v), 6) for v in np.asarray(pos, dtype=float).reshape(3)],
        "q": float(strength),
        "surface": str(surface),
        "age": int(ages[source_idx]) if source_idx < ages.size else 0,
        "support": float(support[source_idx]) if source_idx < support.size else 0.0,
        "tentative": bool(tentative[source_idx])
        if source_idx < tentative.size
        else False,
        "verification_fail_streak": int(fail_streak[source_idx])
        if source_idx < fail_streak.size
        else 0,
    }
    payload.update(truth_diag)
    return payload


def _weighted_mean_std(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return weighted mean and standard deviation along the first axis."""
    arr = np.asarray(values, dtype=float)
    weight_arr = np.asarray(weights, dtype=float).reshape(-1)
    if arr.shape[0] == 0 or weight_arr.size != arr.shape[0]:
        return np.zeros(arr.shape[1:], dtype=float), np.zeros(
            arr.shape[1:], dtype=float
        )
    total = max(float(np.sum(weight_arr)), 1.0e-300)
    normalized = weight_arr / total
    reshape = (-1,) + (1,) * (arr.ndim - 1)
    mean = np.sum(arr * normalized.reshape(reshape), axis=0)
    variance = np.sum(((arr - mean) ** 2) * normalized.reshape(reshape), axis=0)
    return np.asarray(mean, dtype=float), np.sqrt(np.maximum(variance, 0.0))


def _log_particle_cloud_diagnostics(
    estimator: RotatingShieldPFEstimator,
    true_sources: dict[str, NDArray[np.float64]],
    true_strengths: dict[str, float | Sequence[float]],
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    *,
    obstacle_height_m: float,
    step_index: int,
    particle_log_limit: int,
) -> None:
    """Log posterior particle-cloud positions, strengths, and slot-wise spread."""
    for isotope, filt in sorted(estimator.filters.items()):
        particles = list(getattr(filt, "continuous_particles", []))
        if not particles:
            print(f"[step {step_index}] particle_cloud[{isotope}]: no particles")
            continue
        weights = np.asarray(filt.continuous_weights, dtype=float).reshape(-1)
        if weights.size != len(particles) or float(np.sum(weights)) <= 0.0:
            weights = np.ones(len(particles), dtype=float)
            weights /= max(float(weights.size), 1.0)
        weights = weights / max(float(np.sum(weights)), 1.0e-300)
        source_counts = np.asarray(
            [
                max(0, int(getattr(particle.state, "num_sources", 0)))
                for particle in particles
            ],
            dtype=int,
        )
        source_count_mass: dict[str, float] = {}
        for count in sorted(set(int(v) for v in source_counts.tolist())):
            source_count_mass[str(count)] = float(
                np.sum(weights[source_counts == count])
            )
        ess = float(1.0 / max(float(np.sum(weights**2)), 1.0e-300))
        order = np.argsort(weights)[::-1]
        log_order = _diagnostic_detail_order(order, particle_log_limit)
        print(
            f"[step {step_index}] particle_cloud[{isotope}] "
            f"particles={len(particles)} logged_particles={int(log_order.size)} "
            f"ess={ess:.3f} ess_ratio={ess / max(len(particles), 1):.4f} "
            f"max_weight={float(np.max(weights)):.6g} "
            f"source_count_mean={float(np.sum(weights * source_counts)):.3f} "
            f"source_count_mass={_safe_json_dumps(source_count_mass)}"
        )
        if int(particle_log_limit) == 0:
            continue
        max_slots = int(np.max(source_counts)) if source_counts.size else 0
        for slot_idx in range(max_slots):
            slot_positions: list[NDArray[np.float64]] = []
            slot_strengths: list[float] = []
            slot_weights: list[float] = []
            slot_runtime_weights: list[float] = []
            slot_ages: list[float] = []
            slot_tentative_weights: list[float] = []
            for particle_idx, particle in enumerate(particles):
                state = particle.state
                raw_positions = np.asarray(
                    getattr(state, "positions", np.zeros((0, 3))),
                    dtype=float,
                ).reshape(-1, 3)
                raw_strengths = np.asarray(
                    getattr(state, "strengths", np.zeros(0)),
                    dtype=float,
                ).reshape(-1)
                count = min(
                    max(0, int(getattr(state, "num_sources", 0))),
                    raw_positions.shape[0],
                    raw_strengths.size,
                )
                if slot_idx >= count:
                    continue
                source_weight = float(weights[particle_idx])
                slot_positions.append(raw_positions[slot_idx])
                slot_strengths.append(float(raw_strengths[slot_idx]))
                slot_weights.append(source_weight)
                slot_runtime_weights.append(source_weight)
                ages = _metadata_value_array(state, "ages", count, fill=0, dtype=int)
                tentative = _metadata_value_array(
                    state,
                    "tentative_sources",
                    count,
                    fill=False,
                    dtype=bool,
                )
                slot_ages.append(float(ages[slot_idx]) if slot_idx < ages.size else 0.0)
                slot_tentative_weights.append(
                    source_weight
                    if slot_idx < tentative.size and bool(tentative[slot_idx])
                    else 0.0
                )
            if not slot_positions:
                continue
            pos_arr = np.vstack(slot_positions)
            strength_arr = np.asarray(slot_strengths, dtype=float)
            weight_arr = np.asarray(slot_weights, dtype=float)
            mean_pos, std_pos = _weighted_mean_std(pos_arr, weight_arr)
            mean_q, std_q = _weighted_mean_std(strength_arr[:, None], weight_arr)
            age_mean, _age_std = _weighted_mean_std(
                np.asarray(slot_ages, dtype=float)[:, None],
                weight_arr,
            )
            surfaces = source_surface_kinds(
                pos_arr,
                env,
                obstacle_grid,
                obstacle_height_m=obstacle_height_m,
                tolerance_m=1.0e-5,
            )
            surface_mass: dict[str, float] = {}
            for surface, weight in zip(surfaces, weight_arr):
                key = str(surface)
                surface_mass[key] = surface_mass.get(key, 0.0) + float(weight)
            truth_diag = _nearest_truth_diagnostic(
                isotope,
                mean_pos,
                true_sources,
                true_strengths,
            )
            print(
                f"[step {step_index}] particle_slot_cloud[{isotope}] "
                f"slot={slot_idx} source_mass={float(np.sum(weight_arr)):.6f} "
                f"runtime_active_mass={float(np.sum(slot_runtime_weights)):.6f} "
                f"tentative_mass={float(np.sum(slot_tentative_weights)):.6f} "
                f"samples={len(slot_positions)} "
                f"mean_pos={_fmt_pos(mean_pos)} std_pos={_fmt_pos(std_pos)} "
                f"min_pos={_fmt_pos(np.min(pos_arr, axis=0))} "
                f"max_pos={_fmt_pos(np.max(pos_arr, axis=0))} "
                f"mean_q={float(mean_q[0]):.3f} std_q={float(std_q[0]):.3f} "
                f"mean_age={float(age_mean[0]):.3f} "
                f"surface_mass={_safe_json_dumps(surface_mass)} "
                f"nearest_truth={truth_diag.get('nearest_truth_index')} "
                f"truth_d={_fmt_optional_float(truth_diag.get('nearest_truth_distance_m'))}m "
                f"truth_q={_fmt_optional_float(truth_diag.get('nearest_truth_strength'))}"
            )
        for rank, particle_idx_raw in enumerate(log_order, start=1):
            particle_idx = int(particle_idx_raw)
            particle = particles[particle_idx]
            state = particle.state
            positions, strengths, background = _state_source_arrays(state)
            surfaces = (
                source_surface_kinds(
                    positions,
                    env,
                    obstacle_grid,
                    obstacle_height_m=obstacle_height_m,
                    tolerance_m=1.0e-5,
                )
                if positions.size
                else []
            )
            sources = [
                _particle_source_payload(
                    isotope,
                    source_idx,
                    positions[source_idx],
                    float(strengths[source_idx]),
                    str(surfaces[source_idx])
                    if source_idx < len(surfaces)
                    else "unknown",
                    state,
                    true_sources,
                    true_strengths,
                )
                for source_idx in range(positions.shape[0])
            ]
            print(
                f"[step {step_index}] particle_source[{isotope}] "
                f"rank={rank} particle_idx={particle_idx} "
                f"weight={float(weights[particle_idx]):.8g} "
                f"log_weight={float(getattr(particle, 'log_weight', 0.0)):.8g} "
                f"source_count={positions.shape[0]} background={background:.6g} "
                f"sources={_safe_json_dumps(sources)}"
            )


def _candidate_max_weighted_correlation(
    candidate_counts: NDArray[np.float64],
    existing_counts: NDArray[np.float64],
    observation_variances: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return maximum whitened response correlation for each candidate."""
    candidates = np.maximum(np.asarray(candidate_counts, dtype=float), 0.0)
    existing = np.maximum(np.asarray(existing_counts, dtype=float), 0.0)
    if candidates.ndim != 2 or candidates.size == 0:
        return np.zeros(0, dtype=float)
    if existing.ndim != 2 or existing.size == 0 or existing.shape[1] == 0:
        return np.zeros(candidates.shape[1], dtype=float)
    variances = np.asarray(observation_variances, dtype=float).reshape(-1)
    if variances.size != candidates.shape[0]:
        variances = np.ones(candidates.shape[0], dtype=float)
    scale = 1.0 / np.sqrt(np.maximum(variances, 1.0e-12))
    cand_w = candidates * scale[:, None]
    exist_w = existing * scale[:, None]
    cand_norm = np.maximum(np.linalg.norm(cand_w, axis=0), 1.0e-12)
    exist_norm = np.maximum(np.linalg.norm(exist_w, axis=0), 1.0e-12)
    corr = np.abs(exist_w.T @ cand_w) / (exist_norm[:, None] * cand_norm[None, :])
    if corr.size == 0:
        return np.zeros(candidates.shape[1], dtype=float)
    return np.asarray(np.max(corr, axis=0), dtype=float)


def _birth_support_detail_counts(
    filt: object,
    *,
    candidate_counts: NDArray[np.float64],
    residual: NDArray[np.float64],
    observation_variances: NDArray[np.float64],
    detector_positions: NDArray[np.float64],
    fe_indices: NDArray[np.int64],
    pb_indices: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Return raw support, distinct-view, and distinct-station counts."""
    counts = np.asarray(candidate_counts, dtype=float)
    if counts.ndim != 2 or counts.size == 0:
        empty = np.zeros(0, dtype=int)
        return empty, empty, empty
    residual_arr = np.asarray(residual, dtype=float).reshape(-1)
    variances = np.asarray(observation_variances, dtype=float).reshape(-1)
    if residual_arr.size != counts.shape[0]:
        residual_arr = np.zeros(counts.shape[0], dtype=float)
    if variances.size != counts.shape[0]:
        variances = np.ones(counts.shape[0], dtype=float)
    sigma = np.sqrt(np.maximum(variances, 1.0e-12))
    threshold_sigma = max(
        float(getattr(filt.config, "birth_residual_support_sigma", 0.0)),
        0.0,
    )
    residual_support = residual_arr / sigma >= threshold_sigma
    if not np.any(residual_support):
        empty = np.zeros(counts.shape[1], dtype=int)
        return empty, empty, empty
    overlap = np.maximum(counts, 0.0) * residual_arr[:, None]
    max_overlap = np.max(overlap, axis=0)
    fraction = float(
        np.clip(
            getattr(filt.config, "birth_candidate_support_fraction", 0.05), 0.0, 1.0
        )
    )
    support = (overlap >= max_overlap[None, :] * fraction) & (
        max_overlap[None, :] > 0.0
    )
    support &= residual_support[:, None]
    raw_counts = np.sum(support, axis=0).astype(int)
    view_labels = filt._support_view_labels(
        detector_positions,
        fe_indices,
        pb_indices,
        counts.shape[0],
    )
    station_labels = filt._support_station_labels(
        detector_positions,
        counts.shape[0],
    )
    view_counts = filt._distinct_label_counts_for_support_matrix(
        support,
        view_labels,
    )
    station_counts = filt._distinct_label_counts_for_support_matrix(
        support,
        station_labels,
    )
    return raw_counts, view_counts, station_counts


def _birth_rejection_reason(
    *,
    support: bool,
    distance: bool,
    score_valid: bool,
) -> str:
    """Return the first failed birth-candidate diagnostic gate."""
    if not support:
        return "support_gate_failed"
    if not distance:
        return "distance_gate_failed"
    if not score_valid:
        return "invalid_residual_score"
    return "kept"


def _log_birth_candidate_diagnostics(
    estimator: RotatingShieldPFEstimator,
    true_sources: dict[str, NDArray[np.float64]],
    true_strengths: dict[str, float | Sequence[float]],
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    *,
    obstacle_height_m: float,
    step_index: int,
    candidate_log_limit: int,
) -> None:
    """Log residual-birth candidate scores and gate outcomes."""
    candidates_all = np.asarray(
        getattr(estimator, "candidate_sources", np.zeros((0, 3))),
        dtype=float,
    ).reshape(-1, 3)
    if candidates_all.size == 0:
        print(f"[step {step_index}] birth_candidates: no candidate source grid")
        return
    for isotope, filt in sorted(estimator.filters.items()):
        data = estimator._measurement_data_for_iso(
            isotope,
            None,
        )
        if data is None or data.z_k.size == 0:
            print(f"[step {step_index}] birth_candidates[{isotope}]: no data")
            continue
        state = _active_state_for_diagnostics(filt)
        positions, strengths, background = _state_source_arrays(state)
        existing_unit = (
            filt._unit_response_counts_for_state(state, data)
            if state is not None
            else np.zeros((data.z_k.size, 0), dtype=float)
        )
        source_count = min(int(existing_unit.shape[1]), int(strengths.size))
        existing_unit = existing_unit[:, :source_count]
        strengths = strengths[:source_count]
        positions = positions[:source_count]
        lambda_total = (
            np.asarray(data.live_times, dtype=float) * float(background)
            + existing_unit @ strengths
        )
        residual = np.maximum(np.asarray(data.z_k, dtype=float) - lambda_total, 0.0)
        residual_sum = float(np.sum(residual))
        candidate_counts = expected_counts_per_source(
            kernel=filt.continuous_kernel,
            isotope=isotope,
            detector_positions=data.detector_positions,
            sources=candidates_all,
            strengths=np.ones(candidates_all.shape[0], dtype=float),
            live_times=data.live_times,
            fe_indices=data.fe_indices,
            pb_indices=data.pb_indices,
            source_scale=filt._measurement_source_scale_vector(
                data.fe_indices,
                data.pb_indices,
            ),
        )
        support_mask = filt._birth_candidate_support_mask(
            candidate_counts=candidate_counts,
            residual_mix=residual,
            observation_variances=data.observation_variances,
            detector_positions=data.detector_positions,
            fe_indices=data.fe_indices,
            pb_indices=data.pb_indices,
        )
        distance_mask = np.ones(candidates_all.shape[0], dtype=bool)
        if positions.size:
            distances = np.linalg.norm(
                candidates_all[:, None, :] - positions[None, :, :],
                axis=2,
            )
            distance_mask = np.min(distances, axis=1) >= float(
                getattr(filt.config, "birth_min_sep_m", 0.0)
            )
        scores, q_hat = filt._birth_residual_candidate_scores(
            candidate_counts=candidate_counts,
            residual_mix=residual,
            observation_variances=data.observation_variances,
        )
        score_valid = (
            np.isfinite(scores) & np.isfinite(q_hat) & (scores > 0.0) & (q_hat > 0.0)
        )
        keep = support_mask & distance_mask & score_valid
        support_counts, view_counts, station_counts = _birth_support_detail_counts(
            filt,
            candidate_counts=candidate_counts,
            residual=residual,
            observation_variances=data.observation_variances,
            detector_positions=data.detector_positions,
            fe_indices=data.fe_indices,
            pb_indices=data.pb_indices,
        )
        max_corr = _candidate_max_weighted_correlation(
            candidate_counts,
            existing_unit,
            data.observation_variances,
        )
        order = _diagnostic_detail_order(
            np.argsort(np.where(np.isfinite(scores), scores, -np.inf))[::-1],
            candidate_log_limit,
        )
        surface_kinds = source_surface_kinds(
            candidates_all,
            env,
            obstacle_grid,
            obstacle_height_m=obstacle_height_m,
            tolerance_m=1.0e-5,
        )
        print(
            f"[step {step_index}] birth_candidates[{isotope}] "
            f"candidate_count={candidates_all.shape[0]} "
            f"logged={order.size} residual_sum={residual_sum:.3f} "
            f"residual_max={float(np.max(residual)) if residual.size else 0.0:.3f} "
            f"support_pass={int(np.sum(support_mask))} "
            f"distance_pass={int(np.sum(distance_mask))} "
            f"score_valid={int(np.sum(score_valid))} "
            f"kept={int(np.sum(keep))} "
            f"existing_sources={positions.shape[0]} "
            f"window_measurements={data.z_k.size}"
        )
        for rank, cand_idx in enumerate(order, start=1):
            idx = int(cand_idx)
            pos = candidates_all[idx]
            reason = _birth_rejection_reason(
                support=bool(support_mask[idx]),
                distance=bool(distance_mask[idx]),
                score_valid=bool(score_valid[idx]),
            )
            truth_diag = _nearest_truth_diagnostic(
                isotope,
                pos,
                true_sources,
                true_strengths,
            )
            surface_kind = (
                str(surface_kinds[idx]) if idx < len(surface_kinds) else "unknown"
            )
            print(
                f"[step {step_index}] birth_candidate[{isotope}] "
                f"rank={rank} idx={idx} pos={_fmt_pos(pos)} "
                f"surface={surface_kind} score={float(scores[idx]):.6g} "
                f"q_hat={float(q_hat[idx]):.6g} kept={bool(keep[idx])} "
                f"reason={reason} support={bool(support_mask[idx])} "
                f"support_count={int(support_counts[idx]) if idx < support_counts.size else 0} "
                f"view_count={int(view_counts[idx]) if idx < view_counts.size else 0} "
                f"station_count={int(station_counts[idx]) if idx < station_counts.size else 0} "
                f"max_response_corr={float(max_corr[idx]):.6g} "
                f"distance_pass={bool(distance_mask[idx])} "
                f"nearest_truth={truth_diag.get('nearest_truth_index')} "
                f"truth_d={_fmt_optional_float(truth_diag.get('nearest_truth_distance_m'))}m "
                f"truth_q={_fmt_optional_float(truth_diag.get('nearest_truth_strength'))}"
            )


def _log_source_event_diagnostics(
    estimator: RotatingShieldPFEstimator,
    true_sources: dict[str, NDArray[np.float64]],
    true_strengths: dict[str, float | Sequence[float]],
    *,
    step_index: int,
    event_log_limit: int = 64,
) -> None:
    """Log bounded source-event details plus a complete deterministic summary."""
    any_event = False
    for isotope, filt in sorted(estimator.filters.items()):
        events = [
            dict(event)
            for event in getattr(filt, "last_source_event_diagnostics", [])
        ]
        if not events:
            continue
        any_event = True
        event_counts = Counter(
            str(event.get("event", "unspecified")) for event in events
        )
        reason_counts = Counter(
            str(event.get("reason", "unspecified")) for event in events
        )
        event_digest = hashlib.sha256()
        for event in events:
            event_digest.update(_safe_json_dumps(event).encode("utf-8"))
            event_digest.update(b"\n")
        detail_indices = _diagnostic_detail_order(
            np.arange(len(events), dtype=np.int64),
            event_log_limit,
        )
        print(
            f"[step {step_index}] source_event_summary[{isotope}] "
            f"total={len(events)} logged={int(detail_indices.size)} "
            f"omitted={len(events) - int(detail_indices.size)} "
            f"raw_sha256={event_digest.hexdigest()} "
            "event_counts="
            f"{_safe_json_dumps(dict(sorted(event_counts.items())))} "
            "reason_counts="
            f"{_safe_json_dumps(dict(sorted(reason_counts.items())))}"
        )
        for raw_event_idx in detail_indices:
            event_idx = int(raw_event_idx)
            event = dict(events[event_idx])
            position = np.asarray(event.get("position", np.zeros(3)), dtype=float)
            if position.size >= 3:
                event.update(
                    _nearest_truth_diagnostic(
                        isotope,
                        position[:3],
                        true_sources,
                        true_strengths,
                    )
                )
            print(
                f"[step {step_index}] source_event[{isotope}] "
                f"event_idx={event_idx} {_safe_json_dumps(event)}"
            )
    if not any_event:
        print(f"[step {step_index}] source_event: none")


def _run_precision_diagnostic_block(
    label: str,
    callback: Callable[[], None],
) -> None:
    """Run one diagnostic callback and log failures without interrupting the simulation."""
    try:
        callback()
    except Exception as exc:  # pragma: no cover - defensive logging path
        print(f"precision_diagnostic_error[{label}]: {type(exc).__name__}: {exc}")


def _format_pf_timing_item(key: str, value: float) -> str:
    """Format PF structural diagnostics without treating counters as seconds."""
    key_str = str(key)
    numeric = float(value)
    if (
        key_str == "total"
        or key_str.endswith("_s")
        or "wall" in key_str
        or "elapsed" in key_str
        or "duration" in key_str
    ):
        return f"{key_str}={numeric:.3f}s"
    if np.isfinite(numeric) and abs(numeric - round(numeric)) < 1.0e-9:
        return f"{key_str}={int(round(numeric))}"
    return f"{key_str}={numeric:.3f}"


def _diagnostic_detail_order(
    ordered_indices: NDArray[np.integer] | Sequence[int],
    limit: int,
) -> NDArray[np.int64]:
    """Return diagnostic detail indices using 0=none, positive=N, negative=all."""
    order = np.asarray(ordered_indices, dtype=np.int64).reshape(-1)
    limit_int = int(limit)
    if limit_int < 0:
        return order
    if limit_int == 0:
        return order[:0]
    return order[: min(limit_int, order.size)]


def _log_precision_degradation_diagnostics(
    estimator: RotatingShieldPFEstimator,
    decomposer: SpectralDecomposer,
    measurement: Measurement | None,
    true_sources: dict[str, NDArray[np.float64]],
    true_strengths: dict[str, float | Sequence[float]],
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    *,
    obstacle_height_m: float,
    step_index: int,
    candidate_log_limit: int,
    particle_log_limit: int,
    source_event_log_limit: int = 64,
    birth_candidate_diagnostics_enabled: bool = False,
    full_spectrum_response_diagnostics_enabled: bool = False,
) -> None:
    """Log high-detail diagnostics for identifying PF accuracy degradation."""
    if bool(full_spectrum_response_diagnostics_enabled):
        _run_precision_diagnostic_block(
            "spectrum_response_poisson",
            lambda: _log_spectrum_response_poisson_diagnostics(
                decomposer,
                step_index=step_index,
            ),
        )
    else:
        selected_counts = (
            {
                str(key): float(value)
                for key, value in measurement.counts_by_isotope.items()
            }
            if measurement is not None
            else {}
        )
        selected_variances = (
            {
                str(key): float(value)
                for key, value in measurement.count_variance_by_isotope.items()
            }
            if measurement is not None
            and measurement.count_variance_by_isotope is not None
            else {}
        )
        _run_precision_diagnostic_block(
            "spectrum_isotope_channels",
            lambda: _log_spectrum_isotope_channel_diagnostics(
                decomposer,
                step_index=step_index,
                selected_counts=selected_counts,
                selected_variances=selected_variances,
            ),
        )
    if measurement is None:
        print(f"[step {step_index}] precision_diagnostics: no latest measurement")
    else:
        _run_precision_diagnostic_block(
            "map_prediction_residuals",
            lambda: _log_current_map_prediction_residuals(
                estimator,
                measurement,
                true_sources,
                true_strengths,
                env,
                obstacle_grid,
                obstacle_height_m=obstacle_height_m,
                step_index=step_index,
            ),
        )
        _run_precision_diagnostic_block(
            "truth_observability",
            lambda: _log_truth_observability_diagnostics(
                estimator,
                measurement,
                true_sources,
                true_strengths,
                env,
                obstacle_grid,
                obstacle_height_m=obstacle_height_m,
                step_index=step_index,
            ),
        )
    _run_precision_diagnostic_block(
        "posterior_truth_mass",
        lambda: _log_posterior_truth_mass_diagnostics(
            estimator,
            true_sources,
            true_strengths,
            step_index=step_index,
        ),
    )
    _run_precision_diagnostic_block(
        "particle_cloud",
        lambda: _log_particle_cloud_diagnostics(
            estimator,
            true_sources,
            true_strengths,
            env,
            obstacle_grid,
            obstacle_height_m=obstacle_height_m,
            step_index=step_index,
            particle_log_limit=particle_log_limit,
        ),
    )
    _run_precision_diagnostic_block(
        "source_events",
        lambda: _log_source_event_diagnostics(
            estimator,
            true_sources,
            true_strengths,
            step_index=step_index,
            event_log_limit=source_event_log_limit,
        ),
    )
    if bool(birth_candidate_diagnostics_enabled) and int(candidate_log_limit) != 0:
        _run_precision_diagnostic_block(
            "birth_candidates",
            lambda: _log_birth_candidate_diagnostics(
                estimator,
                true_sources,
                true_strengths,
                env,
                obstacle_grid,
                obstacle_height_m=obstacle_height_m,
                step_index=step_index,
                candidate_log_limit=candidate_log_limit,
            ),
        )


def _resolve_ig_threshold(
    mode: str,
    ig_floor: float,
    ig_rel: float,
    ig_max_global: float,
    ig_max_pose: float,
) -> float:
    """Return the active IG threshold for the selected mode."""
    mode = mode.lower()
    if mode == "absolute":
        return float(ig_floor)
    if mode == "relative_max":
        return float(max(ig_floor, ig_rel * ig_max_global))
    if mode == "relative_pose":
        return float(max(ig_floor, ig_rel * ig_max_pose))
    raise ValueError(f"Unknown IG threshold mode: {mode}")


def _default_use_gpu() -> bool:
    """Return True if CUDA is available for torch acceleration."""
    try:
        from pf import gpu_utils
    except ImportError:
        return False
    return gpu_utils.torch_available()


def _resolve_runtime_use_gpu(runtime_config: Mapping[str, object]) -> bool:
    """Return the configured GPU policy, defaulting to automatic CUDA detection."""
    configured = _optional_runtime_bool(runtime_config, "use_gpu")
    if configured is not None:
        return configured
    return _default_use_gpu()


def _resolve_python_worker_count(worker_count: object | None) -> int:
    """Return a Python CPU worker count, using all logical CPUs for auto."""
    if worker_count is None:
        return max(1, os.cpu_count() or 1)
    workers = int(worker_count)
    if workers <= 0:
        cpu_count = os.cpu_count() or 1
        return max(1, cpu_count)
    return max(1, workers)


def _resolve_ig_workers(ig_workers: int | None) -> int:
    """Return a worker count for IG evaluation (0 or None means all CPUs)."""
    return _resolve_python_worker_count(ig_workers)


def _coerce_live_visualization(live: bool) -> bool:
    """Return whether live Matplotlib updates can run in this process."""
    if not live:
        return False
    backend = str(matplotlib.get_backend()).lower()
    if "agg" in backend or not _has_display():
        print("GUI display unavailable; running in CUI/headless mode.")
        return False
    return True


def _resolve_structural_trial_parallelism(
    runtime_config: Mapping[str, object],
) -> tuple[int, int]:
    """Return PF structural trial worker count and minimum trial threshold."""
    try:
        workers = int(runtime_config.get("structural_trial_workers", 1))
    except (TypeError, ValueError):
        workers = 1
    try:
        min_trials = int(runtime_config.get("structural_trial_parallel_min_trials", 8))
    except (TypeError, ValueError):
        min_trials = 8
    return max(1, workers), max(1, min_trials)


def _resolve_plot_save_interval(
    runtime_config: dict[str, object],
    key: str,
    *,
    default: int = 1,
    allow_disable: bool = False,
) -> int:
    """Return a plot-save interval from runtime config."""
    try:
        interval = int(runtime_config.get(key, default))
    except (TypeError, ValueError):
        interval = int(default)
    if allow_disable and interval <= 0:
        return 0
    return max(1, interval)


def _spectrum_config_from_runtime_config(
    runtime_config: dict[str, object],
) -> SpectrumConfig:
    """Build a spectrum config from runtime keys while preserving defaults."""
    return build_spectrum_config_from_runtime_config(runtime_config)


@dataclass(frozen=True)
class DetectorHeightPlanningConfig:
    """Describe the detector mast workspace used by pose planning."""

    mode: str
    ground_z_m: float
    initial_mast_height_m: float
    minimum_mast_height_m: float
    maximum_mast_height_m: float
    discrete_mast_actions_m: tuple[float, ...] = ()

    @property
    def initial_world_z_m(self) -> float:
        """Return the initial detector height in world coordinates."""
        return float(self.ground_z_m + self.initial_mast_height_m)

    @property
    def minimum_world_z_m(self) -> float:
        """Return the minimum detector world height."""
        return float(self.ground_z_m + self.minimum_mast_height_m)

    @property
    def maximum_world_z_m(self) -> float:
        """Return the maximum detector world height."""
        return float(self.ground_z_m + self.maximum_mast_height_m)

    @property
    def discrete_world_actions_m(self) -> tuple[float, ...]:
        """Return configured discrete world heights, or an empty tuple."""
        return tuple(
            float(self.ground_z_m + value) for value in self.discrete_mast_actions_m
        )

    @property
    def candidate_world_heights_m(self) -> tuple[float, ...] | None:
        """Return discrete candidate heights or None for continuous sampling."""
        if self.mode == "continuous":
            return None
        return self.discrete_world_actions_m

    @property
    def candidate_world_z_bounds_m(self) -> tuple[float, float]:
        """Return the z interval sampled by the pose candidate generator."""
        if self.mode == "continuous":
            return self.minimum_world_z_m, self.maximum_world_z_m
        actions = self.discrete_world_actions_m
        return float(min(actions)), float(max(actions))


def _resolve_detector_height_planning_config(
    runtime_config: Mapping[str, object],
    *,
    room_height_m: float,
) -> DetectorHeightPlanningConfig:
    """Resolve continuous or legacy discrete detector-height planning settings."""
    room_height = float(room_height_m)
    ground_z = float(runtime_config.get("robot_ground_z_m", 0.0))
    if not np.isfinite(ground_z):
        raise ValueError("robot_ground_z_m must be finite.")
    initial_mast_height = float(runtime_config.get("detector_height_m", 0.5))
    minimum_mast_height = max(
        0.0,
        float(runtime_config.get("detector_height_min_m", 0.0)),
    )
    maximum_mast_height = min(
        max(room_height - ground_z, 0.0),
        float(
            runtime_config.get(
                "detector_height_max_m",
                max(room_height - ground_z, 0.0),
            )
        ),
    )
    if not np.isfinite(initial_mast_height):
        raise ValueError("detector_height_m must be finite.")
    if not np.isfinite(minimum_mast_height) or not np.isfinite(maximum_mast_height):
        raise ValueError("detector height bounds must be finite.")
    if maximum_mast_height < minimum_mast_height:
        raise ValueError("detector_height_max_m must be >= detector_height_min_m.")
    if not minimum_mast_height <= initial_mast_height <= maximum_mast_height:
        raise ValueError(
            "detector_height_m must lie within the detector height bounds."
        )
    height_payload = runtime_config.get(
        "detector_height_actions_m",
        runtime_config.get("detector_heights_m"),
    )
    raw_mode = runtime_config.get("detector_height_sampling_mode")
    if raw_mode is None:
        mode = "discrete"
    else:
        mode = str(raw_mode).strip().lower().replace("-", "_")
    aliases = {
        "continuous": "continuous",
        "continuous_sobol": "continuous",
        "sobol": "continuous",
        "discrete": "discrete",
        "fixed": "discrete",
    }
    if mode not in aliases:
        raise ValueError(
            "detector_height_sampling_mode must be 'continuous' or 'discrete'."
        )
    mode = aliases[mode]
    if mode == "continuous":
        if height_payload is not None:
            raise ValueError(
                "detector_height_actions_m must be omitted when continuous detector "
                "height sampling is enabled."
            )
        return DetectorHeightPlanningConfig(
            mode=mode,
            ground_z_m=ground_z,
            initial_mast_height_m=initial_mast_height,
            minimum_mast_height_m=minimum_mast_height,
            maximum_mast_height_m=maximum_mast_height,
        )
    mast_actions = resolve_detector_height_actions(
        height_payload,
        default_height_m=initial_mast_height,
        bounds_z=(minimum_mast_height, maximum_mast_height),
    )
    if not np.any(np.isclose(mast_actions, initial_mast_height)):
        mast_actions = resolve_detector_height_actions(
            [*mast_actions.tolist(), initial_mast_height],
            default_height_m=initial_mast_height,
            bounds_z=(minimum_mast_height, maximum_mast_height),
        )
    world_actions = np.asarray(mast_actions, dtype=float) + ground_z
    if np.any(world_actions < 0.0) or np.any(world_actions > room_height):
        raise ValueError(
            "Detector mast-height actions plus robot_ground_z_m must lie inside the room."
        )
    return DetectorHeightPlanningConfig(
        mode=mode,
        ground_z_m=ground_z,
        initial_mast_height_m=initial_mast_height,
        minimum_mast_height_m=minimum_mast_height,
        maximum_mast_height_m=maximum_mast_height,
        discrete_mast_actions_m=tuple(float(value) for value in mast_actions),
    )


def _resolve_detector_height_world_actions(
    runtime_config: Mapping[str, object],
    *,
    room_height_m: float,
) -> tuple[float, float, NDArray[np.float64], NDArray[np.float64]]:
    """Resolve legacy discrete mast settings into world-z planning actions."""
    config = _resolve_detector_height_planning_config(
        runtime_config,
        room_height_m=room_height_m,
    )
    if config.mode != "discrete":
        raise ValueError(
            "_resolve_detector_height_world_actions only supports discrete mode."
        )
    return (
        float(config.ground_z_m),
        float(config.initial_world_z_m),
        np.asarray(config.discrete_mast_actions_m, dtype=float),
        np.asarray(config.discrete_world_actions_m, dtype=float),
    )


_DEFAULT_ROBOT_BASE_RADIUS_M = float(np.hypot(0.31, 0.32))
_DEFAULT_ROBOT_BASE_HEIGHT_M = 0.23
_DEFAULT_ROBOT_MAST_RADIUS_M = float(np.hypot(0.04, 0.04))


def _resolve_measurement_clearance_radius_m(
    runtime_config: Mapping[str, object],
    *,
    requested_robot_radius_m: float,
) -> float:
    """Return the conservative floor-planning radius for the physical robot."""
    requested_radius = float(requested_robot_radius_m)
    if not np.isfinite(requested_radius) or requested_radius < 0.0:
        raise ValueError("robot_radius_m must be finite and non-negative.")
    if not bool(runtime_config.get("measurement_pose_clearance_enabled", True)):
        return requested_radius
    physical_radius = float(
        runtime_config.get(
            "robot_base_physical_radius_m",
            _DEFAULT_ROBOT_BASE_RADIUS_M,
        )
    )
    clearance_margin = float(
        runtime_config.get("measurement_pose_clearance_margin_m", 0.02)
    )
    if not np.isfinite(physical_radius) or physical_radius <= 0.0:
        raise ValueError("robot_base_physical_radius_m must be finite and positive.")
    if not np.isfinite(clearance_margin) or clearance_margin < 0.0:
        raise ValueError(
            "measurement_pose_clearance_margin_m must be finite and non-negative."
        )
    return float(max(requested_radius, physical_radius + clearance_margin))


def _measurement_collision_boxes(
    obstacle_grid: ObstacleGrid | None,
    *,
    ground_z_m: float,
    obstacle_height_m: float,
) -> tuple[tuple[float, float, float, float, float, float], ...]:
    """Return explicit 3D collision boxes, with grid columns as a fallback."""
    if obstacle_grid is None:
        return ()
    explicit_boxes = tuple(obstacle_grid.collision_boxes_m)
    if explicit_boxes:
        return explicit_boxes
    return tuple(
        obstacle_grid.blocked_boxes(
            z_min=float(ground_z_m),
            z_max=float(ground_z_m) + max(float(obstacle_height_m), 0.0),
        )
    )


def _build_measurement_workspace(
    runtime_config: Mapping[str, object],
    *,
    environment_size_xyz: Sequence[float],
    detector_height_config: DetectorHeightPlanningConfig,
    obstacle_grid: ObstacleGrid | None,
    base_map: object | None,
    shield_params: object,
    effective_robot_radius_m: float,
) -> tuple[object | None, dict[str, object]]:
    """Build the 3D measurement workspace and its serialized diagnostics."""
    enabled = bool(runtime_config.get("measurement_pose_clearance_enabled", True))
    if not enabled:
        return base_map, {
            "enabled": False,
            "effective_robot_radius_m": float(effective_robot_radius_m),
        }
    environment_size = np.asarray(environment_size_xyz, dtype=float).reshape(-1)
    if environment_size.shape != (3,) or np.any(~np.isfinite(environment_size)):
        raise ValueError("environment_size_xyz must be a finite three-vector.")
    margin = float(runtime_config.get("measurement_pose_clearance_margin_m", 0.02))
    base_height = float(
        runtime_config.get(
            "robot_base_physical_height_m",
            _DEFAULT_ROBOT_BASE_HEIGHT_M,
        )
    )
    mast_radius = float(
        runtime_config.get(
            "detector_mast_physical_radius_m",
            _DEFAULT_ROBOT_MAST_RADIUS_M,
        )
    )
    geometry_values = np.asarray([margin, base_height, mast_radius], dtype=float)
    if np.any(~np.isfinite(geometry_values)) or margin < 0.0:
        raise ValueError("Measurement-clearance dimensions must be finite.")
    if base_height <= 0.0 or mast_radius < 0.0:
        raise ValueError(
            "Robot base height must be positive and mast radius non-negative."
        )
    shield_outer_radius_m = 0.01 * max(
        float(getattr(shield_params, "inner_radius_fe_cm"))
        + float(getattr(shield_params, "thickness_fe_cm")),
        float(getattr(shield_params, "inner_radius_pb_cm"))
        + float(getattr(shield_params, "thickness_pb_cm")),
    )
    if not np.isfinite(shield_outer_radius_m) or shield_outer_radius_m <= 0.0:
        raise ValueError("Shield outer radius must be finite and positive.")
    transport_mast_height = float(
        runtime_config.get(
            "detector_transport_height_m",
            detector_height_config.initial_mast_height_m,
        )
    )
    if not (
        detector_height_config.minimum_mast_height_m
        <= transport_mast_height
        <= detector_height_config.maximum_mast_height_m
    ):
        raise ValueError(
            "detector_transport_height_m must lie inside the detector mast range."
        )
    obstacle_height = float(runtime_config.get("obstacle_height_m", 2.0))
    collision_boxes = _measurement_collision_boxes(
        obstacle_grid,
        ground_z_m=detector_height_config.ground_z_m,
        obstacle_height_m=obstacle_height,
    )
    assembly = DetectorAssemblyGeometry(
        base_radius_m=float(effective_robot_radius_m),
        base_height_m=base_height + margin,
        mast_radius_m=mast_radius + margin,
        head_radius_m=shield_outer_radius_m + margin,
    )
    workspace = MeasurementWorkspace(
        room_bounds=AxisAlignedRoomBounds(
            lower_xyz=(0.0, 0.0, detector_height_config.ground_z_m),
            upper_xyz=tuple(float(value) for value in environment_size),
        ),
        assembly=assembly,
        ground_z_m=detector_height_config.ground_z_m,
        detector_transport_world_z_m=(
            detector_height_config.ground_z_m + transport_mast_height
        ),
        collision_boxes_m=collision_boxes,
        base_map=base_map,
        motion_worker_count=max(
            0,
            int(runtime_config.get("measurement_route_workers", 0)),
        ),
        motion_grid_cell_size_m=float(
            runtime_config.get("measurement_route_grid_cell_size_m", 0.25)
        ),
    )
    diagnostics: dict[str, object] = {
        "enabled": True,
        "continuous_measurement_volume": (detector_height_config.mode == "continuous"),
        "height_sampling_mode": detector_height_config.mode,
        "collision_box_count": int(len(collision_boxes)),
        "effective_robot_radius_m": float(assembly.base_radius_m),
        "base_height_m": float(assembly.base_height_m),
        "mast_radius_m": float(assembly.mast_radius_m),
        "head_radius_m": float(assembly.head_radius_m),
        "clearance_margin_m": float(margin),
        "transport_world_z_m": float(workspace.detector_transport_world_z_m),
        "motion_policy": "retract_translate_extend",
        "route_workers": int(workspace.motion_worker_count),
        "route_grid_cell_size_m": float(workspace.motion_grid_cell_size_m),
    }
    return workspace, diagnostics






def _count_error_model_diagnostics(
    pf_config: RotatingShieldPFConfig,
    *,
    obstacle_attenuation_active: bool,
) -> dict[str, object]:
    """Describe statistical, calibrated-bias, and forward-model error layers."""
    isotope_scales = pf_config.measurement_scale_by_isotope or {}
    pair_scales = pf_config.measurement_scale_by_isotope_and_pair or {}
    return {
        "statistical_uncertainty": {
            "count_likelihood_model": str(pf_config.count_likelihood_model),
            "student_t_degrees_of_freedom": float(pf_config.count_likelihood_df),
            "spectrum_covariance_propagated": True,
            "observation_count_variance_includes_counting_noise": bool(
                pf_config.observation_count_variance_includes_counting_noise
            ),
            "observation_count_variance_semantics": str(
                pf_config.observation_count_variance_semantics
            ),
            "direct_spectrum_likelihood_enabled": bool(
                pf_config.direct_spectrum_likelihood_enable
            ),
            "station_view_covariance_enabled": bool(
                pf_config.station_view_covariance_enable
            ),
            "shield_view_ratio_enabled": bool(
                pf_config.shield_view_ratio_likelihood_enable
            ),
            "shield_contrast_enabled": bool(
                pf_config.shield_contrast_likelihood_enable
            ),
        },
        "calibrated_systematic_response": {
            "isotope_scale_configured": bool(isotope_scales),
            "shield_pair_scale_configured": bool(pair_scales),
            "calibration_scope": "external_calibration_only",
        },
        "forward_model_mismatch": {
            "transport_relative_sigma": pf_config.transport_model_rel_sigma,
            "transport_absolute_sigma": pf_config.transport_model_abs_sigma,
            "spectrum_relative_sigma": pf_config.spectrum_count_rel_sigma,
            "spectrum_absolute_sigma": pf_config.spectrum_count_abs_sigma,
            "low_count_absolute_sigma": float(pf_config.low_count_abs_sigma),
            "obstacle_attenuation_active": bool(obstacle_attenuation_active),
        },
    }




def _compute_ig_grid(
    estimator: RotatingShieldPFEstimator,
    rot_mats: Sequence[np.ndarray],
    *,
    pose_idx: int,
    live_time_s: float,
    planning_isotopes: Sequence[str] | None = None,
) -> np.ndarray:
    """
    Compute expected IG over all Fe/Pb orientation pairs for the current PF state.
    """
    eig_samples = estimator.pf_config.planning_eig_samples
    if eig_samples is None:
        eig_samples = estimator.pf_config.eig_num_samples
    rollout_particles = estimator.pf_config.planning_rollout_particles
    if rollout_particles is None:
        rollout_particles = estimator.pf_config.planning_particles
    rollout_method = (
        estimator.pf_config.planning_rollout_method
        or estimator.pf_config.planning_method
    )
    particles_by_iso = estimator.planning_particles(
        max_particles=rollout_particles,
        method=rollout_method,
    )
    alpha_weights = estimator.pf_config.alpha_weights
    if planning_isotopes is not None:
        planning_set = set(planning_isotopes)
        particles_by_iso = {
            iso: val for iso, val in particles_by_iso.items() if iso in planning_set
        }
        if alpha_weights is None:
            alpha_weights = {iso: 1.0 for iso in planning_set}
        else:
            alpha_weights = {
                iso: float(alpha_weights.get(iso, 1.0)) for iso in planning_set
            }
    size = len(rot_mats)
    scores = np.zeros((size, size), dtype=float)

    def _ig_for_pair(
        fe_idx: int, pb_idx: int, RFe: np.ndarray, RPb: np.ndarray
    ) -> float:
        """Compute expected IG for a single Fe/Pb orientation pair."""
        return float(
            estimator.orientation_expected_information_gain(
                pose_idx=pose_idx,
                RFe=RFe,
                RPb=RPb,
                live_time_s=live_time_s,
                num_samples=eig_samples,
                alpha_by_isotope=alpha_weights,
                particles_by_isotope=particles_by_iso,
            )
        )

    total_pairs = size * size
    workers = _resolve_ig_workers(getattr(estimator.pf_config, "ig_workers", None))
    if getattr(estimator, "_can_use_gpu", lambda: False)():
        if hasattr(
            estimator, "orientation_expected_information_gain_grid"
        ) and size == int(estimator.num_orientations):
            return estimator.orientation_expected_information_gain_grid(
                pose_idx=pose_idx,
                live_time_s=live_time_s,
                num_samples=eig_samples,
                alpha_by_isotope=alpha_weights,
                particles_by_isotope=particles_by_iso,
            )
        # A single CUDA device cannot safely evaluate many independent Python
        # workers at once for dense obstacle scenes; serialize fallback GPU
        # calls while preserving the same expected-count model.
        workers = 1
    if workers <= 1 or total_pairs <= 1:
        for fe_idx, RFe in enumerate(rot_mats):
            for pb_idx, RPb in enumerate(rot_mats):
                scores[fe_idx, pb_idx] = _ig_for_pair(fe_idx, pb_idx, RFe, RPb)
        return scores

    max_workers = min(workers, total_pairs)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for fe_idx, RFe in enumerate(rot_mats):
            for pb_idx, RPb in enumerate(rot_mats):
                future = executor.submit(_ig_for_pair, fe_idx, pb_idx, RFe, RPb)
                futures[future] = (fe_idx, pb_idx)
        for future in as_completed(futures):
            fe_idx, pb_idx = futures[future]
            scores[fe_idx, pb_idx] = float(future.result())
    return scores


def _select_best_pair_from_scores(
    scores: NDArray[np.float64],
    allowed_indices: set[int] | None,
) -> tuple[int, float]:
    """Return the best (fe,pb) pair index and score from a full IG grid."""
    if scores.size == 0:
        return -1, 0.0
    size = int(scores.shape[0])
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must be a square 2D array.")
    if allowed_indices is None:
        allowed_iter = range(size * size)
    else:
        allowed_iter = sorted(allowed_indices)
    best_idx = -1
    best_score = -np.inf
    for oid in allowed_iter:
        fe_idx = int(oid) // size
        pb_idx = int(oid) % size
        score = float(scores[fe_idx, pb_idx])
        if np.isnan(score):
            continue
        if score > best_score:
            best_score = score
            best_idx = int(oid)
    if best_idx < 0:
        return -1, 0.0
    return best_idx, float(best_score)


def _transition_cost_for_pair(
    rot_mats: Sequence[np.ndarray],
    current_pair_id: int | None,
    candidate_pair_id: int,
) -> float:
    """Return a normalized Fe/Pb angular transition cost for one pair."""
    if current_pair_id is None or int(current_pair_id) < 0:
        return 0.0
    num_orients = len(rot_mats)
    if num_orients <= 0:
        return 0.0
    prev_fe = int(current_pair_id) // num_orients
    prev_pb = int(current_pair_id) % num_orients
    next_fe = int(candidate_pair_id) // num_orients
    next_pb = int(candidate_pair_id) % num_orients
    normals = np.asarray([np.asarray(mat, dtype=float)[:, 2] for mat in rot_mats])
    cost = 0.0
    for prev_idx, next_idx in ((prev_fe, next_fe), (prev_pb, next_pb)):
        dot = float(np.clip(np.dot(normals[prev_idx], normals[next_idx]), -1.0, 1.0))
        cost += float(np.arccos(dot) / np.pi)
    return cost


def _log_utility_grid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a nonnegative log-scaled utility grid for planner score terms."""
    arr = np.asarray(values, dtype=float)
    finite = np.where(np.isfinite(arr), arr, 0.0)
    return np.log1p(np.maximum(finite, 0.0))


def _isotope_count_balance_penalty(counts: dict[str, float]) -> float:
    """Return an isotope-agnostic penalty for single-isotope dominated counts."""
    values = np.asarray(
        [max(float(value), 0.0) for value in counts.values()],
        dtype=float,
    )
    if values.size <= 1:
        return 0.0
    total = float(np.sum(values))
    if total <= 0.0:
        return 1.0
    probabilities = values / total
    positive = probabilities > 0.0
    entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
    normalized_entropy = entropy / max(float(np.log(values.size)), 1e-12)
    return float(np.clip(1.0 - normalized_entropy, 0.0, 1.0))


def _compute_shield_selection_grid(
    estimator: RotatingShieldPFEstimator,
    rot_mats: Sequence[np.ndarray],
    *,
    pose_idx: int,
    live_time_s: float,
    ig_scores: NDArray[np.float64],
    current_pair_id: int | None,
    min_observation_counts: float,
    signature_weight: float,
    low_count_penalty_weight: float,
    count_balance_weight: float,
    rotation_cost_weight: float,
    variance_floor: float,
    max_particles: int | None,
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    """
    Score shield postures by EIG, signature separation, observability, and rotation cost.

    The score is a planner utility only; PF updates still use measured spectra.
    """
    size = int(len(rot_mats))
    scores = np.asarray(ig_scores, dtype=float).copy()
    signature_grid = np.zeros((size, size), dtype=float)
    signature_utility_grid = np.zeros((size, size), dtype=float)
    penalty_grid = np.zeros((size, size), dtype=float)
    balance_grid = np.zeros((size, size), dtype=float)
    rotation_grid = np.zeros((size, size), dtype=float)
    planning_particles = estimator.planning_particles(
        max_particles=max_particles,
        method=estimator.pf_config.planning_method,
    )
    need_count_grids = min_observation_counts > 0.0 or count_balance_weight > 0.0
    count_grids: dict[str, NDArray[np.float64]] = {}
    if (signature_weight > 0.0 or need_count_grids) and hasattr(
        estimator,
        "shield_selection_batch_grids",
    ):
        signature_grid, count_grids = estimator.shield_selection_batch_grids(
            pose_idx=pose_idx,
            live_time_s=live_time_s,
            max_particles=max_particles,
            particles_by_isotope=planning_particles,
            alpha_by_isotope=estimator.pf_config.alpha_weights,
            variance_floor=variance_floor,
            include_count_quantiles=need_count_grids,
        )
    if need_count_grids and count_grids:
        count_stack = np.stack(
            [np.asarray(grid, dtype=float) for grid in count_grids.values()],
            axis=0,
        )
        if min_observation_counts > 0.0:
            min_counts = float(min_observation_counts)
            shortfalls = np.maximum(
                0.0, 1.0 - np.maximum(count_stack, 0.0) / min_counts
            )
            penalty_grid = np.mean(shortfalls * shortfalls, axis=0)
        if count_balance_weight > 0.0:
            totals = np.sum(np.maximum(count_stack, 0.0), axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                probabilities = np.divide(
                    np.maximum(count_stack, 0.0),
                    totals[None, :, :],
                    out=np.zeros_like(count_stack),
                    where=totals[None, :, :] > 0.0,
                )
                entropy_terms = np.where(
                    probabilities > 0.0,
                    probabilities * np.log(probabilities),
                    0.0,
                )
            entropy = -np.sum(entropy_terms, axis=0)
            norm = max(float(np.log(count_stack.shape[0])), 1e-12)
            balance_grid = np.where(
                totals > 0.0,
                np.clip(1.0 - entropy / norm, 0.0, 1.0),
                1.0,
            )
    elif signature_weight > 0.0 or need_count_grids:
        for fe_idx in range(size):
            for pb_idx in range(size):
                if signature_weight > 0.0:
                    signature_grid[fe_idx, pb_idx] = (
                        estimator.orientation_signature_separation_score(
                            pose_idx=pose_idx,
                            fe_index=fe_idx,
                            pb_index=pb_idx,
                            live_time_s=live_time_s,
                            particles_by_isotope=planning_particles,
                            alpha_by_isotope=estimator.pf_config.alpha_weights,
                            variance_floor=variance_floor,
                        )
                    )
                counts = None
                if need_count_grids:
                    counts = estimator.expected_observation_counts_by_isotope_at_pair(
                        pose_idx=pose_idx,
                        fe_index=fe_idx,
                        pb_index=pb_idx,
                        live_time_s=live_time_s,
                        max_particles=max_particles,
                    )
                if min_observation_counts > 0.0 and counts is not None:
                    penalty_grid[fe_idx, pb_idx] = minimum_observation_shortfall(
                        counts,
                        min_counts=float(min_observation_counts),
                    )
                if count_balance_weight > 0.0 and counts is not None:
                    balance_grid[fe_idx, pb_idx] = _isotope_count_balance_penalty(
                        counts
                    )
    for fe_idx in range(size):
        for pb_idx in range(size):
            pair_id = fe_idx * size + pb_idx
            rotation_grid[fe_idx, pb_idx] = _transition_cost_for_pair(
                rot_mats,
                current_pair_id,
                pair_id,
            )
    signature_utility_grid = _log_utility_grid(signature_grid)
    scores += float(signature_weight) * signature_utility_grid
    scores -= float(low_count_penalty_weight) * penalty_grid
    scores -= float(count_balance_weight) * balance_grid
    scores -= float(rotation_cost_weight) * rotation_grid
    diagnostics = {
        "eig": np.asarray(ig_scores, dtype=float),
        "signature": signature_grid,
        "signature_utility": signature_utility_grid,
        "low_count_penalty": penalty_grid,
        "count_balance_penalty": balance_grid,
        "rotation_cost": rotation_grid,
    }
    return scores, diagnostics


def _polyline_distance(points_xyz: NDArray[np.float64] | None) -> float:
    """Return the total length of a 3D polyline."""
    if points_xyz is None:
        return float("inf")
    arr = np.asarray(points_xyz, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] != 3:
        return float("inf")
    if arr.shape[0] < 2:
        return 0.0
    deltas = np.diff(arr, axis=0)
    return float(np.sum(np.linalg.norm(deltas, axis=1)))


def _obstacle_aware_waypoints(
    map_api: object | None,
    start_xyz: NDArray[np.float64],
    goal_xyz: NDArray[np.float64],
) -> tuple[NDArray[np.float64], bool]:
    """Return travel waypoints, preferring a grid path when available."""
    start = np.asarray(start_xyz, dtype=float).reshape(3)
    goal = np.asarray(goal_xyz, dtype=float).reshape(3)
    if map_api is not None:
        motion_waypoints = getattr(map_api, "motion_waypoints", None)
        if callable(motion_waypoints):
            path = motion_waypoints(start, goal)
            if path is None:
                return np.zeros((0, 3), dtype=float), True
            path_array = np.asarray(path, dtype=float)
            if path_array.ndim == 2 and path_array.shape[0] >= 2:
                return path_array, True
        path = shortest_grid_path_points(map_api, start, goal, allow_diagonal=True)
        if path is not None and path.shape[0] >= 2:
            return np.asarray(path, dtype=float), True
        if _supports_grid_path(map_api):
            return np.zeros((0, 3), dtype=float), True
    return np.vstack([start, goal]).astype(float), False


def _supports_grid_path(map_api: object | None) -> bool:
    """Return True when a map API supports grid path planning."""
    if map_api is None:
        return False
    cell_index = getattr(map_api, "cell_index", None)
    grid_shape = getattr(map_api, "grid_shape", None)
    has_cell_free = any(
        callable(getattr(map_api, attr, None))
        for attr in ("is_free_cell", "is_cell_free")
    )
    return callable(cell_index) and grid_shape is not None and has_cell_free


def _filter_reachable_candidates(
    *,
    current_pose_xyz: NDArray[np.float64],
    map_api: object | None,
    candidates: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Keep only candidates connected to the current pose on the traversability grid."""
    candidate_arr = np.asarray(candidates, dtype=float)
    if candidate_arr.size == 0:
        return candidate_arr
    motion_reachable_batch = getattr(map_api, "is_motion_reachable_batch", None)
    if callable(motion_reachable_batch):
        reachable_mask = np.asarray(
            motion_reachable_batch(current_pose_xyz, candidate_arr),
            dtype=bool,
        ).reshape(-1)
        if reachable_mask.size != candidate_arr.shape[0]:
            raise ValueError(
                "is_motion_reachable_batch returned the wrong number of flags."
            )
        return candidate_arr[reachable_mask]
    motion_waypoints = getattr(map_api, "motion_waypoints", None)
    if callable(motion_waypoints):
        reachable_mask = [
            motion_waypoints(current_pose_xyz, candidate) is not None
            for candidate in candidate_arr
        ]
        return candidate_arr[np.asarray(reachable_mask, dtype=bool)]
    if not _supports_grid_path(map_api):
        return candidate_arr
    reachable_mask = [
        shortest_grid_path_points(map_api, current_pose_xyz, candidate) is not None
        for candidate in candidate_arr
    ]
    return candidate_arr[np.asarray(reachable_mask, dtype=bool)]


def _build_robot_path_segment(
    *,
    map_api: object | None,
    from_pose_xyz: NDArray[np.float64],
    to_pose_xyz: NDArray[np.float64],
    nominal_motion_speed_m_s: float,
    path_planner: str,
    planned_shield_program: tuple[int, ...] | None,
    dss_diagnostics: dict[str, Any] | None,
) -> dict[str, object]:
    """Build an obstacle-aware robot travel segment for timing and rendering."""
    waypoints, obstacle_aware = _obstacle_aware_waypoints(
        map_api,
        np.asarray(from_pose_xyz, dtype=float),
        np.asarray(to_pose_xyz, dtype=float),
    )
    distance_m = _polyline_distance(waypoints)
    if not np.isfinite(distance_m):
        raise RuntimeError(
            "Selected robot travel segment is not connected on the traversability grid."
        )
    motion_time_s = distance_m / max(float(nominal_motion_speed_m_s), 1e-9)
    return {
        "from_pose_xyz": [float(v) for v in np.asarray(from_pose_xyz, dtype=float)],
        "to_pose_xyz": [float(v) for v in np.asarray(to_pose_xyz, dtype=float)],
        "waypoints_xyz": [
            [float(coord) for coord in waypoint]
            for waypoint in np.asarray(waypoints, dtype=float)
        ],
        "distance_m": float(distance_m),
        "euclidean_distance_m": float(
            np.linalg.norm(
                np.asarray(to_pose_xyz, dtype=float)
                - np.asarray(from_pose_xyz, dtype=float)
            )
        ),
        "travel_time_s": float(motion_time_s),
        "speed_m_s": float(nominal_motion_speed_m_s),
        "obstacle_aware": bool(obstacle_aware),
        "path_planner": path_planner,
        "planned_shield_program": None
        if planned_shield_program is None
        else [int(v) for v in planned_shield_program],
        "dss_diagnostics": dss_diagnostics,
    }


def _estimate_best_next_pose_gain_rate(
    estimator: RotatingShieldPFEstimator,
    *,
    candidates: NDArray[np.float64],
    current_pose_xyz: NDArray[np.float64],
    map_api: object | None,
    live_time_s: float,
    rotation_limit: int,
    nominal_motion_speed_m_s: float,
    rotation_overhead_s: float,
    max_candidates: int,
) -> tuple[float, float, int]:
    """Estimate the best next-pose information gain rate for rotation stopping."""
    if candidates.size == 0:
        return 0.0, 0.0, -1
    candidate_arr = np.asarray(candidates, dtype=float)
    if candidate_arr.ndim != 2 or candidate_arr.shape[1] != 3:
        return 0.0, 0.0, -1
    limit = min(max(1, int(max_candidates)), int(candidate_arr.shape[0]))
    current_uncertainty = max(float(estimator.global_uncertainty()), 0.0)
    current_information = float(np.log1p(current_uncertainty))
    best_rate = 0.0
    best_gain = 0.0
    best_idx = -1
    for idx, candidate in enumerate(candidate_arr[:limit]):
        try:
            after_uncertainty = float(
                estimator.expected_uncertainty_after_rotation(
                    pose_xyz=candidate,
                    live_time_per_rot_s=live_time_s,
                    tau_ig=float(estimator.pf_config.ig_threshold),
                    tmax_s=float(max(1, rotation_limit)) * float(live_time_s),
                    n_rollouts=0,
                    orient_selection="IG",
                    rng_seed=idx,
                )
            )
        except RuntimeError:
            continue
        after_information = float(np.log1p(max(after_uncertainty, 0.0)))
        gain = max(current_information - after_information, 0.0)
        waypoints, _ = _obstacle_aware_waypoints(map_api, current_pose_xyz, candidate)
        travel = _polyline_distance(waypoints)
        if not np.isfinite(travel):
            continue
        travel_time = travel / max(float(nominal_motion_speed_m_s), 1e-9)
        cost_time = travel_time + float(rotation_overhead_s) + float(live_time_s)
        rate = gain / max(cost_time, 1e-9)
        if rate > best_rate:
            best_rate = float(rate)
            best_gain = float(gain)
            best_idx = int(idx)
    return best_rate, best_gain, best_idx


def _generate_planning_candidates(
    *,
    current_pose_xyz: NDArray[np.float64],
    map_api: object | None,
    n_candidates: int,
    min_dist_from_visited: float,
    visited_poses_xyz: NDArray[np.float64] | None,
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]],
    detector_heights_m: Sequence[float] | None = None,
    continuous_height_anchor_count: int = 0,
    height_partner_xy_tolerance_m: float = 1.0e-9,
    height_partner_z_tolerance_m: float = 1.0e-9,
    height_partner_min_z_separation_m: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], bool, float]:
    """Generate next-pose actions with one lateral-spacing retry.

    The local height-action budget must not crowd lateral stations out of the
    candidate batch. A height action is also a one-step partner measurement:
    after taking one, the next station must move laterally instead of chaining
    another height action at the same xy location.
    """
    min_dist = max(float(min_dist_from_visited), 0.0)
    height_partners_requested = bool(
        detector_heights_m is not None or int(continuous_height_anchor_count) > 0
    )
    visited = (
        np.zeros((0, 3), dtype=float)
        if visited_poses_xyz is None
        else np.asarray(visited_poses_xyz, dtype=float).reshape(-1, 3)
    )
    previous_move_was_height_partner = _previous_move_was_height_partner(
        visited,
        xy_tolerance_m=height_partner_xy_tolerance_m,
        z_tolerance_m=height_partner_z_tolerance_m,
        min_z_separation_m=height_partner_min_z_separation_m,
    )
    height_partners_enabled = bool(
        height_partners_requested and not previous_move_was_height_partner
    )
    if height_partners_enabled:
        if detector_heights_m is not None:
            height_action_budget = len(tuple(detector_heights_m))
        else:
            height_action_budget = max(
                int(continuous_height_anchor_count),
                0,
            )
    else:
        height_action_budget = 0
    required_lateral_count = max(
        int(n_candidates) - min(height_action_budget, int(n_candidates) - 1),
        1,
    )

    def _lateral_count(candidate_rows: NDArray[np.float64]) -> int:
        """Return the number of candidates that change detector xy."""
        rows = np.asarray(candidate_rows, dtype=float).reshape(-1, 3)
        if rows.shape[0] == 0:
            return 0
        lateral_distance = np.linalg.norm(
            rows[:, :2] - np.asarray(current_pose_xyz, dtype=float)[None, :2],
            axis=1,
        )
        return int(
            np.count_nonzero(
                lateral_distance > max(float(height_partner_xy_tolerance_m), 1.0e-9)
            )
        )

    candidates = generate_candidate_poses(
        current_pose_xyz=current_pose_xyz,
        map_api=map_api,
        n_candidates=int(n_candidates),
        strategy="free_space_sobol",
        min_dist_from_visited=min_dist,
        visited_poses_xyz=visited_poses_xyz,
        bounds_xyz=bounds_xyz,
        detector_heights_m=detector_heights_m,
        include_current_xy_height_actions=height_partners_enabled,
        continuous_height_anchor_count=max(int(continuous_height_anchor_count), 0),
        allow_height_partners=height_partners_enabled,
        height_partner_xy_tolerance_m=height_partner_xy_tolerance_m,
        height_partner_z_tolerance_m=height_partner_z_tolerance_m,
        height_partner_min_z_separation_m=height_partner_min_z_separation_m,
        require_motion_reachable=True,
        rng=rng,
    )
    candidates = _filter_reachable_candidates(
        current_pose_xyz=current_pose_xyz,
        map_api=map_api,
        candidates=candidates,
    )
    lateral_count = _lateral_count(candidates)
    if lateral_count >= required_lateral_count or min_dist <= 0.0:
        return candidates, False, min_dist
    relaxed_dist = max(min_dist * 0.5, 0.5)
    candidates = generate_candidate_poses(
        current_pose_xyz=current_pose_xyz,
        map_api=map_api,
        n_candidates=max(int(n_candidates) * 2, int(n_candidates)),
        strategy="free_space_sobol",
        min_dist_from_visited=relaxed_dist,
        visited_poses_xyz=visited_poses_xyz,
        bounds_xyz=bounds_xyz,
        detector_heights_m=detector_heights_m,
        include_current_xy_height_actions=height_partners_enabled,
        continuous_height_anchor_count=max(int(continuous_height_anchor_count), 0),
        allow_height_partners=height_partners_enabled,
        height_partner_xy_tolerance_m=height_partner_xy_tolerance_m,
        height_partner_z_tolerance_m=height_partner_z_tolerance_m,
        height_partner_min_z_separation_m=height_partner_min_z_separation_m,
        require_motion_reachable=True,
        rng=rng,
    )
    candidates = _filter_reachable_candidates(
        current_pose_xyz=current_pose_xyz,
        map_api=map_api,
        candidates=candidates,
    )
    return candidates, True, relaxed_dist


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
    estimator: RotatingShieldPFEstimator,
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
    last_counts = {
        str(isotope): float((last.isotope_counts or {}).get(str(isotope), 0.0))
        for isotope in isotopes
    }
    representative_counts = {
        str(isotope): float(
            (representative.isotope_counts or {}).get(str(isotope), 0.0)
        )
        for isotope in isotopes
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
        total_motion_distance_m=motion_time
        * max(float(nominal_motion_speed_m_s), 0.0),
        total_motion_time_s=motion_time,
        total_rotation_time_s=rotation_time,
        measurement_live_times_s=live_times,
        last_spectrum=np.asarray(last.spectrum_counts, dtype=float).copy(),
        last_counts=last_counts,
        representative_spectrum=np.asarray(
            representative.spectrum_counts,
            dtype=float,
        ).copy(),
        representative_counts=representative_counts,
        representative_step_index=int(representative.step_id),
    )


def _advance_resume_planning_candidate_rng(
    *,
    rng: np.random.Generator,
    records: Sequence[MeasurementLogRecord],
    map_api: object | None,
    n_candidates: int,
    min_dist_from_visited: float,
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]],
    detector_heights_m: Sequence[float] | None,
    continuous_height_anchor_count: int,
    height_partner_xy_tolerance_m: float,
    height_partner_z_tolerance_m: float,
    height_partner_min_z_separation_m: float,
    expected_candidate_counts: Sequence[tuple[int, int, int]] | None = None,
) -> None:
    """Replay candidate generation and verify logged total/lateral/height counts."""
    stations = _records_by_station(records)
    transition_count = max(len(stations) - 1, 0)
    if expected_candidate_counts is not None and len(expected_candidate_counts) != (
        transition_count
    ):
        raise RuntimeError(
            "Resume candidate diagnostics do not match the transition count."
        )
    visited: list[NDArray[np.float64]] = []
    for station_index, station in enumerate(stations[:-1]):
        current_pose = np.asarray(station[0].detector_pose_xyz, dtype=float)
        visited.append(current_pose.copy())
        visited_arr = np.vstack(visited)
        candidates, _, _ = _generate_planning_candidates(
            current_pose_xyz=current_pose,
            map_api=map_api,
            n_candidates=int(n_candidates),
            min_dist_from_visited=float(min_dist_from_visited),
            visited_poses_xyz=visited_arr,
            bounds_xyz=bounds_xyz,
            detector_heights_m=detector_heights_m,
            continuous_height_anchor_count=int(continuous_height_anchor_count),
            height_partner_xy_tolerance_m=float(
                height_partner_xy_tolerance_m
            ),
            height_partner_z_tolerance_m=float(
                height_partner_z_tolerance_m
            ),
            height_partner_min_z_separation_m=float(
                height_partner_min_z_separation_m
            ),
            rng=rng,
        )
        if expected_candidate_counts is not None:
            candidates_array = np.asarray(candidates, dtype=float).reshape(-1, 3)
            lateral_distance = np.linalg.norm(
                candidates_array[:, :2] - current_pose[None, :2],
                axis=1,
            )
            lateral_count = int(
                np.count_nonzero(
                    lateral_distance
                    > max(float(height_partner_xy_tolerance_m), 1.0e-9)
                )
            )
            actual_counts = (
                int(candidates_array.shape[0]),
                lateral_count,
                int(candidates_array.shape[0] - lateral_count),
            )
            if actual_counts != tuple(expected_candidate_counts[station_index]):
                raise RuntimeError(
                    "Regenerated candidate total/lateral/height counts differ "
                    f"after station {station_index}: actual={actual_counts}, "
                    f"expected={expected_candidate_counts[station_index]}."
                )


def _parse_candidate_generation_counts(
    runtime_log_path: str | Path,
) -> tuple[tuple[int, int, int], ...]:
    """Parse post-station candidate total/lateral/height diagnostics."""
    path = Path(runtime_log_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Resume runtime log must be one regular file.")
    pattern = re.compile(
        r"Generated ([0-9]+) candidate poses "
        r"\(lateral=([0-9]+), height=([0-9]+)\)\."
    )
    result: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.search(line)
            if match is None:
                continue
            result.append(tuple(int(value) for value in match.groups()))
    return tuple(result)


def _parse_remaining_measurement_details(
    runtime_log_path: str | Path,
) -> dict[int, dict[str, Any]]:
    """Parse structured estimator-only remaining-budget details from a live log."""
    path = Path(runtime_log_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Resume runtime log must be one regular file.")
    parsed: dict[int, dict[str, Any]] = {}
    marker = "Remaining measurement detail[pose_"
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            marker_index = line.find(marker)
            if marker_index < 0:
                continue
            payload_line = line[marker_index:].rstrip("\n")
            label, separator, payload = payload_line.partition("]: components=")
            if not separator or not label.endswith("_next"):
                continue
            raw_index = label[len(marker) : -len("_next")]
            try:
                station_index = int(raw_index)
                components_text, payload = payload.split(" gains=", 1)
                gains_text, isotope_text = payload.split(" isotopes=", 1)
                components = json.loads(components_text)
                gains = json.loads(gains_text)
                isotope_details = json.loads(isotope_text)
            except (ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "Malformed remaining-measurement detail in resume runtime log "
                    f"at line {line_number}."
                ) from exc
            if station_index in parsed:
                raise ValueError(
                    "Resume runtime log contains duplicate next-budget details for "
                    f"station {station_index}."
                )
            if not all(
                isinstance(value, dict)
                for value in (components, gains, isotope_details)
            ):
                raise ValueError(
                    "Resume remaining-measurement details must be JSON objects."
                )
            parsed[station_index] = {
                "components": components,
                "gains": gains,
                "isotope_details": isotope_details,
            }
    return parsed


def _restore_remaining_measurement_history(
    *,
    estimator: RotatingShieldPFEstimator,
    records: Sequence[MeasurementLogRecord],
    runtime_log_path: str | Path,
    config: RemainingMeasurementConfig,
) -> list[dict[str, Any]]:
    """Restore exact empirical-eta history from logged structured diagnostics."""
    stations = _records_by_station(records)
    expected_indices = list(range(max(len(stations) - 1, 0)))
    details_by_station = _parse_remaining_measurement_details(runtime_log_path)
    if sorted(details_by_station) != expected_indices:
        raise RuntimeError(
            "Resume runtime log does not contain exactly one next-budget detail "
            "for every completed historical planning transition."
        )
    history: list[dict[str, float]] = []
    eta_ratios: list[float] = []
    payloads: list[dict[str, Any]] = []
    for station_index in expected_indices:
        detail = details_by_station[station_index]
        components = {
            str(key): float(value)
            for key, value in dict(detail["components"]).items()
        }
        gains = {
            str(key): float(value) for key, value in dict(detail["gains"]).items()
        }
        budget = (
            float(config.uncertainty_weight) * components.get("uncertainty", 0.0)
            + float(config.cardinality_weight)
            * components.get("cardinality", 0.0)
            + float(config.separation_weight)
            * components.get("same_isotope_separation", 0.0)
            + float(config.verification_weight)
            * components.get("pseudo_source_verification", 0.0)
            + float(config.residual_weight) * components.get("residual", 0.0)
            + float(config.high_surface_ambiguity_weight)
            * components.get("high_surface_ambiguity", 0.0)
            + components.get("isotope_absence", 0.0)
        )
        predicted_gain = (
            float(config.uncertainty_weight) * gains.get("uncertainty", 0.0)
            + float(config.separation_weight)
            * gains.get("same_isotope_separation", 0.0)
            + float(config.verification_weight)
            * gains.get("pseudo_source_verification", 0.0)
            + float(config.residual_weight) * gains.get("residual", 0.0)
            + float(config.high_surface_ambiguity_weight)
            * gains.get("high_surface_ambiguity", 0.0)
            + float(config.dss_information_gain_weight)
            * gains.get("dss_information", 0.0)
            + float(config.dss_count_utility_weight)
            * gains.get("residual", 0.0)
        )
        if history:
            previous = history[-1]
            realized = max(float(previous["budget"]) - float(budget), 0.0)
            denominator = max(float(previous["predicted_gain"]), 1.0e-12)
            eta_ratios.append(realized / denominator)
            eta_ratios = eta_ratios[-8:]
        history.append(
            {
                "budget": float(budget),
                "predicted_gain": float(predicted_gain),
            }
        )
        history = history[-8:]
        eta = (
            float(np.median(np.asarray(eta_ratios, dtype=float)))
            if eta_ratios
            else float(config.eta_default)
        )
        eta = float(np.clip(eta, float(config.eta_min), float(config.eta_max)))
        remaining_budget = max(float(budget) - float(config.stop_budget), 0.0)
        denominator = max(
            eta * float(predicted_gain),
            float(config.gain_epsilon),
        )
        estimate = (
            int(np.ceil(remaining_budget / denominator))
            if remaining_budget > 0.0
            else 0
        )
        estimate = min(
            max(estimate, 0),
            max(0, int(config.max_reported_stations)),
        )
        low = (
            max(1, int(np.floor(float(estimate) / float(config.range_scale))))
            if estimate > 0
            else 0
        )
        high = min(
            max(0, int(config.max_reported_stations)),
            max(
                estimate,
                int(np.ceil(float(estimate) * float(config.range_scale))),
            ),
        )
        next_station = stations[station_index + 1]
        program = [
            int(record.fe_orientation_index) * 8
            + int(record.pb_orientation_index)
            for record in next_station
        ]
        unresolved = [
            key
            for key, value in sorted(components.items())
            if float(value) > 1.0e-9
        ]
        bottleneck = (
            "none"
            if not unresolved
            else max(components, key=lambda key: float(components[key]))
        )
        next_pose = [
            float(value) for value in next_station[0].detector_pose_xyz
        ]
        payloads.append(
            {
                "current_station_count": int(station_index + 1),
                "estimated_remaining_stations": int(estimate),
                "estimated_remaining_station_low": int(low),
                "estimated_remaining_station_high": int(high),
                "estimated_remaining_spectra_low": int(low * len(program)),
                "estimated_remaining_spectra_high": int(high * len(program)),
                "program_length": int(len(program)),
                "current_budget": float(budget),
                "stop_budget": float(config.stop_budget),
                "predicted_gain": float(predicted_gain),
                "empirical_eta": float(eta),
                "bottleneck": str(bottleneck),
                "unresolved_factors": unresolved,
                "components": components,
                "gains": gains,
                "isotope_details": dict(detail["isotope_details"]),
                "timings": {"restored_from_runtime_log": 1.0},
                "next_pose": next_pose,
                "planned_pairs": program,
            }
        )
    setattr(estimator, "_remaining_measurement_budget_history", history)
    setattr(estimator, "_remaining_measurement_eta_ratios", eta_ratios)
    return payloads


def _previous_move_was_height_partner(
    visited_poses_xyz: NDArray[np.float64] | None,
    *,
    xy_tolerance_m: float,
    z_tolerance_m: float = 1.0e-9,
    min_z_separation_m: float = 0.0,
) -> bool:
    """Return whether the latest completed station move changed only height."""
    if visited_poses_xyz is None:
        return False
    visited = np.asarray(visited_poses_xyz, dtype=float).reshape(-1, 3)
    return bool(
        visited.shape[0] >= 2
        and _is_detector_height_partner(
            visited[-2],
            visited[-1],
            xy_tolerance_m=xy_tolerance_m,
            z_tolerance_m=z_tolerance_m,
            min_z_separation_m=min_z_separation_m,
        )
    )


def _validate_selected_station_action(
    *,
    current_pose_xyz: NDArray[np.float64],
    next_pose_xyz: NDArray[np.float64],
    previous_move_was_height_partner: bool,
    xy_tolerance_m: float,
    z_tolerance_m: float = 1.0e-9,
    min_z_separation_m: float = 0.0,
) -> bool:
    """Validate the selected action and return whether it changes only height."""
    is_height_partner_action = _is_detector_height_partner(
        current_pose_xyz,
        next_pose_xyz,
        xy_tolerance_m=xy_tolerance_m,
        z_tolerance_m=z_tolerance_m,
        min_z_separation_m=min_z_separation_m,
    )
    if bool(previous_move_was_height_partner) and is_height_partner_action:
        raise RuntimeError(
            "Planner selected consecutive same-xy height actions after the "
            "height-action lock was enabled."
        )
    return bool(is_height_partner_action)


def _is_detector_height_partner(
    first_pose_xyz: NDArray[np.float64],
    second_pose_xyz: NDArray[np.float64],
    *,
    xy_tolerance_m: float,
    z_tolerance_m: float = 1.0e-9,
    min_z_separation_m: float = 0.0,
) -> bool:
    """Return whether two actions share xy but use distinct detector heights."""
    first = np.asarray(first_pose_xyz, dtype=float).reshape(3)
    second = np.asarray(second_pose_xyz, dtype=float).reshape(3)
    z_distance = abs(float(first[2]) - float(second[2]))
    return bool(
        np.linalg.norm(first[:2] - second[:2]) <= max(float(xy_tolerance_m), 0.0)
        and z_distance > max(float(z_tolerance_m), 0.0)
        and z_distance >= max(float(min_z_separation_m), 0.0)
    )


def _height_partner_program_for_scoring(
    *,
    reuse_enabled: bool,
    executed_pair_ids: Sequence[int],
    baseline_shield_policy: Mapping[str, object] | None,
) -> tuple[int, ...] | None:
    """Return an explicitly requested legacy height-partner shield program."""
    if not reuse_enabled or baseline_shield_policy is not None:
        return None
    pair_ids = tuple(int(value) for value in executed_pair_ids)
    return pair_ids or None


def _measurement_detector_positions(
    measurements: Sequence[object],
    registered_poses: Sequence[NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Return the detector position actually stored for every measurement row."""
    positions: list[NDArray[np.float64]] = []
    for record in measurements:
        stored_position = getattr(record, "detector_position_xyz_m", None)
        if stored_position is None:
            pose_index = int(getattr(record, "pose_idx"))
            if pose_index < 0 or pose_index >= len(registered_poses):
                raise ValueError("measurement pose_idx is outside registered poses.")
            stored_position = registered_poses[pose_index]
        position = np.asarray(stored_position, dtype=float).reshape(-1)
        if position.shape != (3,) or np.any(~np.isfinite(position)):
            raise ValueError(
                "measurement detector positions must be finite XYZ vectors."
            )
        positions.append(position.copy())
    if not positions:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(positions)


def _pose_tolerance_component_labels(
    positions_xyz: NDArray[np.float64],
    *,
    xy_tolerance_m: float,
    z_tolerance_m: float | None,
) -> NDArray[np.int64]:
    """Return connected-component labels under planner XY/Z pose tolerances."""
    positions = np.asarray(positions_xyz, dtype=float).reshape(-1, 3)
    count = int(positions.shape[0])
    if count == 0:
        return np.zeros(0, dtype=np.int64)
    xy_tolerance = max(float(xy_tolerance_m), 0.0)
    z_tolerance = None if z_tolerance_m is None else max(float(z_tolerance_m), 0.0)
    parents = np.arange(count, dtype=np.int64)

    def _find(index: int) -> int:
        """Return and compress one disjoint-set root."""
        root = int(index)
        while int(parents[root]) != root:
            root = int(parents[root])
        while int(parents[index]) != index:
            next_index = int(parents[index])
            parents[index] = root
            index = next_index
        return root

    def _union(first: int, second: int) -> None:
        """Join two tolerance-connected pose rows."""
        first_root = _find(first)
        second_root = _find(second)
        if first_root != second_root:
            parents[max(first_root, second_root)] = min(first_root, second_root)

    for first in range(count - 1):
        differences = positions[first + 1 :, :2] - positions[first, :2]
        within = np.linalg.norm(differences, axis=1) <= xy_tolerance
        if z_tolerance is not None:
            within &= (
                np.abs(positions[first + 1 :, 2] - positions[first, 2]) <= z_tolerance
            )
        for offset in np.flatnonzero(within):
            _union(first, first + 1 + int(offset))
    roots = np.asarray([_find(index) for index in range(count)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return np.asarray(labels, dtype=np.int64)


def _operational_station_height_metrics(
    measurements: Sequence[object],
    registered_poses: Sequence[NDArray[np.float64]],
    *,
    xy_tolerance_m: float,
    z_tolerance_m: float,
) -> dict[str, object]:
    """Return tolerance-aware station visits, unique actions, and height changes."""
    positions = _measurement_detector_positions(measurements, registered_poses)
    xy_tolerance = max(float(xy_tolerance_m), 0.0)
    z_tolerance = max(float(z_tolerance_m), 0.0)
    if positions.size == 0:
        observed_heights: list[float] = []
        station_visit_count = 0
        unique_xy_station_count = 0
        unique_xyz_action_count = 0
        height_pair_station_count = 0
        height_transition_count = 0
    else:
        xy_labels = _pose_tolerance_component_labels(
            positions,
            xy_tolerance_m=xy_tolerance,
            z_tolerance_m=None,
        )
        xyz_labels = _pose_tolerance_component_labels(
            positions,
            xy_tolerance_m=xy_tolerance,
            z_tolerance_m=z_tolerance,
        )
        height_only_positions = np.column_stack(
            (
                np.zeros((positions.shape[0], 2), dtype=float),
                positions[:, 2],
            )
        )
        height_labels = _pose_tolerance_component_labels(
            height_only_positions,
            xy_tolerance_m=0.0,
            z_tolerance_m=z_tolerance,
        )
        observed_heights = sorted(
            float(np.mean(positions[height_labels == label, 2]))
            for label in np.unique(height_labels)
        )
        station_visit_count = 1 + int(np.count_nonzero(np.diff(xy_labels) != 0))
        unique_xy_station_count = int(np.unique(xy_labels).size)
        unique_xyz_action_count = int(np.unique(xyz_labels).size)
        height_pair_station_count = 0
        for xy_label in np.unique(xy_labels):
            member_positions = positions[xy_labels == xy_label]
            normalized_members = np.column_stack(
                (
                    np.zeros((member_positions.shape[0], 2), dtype=float),
                    member_positions[:, 2],
                )
            )
            member_height_labels = _pose_tolerance_component_labels(
                normalized_members,
                xy_tolerance_m=0.0,
                z_tolerance_m=z_tolerance,
            )
            if np.unique(member_height_labels).size > 1:
                height_pair_station_count += 1
        height_transition_count = int(
            np.count_nonzero(np.abs(np.diff(positions[:, 2])) > z_tolerance)
        )
    definitions = {
        "station_visit_count": (
            "Number of contiguous measurement-sequence visits to an XY station; "
            "same-XY height actions remain one visit, and a later revisit is counted."
        ),
        "unique_xy_station_count": (
            "Number of tolerance-connected unique detector XY stations."
        ),
        "unique_xyz_action_count": (
            "Number of tolerance-connected unique detector XYZ actions."
        ),
        "height_pair_station_count": (
            "Number of unique XY stations observed at more than one detector height."
        ),
        "height_transition_count": (
            "Number of consecutive measurement rows whose detector heights differ "
            "by more than the planner Z tolerance."
        ),
        "station_count": "Compatibility alias of unique_xy_station_count.",
        "detector_pose_station_count": (
            "Compatibility alias of unique_xyz_action_count."
        ),
        "height_change_count": "Compatibility alias of height_transition_count.",
        "position_source": (
            "MeasurementRecord.detector_position_xyz_m, falling back to the "
            "registered pose only when the record has no explicit detector position."
        ),
    }
    return {
        "observed_detector_heights_m": observed_heights,
        "station_visit_count": int(station_visit_count),
        "unique_xy_station_count": int(unique_xy_station_count),
        "unique_xyz_action_count": int(unique_xyz_action_count),
        "height_pair_station_count": int(height_pair_station_count),
        "height_transition_count": int(height_transition_count),
        "station_count": int(unique_xy_station_count),
        "detector_pose_station_count": int(unique_xyz_action_count),
        "height_change_count": int(height_transition_count),
        "station_height_count_definitions": definitions,
        "station_height_xy_tolerance_m": float(xy_tolerance),
        "station_height_z_tolerance_m": float(z_tolerance),
    }


def _coverage_fraction_for_poses(
    map_api: object | None,
    poses_xyz: Sequence[NDArray[np.float64]],
    *,
    radius_m: float,
) -> float:
    """Return traversable-map coverage by measurement poses within a radius."""
    if map_api is None or not poses_xyz:
        return 0.0
    grid_shape = getattr(map_api, "grid_shape", None)
    cell_size = getattr(map_api, "cell_size", None)
    origin = getattr(map_api, "origin", None)
    if grid_shape is None or cell_size is None or origin is None:
        return 0.0
    cell_center = getattr(map_api, "cell_center", None)
    traversable_cells = getattr(map_api, "traversable_cells", None)
    if traversable_cells is not None:
        cells = list(traversable_cells)
    else:
        is_cell_free = getattr(map_api, "is_free_cell", None)
        if not callable(is_cell_free):
            is_cell_free = getattr(map_api, "is_cell_free", None)
        if not callable(is_cell_free):
            return 0.0
        cells = [
            (ix, iy)
            for ix in range(int(grid_shape[0]))
            for iy in range(int(grid_shape[1]))
            if bool(is_cell_free((ix, iy)))
        ]
    if not cells:
        return 0.0
    pose_xy = np.asarray([np.asarray(pose, dtype=float)[:2] for pose in poses_xyz])
    radius = max(float(radius_m), 0.0)
    covered = 0
    for cell in cells:
        if callable(cell_center):
            center_xy = np.asarray(cell_center(cell), dtype=float)
        else:
            center_xy = np.asarray(
                [
                    float(origin[0]) + (float(cell[0]) + 0.5) * float(cell_size),
                    float(origin[1]) + (float(cell[1]) + 0.5) * float(cell_size),
                ],
                dtype=float,
            )
        if float(np.min(np.linalg.norm(pose_xy - center_xy, axis=1))) <= radius:
            covered += 1
    return float(covered) / float(len(cells))


def _adaptive_mission_stop_reason(
    estimator: RotatingShieldPFEstimator,
    *,
    current_pose_idx: int,
    visited_poses_xyz: Sequence[NDArray[np.float64]],
    map_api: object | None,
    min_poses: int,
    coverage_radius_m: float,
    coverage_fraction_threshold: float,
    ig_threshold: float,
    planning_live_time_s: float,
    require_quiet_birth_residual: bool = True,
    birth_residual_min_support: int = 1,
    require_pf_convergence_for_coverage: bool = False,
    require_no_unresolved_discriminative_failures: bool = True,
    unresolved_discriminative_failure_min_count: int = 1,
    require_pf_cardinality_ready: bool = True,
    remaining_measurement_estimate: Mapping[str, Any] | None = None,
    require_remaining_measurement_ready: bool = True,
) -> str | None:
    """Return an adaptive mission-stop reason when exploration is sufficiently complete."""
    if len(visited_poses_xyz) < max(1, int(min_poses)):
        return None
    if bool(require_no_unresolved_discriminative_failures) and (
        _has_unresolved_discriminative_pseudo_failures(
            estimator,
            min_count=int(unresolved_discriminative_failure_min_count),
        )
    ):
        return None
    cardinality_ready, _cardinality_reason = _source_cardinality_dwell_status(
        estimator,
        refresh_estimates=False,
    )
    if bool(require_pf_cardinality_ready) and not cardinality_ready:
        return None
    if bool(require_remaining_measurement_ready) and not (
        _remaining_measurement_ready_for_stop(remaining_measurement_estimate)
    ):
        return None
    if _all_pf_filters_converged(estimator, refresh_estimates=False):
        if bool(require_quiet_birth_residual) and _has_birth_residual_evidence(
            estimator,
            min_support=int(birth_residual_min_support),
        ):
            return None
        return "pf_filters_converged"
    if estimator.should_stop_exploration(
        ig_threshold=float(ig_threshold),
        live_time_s=float(planning_live_time_s),
    ):
        return "pf_converged_low_information_gain"
    coverage = _coverage_fraction_for_poses(
        map_api,
        visited_poses_xyz,
        radius_m=float(coverage_radius_m),
    )
    if coverage >= float(coverage_fraction_threshold):
        if bool(require_quiet_birth_residual) and _has_birth_residual_evidence(
            estimator,
            min_support=int(birth_residual_min_support),
        ):
            return None
        if bool(
            require_pf_convergence_for_coverage
        ) and not estimator.should_stop_exploration(
            ig_threshold=float(ig_threshold),
            live_time_s=float(planning_live_time_s),
        ):
            return None
        return f"environment_coverage:{coverage:.3f}"
    if current_pose_idx >= 0 and estimator.should_stop_shield_rotation(
        pose_idx=int(current_pose_idx),
        ig_threshold=float(ig_threshold),
        live_time_s=float(planning_live_time_s),
    ):
        return "current_pose_converged"
    return None


def _posterior_cardinality_summary(filt: object) -> tuple[float, float]:
    """Return posterior mean and variance of physical PF source count."""
    particles = list(getattr(filt, "continuous_particles", []) or [])
    if not particles:
        return 0.0, 0.0
    weights = np.asarray(getattr(filt, "continuous_weights", []), dtype=float)
    if weights.size != len(particles):
        weights = np.ones(len(particles), dtype=float)
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        weights = np.full(len(particles), 1.0 / max(len(particles), 1), dtype=float)
    else:
        weights = weights / total_weight
    counts: list[float] = []
    for particle in particles:
        state = getattr(particle, "state", None)
        if state is None:
            counts.append(0.0)
            continue
        counts.append(float(max(0, int(getattr(state, "num_sources", 0)))))
    counts_arr = np.asarray(counts, dtype=float)
    mean = float(np.sum(weights * counts_arr))
    variance = float(np.sum(weights * (counts_arr - mean) ** 2))
    return mean, variance




def _all_pf_filters_converged(
    estimator: RotatingShieldPFEstimator,
    *,
    refresh_estimates: bool = True,
) -> bool:
    """Return True when every enabled isotope PF reports convergence."""
    del refresh_estimates
    pf_config = getattr(estimator, "pf_config", None)
    if not bool(getattr(pf_config, "converge_enable", False)):
        return False
    filters = getattr(estimator, "filters", {})
    if not filters:
        return False
    for filt in filters.values():
        filt_config = getattr(filt, "config", pf_config)
        if not bool(getattr(filt_config, "converge_enable", False)):
            return False
        if not bool(getattr(filt, "is_converged", False)):
            return False
    return True


def _source_cardinality_dwell_status(
    estimator: RotatingShieldPFEstimator,
    *,
    refresh_estimates: bool = True,
) -> tuple[bool, str]:
    """Return whether PF structural evidence supports ending adaptive dwell."""
    if bool(refresh_estimates):
        try:
            estimator.estimates()
        except RuntimeError:
            return False, "pf_posterior_unavailable"
    unresolved_getter = getattr(estimator, "unresolved_structural_evidence", None)
    if callable(unresolved_getter):
        try:
            unresolved = unresolved_getter()
        except (RuntimeError, ValueError, TypeError):
            unresolved = {}
        if isinstance(unresolved, dict) and unresolved:
            labels = ",".join(str(key) for key in sorted(unresolved))
            return False, f"unresolved_structural:{labels}"
    filters = getattr(estimator, "filters", {})
    if not isinstance(filters, dict) or not filters:
        return False, "no_pf_posterior"
    pf_config = getattr(estimator, "pf_config", None)
    variance_limit = max(
        float(getattr(pf_config, "converge_cardinality_var_max", 0.05)),
        0.0,
    )
    pending = [
        str(isotope)
        for isotope, filt in sorted(filters.items())
        if _posterior_cardinality_summary(filt)[1] > variance_limit + 1.0e-9
    ]
    if pending:
        return False, f"pf_cardinality_variance:{','.join(pending)}"
    return True, "pf_cardinality_ready"




def _has_unresolved_discriminative_pseudo_failures(
    estimator: RotatingShieldPFEstimator,
    *,
    min_count: int,
) -> bool:
    """Return True when pseudo-source failures still request discriminative views."""
    count_floor = max(1, int(min_count))
    unresolved_reasons = {
        "needs_discriminative_views",
        "insufficient_distinct_views",
        "high_response_corr",
    }
    filters = getattr(estimator, "filters", {})
    for filt in filters.values():
        reasons = getattr(filt, "last_pseudo_source_fail_reasons", {})
        if isinstance(reasons, dict):
            unresolved = sum(
                int(reasons.get(reason, 0)) for reason in unresolved_reasons
            )
            if unresolved >= count_floor:
                return True
        elif int(getattr(filt, "last_pseudo_source_failed", 0)) >= count_floor:
            return True
    return False


def _final_pf_cardinality_status(estimator: object) -> dict[str, Any]:
    """Return PF cardinality and unresolved structural evidence for JSON output."""
    getter = getattr(estimator, "posterior_cardinality_distribution", None)
    distributions = dict(getter()) if callable(getter) else {}
    cardinality: dict[str, dict[str, Any]] = {}
    for isotope, distribution_raw in sorted(distributions.items()):
        distribution = {
            int(key): max(float(value), 0.0)
            for key, value in dict(distribution_raw).items()
        }
        total = float(sum(distribution.values()))
        if total > 0.0:
            distribution = {
                key: value / total for key, value in distribution.items()
            }
        counts = np.asarray(list(distribution), dtype=float)
        probabilities = np.asarray(list(distribution.values()), dtype=float)
        mean = float(np.sum(counts * probabilities)) if counts.size else 0.0
        variance = (
            float(np.sum(probabilities * (counts - mean) ** 2))
            if counts.size
            else 0.0
        )
        positive = probabilities[probabilities > 0.0]
        entropy = float(-np.sum(positive * np.log(positive)))
        cardinality[str(isotope)] = {
            "distribution": {
                str(key): float(value)
                for key, value in sorted(distribution.items())
            },
            "mean": mean,
            "variance": variance,
            "entropy_nats": entropy,
        }
    pseudo_by_isotope: dict[str, dict[str, Any]] = {}
    unresolved_reason_totals: dict[str, int] = {}
    unresolved_structural_evidence: dict[str, Any] = {}
    if hasattr(estimator, "unresolved_structural_evidence"):
        try:
            unresolved_structural_evidence = dict(
                estimator.unresolved_structural_evidence()
            )
        except RuntimeError:
            unresolved_structural_evidence = {}
    filters = getattr(estimator, "filters", {})
    if isinstance(filters, dict):
        for isotope, filt in sorted(filters.items()):
            reasons = getattr(filt, "last_pseudo_source_fail_reasons", {})
            reason_payload = (
                {str(reason): int(count) for reason, count in dict(reasons).items()}
                if isinstance(reasons, dict)
                else {}
            )
            for reason, count in reason_payload.items():
                unresolved_reason_totals[reason] = int(
                    unresolved_reason_totals.get(reason, 0)
                ) + int(count)
            pseudo_by_isotope[str(isotope)] = {
                "verified": int(getattr(filt, "last_pseudo_source_verified", 0)),
                "failed": int(getattr(filt, "last_pseudo_source_failed", 0)),
                "pruned": int(getattr(filt, "last_pseudo_source_pruned", 0)),
                "quarantined": int(getattr(filt, "last_pseudo_source_quarantined", 0)),
                "quarantine_active": int(
                    getattr(filt, "last_pseudo_source_quarantine_active", 0)
                ),
                "fail_reasons": reason_payload,
            }
    return {
        "source": "pf_posterior",
        "all_structural_evidence_resolved": not unresolved_structural_evidence,
        "pf_cardinality": cardinality,
        "unresolved_pseudo_source_reasons": unresolved_reason_totals,
        "unresolved_structural_evidence": unresolved_structural_evidence,
        "pseudo_source_verification": pseudo_by_isotope,
    }


def _signature_vector_is_dependent(
    vector: NDArray[np.float64],
    previous_vectors: Sequence[NDArray[np.float64]],
    *,
    cosine_threshold: float,
    min_norm: float = 1e-9,
) -> bool:
    """Return True when a shield signature vector adds little new direction."""
    threshold = float(cosine_threshold)
    if threshold <= 0.0 or threshold >= 1.0:
        return False
    candidate = np.asarray(vector, dtype=float).ravel()
    candidate_norm = float(np.linalg.norm(candidate))
    if candidate_norm <= float(min_norm):
        return False
    for previous in previous_vectors:
        prev = np.asarray(previous, dtype=float).ravel()
        if prev.shape != candidate.shape:
            continue
        prev_norm = float(np.linalg.norm(prev))
        if prev_norm <= float(min_norm):
            continue
        cosine = float(
            np.dot(candidate, prev) / max(candidate_norm * prev_norm, min_norm)
        )
        if abs(cosine) >= threshold:
            return True
    return False


def _save_spectrum_plot(
    decomposer: SpectralDecomposer,
    spectrum: np.ndarray,
    output_path: Path,
    peak_tolerance_keV: float = 10.0,
    highlight_isotopes: set[str] | None = None,
    counts_by_isotope: dict[str, float] | None = None,
    component_spectra_by_isotope: dict[str, NDArray[np.float64]] | None = None,
    use_detection_lines: bool = True,
    window_keV: float | None = None,
    window_sigma: float = 3.0,
    title: str = "Final measurement spectrum",
) -> None:
    """Save the measurement spectrum with nuclide lines and colored count windows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    energy_axis = decomposer.energy_axis
    library = decomposer.library
    if highlight_isotopes is not None:
        library = {iso: library[iso] for iso in library if iso in highlight_isotopes}
    line_map: dict[str, list[float]] = {}
    for iso, nuclide in library.items():
        if use_detection_lines:
            lines = get_detection_lines_keV(iso)
        else:
            lines = [line.energy_keV for line in nuclide.lines]
        if lines:
            line_map[iso] = lines
    colors = _build_isotope_colors(list(library.keys()))
    smoothed = gaussian_smooth(
        spectrum,
        sigma_bins=2.0,
        use_gpu=decomposer.use_gpu,
        gpu_device=decomposer.gpu_device,
        gpu_dtype=decomposer.gpu_dtype,
    )
    baseline = baseline_als(
        smoothed,
        lam=decomposer.config.baseline_lam,
        p=decomposer.config.baseline_p,
        niter=decomposer.config.baseline_niter,
    )
    corrected = np.clip(smoothed - baseline, a_min=0.0, a_max=None)
    peak_indices = detect_peaks(corrected, prominence=0.05, distance=5)
    line_energies = (
        {iso: np.array(lines, dtype=float) for iso, lines in line_map.items()}
        if use_detection_lines
        else None
    )
    peaks_by_iso, unassigned = decomposer._assign_peak_indices(
        energy_axis,
        peak_indices,
        library,
        tolerance_keV=peak_tolerance_keV,
        line_energies=line_energies,
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    component_labels: list[str] = []
    component_values: list[NDArray[np.float64]] = []
    component_colors: list[object] = []
    if component_spectra_by_isotope:
        for iso in library:
            component_raw = component_spectra_by_isotope.get(iso)
            if component_raw is None:
                continue
            component = np.clip(
                np.asarray(component_raw, dtype=float)[: energy_axis.size],
                a_min=0.0,
                a_max=None,
            )
            if component.size != energy_axis.size or float(np.sum(component)) <= 0.0:
                continue
            component_values.append(component)
            component_colors.append(colors.get(iso, "gray"))
            component_labels.append(f"{iso} photopeak={float(np.sum(component)):.1f}")
    if component_values:
        ax.stackplot(
            energy_axis,
            component_values,
            labels=component_labels,
            colors=component_colors,
            alpha=0.45,
        )
    ax.plot(
        energy_axis, smoothed, color="black", linewidth=1.0, label="Processed spectrum"
    )
    for iso, nuclide in library.items():
        if iso not in line_map:
            continue
        if counts_by_isotope is not None and counts_by_isotope.get(iso, 0.0) <= 0.0:
            continue
        color = colors.get(iso, "gray")
        for line_keV in line_map[iso]:
            half_width = window_keV
            if half_width is None:
                sigma = float(decomposer.resolution_fn(line_keV))
                sigma_width = max(window_sigma * sigma, 1e-6)
                if use_detection_lines:
                    half_width = max(
                        float(decomposer.config.detect_half_window_keV), sigma_width
                    )
                else:
                    half_width = sigma_width
            mask = np.abs(energy_axis - line_keV) <= float(half_width)
            if np.any(mask):
                ax.fill_between(
                    energy_axis[mask],
                    baseline[mask],
                    smoothed[mask],
                    color=color,
                    alpha=0.2,
                    linewidth=0.0,
                )
    for iso, nuclide in library.items():
        if iso not in line_map:
            continue
        color = colors.get(iso, "gray")
        labeled = False
        for line_keV in line_map[iso]:
            label = iso if not labeled else None
            ax.axvline(
                line_keV,
                color=color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
                label=label,
            )
            labeled = True
    for iso, idxs in peaks_by_iso.items():
        if highlight_isotopes is not None and iso not in highlight_isotopes:
            continue
        if idxs:
            ax.scatter(
                energy_axis[idxs],
                spectrum[idxs],
                color=colors.get(iso, "gray"),
                s=28,
                zorder=3,
            )
    if unassigned and highlight_isotopes is None:
        ax.scatter(
            energy_axis[unassigned],
            spectrum[unassigned],
            color="gray",
            s=20,
            zorder=3,
            alpha=0.6,
        )
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if library:
        ax.legend(loc="upper right", fontsize=8, title="Nuclide lines")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _validate_measurement_timing(
    *,
    measurement_time_s: float,
    adaptive_dwell: bool,
    adaptive_dwell_chunk_s: float,
    adaptive_min_dwell_s: float,
    adaptive_ready_min_counts: float,
    adaptive_ready_min_isotopes: int,
    adaptive_ready_min_snr: float,
) -> None:
    """Validate fixed and adaptive dwell-time settings."""
    if measurement_time_s <= 0.0 and not adaptive_dwell:
        raise ValueError("measurement_time_s must be positive for fixed dwell.")
    if not adaptive_dwell:
        return
    if adaptive_dwell_chunk_s <= 0.0:
        raise ValueError("adaptive_dwell_chunk_s must be positive.")
    if adaptive_min_dwell_s <= 0.0:
        raise ValueError("adaptive_min_dwell_s must be positive.")
    has_dwell_cap = measurement_time_s > 0.0 and np.isfinite(measurement_time_s)
    if has_dwell_cap and adaptive_min_dwell_s > measurement_time_s:
        raise ValueError("adaptive_min_dwell_s cannot exceed measurement_time_s.")
    if adaptive_ready_min_counts < 0.0:
        raise ValueError("adaptive_ready_min_counts cannot be negative.")
    if adaptive_ready_min_isotopes < 0:
        raise ValueError("adaptive_ready_min_isotopes cannot be negative.")
    if adaptive_ready_min_snr < 0.0:
        raise ValueError("adaptive_ready_min_snr cannot be negative.")


def _observation_spectrum_array(
    observation: SimulationObservation,
    decomposer: SpectralDecomposer,
) -> NDArray[np.float64]:
    """Return a validated spectrum array from a simulator observation."""
    spectrum = np.asarray(observation.spectrum_counts, dtype=float)
    if spectrum.shape != decomposer.energy_axis.shape:
        raise ValueError(
            "Simulator returned an unexpected spectrum shape: "
            f"{spectrum.shape} != {decomposer.energy_axis.shape}"
        )
    return spectrum


def _metadata_spectrum_variance(
    metadata: dict[str, object],
    expected_shape: tuple[int, ...],
) -> NDArray[np.float64] | None:
    """Return a validated per-bin spectrum variance array from metadata."""
    raw = metadata.get("spectrum_count_variance")
    if raw is None:
        return None
    variance = np.asarray(raw, dtype=float)
    if variance.shape != expected_shape:
        return None
    return np.clip(variance, a_min=0.0, a_max=None)


def _should_fold_incident_gamma_detector_response(
    observation: SimulationObservation,
    decomposer: SpectralDecomposer,
) -> bool:
    """Return whether fast Geant4 incident-gamma spectra need detector response folding."""
    if not bool(decomposer.config.apply_incident_gamma_detector_response):
        return False
    metadata = observation.metadata
    scoring_mode = str(metadata.get("detector_scoring_mode", "")).strip().lower()
    fast_scoring = str(metadata.get("detector_fast_scoring", "")).strip().lower()
    return scoring_mode == "incident_gamma_energy" or fast_scoring == "true"


def _analysis_spectrum_array(
    observation: SimulationObservation,
    decomposer: SpectralDecomposer,
) -> NDArray[np.float64]:
    """Return the pulse-height spectrum used for display and isotope count extraction."""
    spectrum = _observation_spectrum_array(observation, decomposer)
    if _should_fold_incident_gamma_detector_response(observation, decomposer):
        return decomposer.fold_incident_gamma_spectrum(spectrum)
    return spectrum


def _analysis_spectrum_variance(
    observation: SimulationObservation,
    decomposer: SpectralDecomposer,
) -> NDArray[np.float64] | None:
    """Return the variance of the analysis spectrum when simulator metadata provides it."""
    raw_spectrum = _observation_spectrum_array(observation, decomposer)
    variance = _metadata_spectrum_variance(observation.metadata, raw_spectrum.shape)
    if variance is None:
        return None
    if _should_fold_incident_gamma_detector_response(observation, decomposer):
        return decomposer.fold_incident_gamma_spectrum_variance(variance)
    return variance


def _spectrum_evidence_payload(
    decomposer: SpectralDecomposer,
    spectrum: NDArray[np.float64],
    *,
    live_time_s: float,
    spectrum_variance: NDArray[np.float64] | None,
    isotopes: Sequence[str],
) -> dict[str, object] | None:
    """Return direct spectrum-bin evidence payload for one runtime measurement."""
    spectrum_arr = np.asarray(spectrum, dtype=float).reshape(-1)
    if spectrum_arr.size == 0:
        return None
    template_getter = getattr(decomposer, "count_response_templates", None)
    if not callable(template_getter):
        return None
    templates = template_getter([str(isotope) for isotope in isotopes])
    template_payload = {
        str(isotope): np.asarray(template, dtype=float).reshape(-1).copy()
        for isotope, template in templates.items()
        if np.asarray(template, dtype=float).reshape(-1).size == spectrum_arr.size
    }
    if not template_payload:
        return None
    background_getter = getattr(decomposer, "configured_background_spectrum", None)
    background = (
        background_getter(float(live_time_s)) if callable(background_getter) else None
    )
    background_arr = None
    if background is not None:
        candidate_background = np.asarray(background, dtype=float).reshape(-1)
        if candidate_background.size == spectrum_arr.size:
            background_arr = candidate_background.copy()
    variance_arr = None
    if spectrum_variance is not None:
        candidate_variance = np.asarray(spectrum_variance, dtype=float).reshape(-1)
        if candidate_variance.size == spectrum_arr.size:
            variance_arr = candidate_variance.copy()
    return {
        "spectrum_counts": spectrum_arr.copy(),
        "spectrum_variance": variance_arr,
        "spectrum_background": background_arr,
        "spectrum_background_source": (
            "configured_rate_and_detector_background_shape"
            if background_arr is not None
            else "joint_nuisance_only"
        ),
        "spectrum_background_observation_independent": True,
        "spectrum_response_templates_by_isotope": template_payload,
    }


def _callable_accepts_keyword(
    callable_obj: Callable[..., object], keyword: str
) -> bool:
    """Return True when a callable accepts the requested keyword argument."""
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    if keyword in signature.parameters:
        return True
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _evaluate_spectrum_counts(
    decomposer: SpectralDecomposer,
    spectrum: NDArray[np.float64],
    *,
    live_time_s: float,
    spectrum_count_method: str,
    detect_threshold_abs: float,
    detect_threshold_rel: float,
    detect_threshold_rel_by_isotope: dict[str, float],
    min_peaks_by_isotope: dict[str, int] | None,
    spectrum_variance: NDArray[np.float64] | None = None,
    transport_metadata: dict[str, object] | None = None,
    transport_spectrum: NDArray[np.float64] | None = None,
    transport_covariance_chunks: tuple[ResponsePoissonCovarianceChunk, ...] = (),
    candidate_isotopes: Sequence[str] | None = None,
) -> tuple[dict[str, float], dict[str, float], set[str]]:
    """Extract isotope counts, count variances, and detected labels."""
    result = _evaluate_spectrum_count_result(
        decomposer,
        spectrum,
        live_time_s=live_time_s,
        spectrum_count_method=spectrum_count_method,
        detect_threshold_abs=detect_threshold_abs,
        detect_threshold_rel=detect_threshold_rel,
        detect_threshold_rel_by_isotope=detect_threshold_rel_by_isotope,
        min_peaks_by_isotope=min_peaks_by_isotope,
        spectrum_variance=spectrum_variance,
        transport_metadata=transport_metadata,
        transport_spectrum=transport_spectrum,
        transport_covariance_chunks=transport_covariance_chunks,
        candidate_isotopes=candidate_isotopes,
    )
    return result.counts, result.variances, result.detected


def _evaluate_spectrum_count_result(
    decomposer: SpectralDecomposer,
    spectrum: NDArray[np.float64],
    *,
    live_time_s: float,
    spectrum_count_method: str,
    detect_threshold_abs: float,
    detect_threshold_rel: float,
    detect_threshold_rel_by_isotope: dict[str, float],
    min_peaks_by_isotope: dict[str, int] | None,
    spectrum_variance: NDArray[np.float64] | None = None,
    transport_metadata: dict[str, object] | None = None,
    transport_spectrum: NDArray[np.float64] | None = None,
    transport_covariance_chunks: tuple[ResponsePoissonCovarianceChunk, ...] = (),
    candidate_isotopes: Sequence[str] | None = None,
) -> RuntimeCountResult:
    """Extract PF-ready count means, variances, detections, and covariance."""
    extractor = RuntimeCountExtractor(
        decomposer,
        count_method=spectrum_count_method,
    )
    result = extractor.extract(
        spectrum,
        live_time_s=live_time_s,
        detect_threshold_abs=detect_threshold_abs,
        detect_threshold_rel=detect_threshold_rel,
        detect_threshold_rel_by_isotope=detect_threshold_rel_by_isotope,
        min_peaks_by_isotope=min_peaks_by_isotope,
        spectrum_variance=spectrum_variance,
        transport_metadata=transport_metadata,
        transport_spectrum=transport_spectrum,
        transport_covariance_chunks=transport_covariance_chunks,
    )
    if candidate_isotopes is None:
        return result
    candidate_set = {str(isotope) for isotope in candidate_isotopes}
    counts = {
        isotope: value
        for isotope, value in result.counts.items()
        if isotope in candidate_set
    }
    variances = {
        isotope: value
        for isotope, value in result.variances.items()
        if isotope in candidate_set
    }
    detected = {isotope for isotope in result.detected if isotope in candidate_set}
    covariance = _filter_count_covariance(result.covariance, candidate_set)
    return RuntimeCountResult(counts, variances, detected, covariance)


def _filter_count_covariance(
    covariance: Mapping[str, Mapping[str, float]] | None,
    isotope_set: set[str],
) -> dict[str, dict[str, float]] | None:
    """Return a covariance payload restricted to selected isotope channels."""
    if covariance is None:
        return None
    filtered: dict[str, dict[str, float]] = {}
    for row_iso, row_payload in covariance.items():
        row_key = str(row_iso)
        if row_key not in isotope_set or not isinstance(row_payload, Mapping):
            continue
        row: dict[str, float] = {}
        for col_iso, value in row_payload.items():
            col_key = str(col_iso)
            if col_key not in isotope_set:
                continue
            try:
                row[col_key] = float(value)
            except (TypeError, ValueError):
                continue
        filtered[row_key] = row
    return filtered


def _metadata_count_covariance(
    metadata: Mapping[str, object],
    isotope_set: set[str] | None = None,
) -> dict[str, dict[str, float]] | None:
    """Return a count covariance payload stored on simulation metadata."""
    payload = metadata.get("count_covariance_by_isotope")
    if not isinstance(payload, Mapping):
        payload = metadata.get("adaptive_dwell_count_covariance_by_isotope")
    if not isinstance(payload, Mapping):
        return None
    allowed = isotope_set if isotope_set is not None else {str(key) for key in payload}
    covariance = _filter_count_covariance(payload, allowed)
    return covariance


def _complete_count_covariance(
    variances: Mapping[str, float],
    covariance: Mapping[str, Mapping[str, float]] | None,
    isotopes: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Return the symmetric covariance used identically by live PF and replay."""
    isotope_order = [str(isotope) for isotope in isotopes]
    matrix = np.diag(
        [float(variances.get(isotope, 1.0)) for isotope in isotope_order]
    ).astype(np.float64)
    if isinstance(covariance, Mapping):
        for row_index, row_isotope in enumerate(isotope_order):
            row_payload = covariance.get(row_isotope, {})
            if not isinstance(row_payload, Mapping):
                continue
            if row_isotope in row_payload:
                value = float(row_payload[row_isotope])
                if np.isfinite(value):
                    matrix[row_index, row_index] = value
        for row_index, row_isotope in enumerate(isotope_order):
            row_payload = covariance.get(row_isotope, {})
            if not isinstance(row_payload, Mapping):
                row_payload = {}
            for column_index in range(row_index + 1, len(isotope_order)):
                column_isotope = isotope_order[column_index]
                column_payload = covariance.get(column_isotope, {})
                if not isinstance(column_payload, Mapping):
                    column_payload = {}
                pair_values = []
                for payload, key in (
                    (row_payload, column_isotope),
                    (column_payload, row_isotope),
                ):
                    if key not in payload:
                        continue
                    value = float(payload[key])
                    if np.isfinite(value):
                        pair_values.append(value)
                if pair_values:
                    symmetric_value = float(np.mean(pair_values))
                    matrix[row_index, column_index] = symmetric_value
                    matrix[column_index, row_index] = symmetric_value
    return {
        row_isotope: {
            column_isotope: float(matrix[row_index, column_index])
            for column_index, column_isotope in enumerate(isotope_order)
        }
        for row_index, row_isotope in enumerate(isotope_order)
    }


def _store_count_covariance_metadata(
    metadata: dict[str, object],
    covariance: Mapping[str, Mapping[str, float]] | None,
) -> None:
    """Store JSON-safe isotope count covariance on an observation metadata dict."""
    if covariance is None:
        return
    metadata["count_covariance_by_isotope"] = {
        str(row_iso): {
            str(col_iso): float(value) for col_iso, value in row_payload.items()
        }
        for row_iso, row_payload in covariance.items()
        if isinstance(row_payload, Mapping)
    }


def _is_adaptive_spectrum_ready(
    counts_by_isotope: dict[str, float],
    count_variance_by_isotope: dict[str, float] | None = None,
    *,
    live_time_s: float,
    min_live_time_s: float,
    min_counts_per_detected_isotope: float,
    min_detected_isotopes: int,
    candidate_isotopes: list[str] | tuple[str, ...] | None = None,
    min_snr: float = 3.0,
    total_spectrum_counts: float | None = None,
    informative_low_total_factor: float = 20.0,
    informative_low_fraction: float = 0.01,
    informative_low_min_live_s: float = 20.0,
    informative_low_count_fraction: float = 0.5,
    allow_informative_low: bool = True,
    allow_low_signal_stop: bool = False,
    low_signal_min_live_s: float = 120.0,
    low_signal_upper_sigma: float = 3.0,
    low_signal_count_fraction: float = 0.05,
    low_signal_projected_live_factor: float = 4.0,
) -> tuple[bool, str]:
    """Return whether an accumulated spectrum is usable for isotope counts."""
    if live_time_s + 1e-12 < min_live_time_s:
        return False, "below_min_live_time"
    if min_detected_isotopes <= 0:
        return True, "min_live_time_reached"
    if candidate_isotopes is None:
        candidate_isotopes = tuple(sorted(counts_by_isotope))
    required = min(int(min_detected_isotopes), len(candidate_isotopes))
    if required <= 0:
        return True, "min_live_time_reached"
    min_counts = float(min_counts_per_detected_isotope)
    min_snr = max(float(min_snr), 0.0)
    total_counts = (
        None
        if total_spectrum_counts is None
        else max(float(total_spectrum_counts), 0.0)
    )
    informative_total_floor = max(min_counts * float(informative_low_total_factor), 0.0)
    informative_fraction = max(float(informative_low_fraction), 0.0)
    informative_count_ceiling = max(
        1.0,
        min_counts * max(float(informative_low_count_fraction), 0.0),
    )
    enough_live_for_low = live_time_s + 1e-12 >= max(
        min_live_time_s,
        float(informative_low_min_live_s),
    )
    variances = count_variance_by_isotope or {}
    usable = []
    informative_low = []
    unresolved_upper_below = []
    unresolved_count_below = []
    projected_unproductive = []
    low_count_floor = max(min_counts * max(float(low_signal_count_fraction), 0.0), 0.0)
    projected_limit = max(
        live_time_s,
        max(min_live_time_s, float(low_signal_min_live_s))
        * max(float(low_signal_projected_live_factor), 1.0),
    )
    for iso in candidate_isotopes:
        count = max(float(counts_by_isotope.get(iso, 0.0)), 0.0)
        variance = max(float(variances.get(iso, max(count, 1.0))), 1.0)
        snr = count / np.sqrt(variance)
        count_ready = count >= min_counts
        snr_ready = True if min_counts <= 0.0 else snr >= min_snr
        if count_ready and snr_ready:
            usable.append(str(iso))
            continue
        upper_bound = count + max(float(low_signal_upper_sigma), 0.0) * np.sqrt(
            variance
        )
        if min_counts > 0.0 and upper_bound < min_counts:
            unresolved_upper_below.append(str(iso))
        if min_counts > 0.0 and count <= low_count_floor:
            unresolved_count_below.append(str(iso))
        if allow_informative_low:
            has_spectrum_evidence = (
                total_counts is not None
                and total_counts >= informative_total_floor
                and total_counts > 0.0
            )
            low_fraction = (
                enough_live_for_low
                and has_spectrum_evidence
                and count < min_counts
                and count <= informative_count_ceiling
                and (count / max(total_counts, 1.0)) <= informative_fraction
            )
            if low_fraction:
                informative_low.append(str(iso))
                continue
        if allow_low_signal_stop and live_time_s + 1e-12 >= max(
            min_live_time_s, float(low_signal_min_live_s)
        ):
            projected_live_s = 0.0
            if min_counts > 0.0 and not count_ready:
                projected_live_s = (
                    float("inf")
                    if count <= 0.0
                    else live_time_s * min_counts / max(count, 1.0e-12)
                )
            elif count_ready and not snr_ready and min_snr > 0.0:
                projected_live_s = (
                    float("inf")
                    if snr <= 0.0
                    else live_time_s * (min_snr / max(snr, 1.0e-12)) ** 2
                )
            if projected_live_s > projected_limit:
                projected_unproductive.append(
                    (str(iso), float(count), float(projected_live_s))
                )
    if len(usable) + len(informative_low) >= required:
        if informative_low:
            return (
                True,
                "isotope_count_estimates_ready:"
                f"positive={len(usable)},informative_low={len(informative_low)}",
            )
        return True, "isotope_count_estimates_ready"
    unresolved_count = len(candidate_isotopes) - len(usable)
    if (
        allow_low_signal_stop
        and min_counts > 0.0
        and unresolved_count > 0
        and live_time_s + 1e-12 >= max(min_live_time_s, float(low_signal_min_live_s))
        and (
            len(unresolved_upper_below) >= unresolved_count
            or len(unresolved_count_below) >= unresolved_count
        )
    ):
        if len(unresolved_upper_below) >= unresolved_count:
            reason_kind = "upper_bound"
            evidence_count = len(unresolved_upper_below)
        else:
            reason_kind = "count_floor"
            evidence_count = len(unresolved_count_below)
        return (
            True,
            f"low_signal_{reason_kind}:positive={len(usable)},below={evidence_count}",
        )
    if (
        allow_low_signal_stop
        and min_counts > 0.0
        and live_time_s + 1e-12 >= max(min_live_time_s, float(low_signal_min_live_s))
        and projected_unproductive
    ):
        projected_blocked = {iso for iso, _, _ in projected_unproductive}
        available_count = sum(
            1 for iso in candidate_isotopes if str(iso) not in projected_blocked
        )
        if available_count < required:
            best_iso, best_count, best_projected = max(
                projected_unproductive,
                key=lambda item: item[1],
            )
            return (
                True,
                "low_signal_projected_time:"
                f"positive={len(usable)},available={available_count},"
                f"blocked={len(projected_blocked)},best={best_count:.3f},"
                f"best_iso={best_iso},projected={best_projected:.1f}",
            )
    if (
        allow_low_signal_stop
        and min_counts > 0.0
        and not usable
        and live_time_s + 1e-12 >= max(min_live_time_s, float(low_signal_min_live_s))
    ):
        max_count = max(
            (
                max(float(counts_by_isotope.get(iso, 0.0)), 0.0)
                for iso in candidate_isotopes
            ),
            default=0.0,
        )
        if max_count <= 0.0:
            return (
                True,
                "low_signal_projected_time:positive=0,best=0.000,projected=inf",
            )
        projected_live_s = live_time_s * min_counts / max(max_count, 1e-12)
        if projected_live_s > projected_limit:
            return (
                True,
                "low_signal_projected_time:"
                f"positive=0,best={max_count:.3f},projected={projected_live_s:.1f}",
            )
    return (
        False,
        "insufficient_isotope_count_estimates:"
        f"{len(usable) + len(informative_low)}/{required}",
    )


def _adaptive_dwell_progress_message(
    *,
    step_id: int,
    chunk_index: int,
    live_time_s: float,
    counts_by_isotope: dict[str, float],
    count_variance_by_isotope: dict[str, float],
    reason: str,
) -> str:
    """Return a concise adaptive-dwell progress diagnostic message."""
    snr_by_isotope = {}
    for iso, count in counts_by_isotope.items():
        variance = max(float(count_variance_by_isotope.get(iso, max(count, 1.0))), 1.0)
        snr_by_isotope[iso] = float(max(count, 0.0) / np.sqrt(variance))
    return (
        f"[adaptive dwell step {step_id}] "
        f"chunks={chunk_index + 1} live={live_time_s:.1f}s "
        f"counts={_fmt_counts(counts_by_isotope)} "
        f"snr={_fmt_counts(snr_by_isotope)} "
        f"reason={reason}"
    )


def _inflate_low_signal_variances(
    counts_by_isotope: dict[str, float],
    count_variance_by_isotope: dict[str, float],
    *,
    min_counts_per_detected_isotope: float,
    ready_reason: str,
) -> dict[str, float]:
    """Return count variances with censored low-signal observations softened."""
    reason = str(ready_reason)
    should_soften_subthreshold = (
        reason.startswith("low_signal_")
        or reason.startswith("max_dwell_reached")
        or reason == "isotope_count_estimates_ready"
    )
    if not should_soften_subthreshold:
        return {
            iso: float(max(var, 1.0)) for iso, var in count_variance_by_isotope.items()
        }
    threshold_var = max(float(min_counts_per_detected_isotope), 1.0) ** 2
    inflated: dict[str, float] = {}
    for isotope, count in counts_by_isotope.items():
        base_var = float(count_variance_by_isotope.get(isotope, max(float(count), 1.0)))
        if max(float(count), 0.0) < float(min_counts_per_detected_isotope):
            inflated[isotope] = float(max(base_var, threshold_var, 1.0))
        else:
            inflated[isotope] = float(max(base_var, 1.0))
    return inflated


def _merge_adaptive_observation_chunks(
    *,
    logical_step_id: int,
    observations: list[SimulationObservation],
    chunk_live_times_s: list[float],
    ready_reason: str,
    counts_by_isotope: dict[str, float],
    count_variance_by_isotope: dict[str, float],
    detected_isotopes: set[str],
    count_covariance_by_isotope: dict[str, dict[str, float]] | None = None,
) -> SimulationObservation:
    """Combine multiple simulator observations into one logical measurement."""
    if not observations:
        raise ValueError("At least one observation chunk is required.")
    first = observations[0]
    edge_ref = np.asarray(first.energy_bin_edges_keV, dtype=float)
    spectrum_total = np.zeros_like(np.asarray(first.spectrum_counts, dtype=float))
    spectrum_variance_total = np.zeros_like(spectrum_total, dtype=float)
    has_spectrum_variance = False
    for observation in observations:
        edges = np.asarray(observation.energy_bin_edges_keV, dtype=float)
        if edges.shape != edge_ref.shape or not np.allclose(edges, edge_ref):
            raise ValueError("Adaptive dwell chunks returned inconsistent energy bins.")
        spectrum_chunk = np.asarray(observation.spectrum_counts, dtype=float)
        spectrum_total += spectrum_chunk
        chunk_variance = _metadata_spectrum_variance(
            observation.metadata,
            spectrum_chunk.shape,
        )
        if chunk_variance is not None:
            spectrum_variance_total += chunk_variance
            has_spectrum_variance = True
    metadata = dict(observations[-1].metadata)
    chunk_transport_provenance: list[dict[str, object]] = []
    for chunk_index, (observation, chunk_live_time_s) in enumerate(
        zip(observations, chunk_live_times_s, strict=True)
    ):
        provenance: dict[str, object] = {
            "chunk_index": int(chunk_index),
            "step_id": int(observation.step_id),
            "commanded_dwell_time_s": float(chunk_live_time_s),
        }
        provenance.update(_measurement_transport_provenance(observation.metadata))
        chunk_transport_provenance.append(provenance)
    additive_metadata_keys = {
        "dwell_time_s",
        "num_primaries",
        "expected_physical_primaries",
        "expected_detector_equivalent_primaries",
        "expected_unthinned_primaries",
        "expected_sampled_primaries",
        "total_spectrum_counts",
        "total_track_steps",
        "detector_hit_events",
        "detector_hit_steps",
        "secondary_count",
        "killed_non_gamma_secondary_count",
        "process_count_compton",
        "process_count_rayleigh",
        "process_count_photoelectric",
        "pre_dead_time_total_spectrum_counts",
        "pre_dead_time_weighted_spectrum_sumw2",
        "weighted_spectrum_sumw2",
        "run_time_s",
    }
    transport_count_prefixes = (
        "transport_detected_counts_",
        "transport_uncollided_primary_counts_",
        "transport_interacted_primary_counts_",
        "transport_secondary_counts_",
    )
    additive_metadata_keys.update(
        key
        for observation in observations
        for key in observation.metadata
        if str(key).startswith("source_equivalent_counts_")
    )
    additive_metadata_keys.update(
        key
        for observation in observations
        for key in observation.metadata
        if str(key).startswith(transport_count_prefixes)
    )
    for key in sorted(additive_metadata_keys):
        values = [
            _metadata_float(observation.metadata, key) for observation in observations
        ]
        finite_values = [value for value in values if value is not None]
        optional_counter = str(key).startswith("source_equivalent_counts_")
        if len(finite_values) == len(observations) or (
            optional_counter and finite_values
        ):
            metadata[key] = float(sum(finite_values))
        elif key in metadata:
            metadata.pop(key)
    metadata["dwell_time_s"] = float(sum(chunk_live_times_s))

    chunk_sampling_fractions = [
        _metadata_float(observation.metadata, "primary_sampling_fraction")
        for observation in observations
    ]
    chunk_history_weights = [
        _metadata_float(observation.metadata, "primary_history_weight")
        for observation in observations
    ]
    finite_sampling_fractions = [
        float(value) for value in chunk_sampling_fractions if value is not None
    ]
    finite_history_weights = [
        float(value) for value in chunk_history_weights if value is not None
    ]
    if len(finite_sampling_fractions) == len(observations):
        metadata["adaptive_dwell_chunk_primary_sampling_fractions"] = (
            finite_sampling_fractions
        )
    else:
        metadata.pop("primary_sampling_fraction", None)
    if len(finite_history_weights) == len(observations):
        metadata["adaptive_dwell_chunk_primary_history_weights"] = (
            finite_history_weights
        )
    else:
        metadata.pop("primary_history_weight", None)

    aggregate_unthinned = _metadata_float(metadata, "expected_unthinned_primaries")
    aggregate_sampled = _metadata_float(metadata, "expected_sampled_primaries")
    aggregate_fraction: float | None = None
    if (
        aggregate_unthinned is not None
        and aggregate_sampled is not None
        and aggregate_unthinned > 0.0
    ):
        aggregate_fraction = float(aggregate_sampled / aggregate_unthinned)
    elif finite_sampling_fractions and np.allclose(
        finite_sampling_fractions,
        finite_sampling_fractions[0],
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        aggregate_fraction = float(finite_sampling_fractions[0])
    if (
        aggregate_fraction is not None
        and np.isfinite(aggregate_fraction)
        and aggregate_fraction > 0.0
        and aggregate_fraction <= 1.0
    ):
        metadata["primary_sampling_fraction"] = aggregate_fraction
        metadata["primary_history_weight"] = float(1.0 / aggregate_fraction)
        metadata["adaptive_dwell_effective_primary_sampling_fraction"] = (
            aggregate_fraction
        )
    else:
        metadata.pop("primary_sampling_fraction", None)
        metadata.pop("primary_history_weight", None)

    if len(observations) > 1:
        metadata["primary_sampling_fraction_resolution"] = "adaptive_chunk_aggregate"
        metadata["adaptive_dwell_primary_sampling_fraction_semantics"] = (
            "expected_primary_weighted_aggregate_across_independent_chunks"
        )
        metadata["adaptive_dwell_primary_history_weight_semantics"] = (
            "inverse_aggregate_fraction_diagnostic_only; exact_weights_are_per_chunk"
        )
        metadata.pop("dead_time_observed_scale", None)
    budget_flags = [
        _metadata_bool(observation.metadata, "primary_sampling_budget_enabled")
        for observation in observations
    ]
    if len([value for value in budget_flags if value is not None]) == len(observations):
        metadata["primary_sampling_budget_enabled"] = bool(any(budget_flags))
    else:
        metadata.pop("primary_sampling_budget_enabled", None)
    history_thinning_flags = [
        _metadata_bool(observation.metadata, "history_thinning_enabled")
        for observation in observations
    ]
    if len([value for value in history_thinning_flags if value is not None]) == len(
        observations
    ):
        any_history_thinning = bool(any(history_thinning_flags))
        metadata["history_thinning_enabled"] = any_history_thinning
        metadata["transport_history_mode"] = (
            "adaptive_mixed_weighted_thinning"
            if any_history_thinning and not all(history_thinning_flags)
            else "weighted_thinning"
            if any_history_thinning
            else "full_unit_weight"
        )
    else:
        metadata.pop("history_thinning_enabled", None)
        metadata.pop("transport_history_mode", None)
    target_values = [
        _metadata_float(observation.metadata, "target_sampled_primaries")
        for observation in observations
    ]
    finite_targets = [float(value) for value in target_values if value is not None]
    targets_consistent = len(finite_targets) == len(observations) and np.allclose(
        finite_targets,
        finite_targets[0],
        rtol=0.0,
        atol=0.0,
    )
    if targets_consistent:
        metadata["target_sampled_primaries"] = int(finite_targets[0])
    else:
        metadata.pop("target_sampled_primaries", None)
    if any(value is True for value in budget_flags):
        metadata["adaptive_dwell_target_sampled_primaries_semantics"] = (
            "per_geant4_transport_invocation_not_per_logical_observation"
        )
    spectrum_total_sum = float(np.sum(np.clip(spectrum_total, a_min=0.0, a_max=None)))
    metadata["total_spectrum_counts"] = spectrum_total_sum
    dead_time_scales: list[float] = []
    for observation in observations:
        dead_time_scale = _metadata_float(
            observation.metadata,
            "dead_time_observed_scale",
        )
        if dead_time_scale is not None:
            dead_time_scales.append(float(dead_time_scale))
    metadata.update(
        {
            "adaptive_dwell": True,
            "adaptive_dwell_chunks": int(len(observations)),
            "adaptive_dwell_child_step_ids": [
                int(observation.step_id) for observation in observations
            ],
            "adaptive_dwell_chunk_live_times_s": [
                float(value) for value in chunk_live_times_s
            ],
            "adaptive_dwell_live_time_s": float(sum(chunk_live_times_s)),
            "adaptive_dwell_ready_reason": str(ready_reason),
            "adaptive_dwell_detected_isotopes": sorted(detected_isotopes),
            "adaptive_dwell_dead_time_observed_scales": [
                float(value) for value in dead_time_scales
            ],
            "adaptive_dwell_transport_chunk_provenance": (chunk_transport_provenance),
            "adaptive_dwell_counts_by_isotope": {
                iso: float(value) for iso, value in counts_by_isotope.items()
            },
            "adaptive_dwell_count_variance_by_isotope": {
                iso: float(value) for iso, value in count_variance_by_isotope.items()
            },
        }
    )
    if len(observations) > 1 and dead_time_scales:
        metadata["spectrum_variance_dead_time_propagation"] = (
            "independent_chunk_factored_dead_time_jacobians"
        )
    if count_covariance_by_isotope is not None:
        metadata["adaptive_dwell_count_covariance_by_isotope"] = {
            str(row_iso): {
                str(col_iso): float(value) for col_iso, value in row_payload.items()
            }
            for row_iso, row_payload in count_covariance_by_isotope.items()
        }
        metadata["count_covariance_by_isotope"] = metadata[
            "adaptive_dwell_count_covariance_by_isotope"
        ]
    if has_spectrum_variance:
        metadata["spectrum_count_variance"] = spectrum_variance_total.tolist()
        variance_total = float(np.sum(spectrum_variance_total))
        metadata["spectrum_count_variance_total"] = variance_total
        if spectrum_total_sum > 0.0 and variance_total > 0.0:
            metadata["weighted_spectrum_effective_entries"] = float(
                (spectrum_total_sum * spectrum_total_sum) / variance_total
            )
    run_time_total = _metadata_float(metadata, "run_time_s")
    if run_time_total is not None and run_time_total > 0.0:
        primaries_total = _metadata_float(metadata, "num_primaries")
        effective_entries = _metadata_float(
            metadata, "weighted_spectrum_effective_entries"
        )
        if primaries_total is not None:
            metadata["primaries_per_sec"] = float(primaries_total / run_time_total)
        if effective_entries is not None:
            metadata["effective_entries_per_sec"] = float(
                effective_entries / run_time_total
            )
    return SimulationObservation(
        step_id=int(logical_step_id),
        detector_pose_xyz=first.detector_pose_xyz,
        detector_quat_wxyz=first.detector_quat_wxyz,
        fe_orientation_index=first.fe_orientation_index,
        pb_orientation_index=first.pb_orientation_index,
        spectrum_counts=spectrum_total.tolist(),
        energy_bin_edges_keV=edge_ref.tolist(),
        metadata=metadata,
    )


def _acquire_spectrum_observation(
    *,
    simulation_runtime: SimulationRuntime,
    decomposer: SpectralDecomposer,
    step_id: int,
    pose_xyz: NDArray[np.float64],
    fe_idx: int,
    pb_idx: int,
    live_time_s: float,
    travel_time_s: float,
    shield_actuation_time_s: float,
    adaptive_dwell: bool,
    adaptive_dwell_chunk_s: float,
    adaptive_min_dwell_s: float,
    adaptive_ready_min_counts: float,
    adaptive_ready_min_isotopes: int,
    adaptive_ready_min_snr: float,
    spectrum_count_method: str,
    detect_threshold_abs: float,
    detect_threshold_rel: float,
    detect_threshold_rel_by_isotope: dict[str, float],
    min_peaks_by_isotope: dict[str, int] | None,
    adaptive_progress_every_chunks: int = 0,
    adaptive_ready_allow_informative_low: bool = True,
    adaptive_allow_low_signal_stop: bool = False,
    adaptive_low_signal_min_live_s: float = 120.0,
    adaptive_low_signal_upper_sigma: float = 3.0,
    adaptive_low_signal_count_fraction: float = 0.05,
    adaptive_low_signal_projected_live_factor: float = 4.0,
    source_cardinality_ready: bool = True,
    source_cardinality_min_live_s: float = 0.0,
    candidate_isotopes: Sequence[str] | None = None,
    travel_waypoints_xyz: Sequence[Sequence[float]] | None = None,
) -> tuple[
    SimulationObservation,
    float,
    dict[str, float],
    dict[str, float],
    set[str],
    str,
    int,
]:
    """Acquire one logical spectrum, optionally stopping adaptive dwell early."""
    target_pose = tuple(float(v) for v in pose_xyz)
    command_waypoints = (
        None
        if travel_waypoints_xyz is None
        else tuple(
            tuple(float(value) for value in waypoint)
            for waypoint in travel_waypoints_xyz
        )
    )
    if not adaptive_dwell:
        observation = simulation_runtime.step(
            SimulationCommand(
                step_id=int(step_id),
                target_pose_xyz=target_pose,
                target_base_yaw_rad=0.0,
                fe_orientation_index=int(fe_idx),
                pb_orientation_index=int(pb_idx),
                dwell_time_s=float(live_time_s),
                travel_time_s=float(travel_time_s),
                shield_actuation_time_s=float(shield_actuation_time_s),
                travel_waypoints_xyz=command_waypoints,
            )
        )
        spectrum = _analysis_spectrum_array(observation, decomposer)
        spectrum_variance = _analysis_spectrum_variance(observation, decomposer)
        transport_spectrum = _observation_spectrum_array(observation, decomposer)
        count_result = _evaluate_spectrum_count_result(
            decomposer,
            spectrum,
            live_time_s=float(live_time_s),
            spectrum_count_method=spectrum_count_method,
            detect_threshold_abs=detect_threshold_abs,
            detect_threshold_rel=detect_threshold_rel,
            detect_threshold_rel_by_isotope=detect_threshold_rel_by_isotope,
            min_peaks_by_isotope=min_peaks_by_isotope,
            spectrum_variance=spectrum_variance,
            transport_metadata=observation.metadata,
            transport_spectrum=transport_spectrum,
            candidate_isotopes=candidate_isotopes,
        )
        _store_count_covariance_metadata(
            observation.metadata,
            count_result.covariance,
        )
        return (
            observation,
            float(live_time_s),
            count_result.counts,
            count_result.variances,
            count_result.detected,
            "fixed_dwell",
            1,
        )

    observations: list[SimulationObservation] = []
    chunk_live_times_s: list[float] = []
    accumulated_spectrum = np.zeros_like(decomposer.energy_axis, dtype=float)
    accumulated_spectrum_variance = np.zeros_like(decomposer.energy_axis, dtype=float)
    has_spectrum_variance = False
    native_covariance_chunks: list[ResponsePoissonCovarianceChunk] = []
    native_covariance_mode: bool | None = None
    accumulated_live_time_s = 0.0
    last_counts: dict[str, float] = {}
    last_variances: dict[str, float] = {}
    last_covariance: dict[str, dict[str, float]] | None = None
    last_detected: set[str] = set()
    has_dwell_cap = np.isfinite(float(live_time_s)) and float(live_time_s) > 0.0
    ready_reason = "max_dwell_reached" if has_dwell_cap else "uncapped_dwell_not_ready"
    last_ready = False
    chunk_index = 0
    while True:
        if has_dwell_cap and accumulated_live_time_s + 1e-12 >= float(live_time_s):
            break
        remaining_s = (
            float(live_time_s) - accumulated_live_time_s
            if has_dwell_cap
            else float("inf")
        )
        chunk_live_time_s = min(float(adaptive_dwell_chunk_s), remaining_s)
        internal_step_id = int(step_id) * ADAPTIVE_STEP_ID_STRIDE + int(chunk_index)
        observation = simulation_runtime.step(
            SimulationCommand(
                step_id=internal_step_id,
                target_pose_xyz=target_pose,
                target_base_yaw_rad=0.0,
                fe_orientation_index=int(fe_idx),
                pb_orientation_index=int(pb_idx),
                dwell_time_s=chunk_live_time_s,
                travel_time_s=float(travel_time_s) if chunk_index == 0 else 0.0,
                shield_actuation_time_s=(
                    float(shield_actuation_time_s) if chunk_index == 0 else 0.0
                ),
                travel_waypoints_xyz=command_waypoints if chunk_index == 0 else None,
            )
        )
        spectrum = _analysis_spectrum_array(observation, decomposer)
        spectrum_variance = _analysis_spectrum_variance(observation, decomposer)
        transport_spectrum = _observation_spectrum_array(observation, decomposer)
        observations.append(observation)
        chunk_live_times_s.append(chunk_live_time_s)
        accumulated_spectrum += spectrum
        if spectrum_variance is not None:
            accumulated_spectrum_variance += spectrum_variance
            has_spectrum_variance = True
        accumulated_live_time_s += chunk_live_time_s
        evaluation_metadata = dict(observation.metadata)
        evaluation_transport_spectrum: NDArray[np.float64] | None = transport_spectrum
        native_covariance = (
            str(observation.metadata.get("spectrum_variance_semantics", ""))
            == "compound_poisson_sumw2_includes_counting"
            and str(
                observation.metadata.get(
                    "spectrum_variance_dead_time_propagation",
                    "",
                )
            )
            == "fixed_observed_scale"
        )
        if native_covariance_mode is None:
            native_covariance_mode = bool(native_covariance)
        elif native_covariance_mode != bool(native_covariance):
            raise ValueError(
                "Adaptive dwell cannot mix native and approximate covariance chunks."
            )
        if native_covariance:
            if spectrum_variance is None:
                raise ValueError(
                    "Native adaptive dwell chunk is missing spectrum variance."
                )
            native_covariance_chunks.append(
                ResponsePoissonCovarianceChunk(
                    analysis_spectrum=np.asarray(spectrum, dtype=float),
                    analysis_variance=np.asarray(spectrum_variance, dtype=float),
                    transport_spectrum=np.asarray(transport_spectrum, dtype=float),
                    transport_metadata=dict(observation.metadata),
                    live_time_s=float(chunk_live_time_s),
                )
            )
        evaluation_covariance_chunks: tuple[ResponsePoissonCovarianceChunk, ...] = ()
        if len(native_covariance_chunks) == len(observations) > 1:
            evaluation_covariance_chunks = tuple(native_covariance_chunks)
            evaluation_transport_spectrum = None
        elif len(observations) > 1:
            evaluation_metadata["spectrum_variance_dead_time_propagation"] = (
                "independent_chunk_sum_post_transform"
            )
            evaluation_transport_spectrum = None
        count_result = _evaluate_spectrum_count_result(
            decomposer,
            accumulated_spectrum,
            live_time_s=accumulated_live_time_s,
            spectrum_count_method=spectrum_count_method,
            detect_threshold_abs=detect_threshold_abs,
            detect_threshold_rel=detect_threshold_rel,
            detect_threshold_rel_by_isotope=detect_threshold_rel_by_isotope,
            min_peaks_by_isotope=min_peaks_by_isotope,
            spectrum_variance=(
                accumulated_spectrum_variance if has_spectrum_variance else None
            ),
            transport_metadata=evaluation_metadata,
            transport_spectrum=evaluation_transport_spectrum,
            transport_covariance_chunks=evaluation_covariance_chunks,
            candidate_isotopes=candidate_isotopes,
        )
        last_counts = count_result.counts
        last_variances = count_result.variances
        last_covariance = count_result.covariance
        last_detected = count_result.detected
        ready, reason = _is_adaptive_spectrum_ready(
            last_counts,
            last_variances,
            live_time_s=accumulated_live_time_s,
            min_live_time_s=float(adaptive_min_dwell_s),
            min_counts_per_detected_isotope=float(adaptive_ready_min_counts),
            min_detected_isotopes=int(adaptive_ready_min_isotopes),
            candidate_isotopes=(
                list(candidate_isotopes)
                if candidate_isotopes is not None
                else list(decomposer.isotope_names)
            ),
            min_snr=float(adaptive_ready_min_snr),
            total_spectrum_counts=float(
                np.sum(np.clip(accumulated_spectrum, 0.0, None))
            ),
            allow_informative_low=bool(adaptive_ready_allow_informative_low),
            allow_low_signal_stop=bool(adaptive_allow_low_signal_stop),
            low_signal_min_live_s=float(adaptive_low_signal_min_live_s),
            low_signal_upper_sigma=float(adaptive_low_signal_upper_sigma),
            low_signal_count_fraction=float(adaptive_low_signal_count_fraction),
            low_signal_projected_live_factor=float(
                adaptive_low_signal_projected_live_factor
            ),
        )
        last_ready = ready
        ready_reason = reason
        if (
            ready
            and not bool(source_cardinality_ready)
            and accumulated_live_time_s + 1.0e-12
            < max(float(source_cardinality_min_live_s), float(adaptive_min_dwell_s))
        ):
            last_ready = False
            ready = False
            ready_reason = f"source_cardinality_pending:{reason}"
        progress_every = int(adaptive_progress_every_chunks)
        should_log_progress = progress_every > 0 and (
            chunk_index == 0 or ready or (chunk_index + 1) % progress_every == 0
        )
        if should_log_progress:
            print(
                _adaptive_dwell_progress_message(
                    step_id=step_id,
                    chunk_index=chunk_index,
                    live_time_s=accumulated_live_time_s,
                    counts_by_isotope=last_counts,
                    count_variance_by_isotope=last_variances,
                    reason=reason,
                ),
                flush=True,
            )
        if ready:
            break
        chunk_index += 1
    reached_dwell_cap = has_dwell_cap and accumulated_live_time_s + 1e-12 >= live_time_s
    if reached_dwell_cap and not last_ready:
        ready_reason = f"max_dwell_reached:{ready_reason}"
    last_variances = _inflate_low_signal_variances(
        last_counts,
        last_variances,
        min_counts_per_detected_isotope=float(adaptive_ready_min_counts),
        ready_reason=ready_reason,
    )
    observation = _merge_adaptive_observation_chunks(
        logical_step_id=int(step_id),
        observations=observations,
        chunk_live_times_s=chunk_live_times_s,
        ready_reason=ready_reason,
        counts_by_isotope=last_counts,
        count_variance_by_isotope=last_variances,
        count_covariance_by_isotope=last_covariance,
        detected_isotopes=last_detected,
    )
    return (
        observation,
        float(accumulated_live_time_s),
        last_counts,
        last_variances,
        last_detected,
        ready_reason,
        len(observations),
    )


def run_live_pf(
    live: bool = True,
    max_steps: int | None = None,
    max_poses: int | None = 8,
    sources: list[PointSource] | None = None,
    detect_threshold_abs: float = 50.0,
    detect_threshold_rel: float = 0.3,
    detect_consecutive: int = 10,
    detect_min_steps: int | None = None,
    min_peaks_by_isotope: dict[str, int] | None = None,
    ig_threshold_mode: str = "relative_pose",
    ig_threshold_rel: float = 0.02,
    ig_threshold_min: float | None = None,
    environment_mode: str = DEFAULT_ENVIRONMENT_MODE,
    obstacle_layout_path: str | None = DEFAULT_OBSTACLE_CONFIG.as_posix(),
    obstacle_seed: int | None = None,
    eval_match_radius_m: float = 0.5,
    candidate_grid_spacing: tuple[float, float, float] | None = None,
    candidate_grid_margin: float = CANDIDATE_GRID_MARGIN,
    birth_enabled: bool = False,
    num_particles: int = 2000,
    pf_config_overrides: dict[str, object] | None = None,
    save_outputs: bool = True,
    output_tag: str | None = None,
    measurement_log_output: str | None = None,
    resume_measurement_stage: str | None = None,
    resume_runtime_log: str | None = None,
    resume_compatible_code_paths: Sequence[str] | None = None,
    resume_compatibility_basis: str | None = None,
    pose_candidates: int = 64,
    pose_min_dist: float = 3.0,
    return_state: bool = False,
    converge: bool = False,
    sim_backend: str = "analytic",
    sim_config_path: str | None = None,
    blender_executable: str | None = None,
    blender_output_path: str | None = None,
    blender_timeout_s: float = 120.0,
    passage_width_m: float = 1.0,
    robot_radius_m: float = 0.35,
    nominal_motion_speed_m_s: float = DEFAULT_ROBOT_SPEED_M_S,
    rotation_overhead_s: float = DEFAULT_ROTATION_OVERHEAD_S,
    measurement_time_s: float = DEFAULT_MEASUREMENT_TIME_S,
    adaptive_dwell: bool = False,
    adaptive_dwell_chunk_s: float = 2.0,
    adaptive_min_dwell_s: float = 2.0,
    adaptive_ready_min_counts: float = 100.0,
    adaptive_ready_min_isotopes: int = 1,
    adaptive_ready_min_snr: float = 0.0,
    pose_min_observation_counts: float | None = None,
    pose_min_observation_penalty_scale: float = 1.0,
    pose_min_observation_aggregate: str = "max",
    path_planner: str | None = None,
    dss_horizon: int | None = None,
    dss_beam_width: int | None = None,
    dss_program_length: int | None = None,
    dss_signature_weight: float | None = None,
    dss_differential_weight: float | None = None,
    dss_rotation_weight: float | None = None,
    source_generation_mode: str = "demo",
    random_source_seed: int | None = None,
    random_source_count: int = DEFAULT_RANDOM_SOURCE_COUNT,
    random_source_isotopes: str | Sequence[str] | None = None,
    random_source_intensity_cps_1m: float = DEFAULT_RANDOM_SOURCE_INTENSITY_CPS_1M,
    random_source_intensity_min_cps_1m: float | None = None,
    random_source_intensity_max_cps_1m: float | None = None,
    notification_config: PiplupNotificationConfig | None = None,
    notify_spectrum: bool = False,
    notify_spectrum_every: int = 1,
    notify_spectrum_max_bins: int = 800,
) -> RotatingShieldPFEstimator | None:
    """
    Run a simple PF loop with live visualization (active pose/orientation selection).

    If max_steps is None, run until the information-gain threshold is met.
    If max_poses is None, run without a pose-count limit.
    If obstacle_layout_path is provided, blocked grid cells are excluded and shown
    in black.

    Args:
        pf_config_overrides: Optional overrides applied to the PF configuration.
        save_outputs: When False, skip writing plots and snapshot images.
        output_tag: Optional tag appended to result output filenames.
        measurement_log_output: Truth-free log directory. Pure runs require this
            argument or runtime_config.measurement_log_output_dir.
        resume_measurement_stage: Hidden stream stage to adopt at a completed
            station boundary before pure-PF replay.
        resume_runtime_log: Original live stdout log used only to restore
            estimator-neutral remaining-budget controller history.
        resume_compatible_code_paths: Runtime paths whose commit delta passed an
            external state-equivalence gate.
        resume_compatibility_basis: Description of the equivalence evidence for
            explicitly compatible runtime paths.
        pose_candidates: Number of pose candidates to generate per step.
        pose_min_dist: Minimum distance from visited poses for candidates (meters).
        return_state: When True, return the estimator for inspection/testing.
        candidate_grid_spacing: Optional (x, y, z) spacing for birth candidate grid.
        candidate_grid_margin: Margin from the environment bounds for candidate sources.
        birth_enabled: Enable birth/death/split/merge moves.
        num_particles: Particle count used by each isotope filter.
        converge: Enable per-isotope convergence gating.
        environment_mode: Obstacle environment mode ("fixed" or "random").
        sim_backend: Simulation backend name ("analytic", "isaacsim", or "geant4").
        sim_config_path: Optional JSON config for the selected simulation backend.
        blender_executable: Optional Blender executable path for random mode.
        blender_output_path: Optional USD path written by Blender in random mode.
        blender_timeout_s: Timeout for Blender environment generation.
        passage_width_m: Minimum reserved corridor width in random mode.
        robot_radius_m: Robot footprint radius used for 2D traversability maps.
        nominal_motion_speed_m_s: Nominal robot speed used for mission-time estimates.
        rotation_overhead_s: Fixed shield-actuation overhead per measurement.
        measurement_time_s: Fixed dwell time or adaptive maximum dwell time.
            Values <= 0 remove the adaptive dwell cap.
        adaptive_dwell: Stop each measurement once isotope counts are reliable enough.
        adaptive_dwell_chunk_s: Geant4 dwell duration for each adaptive chunk.
        adaptive_min_dwell_s: Minimum accumulated dwell before early stopping.
        adaptive_ready_min_counts: Minimum count estimate per detected isotope.
        adaptive_ready_min_isotopes: Required number of detected isotopes for readiness.
        adaptive_ready_min_snr: Optional minimum count-estimate SNR for dwell readiness.
        pose_min_observation_counts: Minimum posterior-predicted counts per isotope
            used as a soft pose-selection constraint; None uses runtime config
            or the configured observation-SNR floor.
        pose_min_observation_penalty_scale: Relative weight of the pose
            observability soft constraint.
        pose_min_observation_aggregate: Orientation aggregation for pose
            observability ("max" or "mean").
        path_planner: Pose planner name. Use "one_step" for the original
            selector or "dss_pp" for joint pose-shield planning.
        dss_horizon: DSS-PP receding-horizon length.
        dss_beam_width: DSS-PP beam width.
        dss_program_length: Number of shield postures in each DSS program.
        dss_signature_weight: DSS shield-signature separation weight.
        dss_differential_weight: DSS differential-observability penalty weight.
        dss_rotation_weight: DSS shield-transition penalty weight.
        source_generation_mode: Source layout mode ("demo" or "surface_random").
        random_source_seed: RNG seed for surface-random source generation.
        random_source_count: Number of surface-random sources to generate.
        random_source_isotopes: Optional isotope list for surface-random sources.
        random_source_intensity_cps_1m: Detector-cps@1m strength for random sources.
        random_source_intensity_min_cps_1m: Optional minimum random source strength.
        random_source_intensity_max_cps_1m: Optional maximum random source strength.
        notification_config: Optional piplup-notify delivery settings.
        notify_spectrum: Send per-measurement spectrum events through piplup.
        notify_spectrum_every: Send one spectrum event every N measurements.
        notify_spectrum_max_bins: Maximum number of spectrum bins per event.
    """
    _validate_measurement_timing(
        measurement_time_s=float(measurement_time_s),
        adaptive_dwell=bool(adaptive_dwell),
        adaptive_dwell_chunk_s=float(adaptive_dwell_chunk_s),
        adaptive_min_dwell_s=float(adaptive_min_dwell_s),
        adaptive_ready_min_counts=float(adaptive_ready_min_counts),
        adaptive_ready_min_isotopes=int(adaptive_ready_min_isotopes),
        adaptive_ready_min_snr=float(adaptive_ready_min_snr),
    )
    notifier = PiplupNotifier(notification_config)
    live = _coerce_live_visualization(live)
    if sim_config_path is None:
        input_config_hash = sha256_json({})
    else:
        input_config_hash = hashlib.sha256(
            Path(sim_config_path).expanduser().read_bytes()
        ).hexdigest()
    runtime_config = enforce_pure_runtime_settings(load_runtime_config(sim_config_path))
    _validate_adaptive_primary_budget_contract(
        runtime_config,
        adaptive_dwell=bool(adaptive_dwell),
    )
    joint_observation_update, delayed_resample_update = _resolve_station_update_modes(
        runtime_config
    )
    adaptive_ready_allow_informative_low = bool(
        runtime_config.get("adaptive_ready_allow_informative_low", False)
    )
    adaptive_allow_low_signal_stop = bool(
        runtime_config.get("adaptive_allow_low_signal_stop", False)
    )
    adaptive_low_signal_min_live_s = max(
        0.0,
        float(runtime_config.get("adaptive_low_signal_min_live_s", 120.0)),
    )
    adaptive_low_signal_upper_sigma = max(
        0.0,
        float(runtime_config.get("adaptive_low_signal_upper_sigma", 3.0)),
    )
    adaptive_low_signal_count_fraction = max(
        0.0,
        float(runtime_config.get("adaptive_low_signal_count_fraction", 0.05)),
    )
    adaptive_low_signal_projected_live_factor = max(
        1.0,
        float(runtime_config.get("adaptive_low_signal_projected_live_factor", 4.0)),
    )
    adaptive_cardinality_dwell_enable = bool(
        runtime_config.get("adaptive_cardinality_dwell_enable", True)
    )
    adaptive_cardinality_min_live_s = max(
        float(adaptive_min_dwell_s),
        float(
            runtime_config.get("adaptive_cardinality_min_live_s", adaptive_min_dwell_s)
        ),
    )
    effective_robot_radius_m = _resolve_measurement_clearance_radius_m(
        runtime_config,
        requested_robot_radius_m=float(robot_radius_m),
    )
    environment_size_z_m = 10.0
    detector_height_config = _resolve_detector_height_planning_config(
        runtime_config,
        room_height_m=environment_size_z_m,
    )
    robot_ground_z_m = float(detector_height_config.ground_z_m)
    initial_detector_world_z_m = float(detector_height_config.initial_world_z_m)
    detector_height_candidates = detector_height_config.candidate_world_heights_m
    detector_height_min_world_z_m, detector_height_max_world_z_m = (
        detector_height_config.candidate_world_z_bounds_m
    )
    detector_pose_consistency_tolerance_m = max(
        0.0,
        float(runtime_config.get("detector_pose_consistency_tolerance_m", 1.0e-4)),
    )
    height_partner_reuse_shield_program = bool(
        runtime_config.get("height_partner_reuse_shield_program", False)
    )
    detector_height_pair_xy_tolerance_m = max(
        0.0,
        float(runtime_config.get("detector_height_pair_xy_tolerance_m", 1.0e-6)),
    )
    detector_height_pair_z_tolerance_m = max(
        0.0,
        float(runtime_config.get("detector_height_pair_z_tolerance_m", 1.0e-9)),
    )
    detector_height_pair_min_separation_m = max(
        0.0,
        float(runtime_config.get("detector_height_pair_min_separation_m", 0.0)),
    )
    detector_continuous_height_partner_candidates = (
        max(
            0,
            int(
                runtime_config.get(
                    "detector_continuous_height_partner_candidates",
                    8,
                )
            ),
        )
        if detector_height_config.mode == "continuous"
        else 0
    )
    continuous_height_bounds_for_dss = (
        (
            detector_height_min_world_z_m,
            detector_height_max_world_z_m,
        )
        if detector_height_config.mode == "continuous"
        else None
    )
    env = EnvironmentConfig(
        size_x=10.0,
        size_y=20.0,
        size_z=environment_size_z_m,
        detector_position=(1.0, 1.0, initial_detector_world_z_m),
    )
    if detector_height_config.mode == "continuous":
        print(
            "Detector height workspace: "
            f"mode=continuous ground_z={robot_ground_z_m:.3f}m "
            "mast_range="
            f"[{detector_height_config.minimum_mast_height_m:.3f}, "
            f"{detector_height_config.maximum_mast_height_m:.3f}]m "
            "world_z_range="
            f"[{detector_height_min_world_z_m:.3f}, "
            f"{detector_height_max_world_z_m:.3f}]m"
        )
    else:
        print(
            "Detector height workspace: "
            f"mode=discrete ground_z={robot_ground_z_m:.3f}m "
            f"mast={list(detector_height_config.discrete_mast_actions_m)} "
            f"world_z={list(detector_height_config.discrete_world_actions_m)}"
        )
    normalized_source_generation_mode = source_generation_mode.strip().lower()
    if normalized_source_generation_mode not in {"demo", "surface_random"}:
        raise ValueError("source_generation_mode must be 'demo' or 'surface_random'.")
    spectrum_config = _spectrum_config_from_runtime_config(runtime_config)
    decomposer = SpectralDecomposer(spectrum_config=spectrum_config)
    diagnostic_decomposer = SpectralDecomposer(
        spectrum_config=spectrum_config,
        library=decomposer.library,
    )
    default_count_method = "response_poisson"
    spectrum_count_method = (
        str(runtime_config.get("spectrum_count_method", default_count_method))
        .strip()
        .lower()
    )
    RuntimeCountExtractor.validate_count_method(spectrum_count_method)
    if spectrum_count_method != "response_poisson":
        raise ValueError(
            "Pure PF live runs require spectrum_count_method='response_poisson'."
        )
    measurement_log_target = _resolve_required_measurement_log_target(
        measurement_log_output,
        runtime_config,
        repository_root=ROOT,
    )
    if min_peaks_by_isotope is None:
        min_peaks_by_isotope = dict(DETECT_MIN_PEAKS_BY_ISOTOPE)
    detect_threshold_rel_by_isotope = dict(DETECT_REL_THRESH_BY_ISOTOPE)
    obstacle_environment = build_runtime_obstacle_environment(
        root=ROOT,
        environment_mode=environment_mode,
        obstacle_layout_path=obstacle_layout_path,
        room_size_xyz=(env.size_x, env.size_y, env.size_z),
        detector_position_xy=env.detector_position,
        obstacle_seed=obstacle_seed,
        blocked_fraction=0.4,
        passage_width_m=passage_width_m,
        attach_known_transport=True,
        obstacle_height_m=float(runtime_config.get("obstacle_height_m", 2.0)),
        include_room_boundaries=bool(
            runtime_config.get("author_room_boundary_prims", False)
        ),
        room_boundary_thickness_m=float(
            runtime_config.get("room_boundary_thickness_m", 0.1)
        ),
    )
    obstacle_grid = obstacle_environment.grid
    normalized_environment_mode = obstacle_environment.mode
    known_obstacle_instances = obstacle_environment.known_obstacle_instances
    measurement_log_obstacle_layout_path = _measurement_log_obstacle_layout_path(
        obstacle_environment,
        repository_root=ROOT,
    )
    runtime_obstacle_material = str(runtime_config.get("obstacle_material", "concrete"))
    if obstacle_environment.message is not None:
        print(obstacle_environment.message)
    obstacle_asset_summary = obstacle_environment.asset_summary()
    if obstacle_asset_summary is not None:
        print(obstacle_asset_summary)
    random_source_visibility_points: NDArray[np.float64] | None = None
    random_source_visibility_diagnostics: dict[str, object] = {
        "enabled": False,
        "reference_point_count": 0,
    }
    if normalized_source_generation_mode == "surface_random":
        source_rng_seed = (
            obstacle_seed if random_source_seed is None else random_source_seed
        )
        source_rng = np.random.default_rng(source_rng_seed)
        source_isotopes = _resolve_random_source_isotopes(
            random_source_isotopes,
            runtime_config,
            tuple(decomposer.isotope_names),
        )
        source_visibility_enabled = bool(
            runtime_config.get("random_source_visibility_filter", True)
        )
        source_visibility_min_fraction = max(
            0.0,
            float(runtime_config.get("random_source_min_visible_fraction", 0.10)),
        )
        source_visibility_clear_path_max_m = max(
            0.0,
            float(runtime_config.get("random_source_clear_path_max_m", 0.01)),
        )
        source_visibility_batch_size = max(
            1,
            int(runtime_config.get("random_source_visibility_batch_size", 256)),
        )
        source_visibility_max_attempts = max(
            1,
            int(
                runtime_config.get(
                    "random_source_visibility_max_attempts_per_source",
                    4096,
                )
            ),
        )
        source_visibility_detector_height_m = float(env.detector_position[2])
        source_response_filter_enabled = bool(
            runtime_config.get("random_source_response_observability_filter", False)
        )
        source_response_max_corr = float(
            runtime_config.get("random_source_response_max_pairwise_corr", 0.995)
        )
        source_response_max_condition = float(
            runtime_config.get("random_source_response_condition_max", 1.0e3)
        )
        source_response_max_sets = max(
            1,
            int(runtime_config.get("random_source_response_max_set_attempts", 16)),
        )
        max_ceiling_payload = runtime_config.get(
            "random_source_max_ceiling_sources",
            1,
        )
        random_source_max_ceiling_sources = (
            None if max_ceiling_payload is None else max(0, int(max_ceiling_payload))
        )
        preferred_z_payload = runtime_config.get(
            "random_source_preferred_max_z_m",
            5.0,
        )
        random_source_preferred_max_z_m = (
            None if preferred_z_payload is None else float(preferred_z_payload)
        )
        random_source_same_isotope_min_distance_m = max(
            0.0,
            float(runtime_config.get("random_source_same_isotope_min_distance_m", 2.0)),
        )
        intensity_min_payload = runtime_config.get(
            "random_source_intensity_min_cps_1m",
            random_source_intensity_min_cps_1m,
        )
        intensity_max_payload = runtime_config.get(
            "random_source_intensity_max_cps_1m",
            random_source_intensity_max_cps_1m,
        )
        random_source_intensity_spec: float | tuple[float, float]
        if intensity_min_payload is not None or intensity_max_payload is not None:
            if intensity_min_payload is None or intensity_max_payload is None:
                raise ValueError(
                    "random source intensity min/max must be provided together."
                )
            random_source_intensity_spec = (
                float(intensity_min_payload),
                float(intensity_max_payload),
            )
        else:
            random_source_intensity_spec = float(random_source_intensity_cps_1m)
        print(
            "Random source placement constraints: "
            f"max_ceiling_sources={random_source_max_ceiling_sources} "
            f"preferred_max_z_m={random_source_preferred_max_z_m} "
            "same_isotope_min_distance_m="
            f"{random_source_same_isotope_min_distance_m:.3f}"
        )
        if (
            source_visibility_enabled
            and source_visibility_min_fraction > 0.0
            and obstacle_grid is not None
            and obstacle_grid.blocked_cells
        ):
            random_source_visibility_points = _ground_visibility_reference_points(
                env,
                obstacle_grid,
                robot_radius_m=float(robot_radius_m),
                detector_height_m=source_visibility_detector_height_m,
            )
            print(
                "Random source ground-visibility filter: "
                f"min_visible_fraction={source_visibility_min_fraction:.3f}, "
                f"reference_points={random_source_visibility_points.shape[0]}, "
                f"clear_path_max_m={source_visibility_clear_path_max_m:.3f}"
            )
        source_set_attempts = (
            source_response_max_sets if source_response_filter_enabled else 1
        )
        used_source_set_attempts = 0
        last_sources: list[PointSource] | None = None
        last_visibility_diagnostics: dict[str, object] | None = None
        for source_set_attempt in range(source_set_attempts):
            used_source_set_attempts = source_set_attempt + 1
            trial_sources = generate_surface_sources(
                env=env,
                obstacle_grid=obstacle_grid,
                isotopes=source_isotopes,
                intensity_cps_1m=random_source_intensity_spec,
                rng=source_rng,
                count=max(1, int(random_source_count)),
                obstacle_height_m=float(runtime_config.get("obstacle_height_m", 2.0)),
                visibility_measurement_points=random_source_visibility_points,
                visibility_min_fraction=(
                    source_visibility_min_fraction if source_visibility_enabled else 0.0
                ),
                visibility_detector_height_m=source_visibility_detector_height_m,
                visibility_clear_path_max_m=source_visibility_clear_path_max_m,
                visibility_batch_size=source_visibility_batch_size,
                visibility_max_attempts_per_source=source_visibility_max_attempts,
                max_ceiling_sources=random_source_max_ceiling_sources,
                preferred_max_z_m=random_source_preferred_max_z_m,
                same_isotope_min_distance_m=random_source_same_isotope_min_distance_m,
            )
            trial_visibility = _source_ground_visibility_payload(
                trial_sources,
                obstacle_grid,
                random_source_visibility_points,
                obstacle_height_m=float(runtime_config.get("obstacle_height_m", 2.0)),
                detector_height_m=source_visibility_detector_height_m,
                clear_path_max_m=source_visibility_clear_path_max_m,
                min_visible_fraction=(
                    source_visibility_min_fraction if source_visibility_enabled else 0.0
                ),
            )
            last_sources = trial_sources
            last_visibility_diagnostics = trial_visibility
            if not source_response_filter_enabled:
                break
            response_diag = trial_visibility.get("response_observability", {})
            if not isinstance(response_diag, Mapping):
                break
            max_corr = float(response_diag.get("max_pairwise_correlation", 0.0))
            same_pair_count = int(response_diag.get("same_isotope_pair_count", 0))
            same_max_corr = float(
                response_diag.get("same_isotope_max_pairwise_correlation", 0.0)
            )
            screening_corr = same_max_corr if same_pair_count > 0 else max_corr
            condition = float(response_diag.get("condition_number", 1.0))
            if (
                screening_corr <= source_response_max_corr
                and condition <= source_response_max_condition
            ):
                break
            if source_set_attempt + 1 == source_set_attempts:
                print(
                    "Random source response-observability filter reached "
                    f"attempt limit={source_set_attempts}; using last set "
                    f"(screening_corr={screening_corr:.3f}, "
                    f"max_corr={max_corr:.3f}, same_iso_corr={same_max_corr:.3f}, "
                    f"condition={condition:.3g})."
                )
        sources = last_sources or []
        random_source_visibility_diagnostics = last_visibility_diagnostics or {
            "enabled": False,
            "reference_point_count": 0,
        }
        random_source_visibility_diagnostics["response_filter_enabled"] = bool(
            source_response_filter_enabled
        )
        random_source_visibility_diagnostics["response_filter_max_pairwise_corr"] = (
            float(source_response_max_corr)
        )
        random_source_visibility_diagnostics["response_filter_condition_max"] = float(
            source_response_max_condition
        )
        random_source_visibility_diagnostics["response_filter_set_attempts"] = int(
            used_source_set_attempts
        )
        print(
            "Generated surface-constrained random sources: "
            f"count={len(sources)}, seed={source_rng_seed}, "
            f"isotopes={list(source_isotopes)}, "
            "intensity_cps_1m="
            f"{_format_random_source_intensity_spec(random_source_intensity_spec)}"
        )
        if random_source_visibility_diagnostics.get("enabled"):
            print(
                "Random source ground visibility: "
                "min_fraction="
                f"{random_source_visibility_diagnostics['min_source_visible_fraction']:.3f}, "
                "mean_fraction="
                f"{random_source_visibility_diagnostics['mean_source_visible_fraction']:.3f}"
            )
    elif sources is None:
        sources = _build_demo_sources()
    normals = generate_octant_orientations()
    rot_mats = generate_octant_rotation_matrices()
    num_orients = len(rot_mats)
    if save_outputs:
        PF_DIR.mkdir(parents=True, exist_ok=True)
    output_suffix = ""
    cleaned_tag = ""
    if output_tag:
        cleaned_tag = output_tag.strip().replace(" ", "_")
        cleaned_tag = cleaned_tag.replace("/", "_").replace("\\", "_")
        cleaned_tag = cleaned_tag.lstrip("_")
        if cleaned_tag:
            output_suffix = f"_{cleaned_tag}"
    estimate_trace_enabled = bool(
        runtime_config.get("intermediate_estimate_trace", True)
    )
    estimate_trace_log_enabled = bool(
        runtime_config.get("intermediate_estimate_trace_log", True)
    )
    estimate_trace_log_every = max(
        1,
        int(runtime_config.get("intermediate_estimate_trace_log_every", 1)),
    )
    estimate_trace_max_log_records = max(
        0,
        int(runtime_config.get("intermediate_estimate_trace_max_log_records", 6)),
    )
    precision_diagnostic_birth_candidate_log_limit = int(
        runtime_config.get("precision_diagnostic_birth_candidate_log_limit", 0)
    )
    precision_diagnostic_birth_candidate_enable = bool(
        runtime_config.get("precision_diagnostic_birth_candidate_enable", False)
    )
    precision_diagnostic_full_spectrum_response_enable = bool(
        runtime_config.get("precision_diagnostic_full_spectrum_response_enable", False)
    )
    precision_diagnostic_particle_log_limit = int(
        runtime_config.get("precision_diagnostic_particle_log_limit", 0)
    )
    precision_diagnostic_source_event_log_limit = int(
        runtime_config.get("precision_diagnostic_source_event_log_limit", 64)
    )
    surface_observability_diagnostic_candidates = max(
        0,
        int(runtime_config.get("surface_observability_diagnostic_candidates", 0)),
    )
    estimate_trace_out_path: Path | None = None
    if save_outputs and estimate_trace_enabled:
        estimate_trace_out_path = (
            RESULTS_DIR
            / "estimate_traces"
            / f"intermediate_estimates{output_suffix}.jsonl"
        )
        estimate_trace_out_path.parent.mkdir(parents=True, exist_ok=True)
        estimate_trace_out_path.write_text("", encoding="utf-8")
    cui_split_view_enabled = _resolve_cui_split_view_enabled(
        runtime_config,
        save_outputs=save_outputs,
    )
    cui_split_view_dir_raw = runtime_config.get(
        "cui_split_view_dir",
        DEFAULT_CUI_SPLIT_VIEW_DIR,
    )
    cui_split_view_dir = Path(str(cui_split_view_dir_raw)).expanduser()
    if not cui_split_view_dir.is_absolute():
        cui_split_view_dir = ROOT / cui_split_view_dir
    cui_split_max_particles_raw = runtime_config.get(
        "cui_split_view_max_particles_per_isotope",
        None,
    )
    cui_split_max_particles = (
        None
        if cui_split_max_particles_raw is None
        else int(cui_split_max_particles_raw)
    )
    spectrum_plot_save_every = _resolve_plot_save_interval(
        runtime_config,
        "spectrum_plot_save_every",
        default=1,
    )
    pf_plot_save_every = _resolve_plot_save_interval(
        runtime_config,
        "pf_plot_save_every",
        default=1,
        allow_disable=True,
    )
    generated_blender_usd_path: Path | None = None
    traversability_map: TraversabilityMap | None = None
    traversability_map_path: Path | None = None
    traversability_map_png_path: Path | None = None
    if obstacle_grid is not None and normalized_environment_mode == "random":
        if blender_output_path:
            generated_output_path = Path(blender_output_path)
            if not generated_output_path.is_absolute():
                generated_output_path = (ROOT / generated_output_path).resolve()
        else:
            if obstacle_seed is None:
                path_token = f"random_{int(time.time() * 1000)}"
            else:
                path_token = f"random_seed_{int(obstacle_seed)}"
            if cleaned_tag:
                path_token = f"{path_token}_{cleaned_tag}"
            generated_output_path = BLENDER_ENV_DIR / f"{path_token}.usda"
        base_usd_path = _resolve_config_relative_path(
            runtime_config.get(
                "random_environment_base_usd_path",
                runtime_config.get("usd_path"),
            ),
            sim_config_path,
        )
        traversability_map_path = generated_output_path.with_suffix(
            ".traversability.json"
        )
        traversability_map_png_path = generated_output_path.with_suffix(
            ".traversability.png"
        )
        generated_blender_usd_path = generate_blender_environment_usd(
            grid=obstacle_grid,
            output_path=generated_output_path,
            room_size_xyz=(env.size_x, env.size_y, env.size_z),
            obstacle_height_m=float(runtime_config.get("obstacle_height_m", 2.0)),
            obstacle_material=runtime_obstacle_material,
            obstacle_instances=known_obstacle_instances,
            obstacle_asset_seed=obstacle_seed,
            base_usd_path=base_usd_path,
            traversability_output_path=traversability_map_path,
            robot_radius_m=float(effective_robot_radius_m),
            traversability_reachable_from_xy=(
                float(env.detector_position[0]),
                float(env.detector_position[1]),
            ),
            blender_executable=blender_executable,
            timeout_s=blender_timeout_s,
        )
        print(f"Generated Blender random environment: {generated_blender_usd_path}")
        if traversability_map_path.exists():
            traversability_map = TraversabilityMap.load(traversability_map_path)
        if traversability_map is None or float(
            traversability_map.robot_radius_m
        ) + 1.0e-9 < float(effective_robot_radius_m):
            traversability_map = build_traversability_map_from_obstacle_grid(
                obstacle_grid,
                robot_radius_m=float(effective_robot_radius_m),
                reachable_from=env.detector_position,
            )
            traversability_map.save(traversability_map_path)
        render_traversability_map(traversability_map, traversability_map_png_path)
        print(
            "Generated 2D robot traversability map: "
            f"{traversability_map_path} "
            f"(free_fraction={traversability_map.traversable_fraction:.3f}, "
            f"robot_radius_m={float(effective_robot_radius_m):.2f})"
        )
    elif obstacle_grid is not None and obstacle_grid.blocked_cells:
        traversability_map = build_traversability_map_from_obstacle_grid(
            obstacle_grid,
            robot_radius_m=float(effective_robot_radius_m),
            reachable_from=env.detector_position,
        )
    planning_map = (
        traversability_map if traversability_map is not None else obstacle_grid
    )
    pf_obstacle_attenuation_enabled = _pf_obstacle_attenuation_enabled(runtime_config)
    pf_obstacle_grid = _pf_obstacle_grid_for_runtime(obstacle_grid, runtime_config)

    # Candidate sources: known environment surfaces used by birth proposals.
    spacing = candidate_grid_spacing or CANDIDATE_GRID_SPACING
    source_surface_prior = _source_surface_prior_enabled(runtime_config)
    if source_surface_prior:
        _validate_surface_constrained_sources(
            sources,
            env,
            obstacle_grid,
            obstacle_height_m=float(runtime_config.get("obstacle_height_m", 2.0)),
            tolerance_m=max(
                0.0,
                float(runtime_config.get("posterior_surface_tolerance_m", 1.0e-5)),
            ),
        )
    source_position_min, source_position_max = _resolve_source_position_bounds(
        env,
        runtime_config,
    )
    grid = _build_source_candidate_grid(
        env,
        obstacle_grid,
        spacing=spacing,
        margin=float(candidate_grid_margin),
        position_min=source_position_min,
        position_max=source_position_max,
        source_surface_prior=source_surface_prior,
        obstacle_height_m=float(runtime_config.get("obstacle_height_m", 2.0)),
    )

    bounds_lo = np.array(
        [0.0, 0.0, detector_height_min_world_z_m],
        dtype=float,
    )
    bounds_hi = np.array(
        [env.size_x, env.size_y, detector_height_max_world_z_m],
        dtype=float,
    )

    isotopes = list(
        _resolve_candidate_isotopes(runtime_config, decomposer.isotope_names)
    )
    measurement_log_runtime_config = _truth_free_live_runtime_config(runtime_config)
    measurement_log_runtime_config.update(
        {
            "sim_backend": str(sim_backend),
            "spectrum_count_method": str(spectrum_count_method),
            "candidate_isotopes": [str(value) for value in isotopes],
            "source_rate_model": "detector_cps_1m",
            "environment_mode": str(normalized_environment_mode),
            "joint_observation_update": bool(joint_observation_update),
            "delayed_resample_update": bool(delayed_resample_update),
        }
    )
    pf_random_seed = int(
        runtime_config.get(
            "pf_random_seed",
            runtime_config.get("random_seed", runtime_config.get("rng_seed", 0)),
        )
    )
    _seed_pf_random_generators(pf_random_seed)
    planning_candidate_seed = int(
        runtime_config.get("planning_candidate_seed", pf_random_seed + 1000003)
    )
    planning_candidate_rng = np.random.default_rng(planning_candidate_seed)
    measurement_log_runtime_config["planning_candidate_seed"] = int(
        planning_candidate_seed
    )
    print(
        "PF candidate isotopes: "
        f"{isotopes} (spectrum_library={list(decomposer.isotope_names)})"
    )
    detect_min_steps = (
        detect_consecutive if detect_min_steps is None else detect_min_steps
    )
    detect_counts = {iso: 0 for iso in isotopes}
    miss_counts = {iso: 0 for iso in isotopes}
    detected_isotopes: set[str] = set()
    active_isotopes: set[str] = set()
    last_candidates: set[str] = set()
    pf_detected_isotopes_only = bool(
        runtime_config.get("pf_detected_isotopes_only", False)
    )
    pf_detection_activation_only = bool(
        runtime_config.get("pf_detected_isotope_activation_only", False)
    )
    pf_detect_consecutive = max(
        1,
        int(
            runtime_config.get(
                "pf_detected_isotope_consecutive",
                1 if pf_detected_isotopes_only else max(int(detect_consecutive), 1),
            )
        ),
    )
    pf_detect_consecutive_by_isotope_raw = runtime_config.get(
        "pf_detected_isotope_consecutive_by_isotope",
    )
    pf_detect_consecutive_by_isotope = (
        {
            str(iso): max(1, int(value))
            for iso, value in pf_detect_consecutive_by_isotope_raw.items()
        }
        if isinstance(pf_detect_consecutive_by_isotope_raw, dict)
        else None
    )
    num_particles = max(1, int(num_particles))
    detector_model_payload = runtime_config.get("detector_model", {})
    if not isinstance(detector_model_payload, dict):
        detector_model_payload = {}
    observation_model = build_runtime_observation_model(
        runtime_config,
        isotopes=isotopes,
    )
    detector_geometry = observation_model.detector_geometry
    shield_thickness = resolve_shield_thickness_config(runtime_config)
    shield_params = observation_model.shield_params
    planning_map, measurement_workspace_diagnostics = _build_measurement_workspace(
        runtime_config,
        environment_size_xyz=(env.size_x, env.size_y, env.size_z),
        detector_height_config=detector_height_config,
        obstacle_grid=obstacle_grid,
        base_map=planning_map,
        shield_params=shield_params,
        effective_robot_radius_m=effective_robot_radius_m,
    )
    if isinstance(planning_map, MeasurementWorkspace):
        initial_pose = np.asarray(env.detector_position, dtype=float).reshape(1, 3)
        initial_validity = planning_map.endpoint_validity_masks(initial_pose)
        if not bool(initial_validity["valid"][0]):
            failed_checks = sorted(
                name
                for name, values in initial_validity.items()
                if name != "valid" and not bool(values[0])
            )
            raise ValueError(
                "Initial detector pose is not collision-free: "
                f"failed_checks={failed_checks}."
            )
        print(
            "3D measurement workspace: collision-aware free-volume planning enabled "
            f"(collision_boxes={measurement_workspace_diagnostics['collision_box_count']}, "
            f"base_radius={measurement_workspace_diagnostics['effective_robot_radius_m']:.3f}m, "
            f"head_radius={measurement_workspace_diagnostics['head_radius_m']:.3f}m, "
            f"transport_z={measurement_workspace_diagnostics['transport_world_z_m']:.3f}m)"
        )
    obstacle_buildup_coeff = observation_model.obstacle_buildup_coeff
    pf_obstacle_buildup_coeff = (
        obstacle_buildup_coeff if pf_obstacle_grid is not None else 0.0
    )
    print(
        "Shield thickness model: "
        f"scale={float(shield_thickness.thickness_scale):.6g} "
        f"target_transmission={shield_thickness.transmission_target} "
        f"Fe={float(shield_params.thickness_fe_cm):.4f}cm "
        f"Pb={float(shield_params.thickness_pb_cm):.4f}cm "
        f"inner_radii=(Fe {shield_params.inner_radius_fe_cm:.4f}cm, "
        f"Pb {shield_params.inner_radius_pb_cm:.4f}cm) "
        f"buildup=(Fe {shield_params.buildup_fe_coeff:.3g}, "
        f"Pb {shield_params.buildup_pb_coeff:.3g}, "
        f"obstacle {obstacle_buildup_coeff:.3g})"
    )
    print(
        "PF obstacle attenuation: "
        f"{'enabled' if pf_obstacle_attenuation_enabled else 'disabled'} "
        f"(environment_obstacles={_has_environment_obstacles(obstacle_grid)}, "
        f"pf_grid_active={_has_environment_obstacles(pf_obstacle_grid)}, "
        f"buildup_coeff={pf_obstacle_buildup_coeff:.3g})"
    )
    mu_by_isotope = observation_model.mu_by_isotope
    line_mu_by_isotope = observation_model.line_mu_by_isotope
    transport_response_model = observation_model.transport_response_model
    obstacle_mu_by_isotope = observation_model.obstacle_mu_by_isotope
    if line_mu_by_isotope is not None:
        print(
            "PF line-resolved shield attenuation: enabled "
            f"(isotopes={','.join(sorted(line_mu_by_isotope))})"
        )
    else:
        print("PF line-resolved shield attenuation: disabled")
    print(
        "PF transport response model: "
        f"{'enabled' if transport_response_model is not None else 'disabled'}"
    )
    use_gpu = _resolve_runtime_use_gpu(runtime_config)
    background_by_isotope = {iso: 5.0 for iso in isotopes}
    live_time = float(measurement_time_s)
    has_live_time_cap = np.isfinite(live_time) and live_time > 0.0
    planning_live_time = (
        live_time
        if has_live_time_cap
        else max(float(adaptive_min_dwell_s), float(adaptive_dwell_chunk_s))
    )
    observation_snr_floor = max(
        0.0,
        float(runtime_config.get("pose_min_observation_snr", 5.0)),
    )
    default_min_observation_counts = observation_snr_floor * observation_snr_floor
    if pose_min_observation_counts is None:
        pose_min_observation_counts_resolved = runtime_config.get(
            "pose_min_observation_counts",
            default_min_observation_counts,
        )
    else:
        pose_min_observation_counts_resolved = pose_min_observation_counts
    pose_min_observation_counts_resolved = max(
        float(pose_min_observation_counts_resolved),
        0.0,
    )
    pose_min_observation_penalty_scale = max(
        float(
            runtime_config.get(
                "pose_min_observation_penalty_scale",
                pose_min_observation_penalty_scale,
            )
        ),
        0.0,
    )
    pose_min_observation_aggregate = (
        str(
            runtime_config.get(
                "pose_min_observation_aggregate",
                pose_min_observation_aggregate,
            )
        )
        .strip()
        .lower()
    )
    pose_min_observation_max_particles = runtime_config.get(
        "pose_min_observation_max_particles",
        None,
    )
    if pose_min_observation_max_particles is not None:
        pose_min_observation_max_particles = int(pose_min_observation_max_particles)
    pose_min_observation_quantile = float(
        runtime_config.get("pose_min_observation_quantile", 0.25)
    )
    path_planner_resolved = (
        str(
            path_planner
            if path_planner is not None
            else runtime_config.get("path_planner", "one_step")
        )
        .strip()
        .lower()
    )
    if path_planner_resolved in {"dss", "dss-pp", "dsspp"}:
        path_planner_resolved = "dss_pp"
    if path_planner_resolved not in {"one_step", "dss_pp"}:
        raise ValueError("path_planner must be 'one_step' or 'dss_pp'.")
    dss_runtime = runtime_config.get("dss_pp", {})
    if not isinstance(dss_runtime, dict):
        dss_runtime = {}
    python_worker_count_resolved = _resolve_python_worker_count(
        runtime_config.get(
            "python_worker_count",
            runtime_config.get("cpu_worker_count", 0),
        )
    )

    def _dss_value(key: str, default: object) -> object:
        """Read a DSS-PP setting from CLI override or runtime config."""
        return dss_runtime.get(key, runtime_config.get(f"dss_{key}", default))

    dss_horizon_resolved = int(
        dss_horizon if dss_horizon is not None else _dss_value("horizon", 2)
    )
    dss_beam_width_resolved = int(
        dss_beam_width if dss_beam_width is not None else _dss_value("beam_width", 8)
    )
    dss_program_length_resolved = int(
        dss_program_length
        if dss_program_length is not None
        else _dss_value("program_length", 2)
    )
    dss_residual_program_length_resolved = max(
        dss_program_length_resolved,
        int(_dss_value("residual_program_length", 16)),
    )
    dss_adaptive_program_length_enabled = bool(
        _dss_value("adaptive_program_length_enable", True)
    )
    dss_adaptive_simple_program_length = max(
        1,
        int(_dss_value("adaptive_simple_program_length", 2)),
    )
    dss_adaptive_residual_budget_threshold = max(
        0.0,
        float(_dss_value("adaptive_residual_budget_threshold", 1.0e-9)),
    )
    dss_adaptive_ambiguity_budget_threshold = max(
        0.0,
        float(_dss_value("adaptive_ambiguity_budget_threshold", 1.0e-9)),
    )
    dss_residual_extension_requires_cardinality_evidence = bool(
        _dss_value("residual_extension_requires_cardinality_evidence", False)
    )
    pose_selection_workers_resolved = max(
        1,
        int(
            runtime_config.get(
                "pose_selection_workers",
                runtime_config.get(
                    "one_step_pose_eval_workers",
                    python_worker_count_resolved,
                ),
            )
        ),
    )
    one_step_pose_eval_use_gpu = _optional_runtime_bool(
        runtime_config,
        "one_step_pose_eval_use_gpu",
    )
    dss_one_step_guard_enabled = bool(_dss_value("one_step_guard_enable", True))
    dss_one_step_guard_abs_margin = max(
        0.0,
        float(_dss_value("one_step_guard_score_abs_margin", 0.0)),
    )
    dss_one_step_guard_rel_margin = max(
        0.0,
        float(_dss_value("one_step_guard_score_rel_margin", 0.0)),
    )
    dss_one_step_guard_use_gpu_payload = _dss_value("one_step_guard_use_gpu", None)
    dss_one_step_guard_use_gpu = (
        one_step_pose_eval_use_gpu
        if dss_one_step_guard_use_gpu_payload is None
        else _optional_runtime_bool(
            {"one_step_guard_use_gpu": dss_one_step_guard_use_gpu_payload},
            "one_step_guard_use_gpu",
        )
    )
    dss_signature_weight_resolved = float(
        dss_signature_weight
        if dss_signature_weight is not None
        else _dss_value("signature_weight", 1.0)
    )
    dss_differential_weight_resolved = float(
        dss_differential_weight
        if dss_differential_weight is not None
        else _dss_value("differential_weight", 1.0)
    )
    dss_rotation_weight_resolved = float(
        dss_rotation_weight
        if dss_rotation_weight is not None
        else _dss_value("rotation_weight", 0.15)
    )
    dss_planning_particles_resolved = _dss_value(
        "planning_particles",
        runtime_config.get("planning_rollout_particles", 512),
    )
    dss_planning_method_resolved = _dss_value(
        "planning_method",
        runtime_config.get("planning_rollout_method", "resample"),
    )
    dss_config = DSSPPConfig(
        horizon=max(1, dss_horizon_resolved),
        beam_width=max(1, dss_beam_width_resolved),
        max_programs=max(1, int(_dss_value("max_programs", 40))),
        program_length=max(1, dss_program_length_resolved),
        mode_cluster_radius_m=float(_dss_value("mode_cluster_radius_m", 1.5)),
        max_modes_per_isotope=max(1, int(_dss_value("max_modes_per_isotope", 4))),
        planning_particles=(
            None
            if dss_planning_particles_resolved is None
            else int(dss_planning_particles_resolved)
        ),
        planning_method=(
            None
            if dss_planning_method_resolved is None
            else str(dss_planning_method_resolved)
        ),
        live_time_s=planning_live_time,
        primary_history_weight=_planning_primary_history_weight(runtime_config),
        target_sampled_primaries=_target_sampled_primaries(runtime_config),
        transport_detector_radius_m=_transport_detector_budget_radius_m(runtime_config),
        lambda_eig=float(_dss_value("eig_weight", 1.0)),
        lambda_signature=max(0.0, dss_signature_weight_resolved),
        lambda_distance=(
            None
            if _dss_value("distance_weight", None) is None
            else float(_dss_value("distance_weight", 0.0))
        ),
        lambda_time=max(0.0, float(_dss_value("time_weight", 0.0))),
        lambda_rotation=max(0.0, dss_rotation_weight_resolved),
        lambda_dose=max(0.0, float(_dss_value("dose_weight", 0.0))),
        lambda_coverage=max(0.0, float(_dss_value("coverage_weight", 0.0))),
        lambda_bearing_diversity=max(
            0.0,
            float(_dss_value("bearing_diversity_weight", 0.0)),
        ),
        lambda_frontier=max(
            0.0,
            float(_dss_value("frontier_weight", 0.0)),
        ),
        lambda_turn_smoothness=max(
            0.0,
            float(_dss_value("turn_smoothness_weight", 0.0)),
        ),
        lambda_temporal_separation=max(
            0.0,
            float(_dss_value("temporal_separation_weight", 0.0)),
        ),
        lambda_count_utility=max(
            0.0,
            float(_dss_value("count_utility_weight", 0.75)),
        ),
        lambda_local_orbit=max(
            0.0,
            float(_dss_value("local_orbit_weight", 0.75)),
        ),
        lambda_station_condition=max(
            0.0,
            float(_dss_value("station_condition_weight", 0.75)),
        ),
        lambda_correlation_reduction=max(
            0.0,
            float(_dss_value("correlation_reduction_weight", 0.0)),
        ),
        lambda_isotope_balance=max(
            0.0,
            float(_dss_value("isotope_balance_weight", 0.0)),
        ),
        lambda_environment_signature=max(
            0.0,
            float(_dss_value("environment_signature_weight", 0.0)),
        ),
        lambda_occlusion_boundary=max(
            0.0,
            float(_dss_value("occlusion_boundary_weight", 0.0)),
        ),
        lambda_elevation_signature=max(
            0.0,
            float(_dss_value("elevation_signature_weight", 0.0)),
        ),
        lambda_elevation_condition=max(
            0.0,
            float(_dss_value("elevation_condition_weight", 0.0)),
        ),
        lambda_vertical_environment_signature=max(
            0.0,
            float(_dss_value("vertical_environment_signature_weight", 0.0)),
        ),
        residual_signature_weight=max(
            0.0,
            float(_dss_value("residual_signature_weight", 1.0)),
        ),
        eta_observation=max(
            0.0,
            float(
                _dss_value(
                    "observation_weight",
                    pose_min_observation_penalty_scale,
                )
            ),
        ),
        eta_differential=max(0.0, dss_differential_weight_resolved),
        eta_count_balance=max(
            0.0,
            float(
                _dss_value(
                    "count_balance_weight",
                    runtime_config.get("shield_count_balance_weight", 0.5),
                )
            ),
        ),
        eta_revisit=max(
            0.0,
            float(_dss_value("revisit_penalty_weight", 0.0)),
        ),
        min_observation_counts=pose_min_observation_counts_resolved,
        enforce_min_observation=bool(_dss_value("enforce_min_observation", True)),
        signature_std_min_counts=max(
            0.0,
            float(_dss_value("signature_std_min_counts", 1.0)),
        ),
        count_variance_floor=max(
            1e-12,
            float(_dss_value("count_variance_floor", 1.0)),
        ),
        coverage_radius_m=max(
            0.0,
            float(_dss_value("coverage_radius_m", 3.0)),
        ),
        coverage_grid_max_cells=max(
            0,
            int(_dss_value("coverage_grid_max_cells", 5000)),
        ),
        coverage_floor_quantile=float(_dss_value("coverage_floor_quantile", 0.0)),
        coverage_floor_weight=max(
            0.0,
            float(_dss_value("coverage_floor_weight", 0.0)),
        ),
        min_station_separation_m=max(
            0.0,
            float(_dss_value("min_station_separation_m", pose_min_dist)),
        ),
        detector_aperture_samples=max(
            1,
            int(
                _dss_value(
                    "detector_aperture_samples",
                    detector_geometry.aperture_samples,
                )
            ),
        ),
        robot_speed_m_s=float(nominal_motion_speed_m_s),
        rotation_overhead_s=float(rotation_overhead_s),
        augment_candidates=bool(_dss_value("augment_candidates", True)),
        max_augmented_candidates=max(
            pose_candidates,
            int(_dss_value("max_augmented_candidates", 256)),
        ),
        count_utility_saturation_counts=max(
            1.0e-12,
            float(
                _dss_value(
                    "count_utility_saturation_counts",
                    max(100.0, 5.0 * float(pose_min_observation_counts_resolved)),
                )
            ),
        ),
        local_orbit_sigma_m=max(
            1.0e-6,
            float(_dss_value("local_orbit_sigma_m", 0.75)),
        ),
        station_condition_ridge=max(
            1.0e-12,
            float(_dss_value("station_condition_ridge", 1.0e-3)),
        ),
        station_condition_min_singular_weight=max(
            0.0,
            float(_dss_value("station_condition_min_singular_weight", 0.0)),
        ),
        station_condition_inverse_condition_weight=max(
            0.0,
            float(_dss_value("station_condition_inverse_condition_weight", 0.0)),
        ),
        station_condition_coherence_weight=max(
            0.0,
            float(_dss_value("station_condition_coherence_weight", 0.0)),
        ),
        environment_contrast_threshold=max(
            1.0e-12,
            float(_dss_value("environment_contrast_threshold", 0.25)),
        ),
        environment_signature_score_clip=max(
            1.0e-12,
            float(_dss_value("environment_signature_score_clip", 3.0)),
        ),
        occlusion_boundary_step_m=max(
            0.0,
            float(_dss_value("occlusion_boundary_step_m", 0.5)),
        ),
        elevation_pair_z_scale_m=max(
            1.0e-9,
            float(_dss_value("elevation_pair_z_scale_m", 2.0)),
        ),
        elevation_pair_xy_scale_m=max(
            1.0e-9,
            float(_dss_value("elevation_pair_xy_scale_m", 4.0)),
        ),
        elevation_angle_threshold_deg=max(
            1.0e-6,
            float(_dss_value("elevation_angle_threshold_deg", 15.0)),
        ),
        eig_candidate_limit=(
            None
            if _dss_value("eig_candidate_limit", 8) is None
            else int(_dss_value("eig_candidate_limit", 8))
        ),
        temporal_cover_weight=max(
            0.0,
            float(_dss_value("temporal_cover_weight", 1.0)),
        ),
        temporal_logdet_weight=max(
            0.0,
            float(_dss_value("temporal_logdet_weight", 0.25)),
        ),
        temporal_decorrelation_weight=max(
            0.0,
            float(_dss_value("temporal_decorrelation_weight", 0.5)),
        ),
        temporal_pair_contrast_threshold=max(
            1.0e-12,
            float(_dss_value("temporal_pair_contrast_threshold", 0.25)),
        ),
        temporal_logdet_ridge=max(
            1.0e-12,
            float(_dss_value("temporal_logdet_ridge", 1.0e-3)),
        ),
        temporal_cover_programs=max(
            0,
            int(_dss_value("temporal_cover_programs", 1)),
        ),
        temporal_cover_beam_width=max(
            1,
            int(_dss_value("temporal_cover_beam_width", 4)),
        ),
        program_eval_workers=(
            python_worker_count_resolved
            if _dss_value("program_eval_workers", None) is None
            else max(1, int(_dss_value("program_eval_workers", 1)))
        ),
        candidate_preselect_enable=bool(_dss_value("candidate_preselect_enable", True)),
        candidate_preselect_min=max(
            1,
            int(_dss_value("candidate_preselect_min", 32)),
        ),
        candidate_preselect_multiplier=max(
            1,
            int(_dss_value("candidate_preselect_multiplier", 8)),
        ),
        remaining_budget_guidance=bool(_dss_value("remaining_budget_guidance", True)),
        remaining_budget_urgency_stations=max(
            1,
            int(_dss_value("remaining_budget_urgency_stations", 4)),
        ),
        remaining_route_weight=max(
            0.0,
            float(_dss_value("remaining_route_weight", 2.0)),
        ),
        remaining_route_distance_weight=max(
            0.0,
            float(_dss_value("remaining_route_distance_weight", 0.5)),
        ),
        remaining_route_revisit_weight=max(
            0.0,
            float(_dss_value("remaining_route_revisit_weight", 1.0)),
        ),
        remaining_route_turn_weight=max(
            0.0,
            float(_dss_value("remaining_route_turn_weight", 0.75)),
        ),
        remaining_route_backtrack_weight=max(
            0.0,
            float(_dss_value("remaining_route_backtrack_weight", 1.0)),
        ),
        remaining_route_coverage_weight=max(
            0.0,
            float(_dss_value("remaining_route_coverage_weight", 0.5)),
        ),
        remaining_route_frontier_weight=max(
            0.0,
            float(_dss_value("remaining_route_frontier_weight", 0.5)),
        ),
        same_isotope_direct_separation_guard=bool(
            _dss_value("same_isotope_direct_separation_guard", True)
        ),
        same_isotope_direct_separation_epsilon=max(
            0.0,
            float(_dss_value("same_isotope_direct_separation_epsilon", 1.0e-9)),
        ),
        recovery_isotope_mode_weight_multiplier=max(
            1.0,
            float(_dss_value("recovery_isotope_mode_weight_multiplier", 2.0)),
        ),
        weak_mode_weight_floor=max(
            0.0,
            float(_dss_value("weak_mode_weight_floor", 0.0)),
        ),
        dominant_mode_weight_cap=float(
            np.clip(float(_dss_value("dominant_mode_weight_cap", 1.0)), 0.0, 1.0)
        ),
        high_surface_pair_boost=max(
            1.0,
            float(_dss_value("high_surface_pair_boost", 1.0)),
        ),
        high_surface_cross_stratum_boost=max(
            1.0,
            float(_dss_value("high_surface_cross_stratum_boost", 1.0)),
        ),
        high_surface_z_fraction=float(
            np.clip(float(_dss_value("high_surface_z_fraction", 0.75)), 0.0, 1.0)
        ),
        high_surface_pair_distance_m=max(
            0.0,
            float(_dss_value("high_surface_pair_distance_m", 0.0)),
        ),
        diagnostic_ranked_node_limit=int(
            _dss_value("diagnostic_ranked_node_limit", 64)
        ),
        explicit_mode_switch=bool(_dss_value("explicit_mode_switch", False)),
        planner_mode=str(_dss_value("planner_mode", "balanced")),
        rng_seed=obstacle_seed,
    )
    remaining_runtime = runtime_config.get("remaining_measurement_estimate", {})
    if not isinstance(remaining_runtime, dict):
        remaining_runtime = {}

    def _remaining_value(key: str, default: object) -> object:
        """Read remaining-measurement estimator settings from runtime config."""
        if key in remaining_runtime:
            return remaining_runtime[key]
        legacy_key = f"remaining_measurement_{key}"
        return runtime_config.get(legacy_key, default)

    remaining_measurement_config = RemainingMeasurementConfig(
        enabled=bool(_remaining_value("enabled", True)),
        mode_cluster_radius_m=max(
            1.0e-6,
            float(
                _remaining_value(
                    "mode_cluster_radius_m",
                    dss_config.mode_cluster_radius_m,
                )
            ),
        ),
        max_modes_per_isotope=max(
            1,
            int(
                _remaining_value(
                    "max_modes_per_isotope",
                    dss_config.max_modes_per_isotope,
                )
            ),
        ),
        max_particles=(
            None
            if _remaining_value("max_particles", dss_config.planning_particles) is None
            else max(
                1,
                int(_remaining_value("max_particles", dss_config.planning_particles)),
            )
        ),
        planning_method=(
            None
            if _remaining_value("planning_method", dss_config.planning_method) is None
            else str(_remaining_value("planning_method", dss_config.planning_method))
        ),
        target_position_spread_m=max(
            1.0e-6,
            float(_remaining_value("target_position_spread_m", 1.0)),
        ),
        target_strength_cv=max(
            1.0e-6,
            float(_remaining_value("target_strength_cv", 0.5)),
        ),
        target_cardinality_confidence=float(
            np.clip(
                float(_remaining_value("target_cardinality_confidence", 0.9)),
                0.0,
                1.0,
            )
        ),
        pairwise_separation_threshold=max(
            0.0,
            float(_remaining_value("pairwise_separation_threshold", 9.0)),
        ),
        residual_chi2_threshold=max(
            1.0e-12,
            float(_remaining_value("residual_chi2_threshold", 9.0)),
        ),
        count_variance_floor=max(
            1.0e-12,
            float(
                _remaining_value(
                    "count_variance_floor",
                    dss_config.count_variance_floor,
                )
            ),
        ),
        stop_budget=max(0.0, float(_remaining_value("stop_budget", 0.0))),
        eta_default=float(_remaining_value("eta_default", 0.7)),
        eta_min=float(_remaining_value("eta_min", 0.3)),
        eta_max=float(_remaining_value("eta_max", 1.0)),
        gain_epsilon=max(
            1.0e-12,
            float(_remaining_value("gain_epsilon", 1.0e-6)),
        ),
        max_reported_stations=max(
            0,
            int(_remaining_value("max_reported_stations", 99)),
        ),
        uncertainty_weight=max(
            0.0,
            float(_remaining_value("uncertainty_weight", 1.0)),
        ),
        cardinality_weight=max(
            0.0,
            float(_remaining_value("cardinality_weight", 1.0)),
        ),
        separation_weight=max(
            0.0,
            float(_remaining_value("separation_weight", 1.5)),
        ),
        verification_weight=max(
            0.0,
            float(_remaining_value("verification_weight", 1.0)),
        ),
        residual_weight=max(
            0.0,
            float(_remaining_value("residual_weight", 1.0)),
        ),
        high_surface_ambiguity_weight=max(
            0.0,
            float(_remaining_value("high_surface_ambiguity_weight", 1.0)),
        ),
        high_surface_z_fraction=float(
            np.clip(
                float(_remaining_value("high_surface_z_fraction", 0.75)),
                0.0,
                1.0,
            )
        ),
        high_surface_pairwise_separation_threshold=max(
            0.0,
            float(
                _remaining_value(
                    "high_surface_pairwise_separation_threshold",
                    _remaining_value("pairwise_separation_threshold", 9.0),
                )
            ),
        ),
        dss_information_gain_weight=max(
            0.0,
            float(_remaining_value("dss_information_gain_weight", 1.0)),
        ),
        dss_count_utility_weight=max(
            0.0,
            float(_remaining_value("dss_count_utility_weight", 0.25)),
        ),
        range_scale=max(1.0, float(_remaining_value("range_scale", 1.35))),
        unresolved_absent_min_total_counts=max(
            0.0,
            float(_remaining_value("unresolved_absent_min_total_counts", 25.0)),
        ),
        unresolved_absent_min_max_counts=max(
            0.0,
            float(_remaining_value("unresolved_absent_min_max_counts", 5.0)),
        ),
        unresolved_absent_min_snr=max(
            0.0,
            float(_remaining_value("unresolved_absent_min_snr", 2.0)),
        ),
        unresolved_absent_budget_weight=max(
            0.0,
            float(_remaining_value("unresolved_absent_budget_weight", 1.0)),
        ),
    )
    likelihood_runtime = runtime_config.get("pf_count_likelihood", {})
    if not isinstance(likelihood_runtime, dict):
        likelihood_runtime = {}
    geant4_likelihood_defaults = sim_backend.strip().lower() == "geant4"
    spectrum_estimate_likelihood_defaults = (
        spectrum_count_method == RuntimeCountExtractor.STANDARD_METHOD
    )

    def _likelihood_config_value(key: str, default: object) -> object:
        """Read a PF likelihood setting from nested or legacy runtime config keys."""
        legacy_key = f"pf_{key}"
        if key in likelihood_runtime:
            return likelihood_runtime[key]
        return runtime_config.get(legacy_key, default)

    count_likelihood_model = str(
        _likelihood_config_value(
            "count_likelihood_model",
            DEFAULT_GEANT4_COUNT_LIKELIHOOD_MODEL
            if geant4_likelihood_defaults or spectrum_estimate_likelihood_defaults
            else "poisson",
        )
    )
    transport_model_rel_sigma = _likelihood_config_value(
        "transport_model_rel_sigma",
        DEFAULT_GEANT4_TRANSPORT_MODEL_REL_SIGMA if geant4_likelihood_defaults else 0.0,
    )
    transport_model_abs_sigma = _likelihood_config_value(
        "transport_model_abs_sigma",
        DEFAULT_GEANT4_TRANSPORT_MODEL_ABS_SIGMA if geant4_likelihood_defaults else 0.0,
    )
    spectrum_count_rel_sigma = _likelihood_config_value(
        "spectrum_count_rel_sigma",
        DEFAULT_GEANT4_SPECTRUM_COUNT_REL_SIGMA if geant4_likelihood_defaults else 0.0,
    )
    spectrum_count_abs_sigma = _likelihood_config_value(
        "spectrum_count_abs_sigma",
        DEFAULT_GEANT4_SPECTRUM_COUNT_ABS_SIGMA if geant4_likelihood_defaults else 0.0,
    )
    low_count_abs_sigma = _likelihood_config_value(
        "low_count_abs_sigma",
        DEFAULT_GEANT4_LOW_COUNT_ABS_SIGMA if geant4_likelihood_defaults else 0.0,
    )
    low_count_transition_counts = _likelihood_config_value(
        "low_count_transition_counts",
        DEFAULT_GEANT4_LOW_COUNT_TRANSITION_COUNTS
        if geant4_likelihood_defaults
        else 0.0,
    )
    observation_count_variance_includes_counting_noise = bool(
        _likelihood_config_value(
            "observation_count_variance_includes_counting_noise",
            False,
        )
    )
    observation_count_variance_semantics = str(
        _likelihood_config_value(
            "observation_count_variance_semantics",
            "",
        )
    )
    direct_spectrum_likelihood_enable = bool(
        _likelihood_config_value(
            "direct_spectrum_likelihood_enable",
            True,
        )
    )
    count_likelihood_df = float(
        _likelihood_config_value(
            "count_likelihood_df",
            DEFAULT_GEANT4_COUNT_LIKELIHOOD_DF,
        )
    )
    station_view_covariance_enable = bool(
        _likelihood_config_value(
            "station_view_covariance_enable",
            bool(geant4_likelihood_defaults or spectrum_estimate_likelihood_defaults),
        )
    )
    station_view_correlated_spectrum_fraction = float(
        _likelihood_config_value(
            "station_view_correlated_spectrum_fraction",
            1.0
            if geant4_likelihood_defaults or spectrum_estimate_likelihood_defaults
            else 0.0,
        )
    )
    shield_contrast_runtime = runtime_config.get("pf_shield_contrast_likelihood", {})
    if not isinstance(shield_contrast_runtime, dict):
        shield_contrast_runtime = {}

    def _shield_contrast_config_value(key: str, default: object) -> object:
        """Read same-station shield-contrast likelihood settings."""
        legacy_key = f"pf_shield_contrast_{key}"
        if key in shield_contrast_runtime:
            return shield_contrast_runtime[key]
        return runtime_config.get(legacy_key, default)

    shield_contrast_likelihood_enable = bool(
        _shield_contrast_config_value("enabled", geant4_likelihood_defaults)
    )
    shield_contrast_likelihood_weight = max(
        0.0,
        float(_shield_contrast_config_value("weight", 1.0)),
    )
    shield_contrast_log_sigma_floor = max(
        1.0e-6,
        float(_shield_contrast_config_value("log_sigma_floor", 0.5)),
    )
    shield_contrast_log_sigma_ceiling = max(
        shield_contrast_log_sigma_floor,
        float(_shield_contrast_config_value("log_sigma_ceiling", 2.0)),
    )
    shield_contrast_min_count = max(
        1.0e-6,
        float(_shield_contrast_config_value("min_count", 25.0)),
    )
    shield_contrast_min_views = max(
        2,
        int(_shield_contrast_config_value("min_views", 2)),
    )
    shield_contrast_likelihood_df = max(
        1.0,
        float(_shield_contrast_config_value("df", 5.0)),
    )
    shield_view_ratio_runtime = runtime_config.get(
        "pf_shield_view_ratio_likelihood", {}
    )
    if not isinstance(shield_view_ratio_runtime, dict):
        shield_view_ratio_runtime = {}

    def _shield_view_ratio_config_value(key: str, default: object) -> object:
        """Read same-station shield-view ratio likelihood settings."""
        legacy_key = f"pf_shield_view_ratio_{key}"
        if key in shield_view_ratio_runtime:
            return shield_view_ratio_runtime[key]
        return runtime_config.get(legacy_key, default)

    shield_view_ratio_likelihood_enable = bool(
        _shield_view_ratio_config_value("enabled", geant4_likelihood_defaults)
    )
    shield_view_ratio_likelihood_weight = max(
        0.0,
        float(_shield_view_ratio_config_value("weight", 1.0)),
    )
    shield_view_ratio_likelihood_concentration = max(
        1.0e-6,
        float(_shield_view_ratio_config_value("concentration", 128.0)),
    )
    shield_view_ratio_likelihood_min_total_count = max(
        0.0,
        float(_shield_view_ratio_config_value("min_total_count", 25.0)),
    )
    shield_view_ratio_likelihood_min_views = max(
        2,
        int(_shield_view_ratio_config_value("min_views", 2)),
    )
    _validate_weighted_pf_runtime_contract(
        runtime_config,
        count_likelihood_model=count_likelihood_model,
        observation_variance_semantics=observation_count_variance_semantics,
        direct_spectrum_likelihood_enable=direct_spectrum_likelihood_enable,
        shield_contrast_likelihood_enable=shield_contrast_likelihood_enable,
        shield_view_ratio_likelihood_enable=shield_view_ratio_likelihood_enable,
        planning_primary_history_weight=_planning_primary_history_weight(
            runtime_config
        ),
    )
    adaptive_mission_stop = bool(runtime_config.get("adaptive_mission_stop", False))
    max_steps = _resolve_mission_max_steps(max_steps, runtime_config)
    max_poses = _resolve_mission_max_poses(max_poses, runtime_config)
    mission_stop_min_convergence_poses = max(
        1,
        int(
            runtime_config.get(
                "mission_stop_min_convergence_poses",
                runtime_config.get("mission_stop_convergence_min_poses", 4),
            )
        ),
    )
    if max_poses is not None and int(max_poses) > 0:
        mission_stop_min_convergence_poses = min(
            mission_stop_min_convergence_poses,
            int(max_poses),
        )
    mission_stop_coverage_radius_m = max(
        0.0,
        float(runtime_config.get("mission_stop_coverage_radius_m", 4.0)),
    )
    mission_stop_coverage_fraction = float(
        np.clip(
            float(runtime_config.get("mission_stop_coverage_fraction", 0.85)),
            0.0,
            1.0,
        )
    )
    mission_stop_require_quiet_birth_residual = bool(
        runtime_config.get("mission_stop_require_quiet_birth_residual", True)
    )
    mission_stop_require_pf_convergence_for_coverage = bool(
        runtime_config.get("mission_stop_require_pf_convergence_for_coverage", False)
    )
    mission_stop_birth_residual_min_support = max(
        1,
        int(runtime_config.get("mission_stop_birth_residual_min_support", 1)),
    )
    mission_stop_require_no_unresolved_discriminative_failures = bool(
        runtime_config.get(
            "mission_stop_require_no_unresolved_discriminative_failures",
            True,
        )
    )
    mission_stop_require_pf_cardinality_ready = bool(
        runtime_config.get("mission_stop_require_pf_cardinality_ready", True)
    )
    mission_stop_require_remaining_measurement_ready = bool(
        runtime_config.get("mission_stop_require_remaining_measurement_ready", True)
    )
    mission_stop_soft_extend_on_unresolved = bool(
        runtime_config.get("mission_stop_soft_extend_on_unresolved", False)
    )
    mission_stop_soft_extension_poses = max(
        0,
        int(runtime_config.get("mission_stop_soft_extension_poses", 4)),
    )
    mission_stop_soft_extension_require_progress = bool(
        runtime_config.get(
            "mission_stop_soft_extension_require_progress",
            True,
        )
    )
    mission_stop_unresolved_discriminative_fail_min_count = max(
        1,
        int(
            runtime_config.get(
                "mission_stop_unresolved_discriminative_fail_min_count",
                runtime_config.get(
                    "mission_stop_unresolved_discriminative_failure_min_count",
                    1,
                ),
            )
        ),
    )
    simulation_runtime = create_simulation_runtime(
        sim_backend,
        sources=sources,
        decomposer=decomposer,
        mu_by_isotope=mu_by_isotope,
        shield_params=shield_params,
        runtime_config=runtime_config,
        runtime_config_path=sim_config_path,
    )
    isaac_pf_visualization_enabled = bool(
        runtime_config.get("isaacsim_show_pf_particles", True)
    )
    isaac_pf_max_particles_raw = runtime_config.get(
        "isaacsim_pf_max_particles_per_isotope",
        runtime_config.get("pf_visual_max_particles_per_isotope", 800),
    )
    isaac_pf_max_particles = (
        None
        if isaac_pf_max_particles_raw is None
        else max(1, int(isaac_pf_max_particles_raw))
    )
    isaac_pf_visualization_warning_printed = False

    def _send_isaac_pf_visualization(frame: PFFrame) -> None:
        """Send a PF frame to an Isaac Sim runtime when available."""
        nonlocal isaac_pf_visualization_warning_printed
        if not isaac_pf_visualization_enabled:
            return
        visualizer = getattr(simulation_runtime, "visualize_pf_state", None)
        if visualizer is None:
            return
        payload = frame_to_isaac_pf_payload(
            frame,
            max_particles_per_isotope=isaac_pf_max_particles,
        )
        try:
            visualizer(payload)
        except Exception as exc:
            if not isaac_pf_visualization_warning_printed:
                print(f"Isaac Sim PF visualization disabled after error: {exc}")
                isaac_pf_visualization_warning_printed = True

    orientation_limit_resolved = max(
        1,
        int(
            runtime_config.get(
                "orientation_k",
                runtime_config.get("rotations_per_pose", 2),
            )
        ),
    )
    min_rotations_resolved = max(
        0,
        int(
            runtime_config.get(
                "min_rotations_per_pose",
                min(2, orientation_limit_resolved),
            )
        ),
    )
    pf_max_sources_raw = runtime_config.get(
        "pf_max_sources",
        DEFAULT_MAX_SOURCES_PER_ISOTOPE,
    )
    pf_max_sources = (
        DEFAULT_MAX_SOURCES_PER_ISOTOPE
        if pf_max_sources_raw is None
        else max(1, int(pf_max_sources_raw))
    )
    init_num_sources_raw = runtime_config.get("init_num_sources", None)
    if (
        isinstance(init_num_sources_raw, (list, tuple))
        and len(init_num_sources_raw) == 2
    ):
        init_num_sources = (
            max(0, int(init_num_sources_raw[0])),
            max(0, int(init_num_sources_raw[1])),
        )
    else:
        default_init_max = min(DEFAULT_MAX_SOURCES_PER_ISOTOPE, pf_max_sources)
        init_min_raw = runtime_config.get("init_num_sources_min", None)
        init_max_raw = runtime_config.get("init_num_sources_max", None)
        init_min = (
            (0 if birth_enabled else 1) if init_min_raw is None else int(init_min_raw)
        )
        init_max = default_init_max if init_max_raw is None else int(init_max_raw)
        init_num_sources = (
            max(0, init_min),
            max(0, init_max),
        )
    if init_num_sources[1] < init_num_sources[0]:
        init_num_sources = (init_num_sources[1], init_num_sources[0])
    if pf_max_sources is not None:
        init_num_sources = (
            min(init_num_sources[0], pf_max_sources),
            min(init_num_sources[1], pf_max_sources),
        )
    if init_num_sources[1] <= 0 and not birth_enabled:
        init_num_sources = (1, 1)
    (
        init_strength_prior,
        init_strength_min,
        init_strength_max,
    ) = _resolve_pf_initial_strength_prior(runtime_config)
    parallel_isotope_workers_raw = runtime_config.get(
        "parallel_isotope_workers",
        python_worker_count_resolved,
    )
    parallel_isotope_workers = (
        None
        if parallel_isotope_workers_raw is None
        else max(1, int(parallel_isotope_workers_raw))
    )
    birth_jitter_topk_raw = runtime_config.get("birth_jitter_topk_candidates", 512)
    birth_jitter_topk_candidates = (
        None if birth_jitter_topk_raw is None else max(1, int(birth_jitter_topk_raw))
    )
    structural_proposal_topk_raw = runtime_config.get(
        "structural_proposal_topk_particles",
        None,
    )
    structural_proposal_topk_particles = (
        None
        if structural_proposal_topk_raw is None
        else max(1, int(structural_proposal_topk_raw))
    )
    (
        structural_trial_workers,
        structural_trial_parallel_min_trials,
    ) = _resolve_structural_trial_parallelism(runtime_config)
    pf_conf = RotatingShieldPFConfig(
        estimator_profile=str(runtime_config.get("estimator_profile", "pf_strict")),
        num_particles=num_particles,
        min_particles=num_particles,
        max_particles=num_particles,
        max_sources=pf_max_sources,
        resample_threshold=0.7,
        position_sigma=0.5,
        background_level=background_by_isotope,
        measurement_scale_by_isotope=runtime_config.get("measurement_scale_by_isotope"),
        measurement_scale_by_isotope_and_pair=runtime_config.get(
            "measurement_scale_by_isotope_and_pair"
        ),
        count_likelihood_model=count_likelihood_model,
        transport_model_rel_sigma=transport_model_rel_sigma,
        transport_model_abs_sigma=transport_model_abs_sigma,
        spectrum_count_rel_sigma=spectrum_count_rel_sigma,
        spectrum_count_abs_sigma=spectrum_count_abs_sigma,
        low_count_abs_sigma=low_count_abs_sigma,
        low_count_transition_counts=low_count_transition_counts,
        observation_count_variance_includes_counting_noise=(
            observation_count_variance_includes_counting_noise
        ),
        observation_count_variance_semantics=(observation_count_variance_semantics),
        count_likelihood_df=count_likelihood_df,
        station_view_covariance_enable=station_view_covariance_enable,
        station_view_correlated_spectrum_fraction=(
            station_view_correlated_spectrum_fraction
        ),
        direct_spectrum_likelihood_enable=direct_spectrum_likelihood_enable,
        spectrum_likelihood_bin_chunk=max(
            1,
            int(runtime_config.get("spectrum_likelihood_bin_chunk", 512)),
        ),
        shield_contrast_likelihood_enable=shield_contrast_likelihood_enable,
        shield_contrast_likelihood_weight=shield_contrast_likelihood_weight,
        shield_contrast_log_sigma_floor=shield_contrast_log_sigma_floor,
        shield_contrast_log_sigma_ceiling=shield_contrast_log_sigma_ceiling,
        shield_contrast_min_count=shield_contrast_min_count,
        shield_contrast_min_views=shield_contrast_min_views,
        shield_contrast_likelihood_df=shield_contrast_likelihood_df,
        shield_view_ratio_likelihood_enable=shield_view_ratio_likelihood_enable,
        shield_view_ratio_likelihood_weight=shield_view_ratio_likelihood_weight,
        shield_view_ratio_likelihood_concentration=(
            shield_view_ratio_likelihood_concentration
        ),
        shield_view_ratio_likelihood_min_total_count=(
            shield_view_ratio_likelihood_min_total_count
        ),
        shield_view_ratio_likelihood_min_views=(shield_view_ratio_likelihood_min_views),
        min_strength=5.0,
        p_birth=0.05,
        p_kill=0.1,
        short_time_s=planning_live_time,
        max_dwell_time_s=10000.0,
        position_min=source_position_min,
        position_max=source_position_max,
        source_position_prior="surface" if source_surface_prior else "volume",
        surface_rejuvenation_enable=bool(
            runtime_config.get("surface_rejuvenation_enable", True)
        ),
        init_num_sources=init_num_sources,
        init_grid_spacing_m=1.0,
        init_grid_repeats=max(1, int(runtime_config.get("init_grid_repeats", 1))),
        init_joint_position_design=str(
            runtime_config.get("pf_init_joint_position_design", "latin_hypercube")
        ),
        init_joint_position_retries=max(
            1,
            int(runtime_config.get("pf_init_joint_position_retries", 16)),
        ),
        init_source_min_separation_m=max(
            0.0,
            float(
                runtime_config.get(
                    "pf_init_source_min_separation_m",
                    runtime_config.get(
                        "random_source_same_isotope_min_distance_m",
                        0.0,
                    ),
                )
            ),
        ),
        init_strength_prior=init_strength_prior,
        init_strength_min=init_strength_min,
        init_strength_max=init_strength_max,
        init_strength_log_mean=float(
            runtime_config.get("pf_init_strength_log_mean", 9.0)
        ),
        init_strength_log_sigma=max(
            0.0,
            float(runtime_config.get("pf_init_strength_log_sigma", 1.0)),
        ),
        cardinality_preserving_resample=bool(
            runtime_config.get("cardinality_preserving_resample", True)
        ),
        cardinality_preserving_min_stations=max(
            0,
            int(runtime_config.get("cardinality_preserving_min_stations", 0)),
        ),
        cardinality_preserving_require_confirmed_structure=bool(
            runtime_config.get(
                "cardinality_preserving_require_confirmed_structure",
                False,
            )
        ),
        mode_preserving_resample=bool(
            runtime_config.get("mode_preserving_resample", True)
        ),
        mode_preserving_max_modes=max(
            0,
            int(runtime_config.get("mode_preserving_max_modes", 6)),
        ),
        mode_preserving_particles_per_mode=max(
            0,
            int(runtime_config.get("mode_preserving_particles_per_mode", 3)),
        ),
        mode_preserving_radius_m=max(
            0.05,
            float(runtime_config.get("mode_preserving_radius_m", 1.5)),
        ),
        mode_preserving_min_weight_fraction=max(
            0.0,
            float(runtime_config.get("mode_preserving_min_weight_fraction", 1e-4)),
        ),
        mode_preserving_surface_strata=bool(
            runtime_config.get("mode_preserving_surface_strata", True)
        ),
        mode_preserving_height_bin_m=max(
            0.0,
            float(runtime_config.get("mode_preserving_height_bin_m", 2.0)),
        ),
        mode_preserving_high_surface_extra_particles=max(
            0,
            int(runtime_config.get("mode_preserving_high_surface_extra_particles", 0)),
        ),
        mode_preserving_high_surface_z_fraction=float(
            np.clip(
                float(
                    runtime_config.get(
                        "mode_preserving_high_surface_z_fraction",
                        0.75,
                    )
                ),
                0.0,
                1.0,
            )
        ),
        mode_preserving_support_score_weight=max(
            0.0,
            float(runtime_config.get("mode_preserving_support_score_weight", 0.0)),
        ),
        mode_preserving_tentative_boost=max(
            1.0,
            float(runtime_config.get("mode_preserving_tentative_boost", 1.0)),
        ),
        mode_preserving_residual_boost=max(
            1.0,
            float(runtime_config.get("mode_preserving_residual_boost", 1.0)),
        ),
        mode_preserving_cardinality_strata=bool(
            runtime_config.get("mode_preserving_cardinality_strata", True)
        ),
        mode_preserving_min_particles_per_cardinality=max(
            0,
            int(runtime_config.get("mode_preserving_min_particles_per_cardinality", 2)),
        ),
        mode_preserving_dynamic_cardinality_allocation=bool(
            runtime_config.get(
                "mode_preserving_dynamic_cardinality_allocation",
                False,
            )
        ),
        mode_preserving_dynamic_cardinality_extra_particles=max(
            0,
            int(
                runtime_config.get(
                    "mode_preserving_dynamic_cardinality_extra_particles",
                    0,
                )
            ),
        ),
        mode_preserving_dynamic_cardinality_min_mass=max(
            0.0,
            float(
                runtime_config.get(
                    "mode_preserving_dynamic_cardinality_min_mass",
                    0.02,
                )
            ),
        ),
        mode_preserving_dynamic_cardinality_entropy_min=max(
            0.0,
            float(
                runtime_config.get(
                    "mode_preserving_dynamic_cardinality_entropy_min",
                    0.5,
                )
            ),
        ),
        mode_preserving_dynamic_spatial_allocation=bool(
            runtime_config.get(
                "mode_preserving_dynamic_spatial_allocation",
                False,
            )
        ),
        mode_preserving_dynamic_spatial_extra_particles=max(
            0,
            int(
                runtime_config.get(
                    "mode_preserving_dynamic_spatial_extra_particles",
                    0,
                )
            ),
        ),
        mode_preserving_dynamic_spatial_min_score_fraction=max(
            0.0,
            float(
                runtime_config.get(
                    "mode_preserving_dynamic_spatial_min_score_fraction",
                    0.005,
                )
            ),
        ),
        deferred_resample_roughening_scale=max(
            0.0,
            float(runtime_config.get("deferred_resample_roughening_scale", 0.15)),
        ),
        pose_min_observation_counts=pose_min_observation_counts_resolved,
        pose_min_observation_penalty_scale=pose_min_observation_penalty_scale,
        pose_min_observation_aggregate=pose_min_observation_aggregate,
        pose_min_observation_max_particles=pose_min_observation_max_particles,
        pose_min_observation_quantile=pose_min_observation_quantile,
        split_prob=max(0.0, float(runtime_config.get("split_prob", 0.05))),
        split_residual_guided=bool(runtime_config.get("split_residual_guided", True)),
        split_complexity_penalty=max(
            0.0,
            float(runtime_config.get("split_complexity_penalty", 0.0)),
        ),
        split_residual_candidate_count=max(
            1,
            int(runtime_config.get("split_residual_candidate_count", 8)),
        ),
        merge_prob=max(0.0, float(runtime_config.get("merge_prob", 0.05))),
        merge_distance_max=max(
            0.0,
            float(runtime_config.get("merge_distance_max", 0.5)),
        ),
        merge_delta_ll_threshold=float(
            runtime_config.get("merge_delta_ll_threshold", 0.0)
        ),
        merge_response_corr_min=float(
            np.clip(
                float(runtime_config.get("merge_response_corr_min", 0.995)),
                0.0,
                1.0,
            )
        ),
        merge_search_topk_pairs=max(
            1,
            int(runtime_config.get("merge_search_topk_pairs", 8)),
        ),
        structural_proposal_topk_particles=structural_proposal_topk_particles,
        structural_trial_workers=structural_trial_workers,
        structural_trial_parallel_min_trials=structural_trial_parallel_min_trials,
        birth_delta_ll_threshold=float(
            runtime_config.get("birth_delta_ll_threshold", 0.0)
        ),
        birth_complexity_penalty=max(
            0.0,
            float(runtime_config.get("birth_complexity_penalty", 0.0)),
        ),
        birth_bic_penalty_params=max(
            0,
            int(runtime_config.get("birth_bic_penalty_params", 4)),
        ),
        birth_max_per_update=(
            None
            if runtime_config.get("birth_max_per_update", None) is None
            else int(runtime_config.get("birth_max_per_update", 0))
        ),
        birth_min_distinct_poses=max(
            1,
            int(runtime_config.get("birth_min_distinct_poses", 1)),
        ),
        birth_residual_min_support=max(
            1,
            int(runtime_config.get("birth_residual_min_support", 2)),
        ),
        birth_residual_support_sigma=max(
            0.0,
            float(runtime_config.get("birth_residual_support_sigma", 1.0)),
        ),
        birth_min_distinct_stations=max(
            1,
            int(runtime_config.get("birth_min_distinct_stations", 1)),
        ),
        source_detector_exclusion_m=max(
            0.0,
            float(runtime_config.get("source_detector_exclusion_m", 0.0)),
        ),
        birth_residual_gate_p_value=float(
            runtime_config.get("birth_residual_gate_p_value", 0.05)
        ),
        birth_candidate_support_fraction=float(
            runtime_config.get("birth_candidate_support_fraction", 0.05)
        ),
        birth_use_shield_coded_residual=bool(
            runtime_config.get("birth_use_shield_coded_residual", True)
        ),
        birth_count_distance_prior_weight=max(
            0.0,
            float(runtime_config.get("birth_count_distance_prior_weight", 0.5)),
        ),
        birth_count_distance_strength_weight=max(
            0.0,
            float(runtime_config.get("birth_count_distance_strength_weight", 0.25)),
        ),
        birth_count_distance_log_clip=max(
            0.0,
            float(runtime_config.get("birth_count_distance_log_clip", 3.0)),
        ),
        birth_count_distance_strength_sigma=max(
            1.0e-12,
            float(runtime_config.get("birth_count_distance_strength_sigma", 2.0)),
        ),
        birth_residual_expand_structural_particles=bool(
            runtime_config.get("birth_residual_expand_structural_particles", True)
        ),
        birth_residual_expanded_structural_topk_particles=(
            None
            if runtime_config.get(
                "birth_residual_expanded_structural_topk_particles",
                256,
            )
            is None
            else max(
                1,
                int(
                    runtime_config.get(
                        "birth_residual_expanded_structural_topk_particles",
                        256,
                    )
                ),
            )
        ),
        birth_matching_pursuit_max_new_sources=max(
            1,
            int(runtime_config.get("birth_matching_pursuit_max_new_sources", 3)),
        ),
        birth_matching_pursuit_topk_candidates=max(
            1,
            int(runtime_config.get("birth_matching_pursuit_topk_candidates", 16)),
        ),
        birth_q_max=max(
            0.0,
            float(
                runtime_config.get(
                    "birth_q_max",
                    init_strength_max
                    if init_strength_prior in {"uniform", "log_uniform"}
                    and init_strength_max is not None
                    else 5.0e6,
                )
            ),
        ),
        birth_q_min=max(
            0.0,
            float(
                runtime_config.get(
                    "birth_q_min",
                    init_strength_min
                    if init_strength_prior in {"uniform", "log_uniform"}
                    else 1.0e2,
                )
            ),
        ),
        birth_orthogonalize_residual_candidates=bool(
            runtime_config.get("birth_orthogonalize_residual_candidates", False)
        ),
        birth_orthogonal_candidate_corr_max=float(
            np.clip(
                float(runtime_config.get("birth_orthogonal_candidate_corr_max", 0.98)),
                0.0,
                1.0,
            )
        ),
        birth_jitter_topk_candidates=birth_jitter_topk_candidates,
        residual_decomposition_enable=bool(
            runtime_config.get("residual_decomposition_enable", True)
        ),
        peak_suppression_enable=bool(
            runtime_config.get("peak_suppression_enable", True)
        ),
        peak_suppression_min_source_fraction=float(
            np.clip(
                float(runtime_config.get("peak_suppression_min_source_fraction", 0.25)),
                0.0,
                1.0,
            )
        ),
        peak_suppression_factor=float(
            np.clip(float(runtime_config.get("peak_suppression_factor", 1.0)), 0.0, 1.0)
        ),
        residual_decomposition_max_layers=max(
            1,
            int(runtime_config.get("residual_decomposition_max_layers", 4)),
        ),
        pseudo_source_verification_enable=bool(
            runtime_config.get("pseudo_source_verification_enable", True)
        ),
        pseudo_source_min_delta_ll=float(
            runtime_config.get("pseudo_source_min_delta_ll", 0.0)
        ),
        pseudo_source_min_distinct_views=max(
            1,
            int(runtime_config.get("pseudo_source_min_distinct_views", 2)),
        ),
        pseudo_source_fail_grace_stations=max(
            0,
            int(runtime_config.get("pseudo_source_fail_grace_stations", 2)),
        ),
        pseudo_source_corr_max=float(
            np.clip(
                float(runtime_config.get("pseudo_source_corr_max", 0.995)), 0.0, 1.0
            )
        ),
        pseudo_source_temporal_sep_min=max(
            0.0,
            float(runtime_config.get("pseudo_source_temporal_sep_min", 0.0)),
        ),
        pseudo_source_quarantine_on_suppress=bool(
            runtime_config.get("pseudo_source_quarantine_on_suppress", True)
        ),
        source_prune_min_distinct_stations=max(
            1,
            int(runtime_config.get("source_prune_min_distinct_stations", 2)),
        ),
        source_prune_min_distinct_views=max(
            1,
            int(runtime_config.get("source_prune_min_distinct_views", 2)),
        ),
        source_prune_fail_grace_stations=max(
            1,
            int(runtime_config.get("source_prune_fail_grace_stations", 2)),
        ),
        source_prune_delta_ll_threshold=float(
            runtime_config.get("source_prune_delta_ll_threshold", 0.0)
        ),
        source_prune_bic_penalty_params=max(
            0,
            int(runtime_config.get("source_prune_bic_penalty_params", 4)),
        ),
        birth_stage_single_station_as_quarantine=bool(
            runtime_config.get("birth_stage_single_station_as_quarantine", True)
        ),
        history_estimate_interval=max(
            0,
            int(runtime_config.get("history_estimate_interval", 1)),
        ),
        candidate_response_cache_max_entries=max(
            0,
            int(runtime_config.get("candidate_response_cache_max_entries", 24)),
        ),
        orientation_k=orientation_limit_resolved,
        min_rotations_per_pose=min_rotations_resolved,
        planning_eig_samples=int(runtime_config.get("planning_eig_samples", 50)),
        planning_rollout_particles=(
            None
            if runtime_config.get("planning_rollout_particles", 512) is None
            else int(runtime_config.get("planning_rollout_particles", 512))
        ),
        planning_rollout_method=str(
            runtime_config.get("planning_rollout_method", "resample")
        ),
        use_fast_gpu_rollout=True,
        use_gpu=use_gpu,
        gpu_device="cuda",
        gpu_dtype="float64",
        ig_workers=_resolve_ig_workers(
            runtime_config.get("ig_workers", python_worker_count_resolved)
        ),
        parallel_isotope_updates=bool(
            runtime_config.get("parallel_isotope_updates", True)
        ),
        parallel_isotope_workers=parallel_isotope_workers,
    )
    pf_conf.use_tempering = True
    pf_conf.max_temper_steps = 8
    pf_conf.min_delta_beta = 0.01
    pf_conf.target_ess_ratio = 0.4
    pf_conf.converge_enable = bool(
        converge
        or runtime_config.get("converge_enable", False)
        or adaptive_mission_stop
    )
    pf_conf.converge_freeze_updates = bool(
        runtime_config.get("converge_freeze_updates", False)
    )
    pf_conf.converge_cardinality_var_max = max(
        0.0,
        float(runtime_config.get("converge_cardinality_var_max", 0.05)),
    )
    pf_conf.converge_require_no_tentative = bool(
        runtime_config.get("converge_require_no_tentative", True)
    )
    pf_conf.converge_cluster_spread_max_m = max(
        0.0,
        float(runtime_config.get("converge_cluster_spread_max_m", 0.0)),
    )
    pf_conf.converge_cluster_min_support_fraction = float(
        np.clip(
            float(runtime_config.get("converge_cluster_min_support_fraction", 0.0)),
            0.0,
            1.0,
        )
    )
    if pf_config_overrides:
        for key, value in pf_config_overrides.items():
            if not hasattr(pf_conf, key):
                raise ValueError(f"Unknown PF config override: {key}")
            setattr(pf_conf, key, value)
    pf_conf.birth_enable = bool(birth_enabled)
    if birth_enabled:
        if not pf_config_overrides or "p_birth" not in pf_config_overrides:
            if pf_conf.p_birth <= 0.0:
                pf_conf.p_birth = 0.05
        if not pf_config_overrides or "p_kill" not in pf_config_overrides:
            if pf_conf.p_kill <= 0.0:
                pf_conf.p_kill = 0.1
        if not pf_config_overrides or "split_prob" not in pf_config_overrides:
            if pf_conf.split_prob <= 0.0:
                pf_conf.split_prob = 0.05
        if not pf_config_overrides or "merge_prob" not in pf_config_overrides:
            if pf_conf.merge_prob <= 0.0:
                pf_conf.merge_prob = 0.05
    if not birth_enabled:
        pf_conf.p_birth = 0.0
        pf_conf.p_kill = 0.0
        pf_conf.split_prob = 0.0
        pf_conf.merge_prob = 0.0
        pf_conf.max_sources = 1
        if not pf_config_overrides or "init_num_sources" not in pf_config_overrides:
            pf_conf.init_num_sources = (1, 1)
    if ig_threshold_min is not None:
        pf_conf.ig_threshold = float(ig_threshold_min)
    strict_planned_shield_program = bool(
        runtime_config.get(
            "strict_planned_shield_program",
            path_planner_resolved == "dss_pp",
        )
    )
    baseline_shield_policy = runtime_config.get("baseline_shield_policy")
    baseline_path_policy = runtime_config.get("baseline_path_policy")
    shield_signature_weight = max(
        0.0,
        float(runtime_config.get("shield_signature_weight", 0.25)),
    )
    shield_low_count_penalty_weight = max(
        0.0,
        float(runtime_config.get("shield_low_count_penalty_weight", 1.0)),
    )
    shield_count_balance_weight = max(
        0.0,
        float(runtime_config.get("shield_count_balance_weight", 0.25)),
    )
    shield_rotation_cost_weight = max(
        0.0,
        float(runtime_config.get("shield_rotation_cost_weight", 0.05)),
    )
    shield_signature_variance_floor = max(
        1e-12,
        float(runtime_config.get("shield_signature_variance_floor", 1.0)),
    )
    shield_stop_min_gain = max(
        0.0,
        float(runtime_config.get("shield_stop_min_gain", 0.0)),
    )
    shield_stop_compare_next_pose = bool(
        runtime_config.get("shield_stop_compare_next_pose", True)
    )
    shield_stop_pose_candidates = max(
        1,
        int(
            runtime_config.get("shield_stop_pose_candidates", min(pose_candidates, 16))
        ),
    )
    shield_stop_rate_margin = max(
        0.0,
        float(runtime_config.get("shield_stop_rate_margin", 1.0)),
    )
    shield_stop_signature_cosine = float(
        runtime_config.get("shield_stop_signature_cosine", 0.995)
    )
    shield_selection_max_particles_raw = runtime_config.get(
        "shield_selection_max_particles",
        None,
    )
    if shield_selection_max_particles_raw is None:
        shield_selection_max_particles = (
            pf_conf.pose_min_observation_max_particles
            if pf_conf.pose_min_observation_max_particles is not None
            else (
                pf_conf.planning_rollout_particles or pf_conf.planning_particles or 256
            )
        )
    else:
        shield_selection_max_particles = int(shield_selection_max_particles_raw)
    if shield_selection_max_particles is not None:
        shield_selection_max_particles = max(1, int(shield_selection_max_particles))
    init_support_prob = (
        None
        if str(pf_conf.source_position_prior).strip().lower() == "surface"
        else _initial_particle_nearby_probability(
            num_particles=int(pf_conf.num_particles),
            position_min=pf_conf.position_min,
            position_max=pf_conf.position_max,
            radius_m=float(eval_match_radius_m),
            init_num_sources=pf_conf.init_num_sources,
        )
    )

    # Build true sources dict for visualization
    true_src = {}
    true_strengths = {}
    for iso in isotopes:
        positions = [
            np.array(src.position, dtype=float) for src in sources if src.isotope == iso
        ]
        strengths = [src.intensity_cps_1m for src in sources if src.isotope == iso]
        if positions:
            true_src[iso] = np.vstack(positions)
        if strengths:
            true_strengths[iso] = [float(val) for val in strengths]
    estimate_mode = "mmse"
    estimate_min_strength = 100.0
    estimate_min_existence_prob = None
    pf_detector_radius_m = detector_geometry.count_radius_m
    pf_detector_aperture_radius_m = detector_geometry.aperture_radius_m
    pf_detector_aperture_samples = detector_geometry.aperture_samples
    pf_detector_aperture_sampling = detector_geometry.aperture_sampling
    pf_source_extent_radius_m = observation_model.source_extent_radius_m
    pf_source_extent_samples = observation_model.source_extent_samples

    # This is the sole effective configuration hashed into live PF provenance,
    # the MeasurementLog, and the forward-model manifest.  Apply the pure
    # capability boundary before serializing it so hostile API overrides cannot
    # survive in provenance while being disabled later by the estimator.
    apply_profile_to_config(pf_conf)
    measurement_log_runtime_config = _build_effective_live_runtime_config(
        measurement_log_runtime_config,
        pf_config=pf_conf,
        candidate_sources_xyz=np.asarray(grid, dtype=np.float64),
        source_position_bounds=(
            np.asarray(source_position_min, dtype=np.float64),
            np.asarray(source_position_max, dtype=np.float64),
        ),
        api_settings={
            "max_steps": max_steps,
            "max_poses": max_poses,
            "birth_enabled": bool(birth_enabled),
            "num_particles": int(num_particles),
            "candidate_grid_spacing_m": list(spacing),
            "candidate_grid_margin_m": float(candidate_grid_margin),
            "source_surface_prior": bool(source_surface_prior),
            "obstacle_height_m": float(runtime_config.get("obstacle_height_m", 2.0)),
            "pose_candidates": int(pose_candidates),
            "pose_min_dist_m": float(pose_min_dist),
            "path_planner": str(path_planner_resolved),
            "dss_pp": json_safe(dss_config),
            "measurement_time_s": float(live_time),
            "adaptive_dwell": bool(adaptive_dwell),
            "adaptive_dwell_chunk_s": float(adaptive_dwell_chunk_s),
            "adaptive_min_dwell_s": float(adaptive_min_dwell_s),
            "adaptive_ready_min_counts": float(adaptive_ready_min_counts),
            "adaptive_ready_min_isotopes": int(adaptive_ready_min_isotopes),
            "adaptive_ready_min_snr": float(adaptive_ready_min_snr),
            "nominal_motion_speed_m_s": float(nominal_motion_speed_m_s),
            "rotation_overhead_s": float(rotation_overhead_s),
            "joint_observation_update": bool(joint_observation_update),
            "delayed_resample_update": bool(delayed_resample_update),
            "pf_random_seed": int(pf_random_seed),
            "sim_backend": str(sim_backend),
            "environment_mode": str(normalized_environment_mode),
        },
    )
    measurement_log_config_hash = sha256_json(measurement_log_runtime_config)

    def _build_estimator() -> tuple[
        RotatingShieldPFEstimator, NDArray[np.float64], int
    ]:
        """Create a fresh estimator and register the initial pose."""
        estimator_local = RotatingShieldPFEstimator(
            isotopes=isotopes,
            candidate_sources=grid,
            shield_normals=normals,
            mu_by_isotope=mu_by_isotope,
            pf_config=pf_conf,
            obstacle_grid=pf_obstacle_grid,
            obstacle_height_m=float(runtime_config.get("obstacle_height_m", 2.0)),
            obstacle_mu_by_isotope=obstacle_mu_by_isotope,
            obstacle_buildup_coeff=pf_obstacle_buildup_coeff,
            detector_radius_m=pf_detector_radius_m,
            detector_aperture_radius_m=pf_detector_aperture_radius_m,
            detector_aperture_samples=pf_detector_aperture_samples,
            detector_aperture_sampling=pf_detector_aperture_sampling,
            source_extent_radius_m=pf_source_extent_radius_m,
            source_extent_samples=pf_source_extent_samples,
            line_mu_by_isotope=line_mu_by_isotope,
            transport_response_model=transport_response_model,
            config_hash=input_config_hash,
            resolved_config_hash=measurement_log_config_hash,
            random_seed=pf_random_seed,
        )
        pose_local = np.array(env.detector_position, dtype=float)
        estimator_local.add_measurement_pose(pose_local)
        if pf_detected_isotopes_only:
            estimator_local.restrict_isotopes([], allow_empty=True)
        pose_idx_local = len(estimator_local.poses) - 1
        return estimator_local, pose_local, pose_idx_local

    def _build_visualizer() -> object:
        """Create a new PF visualizer."""
        visualizer_args = {
            "isotopes": isotopes,
            "world_bounds": (0, env.size_x, 0, env.size_y, 0, env.size_z),
            "true_sources": true_src,
            "true_strengths": true_strengths,
            "obstacle_grid": obstacle_grid,
            "show_counts": False,
        }
        if not live and bool(runtime_config.get("headless_visualizer_defer", True)):
            return DeferredPFVisualizer(RealTimePFVisualizer, **visualizer_args)
        return RealTimePFVisualizer(
            **visualizer_args,
        )

    def _build_cui_split_visualizer() -> (
        CUISplitPFVisualizer | AsyncCUISplitPFVisualizer | None
    ):
        """Create the CUI split visualizer when enabled."""
        if not cui_split_view_enabled:
            return None
        split_cls = (
            AsyncCUISplitPFVisualizer
            if bool(runtime_config.get("cui_split_view_async", True))
            else CUISplitPFVisualizer
        )
        split_viz = split_cls(
            isotopes=isotopes,
            output_dir=cui_split_view_dir,
            world_bounds=(0, env.size_x, 0, env.size_y, 0, env.size_z),
            true_sources=true_src,
            true_strengths=true_strengths,
            obstacle_grid=obstacle_grid,
            max_particles_per_isotope=cui_split_max_particles,
        )
        serve_cui = bool(runtime_config.get("cui_split_view_serve", True))
        split_url = None
        if serve_cui:
            split_url = _ensure_cui_view_server(
                split_viz.output_dir,
                host=str(runtime_config.get("cui_split_view_host", "0.0.0.0")),
                port=int(runtime_config.get("cui_split_view_port", 8877)),
                public_host=(
                    None
                    if runtime_config.get("cui_split_view_public_host") is None
                    else str(runtime_config.get("cui_split_view_public_host"))
                ),
            )
        print(
            "CUI split visualization enabled: "
            f"{split_viz.index_path.as_posix()} "
            "(latest_robot_2d.png, latest_pf_3d.png)"
        )
        if isinstance(split_viz, AsyncCUISplitPFVisualizer):
            print("CUI split visualization rendering: async process")
        if split_url is not None:
            print(f"CUI split visualization URL: {split_url}")
        return split_viz

    def _grid_centers() -> NDArray[np.float64]:
        """Return 1m grid-center positions for the environment bounds."""
        spacing = 1.0
        xs = np.arange(0.5, env.size_x, spacing)
        ys = np.arange(0.5, env.size_y, spacing)
        zs = np.arange(0.5, env.size_z, spacing)
        grid_pos = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
        return grid_pos.reshape(-1, 3)

    def _build_pf_posterior_estimates(
        estimator_final: RotatingShieldPFEstimator,
        isotope_list: list[str],
    ) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Project the current sequential PF posterior without a batch report stage."""
        estimates = _pure_pf_primary_estimates(estimator_final, isotope_list)
        if estimates is None:
            raise RuntimeError("The realtime estimator must use the strict PF profile.")
        return estimates

    def _serialize_estimate_stage(
        estimates_in: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    ) -> dict[str, list[dict[str, float | list[float]]]]:
        """Return JSON-serializable source estimates for one reporting stage."""
        serialized: dict[str, list[dict[str, float | list[float]]]] = {}
        for iso, estimate in sorted(estimates_in.items()):
            positions = np.asarray(estimate[0], dtype=float).reshape(-1, 3)
            strengths = np.asarray(estimate[1], dtype=float).reshape(-1)
            entries: list[dict[str, float | list[float]]] = []
            for pos, strength in zip(positions, strengths):
                entries.append(
                    {
                        "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                        "strength": float(strength),
                    }
                )
            serialized[iso] = entries
        return serialized

    if live:
        plt.ion()
        plt.show(block=False)
        plt.pause(0.1)
        preview_pose = np.array(env.detector_position, dtype=float)
        preview_viz = RealTimePFVisualizer(
            isotopes=["Cs-137"],
            world_bounds=(0, env.size_x, 0, env.size_y, 0, env.size_z),
            true_sources={},
            true_strengths={},
            obstacle_grid=obstacle_grid,
            show_counts=False,
        )
        grid_pos = _grid_centers()
        preview_frame = PFFrame(
            step_index=-1,
            time=0.0,
            robot_position=preview_pose,
            robot_orientation=None,
            RFe=np.eye(3),
            RPb=np.eye(3),
            duration=0.0,
            counts_by_isotope={},
            particle_positions={"Cs-137": grid_pos},
            particle_weights={"Cs-137": np.ones(grid_pos.shape[0], dtype=float)},
            estimated_sources={"Cs-137": np.zeros((0, 3), dtype=float)},
            estimated_strengths={"Cs-137": np.zeros(0, dtype=float)},
        )
        preview_viz.update(preview_frame)
        preview_viz.fig.canvas.draw()
        if hasattr(preview_viz.fig.canvas, "flush_events"):
            preview_viz.fig.canvas.flush_events()
        plt.pause(5.0)
        plt.close(preview_viz.fig)

    estimator: RotatingShieldPFEstimator
    current_pose: NDArray[np.float64]
    current_pose_idx: int
    measurement_log_writer: MeasurementLogStreamWriter | None = None
    resume_controller_state: _LiveResumeControllerState | None = None
    resume_controller_checkpoint: _LiveControllerCheckpoint | None = None
    resume_remaining_measurement_estimates: list[dict[str, Any]] = []
    resume_replay_wall_s = 0.0
    planning_candidate_checkpoint_parameters = (
        _planning_candidate_checkpoint_parameters(
            pose_candidates=int(pose_candidates),
            pose_min_dist=float(pose_min_dist),
            bounds_xyz=(bounds_lo, bounds_hi),
            detector_heights_m=detector_height_candidates,
            continuous_height_anchor_count=(
                detector_continuous_height_partner_candidates
            ),
            height_partner_xy_tolerance_m=(
                detector_height_pair_xy_tolerance_m
            ),
            height_partner_z_tolerance_m=(
                detector_height_pair_z_tolerance_m
            ),
            height_partner_min_z_separation_m=(
                detector_height_pair_min_separation_m
            ),
        )
    )
    if measurement_log_target is not None:
        environment_payload: dict[str, Any] = {
            "environment_model_id": str(
                runtime_config.get(
                    "environment_model_id",
                    f"{normalized_environment_mode}_environment.v1",
                )
            ),
            "size_x": float(env.size_x),
            "size_y": float(env.size_y),
            "size_z": float(env.size_z),
            "detector_position": [float(value) for value in env.detector_position],
            "environment_mode": str(normalized_environment_mode),
            "obstacle_grid": (
                None if obstacle_grid is None else obstacle_grid.to_dict()
            ),
        }
        execution_commit = repository_commit()
        if not _full_git_commit(execution_commit):
            raise RuntimeError(
                "Pure live acquisition requires an available full Git commit."
            )
        compatibility: dict[str, Any] | None = None
        if resume_measurement_stage is None:
            commit = execution_commit
        else:
            commit = _resume_stage_repository_commit(resume_measurement_stage)
            compatibility = _build_resume_compatibility_provenance(
                repository_root=ROOT,
                prefix_commit=commit,
                execution_commit=execution_commit,
                additional_compatible_code_paths=resume_compatible_code_paths,
                compatibility_basis=resume_compatibility_basis,
            )
            if resume_runtime_log is not None:
                compatibility["controller_history_log_sha256"] = _sha256_path(
                    resume_runtime_log
                )
        forward_manifest = build_forward_model_manifest(
            runtime_config=measurement_log_runtime_config,
            environment=environment_payload,
            obstacle_layout_path=measurement_log_obstacle_layout_path,
            isotopes=isotopes,
            repository_commit=commit,
            resolved_config_sha256=measurement_log_config_hash,
            run_root=measurement_log_target,
            repository_root=ROOT,
        )
        log_run_id = str(
            runtime_config.get(
                "measurement_log_run_id",
                measurement_log_target.name,
            )
        )
        writer_arguments = {
            "run_id": log_run_id,
            "repository_commit": commit,
            "runtime_config": measurement_log_runtime_config,
            "environment": environment_payload,
            "forward_model_manifest": forward_manifest,
            "isotopes": isotopes,
            "obstacle_layout_path": measurement_log_obstacle_layout_path,
        }
        if resume_measurement_stage is None:
            estimator, current_pose, current_pose_idx = _build_estimator()
            measurement_log_writer = MeasurementLogStreamWriter(
                measurement_log_target,
                metadata={"acquisition": "live_append_before_pf_update"},
                **writer_arguments,
            )
        else:
            if pf_detected_isotopes_only:
                raise RuntimeError(
                    "Exact station-boundary resume currently requires "
                    "pf_detected_isotopes_only=false."
                )
            assert compatibility is not None
            measurement_log_writer = MeasurementLogStreamWriter.resume_from_stage(
                measurement_log_target,
                stage_dir=resume_measurement_stage,
                metadata={"acquisition": "live_station_boundary_resume"},
                resume_execution_commit=execution_commit,
                resume_compatibility=compatibility,
                **writer_arguments,
            )
            replay_start = time.perf_counter()
            with tempfile.TemporaryDirectory(
                prefix=f".{measurement_log_target.name}.resume-prefix-",
                dir=measurement_log_target.parent,
            ) as temporary_root:
                prefix_path = Path(temporary_root) / "measurement-log"
                prefix_log = measurement_log_writer.write_canonical_prefix(
                    prefix_path
                )
                estimator = _build_resume_replay_estimator(
                    prefix_log,
                    profile=str(pf_conf.estimator_profile),
                    seed=int(pf_random_seed),
                    config_hash=input_config_hash,
                    resolved_config_hash=measurement_log_config_hash,
                )

                def _report_replayed_station(
                    replay_estimator: RotatingShieldPFEstimator,
                    record: MeasurementLogRecord,
                    record_index: int,
                ) -> None:
                    """Report durable replay progress only at station boundaries."""
                    del replay_estimator
                    print(
                        "Resume PF replay completed station "
                        f"{int(record.station_id)} at record {record_index + 1}/"
                        f"{len(prefix_log.records)}.",
                        flush=True,
                    )

                replay_trace = replay_records(
                    prefix_log,
                    estimator,
                    station_complete_callback=_report_replayed_station,
                )
            resume_replay_wall_s = float(time.perf_counter() - replay_start)
            if len(replay_trace) != len(measurement_log_writer.records):
                raise RuntimeError(
                    "Pure PF replay did not consume the complete staged prefix."
                )
            if len(estimator.measurements) != len(measurement_log_writer.records):
                raise RuntimeError(
                    "Pure PF replay measurement count differs from the staged prefix."
                )
            if not _pure_pf_profile_active(estimator):
                raise RuntimeError("Live resume requires the strict pure-PF profile.")
            resume_controller_state = _reconstruct_resume_controller_state(
                records=measurement_log_writer.records,
                estimator=estimator,
                isotopes=isotopes,
                nominal_motion_speed_m_s=nominal_motion_speed_m_s,
                expected_program_length=int(estimator.pf_config.orientation_k),
            )
            current_pose = resume_controller_state.current_pose.copy()
            current_pose_idx = int(resume_controller_state.current_pose_idx)
            resume_controller_checkpoint = _restore_live_controller_checkpoint(
                record=measurement_log_writer.records[-1],
                planning_candidate_rng=planning_candidate_rng,
                expected_planning_candidate_parameters=(
                    planning_candidate_checkpoint_parameters
                ),
            )
            if resume_controller_checkpoint is None:
                if resume_runtime_log is None:
                    raise RuntimeError(
                        "A legacy stage without a controller checkpoint requires "
                        "its original runtime log for exact candidate-RNG replay."
                    )
                _advance_resume_planning_candidate_rng(
                    rng=planning_candidate_rng,
                    records=measurement_log_writer.records,
                    map_api=planning_map,
                    n_candidates=int(pose_candidates),
                    min_dist_from_visited=float(pose_min_dist),
                    bounds_xyz=(bounds_lo, bounds_hi),
                    detector_heights_m=detector_height_candidates,
                    continuous_height_anchor_count=(
                        detector_continuous_height_partner_candidates
                    ),
                    height_partner_xy_tolerance_m=(
                        detector_height_pair_xy_tolerance_m
                    ),
                    height_partner_z_tolerance_m=(
                        detector_height_pair_z_tolerance_m
                    ),
                    height_partner_min_z_separation_m=(
                        detector_height_pair_min_separation_m
                    ),
                    expected_candidate_counts=(
                        _parse_candidate_generation_counts(resume_runtime_log)
                    ),
                )
                if bool(remaining_measurement_config.enabled):
                    resume_remaining_measurement_estimates = (
                        _restore_remaining_measurement_history(
                            estimator=estimator,
                            records=measurement_log_writer.records,
                            runtime_log_path=resume_runtime_log,
                            config=remaining_measurement_config,
                        )
                    )
            else:
                resume_remaining_measurement_estimates = [
                    dict(value)
                    for value in (
                        resume_controller_checkpoint.remaining_measurement_estimates
                    )
                ]
                setattr(
                    estimator,
                    "_remaining_measurement_budget_history",
                    [
                        dict(value)
                        for value in (
                            resume_controller_checkpoint
                            .remaining_measurement_budget_history
                        )
                    ],
                )
                setattr(
                    estimator,
                    "_remaining_measurement_eta_ratios",
                    list(
                        resume_controller_checkpoint
                        .remaining_measurement_eta_ratios
                    ),
                )
            print(
                "Station-boundary resume restored "
                f"records={resume_controller_state.step_counter} "
                f"stations={resume_controller_state.pose_counter + 1} "
                f"next_step={resume_controller_state.step_counter} "
                "controller_state="
                f"{'checkpoint' if resume_controller_checkpoint else 'replayed'} "
                f"replay_wall_s={resume_replay_wall_s:.3f}.",
                flush=True,
            )
    else:
        estimator, current_pose, current_pose_idx = _build_estimator()
    if resume_controller_checkpoint is not None:
        max_poses = resume_controller_checkpoint.max_poses
    viz = _build_visualizer()
    cui_split_viz = _build_cui_split_visualizer()
    if live:
        plt.ion()
        plt.show(block=False)
        plt.pause(0.1)

    elapsed = (
        0.0
        if resume_controller_state is None
        else float(resume_controller_state.elapsed_s)
    )
    last_frame: PFFrame | None = None
    step_counter = (
        0
        if resume_controller_state is None
        else int(resume_controller_state.step_counter)
    )
    total_pairs = num_orients * num_orients
    visited_poses: list[NDArray[np.float64]] = (
        []
        if resume_controller_state is None
        else [pose.copy() for pose in resume_controller_state.visited_poses]
    )
    last_spectrum: np.ndarray | None = (
        None
        if resume_controller_state is None
        else resume_controller_state.last_spectrum.copy()
    )
    last_spectrum_components: dict[str, NDArray[np.float64]] = {}
    last_counts: dict[str, float] | None = (
        None
        if resume_controller_state is None
        else dict(resume_controller_state.last_counts)
    )
    last_measurement_for_diagnostics: Measurement | None = None
    representative_spectrum: np.ndarray | None = (
        None
        if resume_controller_state is None
        else resume_controller_state.representative_spectrum.copy()
    )
    representative_spectrum_components: dict[str, NDArray[np.float64]] = {}
    representative_counts: dict[str, float] | None = (
        None
        if resume_controller_state is None
        else dict(resume_controller_state.representative_counts)
    )
    representative_candidates: set[str] = set()
    representative_step_index: int | None = (
        None
        if resume_controller_state is None
        else int(resume_controller_state.representative_step_index)
    )
    representative_total_counts = (
        -np.inf
        if representative_spectrum is None
        else float(np.sum(representative_spectrum))
    )
    last_max_ig: float | None = (
        None
        if resume_controller_checkpoint is None
        else resume_controller_checkpoint.last_max_ig
    )
    total_motion_distance_m = (
        0.0
        if resume_controller_state is None
        else float(resume_controller_state.total_motion_distance_m)
    )
    total_motion_time_s = (
        0.0
        if resume_controller_state is None
        else float(resume_controller_state.total_motion_time_s)
    )
    total_rotation_time_s = (
        0.0
        if resume_controller_state is None
        else float(resume_controller_state.total_rotation_time_s)
    )
    pending_motion_distance_m = 0.0
    pending_motion_time_s = 0.0
    pending_path_segment: dict[str, object] | None = None
    path_segments: list[dict[str, object]] = []
    remaining_measurement_estimates: list[dict[str, Any]] = list(
        resume_remaining_measurement_estimates
    )
    soft_pose_extension_used = (
        0
        if resume_controller_checkpoint is None
        else int(resume_controller_checkpoint.soft_pose_extension_used)
    )
    max_pose_stop_unresolved = False
    max_pose_stop_diagnostics: dict[str, object] = {}
    measurement_live_times_s: list[float] = (
        []
        if resume_controller_state is None
        else list(resume_controller_state.measurement_live_times_s)
    )
    total_ig_wall_s = 0.0
    total_pf_wall_s = 0.0
    total_viz_wall_s = 0.0
    total_path_planning_wall_s = 0.0
    ig_wall_samples_s: list[float] = []
    pf_wall_samples_s: list[float] = []
    path_planning_wall_samples_s: list[float] = []
    if resume_controller_state is not None:
        assert measurement_log_writer is not None
        resumed_stations = _records_by_station(measurement_log_writer.records)
        for station_index in range(len(resumed_stations) - 1):
            next_station = resumed_stations[station_index + 1]
            planned_pairs = tuple(
                int(record.fe_orientation_index) * num_orients
                + int(record.pb_orientation_index)
                for record in next_station
            )
            segment = _build_robot_path_segment(
                map_api=planning_map,
                from_pose_xyz=np.asarray(
                    resumed_stations[station_index][0].detector_pose_xyz,
                    dtype=float,
                ),
                to_pose_xyz=np.asarray(
                    next_station[0].detector_pose_xyz,
                    dtype=float,
                ),
                nominal_motion_speed_m_s=nominal_motion_speed_m_s,
                path_planner=path_planner_resolved,
                planned_shield_program=planned_pairs,
                dss_diagnostics=None,
            )
            logged_travel = float(next_station[0].travel_time_s)
            if not np.isclose(
                float(segment["travel_time_s"]),
                logged_travel,
                rtol=0.0,
                atol=1.0e-9,
            ):
                raise RuntimeError(
                    "Reconstructed robot route time does not match the staged "
                    f"transition into station {station_index + 1}."
                )
            path_segments.append(segment)
    gpu_runtime_enabled = False
    if bool(pf_conf.use_gpu):
        try:
            gpu_runtime_enabled = bool(estimator._gpu_enabled())
        except RuntimeError:
            gpu_runtime_enabled = False
    gpu_memory_baseline = start_gpu_memory_tracking(
        str(pf_conf.gpu_device) if gpu_runtime_enabled else None
    )
    run_wall_start = time.perf_counter()

    def _select_one_step_pose_for_planning(
        *,
        candidate_poses_xyz: NDArray[np.float64],
        current_pose_xyz: NDArray[np.float64],
        program_length_budget: int,
        use_gpu: bool | None,
    ) -> tuple[int, float]:
        """Select a greedy one-step pose and record planning wall time."""
        nonlocal total_path_planning_wall_s
        one_step_start = time.perf_counter()
        next_idx_local = select_next_pose_from_candidates(
            estimator=estimator,
            candidate_poses_xyz=candidate_poses_xyz,
            current_pose_xyz=current_pose_xyz,
            criterion="after_rotation",
            t_max_s=float(max(1, int(program_length_budget)))
            * float(planning_live_time),
            verbose=True,
            progress_every=0,
            auto_lambda_cost=True,
            num_rollouts=DEFAULT_PLANNING_ROLLOUTS,
            min_observation_counts=float(pf_conf.pose_min_observation_counts),
            min_observation_penalty_scale=float(
                pf_conf.pose_min_observation_penalty_scale
            ),
            min_observation_aggregate=pf_conf.pose_min_observation_aggregate,
            min_observation_max_particles=pf_conf.pose_min_observation_max_particles,
            worker_count=int(pose_selection_workers_resolved),
            use_gpu=use_gpu,
        )
        one_step_elapsed = time.perf_counter() - one_step_start
        total_path_planning_wall_s += float(one_step_elapsed)
        path_planning_wall_samples_s.append(float(one_step_elapsed))
        return int(next_idx_local), float(one_step_elapsed)

    def _latest_remaining_estimate_for_planning(
        preferred: Mapping[str, Any] | object | None = None,
    ) -> Mapping[str, Any] | object | None:
        """Return the best available remaining-measurement estimate for planning."""
        if preferred is not None:
            return preferred
        if (
            bool(remaining_measurement_config.enabled)
            and remaining_measurement_estimates
        ):
            return remaining_measurement_estimates[-1]
        return None

    def _pf_priority_isotopes_from_remaining(
        estimate: Mapping[str, Any] | object | None,
        *,
        max_isotopes: int = 2,
    ) -> tuple[str, ...]:
        """Return isotopes prioritized by PF posterior uncertainty and residuals."""
        payload = _remaining_measurement_payload(estimate)
        details = payload.get("isotope_details", {})
        if not isinstance(details, Mapping):
            return tuple()
        ranked: list[tuple[float, str]] = []
        for isotope, detail_raw in details.items():
            if not isinstance(detail_raw, Mapping):
                continue
            detail = dict(detail_raw)

            def _detail_float(key: str) -> float:
                """Return a numeric isotope-detail value with a zero fallback."""
                try:
                    return float(detail.get(key, 0.0))
                except (TypeError, ValueError):
                    return 0.0

            score = _detail_float("cardinality_entropy")
            score += min(
                _detail_float("posterior_predictive_positive_chi2") / 100.0,
                10.0,
            )
            score += float(int(_detail_float("unresolved_pair_count") > 0.0))
            score += float(
                int(_detail_float("high_surface_unresolved_pair_count") > 0.0)
            )
            score += _detail_float("verification_budget")
            score += _detail_float("unresolved_absent_budget")
            if score > 0.0:
                ranked.append((float(score), str(isotope)))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return tuple(isotope for _score, isotope in ranked[: max(0, int(max_isotopes))])

    def _pf_planner_mode_from_remaining(
        estimate: Mapping[str, Any] | object | None,
    ) -> tuple[str, dict[str, float]]:
        """Return a planner mode from PF posterior budget components."""
        payload = _remaining_measurement_payload(estimate)
        components = payload.get("components", {})
        totals = {
            "residual": 0.0,
            "absent": 0.0,
            "verification": 0.0,
            "pairwise": 0.0,
            "high_surface": 0.0,
            "cardinality": 0.0,
            "uncertainty": 0.0,
        }
        if isinstance(components, Mapping):

            def _component(key: str) -> float:
                """Read a finite PF posterior budget component."""
                try:
                    value = float(components.get(key, 0.0))
                except (TypeError, ValueError):
                    return 0.0
                return value if np.isfinite(value) else 0.0

            totals["residual"] = _component("residual")
            totals["absent"] = _component("isotope_absence")
            totals["verification"] = _component("pseudo_source_verification")
            totals["pairwise"] = _component("same_isotope_separation")
            totals["high_surface"] = _component("high_surface_ambiguity")
            totals["cardinality"] = _component("cardinality")
            totals["uncertainty"] = _component("uncertainty")
        if totals["absent"] > 0.0 or totals["residual"] >= max(
            1.0,
            totals["pairwise"] + totals["verification"] + totals["cardinality"],
        ):
            return "global_recovery", totals
        if totals["verification"] > 0.0:
            return "verification", totals
        if (
            totals["pairwise"] > 0.0
            or totals["high_surface"] > 0.0
            or totals["cardinality"] > 0.0
        ):
            return "local_disambiguation", totals
        return "balanced", totals

    def _adaptive_dss_selection_config(
        config: DSSPPConfig,
        *,
        residual_burst_active: bool,
        remaining_estimate: Mapping[str, Any] | object | None,
        label: str,
    ) -> DSSPPConfig:
        """Adapt DSS-PP using only PF posterior-predictive budget evidence."""
        if config.forced_program_pair_ids is not None:
            return config
        payload = _remaining_measurement_payload(remaining_estimate)
        components = payload.get("components", {})
        component_values = dict(components) if isinstance(components, Mapping) else {}

        def _budget(key: str) -> float:
            """Return a finite PF remaining-budget component."""
            try:
                value = float(component_values.get(key, 0.0))
            except (TypeError, ValueError):
                return 0.0
            return value if np.isfinite(value) else 0.0

        cardinality_ready, _cardinality_reason = _source_cardinality_dwell_status(
            estimator,
            refresh_estimates=False,
        )
        residual_unresolved = bool(
            _budget("residual") > dss_adaptive_residual_budget_threshold
            or _budget("isotope_absence") > dss_adaptive_residual_budget_threshold
            or bool(residual_burst_active)
        )
        ambiguity_unresolved = bool(
            _budget("same_isotope_separation")
            > dss_adaptive_ambiguity_budget_threshold
            or _budget("pseudo_source_verification")
            > dss_adaptive_ambiguity_budget_threshold
            or _budget("high_surface_ambiguity")
            > dss_adaptive_ambiguity_budget_threshold
            or _budget("cardinality") > dss_adaptive_ambiguity_budget_threshold
        )
        require_cardinality_gap = bool(
            dss_residual_extension_requires_cardinality_evidence
        )
        priority_isotopes = _pf_priority_isotopes_from_remaining(remaining_estimate)
        planner_mode, mode_scores = (
            _pf_planner_mode_from_remaining(remaining_estimate)
            if bool(config.explicit_mode_switch)
            else (str(config.planner_mode), {})
        )
        adapted, reason = _adapt_dss_config_from_pf_evidence(
            config,
            adaptive_program_length_enabled=bool(
                dss_adaptive_program_length_enabled
            ),
            residual_unresolved=residual_unresolved,
            ambiguity_unresolved=ambiguity_unresolved,
            require_cardinality_gap=require_cardinality_gap,
            cardinality_ready=cardinality_ready,
            remaining_ready=_remaining_measurement_ready_for_stop(payload),
            residual_program_length=int(dss_residual_program_length_resolved),
            simple_program_length=int(dss_adaptive_simple_program_length),
            priority_isotopes=priority_isotopes,
            planner_mode=planner_mode,
        )
        if bool(dss_adaptive_program_length_enabled) and reason != "full":
            print(
                "DSS-PP adaptive shield program length: "
                f"context={label} reason={reason} "
                f"program_length={int(adapted.program_length)} "
                f"base={int(config.program_length)} "
                f"residual={int(dss_residual_program_length_resolved)}"
            )
        if priority_isotopes:
            print(
                "DSS-PP PF-priority isotopes: "
                f"context={label} isotopes={list(priority_isotopes)}"
            )
        if bool(adapted.explicit_mode_switch):
            print(
                "DSS-PP explicit planner mode: "
                f"context={label} mode={adapted.planner_mode} "
                f"scores={_safe_json_dumps(mode_scores)} "
                f"signature_w={float(adapted.lambda_signature):.2f} "
                f"temporal_w={float(adapted.lambda_temporal_separation):.2f} "
                f"condition_w={float(adapted.lambda_station_condition):.2f}"
            )
        return adapted

    def _forced_baseline_program_for_planned_station(
        *,
        label: str,
    ) -> tuple[tuple[int, ...] | None, DSSPPConfig, BaselineShieldProgram | None]:
        """Return a deterministic baseline shield program for the next station."""
        dss_selection_config = dss_config
        if baseline_shield_policy is None:
            return None, dss_selection_config, None
        residual_burst_active = _has_birth_residual_evidence(
            estimator,
            min_support=max(
                1,
                int(estimator.pf_config.birth_residual_min_support),
            ),
        )
        dss_selection_config = _adaptive_dss_selection_config(
            dss_selection_config,
            residual_burst_active=residual_burst_active,
            remaining_estimate=_latest_remaining_estimate_for_planning(),
            label=label,
        )
        dss_selection_config, baseline_program = (
            _apply_baseline_shield_program_to_dss_config(
                dss_selection_config,
                baseline_shield_policy,
                total_pairs=total_pairs,
                pose_index=pose_counter,
                current_pair_id=current_shield_pair_id,
            )
        )
        if baseline_program is None:
            return None, dss_selection_config, None
        forced_pairs = tuple(int(pair_id) for pair_id in baseline_program.pair_ids)
        print(
            "Planned baseline shield program for next station: "
            f"context={label} program={baseline_program.name} "
            f"pairs={list(forced_pairs)} "
            f"program_length={int(dss_selection_config.program_length)}"
        )
        return forced_pairs, dss_selection_config, baseline_program

    if max_steps is not None and max_steps <= 0:
        max_steps = None
    if max_poses is not None and max_poses <= 0:
        max_poses = None
    gpu_status = "disabled"
    if bool(estimator.pf_config.use_gpu):
        gpu_status = "enabled" if estimator._gpu_enabled() else "disabled"
    cfg = decomposer.config
    dwell_cap_label = f"{live_time:.3f}" if has_live_time_cap else "unbounded"
    dwell_step_label = f"{live_time:.1f}" if has_live_time_cap else "unbounded"
    if save_outputs:
        IG_DIR.mkdir(parents=True, exist_ok=True)
    print(
        "Spectrum config: "
        f"bin_width_keV={cfg.bin_width_keV}, live_time_s={dwell_cap_label}, "
        f"smooth_sigma_bins={cfg.smooth_sigma_bins}, "
        f"als_lambda={cfg.als_lambda}, als_p={cfg.als_p}, als_niter={cfg.als_niter}, "
        f"resolution_a={cfg.resolution_a}, resolution_b={cfg.resolution_b}, "
        f"peak_window_sigma={cfg.peak_window_sigma}, dead_time_tau_s={cfg.dead_time_tau_s}, "
        f"response_continuum_to_peak={cfg.response_continuum_to_peak}, "
        f"response_backscatter_fraction={cfg.response_backscatter_fraction}, "
        f"apply_incident_gamma_detector_response={cfg.apply_incident_gamma_detector_response}"
    )
    print(
        "Dwell control: "
        f"adaptive={bool(adaptive_dwell)} "
        f"cap_s={dwell_cap_label} "
        f"chunk_s={float(adaptive_dwell_chunk_s):.3f} "
        f"min_s={float(adaptive_min_dwell_s):.3f} "
        f"ready_min_counts={float(adaptive_ready_min_counts):.3f} "
        f"ready_min_isotopes={int(adaptive_ready_min_isotopes)} "
        f"ready_min_snr={float(adaptive_ready_min_snr):.3f} "
        f"allow_informative_low={bool(adaptive_ready_allow_informative_low)} "
        f"allow_low_signal_stop={bool(adaptive_allow_low_signal_stop)} "
        f"low_signal_min_s={float(adaptive_low_signal_min_live_s):.1f} "
        f"low_signal_sigma={float(adaptive_low_signal_upper_sigma):.1f} "
        f"low_signal_count_fraction={float(adaptive_low_signal_count_fraction):.3f} "
        "low_signal_projected_live_factor="
        f"{float(adaptive_low_signal_projected_live_factor):.2f} "
        f"cardinality_dwell={bool(adaptive_cardinality_dwell_enable)} "
        f"cardinality_min_s={float(adaptive_cardinality_min_live_s):.1f}"
    )
    print(
        "Output rendering: "
        f"spectrum_plot_save_every={int(spectrum_plot_save_every)}; "
        f"pf_plot_save_every={int(pf_plot_save_every)}; "
        f"headless_visualizer_defer={bool(runtime_config.get('headless_visualizer_defer', True))}; "
        "estimates use the current PF posterior projection"
    )
    print(
        "Pose observability constraint: "
        f"min_counts={float(pf_conf.pose_min_observation_counts):.3f} "
        f"penalty_scale={float(pf_conf.pose_min_observation_penalty_scale):.3f} "
        f"aggregate={pf_conf.pose_min_observation_aggregate} "
        f"quantile={float(pf_conf.pose_min_observation_quantile):.3f}"
    )
    print(
        "Path planner: "
        f"mode={path_planner_resolved} "
        f"dss_horizon={int(dss_config.horizon)} "
        f"dss_beam={int(dss_config.beam_width)} "
        f"dss_program_len={int(dss_config.program_length)} "
        "dss_min_primary_history_weight="
        f"{float(dss_config.primary_history_weight):.6g} "
        "dss_target_sampled_primaries="
        f"{_target_sampled_primaries(runtime_config)} "
        f"signature_w={float(dss_config.lambda_signature):.3f} "
        f"temporal_sep_w={float(dss_config.lambda_temporal_separation):.3f} "
        f"residual_signature_w={float(dss_config.residual_signature_weight):.3f} "
        f"differential_w={float(dss_config.eta_differential):.3f} "
        f"count_balance_w={float(dss_config.eta_count_balance):.3f} "
        f"rotation_w={float(dss_config.lambda_rotation):.3f} "
        f"coverage_w={float(dss_config.lambda_coverage):.3f} "
        f"bearing_w={float(dss_config.lambda_bearing_diversity):.3f} "
        f"frontier_w={float(dss_config.lambda_frontier):.3f} "
        f"count_util_w={float(dss_config.lambda_count_utility):.3f} "
        f"local_orbit_w={float(dss_config.lambda_local_orbit):.3f} "
        f"station_cond_w={float(dss_config.lambda_station_condition):.3f} "
        "station_cond_min_sv_w="
        f"{float(dss_config.station_condition_min_singular_weight):.3f} "
        "station_cond_inv_cond_w="
        f"{float(dss_config.station_condition_inverse_condition_weight):.3f} "
        "station_cond_coherence_w="
        f"{float(dss_config.station_condition_coherence_weight):.3f} "
        f"corr_reduction_w={float(dss_config.lambda_correlation_reduction):.3f} "
        f"isotope_balance_w={float(dss_config.lambda_isotope_balance):.3f} "
        f"env_sig_w={float(dss_config.lambda_environment_signature):.3f} "
        f"occ_boundary_w={float(dss_config.lambda_occlusion_boundary):.3f} "
        f"turn_w={float(dss_config.lambda_turn_smoothness):.3f} "
        f"revisit_w={float(dss_config.eta_revisit):.3f} "
        f"remaining_guidance={bool(dss_config.remaining_budget_guidance)} "
        f"remaining_route_w={float(dss_config.remaining_route_weight):.3f} "
        f"remaining_urgency={int(dss_config.remaining_budget_urgency_stations)} "
        "adaptive_program_len="
        f"{bool(dss_adaptive_program_length_enabled)} "
        "residual_requires_cardinality_gap="
        f"{bool(dss_residual_extension_requires_cardinality_evidence)} "
        f"simple_len={int(dss_adaptive_simple_program_length)} "
        f"residual_len={int(dss_residual_program_length_resolved)} "
        f"pose_eval_workers={int(pose_selection_workers_resolved)} "
        f"one_step_pose_eval_use_gpu={one_step_pose_eval_use_gpu} "
        f"one_step_guard={bool(dss_one_step_guard_enabled)} "
        f"one_step_guard_use_gpu={dss_one_step_guard_use_gpu} "
        f"explicit_mode_switch={bool(dss_config.explicit_mode_switch)} "
        f"planner_mode={dss_config.planner_mode} "
        "same_iso_direct_guard="
        f"{bool(dss_config.same_isotope_direct_separation_guard)} "
        "same_iso_direct_eps="
        f"{float(dss_config.same_isotope_direct_separation_epsilon):.3g} "
        f"high_surface_pair_boost={float(dss_config.high_surface_pair_boost):.2f} "
        "high_surface_cross_stratum_boost="
        f"{float(dss_config.high_surface_cross_stratum_boost):.2f} "
        f"high_surface_z_frac={float(dss_config.high_surface_z_fraction):.2f} "
        f"min_station_sep={float(dss_config.min_station_separation_m):.2f}m "
        f"enforce_min_obs={bool(dss_config.enforce_min_observation)}"
    )
    print(
        "Remaining measurement estimator: "
        f"enabled={bool(remaining_measurement_config.enabled)} "
        f"target_spread={float(remaining_measurement_config.target_position_spread_m):.2f}m "
        f"target_card_conf={float(remaining_measurement_config.target_cardinality_confidence):.2f} "
        f"sep_threshold={float(remaining_measurement_config.pairwise_separation_threshold):.2f} "
        "high_surface_sep_threshold="
        f"{float(remaining_measurement_config.high_surface_pairwise_separation_threshold):.2f} "
        f"high_surface_w={float(remaining_measurement_config.high_surface_ambiguity_weight):.2f} "
        f"eta_default={float(remaining_measurement_config.eta_default):.2f}"
    )
    print(
        "PF count likelihood: "
        f"model={pf_conf.count_likelihood_model} "
        f"transport_rel_sigma={pf_conf.transport_model_rel_sigma} "
        f"transport_abs_sigma={pf_conf.transport_model_abs_sigma} "
        f"spectrum_rel_sigma={pf_conf.spectrum_count_rel_sigma} "
        f"spectrum_abs_sigma={pf_conf.spectrum_count_abs_sigma} "
        f"low_count_abs_sigma={pf_conf.low_count_abs_sigma} "
        f"low_count_transition={pf_conf.low_count_transition_counts} "
        f"df={float(pf_conf.count_likelihood_df):.2f} "
        "obs_var_includes_counting_noise="
        f"{bool(pf_conf.observation_count_variance_includes_counting_noise)} "
        "obs_var_semantics="
        f"{pf_conf.observation_count_variance_semantics} "
        "direct_spectrum_likelihood="
        f"{bool(pf_conf.direct_spectrum_likelihood_enable)} "
        "station_view_cov="
        f"{bool(pf_conf.station_view_covariance_enable)} "
        "station_view_spectrum_frac="
        f"{float(pf_conf.station_view_correlated_spectrum_fraction):.2f} "
        "shield_view_ratio="
        f"{bool(pf_conf.shield_view_ratio_likelihood_enable)} "
        "shield_view_ratio_weight="
        f"{float(pf_conf.shield_view_ratio_likelihood_weight):.2f} "
        "shield_view_ratio_conc="
        f"{float(pf_conf.shield_view_ratio_likelihood_concentration):.1f}"
    )
    print(
        "PF shield-program update: "
        f"delayed_resample={bool(delayed_resample_update)} "
        f"legacy_joint={bool(joint_observation_update)}"
    )
    print(
        "Shield posture selector: "
        f"signature_w={float(shield_signature_weight):.3f} "
        f"low_count_w={float(shield_low_count_penalty_weight):.3f} "
        f"count_balance_w={float(shield_count_balance_weight):.3f} "
        f"rotation_w={float(shield_rotation_cost_weight):.3f} "
        f"max_particles={shield_selection_max_particles} "
        f"stop_min_gain={float(shield_stop_min_gain):.6g} "
        f"compare_next_pose={bool(shield_stop_compare_next_pose)} "
        f"rate_margin={float(shield_stop_rate_margin):.3f} "
        f"signature_cosine_stop={float(shield_stop_signature_cosine):.3f}"
    )
    print(
        "Rotation IG threshold: "
        f"mode={ig_threshold_mode}, floor={estimator.pf_config.ig_threshold:.6g}, "
        f"rel={ig_threshold_rel:.6g}"
    )
    ig_workers = _resolve_ig_workers(estimator.pf_config.ig_workers)
    print(
        "Python CPU workers: "
        f"general={python_worker_count_resolved} "
        f"ig_grid={ig_workers} "
        f"dss_program_eval={dss_config.program_eval_workers} "
        f"pf_structural_trials={pf_conf.structural_trial_workers}"
    )
    print(
        "Candidate grid: "
        f"mode={pf_conf.source_position_prior} "
        f"spacing={spacing} margin={float(candidate_grid_margin):.2f} "
        f"points={grid.shape[0]}"
    )
    print(
        "PF source-position support: "
        f"min={tuple(round(float(v), 3) for v in source_position_min)} "
        f"max={tuple(round(float(v), 3) for v in source_position_max)}"
    )
    if init_support_prob is None:
        print(
            "Init support probability: "
            f"surface prior active, candidates={grid.shape[0]} "
            f"(init_num_sources={pf_conf.init_num_sources})"
        )
    else:
        print(
            "Init support probability: "
            f"radius={float(eval_match_radius_m):.2f}m "
            f"prob≈{init_support_prob:.3f} "
            f"(init_num_sources={pf_conf.init_num_sources})"
        )
    print(
        "PF init prior: "
        f"init_num_sources={pf_conf.init_num_sources}, "
        f"init_grid_spacing_m={pf_conf.init_grid_spacing_m}, "
        f"init_grid_repeats={int(pf_conf.init_grid_repeats)}, "
        f"init_strength_log_mean={pf_conf.init_strength_log_mean:.2f}, "
        f"init_strength_log_sigma={pf_conf.init_strength_log_sigma:.2f}, "
        f"max_sources={pf_conf.max_sources}"
    )
    print(
        "Tempering settings: "
        f"max_resamples_per_observation={pf_conf.max_resamples_per_observation} "
        f"disable_regularize_on_temper_resample={pf_conf.disable_regularize_on_temper_resample} "
        f"deferred_resample_roughening_scale={pf_conf.deferred_resample_roughening_scale:.3f} "
        f"cardinality_preserving_resample={bool(pf_conf.cardinality_preserving_resample)} "
        f"cardinality_preserving_min_stations={int(pf_conf.cardinality_preserving_min_stations)} "
        f"mode_preserve_high_extra={int(pf_conf.mode_preserving_high_surface_extra_particles)} "
        f"mode_preserve_high_z_frac={float(pf_conf.mode_preserving_high_surface_z_fraction):.2f} "
        "cardinality_preserving_require_confirmed="
        f"{bool(pf_conf.cardinality_preserving_require_confirmed_structure)}"
    )
    print(
        "Roughening settings: "
        f"k={pf_conf.roughening_k:.3f} "
        f"min_sigma_pos={pf_conf.min_sigma_pos:.3f} "
        f"max_sigma_pos={pf_conf.max_sigma_pos:.3f}"
    )
    print(
        "Planning rollout settings: "
        f"eig_samples={estimator.pf_config.planning_eig_samples}, "
        f"particles={estimator.pf_config.planning_rollout_particles}, "
        f"method={estimator.pf_config.planning_rollout_method}, "
        f"rollouts={DEFAULT_PLANNING_ROLLOUTS}"
    )
    print(
        "GPU acceleration: "
        f"{gpu_status} (device={estimator.pf_config.gpu_device}, dtype={estimator.pf_config.gpu_dtype})"
    )
    print(
        "PF parallelism: "
        f"parallel_isotope_updates={bool(estimator.pf_config.parallel_isotope_updates)} "
        f"parallel_isotope_workers={estimator.pf_config.parallel_isotope_workers} "
        f"structural_trial_workers={estimator.pf_config.structural_trial_workers} "
        f"structural_trial_parallel_min_trials="
        f"{estimator.pf_config.structural_trial_parallel_min_trials}"
    )
    print(f"Simulation backend: {sim_backend}")
    print(
        "Mission timing model: "
        f"robot_speed={float(nominal_motion_speed_m_s):.3f}m/s "
        f"shield_overhead={float(rotation_overhead_s):.3f}s/measurement "
        "mission_time=travel+shield+live"
    )
    print(
        "Convergence gating: "
        f"enabled={estimator.pf_config.converge_enable} "
        f"window={estimator.pf_config.converge_window} "
        f"min_steps={estimator.pf_config.converge_min_steps} "
        f"map_eps={estimator.pf_config.converge_map_move_eps_m:.3f} "
        f"ess_ratio_high={estimator.pf_config.converge_ess_ratio_high:.2f} "
        f"ll_improve_eps={estimator.pf_config.converge_ll_improve_eps:.3f} "
        f"require_all={estimator.pf_config.converge_require_all} "
        f"freeze_updates={estimator.pf_config.converge_freeze_updates}"
    )
    print(
        "Adaptive mission stop: "
        f"enabled={adaptive_mission_stop} "
        f"min_convergence_poses={mission_stop_min_convergence_poses} "
        f"max_poses={max_poses if max_poses is not None else 'none'} "
        f"coverage_radius={mission_stop_coverage_radius_m:.2f}m "
        f"coverage_threshold={mission_stop_coverage_fraction:.3f} "
        f"quiet_birth_residual={mission_stop_require_quiet_birth_residual} "
        f"coverage_requires_pf_convergence="
        f"{mission_stop_require_pf_convergence_for_coverage} "
        f"birth_residual_min_support={mission_stop_birth_residual_min_support} "
        "require_no_unresolved_discriminative_failures="
        f"{mission_stop_require_no_unresolved_discriminative_failures} "
        "require_pf_cardinality_ready="
        f"{mission_stop_require_pf_cardinality_ready} "
        "require_remaining_measurement_ready="
        f"{mission_stop_require_remaining_measurement_ready} "
        "unresolved_fail_min_count="
        f"{mission_stop_unresolved_discriminative_fail_min_count}"
    )
    has_environment_obstacles = _has_environment_obstacles(obstacle_grid)
    reset_usd_path = (
        generated_blender_usd_path.as_posix()
        if generated_blender_usd_path is not None
        else (None if has_environment_obstacles else "")
    )
    simulation_runtime.reset(
        {
            "usd_path": reset_usd_path,
            "room_size_xyz": [env.size_x, env.size_y, env.size_z],
            "source_count": len(sources),
            "random_source_visibility": random_source_visibility_diagnostics,
            "sources": [
                {
                    "isotope": source.isotope,
                    "position": [
                        float(source.position[0]),
                        float(source.position[1]),
                        float(source.position[2]),
                    ],
                    "intensity_cps_1m": float(source.intensity_cps_1m),
                }
                for source in sources
            ],
            "obstacle_origin_xy": (
                [0.0, 0.0] if obstacle_grid is None else list(obstacle_grid.origin)
            ),
            "obstacle_cell_size_m": 1.0
            if obstacle_grid is None
            else float(obstacle_grid.cell_size),
            "obstacle_material": runtime_obstacle_material,
            "obstacle_grid_shape": [0, 0]
            if obstacle_grid is None
            else list(obstacle_grid.grid_shape),
            "obstacle_cells": []
            if obstacle_grid is None
            else list(obstacle_grid.blocked_cells),
            "collision_boxes_m": []
            if obstacle_grid is None
            else [list(box) for box in obstacle_grid.collision_boxes_m],
            "transport_boxes_m": []
            if obstacle_grid is None
            else [list(box) for box in obstacle_grid.transport_boxes_m],
            "transport_mu_by_isotope": {}
            if obstacle_grid is None
            else {
                str(isotope): [float(value) for value in values]
                for isotope, values in obstacle_grid.transport_mu_by_isotope.items()
            },
            "transport_line_mu_by_isotope": {}
            if obstacle_grid is None
            else {
                str(isotope): [[float(value) for value in row] for row in rows]
                for isotope, rows in (
                    obstacle_grid.transport_line_mu_by_isotope.items()
                )
            },
            "obstacle_instances": []
            if known_obstacle_instances is None
            else obstacle_instances_to_dicts(known_obstacle_instances),
            "traversability_map_path": None
            if traversability_map_path is None
            else traversability_map_path.as_posix(),
            "traversability_map_png_path": None
            if traversability_map_png_path is None
            else traversability_map_png_path.as_posix(),
            "robot_radius_m": float(effective_robot_radius_m),
            "measurement_workspace": measurement_workspace_diagnostics,
            "author_obstacle_prims": (
                known_obstacle_instances is not None
                or generated_blender_usd_path is None
            ),
            "use_config_usd_fallback": bool(
                generated_blender_usd_path is not None or has_environment_obstacles
            ),
        }
    )
    notifier.notify_started(
        {
            "backend": sim_backend,
            "sim_config_path": sim_config_path,
            "max_steps": max_steps,
            "max_poses": max_poses,
            "environment_mode": normalized_environment_mode,
            "obstacle_layout_path": obstacle_layout_path,
            "obstacle_seed": obstacle_seed,
            "obstacle_blocked_fraction": (
                None if obstacle_grid is None else float(obstacle_grid.blocked_fraction)
            ),
            "pf_obstacle_attenuation": bool(pf_obstacle_attenuation_enabled),
            "pf_obstacle_grid_active": _has_environment_obstacles(pf_obstacle_grid),
            "remaining_measurement_estimator": {
                "enabled": bool(remaining_measurement_config.enabled),
                "target_position_spread_m": float(
                    remaining_measurement_config.target_position_spread_m
                ),
                "target_cardinality_confidence": float(
                    remaining_measurement_config.target_cardinality_confidence
                ),
                "pairwise_separation_threshold": float(
                    remaining_measurement_config.pairwise_separation_threshold
                ),
                "eta_default": float(remaining_measurement_config.eta_default),
            },
            "source_count": len(sources),
            "sources": [
                {
                    "isotope": source.isotope,
                    "position": [
                        float(source.position[0]),
                        float(source.position[1]),
                        float(source.position[2]),
                    ],
                    "intensity_cps_1m": float(source.intensity_cps_1m),
                }
                for source in sources
            ],
            "isotopes": isotopes,
            "birth_enabled": birth_enabled,
            "birth_max_per_update": pf_conf.birth_max_per_update,
            "converge": converge,
            "pose_candidates": int(pose_candidates),
            "pose_min_dist_m": float(pose_min_dist),
            "detector_height_sampling_mode": detector_height_config.mode,
            "detector_height_min_m": float(
                detector_height_config.minimum_mast_height_m
            ),
            "detector_height_max_m": float(
                detector_height_config.maximum_mast_height_m
            ),
            "detector_height_actions_m": list(
                detector_height_config.discrete_mast_actions_m
            ),
            "detector_height_action_world_z_m": list(
                detector_height_config.discrete_world_actions_m
            ),
            "robot_ground_z_m": float(robot_ground_z_m),
            "measurement_workspace": measurement_workspace_diagnostics,
            "height_partner_reuse_shield_program": bool(
                height_partner_reuse_shield_program
            ),
            "candidate_grid_points": int(grid.shape[0]),
            "source_position_prior": str(pf_conf.source_position_prior),
            "pf_num_particles": int(pf_conf.num_particles),
            "pf_max_sources": (
                None if pf_conf.max_sources is None else int(pf_conf.max_sources)
            ),
            "python_worker_count": int(python_worker_count_resolved),
            "ig_workers": int(ig_workers),
            "dss_program_eval_workers": int(dss_config.program_eval_workers or 1),
            "robot_speed_m_s": float(nominal_motion_speed_m_s),
            "rotation_overhead_s": float(rotation_overhead_s),
            "measurement_time_s": float(live_time),
            "measurement_time_cap_s": float(live_time) if has_live_time_cap else None,
            "adaptive_dwell": bool(adaptive_dwell),
            "adaptive_dwell_chunk_s": float(adaptive_dwell_chunk_s),
            "adaptive_min_dwell_s": float(adaptive_min_dwell_s),
            "adaptive_ready_min_counts": float(adaptive_ready_min_counts),
            "adaptive_ready_min_isotopes": int(adaptive_ready_min_isotopes),
            "adaptive_ready_min_snr": float(adaptive_ready_min_snr),
            "adaptive_ready_allow_informative_low": bool(
                adaptive_ready_allow_informative_low
            ),
            "adaptive_allow_low_signal_stop": bool(adaptive_allow_low_signal_stop),
            "adaptive_low_signal_min_live_s": float(adaptive_low_signal_min_live_s),
            "adaptive_low_signal_upper_sigma": float(adaptive_low_signal_upper_sigma),
            "adaptive_low_signal_count_fraction": float(
                adaptive_low_signal_count_fraction
            ),
            "adaptive_low_signal_projected_live_factor": float(
                adaptive_low_signal_projected_live_factor
            ),
            "adaptive_cardinality_dwell_enable": bool(
                adaptive_cardinality_dwell_enable
            ),
            "adaptive_cardinality_min_live_s": float(adaptive_cardinality_min_live_s),
            "delayed_resample_update": bool(delayed_resample_update),
            "joint_observation_update": bool(joint_observation_update),
            "strict_planned_shield_program": bool(strict_planned_shield_program),
            "baseline_shield_policy": baseline_shield_policy,
            "baseline_path_policy": baseline_path_policy,
            "shield_signature_weight": float(shield_signature_weight),
            "shield_low_count_penalty_weight": float(shield_low_count_penalty_weight),
            "shield_count_balance_weight": float(shield_count_balance_weight),
            "shield_rotation_cost_weight": float(shield_rotation_cost_weight),
            "shield_selection_max_particles": None
            if shield_selection_max_particles is None
            else int(shield_selection_max_particles),
            "shield_signature_variance_floor": float(shield_signature_variance_floor),
            "shield_stop_min_gain": float(shield_stop_min_gain),
            "shield_stop_compare_next_pose": bool(shield_stop_compare_next_pose),
            "shield_stop_pose_candidates": int(shield_stop_pose_candidates),
            "shield_stop_rate_margin": float(shield_stop_rate_margin),
            "shield_stop_signature_cosine": float(shield_stop_signature_cosine),
            "pose_min_observation_counts": float(pf_conf.pose_min_observation_counts),
            "pose_min_observation_penalty_scale": float(
                pf_conf.pose_min_observation_penalty_scale
            ),
            "pose_min_observation_aggregate": pf_conf.pose_min_observation_aggregate,
            "pose_min_observation_quantile": float(
                pf_conf.pose_min_observation_quantile
            ),
            "path_planner": path_planner_resolved,
            "dss_horizon": int(dss_config.horizon),
            "dss_beam_width": int(dss_config.beam_width),
            "dss_program_length": int(dss_config.program_length),
            "dss_primary_history_weight": float(dss_config.primary_history_weight),
            "dss_minimum_primary_history_weight": float(
                dss_config.primary_history_weight
            ),
            "dss_primary_history_weight_semantics": (
                "minimum_from_maximum_sampling_fraction"
                if _target_sampled_primaries(runtime_config) is not None
                else "fixed_transport_history_weight"
            ),
            "dss_target_sampled_primaries": _target_sampled_primaries(runtime_config),
            "dss_signature_weight": float(dss_config.lambda_signature),
            "dss_temporal_separation_weight": float(
                dss_config.lambda_temporal_separation
            ),
            "dss_temporal_cover_programs": int(dss_config.temporal_cover_programs),
            "dss_differential_weight": float(dss_config.eta_differential),
            "dss_rotation_weight": float(dss_config.lambda_rotation),
            "dss_coverage_weight": float(dss_config.lambda_coverage),
            "dss_count_utility_weight": float(dss_config.lambda_count_utility),
            "dss_count_utility_saturation_counts": float(
                dss_config.count_utility_saturation_counts
            ),
            "dss_local_orbit_weight": float(dss_config.lambda_local_orbit),
            "dss_station_condition_weight": float(dss_config.lambda_station_condition),
            "dss_correlation_reduction_weight": float(
                dss_config.lambda_correlation_reduction
            ),
            "dss_isotope_balance_weight": float(dss_config.lambda_isotope_balance),
            "dss_environment_signature_weight": float(
                dss_config.lambda_environment_signature
            ),
            "dss_occlusion_boundary_weight": float(
                dss_config.lambda_occlusion_boundary
            ),
            "dss_environment_contrast_threshold": float(
                dss_config.environment_contrast_threshold
            ),
            "dss_occlusion_boundary_step_m": float(
                dss_config.occlusion_boundary_step_m
            ),
            "dss_revisit_penalty_weight": float(dss_config.eta_revisit),
            "dss_min_station_separation_m": float(dss_config.min_station_separation_m),
            "dss_remaining_budget_guidance": bool(dss_config.remaining_budget_guidance),
            "dss_remaining_route_weight": float(dss_config.remaining_route_weight),
            "dss_remaining_budget_urgency_stations": int(
                dss_config.remaining_budget_urgency_stations
            ),
            "dss_one_step_guard_enable": bool(dss_one_step_guard_enabled),
            "dss_one_step_guard_score_abs_margin": float(dss_one_step_guard_abs_margin),
            "dss_one_step_guard_score_rel_margin": float(dss_one_step_guard_rel_margin),
        }
    )
    ig_max_global = (
        0.0
        if resume_controller_checkpoint is None
        else float(resume_controller_checkpoint.ig_max_global)
    )
    pose_counter = (
        0
        if resume_controller_state is None
        else int(resume_controller_state.pose_counter)
    )
    current_shield_pair_id: int | None = (
        None
        if resume_controller_state is None
        else int(resume_controller_state.current_shield_pair_id)
    )
    pending_shield_program: tuple[int, ...] | None = None
    pending_force_strict_shield_program = False
    resume_station_boundary_pending = resume_controller_state is not None
    try:
        while True:
            resume_station_boundary = bool(resume_station_boundary_pending)
            resume_station_boundary_pending = False
            pose = current_pose
            stop_run = bool(
                resume_station_boundary
                and max_steps is not None
                and step_counter >= max_steps
            )
            pose_elapsed = 0.0
            zero_ig_override = False
            active_shield_program = pending_shield_program
            force_strict_active_shield_program = pending_force_strict_shield_program
            pending_shield_program = None
            pending_force_strict_shield_program = False
            if active_shield_program:
                planned_label = (
                    "planned baseline/DSS-PP"
                    if baseline_shield_policy is not None
                    else "planned DSS-PP"
                )
                print(
                    f"Executing {planned_label} shield program at this pose: "
                    f"{list(active_shield_program)}"
                )
            remaining_orientations = set(range(total_pairs))
            rotation_limit = max(1, int(estimator.pf_config.orientation_k))
            if active_shield_program:
                rotation_limit = _resolve_rotation_limit_for_active_program(
                    base_rotation_limit=rotation_limit,
                    active_shield_program=active_shield_program,
                    strict_planned_shield_program=strict_planned_shield_program,
                    baseline_shield_policy=baseline_shield_policy,
                    force_strict_program=force_strict_active_shield_program,
                )
            if not active_shield_program:
                baseline_program = select_baseline_shield_program(
                    baseline_shield_policy,
                    total_pairs=total_pairs,
                    program_length=rotation_limit,
                    pose_index=pose_counter,
                    current_pair_id=current_shield_pair_id,
                )
                if baseline_program is not None:
                    active_shield_program = tuple(
                        int(v) for v in baseline_program.pair_ids
                    )
                    rotation_limit = _resolve_rotation_limit_for_active_program(
                        base_rotation_limit=rotation_limit,
                        active_shield_program=active_shield_program,
                        strict_planned_shield_program=strict_planned_shield_program,
                        baseline_shield_policy=baseline_shield_policy,
                        force_strict_program=force_strict_active_shield_program,
                    )
                    used_name = str(baseline_program.name)
                    print(
                        "Executing baseline shield program: "
                        f"{used_name} pairs={list(active_shield_program)}"
                    )
            force_active_shield_program = bool(active_shield_program)
            joint_update_records: list[tuple[object, ...]] = []
            executed_pair_ids_this_pose: list[int] = (
                list(resume_controller_state.last_station_pair_ids)
                if resume_station_boundary and resume_controller_state is not None
                else []
            )
            deferred_update_records = 0
            executed_signature_vectors: list[NDArray[np.float64]] = []
            if delayed_resample_update:
                estimator.begin_deferred_pose_update()
            min_rotations_this_pose = min(
                rotation_limit,
                max(2, int(estimator.pf_config.min_rotations_per_pose)),
            )
            rotation_count = rotation_limit if resume_station_boundary else 0
            ig_max_pose = (
                float(resume_controller_checkpoint.ig_max_pose)
                if (
                    resume_station_boundary
                    and resume_controller_checkpoint is not None
                )
                else 0.0
            )
            ig_threshold_current = (
                float(resume_controller_checkpoint.ig_threshold_current)
                if (
                    resume_station_boundary
                    and resume_controller_checkpoint is not None
                )
                else estimator.pf_config.ig_threshold
            )
            used_planned_program_this_pose = bool(
                force_active_shield_program or resume_station_boundary
            )
            while True:
                if rotation_count >= rotation_limit:
                    print(
                        f"Reached max rotations per pose ({rotation_limit}); "
                        "moving to the next pose."
                    )
                    break
                if not remaining_orientations:
                    print("All orientation pairs exhausted; moving to the next pose.")
                    break
                planned_pair = None
                if (
                    force_active_shield_program
                    and active_shield_program
                    and rotation_count < len(active_shield_program)
                ):
                    # Explicit baseline/DSS programs may intentionally repeat a
                    # posture, e.g. a fixed-shield baseline with equal live-time
                    # budget. Do not apply the adaptive selector's uniqueness
                    # constraint to an explicitly supplied program.
                    planned_pair = int(active_shield_program[rotation_count])
                if planned_pair is None:
                    ig_start = time.perf_counter()
                    ig_scores = _compute_ig_grid(
                        estimator,
                        rot_mats,
                        pose_idx=current_pose_idx,
                        live_time_s=planning_live_time,
                        planning_isotopes=None,
                    )
                    ig_elapsed = time.perf_counter() - ig_start
                    total_ig_wall_s += ig_elapsed
                    ig_wall_samples_s.append(float(ig_elapsed))
                    print(f"IG grid computed in {ig_elapsed:.3f}s.")
                    shield_selection_start = time.perf_counter()
                    shield_scores, shield_score_parts = _compute_shield_selection_grid(
                        estimator,
                        rot_mats,
                        pose_idx=current_pose_idx,
                        live_time_s=planning_live_time,
                        ig_scores=ig_scores,
                        current_pair_id=current_shield_pair_id,
                        min_observation_counts=float(
                            pf_conf.pose_min_observation_counts
                        ),
                        signature_weight=float(shield_signature_weight),
                        low_count_penalty_weight=float(shield_low_count_penalty_weight),
                        count_balance_weight=float(shield_count_balance_weight),
                        rotation_cost_weight=float(shield_rotation_cost_weight),
                        variance_floor=float(shield_signature_variance_floor),
                        max_particles=shield_selection_max_particles,
                    )
                    shield_selection_elapsed = (
                        time.perf_counter() - shield_selection_start
                    )
                    print(
                        "Shield selection grid computed in "
                        f"{shield_selection_elapsed:.3f}s."
                    )
                    best_pair_idx, shield_score = _select_best_pair_from_scores(
                        shield_scores,
                        remaining_orientations,
                    )
                    using_planned_pair = False
                else:
                    best_pair_idx = int(planned_pair)
                    ig_elapsed = 0.0
                    shield_selection_elapsed = 0.0
                    shield_score = 0.0
                    using_planned_pair = True
                if best_pair_idx < 0:
                    print("No valid orientation candidates; moving to the next pose.")
                    break
                fe_for_score = best_pair_idx // num_orients
                pb_for_score = best_pair_idx % num_orients
                if using_planned_pair:
                    raw_ig_val = 0.0
                    signature_val = 0.0
                    signature_utility_val = 0.0
                    low_count_penalty = 0.0
                    count_balance_penalty = 0.0
                    rotation_cost = 0.0
                else:
                    raw_ig_val = max(float(ig_scores[fe_for_score, pb_for_score]), 0.0)
                    signature_val = float(
                        shield_score_parts["signature"][fe_for_score, pb_for_score]
                    )
                    signature_utility_val = float(
                        shield_score_parts["signature_utility"][
                            fe_for_score,
                            pb_for_score,
                        ]
                    )
                    low_count_penalty = float(
                        shield_score_parts["low_count_penalty"][
                            fe_for_score,
                            pb_for_score,
                        ]
                    )
                    count_balance_penalty = float(
                        shield_score_parts["count_balance_penalty"][
                            fe_for_score,
                            pb_for_score,
                        ]
                    )
                    rotation_cost = float(
                        shield_score_parts["rotation_cost"][
                            fe_for_score,
                            pb_for_score,
                        ]
                    )
                shield_gain = float(shield_score)
                ig_val = max(raw_ig_val, shield_gain, 0.0)
                last_max_ig = ig_val
                ig_max_global = max(ig_max_global, ig_val)
                ig_max_pose = max(ig_max_pose, ig_val)
                ig_threshold_current = _resolve_ig_threshold(
                    mode=ig_threshold_mode,
                    ig_floor=estimator.pf_config.ig_threshold,
                    ig_rel=ig_threshold_rel,
                    ig_max_global=ig_max_global,
                    ig_max_pose=ig_max_pose,
                )
                if ig_max_pose <= 0.0 and ig_threshold_current > 0.0:
                    if not zero_ig_override:
                        print(
                            "IG grid returned zero; forcing rotation despite threshold."
                        )
                        zero_ig_override = True
                    ig_threshold_current = 0.0
                if using_planned_pair:
                    signature_vector = np.zeros(len(estimator.isotopes), dtype=float)
                    signature_dependent = False
                else:
                    predicted_counts = (
                        estimator.expected_observation_counts_by_isotope_at_pair(
                            pose_idx=current_pose_idx,
                            fe_index=fe_for_score,
                            pb_index=pb_for_score,
                            live_time_s=planning_live_time,
                            max_particles=shield_selection_max_particles,
                        )
                    )
                    signature_vector = np.asarray(
                        [
                            float(predicted_counts.get(iso, 0.0))
                            for iso in estimator.isotopes
                        ],
                        dtype=float,
                    )
                    signature_dependent = _signature_vector_is_dependent(
                        signature_vector,
                        executed_signature_vectors,
                        cosine_threshold=float(shield_stop_signature_cosine),
                    )
                shield_gain_rate = max(shield_gain, 0.0) / max(
                    float(planning_live_time) + float(rotation_overhead_s),
                    1e-9,
                )
                next_pose_gain_rate = 0.0
                next_pose_gain = 0.0
                next_pose_candidate_idx = -1
                stop_reason = "continue"
                if (
                    rotation_count >= min_rotations_this_pose
                    and not using_planned_pair
                    and shield_gain < float(shield_stop_min_gain)
                ):
                    stop_reason = "shield_gain_below_absolute_threshold"
                elif (
                    ig_val < ig_threshold_current
                    and rotation_count >= min_rotations_this_pose
                    and not using_planned_pair
                ):
                    stop_reason = "shield_gain_below_ig_threshold"
                elif (
                    signature_dependent
                    and rotation_count >= min_rotations_this_pose
                    and not using_planned_pair
                ):
                    stop_reason = "signature_linearly_dependent"
                elif (
                    shield_stop_compare_next_pose
                    and rotation_count >= min_rotations_this_pose
                    and not using_planned_pair
                    and (max_poses is None or pose_counter + 1 < max_poses)
                ):
                    visited_for_stop = list(visited_poses) + [pose.copy()]
                    visited_stop_arr = (
                        np.vstack(visited_for_stop) if visited_for_stop else None
                    )
                    stop_candidates, _, _ = _generate_planning_candidates(
                        current_pose_xyz=pose,
                        map_api=planning_map,
                        n_candidates=int(shield_stop_pose_candidates),
                        min_dist_from_visited=pose_min_dist,
                        visited_poses_xyz=visited_stop_arr,
                        bounds_xyz=(bounds_lo, bounds_hi),
                        detector_heights_m=detector_height_candidates,
                        continuous_height_anchor_count=(
                            detector_continuous_height_partner_candidates
                        ),
                        height_partner_xy_tolerance_m=(
                            detector_height_pair_xy_tolerance_m
                        ),
                        height_partner_z_tolerance_m=(
                            detector_height_pair_z_tolerance_m
                        ),
                        height_partner_min_z_separation_m=(
                            detector_height_pair_min_separation_m
                        ),
                        rng=planning_candidate_rng,
                    )
                    (
                        next_pose_gain_rate,
                        next_pose_gain,
                        next_pose_candidate_idx,
                    ) = _estimate_best_next_pose_gain_rate(
                        estimator,
                        candidates=stop_candidates,
                        current_pose_xyz=pose,
                        map_api=planning_map,
                        live_time_s=planning_live_time,
                        rotation_limit=rotation_limit,
                        nominal_motion_speed_m_s=nominal_motion_speed_m_s,
                        rotation_overhead_s=rotation_overhead_s,
                        max_candidates=int(shield_stop_pose_candidates),
                    )
                    if (
                        next_pose_gain_rate > 0.0
                        and shield_gain_rate
                        < next_pose_gain_rate * float(shield_stop_rate_margin)
                    ):
                        stop_reason = "next_pose_gain_rate_higher"
                if stop_reason != "continue":
                    print(
                        "Stopping rotation at this pose "
                        f"(reason={stop_reason}, "
                        f"shield_gain={shield_gain:.6g}, "
                        f"raw_ig={raw_ig_val:.6g}, "
                        f"threshold={ig_threshold_current:.6g}, "
                        f"shield_gain_rate={shield_gain_rate:.6g}, "
                        f"next_pose_gain={next_pose_gain:.6g}, "
                        f"next_pose_gain_rate={next_pose_gain_rate:.6g}, "
                        f"next_pose_candidate={next_pose_candidate_idx})."
                    )
                    break
                fe_idx = best_pair_idx // num_orients
                pb_idx = best_pair_idx % num_orients
                RFe_sel = rot_mats[fe_idx]
                RPb_sel = rot_mats[pb_idx]
                if (
                    not using_planned_pair
                    and save_outputs
                    and SAVE_IG_GRIDS
                    and (step_counter + 1) % 10 == 0
                ):
                    ig_path = IG_DIR / f"ig_grid_step_{step_counter:04d}.png"
                    render_octant_grid(
                        ig_path,
                        ig_scores=ig_scores,
                        highlight_idx=(fe_idx, pb_idx),
                        highlight_max=False,
                        font_size=12,
                    )
                step_motion_distance_m = float(pending_motion_distance_m)
                step_motion_time_s = float(pending_motion_time_s)
                step_rotation_time_s = float(rotation_overhead_s)
                step_travel_waypoints: list[list[float]] | None = None
                if pending_path_segment is not None:
                    waypoint_payload = pending_path_segment.get("waypoints_xyz")
                    if waypoint_payload is not None:
                        waypoint_array = np.asarray(waypoint_payload, dtype=float)
                        if waypoint_array.ndim == 2 and waypoint_array.shape[1] == 3:
                            step_travel_waypoints = waypoint_array.tolist()
                cardinality_ready = True
                cardinality_reason = "disabled"
                if bool(adaptive_cardinality_dwell_enable):
                    cardinality_ready, cardinality_reason = (
                        _source_cardinality_dwell_status(
                            estimator,
                            refresh_estimates=False,
                        )
                    )
                (
                    observation,
                    actual_live_time_s,
                    z_detected,
                    z_detected_variance,
                    detected,
                    dwell_ready_reason,
                    dwell_chunks,
                ) = _acquire_spectrum_observation(
                    simulation_runtime=simulation_runtime,
                    decomposer=decomposer,
                    step_id=step_counter,
                    pose_xyz=pose,
                    fe_idx=fe_idx,
                    pb_idx=pb_idx,
                    live_time_s=live_time,
                    travel_time_s=step_motion_time_s,
                    shield_actuation_time_s=step_rotation_time_s,
                    adaptive_dwell=bool(adaptive_dwell),
                    adaptive_dwell_chunk_s=float(adaptive_dwell_chunk_s),
                    adaptive_min_dwell_s=float(adaptive_min_dwell_s),
                    adaptive_ready_min_counts=float(adaptive_ready_min_counts),
                    adaptive_ready_min_isotopes=int(adaptive_ready_min_isotopes),
                    adaptive_ready_min_snr=float(adaptive_ready_min_snr),
                    spectrum_count_method=spectrum_count_method,
                    detect_threshold_abs=detect_threshold_abs,
                    detect_threshold_rel=detect_threshold_rel,
                    detect_threshold_rel_by_isotope=detect_threshold_rel_by_isotope,
                    min_peaks_by_isotope=min_peaks_by_isotope,
                    adaptive_progress_every_chunks=(
                        10 if bool(adaptive_dwell) and not has_live_time_cap else 0
                    ),
                    adaptive_ready_allow_informative_low=(
                        adaptive_ready_allow_informative_low
                    ),
                    adaptive_allow_low_signal_stop=adaptive_allow_low_signal_stop,
                    adaptive_low_signal_min_live_s=adaptive_low_signal_min_live_s,
                    adaptive_low_signal_upper_sigma=adaptive_low_signal_upper_sigma,
                    adaptive_low_signal_count_fraction=(
                        adaptive_low_signal_count_fraction
                    ),
                    adaptive_low_signal_projected_live_factor=(
                        adaptive_low_signal_projected_live_factor
                    ),
                    source_cardinality_ready=cardinality_ready,
                    source_cardinality_min_live_s=(adaptive_cardinality_min_live_s),
                    candidate_isotopes=list(isotopes),
                    travel_waypoints_xyz=step_travel_waypoints,
                )
                executed_pair_ids_this_pose.append(int(fe_idx * num_orients + pb_idx))
                pending_motion_distance_m = 0.0
                pending_motion_time_s = 0.0
                step_path_segment = pending_path_segment
                pending_path_segment = None
                total_motion_distance_m += step_motion_distance_m
                total_motion_time_s += step_motion_time_s
                total_rotation_time_s += step_rotation_time_s
                if step_path_segment is not None:
                    path_segments.append(step_path_segment)
                elapsed += step_motion_time_s + step_rotation_time_s
                spectrum = _analysis_spectrum_array(observation, decomposer)
                spectrum_variance = _analysis_spectrum_variance(
                    observation,
                    decomposer,
                )
                last_spectrum = spectrum.copy()
                last_counts = {iso: float(val) for iso, val in z_detected.items()}
                last_spectrum_components = {
                    iso: np.asarray(component, dtype=float).copy()
                    for iso, component in getattr(
                        decomposer,
                        "last_response_poisson_components",
                        {},
                    ).items()
                }
                last_candidates = set(detected)
                spectrum_total_counts = float(np.sum(spectrum))
                if spectrum_total_counts > representative_total_counts:
                    representative_total_counts = spectrum_total_counts
                    representative_spectrum = spectrum.copy()
                    representative_spectrum_components = {
                        iso: component.copy()
                        for iso, component in last_spectrum_components.items()
                    }
                    representative_counts = dict(last_counts)
                    representative_candidates = set(last_candidates)
                    representative_step_index = int(step_counter)
                spectrum_notify_every = max(1, int(notify_spectrum_every))
                if notify_spectrum and step_counter % spectrum_notify_every == 0:
                    notifier.notify_spectrum(
                        step_counter,
                        _build_spectrum_notification_payload(
                            decomposer=decomposer,
                            spectrum=spectrum,
                            step_index=step_counter,
                            pose_xyz=np.asarray(
                                observation.detector_pose_xyz, dtype=float
                            ),
                            fe_index=fe_idx,
                            pb_index=pb_idx,
                            live_time_s=actual_live_time_s,
                            counts_by_isotope=last_counts,
                            detected_isotopes=set(detected),
                            count_method=spectrum_count_method,
                            max_bins=int(notify_spectrum_max_bins),
                        ),
                    )
                raw_detected_isotopes = set(detected)
                if pf_detected_isotopes_only:
                    previous_active_isotopes = set(active_isotopes)
                    updated_active = _update_detection_hysteresis(
                        raw_detected_isotopes,
                        detect_counts,
                        miss_counts,
                        active_isotopes,
                        consecutive=pf_detect_consecutive,
                        miss_consecutive=DETECT_MISS_AFTER_LOCK,
                        consecutive_by_isotope=pf_detect_consecutive_by_isotope,
                    )
                    if pf_detection_activation_only:
                        active_isotopes = previous_active_isotopes | updated_active
                    else:
                        active_isotopes = updated_active
                    detected_isotopes = set(active_isotopes)
                    last_candidates = set(detected_isotopes)
                    newly_active = active_isotopes - previous_active_isotopes
                    if newly_active:
                        ordered_new = [iso for iso in isotopes if iso in newly_active]
                        estimator.add_isotopes(ordered_new)
                    if not pf_detection_activation_only:
                        ordered_active = [
                            iso for iso in isotopes if iso in active_isotopes
                        ]
                        if set(estimator.isotopes) != set(ordered_active):
                            estimator.restrict_isotopes(
                                ordered_active,
                                allow_empty=True,
                            )
                    should_report_detection = (
                        step_counter + 1 >= detect_min_steps
                        and active_isotopes != previous_active_isotopes
                    )
                    if should_report_detection:
                        print(
                            "Spectrum-detected isotope PF set active: "
                            f"{sorted(active_isotopes)} "
                            "(activation-only; inactive isotopes are excluded from PF/planning)."
                        )
                elif detect_consecutive > 0:
                    previous_active_isotopes = set(active_isotopes)
                    active_isotopes = _update_detection_hysteresis(
                        raw_detected_isotopes,
                        detect_counts,
                        miss_counts,
                        active_isotopes,
                        consecutive=detect_consecutive,
                        miss_consecutive=DETECT_MISS_AFTER_LOCK,
                        consecutive_by_isotope=DETECT_CONSECUTIVE_BY_ISOTOPE,
                    )
                    detected_isotopes = set(active_isotopes)
                    last_candidates = set(detected_isotopes)
                    should_report_detection = (
                        step_counter + 1 >= detect_min_steps
                        and active_isotopes != previous_active_isotopes
                    )
                    if should_report_detection:
                        print(
                            "Detected isotope diagnostics active: "
                            f"{sorted(active_isotopes)} "
                            "(PF/planning gating disabled by config)."
                        )
                pf_isotopes = list(estimator.isotopes)
                # The scientific baseline is a response_poisson count-domain PF.
                # Raw bins remain in MeasurementLog for audit and replay and must
                # never enter the online PF likelihood a second time.
                spectrum_payload = None
                z_k_full = {iso: float(z_detected.get(iso, 0.0)) for iso in pf_isotopes}
                z_variance_full = {
                    iso: float(
                        max(
                            z_detected_variance.get(iso, max(z_k_full[iso], 1.0)),
                            1.0,
                        )
                    )
                    for iso in pf_isotopes
                }
                history_z_k = {iso: float(z_detected.get(iso, 0.0)) for iso in isotopes}
                history_z_variance = {
                    iso: float(
                        max(
                            z_detected_variance.get(
                                iso,
                                max(history_z_k[iso], 1.0),
                            ),
                            1.0,
                        )
                    )
                    for iso in isotopes
                }
                metadata_count_covariance = _metadata_count_covariance(
                    observation.metadata,
                    {str(isotope) for isotope in isotopes},
                )
                z_covariance_full = _complete_count_covariance(
                    history_z_variance,
                    metadata_count_covariance,
                    isotopes,
                )
                z_counts = z_k_full
                z_k = z_k_full
                if str(observation.metadata.get("backend", "")).lower() == "geant4":
                    if spectrum_count_method == "response_poisson":
                        response_poisson_counts = {
                            iso: float(z_k_full.get(iso, 0.0)) for iso in pf_isotopes
                        }
                    else:
                        response_poisson_counts = (
                            _response_poisson_counts_for_diagnostics(
                                diagnostic_decomposer,
                                spectrum,
                                pf_isotopes,
                            )
                        )
                    source_equivalent_counts = _source_equivalent_counts_from_metadata(
                        observation.metadata,
                        pf_isotopes,
                    )
                    transport_detected_counts = (
                        _transport_detected_counts_from_metadata(
                            observation.metadata,
                            pf_isotopes,
                        )
                    )
                    _log_geant4_transport_decomposition_diagnostics(
                        step_index=step_counter,
                        metadata=observation.metadata,
                        spectrum_total_counts=spectrum_total_counts,
                        selected_count_method=spectrum_count_method,
                        selected_counts=z_k_full,
                        response_poisson_counts=response_poisson_counts,
                        source_equivalent_counts=source_equivalent_counts,
                        transport_detected_counts=transport_detected_counts,
                    )
                if spectrum_count_method == "response_poisson":
                    _log_spectrum_isotope_channel_diagnostics(
                        decomposer,
                        step_index=step_counter,
                        selected_counts=z_k_full,
                        selected_variances=z_variance_full,
                    )
                pose_for_pf = np.asarray(observation.detector_pose_xyz, dtype=float)
                planned_pose_error_m = float(
                    np.linalg.norm(pose_for_pf - np.asarray(pose, dtype=float))
                )
                if rotation_count == 0:
                    if planned_pose_error_m > detector_pose_consistency_tolerance_m:
                        raise RuntimeError(
                            "Simulator detector pose does not match the planned PF "
                            f"pose: planned={np.asarray(pose, dtype=float).tolist()} "
                            f"observed={pose_for_pf.tolist()} "
                            f"error_m={planned_pose_error_m:.6g} "
                            "(check detector-height actuation and simulator pose "
                            "wiring)."
                        )
                    # The observation pose is the scientific likelihood input.
                    # Synchronize the still-unused station pose before the first
                    # update so the log and replay use bit-identical geometry.
                    estimator.poses[current_pose_idx] = pose_for_pf.copy()
                    estimator.kernel_cache = None
                    pose = pose_for_pf.copy()
                    current_pose = pose_for_pf.copy()
                elif not np.array_equal(
                    np.asarray(estimator.poses[current_pose_idx], dtype=float),
                    pose_for_pf,
                ):
                    raise RuntimeError(
                        "Simulator detector pose changed within one PF station; "
                        "pure replay requires an exact station pose after the "
                        "first observation."
                    )
                meas = Measurement(
                    counts_by_isotope=z_k,
                    count_variance_by_isotope=z_variance_full,
                    pose_idx=current_pose_idx,
                    orient_idx=best_pair_idx,
                    live_time_s=actual_live_time_s,
                    fe_index=fe_idx,
                    pb_index=pb_idx,
                    RFe=RFe_sel,
                    RPb=RPb_sel,
                    detector_position=pose_for_pf,
                )
                last_measurement_for_diagnostics = meas
                if measurement_log_writer is not None:
                    log_covariance = np.asarray(
                        [
                            [
                                z_covariance_full[row_isotope][column_isotope]
                                for column_isotope in isotopes
                            ]
                            for row_isotope in isotopes
                        ],
                        dtype=np.float64,
                    )
                    measurement_log_writer.append_before_update(
                        MeasurementLogRecord(
                            step_id=int(step_counter),
                            action_id=int(step_counter),
                            station_id=int(pose_counter),
                            detector_pose_xyz=tuple(
                                float(value) for value in pose_for_pf
                            ),
                            detector_quat_wxyz=tuple(
                                float(value) for value in observation.detector_quat_wxyz
                            ),
                            fe_orientation_index=int(fe_idx),
                            pb_orientation_index=int(pb_idx),
                            live_time_s=float(actual_live_time_s),
                            travel_time_s=float(step_motion_time_s),
                            shield_actuation_time_s=float(step_rotation_time_s),
                            energy_bin_edges_keV=np.asarray(
                                observation.energy_bin_edges_keV,
                                dtype=np.float64,
                            ),
                            spectrum_counts=np.asarray(spectrum, dtype=np.float64),
                            spectrum_variance=(
                                None
                                if spectrum_variance is None
                                else np.asarray(
                                    spectrum_variance,
                                    dtype=np.float64,
                                )
                            ),
                            isotope_counts=dict(history_z_k),
                            isotope_count_covariance=log_covariance,
                            metadata={
                                "backend": str(
                                    observation.metadata.get("backend", sim_backend)
                                ),
                                "spectrum_count_method": str(spectrum_count_method),
                                "dwell_ready_reason": str(dwell_ready_reason),
                                "dwell_chunks": int(dwell_chunks),
                                **_measurement_transport_provenance(
                                    observation.metadata
                                ),
                            },
                        )
                    )
                pf_start = time.perf_counter()
                if joint_observation_update:
                    joint_record: tuple[object, ...] = (
                        dict(history_z_k),
                        int(fe_idx),
                        int(pb_idx),
                        float(actual_live_time_s),
                        dict(history_z_variance),
                    )
                    joint_record = (
                        *joint_record,
                        {
                            row_iso: dict(row_payload)
                            for row_iso, row_payload in z_covariance_full.items()
                        },
                    )
                    if spectrum_payload is not None:
                        joint_record = (*joint_record, spectrum_payload)
                    joint_update_records.append(joint_record)
                else:
                    update_kwargs: dict[str, object] = {
                        "z_covariance_k": z_covariance_full,
                    }
                    if spectrum_payload is not None and _callable_accepts_keyword(
                        estimator.update_pair,
                        "spectrum_payload",
                    ):
                        update_kwargs["spectrum_payload"] = spectrum_payload
                    estimator.update_pair(
                        z_k=history_z_k,
                        pose_idx=current_pose_idx,
                        fe_index=fe_idx,
                        pb_index=pb_idx,
                        live_time_s=actual_live_time_s,
                        z_variance_k=history_z_variance,
                        **update_kwargs,
                    )
                    if delayed_resample_update:
                        deferred_update_records += 1
                pf_elapsed = time.perf_counter() - pf_start
                if not joint_observation_update:
                    total_pf_wall_s += pf_elapsed
                    pf_wall_samples_s.append(float(pf_elapsed))
                measurement_live_times_s.append(float(actual_live_time_s))
                elapsed += actual_live_time_s
                viz_elapsed = 0.0
                viz_start = time.perf_counter()
                frame = build_frame_from_pf(
                    estimator,
                    meas,
                    step_index=step_counter,
                    time_sec=elapsed,
                    estimate_mode=estimate_mode,
                    min_est_strength=estimate_min_strength,
                    min_existence_prob=estimate_min_existence_prob,
                )
                if step_path_segment is not None:
                    waypoints_payload = step_path_segment.get("waypoints_xyz")
                    if waypoints_payload is not None:
                        waypoint_array = np.asarray(
                            waypoints_payload,
                            dtype=float,
                        )
                        if isinstance(frame, dict):
                            frame["path_waypoints_xyz"] = waypoint_array
                        else:
                            frame.path_waypoints_xyz = waypoint_array
                spectrum_components_payload = {
                    iso: component.copy()
                    for iso, component in last_spectrum_components.items()
                }
                if isinstance(frame, dict):
                    frame["spectrum_energy_keV"] = decomposer.energy_axis.copy()
                    frame["spectrum_counts"] = spectrum.copy()
                    frame["spectrum_components_by_isotope"] = (
                        spectrum_components_payload
                    )
                else:
                    frame.spectrum_energy_keV = decomposer.energy_axis.copy()
                    frame.spectrum_counts = spectrum.copy()
                    frame.spectrum_components_by_isotope = spectrum_components_payload
                viz_elapsed += time.perf_counter() - viz_start
                viz_start = time.perf_counter()
                viz.update(frame)
                if cui_split_viz is not None:
                    cui_split_viz.update(frame)
                _send_isaac_pf_visualization(frame)
                last_frame = frame
                print(
                    f"[step {step_counter}] pose={_fmt_pos(pose_for_pf)} "
                    f"orient_pair={best_pair_idx} "
                    f"planned_pair={using_planned_pair} "
                    f"shield_gain={shield_gain:.6g} raw_ig={raw_ig_val:.6g} "
                    f"signature={signature_val:.6g} "
                    f"signature_utility={signature_utility_val:.6g} "
                    f"low_count_penalty={low_count_penalty:.6g} "
                    f"count_balance_penalty={count_balance_penalty:.6g} "
                    f"rotation_cost={rotation_cost:.6g} "
                    f"shield_gain_rate={shield_gain_rate:.6g} "
                    f"next_pose_gain={next_pose_gain:.6g} "
                    f"next_pose_gain_rate={next_pose_gain_rate:.6g} "
                    f"stop_reason={stop_reason} "
                    f"ig_threshold={ig_threshold_current:.6g} "
                    f"fe_idx={fe_idx} pb_idx={pb_idx} "
                    f"travel_distance_m={step_motion_distance_m:.3f} "
                    f"travel_time_s={step_motion_time_s:.1f} "
                    f"shield_time_s={step_rotation_time_s:.1f} "
                    f"live_time_s={actual_live_time_s:.1f}/{dwell_step_label} "
                    f"dwell_chunks={dwell_chunks} "
                    f"dwell_reason={dwell_ready_reason} "
                    f"cardinality_ready={cardinality_ready} "
                    f"cardinality_reason={cardinality_reason} "
                    f"mission_time_s={elapsed:.1f} "
                    f"z_keys={sorted(z_k.keys())} "
                    f"z_obs={_fmt_counts(z_counts)}"
                )
                if live:
                    plt.pause(0.05)
                viz_elapsed += time.perf_counter() - viz_start
                total_viz_wall_s += viz_elapsed
                _log_pf_diagnostics(estimator, step_counter)
                _log_precision_degradation_diagnostics(
                    estimator,
                    decomposer,
                    last_measurement_for_diagnostics,
                    true_src,
                    true_strengths,
                    env,
                    obstacle_grid,
                    obstacle_height_m=float(
                        runtime_config.get("obstacle_height_m", 2.0)
                    ),
                    step_index=step_counter,
                    candidate_log_limit=precision_diagnostic_birth_candidate_log_limit,
                    particle_log_limit=precision_diagnostic_particle_log_limit,
                    source_event_log_limit=(
                        precision_diagnostic_source_event_log_limit
                    ),
                    birth_candidate_diagnostics_enabled=(
                        precision_diagnostic_birth_candidate_enable
                    ),
                    full_spectrum_response_diagnostics_enabled=(
                        precision_diagnostic_full_spectrum_response_enable
                    ),
                )
                if estimate_trace_out_path is not None or estimate_trace_log_enabled:
                    _emit_intermediate_estimate_trace(
                        estimator,
                        isotopes,
                        frame,
                        true_src,
                        true_strengths,
                        env,
                        obstacle_grid,
                        step_index=step_counter,
                        elapsed_s=elapsed,
                        counts_by_isotope={
                            str(key): float(value) for key, value in z_counts.items()
                        },
                        obstacle_height_m=float(
                            runtime_config.get("obstacle_height_m", 2.0)
                        ),
                        match_radius_m=float(eval_match_radius_m),
                        trace_path=estimate_trace_out_path,
                        log_enabled=estimate_trace_log_enabled,
                        log_every=estimate_trace_log_every,
                        max_log_records=estimate_trace_max_log_records,
                        estimate_source="current_pf_posterior",
                    )
                print(
                    f"[timing step {step_counter}] ig={ig_elapsed:.3f}s pf={pf_elapsed:.3f}s "
                    f"viz={viz_elapsed:.3f}s "
                    f"travel={step_motion_time_s:.1f}s "
                    f"shield={step_rotation_time_s:.1f}s "
                    f"live={actual_live_time_s:.1f}s"
                )
                step_counter += 1
                rotation_count += 1
                remaining_orientations.discard(best_pair_idx)
                current_shield_pair_id = int(best_pair_idx)
                executed_signature_vectors.append(signature_vector.copy())
                if (
                    save_outputs
                    and last_spectrum is not None
                    and step_counter % spectrum_plot_save_every == 0
                ):
                    highlight = set(last_candidates)
                    spectrum_path = (
                        SPECTRUM_DIR / f"spectrum_step_{step_counter:04d}.png"
                    )
                    _save_spectrum_plot(
                        decomposer,
                        last_spectrum,
                        spectrum_path,
                        highlight_isotopes=highlight,
                        counts_by_isotope=last_counts,
                        component_spectra_by_isotope=last_spectrum_components,
                        title=f"Processed measurement spectrum (step {step_counter})",
                    )
                if max_steps is not None and step_counter >= max_steps:
                    stop_run = True
                    break
                pose_elapsed += actual_live_time_s + step_rotation_time_s
                if pose_elapsed >= estimator.pf_config.max_dwell_time_s:
                    break
            if (
                measurement_log_writer is not None
                and rotation_count > 0
                and not resume_station_boundary
            ):
                measurement_log_writer.mark_station_complete_before_update(
                    int(pose_counter),
                    completion_metadata={
                        _LIVE_CONTROLLER_CHECKPOINT_KEY: (
                            _build_live_controller_checkpoint(
                                planning_candidate_rng=planning_candidate_rng,
                                planning_candidate_parameters=(
                                    planning_candidate_checkpoint_parameters
                                ),
                                remaining_measurement_estimates=(
                                    remaining_measurement_estimates
                                ),
                                estimator=estimator,
                                max_poses=max_poses,
                                soft_pose_extension_used=(
                                    soft_pose_extension_used
                                ),
                                ig_max_global=ig_max_global,
                                ig_max_pose=ig_max_pose,
                                ig_threshold_current=ig_threshold_current,
                                last_max_ig=last_max_ig,
                            )
                        )
                    },
                )
            if delayed_resample_update:
                pf_start = time.perf_counter()
                finalized_measurements = estimator.finalize_deferred_pose_update()
                pf_elapsed = time.perf_counter() - pf_start
                if finalized_measurements > 0:
                    total_pf_wall_s += pf_elapsed
                    per_measurement_pf = pf_elapsed / max(finalized_measurements, 1)
                    pf_wall_samples_s.extend(
                        [float(per_measurement_pf)] * finalized_measurements
                    )
                    print(
                        f"[pose {current_pose_idx}] delayed_pf_finalize "
                        f"measurements={finalized_measurements} "
                        f"likelihood_updates={deferred_update_records} "
                        f"pf={pf_elapsed:.3f}s "
                        f"per_measurement={per_measurement_pf:.3f}s"
                    )
                    finalize_step_index = max(step_counter - 1, 0)
                    _log_pf_diagnostics(estimator, finalize_step_index)
                    _log_precision_degradation_diagnostics(
                        estimator,
                        decomposer,
                        last_measurement_for_diagnostics,
                        true_src,
                        true_strengths,
                        env,
                        obstacle_grid,
                        obstacle_height_m=float(
                            runtime_config.get("obstacle_height_m", 2.0)
                        ),
                        step_index=finalize_step_index,
                        candidate_log_limit=(
                            precision_diagnostic_birth_candidate_log_limit
                        ),
                        particle_log_limit=precision_diagnostic_particle_log_limit,
                        source_event_log_limit=(
                            precision_diagnostic_source_event_log_limit
                        ),
                        birth_candidate_diagnostics_enabled=(
                            precision_diagnostic_birth_candidate_enable
                        ),
                        full_spectrum_response_diagnostics_enabled=(
                            precision_diagnostic_full_spectrum_response_enable
                        ),
                    )
                    _log_surface_candidate_observability_diagnostics(
                        estimator,
                        finalize_step_index,
                        label=f"pose_{current_pose_idx}_finalize",
                        max_candidates=surface_observability_diagnostic_candidates,
                    )
                    if last_frame is not None and (
                        estimate_trace_out_path is not None
                        or estimate_trace_log_enabled
                    ):
                        _emit_intermediate_estimate_trace(
                            estimator,
                            isotopes,
                            last_frame,
                            true_src,
                            true_strengths,
                            env,
                            obstacle_grid,
                            step_index=finalize_step_index,
                            elapsed_s=elapsed,
                            counts_by_isotope=dict(last_counts or {}),
                            obstacle_height_m=float(
                                runtime_config.get("obstacle_height_m", 2.0)
                            ),
                            match_radius_m=float(eval_match_radius_m),
                            trace_path=estimate_trace_out_path,
                            log_enabled=estimate_trace_log_enabled,
                            log_every=estimate_trace_log_every,
                            max_log_records=estimate_trace_max_log_records,
                            estimate_source="post_finalize_pf_posterior",
                        )
            elif joint_observation_update and joint_update_records:
                pf_start = time.perf_counter()
                estimator.update_pair_sequence(
                    joint_update_records,
                    pose_idx=current_pose_idx,
                )
                pf_elapsed = time.perf_counter() - pf_start
                total_pf_wall_s += pf_elapsed
                per_measurement_pf = pf_elapsed / max(len(joint_update_records), 1)
                pf_wall_samples_s.extend(
                    [float(per_measurement_pf)] * len(joint_update_records)
                )
                print(
                    f"[pose {current_pose_idx}] joint_pf_update "
                    f"measurements={len(joint_update_records)} "
                    f"pf={pf_elapsed:.3f}s "
                    f"per_measurement={per_measurement_pf:.3f}s "
                    "isotope_update_workers="
                    f"{int(getattr(estimator, 'last_pair_sequence_update_workers', 1))} "
                    "isotope_update_wall="
                    f"{float(getattr(estimator, 'last_pair_sequence_update_wall_s', 0.0)):.3f}s "
                    "structural_workers="
                    f"{int(getattr(estimator, 'last_structural_update_workers', 1))} "
                    "structural_wall="
                    f"{float(getattr(estimator, 'last_structural_update_wall_s', 0.0)):.3f}s"
                )
                pair_stage_wall = getattr(
                    estimator,
                    "last_pair_sequence_stage_wall_s",
                    {},
                )
                if isinstance(pair_stage_wall, dict) and pair_stage_wall:
                    print(
                        f"[pose {current_pose_idx}] joint_pf_update_stages "
                        + " ".join(
                            f"{key}={float(value):.3f}s"
                            for key, value in sorted(pair_stage_wall.items())
                        )
                    )
                joint_step_index = max(step_counter - 1, 0)
                _log_pf_diagnostics(estimator, joint_step_index)
                _log_precision_degradation_diagnostics(
                    estimator,
                    decomposer,
                    last_measurement_for_diagnostics,
                    true_src,
                    true_strengths,
                    env,
                    obstacle_grid,
                    obstacle_height_m=float(
                        runtime_config.get("obstacle_height_m", 2.0)
                    ),
                    step_index=joint_step_index,
                    candidate_log_limit=precision_diagnostic_birth_candidate_log_limit,
                    particle_log_limit=precision_diagnostic_particle_log_limit,
                    source_event_log_limit=(
                        precision_diagnostic_source_event_log_limit
                    ),
                    birth_candidate_diagnostics_enabled=(
                        precision_diagnostic_birth_candidate_enable
                    ),
                    full_spectrum_response_diagnostics_enabled=(
                        precision_diagnostic_full_spectrum_response_enable
                    ),
                )
                _log_surface_candidate_observability_diagnostics(
                    estimator,
                    joint_step_index,
                    label=f"pose_{current_pose_idx}_joint_update",
                    max_candidates=surface_observability_diagnostic_candidates,
                )
                if last_frame is not None and (
                    estimate_trace_out_path is not None or estimate_trace_log_enabled
                ):
                    _emit_intermediate_estimate_trace(
                        estimator,
                        isotopes,
                        last_frame,
                        true_src,
                        true_strengths,
                        env,
                        obstacle_grid,
                        step_index=joint_step_index,
                        elapsed_s=elapsed,
                        counts_by_isotope=dict(last_counts or {}),
                        obstacle_height_m=float(
                            runtime_config.get("obstacle_height_m", 2.0)
                        ),
                        match_radius_m=float(eval_match_radius_m),
                        trace_path=estimate_trace_out_path,
                        log_enabled=estimate_trace_log_enabled,
                        log_every=estimate_trace_log_every,
                        max_log_records=estimate_trace_max_log_records,
                        estimate_source="post_joint_update_pf_posterior",
                    )
            if (
                save_outputs
                and not resume_station_boundary
                and estimator.measurements
                and estimator.measurements[-1].pose_idx == current_pose_idx
                and pf_plot_save_every > 0
                and (current_pose_idx + 1) % pf_plot_save_every == 0
            ):
                pf_step = current_pose_idx + 1
                pf_path = PF_DIR / f"pf_step_{pf_step:03d}.png"
                viz.save_final(pf_path.as_posix())
            if stop_run:
                print(f"Reached max steps ({max_steps}); stopping exploration.")
                break
            if (
                not used_planned_program_this_pose
                and last_max_ig is not None
                and last_max_ig < ig_threshold_current
                and (pose_counter + 1) >= mission_stop_min_convergence_poses
            ):
                remaining_stop_estimate = (
                    remaining_measurement_estimates[-1]
                    if (
                        bool(remaining_measurement_config.enabled)
                        and remaining_measurement_estimates
                    )
                    else None
                )
                cardinality_ready, cardinality_reason = (
                    _source_cardinality_dwell_status(
                        estimator,
                        refresh_estimates=False,
                    )
                )
                if (
                    (
                        not bool(mission_stop_require_pf_cardinality_ready)
                        or cardinality_ready
                    )
                    and (
                        not bool(mission_stop_require_remaining_measurement_ready)
                        or _remaining_measurement_ready_for_stop(
                            remaining_stop_estimate
                        )
                    )
                    and estimator.should_stop_exploration(
                        ig_threshold=ig_threshold_current,
                        live_time_s=planning_live_time,
                    )
                ):
                    print(
                        "Converged; stopping exploration "
                        f"(max IG {last_max_ig:.6g} < threshold "
                        f"{ig_threshold_current:.6g}; "
                        f"cardinality={cardinality_reason})."
                    )
                    break
            visited_poses.append(pose.copy())
            pose_counter += 1
            if adaptive_mission_stop:
                remaining_stop_estimate = (
                    remaining_measurement_estimates[-1]
                    if (
                        bool(remaining_measurement_config.enabled)
                        and remaining_measurement_estimates
                    )
                    else None
                )
                stop_reason = _adaptive_mission_stop_reason(
                    estimator,
                    current_pose_idx=current_pose_idx,
                    visited_poses_xyz=visited_poses,
                    map_api=planning_map,
                    min_poses=mission_stop_min_convergence_poses,
                    coverage_radius_m=mission_stop_coverage_radius_m,
                    coverage_fraction_threshold=mission_stop_coverage_fraction,
                    ig_threshold=ig_threshold_current,
                    planning_live_time_s=planning_live_time,
                    require_quiet_birth_residual=mission_stop_require_quiet_birth_residual,
                    birth_residual_min_support=mission_stop_birth_residual_min_support,
                    require_pf_convergence_for_coverage=(
                        mission_stop_require_pf_convergence_for_coverage
                    ),
                    require_no_unresolved_discriminative_failures=(
                        mission_stop_require_no_unresolved_discriminative_failures
                    ),
                    unresolved_discriminative_failure_min_count=(
                        mission_stop_unresolved_discriminative_fail_min_count
                    ),
                    require_pf_cardinality_ready=(
                        mission_stop_require_pf_cardinality_ready
                    ),
                    remaining_measurement_estimate=remaining_stop_estimate,
                    require_remaining_measurement_ready=(
                        bool(mission_stop_require_remaining_measurement_ready)
                        and bool(remaining_measurement_config.enabled)
                    ),
                )
                if stop_reason is not None:
                    print(f"Adaptive mission stop: {stop_reason}.")
                    break
            if max_poses is not None and pose_counter >= max_poses:
                remaining_stop_estimate = (
                    remaining_measurement_estimates[-1]
                    if (
                        bool(remaining_measurement_config.enabled)
                        and remaining_measurement_estimates
                    )
                    else None
                )
                remaining_unresolved = not _remaining_measurement_ready_for_stop(
                    remaining_stop_estimate
                )
                pf_cardinality_ready, pf_cardinality_reason = (
                    _source_cardinality_dwell_status(
                        estimator,
                        refresh_estimates=False,
                    )
                )
                pf_cardinality_unresolved = not pf_cardinality_ready
                can_extend = (
                    bool(mission_stop_soft_extend_on_unresolved)
                    and soft_pose_extension_used
                    < int(mission_stop_soft_extension_poses)
                    and (remaining_unresolved or pf_cardinality_unresolved)
                )
                remaining_progress = {}
                if (
                    can_extend
                    and mission_stop_soft_extension_require_progress
                    and not pf_cardinality_unresolved
                ):
                    remaining_progress = _remaining_measurement_progress(
                        remaining_measurement_estimates
                    )
                    can_extend = bool(remaining_progress.get("has_progress", False))
                    if not can_extend:
                        print(
                            "Soft extension denied because the PF posterior "
                            "remaining-measurement budget did not improve: "
                            f"remaining_progress={remaining_progress}"
                        )
                if can_extend:
                    requested_extension = 1
                    if isinstance(remaining_stop_estimate, Mapping):
                        requested_extension = max(
                            requested_extension,
                            int(
                                remaining_stop_estimate.get(
                                    "estimated_remaining_stations",
                                    1,
                                )
                            ),
                        )
                    extension = min(
                        requested_extension,
                        int(mission_stop_soft_extension_poses)
                        - int(soft_pose_extension_used),
                    )
                    if extension > 0:
                        soft_pose_extension_used += int(extension)
                        max_poses += int(extension)
                        print(
                            "Soft-extending max poses for unresolved PF structure: "
                            f"extension={int(extension)} "
                            f"new_max_poses={int(max_poses)} "
                            f"remaining_unresolved={bool(remaining_unresolved)} "
                            "pf_cardinality_unresolved="
                            f"{bool(pf_cardinality_unresolved)} "
                            f"pf_cardinality_reason={pf_cardinality_reason} "
                            f"remaining_progress={remaining_progress}"
                        )
                    else:
                        max_pose_stop_unresolved = bool(
                            remaining_unresolved or pf_cardinality_unresolved
                        )
                        max_pose_stop_diagnostics = {
                            "max_poses": int(max_poses),
                            "remaining_unresolved": bool(remaining_unresolved),
                            "pf_cardinality_unresolved": bool(
                                pf_cardinality_unresolved
                            ),
                            "pf_cardinality_reason": pf_cardinality_reason,
                            "soft_pose_extension_used": int(soft_pose_extension_used),
                            "soft_pose_extension_limit": int(
                                mission_stop_soft_extension_poses
                            ),
                            "soft_extension_remaining_progress": remaining_progress,
                        }
                        print(f"Reached max poses ({max_poses}); stopping exploration.")
                        break
                else:
                    max_pose_stop_unresolved = bool(
                        remaining_unresolved or pf_cardinality_unresolved
                    )
                    max_pose_stop_diagnostics = {
                        "max_poses": int(max_poses),
                        "remaining_unresolved": bool(remaining_unresolved),
                        "pf_cardinality_unresolved": bool(
                            pf_cardinality_unresolved
                        ),
                        "pf_cardinality_reason": pf_cardinality_reason,
                        "soft_pose_extension_used": int(soft_pose_extension_used),
                        "soft_pose_extension_limit": int(
                            mission_stop_soft_extension_poses
                        ),
                        "soft_extension_remaining_progress": remaining_progress,
                    }
                    print(f"Reached max poses ({max_poses}); stopping exploration.")
                    break
            visited_arr = np.vstack(visited_poses) if visited_poses else None
            previous_move_was_height_partner = _previous_move_was_height_partner(
                visited_arr,
                xy_tolerance_m=detector_height_pair_xy_tolerance_m,
                z_tolerance_m=detector_height_pair_z_tolerance_m,
                min_z_separation_m=detector_height_pair_min_separation_m,
            )
            allow_height_partner_first_action = bool(
                not previous_move_was_height_partner
            )
            print("Generating candidate poses for next measurement point...")
            candidates, relaxed_retry, candidate_min_dist = (
                _generate_planning_candidates(
                    current_pose_xyz=pose,
                    map_api=planning_map,
                    n_candidates=pose_candidates,
                    min_dist_from_visited=pose_min_dist,
                    visited_poses_xyz=visited_arr,
                    bounds_xyz=(bounds_lo, bounds_hi),
                    detector_heights_m=detector_height_candidates,
                    continuous_height_anchor_count=(
                        detector_continuous_height_partner_candidates
                    ),
                    height_partner_xy_tolerance_m=(detector_height_pair_xy_tolerance_m),
                    height_partner_z_tolerance_m=(detector_height_pair_z_tolerance_m),
                    height_partner_min_z_separation_m=(
                        detector_height_pair_min_separation_m
                    ),
                    rng=planning_candidate_rng,
                )
            )
            if relaxed_retry:
                print(
                    "Insufficient reachable lateral candidates with current spacing; "
                    f"retrying with min_dist={candidate_min_dist:.2f}."
                )
            if candidates.size == 0:
                print("No candidate poses available; stopping exploration.")
                break
            candidate_xy_distance = np.linalg.norm(
                candidates[:, :2] - pose[None, :2],
                axis=1,
            )
            lateral_candidate_count = int(
                np.count_nonzero(
                    candidate_xy_distance
                    > max(float(detector_height_pair_xy_tolerance_m), 1.0e-9)
                )
            )
            print(
                f"Generated {len(candidates)} candidate poses "
                f"(lateral={lateral_candidate_count}, "
                f"height={len(candidates) - lateral_candidate_count}). "
                "Computing best next pose..."
            )
            planned_program_for_next: tuple[int, ...] | None = None
            dss_diagnostics: dict[str, Any] | None = None
            dss_first_node = None
            height_partner_program_for_scoring = _height_partner_program_for_scoring(
                reuse_enabled=height_partner_reuse_shield_program,
                executed_pair_ids=executed_pair_ids_this_pose,
                baseline_shield_policy=baseline_shield_policy,
            )
            baseline_path_selection = select_baseline_next_pose(
                baseline_path_policy,
                candidate_poses_xyz=candidates,
                current_pose_xyz=pose,
                visited_poses_xyz=visited_arr,
                bounds_xyz=(bounds_lo, bounds_hi),
            )
            if baseline_path_selection is not None:
                next_pose = baseline_path_selection.next_pose
                print(
                    "Baseline path policy selected next station: "
                    f"policy={baseline_path_selection.name} "
                    f"idx={baseline_path_selection.candidate_index} "
                    f"score={baseline_path_selection.score:.6g} "
                    f"pose={next_pose.tolist()}"
                )
                forced_baseline_program, _, _ = (
                    _forced_baseline_program_for_planned_station(
                        label="baseline_path_fixed_station",
                    )
                )
                if forced_baseline_program is not None:
                    planned_program_for_next = forced_baseline_program
                elif baseline_shield_policy is None:
                    dss_selection_config = dss_config
                    residual_burst_active = _has_birth_residual_evidence(
                        estimator,
                        min_support=max(
                            1,
                            int(estimator.pf_config.birth_residual_min_support),
                        ),
                    )
                    dss_selection_config = _adaptive_dss_selection_config(
                        dss_selection_config,
                        residual_burst_active=residual_burst_active,
                        remaining_estimate=_latest_remaining_estimate_for_planning(),
                        label="baseline_path_fixed_station",
                    )
                    dss_start = time.perf_counter()
                    dss_result = select_dss_pp_next_station(
                        estimator=estimator,
                        candidate_poses_xyz=np.asarray([next_pose], dtype=float),
                        current_pose_xyz=pose,
                        current_pair_id=current_shield_pair_id,
                        visited_poses_xyz=visited_arr,
                        map_api=planning_map,
                        bounds_xyz=(bounds_lo, bounds_hi),
                        continuous_height_bounds_m=(continuous_height_bounds_for_dss),
                        config=dss_selection_config,
                        height_partner_forced_program_pair_ids=(
                            height_partner_program_for_scoring
                        ),
                        height_partner_xy_tolerance_m=(
                            detector_height_pair_xy_tolerance_m
                        ),
                        height_partner_z_tolerance_m=(
                            detector_height_pair_z_tolerance_m
                        ),
                        height_partner_min_z_separation_m=(
                            detector_height_pair_min_separation_m
                        ),
                        allow_height_partner_first_action=(
                            allow_height_partner_first_action
                        ),
                    )
                    dss_elapsed = time.perf_counter() - dss_start
                    total_path_planning_wall_s += float(dss_elapsed)
                    path_planning_wall_samples_s.append(float(dss_elapsed))
                    planned_program_for_next = tuple(
                        int(pair_id) for pair_id in dss_result.shield_program.pair_ids
                    )
                    dss_diagnostics = dict(dss_result.diagnostics)
                    dss_first_node = (
                        dss_result.sequence[0] if dss_result.sequence else None
                    )
                    print(
                        "DSS-PP fixed-station shield program: "
                        f"program={dss_result.shield_program.name} "
                        f"pairs={list(planned_program_for_next)} "
                        f"score={float(dss_result.score):.6g} "
                        f"signature={float(dss_result.sequence[0].signature_score):.6g} "
                        "temporal_sep="
                        f"{float(dss_result.sequence[0].temporal_separation_score):.6g} "
                        f"workers={int(dss_result.diagnostics.get('program_eval_workers', 1))} "
                        f"compute={dss_elapsed:.3f}s"
                    )
                    _log_dss_ranked_node_diagnostics(
                        dss_diagnostics,
                        label=f"pose_{current_pose_idx}_baseline_path_fixed_station",
                    )
                    _log_dss_pairwise_ambiguity_diagnostics(
                        dss_diagnostics,
                        label=f"pose_{current_pose_idx}_baseline_path_fixed_station",
                    )
                    _log_dss_component_leader_diagnostics(
                        dss_diagnostics,
                        label=f"pose_{current_pose_idx}_baseline_path_fixed_station",
                    )
            elif path_planner_resolved == "dss_pp":
                dss_selection_config = dss_config
                remaining_guidance_estimate = None
                if (
                    bool(dss_selection_config.remaining_budget_guidance)
                    and bool(remaining_measurement_config.enabled)
                    and estimator.measurements
                ):
                    fallback_pair = (
                        0
                        if current_shield_pair_id is None
                        else int(current_shield_pair_id)
                    )
                    remaining_guidance_estimate = estimate_remaining_measurement_budget(
                        estimator,
                        shield_program_pair_ids=(fallback_pair,),
                        live_time_s=planning_live_time,
                        config=remaining_measurement_config,
                        current_station_count=pose_counter,
                        update_history=False,
                    )
                    dss_selection_config = replace(
                        dss_selection_config,
                        remaining_station_estimate=(
                            remaining_guidance_estimate.estimated_remaining_stations
                        ),
                    )
                    print(
                        "DSS-PP remaining-budget guidance: "
                        "estimated_remaining_stations="
                        f"{remaining_guidance_estimate.estimated_remaining_stations} "
                        f"range={remaining_guidance_estimate.estimated_remaining_station_low}-"
                        f"{remaining_guidance_estimate.estimated_remaining_station_high} "
                        f"bottleneck={remaining_guidance_estimate.bottleneck}"
                    )
                    _log_remaining_measurement_detail(
                        remaining_guidance_estimate,
                        label=f"pose_{current_pose_idx}_guidance",
                    )
                residual_burst_active = _has_birth_residual_evidence(
                    estimator,
                    min_support=max(
                        1,
                        int(estimator.pf_config.birth_residual_min_support),
                    ),
                )
                dss_selection_config = _adaptive_dss_selection_config(
                    dss_selection_config,
                    residual_burst_active=residual_burst_active,
                    remaining_estimate=_latest_remaining_estimate_for_planning(
                        remaining_guidance_estimate
                    ),
                    label="next_station",
                )
                dss_selection_config, baseline_program_for_planning = (
                    _apply_baseline_shield_program_to_dss_config(
                        dss_selection_config,
                        baseline_shield_policy,
                        total_pairs=total_pairs,
                        pose_index=pose_counter,
                        current_pair_id=current_shield_pair_id,
                    )
                )
                if baseline_program_for_planning is not None:
                    forced_pairs = tuple(
                        int(pair_id)
                        for pair_id in baseline_program_for_planning.pair_ids
                    )
                    print(
                        "DSS-PP scoring forced baseline shield program: "
                        f"{baseline_program_for_planning.name} "
                        f"pairs={list(forced_pairs)} "
                        f"program_length={int(dss_selection_config.program_length)}"
                    )
                dss_start = time.perf_counter()
                dss_result = select_dss_pp_next_station(
                    estimator=estimator,
                    candidate_poses_xyz=candidates,
                    current_pose_xyz=pose,
                    current_pair_id=current_shield_pair_id,
                    visited_poses_xyz=visited_arr,
                    map_api=planning_map,
                    bounds_xyz=(bounds_lo, bounds_hi),
                    continuous_height_bounds_m=continuous_height_bounds_for_dss,
                    config=dss_selection_config,
                    height_partner_forced_program_pair_ids=(
                        height_partner_program_for_scoring
                    ),
                    height_partner_xy_tolerance_m=(detector_height_pair_xy_tolerance_m),
                    height_partner_z_tolerance_m=(detector_height_pair_z_tolerance_m),
                    height_partner_min_z_separation_m=(
                        detector_height_pair_min_separation_m
                    ),
                    allow_height_partner_first_action=(
                        allow_height_partner_first_action
                    ),
                )
                dss_elapsed = time.perf_counter() - dss_start
                total_path_planning_wall_s += float(dss_elapsed)
                path_planning_wall_samples_s.append(float(dss_elapsed))
                next_pose = dss_result.next_pose
                planned_program_for_next = tuple(
                    int(pair_id) for pair_id in dss_result.shield_program.pair_ids
                )
                dss_diagnostics = dict(dss_result.diagnostics)
                dss_first_node = dss_result.sequence[0] if dss_result.sequence else None
                if bool(dss_one_step_guard_enabled):
                    guard_start = time.perf_counter()
                    diagnostic_guard = _best_dss_first_step_guard_candidate(
                        dss_diagnostics,
                        candidate_poses_xyz=candidates,
                    )
                    guard_source = "dss_ranked_nodes"
                    if diagnostic_guard is None and dss_first_node is not None:
                        try:
                            first_pose = np.asarray(
                                dss_first_node.pose_xyz, dtype=float
                            ).reshape(-1)
                            first_score = float(dss_first_node.score)
                            if first_pose.size == 3 and np.isfinite(first_score):
                                diagnostic_guard = (
                                    int(dss_first_node.pose_index),
                                    first_score,
                                    first_pose.copy(),
                                )
                                guard_source = "dss_first_node"
                        except (TypeError, ValueError, AttributeError):
                            diagnostic_guard = None
                    guard_program_length = max(
                        1,
                        int(
                            len(planned_program_for_next)
                            if planned_program_for_next
                            else dss_selection_config.program_length
                        ),
                    )
                    if diagnostic_guard is None:
                        guard_source = "after_rotation_fallback"
                        guard_idx, guard_elapsed = _select_one_step_pose_for_planning(
                            candidate_poses_xyz=candidates,
                            current_pose_xyz=pose,
                            program_length_budget=guard_program_length,
                            use_gpu=dss_one_step_guard_use_gpu,
                        )
                        guard_candidate_score = -np.inf
                        guard_pose = np.asarray(candidates[int(guard_idx)], dtype=float)
                    else:
                        guard_idx, guard_candidate_score, guard_pose = diagnostic_guard
                        guard_elapsed = time.perf_counter() - guard_start
                        total_path_planning_wall_s += float(guard_elapsed)
                    same_guard_pose = bool(
                        np.allclose(guard_pose, np.asarray(next_pose, dtype=float))
                    )
                    guard_dss_result = None
                    guard_first_score = float(guard_candidate_score)
                    selected_first_score = (
                        float(dss_first_node.score)
                        if dss_first_node is not None
                        else float(dss_result.score)
                    )
                    score_threshold = (
                        selected_first_score
                        + float(dss_one_step_guard_abs_margin)
                        + abs(selected_first_score)
                        * float(dss_one_step_guard_rel_margin)
                    )
                    should_verify_guard = bool(
                        not same_guard_pose
                        and (
                            guard_source == "after_rotation_fallback"
                            or guard_first_score > score_threshold
                        )
                    )
                    if should_verify_guard:
                        guard_dss_start = time.perf_counter()
                        guard_dss_result = select_dss_pp_next_station(
                            estimator=estimator,
                            candidate_poses_xyz=np.asarray([guard_pose], dtype=float),
                            current_pose_xyz=pose,
                            current_pair_id=current_shield_pair_id,
                            visited_poses_xyz=visited_arr,
                            map_api=planning_map,
                            bounds_xyz=(bounds_lo, bounds_hi),
                            continuous_height_bounds_m=(
                                continuous_height_bounds_for_dss
                            ),
                            config=dss_selection_config,
                            height_partner_forced_program_pair_ids=(
                                height_partner_program_for_scoring
                            ),
                            height_partner_xy_tolerance_m=(
                                detector_height_pair_xy_tolerance_m
                            ),
                            height_partner_z_tolerance_m=(
                                detector_height_pair_z_tolerance_m
                            ),
                            height_partner_min_z_separation_m=(
                                detector_height_pair_min_separation_m
                            ),
                            allow_height_partner_first_action=(
                                allow_height_partner_first_action
                            ),
                        )
                        guard_dss_elapsed = time.perf_counter() - guard_dss_start
                        total_path_planning_wall_s += float(guard_dss_elapsed)
                        path_planning_wall_samples_s.append(float(guard_dss_elapsed))
                        guard_first_node = (
                            guard_dss_result.sequence[0]
                            if guard_dss_result.sequence
                            else None
                        )
                        guard_first_score = (
                            float(guard_first_node.score)
                            if guard_first_node is not None
                            else float(guard_dss_result.score)
                        )
                    guard_selected = bool(
                        guard_dss_result is not None
                        and guard_first_score > score_threshold
                    )
                    if guard_selected and guard_dss_result is not None:
                        print(
                            "DSS-PP one-step guard selected local station: "
                            f"source={guard_source} "
                            f"one_step_idx={guard_idx} "
                            f"one_step_pose={guard_pose.tolist()} "
                            f"one_step_score={guard_first_score:.6g} "
                            f"dss_first_score={selected_first_score:.6g} "
                            f"guard_compute={guard_elapsed:.3f}s"
                        )
                        dss_result = guard_dss_result
                        next_pose = dss_result.next_pose
                        planned_program_for_next = tuple(
                            int(pair_id)
                            for pair_id in dss_result.shield_program.pair_ids
                        )
                        dss_diagnostics = dict(dss_result.diagnostics)
                        dss_first_node = (
                            dss_result.sequence[0] if dss_result.sequence else None
                        )
                    else:
                        print(
                            "DSS-PP one-step guard kept DSS station: "
                            f"source={guard_source} "
                            f"one_step_idx={guard_idx} "
                            f"same_pose={same_guard_pose} "
                            f"one_step_score={guard_first_score:.6g} "
                            f"dss_first_score={selected_first_score:.6g} "
                            f"threshold={score_threshold:.6g} "
                            f"guard_compute={guard_elapsed:.3f}s"
                        )
                    dss_diagnostics["one_step_guard_enabled"] = True
                    dss_diagnostics["one_step_guard_selected"] = guard_selected
                    dss_diagnostics["one_step_guard_source"] = guard_source
                    dss_diagnostics["one_step_guard_index"] = int(guard_idx)
                    dss_diagnostics["one_step_guard_pose"] = [
                        float(value) for value in guard_pose
                    ]
                    dss_diagnostics["one_step_guard_score"] = float(guard_first_score)
                    dss_diagnostics["one_step_guard_selected_first_score"] = float(
                        selected_first_score
                    )
                print(
                    "DSS-PP selected next station: "
                    f"pose={next_pose.tolist()} "
                    f"program={dss_result.shield_program.name} "
                    f"pairs={list(planned_program_for_next)} "
                    f"score={float(dss_result.score):.6g} "
                    f"signature={float(dss_result.sequence[0].signature_score):.6g} "
                    f"temporal_sep={float(dss_result.sequence[0].temporal_separation_score):.6g} "
                    f"elevation_sep={float(dss_result.sequence[0].elevation_signature_score):.6g} "
                    f"obs_penalty={float(dss_result.sequence[0].observation_penalty):.6g} "
                    f"diff_penalty={float(dss_result.sequence[0].differential_penalty):.6g} "
                    f"count_util={float(dss_result.sequence[0].count_utility):.6g} "
                    f"coverage_gain={float(dss_result.sequence[0].coverage_gain):.6g} "
                    f"revisit_penalty={float(dss_result.sequence[0].revisit_penalty):.6g} "
                    f"bearing_gain={float(dss_result.sequence[0].bearing_diversity_gain):.6g} "
                    f"frontier_gain={float(dss_result.sequence[0].frontier_gain):.6g} "
                    f"local_orbit={float(dss_result.sequence[0].local_orbit_gain):.6g} "
                    f"station_cond={float(dss_result.sequence[0].station_condition_gain):.6g} "
                    f"corr_reduction={float(dss_result.sequence[0].correlation_reduction_gain):.6g} "
                    f"isotope_balance={float(dss_result.sequence[0].isotope_balance_gain):.6g} "
                    f"elevation_cond={float(dss_result.sequence[0].elevation_condition_gain):.6g} "
                    f"env_sig={float(dss_result.sequence[0].environment_signature_score):.6g} "
                    f"vertical_env_sig={float(dss_result.sequence[0].vertical_environment_signature_score):.6g} "
                    f"occ_boundary={float(dss_result.sequence[0].occlusion_boundary_gain):.6g} "
                    f"turn_penalty={float(dss_result.sequence[0].turn_penalty):.6g} "
                    f"remaining_route_pressure={float(dss_result.sequence[0].remaining_route_pressure):.6g} "
                    f"remaining_route_penalty={float(dss_result.sequence[0].remaining_route_penalty):.6g} "
                    f"remaining_route_gain={float(dss_result.sequence[0].remaining_route_gain):.6g} "
                    f"planner_mode={dss_result.diagnostics.get('planner_mode', 'balanced')} "
                    f"workers={int(dss_result.diagnostics.get('program_eval_workers', 1))} "
                    f"compute={dss_elapsed:.3f}s"
                )
                _log_dss_ranked_node_diagnostics(
                    dss_diagnostics,
                    label=f"pose_{current_pose_idx}_next",
                )
                _log_dss_pairwise_ambiguity_diagnostics(
                    dss_diagnostics,
                    label=f"pose_{current_pose_idx}_next",
                )
                _log_dss_component_leader_diagnostics(
                    dss_diagnostics,
                    label=f"pose_{current_pose_idx}_next",
                )
            else:
                next_idx, one_step_elapsed = _select_one_step_pose_for_planning(
                    candidate_poses_xyz=candidates,
                    current_pose_xyz=pose,
                    program_length_budget=rotation_limit,
                    use_gpu=one_step_pose_eval_use_gpu,
                )
                next_pose = candidates[next_idx]
                print(
                    "One-step path policy selected next station: "
                    f"idx={int(next_idx)} pose={next_pose.tolist()} "
                    f"workers={int(pose_selection_workers_resolved)} "
                    f"compute={one_step_elapsed:.3f}s"
                )
                forced_baseline_program, _, _ = (
                    _forced_baseline_program_for_planned_station(
                        label="one_step_fixed_station",
                    )
                )
                if forced_baseline_program is not None:
                    planned_program_for_next = forced_baseline_program
                elif baseline_shield_policy is None:
                    dss_selection_config = dss_config
                    residual_burst_active = _has_birth_residual_evidence(
                        estimator,
                        min_support=max(
                            1,
                            int(estimator.pf_config.birth_residual_min_support),
                        ),
                    )
                    dss_selection_config = _adaptive_dss_selection_config(
                        dss_selection_config,
                        residual_burst_active=residual_burst_active,
                        remaining_estimate=_latest_remaining_estimate_for_planning(),
                        label="one_step_fixed_station",
                    )
                    dss_start = time.perf_counter()
                    dss_result = select_dss_pp_next_station(
                        estimator=estimator,
                        candidate_poses_xyz=np.asarray([next_pose], dtype=float),
                        current_pose_xyz=pose,
                        current_pair_id=current_shield_pair_id,
                        visited_poses_xyz=visited_arr,
                        map_api=planning_map,
                        bounds_xyz=(bounds_lo, bounds_hi),
                        continuous_height_bounds_m=(continuous_height_bounds_for_dss),
                        config=dss_selection_config,
                        height_partner_forced_program_pair_ids=(
                            height_partner_program_for_scoring
                        ),
                        height_partner_xy_tolerance_m=(
                            detector_height_pair_xy_tolerance_m
                        ),
                        height_partner_z_tolerance_m=(
                            detector_height_pair_z_tolerance_m
                        ),
                        height_partner_min_z_separation_m=(
                            detector_height_pair_min_separation_m
                        ),
                        allow_height_partner_first_action=(
                            allow_height_partner_first_action
                        ),
                    )
                    dss_elapsed = time.perf_counter() - dss_start
                    total_path_planning_wall_s += float(dss_elapsed)
                    path_planning_wall_samples_s.append(float(dss_elapsed))
                    planned_program_for_next = tuple(
                        int(pair_id) for pair_id in dss_result.shield_program.pair_ids
                    )
                    dss_diagnostics = dict(dss_result.diagnostics)
                    dss_first_node = (
                        dss_result.sequence[0] if dss_result.sequence else None
                    )
                    print(
                        "DSS-PP fixed-station shield program: "
                        f"program={dss_result.shield_program.name} "
                        f"pairs={list(planned_program_for_next)} "
                        f"score={float(dss_result.score):.6g} "
                        f"signature={float(dss_result.sequence[0].signature_score):.6g} "
                        "temporal_sep="
                        f"{float(dss_result.sequence[0].temporal_separation_score):.6g} "
                        f"workers={int(dss_result.diagnostics.get('program_eval_workers', 1))} "
                        f"compute={dss_elapsed:.3f}s"
                    )
                    _log_dss_ranked_node_diagnostics(
                        dss_diagnostics,
                        label=f"pose_{current_pose_idx}_one_step_fixed_station",
                    )
                    _log_dss_pairwise_ambiguity_diagnostics(
                        dss_diagnostics,
                        label=f"pose_{current_pose_idx}_one_step_fixed_station",
                    )
                    _log_dss_component_leader_diagnostics(
                        dss_diagnostics,
                        label=f"pose_{current_pose_idx}_one_step_fixed_station",
                    )
            is_height_partner_action = _validate_selected_station_action(
                current_pose_xyz=np.asarray(pose, dtype=float),
                next_pose_xyz=np.asarray(next_pose, dtype=float),
                previous_move_was_height_partner=(previous_move_was_height_partner),
                xy_tolerance_m=detector_height_pair_xy_tolerance_m,
                z_tolerance_m=detector_height_pair_z_tolerance_m,
                min_z_separation_m=detector_height_pair_min_separation_m,
            )
            if (
                is_height_partner_action
                and height_partner_program_for_scoring is not None
            ):
                if tuple(planned_program_for_next or ()) != tuple(
                    height_partner_program_for_scoring
                ):
                    raise RuntimeError(
                        "DSS-PP selected a height action with a shield program "
                        "different from the program used to score that action."
                    )
                if dss_diagnostics is None:
                    dss_diagnostics = {}
                dss_diagnostics["height_partner_action"] = True
                dss_diagnostics["height_partner_reused_shield_program"] = True
                dss_diagnostics["height_partner_strict_execution"] = True
                dss_diagnostics["height_partner_pair_ids"] = [
                    int(value) for value in height_partner_program_for_scoring
                ]
                print(
                    "Height-pair action was scored with the reused shield program: "
                    f"z={float(pose[2]):.3f}m->{float(next_pose[2]):.3f}m "
                    f"pairs={list(height_partner_program_for_scoring)}"
                )
            if bool(remaining_measurement_config.enabled) and estimator.measurements:
                remaining_program = planned_program_for_next
                if remaining_program is None:
                    fallback_pair = (
                        0
                        if current_shield_pair_id is None
                        else int(current_shield_pair_id)
                    )
                    remaining_program = (fallback_pair,)
                remaining_estimate = estimate_remaining_measurement_budget(
                    estimator,
                    next_pose_xyz=next_pose,
                    shield_program_pair_ids=remaining_program,
                    live_time_s=planning_live_time,
                    dss_node=dss_first_node,
                    dss_diagnostics=dss_diagnostics,
                    config=remaining_measurement_config,
                    current_station_count=pose_counter,
                )
                remaining_payload = remaining_estimate.to_dict()
                remaining_payload["next_pose"] = [
                    float(value) for value in np.asarray(next_pose, dtype=float)
                ]
                remaining_payload["planned_pairs"] = [
                    int(value) for value in remaining_program
                ]
                remaining_measurement_estimates.append(remaining_payload)
                print(format_remaining_measurement_estimate(remaining_estimate))
                _log_remaining_measurement_detail(
                    remaining_estimate,
                    label=f"pose_{current_pose_idx}_next",
                )
            pending_path_segment = _build_robot_path_segment(
                map_api=planning_map,
                from_pose_xyz=pose,
                to_pose_xyz=next_pose,
                nominal_motion_speed_m_s=nominal_motion_speed_m_s,
                path_planner=path_planner_resolved,
                planned_shield_program=planned_program_for_next,
                dss_diagnostics=dss_diagnostics,
            )
            motion_distance_m = float(pending_path_segment["distance_m"])
            motion_time_s = float(pending_path_segment["travel_time_s"])
            pending_motion_distance_m = motion_distance_m
            pending_motion_time_s = motion_time_s
            pending_shield_program = planned_program_for_next
            pending_force_strict_shield_program = bool(
                is_height_partner_action
                and height_partner_program_for_scoring is not None
            )
            print(
                "Robot travel segment: "
                f"distance={motion_distance_m:.3f}m "
                f"euclidean={float(pending_path_segment['euclidean_distance_m']):.3f}m "
                f"time={motion_time_s:.1f}s "
                f"speed={float(nominal_motion_speed_m_s):.3f}m/s "
                f"obstacle_aware={bool(pending_path_segment['obstacle_aware'])}"
            )
            current_pose = next_pose
            estimator.add_measurement_pose(current_pose, reset_filters=False)
            current_pose_idx = len(estimator.poses) - 1
    except Exception as exc:
        notifier.notify_failed(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "measurements_completed": int(step_counter),
                "mission_time_s": float(elapsed),
                "last_counts": last_counts,
                "last_max_ig": last_max_ig,
            }
        )
        raise
    finally:
        simulation_runtime.close()

    published_measurement_log = None
    if measurement_log_writer is not None:
        if not measurement_log_writer.records:
            raise RuntimeError(
                "Pure PF run produced no MeasurementLog records; refusing to "
                "return an estimator with unavailable input provenance."
            )
        published_measurement_log = measurement_log_writer.finalize()
        estimator.measurement_log_sha256 = published_measurement_log.log_sha256
        print(
            "MeasurementLog published: "
            f"{published_measurement_log.path} "
            f"sha256={published_measurement_log.log_sha256}"
        )

    online_wall_clock_s = float(time.perf_counter() - run_wall_start)
    wall_clock_runtime_s = online_wall_clock_s

    # Save final snapshots
    result_paths: dict[str, str] = {}
    summary_out_path: Path | None = None
    final_estimates_for_run: (
        dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] | None
    ) = None
    final_estimate_stages_for_run: (
        dict[str, dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]] | None
    ) = None
    final_posterior_projection_time_s = 0.0
    if save_outputs:
        pf_out_path = RESULTS_DIR / f"result_pf{output_suffix}.png"
        spectrum_out_path = RESULTS_DIR / f"result_spectrum{output_suffix}.png"
        last_spectrum_out_path = (
            RESULTS_DIR / f"result_spectrum_last{output_suffix}.png"
        )
        estimates_out_path = RESULTS_DIR / f"result_estimates{output_suffix}.png"
        summary_out_path = RESULTS_DIR / f"result_summary{output_suffix}.json"
        pf_posterior_out_path = RESULTS_DIR / f"pf_posterior{output_suffix}.json"
        result_paths = {
            "pf_plot": pf_out_path.as_posix(),
            "estimates_plot": estimates_out_path.as_posix(),
            "spectrum_plot": spectrum_out_path.as_posix(),
            "last_spectrum_plot": last_spectrum_out_path.as_posix(),
            "summary_json": summary_out_path.as_posix(),
            "pf_posterior_json": pf_posterior_out_path.as_posix(),
        }
        if published_measurement_log is not None:
            result_paths["measurement_log"] = str(published_measurement_log.path)
        if estimate_trace_out_path is not None:
            result_paths["intermediate_estimate_trace_jsonl"] = (
                estimate_trace_out_path.as_posix()
            )
        if cui_split_viz is not None:
            result_paths.update(
                {
                    "cui_split_view": cui_split_viz.index_path.as_posix(),
                    "cui_robot_2d_latest": cui_split_viz.latest_robot_path.as_posix(),
                    "cui_pf_3d_latest": cui_split_viz.latest_pf_path.as_posix(),
                }
            )
        pf_out_path.parent.mkdir(parents=True, exist_ok=True)
        posterior_projection_started_at = time.perf_counter()
        if last_frame is not None:
            last_frame.step_index = max(0, int(step_counter) - 1)
            last_frame.time = float(elapsed)
            final_estimates = _build_pf_posterior_estimates(
                estimator,
                isotopes,
            )
            final_estimate_stages_for_run = {
                "pf_posterior_projection": final_estimates,
            }
            _validate_surface_constrained_estimates(
                final_estimates,
                env,
                obstacle_grid,
                obstacle_height_m=float(runtime_config.get("obstacle_height_m", 2.0)),
                tolerance_m=max(
                    0.0,
                    float(
                        runtime_config.get(
                            "posterior_surface_tolerance_m",
                            1.0e-5,
                        )
                    ),
                ),
                surface_prior_active=bool(source_surface_prior),
            )
            final_estimates_for_run = final_estimates
        final_posterior_projection_time_s += float(
            time.perf_counter() - posterior_projection_started_at
        )
        if last_frame is not None:
            last_frame.estimated_sources = {
                iso: pos for iso, (pos, _) in final_estimates.items()
            }
            last_frame.estimated_strengths = {
                iso: strg for iso, (_, strg) in final_estimates.items()
            }
            viz.update(last_frame)
            if cui_split_viz is not None:
                cui_split_viz.update(last_frame)
        viz.save_final(pf_out_path.as_posix())
        if last_frame is not None:
            viz.save_estimates_only(estimates_out_path.as_posix())
        if representative_spectrum is not None:
            highlight = set(representative_candidates)
            title = "Representative measurement spectrum"
            if representative_step_index is not None:
                title = f"{title} (step {representative_step_index})"
            _save_spectrum_plot(
                decomposer,
                representative_spectrum,
                spectrum_out_path,
                highlight_isotopes=highlight,
                counts_by_isotope=representative_counts,
                component_spectra_by_isotope=representative_spectrum_components,
                title=title,
            )
        if last_spectrum is not None:
            highlight = set(last_candidates)
            _save_spectrum_plot(
                decomposer,
                last_spectrum,
                last_spectrum_out_path,
                highlight_isotopes=highlight,
                counts_by_isotope=last_counts,
                component_spectra_by_isotope=last_spectrum_components,
                title="Last measurement spectrum",
            )
        if cui_split_viz is not None and hasattr(cui_split_viz, "close"):
            cui_split_viz.close()
    total_meas_time = float(sum(measurement_live_times_s))
    total_mission_time_s = float(
        total_meas_time + total_motion_time_s + total_rotation_time_s
    )
    mean_live_time_s = (
        float(np.mean(measurement_live_times_s)) if measurement_live_times_s else 0.0
    )
    min_live_time_s = (
        float(np.min(measurement_live_times_s)) if measurement_live_times_s else 0.0
    )
    max_live_time_s = (
        float(np.max(measurement_live_times_s)) if measurement_live_times_s else 0.0
    )
    mean_ig_wall_s = float(np.mean(ig_wall_samples_s)) if ig_wall_samples_s else 0.0
    max_ig_wall_s = float(np.max(ig_wall_samples_s)) if ig_wall_samples_s else 0.0
    mean_pf_wall_s = float(np.mean(pf_wall_samples_s)) if pf_wall_samples_s else 0.0
    max_pf_wall_s = float(np.max(pf_wall_samples_s)) if pf_wall_samples_s else 0.0
    median_pf_wall_s = float(np.median(pf_wall_samples_s)) if pf_wall_samples_s else 0.0
    p95_pf_wall_s = (
        float(np.percentile(pf_wall_samples_s, 95.0)) if pf_wall_samples_s else 0.0
    )
    mean_path_planning_wall_s = (
        float(np.mean(path_planning_wall_samples_s))
        if path_planning_wall_samples_s
        else 0.0
    )
    max_path_planning_wall_s = (
        float(np.max(path_planning_wall_samples_s))
        if path_planning_wall_samples_s
        else 0.0
    )
    total_compute_time_s = float(
        total_ig_wall_s
        + total_pf_wall_s
        + total_viz_wall_s
        + total_path_planning_wall_s
    )
    resume_prefix_measurement_count = (
        0
        if resume_controller_state is None
        else int(resume_controller_state.step_counter)
    )
    station_height_metrics = _operational_station_height_metrics(
        estimator.measurements,
        estimator.poses,
        xy_tolerance_m=detector_height_pair_xy_tolerance_m,
        z_tolerance_m=detector_height_pair_z_tolerance_m,
    )
    mission_metrics = {
        "total_measurements": int(step_counter),
        "total_live_time_s": float(total_meas_time),
        "measurement_live_times_s": [
            float(value) for value in measurement_live_times_s
        ],
        "mean_live_time_s": mean_live_time_s,
        "min_live_time_s": min_live_time_s,
        "max_live_time_s": max_live_time_s,
        "measurement_time_cap_s": float(live_time) if has_live_time_cap else None,
        "adaptive_dwell_enabled": bool(adaptive_dwell),
        "adaptive_dwell_chunk_s": float(adaptive_dwell_chunk_s),
        "adaptive_min_dwell_s": float(adaptive_min_dwell_s),
        "adaptive_ready_min_counts": float(adaptive_ready_min_counts),
        "adaptive_ready_min_isotopes": int(adaptive_ready_min_isotopes),
        "adaptive_ready_min_snr": float(adaptive_ready_min_snr),
        "detector_height_sampling_mode": detector_height_config.mode,
        "detector_height_min_m": float(detector_height_config.minimum_mast_height_m),
        "detector_height_max_m": float(detector_height_config.maximum_mast_height_m),
        "detector_height_actions_m": list(
            detector_height_config.discrete_mast_actions_m
        ),
        "detector_height_action_world_z_m": list(
            detector_height_config.discrete_world_actions_m
        ),
        "robot_ground_z_m": float(robot_ground_z_m),
        "measurement_workspace": measurement_workspace_diagnostics,
        **station_height_metrics,
        "detector_pose_consistency_tolerance_m": float(
            detector_pose_consistency_tolerance_m
        ),
        "total_motion_distance_m": float(total_motion_distance_m),
        "nominal_motion_speed_m_s": float(nominal_motion_speed_m_s),
        "total_travel_time_s": float(total_motion_time_s),
        "estimated_motion_time_s": float(total_motion_time_s),
        "rotation_overhead_s_per_measurement": float(rotation_overhead_s),
        "total_shield_actuation_time_s": float(total_rotation_time_s),
        "estimated_rotation_time_s": float(total_rotation_time_s),
        "total_mission_time_s": float(total_mission_time_s),
        "estimated_end_to_end_time_s": float(total_mission_time_s),
        "total_move_measure_time_s": float(total_mission_time_s),
        "path_segments": path_segments,
        "num_motion_segments": int(len(path_segments)),
        "path_planner": path_planner_resolved,
        "dss_horizon": int(dss_config.horizon),
        "dss_beam_width": int(dss_config.beam_width),
        "dss_program_length": int(dss_config.program_length),
        "dss_primary_history_weight": float(dss_config.primary_history_weight),
        "dss_minimum_primary_history_weight": float(dss_config.primary_history_weight),
        "dss_primary_history_weight_semantics": (
            "minimum_from_maximum_sampling_fraction"
            if _target_sampled_primaries(runtime_config) is not None
            else "fixed_transport_history_weight"
        ),
        "dss_target_sampled_primaries": _target_sampled_primaries(runtime_config),
        "dss_signature_weight": float(dss_config.lambda_signature),
        "dss_differential_weight": float(dss_config.eta_differential),
        "dss_rotation_weight": float(dss_config.lambda_rotation),
        "dss_correlation_reduction_weight": float(
            dss_config.lambda_correlation_reduction
        ),
        "dss_isotope_balance_weight": float(dss_config.lambda_isotope_balance),
        "dss_environment_signature_weight": float(
            dss_config.lambda_environment_signature
        ),
        "dss_occlusion_boundary_weight": float(dss_config.lambda_occlusion_boundary),
        "dss_remaining_budget_guidance": bool(dss_config.remaining_budget_guidance),
        "dss_remaining_route_weight": float(dss_config.remaining_route_weight),
        "dss_same_isotope_direct_separation_guard": bool(
            dss_config.same_isotope_direct_separation_guard
        ),
        "dss_same_isotope_direct_separation_epsilon": float(
            dss_config.same_isotope_direct_separation_epsilon
        ),
        "mission_stop_require_pf_cardinality_ready": bool(
            mission_stop_require_pf_cardinality_ready
        ),
        "mission_stop_require_remaining_measurement_ready": bool(
            mission_stop_require_remaining_measurement_ready
        ),
        "pf_obstacle_attenuation": bool(pf_obstacle_attenuation_enabled),
        "pf_obstacle_grid_active": _has_environment_obstacles(pf_obstacle_grid),
        **_online_compute_timing_provenance(resume_prefix_measurement_count),
        "total_compute_time_s": total_compute_time_s,
        "ig_compute_time_s": float(total_ig_wall_s),
        "mean_orientation_selection_time_s": mean_ig_wall_s,
        "max_orientation_selection_time_s": max_ig_wall_s,
        "pf_compute_time_s": float(total_pf_wall_s),
        "pf_update_count": int(len(pf_wall_samples_s)),
        "mean_pf_update_time_s": mean_pf_wall_s,
        "median_pf_update_time_s": median_pf_wall_s,
        "p95_pf_update_time_s": p95_pf_wall_s,
        "max_pf_update_time_s": max_pf_wall_s,
        "path_planning_compute_time_s": float(total_path_planning_wall_s),
        "mean_path_planning_time_s": mean_path_planning_wall_s,
        "max_path_planning_time_s": max_path_planning_wall_s,
        "viz_time_s": float(total_viz_wall_s),
        "resumed_from_station_boundary": bool(
            resume_controller_state is not None
        ),
        "resume_prefix_measurement_count": (
            resume_prefix_measurement_count
        ),
        "resume_prefix_station_count": (
            0
            if resume_controller_state is None
            else int(resume_controller_state.pose_counter + 1)
        ),
        "resume_pf_replay_wall_s": float(resume_replay_wall_s),
        "online_wall_clock_s": float(online_wall_clock_s),
        "wall_clock_runtime_s": wall_clock_runtime_s,
        "operational_timing_definitions": {
            "online_wall_clock_s": (
                "Wall time from online-loop initialization through simulator close; "
                "posterior projection, plotting, and evaluation are excluded."
            ),
            "end_to_end_wall_clock_s": (
                "Wall time from online-loop initialization through final evaluation "
                "and strict payload sanitization; final JSON I/O and notification "
                "are excluded."
            ),
            "wall_clock_runtime_s": ("Compatibility alias of online_wall_clock_s."),
            "final_posterior_projection_time_s": (
                "Time to project final source estimates from the current sequential "
                "PF posterior; visualization and file output are excluded."
            ),
        },
    }
    setattr(estimator, "mission_metrics", mission_metrics)
    if save_outputs:
        print(f"Final PF visualization saved to: {pf_out_path}")
        print(f"Final estimates-only visualization saved to: {estimates_out_path}")
        if cui_split_viz is not None:
            print(f"CUI split view saved to: {cui_split_viz.index_path}")
        if representative_spectrum is not None:
            print(f"Representative spectrum saved to: {spectrum_out_path}")
        if last_spectrum is not None:
            print(f"Last spectrum saved to: {last_spectrum_out_path}")
    print(
        f"Total measurements: {step_counter}, "
        f"live={total_meas_time:.1f}s, "
        f"travel={total_motion_time_s:.1f}s, "
        f"shield={total_rotation_time_s:.1f}s, "
        f"mission={total_mission_time_s:.1f}s"
    )
    print(
        "Mission timing summary: "
        f"distance={total_motion_distance_m:.2f}m "
        f"motion={total_motion_time_s:.1f}s "
        f"rotation={total_rotation_time_s:.1f}s "
        f"end_to_end={mission_metrics['estimated_end_to_end_time_s']:.1f}s "
        f"compute={total_compute_time_s:.3f}s "
        f"compute_scope={mission_metrics['online_compute_timing_scope']} "
        f"path_plan={total_path_planning_wall_s:.3f}s "
        f"ig_mean={mean_ig_wall_s:.3f}s "
        f"pf_mean={mean_pf_wall_s:.3f}s "
        f"online_wall_clock={online_wall_clock_s:.2f}s"
    )
    surface_obstacle_height_m = float(runtime_config.get("obstacle_height_m", 2.0))
    posterior_surface_tolerance_m = max(
        0.0,
        float(runtime_config.get("posterior_surface_tolerance_m", 1.0e-5)),
    )
    source_positions_for_surface = np.asarray(
        [source.position for source in sources],
        dtype=float,
    ).reshape(-1, 3)
    source_kinds_for_evaluation = source_surface_kinds(
        source_positions_for_surface,
        env,
        obstacle_grid,
        obstacle_height_m=surface_obstacle_height_m,
        tolerance_m=posterior_surface_tolerance_m,
    )
    gt_by_iso: dict[str, list[dict[str, Any]]] = {}
    for src, surface_kind in zip(
        sources,
        source_kinds_for_evaluation,
        strict=True,
    ):
        gt_by_iso.setdefault(src.isotope, []).append(
            {
                "pos": [
                    float(src.position[0]),
                    float(src.position[1]),
                    float(src.position[2]),
                ],
                "strength": float(src.intensity_cps_1m),
                "surface_kind": (
                    "off_surface" if surface_kind is None else str(surface_kind)
                ),
            }
        )
    late_posterior_projection_start = (
        time.perf_counter() if final_estimates_for_run is None else None
    )
    if final_estimates_for_run is None:
        estimates = _build_pf_posterior_estimates(estimator, isotopes)
        final_estimate_stages = {
            "pf_posterior_projection": estimates,
        }
        _validate_surface_constrained_estimates(
            estimates,
            env,
            obstacle_grid,
            obstacle_height_m=float(runtime_config.get("obstacle_height_m", 2.0)),
            tolerance_m=max(
                0.0,
                float(
                    runtime_config.get(
                        "posterior_surface_tolerance_m",
                        1.0e-5,
                    )
                ),
            ),
            surface_prior_active=bool(source_surface_prior),
        )
    else:
        estimates = final_estimates_for_run
        final_estimate_stages = final_estimate_stages_for_run or {
            "pf_posterior_projection": estimates,
        }
    if late_posterior_projection_start is not None:
        final_posterior_projection_time_s += float(
            time.perf_counter() - late_posterior_projection_start
        )
    est_by_iso: dict[str, list[dict[str, Any]]] = {}
    for iso, estimate in estimates.items():
        positions = np.asarray(estimate[0], dtype=float)
        strengths = np.asarray(estimate[1], dtype=float)
        surface_kinds = source_surface_kinds(
            positions.reshape(-1, 3),
            env,
            obstacle_grid,
            obstacle_height_m=surface_obstacle_height_m,
            tolerance_m=posterior_surface_tolerance_m,
        )
        est_list: list[dict[str, Any]] = []
        for pos, strength, surface_kind in zip(
            positions,
            strengths,
            surface_kinds,
            strict=True,
        ):
            est_list.append(
                {
                    "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                    "strength": float(strength),
                    "surface_kind": (
                        "off_surface" if surface_kind is None else str(surface_kind)
                    ),
                }
            )
        est_by_iso[iso] = est_list
    estimated_source_uncertainty = estimator.posterior_source_uncertainty(
        estimates,
        match_radius_m=(
            None
            if runtime_config.get("posterior_uncertainty_match_radius_m") is None
            else max(
                0.0,
                float(runtime_config["posterior_uncertainty_match_radius_m"]),
            )
        ),
        surface_tolerance_m=posterior_surface_tolerance_m,
    )
    for isotope_diagnostics in estimated_source_uncertainty.values():
        for diagnostic in isotope_diagnostics:
            diagnostic["posterior_reference"] = "current_final_pf_particle_cloud"
            diagnostic["reported_estimate_reference"] = (
                "current_pf_posterior_projection"
            )
            diagnostic["reference_consistent"] = True
    final_source_status = _final_estimate_source_status(estimator, estimates)
    confirmed_est_by_iso = _filter_serialized_sources_by_status(
        est_by_iso,
        final_source_status,
        status="confirmed",
    )
    tentative_est_by_iso = _filter_serialized_sources_by_status(
        est_by_iso,
        final_source_status,
        status="tentative",
    )
    candidate_surface_payload = _surface_count_payload(
        grid,
        env,
        obstacle_grid,
        obstacle_height_m=surface_obstacle_height_m,
        tolerance_m=posterior_surface_tolerance_m,
    )
    candidate_surface_payload["total_candidates"] = candidate_surface_payload.pop(
        "total_sources"
    )
    source_surface_diagnostics = {
        "source_surface_prior_active": bool(source_surface_prior),
        "surface_annotation_tolerance_m": float(posterior_surface_tolerance_m),
        "candidate_grid": candidate_surface_payload,
        "estimated_sources": _estimate_surface_diagnostics(
            estimates,
            env,
            obstacle_grid,
            obstacle_height_m=surface_obstacle_height_m,
            tolerance_m=posterior_surface_tolerance_m,
        ),
        "particles": _particle_surface_diagnostics(
            estimator,
            env,
            obstacle_grid,
            obstacle_height_m=surface_obstacle_height_m,
            tolerance_m=posterior_surface_tolerance_m,
        ),
    }
    pf_obstacle_diagnostics = {
        "pf_obstacle_attenuation_active": bool(pf_obstacle_attenuation_enabled),
        "environment_obstacles_active": _has_environment_obstacles(obstacle_grid),
        "pf_obstacle_grid_active": _has_environment_obstacles(pf_obstacle_grid),
        "obstacle_buildup_coeff": float(pf_obstacle_buildup_coeff),
    }
    final_surface_observability = estimator.surface_candidate_observability_diagnostics(
        window=None,
        max_candidates=int(
            runtime_config.get("final_surface_observability_candidates", 1024)
        ),
    )
    gpu_memory_metrics = finish_gpu_memory_tracking(gpu_memory_baseline)
    mission_metrics.update(
        {
            "final_posterior_projection_time_s": float(
                final_posterior_projection_time_s
            ),
            "gpu_memory": gpu_memory_metrics,
        }
    )
    count_error_model = _count_error_model_diagnostics(
        pf_conf,
        obstacle_attenuation_active=bool(pf_obstacle_attenuation_enabled),
    )
    metrics = compute_metrics(
        gt_by_iso,
        est_by_iso,
        match_radius_m=eval_match_radius_m,
        close_pair_distance_m=max(
            0.0,
            float(runtime_config.get("evaluation_close_pair_distance_m", 2.0)),
        ),
        close_pair_min_estimated_separation_m=max(
            0.0,
            float(
                runtime_config.get(
                    "evaluation_close_pair_min_estimated_separation_m",
                    0.5,
                )
            ),
        ),
        uncertainty_by_iso=estimated_source_uncertainty,
    )
    print_metrics_report(metrics)
    final_isotope_count_diagnostics = _final_isotope_count_residual_diagnostics(
        estimator,
        estimates,
    )
    for iso, stats in sorted(final_isotope_count_diagnostics.items()):
        print(
            f"Final count residual[{iso}]: "
            f"sources={int(stats.get('reported_source_count', 0))} "
            f"obs_total={float(stats.get('observed_total_counts', 0.0)):.3f} "
            f"pred_total={float(stats.get('predicted_total_counts', 0.0)):.3f} "
            f"pos_resid={float(stats.get('positive_residual_total_counts', 0.0)):.3f} "
            f"neg_resid={float(stats.get('negative_residual_total_counts', 0.0)):.3f} "
            f"chi2={float(stats.get('residual_chi2', 0.0)):.3f}"
        )
    online_metric_summary = _online_estimate_metric_summary(
        estimator.history_estimates,
        gt_by_iso,
        match_radius_m=float(eval_match_radius_m),
    )
    cluster_stability = summarize_cluster_stability(
        estimator.history_estimates,
        final_window=max(
            1,
            int(runtime_config.get("evaluation_cluster_stability_window", 5)),
        ),
        match_gate_m=max(
            0.0,
            float(runtime_config.get("evaluation_cluster_match_gate_m", 0.5)),
        ),
    )
    count_bias_diagnostics = _final_count_bias_diagnostics(
        estimator,
        estimates,
        count_regime_lower_edges=runtime_config.get(
            "evaluation_count_regime_lower_edges",
            (0.0, 10.0, 100.0, 1000.0),
        ),
    )
    remaining_trace_summary = _remaining_measurement_trace_summary(
        remaining_measurement_estimates
    )
    final_pf_cardinality_status = _final_pf_cardinality_status(estimator)
    final_payload = {
        **_pure_pf_summary_provenance(estimator),
        "measurements_completed": int(step_counter),
        "mission_metrics": {
            **{
                key: value
                for key, value in mission_metrics.items()
                if key != "path_segments"
            },
            "path_segments": _compact_path_segments(path_segments),
        },
        "match_metrics": metrics,
        "evaluation_metrics": {
            "schema_version": 1,
            "accuracy": metrics.get("global", {}),
            "count_bias": count_bias_diagnostics,
            "pf_structural_evidence": final_pf_cardinality_status,
            "cluster_stability": cluster_stability,
            "operational": {
                "mean_pf_update_time_s": mission_metrics.get("mean_pf_update_time_s"),
                "pf_update_count": mission_metrics.get("pf_update_count"),
                "median_pf_update_time_s": mission_metrics.get(
                    "median_pf_update_time_s"
                ),
                "p95_pf_update_time_s": mission_metrics.get("p95_pf_update_time_s"),
                "max_pf_update_time_s": mission_metrics.get("max_pf_update_time_s"),
                "final_posterior_projection_time_s": mission_metrics.get(
                    "final_posterior_projection_time_s"
                ),
                "gpu_memory": mission_metrics.get("gpu_memory"),
                "mission_time_s": mission_metrics.get("total_mission_time_s"),
                "online_wall_clock_s": mission_metrics.get("online_wall_clock_s"),
                "end_to_end_wall_clock_s": mission_metrics.get(
                    "end_to_end_wall_clock_s"
                ),
                "station_visit_count": mission_metrics.get("station_visit_count"),
                "unique_xy_station_count": mission_metrics.get(
                    "unique_xy_station_count"
                ),
                "unique_xyz_action_count": mission_metrics.get(
                    "unique_xyz_action_count"
                ),
                "height_transition_count": mission_metrics.get(
                    "height_transition_count"
                ),
                "station_count": mission_metrics.get("station_count"),
                "detector_pose_station_count": mission_metrics.get(
                    "detector_pose_station_count"
                ),
                "height_change_count": mission_metrics.get("height_change_count"),
                "station_height_count_definitions": mission_metrics.get(
                    "station_height_count_definitions"
                ),
                "operational_timing_definitions": mission_metrics.get(
                    "operational_timing_definitions"
                ),
            },
        },
        "online_estimate_metrics": online_metric_summary,
        "remaining_measurement_trace_summary": remaining_trace_summary,
        "estimated_sources": est_by_iso,
        "estimated_source_uncertainty": estimated_source_uncertainty,
        "estimated_source_uncertainty_reference": {
            "posterior_reference": "current_final_pf_particle_cloud",
            "reported_estimate_reference": "current_pf_posterior_projection",
            "reference_consistent": True,
        },
        "estimated_sources_confirmed": confirmed_est_by_iso,
        "estimated_sources_tentative": tentative_est_by_iso,
        "estimated_source_status": final_source_status,
        "final_particle_cloud": _final_particle_cloud_payload(estimator),
        "remaining_measurement_estimates": remaining_measurement_estimates,
        "last_remaining_measurement_estimate": (
            None
            if not remaining_measurement_estimates
            else remaining_measurement_estimates[-1]
        ),
        "max_pose_stop_unresolved": bool(max_pose_stop_unresolved),
        "max_pose_stop_diagnostics": max_pose_stop_diagnostics,
        "random_source_visibility": random_source_visibility_diagnostics,
        "source_surface_diagnostics": source_surface_diagnostics,
        "pf_obstacle_diagnostics": pf_obstacle_diagnostics,
        "count_error_model": count_error_model,
        "surface_candidate_observability_diagnostics": final_surface_observability,
        "isotope_count_residual_diagnostics": final_isotope_count_diagnostics,
        "final_estimate_diagnostics": {
            "stages": {
                stage: _serialize_estimate_stage(stage_estimates)
                for stage, stage_estimates in final_estimate_stages.items()
            },
            "pf_cardinality_status": final_pf_cardinality_status,
        },
        "ground_truth_sources": gt_by_iso,
        "last_counts": last_counts,
        "output_paths": result_paths,
        "backend": sim_backend,
        "sim_config_path": sim_config_path,
        "environment_mode": normalized_environment_mode,
    }
    sanitized_final_payload = _sanitize_json_payload(final_payload)
    if not isinstance(sanitized_final_payload, dict):
        raise TypeError("Final run summary must sanitize to a JSON object.")
    final_payload = sanitized_final_payload
    end_to_end_wall_clock_s = float(time.perf_counter() - run_wall_start)
    mission_metrics["end_to_end_wall_clock_s"] = end_to_end_wall_clock_s
    final_payload["mission_metrics"]["end_to_end_wall_clock_s"] = (
        end_to_end_wall_clock_s
    )
    final_payload["evaluation_metrics"]["operational"]["end_to_end_wall_clock_s"] = (
        end_to_end_wall_clock_s
    )
    setattr(estimator, "final_run_summary", final_payload)
    if save_outputs and summary_out_path is not None:
        posterior_getter = getattr(estimator, "posterior_snapshot", None)
        if callable(posterior_getter):
            pf_posterior_out_path.write_text(
                json.dumps(
                    posterior_getter().to_dict(),
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
        summary_out_path.write_text(
            json.dumps(
                final_payload,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
    notifier.notify_finished(
        {
            "summary": (
                f"{step_counter} measurements, "
                f"mission_time_s={total_mission_time_s:.1f}, "
                f"end_to_end_wall_clock_s={end_to_end_wall_clock_s:.2f}"
            ),
            **final_payload,
        }
    )
    if live:
        plt.ioff()
        plt.pause(0.1)
    plt.close("all")
    if return_state:
        return estimator
    return None


def run_realtime_pf() -> None:
    """Entry point for real-time PF + visualization with built-in demo settings."""
    run_live_pf(live=True, max_steps=10)
