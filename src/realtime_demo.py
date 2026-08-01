"""Real-time demo for the rotating-shield particle filter with visualization."""
# ruff: noqa: E402

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
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


def _resolve_variable_cardinality(
    requested: bool | None,
    runtime_config: Mapping[str, Any],
) -> bool:
    """Resolve an explicit variable-cardinality override or runtime default."""
    if requested is not None:
        return _strict_json_bool(
            requested,
            name="variable_cardinality override",
        )
    return _runtime_bool(runtime_config, "variable_cardinality", False)


def _deep_merge_runtime_config(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Recursively merge physical and estimator configuration objects."""
    merged = dict(base)
    for key, value in overrides.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge_runtime_config(existing, value)
        else:
            merged[key] = value
    return merged


def load_online_runtime_configs(
    sim_config_path: str | Path | None,
    pf_config_path: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load physical config and overlay it on PF-owned estimator defaults."""
    physical_config = load_runtime_config(sim_config_path)
    estimator_defaults = (
        {}
        if pf_config_path is None
        else load_runtime_config(pf_config_path)
    )
    merged = _deep_merge_runtime_config(estimator_defaults, physical_config)
    return physical_config, enforce_pure_runtime_settings(merged)


def resolve_runtime_variable_cardinality(
    requested: bool | None,
    sim_config_path: str | Path | None,
    pf_config_path: str | Path | None = None,
) -> bool:
    """Resolve variable cardinality from a CLI override and runtime config."""
    _, runtime_config = load_online_runtime_configs(
        sim_config_path,
        pf_config_path,
    )
    return _resolve_variable_cardinality(requested, runtime_config)


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
        """Return whether one configuration key discloses truth realization."""
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
        """Recursively remove truth-realization fields from one payload."""
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
    surface_diagnostic_points_xyz: NDArray[np.float64],
    surface_atlas_diagnostics: Mapping[str, Any],
    api_settings: Mapping[str, Any],
    isotopes: Sequence[str],
) -> dict[str, Any]:
    """Return one canonical config binding every resolved live-PF input."""
    diagnostic_points = np.asarray(
        surface_diagnostic_points_xyz,
        dtype=np.float64,
    )
    if diagnostic_points.ndim != 2 or diagnostic_points.shape[1] != 3:
        raise ValueError(
            "surface_diagnostic_points_xyz must have shape (N, 3)."
        )
    api_payload = dict(api_settings)
    api_payload["pf_rng_provenance"] = pf_rng_provenance(
        api_payload.get("pf_random_seed"),
        isotopes,
    )
    planning_seed = api_payload.get(
        "planning_random_seed",
        api_payload.get(
            "planning_candidate_seed",
            api_payload.get("pf_random_seed"),
        ),
    )
    api_payload["planning_rng_provenance"] = named_rng_provenance(
        planning_seed,
        (
            "live_planning_candidate",
            "live_planning_dss_eig",
        ),
    )
    payload = _truth_free_live_runtime_config(runtime_config)
    payload["effective_pf_replay"] = {
        "api_settings": json_safe(api_payload),
        "pf_config": json_safe(pf_config),
        "surface_atlas_diagnostics": {
            **json_safe(dict(surface_atlas_diagnostics)),
            "point_count": int(diagnostic_points.shape[0]),
            "xyz_sha256": sha256_json(diagnostic_points),
        },
    }
    return dict(json_safe(payload))


def _physical_surface_atlas_diagnostic_points(
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    *,
    chart_max_edge_m: float,
    point_count: int,
) -> tuple[
    NDArray[np.float64],
    dict[str, Any],
    SurfaceChartGeometry,
]:
    """Sample the same physical continuous surface atlas used by the PF."""
    chart_geometry = build_surface_chart_geometry(
        env,
        obstacle_grid,
        max_edge_m=float(chart_max_edge_m),
    )
    if not chart_geometry.obstacle_surfaces_available:
        warning = chart_geometry.obstacle_geometry_warning or (
            "Obstacle component surfaces are unavailable."
        )
        raise ValueError(
            "Surface diagnostics require the PF physical transport-box atlas: "
            f"{warning}"
        )
    atlas = ContinuousSurfaceAtlas(chart_geometry)
    count = max(1, int(point_count))
    quantiles = (
        np.arange(count, dtype=np.float64) + 0.5
    ) / float(count)
    chart_ids = np.searchsorted(
        np.cumsum(atlas.chart_probabilities),
        quantiles,
        side="right",
    ).astype(np.int64)
    if np.any(chart_ids < 0) or np.any(chart_ids >= atlas.chart_count):
        raise RuntimeError(
            "Area-weighted surface diagnostics produced an invalid chart ID."
        )
    sequence = np.arange(count, dtype=np.float64) + 0.5
    uv = np.column_stack(
        (
            np.mod(sequence * ((np.sqrt(5.0) - 1.0) / 2.0), 1.0),
            np.mod(sequence * (np.sqrt(2.0) - 1.0), 1.0),
        )
    )
    points = np.ascontiguousarray(atlas.positions_xyz(chart_ids, uv))
    diagnostics = {
        "generator": "continuous_surface_atlas_area_uniform.v1",
        "sampling_domain": "pf_physical_continuous_surface_atlas",
        "chart_max_edge_m": float(chart_max_edge_m),
        "chart_count": int(atlas.chart_count),
        "total_area_m2": float(atlas.total_area_m2),
        "surface_atlas_contract_sha256": (
            surface_chart_geometry_sha256(chart_geometry)
        ),
        "ordered_vertices_sha256": sha256_json(chart_geometry.vertices_xyz),
        "ordered_areas_sha256": sha256_json(chart_geometry.areas_m2),
        **chart_geometry.geometry_metadata,
    }
    return points, diagnostics, chart_geometry


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
        if any(not isinstance(name, str) for name in raw_requested):
            raise TypeError(
                "random_source_isotopes sequence entries must be JSON strings."
            )
        names = [name.strip() for name in raw_requested if name.strip()]
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
    if len(set(names)) != len(names):
        raise ValueError("random_source_isotopes must not contain duplicates.")
    return tuple(sorted(names))


def _runtime_float(
    runtime_config: Mapping[str, Any],
    key: str,
    default: float,
) -> float:
    """Return one exact JSON number, treating explicit null as the default."""
    value = runtime_config.get(key)
    if value is None:
        return float(default)
    return _strict_json_number(value, name=key)


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
        raise TypeError("detector_model must be a JSON object.")
    return detector_active_radius_m(
        detector_model,
        default_radius_m=DEFAULT_CRYSTAL_RADIUS_M,
    )


def _validate_weighted_pf_runtime_contract(
    runtime_config: Mapping[str, Any],
    *,
    planning_primary_history_weight: float,
) -> None:
    """Fail closed when weighted transport violates its transport contract."""
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


def _resolve_pf_strength_prior_bounds(
    runtime_config: Mapping[str, Any],
    *,
    generated_population_bounds: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Resolve one bounded physical strength prior declared before inference."""
    explicit_minimum = runtime_config.get("pf_strength_prior_min_cps_1m")
    explicit_maximum = runtime_config.get("pf_strength_prior_max_cps_1m")
    if (explicit_minimum is None) != (explicit_maximum is None):
        raise ValueError(
            "PF strength-prior minimum and maximum must be provided together."
        )
    if explicit_minimum is not None and explicit_maximum is not None:
        minimum = _strict_json_number(
            explicit_minimum,
            name="pf_strength_prior_min_cps_1m",
        )
        maximum = _strict_json_number(
            explicit_maximum,
            name="pf_strength_prior_max_cps_1m",
        )
    elif generated_population_bounds is not None:
        minimum, maximum = (
            float(generated_population_bounds[0]),
            float(generated_population_bounds[1]),
        )
    else:
        generator_minimum = runtime_config.get(
            "random_source_intensity_min_cps_1m"
        )
        generator_maximum = runtime_config.get(
            "random_source_intensity_max_cps_1m"
        )
        if generator_minimum is None or generator_maximum is None:
            raise ValueError(
                "Pure PF requires explicit pf_strength_prior_min_cps_1m and "
                "pf_strength_prior_max_cps_1m bounds."
            )
        minimum = _strict_json_number(
            generator_minimum,
            name="random_source_intensity_min_cps_1m",
        )
        maximum = _strict_json_number(
            generator_maximum,
            name="random_source_intensity_max_cps_1m",
        )
    if not np.isfinite(minimum) or minimum < 0.0:
        raise ValueError(
            "pf_strength_prior_min_cps_1m must be finite and nonnegative."
        )
    if not np.isfinite(maximum) or maximum <= minimum:
        raise ValueError(
            "pf_strength_prior_max_cps_1m must be finite and greater than "
            "pf_strength_prior_min_cps_1m."
        )
    return minimum, maximum


def _resolve_candidate_isotopes(
    runtime_config: Mapping[str, Any],
    library_isotopes: Sequence[str],
) -> tuple[str, ...]:
    """Return isotope names that the online PF should estimate."""
    raw_requested = runtime_config.get("candidate_isotopes")
    if raw_requested is None:
        names = [str(name) for name in library_isotopes]
    elif isinstance(raw_requested, str):
        names = [name.strip() for name in raw_requested.split(",") if name.strip()]
    elif isinstance(raw_requested, Sequence):
        if any(not isinstance(name, str) for name in raw_requested):
            raise TypeError(
                "candidate_isotopes sequence entries must be JSON strings."
            )
        names = [name.strip() for name in raw_requested if name.strip()]
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
    if len(set(names)) != len(names):
        raise ValueError("candidate_isotopes must not contain duplicates.")
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
from measurement.source_boundary import (
    SURFACE_SOURCE_RUNTIME_KEYS,
    canonical_surface_source_runtime_payload,
    surface_emission_policy_payload,
    surface_emission_policy_sha256,
    surface_source_runtime_contract_sha256,
    surface_transport_positions,
    validate_air_facing_surface_normals,
)
from measurement.source_surfaces import (
    SOURCE_SURFACE_REPORT_LABELS,
    generate_surface_sources,
    source_surface_kinds,
    validate_area_uniform_source_config,
)
from measurement.surface_charts import (
    SurfaceChartGeometry,
    build_surface_chart_geometry,
    surface_chart_geometry_sha256,
)
from measurement.shielding import (
    generate_octant_orientations,
    generate_octant_rotation_matrices,
)
from spectrum.library import default_library, get_detection_lines_keV
from spectrum.transport_spectral import (
    GeometryConditionedSpectralModel,
    geometry_conditioned_model_from_runtime_config,
)
from spectrum.response_matrix import (
    NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256,
)
from pf.full_spectrum import FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY
from pf.posterior import (
    validated_probability_distribution,
    validated_state_cardinality,
)
from pf.pure_estimator import PurePFEstimator, RotatingShieldPFConfig
from pf.profiles import apply_profile_to_config, enforce_pure_runtime_settings
from pf.provenance import json_safe, sha256_json
from pf.randomness import (
    named_random_generator,
    named_rng_provenance,
    named_stream_seed,
    pf_rng_provenance,
)
from pf.replay import build_replay_estimator, replay_records
from measurement.surface_atlas import ContinuousSurfaceAtlas
from planning.candidate_generation import generate_candidate_poses
from planning.dss_pp import DSSPPConfig, select_dss_pp_next_station
from planning.measurement_workspace import (
    AxisAlignedRoomBounds,
    DetectorAssemblyGeometry,
    MeasurementWorkspace,
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
from evaluation_diagnostics import (
    finish_gpu_memory_tracking,
    start_gpu_memory_tracking,
    summarize_cluster_stability,
)
from evaluation_metrics import compute_metrics, print_metrics_report
from piplup_notify import PiplupNotificationConfig, PiplupNotifier
from cui_runtime import (
    ensure_cui_view_server as _ensure_cui_view_server,
    resolve_cui_split_view_enabled as _resolve_cui_split_view_enabled,
)
from mission_control import (
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
from runtime.assets import simulation_runtime_root
from runtime.provenance import repository_commit as simulation_repository_commit
from runtime.session import estimator_neutral_runtime_config


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
_TRUTH_SURFACE_SOURCE_RNG_DOMAIN = "truth_surface_sources"


def _strict_json_bool(value: object, *, name: str) -> bool:
    """Return an exact JSON boolean without truth-value coercion."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a JSON boolean.")
    return value


def _runtime_bool(
    runtime_config: Mapping[str, object],
    key: str,
    default: bool,
) -> bool:
    """Return one optional runtime boolean under a fail-closed type contract."""
    return _strict_json_bool(
        runtime_config.get(key, default),
        name=key,
    )


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


def _strict_json_number(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_exclusive: bool = False,
    maximum_exclusive: bool = False,
) -> float:
    """Return a finite JSON number inside optional numeric bounds."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a JSON number.")
    resolved = float(value)
    if not np.isfinite(resolved):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None:
        invalid_minimum = (
            resolved <= float(minimum)
            if minimum_exclusive
            else resolved < float(minimum)
        )
        if invalid_minimum:
            relation = "greater than" if minimum_exclusive else "at least"
            raise ValueError(f"{name} must be {relation} {float(minimum)}.")
    if maximum is not None:
        invalid_maximum = (
            resolved >= float(maximum)
            if maximum_exclusive
            else resolved > float(maximum)
        )
        if invalid_maximum:
            relation = "less than" if maximum_exclusive else "at most"
            raise ValueError(f"{name} must be {relation} {float(maximum)}.")
    return resolved


def _strict_json_string(
    value: object,
    *,
    name: str,
    allow_empty: bool = False,
) -> str:
    """Return an exact JSON string without stringification coercion."""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a JSON string.")
    if not allow_empty and not value.strip():
        raise ValueError(f"{name} must be a nonempty JSON string.")
    return value


def _build_resume_replay_estimator(
    prefix_log: MeasurementLog,
    *,
    pf_config: Mapping[str, Any],
    profile: str,
    seed: int,
    config_hash: str,
) -> PurePFEstimator:
    """Build a resume estimator from explicit PF settings and a raw prefix."""
    return build_replay_estimator(
        prefix_log,
        pf_config,
        profile=profile,
        seed=seed,
        config_hash=config_hash,
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
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


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
        "planning_candidate_parameters": json_safe(
            dict(planning_candidate_parameters)
        ),
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
    if sha256_json(
        json_safe(dss_eig_rng.bit_generator.state)
    ) != sha256_json(dict(dss_rng_state)):
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
    if not isinstance(raw, Mapping):
        raise RuntimeError("Pure PF estimates must be an isotope mapping.")
    expected = tuple(str(isotope) for isotope in isotopes)
    if len(set(expected)) != len(expected) or set(raw) != set(expected):
        raise RuntimeError(
            "Pure PF estimates must contain exactly every configured isotope."
        )
    resolved: dict[
        str,
        tuple[NDArray[np.float64], NDArray[np.float64]],
    ] = {}
    for isotope in expected:
        estimate = raw[isotope]
        if (
            not isinstance(estimate, Sequence)
            or isinstance(estimate, (str, bytes))
            or len(estimate) != 2
        ):
            raise RuntimeError(
                f"Pure PF estimate for isotope {isotope} must be (positions, strengths)."
            )
        positions = np.asarray(estimate[0], dtype=float)
        strengths = np.asarray(estimate[1], dtype=float)
        if (
            positions.ndim != 2
            or positions.shape[1:] != (3,)
            or strengths.ndim != 1
            or positions.shape[0] != strengths.size
            or np.any(~np.isfinite(positions))
            or np.any(~np.isfinite(strengths))
            or np.any(strengths <= 0.0)
        ):
            raise RuntimeError(
                f"Pure PF estimate arrays are invalid for isotope {isotope}."
            )
        resolved[isotope] = (positions.copy(), strengths.copy())
    return resolved


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
    estimator: PurePFEstimator | None = None,
) -> None:
    """Fail closed when a surface-prior report leaves its authoritative support."""
    if not surface_prior_active:
        return
    total_positions = 0
    off_surface_count = 0
    for isotope, estimate in estimates.items():
        positions = np.asarray(estimate[0], dtype=float).reshape(-1, 3)
        total_positions += int(positions.shape[0])
        if positions.size == 0:
            continue
        if estimator is not None:
            try:
                surface_kinds = estimator.structural_surface_kinds(
                    str(isotope),
                    positions,
                    strict=True,
                )
            except ValueError as error:
                raise RuntimeError(
                    "Surface-prior report contains off-surface positions "
                    f"for isotope {isotope}."
                ) from error
        else:
            surface_kinds = source_surface_kinds(
                positions,
                environment,
                obstacle_grid,
                obstacle_height_m=obstacle_height_m,
                tolerance_m=tolerance_m,
            )
        off_surface_count += int(
            np.count_nonzero(np.equal(surface_kinds, None))
        )
    if off_surface_count:
        raise RuntimeError(
            "Surface-constrained PF report contains "
            f"{off_surface_count}/{total_positions} off-surface positions."
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


def _bind_sources_to_surface_transport(
    sources: Sequence[PointSource],
    chart_geometry: SurfaceChartGeometry,
) -> list[PointSource]:
    """Bind truth anchors to chart identity and shared air-side transport XYZ."""
    if not sources:
        return []
    validate_air_facing_surface_normals(chart_geometry)
    atlas = ContinuousSurfaceAtlas(chart_geometry)
    anchors = np.asarray([source.position for source in sources], dtype=np.float64)
    chart_ids = np.empty(len(sources), dtype=np.int64)
    surface_uv = np.empty((len(sources), 2), dtype=np.float64)
    unresolved_indices = np.asarray(
        [
            index
            for index, source in enumerate(sources)
            if source.surface_chart_id is None
        ],
        dtype=np.int64,
    )
    if unresolved_indices.size:
        located_chart_ids, located_uv = atlas.locate_positions(
            anchors[unresolved_indices]
        )
        chart_ids[unresolved_indices] = located_chart_ids
        surface_uv[unresolved_indices] = located_uv
    authoritative_indices = np.asarray(
        [
            index
            for index, source in enumerate(sources)
            if source.surface_chart_id is not None
        ],
        dtype=np.int64,
    )
    if authoritative_indices.size:
        authoritative_chart_ids = np.asarray(
            [
                sources[index].surface_chart_id
                for index in authoritative_indices
            ],
            dtype=np.int64,
        )
        authoritative_uv = np.asarray(
            [
                sources[index].surface_uv
                for index in authoritative_indices
            ],
            dtype=np.float64,
        )
        mapped = atlas.positions_xyz(
            authoritative_chart_ids,
            authoritative_uv,
        )
        coordinate_match = np.all(
            np.isclose(
                mapped,
                anchors[authoritative_indices],
                rtol=0.0,
                atol=1.0e-9,
            ),
            axis=1,
        )
        if np.any(~coordinate_match):
            first_bad = int(
                authoritative_indices[np.flatnonzero(~coordinate_match)[0]]
            )
            raise ValueError(
                "PointSource chart/UV metadata does not map to its exact "
                f"surface anchor (first invalid source {first_bad})."
            )
        chart_ids[authoritative_indices] = authoritative_chart_ids
        surface_uv[authoritative_indices] = authoritative_uv
    policy_hash = surface_emission_policy_sha256()
    bound: list[PointSource] = []
    for index, source in enumerate(sources):
        chart_id = int(chart_ids[index])
        uv = surface_uv[index]
        normal = np.asarray(
            chart_geometry.normals_xyz[chart_id],
            dtype=np.float64,
        )
        transport = surface_transport_positions(
            anchors[index].reshape(1, 3),
            normal.reshape(1, 3),
        )[0]
        if source.surface_normal is not None and not np.array_equal(
            np.asarray(source.surface_normal, dtype=np.float64),
            normal,
        ):
            raise ValueError("PointSource surface normal conflicts with the atlas.")
        if source.transport_position is not None and not np.allclose(
            np.asarray(source.transport_position, dtype=np.float64),
            transport,
            rtol=0.0,
            atol=1.0e-15,
        ):
            raise ValueError(
                "PointSource transport position conflicts with the shared "
                "surface-emission policy."
            )
        if (
            source.surface_emission_policy_sha256 is not None
            and source.surface_emission_policy_sha256 != policy_hash
        ):
            raise ValueError("PointSource surface-emission policy hash is stale.")
        bound.append(
            PointSource(
                isotope=source.isotope,
                position=tuple(float(value) for value in anchors[index]),
                intensity_cps_1m=float(source.intensity_cps_1m),
                surface_chart_id=chart_id,
                surface_uv=(float(uv[0]), float(uv[1])),
                surface_normal=tuple(float(value) for value in normal),
                transport_position=tuple(float(value) for value in transport),
                surface_emission_policy_sha256=policy_hash,
            )
        )
    return bound


def _source_runtime_payload(source: PointSource) -> dict[str, object]:
    """Serialize one truth anchor and its distinct native transport position."""
    payload: dict[str, object] = {
        "isotope": str(source.isotope),
        "position": [float(value) for value in source.position],
        "transport_position": [
            float(value) for value in source.transport_position_array()
        ],
        "intensity_cps_1m": float(source.intensity_cps_1m),
    }
    if source.surface_chart_id is not None:
        payload.update(
            {
                "surface_chart_id": int(source.surface_chart_id),
                "surface_uv": [float(value) for value in source.surface_uv],
                "surface_normal": [
                    float(value) for value in source.surface_normal
                ],
                "surface_emission_policy_sha256": str(
                    source.surface_emission_policy_sha256
                ),
            }
        )
    return payload


def _validate_provided_surface_source_contract(
    provenance: Mapping[str, object] | None,
    sources: Sequence[PointSource],
    *,
    chart_geometry: SurfaceChartGeometry,
    obstacle_seed: int | None,
    chart_max_edge_m: float,
) -> None:
    """Verify a declared area-uniform source file against the runtime atlas."""
    if provenance is None:
        return
    declared = provenance.get("provided_file_declared_metadata")
    if not isinstance(declared, Mapping):
        raise ValueError(
            "Provided source provenance is missing declared metadata."
        )
    raw_schema = declared.get("source_surface_sampling_schema_version")
    if raw_schema is None:
        return
    schema = _strict_json_integer(
        raw_schema,
        name="source_surface_sampling_schema_version",
        minimum=3,
        maximum=3,
    )
    if schema != 3:
        raise ValueError(
            "Area-uniform source files must use surface sampling schema 3."
        )
    required = {
        "obstacle_seed",
        "sampling_measure",
        "selection_conditioning",
        "surface_atlas_contract_sha256",
        "surface_chart_max_edge_m",
        "surface_emission_policy_sha256",
        "surface_source_runtime_contract_sha256",
    }
    missing = sorted(required.difference(declared))
    if missing:
        raise ValueError(
            "Area-uniform source metadata is incomplete: "
            f"missing={missing}."
        )
    if declared["sampling_measure"] != "continuous_area_uniform":
        raise ValueError(
            "Area-uniform source metadata has the wrong sampling measure."
        )
    if declared["selection_conditioning"] != "none_physical_area_only":
        raise ValueError(
            "Area-uniform source metadata must not declare truth selection."
        )
    declared_obstacle_seed = _strict_json_integer(
        declared["obstacle_seed"],
        name="source metadata obstacle_seed",
        minimum=0,
    )
    if obstacle_seed is None or declared_obstacle_seed != int(obstacle_seed):
        raise ValueError(
            "Provided area-uniform sources were generated for a different "
            "obstacle seed."
        )
    declared_edge = _strict_json_number(
        declared["surface_chart_max_edge_m"],
        name="source metadata surface_chart_max_edge_m",
        minimum=0.0,
        minimum_exclusive=True,
    )
    if declared_edge != float(chart_max_edge_m):
        raise ValueError(
            "Provided area-uniform sources use a different surface chart "
            "tessellation."
        )

    def _sha256(value: object, *, name: str) -> str:
        """Return one strict lowercase SHA-256 metadata value."""
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
        return value

    declared_atlas = _sha256(
        declared["surface_atlas_contract_sha256"],
        name="source metadata surface_atlas_contract_sha256",
    )
    runtime_atlas = surface_chart_geometry_sha256(chart_geometry)
    if declared_atlas != runtime_atlas:
        raise ValueError(
            "Provided area-uniform sources were generated on a different "
            "continuous surface atlas."
        )
    declared_policy = _sha256(
        declared["surface_emission_policy_sha256"],
        name="source metadata surface_emission_policy_sha256",
    )
    if declared_policy != surface_emission_policy_sha256():
        raise ValueError(
            "Provided area-uniform sources use a stale surface-emission policy."
        )
    if any(source.surface_chart_id is None for source in sources):
        raise ValueError(
            "Area-uniform source schema 3 requires authoritative chart/UV "
            "metadata for every source."
        )
    payloads = [_source_runtime_payload(source) for source in sources]
    if any(set(payload) != SURFACE_SOURCE_RUNTIME_KEYS for payload in payloads):
        raise ValueError(
            "Area-uniform source schema 3 contains an incomplete source entry."
        )
    declared_source_contract = _sha256(
        declared["surface_source_runtime_contract_sha256"],
        name="source metadata surface_source_runtime_contract_sha256",
    )
    if (
        surface_source_runtime_contract_sha256(payloads)
        != declared_source_contract
    ):
        raise ValueError(
            "Provided area-uniform source entries differ from their declared "
            "runtime contract."
        )


def _validate_truth_within_pf_state_support(
    sources: Sequence[PointSource],
    *,
    candidate_isotopes: Sequence[str],
    max_sources_per_isotope: int,
    strength_prior_min_cps_1m: float,
    strength_prior_max_cps_1m: float,
) -> None:
    """Reject truth that the configured PF state space cannot represent."""
    maximum_cardinality = _strict_json_integer(
        max_sources_per_isotope,
        name="max_sources_per_isotope",
        minimum=0,
    )
    minimum_strength = _strict_json_number(
        strength_prior_min_cps_1m,
        name="strength_prior_min_cps_1m",
        minimum=0.0,
    )
    maximum_strength = _strict_json_number(
        strength_prior_max_cps_1m,
        name="strength_prior_max_cps_1m",
        minimum=minimum_strength,
        minimum_exclusive=True,
    )

    isotope_support = {str(name) for name in candidate_isotopes}
    cardinality_by_isotope = Counter(str(source.isotope) for source in sources)
    unsupported_isotopes = sorted(
        set(cardinality_by_isotope).difference(isotope_support)
    )
    if unsupported_isotopes:
        raise ValueError(
            "Simulation truth contains isotopes outside the PF state support: "
            f"{unsupported_isotopes}"
        )
    excess_cardinalities = {
        isotope: int(cardinality)
        for isotope, cardinality in sorted(cardinality_by_isotope.items())
        if int(cardinality) > maximum_cardinality
    }
    if excess_cardinalities:
        raise ValueError(
            "Simulation truth cardinality exceeds PF max_sources for these "
            f"isotopes: {excess_cardinalities}; max={maximum_cardinality}"
        )

    invalid_strengths = [
        {
            "source_index": int(index),
            "isotope": str(source.isotope),
            "intensity_cps_1m": float(source.intensity_cps_1m),
        }
        for index, source in enumerate(sources)
        if (
            not np.isfinite(float(source.intensity_cps_1m))
            or float(source.intensity_cps_1m) < minimum_strength
            or float(source.intensity_cps_1m) > maximum_strength
        )
    ]
    if invalid_strengths:
        raise ValueError(
            "Simulation truth contains strengths outside the PF prior support "
            f"[{minimum_strength}, {maximum_strength}]: {invalid_strengths}"
        )
    positions_by_isotope: dict[str, set[tuple[float, float, float]]] = {}
    duplicate_sources: list[dict[str, object]] = []
    for index, source in enumerate(sources):
        isotope = str(source.isotope)
        position = tuple(float(value) for value in source.position)
        isotope_positions = positions_by_isotope.setdefault(isotope, set())
        if position in isotope_positions:
            duplicate_sources.append(
                {
                    "source_index": int(index),
                    "isotope": isotope,
                    "position_xyz_m": list(position),
                }
            )
        isotope_positions.add(position)
    if duplicate_sources:
        raise ValueError(
            "Simulation truth contains exactly co-located same-isotope "
            "sources whose cardinality is not identifiable: "
            f"{duplicate_sources}"
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


def _pure_pf_summary_provenance(
    estimator: object,
    *,
    posterior_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Embed mandatory PF provenance in every runtime summary result."""
    if not _pure_pf_profile_active(estimator):
        return {}
    if posterior_payload is None:
        snapshot_getter = getattr(estimator, "posterior_snapshot", None)
        if not callable(snapshot_getter):
            raise RuntimeError(
                "A pure PF result requires posterior_snapshot provenance."
            )
        snapshot = snapshot_getter()
        serializer = getattr(snapshot, "to_dict", None)
        if not callable(serializer):
            raise RuntimeError(
                "A pure PF posterior snapshot must be serializable."
            )
        payload = dict(serializer())
    else:
        payload = dict(posterior_payload)
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
SIMULATION_RUNTIME_ROOT = simulation_runtime_root()
RESULTS_DIR = ROOT / "results"
SPECTRUM_DIR = RESULTS_DIR / "spectrum"
PF_DIR = RESULTS_DIR / "pf"
IG_DIR = RESULTS_DIR / "IG"
BLENDER_ENV_DIR = RESULTS_DIR / "blender_environments"
SAVE_IG_GRIDS = False
OBSTACLE_LAYOUT_DIR = ROOT / "obstacle_layouts"
DEFAULT_SOURCE_CONFIG = (
    SIMULATION_RUNTIME_ROOT / "source_layouts" / "demo_sources.json"
)
DEFAULT_OBSTACLE_CONFIG = SIMULATION_RUNTIME_ROOT / DEFAULT_FIXED_OBSTACLE_CONFIG
DEFAULT_PF_CONFIG = ROOT / "configs" / "pf" / "pf_strict_3d.json"
HEALTH_LOG_TOP_K = 0


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


def _pf_obstacle_attenuation_enabled(
    runtime_config: Mapping[str, object],
) -> bool:
    """Return the explicit Boolean PF obstacle-attenuation policy."""
    raw = runtime_config.get("pf_obstacle_attenuation", True)
    if not isinstance(raw, bool):
        raise ValueError("pf_obstacle_attenuation must be a boolean.")
    return raw


def _optional_runtime_bool(
    runtime_config: Mapping[str, object],
    key: str,
) -> bool | None:
    """Return an optional strict Boolean runtime setting."""
    if key not in runtime_config or runtime_config[key] is None:
        return None
    raw = runtime_config[key]
    if isinstance(raw, bool):
        return raw
    raise ValueError(f"Runtime config key {key!r} must be a boolean.")


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


def _resolve_rotation_limit_for_active_program(
    *,
    base_rotation_limit: int,
    active_shield_program: Sequence[int] | None,
    strict_planned_shield_program: bool,
    baseline_shield_policy: Mapping[str, Any] | str | None,
) -> int:
    """Return the rotation limit for a station with an explicit shield program."""
    base_limit = max(1, int(base_rotation_limit))
    if not active_shield_program:
        return base_limit
    program_limit = max(1, len(active_shield_program))
    if (
        strict_planned_shield_program
        or baseline_shield_policy is not None
    ):
        return program_limit
    return max(base_limit, program_limit)


def _pf_obstacle_grid_for_runtime(
    obstacle_grid: ObstacleGrid | None,
    runtime_config: Mapping[str, object],
) -> ObstacleGrid | None:
    """Return the obstacle grid used by the PF observation model."""
    if _pf_obstacle_attenuation_enabled(runtime_config):
        return obstacle_grid
    if _has_environment_obstacles(obstacle_grid):
        raise ValueError(
            "pf_obstacle_attenuation=false is invalid when physical "
            "environment obstacles are active."
        )
    return None


_SURFACE_REPORT_LABELS = SOURCE_SURFACE_REPORT_LABELS


def _surface_kind_counts_from_array(
    surface_kinds: NDArray[np.object_],
) -> dict[str, int]:
    """Return report counts from authoritative or legacy surface-kind labels."""
    kinds = np.asarray(surface_kinds, dtype=object).reshape(-1)
    counts = {label: 0 for label in _SURFACE_REPORT_LABELS}
    for label in _SURFACE_REPORT_LABELS[:-1]:
        counts[label] = int(np.count_nonzero(kinds == label))
    counts["off_surface"] = int(np.count_nonzero(np.equal(kinds, None)))
    return counts


def _surface_count_payload(
    positions: NDArray[np.float64],
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    *,
    obstacle_height_m: float,
    tolerance_m: float = 1.0e-5,
    surface_kinds: NDArray[np.object_] | None = None,
) -> dict[str, object]:
    """Return serializable surface-kind counts for source positions."""
    pos_arr = np.asarray(positions, dtype=float).reshape(-1, 3)
    resolved_kinds = (
        source_surface_kinds(
            pos_arr,
            env,
            obstacle_grid,
            obstacle_height_m=obstacle_height_m,
            tolerance_m=max(float(tolerance_m), 0.0),
        )
        if surface_kinds is None
        else np.asarray(surface_kinds, dtype=object).reshape(-1)
    )
    if resolved_kinds.size != pos_arr.shape[0]:
        raise ValueError("surface_kinds must have one label per source position.")
    counts = _surface_kind_counts_from_array(resolved_kinds)
    return {
        "total_sources": int(pos_arr.shape[0]),
        "surface_counts": counts,
        "off_surface_count": int(counts.get("off_surface", 0)),
    }


def _estimate_surface_diagnostics(
    estimates: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    *,
    obstacle_height_m: float,
    tolerance_m: float = 1.0e-5,
    estimator: PurePFEstimator | None = None,
) -> dict[str, dict[str, object]]:
    """Return per-isotope surface diagnostics for final reported estimates."""
    return {
        isotope: _surface_count_payload(
            np.asarray(positions, dtype=float),
            env,
            obstacle_grid,
            obstacle_height_m=obstacle_height_m,
            tolerance_m=tolerance_m,
            surface_kinds=(
                None
                if estimator is None
                else estimator.structural_surface_kinds(
                    isotope,
                    np.asarray(positions, dtype=float).reshape(-1, 3),
                    strict=True,
                )
            ),
        )
        for isotope, (positions, _strengths) in estimates.items()
    }


def _validated_reporting_particle_state(
    filt: object,
    state: object,
    *,
    name: str,
) -> tuple[
    int,
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.object_],
]:
    """Return one exact continuous-surface state for scientific reporting."""
    config = filt.config
    maximum_cardinality = config.max_sources
    if maximum_cardinality is None:
        raise RuntimeError(
            "PF reporting requires a finite configured source-count support."
        )
    try:
        cardinality = validated_state_cardinality(
            state,
            name=name,
            max_cardinality=maximum_cardinality,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(str(exc)) from exc
    strengths = np.asarray(state.strengths, dtype=np.float64)
    raw_chart_ids = np.asarray(state.surface_chart_ids)
    surface_uv = np.asarray(state.surface_uv, dtype=np.float64)
    if (
        strengths.shape != (cardinality,)
        or np.any(~np.isfinite(strengths))
        or np.any(strengths <= 0.0)
        or not np.issubdtype(raw_chart_ids.dtype, np.integer)
        or raw_chart_ids.shape != (cardinality,)
        or np.any(raw_chart_ids < 0)
        or surface_uv.shape != (cardinality, 2)
        or np.any(~np.isfinite(surface_uv))
        or np.any(surface_uv < 0.0)
        or np.any(surface_uv > 1.0)
    ):
        raise RuntimeError(
            f"{name} has state arrays inconsistent with num_sources."
        )
    chart_ids = np.asarray(raw_chart_ids, dtype=np.int64)
    positions = np.asarray(
        filt.continuous_state_positions(state),
        dtype=np.float64,
    )
    if positions.shape != (cardinality, 3) or np.any(~np.isfinite(positions)):
        raise RuntimeError(
            f"{name} does not resolve to one finite XYZ point per source."
        )
    atlas = filt._structural_rj_surface_atlas
    if atlas is None:
        raise RuntimeError("PF reporting requires a continuous surface atlas.")
    if np.any(chart_ids >= atlas.chart_count):
        raise RuntimeError(f"{name} contains a chart outside the surface atlas.")
    kinds = np.asarray(atlas.geometry.kinds, dtype=object)[chart_ids]
    return cardinality, positions, chart_ids, surface_uv, kinds


def _particle_surface_diagnostics(
    estimator: PurePFEstimator,
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
        kinds_by_state: list[NDArray[np.object_]] = []
        posterior_slots = 0
        filt.validate_continuous_surface_states()
        raw_particle_weights = np.asarray(filt.continuous_weights)
        if raw_particle_weights.ndim != 1:
            raise RuntimeError("PF particle weights must be a one-dimensional array.")
        particle_weights = np.asarray(raw_particle_weights, dtype=np.float64)
        particles = list(filt.continuous_particles)
        if not particles or particle_weights.size != len(particles):
            raise RuntimeError(
                "PF particle diagnostics require one posterior weight per particle."
            )
        try:
            particle_weights = validated_probability_distribution(
                particle_weights,
                name=f"{isotope} final PF particle weights",
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        for particle_index, (particle, weight) in enumerate(
            zip(particles, particle_weights, strict=True)
        ):
            state = particle.state
            (
                count,
                state_positions,
                _chart_ids,
                _surface_uv,
                state_kinds,
            ) = _validated_reporting_particle_state(
                filt,
                state,
                name=f"{isotope} particle[{particle_index}]",
            )
            posterior_slots += count
            if count == 0:
                continue
            positions.append(state_positions)
            weights.append(np.full(count, float(weight), dtype=float))
            kinds_by_state.append(state_kinds)
        if positions:
            weight_arr = np.concatenate(weights)
            kinds = np.concatenate(kinds_by_state)
        else:
            weight_arr = np.zeros(0, dtype=float)
            kinds = np.zeros(0, dtype=object)
        counts = _surface_kind_counts_from_array(kinds)
        weighted = {
            label: float(np.sum(weight_arr[kinds == label]))
            for label in _SURFACE_REPORT_LABELS[:-1]
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
    estimator: PurePFEstimator,
    *,
    max_points_per_isotope: int = 1200,
) -> dict[str, dict[str, object]]:
    """Return deterministic final PF source-slot samples for paper figures."""
    output: dict[str, dict[str, object]] = {}
    max_points = _strict_json_integer(
        max_points_per_isotope,
        name="max_points_per_isotope",
        minimum=1,
    )
    for isotope, filt in estimator.filters.items():
        filt.validate_continuous_surface_states()
        raw_particle_weights = np.asarray(filt.continuous_weights)
        if raw_particle_weights.ndim != 1:
            raise RuntimeError("PF particle weights must be a one-dimensional array.")
        particle_weights = np.asarray(raw_particle_weights, dtype=np.float64)
        particles = list(filt.continuous_particles)
        if not particles or particle_weights.size != len(particles):
            raise RuntimeError(
                "Final particle cloud requires one valid weight per particle."
            )
        try:
            particle_weights = validated_probability_distribution(
                particle_weights,
                name=f"{isotope} final PF particle weights",
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        positions: list[NDArray[np.float64]] = []
        weights: list[NDArray[np.float64]] = []
        chart_ids_by_state: list[NDArray[np.int64]] = []
        surface_uv_by_state: list[NDArray[np.float64]] = []
        for particle_index, (particle, weight) in enumerate(
            zip(particles, particle_weights, strict=True)
        ):
            state = particle.state
            (
                count,
                active_positions,
                active_chart_ids,
                active_surface_uv,
                _surface_kinds,
            ) = _validated_reporting_particle_state(
                filt,
                state,
                name=f"{isotope} particle[{particle_index}]",
            )
            if count == 0:
                continue
            positions.append(active_positions)
            weights.append(
                np.full(active_positions.shape[0], float(weight), dtype=float)
            )
            chart_ids_by_state.append(active_chart_ids)
            surface_uv_by_state.append(active_surface_uv)
        if positions:
            position_arr = np.vstack(positions)
            weight_arr = np.concatenate(weights)
            chart_id_arr = np.concatenate(chart_ids_by_state)
            surface_uv_arr = np.vstack(surface_uv_by_state)
            if position_arr.shape[0] > max_points:
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
                chart_id_arr = chart_id_arr[order]
                surface_uv_arr = surface_uv_arr[order]
        else:
            position_arr = np.zeros((0, 3), dtype=float)
            weight_arr = np.zeros(0, dtype=float)
            chart_id_arr = np.zeros(0, dtype=np.int64)
            surface_uv_arr = np.zeros((0, 2), dtype=np.float64)
        output[isotope] = {
            "positions": position_arr.tolist(),
            "weights": weight_arr.tolist(),
            "surface_chart_ids": chart_id_arr.tolist(),
            "surface_uv": surface_uv_arr.tolist(),
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


@dataclass(frozen=True)
class LoadedSourceConfiguration:
    """Hold explicit source truth and provenance from the exact parsed bytes."""

    sources: tuple[PointSource, ...]
    provenance: Mapping[str, object]


def _normalized_source_file_path(
    path: Path,
    *,
    repository_root: Path,
) -> tuple[str, str]:
    """Return an auditable repository-relative or resolved-absolute path."""
    resolved_path = path.expanduser().resolve(strict=True)
    resolved_root = repository_root.expanduser().resolve(strict=True)
    try:
        relative_path = resolved_path.relative_to(resolved_root)
    except ValueError:
        return resolved_path.as_posix(), "resolved_absolute"
    return relative_path.as_posix(), "repository_relative"


def _reject_nonfinite_json_constant(value: str) -> object:
    """Reject non-standard NaN and infinity constants in source truth JSON."""
    raise ValueError(
        f"Source config contains non-standard non-finite JSON constant {value!r}."
    )


def _strict_source_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    """Build one source JSON object while rejecting duplicate member names."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(
                f"Source config contains duplicate JSON object key {key!r}."
            )
        result[key] = value
    return result


def load_sources_from_json(
    path: Path,
    *,
    repository_root: Path = ROOT,
) -> LoadedSourceConfiguration:
    """Load a strict explicit source file and retain its exact-byte provenance."""
    resolved_path = path.expanduser().resolve(strict=True)
    raw_bytes = resolved_path.read_bytes()
    try:
        data = json.loads(
            raw_bytes.decode("utf-8"),
            parse_constant=_reject_nonfinite_json_constant,
            object_pairs_hook=_strict_source_json_object,
        )
    except UnicodeDecodeError as exc:
        raise ValueError("Source config must be UTF-8 JSON.") from exc
    if isinstance(data, dict):
        allowed_top_level_keys = {"name", "metadata", "sources"}
        unexpected_top_level_keys = sorted(
            set(data).difference(allowed_top_level_keys)
        )
        if unexpected_top_level_keys:
            raise ValueError(
                "Source config contains unsupported top-level fields: "
                f"{unexpected_top_level_keys}."
            )
        if "sources" not in data:
            raise ValueError(
                "Source config object must include an explicit 'sources' list."
            )
        entries = data["sources"]
        declared_metadata = data.get("metadata", {})
        if not isinstance(declared_metadata, dict):
            raise ValueError("Source config top-level 'metadata' must be an object.")
    elif isinstance(data, list):
        entries = data
        declared_metadata = {}
    else:
        raise ValueError("Source config must be a list or include a 'sources' list.")
    if not isinstance(entries, list):
        raise ValueError("Source config 'sources' must be a list.")
    sources: list[PointSource] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Source entry {idx} must be an object.")
        legacy_keys = sorted(
            key for key in ("strength_cps_1m", "intensity") if key in entry
        )
        if legacy_keys:
            raise ValueError(
                f"Source entry {idx} uses removed intensity field(s) "
                f"{legacy_keys}; use only 'intensity_cps_1m'."
            )
        minimal_keys = {"isotope", "position", "intensity_cps_1m"}
        entry_keys = set(entry)
        if entry_keys == SURFACE_SOURCE_RUNTIME_KEYS:
            normalized = canonical_surface_source_runtime_payload([entry])[0]
            sources.append(
                PointSource(
                    isotope=str(normalized["isotope"]),
                    position=tuple(normalized["position"]),
                    intensity_cps_1m=float(normalized["intensity_cps_1m"]),
                    surface_chart_id=int(normalized["surface_chart_id"]),
                    surface_uv=tuple(normalized["surface_uv"]),
                    surface_normal=tuple(normalized["surface_normal"]),
                    transport_position=tuple(
                        normalized["transport_position"]
                    ),
                    surface_emission_policy_sha256=str(
                        normalized["surface_emission_policy_sha256"]
                    ),
                )
            )
            continue
        if entry_keys != minimal_keys:
            unexpected_keys = sorted(
                entry_keys - minimal_keys - SURFACE_SOURCE_RUNTIME_KEYS
            )
            missing_minimal = sorted(minimal_keys - entry_keys)
            raise ValueError(
                f"Source entry {idx} must use either the exact minimal "
                "anchor schema or the complete surface chart/UV runtime "
                f"schema; missing_minimal={missing_minimal}, "
                f"unexpected={unexpected_keys}."
            )
        isotope = entry.get("isotope")
        position = entry.get("position")
        intensity = entry.get("intensity_cps_1m")
        if isotope is None or position is None or intensity is None:
            raise ValueError(
                "Each source must include 'isotope', 'position', and 'intensity_cps_1m'."
            )
        if not isinstance(isotope, str) or not isotope.strip():
            raise ValueError(f"Source entry {idx} isotope must be a non-empty string.")
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            raise ValueError(f"Source entry {idx} position must be a 3-element list.")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in position
        ):
            raise ValueError(
                f"Source entry {idx} position must contain three JSON numbers."
            )
        if isinstance(intensity, bool) or not isinstance(intensity, (int, float)):
            raise ValueError(
                f"Source entry {idx} intensity_cps_1m must be a JSON number."
            )
        sources.append(
            PointSource(
                isotope=isotope.strip(),
                position=(float(position[0]), float(position[1]), float(position[2])),
                intensity_cps_1m=float(intensity),
            )
        )
    normalized_path, path_kind = _normalized_source_file_path(
        resolved_path,
        repository_root=repository_root,
    )
    provenance: dict[str, object] = {
        "provided_file_path": normalized_path,
        "provided_file_path_kind": path_kind,
        "provided_file_bytes_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "provided_file_declared_metadata": dict(declared_metadata),
    }
    return LoadedSourceConfiguration(
        sources=tuple(sources),
        provenance=provenance,
    )


def _validated_provided_source_provenance(
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Validate explicit source-file provenance before publishing run metadata."""
    required_keys = {
        "provided_file_path",
        "provided_file_path_kind",
        "provided_file_bytes_sha256",
        "provided_file_declared_metadata",
    }
    if set(provenance) != required_keys:
        missing = sorted(required_keys - set(provenance))
        unexpected = sorted(set(provenance) - required_keys)
        raise ValueError(
            "provided-file source provenance has incompatible fields: "
            f"missing={missing}, unexpected={unexpected}."
        )
    path_value = provenance["provided_file_path"]
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("provided_file_path must be a non-empty string.")
    if "\\" in path_value:
        raise ValueError("provided_file_path must use normalized POSIX separators.")
    path_kind = provenance["provided_file_path_kind"]
    if path_kind not in {"repository_relative", "resolved_absolute"}:
        raise ValueError(
            "provided_file_path_kind must be 'repository_relative' or "
            "'resolved_absolute'."
        )
    normalized_path = Path(path_value)
    if path_kind == "repository_relative":
        if normalized_path.is_absolute() or ".." in normalized_path.parts:
            raise ValueError(
                "repository-relative provided_file_path must remain within the "
                "repository."
            )
    elif not normalized_path.is_absolute():
        raise ValueError("resolved-absolute provided_file_path must be absolute.")
    digest = provenance["provided_file_bytes_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(
            "provided_file_bytes_sha256 must be a lowercase SHA-256 digest."
        )
    declared_metadata = provenance["provided_file_declared_metadata"]
    if not isinstance(declared_metadata, Mapping):
        raise ValueError("provided_file_declared_metadata must be an object.")
    sanitized_metadata = _sanitize_json_payload(
        dict(declared_metadata),
        _unsafe_integers_as_decimal_strings=True,
    )
    if not isinstance(sanitized_metadata, dict):
        raise TypeError("provided_file_declared_metadata must sanitize to an object.")
    return {
        "provided_file_path": path_value,
        "provided_file_path_kind": path_kind,
        "provided_file_bytes_sha256": digest,
        "provided_file_declared_metadata": sanitized_metadata,
    }


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


def _estimate_map_to_metric_sources(
    estimates: Mapping[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
) -> dict[str, list[dict[str, float | list[float]]]]:
    """Convert an estimator estimate map into compute_metrics source records."""
    payload: dict[str, list[dict[str, float | list[float]]]] = {}
    for isotope, (positions, strengths) in estimates.items():
        pos_arr = np.asarray(positions, dtype=float)
        q_arr = np.asarray(strengths, dtype=float)
        isotope_name = str(isotope)
        if pos_arr.ndim != 2 or pos_arr.shape[1:] != (3,):
            raise ValueError(
                "Online metric positions must have shape (N, 3) for isotope "
                f"{isotope_name!r}."
            )
        if q_arr.ndim != 1:
            raise ValueError(
                "Online metric strengths must have shape (N,) for isotope "
                f"{isotope_name!r}."
            )
        if pos_arr.shape[0] != q_arr.size:
            raise ValueError(
                "Online metric positions and strengths must contain the same "
                f"number of sources for isotope {isotope_name!r}; got "
                f"{pos_arr.shape[0]} positions and {q_arr.size} strengths."
            )
        if np.any(~np.isfinite(pos_arr)):
            raise ValueError(
                f"Online metric positions must be finite for isotope {isotope_name!r}."
            )
        if np.any(~np.isfinite(q_arr)) or np.any(q_arr < 0.0):
            raise ValueError(
                "Online metric strengths must be finite and non-negative for "
                f"isotope {isotope_name!r}."
            )
        payload[str(isotope)] = [
            {
                "pos": [float(value) for value in pos_arr[idx]],
                "strength": float(q_arr[idx]),
            }
            for idx in range(pos_arr.shape[0])
        ]
    return payload


def _online_estimate_metric_summary(
    history_estimates: Sequence[
        Mapping[str, tuple[NDArray[np.float64], NDArray[np.float64]]]
    ],
    gt_by_iso: Mapping[str, list[dict[str, float | list[float]]]],
    *,
    match_radius_m: float,
    surface_atlas: object,
) -> dict[str, dict[str, float | int | bool | str | None]]:
    """Summarize online source-count and localization stability over PF history."""
    summaries: dict[str, dict[str, float | int | bool | str | None]] = {}
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
                "mean_online_surface_path_error_m": None,
                "final_online_surface_path_error_m": None,
                "final_online_surface_path_error_available": False,
                "final_online_surface_path_error_unavailable_reason": "no_history",
                "last_evaluable_online_surface_path_error_m": None,
                "last_evaluable_online_surface_path_error_step": None,
            }
            for isotope in isotopes
        }
    cardinality_errors: dict[str, list[int]] = {isotope: [] for isotope in isotopes}
    mean_errors: dict[str, list[float | None]] = {isotope: [] for isotope in isotopes}
    for step_index, estimate_map in enumerate(history_estimates):
        metrics = compute_metrics(
            dict(gt_by_iso),
            _estimate_map_to_metric_sources(estimate_map),
            match_radius_m=float(match_radius_m),
            surface_atlas=surface_atlas,
        )
        if not isinstance(metrics, Mapping):
            raise RuntimeError(
                f"Online metric step {step_index} did not return a metric mapping."
            )
        isotope_metrics = metrics.get("isotopes")
        if not isinstance(isotope_metrics, Mapping):
            raise RuntimeError(
                f"Online metric step {step_index} is missing the required "
                "'isotopes' mapping; legacy metric schemas are unsupported."
            )
        for isotope in isotopes:
            if isotope not in isotope_metrics:
                raise RuntimeError(
                    f"Online metric step {step_index} is missing required isotope "
                    f"{isotope!r}; legacy metric schemas are unsupported."
                )
            data = isotope_metrics[isotope]
            if not isinstance(data, Mapping):
                raise RuntimeError(
                    f"Online metric step {step_index} isotope {isotope!r} must "
                    "contain a metric mapping."
                )
            counts = data.get("counts")
            if not isinstance(counts, Mapping) or "source_count_error" not in counts:
                raise RuntimeError(
                    f"Online metric step {step_index} isotope {isotope!r} is "
                    "missing required counts.source_count_error; legacy metric "
                    "schemas are unsupported."
                )
            raw_count_error = counts["source_count_error"]
            if isinstance(raw_count_error, bool) or not isinstance(
                raw_count_error,
                int,
            ):
                raise RuntimeError(
                    f"Online metric step {step_index} isotope {isotope!r} "
                    "counts.source_count_error must be a JSON integer."
                )
            pos_summary = data.get("surface_path_error")
            if not isinstance(pos_summary, Mapping) or "mean" not in pos_summary:
                raise RuntimeError(
                    f"Online metric step {step_index} isotope {isotope!r} is "
                    "missing required surface_path_error.mean; legacy metric "
                    "schemas are unsupported."
                )
            raw_mean_error = pos_summary["mean"]
            if raw_mean_error is None:
                mean_error = None
            else:
                try:
                    mean_error = _strict_json_number(
                        raw_mean_error,
                        name=(
                            f"online metric step {step_index} isotope "
                            f"{isotope!r} surface_path_error.mean"
                        ),
                    )
                except ValueError as exc:
                    raise RuntimeError(str(exc)) from exc
            cardinality_errors[isotope].append(int(raw_count_error))
            mean_errors[isotope].append(mean_error)
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
        evaluable_mean_errors = [
            (step_index, float(value))
            for step_index, value in enumerate(mean_errors[isotope])
            if value is not None
        ]
        final_mean_error = mean_errors[isotope][-1] if errors else None
        last_evaluable_step, last_evaluable_error = (
            evaluable_mean_errors[-1]
            if evaluable_mean_errors
            else (None, None)
        )
        mean_online_error = (
            float(
                np.mean(
                    [value for _step_index, value in evaluable_mean_errors],
                    dtype=np.float64,
                )
            )
            if evaluable_mean_errors
            else None
        )
        if mean_online_error is not None and not np.isfinite(mean_online_error):
            raise RuntimeError(
                f"Online metric mean overflowed for isotope {isotope!r}."
            )
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
            "mean_online_surface_path_error_m": mean_online_error,
            "final_online_surface_path_error_m": final_mean_error,
            "final_online_surface_path_error_available": (
                final_mean_error is not None
            ),
            "final_online_surface_path_error_unavailable_reason": (
                None
                if final_mean_error is not None
                else "no_gated_localization_match"
            ),
            "last_evaluable_online_surface_path_error_m": last_evaluable_error,
            "last_evaluable_online_surface_path_error_step": last_evaluable_step,
        }
    return summaries


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


def _thin_spectrum_for_notification(
    energy_keV: NDArray[np.float64],
    counts: NDArray[np.float64],
    max_bins: int,
) -> tuple[list[float], list[float]]:
    """Return spectrum arrays thinned to a notification-friendly size."""
    energy_raw = np.asarray(energy_keV)
    values_raw = np.asarray(counts)
    if energy_raw.ndim != 1 or values_raw.ndim != 1:
        raise ValueError("Notification spectrum arrays must both be one-dimensional.")
    energy = np.asarray(energy_raw, dtype=float)
    values = np.asarray(values_raw, dtype=float)
    if energy.shape != values.shape or energy.size == 0:
        raise ValueError(
            "Notification energy and spectrum arrays must have the same "
            "nonempty shape."
        )
    if (
        np.any(~np.isfinite(energy))
        or np.any(~np.isfinite(values))
        or np.any(values < 0.0)
    ):
        raise ValueError(
            "Notification energy and spectrum values must be finite, with "
            "nonnegative spectrum counts."
        )
    limit = _strict_json_integer(
        max_bins,
        name="notification max_bins",
        minimum=0,
    )
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
    energy_axis_keV: NDArray[np.float64],
    spectrum: NDArray[np.int64],
    step_index: int,
    pose_xyz: NDArray[np.float64],
    fe_index: int,
    pb_index: int,
    live_time_s: float,
    full_spectrum_contract_hash_sha256: str,
    max_bins: int,
) -> dict[str, object]:
    """Build a compact raw-spectrum payload for piplup/Railway display."""
    spectrum_values = np.asarray(spectrum, dtype=float)
    energy_keV, spectrum_counts = _thin_spectrum_for_notification(
        np.asarray(energy_axis_keV, dtype=float),
        spectrum_values,
        max_bins,
    )
    return {
        "step_index": int(step_index),
        "pose_xyz": [float(v) for v in np.asarray(pose_xyz, dtype=float)],
        "fe_index": int(fe_index),
        "pb_index": int(pb_index),
        "live_time_s": float(live_time_s),
        FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY: str(
            full_spectrum_contract_hash_sha256
        ),
        "observation_semantics": "raw_joint_full_spectrum_integer_counts",
        "total_spectrum_counts": float(np.sum(spectrum_values)),
        "max_bin_count": float(np.max(spectrum_values))
        if spectrum_values.size
        else 0.0,
        "energy_keV": energy_keV,
        "spectrum_counts": spectrum_counts,
    }


def _fmt_sources(positions: NDArray[np.float64], strengths: NDArray[np.float64]) -> str:
    """Format a list of source positions/strengths for logging."""
    positions_raw = np.asarray(positions)
    strengths_raw = np.asarray(strengths)
    if positions_raw.ndim != 2 or positions_raw.shape[1:] != (3,):
        raise ValueError("Logged source positions must have shape (N, 3).")
    if strengths_raw.ndim != 1:
        raise ValueError("Logged source strengths must have shape (N,).")
    if positions_raw.shape[0] != strengths_raw.size:
        raise ValueError(
            "Logged source positions and strengths must contain the same count."
        )
    positions = np.asarray(positions_raw, dtype=float)
    strengths = np.asarray(strengths_raw, dtype=float)
    if (
        np.any(~np.isfinite(positions))
        or np.any(~np.isfinite(strengths))
        or np.any(strengths < 0.0)
    ):
        raise ValueError(
            "Logged source states must be finite with nonnegative strengths."
        )
    if positions.shape[0] == 0:
        return "[]"
    count = min(int(positions.shape[0]), 8)
    positions = positions[:count]
    strengths = strengths[:count]
    chunks = []
    for pos, strength in zip(positions, strengths, strict=True):
        pos_str = np.array2string(pos, precision=2, floatmode="fixed", separator=", ")
        chunks.append(f"{pos_str}|{float(strength):.2f}")
    return "[" + ", ".join(chunks) + "]"


def _frame_field(
    frame: PFFrame | dict[str, object], name: str, default: object
) -> object:
    """Return a PFFrame field from either a dataclass frame or test stub dict."""
    if isinstance(frame, dict):
        return frame.get(name, default)
    return getattr(frame, name, default)


def _current_pf_posterior_estimate_trace_frame(
    estimator: PurePFEstimator,
    isotopes: Sequence[str],
    frame: PFFrame | dict[str, object],
    *,
    step_index: int,
    elapsed_s: float,
    estimate_source: str = "current_pf_posterior",
) -> dict[str, object]:
    """Return a trace frame from the canonical MAP-cardinality PF posterior."""
    estimated_sources: dict[str, NDArray[np.float64]] = {}
    estimated_strengths: dict[str, NDArray[np.float64]] = {}
    posterior_estimates = estimator.estimates()
    if not isinstance(posterior_estimates, Mapping):
        raise RuntimeError("PF posterior trace requires an isotope mapping.")
    expected = tuple(str(name) for name in isotopes)
    if (
        len(set(expected)) != len(expected)
        or set(posterior_estimates) != set(expected)
    ):
        raise RuntimeError(
            "PF posterior trace must contain exactly every configured isotope."
        )
    for isotope in sorted(expected):
        estimate = posterior_estimates[isotope]
        if (
            not isinstance(estimate, Sequence)
            or isinstance(estimate, (str, bytes))
            or len(estimate) != 2
        ):
            raise ValueError(
                "PF posterior trace estimates must be position/strength pairs."
            )
        positions = np.asarray(estimate[0], dtype=float)
        strengths = np.asarray(estimate[1], dtype=float)
        if (
            positions.ndim != 2
            or positions.shape[1:] != (3,)
            or strengths.ndim != 1
            or strengths.size != positions.shape[0]
        ):
            raise ValueError(
                "PF posterior trace requires one strength per source position."
            )
        if (
            np.any(~np.isfinite(positions))
            or np.any(~np.isfinite(strengths))
            or np.any(strengths <= 0.0)
        ):
            raise ValueError(
                "PF posterior trace values must be finite with positive strengths."
            )
        estimated_sources[isotope] = positions.copy()
        estimated_strengths[isotope] = strengths.copy()
    robot_position = np.asarray(
        _frame_field(frame, "robot_position", np.zeros(3)),
        dtype=float,
    )
    if robot_position.shape != (3,) or np.any(~np.isfinite(robot_position)):
        raise ValueError("PF posterior trace robot_position must be finite XYZ.")
    return {
        "estimate_source": str(estimate_source),
        "step_index": int(step_index),
        "time": float(elapsed_s),
        "robot_position": robot_position.copy(),
        "estimated_sources": estimated_sources,
        "estimated_strengths": estimated_strengths,
    }


def _build_intermediate_estimate_trace_payload(
    frame: PFFrame | dict[str, object],
    *,
    surface_kinds_by_isotope: Mapping[str, NDArray[np.object_]] | None = None,
) -> dict[str, object]:
    """Build a truth-free JSON payload from the current PF posterior."""
    payload_records: list[dict[str, object]] = []
    summaries: dict[str, dict[str, object]] = {}
    estimated_sources_raw = _frame_field(frame, "estimated_sources", {})
    estimated_strengths_raw = _frame_field(frame, "estimated_strengths", {})
    estimated_sources = (
        dict(estimated_sources_raw) if isinstance(estimated_sources_raw, dict) else {}
    )
    estimated_strengths = (
        dict(estimated_strengths_raw)
        if isinstance(estimated_strengths_raw, dict)
        else {}
    )
    isotope_names = sorted(
        set(estimated_sources) | set(estimated_strengths)
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
        if est_strengths.shape != (est_positions.shape[0],):
            raise ValueError(
                "Intermediate PF trace requires one strength per source."
            )
        if (
            np.any(~np.isfinite(est_positions))
            or np.any(~np.isfinite(est_strengths))
            or np.any(est_strengths <= 0.0)
        ):
            raise ValueError(
                "Intermediate PF trace requires finite positive source states."
            )
        exact_kinds = (
            None
            if surface_kinds_by_isotope is None
            else surface_kinds_by_isotope.get(isotope)
        )
        if exact_kinds is None:
            raise ValueError(
                "Intermediate PF trace requires authoritative surface labels."
            )
        surface_kinds = np.asarray(exact_kinds, dtype=object).reshape(-1)
        if surface_kinds.size != est_positions.shape[0]:
            raise ValueError(
                "surface_kinds_by_isotope must match estimated source counts."
            )
        isotope_records = [
            {
                "isotope": str(isotope),
                "estimate_index": int(index),
                "pos": [
                    float(value)
                    for value in est_positions[index]
                ],
                "strength": float(est_strengths[index]),
                "surface_kind": str(surface_kinds[index]),
            }
            for index in range(est_positions.shape[0])
        ]
        payload_records.extend(isotope_records)
        summaries[str(isotope)] = {
            "estimate_count": int(est_positions.shape[0]),
            "total_est_strength": float(np.sum(est_strengths)),
        }
    robot_position = np.asarray(
        _frame_field(frame, "robot_position", np.zeros(3)),
        dtype=float,
    )
    if robot_position.shape != (3,) or np.any(~np.isfinite(robot_position)):
        raise ValueError("Intermediate PF trace robot_position must be finite XYZ.")
    return {
        "estimate_source": str(_frame_field(frame, "estimate_source", "frame")),
        "step_index": int(_frame_field(frame, "step_index", -1)),
        "time_s": float(_frame_field(frame, "time", 0.0)),
        "robot_position": [float(v) for v in robot_position],
        "isotopes": summaries,
        "estimates": payload_records,
    }


def _format_estimate_trace_log_line(
    step_index: int,
    isotope: str,
    summary: dict[str, object],
    records: list[dict[str, object]],
    *,
    max_records: int = 6,
) -> str:
    """Format one truth-free posterior trace line for the console log."""
    estimate_source = str(summary.get("estimate_source", "frame"))
    chunks: list[str] = []
    for record in records[: max(0, int(max_records))]:
        pos = np.asarray(record.get("pos", np.zeros(3)), dtype=float)
        chunks.append(
            "#"
            f"{int(record.get('estimate_index', 0))}"
            f" pos={_fmt_pos(pos)}"
            f" q={float(record.get('strength', 0.0)):.1f}"
            f" surface={record.get('surface_kind')}"
        )
    return (
        f"[step {step_index}] pf_estimates[{isotope}] "
        f"mode={estimate_source} "
        f"n={int(summary.get('estimate_count', 0))} "
        f"total_q={float(summary.get('total_est_strength', 0.0)):.1f} "
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
    estimator: PurePFEstimator,
    isotopes: Sequence[str],
    frame: PFFrame | dict[str, object],
    *,
    step_index: int,
    elapsed_s: float,
    trace_path: Path | None,
    log_enabled: bool,
    log_every: int,
    max_log_records: int,
    estimate_source: str,
) -> None:
    """Write one station-complete canonical PF-posterior trace diagnostic."""
    if estimate_source != "post_joint_update_pf_posterior":
        raise ValueError(
            "Scientific intermediate traces may only be emitted after the "
            "joint station PF update."
        )
    estimate_trace_frame = _current_pf_posterior_estimate_trace_frame(
        estimator,
        isotopes,
        frame,
        step_index=step_index,
        elapsed_s=elapsed_s,
        estimate_source=estimate_source,
    )
    trace_sources = dict(estimate_trace_frame["estimated_sources"])
    surface_kinds_by_isotope = {
        str(isotope): estimator.structural_surface_kinds(
            str(isotope),
            np.asarray(positions, dtype=float).reshape(-1, 3),
            strict=True,
        )
        for isotope, positions in trace_sources.items()
    }
    estimate_trace_payload = _build_intermediate_estimate_trace_payload(
        estimate_trace_frame,
        surface_kinds_by_isotope=surface_kinds_by_isotope,
    )
    if trace_path is not None:
        _append_estimate_trace_jsonl(trace_path, estimate_trace_payload)
    if not log_enabled or step_index % max(1, int(log_every)) != 0:
        return
    estimate_records = list(estimate_trace_payload.get("estimates", []))
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


def _measurement_transport_provenance(
    metadata: dict[str, object],
) -> dict[str, object]:
    """Return Geant4 fidelity fields that must survive in measurement logs."""
    keys = (
        "accelerated_weighted_transport_enable",
        "background_cps",
        "background_spectrum_model_id",
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
        "scene_hash",
        "secondary_transport_mode",
        "source_anchor_semantics",
        "source_bias_weighted_transport",
        "source_bias_mode",
        "source_position_semantics",
        "source_rate_model",
        "all_sources_surface_bound",
        "surface_emission_epsilon_m",
        "surface_emission_policy_sha256",
        "surface_source_contract_sha256",
        "spectrum_variance_dead_time_propagation",
        "spectrum_variance_semantics",
        "spectrum_energy_min_keV",
        "spectrum_energy_max_keV",
        "spectrum_bin_width_keV",
        "spectrum_bin_count",
        "target_sampled_primaries",
        "theory_tvl_attenuation",
        "transport_history_mode",
        "transport_tally_weighted",
        "weighted_spectrum_effective_entries",
        "weighted_spectrum_sumw2",
        "weighted_transport",
    )
    return {key: metadata[key] for key in keys if key in metadata}


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


def _log_pf_diagnostics(
    estimator: PurePFEstimator,
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
        current_ess = float(stats["current_ess"])
        current_ess_ratio = float(stats["current_ess_ratio"])
        particle_count = int(stats["particle_count"])
        resamples = int(stats["resample_count"])
        births = int(stats["birth_count"])
        deaths = int(stats["death_count"])
        structural_timing = stats.get("structural_timing_s", {})
        transition_weight_mass = {
            str(key): float(value)
            for key, value in dict(
                stats.get("transition_weight_mass", {})
            ).items()
        }
        temper_steps = stats.get("temper_steps", [])
        temper_resamples = int(stats.get("temper_resamples", 0))
        temper_min_ess = stats.get("temper_min_ess")
        unique_ancestor_count = stats.get("unique_ancestor_count")
        r_mean = float(stats["r_mean"])
        r_var = float(stats["r_var"])
        r_weighted_mean = float(stats.get("r_weighted_mean", r_mean))
        r_weighted_var = float(stats.get("r_weighted_var", r_var))
        r_probabilities = dict(stats.get("r_probability_by_count", {}))
        r_particle_counts = dict(stats.get("r_particle_count_by_count", {}))
        posterior_pos, posterior_str = stats["posterior"]
        top_entries = stats["top_k"]
        variable_cardinality = bool(
            getattr(getattr(filt, "config", None), "variable_cardinality", False)
        )
        max_sources = getattr(getattr(filt, "config", None), "max_sources", None)
        p_birth = float(
            getattr(
                getattr(filt, "config", None),
                "structural_rj_birth_probability",
                0.0,
            )
        )
        p_death = float(
            getattr(
                getattr(filt, "config", None),
                "structural_rj_death_probability",
                0.0,
            )
        )
        structural_kernel = "exact_rj" if variable_cardinality else "fixed_k"
        print(
            f"[step {step_index}] pf[{iso}] ess_pre={ess_pre:.2f} resampled={resampled} "
            f"ess_post={_fmt_optional_float(ess_post)} "
            f"current_ess={current_ess:.2f} "
            f"current_ess_ratio={current_ess_ratio:.3f} "
            f"particles={particle_count} "
            f"temper_min_ess={_fmt_optional_float(temper_min_ess)} "
            f"unique_ancestors={unique_ancestor_count} "
            f"resamples={resamples} births={births} deaths={deaths} "
            f"r_mean={r_mean:.2f} r_var={r_var:.2f} "
            f"r_weighted_mean={r_weighted_mean:.2f} "
            f"r_weighted_var={r_weighted_var:.2f} "
            f"r_posterior={_fmt_probability_map(r_probabilities)} "
            f"r_particles={r_particle_counts} "
            f"weighted_moves={_safe_json_dumps(transition_weight_mass)} "
            f"variable_cardinality={variable_cardinality} max_sources={max_sources} "
            f"structural_kernel={structural_kernel} "
            f"p_birth={p_birth:.3f} p_death={p_death:.3f}"
        )
        if structural_timing:
            timing_items = {
                key: float(value)
                for key, value in dict(structural_timing).items()
                if (
                    (str(key).startswith("rj_") or key == "total")
                    and (float(value) > 0.0 or key == "total")
                )
            }
            if timing_items:
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
            f"[step {step_index}] pf[{iso}] "
            f"posterior={_fmt_sources(posterior_pos, posterior_str)}"
        )
        if top_entries:
            print(f"[step {step_index}] pf[{iso}] top_k={_fmt_top_k(top_entries)}")




def _log_surface_atlas_observability_diagnostics(
    estimator: PurePFEstimator,
    step_index: int,
    *,
    label: str,
    window: int | None = None,
    max_candidates: int = 256,
) -> None:
    """Log truth-independent surface-candidate observability diagnostics."""
    if int(max_candidates) <= 0:
        return
    if not hasattr(estimator, "surface_atlas_observability_diagnostics"):
        return
    diagnostics = estimator.surface_atlas_observability_diagnostics(
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
            f"coverage={float(entry.get('coverage_gain', 0.0)):.6g} "
            f"revisit={float(entry.get('revisit_penalty', 0.0)):.6g} "
            f"bearing={float(entry.get('bearing_diversity_gain', 0.0)):.6g} "
            f"frontier={float(entry.get('frontier_gain', 0.0)):.6g} "
            f"turn={float(entry.get('turn_penalty', 0.0)):.6g} "
            f"local_orbit={float(entry.get('local_orbit_gain', 0.0)):.6g} "
            f"elev_cond={float(entry.get('elevation_condition_gain', 0.0)):.6g} "
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


def _sanitize_json_payload(
    payload: object,
    *,
    _path: str = "$",
    _unsafe_integers_as_decimal_strings: bool = False,
) -> object:
    """Return recursively plain, strict-JSON-compatible data.

    NumPy containers and scalars are converted to their Python equivalents,
    mappings receive string keys, and non-finite floating-point values fail with
    their exact payload path. Unsupported objects raise instead of being
    silently stringified.
    """
    if payload is None or isinstance(payload, (str, bool)):
        return payload
    if isinstance(payload, np.bool_):
        return bool(payload)
    if isinstance(payload, (int, np.integer)):
        value = int(payload)
        if _unsafe_integers_as_decimal_strings and abs(value) > 2**53:
            return str(value)
        return value
    if isinstance(payload, (float, np.floating)):
        value = float(payload)
        if not np.isfinite(value):
            raise ValueError(
                f"Strict JSON payload contains a non-finite number at {_path}."
            )
        return value
    if isinstance(payload, Path):
        return payload.as_posix()
    if isinstance(payload, np.ndarray):
        return _sanitize_json_payload(
            payload.tolist(),
            _path=_path,
            _unsafe_integers_as_decimal_strings=(
                _unsafe_integers_as_decimal_strings
            ),
        )
    if isinstance(payload, np.generic):
        return _sanitize_json_payload(
            payload.item(),
            _path=_path,
            _unsafe_integers_as_decimal_strings=(
                _unsafe_integers_as_decimal_strings
            ),
        )
    if isinstance(payload, Mapping):
        result: dict[str, object] = {}
        for key, value in payload.items():
            resolved_key = str(key)
            if resolved_key in result:
                raise ValueError(
                    "Strict JSON payload contains colliding stringified keys at "
                    f"{_path}: {resolved_key!r}."
                )
            result[resolved_key] = _sanitize_json_payload(
                value,
                _path=f"{_path}[{resolved_key!r}]",
                _unsafe_integers_as_decimal_strings=(
                    _unsafe_integers_as_decimal_strings
                ),
            )
        return result
    if isinstance(payload, (list, tuple)):
        return [
            _sanitize_json_payload(
                value,
                _path=f"{_path}[{index}]",
                _unsafe_integers_as_decimal_strings=(
                    _unsafe_integers_as_decimal_strings
                ),
            )
            for index, value in enumerate(payload)
        ]
    if isinstance(payload, (set, frozenset)):
        return [
            _sanitize_json_payload(
                value,
                _path=f"{_path}[{index}]",
                _unsafe_integers_as_decimal_strings=(
                    _unsafe_integers_as_decimal_strings
                ),
            )
            for index, value in enumerate(
                sorted(payload, key=lambda value: repr(value))
            )
        ]
    raise TypeError(
        f"Unsupported value in JSON payload at {_path}: "
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


def _atomic_write_json(
    target: Path,
    payload: Mapping[str, Any],
) -> None:
    """Atomically replace one JSON artifact after strict serialization."""
    serialized = (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.tmp-",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _render_optional_outputs_after_artifacts(
    *,
    required_artifacts: Sequence[Path],
    renderers: Sequence[tuple[str, Callable[[], None]]],
) -> tuple[dict[str, str], ...]:
    """Render optional plots only after all scientific artifacts exist.

    Plot failures are isolated from the already-published MeasurementLog,
    canonical posterior, and evaluation summary. Invalid scientific payloads
    still fail before this function is entered.
    """
    missing = [
        Path(path)
        for path in required_artifacts
        if not Path(path).exists()
    ]
    if missing:
        raise RuntimeError(
            "Optional rendering requires published scientific artifacts: "
            + ", ".join(path.as_posix() for path in missing)
        )
    failures: list[dict[str, str]] = []
    for label, renderer in renderers:
        try:
            renderer()
        except Exception as exc:
            failure = {
                "label": str(label),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            print(
                "Optional final rendering failed after scientific artifacts "
                f"were published: {_safe_json_dumps(failure)}",
                flush=True,
            )
    return tuple(failures)


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


def _preflight_pure_pf_compute_backend(
    *,
    use_gpu: bool,
    gpu_device: str,
    gpu_dtype: str,
) -> str:
    """Validate the selected PF compute backend before external transport."""
    dtype_name = str(gpu_dtype).strip().lower()
    if dtype_name != "float64":
        raise ValueError(
            "Production pure-PF runtime requires gpu_dtype='float64'; "
            "lower-precision posterior dynamics are forbidden."
        )
    if not bool(use_gpu):
        return "batched_numpy_float64"
    from pf import gpu_utils

    gpu_utils.require_torch_compute_device(
        str(gpu_device),
        dtype_name,
    )
    return "batched_torch_float64"


def _resolve_python_worker_count(worker_count: object | None) -> int:
    """Return a Python CPU worker count, using all logical CPUs for auto."""
    if worker_count is None:
        return max(1, os.cpu_count() or 1)
    workers = _strict_json_integer(
        worker_count,
        name="python_worker_count",
        minimum=0,
    )
    if workers == 0:
        cpu_count = os.cpu_count() or 1
        return max(1, cpu_count)
    return workers


def _coerce_live_visualization(live: bool) -> bool:
    """Return whether live Matplotlib updates can run in this process."""
    requested_live = _strict_json_bool(live, name="live")
    if not requested_live:
        return False
    backend = str(matplotlib.get_backend()).lower()
    if "agg" in backend or not _has_display():
        print("GUI display unavailable; running in CUI/headless mode.")
        return False
    return True


def _resolve_plot_save_interval(
    runtime_config: dict[str, object],
    key: str,
    *,
    default: int = 1,
    allow_disable: bool = False,
) -> int:
    """Return a strictly typed plot-save interval from runtime config."""
    interval = _strict_json_integer(
        runtime_config.get(key, default),
        name=key,
        minimum=0 if allow_disable else 1,
    )
    if allow_disable and interval == 0:
        return 0
    return interval


@dataclass(frozen=True)
class DetectorHeightPlanningConfig:
    """Describe the continuous detector mast workspace used by pose planning."""

    ground_z_m: float
    initial_mast_height_m: float
    minimum_mast_height_m: float
    maximum_mast_height_m: float

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
    def candidate_world_heights_m(self) -> tuple[float, ...] | None:
        """Return None because production samples height continuously."""
        return None

    @property
    def candidate_world_z_bounds_m(self) -> tuple[float, float]:
        """Return the continuous z interval sampled by the pose generator."""
        return self.minimum_world_z_m, self.maximum_world_z_m


def _resolve_detector_height_planning_config(
    runtime_config: Mapping[str, object],
    *,
    room_height_m: float,
) -> DetectorHeightPlanningConfig:
    """Resolve the sole continuous detector-height planning contract."""
    room_height = _strict_json_number(
        room_height_m,
        name="room_height_m",
        minimum=0.0,
        minimum_exclusive=True,
    )
    ground_z = _strict_json_number(
        runtime_config.get("robot_ground_z_m", 0.0),
        name="robot_ground_z_m",
        minimum=0.0,
        maximum=room_height,
    )
    available_height = room_height - ground_z
    initial_mast_height = _strict_json_number(
        runtime_config.get("detector_height_m", 0.5),
        name="detector_height_m",
        minimum=0.0,
        maximum=available_height,
    )
    minimum_mast_height = _strict_json_number(
        runtime_config.get("detector_height_min_m", 0.0),
        name="detector_height_min_m",
        minimum=0.0,
        maximum=available_height,
    )
    maximum_mast_height = _strict_json_number(
        runtime_config.get("detector_height_max_m", available_height),
        name="detector_height_max_m",
        minimum=0.0,
        maximum=available_height,
    )
    if maximum_mast_height < minimum_mast_height:
        raise ValueError("detector_height_max_m must be >= detector_height_min_m.")
    if not minimum_mast_height <= initial_mast_height <= maximum_mast_height:
        raise ValueError(
            "detector_height_m must lie within the detector height bounds."
        )
    retired_keys = {
        key
        for key in ("detector_height_actions_m", "detector_heights_m")
        if key in runtime_config
    }
    if retired_keys:
        raise ValueError(
            "Discrete detector-height actions were removed; omit "
            f"{sorted(retired_keys)} and use continuous bounds."
        )
    if runtime_config.get("detector_height_sampling_mode") != "continuous":
        raise ValueError(
            "Production pure-PF planning requires "
            "detector_height_sampling_mode='continuous'."
        )
    return DetectorHeightPlanningConfig(
        ground_z_m=ground_z,
        initial_mast_height_m=initial_mast_height,
        minimum_mast_height_m=minimum_mast_height,
        maximum_mast_height_m=maximum_mast_height,
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
    if not _runtime_bool(
        runtime_config,
        "measurement_pose_clearance_enabled",
        True,
    ):
        return requested_radius
    physical_radius = _strict_json_number(
        runtime_config.get(
            "robot_base_physical_radius_m",
            _DEFAULT_ROBOT_BASE_RADIUS_M,
        ),
        name="robot_base_physical_radius_m",
        minimum=0.0,
        minimum_exclusive=True,
    )
    clearance_margin = _strict_json_number(
        runtime_config.get("measurement_pose_clearance_margin_m", 0.02),
        name="measurement_pose_clearance_margin_m",
        minimum=0.0,
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
    height = _strict_json_number(
        obstacle_height_m,
        name="obstacle_height_m",
        minimum=0.0,
    )
    return tuple(
        obstacle_grid.blocked_boxes(
            z_min=float(ground_z_m),
            z_max=float(ground_z_m) + height,
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
    enabled = _runtime_bool(
        runtime_config,
        "measurement_pose_clearance_enabled",
        True,
    )
    if not enabled:
        return base_map, {
            "enabled": False,
            "effective_robot_radius_m": float(effective_robot_radius_m),
        }
    environment_size = np.asarray(environment_size_xyz, dtype=float).reshape(-1)
    if environment_size.shape != (3,) or np.any(~np.isfinite(environment_size)):
        raise ValueError("environment_size_xyz must be a finite three-vector.")
    margin = _strict_json_number(
        runtime_config.get("measurement_pose_clearance_margin_m", 0.02),
        name="measurement_pose_clearance_margin_m",
        minimum=0.0,
    )
    base_height = _strict_json_number(
        runtime_config.get(
            "robot_base_physical_height_m",
            _DEFAULT_ROBOT_BASE_HEIGHT_M,
        ),
        name="robot_base_physical_height_m",
        minimum=0.0,
        minimum_exclusive=True,
    )
    mast_radius = _strict_json_number(
        runtime_config.get(
            "detector_mast_physical_radius_m",
            _DEFAULT_ROBOT_MAST_RADIUS_M,
        ),
        name="detector_mast_physical_radius_m",
        minimum=0.0,
    )
    shield_outer_radius_m = 0.01 * max(
        float(getattr(shield_params, "inner_radius_fe_cm"))
        + float(getattr(shield_params, "thickness_fe_cm")),
        float(getattr(shield_params, "inner_radius_pb_cm"))
        + float(getattr(shield_params, "thickness_pb_cm")),
    )
    if not np.isfinite(shield_outer_radius_m) or shield_outer_radius_m <= 0.0:
        raise ValueError("Shield outer radius must be finite and positive.")
    transport_mast_height = _strict_json_number(
        runtime_config.get(
            "detector_transport_height_m",
            detector_height_config.initial_mast_height_m,
        ),
        name="detector_transport_height_m",
        minimum=detector_height_config.minimum_mast_height_m,
        maximum=detector_height_config.maximum_mast_height_m,
    )
    obstacle_height = _strict_json_number(
        runtime_config.get("obstacle_height_m", 2.0),
        name="obstacle_height_m",
        minimum=0.0,
    )
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
        motion_worker_count=_strict_json_integer(
            runtime_config.get("measurement_route_workers", 0),
            name="measurement_route_workers",
            minimum=0,
        ),
        motion_grid_cell_size_m=_strict_json_number(
            runtime_config.get("measurement_route_grid_cell_size_m", 0.25),
            name="measurement_route_grid_cell_size_m",
            minimum=0.0,
            minimum_exclusive=True,
        ),
    )
    diagnostics: dict[str, object] = {
        "enabled": True,
        "continuous_measurement_volume": True,
        "height_sampling_mode": "continuous",
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






def _full_spectrum_model_diagnostics(
    full_spectrum_model: GeometryConditionedSpectralModel,
    *,
    obstacle_attenuation_active: bool,
) -> dict[str, object]:
    """Describe the sole joint full-spectrum observation distribution."""
    return {
        "observation_likelihood": {
            "family": (
                "nonparalyzable_renewal_total_conditional_full_spectrum_marks"
            ),
            "contract_hash_sha256": str(
                full_spectrum_model.contract_hash_sha256
            ),
            "raw_integer_spectrum": True,
            "background_owned_once_by_generative_model": True,
            "projected_isotope_counts": False,
            "contrast_term": False,
            "view_ratio_term": False,
            "dead_time_tau_s": float(full_spectrum_model.dead_time_tau_s),
            "background_rate_cps": float(
                full_spectrum_model.background_rate_cps
            ),
            "obstacle_attenuation_active": bool(obstacle_attenuation_active),
        },
    }




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


def _generate_planning_candidates(
    *,
    current_pose_xyz: NDArray[np.float64],
    map_api: object | None,
    n_candidates: int,
    min_dist_from_visited: float,
    visited_poses_xyz: NDArray[np.float64] | None,
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]],
    detector_heights_m: Sequence[float] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], dict[str, object]]:
    """Generate one global reachable 3-D Sobol action pool.

    Candidate generation enforces only physical bounds, free space, motion
    reachability, and the declared three-dimensional separation from already
    visited poses. It does not impose horizontal-count or extent gates and does
    not relax physical separation to manufacture a preferred XY distribution.
    Surface-atlas coverage and the shared full-spectrum EIG rank the resulting
    globally sampled action set.
    """
    if rng is None:
        raise ValueError("Planning candidate generation requires an explicit RNG.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")
    bounds_lo = np.asarray(bounds_xyz[0], dtype=np.float64)
    bounds_hi = np.asarray(bounds_xyz[1], dtype=np.float64)
    if (
        bounds_lo.shape != (3,)
        or bounds_hi.shape != (3,)
        or np.any(~np.isfinite(bounds_lo))
        or np.any(~np.isfinite(bounds_hi))
        or np.any(bounds_hi < bounds_lo)
    ):
        raise ValueError("bounds_xyz must contain finite ordered 3-D bounds.")
    candidates = generate_candidate_poses(
        current_pose_xyz=current_pose_xyz,
        map_api=map_api,
        n_candidates=n_candidates,
        strategy="free_space_sobol",
        min_dist_from_visited=min_dist_from_visited,
        visited_poses_xyz=visited_poses_xyz,
        bounds_xyz=(bounds_lo, bounds_hi),
        detector_heights_m=detector_heights_m,
        require_motion_reachable=True,
        rng=rng,
    )
    candidates = np.asarray(candidates, dtype=np.float64)
    if (
        candidates.ndim != 2
        or candidates.shape[1] != 3
        or np.any(~np.isfinite(candidates))
    ):
        raise RuntimeError(
            "Global candidate generation returned an invalid 3-D action pool."
        )
    if candidates.shape[0] == 0:
        raise RuntimeError(
            "No globally sampled candidate satisfies bounds, free-space, "
            "reachability, and physical separation."
        )
    return candidates, {
        "contract": "global_reachable_3d_sobol_pool_v1",
        "candidate_count": int(candidates.shape[0]),
        "requested_candidate_count": int(n_candidates),
        "minimum_3d_separation_m": float(min_dist_from_visited),
        "physical_separation_relaxed": False,
        "horizontal_quality_gate": False,
        "bounds_lo_xyz_m": [float(value) for value in bounds_lo],
        "bounds_hi_xyz_m": [float(value) for value in bounds_hi],
        "detector_heights_m": (
            None
            if detector_heights_m is None
            else [float(value) for value in detector_heights_m]
        ),
    }


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
    # Schema v2 stores no projected isotope counts.  Preserve only the raw
    # spectrum total for display/resume status, never for PF inference.
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
        total_motion_distance_m=motion_time
        * max(float(nominal_motion_speed_m_s), 0.0),
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



def _adaptive_mission_stop_reason(
    estimator: PurePFEstimator,
    *,
    visited_poses_xyz: Sequence[NDArray[np.float64]],
    min_poses: int,
    require_pf_cardinality_ready: bool = True,
) -> str | None:
    """Stop only when the intrinsic surface posterior contract is converged."""
    if isinstance(min_poses, (bool, np.bool_)) or not isinstance(
        min_poses,
        (int, np.integer),
    ):
        raise ValueError("min_poses must be a positive integer.")
    resolved_minimum_poses = int(min_poses)
    if resolved_minimum_poses <= 0:
        raise ValueError("min_poses must be a positive integer.")
    if type(require_pf_cardinality_ready) is not bool:
        raise ValueError(
            "require_pf_cardinality_ready must be a JSON boolean."
        )
    if len(visited_poses_xyz) < resolved_minimum_poses:
        return None
    cardinality_ready, _cardinality_reason = _source_cardinality_dwell_status(
        estimator,
        refresh_estimates=False,
    )
    if require_pf_cardinality_ready and not cardinality_ready:
        return None
    posterior_convergence = estimator.posterior_convergence_diagnostics()
    if not bool(posterior_convergence.get("ready", False)):
        return None
    return "intrinsic_surface_posterior_converged"


def _validated_cardinality_distribution(
    value: object,
    *,
    name: str,
) -> dict[int, float]:
    """Return one strict integer-cardinality probability distribution."""
    if not isinstance(value, Mapping) or not value:
        raise RuntimeError(f"{name} must be a nonempty mapping.")
    items: list[tuple[int, object]] = []
    for raw_key, raw_probability in value.items():
        if isinstance(raw_key, bool) or not isinstance(
            raw_key,
            (int, np.integer),
        ):
            raise RuntimeError(f"{name} keys must be nonnegative integers.")
        cardinality = int(raw_key)
        if cardinality < 0:
            raise RuntimeError(f"{name} keys must be nonnegative integers.")
        items.append((cardinality, raw_probability))
    items.sort(key=lambda item: item[0])
    if len({item[0] for item in items}) != len(items):
        raise RuntimeError(f"{name} contains duplicate cardinality keys.")
    try:
        probabilities = validated_probability_distribution(
            [item[1] for item in items],
            name=name,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    return {
        cardinality: float(probability)
        for (cardinality, _), probability in zip(
            items,
            probabilities,
            strict=True,
        )
    }


def _validated_joint_cardinality_distribution(
    value: object,
    *,
    isotope_count: int,
) -> dict[tuple[int, ...], float]:
    """Return strict posterior mass over aligned joint cardinality tuples."""
    if not isinstance(value, Mapping) or not value:
        raise RuntimeError(
            "Joint PF cardinality posterior must be a nonempty mapping."
        )
    items: list[tuple[tuple[int, ...], object]] = []
    for raw_tuple, raw_probability in value.items():
        if not isinstance(raw_tuple, tuple) or len(raw_tuple) != isotope_count:
            raise RuntimeError(
                "Joint PF cardinality keys must match joint_isotope_order."
            )
        if any(
            isinstance(raw_value, bool)
            or not isinstance(raw_value, (int, np.integer))
            or int(raw_value) < 0
            for raw_value in raw_tuple
        ):
            raise RuntimeError(
                "Joint PF cardinality tuples must contain nonnegative integers."
            )
        items.append(
            (
                tuple(int(raw_value) for raw_value in raw_tuple),
                raw_probability,
            )
        )
    items.sort(key=lambda item: item[0])
    if len({item[0] for item in items}) != len(items):
        raise RuntimeError("Joint PF cardinality posterior contains duplicate tuples.")
    try:
        probabilities = validated_probability_distribution(
            [item[1] for item in items],
            name="joint PF cardinality posterior",
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    return {
        cardinality_tuple: float(probability)
        for (cardinality_tuple, _), probability in zip(
            items,
            probabilities,
            strict=True,
        )
    }




def _source_cardinality_dwell_status(
    estimator: PurePFEstimator,
    *,
    refresh_estimates: bool = True,
) -> tuple[bool, str]:
    """Return whether aligned joint PF evidence supports ending adaptive dwell."""
    if bool(refresh_estimates):
        estimator.estimates()
    filters = getattr(estimator, "filters", {})
    if not isinstance(filters, dict) or not filters:
        return False, "no_pf_posterior"
    pf_config = getattr(estimator, "pf_config", None)
    status = _final_pf_cardinality_status(estimator)
    joint_status = status["joint_cardinality"]
    joint_probability = float(joint_status["map_probability"])
    minimum_joint_probability = float(
        getattr(pf_config, "converge_cardinality_min_probability", 0.95)
    )
    if joint_probability + 1.0e-12 < minimum_joint_probability:
        return False, "pf_joint_cardinality_probability"
    variance_limit = max(
        float(getattr(pf_config, "converge_cardinality_var_max", 0.05)),
        0.0,
    )
    pending = [
        str(isotope)
        for isotope, isotope_status in sorted(
            status["pf_cardinality"].items()
        )
        if float(isotope_status["variance"]) > variance_limit + 1.0e-9
    ]
    if pending:
        return False, f"pf_cardinality_variance:{','.join(pending)}"
    isotope_order = tuple(str(value) for value in joint_status["isotope_order"])
    first_filter = filters.get(isotope_order[0])
    if first_filter is None:
        raise RuntimeError(
            "Joint cardinality isotope order does not match initialized filters."
        )
    try:
        weights = validated_probability_distribution(
            np.asarray(
                getattr(first_filter, "continuous_weights", ()),
                dtype=float,
            ),
            name="joint PF stopping weights",
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    current_ess = float(1.0 / np.sum(np.square(weights)))
    current_ess_ratio = current_ess / float(weights.size)
    minimum_ess_ratio = float(
        getattr(pf_config, "converge_min_ess_ratio", 0.4)
    )
    if current_ess_ratio + 1.0e-12 < minimum_ess_ratio:
        return False, "pf_current_ess"
    return True, "pf_cardinality_ready"




def _final_pf_cardinality_status(estimator: object) -> dict[str, Any]:
    """Return strict marginal and aligned-joint PF cardinality posteriors."""
    getter = getattr(estimator, "posterior_cardinality_distribution", None)
    if not callable(getter):
        raise RuntimeError(
            "Final PF reporting requires marginal cardinality posteriors."
        )
    distributions_raw = getter()
    if not isinstance(distributions_raw, Mapping) or not distributions_raw:
        raise RuntimeError(
            "Final PF cardinality posterior must contain every isotope."
        )
    cardinality: dict[str, dict[str, Any]] = {}
    validated_marginals: dict[str, dict[int, float]] = {}
    for isotope, distribution_raw in sorted(distributions_raw.items()):
        distribution = _validated_cardinality_distribution(
            distribution_raw,
            name=f"PF cardinality posterior[{isotope}]",
        )
        validated_marginals[str(isotope)] = distribution
        counts = np.asarray(list(distribution), dtype=float)
        probabilities = np.asarray(list(distribution.values()), dtype=float)
        mean = float(np.sum(counts * probabilities))
        variance = float(np.sum(probabilities * (counts - mean) ** 2))
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
    order_getter = getattr(estimator, "joint_isotope_order", None)
    joint_getter = getattr(
        estimator,
        "posterior_joint_cardinality_distribution",
        None,
    )
    if not callable(order_getter) or not callable(joint_getter):
        raise RuntimeError(
            "Final PF reporting requires aligned joint cardinality mass."
        )
    isotope_order = tuple(str(value) for value in order_getter())
    if (
        not isotope_order
        or len(set(isotope_order)) != len(isotope_order)
        or set(isotope_order) != set(validated_marginals)
    ):
        raise RuntimeError(
            "Joint PF isotope order does not match marginal posteriors."
        )
    joint_distribution = _validated_joint_cardinality_distribution(
        joint_getter(),
        isotope_count=len(isotope_order),
    )
    for isotope_index, isotope in enumerate(isotope_order):
        joint_marginal: dict[int, float] = {}
        for cardinality_tuple, probability in joint_distribution.items():
            cardinality_value = int(cardinality_tuple[isotope_index])
            joint_marginal[cardinality_value] = (
                joint_marginal.get(cardinality_value, 0.0)
                + float(probability)
            )
        marginal = validated_marginals[isotope]
        support = sorted(set(joint_marginal) | set(marginal))
        if any(
            not np.isclose(
                joint_marginal.get(value, 0.0),
                marginal.get(value, 0.0),
                rtol=0.0,
                atol=1.0e-12,
            )
            for value in support
        ):
            raise RuntimeError(
                "Joint PF cardinality mass does not reproduce isotope "
                f"marginal {isotope!r}."
            )
    maximum_mass = max(joint_distribution.values())
    map_tuple = min(
        cardinality_tuple
        for cardinality_tuple, probability in joint_distribution.items()
        if np.isclose(
            probability,
            maximum_mass,
            rtol=0.0,
            atol=1.0e-15,
        )
    )
    positive_joint = np.asarray(
        [
            probability
            for probability in joint_distribution.values()
            if probability > 0.0
        ],
        dtype=float,
    )
    joint_payload = {
        "isotope_order": list(isotope_order),
        "distribution": [
            {
                "cardinality_tuple": [
                    int(value) for value in cardinality_tuple
                ],
                "probability": float(probability),
            }
            for cardinality_tuple, probability in joint_distribution.items()
        ],
        "map_cardinality_tuple": [int(value) for value in map_tuple],
        "map_probability": float(maximum_mass),
        "entropy_nats": float(
            -np.sum(positive_joint * np.log(positive_joint))
        ),
    }
    return {
        "source": "pf_posterior",
        "pf_cardinality": cardinality,
        "joint_cardinality": joint_payload,
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
    energy_axis_keV: NDArray[np.float64],
    spectrum: np.ndarray,
    output_path: Path,
    highlight_isotopes: set[str] | None = None,
    use_detection_lines: bool = True,
    title: str = "Final measurement spectrum",
) -> None:
    """Save the raw measurement spectrum with fixed nuclide line markers."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    energy_axis = np.asarray(energy_axis_keV, dtype=np.float64)
    library = default_library()
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
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        energy_axis,
        np.asarray(spectrum, dtype=float),
        color="black",
        linewidth=1.0,
        label="Observed spectrum",
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
) -> None:
    """Require one predeclared finite physical live time for every action."""
    if not np.isfinite(measurement_time_s) or measurement_time_s <= 0.0:
        raise ValueError("measurement_time_s must be finite and positive.")


def _validate_observation_command_binding(
    observation: SimulationObservation,
    command: SimulationCommand,
    *,
    require_native_contract: bool,
) -> None:
    """Require one simulator response to match the exact submitted action."""
    # This acquisition path commands zero base yaw and an identity detector
    # local orientation, so the detector world quaternion is also identity.
    top_level_fields = {
        "step_id": (command.step_id, observation.step_id),
        "detector_pose_xyz": (
            command.target_pose_xyz,
            observation.detector_pose_xyz,
        ),
        "detector_quat_wxyz": (
            (1.0, 0.0, 0.0, 0.0),
            observation.detector_quat_wxyz,
        ),
        "fe_orientation_index": (
            command.fe_orientation_index,
            observation.fe_orientation_index,
        ),
        "pb_orientation_index": (
            command.pb_orientation_index,
            observation.pb_orientation_index,
        ),
    }
    top_level_mismatches = {
        key: (expected, actual)
        for key, (expected, actual) in top_level_fields.items()
        if actual != expected
    }
    if top_level_mismatches:
        raise RuntimeError(
            "Simulator response does not match the submitted action: "
            f"{top_level_mismatches}."
        )
    if not _strict_json_bool(
        require_native_contract,
        name="require_native_contract",
    ):
        return

    expected_pair_id = (
        int(command.fe_orientation_index) * 8
        + int(command.pb_orientation_index)
    )
    metadata = observation.metadata
    native_integer_fields = {
        "fe_orientation_index": command.fe_orientation_index,
        "pb_orientation_index": command.pb_orientation_index,
        "shield_num_orientations": 8,
        "shield_pair_id": expected_pair_id,
    }
    native_mismatches: dict[str, tuple[object, object]] = {}
    for key, expected in native_integer_fields.items():
        actual = _strict_json_integer(
            metadata.get(key),
            name=f"native metadata.{key}",
        )
        if actual != expected:
            native_mismatches[key] = (expected, actual)
    actual_dwell_time_s = _strict_json_number(
        metadata.get("dwell_time_s"),
        name="native metadata.dwell_time_s",
        minimum=0.0,
        minimum_exclusive=True,
    )
    if actual_dwell_time_s != command.dwell_time_s:
        native_mismatches["dwell_time_s"] = (
            command.dwell_time_s,
            actual_dwell_time_s,
        )
    if native_mismatches:
        raise RuntimeError(
            "Native observation metadata does not match the submitted action: "
            f"{native_mismatches}."
        )


def _analysis_spectrum_array(
    observation: SimulationObservation,
    model: GeometryConditionedSpectralModel,
    *,
    require_native_contract: bool,
) -> NDArray[np.int64]:
    """Return a fail-closed native raw spectrum for the joint PF likelihood."""
    native_contract_required = _strict_json_bool(
        require_native_contract,
        name="require_native_contract",
    )
    raw = np.asarray(observation.spectrum_counts)
    values = np.asarray(raw, dtype=np.float64)
    expected_axis = np.asarray(model.energy_axis_keV, dtype=np.float64)
    if values.ndim != 1 or values.shape != expected_axis.shape:
        raise ValueError(
            "Simulator spectrum shape disagrees with the approved full-spectrum "
            f"contract: {values.shape} != {expected_axis.shape}."
        )
    if (
        np.any(~np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values != np.rint(values))
        or np.any(values > np.iinfo(np.int64).max)
    ):
        raise ValueError(
            "Production observations must be exact nonnegative unit-weight "
            "integer event counts."
        )
    bin_width = float(expected_axis[1] - expected_axis[0])
    expected_edges = np.concatenate(
        (expected_axis, [float(expected_axis[-1] + bin_width)])
    )
    actual_edges = np.asarray(
        observation.energy_bin_edges_keV,
        dtype=np.float64,
    )
    if not np.array_equal(actual_edges, expected_edges):
        raise ValueError(
            "Simulator energy-bin edges disagree with the approved "
            "full-spectrum contract."
        )
    metadata = observation.metadata
    if not native_contract_required:
        metadata[FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY] = (
            model.contract_hash_sha256
        )
        return np.ascontiguousarray(np.rint(values), dtype=np.int64)
    expected_metadata = {
        "detector_scoring_mode": "incident_gamma_energy",
        "detector_response_sampling_mode": (
            "multinomial_marking_with_nonparalyzable_event_time"
        ),
        "detector_response_sampling_model": (
            "native_incident_gamma_response_v1"
        ),
        "detector_response_sampling_contract_sha256": (
            NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256
        ),
        "intensity_cps_1m_definition": (
            "pre_dead_time_detector_pulse_rate_at_1m"
        ),
        "transport_history_mode": "full_unit_weight",
    }
    mismatches = {
        key: (expected, metadata.get(key))
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "Native observation metadata violates the approved raw-spectrum "
            f"contract: {mismatches}."
        )
    dead_time_tau_s = _strict_json_number(
        metadata.get("dead_time_tau_s"),
        name="native metadata.dead_time_tau_s",
        minimum=0.0,
    )
    background_rate_cps = _strict_json_number(
        metadata.get("background_cps"),
        name="native metadata.background_cps",
        minimum=0.0,
    )
    if (
        not np.isclose(
            dead_time_tau_s,
            float(model.dead_time_tau_s),
            rtol=0.0,
            atol=1.0e-15,
        )
        or not np.isclose(
            background_rate_cps,
            float(model.background_rate_cps),
            rtol=0.0,
            atol=1.0e-15,
        )
    ):
        raise ValueError(
            "Native dead-time/background settings disagree with the approved "
            "full-spectrum model."
        )
    metadata[FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY] = (
        model.contract_hash_sha256
    )
    return np.ascontiguousarray(np.rint(values), dtype=np.int64)


def _acquire_spectrum_observation(
    *,
    simulation_runtime: SimulationRuntime,
    full_spectrum_model: GeometryConditionedSpectralModel,
    step_id: int,
    pose_xyz: NDArray[np.float64],
    fe_idx: int,
    pb_idx: int,
    live_time_s: float,
    travel_time_s: float,
    shield_actuation_time_s: float,
    require_native_contract: bool,
    travel_waypoints_xyz: Sequence[Sequence[float]] | None = None,
) -> tuple[SimulationObservation, float, NDArray[np.int64], str, int]:
    """Acquire one fixed-dwell native raw spectrum for the joint PF."""
    native_contract_required = _strict_json_bool(
        require_native_contract,
        name="require_native_contract",
    )
    resolved_step_id = _strict_json_integer(
        step_id,
        name="step_id",
        minimum=0,
    )
    resolved_fe_idx = _strict_json_integer(
        fe_idx,
        name="fe_idx",
        minimum=0,
        maximum=7,
    )
    resolved_pb_idx = _strict_json_integer(
        pb_idx,
        name="pb_idx",
        minimum=0,
        maximum=7,
    )
    target_pose = tuple(pose_xyz)
    command_waypoints = (
        None
        if travel_waypoints_xyz is None
        else tuple(
            tuple(value for value in waypoint)
            for waypoint in travel_waypoints_xyz
        )
    )
    command = SimulationCommand(
        step_id=resolved_step_id,
        target_pose_xyz=target_pose,
        target_base_yaw_rad=0.0,
        fe_orientation_index=resolved_fe_idx,
        pb_orientation_index=resolved_pb_idx,
        dwell_time_s=live_time_s,
        travel_time_s=travel_time_s,
        shield_actuation_time_s=shield_actuation_time_s,
        travel_waypoints_xyz=command_waypoints,
    )
    observation = simulation_runtime.step(command)
    _validate_observation_command_binding(
        observation,
        command,
        require_native_contract=native_contract_required,
    )
    spectrum = _analysis_spectrum_array(
        observation,
        full_spectrum_model,
        require_native_contract=native_contract_required,
    )
    return (
        observation,
        command.dwell_time_s,
        spectrum,
        "fixed_dwell",
        1,
    )


def run_live_pf(
    live: bool = True,
    max_steps: int | None = None,
    max_poses: int | None = 8,
    sources: list[PointSource] | None = None,
    environment_mode: str = DEFAULT_ENVIRONMENT_MODE,
    obstacle_layout_path: str | None = DEFAULT_OBSTACLE_CONFIG.as_posix(),
    obstacle_seed: int | None = None,
    eval_match_radius_m: float = 0.5,
    variable_cardinality: bool | None = None,
    num_particles: int = 2000,
    pf_config_overrides: dict[str, object] | None = None,
    save_outputs: bool = True,
    output_tag: str | None = None,
    measurement_log_output: str | None = None,
    resume_measurement_stage: str | None = None,
    resume_compatible_code_paths: Sequence[str] | None = None,
    resume_compatibility_basis: str | None = None,
    pose_candidates: int = 64,
    pose_min_dist: float = 3.0,
    return_state: bool = False,
    sim_backend: str | None = None,
    sim_config_path: str | None = None,
    pf_config_path: str | None = DEFAULT_PF_CONFIG.as_posix(),
    blender_executable: str | None = None,
    blender_output_path: str | None = None,
    blender_timeout_s: float = 120.0,
    passage_width_m: float = 1.0,
    robot_radius_m: float = 0.35,
    nominal_motion_speed_m_s: float = DEFAULT_ROBOT_SPEED_M_S,
    rotation_overhead_s: float = DEFAULT_ROTATION_OVERHEAD_S,
    measurement_time_s: float = DEFAULT_MEASUREMENT_TIME_S,
    path_planner: str | None = None,
    dss_program_length: int | None = None,
    dss_rotation_weight: float | None = None,
    source_generation_mode: str = "demo",
    source_config_provenance: Mapping[str, object] | None = None,
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
) -> PurePFEstimator | None:
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
        resume_compatible_code_paths: Runtime paths whose commit delta passed an
            external state-equivalence gate.
        resume_compatibility_basis: Description of the equivalence evidence for
            explicitly compatible runtime paths.
        pose_candidates: Number of pose candidates to generate per step.
        pose_min_dist: Minimum distance from visited poses for candidates (meters).
        return_state: When True, return the estimator for inspection/testing.
        variable_cardinality: Override variable-K RJ; None uses the runtime config.
        num_particles: Particle count used by each isotope filter.
        environment_mode: Obstacle environment mode ("fixed" or "random").
        sim_backend: Explicit simulation backend name ("analytic", "isaacsim",
            or "geant4"). The backend has no implicit fallback.
        sim_config_path: Optional JSON config for the selected simulation backend.
        pf_config_path: PF-owned defaults overlaid beneath the physical config.
        blender_executable: Optional Blender executable path for random mode.
        blender_output_path: Optional USD path written by Blender in random mode.
        blender_timeout_s: Timeout for Blender environment generation.
        passage_width_m: Minimum reserved corridor width in random mode.
        robot_radius_m: Robot footprint radius used for 2D traversability maps.
        nominal_motion_speed_m_s: Nominal robot speed used for mission-time estimates.
        rotation_overhead_s: Fixed shield-actuation overhead per measurement.
        measurement_time_s: Predeclared fixed physical live time per action.
        path_planner: Joint pose-shield planner name. Only "dss_pp" is supported.
        dss_program_length: Number of shield postures in each DSS program.
        dss_rotation_weight: DSS shield-transition penalty weight.
        source_generation_mode: Source layout mode ("demo", "surface_random", or
            "provided_file").
        source_config_provenance: Exact-byte file provenance required for
            ``provided_file`` source layouts.
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
    if sim_backend is None:
        raise ValueError(
            "run_live_pf requires an explicit sim_backend. Use main.py "
            "--full-simulation for Geant4 or --python-cui for the approximate "
            "analytic backend."
        )
    normalized_sim_backend = sim_backend.strip().lower()
    if normalized_sim_backend not in {"analytic", "isaacsim", "geant4"}:
        raise ValueError(
            "sim_backend must be explicitly one of 'analytic', 'isaacsim', "
            "or 'geant4'."
        )
    sim_backend = normalized_sim_backend
    measurement_time_s = _strict_json_number(
        measurement_time_s,
        name="measurement_time_s",
        minimum=0.0,
        minimum_exclusive=True,
    )
    _validate_measurement_timing(measurement_time_s=measurement_time_s)
    notifier = PiplupNotifier(notification_config)
    live = _coerce_live_visualization(live)
    physical_runtime_config, runtime_config = load_online_runtime_configs(
        sim_config_path,
        pf_config_path,
    )
    input_config_hash = sha256_json(
        {
            "physical_runtime_config": physical_runtime_config,
            "online_pf_runtime_config": runtime_config,
        }
    )
    configured_backend = runtime_config.get("backend")
    if (
        not isinstance(configured_backend, str)
        or configured_backend.strip().lower() != sim_backend
    ):
        raise ValueError(
            "Runtime simulation backend does not match the resolved config: "
            f"requested={sim_backend!r}, configured={configured_backend!r}."
        )
    variable_cardinality = _resolve_variable_cardinality(
        variable_cardinality,
        runtime_config,
    )
    effective_robot_radius_m = _resolve_measurement_clearance_radius_m(
        runtime_config,
        requested_robot_radius_m=_strict_json_number(
            robot_radius_m,
            name="robot_radius_m",
            minimum=0.0,
        ),
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
    detector_pose_consistency_tolerance_m = _strict_json_number(
        runtime_config.get("detector_pose_consistency_tolerance_m", 1.0e-4),
        name="detector_pose_consistency_tolerance_m",
        minimum=0.0,
    )
    detector_height_pair_xy_tolerance_m = _strict_json_number(
        runtime_config.get("detector_height_pair_xy_tolerance_m", 1.0e-6),
        name="detector_height_pair_xy_tolerance_m",
        minimum=0.0,
    )
    detector_height_pair_z_tolerance_m = _strict_json_number(
        runtime_config.get("detector_height_pair_z_tolerance_m", 1.0e-9),
        name="detector_height_pair_z_tolerance_m",
        minimum=0.0,
    )
    continuous_height_bounds_for_dss = (
        detector_height_min_world_z_m,
        detector_height_max_world_z_m,
    )
    env = EnvironmentConfig(
        size_x=10.0,
        size_y=20.0,
        size_z=environment_size_z_m,
        detector_position=(1.0, 1.0, initial_detector_world_z_m),
    )
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
    if not isinstance(source_generation_mode, str):
        raise TypeError("source_generation_mode must be a string.")
    normalized_source_generation_mode = source_generation_mode.strip().lower()
    if normalized_source_generation_mode not in {
        "demo",
        "surface_random",
        "provided_file",
    }:
        raise ValueError(
            "source_generation_mode must be 'demo', 'surface_random', or "
            "'provided_file'."
        )
    if normalized_source_generation_mode == "provided_file":
        if sources is None:
            raise ValueError("provided_file source mode requires explicit sources.")
        if source_config_provenance is None:
            raise ValueError(
                "provided_file source mode requires exact source_config_provenance."
            )
        provided_source_provenance = _validated_provided_source_provenance(
            source_config_provenance
        )
    else:
        if source_config_provenance is not None:
            raise ValueError(
                "source_config_provenance is valid only for provided_file mode."
            )
        provided_source_provenance = None
    full_spectrum_model = geometry_conditioned_model_from_runtime_config(
        runtime_config
    )
    full_spectrum_model.require_runtime_ready()
    if not full_spectrum_model.production_ready:
        print(
            "Full-spectrum model is runtime-ready but has not completed "
            "independent production validation; continuing with the exact "
            "configured model contract.",
            flush=True,
        )
    if str(sim_backend).strip().lower() == "geant4":
        from sim.geant4_app.app import Geant4AppConfig

        geant4_contract = Geant4AppConfig.from_dict(runtime_config)
        invalid_native_contract: list[str] = []
        if not geant4_contract.sample_detector_response:
            invalid_native_contract.append("sample_detector_response=true")
        if geant4_contract.detector_scoring_mode != "incident_gamma_energy":
            invalid_native_contract.append(
                "detector_scoring_mode=incident_gamma_energy"
            )
        if geant4_contract.secondary_transport_mode != "full_transport":
            invalid_native_contract.append(
                "secondary_transport_mode=full_transport"
            )
        if geant4_contract.primary_sampling_fraction != 1.0:
            invalid_native_contract.append("primary_sampling_fraction=1.0")
        if geant4_contract.target_sampled_primaries is not None:
            invalid_native_contract.append("target_sampled_primaries=null")
        if geant4_contract.accelerated_weighted_transport_enable:
            invalid_native_contract.append(
                "accelerated_weighted_transport_enable=false"
            )
        if geant4_contract.source_rate_model != "detector_cps_1m":
            invalid_native_contract.append(
                "source_rate_model=detector_cps_1m"
            )
        if not np.isclose(
            float(geant4_contract.background_cps),
            float(full_spectrum_model.background_rate_cps),
            rtol=0.0,
            atol=1.0e-12,
        ):
            invalid_native_contract.append(
                "background_cps=full_spectrum_model.background_rate_cps"
            )
        if not np.isclose(
            float(geant4_contract.dead_time_tau_s),
            float(full_spectrum_model.dead_time_tau_s),
            rtol=0.0,
            atol=1.0e-18,
        ):
            invalid_native_contract.append(
                "dead_time_tau_s=full_spectrum_model.dead_time_tau_s"
            )
        if invalid_native_contract:
            raise ValueError(
                "Production Geant4 pure PF requires native unit-history "
                "full-spectrum response sampling before the sidecar starts; "
                "fix: "
                + ", ".join(invalid_native_contract)
                + "."
            )
    spectrum_isotopes = tuple(
        sorted(
            {
                str(row["isotope"])
                for row in full_spectrum_model.line_identity
            }
        )
    )
    measurement_log_target = _resolve_required_measurement_log_target(
        measurement_log_output,
        runtime_config,
        repository_root=ROOT,
    )
    obstacle_environment = build_runtime_obstacle_environment(
        root=SIMULATION_RUNTIME_ROOT,
        environment_mode=environment_mode,
        obstacle_layout_path=obstacle_layout_path,
        room_size_xyz=(env.size_x, env.size_y, env.size_z),
        detector_position_xy=env.detector_position,
        obstacle_seed=obstacle_seed,
        blocked_fraction=0.4,
        passage_width_m=passage_width_m,
        attach_known_transport=True,
        obstacle_height_m=_strict_json_number(
            runtime_config.get("obstacle_height_m", 2.0),
            name="obstacle_height_m",
            minimum=0.0,
        ),
        include_room_boundaries=_runtime_bool(
            runtime_config,
            "author_room_boundary_prims",
            False,
        ),
        room_boundary_thickness_m=_strict_json_number(
            runtime_config.get("room_boundary_thickness_m", 0.1),
            name="room_boundary_thickness_m",
            minimum=0.0,
            minimum_exclusive=True,
        ),
    )
    obstacle_grid = obstacle_environment.grid
    normalized_environment_mode = obstacle_environment.mode
    known_obstacle_instances = obstacle_environment.known_obstacle_instances
    measurement_log_obstacle_layout_path = _measurement_log_obstacle_layout_path(
        obstacle_environment,
        repository_root=SIMULATION_RUNTIME_ROOT,
    )
    runtime_obstacle_material = _strict_json_string(
        runtime_config.get("obstacle_material", "concrete"),
        name="obstacle_material",
    )
    if obstacle_environment.message is not None:
        print(obstacle_environment.message)
    obstacle_asset_summary = obstacle_environment.asset_summary()
    if obstacle_asset_summary is not None:
        print(obstacle_asset_summary)
    source_population_strength_bounds: tuple[float, float] | None = None
    source_sampling_metadata: dict[str, object] = {
        "mode": normalized_source_generation_mode,
        "measure": None,
        "selection_conditioning": None,
    }
    if provided_source_provenance is not None:
        source_sampling_metadata.update(provided_source_provenance)
    if normalized_source_generation_mode == "surface_random":
        source_surface_sampling_measure = validate_area_uniform_source_config(
            runtime_config
        )
        source_rng_root_seed = (
            int(np.random.SeedSequence().entropy)
            if random_source_seed is None and obstacle_seed is None
            else int(
                obstacle_seed
                if random_source_seed is None
                else random_source_seed
            )
        )
        source_rng_seed = named_stream_seed(
            source_rng_root_seed,
            _TRUTH_SURFACE_SOURCE_RNG_DOMAIN,
        )
        source_rng = named_random_generator(
            source_rng_root_seed,
            _TRUTH_SURFACE_SOURCE_RNG_DOMAIN,
        )
        source_rng_provenance = named_rng_provenance(
            source_rng_root_seed,
            (_TRUTH_SURFACE_SOURCE_RNG_DOMAIN,),
        )
        source_isotopes = _resolve_random_source_isotopes(
            random_source_isotopes,
            runtime_config,
            spectrum_isotopes,
        )
        intensity_min_payload = (
            random_source_intensity_min_cps_1m
            if random_source_intensity_min_cps_1m is not None
            else runtime_config.get("random_source_intensity_min_cps_1m")
        )
        intensity_max_payload = (
            random_source_intensity_max_cps_1m
            if random_source_intensity_max_cps_1m is not None
            else runtime_config.get("random_source_intensity_max_cps_1m")
        )
        random_source_intensity_spec: float | tuple[float, float]
        if intensity_min_payload is not None or intensity_max_payload is not None:
            if intensity_min_payload is None or intensity_max_payload is None:
                raise ValueError(
                    "random source intensity min/max must be provided together."
                )
            random_source_intensity_spec = (
                _strict_json_number(
                    intensity_min_payload,
                    name="random_source_intensity_min_cps_1m",
                    minimum=0.0,
                    minimum_exclusive=True,
                ),
                _strict_json_number(
                    intensity_max_payload,
                    name="random_source_intensity_max_cps_1m",
                    minimum=0.0,
                    minimum_exclusive=True,
                ),
            )
            if (
                random_source_intensity_spec[1]
                < random_source_intensity_spec[0]
            ):
                raise ValueError(
                    "random source intensity maximum must be at least the "
                    "minimum."
                )
            source_population_strength_bounds = random_source_intensity_spec
        else:
            random_source_intensity_spec = _strict_json_number(
                random_source_intensity_cps_1m,
                name="random_source_intensity_cps_1m",
                minimum=0.0,
                minimum_exclusive=True,
            )
        print(
            "Random source surface sampling: "
            f"measure={source_surface_sampling_measure}, "
            "selection_conditioning=none_physical_area_only"
        )
        source_sampling_metadata = {
            "mode": normalized_source_generation_mode,
            "measure": source_surface_sampling_measure,
            "selection_conditioning": "none_physical_area_only",
            "rng_provenance": source_rng_provenance,
        }
        if (
            isinstance(random_source_count, bool)
            or not isinstance(random_source_count, (int, np.integer))
            or int(random_source_count) <= 0
        ):
            raise ValueError("random_source_count must be a positive integer.")
        sources = generate_surface_sources(
            env=env,
            obstacle_grid=obstacle_grid,
            isotopes=source_isotopes,
            intensity_cps_1m=random_source_intensity_spec,
            rng=source_rng,
            count=int(random_source_count),
            obstacle_height_m=_strict_json_number(
                runtime_config.get("obstacle_height_m", 2.0),
                name="obstacle_height_m",
                minimum=0.0,
            ),
            chart_max_edge_m=_strict_json_number(
                runtime_config.get(
                    "structural_rj_surface_chart_max_edge_m",
                    1.0,
                ),
                name="structural_rj_surface_chart_max_edge_m",
                minimum=0.0,
                minimum_exclusive=True,
            ),
        )
        print(
            "Generated continuous area-uniform surface sources: "
            f"count={len(sources)}, root_seed={source_rng_root_seed}, "
            f"domain={_TRUTH_SURFACE_SOURCE_RNG_DOMAIN}, "
            f"derived_seed={source_rng_seed}, "
            f"isotopes={list(source_isotopes)}, "
            "intensity_cps_1m="
            f"{_format_random_source_intensity_spec(random_source_intensity_spec)}"
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
    estimate_trace_enabled = _runtime_bool(
        runtime_config,
        "intermediate_estimate_trace",
        True,
    )
    estimate_trace_log_enabled = _runtime_bool(
        runtime_config,
        "intermediate_estimate_trace_log",
        True,
    )
    estimate_trace_log_every = _strict_json_integer(
        runtime_config.get("intermediate_estimate_trace_log_every", 1),
        name="intermediate_estimate_trace_log_every",
        minimum=1,
    )
    estimate_trace_max_log_records = _strict_json_integer(
        runtime_config.get("intermediate_estimate_trace_max_log_records", 6),
        name="intermediate_estimate_trace_max_log_records",
        minimum=0,
    )
    surface_observability_diagnostic_candidates = _strict_json_integer(
        runtime_config.get("surface_observability_diagnostic_candidates", 0),
        name="surface_observability_diagnostic_candidates",
        minimum=0,
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
            obstacle_height_m=_strict_json_number(
                runtime_config.get("obstacle_height_m", 2.0),
                name="obstacle_height_m",
                minimum=0.0,
            ),
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

    _validate_surface_constrained_sources(
        sources,
        env,
        obstacle_grid,
        obstacle_height_m=_strict_json_number(
            runtime_config.get("obstacle_height_m", 2.0),
            name="obstacle_height_m",
            minimum=0.0,
        ),
        tolerance_m=_strict_json_number(
            runtime_config.get("posterior_surface_tolerance_m", 1.0e-5),
            name="posterior_surface_tolerance_m",
            minimum=0.0,
        ),
    )
    source_position_max = (float(env.size_x), float(env.size_y), float(env.size_z))
    bounds_lo = np.array(
        [0.0, 0.0, detector_height_min_world_z_m],
        dtype=float,
    )
    bounds_hi = np.array(
        [env.size_x, env.size_y, detector_height_max_world_z_m],
        dtype=float,
    )

    isotopes = sorted(
        _resolve_candidate_isotopes(runtime_config, spectrum_isotopes)
    )
    measurement_log_runtime_config = estimator_neutral_runtime_config(
        physical_runtime_config,
        backend=str(sim_backend),
        isotopes=isotopes,
        run_root=SIMULATION_RUNTIME_ROOT,
    )
    measurement_log_runtime_config.update(
        {
            "sim_backend": str(sim_backend),
            "candidate_isotopes": [str(value) for value in isotopes],
            "source_rate_model": "detector_cps_1m",
            "environment_mode": str(normalized_environment_mode),
            "station_update": "joint_full_spectrum",
            "energy_min_keV": float(full_spectrum_model.energy_axis_keV[0]),
            "energy_max_keV": float(full_spectrum_model.energy_axis_keV[-1]),
            "bin_width_keV": float(
                full_spectrum_model.energy_axis_keV[1]
                - full_spectrum_model.energy_axis_keV[0]
            ),
            "energy_bin_count": int(
                full_spectrum_model.energy_axis_keV.size
            ),
            "background_rate_cps": float(
                full_spectrum_model.background_rate_cps
            ),
            "dead_time_tau_s": float(full_spectrum_model.dead_time_tau_s),
            "full_spectrum_generative_model": (
                full_spectrum_model.manifest_payload()
            ),
            "full_spectrum_contract_hash_sha256": (
                full_spectrum_model.contract_hash_sha256
            ),
        }
    )
    pf_random_seed = _strict_json_integer(
        runtime_config.get(
            "pf_random_seed",
            runtime_config.get("random_seed", runtime_config.get("rng_seed", 0)),
        ),
        name="pf_random_seed",
        minimum=0,
        maximum=(1 << 128) - 1,
    )
    planning_root_seed = _strict_json_integer(
        runtime_config.get(
            "planning_random_seed",
            named_stream_seed(pf_random_seed, "live_planning"),
        ),
        name="planning_random_seed",
        minimum=0,
        maximum=(1 << 128) - 1,
    )
    planning_candidate_seed = _strict_json_integer(
        runtime_config.get(
            "planning_candidate_seed",
            named_stream_seed(
                planning_root_seed,
                "live_planning_candidate",
            ),
        ),
        name="planning_candidate_seed",
        minimum=0,
        maximum=(1 << 128) - 1,
    )
    dss_eig_seed = _strict_json_integer(
        runtime_config.get(
            "planning_dss_eig_seed",
            named_stream_seed(
                planning_root_seed,
                "live_planning_dss_eig",
            ),
        ),
        name="planning_dss_eig_seed",
        minimum=0,
        maximum=(1 << 128) - 1,
    )
    planning_candidate_rng = np.random.default_rng(planning_candidate_seed)
    dss_eig_rng = np.random.default_rng(dss_eig_seed)
    print(
        "PF candidate isotopes: "
        f"{isotopes} (full_spectrum_lines={list(spectrum_isotopes)})"
    )
    last_candidates: set[str] = set()
    num_particles = _strict_json_integer(
        num_particles,
        name="num_particles",
        minimum=1,
    )
    pose_candidates = _strict_json_integer(
        pose_candidates,
        name="pose_candidates",
        minimum=2,
    )
    pose_min_dist = _strict_json_number(
        pose_min_dist,
        name="pose_min_dist",
        minimum=0.0,
    )
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
    additive_scatter_response = observation_model.additive_scatter_response
    obstacle_mu_by_isotope = observation_model.obstacle_mu_by_isotope
    if line_mu_by_isotope is not None:
        print(
            "PF line-resolved shield attenuation: enabled "
            f"(isotopes={','.join(sorted(line_mu_by_isotope))})"
        )
    else:
        print("PF line-resolved shield attenuation: disabled")
    print(
        "PF additive noncollided transport response: "
        f"{'enabled' if additive_scatter_response is not None else 'disabled'}"
    )
    use_gpu = _resolve_runtime_use_gpu(runtime_config)
    gpu_dtype_resolved = _strict_json_string(
        runtime_config.get("gpu_dtype", "float64"),
        name="gpu_dtype",
    ).strip().lower()
    if gpu_dtype_resolved != "float64":
        raise ValueError(
            "Production pure-PF runtime requires gpu_dtype='float64'; "
            "lower-precision posterior dynamics are forbidden."
        )
    live_time = float(measurement_time_s)
    planning_live_time = live_time
    path_planner_resolved = _strict_json_string(
        path_planner
        if path_planner is not None
        else runtime_config.get("path_planner", "dss_pp"),
        name="path_planner",
    ).strip().lower()
    if path_planner_resolved != "dss_pp":
        raise ValueError(
            "Pure runtime supports only path_planner='dss_pp'; the legacy "
            "independent-isotope one-step selector is retired."
        )
    dss_runtime = runtime_config.get("dss_pp", {})
    if not isinstance(dss_runtime, dict):
        raise ValueError("dss_pp must be an object.")
    retired_scalar_shield_keys = {
        "shield_signature_weight",
        "shield_low_count_penalty_weight",
        "shield_count_balance_weight",
        "shield_rotation_cost_weight",
        "shield_signature_variance_floor",
        "shield_selection_max_particles",
        "shield_stop_min_gain",
        "shield_stop_compare_next_pose",
        "shield_stop_pose_candidates",
        "shield_stop_rate_margin",
        "shield_stop_signature_cosine",
        "one_step_pose_eval_workers",
        "one_step_pose_eval_use_gpu",
    }
    configured_scalar_shield_keys = sorted(
        retired_scalar_shield_keys.intersection(runtime_config)
    )
    if configured_scalar_shield_keys:
        raise ValueError(
            "Legacy independent-isotope shield selection is retired; every "
            "program must come from joint DSS-PP or an explicit baseline. "
            f"Remove keys: {configured_scalar_shield_keys}."
        )
    retired_dss_information_keys = {
        "beam_width",
        "signature_weight",
        "temporal_separation_weight",
        "count_utility_weight",
        "station_condition_weight",
        "station_condition_min_singular_weight",
        "station_condition_inverse_condition_weight",
        "station_condition_coherence_weight",
        "correlation_reduction_weight",
        "isotope_balance_weight",
        "environment_signature_weight",
        "horizon",
        "environment_signature_score_clip",
        "environment_contrast_threshold",
        "occlusion_boundary_weight",
        "occlusion_boundary_step_m",
        "elevation_signature_weight",
        "vertical_environment_signature_weight",
        "observation_weight",
        "differential_weight",
        "count_balance_weight",
        "enforce_min_observation",
        "signature_std_min_counts",
        "count_variance_floor",
        "count_utility_saturation_counts",
        "temporal_cover_weight",
        "temporal_logdet_weight",
        "temporal_decorrelation_weight",
        "temporal_pair_contrast_threshold",
        "temporal_logdet_ridge",
        "temporal_cover_programs",
        "temporal_cover_beam_width",
    }
    configured_retired_dss_keys = sorted(
        retired_dss_information_keys.intersection(dss_runtime)
    )
    if configured_retired_dss_keys:
        raise ValueError(
            "Retired DSS settings are forbidden because one-step joint "
            "full-spectrum generative EIG is the sole observation-information "
            f"score; remove keys: {configured_retired_dss_keys}."
        )
    python_worker_count_resolved = _resolve_python_worker_count(
        runtime_config.get(
            "python_worker_count",
            runtime_config.get("cpu_worker_count", 0),
        )
    )

    def _dss_value(key: str, default: object) -> object:
        """Read a DSS-PP setting from CLI override or runtime config."""
        return dss_runtime.get(key, default)

    def _dss_integer(
        key: str,
        default: int,
        *,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> int:
        """Return one strictly typed DSS integer from runtime configuration."""
        return _strict_json_integer(
            _dss_value(key, default),
            name=f"dss_pp.{key}",
            minimum=minimum,
            maximum=maximum,
        )

    def _dss_number(
        key: str,
        default: float,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
        minimum_exclusive: bool = False,
        maximum_exclusive: bool = False,
    ) -> float:
        """Return one finite DSS number from runtime configuration."""
        return _strict_json_number(
            _dss_value(key, default),
            name=f"dss_pp.{key}",
            minimum=minimum,
            maximum=maximum,
            minimum_exclusive=minimum_exclusive,
            maximum_exclusive=maximum_exclusive,
        )

    def _dss_bool(key: str, default: bool) -> bool:
        """Return one exact DSS boolean from runtime configuration."""
        return _strict_json_bool(
            _dss_value(key, default),
            name=f"dss_pp.{key}",
        )

    pf_max_sources = _strict_json_integer(
        runtime_config.get(
            "pf_max_sources",
            DEFAULT_MAX_SOURCES_PER_ISOTOPE,
        ),
        name="pf_max_sources",
        minimum=1,
    )
    dss_program_length_resolved = _strict_json_integer(
        dss_program_length
        if dss_program_length is not None
        else _dss_value("program_length", 2),
        name="dss_pp.program_length",
        minimum=1,
    )
    dss_rotation_weight_resolved = _strict_json_number(
        dss_rotation_weight
        if dss_rotation_weight is not None
        else _dss_value("rotation_weight", 0.15),
        name="dss_pp.rotation_weight",
        minimum=0.0,
    )
    dss_planning_particles_resolved = _dss_value(
        "planning_particles",
        512,
    )
    dss_planning_method_resolved = _dss_value(
        "planning_method",
        "resample",
    )
    if dss_planning_particles_resolved is not None:
        dss_planning_particles_resolved = _strict_json_integer(
            dss_planning_particles_resolved,
            name="dss_pp.planning_particles",
            minimum=2,
        )
    if dss_planning_method_resolved != "resample":
        raise ValueError(
            "dss_pp.planning_method must be exactly 'resample' so the "
            "planning particle subset preserves posterior mass."
        )
    dss_max_modes_per_isotope = _dss_integer(
        "max_modes_per_isotope",
        pf_max_sources,
        minimum=1,
    )
    if dss_max_modes_per_isotope < pf_max_sources:
        raise ValueError(
            "dss_pp.max_modes_per_isotope must be at least pf_max_sources; "
            "otherwise the planner silently drops posterior source modes."
        )
    dss_max_augmented_candidates = _dss_integer(
        "max_augmented_candidates",
        256,
        minimum=1,
    )
    if dss_max_augmented_candidates < int(pose_candidates):
        raise ValueError(
            "dss_pp.max_augmented_candidates must be at least pose_candidates."
        )
    dss_config = DSSPPConfig(
        max_programs=_dss_integer("max_programs", 40, minimum=1),
        program_length=dss_program_length_resolved,
        mode_cluster_radius_m=_dss_number(
            "mode_cluster_radius_m",
            1.5,
            minimum=0.0,
            minimum_exclusive=True,
        ),
        max_modes_per_isotope=dss_max_modes_per_isotope,
        planning_particles=(
            None
            if dss_planning_particles_resolved is None
            else dss_planning_particles_resolved
        ),
        planning_method="resample",
        live_time_s=planning_live_time,
        lambda_eig=_dss_number("eig_weight", 1.0, minimum=0.0),
        lambda_distance=(
            None
            if _dss_value("distance_weight", None) is None
            else _dss_number("distance_weight", 0.0, minimum=0.0)
        ),
        lambda_time=_dss_number("time_weight", 0.0, minimum=0.0),
        lambda_rotation=dss_rotation_weight_resolved,
        lambda_coverage=_dss_number("coverage_weight", 0.0, minimum=0.0),
        lambda_bearing_diversity=_dss_number(
            "bearing_diversity_weight",
            0.0,
            minimum=0.0,
        ),
        lambda_frontier=_dss_number(
            "frontier_weight",
            0.0,
            minimum=0.0,
        ),
        lambda_turn_smoothness=_dss_number(
            "turn_smoothness_weight",
            0.0,
            minimum=0.0,
        ),
        lambda_local_orbit=_dss_number(
            "local_orbit_weight",
            0.75,
            minimum=0.0,
        ),
        lambda_elevation_condition=_dss_number(
            "elevation_condition_weight",
            0.0,
            minimum=0.0,
        ),
        eta_revisit=_dss_number(
            "revisit_penalty_weight",
            0.0,
            minimum=0.0,
        ),
        coverage_radius_m=_dss_number(
            "coverage_radius_m",
            3.0,
            minimum=0.0,
        ),
        coverage_surface_quadrature_max_points=_dss_integer(
            "coverage_surface_quadrature_max_points",
            65536,
            minimum=1,
        ),
        coverage_surface_max_hausdorff_m=_dss_number(
            "coverage_surface_max_hausdorff_m",
            0.75,
            minimum=0.0,
            minimum_exclusive=True,
        ),
        coverage_floor_quantile=_dss_number(
            "coverage_floor_quantile",
            0.0,
            minimum=0.0,
            maximum=1.0,
        ),
        coverage_floor_weight=_dss_number(
            "coverage_floor_weight",
            0.0,
            minimum=0.0,
        ),
        min_station_separation_m=_dss_number(
            "min_station_separation_m",
            pose_min_dist,
            minimum=0.0,
        ),
        detector_aperture_samples=_dss_integer(
            "detector_aperture_samples",
            detector_geometry.aperture_samples,
            minimum=1,
        ),
        robot_speed_m_s=_strict_json_number(
            nominal_motion_speed_m_s,
            name="nominal_motion_speed_m_s",
            minimum=0.0,
            minimum_exclusive=True,
        ),
        rotation_overhead_s=_strict_json_number(
            rotation_overhead_s,
            name="rotation_overhead_s",
            minimum=0.0,
        ),
        augment_candidates=_dss_bool("augment_candidates", True),
        max_augmented_candidates=dss_max_augmented_candidates,
        local_orbit_sigma_m=_dss_number(
            "local_orbit_sigma_m",
            0.75,
            minimum=0.0,
            minimum_exclusive=True,
        ),
        elevation_pair_z_scale_m=_dss_number(
            "elevation_pair_z_scale_m",
            2.0,
            minimum=0.0,
            minimum_exclusive=True,
        ),
        elevation_pair_xy_scale_m=_dss_number(
            "elevation_pair_xy_scale_m",
            4.0,
            minimum=0.0,
            minimum_exclusive=True,
        ),
        elevation_angle_threshold_deg=_dss_number(
            "elevation_angle_threshold_deg",
            15.0,
            minimum=0.0,
            maximum=180.0,
            minimum_exclusive=True,
        ),
        diagnostic_ranked_node_limit=_dss_integer(
            "diagnostic_ranked_node_limit",
            64,
            minimum=0,
        ),
        exact_eig_action_limit=_dss_integer(
            "exact_eig_action_limit",
            32,
            minimum=1,
        ),
        exact_eig_coverage_reserve=_dss_integer(
            "exact_eig_coverage_reserve",
            4,
            minimum=0,
        ),
        exact_eig_program_diversity_reserve=_dss_integer(
            "exact_eig_program_diversity_reserve",
            4,
            minimum=0,
        ),
        proxy_memory_budget_bytes=_dss_integer(
            "proxy_memory_budget_bytes",
            256 * 1024 * 1024,
            minimum=1,
        ),
        proxy_planning_particles=_dss_integer(
            "proxy_planning_particles",
            16,
            minimum=2,
        ),
        proxy_eig_samples=_dss_integer(
            "proxy_eig_samples",
            2,
            minimum=1,
        ),
    )
    _validate_weighted_pf_runtime_contract(
        runtime_config,
        planning_primary_history_weight=_planning_primary_history_weight(
            runtime_config
        ),
    )
    adaptive_mission_stop = _runtime_bool(
        runtime_config,
        "adaptive_mission_stop",
        False,
    )
    max_steps = _resolve_mission_max_steps(max_steps, runtime_config)
    max_poses = _resolve_mission_max_poses(max_poses, runtime_config)
    mission_stop_min_convergence_poses = _strict_json_integer(
        runtime_config.get("mission_stop_min_convergence_poses", 4),
        name="mission_stop_min_convergence_poses",
        minimum=1,
    )
    if max_poses is not None and int(max_poses) > 0:
        mission_stop_min_convergence_poses = min(
            mission_stop_min_convergence_poses,
            int(max_poses),
        )
    mission_stop_require_pf_cardinality_ready = _runtime_bool(
        runtime_config,
        "mission_stop_require_pf_cardinality_ready",
        True,
    )
    isaac_pf_visualization_enabled = _runtime_bool(
        runtime_config,
        "isaacsim_show_pf_particles",
        True,
    )
    isaac_pf_max_particles_raw = runtime_config.get(
        "isaacsim_pf_max_particles_per_isotope",
        runtime_config.get("pf_visual_max_particles_per_isotope", 800),
    )
    isaac_pf_max_particles = (
        None
        if isaac_pf_max_particles_raw is None
        else _strict_json_integer(
            isaac_pf_max_particles_raw,
            name="isaacsim_pf_max_particles_per_isotope",
            minimum=1,
        )
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

    orientation_limit_resolved = _strict_json_integer(
        runtime_config.get("orientation_k", 2),
        name="orientation_k",
        minimum=1,
        maximum=64,
    )
    min_rotations_resolved = _strict_json_integer(
        runtime_config.get(
            "min_rotations_per_pose",
            min(2, orientation_limit_resolved),
        ),
        name="min_rotations_per_pose",
        minimum=0,
        maximum=64,
    )
    if min_rotations_resolved > orientation_limit_resolved:
        raise ValueError(
            "min_rotations_per_pose cannot exceed orientation_k."
        )
    init_num_sources_raw = runtime_config.get("init_num_sources", None)
    if init_num_sources_raw is not None:
        init_num_sources = (
            _strict_json_integer(
                init_num_sources_raw[0],
                name="init_num_sources[0]",
                minimum=0,
                maximum=pf_max_sources,
            ),
            _strict_json_integer(
                init_num_sources_raw[1],
                name="init_num_sources[1]",
                minimum=0,
                maximum=pf_max_sources,
            ),
        )
        if init_num_sources[0] > init_num_sources[1]:
            raise ValueError(
                "init_num_sources lower bound cannot exceed its upper bound."
            )
    else:
        if variable_cardinality:
            init_num_sources = (0, pf_max_sources)
        else:
            init_num_sources = (1, 1)
    strength_prior_minimum, strength_prior_maximum = (
        _resolve_pf_strength_prior_bounds(
            runtime_config,
            generated_population_bounds=source_population_strength_bounds,
        )
    )
    pf_conf = RotatingShieldPFConfig(
        estimator_profile=_strict_json_string(
            runtime_config.get("estimator_profile", "pf_strict"),
            name="estimator_profile",
        ),
        num_particles=num_particles,
        max_sources=pf_max_sources,
        variable_cardinality=variable_cardinality,
        structural_rj_surface_chart_max_edge_m=_strict_json_number(
            runtime_config.get("structural_rj_surface_chart_max_edge_m", 1.0),
            name="structural_rj_surface_chart_max_edge_m",
            minimum=0.0,
            minimum_exclusive=True,
        ),
        structural_rj_move_probability=_strict_json_number(
            runtime_config.get("structural_rj_move_probability", 1.0),
            name="structural_rj_move_probability",
            minimum=0.0,
            maximum=1.0,
        ),
        structural_rj_birth_probability=_strict_json_number(
            runtime_config.get("structural_rj_birth_probability", 0.5),
            name="structural_rj_birth_probability",
            minimum=0.0,
            maximum=1.0,
        ),
        structural_rj_death_probability=_strict_json_number(
            runtime_config.get("structural_rj_death_probability", 0.5),
            name="structural_rj_death_probability",
            minimum=0.0,
            maximum=1.0,
        ),
        structural_rj_position_move_probability=_strict_json_number(
            runtime_config.get(
                "structural_rj_position_move_probability",
                1.0,
            ),
            name="structural_rj_position_move_probability",
            minimum=0.0,
            maximum=1.0,
        ),
        structural_rj_position_proposal_prior_weight=_strict_json_number(
            runtime_config.get(
                "structural_rj_position_proposal_prior_weight",
                0.5,
            ),
            name="structural_rj_position_proposal_prior_weight",
            minimum=0.0,
            maximum=1.0,
            minimum_exclusive=True,
        ),
        structural_rj_strength_proposal_prior_weight=_strict_json_number(
            runtime_config.get(
                "structural_rj_strength_proposal_prior_weight",
                0.5,
            ),
            name="structural_rj_strength_proposal_prior_weight",
            minimum=0.0,
            maximum=1.0,
            minimum_exclusive=True,
        ),
        structural_rj_strength_proposal_sigma_fraction=_strict_json_number(
            runtime_config.get(
                "structural_rj_strength_proposal_sigma_fraction",
                0.15,
            ),
            name="structural_rj_strength_proposal_sigma_fraction",
            minimum=0.0,
            minimum_exclusive=True,
        ),
        structural_rj_strength_proposal_grid_size=_strict_json_integer(
            runtime_config.get(
                "structural_rj_strength_proposal_grid_size",
                9,
            ),
            name="structural_rj_strength_proposal_grid_size",
            minimum=2,
        ),
        structural_rj_proposal_chart_batch_size=_strict_json_integer(
            runtime_config.get(
                "structural_rj_proposal_chart_batch_size",
                256,
            ),
            name="structural_rj_proposal_chart_batch_size",
            minimum=1,
        ),
        structural_rj_proposal_score_cache_max_bytes=_strict_json_integer(
            runtime_config.get(
                "structural_rj_proposal_score_cache_max_bytes",
                268_435_456,
            ),
            name="structural_rj_proposal_score_cache_max_bytes",
            minimum=1,
        ),
        structural_rj_local_position_move_probability=_strict_json_number(
            runtime_config.get(
                "structural_rj_local_position_move_probability",
                1.0,
            ),
            name="structural_rj_local_position_move_probability",
            minimum=0.0,
            maximum=1.0,
        ),
        structural_rj_local_position_sigma_m=_strict_json_number(
            runtime_config.get("structural_rj_local_position_sigma_m", 0.5),
            name="structural_rj_local_position_sigma_m",
            minimum=0.0,
            minimum_exclusive=True,
        ),
        structural_rj_strength_move_probability=_strict_json_number(
            runtime_config.get(
                "structural_rj_strength_move_probability",
                1.0,
            ),
            name="structural_rj_strength_move_probability",
            minimum=0.0,
            maximum=1.0,
        ),
        structural_rj_split_merge_probability=_strict_json_number(
            runtime_config.get("structural_rj_split_merge_probability", 1.0),
            name="structural_rj_split_merge_probability",
            minimum=0.0,
            maximum=1.0,
        ),
        structural_rj_block_independence_probability=_strict_json_number(
            runtime_config.get(
                "structural_rj_block_independence_probability",
                0.1,
            ),
            name="structural_rj_block_independence_probability",
            minimum=0.0,
            maximum=1.0,
        ),
        structural_rj_multi_component_probability=_strict_json_number(
            runtime_config.get(
                "structural_rj_multi_component_probability",
                0.1,
            ),
            name="structural_rj_multi_component_probability",
            minimum=0.0,
            maximum=1.0,
        ),
        structural_rj_multi_component_max_group_size=_strict_json_integer(
            runtime_config.get(
                "structural_rj_multi_component_max_group_size",
                4,
            ),
            name="structural_rj_multi_component_max_group_size",
            minimum=3,
        ),
        structural_rj_split_probability=_strict_json_number(
            runtime_config.get("structural_rj_split_probability", 0.5),
            name="structural_rj_split_probability",
            minimum=0.0,
            maximum=1.0,
        ),
        structural_rj_merge_probability=_strict_json_number(
            runtime_config.get("structural_rj_merge_probability", 0.5),
            name="structural_rj_merge_probability",
            minimum=0.0,
            maximum=1.0,
        ),
        structural_rj_split_global_position_probability=_strict_json_number(
            runtime_config.get(
                "structural_rj_split_global_position_probability",
                0.1,
            ),
            name="structural_rj_split_global_position_probability",
            minimum=0.0,
            maximum=1.0,
        ),
        structural_rj_merge_uniform_pair_probability=_strict_json_number(
            runtime_config.get(
                "structural_rj_merge_uniform_pair_probability",
                0.1,
            ),
            name="structural_rj_merge_uniform_pair_probability",
            minimum=0.0,
            maximum=1.0,
        ),
        structural_rj_merge_distance_sigma_m=_strict_json_number(
            runtime_config.get("structural_rj_merge_distance_sigma_m", 0.5),
            name="structural_rj_merge_distance_sigma_m",
            minimum=0.0,
            minimum_exclusive=True,
        ),
        structural_rj_merge_response_sigma=_strict_json_number(
            runtime_config.get("structural_rj_merge_response_sigma", 0.05),
            name="structural_rj_merge_response_sigma",
            minimum=0.0,
            minimum_exclusive=True,
        ),
        structural_cardinality_prior_probs=runtime_config.get(
            "structural_cardinality_prior_probs"
        ),
        structural_cardinality_prior_policy=str(
            runtime_config[
                "structural_cardinality_prior_policy"
            ]
        ),
        structural_cardinality_prior_mean=_strict_json_number(
            runtime_config.get("structural_cardinality_prior_mean", 2.0),
            name="structural_cardinality_prior_mean",
            minimum=0.0,
            minimum_exclusive=True,
        ),
        max_dwell_time_s=10000.0,
        position_max=source_position_max,
        init_num_sources=init_num_sources,
        strength_prior_min_cps_1m=strength_prior_minimum,
        strength_prior_max_cps_1m=strength_prior_maximum,
        history_estimate_interval=_strict_json_integer(
            runtime_config.get("history_estimate_interval", 1),
            name="history_estimate_interval",
            minimum=0,
        ),
        surface_diagnostic_response_cache_max_entries=_strict_json_integer(
            runtime_config.get(
                "surface_diagnostic_response_cache_max_entries",
                24,
            ),
            name="surface_diagnostic_response_cache_max_entries",
            minimum=0,
        ),
        orientation_k=orientation_limit_resolved,
        min_rotations_per_pose=min_rotations_resolved,
        planning_eig_samples=_strict_json_integer(
            runtime_config.get("planning_eig_samples", 50),
            name="planning_eig_samples",
            minimum=1,
        ),
        use_gpu=use_gpu,
        gpu_device=_strict_json_string(
            runtime_config.get("gpu_device", "cuda"),
            name="gpu_device",
        ),
        gpu_dtype=gpu_dtype_resolved,
    )
    pf_conf.max_temper_steps = _strict_json_integer(
        runtime_config.get("max_temper_steps", 256),
        name="max_temper_steps",
        minimum=1,
    )
    pf_conf.min_delta_beta = _strict_json_number(
        runtime_config.get("min_delta_beta", 1.0e-10),
        name="min_delta_beta",
        minimum=0.0,
        maximum=1.0,
        minimum_exclusive=True,
    )
    pf_conf.target_ess_ratio = _strict_json_number(
        runtime_config.get("target_ess_ratio", 0.4),
        name="target_ess_ratio",
        minimum=0.0,
        maximum=1.0,
        minimum_exclusive=True,
    )
    pf_conf.joint_rejuvenation_min_sweeps = _strict_json_integer(
        runtime_config.get("joint_rejuvenation_min_sweeps", 1),
        name="joint_rejuvenation_min_sweeps",
        minimum=1,
    )
    pf_conf.joint_rejuvenation_max_sweeps = _strict_json_integer(
        runtime_config.get("joint_rejuvenation_max_sweeps", 2),
        name="joint_rejuvenation_max_sweeps",
        minimum=1,
    )
    pf_conf.joint_rejuvenation_min_state_change_weight_mass = (
        _strict_json_number(
            runtime_config.get(
                "joint_rejuvenation_min_state_change_weight_mass",
                0.1,
            ),
            name="joint_rejuvenation_min_state_change_weight_mass",
            minimum=0.0,
            maximum=1.0,
        )
    )
    pf_conf.joint_rejuvenation_min_surface_esjd_m2 = _strict_json_number(
        runtime_config.get("joint_rejuvenation_min_surface_esjd_m2", 1.0e-4),
        name="joint_rejuvenation_min_surface_esjd_m2",
        minimum=0.0,
    )
    pf_conf.joint_rejuvenation_min_log_strength_esjd = _strict_json_number(
        runtime_config.get(
            "joint_rejuvenation_min_log_strength_esjd",
            1.0e-4,
        ),
        name="joint_rejuvenation_min_log_strength_esjd",
        minimum=0.0,
    )
    pf_conf.joint_smc_soft_wall_time_s = _strict_json_number(
        runtime_config.get("joint_smc_soft_wall_time_s", 1800.0),
        name="joint_smc_soft_wall_time_s",
        minimum=0.0,
        minimum_exclusive=True,
    )
    pf_conf.joint_guided_initialization = _strict_json_bool(
        runtime_config.get("joint_guided_initialization", True),
        name="joint_guided_initialization",
    )
    pf_conf.joint_guided_initialization_prior_row_probability = (
        _strict_json_number(
            runtime_config.get(
                "joint_guided_initialization_prior_row_probability",
                0.5,
            ),
            name="joint_guided_initialization_prior_row_probability",
            minimum=0.0,
            maximum=1.0,
            minimum_exclusive=True,
        )
    )
    pf_conf.credible_surface_radius_threshold_m = _strict_json_number(
        runtime_config.get("credible_surface_radius_threshold_m", 0.5),
        name="credible_surface_radius_threshold_m",
        minimum=0.0,
        minimum_exclusive=True,
    )
    pf_conf.converge_min_ess_ratio = _strict_json_number(
        runtime_config.get(
            "converge_min_ess_ratio",
            pf_conf.target_ess_ratio,
        ),
        name="converge_min_ess_ratio",
        minimum=0.0,
        maximum=1.0,
        minimum_exclusive=True,
    )
    pf_conf.converge_cardinality_min_probability = _strict_json_number(
        runtime_config.get("converge_cardinality_min_probability", 0.95),
        name="converge_cardinality_min_probability",
        minimum=0.0,
        maximum=1.0,
    )
    pf_conf.converge_max_cardinality_boundary_mass = _strict_json_number(
        runtime_config.get("converge_max_cardinality_boundary_mass", 0.05),
        name="converge_max_cardinality_boundary_mass",
        minimum=0.0,
        maximum=1.0,
    )
    pf_conf.converge_innovation_confidence = _strict_json_number(
        runtime_config.get("converge_innovation_confidence", 0.99),
        name="converge_innovation_confidence",
        minimum=0.0,
        maximum=1.0,
        minimum_exclusive=True,
        maximum_exclusive=True,
    )
    pf_conf.converge_cardinality_var_max = _strict_json_number(
        runtime_config.get("converge_cardinality_var_max", 0.05),
        name="converge_cardinality_var_max",
        minimum=0.0,
    )
    if pf_config_overrides:
        for key, value in pf_config_overrides.items():
            if key == "position_max":
                raise ValueError(
                    "Pure PF derives position_max from the complete environment."
                )
            if not hasattr(pf_conf, key):
                raise ValueError(f"Unknown PF config override: {key}")
            setattr(pf_conf, key, value)
    pf_conf.variable_cardinality = variable_cardinality
    # Overrides and the cardinality mode are applied after construction, so
    # rerun normalization and exact-kernel compatibility checks.
    pf_conf.__post_init__()
    pf_compute_backend = _preflight_pure_pf_compute_backend(
        use_gpu=bool(pf_conf.use_gpu),
        gpu_device=str(pf_conf.gpu_device),
        gpu_dtype=str(pf_conf.gpu_dtype),
    )
    (
        surface_diagnostic_points,
        surface_atlas_diagnostics,
        runtime_surface_chart_geometry,
    ) = (
        _physical_surface_atlas_diagnostic_points(
            env,
            pf_obstacle_grid,
            chart_max_edge_m=float(
                pf_conf.structural_rj_surface_chart_max_edge_m
            ),
            point_count=max(
                _strict_json_integer(
                    runtime_config.get(
                        "final_surface_observability_candidates",
                        1024,
                    ),
                    name="final_surface_observability_candidates",
                    minimum=1,
                ),
                int(surface_observability_diagnostic_candidates),
            ),
        )
    )
    _validate_provided_surface_source_contract(
        provided_source_provenance,
        sources,
        chart_geometry=runtime_surface_chart_geometry,
        obstacle_seed=obstacle_seed,
        chart_max_edge_m=float(
            pf_conf.structural_rj_surface_chart_max_edge_m
        ),
    )
    sources = _bind_sources_to_surface_transport(
        sources,
        runtime_surface_chart_geometry,
    )
    source_sampling_metadata = {
        **source_sampling_metadata,
        "surface_emission_policy": surface_emission_policy_payload(),
        "surface_emission_policy_sha256": (
            surface_emission_policy_sha256()
        ),
        "surface_atlas_contract_sha256": (
            surface_chart_geometry_sha256(runtime_surface_chart_geometry)
        ),
    }
    _validate_truth_within_pf_state_support(
        sources,
        candidate_isotopes=isotopes,
        max_sources_per_isotope=int(pf_conf.max_sources or 0),
        strength_prior_min_cps_1m=float(
            pf_conf.strength_prior_min_cps_1m
        ),
        strength_prior_max_cps_1m=float(
            pf_conf.strength_prior_max_cps_1m
        ),
    )
    # Runtime creation may launch an external Geant4 process.  Keep it after
    # every truth/PF support preflight so an impossible experiment never starts
    # transport.
    simulation_runtime = create_simulation_runtime(
        sim_backend,
        sources=sources,
        mu_by_isotope=mu_by_isotope,
        shield_params=shield_params,
        runtime_config=physical_runtime_config,
        runtime_config_path=sim_config_path,
    )
    strict_planned_shield_program = _runtime_bool(
        runtime_config,
        "strict_planned_shield_program",
        True,
    )
    baseline_shield_policy = runtime_config.get("baseline_shield_policy")
    baseline_path_policy = runtime_config.get("baseline_path_policy")
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
    pf_detector_radius_m = detector_geometry.count_radius_m
    pf_detector_aperture_radius_m = detector_geometry.aperture_radius_m
    pf_detector_aperture_samples = detector_geometry.aperture_samples
    pf_detector_aperture_sampling = detector_geometry.aperture_sampling
    pf_source_extent_radius_m = observation_model.source_extent_radius_m
    pf_source_extent_samples = observation_model.source_extent_samples

    # Keep the PF configuration in estimator provenance. MeasurementLog owns
    # only the estimator-neutral physical acquisition contract.
    apply_profile_to_config(pf_conf)
    effective_pf_runtime_config = _build_effective_live_runtime_config(
        runtime_config,
        pf_config=pf_conf,
        surface_diagnostic_points_xyz=np.asarray(
            surface_diagnostic_points,
            dtype=np.float64,
        ),
        surface_atlas_diagnostics=surface_atlas_diagnostics,
        api_settings={
            "max_steps": max_steps,
            "max_poses": max_poses,
            "variable_cardinality": bool(variable_cardinality),
            "num_particles": int(num_particles),
            "surface_prior_domain": "continuous_environment_surface",
            "obstacle_height_m": _strict_json_number(
                runtime_config.get("obstacle_height_m", 2.0),
                name="obstacle_height_m",
                minimum=0.0,
            ),
            "pose_candidates": int(pose_candidates),
            "pose_min_dist_m": float(pose_min_dist),
            "path_planner": str(path_planner_resolved),
            "dss_pp": json_safe(dss_config),
            "measurement_time_s": float(live_time),
            "dwell_policy": "fixed_predeclared_live_time",
            "nominal_motion_speed_m_s": float(nominal_motion_speed_m_s),
            "rotation_overhead_s": float(rotation_overhead_s),
            "station_update": "joint_sequence",
            "pf_random_seed": int(pf_random_seed),
            "planning_random_seed": int(planning_root_seed),
            "planning_candidate_seed": int(planning_candidate_seed),
            "planning_dss_eig_seed": int(dss_eig_seed),
            "sim_backend": str(sim_backend),
            "environment_mode": str(normalized_environment_mode),
        },
        isotopes=isotopes,
    )
    measurement_log_config_hash = sha256_json(measurement_log_runtime_config)
    effective_pf_config_hash = sha256_json(effective_pf_runtime_config)
    replay_pf_config = {
        **json_safe(asdict(pf_conf)),
        "pure_pf_schema_version": 1,
        "estimator_profile": str(pf_conf.estimator_profile),
    }

    def _build_estimator() -> tuple[
        PurePFEstimator, NDArray[np.float64], int
    ]:
        """Create a fresh estimator and register the initial pose."""
        estimator_local = PurePFEstimator(
            isotopes=isotopes,
            surface_diagnostic_points=surface_diagnostic_points,
            shield_normals=normals,
            mu_by_isotope=mu_by_isotope,
            pf_config=pf_conf,
            obstacle_grid=pf_obstacle_grid,
            obstacle_height_m=_strict_json_number(
                runtime_config.get("obstacle_height_m", 2.0),
                name="obstacle_height_m",
                minimum=0.0,
            ),
            obstacle_mu_by_isotope=obstacle_mu_by_isotope,
            obstacle_buildup_coeff=pf_obstacle_buildup_coeff,
            detector_radius_m=pf_detector_radius_m,
            detector_aperture_radius_m=pf_detector_aperture_radius_m,
            detector_aperture_samples=pf_detector_aperture_samples,
            detector_aperture_sampling=pf_detector_aperture_sampling,
            source_extent_radius_m=pf_source_extent_radius_m,
            source_extent_samples=pf_source_extent_samples,
            line_mu_by_isotope=line_mu_by_isotope,
            config_hash=input_config_hash,
            resolved_config_hash=effective_pf_config_hash,
            random_seed=pf_random_seed,
            full_spectrum_generative_model=full_spectrum_model,
            measurement_log_schema_version=2,
        )
        pose_local = np.array(env.detector_position, dtype=float)
        estimator_local.add_measurement_pose(pose_local)
        pose_idx_local = len(estimator_local.poses) - 1
        return estimator_local, pose_local, pose_idx_local

    def _build_visualizer(*, include_truth: bool = False) -> object:
        """Create a PF visualizer, exposing truth only after termination."""
        visualizer_args = {
            "isotopes": isotopes,
            "world_bounds": (0, env.size_x, 0, env.size_y, 0, env.size_z),
            "true_sources": true_src if include_truth else {},
            "true_strengths": true_strengths if include_truth else {},
            "obstacle_grid": obstacle_grid,
        }
        if not live and _runtime_bool(
            runtime_config,
            "headless_visualizer_defer",
            True,
        ):
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
            if _runtime_bool(runtime_config, "cui_split_view_async", True)
            else CUISplitPFVisualizer
        )
        split_viz = split_cls(
            isotopes=isotopes,
            output_dir=cui_split_view_dir,
            world_bounds=(0, env.size_x, 0, env.size_y, 0, env.size_z),
            true_sources={},
            true_strengths={},
            obstacle_grid=obstacle_grid,
            max_particles_per_isotope=cui_split_max_particles,
        )
        serve_cui = _runtime_bool(
            runtime_config,
            "cui_split_view_serve",
            True,
        )
        split_url = None
        if serve_cui:
            split_url = _ensure_cui_view_server(
                split_viz.output_dir,
                host=_strict_json_string(
                    runtime_config.get("cui_split_view_host", "0.0.0.0"),
                    name="cui_split_view_host",
                ),
                port=_strict_json_integer(
                    runtime_config.get("cui_split_view_port", 8877),
                    name="cui_split_view_port",
                    minimum=1,
                    maximum=65535,
                ),
                public_host=(
                    None
                    if runtime_config.get("cui_split_view_public_host") is None
                    else _strict_json_string(
                        runtime_config.get("cui_split_view_public_host"),
                        name="cui_split_view_public_host",
                    )
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

    def _posterior_snapshot_estimates(
        posterior_snapshot: object,
        isotope_list: list[str],
    ) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Extract the exact source arrays stored in one canonical snapshot."""
        estimates_by_isotope = getattr(posterior_snapshot, "isotopes", None)
        if not isinstance(estimates_by_isotope, Mapping):
            raise RuntimeError(
                "Canonical PF posterior must expose isotope point estimates."
            )
        estimates: dict[
            str,
            tuple[NDArray[np.float64], NDArray[np.float64]],
        ] = {}
        for isotope in isotope_list:
            point_estimate = estimates_by_isotope.get(str(isotope))
            if point_estimate is None:
                raise RuntimeError(
                    f"Canonical PF posterior is missing isotope {isotope}."
                )
            modes = tuple(getattr(point_estimate, "modes", ()))
            positions = np.asarray(
                [mode.position_medoid_xyz for mode in modes],
                dtype=np.float64,
            ).reshape(-1, 3)
            strengths = np.asarray(
                [
                    mode.strength_representative_cps_1m
                    for mode in modes
                ],
                dtype=np.float64,
            ).reshape(-1)
            if (
                positions.shape[0] != strengths.shape[0]
                or np.any(~np.isfinite(positions))
                or np.any(~np.isfinite(strengths))
            ):
                raise RuntimeError(
                    "Canonical PF posterior contains invalid source arrays "
                    f"for isotope {isotope}."
                )
            estimates[str(isotope)] = (positions, strengths)
        return estimates

    def _serialize_estimate_stage(
        estimates_in: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    ) -> dict[str, list[dict[str, float | list[float]]]]:
        """Return JSON-serializable source estimates for one reporting stage."""
        serialized: dict[str, list[dict[str, float | list[float]]]] = {}
        for iso, estimate in sorted(estimates_in.items()):
            positions = np.asarray(estimate[0], dtype=float)
            strengths = np.asarray(estimate[1], dtype=float)
            if (
                positions.ndim != 2
                or positions.shape[1:] != (3,)
                or strengths.ndim != 1
                or positions.shape[0] != strengths.size
                or np.any(~np.isfinite(positions))
                or np.any(~np.isfinite(strengths))
                or np.any(strengths <= 0.0)
            ):
                raise RuntimeError(
                    "Canonical estimate serialization requires matching finite "
                    f"source arrays with positive strengths for isotope {iso}."
                )
            entries: list[dict[str, float | list[float]]] = []
            for pos, strength in zip(positions, strengths, strict=True):
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
        )
        grid_pos = np.asarray(surface_diagnostic_points, dtype=float)
        preview_frame = PFFrame(
            step_index=-1,
            time=0.0,
            robot_position=preview_pose,
            robot_orientation=None,
            RFe=np.eye(3),
            RPb=np.eye(3),
            duration=0.0,
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

    estimator: PurePFEstimator
    current_pose: NDArray[np.float64]
    current_pose_idx: int
    measurement_log_writer: MeasurementLogStreamWriter | None = None
    resume_controller_state: _LiveResumeControllerState | None = None
    resume_controller_checkpoint: _LiveControllerCheckpoint | None = None
    resume_replay_wall_s = 0.0
    planning_candidate_checkpoint_parameters = (
        _planning_candidate_checkpoint_parameters(
            pose_candidates=int(pose_candidates),
            pose_min_dist=float(pose_min_dist),
            bounds_xyz=(bounds_lo, bounds_hi),
            detector_heights_m=detector_height_candidates,
        )
    )
    if measurement_log_target is not None:
        environment_payload: dict[str, Any] = {
            "environment_model_id": _strict_json_string(
                runtime_config.get(
                    "environment_model_id",
                    f"{normalized_environment_mode}_environment.v1",
                ),
                name="environment_model_id",
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
        execution_commit = simulation_repository_commit(
            SIMULATION_RUNTIME_ROOT
        )
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
                repository_root=SIMULATION_RUNTIME_ROOT,
                prefix_commit=commit,
                execution_commit=execution_commit,
                additional_compatible_code_paths=resume_compatible_code_paths,
                compatibility_basis=resume_compatibility_basis,
            )
        forward_manifest = build_forward_model_manifest(
            runtime_config=measurement_log_runtime_config,
            environment=environment_payload,
            obstacle_layout_path=measurement_log_obstacle_layout_path,
            isotopes=isotopes,
            repository_commit=commit,
            resolved_config_sha256=measurement_log_config_hash,
            run_root=measurement_log_target,
            repository_root=SIMULATION_RUNTIME_ROOT,
        )
        log_run_id = _strict_json_string(
            runtime_config.get(
                "measurement_log_run_id",
                measurement_log_target.name,
            ),
            name="measurement_log_run_id",
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
                    pf_config=replay_pf_config,
                    profile=str(pf_conf.estimator_profile),
                    seed=int(pf_random_seed),
                    config_hash=input_config_hash,
                )

                def _report_replayed_station(
                    replay_estimator: PurePFEstimator,
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
                dss_eig_rng=dss_eig_rng,
                expected_planning_candidate_parameters=(
                    planning_candidate_checkpoint_parameters
                ),
            )
            if resume_controller_checkpoint is None:
                raise RuntimeError(
                    "Station-boundary resume requires an embedded live "
                    "controller checkpoint."
                )
            print(
                "Station-boundary resume restored "
                f"records={resume_controller_state.step_counter} "
                f"stations={resume_controller_state.pose_counter + 1} "
                f"next_step={resume_controller_state.step_counter} "
                "controller_state=checkpoint "
                f"replay_wall_s={resume_replay_wall_s:.3f}.",
                flush=True,
            )
    else:
        estimator, current_pose, current_pose_idx = _build_estimator()
    runtime_atlas_sha256 = surface_chart_geometry_sha256(
        runtime_surface_chart_geometry
    )
    estimator_atlas_sha256 = estimator.initialize_joint_particle_filters()
    if estimator_atlas_sha256 != runtime_atlas_sha256:
        raise RuntimeError(
            "Truth binding and joint PF use different continuous surface atlas "
            "contracts."
        )
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
    last_observation_summary: dict[str, float] | None = (
        None
        if resume_controller_state is None
        else dict(resume_controller_state.last_observation_summary)
    )
    representative_spectrum: np.ndarray | None = (
        None
        if resume_controller_state is None
        else resume_controller_state.representative_spectrum.copy()
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
        gpu_runtime_enabled = bool(estimator._gpu_enabled())
    gpu_memory_baseline = start_gpu_memory_tracking(
        str(pf_conf.gpu_device) if gpu_runtime_enabled else None
    )
    run_wall_start = time.perf_counter()

    def _forced_baseline_program_for_planned_station(
        *,
        label: str,
    ) -> tuple[tuple[int, ...] | None, DSSPPConfig, BaselineShieldProgram | None]:
        """Return a deterministic baseline shield program for the next station."""
        dss_selection_config = dss_config
        if baseline_shield_policy is None:
            return None, dss_selection_config, None
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

    gpu_status = "disabled"
    if bool(estimator.pf_config.use_gpu):
        gpu_status = "enabled" if estimator._gpu_enabled() else "disabled"
    dwell_cap_label = f"{live_time:.3f}"
    dwell_step_label = f"{live_time:.1f}"
    if save_outputs:
        IG_DIR.mkdir(parents=True, exist_ok=True)
    print(
        "Spectrum config: "
        "model=geometry_conditioned_joint_full_spectrum "
        f"bins={full_spectrum_model.energy_axis_keV.size} "
        "bin_width_keV="
        f"{float(full_spectrum_model.energy_axis_keV[1] - full_spectrum_model.energy_axis_keV[0]):.6g} "
        f"live_time_s={dwell_cap_label} "
        f"dead_time_tau_s={full_spectrum_model.dead_time_tau_s:.12g} "
        f"background_rate_cps={full_spectrum_model.background_rate_cps:.6g} "
        f"contract={full_spectrum_model.contract_hash_sha256}"
    )
    print(f"Pure PF compute backend: {pf_compute_backend}")
    print(
        "Dwell control: "
        "policy=fixed_predeclared_live_time "
        f"live_time_s={dwell_cap_label}"
    )
    print(
        "Output rendering: "
        f"spectrum_plot_save_every={int(spectrum_plot_save_every)}; "
        f"pf_plot_save_every={int(pf_plot_save_every)}; "
        "headless_visualizer_defer="
        f"{_runtime_bool(runtime_config, 'headless_visualizer_defer', True)}; "
        "estimates use the current PF posterior projection"
    )
    print(
        "Path planner: "
        f"mode={path_planner_resolved} "
        f"dss_program_len={int(dss_config.program_length)} "
        f"rotation_w={float(dss_config.lambda_rotation):.3f} "
        f"coverage_w={float(dss_config.lambda_coverage):.3f} "
        f"bearing_w={float(dss_config.lambda_bearing_diversity):.3f} "
        f"frontier_w={float(dss_config.lambda_frontier):.3f} "
        f"local_orbit_w={float(dss_config.lambda_local_orbit):.3f} "
        f"turn_w={float(dss_config.lambda_turn_smoothness):.3f} "
        f"revisit_w={float(dss_config.eta_revisit):.3f} "
        f"min_station_sep={float(dss_config.min_station_separation_m):.2f}m"
    )
    print(
        "PF observation model: one joint full-spectrum generative likelihood "
        "per station; no projected-count, contrast, or view-ratio term"
    )
    print(
        "Python CPU workers: "
        f"general={python_worker_count_resolved}; "
        "DSS spectrum EIG uses vectorized action batches"
    )
    print(
        "Surface-atlas diagnostics: "
        "support=PF physical continuous atlas "
        f"chart_max_edge={pf_conf.structural_rj_surface_chart_max_edge_m:.3f}m "
        f"charts={int(surface_atlas_diagnostics['chart_count'])} "
        f"points={surface_diagnostic_points.shape[0]}"
    )
    print("PF source-position support: continuous physical surface charts")
    print(
        "Init support: "
        "area-uniform continuous environment surface "
        f"(init_num_sources={pf_conf.init_num_sources})"
    )
    print(
        "PF init prior: "
        f"init_num_sources={pf_conf.init_num_sources}, "
        "strength_uniform_cps_1m="
        f"[{pf_conf.strength_prior_min_cps_1m:.1f}, "
        f"{pf_conf.strength_prior_max_cps_1m:.1f}], "
        f"max_sources={pf_conf.max_sources}"
    )
    print(
        "PF SMC settings: "
        f"target_ess_ratio={pf_conf.target_ess_ratio:.3f}, "
        f"max_temper_steps={pf_conf.max_temper_steps}, "
        f"min_delta_beta={pf_conf.min_delta_beta:.3g}"
    )
    print(
        "PF convergence gates: "
        f"surface_radius95<="
        f"{pf_conf.credible_surface_radius_threshold_m:.3f}m, "
        f"current_ess_ratio>={pf_conf.converge_min_ess_ratio:.3f}, "
        "cardinality_probability>="
        f"{pf_conf.converge_cardinality_min_probability:.3f}, "
        "max_cardinality_boundary_mass<="
        f"{pf_conf.converge_max_cardinality_boundary_mass:.3f}, "
        f"innovation_confidence={pf_conf.converge_innovation_confidence:.3f}"
    )
    print(
        "Planning rollout settings: "
        f"eig_samples={estimator.pf_config.planning_eig_samples}, "
        f"particles={dss_config.planning_particles}, "
        f"method={dss_config.planning_method}"
    )
    print(
        "PF continuous RJ position proposal: "
        "state-independent area-prior + matched-filter mixture "
        f"(prior_weight="
        f"{estimator.pf_config.structural_rj_position_proposal_prior_weight:.3f}, "
        "chart_uv=continuous_uniform, target_response=direct_continuous_xyz)"
    )
    print(
        "GPU acceleration: "
        f"{gpu_status} (device={estimator.pf_config.gpu_device}, dtype={estimator.pf_config.gpu_dtype})"
    )
    print(
        "PF parallelism: "
        "one aligned joint-isotope SMC target; particle/source/view/line "
        "evaluation is vectorized on the configured CPU/GPU backend"
    )
    print(f"Simulation backend: {sim_backend}")
    print(
        "Mission timing model: "
        f"robot_speed={float(nominal_motion_speed_m_s):.3f}m/s "
        f"shield_overhead={float(rotation_overhead_s):.3f}s/measurement "
        "mission_time=travel+shield+live"
    )
    print(
        "Adaptive mission stop: "
        f"enabled={adaptive_mission_stop} "
        f"min_convergence_poses={mission_stop_min_convergence_poses} "
        f"max_poses={max_poses if max_poses is not None else 'none'} "
        "require_pf_cardinality_ready="
        f"{mission_stop_require_pf_cardinality_ready}"
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
            "source_sampling": source_sampling_metadata,
            "sources": [_source_runtime_payload(source) for source in sources],
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
            "transport_line_compton_mu_by_isotope": {}
            if obstacle_grid is None
            else {
                str(isotope): [
                    [float(value) for value in row] for row in rows
                ]
                for isotope, rows in (
                    obstacle_grid
                    .transport_line_compton_mu_by_isotope
                    .items()
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
            "source_count": len(sources),
            "sources": [_source_runtime_payload(source) for source in sources],
            "isotopes": isotopes,
            "variable_cardinality": variable_cardinality,
            "pose_candidates": int(pose_candidates),
            "pose_min_dist_m": float(pose_min_dist),
            "detector_height_sampling_mode": "continuous",
            "detector_height_min_m": float(
                detector_height_config.minimum_mast_height_m
            ),
            "detector_height_max_m": float(
                detector_height_config.maximum_mast_height_m
            ),
            "robot_ground_z_m": float(robot_ground_z_m),
            "measurement_workspace": measurement_workspace_diagnostics,
            "surface_diagnostic_points": int(
                surface_diagnostic_points.shape[0]
            ),
            "source_prior_domain": "continuous_environment_surface",
            "pf_num_particles": int(pf_conf.num_particles),
            "pf_max_sources": (
                None if pf_conf.max_sources is None else int(pf_conf.max_sources)
            ),
            "python_worker_count": int(python_worker_count_resolved),
            "pf_update_schedule": (
                "aligned_joint_isotope_smc_vectorized_particle_source_view_line"
            ),
            "robot_speed_m_s": float(nominal_motion_speed_m_s),
            "rotation_overhead_s": float(rotation_overhead_s),
            "measurement_time_s": float(live_time),
            "measurement_time_cap_s": float(live_time),
            "dwell_policy": "fixed_predeclared_live_time",
            "station_update": "joint_sequence",
            "strict_planned_shield_program": bool(strict_planned_shield_program),
            "baseline_shield_policy": baseline_shield_policy,
            "baseline_path_policy": baseline_path_policy,
            "path_planner": path_planner_resolved,
            "dss_program_length": int(dss_config.program_length),
            "dss_rotation_weight": float(dss_config.lambda_rotation),
            "dss_coverage_weight": float(dss_config.lambda_coverage),
            "dss_local_orbit_weight": float(dss_config.lambda_local_orbit),
            "dss_revisit_penalty_weight": float(dss_config.eta_revisit),
            "dss_min_station_separation_m": float(dss_config.min_station_separation_m),
        }
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
    resume_station_boundary_pending = resume_controller_state is not None

    def _select_joint_dss_program_for_fixed_station(
        pose_xyz: NDArray[np.float64],
    ) -> tuple[int, ...]:
        """Select the current-station program with the shared joint DSS score."""
        fixed_station_config = replace(
            dss_config,
            augment_candidates=False,
            min_station_separation_m=0.0,
            forced_program_pair_ids=None,
        )
        result = select_dss_pp_next_station(
            estimator=estimator,
            rng=dss_eig_rng,
            candidate_poses_xyz=np.asarray([pose_xyz], dtype=float),
            current_pose_xyz=np.asarray(pose_xyz, dtype=float),
            current_pair_id=current_shield_pair_id,
            visited_poses_xyz=None,
            map_api=planning_map,
            bounds_xyz=(bounds_lo, bounds_hi),
            continuous_height_bounds_m=continuous_height_bounds_for_dss,
            config=fixed_station_config,
        )
        return tuple(int(pair_id) for pair_id in result.shield_program.pair_ids)

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
            active_shield_program = pending_shield_program
            pending_shield_program = None
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
                    )
                    used_name = str(baseline_program.name)
                    print(
                        "Executing baseline shield program: "
                        f"{used_name} pairs={list(active_shield_program)}"
                    )
            if not active_shield_program and not resume_station_boundary:
                active_shield_program = _select_joint_dss_program_for_fixed_station(
                    np.asarray(pose, dtype=float)
                )
                rotation_limit = _resolve_rotation_limit_for_active_program(
                    base_rotation_limit=rotation_limit,
                    active_shield_program=active_shield_program,
                    strict_planned_shield_program=True,
                    baseline_shield_policy=None,
                )
                print(
                    "Executing joint DSS-PP shield program at initial station: "
                    f"{list(active_shield_program)}"
                )
            force_active_shield_program = bool(active_shield_program)
            joint_update_records: list[tuple[object, ...]] = []
            executed_pair_ids_this_pose: list[int] = (
                list(resume_controller_state.last_station_pair_ids)
                if resume_station_boundary and resume_controller_state is not None
                else []
            )
            rotation_count = rotation_limit if resume_station_boundary else 0
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
                if (
                    not force_active_shield_program
                    or not active_shield_program
                    or rotation_count >= len(active_shield_program)
                ):
                    raise RuntimeError(
                        "Every pure-PF station must execute a shield program "
                        "selected by joint DSS-PP or an explicit baseline policy."
                    )
                best_pair_idx = int(active_shield_program[rotation_count])
                using_planned_pair = True
                ig_elapsed = 0.0
                fe_idx = best_pair_idx // num_orients
                pb_idx = best_pair_idx % num_orients
                RFe_sel = rot_mats[fe_idx]
                RPb_sel = rot_mats[pb_idx]
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
                (
                    observation,
                    actual_live_time_s,
                    spectrum,
                    dwell_ready_reason,
                    dwell_chunks,
                ) = _acquire_spectrum_observation(
                    simulation_runtime=simulation_runtime,
                    full_spectrum_model=full_spectrum_model,
                    step_id=step_counter,
                    pose_xyz=pose,
                    fe_idx=fe_idx,
                    pb_idx=pb_idx,
                    live_time_s=live_time,
                    travel_time_s=step_motion_time_s,
                    shield_actuation_time_s=step_rotation_time_s,
                    require_native_contract=(
                        str(sim_backend).strip().lower() != "analytic"
                    ),
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
                last_spectrum = spectrum.copy()
                last_observation_summary = {
                    "raw_spectrum_total": float(np.sum(spectrum))
                }
                last_candidates = set()
                spectrum_total_counts = float(np.sum(spectrum))
                if spectrum_total_counts > representative_total_counts:
                    representative_total_counts = spectrum_total_counts
                    representative_spectrum = spectrum.copy()
                    representative_candidates = set(last_candidates)
                    representative_step_index = int(step_counter)
                spectrum_notify_every = max(1, int(notify_spectrum_every))
                if notify_spectrum and step_counter % spectrum_notify_every == 0:
                    notifier.notify_spectrum(
                        step_counter,
                        _build_spectrum_notification_payload(
                            energy_axis_keV=full_spectrum_model.energy_axis_keV,
                            spectrum=spectrum,
                            step_index=step_counter,
                            pose_xyz=np.asarray(
                                observation.detector_pose_xyz, dtype=float
                            ),
                            fe_index=fe_idx,
                            pb_index=pb_idx,
                            live_time_s=actual_live_time_s,
                            full_spectrum_contract_hash_sha256=(
                                full_spectrum_model.contract_hash_sha256
                            ),
                            max_bins=int(notify_spectrum_max_bins),
                        ),
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
                if measurement_log_writer is not None:
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
                            spectrum_counts=np.asarray(
                                spectrum,
                                dtype=np.int64,
                            ),
                            metadata={
                                "backend": str(
                                    observation.metadata.get("backend", sim_backend)
                                ),
                                "dwell_ready_reason": str(dwell_ready_reason),
                                "dwell_chunks": int(dwell_chunks),
                                FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY: (
                                    full_spectrum_model.contract_hash_sha256
                                ),
                                **_measurement_transport_provenance(
                                    observation.metadata
                                ),
                            },
                        )
                    )
                joint_record: tuple[object, ...] = (
                    np.ascontiguousarray(spectrum, dtype=np.int64),
                    int(fe_idx),
                    int(pb_idx),
                    float(actual_live_time_s),
                )
                joint_update_records.append(joint_record)
                measurement_live_times_s.append(float(actual_live_time_s))
                elapsed += actual_live_time_s
                viz_elapsed = 0.0
                viz_start = time.perf_counter()
                frame = build_frame_from_pf(
                    estimator,
                    step_index=step_counter,
                    time_sec=elapsed,
                    detector_position=pose_for_pf,
                    live_time_s=actual_live_time_s,
                    RFe=RFe_sel,
                    RPb=RPb_sel,
                    spectrum_energy_keV=(
                        full_spectrum_model.energy_axis_keV.copy()
                    ),
                    spectrum_counts=spectrum.copy(),
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
                    f"fe_idx={fe_idx} pb_idx={pb_idx} "
                    f"travel_distance_m={step_motion_distance_m:.3f} "
                    f"travel_time_s={step_motion_time_s:.1f} "
                    f"shield_time_s={step_rotation_time_s:.1f} "
                    f"live_time_s={actual_live_time_s:.1f}/{dwell_step_label} "
                    f"dwell_chunks={dwell_chunks} "
                    f"dwell_reason={dwell_ready_reason} "
                    f"mission_time_s={elapsed:.1f} "
                    f"raw_spectrum_total={int(np.sum(spectrum))}"
                )
                if live:
                    plt.pause(0.05)
                viz_elapsed += time.perf_counter() - viz_start
                total_viz_wall_s += viz_elapsed
                _log_pf_diagnostics(estimator, step_counter)
                print(
                    f"[timing step {step_counter}] ig={ig_elapsed:.3f}s "
                    "pf=station_pending "
                    f"viz={viz_elapsed:.3f}s "
                    f"travel={step_motion_time_s:.1f}s "
                    f"shield={step_rotation_time_s:.1f}s "
                    f"live={actual_live_time_s:.1f}s"
                )
                step_counter += 1
                rotation_count += 1
                remaining_orientations.discard(best_pair_idx)
                current_shield_pair_id = int(best_pair_idx)
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
                        full_spectrum_model.energy_axis_keV,
                        last_spectrum,
                        spectrum_path,
                        highlight_isotopes=highlight,
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
                                dss_eig_rng=dss_eig_rng,
                                planning_candidate_parameters=(
                                    planning_candidate_checkpoint_parameters
                                ),
                                max_poses=max_poses,
                            )
                        )
                    },
                )
            if joint_update_records:
                pf_start = time.perf_counter()
                estimator.update_spectrum_station(
                    joint_update_records,
                    pose_idx=current_pose_idx,
                    generative_contract_hash_sha256=(
                        full_spectrum_model.contract_hash_sha256
                    ),
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
                    "joint_update_wall="
                    f"{float(getattr(estimator, 'last_pair_sequence_update_wall_s', 0.0)):.3f}s "
                    "conditional_rj_wall="
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
                _log_surface_atlas_observability_diagnostics(
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
                        step_index=joint_step_index,
                        elapsed_s=elapsed,
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
            visited_poses.append(pose.copy())
            pose_counter += 1
            if adaptive_mission_stop:
                stop_reason = _adaptive_mission_stop_reason(
                    estimator,
                    visited_poses_xyz=visited_poses,
                    min_poses=mission_stop_min_convergence_poses,
                    require_pf_cardinality_ready=(
                        mission_stop_require_pf_cardinality_ready
                    ),
                )
                if stop_reason is not None:
                    print(f"Adaptive mission stop: {stop_reason}.")
                    break
            if max_poses is not None and pose_counter >= max_poses:
                pf_cardinality_ready, pf_cardinality_reason = (
                    _source_cardinality_dwell_status(
                        estimator,
                        refresh_estimates=False,
                    )
                )
                pf_cardinality_unresolved = not pf_cardinality_ready
                max_pose_stop_unresolved = bool(pf_cardinality_unresolved)
                max_pose_stop_diagnostics = {
                    "max_poses": int(max_poses),
                    "pf_cardinality_unresolved": bool(pf_cardinality_unresolved),
                    "pf_cardinality_reason": pf_cardinality_reason,
                }
                print(f"Reached max poses ({max_poses}); stopping exploration.")
                break
            visited_arr = np.vstack(visited_poses) if visited_poses else None
            print("Generating candidate poses for next measurement point...")
            candidates, candidate_generation_diagnostics = (
                _generate_planning_candidates(
                    current_pose_xyz=pose,
                    map_api=planning_map,
                    n_candidates=pose_candidates,
                    min_dist_from_visited=pose_min_dist,
                    visited_poses_xyz=visited_arr,
                    bounds_xyz=(bounds_lo, bounds_hi),
                    detector_heights_m=detector_height_candidates,
                    rng=planning_candidate_rng,
                )
            )
            print(
                f"Generated {len(candidates)} reachable 3-D candidate poses. "
                "Global Sobol pool retained physical separation. "
                "Computing best next pose..."
            )
            planned_program_for_next: tuple[int, ...] | None = None
            dss_diagnostics: dict[str, Any] | None = None
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
                    dss_start = time.perf_counter()
                    dss_result = select_dss_pp_next_station(
                        estimator=estimator,
                        rng=dss_eig_rng,
                        candidate_poses_xyz=np.asarray([next_pose], dtype=float),
                        current_pose_xyz=pose,
                        current_pair_id=current_shield_pair_id,
                        visited_poses_xyz=visited_arr,
                        map_api=planning_map,
                        bounds_xyz=(bounds_lo, bounds_hi),
                        continuous_height_bounds_m=(continuous_height_bounds_for_dss),
                        config=dss_selection_config,
                    )
                    dss_elapsed = time.perf_counter() - dss_start
                    total_path_planning_wall_s += float(dss_elapsed)
                    path_planning_wall_samples_s.append(float(dss_elapsed))
                    planned_program_for_next = tuple(
                        int(pair_id) for pair_id in dss_result.shield_program.pair_ids
                    )
                    dss_diagnostics = dict(dss_result.diagnostics)
                    print(
                        "DSS-PP fixed-station shield program: "
                        f"program={dss_result.shield_program.name} "
                        f"pairs={list(planned_program_for_next)} "
                        f"score={float(dss_result.score):.6g} "
                        f"eig={float(dss_result.sequence[0].information_gain):.6g} "
                        f"compute={dss_elapsed:.3f}s"
                    )
                    _log_dss_ranked_node_diagnostics(
                        dss_diagnostics,
                        label=f"pose_{current_pose_idx}_baseline_path_fixed_station",
                    )
                    _log_dss_component_leader_diagnostics(
                        dss_diagnostics,
                        label=f"pose_{current_pose_idx}_baseline_path_fixed_station",
                    )
            else:
                dss_selection_config = dss_config
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
                    rng=dss_eig_rng,
                    candidate_poses_xyz=candidates,
                    current_pose_xyz=pose,
                    current_pair_id=current_shield_pair_id,
                    visited_poses_xyz=visited_arr,
                    map_api=planning_map,
                    bounds_xyz=(bounds_lo, bounds_hi),
                    continuous_height_bounds_m=continuous_height_bounds_for_dss,
                    config=dss_selection_config,
                )
                dss_elapsed = time.perf_counter() - dss_start
                total_path_planning_wall_s += float(dss_elapsed)
                path_planning_wall_samples_s.append(float(dss_elapsed))
                next_pose = dss_result.next_pose
                planned_program_for_next = tuple(
                    int(pair_id) for pair_id in dss_result.shield_program.pair_ids
                )
                dss_diagnostics = dict(dss_result.diagnostics)
                print(
                    "DSS-PP selected next station: "
                    f"pose={next_pose.tolist()} "
                    f"program={dss_result.shield_program.name} "
                    f"pairs={list(planned_program_for_next)} "
                    f"score={float(dss_result.score):.6g} "
                    f"eig={float(dss_result.sequence[0].information_gain):.6g} "
                    f"coverage_gain={float(dss_result.sequence[0].coverage_gain):.6g} "
                    f"revisit_penalty={float(dss_result.sequence[0].revisit_penalty):.6g} "
                    f"bearing_gain={float(dss_result.sequence[0].bearing_diversity_gain):.6g} "
                    f"frontier_gain={float(dss_result.sequence[0].frontier_gain):.6g} "
                    f"local_orbit={float(dss_result.sequence[0].local_orbit_gain):.6g} "
                    f"elevation_cond={float(dss_result.sequence[0].elevation_condition_gain):.6g} "
                    f"turn_penalty={float(dss_result.sequence[0].turn_penalty):.6g} "
                    f"planner_mode={dss_result.diagnostics.get('planner_mode', 'balanced')} "
                    f"compute={dss_elapsed:.3f}s"
                )
                _log_dss_ranked_node_diagnostics(
                    dss_diagnostics,
                    label=f"pose_{current_pose_idx}_next",
                )
                _log_dss_component_leader_diagnostics(
                    dss_diagnostics,
                    label=f"pose_{current_pose_idx}_next",
                )
            if dss_diagnostics is None:
                dss_diagnostics = {}
            dss_diagnostics["candidate_generation"] = dict(
                candidate_generation_diagnostics
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
                "last_observation_summary": last_observation_summary,
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

    # Resolve final artifact paths before publishing the truth-free posterior.
    result_paths: dict[str, str] = {}
    summary_out_path: Path | None = None
    final_estimates_for_run: dict[
        str,
        tuple[NDArray[np.float64], NDArray[np.float64]],
    ]
    final_estimate_stages_for_run: dict[
        str,
        dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    ]
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

    posterior_getter = getattr(estimator, "posterior_snapshot", None)
    if not callable(posterior_getter):
        raise RuntimeError(
            "A completed pure-PF run requires a canonical posterior snapshot."
        )
    posterior_projection_started_at = time.perf_counter()
    canonical_posterior_snapshot = posterior_getter()
    canonical_posterior_payload = canonical_posterior_snapshot.to_dict()
    final_estimates_for_run = _posterior_snapshot_estimates(
        canonical_posterior_snapshot,
        isotopes,
    )
    final_estimate_stages_for_run = {
        "pf_posterior_projection": final_estimates_for_run,
    }
    _validate_surface_constrained_estimates(
        final_estimates_for_run,
        env,
        obstacle_grid,
        obstacle_height_m=_strict_json_number(
            runtime_config.get("obstacle_height_m", 2.0),
            name="obstacle_height_m",
            minimum=0.0,
        ),
        tolerance_m=_strict_json_number(
            runtime_config.get(
                "posterior_surface_tolerance_m",
                1.0e-5,
            ),
            name="posterior_surface_tolerance_m",
            minimum=0.0,
        ),
        surface_prior_active=True,
        estimator=estimator,
    )
    final_posterior_projection_time_s += float(
        time.perf_counter() - posterior_projection_started_at
    )
    if save_outputs:
        _atomic_write_json(
            pf_posterior_out_path,
            canonical_posterior_payload,
        )
        if last_frame is not None:
            last_frame.step_index = max(0, int(step_counter) - 1)
            last_frame.time = float(elapsed)
            last_frame.estimated_sources = {
                iso: pos
                for iso, (pos, _) in final_estimates_for_run.items()
            }
            last_frame.estimated_strengths = {
                iso: strength
                for iso, (_, strength) in final_estimates_for_run.items()
            }
            registered_poses = np.asarray(estimator.poses, dtype=np.float64)
            if registered_poses.ndim == 2 and registered_poses.shape[1] == 3:
                last_frame.path_waypoints_xyz = registered_poses.copy()
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
        "measurement_time_cap_s": float(live_time),
        "dwell_policy": "fixed_predeclared_live_time",
        "detector_height_sampling_mode": "continuous",
        "detector_height_min_m": float(detector_height_config.minimum_mast_height_m),
        "detector_height_max_m": float(detector_height_config.maximum_mast_height_m),
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
        "dss_program_length": int(dss_config.program_length),
        "dss_rotation_weight": float(dss_config.lambda_rotation),
        "mission_stop_require_pf_cardinality_ready": bool(
            mission_stop_require_pf_cardinality_ready
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
    surface_obstacle_height_m = _strict_json_number(
        runtime_config.get("obstacle_height_m", 2.0),
        name="obstacle_height_m",
        minimum=0.0,
    )
    posterior_surface_tolerance_m = _strict_json_number(
        runtime_config.get("posterior_surface_tolerance_m", 1.0e-5),
        name="posterior_surface_tolerance_m",
        minimum=0.0,
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
    estimates = final_estimates_for_run
    final_estimate_stages = final_estimate_stages_for_run
    est_by_iso: dict[str, list[dict[str, Any]]] = {}
    for iso, estimate in estimates.items():
        positions = np.asarray(estimate[0], dtype=float)
        strengths = np.asarray(estimate[1], dtype=float)
        surface_kinds = estimator.structural_surface_kinds(
            iso,
            positions.reshape(-1, 3),
            strict=True,
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
            else _strict_json_number(
                runtime_config["posterior_uncertainty_match_radius_m"],
                name="posterior_uncertainty_match_radius_m",
                minimum=0.0,
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
    diagnostic_surface_payload = _surface_count_payload(
        surface_diagnostic_points,
        env,
        obstacle_grid,
        obstacle_height_m=surface_obstacle_height_m,
        tolerance_m=posterior_surface_tolerance_m,
    )
    diagnostic_surface_payload["total_diagnostic_points"] = (
        diagnostic_surface_payload.pop("total_sources")
    )
    source_surface_diagnostics = {
        "support_domain": "environment_surface",
        "surface_annotation_tolerance_m": float(posterior_surface_tolerance_m),
        "surface_atlas_diagnostics": diagnostic_surface_payload,
        "estimated_sources": _estimate_surface_diagnostics(
            estimates,
            env,
            obstacle_grid,
            obstacle_height_m=surface_obstacle_height_m,
            tolerance_m=posterior_surface_tolerance_m,
            estimator=estimator,
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
    final_surface_observability = estimator.surface_atlas_observability_diagnostics(
        window=None,
        max_candidates=_strict_json_integer(
            runtime_config.get("final_surface_observability_candidates", 1024),
            name="final_surface_observability_candidates",
            minimum=1,
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
    full_spectrum_model_diagnostics = _full_spectrum_model_diagnostics(
        full_spectrum_model,
        obstacle_attenuation_active=bool(pf_obstacle_attenuation_enabled),
    )
    metrics = compute_metrics(
        gt_by_iso,
        est_by_iso,
        match_radius_m=eval_match_radius_m,
        surface_atlas=estimator.continuous_surface_atlas(),
        close_pair_distance_m=_strict_json_number(
            runtime_config.get("evaluation_close_pair_distance_m", 2.0),
            name="evaluation_close_pair_distance_m",
            minimum=0.0,
        ),
        close_pair_min_estimated_separation_m=_strict_json_number(
            runtime_config.get(
                "evaluation_close_pair_min_estimated_separation_m",
                0.5,
            ),
            name="evaluation_close_pair_min_estimated_separation_m",
            minimum=0.0,
        ),
        uncertainty_by_iso=estimated_source_uncertainty,
    )
    print_metrics_report(metrics)
    online_metric_summary = _online_estimate_metric_summary(
        estimator.history_estimates,
        gt_by_iso,
        match_radius_m=float(eval_match_radius_m),
        surface_atlas=estimator.continuous_surface_atlas(),
    )
    cluster_stability = summarize_cluster_stability(
        estimator.history_estimates,
        final_window=_strict_json_integer(
            runtime_config.get("evaluation_cluster_stability_window", 5),
            name="evaluation_cluster_stability_window",
            minimum=1,
        ),
        match_gate_m=_strict_json_number(
            runtime_config.get("evaluation_cluster_match_gate_m", 0.5),
            name="evaluation_cluster_match_gate_m",
            minimum=0.0,
        ),
    )
    final_pf_cardinality_status = _final_pf_cardinality_status(estimator)
    final_posterior_convergence = estimator.posterior_convergence_diagnostics()
    for isotope, diagnostics in sorted(
        final_posterior_convergence["isotopes"].items()
    ):
        innovation = diagnostics.get("innovation", {})
        mark_tail = innovation.get("conditional_mark_tail_probability")
        innovation_summary = (
            f"total_z={float(innovation['renewal_total_max_abs_z']):.3f},"
            f"mark_p={'unavailable' if mark_tail is None else f'{float(mark_tail):.3g}'}"
            if bool(innovation.get("available", False))
            else "unavailable"
        )
        surface_radius = diagnostics.get(
            "maximum_credible_surface_radius_95_m"
        )
        surface_radius_summary = (
            "disconnected"
            if surface_radius is None
            else f"{float(surface_radius):.3f}m"
        )
        print(
            f"Final convergence gates[{isotope}]: "
            f"ready={bool(diagnostics.get('ready', False))} "
            f"ess_ratio={float(diagnostics.get('current_ess_ratio', 0.0)):.3f} "
            "map_k_probability="
            f"{float(diagnostics.get('map_cardinality_probability', 0.0)):.3f} "
            "max_k_mass="
            f"{float(diagnostics.get('maximum_cardinality_boundary_mass', 0.0)):.3f} "
            f"surface_path_radius95={surface_radius_summary} "
            f"innovation_q={innovation_summary}"
        )
    final_payload = {
        **_pure_pf_summary_provenance(
            estimator,
            posterior_payload=canonical_posterior_payload,
        ),
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
            "pf_structural_evidence": final_pf_cardinality_status,
            "pf_convergence": final_posterior_convergence,
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
        "estimated_sources": est_by_iso,
        "estimated_source_uncertainty": estimated_source_uncertainty,
        "estimated_source_uncertainty_reference": {
            "posterior_reference": "current_final_pf_particle_cloud",
            "reported_estimate_reference": "current_pf_posterior_projection",
            "reference_consistent": True,
        },
        "final_particle_cloud": _final_particle_cloud_payload(estimator),
        "max_pose_stop_unresolved": bool(max_pose_stop_unresolved),
        "max_pose_stop_diagnostics": max_pose_stop_diagnostics,
        "source_sampling": source_sampling_metadata,
        "source_surface_diagnostics": source_surface_diagnostics,
        "pf_obstacle_diagnostics": pf_obstacle_diagnostics,
        "full_spectrum_model_diagnostics": full_spectrum_model_diagnostics,
        "surface_atlas_observability_diagnostics": final_surface_observability,
        "posterior_convergence_diagnostics": final_posterior_convergence,
        "final_estimate_diagnostics": {
            "stages": {
                stage: _serialize_estimate_stage(stage_estimates)
                for stage, stage_estimates in final_estimate_stages.items()
            },
            "pf_cardinality_status": final_pf_cardinality_status,
        },
        "ground_truth_sources": gt_by_iso,
        "last_observation_summary": last_observation_summary,
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
    optional_plot_failures: tuple[dict[str, str], ...] = ()
    if save_outputs and summary_out_path is not None:
        _atomic_write_json(
            summary_out_path,
            final_payload,
        )

        def _render_final_pf_views() -> None:
            """Render final live, CUI, and truth-annotated PF views."""
            if last_frame is not None:
                viz.update(last_frame)
                if cui_split_viz is not None:
                    cui_split_viz.update(last_frame)
            post_run_viz = _build_visualizer(include_truth=True)
            if last_frame is not None:
                post_run_viz.update(last_frame)
            post_run_viz.save_final(pf_out_path.as_posix())
            if last_frame is not None:
                post_run_viz.save_estimates_only(
                    estimates_out_path.as_posix()
                )

        def _render_representative_spectrum() -> None:
            """Render the selected representative raw spectrum."""
            if representative_spectrum is None:
                return
            title = "Representative measurement spectrum"
            if representative_step_index is not None:
                title = f"{title} (step {representative_step_index})"
            _save_spectrum_plot(
                full_spectrum_model.energy_axis_keV,
                representative_spectrum,
                spectrum_out_path,
                highlight_isotopes=set(representative_candidates),
                title=title,
            )

        def _render_last_spectrum() -> None:
            """Render the final raw spectrum."""
            if last_spectrum is None:
                return
            _save_spectrum_plot(
                full_spectrum_model.energy_axis_keV,
                last_spectrum,
                last_spectrum_out_path,
                highlight_isotopes=set(last_candidates),
                title="Last measurement spectrum",
            )

        required_artifacts = [
            pf_posterior_out_path,
            summary_out_path,
        ]
        if published_measurement_log is not None:
            required_artifacts.insert(0, published_measurement_log.path)
        try:
            optional_plot_failures = (
                _render_optional_outputs_after_artifacts(
                    required_artifacts=required_artifacts,
                    renderers=(
                        ("final_pf_views", _render_final_pf_views),
                        (
                            "representative_spectrum",
                            _render_representative_spectrum,
                        ),
                        ("last_spectrum", _render_last_spectrum),
                    ),
                )
            )
        finally:
            if cui_split_viz is not None and hasattr(cui_split_viz, "close"):
                cui_split_viz.close()
        for label, output_path in (
            ("Final PF visualization", pf_out_path),
            ("Final estimates-only visualization", estimates_out_path),
            ("Representative spectrum", spectrum_out_path),
            ("Last spectrum", last_spectrum_out_path),
        ):
            if output_path.is_file():
                print(f"{label} saved to: {output_path}")
        if cui_split_viz is not None and cui_split_viz.index_path.is_file():
            print(f"CUI split view saved to: {cui_split_viz.index_path}")
    elif cui_split_viz is not None and hasattr(cui_split_viz, "close"):
        cui_split_viz.close()
    notification_payload = {
        "summary": (
            f"{step_counter} measurements, "
            f"mission_time_s={total_mission_time_s:.1f}, "
            f"end_to_end_wall_clock_s={end_to_end_wall_clock_s:.2f}"
        ),
        **final_payload,
    }
    if optional_plot_failures:
        notification_payload["optional_plot_failures"] = list(
            optional_plot_failures
        )
    notifier.notify_finished(notification_payload)
    if live:
        plt.ioff()
        plt.pause(0.1)
    plt.close("all")
    if return_state:
        return estimator
    return None
