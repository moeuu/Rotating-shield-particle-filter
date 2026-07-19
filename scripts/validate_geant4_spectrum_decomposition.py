"""Validate Geant4 spectrum decomposition across multi-source cases."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping
import csv
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    from scipy.optimize import least_squares

    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - scipy is an optional runtime dependency.
    least_squares = None
    _SCIPY_AVAILABLE = False

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is an optional runtime dependency.
    torch = None
    _TORCH_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from measurement.continuous_kernels import ContinuousKernel
from measurement.continuous_kernels import capped_transport_response_feature
from measurement.continuous_kernels import finite_sphere_geometric_term
from measurement.continuous_kernels import transport_response_factor_from_payload
from measurement.model import PointSource
from measurement.obstacle_assets import (
    KnownObstacleInstance,
    environment_transport_model,
    generate_manchester_obstacle_instances,
    obstacle_instances_to_dicts,
)
from measurement.obstacles import ObstacleGrid
from measurement.obstacles import build_obstacle_grid
from measurement.observation_model import build_runtime_observation_model
from measurement.observation_model import continuous_kernel_from_observation_model
from measurement.shielding import (
    HVL_TVL_TABLE_MM,
    LOCAL_POSITIVE_OCTANT_CENTER,
    mu_by_isotope_from_tvl_mm,
    rotation_matrix_between_vectors,
)
from pf.estimator import RotatingShieldPFConfig
from pf.likelihood import (
    DEFAULT_GEANT4_COUNT_LIKELIHOOD_DF,
    DEFAULT_GEANT4_COUNT_LIKELIHOOD_MODEL,
    DEFAULT_GEANT4_LOW_COUNT_ABS_SIGMA,
    DEFAULT_GEANT4_LOW_COUNT_TRANSITION_COUNTS,
    DEFAULT_GEANT4_SPECTRUM_COUNT_ABS_SIGMA,
    DEFAULT_GEANT4_SPECTRUM_COUNT_REL_SIGMA,
    DEFAULT_GEANT4_TRANSPORT_MODEL_ABS_SIGMA,
    DEFAULT_GEANT4_TRANSPORT_MODEL_REL_SIGMA,
    CountLikelihoodSpec,
    count_likelihood_variance,
)
from pf.pure_estimator import PurePFEstimator
from sim.geant4_app.app import (
    Geant4Application,
    Geant4AppConfig,
)
from sim.isaacsim_app.scene_builder import SceneDescription, SourceDescription
from sim.isaacsim_app.stage_backend import FakeStageBackend
from sim.protocol import SimulationCommand
from sim.runtime import load_runtime_config
from spectrum.net_response import fit_net_response_calibration
from spectrum.pipeline import SpectralDecomposer, SpectrumConfig
from spectrum.runtime_config import (
    spectrum_config_from_runtime_config as build_spectrum_config_from_runtime_config,
)
from spectrum.runtime_counts import RuntimeCountExtractor
from runtime_defaults import DEFAULT_SOURCE_INTENSITY_RANGE_CPS_1M


ISOTOPES = ("Cs-137", "Co-60", "Eu-154")
DETECTOR_SELECTION_MODES = (
    "balanced",
    "obstacle_extreme",
    "count_imbalance",
    "shield_dynamic_range",
    "mixed_stress",
)
STRESS_SCREEN_SHIELD_PAIR_IDS = (0, 7, 9, 18, 27, 36, 45, 54, 63)
DEFAULT_CALIBRATION_MIN_PAIR_POINTS = 16
DEFAULT_CALIBRATION_PAIR_SHRINKAGE_COUNT = 16.0
DEFAULT_CALIBRATION_HOLDOUT_FRACTION = 0.2
DEFAULT_CALIBRATION_HOLDOUT_SEED = 20260607
DEFAULT_VALIDATION_INTENSITY_MIN_CPS_1M = float(
    DEFAULT_SOURCE_INTENSITY_RANGE_CPS_1M[0]
)
DEFAULT_VALIDATION_INTENSITY_MAX_CPS_1M = float(
    DEFAULT_SOURCE_INTENSITY_RANGE_CPS_1M[1]
)
DEFAULT_TRANSPORT_RESPONSE_TAU_FEATURE_CAPS = {
    "shield": 3.5,
    "fe": 3.5,
    "pb": 3.5,
    "distance_shield": 8.0,
    "distance_fe": 8.0,
    "distance_pb": 8.0,
    "distance_obstacle": 8.0,
}

VALIDATION_METADATA_PREFIXES = (
    "backend",
    "engine_mode",
    "emission_model",
    "physics_profile",
    "accelerated_weighted_transport_enable",
    "source_equivalent",
    "transport_detected",
    "transport_uncollided",
    "transport_interacted",
    "transport_secondary",
    "transport_non_uncollided",
    "geant4_mu_cm_inv",
    "detector_crystal",
    "detector_housing",
    "detector_target",
    "reference_detector",
    "num_primaries",
    "expected_physical_primaries",
    "expected_detector_equivalent_primaries",
    "expected_unthinned_primaries",
    "expected_sampled_primaries",
    "expected_primary_semantics",
    "primary_sampling",
    "requested_primary_sampling",
    "target_sampled_primaries",
    "primary_history",
    "transport_history",
    "history_thinning",
    "transport_tally",
    "weighted",
    "spectrum_variance",
    "dead_time",
    "pre_dead_time",
    "post_dead_time",
    "effective_entries",
    "dwell_time_s",
    "primaries_per_sec",
    "seed",
    "source_rate",
    "source_bias",
    "intensity_cps_1m_definition",
    "line_intensities_normalized",
    "detector_scoring",
    "detector_fast",
    "detector_response_applied",
    "secondary_transport",
    "gamma_only_secondary_transport",
    "theory_tvl",
    "background",
    "poisson_background",
    "fe_shield_normal",
    "pb_shield_normal",
    "runtime_s",
    "run_time_s",
    "thread_count",
    "requested_threads",
    "multithreaded_run_manager",
    "total_track_steps",
    "detector_hit_events",
    "process_count",
    "secondary_count",
    "total_spectrum_counts",
    "volume_step_counts",
)


def validation_observation_metadata(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Return the transport provenance retained in validation artifacts."""
    return {
        str(key): metadata[key]
        for key in sorted(metadata)
        if str(key).startswith(VALIDATION_METADATA_PREFIXES)
    }


def build_runtime_covariance_projector(
    runtime_config: dict[str, Any],
) -> PurePFEstimator:
    """Build a projection-only shell configured like the runtime pure PF."""
    pf_config = RotatingShieldPFConfig(
        observation_covariance_projection_enable=bool(
            runtime_config.get("observation_covariance_projection_enable", True)
        ),
        observation_covariance_projection_weight=float(
            runtime_config.get("observation_covariance_projection_weight", 1.0)
        ),
        observation_covariance_projection_max_corr=float(
            runtime_config.get("observation_covariance_projection_max_corr", 0.999)
        ),
    )
    projector = object.__new__(PurePFEstimator)
    projector.pf_config = pf_config
    return projector


def project_runtime_observation_covariance(
    projector: PurePFEstimator,
    counts: dict[str, float],
    variances: dict[str, float],
    covariance: dict[str, dict[str, float]] | None,
) -> tuple[dict[str, float], dict[str, dict[str, float]] | None]:
    """Call the runtime estimator's shared observation-covariance projection."""
    projected, sanitized = projector._project_observation_covariance_to_variance(
        counts,
        variances,
        covariance,
    )
    return (
        dict(variances) if projected is None else dict(projected),
        sanitized,
    )


def _runtime_count_likelihood_value(
    runtime_config: Mapping[str, Any],
    key: str,
    default: object,
) -> object:
    """Read one nested or legacy PF count-likelihood configuration value."""
    nested = runtime_config.get("pf_count_likelihood", {})
    if isinstance(nested, Mapping) and key in nested:
        return nested[key]
    return runtime_config.get(f"pf_{key}", default)


def _isotope_float_value(
    value: object,
    isotope: str,
    *,
    default: float = 0.0,
) -> float:
    """Resolve a scalar or isotope-indexed nonnegative likelihood setting."""
    selected = value.get(isotope, default) if isinstance(value, Mapping) else value
    try:
        numeric = float(selected)
    except (TypeError, ValueError):
        numeric = float(default)
    return max(numeric, 0.0)


def build_runtime_count_likelihood_specs(
    runtime_config: Mapping[str, Any],
) -> dict[str, CountLikelihoodSpec]:
    """Resolve the same isotope-wise count likelihood used by runtime PFs."""
    geant4_defaults = str(runtime_config.get("backend", "")).lower() == "geant4"
    spectrum_defaults = (
        str(runtime_config.get("spectrum_count_method", ""))
        == RuntimeCountExtractor.STANDARD_METHOD
    )
    robust_defaults = bool(geant4_defaults or spectrum_defaults)
    model = str(
        _runtime_count_likelihood_value(
            runtime_config,
            "count_likelihood_model",
            DEFAULT_GEANT4_COUNT_LIKELIHOOD_MODEL if robust_defaults else "poisson",
        )
    )
    transport_rel = _runtime_count_likelihood_value(
        runtime_config,
        "transport_model_rel_sigma",
        DEFAULT_GEANT4_TRANSPORT_MODEL_REL_SIGMA if geant4_defaults else 0.0,
    )
    transport_abs = _runtime_count_likelihood_value(
        runtime_config,
        "transport_model_abs_sigma",
        DEFAULT_GEANT4_TRANSPORT_MODEL_ABS_SIGMA if geant4_defaults else 0.0,
    )
    spectrum_rel = _runtime_count_likelihood_value(
        runtime_config,
        "spectrum_count_rel_sigma",
        DEFAULT_GEANT4_SPECTRUM_COUNT_REL_SIGMA if geant4_defaults else 0.0,
    )
    spectrum_abs = _runtime_count_likelihood_value(
        runtime_config,
        "spectrum_count_abs_sigma",
        DEFAULT_GEANT4_SPECTRUM_COUNT_ABS_SIGMA if geant4_defaults else 0.0,
    )
    low_count_abs = _runtime_count_likelihood_value(
        runtime_config,
        "low_count_abs_sigma",
        DEFAULT_GEANT4_LOW_COUNT_ABS_SIGMA if geant4_defaults else 0.0,
    )
    low_count_transition = _runtime_count_likelihood_value(
        runtime_config,
        "low_count_transition_counts",
        DEFAULT_GEANT4_LOW_COUNT_TRANSITION_COUNTS if geant4_defaults else 0.0,
    )
    observation_variance_includes_counting_noise = bool(
        _runtime_count_likelihood_value(
            runtime_config,
            "observation_count_variance_includes_counting_noise",
            False,
        )
    )
    observation_variance_semantics = str(
        _runtime_count_likelihood_value(
            runtime_config,
            "observation_count_variance_semantics",
            "",
        )
    )
    student_t_df = float(
        _runtime_count_likelihood_value(
            runtime_config,
            "count_likelihood_df",
            DEFAULT_GEANT4_COUNT_LIKELIHOOD_DF,
        )
    )
    return {
        isotope: CountLikelihoodSpec(
            model=model,
            transport_model_rel_sigma=_isotope_float_value(
                transport_rel,
                isotope,
            ),
            transport_model_abs_sigma=_isotope_float_value(
                transport_abs,
                isotope,
            ),
            spectrum_count_rel_sigma=_isotope_float_value(
                spectrum_rel,
                isotope,
            ),
            spectrum_count_abs_sigma=_isotope_float_value(
                spectrum_abs,
                isotope,
            ),
            low_count_abs_sigma=_isotope_float_value(
                low_count_abs,
                isotope,
            ),
            low_count_transition_counts=_isotope_float_value(
                low_count_transition,
                isotope,
            ),
            observation_count_variance_includes_counting_noise=(
                observation_variance_includes_counting_noise
            ),
            observation_count_variance_semantics=(observation_variance_semantics),
            student_t_df=max(student_t_df, 1.0),
        )
        for isotope in ISOTOPES
    }


def _count_likelihood_spec_payload(spec: CountLikelihoodSpec) -> dict[str, Any]:
    """Return a JSON-safe count-likelihood specification."""
    marginal_factor = (
        float(spec.student_t_df / (spec.student_t_df - 2.0))
        if spec.model == "student_t" and spec.student_t_df > 2.0
        else None
    )
    return {
        "model": spec.model,
        "transport_model_rel_sigma": spec.transport_model_rel_sigma,
        "transport_model_abs_sigma": spec.transport_model_abs_sigma,
        "spectrum_count_rel_sigma": spec.spectrum_count_rel_sigma,
        "spectrum_count_abs_sigma": spec.spectrum_count_abs_sigma,
        "low_count_abs_sigma": spec.low_count_abs_sigma,
        "low_count_transition_counts": spec.low_count_transition_counts,
        "observation_count_variance_includes_counting_noise": bool(
            spec.observation_count_variance_includes_counting_noise
        ),
        "observation_count_variance_semantics": str(
            spec.observation_count_variance_semantics
        ),
        "student_t_df": spec.student_t_df,
        "student_t_marginal_variance_over_scale_squared": marginal_factor,
    }


@dataclass(frozen=True)
class ValidationSource:
    """Describe a source used by a validation case."""

    isotope: str
    position_xyz: tuple[float, float, float]
    intensity_cps_1m: float

    def to_point_source(self) -> PointSource:
        """Convert this source to the measurement-model representation."""
        return PointSource(
            isotope=self.isotope,
            position=self.position_xyz,
            intensity_cps_1m=self.intensity_cps_1m,
        )

    def to_scene_source(self) -> SourceDescription:
        """Convert this source to the Geant4 scene representation."""
        return SourceDescription(
            isotope=self.isotope,
            position_xyz=self.position_xyz,
            intensity_cps_1m=self.intensity_cps_1m,
        )


@dataclass(frozen=True)
class ValidationCase:
    """Describe one Geant4 spectrum-decomposition validation case."""

    name: str
    description: str
    detector_pose_xyz: tuple[float, float, float]
    sources: tuple[ValidationSource, ...]
    fe_index: int = 0
    pb_index: int = 0
    dwell_time_s: float = 30.0
    obstacle_cells: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    obstacle_instances: tuple[KnownObstacleInstance, ...] = field(default_factory=tuple)
    include_in_accuracy_summary: bool = True
    generation_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectorPoseCandidate:
    """Store one detector pose and its inexpensive preselection score."""

    pose_xyz: tuple[float, float, float]
    proxy_score: float
    proxy_features: dict[str, Any]


def _clamp_room_position(
    position: np.ndarray,
    *,
    room_size_xyz: tuple[float, float, float],
    margin_xy_m: float = 0.75,
) -> tuple[float, float, float]:
    """Clamp a candidate source position inside the validation room."""
    room = np.asarray(room_size_xyz, dtype=float)
    pos = np.asarray(position, dtype=float).copy()
    pos[0] = float(np.clip(pos[0], margin_xy_m, room[0] - margin_xy_m))
    pos[1] = float(np.clip(pos[1], margin_xy_m, room[1] - margin_xy_m))
    pos[2] = float(np.clip(pos[2], 0.5, min(room[2] - 0.5, 2.0)))
    return (float(pos[0]), float(pos[1]), float(pos[2]))


def _line_obstacle_cells(
    source_xyz: tuple[float, float, float],
    detector_xyz: tuple[float, float, float],
    *,
    grid_shape: tuple[int, int] = (10, 20),
    max_cells: int = 3,
) -> tuple[tuple[int, int], ...]:
    """Return obstacle cells sampled along a source-detector line segment."""
    source = np.asarray(source_xyz, dtype=float)
    detector = np.asarray(detector_xyz, dtype=float)
    cells: list[tuple[int, int]] = []
    endpoint_cells = {
        (int(np.floor(source[0])), int(np.floor(source[1]))),
        (int(np.floor(detector[0])), int(np.floor(detector[1]))),
    }
    for fraction in np.linspace(0.25, 0.75, 5):
        point = source * (1.0 - float(fraction)) + detector * float(fraction)
        cell = (int(np.floor(point[0])), int(np.floor(point[1])))
        if cell in endpoint_cells or cell in cells:
            continue
        if 0 <= cell[0] < grid_shape[0] and 0 <= cell[1] < grid_shape[1]:
            cells.append(cell)
        if len(cells) >= int(max_cells):
            break
    return tuple(cells)


def _attach_known_obstacle_transport(
    grid: ObstacleGrid,
    instances: tuple[KnownObstacleInstance, ...],
) -> ObstacleGrid:
    """Attach known material-specific transport boxes to an obstacle grid."""
    boxes_m, mu_by_isotope, line_mu_by_isotope = environment_transport_model(
        instances,
        room_size_xyz=(10.0, 20.0, 10.0),
        include_room_boundaries=True,
        isotopes=ISOTOPES,
    )
    return grid.with_transport_model(
        boxes_m=boxes_m,
        mu_by_isotope=mu_by_isotope,
        line_mu_by_isotope=line_mu_by_isotope,
    )


def _free_cell_centers(grid: ObstacleGrid) -> list[tuple[float, float]]:
    """Return centers of free grid cells."""
    centers: list[tuple[float, float]] = []
    for ix in range(int(grid.grid_shape[0])):
        for iy in range(int(grid.grid_shape[1])):
            if not grid.is_cell_free((ix, iy)):
                continue
            centers.append(
                (
                    float(grid.origin[0] + (ix + 0.5) * grid.cell_size),
                    float(grid.origin[1] + (iy + 0.5) * grid.cell_size),
                )
            )
    return centers


def _jittered_free_pose(
    grid: ObstacleGrid,
    rng: np.random.Generator,
    *,
    room_size_xyz: tuple[float, float, float],
    z_m: float = 0.5,
) -> tuple[float, float, float]:
    """Sample a detector pose inside a free cell."""
    centers = _free_cell_centers(grid)
    if not centers:
        raise ValueError("Generated obstacle grid has no free cells.")
    center_xy = centers[int(rng.integers(0, len(centers)))]
    jitter = 0.32 * float(grid.cell_size)
    x = float(center_xy[0] + rng.uniform(-jitter, jitter))
    y = float(center_xy[1] + rng.uniform(-jitter, jitter))
    x = float(np.clip(x, 0.4, room_size_xyz[0] - 0.4))
    y = float(np.clip(y, 0.4, room_size_xyz[1] - 0.4))
    return (x, y, float(z_m))


def _component_top_z(instance: KnownObstacleInstance) -> float:
    """Return the highest z coordinate of a known obstacle instance."""
    if not instance.components:
        return 0.0
    return max(
        float(component.center_xyz[2]) + 0.5 * float(component.size_xyz[2])
        for component in instance.components
    )


def _is_inside_known_obstacle_transport_volume(
    position_xyz: tuple[float, float, float],
    instances: tuple[KnownObstacleInstance, ...],
    *,
    tolerance_m: float = 1.0e-6,
) -> bool:
    """Return True when a point lies strictly inside a known obstacle component."""
    point = np.asarray(position_xyz, dtype=float).reshape(3)
    tol = max(float(tolerance_m), 0.0)
    for instance in instances:
        for component in instance.components:
            box = np.asarray(component.box_m, dtype=float)
            lower = box[:3] + tol
            upper = box[3:] - tol
            if (
                np.all(upper > lower)
                and np.all(point > lower)
                and np.all(point < upper)
            ):
                return True
    return False


def _sample_nonembedded_surface_source_position(
    sampler: Any,
    grid: ObstacleGrid,
    instances: tuple[KnownObstacleInstance, ...],
    rng: np.random.Generator,
    *,
    room_size_xyz: tuple[float, float, float],
    source_index: int,
    max_attempts: int = 256,
) -> tuple[float, float, float]:
    """Sample a surface source while rejecting points embedded in obstacle material."""
    for _ in range(max(1, int(max_attempts))):
        position = sampler(
            grid,
            instances,
            rng,
            room_size_xyz=room_size_xyz,
            source_index=source_index,
        )
        if not _is_inside_known_obstacle_transport_volume(position, instances):
            return position
    raise ValueError("Could not sample a non-embedded source position.")


def _surface_source_position(
    grid: ObstacleGrid,
    instances: tuple[KnownObstacleInstance, ...],
    rng: np.random.Generator,
    *,
    room_size_xyz: tuple[float, float, float],
    source_index: int,
) -> tuple[float, float, float]:
    """Sample a source position on free floor, obstacle side, or obstacle top."""
    mode = int(source_index) % 4
    if instances and mode in {1, 2}:
        instance = instances[int(rng.integers(0, len(instances)))]
        x0, x1, y0, y1 = instance.footprint_xy
        if mode == 1:
            side = int(rng.integers(0, 4))
            if side == 0:
                x, y = x0 - 0.08, float(rng.uniform(y0, y1))
            elif side == 1:
                x, y = x1 + 0.08, float(rng.uniform(y0, y1))
            elif side == 2:
                x, y = float(rng.uniform(x0, x1)), y0 - 0.08
            else:
                x, y = float(rng.uniform(x0, x1)), y1 + 0.08
            z = float(
                rng.uniform(
                    0.45, min(_component_top_z(instance), room_size_xyz[2] - 0.2)
                )
            )
        else:
            x = float(rng.uniform(x0 + 0.12, x1 - 0.12))
            y = float(rng.uniform(y0 + 0.12, y1 - 0.12))
            z = min(_component_top_z(instance) + 0.08, room_size_xyz[2] - 0.2)
        return _clamp_room_position(
            np.asarray((x, y, z), dtype=float),
            room_size_xyz=room_size_xyz,
            margin_xy_m=0.25,
        )

    return _jittered_free_pose(
        grid,
        rng,
        room_size_xyz=room_size_xyz,
        z_m=float(rng.uniform(0.35, 1.8)),
    )


def _multi_isotope_surface_source_position(
    grid: ObstacleGrid,
    instances: tuple[KnownObstacleInstance, ...],
    rng: np.random.Generator,
    *,
    room_size_xyz: tuple[float, float, float],
    source_index: int,
) -> tuple[float, float, float]:
    """Sample a surface source over room and obstacle surfaces."""
    mode = int(source_index) % 6
    eps = 0.08
    if mode == 0:
        pose = _jittered_free_pose(grid, rng, room_size_xyz=room_size_xyz, z_m=eps)
        return (pose[0], pose[1], eps)
    if mode == 1:
        pose = _jittered_free_pose(
            grid,
            rng,
            room_size_xyz=room_size_xyz,
            z_m=room_size_xyz[2] - eps,
        )
        return (pose[0], pose[1], room_size_xyz[2] - eps)
    if mode == 2:
        wall = int(rng.integers(0, 4))
        z = float(rng.uniform(0.35, room_size_xyz[2] - 0.35))
        if wall == 0:
            return (eps, float(rng.uniform(0.35, room_size_xyz[1] - 0.35)), z)
        if wall == 1:
            return (
                room_size_xyz[0] - eps,
                float(rng.uniform(0.35, room_size_xyz[1] - 0.35)),
                z,
            )
        if wall == 2:
            return (float(rng.uniform(0.35, room_size_xyz[0] - 0.35)), eps, z)
        return (
            float(rng.uniform(0.35, room_size_xyz[0] - 0.35)),
            room_size_xyz[1] - eps,
            z,
        )
    if instances and mode in {3, 4}:
        return _surface_source_position(
            grid,
            instances,
            rng,
            room_size_xyz=room_size_xyz,
            source_index=1 if mode == 3 else 2,
        )
    return _jittered_free_pose(
        grid,
        rng,
        room_size_xyz=room_size_xyz,
        z_m=float(rng.uniform(0.35, room_size_xyz[2] - 0.35)),
    )


def _obstacle_tau_for_sources(
    grid: ObstacleGrid,
    detector_xyz: tuple[float, float, float],
    sources: tuple[ValidationSource, ...],
) -> float:
    """Return the maximum obstacle optical depth among source-detector rays."""
    kernel = ContinuousKernel(obstacle_grid=grid, use_gpu=False)
    detector = np.asarray(detector_xyz, dtype=float)
    max_tau = 0.0
    for source in sources:
        tau = kernel.obstacle_optical_depth_pair(
            source.isotope,
            np.asarray(source.position_xyz, dtype=float),
            detector,
        )
        max_tau = max(max_tau, float(tau))
    return max_tau


def _sample_detector_with_tau_mix(
    grid: ObstacleGrid,
    sources: tuple[ValidationSource, ...],
    rng: np.random.Generator,
    *,
    room_size_xyz: tuple[float, float, float],
    prefer_obstacle_crossing: bool,
) -> tuple[float, float, float]:
    """Sample detector poses that alternate direct and obstacle-crossing rays."""
    best_pose = _jittered_free_pose(grid, rng, room_size_xyz=room_size_xyz)
    best_tau = _obstacle_tau_for_sources(grid, best_pose, sources)
    for _ in range(80):
        pose = _jittered_free_pose(grid, rng, room_size_xyz=room_size_xyz)
        tau = _obstacle_tau_for_sources(grid, pose, sources)
        if prefer_obstacle_crossing and tau >= 0.1:
            return pose
        if not prefer_obstacle_crossing and tau <= 0.02:
            return pose
        if prefer_obstacle_crossing and tau > best_tau:
            best_pose, best_tau = pose, tau
        if not prefer_obstacle_crossing and tau < best_tau:
            best_pose, best_tau = pose, tau
    return best_pose


def _effective_detector_selection_mode(mode: str, case_index: int) -> str:
    """Return the concrete detector-selection mode for one generated scenario."""
    normalized = str(mode).strip().lower()
    if normalized != "mixed_stress":
        return normalized if normalized in DETECTOR_SELECTION_MODES else "balanced"
    stress_modes = ("obstacle_extreme", "count_imbalance", "shield_dynamic_range")
    return stress_modes[int(case_index) % len(stress_modes)]


def _detector_sampling_prefers_obstacle(
    mode: str,
    case_index: int,
    detector_attempt: int,
) -> bool:
    """Return whether detector sampling should favor obstacle-crossing rays."""
    if mode == "obstacle_extreme":
        return True
    if mode == "shield_dynamic_range":
        return int(detector_attempt) % 2 == 0
    if mode == "count_imbalance":
        return (int(case_index) + int(detector_attempt)) % 3 != 0
    return (int(case_index) + int(detector_attempt)) % 3 != 0


def _case_with_detector_and_pair(
    base_case: ValidationCase,
    detector_xyz: tuple[float, float, float],
    *,
    fe_index: int,
    pb_index: int,
    description: str = "candidate",
) -> ValidationCase:
    """Return a copy of a case with a candidate detector pose and shield pair."""
    return ValidationCase(
        name=base_case.name,
        description=description,
        detector_pose_xyz=detector_xyz,
        sources=base_case.sources,
        fe_index=int(fe_index),
        pb_index=int(pb_index),
        dwell_time_s=base_case.dwell_time_s,
        obstacle_cells=base_case.obstacle_cells,
        obstacle_instances=base_case.obstacle_instances,
        include_in_accuracy_summary=base_case.include_in_accuracy_summary,
        generation_metadata=dict(base_case.generation_metadata),
    )


def _expected_count_matrix_over_shield_pairs_serial(
    case: ValidationCase,
    runtime_config: dict[str, Any],
    kernel: ContinuousKernel,
    pair_ids: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Return a scalar test oracle for selected Fe/Pb-pair expected counts."""
    selected_pair_ids = (
        tuple(range(64))
        if pair_ids is None
        else tuple(int(pair_id) for pair_id in pair_ids)
    )
    matrix = np.zeros((len(selected_pair_ids), len(ISOTOPES)), dtype=float)
    for row_index, pair_id in enumerate(selected_pair_ids):
        fe_index = int(pair_id) // 8
        pb_index = int(pair_id) % 8
        pair_case = _case_with_detector_and_pair(
            case,
            case.detector_pose_xyz,
            fe_index=fe_index,
            pb_index=pb_index,
            description=case.description,
        )
        counts = expected_pf_counts_with_kernel(pair_case, runtime_config, kernel)
        matrix[row_index, :] = [float(counts[isotope]) for isotope in ISOTOPES]
    return matrix


def _measurement_source_scales_for_pairs(
    isotope: str,
    pair_ids: NDArray[np.int64],
    runtime_config: Mapping[str, Any],
) -> NDArray[np.float64]:
    """Return configured source-count scales for a fixed shield-pair batch."""
    global_scales = runtime_config.get("measurement_scale_by_isotope", {})
    global_scale = (
        max(float(global_scales.get(str(isotope), 1.0)), 0.0)
        if isinstance(global_scales, Mapping)
        else 1.0
    )
    pair_scale_root = runtime_config.get(
        "measurement_scale_by_isotope_and_pair",
        {},
    )
    pair_scales = (
        pair_scale_root.get(str(isotope), {})
        if isinstance(pair_scale_root, Mapping)
        else {}
    )
    if not isinstance(pair_scales, Mapping):
        return np.full(pair_ids.size, global_scale, dtype=float)
    # This is a fixed-size metadata gather, not shield-response computation;
    # the 64-pair physics kernel below remains one batched operation.
    return np.asarray(
        [
            max(
                float(
                    pair_scales.get(
                        str(int(pair_id)),
                        pair_scales.get(int(pair_id), global_scale),
                    )
                ),
                0.0,
            )
            for pair_id in pair_ids
        ],
        dtype=float,
    )


def _expected_count_matrix_over_shield_pairs(
    case: ValidationCase,
    runtime_config: dict[str, Any],
    kernel: ContinuousKernel,
    pair_ids: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Return all selected pair counts using the batched kernel runtime path."""
    num_orientations = int(len(kernel.orientations))
    num_pairs = num_orientations * num_orientations
    selected_pair_ids = np.asarray(
        np.arange(num_pairs, dtype=np.int64)
        if pair_ids is None
        else tuple(int(pair_id) for pair_id in pair_ids),
        dtype=np.int64,
    ).reshape(-1)
    if np.any(selected_pair_ids < 0) or np.any(selected_pair_ids >= num_pairs):
        raise ValueError(f"Shield pair ids must be in [0, {num_pairs - 1}].")
    matrix = np.zeros((selected_pair_ids.size, len(ISOTOPES)), dtype=float)
    detector = np.asarray(case.detector_pose_xyz, dtype=float)
    for isotope_index, isotope in enumerate(ISOTOPES):
        isotope_sources = [
            source for source in case.sources if source.isotope == isotope
        ]
        if not isotope_sources:
            continue
        positions = np.asarray(
            [source.position_xyz for source in isotope_sources],
            dtype=float,
        )
        intensities = np.asarray(
            [source.intensity_cps_1m for source in isotope_sources],
            dtype=float,
        )
        all_pair_kernels = kernel.kernel_values_all_pairs(
            isotope=isotope,
            detector_pos=detector,
            sources=positions,
        )
        selected_kernels = np.asarray(all_pair_kernels, dtype=float)[
            selected_pair_ids,
            :,
        ]
        source_scales = _measurement_source_scales_for_pairs(
            isotope,
            selected_pair_ids,
            runtime_config,
        )
        matrix[:, isotope_index] = (
            float(case.dwell_time_s)
            * source_scales
            * np.asarray(selected_kernels @ intensities, dtype=float)
        )
    return matrix


def _enable_batched_pair_screening(
    kernel: ContinuousKernel,
    runtime_config: Mapping[str, Any],
) -> ContinuousKernel:
    """Select Torch batching for exact all-pair detector-pose validation."""
    if not _TORCH_AVAILABLE or torch is None:
        raise RuntimeError("Exact all-shield-pair screening requires torch batching.")
    requested_device = str(
        runtime_config.get("validation_pair_screening_device", "")
    ).strip()
    if not requested_device:
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    kernel.use_gpu = True
    kernel.gpu_device = requested_device
    kernel.gpu_dtype = str(
        runtime_config.get("validation_pair_screening_dtype", "float64")
    )
    return kernel


def _count_imbalance(counts: np.ndarray) -> float:
    """Return the max/min isotope count ratio for a count vector."""
    arr = np.asarray(counts, dtype=float)
    if arr.size == 0:
        return 1.0
    positive_floor = max(
        float(np.min(arr[arr > 0.0])) if np.any(arr > 0.0) else 1.0,
        1.0,
    )
    return float(np.max(arr) / max(float(np.min(arr)), positive_floor))


def _shield_dynamic_range(count_matrix: np.ndarray) -> float:
    """Return the largest per-isotope response ratio over all shield pairs."""
    matrix = np.asarray(count_matrix, dtype=float)
    if matrix.size == 0:
        return 1.0
    ranges: list[float] = []
    for isotope_index in range(matrix.shape[1]):
        values = matrix[:, isotope_index]
        positive_values = values[values > 0.0]
        floor = max(
            float(np.min(positive_values)) if positive_values.size else 1.0,
            1.0,
        )
        ranges.append(float(np.max(values) / max(float(np.min(values)), floor)))
    return max(ranges) if ranges else 1.0


def _detector_selection_features(
    base_case: ValidationCase,
    detector_xyz: tuple[float, float, float],
    *,
    transport_grid: ObstacleGrid,
    runtime_config: dict[str, Any] | None,
    target_kernel: ContinuousKernel | None,
    evaluate_all_shield_pairs: bool,
    selection_pair_ids: tuple[int, ...] | None = None,
    use_fast_proxy: bool = False,
) -> dict[str, Any]:
    """Return detector-pose features for balanced and stress validation sampling."""
    candidate_case = _case_with_detector_and_pair(
        base_case,
        detector_xyz,
        fe_index=base_case.fe_index,
        pb_index=base_case.pb_index,
    )
    max_tau = _obstacle_tau_for_sources(
        transport_grid,
        detector_xyz,
        base_case.sources,
    )
    features: dict[str, Any] = {
        "max_obstacle_tau": float(max_tau),
        "min_isotope_target": 0.0,
        "target_count_imbalance": 1.0,
        "shield_dynamic_range": 1.0,
        "screened_shield_pair_count": 1,
    }
    if bool(use_fast_proxy):
        features.update(
            _fast_detector_selection_proxy_features(
                base_case,
                detector_xyz,
                transport_grid=transport_grid,
                detector_radius_m=(
                    float(target_kernel.detector_radius_m)
                    if target_kernel is not None
                    else 0.0
                ),
            )
        )
        return features
    if runtime_config is None or target_kernel is None:
        return features
    if bool(evaluate_all_shield_pairs):
        count_matrix = _expected_count_matrix_over_shield_pairs(
            candidate_case,
            runtime_config,
            target_kernel,
            pair_ids=selection_pair_ids,
        )
        reference_counts = np.median(count_matrix, axis=0)
        features["min_isotope_target"] = float(np.min(count_matrix))
        features["target_count_imbalance"] = _count_imbalance(reference_counts)
        features["shield_dynamic_range"] = _shield_dynamic_range(count_matrix)
        features["per_isotope_median_target_counts"] = {
            isotope: float(reference_counts[index])
            for index, isotope in enumerate(ISOTOPES)
        }
        features["per_isotope_min_target_counts"] = {
            isotope: float(np.min(count_matrix[:, index]))
            for index, isotope in enumerate(ISOTOPES)
        }
        features["per_isotope_max_target_counts"] = {
            isotope: float(np.max(count_matrix[:, index]))
            for index, isotope in enumerate(ISOTOPES)
        }
        features["screened_shield_pair_count"] = int(count_matrix.shape[0])
        return features
    counts = expected_pf_counts_with_kernel(
        candidate_case, runtime_config, target_kernel
    )
    count_vector = np.asarray(
        [float(counts[isotope]) for isotope in ISOTOPES],
        dtype=float,
    )
    features["min_isotope_target"] = float(np.min(count_vector))
    features["target_count_imbalance"] = _count_imbalance(count_vector)
    features["per_isotope_target_counts"] = {
        isotope: float(counts[isotope]) for isotope in ISOTOPES
    }
    return features


def _fast_detector_selection_proxy_features(
    base_case: ValidationCase,
    detector_xyz: tuple[float, float, float],
    *,
    transport_grid: ObstacleGrid,
    detector_radius_m: float,
) -> dict[str, Any]:
    """Return cheap geometry features for stress detector-pose screening."""
    detector = np.asarray(detector_xyz, dtype=float)
    counts_by_isotope = {isotope: 0.0 for isotope in ISOTOPES}
    direction_hist = np.zeros(8, dtype=float)
    max_tau = 0.0
    kernel = ContinuousKernel(obstacle_grid=transport_grid, use_gpu=False)
    for source in base_case.sources:
        source_pos = np.asarray(source.position_xyz, dtype=float)
        geom = finite_sphere_geometric_term(
            detector,
            source_pos,
            float(detector_radius_m),
        )
        tau = kernel.obstacle_optical_depth_pair(source.isotope, source_pos, detector)
        max_tau = max(max_tau, float(tau))
        count_proxy = (
            float(base_case.dwell_time_s)
            * float(source.intensity_cps_1m)
            * float(geom)
            * float(np.exp(-float(tau)))
        )
        counts_by_isotope[source.isotope] += count_proxy
        delta = source_pos - detector
        angle = float(np.arctan2(delta[1], delta[0]))
        octant = int(np.floor(((angle + np.pi) / (2.0 * np.pi)) * 8.0)) % 8
        direction_hist[octant] += max(count_proxy, 0.0)
    count_vector = np.asarray([counts_by_isotope[isotope] for isotope in ISOTOPES])
    hist_floor = max(
        float(np.min(direction_hist[direction_hist > 0.0]))
        if np.any(direction_hist > 0.0)
        else 1.0,
        1.0,
    )
    return {
        "max_obstacle_tau": float(max_tau),
        "min_isotope_target": float(np.min(count_vector)) if count_vector.size else 0.0,
        "target_count_imbalance": _count_imbalance(count_vector),
        "shield_dynamic_range": float(np.max(direction_hist) / hist_floor),
        "screened_shield_pair_count": int(0),
        "screening_model": "fast_inverse_square_obstacle_direction_proxy",
        "per_isotope_proxy_target_counts": {
            isotope: float(counts_by_isotope[isotope]) for isotope in ISOTOPES
        },
    }


def _detector_selection_score(
    features: dict[str, Any],
    *,
    mode: str,
    min_target_counts: float,
) -> float:
    """Return a deterministic score for detector-pose selection."""
    min_target = float(features.get("min_isotope_target", 0.0))
    target_gate = 1.0
    if float(min_target_counts) > 0.0:
        target_gate = min(max(min_target / float(min_target_counts), 0.0), 1.0)
    if mode == "obstacle_extreme":
        return (
            np.log1p(max(float(features.get("max_obstacle_tau", 0.0)), 0.0))
            + 0.15 * np.log1p(max(min_target, 0.0))
            + 2.0 * target_gate
        )
    if mode == "count_imbalance":
        return (
            np.log(max(float(features.get("target_count_imbalance", 1.0)), 1.0))
            + 0.05 * np.log1p(max(float(features.get("max_obstacle_tau", 0.0)), 0.0))
            + 2.0 * target_gate
        )
    if mode == "shield_dynamic_range":
        return (
            np.log(max(float(features.get("shield_dynamic_range", 1.0)), 1.0))
            + 0.05 * np.log1p(max(float(features.get("max_obstacle_tau", 0.0)), 0.0))
            + 2.0 * target_gate
        )
    return min_target


def _selection_screen_pair_ids(
    *,
    mode: str,
    screen_all_shield_pairs: bool,
) -> tuple[int, ...] | None:
    """Return shield pairs used for detector-pose screening."""
    if mode == "balanced":
        return None if bool(screen_all_shield_pairs) else None
    return STRESS_SCREEN_SHIELD_PAIR_IDS


def _select_exact_target_qualified_detector(
    candidates: list[DetectorPoseCandidate],
    *,
    base_case: ValidationCase,
    transport_grid: ObstacleGrid,
    runtime_config: dict[str, Any],
    target_kernel: ContinuousKernel,
    min_target_counts: float,
    all_shield_pairs: bool,
) -> tuple[
    DetectorPoseCandidate,
    dict[str, Any],
    int,
]:
    """Select the highest-ranked pose passing the exact PF response gate."""
    threshold = max(float(min_target_counts), 0.0)
    ranked = sorted(candidates, key=lambda item: item.proxy_score, reverse=True)
    best_exact_min = -float("inf")
    for attempt_index, candidate in enumerate(ranked, start=1):
        exact_features = _detector_selection_features(
            base_case,
            candidate.pose_xyz,
            transport_grid=transport_grid,
            runtime_config=runtime_config,
            target_kernel=target_kernel,
            evaluate_all_shield_pairs=bool(all_shield_pairs),
            selection_pair_ids=None,
            use_fast_proxy=False,
        )
        exact_min = float(exact_features.get("min_isotope_target", 0.0))
        best_exact_min = max(best_exact_min, exact_min)
        if exact_min >= threshold:
            return candidate, exact_features, attempt_index
    raise ValueError(
        "No sampled detector pose satisfies the exact PF response gate across "
        f"{'all 64 shield pairs' if all_shield_pairs else 'the selected pair'}: "
        f"required={threshold:.6g}, best={best_exact_min:.6g}, "
        f"candidates={len(ranked)}."
    )


def _source_isotope_program(case_index: int, source_count: int) -> list[str]:
    """Return a deterministic isotope assignment with single and mixed cases."""
    programs = (
        ("Cs-137",),
        ("Co-60",),
        ("Eu-154",),
        ("Cs-137", "Co-60"),
        ("Cs-137", "Eu-154"),
        ("Co-60", "Eu-154"),
        ("Cs-137", "Co-60", "Eu-154"),
        ("Cs-137", "Cs-137", "Co-60"),
        ("Co-60", "Co-60", "Eu-154"),
        ("Cs-137", "Eu-154", "Eu-154"),
    )
    base = list(programs[int(case_index) % len(programs)])
    while len(base) < int(source_count):
        base.append(ISOTOPES[(int(case_index) + len(base)) % len(ISOTOPES)])
    return base[: int(source_count)]


def generated_cases(
    *,
    num_cases: int = 50,
    seed: int = 20260430,
    dwell_time_s: float = 30.0,
    intensity_min_cps_1m: float = DEFAULT_VALIDATION_INTENSITY_MIN_CPS_1M,
    intensity_max_cps_1m: float = DEFAULT_VALIDATION_INTENSITY_MAX_CPS_1M,
) -> list[ValidationCase]:
    """Return deterministic Geant4 validation cases spanning sources, shielding, and obstacles."""
    rng = np.random.default_rng(int(seed))
    room_size = (10.0, 20.0, 10.0)
    detector_positions = (
        (1.5, 1.5, 0.5),
        (2.5, 4.5, 0.5),
        (5.0, 10.0, 0.5),
        (7.5, 15.5, 0.5),
        (8.5, 5.5, 0.5),
    )
    directions = np.asarray(
        [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (1.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (1.0, -1.0, 0.0),
            (-1.0, -1.0, 0.0),
            (1.0, 1.0, 0.7),
            (-1.0, 1.0, 0.7),
            (1.0, -1.0, 0.7),
            (-1.0, -1.0, 0.7),
        ],
        dtype=float,
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    distances = (1.0, 1.5, 2.2, 3.0, 4.0, 5.2, 6.5)
    cases: list[ValidationCase] = []
    for case_index in range(max(0, int(num_cases))):
        detector = np.asarray(
            detector_positions[case_index % len(detector_positions)], dtype=float
        )
        source_count = 1 + (case_index % 5)
        isotope_program = _source_isotope_program(case_index, source_count)
        sources: list[ValidationSource] = []
        for source_index in range(source_count):
            direction = directions[
                (case_index * 3 + source_index * 5) % len(directions)
            ]
            distance = float(
                distances[(case_index + source_index * 2) % len(distances)]
            )
            jitter = rng.normal(0.0, 0.18, size=3)
            jitter[2] *= 0.35
            position = _clamp_room_position(
                detector + distance * direction + jitter,
                room_size_xyz=room_size,
            )
            intensity = float(
                rng.uniform(float(intensity_min_cps_1m), float(intensity_max_cps_1m))
            )
            sources.append(
                ValidationSource(
                    isotope=isotope_program[source_index],
                    position_xyz=position,
                    intensity_cps_1m=intensity,
                )
            )
        obstacle_cells: tuple[tuple[int, int], ...] = ()
        obstacle_mode = case_index % 4
        if obstacle_mode in (1, 2, 3):
            line_cells: list[tuple[int, int]] = []
            for source in sources[: 1 + (obstacle_mode == 3)]:
                line_cells.extend(
                    _line_obstacle_cells(
                        source.position_xyz,
                        tuple(float(v) for v in detector),
                        max_cells=1 + obstacle_mode,
                    )
                )
            obstacle_cells = tuple(sorted(set(line_cells)))
        cases.append(
            ValidationCase(
                name=f"g4_case_{case_index:02d}",
                description=(
                    f"{source_count} source(s), {len(set(isotope_program))} isotope(s), "
                    f"obstacle_cells={len(obstacle_cells)}"
                ),
                detector_pose_xyz=tuple(float(v) for v in detector),
                sources=tuple(sources),
                fe_index=(case_index * 3) % 8,
                pb_index=(case_index * 5 + 1) % 8,
                dwell_time_s=float(dwell_time_s),
                obstacle_cells=obstacle_cells,
                include_in_accuracy_summary=True,
            )
        )
    return cases


def generated_environment_sweep_cases(
    *,
    num_environments: int = 10,
    measurement_points_per_environment: int = 24,
    rotations_per_point: int = 8,
    seed: int = 20260430,
    dwell_time_s: float = 30.0,
    intensity_min_cps_1m: float = DEFAULT_VALIDATION_INTENSITY_MIN_CPS_1M,
    intensity_max_cps_1m: float = DEFAULT_VALIDATION_INTENSITY_MAX_CPS_1M,
    blocked_fraction: float = 0.35,
    passage_width_m: float = 2.0,
) -> list[ValidationCase]:
    """Return random known-obstacle environments with point and shield sweeps."""
    rng = np.random.default_rng(int(seed))
    room_size = (10.0, 20.0, 10.0)
    cases: list[ValidationCase] = []
    for env_index in range(max(0, int(num_environments))):
        env_seed = int(rng.integers(1, 2**31 - 1))
        keep_free = (
            (0.5, 0.5),
            (room_size[0] - 0.5, room_size[1] - 0.5),
            (0.5, room_size[1] - 0.5),
            (room_size[0] - 0.5, 0.5),
            (0.5 * room_size[0], 0.5 * room_size[1]),
        )
        grid = build_obstacle_grid(
            mode="random",
            path=None,
            size_x=room_size[0],
            size_y=room_size[1],
            cell_size=1.0,
            blocked_fraction=float(blocked_fraction),
            rng_seed=env_seed,
            keep_free_points=keep_free,
            passage_width_m=float(passage_width_m),
        )
        instances = generate_manchester_obstacle_instances(
            grid,
            room_size_xyz=room_size,
            obstacle_height_m=2.0,
            rng_seed=env_seed + 17,
        )
        transport_grid = _attach_known_obstacle_transport(grid, instances)
        source_count = 1 + (env_index % 5)
        if env_index in {2, 6, 9}:
            source_count = 3 + (env_index % 3)
        isotope_program = _source_isotope_program(env_index, source_count)
        sources: list[ValidationSource] = []
        for source_index, isotope in enumerate(isotope_program):
            source_pos = _sample_nonembedded_surface_source_position(
                _surface_source_position,
                transport_grid,
                instances,
                rng,
                room_size_xyz=room_size,
                source_index=source_index,
            )
            sources.append(
                ValidationSource(
                    isotope=str(isotope),
                    position_xyz=source_pos,
                    intensity_cps_1m=float(
                        rng.uniform(
                            float(intensity_min_cps_1m),
                            float(intensity_max_cps_1m),
                        )
                    ),
                )
            )
        source_tuple = tuple(sources)
        template_counts: dict[str, int] = {}
        for instance in instances:
            template_counts[instance.template] = (
                template_counts.get(instance.template, 0) + 1
            )
        template_summary = ",".join(
            f"{template}:{count}" for template, count in sorted(template_counts.items())
        )
        measurement_points: list[tuple[float, float, float]] = []
        for point_index in range(max(0, int(measurement_points_per_environment))):
            measurement_points.append(
                _sample_detector_with_tau_mix(
                    transport_grid,
                    source_tuple,
                    rng,
                    room_size_xyz=room_size,
                    prefer_obstacle_crossing=(point_index % 3 != 0),
                )
            )
        for point_index, detector in enumerate(measurement_points):
            max_tau = _obstacle_tau_for_sources(transport_grid, detector, source_tuple)
            for rotation_index in range(max(1, int(rotations_per_point))):
                cases.append(
                    ValidationCase(
                        name=f"env{env_index:02d}_pt{point_index:02d}_rot{rotation_index:02d}",
                        description=(
                            "Random known-material obstacle environment; "
                            f"templates={template_summary}; "
                            f"max_obstacle_tau={max_tau:.3f}"
                        ),
                        detector_pose_xyz=detector,
                        sources=source_tuple,
                        fe_index=int(
                            (rotation_index + 2 * point_index + env_index) % 8
                        ),
                        pb_index=int(
                            (3 * rotation_index + point_index + 2 * env_index) % 8
                        ),
                        dwell_time_s=float(dwell_time_s),
                        obstacle_cells=tuple(grid.blocked_cells),
                        obstacle_instances=instances,
                        include_in_accuracy_summary=True,
                    )
                )
    return cases


def _generated_multi_isotope_source_case(
    case_index: int,
    rng: np.random.Generator,
    *,
    dwell_time_s: float,
    intensity_min_cps_1m: float,
    intensity_max_cps_1m: float,
    sources_per_isotope: int,
    blocked_fraction: float,
    passage_width_m: float,
    min_isotope_target_counts: float,
    detector_attempts: int,
    runtime_config: dict[str, Any] | None,
    mu_by_isotope: dict[str, object] | None,
    screen_all_shield_pairs: bool = False,
    detector_selection_mode: str = "balanced",
) -> ValidationCase:
    """Return one random multi-isotope validation case."""
    room_size = (10.0, 20.0, 10.0)
    source_replicates = max(2, int(sources_per_isotope))
    env_seed = int(rng.integers(1, 2**31 - 1))
    keep_free = (
        (0.5, 0.5),
        (room_size[0] - 0.5, room_size[1] - 0.5),
        (0.5, room_size[1] - 0.5),
        (room_size[0] - 0.5, 0.5),
        (0.5 * room_size[0], 0.5 * room_size[1]),
    )
    grid = build_obstacle_grid(
        mode="random",
        path=None,
        size_x=room_size[0],
        size_y=room_size[1],
        cell_size=1.0,
        blocked_fraction=float(blocked_fraction),
        rng_seed=env_seed,
        keep_free_points=keep_free,
        passage_width_m=float(passage_width_m),
    )
    instances = generate_manchester_obstacle_instances(
        grid,
        room_size_xyz=room_size,
        obstacle_height_m=2.0,
        rng_seed=env_seed + 17,
    )
    transport_grid = _attach_known_obstacle_transport(grid, instances)
    sources: list[ValidationSource] = []
    source_index = 0
    isotope_order = list(ISOTOPES)
    rng.shuffle(isotope_order)
    for isotope in isotope_order:
        for _ in range(source_replicates):
            source_pos = _sample_nonembedded_surface_source_position(
                _multi_isotope_surface_source_position,
                transport_grid,
                instances,
                rng,
                room_size_xyz=room_size,
                source_index=source_index,
            )
            sources.append(
                ValidationSource(
                    isotope=str(isotope),
                    position_xyz=source_pos,
                    intensity_cps_1m=float(
                        rng.uniform(
                            float(intensity_min_cps_1m),
                            float(intensity_max_cps_1m),
                        )
                    ),
                )
            )
            source_index += 1
    source_tuple = tuple(sources)
    # Cycle through all Fe/Pb orientation pairs so large validation sweeps
    # calibrate the same pair space that DSS-PP can select at runtime.  In
    # all-pair sweep mode, the base scenario uses pair 0 only for detector
    # screening and is expanded to all 64 pairs afterwards.
    fe_index = 0 if bool(screen_all_shield_pairs) else int(case_index % 8)
    pb_index = 0 if bool(screen_all_shield_pairs) else int((case_index // 8) % 8)
    requested_selection_mode = str(detector_selection_mode).strip().lower()
    effective_selection_mode = _effective_detector_selection_mode(
        requested_selection_mode,
        case_index,
    )
    can_score_target = runtime_config is not None and mu_by_isotope is not None
    require_target = float(min_isotope_target_counts) > 0.0 and can_score_target
    base_screen_case = ValidationCase(
        name=f"multi_iso_{case_index:04d}",
        description="target-screening-kernel",
        detector_pose_xyz=(0.0, 0.0, 0.0),
        sources=source_tuple,
        fe_index=fe_index,
        pb_index=pb_index,
        dwell_time_s=float(dwell_time_s),
        obstacle_cells=tuple(grid.blocked_cells),
        obstacle_instances=instances,
        include_in_accuracy_summary=False,
    )
    target_kernel = (
        kernel_for_case(
            base_screen_case,
            runtime_config,
            mu_by_isotope,
        )
        if can_score_target
        else None
    )
    if target_kernel is not None and bool(screen_all_shield_pairs):
        target_kernel = _enable_batched_pair_screening(
            target_kernel,
            runtime_config,
        )
    candidates: list[DetectorPoseCandidate] = []
    selection_pair_ids = _selection_screen_pair_ids(
        mode=effective_selection_mode,
        screen_all_shield_pairs=bool(screen_all_shield_pairs),
    )
    if effective_selection_mode == "balanced":
        max_detector_attempts = max(1, int(detector_attempts)) if require_target else 1
    else:
        max_detector_attempts = max(1, int(detector_attempts))
    for detector_attempt in range(max_detector_attempts):
        detector_candidate = _sample_detector_with_tau_mix(
            transport_grid,
            source_tuple,
            rng,
            room_size_xyz=room_size,
            prefer_obstacle_crossing=_detector_sampling_prefers_obstacle(
                effective_selection_mode,
                case_index,
                detector_attempt,
            ),
        )
        features = _detector_selection_features(
            base_screen_case,
            detector_candidate,
            transport_grid=transport_grid,
            runtime_config=runtime_config if can_score_target else None,
            target_kernel=target_kernel if can_score_target else None,
            evaluate_all_shield_pairs=bool(
                screen_all_shield_pairs
                or effective_selection_mode == "shield_dynamic_range"
            ),
            selection_pair_ids=selection_pair_ids,
            use_fast_proxy=(effective_selection_mode != "balanced"),
        )
        min_count = float(features.get("min_isotope_target", 0.0))
        score = _detector_selection_score(
            features,
            mode=effective_selection_mode,
            min_target_counts=float(min_isotope_target_counts)
            if require_target
            else 0.0,
        )
        candidates.append(
            DetectorPoseCandidate(
                pose_xyz=detector_candidate,
                proxy_score=float(score),
                proxy_features=dict(features),
            )
        )
        if (
            effective_selection_mode == "balanced"
            and require_target
            and min_count >= float(min_isotope_target_counts)
        ):
            break
    if not candidates:
        raise RuntimeError("Detector-pose sampling produced no candidates.")
    exact_validation_attempts = 0
    target_gate_passed: bool | None = None
    if require_target:
        if target_kernel is None or runtime_config is None:
            raise RuntimeError("Exact target screening requires a runtime kernel.")
        selected, best_features, exact_validation_attempts = (
            _select_exact_target_qualified_detector(
                candidates,
                base_case=base_screen_case,
                transport_grid=transport_grid,
                runtime_config=runtime_config,
                target_kernel=target_kernel,
                min_target_counts=float(min_isotope_target_counts),
                all_shield_pairs=bool(screen_all_shield_pairs),
            )
        )
        target_gate_passed = True
    else:
        selected = max(candidates, key=lambda item: item.proxy_score)
        best_features = dict(selected.proxy_features)
    best_detector = selected.pose_xyz
    best_score = float(selected.proxy_score)
    best_min_target = float(best_features.get("min_isotope_target", 0.0))
    if require_target and best_min_target < float(min_isotope_target_counts):
        raise AssertionError(
            "Exact detector-pose gate returned a below-threshold candidate."
        )
    template_counts: dict[str, int] = {}
    for instance in instances:
        template_counts[instance.template] = (
            template_counts.get(instance.template, 0) + 1
        )
    template_summary = ",".join(
        f"{template}:{count}" for template, count in sorted(template_counts.items())
    )
    max_tau = _obstacle_tau_for_sources(transport_grid, best_detector, source_tuple)
    min_target_text = (
        f"; min_isotope_target={best_min_target:.1f}" if require_target else ""
    )
    detector_metadata: dict[str, Any] = {
        "detector_selection_requested_mode": requested_selection_mode,
        "detector_selection_effective_mode": effective_selection_mode,
        "detector_selection_score": float(best_score),
        "detector_selection_attempts": int(max_detector_attempts),
        "detector_selection_preselection_features": dict(selected.proxy_features),
        "detector_selection_exact_validation_attempts": int(exact_validation_attempts),
        "detector_selection_target_gate_passed": target_gate_passed,
        "detector_selection_exact_pair_count": (
            64
            if require_target and screen_all_shield_pairs
            else (1 if require_target else 0)
        ),
        "detector_selection_batch_backend": (
            None
            if target_kernel is None or not screen_all_shield_pairs
            else {
                "use_gpu_kernel": bool(target_kernel.use_gpu),
                "device": str(target_kernel.gpu_device),
                "dtype": str(target_kernel.gpu_dtype),
            }
        ),
    }
    detector_metadata.update(best_features)
    return ValidationCase(
        name=f"multi_iso_{case_index:04d}",
        description=(
            f"{source_replicates} sources per isotope; "
            f"detector_selection={requested_selection_mode}->{effective_selection_mode}; "
            f"templates={template_summary}; max_obstacle_tau={max_tau:.3f}"
            f"; count_imbalance={float(best_features.get('target_count_imbalance', 1.0)):.3f}"
            f"; shield_dynamic_range={float(best_features.get('shield_dynamic_range', 1.0)):.3f}"
            f"{min_target_text}"
        ),
        detector_pose_xyz=best_detector,
        sources=source_tuple,
        fe_index=fe_index,
        pb_index=pb_index,
        dwell_time_s=float(dwell_time_s),
        obstacle_cells=tuple(grid.blocked_cells),
        obstacle_instances=instances,
        include_in_accuracy_summary=True,
        generation_metadata=detector_metadata,
    )


def _min_expected_count_over_shield_pairs(
    case: ValidationCase,
    runtime_config: dict[str, Any],
    kernel: ContinuousKernel,
) -> float:
    """Return the minimum isotope target count over all 64 Fe/Pb pairs."""
    count_matrix = _expected_count_matrix_over_shield_pairs(
        case,
        runtime_config,
        kernel,
    )
    if not count_matrix.size:
        return 0.0
    min_count = float(np.min(count_matrix))
    return min_count if np.isfinite(min_count) else 0.0


def _expand_case_over_all_shield_pairs(
    case: ValidationCase,
) -> Iterator[ValidationCase]:
    """Yield copies of one scenario for every Fe/Pb shield orientation pair."""
    for fe_index in range(8):
        for pb_index in range(8):
            pair_id = fe_index * 8 + pb_index
            yield ValidationCase(
                name=f"{case.name}_pair{pair_id:02d}_fe{fe_index}_pb{pb_index}",
                description=f"{case.description}; shield_pair_id={pair_id}",
                detector_pose_xyz=case.detector_pose_xyz,
                sources=case.sources,
                fe_index=fe_index,
                pb_index=pb_index,
                dwell_time_s=case.dwell_time_s,
                obstacle_cells=case.obstacle_cells,
                obstacle_instances=case.obstacle_instances,
                include_in_accuracy_summary=case.include_in_accuracy_summary,
                generation_metadata=dict(case.generation_metadata),
            )


def iter_generated_multi_isotope_source_cases(
    *,
    num_cases: int = 1000,
    seed: int = 20260430,
    dwell_time_s: float = 30.0,
    intensity_min_cps_1m: float = DEFAULT_VALIDATION_INTENSITY_MIN_CPS_1M,
    intensity_max_cps_1m: float = DEFAULT_VALIDATION_INTENSITY_MAX_CPS_1M,
    sources_per_isotope: int = 2,
    blocked_fraction: float = 0.35,
    passage_width_m: float = 2.0,
    min_isotope_target_counts: float = 0.0,
    detector_attempts: int = 24,
    runtime_config: dict[str, Any] | None = None,
    mu_by_isotope: dict[str, object] | None = None,
    all_shield_pairs_per_scenario: bool = False,
    detector_selection_mode: str = "balanced",
) -> Iterator[ValidationCase]:
    """Yield cases where Cs-137, Co-60, and Eu-154 each have multiple sources."""
    rng = np.random.default_rng(int(seed))
    for case_index in range(max(0, int(num_cases))):
        case = _generated_multi_isotope_source_case(
            case_index,
            rng,
            dwell_time_s=float(dwell_time_s),
            intensity_min_cps_1m=float(intensity_min_cps_1m),
            intensity_max_cps_1m=float(intensity_max_cps_1m),
            sources_per_isotope=int(sources_per_isotope),
            blocked_fraction=float(blocked_fraction),
            passage_width_m=float(passage_width_m),
            min_isotope_target_counts=float(min_isotope_target_counts),
            detector_attempts=int(detector_attempts),
            runtime_config=runtime_config,
            mu_by_isotope=mu_by_isotope,
            screen_all_shield_pairs=bool(all_shield_pairs_per_scenario),
            detector_selection_mode=str(detector_selection_mode),
        )
        if (case_index + 1) % 100 == 0:
            print(
                f"generated {case_index + 1}/{int(num_cases)} "
                "multi-isotope validation cases",
                flush=True,
            )
        if bool(all_shield_pairs_per_scenario):
            yield from _expand_case_over_all_shield_pairs(case)
        else:
            yield case


def generated_multi_isotope_source_cases(
    *,
    num_cases: int = 1000,
    seed: int = 20260430,
    dwell_time_s: float = 30.0,
    intensity_min_cps_1m: float = DEFAULT_VALIDATION_INTENSITY_MIN_CPS_1M,
    intensity_max_cps_1m: float = DEFAULT_VALIDATION_INTENSITY_MAX_CPS_1M,
    sources_per_isotope: int = 2,
    blocked_fraction: float = 0.35,
    passage_width_m: float = 2.0,
    min_isotope_target_counts: float = 0.0,
    detector_attempts: int = 24,
    runtime_config: dict[str, Any] | None = None,
    mu_by_isotope: dict[str, object] | None = None,
    all_shield_pairs_per_scenario: bool = False,
    detector_selection_mode: str = "balanced",
) -> list[ValidationCase]:
    """Return cases where Cs-137, Co-60, and Eu-154 each have multiple sources."""
    return list(
        iter_generated_multi_isotope_source_cases(
            num_cases=int(num_cases),
            seed=int(seed),
            dwell_time_s=float(dwell_time_s),
            intensity_min_cps_1m=float(intensity_min_cps_1m),
            intensity_max_cps_1m=float(intensity_max_cps_1m),
            sources_per_isotope=int(sources_per_isotope),
            blocked_fraction=float(blocked_fraction),
            passage_width_m=float(passage_width_m),
            min_isotope_target_counts=float(min_isotope_target_counts),
            detector_attempts=int(detector_attempts),
            runtime_config=runtime_config,
            mu_by_isotope=mu_by_isotope,
            all_shield_pairs_per_scenario=bool(all_shield_pairs_per_scenario),
            detector_selection_mode=str(detector_selection_mode),
        )
    )


def default_cases() -> list[ValidationCase]:
    """Return a small hand-authored compatibility set of Geant4 validation cases."""
    detector = (1.0, 1.0, 0.5)
    near_x = (2.0, 1.0, 0.5)
    near_y = (1.0, 2.0, 0.5)
    near_z = (1.0, 1.0, 1.5)
    far_y = (1.0, 3.0, 0.5)
    blocked_octant = (2.0, 2.0, 1.5)
    free_other_octant = (0.0, 2.0, 1.5)
    return [
        ValidationCase(
            name="single_cs_free",
            description="Single Cs-137 source at 1 m without shield blocking.",
            detector_pose_xyz=detector,
            sources=(ValidationSource("Cs-137", near_x, 100.0),),
        ),
        ValidationCase(
            name="single_co_free",
            description="Single Co-60 source at 1 m without shield blocking.",
            detector_pose_xyz=detector,
            sources=(ValidationSource("Co-60", near_x, 100.0),),
        ),
        ValidationCase(
            name="single_eu_free",
            description="Single Eu-154 source at 1 m without shield blocking.",
            detector_pose_xyz=detector,
            sources=(ValidationSource("Eu-154", near_x, 100.0),),
        ),
        ValidationCase(
            name="two_cs_free",
            description="Two Cs-137 sources at different distances.",
            detector_pose_xyz=detector,
            sources=(
                ValidationSource("Cs-137", near_x, 70.0),
                ValidationSource("Cs-137", far_y, 120.0),
            ),
        ),
        ValidationCase(
            name="cs_co_free",
            description="Two-isotope mixture with Cs-137 and Co-60.",
            detector_pose_xyz=detector,
            sources=(
                ValidationSource("Cs-137", near_x, 80.0),
                ValidationSource("Co-60", near_y, 80.0),
            ),
        ),
        ValidationCase(
            name="cs_eu_free",
            description="Two-isotope mixture with Cs-137 and Eu-154.",
            detector_pose_xyz=detector,
            sources=(
                ValidationSource("Cs-137", near_x, 80.0),
                ValidationSource("Eu-154", near_y, 80.0),
            ),
        ),
        ValidationCase(
            name="co_eu_free",
            description="Two-isotope mixture with Co-60 and Eu-154.",
            detector_pose_xyz=detector,
            sources=(
                ValidationSource("Co-60", near_x, 80.0),
                ValidationSource("Eu-154", near_y, 80.0),
            ),
        ),
        ValidationCase(
            name="three_isotope_balanced_free",
            description="Balanced three-isotope mixture at three directions.",
            detector_pose_xyz=detector,
            sources=(
                ValidationSource("Cs-137", near_x, 60.0),
                ValidationSource("Co-60", near_y, 60.0),
                ValidationSource("Eu-154", near_z, 60.0),
            ),
        ),
        ValidationCase(
            name="three_isotope_imbalanced_free",
            description="Three-isotope mixture with weak Co-60 and Eu-154.",
            detector_pose_xyz=detector,
            sources=(
                ValidationSource("Cs-137", near_x, 120.0),
                ValidationSource("Co-60", near_y, 35.0),
                ValidationSource("Eu-154", near_z, 25.0),
            ),
        ),
        ValidationCase(
            name="three_isotope_fe_blocked",
            description="Three-isotope mixture through the iron shield octant.",
            detector_pose_xyz=detector,
            fe_index=7,
            pb_index=0,
            sources=(
                ValidationSource("Cs-137", blocked_octant, 80.0),
                ValidationSource("Co-60", blocked_octant, 80.0),
                ValidationSource("Eu-154", blocked_octant, 80.0),
            ),
        ),
        ValidationCase(
            name="cs_two_sources_one_fe_blocked",
            description="Two Cs-137 sources with one direction blocked by Fe.",
            detector_pose_xyz=detector,
            fe_index=7,
            pb_index=0,
            sources=(
                ValidationSource("Cs-137", blocked_octant, 80.0),
                ValidationSource("Cs-137", free_other_octant, 80.0),
            ),
        ),
        ValidationCase(
            name="three_isotope_obstacle_stress",
            description="Three-isotope mixture with a concrete obstacle.",
            detector_pose_xyz=(1.0, 1.5, 0.5),
            dwell_time_s=20.0,
            obstacle_cells=((2, 1),),
            include_in_accuracy_summary=True,
            sources=(
                ValidationSource("Cs-137", (4.0, 1.5, 0.5), 80.0),
                ValidationSource("Co-60", (4.0, 1.5, 1.5), 80.0),
                ValidationSource("Eu-154", (4.0, 1.5, 0.8), 80.0),
            ),
        ),
    ]


def resolve_path(path_value: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def build_scene(case: ValidationCase, usd_path: str | None) -> SceneDescription:
    """Build a Geant4 scene for one validation case."""
    return SceneDescription(
        room_size_xyz=(10.0, 20.0, 10.0),
        obstacle_origin_xy=(0.0, 0.0),
        obstacle_cell_size_m=1.0,
        obstacle_grid_shape=(10, 20),
        obstacle_cells=[tuple(cell) for cell in case.obstacle_cells],
        obstacle_instances=case.obstacle_instances,
        author_obstacle_prims=True,
        author_room_boundary_prims=True,
        sources=[source.to_scene_source() for source in case.sources],
        usd_path=usd_path,
        use_config_usd_fallback=usd_path is not None,
    )


def spectrum_config_from_runtime_config(
    runtime_config: dict[str, Any],
) -> SpectrumConfig:
    """Build the spectrum configuration used by the runtime count extractor."""
    return build_spectrum_config_from_runtime_config(runtime_config)


def analysis_spectrum_from_observation(
    raw_spectrum: np.ndarray,
    metadata: dict[str, Any],
    decomposer: SpectralDecomposer,
) -> np.ndarray:
    """Return the pulse-height spectrum used by the runtime spectrum decomposer."""
    spectrum = np.asarray(raw_spectrum, dtype=float)
    scoring_mode = str(metadata.get("detector_scoring_mode", "")).strip().lower()
    fast_scoring = str(metadata.get("detector_fast_scoring", "")).strip().lower()
    should_fold = bool(decomposer.config.apply_incident_gamma_detector_response) and (
        scoring_mode == "incident_gamma_energy" or fast_scoring == "true"
    )
    if should_fold:
        return decomposer.fold_incident_gamma_spectrum(spectrum)
    return spectrum


def analysis_spectrum_variance_from_observation(
    raw_spectrum: np.ndarray,
    metadata: dict[str, Any],
    decomposer: SpectralDecomposer,
) -> np.ndarray | None:
    """Return the analysis-spectrum variance used by runtime count extraction."""
    raw_variance = metadata.get("spectrum_count_variance")
    if raw_variance is None:
        return None
    variance = np.asarray(raw_variance, dtype=float)
    if variance.shape != np.asarray(raw_spectrum, dtype=float).shape:
        return None
    scoring_mode = str(metadata.get("detector_scoring_mode", "")).strip().lower()
    fast_scoring = str(metadata.get("detector_fast_scoring", "")).strip().lower()
    should_fold = bool(decomposer.config.apply_incident_gamma_detector_response) and (
        scoring_mode == "incident_gamma_energy" or fast_scoring == "true"
    )
    variance = np.clip(variance, a_min=0.0, a_max=None)
    if should_fold:
        return decomposer.fold_incident_gamma_spectrum_variance(variance)
    return variance


def obstacle_grid_for_case(case: ValidationCase) -> ObstacleGrid | None:
    """Return the obstacle grid used by the analytic target for a case."""
    if not case.obstacle_cells and not case.obstacle_instances:
        return None
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(10, 20),
        blocked_cells=tuple(case.obstacle_cells),
    )
    if case.obstacle_instances:
        return _attach_known_obstacle_transport(grid, case.obstacle_instances)
    return grid


def kernel_for_case(
    case: ValidationCase,
    runtime_config: dict[str, Any],
    mu_by_isotope: dict[str, object],
) -> ContinuousKernel:
    """Build the inverse-square plus shield/obstacle attenuation target kernel."""
    model = build_runtime_observation_model(runtime_config, isotopes=ISOTOPES)
    return continuous_kernel_from_observation_model(
        model,
        obstacle_grid=obstacle_grid_for_case(case),
        use_gpu=False,
    )


def _energy_metadata_token(energy_keV: float) -> str:
    """Return the Geant4 metadata energy token for one gamma-line energy."""
    return f"e{float(energy_keV):.1f}".replace(".", "p")


def _metadata_float(metadata: dict[str, Any], key: str) -> float | None:
    """Read a finite float metadata field when present."""
    if key not in metadata:
        return None
    try:
        value = float(metadata[key])
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _metadata_vector(
    metadata: dict[str, Any],
    prefix: str,
    components: tuple[str, str, str] = ("x", "y", "z"),
) -> np.ndarray | None:
    """Read a finite metadata vector with ``prefix_component`` keys."""
    values = [
        _metadata_float(metadata, f"{prefix}_{component}") for component in components
    ]
    if any(value is None for value in values):
        return None
    return np.asarray(values, dtype=float)


def _metadata_axis_matrix(metadata: dict[str, Any], prefix: str) -> np.ndarray | None:
    """Read a 3x3 local-axis matrix from Geant4 shield metadata."""
    columns: list[np.ndarray] = []
    for axis_name in ("x", "y", "z"):
        axis = _metadata_vector(metadata, f"{prefix}_shield_axis_{axis_name}")
        if axis is None:
            return None
        columns.append(axis)
    return np.stack(columns, axis=1)


def geant4_shield_line_mu_by_isotope(
    metadata: dict[str, Any],
    kernel: ContinuousKernel,
) -> dict[str, tuple[dict[str, float], ...]]:
    """Return line-resolved Fe/Pb mu values reported by the Geant4 sidecar."""
    table = kernel.line_mu_by_isotope
    if not isinstance(table, dict):
        return {}
    result: dict[str, tuple[dict[str, float], ...]] = {}
    for isotope, entries in table.items():
        parsed: list[dict[str, float]] = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            energy = float(item.get("energy_keV", 0.0))
            token = _energy_metadata_token(energy)
            fe_mu = _metadata_float(
                metadata,
                f"geant4_mu_cm_inv_shield_fe_{isotope}_{token}",
            )
            pb_mu = _metadata_float(
                metadata,
                f"geant4_mu_cm_inv_shield_pb_{isotope}_{token}",
            )
            if fe_mu is None or pb_mu is None:
                continue
            parsed.append(
                {
                    "energy_keV": energy,
                    "weight": float(item.get("weight", 0.0)),
                    "fe": fe_mu,
                    "pb": pb_mu,
                }
            )
        if parsed:
            result[str(isotope)] = tuple(parsed)
    return result


def clone_kernel_with_line_mu(
    kernel: ContinuousKernel,
    line_mu_by_isotope: dict[str, tuple[dict[str, float], ...]],
) -> ContinuousKernel:
    """Return a ContinuousKernel with alternate line-resolved Fe/Pb coefficients."""
    return ContinuousKernel(
        mu_by_isotope=kernel.mu_by_isotope,
        shield_params=kernel.shield_params,
        obstacle_grid=kernel.obstacle_grid,
        obstacle_height_m=kernel.obstacle_height_m,
        obstacle_mu_by_isotope=kernel.obstacle_mu_by_isotope,
        obstacle_buildup_coeff=kernel.obstacle_buildup_coeff,
        detector_radius_m=kernel.detector_radius_m,
        detector_aperture_radius_m=kernel.detector_aperture_radius_m,
        detector_aperture_samples=kernel.detector_aperture_samples,
        detector_aperture_sampling=kernel.detector_aperture_sampling,
        source_extent_radius_m=kernel.source_extent_radius_m,
        source_extent_samples=kernel.source_extent_samples,
        line_mu_by_isotope=line_mu_by_isotope,
        transport_response_model=kernel.transport_response_model,
        use_gpu=False,
    )


def expected_pf_counts_with_geant4_shield_mu(
    case: ValidationCase,
    runtime_config: dict[str, Any],
    kernel: ContinuousKernel,
    metadata: dict[str, Any],
) -> dict[str, float] | None:
    """Compute PF target counts after replacing Fe/Pb mu with Geant4-reported values."""
    line_mu = geant4_shield_line_mu_by_isotope(metadata, kernel)
    if not line_mu:
        return None
    replacement_kernel = clone_kernel_with_line_mu(kernel, line_mu)
    return expected_pf_counts_with_kernel(case, runtime_config, replacement_kernel)


def weighted_mu_diagnostics(
    metadata: dict[str, Any],
    kernel: ContinuousKernel,
) -> dict[str, dict[str, float]]:
    """Return weighted Python-vs-Geant4 Fe/Pb line-mu diagnostics by isotope."""
    table = kernel.line_mu_by_isotope
    if not isinstance(table, dict):
        return {}
    diagnostics: dict[str, dict[str, float]] = {}
    for isotope, entries in table.items():
        total_weight = 0.0
        pf_fe = 0.0
        pf_pb = 0.0
        g4_fe = 0.0
        g4_pb = 0.0
        for item in entries:
            if not isinstance(item, dict):
                continue
            weight = max(float(item.get("weight", 0.0)), 0.0)
            energy = float(item.get("energy_keV", 0.0))
            token = _energy_metadata_token(energy)
            fe_mu = _metadata_float(
                metadata,
                f"geant4_mu_cm_inv_shield_fe_{isotope}_{token}",
            )
            pb_mu = _metadata_float(
                metadata,
                f"geant4_mu_cm_inv_shield_pb_{isotope}_{token}",
            )
            if fe_mu is None or pb_mu is None or weight <= 0.0:
                continue
            total_weight += weight
            pf_fe += weight * float(item.get("fe", 0.0))
            pf_pb += weight * float(item.get("pb", 0.0))
            g4_fe += weight * fe_mu
            g4_pb += weight * pb_mu
        if total_weight <= 0.0:
            continue
        pf_fe /= total_weight
        pf_pb /= total_weight
        g4_fe /= total_weight
        g4_pb /= total_weight
        diagnostics[str(isotope)] = {
            "pf_mu_fe_weighted_cm_inv": float(pf_fe),
            "pf_mu_pb_weighted_cm_inv": float(pf_pb),
            "geant4_mu_fe_weighted_cm_inv": float(g4_fe),
            "geant4_mu_pb_weighted_cm_inv": float(g4_pb),
            "pf_minus_geant4_mu_fe_rel": float(
                (pf_fe - g4_fe) / max(abs(g4_fe), 1.0e-12)
            ),
            "pf_minus_geant4_mu_pb_rel": float(
                (pf_pb - g4_pb) / max(abs(g4_pb), 1.0e-12)
            ),
        }
    return diagnostics


def expected_pf_counts(
    case: ValidationCase,
    runtime_config: dict[str, Any],
    mu_by_isotope: dict[str, object],
) -> dict[str, float]:
    """Compute runtime PF target counts including configured source scales."""
    kernel = kernel_for_case(case, runtime_config, mu_by_isotope)
    return expected_pf_counts_with_kernel(case, runtime_config, kernel)


def expected_pf_counts_with_kernel(
    case: ValidationCase,
    runtime_config: dict[str, Any],
    kernel: ContinuousKernel,
) -> dict[str, float]:
    """Compute runtime PF target counts with a prebuilt continuous kernel."""
    detector = np.asarray(case.detector_pose_xyz, dtype=float)
    counts = {isotope: 0.0 for isotope in ISOTOPES}
    for source in case.sources:
        point_source = source.to_point_source()
        source_pos = point_source.position_array()
        source_scale = measurement_source_scale_for_case(
            source.isotope,
            case,
            runtime_config,
        )
        counts[source.isotope] += (
            float(case.dwell_time_s)
            * float(source.intensity_cps_1m)
            * float(source_scale)
            * kernel.kernel_value_pair(
                source.isotope,
                detector,
                source_pos,
                int(case.fe_index),
                int(case.pb_index),
            )
        )
    return counts


def measurement_source_scale_for_case(
    isotope: str,
    case: ValidationCase,
    runtime_config: dict[str, Any],
) -> float:
    """Return the runtime source-count scale for one validation measurement."""
    pair_id = int(case.fe_index) * 8 + int(case.pb_index)
    pair_scales = runtime_config.get("measurement_scale_by_isotope_and_pair", {})
    if isinstance(pair_scales, dict):
        isotope_pair_scales = pair_scales.get(str(isotope), {})
        if isinstance(isotope_pair_scales, dict):
            value = isotope_pair_scales.get(
                str(pair_id), isotope_pair_scales.get(pair_id)
            )
            if value is not None:
                return max(float(value), 0.0)
    scales = runtime_config.get("measurement_scale_by_isotope", {})
    if isinstance(scales, dict):
        return max(float(scales.get(str(isotope), 1.0)), 0.0)
    return 1.0


def expected_pf_count_diagnostics(
    case: ValidationCase,
    runtime_config: dict[str, Any],
    mu_by_isotope: dict[str, object],
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return per-source PF target and transport-response basis diagnostics."""
    kernel = kernel_for_case(case, runtime_config, mu_by_isotope)
    model = build_runtime_observation_model(runtime_config, isotopes=ISOTOPES)
    shield_only_kernel = ContinuousKernel(
        mu_by_isotope=model.mu_by_isotope,
        shield_params=model.shield_params,
        obstacle_grid=None,
        obstacle_height_m=model.obstacle_height_m,
        obstacle_buildup_coeff=0.0,
        detector_radius_m=kernel.detector_radius_m,
        detector_aperture_radius_m=kernel.detector_aperture_radius_m,
        detector_aperture_samples=kernel.detector_aperture_samples,
        detector_aperture_sampling=kernel.detector_aperture_sampling,
        source_extent_radius_m=kernel.source_extent_radius_m,
        source_extent_samples=kernel.source_extent_samples,
        line_mu_by_isotope=kernel.line_mu_by_isotope,
        use_gpu=False,
    )
    detector = np.asarray(case.detector_pose_xyz, dtype=float)
    rows: list[dict[str, Any]] = []
    for source_index, source in enumerate(case.sources):
        source_token = f"src{int(source_index)}_{source.isotope}"
        source_pos = np.asarray(source.position_xyz, dtype=float)
        geom = finite_sphere_geometric_term(
            detector,
            source_pos,
            kernel.detector_radius_m,
        )
        source_distance_m = float(np.linalg.norm(detector - source_pos))
        shield_att = shield_only_kernel.attenuation_factor_pair(
            source.isotope,
            source_pos,
            detector,
            int(case.fe_index),
            int(case.pb_index),
        )
        obstacle_tau = 0.0
        obstacle_att = 1.0
        obstacle_tau_area = 0.0
        obstacle_att_area = 1.0
        if kernel.obstacle_grid is not None:
            obstacle_tau = kernel.obstacle_optical_depth_pair(
                source.isotope,
                source_pos,
                detector,
            )
            obstacle_att = float(np.exp(-float(obstacle_tau)))
            obstacle_att_area = kernel.obstacle_area_averaged_attenuation_pair(
                source.isotope,
                source_pos,
                detector,
            )
            obstacle_tau_area = kernel.obstacle_area_averaged_optical_depth_pair(
                source.isotope,
                source_pos,
                detector,
            )
        full_kernel = kernel.kernel_value_pair(
            source.isotope,
            detector,
            source_pos,
            int(case.fe_index),
            int(case.pb_index),
        )
        source_scale = measurement_source_scale_for_case(
            source.isotope,
            case,
            runtime_config,
        )
        transport_terms = []
        for term in kernel.transport_response_terms_pair(
            source.isotope,
            detector,
            source_pos,
            int(case.fe_index),
            int(case.pb_index),
        ):
            adjusted_counts = float(
                case.dwell_time_s
                * source.intensity_cps_1m
                * float(term.get("kernel", 0.0))
            )
            base_counts = float(
                case.dwell_time_s
                * source.intensity_cps_1m
                * float(term.get("base_kernel", term.get("kernel", 0.0)))
            )
            transport_terms.append(
                {
                    "counts": base_counts,
                    "scaled_counts": base_counts * source_scale,
                    "base_counts": base_counts,
                    "scaled_base_counts": base_counts * source_scale,
                    "adjusted_counts": adjusted_counts,
                    "scaled_adjusted_counts": adjusted_counts * source_scale,
                    "shield_tau_feature": float(term.get("shield_tau_feature", 0.0)),
                    "fe_tau_feature": float(term.get("fe_tau_feature", 0.0)),
                    "pb_tau_feature": float(term.get("pb_tau_feature", 0.0)),
                    "obstacle_tau_feature": float(
                        term.get("obstacle_tau_feature", 0.0)
                    ),
                    "distance_feature": float(term.get("distance_feature", 0.0)),
                    "distance_shield_feature": float(
                        term.get("distance_shield_feature", 0.0)
                    ),
                    "distance_fe_feature": float(term.get("distance_fe_feature", 0.0)),
                    "distance_pb_feature": float(term.get("distance_pb_feature", 0.0)),
                    "distance_obstacle_feature": float(
                        term.get("distance_obstacle_feature", 0.0)
                    ),
                    "response_factor": float(term.get("response_factor", 1.0)),
                }
            )
        full_target_counts = float(
            case.dwell_time_s * source.intensity_cps_1m * full_kernel
        )
        row = {
            "source_index": int(source_index),
            "isotope": source.isotope,
            "position_xyz": [float(value) for value in source.position_xyz],
            "source_distance_m": float(source_distance_m),
            "intensity_cps_1m": float(source.intensity_cps_1m),
            "measurement_source_scale": float(source_scale),
            "geometric_factor": float(geom),
            "shield_attenuation": float(shield_att),
            "obstacle_tau_center_ray": float(obstacle_tau),
            "obstacle_attenuation_center_ray": float(obstacle_att),
            "obstacle_tau_area_averaged": float(obstacle_tau_area),
            "obstacle_attenuation_area_averaged": float(obstacle_att_area),
            "source_extent_radius_m": float(kernel.source_extent_radius_m),
            "source_extent_samples": int(kernel.source_extent_samples),
            "full_kernel": float(full_kernel),
            "geometric_counts": float(
                case.dwell_time_s * source.intensity_cps_1m * geom
            ),
            "shield_only_counts": float(
                case.dwell_time_s * source.intensity_cps_1m * geom * shield_att
            ),
            "full_target_counts": full_target_counts,
            "scaled_full_target_counts": float(full_target_counts * source_scale),
            "transport_response_terms": transport_terms,
        }
        if metadata is not None:
            detected = _metadata_float(
                metadata,
                f"transport_detected_counts_{source_token}",
            )
            uncollided = _metadata_float(
                metadata,
                f"transport_uncollided_primary_counts_{source_token}",
            )
            interacted = _metadata_float(
                metadata,
                f"transport_interacted_primary_counts_{source_token}",
            )
            secondary = _metadata_float(
                metadata,
                f"transport_secondary_counts_{source_token}",
            )
            row.update(
                {
                    "geant4_transport_detected_counts": detected,
                    "geant4_transport_uncollided_primary_counts": uncollided,
                    "geant4_transport_interacted_primary_counts": interacted,
                    "geant4_transport_secondary_counts": secondary,
                    "geant4_transport_non_uncollided_fraction": _metadata_float(
                        metadata,
                        f"transport_non_uncollided_fraction_{source_token}",
                    ),
                    "pf_target_relative_error_vs_source_uncollided": (
                        None
                        if uncollided is None or uncollided <= 0.0
                        else (float(full_target_counts * source_scale) - uncollided)
                        / max(float(uncollided), 1.0e-12)
                    ),
                    "pf_target_relative_error_vs_source_detected": (
                        None
                        if detected is None or detected <= 0.0
                        else (float(full_target_counts * source_scale) - detected)
                        / max(float(detected), 1.0e-12)
                    ),
                }
            )
        rows.append(row)
    return rows


def source_tally_counts(metadata: dict[str, Any]) -> dict[str, float]:
    """Read native Geant4 source-equivalent tally counts from metadata."""
    return {
        isotope: float(metadata.get(f"source_equivalent_counts_{isotope}", 0.0))
        for isotope in ISOTOPES
    }


def transport_truth_counts(metadata: dict[str, Any]) -> dict[str, float]:
    """Read native Geant4 isotope-labeled counts after transport to the detector."""
    return {
        isotope: float(metadata.get(f"transport_detected_counts_{isotope}", 0.0))
        for isotope in ISOTOPES
    }


def relative_error(value: float, target: float, min_target: float) -> float | None:
    """Return relative error when the target is large enough."""
    target_magnitude = abs(float(target))
    if target_magnitude == 0.0 or target_magnitude < float(min_target):
        return None
    return (float(value) - float(target)) / float(target)


def case_to_dict(case: ValidationCase) -> dict[str, Any]:
    """Return a JSON-compatible representation of a validation case."""
    return {
        "name": case.name,
        "description": case.description,
        "detector_pose_xyz": list(case.detector_pose_xyz),
        "fe_index": int(case.fe_index),
        "pb_index": int(case.pb_index),
        "dwell_time_s": float(case.dwell_time_s),
        "obstacle_cells": [list(cell) for cell in case.obstacle_cells],
        "obstacle_instances": obstacle_instances_to_dicts(case.obstacle_instances),
        "include_in_accuracy_summary": bool(case.include_in_accuracy_summary),
        "generation_metadata": case.generation_metadata,
        "sources": [
            {
                "isotope": source.isotope,
                "position_xyz": list(source.position_xyz),
                "intensity_cps_1m": float(source.intensity_cps_1m),
            }
            for source in case.sources
        ],
    }


def _finite_variance(value: object, *, fallback: float = 1.0) -> float:
    """Return a finite positive variance with a deterministic fallback."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = float(fallback)
    if not np.isfinite(result) or result <= 0.0:
        result = float(fallback)
    return float(max(result, 1.0))


def _covariance_payload(
    covariance: Mapping[str, Mapping[str, float]] | None,
    isotopes: tuple[str, ...],
    *,
    diagonal_variances: Mapping[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """Return a complete JSON-safe covariance mapping in isotope order."""
    source = covariance if isinstance(covariance, Mapping) else {}
    payload: dict[str, dict[str, float]] = {}
    for row_isotope in isotopes:
        raw_row = source.get(row_isotope, {})
        row = raw_row if isinstance(raw_row, Mapping) else {}
        payload[row_isotope] = {}
        for column_isotope in isotopes:
            fallback = (
                _finite_variance(diagonal_variances.get(row_isotope, 1.0))
                if row_isotope == column_isotope and diagonal_variances is not None
                else 0.0
            )
            try:
                value = float(row.get(column_isotope, fallback))
            except (TypeError, ValueError):
                value = float(fallback)
            if not np.isfinite(value):
                value = float(fallback)
            payload[row_isotope][column_isotope] = float(value)
    return payload


def _response_poisson_variance_stages(
    diagnostics: Mapping[str, object],
    runtime_variances: Mapping[str, float],
    projected_variances: Mapping[str, float],
) -> dict[str, dict[str, float | bool]]:
    """Return formal, runtime, and PF-projected variance stages by isotope."""
    formal_diagnostics = diagnostics.get("variances", {})
    formal_by_isotope = (
        formal_diagnostics if isinstance(formal_diagnostics, Mapping) else {}
    )
    component_diagnostics = diagnostics.get("runtime_variance_components", {})
    components_by_isotope = (
        component_diagnostics if isinstance(component_diagnostics, Mapping) else {}
    )
    stages: dict[str, dict[str, float | bool]] = {}
    for isotope in ISOTOPES:
        runtime_variance = _finite_variance(runtime_variances.get(isotope, 1.0))
        projected_variance = _finite_variance(
            projected_variances.get(isotope, runtime_variance),
            fallback=runtime_variance,
        )
        raw_components = components_by_isotope.get(isotope, {})
        components = raw_components if isinstance(raw_components, Mapping) else {}
        formal_variance = _finite_variance(
            components.get(
                "formal_input_variance",
                formal_by_isotope.get(isotope, runtime_variance),
            ),
            fallback=runtime_variance,
        )
        ceilinged_formal_variance = _finite_variance(
            components.get("formal_after_ceiling_variance", formal_variance),
            fallback=formal_variance,
        )
        tolerance = max(abs(formal_variance), 1.0) * 1.0e-12
        stages[isotope] = {
            "formal_variance": formal_variance,
            "ceilinged_formal_variance": ceilinged_formal_variance,
            "runtime_variance": runtime_variance,
            "projected_variance": projected_variance,
            "runtime_over_formal": runtime_variance / formal_variance,
            "runtime_over_ceilinged_formal": (
                runtime_variance / ceilinged_formal_variance
            ),
            "projected_over_runtime": projected_variance / runtime_variance,
            "projected_over_formal": projected_variance / formal_variance,
            "projected_over_ceilinged_formal": (
                projected_variance / ceilinged_formal_variance
            ),
            "formal_ceiling_applied": bool(
                ceilinged_formal_variance < formal_variance - tolerance
            ),
        }
    return stages


def _mahalanobis_squared(
    residual_by_isotope: Mapping[str, float],
    covariance: Mapping[str, Mapping[str, float]],
    isotopes: list[str],
) -> float | None:
    """Return a robust covariance-normalized residual quadratic form."""
    if not isotopes:
        return None
    residual = np.asarray(
        [float(residual_by_isotope[isotope]) for isotope in isotopes],
        dtype=float,
    )
    matrix = np.asarray(
        [[float(covariance[row][column]) for column in isotopes] for row in isotopes],
        dtype=float,
    )
    if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(matrix)):
        return None
    matrix = 0.5 * (matrix + matrix.T)
    try:
        inverse = np.linalg.pinv(matrix, hermitian=True)
    except np.linalg.LinAlgError:
        return None
    value = float(residual @ inverse @ residual)
    if not np.isfinite(value):
        return None
    return float(max(value, 0.0))


def _response_poisson_residual_diagnostics(
    counts: Mapping[str, float],
    truth_counts: Mapping[str, float],
    variance_stages: Mapping[str, Mapping[str, float | bool]],
    formal_covariance: Mapping[str, Mapping[str, float]],
    runtime_covariance: Mapping[str, Mapping[str, float]],
    estimator_covariance: Mapping[str, Mapping[str, float]],
    *,
    min_target: float,
) -> dict[str, Any]:
    """Build per-case residual diagnostics for covariance coverage analysis."""
    included = [
        isotope
        for isotope in ISOTOPES
        if float(truth_counts.get(isotope, 0.0)) >= float(min_target)
    ]
    residuals = {
        isotope: float(counts.get(isotope, 0.0)) - float(truth_counts.get(isotope, 0.0))
        for isotope in ISOTOPES
    }
    normalized: dict[str, dict[str, float]] = {
        "formal": {},
        "runtime": {},
        "projected": {},
    }
    formal_diagonal: dict[str, float] = {}
    runtime_diagonal: dict[str, float] = {}
    projected_diagonal: dict[str, float] = {}
    for isotope in ISOTOPES:
        stage = variance_stages[isotope]
        formal = _finite_variance(stage["formal_variance"])
        runtime = _finite_variance(stage["runtime_variance"])
        projected = _finite_variance(stage["projected_variance"])
        formal_diagonal[isotope] = formal
        runtime_diagonal[isotope] = runtime
        projected_diagonal[isotope] = projected
        normalized["formal"][isotope] = residuals[isotope] / np.sqrt(formal)
        normalized["runtime"][isotope] = residuals[isotope] / np.sqrt(runtime)
        normalized["projected"][isotope] = residuals[isotope] / np.sqrt(projected)
    diagonal_formal_covariance = _covariance_payload(
        None,
        ISOTOPES,
        diagonal_variances=formal_diagonal,
    )
    diagonal_runtime_covariance = _covariance_payload(
        None,
        ISOTOPES,
        diagonal_variances=runtime_diagonal,
    )
    diagonal_projected_covariance = _covariance_payload(
        None,
        ISOTOPES,
        diagonal_variances=projected_diagonal,
    )
    covariance_by_stage = {
        "formal_full": formal_covariance,
        "formal_diagonal": diagonal_formal_covariance,
        "runtime_full": runtime_covariance,
        "runtime_diagonal": diagonal_runtime_covariance,
        "estimator_sanitized_full": estimator_covariance,
        "projected_diagonal": diagonal_projected_covariance,
    }
    mahalanobis = {
        name: _mahalanobis_squared(residuals, covariance, included)
        for name, covariance in covariance_by_stage.items()
    }
    return {
        "truth_reference": "transport_detected_counts_by_isotope",
        "included_isotopes": included,
        "degrees_of_freedom": int(len(included)),
        "residual_counts_by_isotope": residuals,
        "normalized_residual_by_isotope": normalized,
        "mahalanobis_squared": mahalanobis,
    }


def _count_likelihood_scale_squared(
    observed_count: float,
    target_count: float,
    projected_observation_variance: float,
    spec: CountLikelihoodSpec,
) -> float:
    """Return the exact scalar scale squared used by the PF count likelihood."""
    target = max(float(target_count), 1.0e-12)
    if spec.model == "poisson":
        return target
    scale_squared = count_likelihood_variance(
        np.asarray([max(float(observed_count), 0.0)], dtype=float),
        np.asarray([target], dtype=float),
        transport_model_rel_sigma=spec.transport_model_rel_sigma,
        transport_model_abs_sigma=spec.transport_model_abs_sigma,
        spectrum_count_rel_sigma=spec.spectrum_count_rel_sigma,
        spectrum_count_abs_sigma=spec.spectrum_count_abs_sigma,
        low_count_abs_sigma=spec.low_count_abs_sigma,
        low_count_transition_counts=spec.low_count_transition_counts,
        observation_count_variance=max(
            float(projected_observation_variance),
            0.0,
        ),
        observation_count_variance_includes_counting_noise=(
            spec.observation_count_variance_includes_counting_noise
        ),
    )
    return float(np.asarray(scale_squared, dtype=float).reshape(-1)[0])


def _pf_likelihood_target_diagnostic(
    counts: Mapping[str, float],
    targets: Mapping[str, float],
    projected_variances: Mapping[str, float],
    likelihood_specs: Mapping[str, CountLikelihoodSpec],
    *,
    min_target: float,
) -> dict[str, Any]:
    """Build one truth-forward target residual diagnostic at final PF scale."""
    included = [
        isotope
        for isotope in ISOTOPES
        if float(targets.get(isotope, 0.0)) >= float(min_target)
    ]
    residuals: dict[str, float] = {}
    scale_squared_by_isotope: dict[str, float] = {}
    normalized_by_isotope: dict[str, float] = {}
    relative_scale_by_isotope: dict[str, float] = {}
    for isotope in ISOTOPES:
        observed = float(counts.get(isotope, 0.0))
        target = float(targets.get(isotope, 0.0))
        residual = observed - target
        spec = likelihood_specs[isotope]
        scale_squared = _count_likelihood_scale_squared(
            observed,
            target,
            float(projected_variances.get(isotope, 0.0)),
            spec,
        )
        scale = float(np.sqrt(max(scale_squared, 1.0e-12)))
        residuals[isotope] = residual
        scale_squared_by_isotope[isotope] = scale_squared
        normalized_by_isotope[isotope] = residual / scale
        relative_scale_by_isotope[isotope] = scale / max(abs(target), 1.0)
    squared_distance = float(
        sum(normalized_by_isotope[isotope] ** 2 for isotope in included)
    )
    return {
        "included_isotopes": included,
        "degrees_of_freedom": int(len(included)),
        "target_counts_by_isotope": {
            isotope: float(targets.get(isotope, 0.0)) for isotope in ISOTOPES
        },
        "residual_counts_by_isotope": residuals,
        "likelihood_scale_squared_by_isotope": scale_squared_by_isotope,
        "normalized_residual_by_isotope": normalized_by_isotope,
        "likelihood_scale_over_target_by_isotope": relative_scale_by_isotope,
        "diagonal_squared_distance": squared_distance,
    }


def _pf_count_likelihood_diagnostics(
    counts: Mapping[str, float],
    runtime_pf_targets: Mapping[str, float],
    geant4_mu_targets: Mapping[str, float] | None,
    projected_variances: Mapping[str, float],
    likelihood_specs: Mapping[str, CountLikelihoodSpec],
    *,
    min_target: float,
) -> dict[str, Any]:
    """Report truth-forward residuals using the complete PF count scale."""
    targets = {
        "runtime_pf_forward": _pf_likelihood_target_diagnostic(
            counts,
            runtime_pf_targets,
            projected_variances,
            likelihood_specs,
            min_target=float(min_target),
        )
    }
    if geant4_mu_targets is not None:
        targets["geant4_mu_pf_forward"] = _pf_likelihood_target_diagnostic(
            counts,
            geant4_mu_targets,
            projected_variances,
            likelihood_specs,
            min_target=float(min_target),
        )
    return {
        "scale_semantics": (
            "count_likelihood_variance output used as likelihood scale squared; "
            "for Student-t this is not its marginal variance"
        ),
        "observation_variance_semantics": (
            "PF-projected response-Poisson observation variance interpreted by "
            "each likelihood spec: additional adds candidate Poisson variance, "
            "counting_noise_inclusive replaces the observed plug-in term, and "
            "complete_statistical adds no second Poisson term; configured model "
            "discrepancy components remain separate"
        ),
        "likelihood_spec_by_isotope": {
            isotope: _count_likelihood_spec_payload(likelihood_specs[isotope])
            for isotope in ISOTOPES
        },
        "targets": targets,
    }


def run_case(
    app: Geant4Application,
    decomposer: SpectralDecomposer,
    covariance_projector: PurePFEstimator,
    likelihood_specs: Mapping[str, CountLikelihoodSpec],
    case: ValidationCase,
    step_id: int,
    runtime_config: dict[str, Any],
    mu_by_isotope: dict[str, object],
    min_target: float,
) -> tuple[dict[str, Any], np.ndarray]:
    """Run one Geant4 validation case and return metrics plus spectrum."""
    scene = build_scene(case, usd_path=runtime_config.get("validation_usd_path"))
    app.reset(scene)
    start = time.time()
    observation = app.step(
        SimulationCommand(
            step_id=step_id,
            target_pose_xyz=case.detector_pose_xyz,
            target_base_yaw_rad=0.0,
            fe_orientation_index=int(case.fe_index),
            pb_orientation_index=int(case.pb_index),
            dwell_time_s=float(case.dwell_time_s),
        )
    )
    runtime_s = time.time() - start
    raw_spectrum = np.asarray(observation.spectrum_counts, dtype=float)
    metadata = dict(observation.metadata)
    spectrum = analysis_spectrum_from_observation(
        raw_spectrum,
        metadata,
        decomposer,
    )
    spectrum_variance = analysis_spectrum_variance_from_observation(
        raw_spectrum,
        metadata,
        decomposer,
    )
    runtime_counts = RuntimeCountExtractor(decomposer).extract(
        spectrum,
        live_time_s=float(case.dwell_time_s),
        detect_threshold_abs=float(runtime_config.get("detect_threshold_abs", 50.0)),
        detect_threshold_rel=float(runtime_config.get("detect_threshold_rel", 0.0)),
        detect_threshold_rel_by_isotope={
            "Co-60": float(runtime_config.get("detect_threshold_rel_by_co60", 0.1))
        },
        min_peaks_by_isotope={"Co-60": 2, "Eu-154": 2},
        spectrum_variance=spectrum_variance,
        transport_metadata=metadata,
        transport_spectrum=raw_spectrum,
    )
    response_poisson_diagnostics = dict(
        getattr(decomposer, "last_response_poisson_diagnostics", {})
    )
    response_poisson_methods = {
        isotope: str(
            dict(response_poisson_diagnostics.get("methods", {})).get(
                isotope,
                "response_poisson",
            )
        )
        for isotope in ISOTOPES
    }
    response_poisson_counts = {
        isotope: float(runtime_counts.counts.get(isotope, 0.0)) for isotope in ISOTOPES
    }
    response_poisson_variances = {
        isotope: float(runtime_counts.variances.get(isotope, 1.0))
        for isotope in ISOTOPES
    }
    raw_formal_variances = response_poisson_diagnostics.get("variances", {})
    formal_variances = (
        raw_formal_variances if isinstance(raw_formal_variances, Mapping) else {}
    )
    formal_covariance = _covariance_payload(
        getattr(decomposer, "last_count_covariance", None),
        ISOTOPES,
        diagonal_variances={
            isotope: _finite_variance(
                formal_variances.get(
                    isotope,
                    response_poisson_variances[isotope],
                )
            )
            for isotope in ISOTOPES
        },
    )
    runtime_covariance = _covariance_payload(
        runtime_counts.covariance,
        ISOTOPES,
        diagonal_variances=response_poisson_variances,
    )
    projected_variances, estimator_covariance_raw = (
        project_runtime_observation_covariance(
            covariance_projector,
            response_poisson_counts,
            response_poisson_variances,
            runtime_covariance,
        )
    )
    estimator_covariance = _covariance_payload(
        estimator_covariance_raw,
        ISOTOPES,
        diagonal_variances=response_poisson_variances,
    )
    variance_stages = _response_poisson_variance_stages(
        response_poisson_diagnostics,
        response_poisson_variances,
        projected_variances,
    )
    photopeak_counts = decomposer.compute_photopeak_nnls_counts(
        spectrum,
        live_time_s=float(case.dwell_time_s),
        isotopes=ISOTOPES,
    )
    response_counts = decomposer.compute_response_model_counts(
        spectrum,
        isotopes=ISOTOPES,
    )
    peak_window_counts = decomposer.compute_isotope_counts_thesis(
        spectrum,
        live_time_s=float(case.dwell_time_s),
        isotopes=ISOTOPES,
    )
    target_kernel = kernel_for_case(case, runtime_config, mu_by_isotope)
    target_counts = expected_pf_counts_with_kernel(case, runtime_config, target_kernel)
    target_counts_geant4_mu = expected_pf_counts_with_geant4_shield_mu(
        case,
        runtime_config,
        target_kernel,
        metadata,
    )
    pf_count_likelihood_diagnostics = _pf_count_likelihood_diagnostics(
        response_poisson_counts,
        target_counts,
        target_counts_geant4_mu,
        projected_variances,
        likelihood_specs,
        min_target=float(min_target),
    )
    shield_contrast_config = runtime_config.get(
        "pf_shield_contrast_likelihood",
        {},
    )
    shield_ratio_config = runtime_config.get(
        "pf_shield_view_ratio_likelihood",
        {},
    )
    pf_count_likelihood_diagnostics["runtime_likelihood_guards"] = {
        "observation_count_variance_semantics": str(
            runtime_config.get("pf_observation_count_variance_semantics", "")
        ),
        "direct_spectrum_likelihood_enable": bool(
            runtime_config.get("pf_direct_spectrum_likelihood_enable", True)
        ),
        "shield_contrast_likelihood_enable": bool(
            shield_contrast_config.get("enabled", False)
            if isinstance(shield_contrast_config, Mapping)
            else False
        ),
        "shield_view_ratio_likelihood_enable": bool(
            shield_ratio_config.get("enabled", False)
            if isinstance(shield_ratio_config, Mapping)
            else False
        ),
    }
    line_mu_diagnostics = weighted_mu_diagnostics(metadata, target_kernel)
    target_diagnostics = expected_pf_count_diagnostics(
        case,
        runtime_config,
        mu_by_isotope,
        metadata,
    )
    tally_counts = source_tally_counts(dict(observation.metadata))
    truth_counts = transport_truth_counts(dict(observation.metadata))
    covariance_residual_diagnostics = _response_poisson_residual_diagnostics(
        response_poisson_counts,
        truth_counts,
        variance_stages,
        formal_covariance,
        runtime_covariance,
        estimator_covariance,
        min_target=float(min_target),
    )
    uncollided_primary_counts = {
        isotope: float(
            metadata.get(f"transport_uncollided_primary_counts_{isotope}", 0.0)
        )
        for isotope in ISOTOPES
    }
    interacted_primary_counts = {
        isotope: float(
            metadata.get(f"transport_interacted_primary_counts_{isotope}", 0.0)
        )
        for isotope in ISOTOPES
    }
    secondary_counts = {
        isotope: float(metadata.get(f"transport_secondary_counts_{isotope}", 0.0))
        for isotope in ISOTOPES
    }
    non_uncollided_fraction = {
        isotope: float(
            metadata.get(f"transport_non_uncollided_fraction_{isotope}", 0.0)
        )
        for isotope in ISOTOPES
    }
    orientation_count = int(len(target_kernel.orientations))
    pf_fe_normal = -np.asarray(
        target_kernel.orientations[int(case.fe_index) % orientation_count],
        dtype=float,
    )
    pf_pb_normal = -np.asarray(
        target_kernel.orientations[int(case.pb_index) % orientation_count],
        dtype=float,
    )
    native_fe_normal = np.asarray(
        [
            _metadata_float(metadata, "fe_shield_normal_x"),
            _metadata_float(metadata, "fe_shield_normal_y"),
            _metadata_float(metadata, "fe_shield_normal_z"),
        ],
        dtype=object,
    )
    native_pb_normal = np.asarray(
        [
            _metadata_float(metadata, "pb_shield_normal_x"),
            _metadata_float(metadata, "pb_shield_normal_y"),
            _metadata_float(metadata, "pb_shield_normal_z"),
        ],
        dtype=object,
    )
    native_fe_float = (
        None
        if any(value is None for value in native_fe_normal)
        else np.asarray(native_fe_normal, dtype=float)
    )
    native_pb_float = (
        None
        if any(value is None for value in native_pb_normal)
        else np.asarray(native_pb_normal, dtype=float)
    )
    pf_fe_axes = rotation_matrix_between_vectors(
        LOCAL_POSITIVE_OCTANT_CENTER,
        pf_fe_normal,
    )
    pf_pb_axes = rotation_matrix_between_vectors(
        LOCAL_POSITIVE_OCTANT_CENTER,
        pf_pb_normal,
    )
    native_fe_axes = _metadata_axis_matrix(metadata, "fe")
    native_pb_axes = _metadata_axis_matrix(metadata, "pb")
    kernel_diagnostics = {
        "pf_detector_count_radius_m": float(target_kernel.detector_radius_m),
        "pf_detector_aperture_radius_m": float(
            target_kernel.detector_aperture_radius_m or 0.0
        ),
        "pf_detector_aperture_samples": int(target_kernel.detector_aperture_samples),
        "pf_source_extent_radius_m": float(target_kernel.source_extent_radius_m),
        "pf_source_extent_samples": int(target_kernel.source_extent_samples),
        "native_detector_crystal_radius_m": _metadata_float(
            metadata,
            "detector_crystal_radius_m",
        ),
        "native_detector_housing_thickness_m": _metadata_float(
            metadata,
            "detector_housing_thickness_m",
        ),
        "native_detector_target_radius_m": _metadata_float(
            metadata,
            "detector_target_radius_m",
        ),
        "native_reference_detector_acceptance": _metadata_float(
            metadata,
            "reference_detector_acceptance",
        ),
        "pf_expected_fe_normal_x": float(pf_fe_normal[0]),
        "pf_expected_fe_normal_y": float(pf_fe_normal[1]),
        "pf_expected_fe_normal_z": float(pf_fe_normal[2]),
        "pf_expected_pb_normal_x": float(pf_pb_normal[0]),
        "pf_expected_pb_normal_y": float(pf_pb_normal[1]),
        "pf_expected_pb_normal_z": float(pf_pb_normal[2]),
        "native_fe_shield_normal_x": _metadata_float(metadata, "fe_shield_normal_x"),
        "native_fe_shield_normal_y": _metadata_float(metadata, "fe_shield_normal_y"),
        "native_fe_shield_normal_z": _metadata_float(metadata, "fe_shield_normal_z"),
        "native_pb_shield_normal_x": _metadata_float(metadata, "pb_shield_normal_x"),
        "native_pb_shield_normal_y": _metadata_float(metadata, "pb_shield_normal_y"),
        "native_pb_shield_normal_z": _metadata_float(metadata, "pb_shield_normal_z"),
        "fe_shield_normal_dot": (
            None
            if native_fe_float is None
            else float(np.dot(pf_fe_normal, native_fe_float))
        ),
        "pb_shield_normal_dot": (
            None
            if native_pb_float is None
            else float(np.dot(pf_pb_normal, native_pb_float))
        ),
        "fe_shield_normal_max_abs_delta": (
            None
            if native_fe_float is None
            else float(np.max(np.abs(pf_fe_normal - native_fe_float)))
        ),
        "pb_shield_normal_max_abs_delta": (
            None
            if native_pb_float is None
            else float(np.max(np.abs(pf_pb_normal - native_pb_float)))
        ),
        "fe_shield_axis_max_abs_delta": (
            None
            if native_fe_axes is None
            else float(np.max(np.abs(pf_fe_axes - native_fe_axes)))
        ),
        "pb_shield_axis_max_abs_delta": (
            None
            if native_pb_axes is None
            else float(np.max(np.abs(pf_pb_axes - native_pb_axes)))
        ),
        "fe_shield_axis_x_dot": (
            None
            if native_fe_axes is None
            else float(np.dot(pf_fe_axes[:, 0], native_fe_axes[:, 0]))
        ),
        "fe_shield_axis_y_dot": (
            None
            if native_fe_axes is None
            else float(np.dot(pf_fe_axes[:, 1], native_fe_axes[:, 1]))
        ),
        "fe_shield_axis_z_dot": (
            None
            if native_fe_axes is None
            else float(np.dot(pf_fe_axes[:, 2], native_fe_axes[:, 2]))
        ),
        "pb_shield_axis_x_dot": (
            None
            if native_pb_axes is None
            else float(np.dot(pf_pb_axes[:, 0], native_pb_axes[:, 0]))
        ),
        "pb_shield_axis_y_dot": (
            None
            if native_pb_axes is None
            else float(np.dot(pf_pb_axes[:, 1], native_pb_axes[:, 1]))
        ),
        "pb_shield_axis_z_dot": (
            None
            if native_pb_axes is None
            else float(np.dot(pf_pb_axes[:, 2], native_pb_axes[:, 2]))
        ),
    }
    methods = {
        "response_poisson": response_poisson_counts,
        "photopeak_nnls": photopeak_counts,
        "response_matrix": response_counts,
        "peak_window": peak_window_counts,
    }
    per_isotope: dict[str, dict[str, Any]] = {}
    for isotope in ISOTOPES:
        target = float(target_counts.get(isotope, 0.0))
        target_geant4_mu = (
            None
            if target_counts_geant4_mu is None
            else float(target_counts_geant4_mu.get(isotope, 0.0))
        )
        per_isotope[isotope] = {
            "target_pf_counts": target,
            "target_pf_counts_geant4_shield_mu": target_geant4_mu,
            "line_mu_diagnostics": dict(line_mu_diagnostics.get(isotope, {})),
            "target_pf_count_diagnostics": [
                row for row in target_diagnostics if row["isotope"] == isotope
            ],
            "source_tally_counts": float(tally_counts.get(isotope, 0.0)),
            "transport_truth_counts": float(truth_counts.get(isotope, 0.0)),
            "transport_uncollided_primary_counts": float(
                uncollided_primary_counts.get(isotope, 0.0)
            ),
            "transport_interacted_primary_counts": float(
                interacted_primary_counts.get(isotope, 0.0)
            ),
            "transport_secondary_counts": float(secondary_counts.get(isotope, 0.0)),
            "transport_non_uncollided_fraction": float(
                non_uncollided_fraction.get(isotope, 0.0)
            ),
            "method_counts": {
                method: float(values.get(isotope, 0.0))
                for method, values in methods.items()
            },
            "response_poisson_formal_variance": float(
                variance_stages[isotope]["formal_variance"]
            ),
            "response_poisson_ceilinged_formal_variance": float(
                variance_stages[isotope]["ceilinged_formal_variance"]
            ),
            "response_poisson_variance": float(
                response_poisson_variances.get(isotope, 1.0)
            ),
            "response_poisson_projected_variance": float(
                projected_variances.get(
                    isotope,
                    response_poisson_variances.get(isotope, 1.0),
                )
            ),
            "response_poisson_variance_stages": dict(variance_stages[isotope]),
            "response_poisson_method_name": str(
                response_poisson_methods.get(isotope, "response_poisson")
            ),
            "relative_errors": {
                method: relative_error(
                    float(values.get(isotope, 0.0)),
                    target,
                    min_target,
                )
                for method, values in methods.items()
            },
            "relative_errors_vs_transport_truth": {
                method: relative_error(
                    float(values.get(isotope, 0.0)),
                    float(truth_counts.get(isotope, 0.0)),
                    min_target,
                )
                for method, values in methods.items()
            },
            "pf_target_relative_error_vs_transport_truth": relative_error(
                target,
                float(truth_counts.get(isotope, 0.0)),
                min_target,
            ),
            "pf_target_relative_error_vs_uncollided_primary": relative_error(
                target,
                float(uncollided_primary_counts.get(isotope, 0.0)),
                min_target,
            ),
            "pf_geant4_mu_target_relative_error_vs_transport_truth": (
                None
                if target_geant4_mu is None
                else relative_error(
                    float(target_geant4_mu),
                    float(truth_counts.get(isotope, 0.0)),
                    min_target,
                )
            ),
            "pf_geant4_mu_target_relative_error_vs_uncollided_primary": (
                None
                if target_geant4_mu is None
                else relative_error(
                    float(target_geant4_mu),
                    float(uncollided_primary_counts.get(isotope, 0.0)),
                    min_target,
                )
            ),
        }
    result = {
        "case": case_to_dict(case),
        "runtime_s": float(runtime_s),
        "raw_total_spectrum_counts": float(np.sum(raw_spectrum)),
        "total_spectrum_counts": float(np.sum(spectrum)),
        "num_primaries": float(observation.metadata.get("num_primaries", 0.0)),
        "metadata": validation_observation_metadata(observation.metadata),
        "kernel_diagnostics": kernel_diagnostics,
        "target_diagnostics": target_diagnostics,
        "response_poisson_diagnostics": response_poisson_diagnostics,
        "response_poisson_covariance": {
            "isotope_order": list(ISOTOPES),
            "formal": formal_covariance,
            "runtime": runtime_covariance,
            "estimator_sanitized": estimator_covariance,
            "projected_variance_by_isotope": {
                isotope: float(projected_variances[isotope]) for isotope in ISOTOPES
            },
            "projection_config": {
                "enabled": bool(
                    covariance_projector.pf_config.observation_covariance_projection_enable
                ),
                "weight": float(
                    covariance_projector.pf_config.observation_covariance_projection_weight
                ),
                "max_correlation": float(
                    covariance_projector.pf_config.observation_covariance_projection_max_corr
                ),
            },
            "residual_diagnostics": covariance_residual_diagnostics,
        },
        "pf_count_likelihood_diagnostics": pf_count_likelihood_diagnostics,
        "per_isotope": per_isotope,
    }
    return result, spectrum


def flatten_records(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten nested validation results into CSV records."""
    rows: list[dict[str, Any]] = []
    for result in results:
        case = result["case"]
        response_poisson_diagnostics = dict(
            result.get("response_poisson_diagnostics", {})
        )
        background_anchor = dict(
            response_poisson_diagnostics.get("background_anchor", {})
        )
        line_model_selection = dict(
            response_poisson_diagnostics.get("line_model_selection", {})
        )
        crosstalk_guard = dict(
            response_poisson_diagnostics.get("crosstalk_count_guard", {})
        )
        low_snr_suppression = dict(
            response_poisson_diagnostics.get("low_snr_photopeak_suppression", {})
        )
        response_counts = dict(response_poisson_diagnostics.get("counts", {}))
        response_coefficients = dict(
            response_poisson_diagnostics.get("coefficients", {})
        )
        response_photopeak_counts = dict(
            response_poisson_diagnostics.get("photopeak_counts", {})
        )
        response_photopeak_variances = dict(
            response_poisson_diagnostics.get("photopeak_variances", {})
        )
        kernel_diagnostics = dict(result.get("kernel_diagnostics", {}))
        coefficient_corr_by_isotope = dict(
            response_poisson_diagnostics.get(
                "coefficient_correlation_by_isotope",
                {},
            )
        )
        covariance_payload = dict(result.get("response_poisson_covariance", {}))
        formal_covariance = dict(covariance_payload.get("formal", {}))
        runtime_covariance = dict(covariance_payload.get("runtime", {}))
        estimator_covariance = dict(covariance_payload.get("estimator_sanitized", {}))
        residual_diagnostics = dict(covariance_payload.get("residual_diagnostics", {}))
        case_mahalanobis = dict(residual_diagnostics.get("mahalanobis_squared", {}))
        pf_likelihood_payload = dict(result.get("pf_count_likelihood_diagnostics", {}))
        pf_likelihood_targets = dict(pf_likelihood_payload.get("targets", {}))
        runtime_pf_likelihood = dict(
            pf_likelihood_targets.get("runtime_pf_forward", {})
        )
        geant4_mu_pf_likelihood = dict(
            pf_likelihood_targets.get("geant4_mu_pf_forward", {})
        )
        pf_likelihood_specs = dict(
            pf_likelihood_payload.get("likelihood_spec_by_isotope", {})
        )
        for isotope, item in result["per_isotope"].items():
            line_mu_diagnostics = dict(item.get("line_mu_diagnostics", {}))
            transport_feature_diagnostics = _weighted_pf_transport_feature_diagnostics(
                item.get("target_pf_count_diagnostics", []),
                detector_xyz=case.get("detector_pose_xyz"),
            )
            guard_item = dict(crosstalk_guard.get(isotope, {}))
            low_snr_item = dict(low_snr_suppression.get(isotope, {}))
            uncollided_primary_counts = float(
                item.get("transport_uncollided_primary_counts", 0.0)
            )
            interacted_primary_counts = float(
                item.get("transport_interacted_primary_counts", 0.0)
            )
            secondary_counts = float(item.get("transport_secondary_counts", 0.0))
            non_uncollided_fraction = float(
                item.get("transport_non_uncollided_fraction", 0.0)
            )
            for method, value in item["method_counts"].items():
                rel_err = item["relative_errors"][method]
                rows.append(
                    {
                        "case": case["name"],
                        "description": case["description"],
                        "include_in_accuracy_summary": case[
                            "include_in_accuracy_summary"
                        ],
                        "isotope": isotope,
                        "method": method,
                        "target_pf_counts": item["target_pf_counts"],
                        "target_pf_counts_geant4_shield_mu": (
                            ""
                            if item.get("target_pf_counts_geant4_shield_mu") is None
                            else item.get("target_pf_counts_geant4_shield_mu")
                        ),
                        "source_tally_counts": item["source_tally_counts"],
                        "transport_truth_counts": item["transport_truth_counts"],
                        "transport_uncollided_primary_counts": uncollided_primary_counts,
                        "transport_interacted_primary_counts": interacted_primary_counts,
                        "transport_secondary_counts": secondary_counts,
                        "transport_non_uncollided_fraction": non_uncollided_fraction,
                        "estimated_counts": value,
                        "estimated_variance": (
                            item["response_poisson_variance"]
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_formal_variance": (
                            item.get("response_poisson_formal_variance", "")
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_ceilinged_formal_variance": (
                            item.get(
                                "response_poisson_ceilinged_formal_variance",
                                "",
                            )
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_runtime_variance": (
                            item.get("response_poisson_variance", "")
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_projected_variance": (
                            item.get("response_poisson_projected_variance", "")
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_variance_stages": (
                            json.dumps(
                                item.get("response_poisson_variance_stages", {}),
                                sort_keys=True,
                            )
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_formal_covariance_row": (
                            json.dumps(
                                formal_covariance.get(isotope, {}),
                                sort_keys=True,
                            )
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_runtime_covariance_row": (
                            json.dumps(
                                runtime_covariance.get(isotope, {}),
                                sort_keys=True,
                            )
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_estimator_covariance_row": (
                            json.dumps(
                                estimator_covariance.get(isotope, {}),
                                sort_keys=True,
                            )
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_covariance_degrees_of_freedom": (
                            residual_diagnostics.get("degrees_of_freedom", "")
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_mahalanobis_squared": (
                            json.dumps(case_mahalanobis, sort_keys=True)
                            if method == "response_poisson"
                            else ""
                        ),
                        "pf_count_likelihood_spec": (
                            json.dumps(
                                pf_likelihood_specs.get(isotope, {}),
                                sort_keys=True,
                            )
                            if method == "response_poisson"
                            else ""
                        ),
                        "pf_runtime_target_likelihood_scale_squared": (
                            dict(
                                runtime_pf_likelihood.get(
                                    "likelihood_scale_squared_by_isotope",
                                    {},
                                )
                            ).get(isotope, "")
                            if method == "response_poisson"
                            else ""
                        ),
                        "pf_runtime_target_normalized_residual": (
                            dict(
                                runtime_pf_likelihood.get(
                                    "normalized_residual_by_isotope",
                                    {},
                                )
                            ).get(isotope, "")
                            if method == "response_poisson"
                            else ""
                        ),
                        "pf_geant4_mu_target_likelihood_scale_squared": (
                            dict(
                                geant4_mu_pf_likelihood.get(
                                    "likelihood_scale_squared_by_isotope",
                                    {},
                                )
                            ).get(isotope, "")
                            if method == "response_poisson"
                            else ""
                        ),
                        "pf_geant4_mu_target_normalized_residual": (
                            dict(
                                geant4_mu_pf_likelihood.get(
                                    "normalized_residual_by_isotope",
                                    {},
                                )
                            ).get(isotope, "")
                            if method == "response_poisson"
                            else ""
                        ),
                        "count_method_name": (
                            item["response_poisson_method_name"]
                            if method == "response_poisson"
                            else method
                        ),
                        "response_poisson_line_resolved_fit": (
                            response_poisson_diagnostics.get("line_resolved_fit", "")
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_signal_column_count": (
                            response_poisson_diagnostics.get("signal_column_count", "")
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_line_model_selected": (
                            line_model_selection.get("selected", "")
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_line_model_reason": (
                            line_model_selection.get("reason", "")
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_line_bic_delta": (
                            line_model_selection.get(
                                "bic_delta_line_minus_isotope",
                                "",
                            )
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_background_anchor_target": (
                            background_anchor.get("target_counts", "")
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_background_total": (
                            response_poisson_diagnostics.get(
                                "background_total_counts",
                                "",
                            )
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_reduced_chi2": (
                            response_poisson_diagnostics.get("reduced_chi2", "")
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_design_condition": (
                            response_poisson_diagnostics.get(
                                "design_condition_number",
                                "",
                            )
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_fisher_condition": (
                            response_poisson_diagnostics.get(
                                "fisher_condition_number",
                                "",
                            )
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_coeff_corr_max": (
                            response_poisson_diagnostics.get(
                                "coefficient_correlation_max_abs",
                                "",
                            )
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_coeff_corr_isotope": (
                            coefficient_corr_by_isotope.get(isotope, "")
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_raw_coefficient": (
                            response_coefficients.get(isotope, "")
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_final_count": (
                            response_counts.get(isotope, "")
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_photopeak_count": (
                            response_photopeak_counts.get(isotope, "")
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_photopeak_variance": (
                            response_photopeak_variances.get(isotope, "")
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_crosstalk_guard_reason": (
                            guard_item.get("reason", "")
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_crosstalk_guard_poisson_count": (
                            guard_item.get("poisson_count", "")
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_crosstalk_guard_photopeak_count": (
                            guard_item.get("photopeak_count", "")
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_crosstalk_guard_ratio": (
                            guard_item.get("poisson_to_photopeak_ratio", "")
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_low_snr_suppressed": (
                            low_snr_item.get("suppressed", "")
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_low_snr_reason": (
                            low_snr_item.get("reason", "")
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_low_snr_photo_count": (
                            low_snr_item.get("photo_count", "")
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_low_snr_photo_snr": (
                            low_snr_item.get("photo_snr", "")
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_low_snr_poisson_count": (
                            low_snr_item.get("poisson_count", "")
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_low_snr_poisson_snr": (
                            low_snr_item.get("poisson_snr", "")
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_low_snr_poisson_fraction": (
                            low_snr_item.get("poisson_fraction", "")
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_low_snr_photo_to_poisson_ratio": (
                            low_snr_item.get("photo_to_poisson_ratio", "")
                            if method == "response_poisson"
                            else ""
                        ),
                        "response_poisson_low_snr_predicted_photo_snr": (
                            low_snr_item.get("predicted_photo_snr", "")
                            if method == "response_poisson"
                            else ""
                        ),
                        "relative_error": "" if rel_err is None else rel_err,
                        "abs_relative_error": "" if rel_err is None else abs(rel_err),
                        "relative_error_vs_transport_truth": ""
                        if item["relative_errors_vs_transport_truth"][method] is None
                        else item["relative_errors_vs_transport_truth"][method],
                        "abs_relative_error_vs_transport_truth": ""
                        if item["relative_errors_vs_transport_truth"][method] is None
                        else abs(item["relative_errors_vs_transport_truth"][method]),
                        "pf_target_relative_error_vs_transport_truth": ""
                        if item["pf_target_relative_error_vs_transport_truth"] is None
                        else item["pf_target_relative_error_vs_transport_truth"],
                        "pf_target_relative_error_vs_uncollided_primary": ""
                        if item.get("pf_target_relative_error_vs_uncollided_primary")
                        is None
                        else item["pf_target_relative_error_vs_uncollided_primary"],
                        "pf_geant4_mu_target_relative_error_vs_transport_truth": ""
                        if item.get(
                            "pf_geant4_mu_target_relative_error_vs_transport_truth"
                        )
                        is None
                        else item[
                            "pf_geant4_mu_target_relative_error_vs_transport_truth"
                        ],
                        "pf_geant4_mu_target_relative_error_vs_uncollided_primary": ""
                        if item.get(
                            "pf_geant4_mu_target_relative_error_vs_uncollided_primary"
                        )
                        is None
                        else item[
                            "pf_geant4_mu_target_relative_error_vs_uncollided_primary"
                        ],
                        "pf_transport_shield_tau_feature": transport_feature_diagnostics.get(
                            "shield_tau_feature",
                            "",
                        ),
                        "pf_transport_fe_tau_feature": transport_feature_diagnostics.get(
                            "fe_tau_feature",
                            "",
                        ),
                        "pf_transport_pb_tau_feature": transport_feature_diagnostics.get(
                            "pb_tau_feature",
                            "",
                        ),
                        "pf_transport_obstacle_tau_feature": transport_feature_diagnostics.get(
                            "obstacle_tau_feature",
                            "",
                        ),
                        "pf_transport_distance_feature": transport_feature_diagnostics.get(
                            "distance_feature",
                            "",
                        ),
                        "pf_transport_distance_shield_feature": (
                            transport_feature_diagnostics.get(
                                "distance_shield_feature",
                                "",
                            )
                        ),
                        "pf_transport_distance_fe_feature": (
                            transport_feature_diagnostics.get(
                                "distance_fe_feature",
                                "",
                            )
                        ),
                        "pf_transport_distance_pb_feature": (
                            transport_feature_diagnostics.get(
                                "distance_pb_feature",
                                "",
                            )
                        ),
                        "pf_transport_distance_obstacle_feature": (
                            transport_feature_diagnostics.get(
                                "distance_obstacle_feature",
                                "",
                            )
                        ),
                        "pf_transport_response_factor_weighted": transport_feature_diagnostics.get(
                            "response_factor_weighted",
                            "",
                        ),
                        "pf_transport_response_factor_min": transport_feature_diagnostics.get(
                            "response_factor_min",
                            "",
                        ),
                        "pf_transport_response_factor_max": transport_feature_diagnostics.get(
                            "response_factor_max",
                            "",
                        ),
                        "pf_transport_source_count": transport_feature_diagnostics.get(
                            "source_count",
                            "",
                        ),
                        "pf_transport_source_term_count": transport_feature_diagnostics.get(
                            "source_term_count",
                            "",
                        ),
                        "pf_mu_fe_weighted_cm_inv": line_mu_diagnostics.get(
                            "pf_mu_fe_weighted_cm_inv",
                            "",
                        ),
                        "pf_mu_pb_weighted_cm_inv": line_mu_diagnostics.get(
                            "pf_mu_pb_weighted_cm_inv",
                            "",
                        ),
                        "geant4_mu_fe_weighted_cm_inv": line_mu_diagnostics.get(
                            "geant4_mu_fe_weighted_cm_inv",
                            "",
                        ),
                        "geant4_mu_pb_weighted_cm_inv": line_mu_diagnostics.get(
                            "geant4_mu_pb_weighted_cm_inv",
                            "",
                        ),
                        "pf_minus_geant4_mu_fe_rel": line_mu_diagnostics.get(
                            "pf_minus_geant4_mu_fe_rel",
                            "",
                        ),
                        "pf_minus_geant4_mu_pb_rel": line_mu_diagnostics.get(
                            "pf_minus_geant4_mu_pb_rel",
                            "",
                        ),
                        "pf_detector_count_radius_m": kernel_diagnostics.get(
                            "pf_detector_count_radius_m",
                            "",
                        ),
                        "pf_detector_aperture_radius_m": kernel_diagnostics.get(
                            "pf_detector_aperture_radius_m",
                            "",
                        ),
                        "pf_detector_aperture_samples": kernel_diagnostics.get(
                            "pf_detector_aperture_samples",
                            "",
                        ),
                        "pf_source_extent_radius_m": kernel_diagnostics.get(
                            "pf_source_extent_radius_m",
                            "",
                        ),
                        "pf_source_extent_samples": kernel_diagnostics.get(
                            "pf_source_extent_samples",
                            "",
                        ),
                        "native_detector_crystal_radius_m": kernel_diagnostics.get(
                            "native_detector_crystal_radius_m",
                            "",
                        ),
                        "native_detector_housing_thickness_m": kernel_diagnostics.get(
                            "native_detector_housing_thickness_m",
                            "",
                        ),
                        "native_detector_target_radius_m": kernel_diagnostics.get(
                            "native_detector_target_radius_m",
                            "",
                        ),
                        "native_reference_detector_acceptance": kernel_diagnostics.get(
                            "native_reference_detector_acceptance",
                            "",
                        ),
                        "pf_expected_fe_normal_x": kernel_diagnostics.get(
                            "pf_expected_fe_normal_x",
                            "",
                        ),
                        "pf_expected_fe_normal_y": kernel_diagnostics.get(
                            "pf_expected_fe_normal_y",
                            "",
                        ),
                        "pf_expected_fe_normal_z": kernel_diagnostics.get(
                            "pf_expected_fe_normal_z",
                            "",
                        ),
                        "pf_expected_pb_normal_x": kernel_diagnostics.get(
                            "pf_expected_pb_normal_x",
                            "",
                        ),
                        "pf_expected_pb_normal_y": kernel_diagnostics.get(
                            "pf_expected_pb_normal_y",
                            "",
                        ),
                        "pf_expected_pb_normal_z": kernel_diagnostics.get(
                            "pf_expected_pb_normal_z",
                            "",
                        ),
                        "native_fe_shield_normal_x": kernel_diagnostics.get(
                            "native_fe_shield_normal_x",
                            "",
                        ),
                        "native_fe_shield_normal_y": kernel_diagnostics.get(
                            "native_fe_shield_normal_y",
                            "",
                        ),
                        "native_fe_shield_normal_z": kernel_diagnostics.get(
                            "native_fe_shield_normal_z",
                            "",
                        ),
                        "native_pb_shield_normal_x": kernel_diagnostics.get(
                            "native_pb_shield_normal_x",
                            "",
                        ),
                        "native_pb_shield_normal_y": kernel_diagnostics.get(
                            "native_pb_shield_normal_y",
                            "",
                        ),
                        "native_pb_shield_normal_z": kernel_diagnostics.get(
                            "native_pb_shield_normal_z",
                            "",
                        ),
                        "fe_shield_normal_dot": kernel_diagnostics.get(
                            "fe_shield_normal_dot",
                            "",
                        ),
                        "pb_shield_normal_dot": kernel_diagnostics.get(
                            "pb_shield_normal_dot",
                            "",
                        ),
                        "fe_shield_normal_max_abs_delta": kernel_diagnostics.get(
                            "fe_shield_normal_max_abs_delta",
                            "",
                        ),
                        "pb_shield_normal_max_abs_delta": kernel_diagnostics.get(
                            "pb_shield_normal_max_abs_delta",
                            "",
                        ),
                        "fe_shield_axis_max_abs_delta": kernel_diagnostics.get(
                            "fe_shield_axis_max_abs_delta",
                            "",
                        ),
                        "pb_shield_axis_max_abs_delta": kernel_diagnostics.get(
                            "pb_shield_axis_max_abs_delta",
                            "",
                        ),
                        "fe_shield_axis_x_dot": kernel_diagnostics.get(
                            "fe_shield_axis_x_dot",
                            "",
                        ),
                        "fe_shield_axis_y_dot": kernel_diagnostics.get(
                            "fe_shield_axis_y_dot",
                            "",
                        ),
                        "fe_shield_axis_z_dot": kernel_diagnostics.get(
                            "fe_shield_axis_z_dot",
                            "",
                        ),
                        "pb_shield_axis_x_dot": kernel_diagnostics.get(
                            "pb_shield_axis_x_dot",
                            "",
                        ),
                        "pb_shield_axis_y_dot": kernel_diagnostics.get(
                            "pb_shield_axis_y_dot",
                            "",
                        ),
                        "pb_shield_axis_z_dot": kernel_diagnostics.get(
                            "pb_shield_axis_z_dot",
                            "",
                        ),
                        "total_spectrum_counts": result["total_spectrum_counts"],
                        "num_primaries": result["num_primaries"],
                        "runtime_s": result["runtime_s"],
                        "fe_index": case["fe_index"],
                        "pb_index": case["pb_index"],
                        "dwell_time_s": case["dwell_time_s"],
                    }
                )
    return rows


def _finite_distribution(values: list[float]) -> dict[str, float | int]:
    """Return compact finite distribution statistics."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return {
        "num_points": int(array.size),
        "min": float(np.min(array)) if array.size else float("nan"),
        "mean": float(np.mean(array)) if array.size else float("nan"),
        "median": float(np.median(array)) if array.size else float("nan"),
        "p90": float(np.percentile(array, 90.0)) if array.size else float("nan"),
        "p95": float(np.percentile(array, 95.0)) if array.size else float("nan"),
        "p99": float(np.percentile(array, 99.0)) if array.size else float("nan"),
        "max": float(np.max(array)) if array.size else float("nan"),
    }


def _normalized_residual_coverage(values: list[float]) -> dict[str, float | int]:
    """Return normalized-residual moments and empirical sigma coverage."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    absolute = np.abs(array)
    summary = _finite_distribution(absolute.tolist())
    summary.update(
        {
            "mean_signed": (float(np.mean(array)) if array.size else float("nan")),
            "rms": (
                float(np.sqrt(np.mean(array * array))) if array.size else float("nan")
            ),
            "coverage_within_1sigma": (
                float(np.mean(absolute <= 1.0)) if absolute.size else float("nan")
            ),
            "coverage_within_2sigma": (
                float(np.mean(absolute <= 2.0)) if absolute.size else float("nan")
            ),
            "coverage_within_3sigma": (
                float(np.mean(absolute <= 3.0)) if absolute.size else float("nan")
            ),
        }
    )
    return summary


def _covariance_summary_accumulator() -> dict[str, Any]:
    """Return an empty variance-ratio and coverage accumulator."""
    return {
        "num_isotope_records": 0,
        "num_coverage_records": 0,
        "ratios": {
            "runtime_over_formal": [],
            "runtime_over_ceilinged_formal": [],
            "projected_over_runtime": [],
            "projected_over_formal": [],
            "projected_over_ceilinged_formal": [],
        },
        "normalized_residuals": {
            "formal": [],
            "runtime": [],
            "projected": [],
        },
        "ceiling_counts": {
            "formal_ceiling_applied": 0,
            "runtime_above_ceilinged_formal": 0,
            "projected_above_ceilinged_formal": 0,
            "projected_above_runtime": 0,
        },
    }


def _accumulate_covariance_record(
    accumulator: dict[str, Any],
    *,
    count: float,
    truth: float,
    stage: Mapping[str, object],
    include_coverage: bool,
) -> None:
    """Add one isotope record to covariance ratio and coverage diagnostics."""
    formal = _finite_variance(stage.get("formal_variance", 1.0))
    ceilinged = _finite_variance(
        stage.get("ceilinged_formal_variance", formal),
        fallback=formal,
    )
    runtime = _finite_variance(
        stage.get("runtime_variance", formal),
        fallback=formal,
    )
    projected = _finite_variance(
        stage.get("projected_variance", runtime),
        fallback=runtime,
    )
    ratios = {
        "runtime_over_formal": runtime / formal,
        "runtime_over_ceilinged_formal": runtime / ceilinged,
        "projected_over_runtime": projected / runtime,
        "projected_over_formal": projected / formal,
        "projected_over_ceilinged_formal": projected / ceilinged,
    }
    for name, value in ratios.items():
        accumulator["ratios"][name].append(float(value))
    if include_coverage:
        residual = float(count) - float(truth)
        accumulator["normalized_residuals"]["formal"].append(residual / np.sqrt(formal))
        accumulator["normalized_residuals"]["runtime"].append(
            residual / np.sqrt(runtime)
        )
        accumulator["normalized_residuals"]["projected"].append(
            residual / np.sqrt(projected)
        )
        accumulator["num_coverage_records"] += 1
    tolerance = max(formal, ceilinged, runtime, projected, 1.0) * 1.0e-12
    ceiling_counts = accumulator["ceiling_counts"]
    if bool(stage.get("formal_ceiling_applied", ceilinged < formal - tolerance)):
        ceiling_counts["formal_ceiling_applied"] += 1
    if runtime > ceilinged + tolerance:
        ceiling_counts["runtime_above_ceilinged_formal"] += 1
    if projected > ceilinged + tolerance:
        ceiling_counts["projected_above_ceilinged_formal"] += 1
    if projected > runtime + tolerance:
        ceiling_counts["projected_above_runtime"] += 1
    accumulator["num_isotope_records"] += 1


def _finalize_covariance_accumulator(
    accumulator: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert covariance accumulator lists into JSON summary statistics."""
    num_records = int(accumulator.get("num_isotope_records", 0))
    ceiling_counts = {
        str(name): int(value)
        for name, value in dict(accumulator.get("ceiling_counts", {})).items()
    }
    ceiling_fractions = {
        name: (float(value) / num_records if num_records else float("nan"))
        for name, value in ceiling_counts.items()
    }
    return {
        "num_isotope_records": num_records,
        "num_coverage_records": int(accumulator.get("num_coverage_records", 0)),
        "variance_ratios": {
            str(name): _finite_distribution(list(values))
            for name, values in dict(accumulator.get("ratios", {})).items()
        },
        "ceiling_exceedance_counts": ceiling_counts,
        "ceiling_exceedance_fractions": ceiling_fractions,
        "normalized_residual_coverage_vs_transport_truth": {
            str(name): _normalized_residual_coverage(list(values))
            for name, values in dict(
                accumulator.get("normalized_residuals", {})
            ).items()
        },
    }


def _mahalanobis_distribution(
    values: list[tuple[float, int]],
) -> dict[str, Any]:
    """Summarize case-wise Mahalanobis squared values and degrees of freedom."""
    finite = [
        (float(value), int(degrees))
        for value, degrees in values
        if np.isfinite(float(value)) and int(degrees) > 0
    ]
    squared = [value for value, _degrees in finite]
    per_degree = [value / degrees for value, degrees in finite]
    total_degrees = int(sum(degrees for _value, degrees in finite))
    total_squared = float(sum(squared))
    return {
        "num_cases": int(len(finite)),
        "total_degrees_of_freedom": total_degrees,
        "pooled_squared_per_degree_of_freedom": (
            total_squared / total_degrees if total_degrees else float("nan")
        ),
        "mahalanobis_squared": _finite_distribution(squared),
        "squared_per_degree_of_freedom": _finite_distribution(per_degree),
    }


def summarize_response_poisson_covariance(
    results: list[dict[str, Any]],
    min_target: float,
) -> dict[str, Any]:
    """Summarize formal, runtime, and PF-projected count uncertainty."""
    total = _covariance_summary_accumulator()
    by_isotope = {isotope: _covariance_summary_accumulator() for isotope in ISOTOPES}
    mahalanobis: dict[str, list[tuple[float, int]]] = {
        "formal_full": [],
        "formal_diagonal": [],
        "runtime_full": [],
        "runtime_diagonal": [],
        "estimator_sanitized_full": [],
        "projected_diagonal": [],
    }
    num_cases = 0
    for result in results:
        case = result.get("case", {})
        if not bool(case.get("include_in_accuracy_summary", True)):
            continue
        covariance_payload = result.get("response_poisson_covariance", {})
        covariance_payload = (
            covariance_payload if isinstance(covariance_payload, Mapping) else {}
        )
        residual_diagnostics = covariance_payload.get("residual_diagnostics", {})
        residual_diagnostics = (
            residual_diagnostics if isinstance(residual_diagnostics, Mapping) else {}
        )
        degrees = int(residual_diagnostics.get("degrees_of_freedom", 0))
        case_mahalanobis = residual_diagnostics.get("mahalanobis_squared", {})
        if isinstance(case_mahalanobis, Mapping) and degrees > 0:
            for name in mahalanobis:
                value = case_mahalanobis.get(name)
                if value is not None:
                    mahalanobis[name].append((float(value), degrees))
            num_cases += 1
        for isotope, item in result.get("per_isotope", {}).items():
            if isotope not in by_isotope:
                by_isotope[isotope] = _covariance_summary_accumulator()
            truth = float(item.get("transport_truth_counts", 0.0))
            include_coverage = truth >= float(min_target)
            runtime = _finite_variance(item.get("response_poisson_variance", 1.0))
            raw_stage = item.get("response_poisson_variance_stages", {})
            stage = dict(raw_stage) if isinstance(raw_stage, Mapping) else {}
            stage.setdefault(
                "formal_variance",
                item.get("response_poisson_formal_variance", runtime),
            )
            stage.setdefault(
                "ceilinged_formal_variance",
                item.get("response_poisson_ceilinged_formal_variance", runtime),
            )
            stage.setdefault("runtime_variance", runtime)
            stage.setdefault(
                "projected_variance",
                item.get("response_poisson_projected_variance", runtime),
            )
            count = float(
                dict(item.get("method_counts", {})).get("response_poisson", 0.0)
            )
            _accumulate_covariance_record(
                total,
                count=count,
                truth=truth,
                stage=stage,
                include_coverage=include_coverage,
            )
            _accumulate_covariance_record(
                by_isotope[isotope],
                count=count,
                truth=truth,
                stage=stage,
                include_coverage=include_coverage,
            )
    return {
        "metric_definitions": {
            "formal_variance": (
                "response-regression covariance before runtime ceiling/floors"
            ),
            "runtime_variance": (
                "RuntimeCountExtractor diagonal after ceiling and all floors"
            ),
            "projected_variance": (
                "per-isotope variance returned by the runtime PF shared "
                "observation-covariance projection"
            ),
            "mahalanobis_squared": (
                "quadratic residual against Geant4 isotope-labeled transport "
                "truth, with the named covariance stage"
            ),
        },
        "min_transport_truth_counts": float(min_target),
        "num_cases_with_covariance_residuals": int(num_cases),
        **_finalize_covariance_accumulator(total),
        "mahalanobis_vs_transport_truth": {
            name: _mahalanobis_distribution(values)
            for name, values in mahalanobis.items()
        },
        "by_isotope": {
            isotope: _finalize_covariance_accumulator(accumulator)
            for isotope, accumulator in sorted(by_isotope.items())
        },
    }


def _pf_likelihood_summary_accumulator() -> dict[str, Any]:
    """Return an empty final-PF-likelihood calibration accumulator."""
    return {
        "normalized_residuals": [],
        "relative_scales": [],
        "case_squared_distances": [],
        "by_isotope": {
            isotope: {"normalized_residuals": [], "relative_scales": []}
            for isotope in ISOTOPES
        },
    }


def _finalize_pf_likelihood_summary(
    accumulator: Mapping[str, Any],
) -> dict[str, Any]:
    """Finalize normalized residual and scale summaries for one PF target."""
    normalized = list(accumulator.get("normalized_residuals", []))
    relative_scales = list(accumulator.get("relative_scales", []))
    case_distances = list(accumulator.get("case_squared_distances", []))
    by_isotope_payload = accumulator.get("by_isotope", {})
    by_isotope = by_isotope_payload if isinstance(by_isotope_payload, Mapping) else {}
    return {
        "num_records": int(len(normalized)),
        "num_cases": int(len(case_distances)),
        "normalized_residual_coverage": _normalized_residual_coverage(normalized),
        "likelihood_scale_over_target": _finite_distribution(relative_scales),
        "diagonal_squared_distance": _mahalanobis_distribution(case_distances),
        "by_isotope": {
            isotope: {
                "num_records": int(
                    len(
                        dict(by_isotope.get(isotope, {})).get(
                            "normalized_residuals",
                            [],
                        )
                    )
                ),
                "normalized_residual_coverage": _normalized_residual_coverage(
                    list(
                        dict(by_isotope.get(isotope, {})).get(
                            "normalized_residuals",
                            [],
                        )
                    )
                ),
                "likelihood_scale_over_target": _finite_distribution(
                    list(
                        dict(by_isotope.get(isotope, {})).get(
                            "relative_scales",
                            [],
                        )
                    )
                ),
            }
            for isotope in ISOTOPES
        },
    }


def summarize_pf_count_likelihood_diagnostics(
    results: list[dict[str, Any]],
    min_target: float,
) -> dict[str, Any]:
    """Summarize residual calibration at the final runtime PF likelihood scale."""
    accumulators: dict[str, dict[str, Any]] = {}
    likelihood_specs: dict[str, Any] = {}
    scale_semantics = ""
    observation_semantics = ""
    for result in results:
        case = result.get("case", {})
        if not bool(case.get("include_in_accuracy_summary", True)):
            continue
        payload = result.get("pf_count_likelihood_diagnostics", {})
        if not isinstance(payload, Mapping):
            continue
        if not likelihood_specs:
            raw_specs = payload.get("likelihood_spec_by_isotope", {})
            if isinstance(raw_specs, Mapping):
                likelihood_specs = dict(raw_specs)
        scale_semantics = str(payload.get("scale_semantics", scale_semantics))
        observation_semantics = str(
            payload.get("observation_variance_semantics", observation_semantics)
        )
        targets = payload.get("targets", {})
        if not isinstance(targets, Mapping):
            continue
        for target_name, raw_target in targets.items():
            if not isinstance(raw_target, Mapping):
                continue
            accumulator = accumulators.setdefault(
                str(target_name),
                _pf_likelihood_summary_accumulator(),
            )
            target_counts = raw_target.get("target_counts_by_isotope", {})
            normalized = raw_target.get("normalized_residual_by_isotope", {})
            relative_scales = raw_target.get(
                "likelihood_scale_over_target_by_isotope",
                {},
            )
            if not all(
                isinstance(value, Mapping)
                for value in (target_counts, normalized, relative_scales)
            ):
                continue
            case_values: list[float] = []
            for isotope in ISOTOPES:
                target = float(target_counts.get(isotope, 0.0))
                if target < float(min_target):
                    continue
                normalized_value = float(normalized.get(isotope, float("nan")))
                relative_scale = float(relative_scales.get(isotope, float("nan")))
                if not np.isfinite(normalized_value) or not np.isfinite(relative_scale):
                    continue
                accumulator["normalized_residuals"].append(normalized_value)
                accumulator["relative_scales"].append(relative_scale)
                isotope_accumulator = accumulator["by_isotope"][isotope]
                isotope_accumulator["normalized_residuals"].append(normalized_value)
                isotope_accumulator["relative_scales"].append(relative_scale)
                case_values.append(normalized_value)
            if case_values:
                accumulator["case_squared_distances"].append(
                    (float(np.sum(np.square(case_values))), len(case_values))
                )
    return {
        "metric_definitions": {
            "normalized_residual": (
                "(response_poisson - truth-source PF forward target) divided "
                "by the configured count-likelihood scale"
            ),
            "diagonal_squared_distance": (
                "sum of squared normalized count residuals; this is a scale "
                "diagnostic, not a Student-t chi-square statistic"
            ),
        },
        "min_pf_target_counts": float(min_target),
        "scale_semantics": scale_semantics,
        "observation_variance_semantics": observation_semantics,
        "likelihood_spec_by_isotope": likelihood_specs,
        "targets": {
            target_name: _finalize_pf_likelihood_summary(accumulator)
            for target_name, accumulator in sorted(accumulators.items())
        },
    }


def summarize_accuracy(
    results: list[dict[str, Any]],
    min_target: float,
) -> dict[str, dict[str, float]]:
    """Summarize accuracy metrics by decomposition method."""
    values_by_method: dict[str, list[float]] = {
        "response_poisson": [],
        "photopeak_nnls": [],
        "response_matrix": [],
        "peak_window": [],
    }
    false_positive_by_method: dict[str, list[float]] = {
        "response_poisson": [],
        "photopeak_nnls": [],
        "response_matrix": [],
        "peak_window": [],
    }
    values_vs_truth_by_method: dict[str, list[float]] = {
        "response_poisson": [],
        "photopeak_nnls": [],
        "response_matrix": [],
        "peak_window": [],
    }
    false_positive_vs_truth_by_method: dict[str, list[float]] = {
        "response_poisson": [],
        "photopeak_nnls": [],
        "response_matrix": [],
        "peak_window": [],
    }
    pf_target_vs_truth: list[float] = []
    pf_target_vs_uncollided: list[float] = []
    pf_geant4_mu_target_vs_truth: list[float] = []
    pf_geant4_mu_target_vs_uncollided: list[float] = []
    non_uncollided_fractions: list[float] = []
    for result in results:
        if not bool(result["case"]["include_in_accuracy_summary"]):
            continue
        for item in result["per_isotope"].values():
            target = float(item["target_pf_counts"])
            truth = float(item["transport_truth_counts"])
            uncollided = float(item.get("transport_uncollided_primary_counts", 0.0))
            non_uncollided_fraction = float(
                item.get("transport_non_uncollided_fraction", 0.0)
            )
            target_truth_err = item.get("pf_target_relative_error_vs_transport_truth")
            if target_truth_err is not None:
                pf_target_vs_truth.append(abs(float(target_truth_err)))
                non_uncollided_fractions.append(non_uncollided_fraction)
            target_uncollided_err = relative_error(target, uncollided, min_target)
            if target_uncollided_err is not None:
                pf_target_vs_uncollided.append(abs(float(target_uncollided_err)))
            geant4_mu_truth_err = item.get(
                "pf_geant4_mu_target_relative_error_vs_transport_truth"
            )
            if geant4_mu_truth_err is not None:
                pf_geant4_mu_target_vs_truth.append(abs(float(geant4_mu_truth_err)))
            geant4_mu_uncollided_err = item.get(
                "pf_geant4_mu_target_relative_error_vs_uncollided_primary"
            )
            if geant4_mu_uncollided_err is not None:
                pf_geant4_mu_target_vs_uncollided.append(
                    abs(float(geant4_mu_uncollided_err))
                )
            for method, estimate in item["method_counts"].items():
                if target >= float(min_target):
                    err = relative_error(float(estimate), target, min_target)
                    if err is not None:
                        values_by_method[method].append(abs(err))
                else:
                    false_positive_by_method[method].append(float(estimate))
                truth_err = item["relative_errors_vs_transport_truth"][method]
                if truth >= float(min_target):
                    if truth_err is not None:
                        values_vs_truth_by_method[method].append(abs(float(truth_err)))
                else:
                    false_positive_vs_truth_by_method[method].append(float(estimate))

    summary: dict[str, dict[str, float]] = {}
    for method, values in values_by_method.items():
        arr = np.asarray(values, dtype=float)
        fp = np.asarray(false_positive_by_method[method], dtype=float)
        truth_arr = np.asarray(values_vs_truth_by_method[method], dtype=float)
        truth_fp = np.asarray(false_positive_vs_truth_by_method[method], dtype=float)
        summary[method] = {
            "num_accuracy_points": float(arr.size),
            "mean_abs_relative_error": float(np.mean(arr))
            if arr.size
            else float("nan"),
            "median_abs_relative_error": float(np.median(arr))
            if arr.size
            else float("nan"),
            "max_abs_relative_error": float(np.max(arr)) if arr.size else float("nan"),
            "num_absent_isotope_points": float(fp.size),
            "max_absent_isotope_counts": float(np.max(fp)) if fp.size else 0.0,
            "mean_absent_isotope_counts": float(np.mean(fp)) if fp.size else 0.0,
            "num_accuracy_points_vs_transport_truth": float(truth_arr.size),
            "mean_abs_relative_error_vs_transport_truth": (
                float(np.mean(truth_arr)) if truth_arr.size else float("nan")
            ),
            "median_abs_relative_error_vs_transport_truth": (
                float(np.median(truth_arr)) if truth_arr.size else float("nan")
            ),
            "max_abs_relative_error_vs_transport_truth": (
                float(np.max(truth_arr)) if truth_arr.size else float("nan")
            ),
            "num_absent_truth_isotope_points": float(truth_fp.size),
            "max_absent_truth_isotope_counts": float(np.max(truth_fp))
            if truth_fp.size
            else 0.0,
            "mean_absent_truth_isotope_counts": float(np.mean(truth_fp))
            if truth_fp.size
            else 0.0,
        }
    pf_arr = np.asarray(pf_target_vs_truth, dtype=float)
    summary["pf_theory_target_vs_transport_truth"] = {
        "num_accuracy_points": float(pf_arr.size),
        "mean_abs_relative_error": float(np.mean(pf_arr))
        if pf_arr.size
        else float("nan"),
        "median_abs_relative_error": float(np.median(pf_arr))
        if pf_arr.size
        else float("nan"),
        "max_abs_relative_error": float(np.max(pf_arr))
        if pf_arr.size
        else float("nan"),
    }
    pf_uncollided_arr = np.asarray(pf_target_vs_uncollided, dtype=float)
    non_uncollided_arr = np.asarray(non_uncollided_fractions, dtype=float)
    summary["pf_theory_target_vs_uncollided_primary"] = {
        "num_accuracy_points": float(pf_uncollided_arr.size),
        "mean_abs_relative_error": (
            float(np.mean(pf_uncollided_arr))
            if pf_uncollided_arr.size
            else float("nan")
        ),
        "median_abs_relative_error": (
            float(np.median(pf_uncollided_arr))
            if pf_uncollided_arr.size
            else float("nan")
        ),
        "max_abs_relative_error": (
            float(np.max(pf_uncollided_arr)) if pf_uncollided_arr.size else float("nan")
        ),
    }
    pf_g4_truth_arr = np.asarray(pf_geant4_mu_target_vs_truth, dtype=float)
    summary["pf_geant4_mu_target_vs_transport_truth"] = {
        "num_accuracy_points": float(pf_g4_truth_arr.size),
        "mean_abs_relative_error": (
            float(np.mean(pf_g4_truth_arr)) if pf_g4_truth_arr.size else float("nan")
        ),
        "median_abs_relative_error": (
            float(np.median(pf_g4_truth_arr)) if pf_g4_truth_arr.size else float("nan")
        ),
        "max_abs_relative_error": (
            float(np.max(pf_g4_truth_arr)) if pf_g4_truth_arr.size else float("nan")
        ),
    }
    pf_g4_uncollided_arr = np.asarray(pf_geant4_mu_target_vs_uncollided, dtype=float)
    summary["pf_geant4_mu_target_vs_uncollided_primary"] = {
        "num_accuracy_points": float(pf_g4_uncollided_arr.size),
        "mean_abs_relative_error": (
            float(np.mean(pf_g4_uncollided_arr))
            if pf_g4_uncollided_arr.size
            else float("nan")
        ),
        "median_abs_relative_error": (
            float(np.median(pf_g4_uncollided_arr))
            if pf_g4_uncollided_arr.size
            else float("nan")
        ),
        "max_abs_relative_error": (
            float(np.max(pf_g4_uncollided_arr))
            if pf_g4_uncollided_arr.size
            else float("nan")
        ),
    }
    summary["transport_non_uncollided_fraction"] = {
        "num_accuracy_points": float(non_uncollided_arr.size),
        "mean": float(np.mean(non_uncollided_arr))
        if non_uncollided_arr.size
        else float("nan"),
        "median": float(np.median(non_uncollided_arr))
        if non_uncollided_arr.size
        else float("nan"),
        "max": float(np.max(non_uncollided_arr))
        if non_uncollided_arr.size
        else float("nan"),
    }
    return summary


def summarize_shield_pair_diagnostics(
    results: list[dict[str, Any]],
    min_target: float,
    *,
    num_orientations: int = 8,
) -> dict[str, Any]:
    """Summarize target/observation mismatch by Fe/Pb shield-pair id."""
    grouped: dict[str, dict[str, dict[str, list[float]]]] = {}
    for result in results:
        case = result["case"]
        if not bool(case.get("include_in_accuracy_summary", True)):
            continue
        pair_id = int(case["fe_index"]) * max(int(num_orientations), 1) + int(
            case["pb_index"]
        )
        pair_key = str(pair_id)
        for isotope, item in result["per_isotope"].items():
            target = float(item.get("target_pf_counts", 0.0))
            truth = float(item.get("transport_truth_counts", 0.0))
            response = float(item.get("method_counts", {}).get("response_poisson", 0.0))
            if target < float(min_target) or truth < float(min_target):
                continue
            isotope_metrics = grouped.setdefault(pair_key, {}).setdefault(
                str(isotope),
                {
                    "response_vs_pf_target": [],
                    "response_vs_transport_truth": [],
                    "pf_target_vs_transport_truth": [],
                },
            )
            isotope_metrics["response_vs_pf_target"].append(
                abs(response - target) / max(abs(target), 1.0)
            )
            isotope_metrics["response_vs_transport_truth"].append(
                abs(response - truth) / max(abs(truth), 1.0)
            )
            isotope_metrics["pf_target_vs_transport_truth"].append(
                abs(target - truth) / max(abs(truth), 1.0)
            )
    by_pair: dict[str, Any] = {}
    by_pair_isotope: dict[str, Any] = {}
    for pair_key, isotope_payload in sorted(
        grouped.items(), key=lambda item: int(item[0])
    ):
        merged: dict[str, list[float]] = {
            "response_vs_pf_target": [],
            "response_vs_transport_truth": [],
            "pf_target_vs_transport_truth": [],
        }
        by_pair_isotope[pair_key] = {}
        for isotope, metrics in sorted(isotope_payload.items()):
            by_pair_isotope[pair_key][isotope] = {
                metric: _relative_error_distribution(values)
                for metric, values in metrics.items()
            }
            for metric, values in metrics.items():
                merged[metric].extend(values)
        by_pair[pair_key] = {
            metric: _relative_error_distribution(values)
            for metric, values in merged.items()
        }
    return {
        "metric_definitions": {
            "response_vs_pf_target": (
                "abs(response_poisson - PF expected count) / PF expected count"
            ),
            "response_vs_transport_truth": (
                "abs(response_poisson - Geant4 isotope-labeled count) / truth"
            ),
            "pf_target_vs_transport_truth": (
                "abs(PF expected count - Geant4 isotope-labeled count) / truth"
            ),
        },
        "by_pair": by_pair,
        "by_pair_isotope": by_pair_isotope,
        "top_pairs": {
            "response_vs_pf_target": _top_pairs_by_metric(
                by_pair,
                "response_vs_pf_target",
            ),
            "response_vs_transport_truth": _top_pairs_by_metric(
                by_pair,
                "response_vs_transport_truth",
            ),
            "pf_target_vs_transport_truth": _top_pairs_by_metric(
                by_pair,
                "pf_target_vs_transport_truth",
            ),
        },
    }


def _relative_error_distribution(values: list[float]) -> dict[str, float]:
    """Return compact distribution statistics for relative-error values."""
    arr = np.asarray(values, dtype=float)
    return {
        "num_points": float(arr.size),
        "median": float(np.median(arr)) if arr.size else float("nan"),
        "p90": float(np.percentile(arr, 90.0)) if arr.size else float("nan"),
        "p99": float(np.percentile(arr, 99.0)) if arr.size else float("nan"),
        "max": float(np.max(arr)) if arr.size else float("nan"),
    }


def _top_pairs_by_metric(
    by_pair: dict[str, Any],
    metric: str,
    *,
    limit: int = 10,
) -> list[dict[str, float]]:
    """Return pairs sorted by p90 then maximum relative error for one metric."""
    rows: list[dict[str, float]] = []
    for pair_key, payload in by_pair.items():
        stats = payload.get(metric, {})
        rows.append(
            {
                "shield_pair_id": float(pair_key),
                "num_points": float(stats.get("num_points", 0.0)),
                "median": float(stats.get("median", float("nan"))),
                "p90": float(stats.get("p90", float("nan"))),
                "p99": float(stats.get("p99", float("nan"))),
                "max": float(stats.get("max", float("nan"))),
            }
        )
    rows.sort(key=lambda row: (row["p90"], row["max"]), reverse=True)
    return rows[: max(int(limit), 0)]


def summarize_response_poisson_calibration(
    results: list[dict[str, Any]],
    min_target: float,
    runtime_config: dict[str, Any] | None = None,
    *,
    num_orientations: int = 8,
    min_pair_fit_points: int = DEFAULT_CALIBRATION_MIN_PAIR_POINTS,
    pair_shrinkage_count: float = DEFAULT_CALIBRATION_PAIR_SHRINKAGE_COUNT,
    holdout_fraction: float = DEFAULT_CALIBRATION_HOLDOUT_FRACTION,
    holdout_seed: int = DEFAULT_CALIBRATION_HOLDOUT_SEED,
) -> dict[str, Any]:
    """Fit PF-target to response-Poisson scale factors from validation results."""
    records = _response_poisson_calibration_records(
        results,
        min_target,
        num_orientations=num_orientations,
    )
    calibration = fit_net_response_calibration(
        records,
        isotopes=ISOTOPES,
        min_theory_counts=float(min_target),
        min_pair_fit_points=int(min_pair_fit_points),
        pair_shrinkage_count=float(pair_shrinkage_count),
        metadata={
            "source": "validate_geant4_spectrum_decomposition",
            "theory_counts": "target_pf_counts",
            "net_counts": "response_poisson",
            "num_fit_records": len(records),
        },
    )
    payload = calibration.to_dict()
    payload["runtime_config_snippet"] = {
        "measurement_scale_by_isotope": payload.get("scale_by_isotope", {}),
        "measurement_scale_by_isotope_and_pair": payload.get(
            "scale_by_isotope_and_pair",
            {},
        ),
    }
    if runtime_config is not None:
        payload["runtime_config_effective_snippet"] = (
            _effective_measurement_scale_snippet(
                runtime_config,
                payload,
            )
        )
    payload["pair_coverage"] = _response_poisson_pair_coverage(records)
    payload["calibrated_residual_summary"] = _calibrated_residual_summary(
        results,
        payload,
        min_target,
        num_orientations=num_orientations,
    )
    train_records, holdout_records = _split_calibration_records(
        records,
        holdout_fraction=float(holdout_fraction),
        seed=int(holdout_seed),
    )
    if train_records and holdout_records:
        holdout_calibration = fit_net_response_calibration(
            train_records,
            isotopes=ISOTOPES,
            min_theory_counts=float(min_target),
            min_pair_fit_points=int(min_pair_fit_points),
            pair_shrinkage_count=float(pair_shrinkage_count),
            metadata={
                "source": "validate_geant4_spectrum_decomposition",
                "split": "train",
                "theory_counts": "target_pf_counts",
                "net_counts": "response_poisson",
                "num_fit_records": len(train_records),
            },
        )
        payload["holdout_validation"] = {
            "train_records": len(train_records),
            "holdout_records": len(holdout_records),
            "holdout_fraction": float(holdout_fraction),
            "holdout_seed": int(holdout_seed),
            "split_semantics": (
                "Scenario-level split when calibration records contain scenario "
                "keys; otherwise deterministic record-level fallback."
            ),
            "calibrated_residual_summary": _calibrated_record_residual_summary(
                holdout_records,
                holdout_calibration.to_dict(),
            ),
            "fit_statistics": holdout_calibration.fit_statistics,
        }
    return payload


def _response_poisson_calibration_records(
    results: list[dict[str, Any]],
    min_target: float,
    *,
    num_orientations: int = 8,
) -> list[dict[str, Any]]:
    """Return weighted PF-target to response-Poisson calibration records."""
    records: list[dict[str, Any]] = []
    for result in results:
        case = result["case"]
        if not bool(case.get("include_in_accuracy_summary", True)):
            continue
        pair_id = int(case["fe_index"]) * max(int(num_orientations), 1) + int(
            case["pb_index"]
        )
        for isotope, item in result["per_isotope"].items():
            theory = float(item.get("target_pf_counts", 0.0))
            net = float(item.get("method_counts", {}).get("response_poisson", 0.0))
            method_name = str(
                item.get("response_poisson_method_name", "response_poisson")
            )
            if theory < float(min_target):
                continue
            if "low_snr_photopeak_uncertain" in method_name:
                continue
            variance = max(float(item.get("response_poisson_variance", 1.0)), 1.0)
            records.append(
                {
                    "isotope": str(isotope),
                    "case": str(case.get("name", "")),
                    "scenario": _transport_response_scenario_key(case),
                    "theory_counts": theory,
                    "net_counts": max(net, 0.0),
                    "shield_pair_id": pair_id,
                    "weight": 1.0 / variance,
                }
            )
    return records


def _split_calibration_records(
    records: list[dict[str, Any]],
    *,
    holdout_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split calibration records into deterministic scenario-level holdout lists."""
    frac = min(max(float(holdout_fraction), 0.0), 0.9)
    if not records or frac <= 0.0:
        return list(records), []
    scenarios = sorted(
        {
            str(record.get("scenario", ""))
            for record in records
            if str(record.get("scenario", ""))
        }
    )
    if len(scenarios) >= 2:
        rng = np.random.default_rng(int(seed))
        shuffled = np.asarray(scenarios, dtype=object)
        rng.shuffle(shuffled)
        holdout_count = int(round(float(len(scenarios)) * frac))
        holdout_count = min(max(holdout_count, 1), len(scenarios) - 1)
        holdout_scenarios = {str(value) for value in shuffled[:holdout_count]}
        train: list[dict[str, Any]] = []
        holdout: list[dict[str, Any]] = []
        for record in records:
            if str(record.get("scenario", "")) in holdout_scenarios:
                holdout.append(record)
            else:
                train.append(record)
        return train, holdout
    rng = np.random.default_rng(int(seed))
    indices = np.arange(len(records), dtype=int)
    rng.shuffle(indices)
    holdout_count = int(round(float(len(records)) * frac))
    holdout_count = min(max(holdout_count, 1), len(records) - 1)
    holdout_idx = set(int(value) for value in indices[:holdout_count])
    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if index in holdout_idx:
            holdout.append(record)
        else:
            train.append(record)
    return train, holdout


def _calibrated_record_residual_summary(
    records: list[dict[str, Any]],
    calibration_payload: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """Summarize calibrated residuals for flat calibration records."""
    scale_by_isotope = calibration_payload.get("scale_by_isotope", {})
    scale_by_pair = calibration_payload.get("scale_by_isotope_and_pair", {})
    values_by_iso: dict[str, list[float]] = {iso: [] for iso in ISOTOPES}
    sigma_by_iso: dict[str, list[float]] = {iso: [] for iso in ISOTOPES}
    for record in records:
        isotope = str(record["isotope"])
        pair_key = str(int(record.get("shield_pair_id", -1)))
        theory = float(record["theory_counts"])
        net = float(record["net_counts"])
        iso_pair_scales = scale_by_pair.get(isotope, {})
        if isinstance(iso_pair_scales, dict) and pair_key in iso_pair_scales:
            scale = float(iso_pair_scales[pair_key])
        else:
            scale = float(scale_by_isotope.get(isotope, 1.0))
        prediction = max(float(scale) * theory, 1.0e-12)
        residual = net - prediction
        values_by_iso.setdefault(isotope, []).append(abs(residual) / max(abs(net), 1.0))
        weight = max(float(record.get("weight", 0.0)), 0.0)
        if weight > 0.0:
            sigma_by_iso.setdefault(isotope, []).append(abs(residual) * np.sqrt(weight))
    summary: dict[str, dict[str, float]] = {}
    for isotope in sorted(values_by_iso):
        rel = np.asarray(values_by_iso.get(isotope, []), dtype=float)
        sigma = np.asarray(sigma_by_iso.get(isotope, []), dtype=float)
        summary[isotope] = {
            "num_points": float(rel.size),
            "median_abs_relative_error": (
                float(np.median(rel)) if rel.size else float("nan")
            ),
            "p90_abs_relative_error": (
                float(np.percentile(rel, 90.0)) if rel.size else float("nan")
            ),
            "p99_abs_relative_error": (
                float(np.percentile(rel, 99.0)) if rel.size else float("nan")
            ),
            "max_abs_relative_error": (
                float(np.max(rel)) if rel.size else float("nan")
            ),
            "median_abs_sigma_residual": (
                float(np.median(sigma)) if sigma.size else float("nan")
            ),
            "p99_abs_sigma_residual": (
                float(np.percentile(sigma, 99.0)) if sigma.size else float("nan")
            ),
        }
    return summary


def summarize_pf_transport_calibration(
    results: list[dict[str, Any]],
    min_target: float,
    runtime_config: dict[str, Any],
    *,
    num_orientations: int = 8,
    min_pair_fit_points: int = DEFAULT_CALIBRATION_MIN_PAIR_POINTS,
    pair_shrinkage_count: float = DEFAULT_CALIBRATION_PAIR_SHRINKAGE_COUNT,
    holdout_fraction: float = DEFAULT_CALIBRATION_HOLDOUT_FRACTION,
    holdout_seed: int = DEFAULT_CALIBRATION_HOLDOUT_SEED,
) -> dict[str, Any]:
    """Fit current PF target counts onto Geant4 transport-labeled counts."""
    records: list[dict[str, Any]] = []
    for result in results:
        case = result["case"]
        if not bool(case.get("include_in_accuracy_summary", True)):
            continue
        pair_id = int(case["fe_index"]) * max(int(num_orientations), 1) + int(
            case["pb_index"]
        )
        for isotope, item in result["per_isotope"].items():
            theory = float(item.get("target_pf_counts", 0.0))
            truth = float(item.get("transport_truth_counts", 0.0))
            if theory < float(min_target) or truth < float(min_target):
                continue
            records.append(
                {
                    "isotope": str(isotope),
                    "case": str(case.get("name", "")),
                    "scenario": _transport_response_scenario_key(case),
                    "theory_counts": theory,
                    "net_counts": max(truth, 0.0),
                    "shield_pair_id": pair_id,
                    "weight": 1.0 / max(truth, 1.0),
                }
            )
    response_groups = _pf_transport_response_groups(
        results,
        min_target,
        num_orientations=num_orientations,
    )
    calibration = fit_net_response_calibration(
        records,
        isotopes=ISOTOPES,
        min_theory_counts=float(min_target),
        min_pair_fit_points=int(min_pair_fit_points),
        pair_shrinkage_count=float(pair_shrinkage_count),
        metadata={
            "source": "validate_geant4_spectrum_decomposition",
            "model": "transport_counts = scale * current_pf_target_counts",
            "theory_counts": "target_pf_counts",
            "net_counts": "transport_detected_counts",
            "num_fit_records": len(records),
        },
    )
    payload = calibration.to_dict()
    payload["pair_coverage"] = _response_poisson_pair_coverage(records)
    payload["runtime_config_multiplier"] = {
        "measurement_scale_by_isotope": payload.get("scale_by_isotope", {}),
        "measurement_scale_by_isotope_and_pair": payload.get(
            "scale_by_isotope_and_pair",
            {},
        ),
    }
    payload["runtime_config_effective_snippet"] = _effective_transport_scale_snippet(
        runtime_config,
        payload,
    )
    payload["calibrated_residual_summary"] = _pf_transport_residual_summary(
        results,
        payload,
        min_target,
        num_orientations=num_orientations,
    )
    transport_model = _fit_pf_transport_response_model(
        response_groups,
        min_pair_fit_points=int(min_pair_fit_points),
        pair_shrinkage_count=float(pair_shrinkage_count),
    )
    payload["transport_response_model"] = transport_model
    payload["runtime_config_transport_response_snippet"] = {
        "pf_transport_response_model": transport_model
    }
    payload["transport_response_residual_summary"] = (
        _pf_transport_response_model_residual_summary(
            response_groups,
            transport_model,
        )
    )
    train_groups, holdout_groups = _split_transport_response_groups(
        response_groups,
        holdout_fraction=float(holdout_fraction),
        seed=int(holdout_seed),
    )
    if train_groups and holdout_groups:
        holdout_model = _fit_pf_transport_response_model(
            train_groups,
            min_pair_fit_points=int(min_pair_fit_points),
            pair_shrinkage_count=float(pair_shrinkage_count),
        )
        payload["transport_response_holdout_validation"] = {
            "train_groups": len(train_groups),
            "holdout_groups": len(holdout_groups),
            "holdout_fraction": float(holdout_fraction),
            "holdout_seed": int(holdout_seed),
            "split_semantics": (
                "Scenario-level split: all shield pairs from one environment, "
                "source layout, and detector pose stay together."
            ),
            "train_residual_summary": _pf_transport_response_model_residual_summary(
                train_groups,
                holdout_model,
            ),
            "holdout_residual_summary": _pf_transport_response_model_residual_summary(
                holdout_groups,
                holdout_model,
            ),
        }
    return payload


def _pf_transport_response_records(
    results: list[dict[str, Any]],
    min_target: float,
    *,
    num_orientations: int = 8,
) -> list[dict[str, float | str | int]]:
    """Return feature records for PF-target to Geant4 transport-response fitting."""
    records: list[dict[str, float | str | int]] = []
    for result in results:
        case = result["case"]
        if not bool(case.get("include_in_accuracy_summary", True)):
            continue
        detector_xyz = case.get("detector_pose_xyz")
        fe_index = int(case["fe_index"])
        pb_index = int(case["pb_index"])
        pair_id = fe_index * max(int(num_orientations), 1) + pb_index
        for isotope, item in result["per_isotope"].items():
            theory = float(item.get("target_pf_counts", 0.0))
            truth = float(item.get("transport_truth_counts", 0.0))
            if theory < float(min_target) or truth < float(min_target):
                continue
            features = _weighted_pf_transport_features(
                item.get("target_pf_count_diagnostics", []),
                detector_xyz=detector_xyz,
            )
            records.append(
                {
                    "isotope": str(isotope),
                    "theory_counts": theory,
                    "transport_truth_counts": truth,
                    "shield_pair_id": pair_id,
                    "shield_tau_feature": features["shield_tau_feature"],
                    "fe_tau_feature": features["fe_tau_feature"],
                    "pb_tau_feature": features["pb_tau_feature"],
                    "obstacle_tau_feature": features["obstacle_tau_feature"],
                    "distance_feature": features["distance_feature"],
                    "distance_shield_feature": features["distance_shield_feature"],
                    "distance_fe_feature": features["distance_fe_feature"],
                    "distance_pb_feature": features["distance_pb_feature"],
                    "distance_obstacle_feature": features["distance_obstacle_feature"],
                    "weight": 1.0 / max(truth, 1.0),
                }
            )
    return records


def _pf_transport_response_groups(
    results: list[dict[str, Any]],
    min_target: float,
    *,
    num_orientations: int = 8,
) -> list[dict[str, Any]]:
    """Return isotope measurement groups with source-wise PF contributions."""
    groups: list[dict[str, Any]] = []
    for result in results:
        case = result["case"]
        if not bool(case.get("include_in_accuracy_summary", True)):
            continue
        detector_xyz = case.get("detector_pose_xyz")
        fe_index = int(case["fe_index"])
        pb_index = int(case["pb_index"])
        pair_id = fe_index * max(int(num_orientations), 1) + pb_index
        for isotope, item in result["per_isotope"].items():
            theory = float(item.get("target_pf_counts", 0.0))
            truth = float(item.get("transport_truth_counts", 0.0))
            if theory < float(min_target) or truth < float(min_target):
                continue
            features = _weighted_pf_transport_features(
                item.get("target_pf_count_diagnostics", []),
                detector_xyz=detector_xyz,
            )
            source_terms = _pf_transport_source_terms(
                item.get("target_pf_count_diagnostics", []),
                detector_xyz=detector_xyz,
            )
            if not source_terms:
                source_terms = [
                    {
                        "counts": theory,
                        "shield_tau_feature": features["shield_tau_feature"],
                        "fe_tau_feature": features["fe_tau_feature"],
                        "pb_tau_feature": features["pb_tau_feature"],
                        "obstacle_tau_feature": features["obstacle_tau_feature"],
                        "distance_feature": features["distance_feature"],
                        "distance_shield_feature": features["distance_shield_feature"],
                        "distance_fe_feature": features["distance_fe_feature"],
                        "distance_pb_feature": features["distance_pb_feature"],
                        "distance_obstacle_feature": features[
                            "distance_obstacle_feature"
                        ],
                    }
                ]
            groups.append(
                {
                    "case": str(case.get("name", "")),
                    "scenario": _transport_response_scenario_key(case),
                    "isotope": str(isotope),
                    "theory_counts": theory,
                    "transport_truth_counts": truth,
                    "shield_pair_id": pair_id,
                    "source_terms": source_terms,
                    "shield_tau_feature": features["shield_tau_feature"],
                    "fe_tau_feature": features["fe_tau_feature"],
                    "pb_tau_feature": features["pb_tau_feature"],
                    "obstacle_tau_feature": features["obstacle_tau_feature"],
                    "distance_feature": features["distance_feature"],
                    "distance_shield_feature": features["distance_shield_feature"],
                    "distance_fe_feature": features["distance_fe_feature"],
                    "distance_pb_feature": features["distance_pb_feature"],
                    "distance_obstacle_feature": features["distance_obstacle_feature"],
                    "weight": 1.0 / max(truth, 1.0),
                }
            )
    return groups


def _transport_response_scenario_key(case: dict[str, Any]) -> str:
    """Return a scenario key that keeps all shield pairs together."""
    name = str(case.get("name", ""))
    if "_pair" in name:
        return name.split("_pair", maxsplit=1)[0]
    return name


def _split_transport_response_groups(
    groups: list[dict[str, Any]],
    *,
    holdout_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split transport-response groups by scenario, not individual rows."""
    frac = min(max(float(holdout_fraction), 0.0), 0.9)
    if not groups or frac <= 0.0:
        return list(groups), []
    scenarios = sorted({str(group.get("scenario", "")) for group in groups})
    if len(scenarios) < 2:
        return list(groups), []
    rng = np.random.default_rng(int(seed))
    shuffled = np.asarray(scenarios, dtype=object)
    rng.shuffle(shuffled)
    holdout_count = int(round(float(len(scenarios)) * frac))
    holdout_count = min(max(holdout_count, 1), len(scenarios) - 1)
    holdout_scenarios = {str(value) for value in shuffled[:holdout_count]}
    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for group in groups:
        if str(group.get("scenario", "")) in holdout_scenarios:
            holdout.append(group)
        else:
            train.append(group)
    return train, holdout


def _source_distance_feature(
    row: dict[str, Any],
    nested: dict[str, Any] | None = None,
    *,
    detector_xyz: Any = None,
) -> float:
    """Return the source-detector distance feature for one diagnostic row."""
    candidates: list[Any] = []
    if isinstance(nested, dict):
        candidates.extend(
            [
                nested.get("distance_feature"),
                nested.get("source_distance_m"),
                nested.get("source_detector_distance_m"),
            ]
        )
    candidates.extend(
        [
            row.get("source_distance_m"),
            row.get("distance_feature"),
            row.get("source_detector_distance_m"),
        ]
    )
    for value in candidates:
        if value is None:
            continue
        distance = float(value)
        if np.isfinite(distance) and distance >= 0.0:
            return float(distance)
    if detector_xyz is None or "position_xyz" not in row:
        return 0.0
    detector = np.asarray(detector_xyz, dtype=float).reshape(-1)
    position = np.asarray(row.get("position_xyz", []), dtype=float).reshape(-1)
    if detector.size != 3 or position.size != 3:
        return 0.0
    distance = float(np.linalg.norm(detector - position))
    return distance if np.isfinite(distance) and distance >= 0.0 else 0.0


def _pf_transport_source_terms(
    diagnostics: Any,
    *,
    detector_xyz: Any = None,
) -> list[dict[str, float]]:
    """Return scaled source terms and local optical-depth features."""
    if not isinstance(diagnostics, list):
        return []
    terms: list[dict[str, float]] = []
    for row in diagnostics:
        if not isinstance(row, dict):
            continue
        nested_terms = row.get("transport_response_terms")
        if isinstance(nested_terms, list) and nested_terms:
            source_scale = max(float(row.get("measurement_source_scale", 1.0)), 0.0)
            for nested in nested_terms:
                if not isinstance(nested, dict):
                    continue
                base_counts = max(
                    float(
                        nested.get(
                            "scaled_base_counts",
                            nested.get("scaled_counts", nested.get("counts", 0.0)),
                        )
                    ),
                    0.0,
                )
                if "scaled_base_counts" not in nested and "scaled_counts" not in nested:
                    base_counts *= source_scale
                if base_counts <= 0.0:
                    continue
                distance = _source_distance_feature(
                    row,
                    nested,
                    detector_xyz=detector_xyz,
                )
                shield_tau = max(
                    float(nested.get("shield_tau_feature", 0.0)),
                    0.0,
                )
                fe_tau = max(float(nested.get("fe_tau_feature", 0.0)), 0.0)
                pb_tau = max(float(nested.get("pb_tau_feature", 0.0)), 0.0)
                obstacle_tau = max(
                    float(nested.get("obstacle_tau_feature", 0.0)),
                    0.0,
                )
                terms.append(
                    {
                        "counts": base_counts,
                        "shield_tau_feature": shield_tau,
                        "fe_tau_feature": fe_tau,
                        "pb_tau_feature": pb_tau,
                        "obstacle_tau_feature": obstacle_tau,
                        "distance_feature": distance,
                        "distance_shield_feature": max(
                            float(
                                nested.get(
                                    "distance_shield_feature",
                                    distance * shield_tau,
                                )
                            ),
                            0.0,
                        ),
                        "distance_fe_feature": max(
                            float(
                                nested.get(
                                    "distance_fe_feature",
                                    distance * fe_tau,
                                )
                            ),
                            0.0,
                        ),
                        "distance_pb_feature": max(
                            float(
                                nested.get(
                                    "distance_pb_feature",
                                    distance * pb_tau,
                                )
                            ),
                            0.0,
                        ),
                        "distance_obstacle_feature": max(
                            float(
                                nested.get(
                                    "distance_obstacle_feature",
                                    distance * obstacle_tau,
                                )
                            ),
                            0.0,
                        ),
                    }
                )
            continue
        base_counts = max(
            float(
                row.get(
                    "scaled_full_target_counts",
                    row.get("full_target_counts", 0.0),
                )
            ),
            0.0,
        )
        source_scale = max(float(row.get("measurement_source_scale", 1.0)), 0.0)
        counts = base_counts
        if "scaled_full_target_counts" not in row:
            counts = base_counts * source_scale
        if counts <= 0.0:
            continue
        shield_att = max(float(row.get("shield_attenuation", 1.0)), 1.0e-12)
        shield_tau = max(-float(np.log(shield_att)), 0.0)
        distance = _source_distance_feature(row, detector_xyz=detector_xyz)
        obstacle_tau = max(
            float(
                row.get(
                    "obstacle_tau_area_averaged",
                    row.get("obstacle_tau_center_ray", 0.0),
                )
            ),
            0.0,
        )
        terms.append(
            {
                "counts": counts,
                "shield_tau_feature": shield_tau,
                "fe_tau_feature": 0.0,
                "pb_tau_feature": 0.0,
                "obstacle_tau_feature": obstacle_tau,
                "distance_feature": distance,
                "distance_shield_feature": distance * shield_tau,
                "distance_fe_feature": 0.0,
                "distance_pb_feature": 0.0,
                "distance_obstacle_feature": distance * obstacle_tau,
            }
        )
    return terms


def _weighted_pf_transport_feature_diagnostics(
    diagnostics: Any,
    *,
    detector_xyz: Any = None,
) -> dict[str, float]:
    """Return CSV-friendly PF transport feature diagnostics."""
    features = _weighted_pf_transport_features(
        diagnostics,
        detector_xyz=detector_xyz,
    )
    summary = dict(features)
    summary.update(
        {
            "response_factor_weighted": 1.0,
            "response_factor_min": 1.0,
            "response_factor_max": 1.0,
            "source_count": 0.0,
            "source_term_count": 0.0,
        }
    )
    if not isinstance(diagnostics, list):
        return summary
    total_weight = 0.0
    factor_sum = 0.0
    factors: list[float] = []
    source_count = 0
    source_term_count = 0
    for row in diagnostics:
        if not isinstance(row, dict):
            continue
        source_count += 1
        source_scale = max(float(row.get("measurement_source_scale", 1.0)), 0.0)
        nested_terms = row.get("transport_response_terms")
        if isinstance(nested_terms, list) and nested_terms:
            for nested in nested_terms:
                if not isinstance(nested, dict):
                    continue
                contribution = max(
                    float(
                        nested.get(
                            "scaled_base_counts",
                            nested.get("scaled_counts", nested.get("counts", 0.0)),
                        )
                    ),
                    0.0,
                )
                if "scaled_base_counts" not in nested and "scaled_counts" not in nested:
                    contribution *= source_scale
                if contribution <= 0.0:
                    continue
                factor = max(float(nested.get("response_factor", 1.0)), 0.0)
                total_weight += contribution
                factor_sum += contribution * factor
                factors.append(factor)
                source_term_count += 1
            continue
        contribution = max(
            float(
                row.get("scaled_full_target_counts", row.get("full_target_counts", 0.0))
            ),
            0.0,
        )
        if "scaled_full_target_counts" not in row:
            contribution *= source_scale
        if contribution <= 0.0:
            continue
        total_weight += contribution
        factor_sum += contribution
        factors.append(1.0)
        source_term_count += 1
    if factors:
        summary["response_factor_weighted"] = float(
            factor_sum / max(total_weight, 1.0e-12)
        )
        summary["response_factor_min"] = float(min(factors))
        summary["response_factor_max"] = float(max(factors))
    summary["source_count"] = float(source_count)
    summary["source_term_count"] = float(source_term_count)
    return summary


def _weighted_pf_transport_features(
    diagnostics: Any,
    *,
    detector_xyz: Any = None,
) -> dict[str, float]:
    """Return source-contribution-weighted transport-response features."""
    if not isinstance(diagnostics, list):
        return {
            "shield_tau_feature": 0.0,
            "fe_tau_feature": 0.0,
            "pb_tau_feature": 0.0,
            "obstacle_tau_feature": 0.0,
            "distance_feature": 0.0,
            "distance_shield_feature": 0.0,
            "distance_fe_feature": 0.0,
            "distance_pb_feature": 0.0,
            "distance_obstacle_feature": 0.0,
        }
    total_weight = 0.0
    shield_sum = 0.0
    fe_sum = 0.0
    pb_sum = 0.0
    obstacle_sum = 0.0
    distance_sum = 0.0
    distance_shield_sum = 0.0
    distance_fe_sum = 0.0
    distance_pb_sum = 0.0
    distance_obstacle_sum = 0.0
    for row in diagnostics:
        if not isinstance(row, dict):
            continue
        nested_terms = row.get("transport_response_terms")
        if isinstance(nested_terms, list) and nested_terms:
            for nested in nested_terms:
                if not isinstance(nested, dict):
                    continue
                contribution = max(
                    float(
                        nested.get(
                            "scaled_base_counts",
                            nested.get("scaled_counts", nested.get("counts", 0.0)),
                        )
                    ),
                    0.0,
                )
                if contribution <= 0.0:
                    continue
                distance = _source_distance_feature(
                    row,
                    nested,
                    detector_xyz=detector_xyz,
                )
                shield_tau = max(
                    float(nested.get("shield_tau_feature", 0.0)),
                    0.0,
                )
                fe_tau = max(float(nested.get("fe_tau_feature", 0.0)), 0.0)
                pb_tau = max(float(nested.get("pb_tau_feature", 0.0)), 0.0)
                obstacle_tau = max(
                    float(nested.get("obstacle_tau_feature", 0.0)),
                    0.0,
                )
                total_weight += contribution
                shield_sum += contribution * shield_tau
                fe_sum += contribution * fe_tau
                pb_sum += contribution * pb_tau
                obstacle_sum += contribution * obstacle_tau
                distance_sum += contribution * distance
                distance_shield_sum += contribution * max(
                    float(
                        nested.get(
                            "distance_shield_feature",
                            distance * shield_tau,
                        )
                    ),
                    0.0,
                )
                distance_fe_sum += contribution * max(
                    float(nested.get("distance_fe_feature", distance * fe_tau)),
                    0.0,
                )
                distance_pb_sum += contribution * max(
                    float(nested.get("distance_pb_feature", distance * pb_tau)),
                    0.0,
                )
                distance_obstacle_sum += contribution * max(
                    float(
                        nested.get(
                            "distance_obstacle_feature",
                            distance * obstacle_tau,
                        )
                    ),
                    0.0,
                )
            continue
        contribution = max(
            float(
                row.get(
                    "scaled_full_target_counts",
                    row.get("full_target_counts", 0.0),
                )
            ),
            0.0,
        )
        if "scaled_full_target_counts" not in row:
            contribution *= max(float(row.get("measurement_source_scale", 1.0)), 0.0)
        if contribution <= 0.0:
            continue
        shield_att = max(float(row.get("shield_attenuation", 1.0)), 1.0e-12)
        shield_tau = max(-float(np.log(shield_att)), 0.0)
        obstacle_tau = max(float(row.get("obstacle_tau_center_ray", 0.0)), 0.0)
        distance = _source_distance_feature(row, detector_xyz=detector_xyz)
        total_weight += contribution
        shield_sum += contribution * shield_tau
        obstacle_sum += contribution * obstacle_tau
        distance_sum += contribution * distance
        distance_shield_sum += contribution * distance * shield_tau
        distance_obstacle_sum += contribution * distance * obstacle_tau
    if total_weight <= 0.0:
        return {
            "shield_tau_feature": 0.0,
            "fe_tau_feature": 0.0,
            "pb_tau_feature": 0.0,
            "obstacle_tau_feature": 0.0,
            "distance_feature": 0.0,
            "distance_shield_feature": 0.0,
            "distance_fe_feature": 0.0,
            "distance_pb_feature": 0.0,
            "distance_obstacle_feature": 0.0,
        }
    return {
        "shield_tau_feature": float(shield_sum / total_weight),
        "fe_tau_feature": float(fe_sum / total_weight),
        "pb_tau_feature": float(pb_sum / total_weight),
        "obstacle_tau_feature": float(obstacle_sum / total_weight),
        "distance_feature": float(distance_sum / total_weight),
        "distance_shield_feature": float(distance_shield_sum / total_weight),
        "distance_fe_feature": float(distance_fe_sum / total_weight),
        "distance_pb_feature": float(distance_pb_sum / total_weight),
        "distance_obstacle_feature": float(distance_obstacle_sum / total_weight),
    }


def _fit_pf_transport_response_model(
    groups: list[dict[str, Any]],
    *,
    ridge: float = 1.0e-3,
    min_pair_fit_points: int = DEFAULT_CALIBRATION_MIN_PAIR_POINTS,
    pair_shrinkage_count: float = DEFAULT_CALIBRATION_PAIR_SHRINKAGE_COUNT,
) -> dict[str, Any]:
    """Fit a low-dimensional optical-depth transport-response model."""
    by_isotope: dict[str, dict[str, Any]] = {}
    for isotope in ISOTOPES:
        isotope_groups = [
            group for group in groups if str(group.get("isotope")) == isotope
        ]
        if len(isotope_groups) < 6:
            by_isotope[isotope] = {
                "scale": 1.0,
                "scale_by_pair": {},
                "tau_coefficients": {
                    "shield": 0.0,
                    "obstacle": 0.0,
                    "shield_squared": 0.0,
                    "obstacle_squared": 0.0,
                    "shield_obstacle": 0.0,
                    "fe": 0.0,
                    "pb": 0.0,
                    "fe_squared": 0.0,
                    "pb_squared": 0.0,
                    "fe_pb": 0.0,
                    "fe_obstacle": 0.0,
                    "pb_obstacle": 0.0,
                    "distance": 0.0,
                    "distance_shield": 0.0,
                    "distance_fe": 0.0,
                    "distance_pb": 0.0,
                    "distance_obstacle": 0.0,
                },
                "tau_feature_caps": dict(DEFAULT_TRANSPORT_RESPONSE_TAU_FEATURE_CAPS),
                "min_log_scale": -2.0,
                "max_log_scale": 2.0,
                "num_fit_records": len(isotope_groups),
            }
            continue
        beta0 = _aggregate_transport_response_initial_beta(
            isotope_groups,
            ridge=float(ridge),
        )
        beta = _fit_sourcewise_transport_response_beta(
            isotope_groups,
            beta0,
            ridge=float(ridge),
        )
        pair_scales = _fit_transport_response_pair_scales(
            isotope_groups,
            beta,
            min_pair_fit_points=int(min_pair_fit_points),
            pair_shrinkage_count=float(pair_shrinkage_count),
        )
        by_isotope[isotope] = {
            "scale": float(np.exp(beta[0])),
            "scale_by_pair": pair_scales,
            "tau_coefficients": {
                "shield": float(beta[1]),
                "obstacle": float(beta[2]),
                "shield_squared": float(beta[3]),
                "obstacle_squared": float(beta[4]),
                "shield_obstacle": float(beta[5]),
                "fe": float(beta[6]),
                "pb": float(beta[7]),
                "fe_squared": float(beta[8]),
                "pb_squared": float(beta[9]),
                "fe_pb": float(beta[10]),
                "fe_obstacle": float(beta[11]),
                "pb_obstacle": float(beta[12]),
                "distance": float(beta[13]),
                "distance_shield": float(beta[14]),
                "distance_fe": float(beta[15]),
                "distance_pb": float(beta[16]),
                "distance_obstacle": float(beta[17]),
            },
            "tau_feature_caps": dict(DEFAULT_TRANSPORT_RESPONSE_TAU_FEATURE_CAPS),
            "min_log_scale": -2.0,
            "max_log_scale": 2.0,
            "num_fit_records": len(isotope_groups),
        }
    return {
        "enabled": True,
        "model": "log(transport_truth_counts/current_pf_target_counts) = "
        "log(scale_pair_or_isotope) plus optical-depth terms for total "
        "shield, Fe, Pb, obstacle, source-detector distance, and "
        "quadratic/cross/distance-material interactions",
        "feature_semantics": {
            "tau_shield": "source-contribution-weighted -log(shield-only transmission)",
            "tau_fe": "source-contribution-weighted Fe-shell optical depth",
            "tau_pb": "source-contribution-weighted Pb-shell optical depth",
            "tau_obstacle": (
                "source-contribution-weighted line-resolved obstacle optical depth "
                "over the same aperture rays used by the PF kernel"
            ),
            "distance_m": (
                "source-contribution-weighted source-detector center distance "
                "used by the detector-cps@1m cone geometry"
            ),
            "distance_shield": (
                "source-contribution-weighted product of source-detector "
                "distance and line-resolved shield optical depth"
            ),
            "distance_fe": (
                "source-contribution-weighted product of source-detector "
                "distance and line-resolved Fe optical depth"
            ),
            "distance_pb": (
                "source-contribution-weighted product of source-detector "
                "distance and line-resolved Pb optical depth"
            ),
            "distance_obstacle": (
                "source-contribution-weighted product of source-detector "
                "distance and line-resolved obstacle optical depth"
            ),
        },
        "by_isotope": by_isotope,
    }


def _fit_transport_response_pair_scales(
    groups: list[dict[str, Any]],
    beta: NDArray[np.float64],
    *,
    min_pair_fit_points: int,
    pair_shrinkage_count: float,
) -> dict[str, float]:
    """Fit shrinkage pair intercepts after isotope-level optical-depth terms."""
    grouped: dict[int, list[float]] = {}
    for group in groups:
        pair_id = int(group["shield_pair_id"])
        predicted_log = _log_sourcewise_transport_prediction(group, beta)
        observed_log = np.log(
            max(float(group["transport_truth_counts"]), 1.0e-12)
            / max(float(group["theory_counts"]), 1.0e-12)
        )
        grouped.setdefault(pair_id, []).append(float(observed_log - predicted_log))
    pair_scales: dict[str, float] = {}
    for pair_id, residuals in sorted(grouped.items()):
        if len(residuals) < max(int(min_pair_fit_points), 1):
            continue
        shrink = len(residuals) / (
            len(residuals) + max(float(pair_shrinkage_count), 0.0)
        )
        residual_log = float(np.mean(np.asarray(residuals, dtype=float))) * shrink
        base_log = float(beta[0]) + residual_log
        pair_scales[str(pair_id)] = float(np.exp(base_log))
    return pair_scales


def _transport_response_cap_value(
    feature_caps: dict[str, float],
    *names: str,
) -> float | None:
    """Return the first finite nonnegative cap from a calibration cap payload."""
    for name in names:
        if name not in feature_caps:
            continue
        value = float(feature_caps[name])
        if np.isfinite(value) and value >= 0.0:
            return value
    return None


def _transport_response_feature_vector(
    term: dict[str, Any],
    feature_caps: dict[str, float] | None = None,
) -> NDArray[np.float64]:
    """Return the runtime-capped optical-depth vector for one contribution."""
    caps = (
        dict(DEFAULT_TRANSPORT_RESPONSE_TAU_FEATURE_CAPS)
        if feature_caps is None
        else dict(feature_caps)
    )
    shield_tau_raw = max(float(term.get("shield_tau_feature", 0.0)), 0.0)
    obstacle_tau_raw = max(float(term.get("obstacle_tau_feature", 0.0)), 0.0)
    fe_tau_raw = max(float(term.get("fe_tau_feature", 0.0)), 0.0)
    pb_tau_raw = max(float(term.get("pb_tau_feature", 0.0)), 0.0)
    distance = max(float(term.get("distance_feature", 0.0)), 0.0)
    distance_shield_raw = max(
        float(term.get("distance_shield_feature", distance * shield_tau_raw)),
        0.0,
    )
    distance_fe_raw = max(
        float(term.get("distance_fe_feature", distance * fe_tau_raw)),
        0.0,
    )
    distance_pb_raw = max(
        float(term.get("distance_pb_feature", distance * pb_tau_raw)),
        0.0,
    )
    distance_obstacle_raw = max(
        float(term.get("distance_obstacle_feature", distance * obstacle_tau_raw)),
        0.0,
    )
    shield_tau = capped_transport_response_feature(
        shield_tau_raw,
        _transport_response_cap_value(caps, "shield", "shield_tau", "tau_shield"),
    )
    obstacle_tau = capped_transport_response_feature(
        obstacle_tau_raw,
        _transport_response_cap_value(
            caps,
            "obstacle",
            "obstacle_tau",
            "tau_obstacle",
        ),
    )
    fe_tau = capped_transport_response_feature(
        fe_tau_raw,
        _transport_response_cap_value(caps, "fe", "fe_tau", "tau_fe"),
    )
    pb_tau = capped_transport_response_feature(
        pb_tau_raw,
        _transport_response_cap_value(caps, "pb", "pb_tau", "tau_pb"),
    )
    distance_shield = capped_transport_response_feature(
        distance_shield_raw,
        _transport_response_cap_value(
            caps,
            "distance_shield",
            "distance_shield_tau",
            "source_distance_shield_tau",
        ),
    )
    distance_fe = capped_transport_response_feature(
        distance_fe_raw,
        _transport_response_cap_value(
            caps,
            "distance_fe",
            "distance_fe_tau",
            "source_distance_fe_tau",
        ),
    )
    distance_pb = capped_transport_response_feature(
        distance_pb_raw,
        _transport_response_cap_value(
            caps,
            "distance_pb",
            "distance_pb_tau",
            "source_distance_pb_tau",
        ),
    )
    distance_obstacle = capped_transport_response_feature(
        distance_obstacle_raw,
        _transport_response_cap_value(
            caps,
            "distance_obstacle",
            "distance_obstacle_tau",
            "source_distance_obstacle_tau",
        ),
    )
    return np.asarray(
        [
            1.0,
            shield_tau,
            obstacle_tau,
            shield_tau * shield_tau,
            obstacle_tau * obstacle_tau,
            shield_tau * obstacle_tau,
            fe_tau,
            pb_tau,
            fe_tau * fe_tau,
            pb_tau * pb_tau,
            fe_tau * pb_tau,
            fe_tau * obstacle_tau,
            pb_tau * obstacle_tau,
            distance,
            distance_shield,
            distance_fe,
            distance_pb,
            distance_obstacle,
        ],
        dtype=float,
    )


def _aggregate_transport_response_initial_beta(
    groups: list[dict[str, Any]],
    *,
    ridge: float,
) -> NDArray[np.float64]:
    """Return aggregate-feature initial coefficients for source-wise fitting."""
    rows: list[list[float]] = []
    targets: list[float] = []
    weights: list[float] = []
    for group in groups:
        source_terms = group.get("source_terms", [])
        if not isinstance(source_terms, list) or not source_terms:
            continue
        total = sum(max(float(term.get("counts", 0.0)), 0.0) for term in source_terms)
        if total <= 0.0:
            continue
        weighted_feature = (
            sum(
                max(float(term.get("counts", 0.0)), 0.0)
                * _transport_response_feature_vector(term)
                for term in source_terms
            )
            / total
        )
        rows.append(weighted_feature.tolist())
        targets.append(
            np.log(
                max(float(group["transport_truth_counts"]), 1.0e-12)
                / max(float(group["theory_counts"]), 1.0e-12)
            )
        )
        weights.append(max(float(group.get("weight", 1.0)), 1.0e-12))
    if not rows:
        return np.zeros(18, dtype=float)
    return _weighted_ridge_fit(
        np.asarray(rows, dtype=float),
        np.asarray(targets, dtype=float),
        np.asarray(weights, dtype=float),
        ridge=float(ridge),
    )


def _fit_sourcewise_transport_response_beta(
    groups: list[dict[str, Any]],
    initial_beta: NDArray[np.float64],
    *,
    ridge: float,
) -> NDArray[np.float64]:
    """Fit source-wise optical-depth response coefficients."""
    beta0 = np.asarray(initial_beta, dtype=float).reshape(-1)
    if beta0.size != 18 or not _SCIPY_AVAILABLE or least_squares is None:
        return beta0
    fit_terms = _sourcewise_transport_fit_terms(groups)
    if not fit_terms:
        return beta0

    def residual(beta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return weighted log-count residuals plus ridge regularization."""
        values: list[float] = []
        for features, log_counts, log_theory, observed_log, root_weight in fit_terms:
            predicted_log, _softmax = _sourcewise_prediction_and_weights(
                features,
                log_counts,
                log_theory,
                beta,
            )
            values.append((predicted_log - observed_log) * root_weight)
        if ridge > 0.0:
            values.extend((np.sqrt(float(ridge)) * beta[1:]).tolist())
        return np.asarray(values, dtype=float)

    def jacobian(beta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the analytic residual Jacobian for source-wise fitting."""
        rows: list[NDArray[np.float64]] = []
        for features, log_counts, log_theory, _observed_log, root_weight in fit_terms:
            _predicted_log, softmax = _sourcewise_prediction_and_weights(
                features,
                log_counts,
                log_theory,
                beta,
            )
            rows.append(float(root_weight) * (softmax @ features))
        if ridge > 0.0:
            ridge_rows = np.zeros((17, beta0.size), dtype=float)
            ridge_rows[:, 1:] = np.eye(17, dtype=float) * np.sqrt(float(ridge))
            rows.extend(ridge_rows)
        return np.vstack(rows).astype(float, copy=False)

    fit = least_squares(
        residual,
        beta0,
        jac=jacobian,
        max_nfev=200,
        xtol=1.0e-10,
        ftol=1.0e-10,
        gtol=1.0e-10,
    )
    return np.asarray(fit.x, dtype=float)


def _sourcewise_transport_fit_terms(
    groups: list[dict[str, Any]],
) -> list[tuple[NDArray[np.float64], NDArray[np.float64], float, float, float]]:
    """Precompute capped source features for source-wise transport fitting."""
    fit_terms: list[
        tuple[NDArray[np.float64], NDArray[np.float64], float, float, float]
    ] = []
    for group in groups:
        source_terms = group.get("source_terms", [])
        if not isinstance(source_terms, list) or not source_terms:
            continue
        features: list[NDArray[np.float64]] = []
        log_counts: list[float] = []
        for term in source_terms:
            if not isinstance(term, dict):
                continue
            counts = max(float(term.get("counts", 0.0)), 0.0)
            if counts <= 0.0:
                continue
            features.append(_transport_response_feature_vector(term))
            log_counts.append(float(np.log(counts)))
        if not features:
            continue
        theory = max(float(group.get("theory_counts", 0.0)), 1.0e-12)
        truth = max(float(group.get("transport_truth_counts", 0.0)), 1.0e-12)
        observed_log = float(np.log(truth / theory))
        root_weight = float(np.sqrt(max(float(group.get("weight", 1.0)), 1.0e-12)))
        fit_terms.append(
            (
                np.vstack(features).astype(float, copy=False),
                np.asarray(log_counts, dtype=float),
                float(np.log(theory)),
                observed_log,
                root_weight,
            )
        )
    return fit_terms


def _sourcewise_prediction_and_weights(
    features: NDArray[np.float64],
    log_counts: NDArray[np.float64],
    log_theory: float,
    beta: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    """Return source-wise log prediction and source contribution weights."""
    values = np.asarray(log_counts, dtype=float) + (
        np.asarray(features, dtype=float) @ np.asarray(beta, dtype=float)
    )
    max_value = float(np.max(values))
    weights = np.exp(values - max_value)
    weight_sum = max(float(np.sum(weights)), 1.0e-300)
    softmax = weights / weight_sum
    log_sum = max_value + float(np.log(weight_sum))
    return float(log_sum - float(log_theory)), np.asarray(softmax, dtype=float)


def _log_sourcewise_transport_prediction(
    group: dict[str, Any],
    beta: NDArray[np.float64],
) -> float:
    """Return log(predicted_count/current_pf_target_count) for one group."""
    beta_arr = np.asarray(beta, dtype=float).reshape(-1)
    if beta_arr.size != 18:
        raise ValueError("transport response beta must contain eighteen coefficients.")
    source_terms = group.get("source_terms", [])
    if not isinstance(source_terms, list) or not source_terms:
        return 0.0
    values: list[float] = []
    for term in source_terms:
        if not isinstance(term, dict):
            continue
        counts = max(float(term.get("counts", 0.0)), 0.0)
        if counts <= 0.0:
            continue
        feature = _transport_response_feature_vector(term)
        values.append(float(np.log(counts) + feature @ beta_arr))
    if not values:
        return 0.0
    max_value = max(values)
    log_sum = max_value + float(
        np.log(sum(np.exp(value - max_value) for value in values))
    )
    theory = max(float(group.get("theory_counts", 0.0)), 1.0e-12)
    return float(log_sum - np.log(theory))


def _weighted_ridge_fit(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    weights: NDArray[np.float64],
    *,
    ridge: float,
) -> NDArray[np.float64]:
    """Return weighted ridge least-squares coefficients."""
    if x.ndim != 2 or y.ndim != 1 or weights.ndim != 1:
        raise ValueError("x must be 2-D and y/weights must be 1-D.")
    if x.shape[0] != y.shape[0] or y.shape[0] != weights.shape[0]:
        raise ValueError("x, y, and weights must contain the same number of rows.")
    root_w = np.sqrt(np.clip(weights, 1.0e-12, None))
    xw = x * root_w[:, None]
    yw = y * root_w
    penalty = np.eye(x.shape[1], dtype=float) * max(float(ridge), 0.0)
    penalty[0, 0] = 0.0
    lhs = xw.T @ xw + penalty
    rhs = xw.T @ yw
    return np.linalg.solve(lhs, rhs)


def _pf_transport_response_model_residual_summary(
    groups: list[dict[str, Any]],
    model: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """Summarize residuals after applying the optical-depth response model."""
    by_isotope: dict[str, list[float]] = {isotope: [] for isotope in ISOTOPES}
    model_by_isotope = model.get("by_isotope", {})
    if not isinstance(model_by_isotope, dict):
        model_by_isotope = {}
    for group in groups:
        isotope = str(group.get("isotope"))
        payload = model_by_isotope.get(isotope, {})
        if not isinstance(payload, dict):
            payload = {}
        expected = _transport_response_model_expected_counts(group, payload)
        truth = max(float(group["transport_truth_counts"]), 1.0e-12)
        by_isotope.setdefault(isotope, []).append(abs(expected - truth) / truth)
    return {
        isotope: _relative_error_distribution(values)
        for isotope, values in sorted(by_isotope.items())
    }


def _transport_response_model_expected_counts(
    group: dict[str, Any],
    payload: dict[str, Any],
) -> float:
    """Return source-summed expected counts under one transport-response payload."""
    pair_id = int(group.get("shield_pair_id", -1))
    total = 0.0
    for term in group.get("source_terms", []):
        if not isinstance(term, dict):
            continue
        counts = max(float(term.get("counts", 0.0)), 0.0)
        if counts <= 0.0:
            continue
        total += counts * transport_response_factor_from_payload(
            payload,
            pair_id=pair_id,
            shield_tau_feature=float(term.get("shield_tau_feature", 0.0)),
            obstacle_tau_feature=float(term.get("obstacle_tau_feature", 0.0)),
            fe_tau_feature=float(term.get("fe_tau_feature", 0.0)),
            pb_tau_feature=float(term.get("pb_tau_feature", 0.0)),
            distance_feature=float(term.get("distance_feature", 0.0)),
            distance_shield_feature=float(
                term.get(
                    "distance_shield_feature",
                    float(term.get("distance_feature", 0.0))
                    * float(term.get("shield_tau_feature", 0.0)),
                )
            ),
        )
    return max(float(total), 1.0e-12)


def _effective_transport_scale_snippet(
    runtime_config: dict[str, Any],
    calibration_payload: dict[str, Any],
) -> dict[str, Any]:
    """Return config scales after applying PF-to-transport calibration."""
    return _effective_measurement_scale_snippet(runtime_config, calibration_payload)


def _effective_measurement_scale_snippet(
    runtime_config: dict[str, Any],
    calibration_payload: dict[str, Any],
) -> dict[str, Any]:
    """Return measurement scales after applying a fitted multiplier payload."""
    base_iso_scales = runtime_config.get("measurement_scale_by_isotope", {})
    if not isinstance(base_iso_scales, dict):
        base_iso_scales = {}
    base_pair_scales = runtime_config.get("measurement_scale_by_isotope_and_pair", {})
    if not isinstance(base_pair_scales, dict):
        base_pair_scales = {}
    fit_iso_scales = calibration_payload.get("scale_by_isotope", {})
    fit_pair_scales = calibration_payload.get("scale_by_isotope_and_pair", {})
    effective_iso: dict[str, float] = {}
    effective_pair: dict[str, dict[str, float]] = {}
    for isotope in ISOTOPES:
        base_iso = float(base_iso_scales.get(isotope, 1.0))
        fit_iso = float(fit_iso_scales.get(isotope, 1.0))
        effective_iso[isotope] = max(base_iso * fit_iso, 0.0)
        base_pairs = base_pair_scales.get(isotope, {})
        if not isinstance(base_pairs, dict):
            base_pairs = {}
        fit_pairs = fit_pair_scales.get(isotope, {})
        if not isinstance(fit_pairs, dict):
            fit_pairs = {}
        pair_payload: dict[str, float] = {}
        for pair_key, fit_value in fit_pairs.items():
            base_value = base_pairs.get(str(pair_key), base_pairs.get(int(pair_key)))
            if base_value is None:
                base_value = base_iso
            pair_payload[str(pair_key)] = max(float(base_value) * float(fit_value), 0.0)
        if pair_payload:
            effective_pair[isotope] = pair_payload
    return {
        "measurement_scale_by_isotope": effective_iso,
        "measurement_scale_by_isotope_and_pair": effective_pair,
    }


def _pf_transport_residual_summary(
    results: list[dict[str, Any]],
    calibration_payload: dict[str, Any],
    min_target: float,
    *,
    num_orientations: int = 8,
) -> dict[str, dict[str, float]]:
    """Summarize PF-target residuals after fitted transport calibration."""
    scale_by_isotope = calibration_payload.get("scale_by_isotope", {})
    scale_by_pair = calibration_payload.get("scale_by_isotope_and_pair", {})
    values_by_iso: dict[str, list[float]] = {iso: [] for iso in ISOTOPES}
    for result in results:
        case = result["case"]
        if not bool(case.get("include_in_accuracy_summary", True)):
            continue
        pair_id = int(case["fe_index"]) * max(int(num_orientations), 1) + int(
            case["pb_index"]
        )
        pair_key = str(pair_id)
        for isotope, item in result["per_isotope"].items():
            theory = float(item.get("target_pf_counts", 0.0))
            truth = float(item.get("transport_truth_counts", 0.0))
            if theory < float(min_target) or truth < float(min_target):
                continue
            iso_scales = scale_by_pair.get(str(isotope), {})
            if isinstance(iso_scales, dict) and pair_key in iso_scales:
                scale = float(iso_scales[pair_key])
            else:
                scale = float(scale_by_isotope.get(str(isotope), 1.0))
            expected = max(theory * scale, 1.0e-12)
            values_by_iso.setdefault(str(isotope), []).append(
                abs(expected - truth) / max(abs(truth), 1.0)
            )
    summary: dict[str, dict[str, float]] = {}
    for isotope in sorted(values_by_iso):
        rel = np.asarray(values_by_iso.get(isotope, []), dtype=float)
        summary[isotope] = {
            "num_points": float(rel.size),
            "median_abs_relative_error": (
                float(np.median(rel)) if rel.size else float("nan")
            ),
            "p90_abs_relative_error": (
                float(np.percentile(rel, 90.0)) if rel.size else float("nan")
            ),
            "p99_abs_relative_error": (
                float(np.percentile(rel, 99.0)) if rel.size else float("nan")
            ),
            "max_abs_relative_error": (
                float(np.max(rel)) if rel.size else float("nan")
            ),
        }
    return summary


def _response_poisson_pair_coverage(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return count coverage by isotope and shield-pair id for calibration."""
    coverage: dict[str, dict[str, int]] = {}
    for record in records:
        isotope = str(record["isotope"])
        pair_id = str(int(record["shield_pair_id"]))
        isotope_coverage = coverage.setdefault(isotope, {})
        isotope_coverage[pair_id] = int(isotope_coverage.get(pair_id, 0)) + 1
    return {
        "num_pairs_by_isotope": {
            isotope: len(pair_counts) for isotope, pair_counts in coverage.items()
        },
        "records_by_isotope_and_pair": coverage,
    }


def _calibrated_residual_summary(
    results: list[dict[str, Any]],
    calibration_payload: dict[str, Any],
    min_target: float,
    *,
    num_orientations: int = 8,
) -> dict[str, dict[str, float]]:
    """Summarize response-Poisson residuals after applying fitted scales."""
    scale_by_isotope = calibration_payload.get("scale_by_isotope", {})
    scale_by_pair = calibration_payload.get("scale_by_isotope_and_pair", {})
    values_by_iso: dict[str, list[float]] = {iso: [] for iso in ISOTOPES}
    normalized_by_iso: dict[str, list[float]] = {iso: [] for iso in ISOTOPES}
    for result in results:
        case = result["case"]
        if not bool(case.get("include_in_accuracy_summary", True)):
            continue
        pair_id = int(case["fe_index"]) * max(int(num_orientations), 1) + int(
            case["pb_index"]
        )
        pair_key = str(pair_id)
        for isotope, item in result["per_isotope"].items():
            theory = float(item.get("target_pf_counts", 0.0))
            if theory < float(min_target):
                continue
            net = float(item.get("method_counts", {}).get("response_poisson", 0.0))
            variance = max(float(item.get("response_poisson_variance", 1.0)), 1.0)
            iso_scales = scale_by_pair.get(str(isotope), {})
            if isinstance(iso_scales, dict) and pair_key in iso_scales:
                scale = float(iso_scales[pair_key])
            else:
                scale = float(scale_by_isotope.get(str(isotope), 1.0))
            expected = max(theory * scale, 1.0e-12)
            values_by_iso.setdefault(str(isotope), []).append(
                abs(net - expected) / expected
            )
            normalized_by_iso.setdefault(str(isotope), []).append(
                abs(net - expected) / np.sqrt(variance)
            )
    summary: dict[str, dict[str, float]] = {}
    for isotope in sorted(values_by_iso):
        rel = np.asarray(values_by_iso.get(isotope, []), dtype=float)
        norm = np.asarray(normalized_by_iso.get(isotope, []), dtype=float)
        summary[isotope] = {
            "num_points": float(rel.size),
            "median_abs_relative_error": (
                float(np.median(rel)) if rel.size else float("nan")
            ),
            "p90_abs_relative_error": (
                float(np.percentile(rel, 90.0)) if rel.size else float("nan")
            ),
            "p99_abs_relative_error": (
                float(np.percentile(rel, 99.0)) if rel.size else float("nan")
            ),
            "max_abs_relative_error": (
                float(np.max(rel)) if rel.size else float("nan")
            ),
            "median_abs_sigma_residual": (
                float(np.median(norm)) if norm.size else float("nan")
            ),
            "p99_abs_sigma_residual": (
                float(np.percentile(norm, 99.0)) if norm.size else float("nan")
            ),
        }
    return summary


def _runtime_transport_response_payload(
    summary: dict[str, Any],
) -> dict[str, Any] | None:
    """Return a runtime-loadable PF transport response payload if available."""
    calibration = summary.get("pf_transport_response_calibration")
    if not isinstance(calibration, dict):
        return None
    model = calibration.get("transport_response_model")
    if not isinstance(model, dict):
        return None
    return {"pf_transport_response_model": model}


def write_outputs(
    output_dir: Path,
    results: list[dict[str, Any]],
    spectra: dict[str, np.ndarray],
    summary: dict[str, Any],
    *,
    cases: list[ValidationCase] | None = None,
    write_detailed_results: bool = True,
) -> None:
    """Write validation outputs to JSON, CSV, and NPZ files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    transport_payload = _runtime_transport_response_payload(summary)
    if transport_payload is not None:
        (output_dir / "pf_transport_response_model.json").write_text(
            json.dumps(transport_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if cases is not None:
        (output_dir / "case_manifest.json").write_text(
            json.dumps([case_to_dict(case) for case in cases], indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    if bool(write_detailed_results):
        (output_dir / "results.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "cases.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    rows = flatten_records(results)
    csv_path = output_dir / "records.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    np.savez_compressed(output_dir / "spectra.npz", **spectra)


def summarize_requested_cases(
    cases: list[ValidationCase],
    *,
    case_total: int | None = None,
) -> dict[str, Any]:
    """Return a compact manifest summary for requested validation cases."""
    pair_counts: dict[str, int] = {}
    source_counts_by_isotope = {isotope: 0 for isotope in ISOTOPES}
    source_histogram: dict[str, int] = {}
    for case in cases:
        pair_key = f"{int(case.fe_index)}:{int(case.pb_index)}"
        pair_counts[pair_key] = int(pair_counts.get(pair_key, 0)) + 1
        source_histogram[str(len(case.sources))] = (
            int(source_histogram.get(str(len(case.sources)), 0)) + 1
        )
        for source in case.sources:
            source_counts_by_isotope[source.isotope] = (
                int(source_counts_by_isotope.get(source.isotope, 0)) + 1
            )
    case_names = [case.name for case in cases]
    return {
        "num_requested_cases": int(case_total)
        if case_total is not None
        else len(cases),
        "first_case_names": case_names[:10],
        "last_case_names": case_names[-10:] if len(case_names) > 10 else case_names,
        "source_counts_by_isotope": source_counts_by_isotope,
        "sources_per_case_histogram": source_histogram,
        "shield_pair_counts": pair_counts,
    }


def summarize_detector_selection_diagnostics(
    cases: list[ValidationCase],
) -> dict[str, Any]:
    """Summarize generated detector-pose stress metrics by base scenario."""
    scenarios: dict[str, ValidationCase] = {}
    for case in cases:
        base_name = str(case.name).split("_pair", maxsplit=1)[0]
        scenarios.setdefault(base_name, case)
    mode_counts: dict[str, int] = {}
    values_by_metric: dict[str, list[float]] = {
        "max_obstacle_tau": [],
        "target_count_imbalance": [],
        "shield_dynamic_range": [],
        "min_isotope_target": [],
    }
    rows: list[dict[str, Any]] = []
    for base_name, case in sorted(scenarios.items()):
        metadata = (
            case.generation_metadata
            if isinstance(case.generation_metadata, dict)
            else {}
        )
        mode = str(metadata.get("detector_selection_effective_mode", "unknown"))
        mode_counts[mode] = int(mode_counts.get(mode, 0)) + 1
        row = {
            "case": base_name,
            "mode": mode,
            "detector_pose_xyz": [float(value) for value in case.detector_pose_xyz],
            "max_obstacle_tau": float(metadata.get("max_obstacle_tau", 0.0)),
            "target_count_imbalance": float(
                metadata.get("target_count_imbalance", 1.0)
            ),
            "shield_dynamic_range": float(metadata.get("shield_dynamic_range", 1.0)),
            "min_isotope_target": float(metadata.get("min_isotope_target", 0.0)),
        }
        rows.append(row)
        for metric in values_by_metric:
            values_by_metric[metric].append(float(row[metric]))
    metric_summary = {
        metric: _relative_error_distribution(values)
        for metric, values in values_by_metric.items()
    }
    return {
        "num_base_scenarios": len(scenarios),
        "mode_counts": mode_counts,
        "metric_summary": metric_summary,
        "top_obstacle_tau": sorted(
            rows,
            key=lambda row: row["max_obstacle_tau"],
            reverse=True,
        )[:10],
        "top_count_imbalance": sorted(
            rows,
            key=lambda row: row["target_count_imbalance"],
            reverse=True,
        )[:10],
        "top_shield_dynamic_range": sorted(
            rows,
            key=lambda row: row["shield_dynamic_range"],
            reverse=True,
        )[:10],
    }


def build_summary(
    *,
    config_path: Path,
    output_dir: Path,
    cases: list[ValidationCase],
    results: list[dict[str, Any]],
    args: argparse.Namespace,
    runtime_config: dict[str, Any],
    sweep_start: float,
    interrupted: bool,
    case_total: int | None = None,
) -> dict[str, Any]:
    """Build a JSON-compatible summary for complete or partial validation output."""
    requested_cases = summarize_requested_cases(cases, case_total=case_total)
    if case_total is not None:
        requested_cases["num_requested_cases"] = int(case_total)
        requested_cases["num_seen_cases"] = int(len(cases))
        requested_cases["complete_manifest"] = bool(len(cases) >= int(case_total))
    return {
        "config": config_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "num_cases": len(results),
        "num_requested_cases": (
            int(case_total) if case_total is not None else len(cases)
        ),
        "interrupted": bool(interrupted),
        "environment_sweep": bool(args.environment_sweep),
        "num_environments": int(args.num_environments)
        if args.environment_sweep
        else None,
        "measurement_points_per_environment": (
            int(args.measurement_points_per_environment)
            if args.environment_sweep
            else None
        ),
        "rotations_per_point": int(args.rotations_per_point)
        if args.environment_sweep
        else None,
        "elapsed_s": float(time.time() - sweep_start),
        "min_target_counts": float(args.min_target_counts),
        "accuracy_summary": summarize_accuracy(results, float(args.min_target_counts)),
        "response_poisson_covariance_summary": (
            summarize_response_poisson_covariance(
                results,
                float(args.min_target_counts),
            )
        ),
        "pf_count_likelihood_summary": (
            summarize_pf_count_likelihood_diagnostics(
                results,
                float(args.min_target_counts),
            )
        ),
        "shield_pair_diagnostics": summarize_shield_pair_diagnostics(
            results,
            float(args.min_target_counts),
        ),
        "response_poisson_pf_target_calibration": (
            summarize_response_poisson_calibration(
                results,
                float(args.min_target_counts),
                runtime_config,
                min_pair_fit_points=int(args.calibration_min_pair_points),
                pair_shrinkage_count=float(args.calibration_pair_shrinkage_count),
                holdout_fraction=float(args.calibration_holdout_fraction),
                holdout_seed=int(args.calibration_holdout_seed),
            )
        ),
        "pf_transport_response_calibration": summarize_pf_transport_calibration(
            results,
            float(args.min_target_counts),
            runtime_config,
            min_pair_fit_points=int(args.calibration_min_pair_points),
            pair_shrinkage_count=float(args.calibration_pair_shrinkage_count),
            holdout_fraction=float(args.calibration_holdout_fraction),
            holdout_seed=int(args.calibration_holdout_seed),
        ),
        "requested_cases": requested_cases,
        "detector_selection_diagnostics": summarize_detector_selection_diagnostics(
            cases
        ),
        "case_generation": {
            "case_seed": int(args.case_seed),
            "num_cases_requested": int(args.num_cases),
            "sources_per_isotope": int(args.sources_per_isotope),
            "blocked_fraction": float(args.blocked_fraction),
            "passage_width_m": float(args.passage_width_m),
            "multi_source_detector_attempts": int(args.multi_source_detector_attempts),
            "multi_source_min_isotope_target_counts": float(
                args.multi_source_min_isotope_target_counts
            ),
            "all_shield_pairs_per_scenario": bool(args.all_shield_pairs_per_scenario),
            "multi_source_detector_selection": str(
                args.multi_source_detector_selection
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/geant4/variance_reduction_external_no_isaac_32threads.json",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help="Run only the named case; repeatable.",
    )
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--min-target-counts", type=float, default=25.0)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--thread-count", type=int, default=None)
    parser.add_argument(
        "--usd-path",
        default=None,
        help=(
            "Optional external USD to import into Geant4. By default validation "
            "uses only the generated scene geometry so the PF target and Geant4 "
            "transport share the same obstacle set."
        ),
    )
    parser.add_argument(
        "--primary-sampling-fraction",
        type=float,
        default=None,
        help=(
            "Override Geant4 primary_sampling_fraction. Values below 1 require "
            "an input config with accelerated_weighted_transport_enable=true."
        ),
    )
    parser.add_argument(
        "--random-seed-base",
        type=int,
        default=None,
        help="Override only the Geant4 transport seed base for repeat diagnostics.",
    )
    parser.add_argument("--num-cases", type=int, default=50)
    parser.add_argument("--case-seed", type=int, default=20260430)
    parser.add_argument("--dwell-time-s", type=float, default=30.0)
    parser.add_argument(
        "--intensity-min-cps-1m",
        type=float,
        default=DEFAULT_VALIDATION_INTENSITY_MIN_CPS_1M,
    )
    parser.add_argument(
        "--intensity-max-cps-1m",
        type=float,
        default=DEFAULT_VALIDATION_INTENSITY_MAX_CPS_1M,
    )
    parser.add_argument(
        "--environment-sweep",
        action="store_true",
        help=(
            "Generate random known-material obstacle environments, multiple "
            "measurement points, and multiple shield rotations per point."
        ),
    )
    parser.add_argument(
        "--multi-isotope-multi-source-cases",
        action="store_true",
        help=(
            "Generate cases where Cs-137, Co-60, and Eu-154 each have multiple "
            "sources on room or obstacle surfaces."
        ),
    )
    parser.add_argument(
        "--all-shield-pairs-per-scenario",
        action="store_true",
        help=(
            "Run all 64 Fe/Pb shield-pair configurations for each generated "
            "multi-isotope scenario while keeping the environment, sources, "
            "and detector pose fixed."
        ),
    )
    parser.add_argument(
        "--sources-per-isotope",
        type=int,
        default=2,
        help="Number of sources for each isotope in multi-isotope cases.",
    )
    parser.add_argument(
        "--multi-source-min-isotope-target-counts",
        type=float,
        default=0.0,
        help=(
            "Minimum PF-theory counts required for every isotope when sampling "
            "multi-isotope multi-source detector poses."
        ),
    )
    parser.add_argument(
        "--multi-source-detector-attempts",
        type=int,
        default=24,
        help="Detector-pose resampling attempts for each multi-isotope case.",
    )
    parser.add_argument(
        "--multi-source-detector-selection",
        choices=DETECTOR_SELECTION_MODES,
        default="balanced",
        help=(
            "Detector-pose selection policy for multi-isotope cases. "
            "balanced keeps all isotope target counts visible; "
            "obstacle_extreme favors high obstacle attenuation; "
            "count_imbalance favors isotope count imbalance; "
            "shield_dynamic_range favors large variation over the 64 shield pairs; "
            "mixed_stress cycles through the stress policies."
        ),
    )
    parser.add_argument("--num-environments", type=int, default=10)
    parser.add_argument("--measurement-points-per-environment", type=int, default=24)
    parser.add_argument("--rotations-per-point", type=int, default=8)
    parser.add_argument("--blocked-fraction", type=float, default=0.35)
    parser.add_argument("--passage-width-m", type=float, default=2.0)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=8,
        help="Write partial validation outputs every N completed cases; set 0 to disable.",
    )
    parser.add_argument(
        "--calibration-min-pair-points",
        type=int,
        default=DEFAULT_CALIBRATION_MIN_PAIR_POINTS,
        help=(
            "Minimum records required before reporting a shield-pair response "
            "scale; pairs below this use the isotope-wise scale."
        ),
    )
    parser.add_argument(
        "--calibration-pair-shrinkage-count",
        type=float,
        default=DEFAULT_CALIBRATION_PAIR_SHRINKAGE_COUNT,
        help=(
            "Pseudo-count strength that shrinks pair response scales toward "
            "the isotope-wise scale."
        ),
    )
    parser.add_argument(
        "--calibration-holdout-fraction",
        type=float,
        default=DEFAULT_CALIBRATION_HOLDOUT_FRACTION,
        help="Fraction of response-Poisson calibration records held out for validation.",
    )
    parser.add_argument(
        "--calibration-holdout-seed",
        type=int,
        default=DEFAULT_CALIBRATION_HOLDOUT_SEED,
        help="Random seed for the deterministic calibration holdout split.",
    )
    parser.add_argument(
        "--hand-authored-cases",
        action="store_true",
        help="Use the legacy small hand-authored case set instead of generated 50-case validation.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the Geant4 spectrum-decomposition validation sweep."""
    args = parse_args()
    config_path = resolve_path(args.config)
    runtime_config = load_runtime_config(config_path.as_posix())
    executable_path = runtime_config.get("executable_path", "build/geant4_sidecar")
    runtime_config["executable_path"] = resolve_path(str(executable_path)).as_posix()
    runtime_config["timeout_s"] = float(args.timeout_s)
    if args.thread_count is not None:
        runtime_config["thread_count"] = int(args.thread_count)
    if args.primary_sampling_fraction is not None:
        runtime_config["primary_sampling_fraction"] = float(
            args.primary_sampling_fraction
        )
    if args.random_seed_base is not None:
        runtime_config["random_seed_base"] = int(args.random_seed_base)
    validated_app_config = Geant4AppConfig.from_dict(runtime_config)
    runtime_config["primary_sampling_fraction"] = (
        validated_app_config.primary_sampling_fraction
    )
    runtime_config["validation_usd_path"] = (
        resolve_path(str(args.usd_path)).as_posix() if args.usd_path else None
    )
    runtime_config["engine_mode"] = "external"
    runtime_config["physics_profile"] = "balanced"

    mu_by_isotope = mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM, isotopes=ISOTOPES)

    stream_multi_isotope_cases = bool(
        args.multi_isotope_multi_source_cases and not args.case
    )
    if stream_multi_isotope_cases:
        base_case_total = max(0, int(args.num_cases))
        if args.max_cases is not None:
            base_case_total = min(base_case_total, max(int(args.max_cases), 0))
        case_total = (
            base_case_total * 64
            if bool(args.all_shield_pairs_per_scenario)
            else base_case_total
        )
        cases_iter = iter_generated_multi_isotope_source_cases(
            num_cases=base_case_total,
            seed=int(args.case_seed),
            dwell_time_s=float(args.dwell_time_s),
            intensity_min_cps_1m=float(args.intensity_min_cps_1m),
            intensity_max_cps_1m=float(args.intensity_max_cps_1m),
            sources_per_isotope=int(args.sources_per_isotope),
            blocked_fraction=float(args.blocked_fraction),
            passage_width_m=float(args.passage_width_m),
            min_isotope_target_counts=float(
                args.multi_source_min_isotope_target_counts
            ),
            detector_attempts=int(args.multi_source_detector_attempts),
            runtime_config=runtime_config,
            mu_by_isotope=mu_by_isotope,
            all_shield_pairs_per_scenario=bool(args.all_shield_pairs_per_scenario),
            detector_selection_mode=str(args.multi_source_detector_selection),
        )
        cases: list[ValidationCase] = []
    else:
        if args.multi_isotope_multi_source_cases:
            all_cases = generated_multi_isotope_source_cases(
                num_cases=int(args.num_cases),
                seed=int(args.case_seed),
                dwell_time_s=float(args.dwell_time_s),
                intensity_min_cps_1m=float(args.intensity_min_cps_1m),
                intensity_max_cps_1m=float(args.intensity_max_cps_1m),
                sources_per_isotope=int(args.sources_per_isotope),
                blocked_fraction=float(args.blocked_fraction),
                passage_width_m=float(args.passage_width_m),
                min_isotope_target_counts=float(
                    args.multi_source_min_isotope_target_counts
                ),
                detector_attempts=int(args.multi_source_detector_attempts),
                runtime_config=runtime_config,
                mu_by_isotope=mu_by_isotope,
                all_shield_pairs_per_scenario=bool(args.all_shield_pairs_per_scenario),
                detector_selection_mode=str(args.multi_source_detector_selection),
            )
        elif args.environment_sweep:
            all_cases = generated_environment_sweep_cases(
                num_environments=int(args.num_environments),
                measurement_points_per_environment=int(
                    args.measurement_points_per_environment
                ),
                rotations_per_point=int(args.rotations_per_point),
                seed=int(args.case_seed),
                dwell_time_s=float(args.dwell_time_s),
                intensity_min_cps_1m=float(args.intensity_min_cps_1m),
                intensity_max_cps_1m=float(args.intensity_max_cps_1m),
                blocked_fraction=float(args.blocked_fraction),
                passage_width_m=float(args.passage_width_m),
            )
        elif args.hand_authored_cases:
            all_cases = default_cases()
        else:
            all_cases = generated_cases(
                num_cases=int(args.num_cases),
                seed=int(args.case_seed),
                dwell_time_s=float(args.dwell_time_s),
                intensity_min_cps_1m=float(args.intensity_min_cps_1m),
                intensity_max_cps_1m=float(args.intensity_max_cps_1m),
            )
        if args.case:
            selected = set(args.case)
            cases = [case for case in all_cases if case.name in selected]
            missing = selected.difference({case.name for case in cases})
            if missing:
                raise ValueError(f"Unknown case names: {sorted(missing)}")
        else:
            cases = all_cases
        if args.max_cases is not None:
            cases = cases[: max(int(args.max_cases), 0)]
        cases_iter = iter(cases)
        case_total = len(cases)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir
        else ROOT
        / "results"
        / "spectrum_validation"
        / f"geant4_photopeak_nnls_sweep_{timestamp}"
    )

    decomposer = SpectralDecomposer(spectrum_config_from_runtime_config(runtime_config))
    covariance_projector = build_runtime_covariance_projector(runtime_config)
    likelihood_specs = build_runtime_count_likelihood_specs(runtime_config)
    results: list[dict[str, Any]] = []
    spectra: dict[str, np.ndarray] = {}
    cases_seen: list[ValidationCase] = []
    sweep_start = time.time()
    app = Geant4Application(app_config=runtime_config, stage_backend=FakeStageBackend())
    interrupted = False
    try:
        for step_id, case in enumerate(cases_iter):
            cases_seen.append(case)
            print(
                f"[{step_id + 1}/{case_total}] running {case.name}: {case.description}",
                flush=True,
            )
            result, spectrum = run_case(
                app,
                decomposer,
                covariance_projector,
                likelihood_specs,
                case,
                step_id,
                runtime_config,
                mu_by_isotope,
                float(args.min_target_counts),
            )
            results.append(result)
            spectra[case.name] = spectrum
            response = {
                isotope: result["per_isotope"][isotope]["method_counts"][
                    "response_poisson"
                ]
                for isotope in ISOTOPES
            }
            target = {
                isotope: result["per_isotope"][isotope]["target_pf_counts"]
                for isotope in ISOTOPES
            }
            truth = {
                isotope: result["per_isotope"][isotope]["transport_truth_counts"]
                for isotope in ISOTOPES
            }
            rel = {
                isotope: result["per_isotope"][isotope]["relative_errors"][
                    "response_poisson"
                ]
                for isotope in ISOTOPES
            }
            rel_truth = {
                isotope: result["per_isotope"][isotope][
                    "relative_errors_vs_transport_truth"
                ]["response_poisson"]
                for isotope in ISOTOPES
            }
            print(
                f"  primaries={result['num_primaries']:.0f} "
                f"runtime={result['runtime_s']:.1f}s target={target} truth={truth} "
                f"response_poisson={response} rel_err={rel} rel_truth={rel_truth}",
                flush=True,
            )
            checkpoint_every = max(int(args.checkpoint_every), 0)
            if checkpoint_every and len(results) % checkpoint_every == 0:
                partial_summary = build_summary(
                    config_path=config_path,
                    output_dir=output_dir,
                    cases=cases_seen,
                    results=results,
                    args=args,
                    runtime_config=runtime_config,
                    sweep_start=sweep_start,
                    interrupted=False,
                    case_total=case_total,
                )
                write_outputs(
                    output_dir,
                    results,
                    spectra,
                    partial_summary,
                    cases=cases_seen,
                    write_detailed_results=False,
                )
                print(
                    f"  checkpoint wrote partial outputs to: {output_dir}", flush=True
                )
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted; writing partial validation outputs.", flush=True)
    finally:
        app.close()

    summary = build_summary(
        config_path=config_path,
        output_dir=output_dir,
        cases=cases_seen,
        results=results,
        args=args,
        runtime_config=runtime_config,
        sweep_start=sweep_start,
        interrupted=interrupted,
        case_total=case_total,
    )
    write_outputs(
        output_dir,
        results,
        spectra,
        summary,
        cases=cases_seen,
        write_detailed_results=True,
    )
    print(json.dumps(summary["accuracy_summary"], indent=2, sort_keys=True))
    print(f"Wrote validation outputs to: {output_dir}")


if __name__ == "__main__":
    main()
