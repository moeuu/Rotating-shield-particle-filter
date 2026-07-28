"""High-level estimator coordinating parallel PFs and shield rotation (Chapter 3)."""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import math
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from measurement.kernels import MeasurementGeometry, ShieldParams
from measurement.model import EnvironmentConfig
from measurement.continuous_kernels import (
    ContinuousKernel,
    validate_orientation_pair_indices,
)
from measurement.obstacles import ObstacleGrid
from measurement.source_surfaces import (
    SOURCE_SURFACE_REPORT_LABELS,
    source_surface_kinds,
)
from pf.defaults import DEFAULT_MAX_SOURCES_PER_ISOTOPE
from pf.full_spectrum import (
    FullSpectrumGenerativeModel,
    validate_full_spectrum_model,
    validate_observed_spectrum,
)
from pf.particle_filter import (
    IsotopeParticle,
    IsotopeParticleFilter,
    JointRowIdentity,
    StructuralGeometryBatch,
    PFConfig,
    TemperingIncrementRequiresRejuvenation,
)
from pf.posterior import (
    PFPointEstimate,
    posterior_point_estimate_from_states,
    validated_probability_distribution,
    validated_state_cardinality,
)
from pf.posterior_uncertainty import posterior_mode_uncertainty_batched
from pf.provenance import sha256_json
from pf.randomness import (
    named_random_generator,
    normalize_pf_random_seed,
    pf_rng_provenance,
)
from pf.resampling import systematic_resample
from pf.state import IsotopeState
from pf.structural_rj import (
    TRUNCATED_POISSON_CARDINALITY_PRIOR_POLICY,
    validate_cardinality_prior_policy,
)
from pf.transport_response import expected_counts_per_source
if TYPE_CHECKING:
    import torch


JOINT_HISTORY_STATION_ACTION_BATCH_SIZE = 4


def _strict_nonnegative_integer(value: object, *, name: str) -> int:
    """Return one exact nonnegative integer without coercion or truncation."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _strict_config_boolean(value: object, *, name: str) -> bool:
    """Return one exact configuration boolean without truthy coercion."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")
    return value


def _strict_config_number(value: object, *, name: str) -> float:
    """Return one finite numeric configuration value without string coercion."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


@dataclass
class RotatingShieldPFConfig:
    """Configure the exact continuous-surface PF and its active planner."""

    estimator_profile: str = "pf_strict"
    num_particles: int = 200
    max_sources: int | None = DEFAULT_MAX_SOURCES_PER_ISOTOPE
    variable_cardinality: bool = True
    history_estimate_interval: int = 1
    surface_diagnostic_response_cache_max_entries: int = 24
    structural_rj_surface_chart_max_edge_m: float = 1.0
    structural_rj_move_probability: float = 1.0
    structural_rj_birth_probability: float = 0.5
    structural_rj_death_probability: float = 0.5
    structural_rj_position_move_probability: float = 1.0
    structural_rj_position_proposal_prior_weight: float = 0.5
    structural_rj_strength_proposal_prior_weight: float = 0.5
    structural_rj_strength_proposal_sigma_fraction: float = 0.15
    structural_rj_strength_proposal_grid_size: int = 9
    structural_rj_proposal_chart_batch_size: int = 256
    structural_rj_proposal_score_cache_max_bytes: int = 268_435_456
    structural_rj_local_position_move_probability: float = 1.0
    structural_rj_local_position_sigma_m: float = 0.5
    structural_rj_strength_move_probability: float = 1.0
    structural_rj_split_merge_probability: float = 1.0
    structural_rj_split_probability: float = 0.5
    structural_rj_merge_probability: float = 0.5
    structural_cardinality_prior_policy: str = (
        TRUNCATED_POISSON_CARDINALITY_PRIOR_POLICY
    )
    structural_cardinality_prior_probs: tuple[float, ...] | list[float] | None = None
    structural_cardinality_prior_mean: float = 2.0
    max_dwell_time_s: float = 5.0  # Max dwell time per pose.
    credible_surface_radius_threshold_m: float = 0.5
    converge_min_ess_ratio: float = 0.5
    converge_cardinality_min_probability: float = 0.95
    converge_max_cardinality_boundary_mass: float = 0.05
    converge_innovation_confidence: float = 0.99
    target_ess_ratio: float = 0.5
    max_temper_steps: int = 256
    min_delta_beta: float = 1e-10
    position_max: Tuple[float, float, float] = (10.0, 10.0, 10.0)
    init_num_sources: Tuple[int, int] = (
        0,
        DEFAULT_MAX_SOURCES_PER_ISOTOPE,
    )
    strength_prior_min_cps_1m: float = 1.0
    strength_prior_max_cps_1m: float = 2_000_000.0
    orientation_k: int = 8
    min_rotations_per_pose: int = 0
    planning_particles: int | None = None
    planning_method: str = "top_weight"
    use_gpu: bool = True
    gpu_device: str = "cuda"
    gpu_dtype: str = "float64"
    planning_eig_samples: int = 50
    converge_cardinality_var_max: float = 0.05

    def __post_init__(self) -> None:
        """Validate and normalize estimator configuration values."""
        integer_fields = (
            ("num_particles", self.num_particles, 1),
            ("history_estimate_interval", self.history_estimate_interval, 0),
            (
                "surface_diagnostic_response_cache_max_entries",
                self.surface_diagnostic_response_cache_max_entries,
                0,
            ),
            (
                "structural_rj_strength_proposal_grid_size",
                self.structural_rj_strength_proposal_grid_size,
                2,
            ),
            (
                "structural_rj_proposal_chart_batch_size",
                self.structural_rj_proposal_chart_batch_size,
                1,
            ),
            (
                "structural_rj_proposal_score_cache_max_bytes",
                self.structural_rj_proposal_score_cache_max_bytes,
                1,
            ),
            ("orientation_k", self.orientation_k, 1),
            ("min_rotations_per_pose", self.min_rotations_per_pose, 0),
            ("planning_eig_samples", self.planning_eig_samples, 1),
            ("max_temper_steps", self.max_temper_steps, 1),
        )
        for name, value, minimum in integer_fields:
            resolved = _strict_nonnegative_integer(value, name=name)
            if resolved < minimum:
                raise ValueError(f"{name} must be at least {minimum}.")
        if self.max_sources is None:
            raise ValueError("Pure PF requires a finite positive max_sources.")
        if _strict_nonnegative_integer(
            self.max_sources,
            name="max_sources",
        ) < 1:
            raise ValueError("Pure PF requires a finite positive max_sources.")
        _strict_config_boolean(
            self.variable_cardinality,
            name="variable_cardinality",
        )
        _strict_config_boolean(self.use_gpu, name="use_gpu")
        if (
            not isinstance(self.init_num_sources, (tuple, list))
            or len(self.init_num_sources) != 2
        ):
            raise TypeError("init_num_sources must contain two integers.")
        for index, value in enumerate(self.init_num_sources):
            _strict_nonnegative_integer(
                value,
                name=f"init_num_sources[{index}]",
            )
        numeric_fields = (
            "strength_prior_min_cps_1m",
            "strength_prior_max_cps_1m",
            "structural_rj_surface_chart_max_edge_m",
            "structural_rj_local_position_sigma_m",
            "structural_rj_position_proposal_prior_weight",
            "structural_rj_strength_proposal_prior_weight",
            "structural_rj_strength_proposal_sigma_fraction",
            "structural_rj_move_probability",
            "structural_rj_birth_probability",
            "structural_rj_death_probability",
            "structural_rj_position_move_probability",
            "structural_rj_local_position_move_probability",
            "structural_rj_strength_move_probability",
            "structural_rj_split_merge_probability",
            "structural_rj_split_probability",
            "structural_rj_merge_probability",
            "structural_cardinality_prior_mean",
            "max_dwell_time_s",
            "credible_surface_radius_threshold_m",
            "converge_min_ess_ratio",
            "converge_cardinality_min_probability",
            "converge_max_cardinality_boundary_mass",
            "converge_innovation_confidence",
            "target_ess_ratio",
            "min_delta_beta",
            "converge_cardinality_var_max",
        )
        for name in numeric_fields:
            _strict_config_number(getattr(self, name), name=name)
        if self.structural_cardinality_prior_probs is not None:
            for index, value in enumerate(
                self.structural_cardinality_prior_probs
            ):
                _strict_config_number(
                    value,
                    name=f"structural_cardinality_prior_probs[{index}]",
                )
        if not isinstance(self.gpu_device, str) or not self.gpu_device.strip():
            raise TypeError("gpu_device must be a nonempty string.")
        if not isinstance(self.gpu_dtype, str):
            raise TypeError("gpu_dtype must be a string.")
        if self.planning_particles is not None:
            if _strict_nonnegative_integer(
                self.planning_particles,
                name="planning_particles",
            ) < 2:
                raise ValueError("planning_particles must be at least two.")
        if self.planning_method not in {"resample", "top_weight"}:
            raise ValueError(
                "planning_method must be 'resample' or 'top_weight'."
            )
        if self.orientation_k > 64:
            raise ValueError("orientation_k cannot exceed 64.")
        if self.min_rotations_per_pose > self.orientation_k:
            raise ValueError(
                "min_rotations_per_pose cannot exceed orientation_k."
            )
        if _strict_config_number(
            self.max_dwell_time_s,
            name="max_dwell_time_s",
        ) <= 0.0:
            raise ValueError("max_dwell_time_s must be positive.")
        target_ess_ratio = _strict_config_number(
            self.target_ess_ratio,
            name="target_ess_ratio",
        )
        if not 0.0 < target_ess_ratio < 1.0:
            raise ValueError(
                "target_ess_ratio must lie strictly between zero and one."
            )
        min_delta_beta = _strict_config_number(
            self.min_delta_beta,
            name="min_delta_beta",
        )
        if not 0.0 < min_delta_beta <= 1.0:
            raise ValueError("min_delta_beta must lie in (0, 1].")
        innovation_confidence = _strict_config_number(
            self.converge_innovation_confidence,
            name="converge_innovation_confidence",
        )
        if not 0.0 < innovation_confidence < 1.0:
            raise ValueError(
                "converge_innovation_confidence must lie in (0, 1)."
            )
        self.num_particles = int(self.num_particles)
        if self.num_particles < 1:
            raise ValueError("num_particles must be positive.")
        if str(self.gpu_dtype).strip().lower() != "float64":
            raise ValueError(
                "Pure PF production kernels require gpu_dtype='float64'."
            )
        self.gpu_dtype = "float64"
        self.strength_prior_min_cps_1m = float(self.strength_prior_min_cps_1m)
        self.strength_prior_max_cps_1m = float(self.strength_prior_max_cps_1m)
        if (
            not np.isfinite(self.strength_prior_min_cps_1m)
            or self.strength_prior_min_cps_1m <= 0.0
        ):
            raise ValueError(
                "strength_prior_min_cps_1m must be finite and positive."
            )
        if (
            not np.isfinite(self.strength_prior_max_cps_1m)
            or self.strength_prior_max_cps_1m
            <= self.strength_prior_min_cps_1m
        ):
            raise ValueError(
                "strength_prior_max_cps_1m must be finite and greater than "
                "strength_prior_min_cps_1m."
            )
        self.structural_rj_surface_chart_max_edge_m = float(
            self.structural_rj_surface_chart_max_edge_m
        )
        if (
            not np.isfinite(self.structural_rj_surface_chart_max_edge_m)
            or self.structural_rj_surface_chart_max_edge_m <= 0.0
        ):
            raise ValueError(
                "structural_rj_surface_chart_max_edge_m must be positive."
            )
        self.structural_rj_local_position_sigma_m = float(
            self.structural_rj_local_position_sigma_m
        )
        if (
            not np.isfinite(self.structural_rj_local_position_sigma_m)
            or self.structural_rj_local_position_sigma_m <= 0.0
        ):
            raise ValueError(
                "structural_rj_local_position_sigma_m must be positive."
            )
        self.structural_rj_position_proposal_prior_weight = float(
            self.structural_rj_position_proposal_prior_weight
        )
        if (
            not np.isfinite(
                self.structural_rj_position_proposal_prior_weight
            )
            or self.structural_rj_position_proposal_prior_weight <= 0.0
            or self.structural_rj_position_proposal_prior_weight > 1.0
        ):
            raise ValueError(
                "structural_rj_position_proposal_prior_weight must lie in "
                "(0, 1]."
            )
        self.structural_rj_strength_proposal_prior_weight = float(
            self.structural_rj_strength_proposal_prior_weight
        )
        if (
            not np.isfinite(
                self.structural_rj_strength_proposal_prior_weight
            )
            or self.structural_rj_strength_proposal_prior_weight <= 0.0
            or self.structural_rj_strength_proposal_prior_weight > 1.0
        ):
            raise ValueError(
                "structural_rj_strength_proposal_prior_weight must lie in "
                "(0, 1]."
            )
        self.structural_rj_strength_proposal_sigma_fraction = float(
            self.structural_rj_strength_proposal_sigma_fraction
        )
        if (
            not np.isfinite(
                self.structural_rj_strength_proposal_sigma_fraction
            )
            or self.structural_rj_strength_proposal_sigma_fraction <= 0.0
        ):
            raise ValueError(
                "structural_rj_strength_proposal_sigma_fraction must be "
                "finite and positive."
            )
        self.structural_rj_strength_proposal_grid_size = int(
            self.structural_rj_strength_proposal_grid_size
        )
        if self.structural_rj_strength_proposal_grid_size < 2:
            raise ValueError(
                "structural_rj_strength_proposal_grid_size must be at least 2."
            )
        self.structural_rj_proposal_chart_batch_size = int(
            self.structural_rj_proposal_chart_batch_size
        )
        if self.structural_rj_proposal_chart_batch_size < 1:
            raise ValueError(
                "structural_rj_proposal_chart_batch_size must be positive."
            )
        self.structural_rj_proposal_score_cache_max_bytes = int(
            self.structural_rj_proposal_score_cache_max_bytes
        )
        if self.structural_rj_proposal_score_cache_max_bytes < 1:
            raise ValueError(
                "structural_rj_proposal_score_cache_max_bytes must be positive."
            )
        for probability_field in (
            "structural_rj_move_probability",
            "structural_rj_birth_probability",
            "structural_rj_death_probability",
            "structural_rj_position_move_probability",
            "structural_rj_local_position_move_probability",
            "structural_rj_strength_move_probability",
            "structural_rj_split_merge_probability",
            "structural_rj_split_probability",
            "structural_rj_merge_probability",
        ):
            probability = float(getattr(self, probability_field))
            if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"{probability_field} must be in [0, 1].")
            setattr(self, probability_field, probability)
        self.structural_cardinality_prior_policy = (
            validate_cardinality_prior_policy(
                self.structural_cardinality_prior_policy,
                has_explicit_probabilities=(
                    self.structural_cardinality_prior_probs is not None
                ),
            )
        )
        if self.structural_cardinality_prior_probs is not None:
            if not isinstance(
                self.structural_cardinality_prior_probs,
                (tuple, list),
            ):
                raise ValueError(
                    "structural_cardinality_prior_probs must be a tuple, list, or None."
                )
            cardinality_prior = np.asarray(
                [float(value) for value in self.structural_cardinality_prior_probs],
                dtype=float,
            )
            if (
                cardinality_prior.size == 0
                or np.any(~np.isfinite(cardinality_prior))
                or np.any(cardinality_prior <= 0.0)
            ):
                raise ValueError(
                    "structural_cardinality_prior_probs must contain only "
                    "positive finite values."
                )
            cardinality_prior /= math.fsum(
                float(value) for value in cardinality_prior
            )
            self.structural_cardinality_prior_probs = tuple(
                float(value) for value in cardinality_prior
            )
        self.structural_cardinality_prior_mean = float(
            self.structural_cardinality_prior_mean
        )
        if (
            not np.isfinite(self.structural_cardinality_prior_mean)
            or self.structural_cardinality_prior_mean <= 0.0
        ):
            raise ValueError(
                "structural_cardinality_prior_mean must be finite and positive."
            )
        self.history_estimate_interval = max(0, int(self.history_estimate_interval))
        self.surface_diagnostic_response_cache_max_entries = max(
            0,
            int(self.surface_diagnostic_response_cache_max_entries),
        )
        self.credible_surface_radius_threshold_m = float(
            self.credible_surface_radius_threshold_m
        )
        if (
            not np.isfinite(self.credible_surface_radius_threshold_m)
            or self.credible_surface_radius_threshold_m < 0.0
        ):
            raise ValueError(
                "credible_surface_radius_threshold_m must be finite and "
                "nonnegative."
            )
        for probability_field, lower_inclusive in (
            ("converge_min_ess_ratio", False),
            ("converge_cardinality_min_probability", False),
            ("converge_max_cardinality_boundary_mass", True),
            ("converge_innovation_confidence", False),
        ):
            probability = float(getattr(self, probability_field))
            lower_valid = (
                probability >= 0.0 if lower_inclusive else probability > 0.0
            )
            if (
                not np.isfinite(probability)
                or not lower_valid
                or probability > 1.0
            ):
                lower_symbol = "[" if lower_inclusive else "("
                raise ValueError(
                    f"{probability_field} must be in {lower_symbol}0, 1]."
                )
            setattr(self, probability_field, probability)
        self.converge_cardinality_var_max = float(
            self.converge_cardinality_var_max
        )
        if (
            not np.isfinite(self.converge_cardinality_var_max)
            or self.converge_cardinality_var_max < 0.0
        ):
            raise ValueError(
                "converge_cardinality_var_max must be finite and nonnegative."
            )
        self.variable_cardinality = bool(self.variable_cardinality)
        if self.max_sources is None or int(self.max_sources) < 1:
            raise ValueError("Pure PF requires a finite positive max_sources.")
        self.max_sources = int(self.max_sources)
        if (
            self.structural_cardinality_prior_probs is not None
            and len(self.structural_cardinality_prior_probs) != self.max_sources + 1
        ):
            raise ValueError(
                "structural_cardinality_prior_probs must contain "
                "max_sources + 1 entries."
            )
        initial_lower, initial_upper = (
            int(self.init_num_sources[0]),
            int(self.init_num_sources[1]),
        )
        if self.variable_cardinality:
            if (
                self.structural_rj_move_probability <= 0.0
                or self.structural_rj_birth_probability <= 0.0
                or self.structural_rj_death_probability <= 0.0
            ):
                raise ValueError(
                    "Variable-cardinality pure PF requires positive structural "
                    "move, birth, and death proposal probabilities."
                )
            if initial_lower != 0 or initial_upper != self.max_sources:
                raise ValueError(
                    "Variable-cardinality pure PF initialization must cover "
                    "every cardinality from zero through max_sources."
                )
            if self.num_particles < self.max_sources + 1:
                raise ValueError(
                    "Variable-cardinality pure PF requires at least one initial "
                    "particle per cardinality from zero through max_sources."
                )
        elif (
            initial_lower != initial_upper
            or initial_lower < 0
            or initial_upper > self.max_sources
        ):
            raise ValueError(
                "Fixed-cardinality pure PF requires "
                "init_num_sources=(K, K) within zero through max_sources."
            )
        self.init_num_sources = (initial_lower, initial_upper)


@dataclass(frozen=True)
class MeasurementRecord:
    """Store one full-spectrum shield-view measurement and provenance."""

    spectrum_counts_b: NDArray[np.float64]
    pose_idx: int
    live_time_s: float
    fe_index: int
    pb_index: int
    detector_position_xyz_m: tuple[float, float, float]
    station_sequence_id: int
    station_view_index: int
    generative_contract_hash_sha256: str


@dataclass(frozen=True)
class JointStationObservation:
    """Store one joint view-major full-spectrum station observation."""

    spectrum_vb: NDArray[np.float64]
    energy_axis_keV: NDArray[np.float64]
    generative_contract_hash_sha256: str
    pose_idx: int
    detector_position_xyz_m: tuple[float, float, float]
    fe_indices: NDArray[np.int64]
    pb_indices: NDArray[np.int64]
    live_times_s: NDArray[np.float64]
    station_sequence_id: int


@dataclass(frozen=True)
class JointPlanningParticles:
    """Expose one aligned joint-particle subset as padded numeric arrays."""

    isotope_order: tuple[str, ...]
    weights_n: NDArray[np.float64]
    positions_nk3_by_isotope: Dict[str, NDArray[np.float64]]
    surface_chart_ids_nk_by_isotope: Dict[str, NDArray[np.int64]]
    surface_uv_nk2_by_isotope: Dict[str, NDArray[np.float64]]
    strengths_nk_by_isotope: Dict[str, NDArray[np.float64]]
    source_mask_nk_by_isotope: Dict[str, NDArray[np.bool_]]
    original_particle_indices: NDArray[np.int64]


@dataclass(frozen=True)
class SurfaceAtlasQuadrature:
    """Represent a complete area-weighted chart-center surface quadrature."""

    positions_s3: NDArray[np.float64]
    area_weights_m2_s: NDArray[np.float64]
    chart_ids_s: NDArray[np.int64]
    chart_count: int
    total_area_m2: float
    maximum_hausdorff_bound_m: float
    kinds: tuple[str, ...]
    face_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate complete one-sample-per-chart quadrature semantics."""
        positions = np.asarray(self.positions_s3, dtype=np.float64)
        weights = np.asarray(self.area_weights_m2_s, dtype=np.float64).reshape(-1)
        chart_ids = np.asarray(self.chart_ids_s, dtype=np.int64).reshape(-1)
        count = int(self.chart_count)
        total_area = float(self.total_area_m2)
        hausdorff = float(self.maximum_hausdorff_bound_m)
        if (
            count <= 0
            or positions.shape != (count, 3)
            or weights.shape != (count,)
            or chart_ids.shape != (count,)
            or len(self.kinds) != count
            or len(self.face_ids) != count
            or np.any(~np.isfinite(positions))
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
            or not np.array_equal(
                chart_ids,
                np.arange(count, dtype=np.int64),
            )
            or not np.isfinite(total_area)
            or total_area <= 0.0
            or not np.isclose(
                float(np.sum(weights, dtype=np.float64)),
                total_area,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
            or not np.isfinite(hausdorff)
            or hausdorff < 0.0
        ):
            raise ValueError(
                "Surface quadrature must contain every chart exactly once "
                "with finite positive physical area."
            )
        object.__setattr__(
            self,
            "positions_s3",
            np.ascontiguousarray(positions),
        )
        object.__setattr__(
            self,
            "area_weights_m2_s",
            np.ascontiguousarray(weights),
        )
        object.__setattr__(
            self,
            "chart_ids_s",
            np.ascontiguousarray(chart_ids),
        )

    def diagnostics(self) -> dict[str, object]:
        """Return JSON-safe completeness and spacing provenance."""
        unique_kinds, kind_counts = np.unique(
            np.asarray(self.kinds, dtype=object),
            return_counts=True,
        )
        return {
            "contract": "complete_chart_center_area_quadrature_v1",
            "sample_count": int(self.chart_count),
            "chart_count": int(self.chart_count),
            "total_area_m2": float(self.total_area_m2),
            "maximum_hausdorff_bound_m": float(
                self.maximum_hausdorff_bound_m
            ),
            "physical_face_count": int(len(set(self.face_ids))),
            "surface_kind_chart_counts": {
                str(kind): int(count)
                for kind, count in zip(
                    unique_kinds,
                    kind_counts,
                    strict=True,
                )
            },
            "every_chart_represented": True,
            "area_weighted": True,
        }


def build_complete_surface_atlas_quadrature(
    atlas: object,
    *,
    max_points: int,
    maximum_hausdorff_bound_m: float,
) -> SurfaceAtlasQuadrature:
    """Build a fail-closed one-center-per-chart physical-area quadrature."""
    budget = int(max_points)
    requested_bound = float(maximum_hausdorff_bound_m)
    if budget <= 0:
        raise ValueError("Surface quadrature max_points must be positive.")
    if (
        not np.isfinite(requested_bound)
        or requested_bound <= 0.0
    ):
        raise ValueError(
            "Surface quadrature Hausdorff bound must be finite and positive."
        )
    chart_count = int(getattr(atlas, "chart_count"))
    if chart_count > budget:
        raise RuntimeError(
            "Surface coverage quadrature budget cannot represent every "
            f"chart ({chart_count} > {budget}). Increase the predeclared "
            "coverage_surface_quadrature_max_points budget."
        )
    geometry = getattr(atlas, "geometry")
    vertices = np.asarray(geometry.vertices_xyz, dtype=np.float64)
    centers = np.asarray(geometry.centers_xyz, dtype=np.float64)
    if (
        vertices.shape != (chart_count, 4, 3)
        or centers.shape != (chart_count, 3)
        or np.any(~np.isfinite(vertices))
        or np.any(~np.isfinite(centers))
    ):
        raise RuntimeError(
            "Surface atlas quadrature requires finite quadrilateral charts."
        )
    chart_center_vertex_radius = np.max(
        np.linalg.norm(
            vertices - centers[:, None, :],
            axis=2,
        ),
        axis=1,
    )
    maximum_bound = float(np.max(chart_center_vertex_radius))
    if maximum_bound > requested_bound + 1.0e-12:
        raise RuntimeError(
            "Surface chart-center quadrature exceeds the predeclared "
            "Hausdorff bound "
            f"({maximum_bound:.6g} m > {requested_bound:.6g} m). "
            "Refine the continuous surface atlas before planning."
        )
    chart_ids = np.arange(chart_count, dtype=np.int64)
    uv = np.full((chart_count, 2), 0.5, dtype=np.float64)
    positions = np.asarray(
        atlas.positions_xyz(chart_ids, uv),
        dtype=np.float64,
    )
    if not np.allclose(
        positions,
        centers,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "Surface atlas center coordinates and chart mapping differ."
        )
    return SurfaceAtlasQuadrature(
        positions_s3=positions,
        area_weights_m2_s=np.asarray(
            geometry.areas_m2,
            dtype=np.float64,
        ).copy(),
        chart_ids_s=chart_ids,
        chart_count=chart_count,
        total_area_m2=float(getattr(atlas, "total_area_m2")),
        maximum_hausdorff_bound_m=maximum_bound,
        kinds=tuple(str(value) for value in geometry.kinds),
        face_ids=tuple(str(value) for value in geometry.face_ids),
    )


class RotatingShieldPFEstimator:
    """
    Online exact-RJ particle filter using rotating shield spectra.

    Isotope state blocks share one outer particle weight and one resampling
    ancestry. Each station is assimilated once through the immutable joint
    full-spectrum generative likelihood.
    """

    def __init__(
        self,
        isotopes: Sequence[str],
        surface_diagnostic_points: NDArray[np.float64],
        shield_normals: NDArray[np.float64] | None,
        mu_by_isotope: Dict[str, object] | None,
        pf_config: RotatingShieldPFConfig | None = None,
        shield_params: ShieldParams | None = None,
        obstacle_grid: ObstacleGrid | None = None,
        obstacle_height_m: float = 2.0,
        obstacle_mu_by_isotope: Dict[str, float] | None = None,
        obstacle_buildup_coeff: float = 0.0,
        detector_radius_m: float = 0.0,
        detector_aperture_radius_m: float | None = None,
        detector_aperture_samples: int = 1,
        detector_aperture_sampling: str = "solid_angle_cone",
        source_extent_radius_m: float = 0.0,
        source_extent_samples: int = 1,
        line_mu_by_isotope: Dict[str, object] | None = None,
        full_spectrum_generative_model: object | None = None,
        random_seed: int = 0,
    ) -> None:
        """Initialize per-isotope filters and shared measurement-model state."""
        configured_isotopes = tuple(isotopes)
        if (
            not configured_isotopes
            or any(
                not isinstance(isotope, str) or not isotope.strip()
                for isotope in configured_isotopes
            )
            or len(set(configured_isotopes)) != len(configured_isotopes)
        ):
            raise ValueError(
                "Joint PF isotopes must be unique nonempty strings."
            )
        self.isotopes = list(configured_isotopes)
        self.random_seed = normalize_pf_random_seed(random_seed)
        self.rng_provenance = pf_rng_provenance(
            self.random_seed,
            self.isotopes,
        )
        self.pf_config = pf_config or RotatingShieldPFConfig()
        self.shield_params = shield_params or ShieldParams()
        if not isinstance(self.shield_params, ShieldParams):
            raise TypeError("shield_params must be a ShieldParams instance.")
        self.obstacle_grid = obstacle_grid
        self.obstacle_height_m = _strict_config_number(
            obstacle_height_m,
            name="obstacle_height_m",
        )
        if self.obstacle_height_m < 0.0:
            raise ValueError("obstacle_height_m must be nonnegative.")
        self.obstacle_mu_by_isotope = obstacle_mu_by_isotope
        self.obstacle_buildup_coeff = _strict_config_number(
            obstacle_buildup_coeff,
            name="obstacle_buildup_coeff",
        )
        if self.obstacle_buildup_coeff < 0.0:
            raise ValueError("obstacle_buildup_coeff must be nonnegative.")
        self.detector_radius_m = _strict_config_number(
            detector_radius_m,
            name="detector_radius_m",
        )
        if self.detector_radius_m < 0.0:
            raise ValueError("detector_radius_m must be nonnegative.")
        if detector_aperture_radius_m is None:
            detector_aperture_radius_m = self.detector_radius_m
        self.detector_aperture_radius_m = _strict_config_number(
            detector_aperture_radius_m,
            name="detector_aperture_radius_m",
        )
        if self.detector_aperture_radius_m < 0.0:
            raise ValueError(
                "detector_aperture_radius_m must be nonnegative."
            )
        self.detector_aperture_samples = _strict_nonnegative_integer(
            detector_aperture_samples,
            name="detector_aperture_samples",
        )
        if self.detector_aperture_samples == 0:
            raise ValueError("detector_aperture_samples must be positive.")
        if not isinstance(detector_aperture_sampling, str):
            raise TypeError("detector_aperture_sampling must be a string.")
        self.detector_aperture_sampling = detector_aperture_sampling
        self.source_extent_radius_m = _strict_config_number(
            source_extent_radius_m,
            name="source_extent_radius_m",
        )
        if self.source_extent_radius_m < 0.0:
            raise ValueError("source_extent_radius_m must be nonnegative.")
        self.source_extent_samples = _strict_nonnegative_integer(
            source_extent_samples,
            name="source_extent_samples",
        )
        if self.source_extent_samples == 0:
            raise ValueError("source_extent_samples must be positive.")
        if (self.source_extent_radius_m == 0.0) != (
            self.source_extent_samples == 1
        ):
            raise ValueError(
                "Source extent requires radius=0 with one sample, or a "
                "positive radius with at least two samples."
            )
        self.line_mu_by_isotope = line_mu_by_isotope
        self.full_spectrum_generative_model = validate_full_spectrum_model(
            full_spectrum_generative_model
        )
        self.additive_scatter_response = getattr(
            self.full_spectrum_generative_model,
            "additive_scatter_response",
            None,
        )
        # Measurement poses are appended incrementally.
        self.poses: List[NDArray[np.float64]] = []
        if shield_normals is None:
            from measurement.shielding import generate_octant_orientations

            self.normals = generate_octant_orientations()
        else:
            self.normals = shield_normals
        self.mu_by_isotope = self._resolve_mu_by_isotope(mu_by_isotope)
        self.kernel_cache: MeasurementGeometry | None = None
        self.filters: Dict[str, IsotopeParticleFilter] = {}
        self._joint_particles_initialized = False
        self._joint_row_identity_root_sha256: str | None = None
        self._joint_row_generation: int | None = None
        diagnostic_points = np.asarray(
            surface_diagnostic_points,
            dtype=np.float64,
        )
        if diagnostic_points.ndim != 2 or diagnostic_points.shape[1] != 3:
            raise ValueError(
                "surface_diagnostic_points must be shaped (N, 3)."
            )
        if not np.all(np.isfinite(diagnostic_points)):
            raise ValueError(
                "surface_diagnostic_points must contain only finite values."
            )
        self.surface_diagnostic_points = np.ascontiguousarray(diagnostic_points)
        self.history_estimates: List[
            Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]
        ] = []
        self.measurements: List[MeasurementRecord] = []
        self._joint_station_history: list[JointStationObservation] = []
        self._active_joint_station_history: (
            tuple[JointStationObservation, ...] | None
        ) = None
        self._active_joint_structural_geometry: (
            StructuralGeometryBatch | None
        ) = None
        self._joint_structural_transport_cache: (
            tuple[
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.float64],
            ]
            | None
        ) = None
        self._joint_birth_proposal_station_score_cache: dict[
            tuple[str, str],
            NDArray[np.float64],
        ] = {}
        self._joint_birth_proposal_station_score_cache_order: list[
            tuple[str, str]
        ] = []
        self._joint_birth_proposal_prefix_scores: dict[
            str,
            NDArray[np.float64],
        ] = {}
        self._joint_birth_proposal_prefix_station_count = 0
        self.last_joint_birth_proposal_cache_hits = 0
        self.last_joint_birth_proposal_cache_misses = 0
        self._joint_random_generator = named_random_generator(
            self.random_seed,
            "joint_isotope_particle_filter",
        )
        self.last_joint_resample_indices = np.zeros(0, dtype=np.int64)
        self.last_joint_temper_steps: list[dict[str, float]] = []
        self.last_joint_unique_ancestor_count: int | None = None
        self.last_joint_station_unique_ancestor_count: int | None = None
        self.last_joint_cumulative_unique_ancestor_count: int | None = None
        self._joint_cumulative_lineage_ids: NDArray[np.int64] | None = None
        self.last_pair_sequence_update_workers = 1
        self.last_pair_sequence_update_wall_s = 0.0
        self.last_pair_sequence_stage_wall_s: Dict[str, float] = {}
        self.last_structural_update_workers = 1
        self.last_structural_update_wall_s = 0.0
        self._surface_diagnostic_response_cache: dict[
            tuple[Any, ...],
            NDArray[np.float64],
        ] = {}
        self._surface_diagnostic_response_cache_order: list[tuple[Any, ...]] = []
        self._surface_diagnostic_response_prefix_cache: dict[
            tuple[Any, ...],
            dict[str, NDArray[np.float64]],
        ] = {}
        self._surface_diagnostic_response_prefix_cache_order: list[tuple[Any, ...]] = []

    def _named_planning_rng(
        self,
        operation: str,
        *components: object,
    ) -> np.random.Generator:
        """Return an order-independent planning stream from the logged PF seed."""
        return named_random_generator(
            self.random_seed,
            "estimator_planning",
            str(operation),
            len(self.measurements),
            *components,
        )

    def _record_history_estimate(self, measurement_count: int) -> None:
        """Record an exact report estimate when the configured history stride allows it."""
        interval = max(
            0,
            int(getattr(self.pf_config, "history_estimate_interval", 1)),
        )
        if interval <= 0:
            return
        count = max(0, int(measurement_count))
        if count <= 0 or count % interval != 0:
            return
        self.history_estimates.append(self.estimates())

    def _surface_diagnostic_response_source_key(
        self,
        sources: NDArray[np.float64],
    ) -> tuple[str, int] | None:
        """Return a stable cache key for the full shared surface-diagnostic atlas sample."""
        source_arr = np.asarray(sources, dtype=float).reshape(-1, 3)
        candidate_arr = np.asarray(self.surface_diagnostic_points, dtype=float).reshape(
            -1,
            3,
        )
        if (
            source_arr.shape == candidate_arr.shape
            and source_arr.size > 0
            and np.shares_memory(source_arr, candidate_arr)
        ):
            return ("surface_diagnostic_points", int(source_arr.shape[0]))
        return None

    @staticmethod
    def _measurement_geometry_digest(data: StructuralGeometryBatch) -> bytes:
        """Return a compact digest for measurement geometry arrays used by responses."""
        digest = hashlib.blake2b(digest_size=16)
        for array in (
            np.asarray(data.detector_positions, dtype=np.float64),
            np.asarray(data.live_times, dtype=np.float64),
            np.asarray(data.fe_indices, dtype=np.int64),
            np.asarray(data.pb_indices, dtype=np.int64),
        ):
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.shape).encode("ascii"))
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(contiguous.tobytes())
        return digest.digest()

    def _store_surface_diagnostic_response_cache(
        self,
        cache_key: tuple[Any, ...],
        counts: NDArray[np.float64],
    ) -> None:
        """Store an exact deterministic surface-diagnostic response with LRU eviction."""
        self._surface_diagnostic_response_cache[cache_key] = np.asarray(
            counts,
            dtype=float,
        ).copy()
        self._surface_diagnostic_response_cache_order.append(cache_key)
        max_entries = max(
            0,
            int(self.pf_config.surface_diagnostic_response_cache_max_entries),
        )
        while len(self._surface_diagnostic_response_cache_order) > max_entries:
            old_key = self._surface_diagnostic_response_cache_order.pop(0)
            if old_key not in self._surface_diagnostic_response_cache_order:
                self._surface_diagnostic_response_cache.pop(old_key, None)

    @staticmethod
    def _surface_diagnostic_response_prefix_matches(
        payload: Mapping[str, NDArray[np.float64]],
        data: StructuralGeometryBatch,
        row_count: int,
    ) -> bool:
        """Return True when cached response rows match the requested prefix."""
        rows = max(0, int(row_count))
        if rows < 0:
            return False
        try:
            return bool(
                np.allclose(
                    np.asarray(payload["detector_positions"], dtype=float)[:rows],
                    np.asarray(data.detector_positions, dtype=float)[:rows],
                    rtol=0.0,
                    atol=1.0e-12,
                )
                and np.allclose(
                    np.asarray(payload["live_times"], dtype=float)[:rows],
                    np.asarray(data.live_times, dtype=float)[:rows],
                    rtol=0.0,
                    atol=1.0e-12,
                )
                and np.array_equal(
                    np.asarray(payload["fe_indices"], dtype=int)[:rows],
                    np.asarray(data.fe_indices, dtype=int)[:rows],
                )
                and np.array_equal(
                    np.asarray(payload["pb_indices"], dtype=int)[:rows],
                    np.asarray(data.pb_indices, dtype=int)[:rows],
                )
            )
        except (KeyError, ValueError, TypeError):
            return False

    def _store_surface_diagnostic_response_prefix_cache(
        self,
        prefix_key: tuple[Any, ...],
        data: StructuralGeometryBatch,
        counts: NDArray[np.float64],
    ) -> None:
        """Store a full-history response prefix for incremental extension."""
        self._surface_diagnostic_response_prefix_cache[prefix_key] = {
            "detector_positions": np.asarray(
                data.detector_positions, dtype=float
            ).copy(),
            "live_times": np.asarray(data.live_times, dtype=float).copy(),
            "fe_indices": np.asarray(data.fe_indices, dtype=int).copy(),
            "pb_indices": np.asarray(data.pb_indices, dtype=int).copy(),
            "counts": np.asarray(counts, dtype=float).copy(),
        }
        self._surface_diagnostic_response_prefix_cache_order.append(prefix_key)
        max_entries = max(
            0,
            int(self.pf_config.surface_diagnostic_response_cache_max_entries),
        )
        while len(self._surface_diagnostic_response_prefix_cache_order) > max_entries:
            old_key = self._surface_diagnostic_response_prefix_cache_order.pop(0)
            if old_key not in self._surface_diagnostic_response_prefix_cache_order:
                self._surface_diagnostic_response_prefix_cache.pop(old_key, None)

    def _cached_expected_counts_for_kernel(
        self,
        *,
        kernel: ContinuousKernel,
        isotope: str,
        data: StructuralGeometryBatch,
        sources: NDArray[np.float64],
        strengths: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return batched expected counts from a particle-independent kernel."""
        source_arr = np.asarray(sources, dtype=float).reshape(-1, 3)
        strength_arr = np.asarray(strengths, dtype=float).reshape(-1)
        source_key = self._surface_diagnostic_response_source_key(source_arr)
        measurement_count = int(np.asarray(data.live_times, dtype=float).size)
        cache_enabled = (
            source_key is not None
            and strength_arr.size == source_arr.shape[0]
            and np.allclose(strength_arr, 1.0)
            and int(self.pf_config.surface_diagnostic_response_cache_max_entries) > 0
        )
        cache_key: tuple[Any, ...] | None = None
        prefix_key: tuple[Any, ...] | None = None
        if cache_enabled:
            cache_key = (
                str(isotope),
                int(id(kernel)),
                source_key,
                self._measurement_geometry_digest(data),
            )
            cached = self._surface_diagnostic_response_cache.get(cache_key)
            if cached is not None:
                return cached.copy()
            prefix_key = (str(isotope), int(id(kernel)), source_key)
            prefix_payload = self._surface_diagnostic_response_prefix_cache.get(prefix_key)
            if isinstance(prefix_payload, dict):
                cached_counts = np.asarray(
                    prefix_payload.get("counts", np.zeros((0, 0))),
                    dtype=float,
                )
                cached_rows = int(cached_counts.shape[0])
                if (
                    cached_rows >= measurement_count
                    and self._surface_diagnostic_response_prefix_matches(
                        prefix_payload,
                        data,
                        measurement_count,
                    )
                ):
                    counts_arr = cached_counts[:measurement_count].copy()
                    self._store_surface_diagnostic_response_cache(cache_key, counts_arr)
                    return counts_arr
                if (
                    cached_rows < measurement_count
                    and self._surface_diagnostic_response_prefix_matches(
                        prefix_payload,
                        data,
                        cached_rows,
                    )
                ):
                    suffix_counts = expected_counts_per_source(
                        kernel=kernel,
                        isotope=isotope,
                        detector_positions=np.asarray(
                            data.detector_positions,
                            dtype=float,
                        )[cached_rows:],
                        sources=source_arr,
                        strengths=strength_arr,
                        live_times=np.asarray(data.live_times, dtype=float)[
                            cached_rows:
                        ],
                        fe_indices=np.asarray(data.fe_indices, dtype=int)[cached_rows:],
                        pb_indices=np.asarray(data.pb_indices, dtype=int)[cached_rows:],
                    )
                    counts_arr = np.vstack(
                        [
                            cached_counts,
                            np.asarray(suffix_counts, dtype=float),
                        ]
                    )
                    self._store_surface_diagnostic_response_prefix_cache(
                        prefix_key,
                        data,
                        counts_arr,
                    )
                    self._store_surface_diagnostic_response_cache(cache_key, counts_arr)
                    return counts_arr.copy()
        counts = expected_counts_per_source(
            kernel=kernel,
            isotope=isotope,
            detector_positions=data.detector_positions,
            sources=source_arr,
            strengths=strength_arr,
            live_times=data.live_times,
            fe_indices=data.fe_indices,
            pb_indices=data.pb_indices,
        )
        counts_arr = np.asarray(counts, dtype=float)
        if cache_enabled and cache_key is not None:
            self._store_surface_diagnostic_response_cache(cache_key, counts_arr)
            if prefix_key is not None:
                self._store_surface_diagnostic_response_prefix_cache(
                    prefix_key,
                    data,
                    counts_arr,
                )
        return counts_arr

    def _cached_expected_counts_per_source(
        self,
        *,
        filt: IsotopeParticleFilter,
        isotope: str,
        data: StructuralGeometryBatch,
        sources: NDArray[np.float64],
        strengths: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return expected counts using one isotope filter's physical kernel."""
        return self._cached_expected_counts_for_kernel(
            kernel=filt.continuous_kernel,
            isotope=isotope,
            data=data,
            sources=sources,
            strengths=strengths,
        )

    def _resolve_mu_by_isotope(
        self, mu_by_isotope: Dict[str, object] | None
    ) -> Dict[str, object]:
        """
        Ensure per-isotope attenuation coefficients are available for all isotopes.

        When missing, attempt to populate values from the HVL/TVL table; otherwise raise.
        """
        from measurement.shielding import HVL_TVL_TABLE_MM, mu_by_isotope_from_tvl_mm

        def _norm_key(name: str) -> str:
            """Return a normalized isotope key for attenuation lookup."""
            return re.sub(r"[^A-Za-z0-9]", "", name).upper()

        canonical_by_norm = {
            "CS137": "Cs-137",
            "CO60": "Co-60",
            "EU154": "Eu-154",
        }

        resolved: Dict[str, object] = {}
        if mu_by_isotope is not None:
            resolved.update(mu_by_isotope)
        normalized: Dict[str, object] = {}
        for key, value in resolved.items():
            normalized[_norm_key(key)] = value
        isotope_names = self.isotopes
        if isotope_names:
            still_missing: List[str] = []
            for iso in isotope_names:
                if iso in resolved:
                    continue
                norm = _norm_key(iso)
                if norm in normalized:
                    resolved[iso] = normalized[norm]
                    continue
                canonical = canonical_by_norm.get(norm)
                if canonical is not None:
                    table_vals = mu_by_isotope_from_tvl_mm(
                        HVL_TVL_TABLE_MM, isotopes=[canonical]
                    )
                    if canonical in table_vals:
                        resolved[iso] = table_vals[canonical]
                        normalized[norm] = table_vals[canonical]
                        if canonical not in resolved:
                            resolved[canonical] = table_vals[canonical]
                        continue
                still_missing.append(iso)
            if still_missing:
                missing_list = ", ".join(still_missing)
                raise ValueError(
                    "mu_by_isotope is missing entries for isotopes: "
                    f"{missing_list}. Ensure isotope names match the HVL/TVL table keys."
                )
        return resolved

    def _ensure_kernel_cache(self) -> None:
        """Build the discrete kernel cache when it is first needed."""
        if self.kernel_cache is not None:
            return
        if len(self.poses) == 0:
            raise ValueError("No poses added; cannot build kernel cache.")
        poses_arr = np.stack(self.poses, axis=0)
        self.kernel_cache = MeasurementGeometry(
            poses=poses_arr,
            orientations=self.normals,
            shield_params=self.shield_params,
            mu_by_isotope=self.mu_by_isotope,
        )
        pf_conf = self._build_pf_config()
        if self.filters:
            expected = set(self.isotopes)
            actual = set(self.filters)
            if actual != expected:
                raise RuntimeError(
                    "Pure PF requires exactly one initialized filter per isotope."
                )
            for iso in self.isotopes:
                self.filters[iso].set_kernel(self.kernel_cache)
        else:
            for iso in self.isotopes:
                self.filters[iso] = self._build_filter(iso, pf_conf)
        self._configure_joint_particle_filters()

    def _build_filter(self, isotope: str, pf_conf: PFConfig) -> IsotopeParticleFilter:
        """Build an isotope filter with shared PF observation-model settings."""
        return IsotopeParticleFilter(
            isotope,
            kernel=self.kernel_cache,
            config=pf_conf,
            obstacle_grid=self.obstacle_grid,
            obstacle_height_m=self.obstacle_height_m,
            obstacle_mu_by_isotope=self.obstacle_mu_by_isotope,
            obstacle_buildup_coeff=self.obstacle_buildup_coeff,
            detector_radius_m=self.detector_radius_m,
            detector_aperture_radius_m=self.detector_aperture_radius_m,
            detector_aperture_samples=self.detector_aperture_samples,
            detector_aperture_sampling=self.detector_aperture_sampling,
            source_extent_radius_m=self.source_extent_radius_m,
            source_extent_samples=self.source_extent_samples,
            line_mu_by_isotope=self.line_mu_by_isotope,
            additive_scatter_response=self.additive_scatter_response,
            random_seed=self.random_seed,
        )

    def _build_pf_config(self) -> PFConfig:
        """Build the per-isotope config from the shared PF field contract."""
        shared_values = {
            field.name: getattr(self.pf_config, field.name)
            for field in fields(PFConfig)
            if hasattr(self.pf_config, field.name)
        }
        return PFConfig(**shared_values)

    def _gpu_enabled(self) -> bool:
        """Return True if GPU computation is enabled and available."""
        from pf import gpu_utils

        if not self.pf_config.use_gpu:
            raise RuntimeError(
                "GPU-only mode: enable use_gpu in RotatingShieldPFConfig."
            )
        gpu_utils.require_torch_compute_device(
            str(self.pf_config.gpu_device),
            str(self.pf_config.gpu_dtype),
        )
        return True

    def _can_use_gpu(self) -> bool:
        """Select explicit NumPy mode or require the configured torch device."""
        if not self.pf_config.use_gpu:
            return False
        return self._gpu_enabled()

    def continuous_kernel(
        self,
        *,
        detector_aperture_samples: int | None = None,
        use_gpu: bool | None = None,
    ) -> ContinuousKernel:
        """Build the shared ContinuousKernel for PF, planning, and diagnostics."""
        requested_aperture_samples = (
            self.detector_aperture_samples
            if detector_aperture_samples is None
            else detector_aperture_samples
        )
        active_aperture_samples = _strict_nonnegative_integer(
            requested_aperture_samples,
            name="detector_aperture_samples",
        )
        if active_aperture_samples < 1:
            raise ValueError("detector_aperture_samples must be positive.")
        active_use_gpu = _strict_config_boolean(
            self.pf_config.use_gpu if use_gpu is None else use_gpu,
            name="use_gpu",
        )
        return ContinuousKernel(
            mu_by_isotope=self.mu_by_isotope,
            shield_params=self.shield_params,
            orientations=self.normals,
            use_gpu=active_use_gpu,
            gpu_device=str(self.pf_config.gpu_device),
            gpu_dtype=str(self.pf_config.gpu_dtype),
            obstacle_grid=self.obstacle_grid,
            obstacle_height_m=self.obstacle_height_m,
            obstacle_mu_by_isotope=self.obstacle_mu_by_isotope,
            obstacle_buildup_coeff=self.obstacle_buildup_coeff,
            detector_radius_m=self.detector_radius_m,
            detector_aperture_radius_m=self.detector_aperture_radius_m,
            detector_aperture_samples=active_aperture_samples,
            detector_aperture_sampling=self.detector_aperture_sampling,
            source_extent_radius_m=self.source_extent_radius_m,
            source_extent_samples=self.source_extent_samples,
            line_mu_by_isotope=self.line_mu_by_isotope,
            additive_scatter_response=self.additive_scatter_response,
        )

    def surface_transport_positions(
        self,
        isotope: str,
        anchors_xyz: NDArray[np.float64],
        chart_ids: NDArray[np.int64],
        surface_uv: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Map exact surface anchors to the shared air-side physics positions."""
        isotope_key = str(isotope)
        try:
            filt = self.filters[isotope_key]
        except KeyError as error:
            raise ValueError(
                f"Unknown isotope for surface transport: {isotope_key!r}."
            ) from error
        atlas = filt._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("The continuous surface atlas is unavailable.")
        anchors = np.asarray(anchors_xyz, dtype=np.float64)
        raw_chart_ids = np.asarray(chart_ids)
        uv = np.asarray(surface_uv, dtype=np.float64)
        if (
            anchors.ndim != 2
            or anchors.shape[1] != 3
            or not np.issubdtype(raw_chart_ids.dtype, np.integer)
            or raw_chart_ids.shape != anchors.shape[:1]
            or uv.shape != anchors.shape[:1] + (2,)
        ):
            raise ValueError(
                "Surface anchors, chart IDs, and UV coordinates are misaligned."
            )
        chart_id_array = np.asarray(raw_chart_ids, dtype=np.int64)
        mapped = atlas.positions_xyz(chart_id_array, uv)
        if not np.allclose(mapped, anchors, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "Surface chart/UV coordinates do not map to their exact anchors."
            )
        return filt._surface_transport_positions(
            anchors,
            chart_ids=chart_id_array,
        )

    def configured_isotope_order(self) -> tuple[str, ...]:
        """Return the stable pure-PF isotope order."""
        return tuple(dict.fromkeys(str(isotope) for isotope in self.isotopes))

    def joint_isotope_order(self) -> tuple[str, ...]:
        """Return the canonical order used by every joint likelihood vector."""
        order = tuple(sorted(self.configured_isotope_order()))
        if not order or len(order) != len(self.isotopes):
            raise RuntimeError("Joint PF requires unique configured isotope names.")
        return order

    def _configure_joint_particle_filters(self) -> None:
        """Attach the conditional joint target and verify aligned initialization."""
        if not self.filters:
            return
        if not self._joint_particles_initialized:
            self._initialize_joint_particles_from_product_prior()
        for isotope in self.joint_isotope_order():
            filt = self.filters[isotope]
            filt.set_joint_target_evaluator(
                self._joint_structural_target_evaluator
            )
            filt.set_joint_proposal_evaluator(
                self._joint_structural_proposal_evaluator
            )
        self._assert_joint_particle_alignment()

    def _joint_row_identity_root(self, *, particle_count: int) -> str:
        """Return the deterministic contract root for joint row identities."""
        atlas_sha256 = self._assert_joint_surface_atlas_alignment()
        model_sha256 = self._full_spectrum_model().contract_hash_sha256
        return sha256_json(
            {
                "schema_version": 1,
                "identity_domain": "pure_pf_joint_row_identity_root_v1",
                "random_seed": self.random_seed,
                "isotope_order": list(self.joint_isotope_order()),
                "particle_count": int(particle_count),
                "surface_atlas_sha256": atlas_sha256,
                "full_spectrum_contract_sha256": model_sha256,
                "pf_config": {
                    config_field.name: getattr(
                        self.pf_config,
                        config_field.name,
                    )
                    for config_field in fields(self.pf_config)
                },
            }
        )

    def _initialize_joint_particles_from_product_prior(self) -> None:
        """Draw aligned rows directly from the independent isotope product prior."""
        order = self.joint_isotope_order()
        if set(self.filters) != set(order):
            raise RuntimeError(
                "Joint prior initialization requires every isotope filter."
            )
        particle_counts = {
            int(filt.config.num_particles) for filt in self.filters.values()
        }
        if len(particle_counts) != 1:
            raise RuntimeError(
                "Joint prior initialization requires one common particle count."
            )
        particle_count = next(iter(particle_counts))
        if particle_count <= 0:
            raise RuntimeError(
                "Joint prior initialization requires positive particle count."
            )
        common_log_weight = float(-np.log(particle_count))
        identity_root = self._joint_row_identity_root(
            particle_count=particle_count
        )
        row_identities = tuple(
            JointRowIdentity.initial(
                root_sha256=identity_root,
                ordinal=row,
            )
            for row in range(particle_count)
        )
        particles_by_isotope: dict[str, list[IsotopeParticle]] = {}
        for isotope in order:
            filt = self.filters[isotope]
            atlas = filt._structural_rj_surface_atlas
            cardinality_prior = filt._structural_rj_cardinality_prior
            if atlas is None or cardinality_prior is None:
                raise RuntimeError(
                    "Joint product-prior initialization requires a surface "
                    "atlas and cardinality prior."
                )
            rng = named_random_generator(
                self.random_seed,
                "joint_product_prior",
                isotope,
            )
            if filt.config.variable_cardinality:
                cardinalities = rng.choice(
                    cardinality_prior.probabilities.size,
                    size=particle_count,
                    replace=True,
                    p=cardinality_prior.probabilities,
                ).astype(np.int64, copy=False)
            else:
                cardinalities = np.full(
                    particle_count,
                    int(filt.config.init_num_sources[0]),
                    dtype=np.int64,
                )
            source_offsets = np.concatenate(
                (
                    np.zeros(1, dtype=np.int64),
                    np.cumsum(cardinalities, dtype=np.int64),
                )
            )
            total_sources = int(source_offsets[-1])
            chart_ids, surface_uv, _ = atlas.sample(
                total_sources,
                rng=rng,
            )
            strengths = np.asarray(
                filt._strength_prior.sample(total_sources, rng=rng),
                dtype=np.float64,
            )
            particles: list[IsotopeParticle] = []
            # Numerical sampling is batched above; this loop only creates the
            # variable-length Python state objects used by the RJ kernel.
            for row, cardinality in enumerate(cardinalities.tolist()):
                begin = int(source_offsets[row])
                end = int(source_offsets[row + 1])
                state = IsotopeState(
                    num_sources=int(cardinality),
                    strengths=np.asarray(
                        strengths[begin:end],
                        dtype=np.float64,
                    ).copy(),
                    surface_chart_ids=np.asarray(
                        chart_ids[begin:end],
                        dtype=np.int64,
                    ).copy(),
                    surface_uv=np.asarray(
                        surface_uv[begin:end],
                        dtype=np.float64,
                    ).copy(),
                )
                filt._canonicalize_structural_rj_state(state)
                particles.append(
                    IsotopeParticle(
                        state=state,
                        log_weight=common_log_weight,
                        joint_row_identity=row_identities[row],
                    )
                )
            particles_by_isotope[isotope] = particles
        for isotope in order:
            filt = self.filters[isotope]
            filt.continuous_particles = particles_by_isotope[isotope]
            filt.N = particle_count
            filt.config.num_particles = particle_count
        self._joint_row_identity_root_sha256 = identity_root
        self._joint_row_generation = 0
        self._joint_cumulative_lineage_ids = np.arange(
            particle_count,
            dtype=np.int64,
        )
        self._joint_particles_initialized = True

    def _assert_joint_particle_alignment(self) -> NDArray[np.float64]:
        """Return common log weights or reject any broken joint-row alignment."""
        order = self.joint_isotope_order()
        if set(self.filters) != set(order):
            raise RuntimeError(
                "Joint PF requires exactly one filter for every configured isotope."
            )
        self._assert_joint_surface_atlas_alignment()
        particle_counts = {
            len(self.filters[isotope].continuous_particles) for isotope in order
        }
        if len(particle_counts) != 1:
            raise RuntimeError(
                "Joint isotope filters must have the same particle count."
            )
        particle_count = next(iter(particle_counts))
        if particle_count <= 0:
            raise RuntimeError("Joint PF requires at least one aligned particle.")
        identity_root = self._joint_row_identity_root_sha256
        identity_generation = self._joint_row_generation
        if identity_root is None or identity_generation is None:
            raise RuntimeError(
                "Joint PF row identity contract is not initialized."
            )
        reference_identities: tuple[JointRowIdentity, ...] | None = None
        for isotope in order:
            identities: list[JointRowIdentity] = []
            for row, particle in enumerate(
                self.filters[isotope].continuous_particles
            ):
                identity = particle.joint_row_identity
                if not isinstance(identity, JointRowIdentity):
                    raise RuntimeError(
                        "Every joint isotope particle requires an immutable "
                        "joint row identity."
                    )
                if (
                    identity.validate() != identity.row_sha256
                    or identity.root_sha256 != identity_root
                    or identity.generation != identity_generation
                    or identity.ordinal != row
                ):
                    raise RuntimeError(
                        "Joint row identity does not match the current "
                        "estimator generation and row ordering."
                    )
                identities.append(identity)
            identity_tuple = tuple(identities)
            if reference_identities is None:
                if len(
                    {identity.row_sha256 for identity in identity_tuple}
                ) != particle_count:
                    raise RuntimeError(
                        "Current joint PF row identities must be unique."
                    )
                reference_identities = identity_tuple
            elif identity_tuple != reference_identities:
                raise RuntimeError(
                    "Joint isotope filters lost their authenticated row "
                    "identity ordering."
                )
        reference = np.asarray(
            [
                particle.log_weight
                for particle in self.filters[order[0]].continuous_particles
            ],
            dtype=np.float64,
        )
        if (
            reference.shape != (particle_count,)
            or np.any(np.isnan(reference))
            or np.any(np.isposinf(reference))
            or not np.any(np.isfinite(reference))
        ):
            raise RuntimeError("Joint PF common log weights are invalid.")
        for isotope in order[1:]:
            candidate = np.asarray(
                [
                    particle.log_weight
                    for particle in self.filters[isotope].continuous_particles
                ],
                dtype=np.float64,
            )
            if not np.array_equal(candidate, reference):
                raise RuntimeError(
                    "Joint isotope filters lost their exact common weights/order."
                )
        return reference

    def _assert_joint_surface_atlas_alignment(self) -> str:
        """Return the shared atlas digest or reject isotope geometry drift."""
        order = self.joint_isotope_order()
        if set(self.filters) != set(order):
            raise RuntimeError(
                "Joint PF requires exactly one filter for every configured isotope."
            )
        digests = tuple(
            str(
                getattr(
                    self.filters[isotope],
                    "structural_rj_surface_atlas_sha256",
                    "",
                )
            ).strip()
            for isotope in order
        )
        if any(
            len(digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in digest.lower()
            )
            for digest in digests
        ):
            raise RuntimeError(
                "Every joint isotope filter requires a valid surface-atlas "
                "digest."
            )
        if len(set(digest.lower() for digest in digests)) != 1:
            raise RuntimeError(
                "Joint isotope rows cannot use different continuous surface "
                "atlases."
            )
        return digests[0].lower()

    def continuous_surface_atlas(self) -> Any:
        """Return the one authoritative atlas shared by all isotope filters."""
        if not self.filters:
            raise RuntimeError(
                "The continuous surface atlas is unavailable before PF "
                "initialization."
            )
        self._assert_joint_surface_atlas_alignment()
        atlas = next(iter(self.filters.values()))._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("The continuous surface atlas is unavailable.")
        return atlas

    def _assign_joint_log_weights(
        self,
        normalized_log_weights: NDArray[np.float64],
    ) -> None:
        """Copy one normalized log-weight vector to every aligned isotope row."""
        values = np.asarray(normalized_log_weights, dtype=np.float64).reshape(-1)
        self._assert_joint_particle_alignment()
        if (
            values.size
            != len(
                self.filters[self.joint_isotope_order()[0]].continuous_particles
            )
            or np.any(np.isnan(values))
            or np.any(np.isposinf(values))
            or not np.any(np.isfinite(values))
        ):
            raise ValueError("Joint normalized log weights are invalid.")
        finite = np.isfinite(values)
        normalizer = float(logsumexp(values[finite]))
        if not np.isclose(normalizer, 0.0, rtol=0.0, atol=1.0e-10):
            raise ValueError("Joint log weights must already be normalized.")
        for isotope in self.joint_isotope_order():
            for particle, log_weight in zip(
                self.filters[isotope].continuous_particles,
                values,
            ):
                particle.log_weight = float(log_weight)
        self._assert_joint_particle_alignment()

    def _strict_joint_particle_weights(self) -> NDArray[np.float64]:
        """Return normalized joint weights or reject invalid posterior mass."""
        log_weights = self._assert_joint_particle_alignment()
        finite = np.isfinite(log_weights)
        weights = np.zeros(log_weights.size, dtype=np.float64)
        weights[finite] = np.exp(
            log_weights[finite] - float(logsumexp(log_weights[finite]))
        )
        return validated_probability_distribution(
            weights,
            name="joint PF particle weights",
        )

    def _full_spectrum_model(self) -> FullSpectrumGenerativeModel:
        """Return the required independently validated generative model."""
        return self.full_spectrum_generative_model

    def _joint_station_from_spectrum_records(
        self,
        records: Sequence[Sequence[object]],
        *,
        pose_idx: int,
        station_sequence_id: int,
        generative_contract_hash_sha256: str,
    ) -> JointStationObservation:
        """Build one strict view-major full-spectrum station observation."""
        if not records:
            raise ValueError("A joint station must contain at least one view.")
        model = self._full_spectrum_model()
        if not isinstance(generative_contract_hash_sha256, str):
            raise TypeError(
                "generative_contract_hash_sha256 must be a JSON string."
            )
        supplied_hash = generative_contract_hash_sha256
        if supplied_hash != model.contract_hash_sha256:
            raise ValueError(
                "Station full-spectrum contract hash differs from the active "
                "generative model."
            )
        bin_count = int(np.asarray(model.energy_axis_keV).size)
        spectra: list[NDArray[np.float64]] = []
        raw_fe_indices: list[int] = []
        raw_pb_indices: list[int] = []
        live_times = np.empty(len(records), dtype=np.float64)
        for view_index, record in enumerate(records):
            if len(record) != 4:
                raise ValueError(
                    "Full-spectrum station records must have exactly four "
                    "fields: (spectrum, Fe, Pb, live time)."
                )
            spectrum, fe_index, pb_index, live_time_s = record
            raw_spectrum = np.asarray(spectrum)
            if raw_spectrum.ndim != 1:
                raise ValueError(
                    "Each full-spectrum station record must contain one "
                    "one-dimensional raw spectrum."
                )
            validated = validate_observed_spectrum(
                raw_spectrum[np.newaxis, :],
                expected_bin_count=bin_count,
            )
            spectra.append(validated[0])
            raw_fe_indices.append(
                _strict_nonnegative_integer(
                    fe_index,
                    name=f"records[{view_index}].fe_index",
                )
            )
            raw_pb_indices.append(
                _strict_nonnegative_integer(
                    pb_index,
                    name=f"records[{view_index}].pb_index",
                )
            )
            resolved_live_time = _strict_config_number(
                live_time_s,
                name=f"records[{view_index}].live_time_s",
            )
            if resolved_live_time <= 0.0:
                raise ValueError(
                    "Full-spectrum station live times must be positive."
                )
            live_times[view_index] = resolved_live_time
        fe_indices, pb_indices = validate_orientation_pair_indices(
            np.asarray(raw_fe_indices),
            np.asarray(raw_pb_indices),
            orientation_count=int(self.num_orientations),
            expected_count=len(records),
        )
        resolved_pose_idx = _strict_nonnegative_integer(
            pose_idx,
            name="pose_idx",
        )
        resolved_station_sequence_id = _strict_nonnegative_integer(
            station_sequence_id,
            name="station_sequence_id",
        )
        return JointStationObservation(
            spectrum_vb=np.ascontiguousarray(np.stack(spectra, axis=0)),
            energy_axis_keV=np.ascontiguousarray(
                np.asarray(model.energy_axis_keV, dtype=np.float64)
            ),
            generative_contract_hash_sha256=supplied_hash,
            pose_idx=resolved_pose_idx,
            detector_position_xyz_m=self._registered_detector_position_xyz(
                resolved_pose_idx
            ),
            fe_indices=np.ascontiguousarray(fe_indices),
            pb_indices=np.ascontiguousarray(pb_indices),
            live_times_s=np.ascontiguousarray(live_times),
            station_sequence_id=resolved_station_sequence_id,
        )

    def _joint_station_expected_means_torch(
        self,
        station: JointStationObservation,
    ) -> "torch.Tensor":
        """Return predicted spectra shaped particle x view x energy bin."""
        model = self._full_spectrum_model()
        total, uncollided, features = (
            self._joint_station_transport_components_torch(station)
        )
        result = model.predict_mean_torch(
            total,
            uncollided,
            features,
            station.live_times_s,
        )
        expected_shape = (
            len(self.filters[self.joint_isotope_order()[0]].continuous_particles),
            int(station.fe_indices.size),
            int(station.energy_axis_keV.size),
        )
        if tuple(result.shape) != expected_shape:
            raise RuntimeError(
                "Joint expected-spectrum tensor has an invalid aligned shape."
            )
        return result

    def _joint_line_layout(
        self,
    ) -> Dict[
        str,
        tuple[
            NDArray[np.int64],
            NDArray[np.int64],
            NDArray[np.float64],
        ],
    ]:
        """Return global columns, isotope line indices, and branching weights."""
        model = self._full_spectrum_model()
        line_identity = tuple(model.line_identity)
        layout: Dict[
            str,
            tuple[
                NDArray[np.int64],
                NDArray[np.int64],
                NDArray[np.float64],
            ],
        ] = {}
        for isotope in self.joint_isotope_order():
            global_columns = np.asarray(
                [
                    column
                    for column, payload in enumerate(line_identity)
                    if str(payload["isotope"]) == isotope
                ],
                dtype=np.int64,
            )
            local_indices = np.asarray(
                [
                    int(line_identity[int(column)]["transport_line_index"])
                    for column in global_columns
                ],
                dtype=np.int64,
            )
            branching_weights = np.asarray(
                [
                    float(line_identity[int(column)]["branching_weight"])
                    for column in global_columns
                ],
                dtype=np.float64,
            )
            if (
                global_columns.size == 0
                or np.unique(local_indices).size != local_indices.size
                or np.any(local_indices < 0)
                or np.any(~np.isfinite(branching_weights))
                or np.any(branching_weights <= 0.0)
            ):
                raise RuntimeError(
                    f"Full-spectrum line layout is invalid for {isotope!r}."
                )
            configured_weights = self.filters[
                isotope
            ].continuous_kernel.line_branching_weights(
                isotope,
                local_indices,
            )
            if not np.allclose(
                configured_weights,
                branching_weights / float(np.sum(branching_weights)),
                rtol=1.0e-12,
                atol=1.0e-15,
            ):
                raise RuntimeError(
                    "Full-spectrum branching weights differ from the physical "
                    f"kernel for {isotope!r}."
                )
            layout[isotope] = (
                global_columns,
                local_indices,
                branching_weights,
            )
        covered = np.concatenate(
            [value[0] for value in layout.values()]
        )
        if not np.array_equal(
            np.sort(covered),
            np.arange(len(line_identity), dtype=np.int64),
        ):
            raise RuntimeError(
                "Full-spectrum line layout does not cover every global line."
            )
        return layout

    def _joint_isotope_station_transport_components_torch(
        self,
        station: JointStationObservation,
        isotope: str,
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Return one isotope's fixed-slot components in global line layout."""
        import torch

        self._assert_joint_particle_alignment()
        model = self._full_spectrum_model()
        layout = self._joint_line_layout()
        line_count = len(tuple(model.line_identity))
        feature_count = len(tuple(model.transport_feature_order))
        isotope_key = str(isotope)
        if isotope_key not in self.joint_isotope_order():
            raise KeyError(f"Unknown joint PF isotope: {isotope_key!r}.")
        global_columns, local_indices, branching_weights = layout[isotope_key]
        components = self.filters[
            isotope_key
        ]._continuous_expected_line_transport_components_pair_sequence_torch(
            pose_idx=int(station.pose_idx),
            fe_indices=station.fe_indices,
            pb_indices=station.pb_indices,
            live_times_s=station.live_times_s,
            positive_line_indices=local_indices,
        )
        total_local = components.total_kernel.to(dtype=torch.float64)
        uncollided_local = components.uncollided_kernel.to(
            device=total_local.device,
            dtype=torch.float64,
        )
        feature_local = torch.stack(
            (
                components.tau_fe,
                components.tau_pb,
                components.tau_obstacle,
                components.distance_m,
            ),
            dim=-1,
        ).to(device=total_local.device, dtype=torch.float64)
        expected_local_shape = tuple(total_local.shape)
        expected_slots = int(self.pf_config.max_sources or 0)
        if (
            total_local.ndim != 4
            or int(total_local.shape[2]) != expected_slots
            or tuple(uncollided_local.shape) != expected_local_shape
            or tuple(feature_local.shape)
            != expected_local_shape + (feature_count,)
            or int(total_local.shape[-1]) != int(local_indices.size)
        ):
            raise RuntimeError(
                "Full-spectrum isotope transport must use the configured "
                "fixed source-slot layout."
            )
        weights = torch.as_tensor(
            branching_weights,
            dtype=torch.float64,
            device=total_local.device,
        ).view(1, 1, 1, -1)
        total_local = total_local * weights
        uncollided_local = uncollided_local * weights
        global_total = torch.zeros(
            (*total_local.shape[:-1], line_count),
            dtype=torch.float64,
            device=total_local.device,
        )
        global_uncollided = torch.zeros_like(global_total)
        global_features = torch.zeros(
            (*total_local.shape[:-1], line_count, feature_count),
            dtype=torch.float64,
            device=total_local.device,
        )
        global_total[..., global_columns] = total_local
        global_uncollided[..., global_columns] = uncollided_local
        global_features[..., global_columns, :] = feature_local
        if (
            bool(torch.any(~torch.isfinite(global_total)))
            or bool(torch.any(~torch.isfinite(global_uncollided)))
            or bool(torch.any(~torch.isfinite(global_features)))
            or bool(torch.any(global_total < 0.0))
            or bool(torch.any(global_uncollided < 0.0))
        ):
            raise RuntimeError(
                "Full-spectrum transport components must be finite, "
                "nonnegative source-slot contributions."
            )
        return global_total, global_uncollided, global_features

    def _joint_station_transport_components_torch(
        self,
        station: JointStationObservation,
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Return source-resolved total, uncollided, and geometry features."""
        total_parts: list["torch.Tensor"] = []
        uncollided_parts: list["torch.Tensor"] = []
        feature_parts: list["torch.Tensor"] = []
        reference_device = None
        for isotope in self.joint_isotope_order():
            total_local, uncollided_local, feature_local = (
                self._joint_isotope_station_transport_components_torch(
                    station,
                    isotope,
                )
            )
            if reference_device is None:
                reference_device = total_local.device
            elif total_local.device != reference_device:
                total_local = total_local.to(device=reference_device)
                uncollided_local = uncollided_local.to(
                    device=reference_device
                )
                feature_local = feature_local.to(device=reference_device)
            total_parts.append(total_local)
            uncollided_parts.append(uncollided_local)
            feature_parts.append(feature_local)
        if not total_parts:
            raise RuntimeError(
                "Joint transport components require configured isotopes."
            )
        import torch

        total = torch.cat(total_parts, dim=2)
        uncollided = torch.cat(uncollided_parts, dim=2)
        features = torch.cat(feature_parts, dim=2)
        if (
            bool(torch.any(~torch.isfinite(total)))
            or bool(torch.any(~torch.isfinite(uncollided)))
            or bool(torch.any(~torch.isfinite(features)))
            or bool(torch.any(total < 0.0))
            or bool(torch.any(uncollided < 0.0))
        ):
            raise RuntimeError(
                "Full-spectrum transport components must be finite, "
                "nonnegative source-slot contributions."
            )
        return total, uncollided, features

    def _joint_station_expected_means_np(
        self,
        station: JointStationObservation,
    ) -> NDArray[np.float64]:
        """Return the NumPy equivalent of aligned station particle means."""
        return (
            self._joint_station_expected_means_torch(station)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )

    def _joint_station_log_likelihood_torch(
        self,
        station: JointStationObservation,
    ) -> "torch.Tensor":
        """Evaluate the sole joint full-spectrum likelihood for all particles."""
        model = self._full_spectrum_model()
        total, uncollided, features = (
            self._joint_station_transport_components_torch(station)
        )
        result = model.log_likelihood_torch(
            station.spectrum_vb,
            total,
            uncollided,
            features,
            station.live_times_s,
        )
        if tuple(result.shape) != (int(total.shape[0]),):
            raise RuntimeError(
                "Full-spectrum likelihood must return one value per particle."
            )
        import torch

        if bool(torch.any(torch.isnan(result)).detach().cpu().item()) or bool(
            torch.any(torch.isinf(result) & (result > 0.0))
            .detach()
            .cpu()
            .item()
        ):
            raise RuntimeError(
                "Full-spectrum likelihood contains NaN or positive infinity."
            )
        if not bool(torch.any(torch.isfinite(result)).detach().cpu().item()):
            raise RuntimeError(
                "Full-spectrum likelihood is negative infinity for every "
                "particle; the observation is outside model support."
            )
        return result

    def _joint_history_structural_geometry(
        self,
        isotope: str,
        stations: Sequence[JointStationObservation],
    ) -> StructuralGeometryBatch:
        """Build geometry-only evidence for exact conditional RJ proposals."""
        isotope_key = str(isotope)
        order = self.joint_isotope_order()
        if isotope_key not in order:
            raise KeyError(f"Unknown joint PF isotope: {isotope_key!r}.")
        total_rows = sum(int(station.fe_indices.size) for station in stations)
        if total_rows <= 0:
            raise ValueError("Joint RJ history requires at least one station row.")
        detector_positions = np.concatenate(
            [
                np.repeat(
                    np.asarray(
                        station.detector_position_xyz_m,
                        dtype=np.float64,
                    ).reshape(1, 3),
                    int(station.fe_indices.size),
                    axis=0,
                )
                for station in stations
            ],
            axis=0,
        )
        fe_indices = np.concatenate(
            [np.asarray(station.fe_indices, dtype=np.int64) for station in stations]
        )
        pb_indices = np.concatenate(
            [np.asarray(station.pb_indices, dtype=np.int64) for station in stations]
        )
        live_times = np.concatenate(
            [
                np.asarray(station.live_times_s, dtype=np.float64)
                for station in stations
            ]
        )
        sequence_ids = np.concatenate(
            [
                np.full(
                    int(station.fe_indices.size),
                    int(station.station_sequence_id),
                    dtype=np.int64,
                )
                for station in stations
            ]
        )
        return StructuralGeometryBatch(
            detector_positions=np.ascontiguousarray(detector_positions),
            fe_indices=np.ascontiguousarray(fe_indices),
            pb_indices=np.ascontiguousarray(pb_indices),
            live_times=np.ascontiguousarray(live_times),
            station_sequence_ids=np.ascontiguousarray(sequence_ids),
        )

    def _validate_joint_structural_geometry(
        self,
        data: StructuralGeometryBatch,
        stations: Sequence[JointStationObservation],
    ) -> None:
        """Require exact row-wise agreement with the active station history."""
        active_geometry = self._active_joint_structural_geometry
        if active_geometry is not None:
            if data is not active_geometry:
                raise ValueError(
                    "Conditional isotope evidence is not the immutable active "
                    "joint-history geometry."
                )
            return
        row_start = 0
        for station in stations:
            row_count = int(np.asarray(station.fe_indices).size)
            row_stop = row_start + row_count
            row_slice = slice(row_start, row_stop)
            expected_positions = np.repeat(
                np.asarray(
                    station.detector_position_xyz_m,
                    dtype=np.float64,
                ).reshape(1, 3),
                row_count,
                axis=0,
            )
            if not (
                np.array_equal(
                    data.detector_positions[row_slice],
                    expected_positions,
                )
                and np.array_equal(
                    data.fe_indices[row_slice],
                    np.asarray(station.fe_indices, dtype=np.int64),
                )
                and np.array_equal(
                    data.pb_indices[row_slice],
                    np.asarray(station.pb_indices, dtype=np.int64),
                )
                and np.array_equal(
                    data.live_times[row_slice],
                    np.asarray(station.live_times_s, dtype=np.float64),
                )
                and np.array_equal(
                    data.station_sequence_ids[row_slice],
                    np.full(
                        row_count,
                        int(station.station_sequence_id),
                        dtype=np.int64,
                    ),
                )
            ):
                raise ValueError(
                    "Conditional isotope evidence geometry differs from the "
                    "active joint station history."
                )
            row_start = row_stop
        if row_start != data.row_count:
            raise ValueError(
                "Conditional isotope evidence row count differs from the "
                "active joint station history."
            )

    def _refresh_joint_structural_transport_cache(
        self,
        stations: Sequence[JointStationObservation],
    ) -> None:
        """Cache source-resolved transport components for conditional RJ."""
        station_components = [
            tuple(
                value.detach().cpu().numpy().astype(np.float64, copy=False)
                for value in self._joint_station_transport_components_torch(
                    station
                )
            )
            for station in stations
        ]
        self._joint_structural_transport_cache = tuple(
            np.concatenate(
                [components[index] for components in station_components],
                axis=1,
            )
            for index in range(3)
        )

    def _refresh_joint_structural_transport_cache_isotope(
        self,
        stations: Sequence[JointStationObservation],
        isotope: str,
    ) -> None:
        """Refresh only one moved isotope's fixed source-slot cache slice."""
        cache = self._joint_structural_transport_cache
        if cache is None:
            raise RuntimeError(
                "Incremental structural transport refresh requires a cache."
            )
        order = self.joint_isotope_order()
        isotope_key = str(isotope)
        if isotope_key not in order:
            raise KeyError(f"Unknown joint PF isotope: {isotope_key!r}.")
        station_components = [
            tuple(
                value.detach().cpu().numpy().astype(
                    np.float64,
                    copy=False,
                )
                for value in (
                    self._joint_isotope_station_transport_components_torch(
                        station,
                        isotope_key,
                    )
                )
            )
            for station in stations
        ]
        refreshed = tuple(
            np.concatenate(
                [components[index] for components in station_components],
                axis=1,
            )
            for index in range(3)
        )
        slots_per_isotope = int(self.pf_config.max_sources or 0)
        slot_start = order.index(isotope_key) * slots_per_isotope
        slot_stop = slot_start + slots_per_isotope
        mutable_cache = [np.asarray(values) for values in cache]
        for cached_values, refreshed_values in zip(
            mutable_cache,
            refreshed,
            strict=True,
        ):
            if (
                cached_values.shape[:2] != refreshed_values.shape[:2]
                or cached_values.shape[2] < slot_stop
                or refreshed_values.shape[2] != slots_per_isotope
                or cached_values.shape[3:] != refreshed_values.shape[3:]
            ):
                raise RuntimeError(
                    "Incremental isotope transport cache shapes disagree."
                )
            cached_values[:, :, slot_start:slot_stop, ...] = refreshed_values
        self._joint_structural_transport_cache = tuple(mutable_cache)

    def _full_spectrum_log_likelihood_numpy(
        self,
        *,
        filt: IsotopeParticleFilter,
        station: JointStationObservation,
        total_nvsl: NDArray[np.float64],
        uncollided_nvsl: NDArray[np.float64],
        features_nvslf: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate one batched station on GPU when available, else NumPy."""
        model = self._full_spectrum_model()
        total = np.asarray(total_nvsl, dtype=np.float64)
        uncollided = np.asarray(uncollided_nvsl, dtype=np.float64)
        features = np.asarray(features_nvslf, dtype=np.float64)
        if filt._can_use_gpu():
            from pf import gpu_utils
            import torch

            device = gpu_utils.resolve_device(filt.config.gpu_device)
            result = (
                model.log_likelihood_torch(
                    station.spectrum_vb,
                    torch.as_tensor(total, dtype=torch.float64, device=device),
                    torch.as_tensor(
                        uncollided,
                        dtype=torch.float64,
                        device=device,
                    ),
                    torch.as_tensor(
                        features,
                        dtype=torch.float64,
                        device=device,
                    ),
                    station.live_times_s,
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            )
        else:
            result = np.asarray(
                model.log_likelihood_numpy(
                    station.spectrum_vb,
                    total,
                    uncollided,
                    features,
                    station.live_times_s,
                ),
                dtype=np.float64,
            )
        expected_shape = (int(total.shape[0]),)
        if np.asarray(result).shape != expected_shape:
            raise RuntimeError(
                "Full-spectrum conditional likelihood must return one value "
                "per candidate row."
            )
        if np.any(np.isnan(result)) or np.any(np.isposinf(result)):
            raise RuntimeError(
                "Full-spectrum conditional likelihood contains NaN or "
                "positive infinity."
            )
        return np.asarray(result, dtype=np.float64)

    def _joint_history_log_likelihood_numpy(
        self,
        *,
        filt: IsotopeParticleFilter,
        stations: Sequence[JointStationObservation],
        total_nvsl: NDArray[np.float64],
        uncollided_nvsl: NDArray[np.float64],
        features_nvslf: NDArray[np.float64],
        target_beta: float,
    ) -> NDArray[np.float64]:
        """Evaluate station-independent latent blocks on one batched action axis."""
        beta = float(target_beta)
        if not np.isfinite(beta) or not 0.0 <= beta <= 1.0:
            raise ValueError("Joint history target_beta must lie in [0, 1].")
        model = self._full_spectrum_model()
        total = np.asarray(total_nvsl, dtype=np.float64)
        uncollided = np.asarray(uncollided_nvsl, dtype=np.float64)
        features = np.asarray(features_nvslf, dtype=np.float64)
        feature_count = len(tuple(model.transport_feature_order))
        total_views = sum(int(station.fe_indices.size) for station in stations)
        if (
            total.ndim != 4
            or uncollided.shape != total.shape
            or features.shape != total.shape + (feature_count,)
            or total.shape[0] <= 0
            or total.shape[1] != total_views
        ):
            raise ValueError(
                "Joint-history transport arrays must align with every station "
                "view and configured transport feature."
            )
        particle_count = int(total.shape[0])
        result = np.zeros(particle_count, dtype=np.float64)
        grouped: dict[
            tuple[int, bytes],
            list[tuple[JointStationObservation, int, int, float]],
        ] = {}
        view_start = 0
        for station_index, station in enumerate(stations):
            view_count = int(station.fe_indices.size)
            view_stop = view_start + view_count
            power = beta if station_index == len(stations) - 1 else 1.0
            live_times = np.ascontiguousarray(
                station.live_times_s,
                dtype=np.float64,
            )
            if live_times.shape != (view_count,):
                raise ValueError(
                    "Joint-history station live times must align with views."
                )
            if power > 0.0:
                key = (view_count, live_times.tobytes(order="C"))
                grouped.setdefault(key, []).append(
                    (station, view_start, view_stop, power)
                )
            view_start = view_stop
        if view_start != total_views:
            raise ValueError(
                "Full-spectrum transport views differ from station history."
            )
        for entries in grouped.values():
            view_count = int(entries[0][2] - entries[0][1])
            first_start = int(entries[0][1])
            last_stop = int(entries[-1][2])
            contiguous = all(
                int(entry[1]) == first_start + index * view_count
                and int(entry[2]) == first_start + (index + 1) * view_count
                for index, entry in enumerate(entries)
            )

            def _station_action_axis(
                values: NDArray[np.float64],
            ) -> NDArray[np.float64]:
                """Return station x particle x view without scalar station work."""
                trailing_shape = tuple(values.shape[2:])
                if contiguous:
                    block = values[:, first_start:last_stop, ...]
                    reshaped = block.reshape(
                        particle_count,
                        len(entries),
                        view_count,
                        *trailing_shape,
                    )
                    return np.moveaxis(reshaped, 1, 0)
                return np.stack(
                    [
                        values[:, int(entry[1]) : int(entry[2]), ...]
                        for entry in entries
                    ],
                    axis=0,
                )

            observed = np.stack(
                [
                    np.asarray(entry[0].spectrum_vb, dtype=np.float64)
                    for entry in entries
                ],
                axis=0,
            )[:, None, :, :]
            total_group = _station_action_axis(total)
            uncollided_group = _station_action_axis(uncollided)
            feature_group = _station_action_axis(features)
            live_times = np.asarray(
                entries[0][0].live_times_s,
                dtype=np.float64,
            )
            action_chunk_size = min(
                len(entries),
                JOINT_HISTORY_STATION_ACTION_BATCH_SIZE,
            )
            if filt._can_use_gpu():
                from pf import gpu_utils
                import torch

                device = gpu_utils.resolve_device(filt.config.gpu_device)
                group_ll = (
                    model.cross_log_likelihood_torch(
                        observed,
                        torch.as_tensor(
                            total_group,
                            dtype=torch.float64,
                            device=device,
                        ),
                        torch.as_tensor(
                            uncollided_group,
                            dtype=torch.float64,
                            device=device,
                        ),
                        torch.as_tensor(
                            feature_group,
                            dtype=torch.float64,
                            device=device,
                        ),
                        live_times,
                        action_chunk_size=action_chunk_size,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=False)
                )
            else:
                group_ll = np.asarray(
                    model.cross_log_likelihood_numpy(
                        observed,
                        total_group,
                        uncollided_group,
                        feature_group,
                        live_times,
                        action_chunk_size=action_chunk_size,
                    ),
                    dtype=np.float64,
                )
            expected_shape = (len(entries), 1, particle_count)
            if (
                group_ll.shape != expected_shape
                or np.any(np.isnan(group_ll))
                or np.any(np.isposinf(group_ll))
            ):
                raise RuntimeError(
                    "Batched station-history likelihood returned invalid "
                    "action/sample/state values."
                )
            powers = np.asarray(
                [entry[3] for entry in entries],
                dtype=np.float64,
            )
            result += np.sum(
                powers[:, None] * group_ll[:, 0, :],
                axis=0,
            )
        if np.any(np.isnan(result)) or np.any(np.isposinf(result)):
            raise RuntimeError(
                "Joint history conditional likelihood is numerically invalid."
            )
        return result

    @staticmethod
    def _joint_birth_proposal_station_digest(
        *,
        filt: IsotopeParticleFilter,
        station: JointStationObservation,
        strength_grid: NDArray[np.float64],
    ) -> str:
        """Hash every immutable input to one station proposal-score grid."""
        digest = hashlib.sha256(
            b"joint_full_spectrum_birth_proposal_station_v1\0"
        )
        for text in (
            str(filt.isotope),
            str(station.generative_contract_hash_sha256),
            str(filt.structural_rj_surface_atlas_sha256),
            str(id(filt.continuous_kernel)),
        ):
            encoded = text.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        digest.update(
            np.asarray(
                [station.pose_idx, station.station_sequence_id],
                dtype="<i8",
            ).tobytes()
        )
        for values in (
            station.spectrum_vb,
            station.energy_axis_keV,
            station.detector_position_xyz_m,
            station.fe_indices,
            station.pb_indices,
            station.live_times_s,
            strength_grid,
        ):
            array = np.ascontiguousarray(values)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
            digest.update(array.tobytes(order="C"))
        return digest.hexdigest()

    @staticmethod
    def _joint_station_structural_geometry(
        station: JointStationObservation,
    ) -> StructuralGeometryBatch:
        """Return the geometry-only carrier for one immutable station."""
        view_count = int(station.fe_indices.size)
        return StructuralGeometryBatch(
            detector_positions=np.repeat(
                np.asarray(
                    station.detector_position_xyz_m,
                    dtype=np.float64,
                ).reshape(1, 3),
                view_count,
                axis=0,
            ),
            fe_indices=np.asarray(station.fe_indices, dtype=np.int64),
            pb_indices=np.asarray(station.pb_indices, dtype=np.int64),
            live_times=np.asarray(station.live_times_s, dtype=np.float64),
            station_sequence_ids=np.full(
                view_count,
                int(station.station_sequence_id),
                dtype=np.int64,
            ),
        )

    @property
    def joint_birth_proposal_cache_bytes(self) -> int:
        """Return bytes occupied by cached chart-by-strength score arrays."""
        return int(
            sum(
                int(values.nbytes)
                for values in (
                    self._joint_birth_proposal_station_score_cache.values()
                )
            )
        )

    def _store_joint_birth_proposal_station_scores(
        self,
        cache_key: tuple[str, str],
        score_grid: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Store one immutable score grid under a strict LRU memory bound."""
        scores = np.ascontiguousarray(score_grid, dtype=np.float64)
        maximum_bytes = int(
            self.pf_config.structural_rj_proposal_score_cache_max_bytes
        )
        if int(scores.nbytes) > maximum_bytes:
            raise MemoryError(
                "One full-spectrum birth-proposal score grid exceeds "
                "structural_rj_proposal_score_cache_max_bytes."
            )
        scores.setflags(write=False)
        self._joint_birth_proposal_station_score_cache[cache_key] = scores
        self._joint_birth_proposal_station_score_cache_order = [
            key
            for key in self._joint_birth_proposal_station_score_cache_order
            if key != cache_key
        ]
        self._joint_birth_proposal_station_score_cache_order.append(cache_key)
        while self.joint_birth_proposal_cache_bytes > maximum_bytes:
            oldest = (
                self._joint_birth_proposal_station_score_cache_order.pop(0)
            )
            self._joint_birth_proposal_station_score_cache.pop(oldest, None)
        if cache_key not in self._joint_birth_proposal_station_score_cache:
            raise RuntimeError(
                "Birth-proposal LRU evicted the score grid being inserted."
            )
        return scores

    def _joint_station_birth_proposal_score_grid(
        self,
        *,
        filt: IsotopeParticleFilter,
        station: JointStationObservation,
        chart_centers_xyz: NDArray[np.float64],
        strength_grid: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return one cached proposal-only chart-by-strength score grid."""
        centers = np.asarray(
            chart_centers_xyz,
            dtype=np.float64,
        ).reshape(-1, 3)
        atlas = filt._structural_rj_surface_atlas
        if atlas is None or int(centers.shape[0]) != int(atlas.chart_count):
            raise ValueError(
                "Birth-proposal centers must contain the authoritative atlas "
                "charts in chart-ID order."
            )
        strengths = np.asarray(
            strength_grid,
            dtype=np.float64,
        ).reshape(-1)
        cache_key = (
            str(filt.isotope),
            self._joint_birth_proposal_station_digest(
                filt=filt,
                station=station,
                strength_grid=strengths,
            ),
        )
        cached = self._joint_birth_proposal_station_score_cache.get(
            cache_key
        )
        expected_shape = (int(centers.shape[0]), int(strengths.size))
        if cached is not None:
            if cached.shape != expected_shape:
                raise RuntimeError(
                    "Cached birth-proposal score grid has an invalid shape."
                )
            self.last_joint_birth_proposal_cache_hits += 1
            self._joint_birth_proposal_station_score_cache_order = [
                key
                for key in (
                    self._joint_birth_proposal_station_score_cache_order
                )
                if key != cache_key
            ]
            self._joint_birth_proposal_station_score_cache_order.append(
                cache_key
            )
            return cached

        self.last_joint_birth_proposal_cache_misses += 1
        model = self._full_spectrum_model()
        line_count = len(tuple(model.line_identity))
        feature_count = len(tuple(model.transport_feature_order))
        layout = self._joint_line_layout()
        global_columns, local_indices, branching_weights = layout[
            str(filt.isotope)
        ]
        target_line_mask = np.zeros(line_count, dtype=np.bool_)
        target_line_mask[global_columns] = True
        view_count = int(station.fe_indices.size)
        geometry = self._joint_station_structural_geometry(station)
        score_cg = np.empty(expected_shape, dtype=np.float64)
        batch_size = int(
            self.pf_config.structural_rj_proposal_chart_batch_size
        )
        for chart_start in range(0, int(centers.shape[0]), batch_size):
            chart_stop = min(
                chart_start + batch_size,
                int(centers.shape[0]),
            )
            batch_centers = centers[chart_start:chart_stop]
            components = (
                filt._continuous_rj_line_transport_component_columns(
                    geometry,
                    batch_centers,
                    local_indices,
                    chart_ids=np.arange(
                        chart_start,
                        chart_stop,
                        dtype=np.int64,
                    ),
                )
            )
            local_shape = (
                view_count,
                int(batch_centers.shape[0]),
                int(local_indices.size),
            )
            unit_total = np.asarray(
                components.total_kernel,
                dtype=np.float64,
            ).reshape(local_shape)
            unit_uncollided = np.asarray(
                components.uncollided_kernel,
                dtype=np.float64,
            ).reshape(local_shape)
            unit_features = np.stack(
                (
                    np.asarray(components.tau_fe, dtype=np.float64),
                    np.asarray(components.tau_pb, dtype=np.float64),
                    np.asarray(
                        components.tau_obstacle,
                        dtype=np.float64,
                    ),
                    np.asarray(components.distance_m, dtype=np.float64),
                ),
                axis=-1,
            ).reshape(local_shape + (feature_count,))
            batch_count = int(batch_centers.shape[0])
            candidate_count = batch_count * int(strengths.size)
            total = np.zeros(
                (candidate_count, view_count, 1, line_count),
                dtype=np.float64,
            )
            uncollided = np.zeros_like(total)
            features = np.zeros(
                (
                    candidate_count,
                    view_count,
                    1,
                    line_count,
                    feature_count,
                ),
                dtype=np.float64,
            )
            scale = (
                strengths[None, :, None, None]
                * branching_weights[None, None, None, :]
            )
            total_local = (
                np.transpose(unit_total, (1, 0, 2))[:, None, :, :]
                * scale
            ).reshape(
                candidate_count,
                view_count,
                int(local_indices.size),
            )
            uncollided_local = (
                np.transpose(unit_uncollided, (1, 0, 2))[:, None, :, :]
                * scale
            ).reshape(
                candidate_count,
                view_count,
                int(local_indices.size),
            )
            feature_local = np.broadcast_to(
                np.transpose(unit_features, (1, 0, 2, 3))[
                    :, None, :, :, :
                ],
                (
                    batch_count,
                    int(strengths.size),
                    view_count,
                    int(local_indices.size),
                    feature_count,
                ),
            ).reshape(
                candidate_count,
                view_count,
                int(local_indices.size),
                feature_count,
            )
            total[..., global_columns] = total_local[:, :, None, :]
            uncollided[..., global_columns] = (
                uncollided_local[:, :, None, :]
            )
            features[..., global_columns, :] = (
                feature_local[:, :, None, :, :]
            )
            if filt._can_use_gpu():
                from pf import gpu_utils
                import torch

                device = gpu_utils.resolve_device(filt.config.gpu_device)
                scores = model.birth_proposal_log_scores_torch(
                    station.spectrum_vb,
                    torch.as_tensor(
                        total,
                        dtype=torch.float64,
                        device=device,
                    ),
                    torch.as_tensor(
                        uncollided,
                        dtype=torch.float64,
                        device=device,
                    ),
                    torch.as_tensor(
                        features,
                        dtype=torch.float64,
                        device=device,
                    ),
                    station.live_times_s,
                    target_line_mask_l=torch.as_tensor(
                        target_line_mask,
                        dtype=torch.bool,
                        device=device,
                    ),
                )
                batch_scores = (
                    scores.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=False)
                )
            else:
                batch_scores = np.asarray(
                    model.birth_proposal_log_scores_numpy(
                        station.spectrum_vb,
                        total,
                        uncollided,
                        features,
                        station.live_times_s,
                        target_line_mask_l=target_line_mask,
                    ),
                    dtype=np.float64,
                )
            if batch_scores.shape != (candidate_count,) or np.any(
                ~np.isfinite(batch_scores)
            ):
                raise RuntimeError(
                    "Full-spectrum birth proposal returned invalid scores."
                )
            score_cg[chart_start:chart_stop, :] = batch_scores.reshape(
                batch_count,
                strengths.size,
            )
        return self._store_joint_birth_proposal_station_scores(
            cache_key,
            score_cg,
        )

    def _joint_structural_proposal_evaluator(
        self,
        *,
        filt: IsotopeParticleFilter,
        data: StructuralGeometryBatch,
        chart_centers_xyz: NDArray[np.float64],
        target_beta: float,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        bool,
    ]:
        """Build one cached state-independent full-spectrum birth proposal."""
        stations = self._active_joint_station_history
        if stations is None:
            raise RuntimeError(
                "Joint full-spectrum proposal target is not active."
            )
        self._validate_joint_structural_geometry(data, stations)
        centers = np.asarray(
            chart_centers_xyz,
            dtype=np.float64,
        ).reshape(-1, 3)
        chart_count = int(centers.shape[0])
        atlas = filt._structural_rj_surface_atlas
        if atlas is None or not np.array_equal(
            centers,
            np.asarray(
                atlas.geometry.centers_xyz,
                dtype=np.float64,
            ),
        ):
            raise ValueError(
                "Birth-proposal chart centers must be the immutable active "
                "continuous-surface atlas centers."
            )
        strength_grid = np.linspace(
            float(self.pf_config.strength_prior_min_cps_1m),
            float(self.pf_config.strength_prior_max_cps_1m),
            int(self.pf_config.structural_rj_strength_proposal_grid_size),
            dtype=np.float64,
        )
        midpoint = float(
            0.5
            * (
                self.pf_config.strength_prior_min_cps_1m
                + self.pf_config.strength_prior_max_cps_1m
            )
        )
        if chart_count == 0:
            return (
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                False,
            )
        total_views = sum(int(station.fe_indices.size) for station in stations)
        if data.row_count != total_views:
            raise ValueError(
                "Residual proposal geometry differs from joint station history."
            )
        beta = float(target_beta)
        has_active_likelihood = len(stations) > 1 or beta > 0.0
        if not has_active_likelihood:
            return (
                np.zeros(chart_count, dtype=np.float64),
                np.full(chart_count, midpoint, dtype=np.float64),
                False,
            )
        completed_count = len(self._joint_station_history)
        if (
            len(stations) != completed_count + 1
            or self._joint_birth_proposal_prefix_station_count
            != completed_count
        ):
            raise RuntimeError(
                "Birth-proposal prefix does not match the completed station "
                "history."
            )
        prefix = self._joint_birth_proposal_prefix_scores.get(
            str(filt.isotope)
        )
        expected_shape = (chart_count, strength_grid.size)
        if prefix is None:
            if completed_count:
                raise RuntimeError(
                    "Birth-proposal prefix is missing for a configured isotope."
                )
            score_cg = np.zeros(expected_shape, dtype=np.float64)
        else:
            score_cg = np.asarray(prefix, dtype=np.float64).copy()
            if score_cg.shape != expected_shape or np.any(
                ~np.isfinite(score_cg)
            ):
                raise RuntimeError(
                    "Birth-proposal prefix score grid is invalid."
                )
        if beta > 0.0:
            score_cg += beta * self._joint_station_birth_proposal_score_grid(
                filt=filt,
                station=stations[-1],
                chart_centers_xyz=centers,
                strength_grid=strength_grid,
            )
        best_grid_indices = np.argmax(score_cg, axis=1)
        best_scores = score_cg[
            np.arange(chart_count, dtype=np.int64),
            best_grid_indices,
        ]
        best_locations = strength_grid[best_grid_indices]
        maximum = float(np.max(best_scores))
        informative = bool(np.isfinite(maximum) and maximum > 1.0e-9)
        if not informative:
            return (
                np.zeros(chart_count, dtype=np.float64),
                np.full(chart_count, midpoint, dtype=np.float64),
                False,
            )
        alignment = np.exp(
            np.clip(best_scores - maximum, -745.0, 0.0)
        )
        return (
            np.asarray(alignment, dtype=np.float64),
            np.asarray(best_locations, dtype=np.float64),
            True,
        )

    def _promote_joint_birth_proposal_station(
        self,
        station: JointStationObservation,
    ) -> None:
        """Add one completed station to each isotope's exact score prefix.

        Birth proposal scores are additive over conditionally independent
        stations.  Keeping the full chart-by-strength sum for completed
        stations is mathematically identical to rescanning every old station,
        while preventing a bounded LRU from cyclically evicting and recomputing
        the entire history during every intermediate rejuvenation.
        """
        completed_count = len(self._joint_station_history)
        if self._joint_birth_proposal_prefix_station_count != completed_count:
            raise RuntimeError(
                "Birth-proposal prefix promotion is out of sequence."
            )
        strength_grid = np.linspace(
            float(self.pf_config.strength_prior_min_cps_1m),
            float(self.pf_config.strength_prior_max_cps_1m),
            int(self.pf_config.structural_rj_strength_proposal_grid_size),
            dtype=np.float64,
        )
        promoted: dict[str, NDArray[np.float64]] = {}
        for isotope in self.joint_isotope_order():
            filt = self.filters[isotope]
            atlas = filt._structural_rj_surface_atlas
            if atlas is None:
                raise RuntimeError(
                    "Birth-proposal prefix requires a continuous surface atlas."
                )
            station_scores = self._joint_station_birth_proposal_score_grid(
                filt=filt,
                station=station,
                chart_centers_xyz=np.asarray(
                    atlas.geometry.centers_xyz,
                    dtype=np.float64,
                ),
                strength_grid=strength_grid,
            )
            previous = self._joint_birth_proposal_prefix_scores.get(isotope)
            if previous is None:
                if completed_count:
                    raise RuntimeError(
                        "Birth-proposal prefix is missing for a configured "
                        "isotope."
                    )
                combined = np.asarray(
                    station_scores,
                    dtype=np.float64,
                ).copy()
            else:
                if previous.shape != station_scores.shape:
                    raise RuntimeError(
                        "Birth-proposal prefix and station grids disagree."
                    )
                combined = (
                    np.asarray(previous, dtype=np.float64)
                    + np.asarray(station_scores, dtype=np.float64)
                )
            if np.any(~np.isfinite(combined)):
                raise RuntimeError(
                    "Birth-proposal prefix contains non-finite scores."
                )
            combined = np.ascontiguousarray(combined, dtype=np.float64)
            combined.setflags(write=False)
            promoted[isotope] = combined
        self._joint_birth_proposal_prefix_scores = promoted
        self._joint_birth_proposal_prefix_station_count = completed_count + 1
        self._joint_birth_proposal_station_score_cache.clear()
        self._joint_birth_proposal_station_score_cache_order.clear()

    def _joint_structural_target_evaluator(
        self,
        *,
        filt: IsotopeParticleFilter,
        data: StructuralGeometryBatch,
        positions_pks: NDArray[np.float64],
        chart_ids_pk: NDArray[np.int64],
        strengths_pk: NDArray[np.float64],
        particle_indices: NDArray[np.int64],
        target_beta: float,
        tempering_start_row: int | None,
    ) -> NDArray[np.float64]:
        """Evaluate a conditional isotope proposal under the full joint target."""
        stations = self._active_joint_station_history
        cache = self._joint_structural_transport_cache
        if stations is None or cache is None:
            raise RuntimeError("Joint structural target is not active.")
        self._validate_joint_structural_geometry(data, stations)
        order = self.joint_isotope_order()
        if filt.isotope not in order:
            raise ValueError("Conditional RJ filter is not a joint isotope.")
        indices = np.asarray(particle_indices, dtype=np.int64).reshape(-1)
        positions = np.asarray(positions_pks, dtype=np.float64)
        raw_chart_ids = np.asarray(chart_ids_pk)
        strengths = np.asarray(strengths_pk, dtype=np.float64)
        total_views = sum(int(station.fe_indices.size) for station in stations)
        if (
            positions.ndim != 3
            or positions.shape[0] != indices.size
            or positions.shape[2] != 3
            or strengths.shape != positions.shape[:2]
            or not np.issubdtype(raw_chart_ids.dtype, np.integer)
            or raw_chart_ids.shape != strengths.shape
        ):
            raise ValueError(
                "Conditional isotope candidates must be aligned surface states."
            )
        chart_ids = np.asarray(raw_chart_ids, dtype=np.int64)
        model = self._full_spectrum_model()
        cached_total, cached_uncollided, cached_features = cache
        line_count = len(tuple(model.line_identity))
        feature_count = len(tuple(model.transport_feature_order))
        slots_per_isotope = int(filt.config.max_sources)
        total_slot_count = slots_per_isotope * len(order)
        if (
            cached_total.shape[1:]
            != (total_views, total_slot_count, line_count)
            or cached_uncollided.shape != cached_total.shape
            or cached_features.shape
            != cached_total.shape + (feature_count,)
            or np.any(indices < 0)
            or np.any(indices >= cached_total.shape[0])
        ):
            raise RuntimeError(
                "Joint structural transport cache is misaligned."
            )
        total = np.asarray(cached_total[indices], dtype=np.float64).copy()
        uncollided = np.asarray(
            cached_uncollided[indices],
            dtype=np.float64,
        ).copy()
        features = np.asarray(
            cached_features[indices],
            dtype=np.float64,
        ).copy()
        layout = self._joint_line_layout()
        global_columns, local_indices, branching_weights = layout[
            str(filt.isotope)
        ]
        components = filt._continuous_rj_line_transport_component_columns(
            data,
            positions.reshape(-1, 3),
            local_indices,
            chart_ids=chart_ids.reshape(-1),
        )
        local_shape = (
            total_views,
            indices.size,
            int(positions.shape[1]),
            int(local_indices.size),
        )
        candidate_total = np.asarray(
            components.total_kernel,
            dtype=np.float64,
        ).reshape(local_shape)
        candidate_uncollided = np.asarray(
            components.uncollided_kernel,
            dtype=np.float64,
        ).reshape(local_shape)
        candidate_features = np.stack(
            (
                np.asarray(components.tau_fe, dtype=np.float64),
                np.asarray(components.tau_pb, dtype=np.float64),
                np.asarray(components.tau_obstacle, dtype=np.float64),
                np.asarray(components.distance_m, dtype=np.float64),
            ),
            axis=-1,
        ).reshape(local_shape + (feature_count,))
        scale = (
            strengths[None, :, :, None]
            * branching_weights.reshape(1, 1, 1, -1)
        )
        candidate_total = np.transpose(
            candidate_total * scale,
            (1, 0, 2, 3),
        )
        candidate_uncollided = np.transpose(
            candidate_uncollided * scale,
            (1, 0, 2, 3),
        )
        candidate_features = np.transpose(
            candidate_features,
            (1, 0, 2, 3, 4),
        )
        isotope_index = order.index(str(filt.isotope))
        slot_start = isotope_index * slots_per_isotope
        slot_stop = slot_start + slots_per_isotope
        total[:, :, slot_start:slot_stop, :] = 0.0
        uncollided[:, :, slot_start:slot_stop, :] = 0.0
        features[:, :, slot_start:slot_stop, :, :] = 0.0
        cardinality = int(positions.shape[1])
        if cardinality > slots_per_isotope:
            raise ValueError(
                "Conditional candidate cardinality exceeds its source slots."
            )
        if cardinality:
            target_slots = slice(slot_start, slot_start + cardinality)
            total_subset = total[:, :, target_slots, :]
            uncollided_subset = uncollided[:, :, target_slots, :]
            feature_subset = features[:, :, target_slots, :, :]
            total_subset[..., global_columns] = candidate_total
            uncollided_subset[..., global_columns] = candidate_uncollided
            feature_subset[..., global_columns, :] = candidate_features
            total[:, :, target_slots, :] = total_subset
            uncollided[:, :, target_slots, :] = uncollided_subset
            features[:, :, target_slots, :, :] = feature_subset
        beta = float(target_beta)
        if not np.isfinite(beta) or not 0.0 <= beta <= 1.0:
            raise ValueError("Joint structural target_beta must lie in [0, 1].")
        expected_start_row = total_views - int(stations[-1].fe_indices.size)
        if (
            tempering_start_row is not None
            and int(tempering_start_row) != expected_start_row
        ):
            raise ValueError(
                "Joint structural tempering must begin at the newest station."
            )
        if data.row_count != total_views:
            raise ValueError(
                "Conditional isotope evidence geometry differs from joint history."
            )
        return self._joint_history_log_likelihood_numpy(
            filt=filt,
            stations=stations,
            total_nvsl=total,
            uncollided_nvsl=uncollided,
            features_nvslf=features,
            target_beta=beta,
        )

    def _resample_joint_particles(
        self,
        normalized_log_weights: NDArray[np.float64],
    ) -> NDArray[np.int64]:
        """Apply one systematic ancestor vector to every isotope state row."""
        log_weights = np.asarray(
            normalized_log_weights,
            dtype=np.float64,
        ).reshape(-1)
        self._assign_joint_log_weights(log_weights)
        raw_indices = np.asarray(
            systematic_resample(
                log_weights,
                rng=self._joint_random_generator,
            )
        )
        if (
            raw_indices.shape != log_weights.shape
            or not np.issubdtype(raw_indices.dtype, np.integer)
            or np.any(raw_indices < 0)
            or np.any(raw_indices >= log_weights.size)
        ):
            raise RuntimeError(
                "Joint systematic resampling returned invalid ancestor indices."
            )
        indices = np.asarray(
            raw_indices,
            dtype=np.int64,
        )
        uniform_log_weight = float(-np.log(max(indices.size, 1)))
        reference_particles = self.filters[
            self.joint_isotope_order()[0]
        ].continuous_particles
        parent_identities = tuple(
            particle.joint_row_identity for particle in reference_particles
        )
        if not all(
            isinstance(identity, JointRowIdentity)
            for identity in parent_identities
        ):
            raise RuntimeError(
                "Joint resampling requires authenticated parent row identities."
            )
        new_identities = tuple(
            parent_identities[int(index)].resampled_child(ordinal=row)
            for row, index in enumerate(indices.tolist())
        )
        new_particles_by_isotope: dict[str, list[IsotopeParticle]] = {}
        for isotope in self.joint_isotope_order():
            filt = self.filters[isotope]
            previous = filt.continuous_particles
            new_particles_by_isotope[isotope] = [
                IsotopeParticle(
                    state=previous[int(index)].state.copy(),
                    log_weight=uniform_log_weight,
                    joint_row_identity=new_identities[row],
                )
                for row, index in enumerate(indices.tolist())
            ]
        for isotope in self.joint_isotope_order():
            filt = self.filters[isotope]
            filt.continuous_particles = new_particles_by_isotope[isotope]
            filt.last_resample_count += 1
            filt._resample_count_in_observation += 1
        self._joint_row_generation = new_identities[0].generation
        self.last_joint_resample_indices = np.asarray(
            indices,
            dtype=np.int64,
        )
        self._assert_joint_particle_alignment()
        return self.last_joint_resample_indices

    def _joint_rejuvenate(
        self,
        stations: Sequence[JointStationObservation],
        *,
        target_beta: float,
    ) -> None:
        """Apply sequential conditional exact-RJ sweeps under one joint target."""
        active = tuple(stations)
        if not active:
            return
        self._active_joint_station_history = active
        newest_start = sum(
            int(station.fe_indices.size) for station in active[:-1]
        )
        try:
            self._refresh_joint_structural_transport_cache(active)
            isotope_order = self.joint_isotope_order()
            for isotope_index, isotope in enumerate(isotope_order):
                filt = self.filters[isotope]
                evidence = self._joint_history_structural_geometry(
                    isotope,
                    active,
                )
                self._validate_joint_structural_geometry(evidence, active)
                self._active_joint_structural_geometry = evidence
                try:
                    filt.apply_structural_moves(
                        evidence,
                        target_beta=float(target_beta),
                        tempering_start_row=int(newest_start),
                    )
                finally:
                    self._active_joint_structural_geometry = None
                self._assert_joint_particle_alignment()
                if isotope_index + 1 < len(isotope_order):
                    self._refresh_joint_structural_transport_cache_isotope(
                        active,
                        isotope,
                    )
        finally:
            self._active_joint_structural_geometry = None
            self._joint_structural_transport_cache = None
            self._active_joint_station_history = None

    def _joint_tempered_station_update(
        self,
        station: JointStationObservation,
    ) -> None:
        """Assimilate one station with common weights and aligned SMC ancestors."""
        import torch

        all_stations = tuple((*self._joint_station_history, station))
        for filt in self.filters.values():
            filt.reset_step_stats()
        self.last_joint_resample_indices = np.empty(0, dtype=np.int64)
        reference_filter = self.filters[self.joint_isotope_order()[0]]
        common_log_weights = self._assert_joint_particle_alignment()
        likelihood = self._joint_station_log_likelihood_torch(station)
        device = likelihood.device
        log_weights = torch.as_tensor(
            common_log_weights,
            dtype=torch.float64,
            device=device,
        )
        log_weights = reference_filter._normalized_log_weights_torch(log_weights)
        self._assign_joint_log_weights(
            log_weights.detach().cpu().numpy()
        )
        initial_ess = reference_filter._ess_from_logw_torch(log_weights)
        target_ess = float(self.pf_config.target_ess_ratio) * int(
            log_weights.numel()
        )
        particle_count = int(log_weights.numel())
        station_ancestor_ids = np.arange(particle_count, dtype=np.int64)
        if self._joint_cumulative_lineage_ids is None:
            if self._joint_station_history:
                raise RuntimeError(
                    "Cumulative PF lineage is missing after prior station updates."
                )
            self._joint_cumulative_lineage_ids = np.arange(
                particle_count,
                dtype=np.int64,
            )
        cumulative_lineage_ids = np.asarray(
            self._joint_cumulative_lineage_ids,
            dtype=np.int64,
        ).reshape(-1).copy()
        if cumulative_lineage_ids.shape != (particle_count,):
            raise RuntimeError(
                "Cumulative PF lineage does not match aligned particle rows."
            )
        beta_total = 0.0
        resamples = 0
        steps: list[dict[str, float]] = []

        def _likelihood() -> "torch.Tensor":
            """Return newest-station likelihood for the current aligned states."""
            return self._joint_station_log_likelihood_torch(station).to(
                device=device,
                dtype=torch.float64,
            )

        likelihood = likelihood.to(device=device, dtype=torch.float64)
        if initial_ess <= target_ess + 1.0e-9:
            indices = self._resample_joint_particles(
                log_weights.detach().cpu().numpy()
            )
            station_ancestor_ids = station_ancestor_ids[indices]
            cumulative_lineage_ids = cumulative_lineage_ids[indices]
            resamples += 1
            self._joint_rejuvenate(all_stations, target_beta=0.0)
            likelihood = _likelihood()
            log_weights = torch.full(
                (int(likelihood.numel()),),
                -math.log(max(int(likelihood.numel()), 1)),
                dtype=torch.float64,
                device=device,
            )
        max_steps = int(self.pf_config.max_temper_steps)
        while beta_total < 1.0 - 1.0e-12:
            if len(steps) >= max_steps:
                raise RuntimeError(
                    "Joint SMC reached max_temper_steps before beta=1."
                )
            try:
                delta_beta, proposed_log_weights, ess = (
                    reference_filter._select_delta_beta(
                        logw_prev=log_weights,
                        ll_t=likelihood,
                        remaining=1.0 - beta_total,
                        target_ess=target_ess,
                    )
                )
            except TemperingIncrementRequiresRejuvenation:
                current_ess = reference_filter._ess_from_logw_torch(
                    log_weights
                )
                resampled = current_ess < particle_count - 1.0e-9
                if resampled:
                    indices = self._resample_joint_particles(
                        log_weights.detach().cpu().numpy()
                    )
                    station_ancestor_ids = station_ancestor_ids[indices]
                    cumulative_lineage_ids = cumulative_lineage_ids[indices]
                    resamples += 1
                    log_weights = torch.full(
                        (particle_count,),
                        -math.log(particle_count),
                        dtype=torch.float64,
                        device=device,
                    )
                self._joint_rejuvenate(
                    all_stations,
                    target_beta=float(beta_total),
                )
                likelihood = _likelihood()
                recovery_step = {
                    "beta_total": float(beta_total),
                    "delta_beta": 0.0,
                    "ess": float(current_ess),
                    "resampled": float(resampled),
                    "recovery_rejuvenation": 1.0,
                    "station_unique_ancestors": float(
                        np.unique(station_ancestor_ids).size
                    ),
                    "cumulative_unique_ancestors": float(
                        np.unique(cumulative_lineage_ids).size
                    ),
                }
                steps.append(recovery_step)
                continue
            log_weights = proposed_log_weights
            self._assign_joint_log_weights(
                log_weights.detach().cpu().numpy()
            )
            beta_total += float(delta_beta)
            step = {
                "beta_total": float(beta_total),
                "delta_beta": float(delta_beta),
                "ess": float(ess),
                "resampled": 0.0,
                "recovery_rejuvenation": 0.0,
                "station_unique_ancestors": float(
                    np.unique(station_ancestor_ids).size
                ),
                "cumulative_unique_ancestors": float(
                    np.unique(cumulative_lineage_ids).size
                ),
            }
            steps.append(step)
            if beta_total >= 1.0 - 1.0e-12:
                beta_total = 1.0
                break
            indices = self._resample_joint_particles(
                log_weights.detach().cpu().numpy()
            )
            station_ancestor_ids = station_ancestor_ids[indices]
            cumulative_lineage_ids = cumulative_lineage_ids[indices]
            resamples += 1
            step["resampled"] = 1.0
            step["station_unique_ancestors"] = float(
                np.unique(station_ancestor_ids).size
            )
            step["cumulative_unique_ancestors"] = float(
                np.unique(cumulative_lineage_ids).size
            )
            self._joint_rejuvenate(
                all_stations,
                target_beta=float(beta_total),
            )
            likelihood = _likelihood()
            log_weights = torch.full(
                (int(likelihood.numel()),),
                -math.log(max(int(likelihood.numel()), 1)),
                dtype=torch.float64,
                device=device,
            )
        self._joint_rejuvenate(all_stations, target_beta=1.0)
        normalized = self._strict_joint_particle_weights()
        final_ess = 1.0 / float(np.sum(normalized**2))
        if final_ess + 1.0e-9 < target_ess:
            raise RuntimeError(
                "Completed joint tempering did not preserve target ESS."
            )
        station_unique_ancestors = int(
            np.unique(station_ancestor_ids).size
        )
        cumulative_unique_ancestors = int(
            np.unique(cumulative_lineage_ids).size
        )
        self._joint_cumulative_lineage_ids = cumulative_lineage_ids
        self.last_joint_temper_steps = steps
        self.last_joint_station_unique_ancestor_count = (
            station_unique_ancestors
        )
        self.last_joint_cumulative_unique_ancestor_count = (
            cumulative_unique_ancestors
        )
        # Backward-compatible field now has the conservative cumulative
        # meaning; station-local ancestry is reported separately.
        self.last_joint_unique_ancestor_count = cumulative_unique_ancestors
        for filt in self.filters.values():
            filt.last_temper_steps = [dict(step) for step in steps]
            filt.last_temper_resample_count = int(resamples)
            filt.last_temper_min_ess = float(
                min((step["ess"] for step in steps), default=final_ess)
            )
            filt.last_station_unique_ancestor_count = (
                station_unique_ancestors
            )
            filt.last_cumulative_unique_ancestor_count = (
                cumulative_unique_ancestors
            )
            filt.last_unique_ancestor_count = cumulative_unique_ancestors
            filt.last_ess_pre = float(initial_ess)
            filt.last_ess = float(final_ess)
            filt.last_ess_post = float(final_ess)
            filt.last_resample_ess = bool(resamples)
        self._promote_joint_birth_proposal_station(station)
        self._joint_station_history.append(station)
        self._assert_joint_particle_alignment()

    def configured_isotope_response_counts(
        self,
        isotope: str,
        data: StructuralGeometryBatch,
        source_positions: NDArray[np.float64],
        strengths: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Return batched responses from the configured isotope PF kernel.

        Measurement rows and source positions are evaluated by the same continuous
        transport, obstacle, aperture, and shield model used by the PF.
        Candidate positions remain batched and particle state is not read.
        """
        isotope_key = str(isotope)
        if isotope_key not in self.configured_isotope_order():
            raise KeyError(f"Isotope {isotope_key!r} is not configured.")
        filt = self.filters.get(isotope_key)
        if filt is None:
            raise RuntimeError(
                "Pure PF isotope filters must be initialized before response "
                "diagnostics are evaluated."
            )
        positions = np.asarray(source_positions, dtype=float).reshape(-1, 3)
        if strengths is None:
            strength_values = np.ones(positions.shape[0], dtype=float)
        else:
            strength_values = np.asarray(strengths, dtype=float).reshape(-1)
        if strength_values.size != positions.shape[0]:
            raise ValueError("strengths must contain one value per source position.")
        return self._cached_expected_counts_for_kernel(
            kernel=filt.continuous_kernel,
            isotope=isotope_key,
            data=data,
            sources=positions,
            strengths=strength_values,
        )

    def planning_joint_particles(
        self,
        max_particles: int | None = None,
        method: str | None = None,
        rng: np.random.Generator | None = None,
    ) -> JointPlanningParticles:
        """Return one common-index numeric snapshot of the joint posterior."""
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        weights = self._strict_joint_particle_weights()
        particle_count = int(weights.size)
        if max_particles is None:
            max_particles = self.pf_config.planning_particles
        method = str(method or self.pf_config.planning_method)
        if (
            max_particles is None
            or int(max_particles) <= 0
            or int(max_particles) >= particle_count
        ):
            indices = np.arange(particle_count, dtype=np.int64)
            selected_weights = weights.copy()
        elif method == "top_weight":
            indices = np.argsort(weights)[::-1][: int(max_particles)].astype(
                np.int64,
                copy=False,
            )
            selected_weights = weights[indices]
            selected_weights /= float(np.sum(selected_weights))
        elif method == "resample":
            if rng is not None and not isinstance(rng, np.random.Generator):
                raise TypeError("rng must be a numpy.random.Generator.")
            if rng is None:
                rng = self._named_planning_rng(
                    "joint_particle_subset",
                    int(max_particles),
                )
            indices = np.asarray(
                rng.choice(
                    particle_count,
                    size=int(max_particles),
                    replace=True,
                    p=weights,
                ),
                dtype=np.int64,
            )
            selected_weights = np.full(
                int(max_particles),
                1.0 / float(max_particles),
                dtype=np.float64,
            )
        else:
            raise ValueError(
                f"Unknown joint planning particle selection method: {method}"
            )
        max_sources = int(self.pf_config.max_sources or 0)
        positions_by_isotope: Dict[str, NDArray[np.float64]] = {}
        chart_ids_by_isotope: Dict[str, NDArray[np.int64]] = {}
        surface_uv_by_isotope: Dict[str, NDArray[np.float64]] = {}
        strengths_by_isotope: Dict[str, NDArray[np.float64]] = {}
        masks_by_isotope: Dict[str, NDArray[np.bool_]] = {}
        for isotope in self.joint_isotope_order():
            filt = self.filters[isotope]
            (
                all_positions,
                all_strengths,
                all_mask,
                all_chart_ids,
                all_surface_uv,
            ) = (
                filt._packed_continuous_surface_state_arrays()
            )
            selected_positions = all_positions[indices]
            selected_strengths = all_strengths[indices]
            selected_mask = all_mask[indices]
            selected_chart_ids = all_chart_ids[indices]
            selected_surface_uv = all_surface_uv[indices]
            current_slots = int(selected_positions.shape[1])
            if current_slots > max_sources:
                raise RuntimeError(
                    "Joint planning state exceeds configured max_sources."
                )
            positions = np.zeros(
                (indices.size, max_sources, 3),
                dtype=np.float64,
            )
            chart_ids = np.zeros(
                (indices.size, max_sources),
                dtype=np.int64,
            )
            surface_uv = np.zeros(
                (indices.size, max_sources, 2),
                dtype=np.float64,
            )
            strengths = np.zeros(
                (indices.size, max_sources),
                dtype=np.float64,
            )
            mask = np.zeros(
                (indices.size, max_sources),
                dtype=bool,
            )
            if current_slots:
                positions[:, :current_slots, :] = selected_positions
                chart_ids[:, :current_slots] = selected_chart_ids
                surface_uv[:, :current_slots, :] = selected_surface_uv
                strengths[:, :current_slots] = selected_strengths
                mask[:, :current_slots] = selected_mask
            positions_by_isotope[isotope] = positions
            chart_ids_by_isotope[isotope] = chart_ids
            surface_uv_by_isotope[isotope] = surface_uv
            strengths_by_isotope[isotope] = strengths
            masks_by_isotope[isotope] = mask
        return JointPlanningParticles(
            isotope_order=self.joint_isotope_order(),
            weights_n=np.ascontiguousarray(selected_weights),
            positions_nk3_by_isotope=positions_by_isotope,
            surface_chart_ids_nk_by_isotope=chart_ids_by_isotope,
            surface_uv_nk2_by_isotope=surface_uv_by_isotope,
            strengths_nk_by_isotope=strengths_by_isotope,
            source_mask_nk_by_isotope=masks_by_isotope,
            original_particle_indices=np.ascontiguousarray(indices),
        )

    def planning_particles(
        self,
        max_particles: int | None = None,
        method: str | None = None,
        rng: np.random.Generator | None = None,
    ) -> Dict[str, Tuple[List[IsotopeState], NDArray[np.float64]]]:
        """
        Select per-isotope particle subsets for orientation evaluation.

        Args:
            max_particles: cap on particles per isotope; None uses config default.
            method: "top_weight" or "resample"; None uses config default.
            rng: optional RNG for resampling.
        """
        joint = self.planning_joint_particles(
            max_particles=max_particles,
            method=method,
            rng=rng,
        )
        return {
            isotope: (
                [
                    self.filters[isotope].continuous_particles[
                        int(index)
                    ].state.copy()
                    for index in joint.original_particle_indices
                ],
                joint.weights_n.copy(),
            )
            for isotope in joint.isotope_order
        }

    def add_measurement_pose(
        self, pose: NDArray[np.float64], reset_filters: bool = False
    ) -> None:
        """Register a pose without erasing posterior state unless requested."""
        if reset_filters and (
            self._joint_station_history or self.measurements
        ):
            raise RuntimeError(
                "Pure PF cannot reset particles after observations have "
                "entered the posterior."
            )
        self.poses.append(np.asarray(pose, dtype=float))
        # Rebuild lazily on the next access.
        self.kernel_cache = None
        if reset_filters:
            self.filters = {}
            self._joint_particles_initialized = False
            self._joint_row_identity_root_sha256 = None
            self._joint_row_generation = None
            self._joint_cumulative_lineage_ids = None
            self._joint_birth_proposal_prefix_scores = {}
            self._joint_birth_proposal_prefix_station_count = 0

    def _registered_detector_position_xyz(
        self,
        pose_idx: int,
    ) -> tuple[float, float, float]:
        """Return the canonical registered detector position for a pose index."""
        resolved_pose_idx = _strict_nonnegative_integer(
            pose_idx,
            name="pose_idx",
        )
        if resolved_pose_idx >= len(self.poses):
            raise IndexError(
                "pose_idx lies outside the registered measurement poses."
            )
        position = np.asarray(
            self.poses[resolved_pose_idx],
            dtype=float,
        ).reshape(-1)
        if position.size != 3 or not np.all(np.isfinite(position)):
            raise ValueError(
                "Registered measurement poses must contain three finite coordinates."
            )
        return tuple(float(value) for value in position)

    def update_spectrum_station(
        self,
        records: Sequence[Sequence[object]],
        *,
        pose_idx: int,
        generative_contract_hash_sha256: str,
    ) -> None:
        """Assimilate one shield program through the sole spectrum likelihood."""
        if not records:
            raise ValueError(
                "A spectrum station must contain at least one shield view."
            )
        sequence_start = time.perf_counter()
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        self._assert_joint_particle_alignment()
        station_sequence_id = len(self._joint_station_history)
        station = self._joint_station_from_spectrum_records(
            records,
            pose_idx=pose_idx,
            station_sequence_id=station_sequence_id,
            generative_contract_hash_sha256=(
                generative_contract_hash_sha256
            ),
        )
        update_start = time.perf_counter()
        self._joint_tempered_station_update(station)
        update_wall = time.perf_counter() - update_start
        self.last_pair_sequence_update_workers = 1
        self.last_pair_sequence_update_wall_s = float(update_wall)
        self.last_structural_update_workers = 1
        self.last_structural_update_wall_s = float(
            sum(
                float(
                    self.filters[isotope].last_structural_timing_s.get(
                        "total",
                        0.0,
                    )
                )
                for isotope in self.joint_isotope_order()
            )
        )
        detector_position = station.detector_position_xyz_m
        for view_index, record in enumerate(records):
            self.measurements.append(
                MeasurementRecord(
                    spectrum_counts_b=np.ascontiguousarray(
                        station.spectrum_vb[view_index]
                    ),
                    pose_idx=int(station.pose_idx),
                    live_time_s=float(station.live_times_s[view_index]),
                    fe_index=int(station.fe_indices[view_index]),
                    pb_index=int(station.pb_indices[view_index]),
                    detector_position_xyz_m=detector_position,
                    station_sequence_id=int(station_sequence_id),
                    station_view_index=int(view_index),
                    generative_contract_hash_sha256=str(
                        generative_contract_hash_sha256
                    ),
                )
            )
        self._record_history_estimate(len(self.measurements))
        self.last_pair_sequence_stage_wall_s = {
            "normalize_and_validate": float(update_start - sequence_start),
            "joint_smc_and_conditional_rj": float(update_wall),
            "total": float(time.perf_counter() - sequence_start),
        }

    @staticmethod
    def _station_sequence_ids_for_records(
        records: Sequence[MeasurementRecord],
    ) -> NDArray[np.int64]:
        """Return the required runtime-likelihood block ID for every history row."""
        return np.fromiter(
            (int(record.station_sequence_id) for record in records),
            dtype=np.int64,
            count=len(records),
        )

    def _structural_geometry_for_records(
        self,
        isotope: str,
        window: int | None,
        records: Sequence[MeasurementRecord] | None = None,
    ) -> StructuralGeometryBatch | None:
        """Return geometry-only rows without an independent isotope target."""
        isotope_key = str(isotope)
        if isotope_key not in self.joint_isotope_order():
            raise KeyError(f"Unknown joint PF isotope: {isotope_key!r}.")
        if records is not None:
            selected_records = list(records)
        elif window is None or int(window) <= 0:
            selected_records = list(self.measurements)
        else:
            selected_records = self.measurements[-int(window) :]
        if not selected_records:
            return None
        return StructuralGeometryBatch(
            detector_positions=np.asarray(
                [
                    record.detector_position_xyz_m
                    for record in selected_records
                ],
                dtype=np.float64,
            ),
            fe_indices=np.asarray(
                [record.fe_index for record in selected_records],
                dtype=np.int64,
            ),
            pb_indices=np.asarray(
                [record.pb_index for record in selected_records],
                dtype=np.int64,
            ),
            live_times=np.asarray(
                [record.live_time_s for record in selected_records],
                dtype=np.float64,
            ),
            station_sequence_ids=self._station_sequence_ids_for_records(
                selected_records
            ),
        )

    def _source_prior_environment(self) -> EnvironmentConfig:
        """Return the room geometry used by surface-candidate diagnostics."""
        hi = np.asarray(self.pf_config.position_max, dtype=float).reshape(3)
        return EnvironmentConfig(
            size_x=float(hi[0]),
            size_y=float(hi[1]),
            size_z=float(hi[2]),
        )

    @staticmethod
    def _response_design_observability_stats(
        design: NDArray[np.float64],
        *,
        eps: float,
    ) -> dict[str, float | int]:
        """Return condition and correlation statistics for a response design."""
        design_arr = np.maximum(np.asarray(design, dtype=float), 0.0)
        if design_arr.ndim != 2 or design_arr.shape[0] == 0:
            return {
                "candidate_count": int(
                    design_arr.shape[1] if design_arr.ndim == 2 else 0
                ),
                "active_candidate_count": 0,
                "weak_column_count": 0,
                "condition_number": 1.0,
                "max_abs_correlation": 0.0,
                "ambiguous_pair_count_corr_ge_0p99": 0,
                "ambiguous_pair_count_corr_ge_0p995": 0,
            }
        column_norm = np.linalg.norm(design_arr, axis=0)
        valid = column_norm > max(float(eps), 1.0e-12)
        weak_count = int(np.count_nonzero(~valid))
        if np.count_nonzero(valid) <= 1:
            return {
                "candidate_count": int(design_arr.shape[1]),
                "active_candidate_count": int(np.count_nonzero(valid)),
                "weak_column_count": weak_count,
                "condition_number": 1.0,
                "max_abs_correlation": 0.0,
                "ambiguous_pair_count_corr_ge_0p99": 0,
                "ambiguous_pair_count_corr_ge_0p995": 0,
            }
        normalized = design_arr[:, valid] / np.maximum(column_norm[valid], eps)
        try:
            singular_values = np.linalg.svd(normalized, compute_uv=False)
            positive = singular_values[singular_values > max(float(eps), 1.0e-12)]
            condition = (
                float(np.max(positive) / max(float(np.min(positive)), eps))
                if positive.size
                else float("inf")
            )
        except np.linalg.LinAlgError:
            condition = float("inf")
        corr = np.abs(normalized.T @ normalized)
        upper = np.triu_indices(corr.shape[0], k=1)
        upper_values = corr[upper] if upper[0].size else np.zeros(0, dtype=float)
        max_corr = float(np.max(upper_values)) if upper_values.size else 0.0
        return {
            "candidate_count": int(design_arr.shape[1]),
            "active_candidate_count": int(np.count_nonzero(valid)),
            "weak_column_count": weak_count,
            "condition_number": condition,
            "max_abs_correlation": max_corr,
            "ambiguous_pair_count_corr_ge_0p99": int(
                np.count_nonzero(upper_values >= 0.99)
            ),
            "ambiguous_pair_count_corr_ge_0p995": int(
                np.count_nonzero(upper_values >= 0.995)
            ),
        }

    def surface_atlas_observability_diagnostics(
        self,
        *,
        window: int | None = None,
        max_candidates: int = 256,
    ) -> dict[str, dict[str, Any]]:
        """Return truth-independent observability diagnostics over surface candidates."""
        diagnostics: dict[str, dict[str, Any]] = {}
        if self.surface_diagnostic_points.size == 0:
            return diagnostics
        self._ensure_kernel_cache()
        pool_all = np.asarray(self.surface_diagnostic_points, dtype=float).reshape(-1, 3)
        sample_count = max(1, min(int(max_candidates), int(pool_all.shape[0])))
        if pool_all.shape[0] > sample_count:
            sample_indices = np.linspace(
                0,
                pool_all.shape[0] - 1,
                sample_count,
                dtype=np.int64,
            )
            pool = pool_all[sample_indices]
        else:
            pool = pool_all
        env = self._source_prior_environment()
        surface_kinds = source_surface_kinds(
            pool,
            env,
            self.obstacle_grid,
            obstacle_height_m=self.obstacle_height_m,
        )
        surface_counts = {
            str(kind): int(np.count_nonzero(surface_kinds == kind))
            for kind in SOURCE_SURFACE_REPORT_LABELS[:-1]
        }
        surface_counts["off_surface"] = int(
            np.count_nonzero(np.equal(surface_kinds, None))
        )
        eps = 1.0e-12
        for isotope, filt in self.filters.items():
            data = self._structural_geometry_for_records(isotope, window)
            if data is None or data.row_count == 0:
                diagnostics[isotope] = {
                    "candidate_count": int(pool_all.shape[0]),
                    "sampled_candidate_count": int(pool.shape[0]),
                    "measurement_count": 0,
                    "surface_counts": surface_counts,
                }
                continue
            candidate_counts = self._cached_expected_counts_per_source(
                filt=filt,
                isotope=isotope,
                data=data,
                sources=pool,
                strengths=np.ones(pool.shape[0], dtype=float),
            )
            stats = self._response_design_observability_stats(
                np.asarray(candidate_counts, dtype=float),
                eps=eps,
            )
            stats.update(
                {
                    "candidate_count": int(pool_all.shape[0]),
                    "sampled_candidate_count": int(pool.shape[0]),
                    "measurement_count": data.row_count,
                    "surface_counts": surface_counts,
                    "window": None if window is None else int(window),
                }
            )
            diagnostics[isotope] = stats
        return diagnostics

    def surface_atlas_area_quadrature(
        self,
        *,
        max_points: int,
        maximum_hausdorff_bound_m: float,
    ) -> SurfaceAtlasQuadrature:
        """Return every chart center with its exact physical chart area.

        A finite area-quantile sample can omit small charts entirely and create
        an exploration absorbing state. Production coverage therefore uses one
        center per continuous chart and fails before planning when its explicit
        budget or chart-center Hausdorff bound cannot cover the atlas.
        """
        if not self.filters:
            raise RuntimeError(
                "Surface-atlas coverage requires initialized PF filters."
            )
        self._assert_joint_surface_atlas_alignment()
        atlases = [
            filt._structural_rj_surface_atlas
            for filt in self.filters.values()
        ]
        if any(atlas is None for atlas in atlases):
            raise RuntimeError(
                "Surface-atlas coverage requires a continuous surface atlas."
            )
        atlas = atlases[0]
        if atlas is None:
            raise RuntimeError("Continuous surface atlas is unavailable.")
        return build_complete_surface_atlas_quadrature(
            atlas,
            max_points=int(max_points),
            maximum_hausdorff_bound_m=float(
                maximum_hausdorff_bound_m
            ),
        )

    def posterior_point_estimate(self) -> Dict[str, PFPointEstimate]:
        """Return deterministic posterior summaries for every isotope.

        The pure runtime subclass overrides this method to select one aligned
        joint-cardinality stratum and one common representative particle row.
        Keeping all report consumers on this virtual method prevents stopping
        and visualization from silently reverting to independent isotope
        marginals.
        """
        estimates: Dict[str, PFPointEstimate] = {}
        for isotope, filt in self.filters.items():
            filt.validate_continuous_surface_states()
            atlas = getattr(filt, "_structural_rj_surface_atlas", None)
            estimates[isotope] = posterior_point_estimate_from_states(
                [particle.state for particle in filt.continuous_particles],
                np.asarray(filt.continuous_weights, dtype=float),
                max_cardinality=self.pf_config.max_sources,
                positions_by_state=[
                    filt.continuous_state_positions(particle.state)
                    for particle in filt.continuous_particles
                ],
                surface_chart_ids_by_state=(
                    None
                    if atlas is None
                    else [
                        np.asarray(
                            particle.state.surface_chart_ids,
                            dtype=np.int64,
                        )
                        for particle in filt.continuous_particles
                    ]
                ),
                surface_uv_by_state=(
                    None
                    if atlas is None
                    else [
                        np.asarray(
                            particle.state.surface_uv,
                            dtype=np.float64,
                        )
                        for particle in filt.continuous_particles
                    ]
                ),
                surface_coordinate_path_distance=(
                    None
                    if atlas is None
                    else atlas.surface_coordinate_path_distance_upper_bound_m
                ),
            )
        return estimates

    def estimates(
        self,
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Return the canonical MAP-cardinality PF posterior projection."""
        estimates: Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
        for isotope, point_estimate in self.posterior_point_estimate().items():
            positions = np.asarray(
                [mode.position_medoid_xyz for mode in point_estimate.modes],
                dtype=float,
            ).reshape(-1, 3)
            strengths = np.asarray(
                [
                    mode.strength_representative_cps_1m
                    for mode in point_estimate.modes
                ],
                dtype=float,
            )
            estimates[isotope] = (
                positions,
                strengths,
            )
        return estimates

    def structural_surface_kinds(
        self,
        isotope: str,
        positions: NDArray[np.float64],
        *,
        strict: bool = True,
    ) -> NDArray[np.object_]:
        """Return authoritative continuous-surface kinds for one isotope."""
        filt = self.filters.get(str(isotope))
        if filt is None:
            raise KeyError(f"Unknown PF isotope: {isotope}")
        return filt.structural_surface_kinds(
            np.asarray(positions, dtype=np.float64),
            strict=strict,
        )

    def _posterior_reporting_particle_indices(
        self,
    ) -> NDArray[np.int64] | None:
        """Return a shared posterior-report stratum or all rows by default."""
        return None

    def posterior_source_uncertainty(
        self,
        reported_estimates: Mapping[
            str,
            Tuple[NDArray[np.float64], NDArray[np.float64]],
        ]
        | None = None,
        *,
        match_radius_m: float | None = None,
        surface_tolerance_m: float = 1.0e-5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Return JSON-safe posterior 3-D diagnostics for reported source modes.

        Particle source slots are matched to the nearest reported mode in one
        batched distance calculation.  Existence mass is unconditional particle
        mass, while location, covariance, z quantiles, ellipsoid, and surface
        probabilities are conditional on a matched source being present.  Each
        mode includes availability flags so downstream evaluation can exclude
        unsupported summaries; the ellipsoid payload identifies itself as a
        Gaussian-equivalent covariance summary rather than an empirical credible
        region.
        """
        point_estimate_map = self.posterior_point_estimate()
        estimate_map = (
            {
                isotope: (
                    np.asarray(
                        [
                            mode.position_medoid_xyz
                            for mode in point_estimate.modes
                        ],
                        dtype=np.float64,
                    ).reshape(-1, 3),
                    np.asarray(
                        [
                            mode.strength_representative_cps_1m
                            for mode in point_estimate.modes
                        ],
                        dtype=np.float64,
                    ),
                )
                for isotope, point_estimate in point_estimate_map.items()
            }
            if reported_estimates is None
            else dict(reported_estimates)
        )
        radius = 0.8 if match_radius_m is None else float(match_radius_m)
        environment = self._source_prior_environment()
        reporting_indices = self._posterior_reporting_particle_indices()
        output: Dict[str, List[Dict[str, Any]]] = {}
        for isotope, estimate in estimate_map.items():
            positions = np.asarray(estimate[0], dtype=float)
            strengths = np.asarray(estimate[1], dtype=float).reshape(-1)
            if positions.size == 0:
                positions = np.zeros((0, 3), dtype=float)
            if positions.ndim != 2 or positions.shape[1] != 3:
                raise ValueError("reported estimate positions must have shape (M, 3).")
            if strengths.size != positions.shape[0]:
                raise ValueError(
                    "reported estimate strengths must have one value per mode."
                )
            if np.any(~np.isfinite(strengths)):
                raise ValueError("reported estimate strengths must be finite.")

            filt = self.filters.get(isotope)
            if filt is None or not filt.continuous_particles:
                packed_positions = np.zeros((0, 0, 3), dtype=float)
                packed_mask = np.zeros((0, 0), dtype=bool)
                packed_chart_ids = np.zeros((0, 0), dtype=np.int64)
                packed_surface_uv = np.zeros((0, 0, 2), dtype=np.float64)
                weights = np.zeros(0, dtype=float)
                atlas = None
            else:
                (
                    packed_positions,
                    _,
                    packed_mask,
                    packed_chart_ids,
                    packed_surface_uv,
                ) = (
                    filt._packed_continuous_surface_state_arrays()
                )
                weights = np.asarray(filt.continuous_weights, dtype=float)
                atlas = getattr(filt, "_structural_rj_surface_atlas", None)
                if atlas is None:
                    raise RuntimeError(
                        "Posterior uncertainty requires the continuous surface "
                        "atlas."
                    )
                if reporting_indices is not None:
                    indices = np.asarray(
                        reporting_indices,
                        dtype=np.int64,
                    ).reshape(-1)
                    if (
                        indices.size == 0
                        or np.any(indices < 0)
                        or np.any(indices >= weights.size)
                    ):
                        raise RuntimeError(
                            "Posterior uncertainty received an invalid shared "
                            "reporting stratum."
                        )
                    packed_positions = packed_positions[indices]
                    packed_mask = packed_mask[indices]
                    packed_chart_ids = packed_chart_ids[indices]
                    packed_surface_uv = packed_surface_uv[indices]
                    reporting_mass = float(np.sum(weights[indices]))
                    if (
                        not np.isfinite(reporting_mass)
                        or reporting_mass <= 0.0
                    ):
                        raise RuntimeError(
                            "Posterior reporting stratum has no probability mass."
                        )
                    weights = validated_probability_distribution(
                        weights[indices] / reporting_mass,
                        name="posterior reporting-stratum weights",
                    )
            packed_surface_kinds = np.full(packed_mask.shape, None, dtype=object)
            if np.any(packed_mask):
                packed_surface_kinds[packed_mask] = filt.structural_surface_kinds(
                    packed_positions[packed_mask],
                    strict=True,
                )
            reported_chart_ids = np.zeros(
                positions.shape[0],
                dtype=np.int64,
            )
            reported_surface_uv = np.zeros(
                (positions.shape[0], 2),
                dtype=np.float64,
            )
            if positions.shape[0] > 0:
                if atlas is None:
                    raise RuntimeError(
                        "Reported continuous-surface modes require an atlas."
                    )
                point_estimate = point_estimate_map.get(isotope)
                point_positions = np.asarray(
                    []
                    if point_estimate is None
                    else [
                        mode.position_medoid_xyz
                        for mode in point_estimate.modes
                    ],
                    dtype=np.float64,
                ).reshape(-1, 3)
                point_modes_have_coordinates = bool(
                    point_estimate is not None
                    and len(point_estimate.modes) == positions.shape[0]
                    and np.array_equal(point_positions, positions)
                    and all(
                        mode.surface_chart_id is not None
                        and mode.surface_uv is not None
                        for mode in point_estimate.modes
                    )
                )
                if point_modes_have_coordinates and point_estimate is not None:
                    reported_chart_ids = np.asarray(
                        [
                            int(mode.surface_chart_id)
                            for mode in point_estimate.modes
                        ],
                        dtype=np.int64,
                    )
                    reported_surface_uv = np.asarray(
                        [
                            mode.surface_uv
                            for mode in point_estimate.modes
                        ],
                        dtype=np.float64,
                    )
                else:
                    reported_chart_ids, reported_surface_uv = (
                        atlas.locate_positions(positions)
                    )

            diagnostics = posterior_mode_uncertainty_batched(
                packed_positions,
                packed_mask,
                weights,
                positions,
                packed_surface_kinds=packed_surface_kinds,
                packed_surface_chart_ids=(
                    None if atlas is None else packed_chart_ids
                ),
                packed_surface_uv=(
                    None if atlas is None else packed_surface_uv
                ),
                reported_surface_chart_ids=(
                    None if atlas is None else reported_chart_ids
                ),
                reported_surface_uv=(
                    None if atlas is None else reported_surface_uv
                ),
                surface_coordinate_path_distance=(
                    None
                    if atlas is None
                    else atlas.surface_coordinate_path_distance_upper_bound_m
                ),
                environment=environment,
                obstacle_grid=self.obstacle_grid,
                obstacle_height_m=self.obstacle_height_m,
                match_radius_m=radius,
                surface_tolerance_m=surface_tolerance_m,
                posterior_reference_mass=(
                    1.0
                    if point_estimate_map.get(isotope) is None
                    else float(
                        point_estimate_map[
                            isotope
                        ].selected_stratum_mass
                    )
                ),
            )
            for mode_index, diagnostic in enumerate(diagnostics):
                diagnostic["reported_strength_cps_1m"] = float(strengths[mode_index])
            output[isotope] = diagnostics
        return output

    def estimate_all(
        self,
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Alias for estimates() to align with visualization helpers."""
        return self.estimates()

    def step_diagnostics(
        self,
        top_k: int = 3,
        *,
        include_estimates: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return per-isotope diagnostics for the current PF state.

        The diagnostics include ESS, resample/birth/death counts, and the source
        count distribution. When include_estimates is false, the routine avoids
        the posterior point-estimate projection.
        """
        diagnostics: Dict[str, Dict[str, Any]] = {}
        eps = 1e-12
        k = max(0, int(top_k))
        posterior_estimates = self.estimates() if include_estimates else {}
        for iso, filt in self.filters.items():
            if not filt.continuous_particles:
                diagnostics[iso] = {
                    "ess_pre": 0.0,
                    "resampled": False,
                    "ess_post": None,
                    "current_ess": 0.0,
                    "current_ess_ratio": 0.0,
                    "particle_count": 0,
                    "resample_count": int(getattr(filt, "last_resample_count", 0)),
                    "birth_count": int(getattr(filt, "last_birth_count", 0)),
                    "death_count": int(getattr(filt, "last_death_count", 0)),
                    "structural_timing_s": dict(
                        getattr(filt, "last_structural_timing_s", {})
                    ),
                    "transition_weight_mass": {},
                    "temper_steps": [],
                    "temper_resamples": 0,
                    "temper_min_ess": None,
                    "unique_ancestor_count": None,
                    "station_unique_ancestor_count": None,
                    "cumulative_unique_ancestor_count": None,
                    "r_mean": 0.0,
                    "r_var": 0.0,
                    "r_weighted_mean": 0.0,
                    "r_weighted_var": 0.0,
                    "r_probability_by_count": {},
                    "r_particle_count_by_count": {},
                    "map": (np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)),
                    "posterior": (
                        np.zeros((0, 3), dtype=float),
                        np.zeros(0, dtype=float),
                    ),
                    "top_k": [],
                }
                continue
            weights = np.asarray(filt.continuous_weights, dtype=float)
            total = float(np.sum(weights))
            if (
                weights.size != len(filt.continuous_particles)
                or not np.isfinite(total)
                or total <= 0.0
            ):
                raise RuntimeError(
                    "Step diagnostics require valid normalized PF weights."
                )
            weights = weights / total
            current_ess = float(1.0 / max(np.sum(weights**2), eps))
            current_ess_ratio = current_ess / float(weights.size)
            maximum_cardinality = self.pf_config.max_sources
            r_int = np.fromiter(
                (
                    validated_state_cardinality(
                        particle.state,
                        name=f"{iso} particle[{index}]",
                        max_cardinality=maximum_cardinality,
                    )
                    for index, particle in enumerate(
                        filt.continuous_particles
                    )
                ),
                dtype=np.int64,
                count=len(filt.continuous_particles),
            )
            r_vals = r_int.astype(np.float64)
            if weights.size != r_vals.size:
                raise RuntimeError(
                    "Step diagnostics require one weight per PF particle."
                )
            r_mean = float(np.mean(r_vals)) if r_vals.size else 0.0
            r_var = float(np.var(r_vals)) if r_vals.size else 0.0
            r_weighted_mean = float(np.sum(weights * r_vals)) if r_vals.size else 0.0
            r_weighted_var = (
                float(np.sum(weights * (r_vals - r_weighted_mean) ** 2))
                if r_vals.size
                else 0.0
            )
            r_probability_by_count = {
                str(int(value)): float(np.sum(weights[r_int == int(value)]))
                for value in sorted(set(int(v) for v in r_int.tolist()))
            }
            r_particle_count_by_count = {
                str(int(value)): int(np.count_nonzero(r_int == int(value)))
                for value in sorted(set(int(v) for v in r_int.tolist()))
            }
            ess_pre = getattr(filt, "last_ess_pre", None)
            if ess_pre is None and weights.size:
                ess_pre = float(1.0 / max(np.sum(weights**2), eps))
            if ess_pre is None:
                ess_pre = 0.0
            resampled = bool(getattr(filt, "last_resample_ess", False))
            ess_post = getattr(filt, "last_ess_post", None)
            particle_count = int(len(filt.continuous_particles))
            if include_estimates:
                posterior_positions, posterior_strengths = posterior_estimates[iso]
            else:
                posterior_positions = np.zeros((0, 3), dtype=float)
                posterior_strengths = np.zeros(0, dtype=float)
            top_entries: List[Dict[str, Any]] = []
            if k > 0 and weights.size:
                order = np.argsort(weights)[::-1][:k]
                for idx in order:
                    state = filt.continuous_particles[int(idx)].state
                    source_count = validated_state_cardinality(
                        state,
                        name=f"{iso} particle[{int(idx)}]",
                        max_cardinality=maximum_cardinality,
                    )
                    top_entries.append(
                        {
                            "weight": float(weights[idx]),
                            "num_sources": source_count,
                            "positions": filt.continuous_state_positions(
                                state
                            )[:source_count],
                            "strengths": state.strengths[:source_count].copy(),
                        }
                    )
            diagnostics[iso] = {
                "ess_pre": float(ess_pre),
                "resampled": resampled,
                "ess_post": ess_post,
                "current_ess": current_ess,
                "current_ess_ratio": current_ess_ratio,
                "particle_count": particle_count,
                "resample_count": int(getattr(filt, "last_resample_count", 0)),
                "birth_count": int(getattr(filt, "last_birth_count", 0)),
                "death_count": int(getattr(filt, "last_death_count", 0)),
                "structural_timing_s": dict(
                    getattr(filt, "last_structural_timing_s", {})
                ),
                "transition_weight_mass": dict(
                    getattr(
                        filt,
                        "last_structural_transition_weight_mass",
                        {},
                    )
                ),
                "temper_steps": list(getattr(filt, "last_temper_steps", [])),
                "temper_resamples": int(getattr(filt, "last_temper_resample_count", 0)),
                "temper_min_ess": getattr(filt, "last_temper_min_ess", None),
                "unique_ancestor_count": getattr(
                    filt,
                    "last_unique_ancestor_count",
                    None,
                ),
                "station_unique_ancestor_count": getattr(
                    filt,
                    "last_station_unique_ancestor_count",
                    None,
                ),
                "cumulative_unique_ancestor_count": getattr(
                    filt,
                    "last_cumulative_unique_ancestor_count",
                    None,
                ),
                "r_mean": r_mean,
                "r_var": r_var,
                "r_weighted_mean": r_weighted_mean,
                "r_weighted_var": r_weighted_var,
                "r_probability_by_count": r_probability_by_count,
                "r_particle_count_by_count": r_particle_count_by_count,
                "posterior": (
                    posterior_positions,
                    posterior_strengths,
                ),
                "top_k": top_entries,
            }
        return diagnostics

    @property
    def num_orientations(self) -> int:
        """Return the number of shield orientation normals."""
        return self.normals.shape[0]

    @staticmethod
    def _normalized_stopping_weights(
        filt: IsotopeParticleFilter,
    ) -> NDArray[np.float64]:
        """Return one filter's valid normalized posterior weights."""
        particle_count = len(filt.continuous_particles)
        weights = np.asarray(filt.continuous_weights, dtype=float).reshape(-1)
        if weights.size != particle_count:
            raise RuntimeError(
                "Stopping diagnostics require one weight per PF particle."
            )
        if (
            weights.size == 0
            or np.any(~np.isfinite(weights))
            or np.any(weights < 0.0)
        ):
            raise RuntimeError(
                "Stopping diagnostics require finite nonnegative PF weights."
            )
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0.0:
            raise RuntimeError(
                "Stopping diagnostics require positive posterior weight mass."
            )
        return weights / total

    def credible_surface_radii(
        self,
        confidence: float = 0.95,
    ) -> Dict[str, List[float]]:
        """Return conservative path radii along connected physical surfaces.

        Each radius is computed inside the MAP-cardinality stratum after
        deterministic source alignment.  Its center is an actual posterior
        surface state. Same-chart and one-portal paths are evaluated directly;
        longer paths use a realizable portal-graph path. Disconnected posterior
        mass makes the radius infinite. This fail-closed statistic cannot
        collapse merely because a broad posterior lies on a two-dimensional
        wall, floor, or ceiling.
        """
        probability = float(confidence)
        if not np.isfinite(probability) or not 0.0 < probability <= 1.0:
            raise ValueError("confidence must be in (0, 1].")
        radii: Dict[str, List[float]] = {}
        point_estimates = self.posterior_point_estimate()
        for isotope, filt in self.filters.items():
            if not filt.continuous_particles:
                radii[isotope] = []
                continue
            atlas = getattr(filt, "_structural_rj_surface_atlas", None)
            if atlas is None:
                raise RuntimeError(
                    "Surface convergence requires a continuous surface atlas."
                )
            filt.validate_continuous_surface_states()
            estimate = point_estimates.get(isotope)
            if estimate is None:
                raise RuntimeError(
                    "Canonical posterior report omitted an initialized isotope."
                )
            if np.isclose(probability, 0.95, rtol=0.0, atol=1.0e-15):
                radii[isotope] = [
                    (
                        float("inf")
                        if mode.credible_surface_path_radius_95_m is None
                        else float(mode.credible_surface_path_radius_95_m)
                    )
                    for mode in estimate.modes
                ]
                continue

            # The reporting contract stores the 95% radius.  Other confidence
            # levels are intentionally unavailable here instead of being
            # inferred from a Gaussian ellipsoid on a non-Euclidean surface.
            raise ValueError(
                "credible_surface_radii currently supports confidence=0.95 only."
            )
        return radii

    def _latest_joint_station_innovation(
        self,
    ) -> Dict[str, float | int | bool | None]:
        """Return strict renewal-total and conditional-mark innovation gates."""
        if not self._joint_station_history:
            return {
                "available": False,
                "passed": False,
                "view_count": 0,
                "dimension": 0,
                "renewal_total_max_abs_z": None,
                "renewal_total_within_confidence": False,
                "conditional_mark_pearson": None,
                "conditional_mark_degrees_of_freedom": 0,
                "conditional_mark_tail_probability": None,
                "confidence": float(
                    self.pf_config.converge_innovation_confidence
                ),
            }
        station = self._joint_station_history[-1]
        weights = self._strict_joint_particle_weights()
        components = tuple(
            value.detach().cpu().numpy().astype(np.float64, copy=False)
            for value in self._joint_station_transport_components_torch(
                station
            )
        )
        raw_result = dict(
            self._full_spectrum_model().posterior_predictive_innovation_numpy(
                station.spectrum_vb,
                components[0],
                components[1],
                components[2],
                station.live_times_s,
                weights,
                confidence=float(
                    self.pf_config.converge_innovation_confidence
                ),
            )
        )
        required_raw = {
            "renewal_total_max_abs_z",
            "renewal_total_within_confidence",
            "conditional_mark_pearson",
            "conditional_mark_degrees_of_freedom",
            "conditional_mark_tail_probability",
            "confidence",
        }
        if set(raw_result) != required_raw:
            raise RuntimeError(
                "Full-spectrum innovation returned an incompatible diagnostic "
                "schema."
            )
        confidence = float(raw_result["confidence"])
        total_z = float(raw_result["renewal_total_max_abs_z"])
        mark_pearson = float(raw_result["conditional_mark_pearson"])
        mark_degrees = raw_result["conditional_mark_degrees_of_freedom"]
        mark_tail_raw = raw_result["conditional_mark_tail_probability"]
        if (
            not np.isfinite(confidence)
            or not np.isclose(
                confidence,
                float(self.pf_config.converge_innovation_confidence),
                rtol=0.0,
                atol=1.0e-15,
            )
            or not np.isfinite(total_z)
            or total_z < 0.0
            or type(raw_result["renewal_total_within_confidence"]) is not bool
            or not np.isfinite(mark_pearson)
            or mark_pearson < 0.0
            or isinstance(mark_degrees, (bool, np.bool_))
            or not isinstance(mark_degrees, (int, np.integer))
        ):
            raise RuntimeError(
                "Full-spectrum innovation contains invalid diagnostics."
            )
        mark_tail: float | None
        if mark_tail_raw is None:
            mark_tail = None
        else:
            mark_tail = float(mark_tail_raw)
            if (
                not np.isfinite(mark_tail)
                or mark_tail < 0.0
                or mark_tail > 1.0
            ):
                raise RuntimeError(
                    "Full-spectrum conditional-mark tail probability is invalid."
                )
        mark_passed = bool(
            mark_tail is not None
            and mark_tail + 1.0e-15 >= 1.0 - confidence
        )
        total_passed = bool(
            raw_result["renewal_total_within_confidence"]
        )
        return {
            "available": True,
            "passed": bool(total_passed and mark_passed),
            "view_count": int(station.spectrum_vb.shape[0]),
            "dimension": int(station.spectrum_vb.size),
            "renewal_total_max_abs_z": total_z,
            "renewal_total_within_confidence": total_passed,
            "conditional_mark_pearson": mark_pearson,
            "conditional_mark_degrees_of_freedom": int(mark_degrees),
            "conditional_mark_tail_probability": mark_tail,
            "confidence": confidence,
        }

    def posterior_convergence_diagnostics(self) -> Dict[str, Any]:
        """Return fail-closed PF convergence gates without using simulation truth."""
        isotope_diagnostics: Dict[str, Dict[str, Any]] = {}
        all_ready = True
        joint_innovation = self._latest_joint_station_innovation()
        point_estimates = self.posterior_point_estimate()
        for isotope, filt in self.filters.items():
            if not filt.continuous_particles:
                isotope_diagnostics[isotope] = {
                    "ready": False,
                    "reason": "missing_particles",
                }
                all_ready = False
                continue
            weights = self._normalized_stopping_weights(filt)
            particle_count = int(weights.size)
            ess = float(1.0 / np.sum(weights**2))
            ess_ratio = ess / float(particle_count)
            atlas = getattr(filt, "_structural_rj_surface_atlas", None)
            if atlas is None:
                raise RuntimeError(
                    "Surface convergence requires a continuous surface atlas."
                )
            filt.validate_continuous_surface_states()
            point_estimate = point_estimates.get(isotope)
            if point_estimate is None:
                raise RuntimeError(
                    "Canonical posterior report omitted an initialized isotope."
                )
            distribution = {
                int(cardinality): float(mass)
                for cardinality, mass in point_estimate.cardinality_distribution.items()
            }
            map_probability = float(point_estimate.selected_stratum_mass)
            cardinality_mean = float(
                sum(
                    float(cardinality) * mass
                    for cardinality, mass in distribution.items()
                )
            )
            cardinality_variance = float(
                sum(
                    mass * (float(cardinality) - cardinality_mean) ** 2
                    for cardinality, mass in distribution.items()
                )
            )
            maximum_cardinality = int(filt.config.max_sources or 0)
            boundary_mass = float(distribution.get(maximum_cardinality, 0.0))
            radii = [
                (
                    None
                    if mode.credible_surface_path_radius_95_m is None
                    else float(mode.credible_surface_path_radius_95_m)
                )
                for mode in point_estimate.modes
            ]
            connected_masses = [
                float(mode.surface_connected_mass)
                for mode in point_estimate.modes
            ]
            maximum_radius = (
                None
                if any(radius is None for radius in radii)
                else max(
                    (float(radius) for radius in radii if radius is not None),
                    default=0.0,
                )
            )
            gates = {
                "current_ess": bool(
                    ess_ratio >= self.pf_config.converge_min_ess_ratio
                ),
                "cardinality_confidence": bool(
                    map_probability
                    >= self.pf_config.converge_cardinality_min_probability
                    and cardinality_variance
                    <= self.pf_config.converge_cardinality_var_max
                ),
                "cardinality_not_at_upper_boundary": bool(
                    not filt.config.variable_cardinality
                    or boundary_mass
                    <= self.pf_config.converge_max_cardinality_boundary_mass
                ),
                "surface_radius": bool(
                    maximum_radius is not None
                    and maximum_radius
                    <= self.pf_config.credible_surface_radius_threshold_m
                ),
                "innovation": bool(joint_innovation["passed"]),
            }
            ready = bool(all(gates.values()))
            isotope_diagnostics[isotope] = {
                "ready": ready,
                "current_ess": ess,
                "particle_count": particle_count,
                "current_ess_ratio": ess_ratio,
                "cardinality_distribution": distribution,
                "map_cardinality_probability": map_probability,
                "cardinality_variance": cardinality_variance,
                "maximum_cardinality_boundary_mass": boundary_mass,
                "credible_surface_radii_95_m": radii,
                "surface_connected_masses": connected_masses,
                "maximum_credible_surface_radius_95_m": maximum_radius,
                "innovation": dict(joint_innovation),
                "gates": gates,
            }
            all_ready &= ready
        return {
            "ready": bool(isotope_diagnostics) and all_ready,
            "metric": "surface_path_upper_bound_credible_distance",
            "isotopes": isotope_diagnostics,
        }
