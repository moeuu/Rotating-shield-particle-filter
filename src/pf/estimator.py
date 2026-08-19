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
from pf.strength_prior import StrengthPrior
from pf.structural_rj import (
    ContinuousBlockStrengthProposal,
    POISSON_GEOMETRIC_TAIL_CARDINALITY_PRIOR_POLICY,
    TRUNCATED_POISSON_CARDINALITY_PRIOR_POLICY,
    cross_isotope_transfer_log_proposal,
    shifted_log_strength_random_walk_log_reverse_ratio,
    validate_cardinality_prior_policy,
)
from pf.transport_response import expected_counts_per_source

if TYPE_CHECKING:
    import torch


JOINT_HISTORY_STATION_ACTION_BATCH_SIZE = 4
JOINT_STRENGTH_GRID_AUTOTUNE_MAX_BATCH_SIZE = 1024
JOINT_STRUCTURAL_UNIT_CACHE_MAX_BYTES = 2_147_483_648
JOINT_STRUCTURAL_UNIT_CACHE_ACTIVE_STATE_MULTIPLIER = 2
JOINT_STRUCTURAL_UNIT_CACHE_KEY_DTYPE = np.dtype(
    [
        ("chart", "<i8"),
        ("x", "<f8"),
        ("y", "<f8"),
        ("z", "<f8"),
    ]
)
JOINT_STRUCTURAL_UNIT_COMPONENT_NAMES = (
    "total_kernel",
    "uncollided_kernel",
    "tau_fe",
    "tau_pb",
    "tau_obstacle",
    "distance_m",
)


def _stratified_categorical_draws(
    probabilities: NDArray[np.float64],
    sample_count: int,
    *,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Draw a randomly permuted stratified categorical sample batch."""
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    count = int(sample_count)
    total = float(np.sum(values, dtype=np.float64))
    if (
        count < 1
        or values.size == 0
        or np.any(~np.isfinite(values))
        or np.any(values < 0.0)
        or not np.isfinite(total)
        or total <= 0.0
        or not isinstance(rng, np.random.Generator)
    ):
        raise ValueError("Stratified categorical inputs are invalid.")
    normalized = values / total
    uniforms = (np.arange(count, dtype=np.float64) + rng.random(count)) / float(count)
    draws = np.searchsorted(
        np.cumsum(normalized, dtype=np.float64),
        uniforms,
        side="right",
    ).astype(np.int64, copy=False)
    draws = np.minimum(draws, values.size - 1)
    return draws[rng.permutation(count)]


def _stratified_joint_cardinality_draws(
    marginal_probabilities: Sequence[NDArray[np.float64]],
    sample_count: int,
    *,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Draw product-prior K vectors with joint stratification.

    The vectorized Cartesian support is tiny for the configured isotope and
    source capacities. Sampling the flattened product distribution preserves
    the independent cardinality prior exactly while stratifying the joint
    vectors rather than only their separate isotope marginals.
    """
    probabilities = tuple(
        np.asarray(values, dtype=np.float64).reshape(-1)
        for values in marginal_probabilities
    )
    if not probabilities:
        raise ValueError("At least one cardinality marginal is required.")
    if any(
        values.size == 0
        or np.any(~np.isfinite(values))
        or np.any(values < 0.0)
        or not np.isclose(np.sum(values), 1.0, rtol=0.0, atol=1.0e-12)
        for values in probabilities
    ):
        raise ValueError("Cardinality marginals must be probability vectors.")
    support_shape = tuple(int(values.size) for values in probabilities)
    support_indices = (
        np.indices(
            support_shape,
            dtype=np.int64,
        )
        .reshape(len(probabilities), -1)
        .T
    )
    product_mass = np.ones(support_indices.shape[0], dtype=np.float64)
    for isotope_index, values in enumerate(probabilities):
        product_mass *= values[support_indices[:, isotope_index]]
    flat_draws = _stratified_categorical_draws(
        product_mass,
        sample_count,
        rng=rng,
    )
    return support_indices[flat_draws]


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
    hard_max_sources: int | None = None
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
    structural_rj_strength_proposal_grid_size: int = 5
    structural_rj_proposal_chart_batch_size: int = 256
    structural_rj_proposal_score_cache_max_bytes: int = 268_435_456
    structural_rj_local_position_move_probability: float = 1.0
    structural_rj_local_position_sigma_m: float = 0.5
    structural_rj_strength_move_probability: float = 1.0
    structural_rj_split_merge_probability: float = 1.0
    structural_rj_block_independence_probability: float = 0.1
    structural_rj_multi_component_probability: float = 0.1
    structural_rj_multi_component_max_group_size: int = 4
    structural_rj_split_probability: float = 0.5
    structural_rj_merge_probability: float = 0.5
    structural_rj_split_global_position_probability: float = 0.1
    structural_rj_merge_uniform_pair_probability: float = 0.1
    structural_rj_merge_distance_sigma_m: float = 0.5
    structural_rj_merge_response_sigma: float = 0.05
    structural_cardinality_prior_policy: str = (
        TRUNCATED_POISSON_CARDINALITY_PRIOR_POLICY
    )
    structural_cardinality_prior_probs: tuple[float, ...] | list[float] | None = None
    structural_cardinality_prior_mean: float = 2.0
    structural_cardinality_tail_ratio: float = 0.05
    max_dwell_time_s: float = 5.0  # Max dwell time per pose.
    credible_surface_radius_threshold_m: float = 0.5
    converge_min_ess_ratio: float = 0.5
    converge_cardinality_min_probability: float = 0.95
    converge_max_cardinality_boundary_mass: float = 0.05
    converge_innovation_confidence: float = 0.99
    target_ess_ratio: float = 0.5
    max_temper_steps: int = 256
    min_delta_beta: float = 1e-10
    joint_rejuvenation_min_sweeps: int = 1
    joint_rejuvenation_max_sweeps: int = 2
    joint_rejuvenation_min_state_change_weight_mass: float = 0.10
    joint_rejuvenation_min_surface_esjd_m2: float = 1.0e-4
    joint_rejuvenation_min_log_strength_esjd: float = 1.0e-4
    joint_rejuvenation_min_k_transition_weight_mass: float = 1.0e-4
    joint_smc_soft_wall_time_s: float = 1800.0
    joint_guided_initialization: bool = True
    joint_guided_initialization_prior_row_probability: float = 0.5
    joint_strength_block_probability: float = 0.0
    joint_strength_block_log_sigma: float = 0.15
    joint_strength_block_batch_size: int = 128
    joint_cross_isotope_transfer_probability: float = 0.0
    joint_cross_isotope_transfer_max_group: int = 3
    joint_cross_isotope_state_block_probability: float = 0.0
    detected_isotopes_only: bool = False
    detected_isotope_false_activation_probability: float = 1.0e-3
    position_max: Tuple[float, float, float] = (10.0, 10.0, 10.0)
    init_num_sources: Tuple[int, int] = (
        0,
        DEFAULT_MAX_SOURCES_PER_ISOTOPE,
    )
    strength_prior_min_cps_1m: float = 1.0
    strength_prior_max_cps_1m: float = 2_000_000.0
    strength_prior_family: str = "bounded_uniform"
    strength_prior_gamma_shape: float = 2.0
    strength_prior_gamma_scale_cps_1m: float = 425_000.0
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
                "joint_strength_block_batch_size",
                self.joint_strength_block_batch_size,
                1,
            ),
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
            (
                "joint_rejuvenation_min_sweeps",
                self.joint_rejuvenation_min_sweeps,
                1,
            ),
            (
                "joint_rejuvenation_max_sweeps",
                self.joint_rejuvenation_max_sweeps,
                1,
            ),
        )
        for name, value, minimum in integer_fields:
            resolved = _strict_nonnegative_integer(value, name=name)
            if resolved < minimum:
                raise ValueError(f"{name} must be at least {minimum}.")
        if self.max_sources is None:
            raise ValueError("Pure PF requires a finite positive max_sources.")
        if (
            _strict_nonnegative_integer(
                self.max_sources,
                name="max_sources",
            )
            < 1
        ):
            raise ValueError("Pure PF requires a finite positive max_sources.")
        if self.hard_max_sources is None:
            self.hard_max_sources = int(self.max_sources)
        if _strict_nonnegative_integer(
            self.hard_max_sources,
            name="hard_max_sources",
        ) < int(self.max_sources):
            raise ValueError("hard_max_sources must be at least max_sources.")
        _strict_config_boolean(
            self.variable_cardinality,
            name="variable_cardinality",
        )
        _strict_config_boolean(self.use_gpu, name="use_gpu")
        _strict_config_boolean(
            self.joint_guided_initialization,
            name="joint_guided_initialization",
        )
        _strict_config_boolean(
            self.detected_isotopes_only,
            name="detected_isotopes_only",
        )
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
            "strength_prior_gamma_shape",
            "strength_prior_gamma_scale_cps_1m",
            "joint_strength_block_probability",
            "joint_strength_block_log_sigma",
            "joint_cross_isotope_state_block_probability",
            "detected_isotope_false_activation_probability",
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
            "structural_rj_block_independence_probability",
            "structural_rj_multi_component_probability",
            "structural_rj_split_probability",
            "structural_rj_merge_probability",
            "structural_rj_split_global_position_probability",
            "structural_rj_merge_uniform_pair_probability",
            "structural_rj_merge_distance_sigma_m",
            "structural_rj_merge_response_sigma",
            "structural_cardinality_prior_mean",
            "structural_cardinality_tail_ratio",
            "max_dwell_time_s",
            "credible_surface_radius_threshold_m",
            "converge_min_ess_ratio",
            "converge_cardinality_min_probability",
            "converge_max_cardinality_boundary_mass",
            "converge_innovation_confidence",
            "target_ess_ratio",
            "min_delta_beta",
            "joint_rejuvenation_min_state_change_weight_mass",
            "joint_rejuvenation_min_surface_esjd_m2",
            "joint_rejuvenation_min_log_strength_esjd",
            "joint_rejuvenation_min_k_transition_weight_mass",
            "joint_smc_soft_wall_time_s",
            "joint_guided_initialization_prior_row_probability",
            "converge_cardinality_var_max",
        )
        for name in numeric_fields:
            _strict_config_number(getattr(self, name), name=name)
        if self.structural_cardinality_prior_probs is not None:
            for index, value in enumerate(self.structural_cardinality_prior_probs):
                _strict_config_number(
                    value,
                    name=f"structural_cardinality_prior_probs[{index}]",
                )
        if not isinstance(self.gpu_device, str) or not self.gpu_device.strip():
            raise TypeError("gpu_device must be a nonempty string.")
        if not isinstance(self.gpu_dtype, str):
            raise TypeError("gpu_dtype must be a string.")
        if self.planning_particles is not None:
            if (
                _strict_nonnegative_integer(
                    self.planning_particles,
                    name="planning_particles",
                )
                < 2
            ):
                raise ValueError("planning_particles must be at least two.")
        if self.planning_method not in {"resample", "top_weight"}:
            raise ValueError("planning_method must be 'resample' or 'top_weight'.")
        if self.orientation_k > 64:
            raise ValueError("orientation_k cannot exceed 64.")
        if self.min_rotations_per_pose > self.orientation_k:
            raise ValueError("min_rotations_per_pose cannot exceed orientation_k.")
        if (
            _strict_config_number(
                self.max_dwell_time_s,
                name="max_dwell_time_s",
            )
            <= 0.0
        ):
            raise ValueError("max_dwell_time_s must be positive.")
        target_ess_ratio = _strict_config_number(
            self.target_ess_ratio,
            name="target_ess_ratio",
        )
        if not 0.0 < target_ess_ratio < 1.0:
            raise ValueError("target_ess_ratio must lie strictly between zero and one.")
        min_delta_beta = _strict_config_number(
            self.min_delta_beta,
            name="min_delta_beta",
        )
        if not 0.0 < min_delta_beta <= 1.0:
            raise ValueError("min_delta_beta must lie in (0, 1].")
        if int(self.joint_rejuvenation_max_sweeps) < int(
            self.joint_rejuvenation_min_sweeps
        ):
            raise ValueError(
                "joint_rejuvenation_max_sweeps must be at least "
                "joint_rejuvenation_min_sweeps."
            )
        state_change_mass = _strict_config_number(
            self.joint_rejuvenation_min_state_change_weight_mass,
            name="joint_rejuvenation_min_state_change_weight_mass",
        )
        if not 0.0 <= state_change_mass <= 1.0:
            raise ValueError(
                "joint_rejuvenation_min_state_change_weight_mass must lie in [0, 1]."
            )
        for name in (
            "joint_rejuvenation_min_surface_esjd_m2",
            "joint_rejuvenation_min_log_strength_esjd",
            "joint_rejuvenation_min_k_transition_weight_mass",
        ):
            if _strict_config_number(getattr(self, name), name=name) < 0.0:
                raise ValueError(f"{name} must be nonnegative.")
        if (
            _strict_config_number(
                self.joint_smc_soft_wall_time_s,
                name="joint_smc_soft_wall_time_s",
            )
            <= 0.0
        ):
            raise ValueError("joint_smc_soft_wall_time_s must be positive.")
        innovation_confidence = _strict_config_number(
            self.converge_innovation_confidence,
            name="converge_innovation_confidence",
        )
        if not 0.0 < innovation_confidence < 1.0:
            raise ValueError("converge_innovation_confidence must lie in (0, 1).")
        self.num_particles = int(self.num_particles)
        if self.num_particles < 1:
            raise ValueError("num_particles must be positive.")
        if str(self.gpu_dtype).strip().lower() != "float64":
            raise ValueError("Pure PF production kernels require gpu_dtype='float64'.")
        self.gpu_dtype = "float64"
        strength_prior = StrengthPrior(
            minimum=float(self.strength_prior_min_cps_1m),
            maximum=float(self.strength_prior_max_cps_1m),
            family=str(self.strength_prior_family),
            gamma_shape=float(self.strength_prior_gamma_shape),
            gamma_scale=float(self.strength_prior_gamma_scale_cps_1m),
        )
        if strength_prior.minimum <= 0.0:
            raise ValueError("strength_prior_min_cps_1m must be finite and positive.")
        self.strength_prior_min_cps_1m = strength_prior.minimum
        self.strength_prior_max_cps_1m = strength_prior.maximum
        self.strength_prior_family = strength_prior.family
        self.strength_prior_gamma_shape = strength_prior.gamma_shape
        self.strength_prior_gamma_scale_cps_1m = strength_prior.gamma_scale
        self.structural_rj_surface_chart_max_edge_m = float(
            self.structural_rj_surface_chart_max_edge_m
        )
        if (
            not np.isfinite(self.structural_rj_surface_chart_max_edge_m)
            or self.structural_rj_surface_chart_max_edge_m <= 0.0
        ):
            raise ValueError("structural_rj_surface_chart_max_edge_m must be positive.")
        self.structural_rj_local_position_sigma_m = float(
            self.structural_rj_local_position_sigma_m
        )
        if (
            not np.isfinite(self.structural_rj_local_position_sigma_m)
            or self.structural_rj_local_position_sigma_m <= 0.0
        ):
            raise ValueError("structural_rj_local_position_sigma_m must be positive.")
        self.structural_rj_position_proposal_prior_weight = float(
            self.structural_rj_position_proposal_prior_weight
        )
        if (
            not np.isfinite(self.structural_rj_position_proposal_prior_weight)
            or self.structural_rj_position_proposal_prior_weight <= 0.0
            or self.structural_rj_position_proposal_prior_weight > 1.0
        ):
            raise ValueError(
                "structural_rj_position_proposal_prior_weight must lie in (0, 1]."
            )
        self.structural_rj_strength_proposal_prior_weight = float(
            self.structural_rj_strength_proposal_prior_weight
        )
        if (
            not np.isfinite(self.structural_rj_strength_proposal_prior_weight)
            or self.structural_rj_strength_proposal_prior_weight <= 0.0
            or self.structural_rj_strength_proposal_prior_weight > 1.0
        ):
            raise ValueError(
                "structural_rj_strength_proposal_prior_weight must lie in (0, 1]."
            )
        self.structural_rj_strength_proposal_sigma_fraction = float(
            self.structural_rj_strength_proposal_sigma_fraction
        )
        if (
            not np.isfinite(self.structural_rj_strength_proposal_sigma_fraction)
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
            "structural_rj_block_independence_probability",
            "structural_rj_multi_component_probability",
            "structural_rj_split_probability",
            "structural_rj_merge_probability",
            "structural_rj_split_global_position_probability",
            "structural_rj_merge_uniform_pair_probability",
        ):
            probability = float(getattr(self, probability_field))
            if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"{probability_field} must be in [0, 1].")
            setattr(self, probability_field, probability)
        for full_support_probability_field in (
            "structural_rj_split_global_position_probability",
            "structural_rj_merge_uniform_pair_probability",
        ):
            if getattr(self, full_support_probability_field) <= 0.0:
                raise ValueError(
                    f"{full_support_probability_field} must lie in (0, 1]."
                )
        guided_prior_probability = float(
            self.joint_guided_initialization_prior_row_probability
        )
        if (
            not np.isfinite(guided_prior_probability)
            or guided_prior_probability <= 0.0
            or guided_prior_probability > 1.0
        ):
            raise ValueError(
                "joint_guided_initialization_prior_row_probability must lie in (0, 1]."
            )
        self.joint_guided_initialization_prior_row_probability = (
            guided_prior_probability
        )
        strength_block_probability = float(self.joint_strength_block_probability)
        if (
            not np.isfinite(strength_block_probability)
            or not 0.0 <= strength_block_probability <= 1.0
        ):
            raise ValueError("joint_strength_block_probability must lie in [0, 1].")
        self.joint_strength_block_probability = strength_block_probability
        self.joint_strength_block_log_sigma = float(self.joint_strength_block_log_sigma)
        if (
            not np.isfinite(self.joint_strength_block_log_sigma)
            or self.joint_strength_block_log_sigma <= 0.0
        ):
            raise ValueError(
                "joint_strength_block_log_sigma must be finite and positive."
            )
        self.joint_strength_block_batch_size = int(self.joint_strength_block_batch_size)
        cross_transfer_probability = float(
            self.joint_cross_isotope_transfer_probability
        )
        if (
            not np.isfinite(cross_transfer_probability)
            or not 0.0 <= cross_transfer_probability <= 1.0
        ):
            raise ValueError(
                "joint_cross_isotope_transfer_probability must lie in [0, 1]."
            )
        self.joint_cross_isotope_transfer_probability = cross_transfer_probability
        cross_state_probability = float(
            self.joint_cross_isotope_state_block_probability
        )
        if (
            not np.isfinite(cross_state_probability)
            or not 0.0 <= cross_state_probability <= 1.0
        ):
            raise ValueError(
                "joint_cross_isotope_state_block_probability must lie in [0, 1]."
            )
        self.joint_cross_isotope_state_block_probability = cross_state_probability
        false_activation_probability = float(
            self.detected_isotope_false_activation_probability
        )
        if (
            not np.isfinite(false_activation_probability)
            or not 0.0 < false_activation_probability < 1.0
        ):
            raise ValueError(
                "detected_isotope_false_activation_probability must lie in (0, 1)."
            )
        self.detected_isotope_false_activation_probability = (
            false_activation_probability
        )
        cross_transfer_max_group = int(self.joint_cross_isotope_transfer_max_group)
        if cross_transfer_max_group < 1:
            raise ValueError("joint_cross_isotope_transfer_max_group must be positive.")
        self.joint_cross_isotope_transfer_max_group = cross_transfer_max_group
        self.structural_rj_merge_distance_sigma_m = float(
            self.structural_rj_merge_distance_sigma_m
        )
        if (
            not np.isfinite(self.structural_rj_merge_distance_sigma_m)
            or self.structural_rj_merge_distance_sigma_m <= 0.0
        ):
            raise ValueError(
                "structural_rj_merge_distance_sigma_m must be finite and positive."
            )
        self.structural_rj_merge_response_sigma = float(
            self.structural_rj_merge_response_sigma
        )
        if (
            not np.isfinite(self.structural_rj_merge_response_sigma)
            or self.structural_rj_merge_response_sigma <= 0.0
        ):
            raise ValueError(
                "structural_rj_merge_response_sigma must be finite and positive."
            )
        self.structural_rj_multi_component_max_group_size = _strict_nonnegative_integer(
            self.structural_rj_multi_component_max_group_size,
            name="structural_rj_multi_component_max_group_size",
        )
        if self.structural_rj_multi_component_max_group_size < 3:
            raise ValueError(
                "structural_rj_multi_component_max_group_size must be at least 3."
            )
        self.structural_cardinality_prior_policy = validate_cardinality_prior_policy(
            self.structural_cardinality_prior_policy,
            has_explicit_probabilities=(
                self.structural_cardinality_prior_probs is not None
            ),
        )
        self.structural_cardinality_tail_ratio = float(
            self.structural_cardinality_tail_ratio
        )
        if (
            self.structural_cardinality_prior_policy
            == POISSON_GEOMETRIC_TAIL_CARDINALITY_PRIOR_POLICY
            and not 0.0 < self.structural_cardinality_tail_ratio < 1.0
        ):
            raise ValueError(
                "structural_cardinality_tail_ratio must lie in (0, 1) for "
                "the thin-tail cardinality policy."
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
            cardinality_prior /= math.fsum(float(value) for value in cardinality_prior)
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
                "credible_surface_radius_threshold_m must be finite and nonnegative."
            )
        for probability_field, lower_inclusive in (
            ("converge_min_ess_ratio", False),
            ("converge_cardinality_min_probability", False),
            ("converge_max_cardinality_boundary_mass", True),
            ("converge_innovation_confidence", False),
        ):
            probability = float(getattr(self, probability_field))
            lower_valid = probability >= 0.0 if lower_inclusive else probability > 0.0
            if not np.isfinite(probability) or not lower_valid or probability > 1.0:
                lower_symbol = "[" if lower_inclusive else "("
                raise ValueError(f"{probability_field} must be in {lower_symbol}0, 1].")
            setattr(self, probability_field, probability)
        self.converge_cardinality_var_max = float(self.converge_cardinality_var_max)
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
            and len(self.structural_cardinality_prior_probs)
            != int(self.hard_max_sources) + 1
        ):
            raise ValueError(
                "structural_cardinality_prior_probs must contain "
                "hard_max_sources + 1 entries."
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

    @property
    def cardinality_capacity(self) -> int:
        """Return the finite state-array capacity for one isotope."""
        return int(
            self.max_sources if self.hard_max_sources is None else self.hard_max_sources
        )


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
            "maximum_hausdorff_bound_m": float(self.maximum_hausdorff_bound_m),
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
    if not np.isfinite(requested_bound) or requested_bound <= 0.0:
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
        raise RuntimeError("Surface atlas center coordinates and chart mapping differ.")
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
        candidate_isotopes: Sequence[str] | None = None,
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
            raise ValueError("Joint PF isotopes must be unique nonempty strings.")
        self.isotopes = list(configured_isotopes)
        configured_candidates = (
            configured_isotopes
            if candidate_isotopes is None
            else tuple(candidate_isotopes)
        )
        if (
            not configured_candidates
            or any(
                not isinstance(isotope, str) or not isotope.strip()
                for isotope in configured_candidates
            )
            or len(set(configured_candidates)) != len(configured_candidates)
            or not set(configured_isotopes).issubset(configured_candidates)
        ):
            raise ValueError(
                "Candidate isotopes must be unique nonempty strings containing "
                "every active PF isotope."
            )
        self.candidate_isotopes = list(configured_candidates)
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
            raise ValueError("detector_aperture_radius_m must be nonnegative.")
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
        if (self.source_extent_radius_m == 0.0) != (self.source_extent_samples == 1):
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
            raise ValueError("surface_diagnostic_points must be shaped (N, 3).")
        if not np.all(np.isfinite(diagnostic_points)):
            raise ValueError(
                "surface_diagnostic_points must contain only finite values."
            )
        self.surface_diagnostic_points = np.ascontiguousarray(diagnostic_points)
        self.history_estimates: List[
            Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]
        ] = []
        self._posterior_point_estimate_cache: Dict[str, PFPointEstimate] | None = None
        self._posterior_point_estimate_cache_fingerprint: str | None = None
        self.measurements: List[MeasurementRecord] = []
        self._joint_station_history: list[JointStationObservation] = []
        self._active_joint_station_history: (
            tuple[JointStationObservation, ...] | None
        ) = None
        self._active_joint_structural_geometry: StructuralGeometryBatch | None = None
        self._active_joint_tempering_prefix_count: int | None = None
        self._joint_structural_transport_cache: (
            tuple[
                object,
                object,
                object,
            ]
            | None
        ) = None
        self._joint_persistent_structural_transport_cache: (
            tuple[object, object, object] | None
        ) = None
        self._joint_persistent_structural_station_signature: tuple[str, ...] = ()
        self._joint_persistent_structural_state_sha256: str | None = None
        self.last_joint_persistent_cache_reuse_count = 0
        self.last_joint_persistent_cache_append_count = 0
        self.last_joint_persistent_cache_reindex_count = 0
        self._joint_structural_unit_transport_cache: dict[
            str,
            dict[str, dict[str, Any]],
        ] = {}
        self._joint_cuda_accepted_unit_transport_cache: dict[
            tuple[str, str], dict[str, object]
        ] = {}
        self._joint_structural_unit_cache_access_generation = 0
        self.last_joint_structural_unit_cache_hits = 0
        self.last_joint_structural_unit_cache_misses = 0
        self.last_joint_strength_grid_source_slots_before = 0
        self.last_joint_strength_grid_source_slots_after = 0
        self._joint_strength_grid_batch_size_cache: dict[
            tuple[object, ...], int
        ] = {}
        self._joint_torch_observation_context_cache: dict[
            tuple[object, ...], object
        ] = {}
        self._joint_torch_context_station_ids: tuple[int, ...] = ()
        self._joint_torch_history_layout_cache: dict[
            tuple[object, ...], tuple[object, ...]
        ] = {}
        self.last_joint_strength_grid_batch_diagnostics: dict[
            str, object
        ] = {}
        self._joint_birth_proposal_station_score_cache: dict[
            tuple[str, str],
            NDArray[np.float64],
        ] = {}
        self._joint_birth_proposal_station_score_cache_order: list[tuple[str, str]] = []
        self._joint_birth_proposal_prefix_scores: dict[
            str,
            NDArray[np.float64],
        ] = {}
        self._joint_birth_proposal_prefix_station_count = 0
        self.last_joint_birth_proposal_cache_hits = 0
        self.last_joint_birth_proposal_cache_misses = 0
        self._joint_birth_proposal_reference_mean_vb: NDArray[np.float64] | None = None
        self._joint_random_generator = named_random_generator(
            self.random_seed,
            "joint_isotope_particle_filter",
        )
        self.last_joint_resample_indices = np.zeros(0, dtype=np.int64)
        self.last_joint_temper_steps: list[dict[str, float]] = []
        self.last_joint_rejuvenation_diagnostics: list[dict[str, float]] = []
        self.last_joint_smc_soft_budget_exceeded = False
        self.last_joint_structural_mixing_incomplete = False
        self._joint_guided_initialization_applied = False
        self.last_joint_guided_initialization_ess: float | None = None
        self.last_joint_cross_isotope_attempted_weight_mass = 0.0
        self.last_joint_cross_isotope_accepted_weight_mass = 0.0
        self.last_joint_cross_isotope_rejection_diagnostics: dict[
            str,
            object,
        ] = {}
        self.last_joint_cross_isotope_state_attempted_weight_mass = 0.0
        self.last_joint_cross_isotope_state_accepted_weight_mass = 0.0
        self.last_joint_cross_isotope_state_rejection_diagnostics: dict[
            str,
            object,
        ] = {}
        self.last_joint_strength_block_attempted_weight_mass = 0.0
        self.last_joint_strength_block_accepted_weight_mass = 0.0
        self.last_joint_device_mh_acceptance_calls = 0
        self.last_joint_device_mh_acceptance_rows = 0
        self._joint_initial_product_prior_state_sha256: str | None = None
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

    def _invalidate_posterior_summary_cache(self) -> None:
        """Discard report-only summaries after posterior state changes."""
        self._posterior_point_estimate_cache = None
        self._posterior_point_estimate_cache_fingerprint = None

    def _posterior_summary_state_fingerprint(self) -> str:
        """Hash aligned weights and continuous states in batched fixed slots."""
        digest = hashlib.blake2b(digest_size=16)
        for isotope in sorted(self.filters):
            filt = self.filters[isotope]
            (
                _positions,
                strengths,
                active_mask,
                chart_ids,
                surface_uv,
            ) = filt._packed_continuous_surface_state_arrays()
            for values in (
                np.asarray(filt.continuous_weights, dtype=np.float64),
                strengths,
                active_mask,
                chart_ids,
                surface_uv,
            ):
                array = np.ascontiguousarray(values)
                digest.update(str(array.dtype).encode("ascii"))
                digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
                digest.update(array.tobytes(order="C"))
        return digest.hexdigest()

    def _cached_posterior_point_estimate(
        self,
        *,
        validate_state: bool = True,
    ) -> Dict[str, PFPointEstimate] | None:
        """Return a shallow copy of the immutable cached posterior summary."""
        if self._posterior_point_estimate_cache is None:
            return None
        if validate_state and (
            self._posterior_point_estimate_cache_fingerprint
            != self._posterior_summary_state_fingerprint()
        ):
            self._invalidate_posterior_summary_cache()
            return None
        return dict(self._posterior_point_estimate_cache)

    def _store_posterior_point_estimate(
        self,
        estimate: Mapping[str, PFPointEstimate],
    ) -> Dict[str, PFPointEstimate]:
        """Store and return one exact posterior summary generation."""
        cached = {
            str(isotope): point_estimate for isotope, point_estimate in estimate.items()
        }
        self._posterior_point_estimate_cache = cached
        self._posterior_point_estimate_cache_fingerprint = (
            self._posterior_summary_state_fingerprint()
        )
        return dict(cached)

    @staticmethod
    def _project_posterior_point_estimates(
        point_estimates: Mapping[str, PFPointEstimate],
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Project immutable posterior summaries into visualization arrays."""
        projected: Dict[
            str,
            Tuple[NDArray[np.float64], NDArray[np.float64]],
        ] = {}
        for isotope, point_estimate in point_estimates.items():
            projected[str(isotope)] = (
                np.asarray(
                    [mode.position_medoid_xyz for mode in point_estimate.modes],
                    dtype=float,
                ).reshape(-1, 3),
                np.asarray(
                    [
                        mode.strength_representative_cps_1m
                        for mode in point_estimate.modes
                    ],
                    dtype=float,
                ),
            )
        return projected

    def visualization_estimates(
        self,
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Return the latest committed exact summary without recomputation.

        Frames collected before the first completed station intentionally show
        no point estimates. Particle clouds remain available, while the live
        rendering path never triggers the quadratic intrinsic-medoid report.
        """
        cached = self._cached_posterior_point_estimate(
            validate_state=False,
        )
        if cached is None:
            return {
                str(isotope): (
                    np.zeros((0, 3), dtype=float),
                    np.zeros(0, dtype=float),
                )
                for isotope in self.filters
            }
        return self._project_posterior_point_estimates(cached)

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
            prefix_payload = self._surface_diagnostic_response_prefix_cache.get(
                prefix_key
            )
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
            filt.set_joint_target_evaluator(self._joint_structural_target_evaluator)
            filt.set_joint_strength_grid_target_evaluator(
                self._joint_structural_strength_grid_target_evaluator
            )
            filt.set_joint_proposal_evaluator(self._joint_structural_proposal_evaluator)
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
        identity_root = self._joint_row_identity_root(particle_count=particle_count)
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
        self._invalidate_posterior_summary_cache()
        state_matrix = np.asarray(
            self._joint_mixing_snapshot()["state_matrix"],
            dtype=np.float64,
        )
        self._joint_initial_product_prior_state_sha256 = hashlib.sha256(
            np.ascontiguousarray(state_matrix).tobytes(order="C")
        ).hexdigest()

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
            raise RuntimeError("Joint PF row identity contract is not initialized.")
        reference_identities: tuple[JointRowIdentity, ...] | None = None
        for isotope in order:
            identities: list[JointRowIdentity] = []
            for row, particle in enumerate(self.filters[isotope].continuous_particles):
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
                if (
                    len({identity.row_sha256 for identity in identity_tuple})
                    != particle_count
                ):
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
            or any(character not in "0123456789abcdef" for character in digest.lower())
            for digest in digests
        ):
            raise RuntimeError(
                "Every joint isotope filter requires a valid surface-atlas digest."
            )
        if len(set(digest.lower() for digest in digests)) != 1:
            raise RuntimeError(
                "Joint isotope rows cannot use different continuous surface atlases."
            )
        return digests[0].lower()

    def initialize_joint_particle_filters(self) -> str:
        """Initialize every joint PF row and return its shared atlas digest.

        Measurement poses are registered before the first station is observed,
        while the particle filters themselves are intentionally built lazily.
        Runtime preflight must use this method instead of inspecting ``filters``
        directly so atlas validation cannot race that lazy initialization.
        """
        self._ensure_kernel_cache()
        self._assert_joint_particle_alignment()
        return self._assert_joint_surface_atlas_alignment()

    def continuous_surface_atlas(self) -> Any:
        """Return the one authoritative atlas shared by all isotope filters."""
        if not self.filters:
            raise RuntimeError(
                "The continuous surface atlas is unavailable before PF initialization."
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
        self._invalidate_posterior_summary_cache()
        values = np.asarray(normalized_log_weights, dtype=np.float64).reshape(-1)
        self._assert_joint_particle_alignment()
        if (
            values.size
            != len(self.filters[self.joint_isotope_order()[0]].continuous_particles)
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
            raise TypeError("generative_contract_hash_sha256 must be a JSON string.")
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
                raise ValueError("Full-spectrum station live times must be positive.")
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
        total, uncollided, features = self._joint_station_transport_components_torch(
            station
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
        covered = np.concatenate([value[0] for value in layout.values()])
        active_names = frozenset(self.joint_isotope_order())
        expected = np.asarray(
            [
                column
                for column, payload in enumerate(line_identity)
                if str(payload["isotope"]) in active_names
            ],
            dtype=np.int64,
        )
        if not np.array_equal(
            np.sort(covered),
            expected,
        ):
            raise RuntimeError(
                "Full-spectrum line layout does not cover every active-isotope "
                "global line."
            )
        return layout

    def _joint_isotope_station_transport_components_torch(
        self,
        station: JointStationObservation,
        isotope: str,
        *,
        particle_indices: NDArray[np.int64] | None = None,
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Return selected isotope rows in the global fixed-slot layout."""
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
        filt = self.filters[isotope_key]
        (
            positions,
            strengths,
            active_mask,
            chart_ids,
            _surface_uv,
        ) = filt._packed_continuous_surface_state_arrays()
        if particle_indices is not None:
            raw_indices = np.asarray(particle_indices)
            if raw_indices.ndim != 1 or not np.issubdtype(
                raw_indices.dtype, np.integer
            ):
                raise ValueError("particle_indices must be a 1-D integer array.")
            indices = np.asarray(raw_indices, dtype=np.int64)
            if (
                np.unique(indices).size != indices.size
                or np.any(indices < 0)
                or np.any(indices >= positions.shape[0])
            ):
                raise ValueError("particle_indices contain invalid PF rows.")
            positions = positions[indices]
            strengths = strengths[indices]
            active_mask = active_mask[indices]
            chart_ids = chart_ids[indices]
        view_count = int(station.fe_indices.size)
        station_geometry = StructuralGeometryBatch(
            detector_positions=np.repeat(
                np.asarray(
                    station.detector_position_xyz_m,
                    dtype=np.float64,
                ).reshape(1, 3),
                view_count,
                axis=0,
            ),
            fe_indices=station.fe_indices,
            pb_indices=station.pb_indices,
            live_times=station.live_times_s,
            station_sequence_ids=np.full(
                view_count,
                int(station.station_sequence_id),
                dtype=np.int64,
            ),
        )
        unit_components = self._joint_cached_continuous_unit_components(
            filt=filt,
            data=station_geometry,
            positions_s3=positions[active_mask],
            chart_ids_s=chart_ids[active_mask],
            positive_line_indices=local_indices,
        )
        particle_count, slot_count = active_mask.shape
        local_line_count = int(local_indices.size)
        dense_components = []
        for values in unit_components:
            dense = np.zeros(
                (
                    particle_count,
                    slot_count,
                    view_count,
                    local_line_count,
                ),
                dtype=np.float64,
            )
            dense[active_mask] = np.transpose(values, (1, 0, 2))
            dense_components.append(np.transpose(dense, (0, 2, 1, 3)))
        total_numpy = (
            dense_components[0]
            * strengths[:, None, :, None]
            * branching_weights.reshape(1, 1, 1, -1)
        )
        uncollided_numpy = (
            dense_components[1]
            * strengths[:, None, :, None]
            * branching_weights.reshape(1, 1, 1, -1)
        )
        feature_numpy = np.stack(
            (
                dense_components[2],
                dense_components[3],
                dense_components[4],
                dense_components[5],
            ),
            axis=-1,
        )
        device = None
        if filt._can_use_gpu():
            from pf import gpu_utils

            device = gpu_utils.resolve_device(filt.config.gpu_device)
        total_local = torch.as_tensor(
            total_numpy,
            dtype=torch.float64,
            device=device,
        )
        uncollided_local = torch.as_tensor(
            uncollided_numpy,
            dtype=torch.float64,
            device=device,
        )
        feature_local = torch.as_tensor(
            feature_numpy,
            dtype=torch.float64,
            device=device,
        )
        expected_local_shape = tuple(total_local.shape)
        expected_slots = self.pf_config.cardinality_capacity
        if (
            total_local.ndim != 4
            or int(total_local.shape[2]) != expected_slots
            or tuple(uncollided_local.shape) != expected_local_shape
            or tuple(feature_local.shape) != expected_local_shape + (feature_count,)
            or int(total_local.shape[-1]) != int(local_indices.size)
        ):
            raise RuntimeError(
                "Full-spectrum isotope transport must use the configured "
                "fixed source-slot layout."
            )
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
                uncollided_local = uncollided_local.to(device=reference_device)
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
        total, uncollided, features = self._joint_station_transport_components_torch(
            station
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
            torch.any(torch.isinf(result) & (result > 0.0)).detach().cpu().item()
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

    def _joint_station_prefix_log_likelihood_torch(
        self,
        station: JointStationObservation,
    ) -> "torch.Tensor":
        """Evaluate exact shared-latent likelihoods for all view prefixes."""
        model = self._full_spectrum_model()
        total, uncollided, features = self._joint_station_transport_components_torch(
            station
        )
        result = model.prefix_log_likelihood_torch(
            station.spectrum_vb,
            total,
            uncollided,
            features,
            station.live_times_s,
        )
        expected_shape = (
            int(station.fe_indices.size) + 1,
            int(total.shape[0]),
        )
        if tuple(result.shape) != expected_shape:
            raise RuntimeError(
                "Full-spectrum prefix likelihood returned an invalid shape."
            )
        import torch

        if bool(torch.any(torch.isnan(result)).item()) or bool(
            torch.any(torch.isinf(result) & (result > 0.0)).item()
        ):
            raise RuntimeError(
                "Full-spectrum prefix likelihood is numerically invalid."
            )
        if not bool(torch.all(result[0] == 0.0).item()):
            raise RuntimeError(
                "The empty full-spectrum prefix must have zero log likelihood."
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
            [np.asarray(station.live_times_s, dtype=np.float64) for station in stations]
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
        """Cache source-resolved transport components for conditional RJ.

        CUDA runs retain the immutable station history on the device for the
        whole Gibbs sweep.  Candidate states are much smaller than this cache,
        so keeping the history resident removes repeated device-to-host and
        host-to-device copies without changing any transport or likelihood
        arithmetic.
        """
        for filt in self.filters.values():
            filt._clear_continuous_rj_device_state()
        active = tuple(stations)
        station_signature = tuple(
            self._joint_station_cache_signature(station) for station in active
        )
        state_sha256 = self._joint_structural_state_sha256()
        persistent = self._joint_persistent_structural_transport_cache
        persistent_signature = self._joint_persistent_structural_station_signature
        if (
            persistent is not None
            and self._joint_persistent_structural_state_sha256 == state_sha256
            and persistent_signature == station_signature
        ):
            self._joint_structural_transport_cache = persistent
            self.last_joint_persistent_cache_reuse_count += 1
            return
        can_append = (
            persistent is not None
            and self._joint_persistent_structural_state_sha256 == state_sha256
            and len(persistent_signature) < len(station_signature)
            and station_signature[: len(persistent_signature)] == persistent_signature
        )
        pending_stations = active[len(persistent_signature) :] if can_append else active
        station_components = [
            self._joint_station_transport_components_torch(station)
            for station in pending_stations
        ]
        if not station_components:
            raise RuntimeError("Structural cache refresh has no station data.")
        if self.pf_config.use_gpu:
            import torch

            appended = tuple(
                torch.cat(
                    [components[index] for components in station_components],
                    dim=1,
                ).contiguous()
                for index in range(3)
            )
        else:
            appended = tuple(
                np.concatenate(
                    [
                        components[index]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64, copy=False)
                        for components in station_components
                    ],
                    axis=1,
                )
                for index in range(3)
            )
        if can_append:
            if hasattr(persistent[0], "detach"):
                import torch

                refreshed = tuple(
                    torch.cat((old, new), dim=1).contiguous()
                    for old, new in zip(persistent, appended, strict=True)
                )
            else:
                refreshed = tuple(
                    np.concatenate((old, new), axis=1)
                    for old, new in zip(persistent, appended, strict=True)
                )
            self.last_joint_persistent_cache_append_count += 1
        else:
            refreshed = appended
        self._joint_structural_transport_cache = refreshed
        self._joint_persistent_structural_transport_cache = refreshed
        self._joint_persistent_structural_station_signature = station_signature
        self._joint_persistent_structural_state_sha256 = state_sha256

    def _joint_structural_state_sha256(self) -> str:
        """Hash compact accepted chart/UV/strength state without transport."""
        digest = hashlib.sha256()
        digest.update(b"joint_structural_accepted_state_v1")
        for isotope in self.joint_isotope_order():
            filt = self.filters[isotope]
            _, strengths, mask, chart_ids, surface_uv = (
                filt._packed_continuous_surface_state_arrays()
            )
            digest.update(str(isotope).encode("utf-8"))
            for values in (strengths, mask, chart_ids, surface_uv):
                array = np.ascontiguousarray(values)
                digest.update(str(array.dtype).encode("ascii"))
                digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
                digest.update(array.tobytes(order="C"))
        return digest.hexdigest()

    @staticmethod
    def _joint_station_cache_signature(
        station: JointStationObservation,
    ) -> str:
        """Return the immutable geometry signature of one station cache slab."""
        digest = hashlib.sha256()
        digest.update(b"joint_station_transport_geometry_v1")
        digest.update(
            np.asarray(
                station.detector_position_xyz_m,
                dtype=np.float64,
            ).tobytes()
        )
        for values in (
            station.fe_indices,
            station.pb_indices,
            station.live_times_s,
        ):
            array = np.ascontiguousarray(values)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(array.tobytes(order="C"))
        digest.update(
            np.asarray(
                [int(station.station_sequence_id)],
                dtype=np.int64,
            ).tobytes()
        )
        return digest.hexdigest()

    @classmethod
    def _joint_station_cache_signatures(
        cls,
        stations: Sequence[JointStationObservation],
    ) -> tuple[str, ...] | None:
        """Return station signatures, or disable persistence for debug stubs."""
        try:
            return tuple(
                cls._joint_station_cache_signature(station) for station in stations
            )
        except AttributeError:
            return None

    def _refresh_joint_structural_transport_cache_isotope(
        self,
        stations: Sequence[JointStationObservation],
        isotope: str,
        *,
        particle_indices: NDArray[np.int64] | None = None,
    ) -> None:
        """Refresh moved rows of one isotope's accepted-state cache slice."""
        cache = self._joint_structural_transport_cache
        if cache is None:
            raise RuntimeError(
                "Incremental structural transport refresh requires a cache."
            )
        order = self.joint_isotope_order()
        isotope_key = str(isotope)
        if isotope_key not in order:
            raise KeyError(f"Unknown joint PF isotope: {isotope_key!r}.")
        if particle_indices is None:
            indices = np.arange(
                int(self.pf_config.num_particles),
                dtype=np.int64,
            )
        else:
            raw_indices = np.asarray(particle_indices)
            if raw_indices.ndim != 1 or not np.issubdtype(
                raw_indices.dtype, np.integer
            ):
                raise ValueError("particle_indices must be a 1-D integer array.")
            indices = np.asarray(raw_indices, dtype=np.int64)
        if indices.size == 0:
            return
        station_components = [
            self._joint_isotope_station_transport_components_torch(
                station,
                isotope_key,
                particle_indices=indices,
            )
            for station in stations
        ]
        cache_is_torch = hasattr(cache[0], "detach")
        if cache_is_torch:
            import torch

            refreshed = tuple(
                torch.cat(
                    [components[index] for components in station_components],
                    dim=1,
                ).contiguous()
                for index in range(3)
            )
        else:
            refreshed = tuple(
                np.concatenate(
                    [
                        components[index]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64, copy=False)
                        for components in station_components
                    ],
                    axis=1,
                )
                for index in range(3)
            )
        slots_per_isotope = self.pf_config.cardinality_capacity
        slot_start = order.index(isotope_key) * slots_per_isotope
        slot_stop = slot_start + slots_per_isotope
        mutable_cache = list(cache)
        for cached_values, refreshed_values in zip(
            mutable_cache, refreshed, strict=True
        ):
            if (
                int(cached_values.shape[0]) != int(self.pf_config.num_particles)
                or int(refreshed_values.shape[0]) != int(indices.size)
                or int(cached_values.shape[1]) != int(refreshed_values.shape[1])
                or int(cached_values.shape[2]) < slot_stop
                or int(refreshed_values.shape[2]) != slots_per_isotope
                or tuple(cached_values.shape[3:]) != tuple(refreshed_values.shape[3:])
            ):
                raise RuntimeError(
                    "Incremental isotope transport cache shapes disagree."
                )
            if cache_is_torch:
                import torch

                index_tensor = torch.as_tensor(
                    indices,
                    device=cached_values.device,
                    dtype=torch.long,
                )
                cached_values[:, :, slot_start:slot_stop, ...].index_copy_(
                    0, index_tensor, refreshed_values
                )
            else:
                cached_values[indices, :, slot_start:slot_stop, ...] = refreshed_values
        self._joint_structural_transport_cache = tuple(mutable_cache)
        self.filters[isotope_key]._clear_continuous_rj_device_state()
        station_signature = self._joint_station_cache_signatures(stations)
        if station_signature is None:
            self._joint_persistent_structural_transport_cache = None
            self._joint_persistent_structural_station_signature = ()
            self._joint_persistent_structural_state_sha256 = None
        else:
            self._joint_persistent_structural_transport_cache = (
                self._joint_structural_transport_cache
            )
            self._joint_persistent_structural_station_signature = station_signature
            self._joint_persistent_structural_state_sha256 = (
                self._joint_structural_state_sha256()
            )

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
        newest_prefix_count: int | None = None,
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
        prefix_count = None if newest_prefix_count is None else int(newest_prefix_count)
        newest_view_count = int(stations[-1].fe_indices.size)
        if prefix_count is not None and not (1 <= prefix_count <= newest_view_count):
            raise ValueError(
                "newest_prefix_count must identify a nonempty newest-station "
                "view prefix."
            )
        layout_key = (
            tuple(id(station) for station in stations),
            bool(prefix_count is not None),
        )
        cached_layout = self._joint_torch_history_layout_cache.get(layout_key)
        if cached_layout is None:
            newest_slice: slice | None = None
            grouped_lists: dict[
                tuple[int, bytes],
                list[tuple[JointStationObservation, int, int, bool]],
            ] = {}
            view_start = 0
            for station_index, station in enumerate(stations):
                view_count = int(station.fe_indices.size)
                view_stop = view_start + view_count
                if (
                    prefix_count is not None
                    and station_index == len(stations) - 1
                ):
                    newest_slice = slice(view_start, view_stop)
                    view_start = view_stop
                    continue
                live_times = np.ascontiguousarray(
                    station.live_times_s,
                    dtype=np.float64,
                )
                if live_times.shape != (view_count,):
                    raise ValueError(
                        "Joint-history station live times must align with views."
                    )
                key = (view_count, live_times.tobytes(order="C"))
                grouped_lists.setdefault(key, []).append(
                    (
                        station,
                        view_start,
                        view_stop,
                        station_index == len(stations) - 1,
                    )
                )
                view_start = view_stop
            grouped = tuple(tuple(entries) for entries in grouped_lists.values())
            cached_layout = (grouped, newest_slice, view_start)
            self._joint_torch_history_layout_cache[layout_key] = cached_layout
        grouped, newest_slice, view_start = cached_layout
        if view_start != total_views:
            raise ValueError(
                "Full-spectrum transport views differ from station history."
            )
        for grouped_entries in grouped:
            entries = tuple(
                entry
                for entry in grouped_entries
                if not (bool(entry[3]) and beta == 0.0)
            )
            if not entries:
                continue
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
                [beta if bool(entry[3]) else 1.0 for entry in entries],
                dtype=np.float64,
            )
            result += np.sum(
                powers[:, None] * group_ll[:, 0, :],
                axis=0,
            )
        if prefix_count is not None:
            if newest_slice is None:
                raise RuntimeError("Newest-station prefix geometry was not selected.")
            station = stations[-1]
            if filt._can_use_gpu():
                from pf import gpu_utils
                import torch

                device = gpu_utils.resolve_device(filt.config.gpu_device)
                prefix_ll = (
                    model.prefix_log_likelihood_torch(
                        station.spectrum_vb,
                        torch.as_tensor(
                            total[:, newest_slice, ...],
                            dtype=torch.float64,
                            device=device,
                        ),
                        torch.as_tensor(
                            uncollided[:, newest_slice, ...],
                            dtype=torch.float64,
                            device=device,
                        ),
                        torch.as_tensor(
                            features[:, newest_slice, ...],
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
                prefix_ll = np.asarray(
                    model.prefix_log_likelihood_numpy(
                        station.spectrum_vb,
                        total[:, newest_slice, ...],
                        uncollided[:, newest_slice, ...],
                        features[:, newest_slice, ...],
                        station.live_times_s,
                    ),
                    dtype=np.float64,
                )
            expected_prefix_shape = (
                newest_view_count + 1,
                particle_count,
            )
            if prefix_ll.shape != expected_prefix_shape:
                raise RuntimeError("Newest-station prefix likelihood shape is invalid.")
            result += (1.0 - beta) * prefix_ll[prefix_count - 1] + beta * prefix_ll[
                prefix_count
            ]
        if np.any(np.isnan(result)) or np.any(np.isposinf(result)):
            raise RuntimeError(
                "Joint history conditional likelihood is numerically invalid."
            )
        return result

    def _joint_history_log_likelihood_torch(
        self,
        *,
        filt: IsotopeParticleFilter,
        stations: Sequence[JointStationObservation],
        total_nvsl: object,
        uncollided_nvsl: object,
        features_nvslf: object,
        target_beta: float,
        newest_prefix_count: int | None = None,
    ) -> object:
        """Evaluate the station history while keeping all state arrays on Torch.

        This is the device-resident equivalent of
        :meth:`_joint_history_log_likelihood_numpy`.  It preserves the same
        station grouping, target powers, model call, and summation order.
        """
        import torch

        if not filt._can_use_gpu():
            raise RuntimeError(
                "Device-resident joint likelihood requires the Torch backend."
            )
        beta = float(target_beta)
        if not np.isfinite(beta) or not 0.0 <= beta <= 1.0:
            raise ValueError("Joint history target_beta must lie in [0, 1].")
        model = self._full_spectrum_model()
        total = torch.as_tensor(total_nvsl)
        uncollided = torch.as_tensor(
            uncollided_nvsl,
            device=total.device,
            dtype=total.dtype,
        )
        features = torch.as_tensor(
            features_nvslf,
            device=total.device,
            dtype=total.dtype,
        )
        feature_count = len(tuple(model.transport_feature_order))
        total_views = sum(int(station.fe_indices.size) for station in stations)
        if (
            total.dtype != torch.float64
            or total.ndim != 4
            or tuple(uncollided.shape) != tuple(total.shape)
            or tuple(features.shape) != tuple(total.shape) + (feature_count,)
            or int(total.shape[0]) <= 0
            or int(total.shape[1]) != total_views
        ):
            raise ValueError(
                "Torch joint-history arrays must align with every station "
                "view and configured transport feature."
            )
        particle_count = int(total.shape[0])
        result = torch.zeros(
            particle_count,
            device=total.device,
            dtype=total.dtype,
        )
        prefix_count = None if newest_prefix_count is None else int(newest_prefix_count)
        newest_view_count = int(stations[-1].fe_indices.size)
        if prefix_count is not None and not (1 <= prefix_count <= newest_view_count):
            raise ValueError(
                "newest_prefix_count must identify a nonempty newest-station "
                "view prefix."
            )
        layout_key = (
            tuple(id(station) for station in stations),
            bool(prefix_count is not None),
        )
        cached_layout = self._joint_torch_history_layout_cache.get(layout_key)
        if cached_layout is None:
            newest_slice = None
            grouped_lists: dict[
                tuple[int, bytes],
                list[tuple[JointStationObservation, int, int, bool]],
            ] = {}
            view_start = 0
            for station_index, station in enumerate(stations):
                view_count = int(station.fe_indices.size)
                view_stop = view_start + view_count
                if (
                    prefix_count is not None
                    and station_index == len(stations) - 1
                ):
                    newest_slice = slice(view_start, view_stop)
                    view_start = view_stop
                    continue
                live_times = np.ascontiguousarray(
                    station.live_times_s,
                    dtype=np.float64,
                )
                if live_times.shape != (view_count,):
                    raise ValueError(
                        "Joint-history station live times must align with views."
                    )
                key = (view_count, live_times.tobytes(order="C"))
                grouped_lists.setdefault(key, []).append(
                    (
                        station,
                        view_start,
                        view_stop,
                        station_index == len(stations) - 1,
                    )
                )
                view_start = view_stop
            grouped = tuple(tuple(entries) for entries in grouped_lists.values())
            cached_layout = (grouped, newest_slice, view_start)
            self._joint_torch_history_layout_cache[layout_key] = cached_layout
        grouped, newest_slice, view_start = cached_layout
        if view_start != total_views:
            raise ValueError(
                "Full-spectrum transport views differ from station history."
            )
        for grouped_entries in grouped:
            entries = tuple(
                entry
                for entry in grouped_entries
                if not (bool(entry[3]) and beta == 0.0)
            )
            if not entries:
                continue
            view_count = int(entries[0][2] - entries[0][1])
            first_start = int(entries[0][1])
            last_stop = int(entries[-1][2])
            contiguous = all(
                int(entry[1]) == first_start + index * view_count
                and int(entry[2]) == first_start + (index + 1) * view_count
                for index, entry in enumerate(entries)
            )

            def _station_action_axis(values: object) -> object:
                """Return station x particle x view without leaving Torch."""
                tensor = torch.as_tensor(values)
                trailing_shape = tuple(int(value) for value in tensor.shape[2:])
                if contiguous:
                    block = tensor[:, first_start:last_stop, ...]
                    reshaped = block.reshape(
                        particle_count,
                        len(entries),
                        view_count,
                        *trailing_shape,
                    )
                    return torch.movedim(reshaped, 1, 0)
                return torch.stack(
                    [
                        tensor[:, int(entry[1]) : int(entry[2]), ...]
                        for entry in entries
                    ],
                    dim=0,
                )

            observation_key = (
                tuple(id(entry[0]) for entry in entries),
                str(total.device),
                str(total.dtype),
            )
            prepared_observation = (
                self._joint_torch_observation_context_cache.get(
                    observation_key
                )
            )
            if prepared_observation is None:
                observed = torch.as_tensor(
                    np.stack(
                        [
                            np.asarray(
                                entry[0].spectrum_vb,
                                dtype=np.float64,
                            )
                            for entry in entries
                        ],
                        axis=0,
                    )[:, None, :, :],
                    device=total.device,
                    dtype=total.dtype,
                )
                prepared_observation = model.prepare_cross_observation_torch(
                    observed,
                    reference=total,
                )
                self._joint_torch_observation_context_cache[
                    observation_key
                ] = prepared_observation
            else:
                observed = prepared_observation.observed_asvb
            group_ll = model.cross_log_likelihood_torch(
                observed,
                _station_action_axis(total),
                _station_action_axis(uncollided),
                _station_action_axis(features),
                entries[0][0].live_times_s,
                action_chunk_size=min(
                    len(entries),
                    JOINT_HISTORY_STATION_ACTION_BATCH_SIZE,
                ),
                prepared_observation=prepared_observation,
            )
            group_ll = torch.as_tensor(
                group_ll,
                device=total.device,
                dtype=total.dtype,
            )
            expected_shape = (len(entries), 1, particle_count)
            if tuple(group_ll.shape) != expected_shape:
                raise RuntimeError(
                    "Torch station-history likelihood shape is invalid."
                )
            powers = torch.as_tensor(
                [beta if bool(entry[3]) else 1.0 for entry in entries],
                device=total.device,
                dtype=total.dtype,
            )
            result = result + torch.sum(
                powers[:, None] * group_ll[:, 0, :],
                dim=0,
            )
        if prefix_count is not None:
            if newest_slice is None:
                raise RuntimeError("Newest-station prefix geometry was not selected.")
            station = stations[-1]
            prefix_ll = model.prefix_log_likelihood_torch(
                station.spectrum_vb,
                total[:, newest_slice, ...],
                uncollided[:, newest_slice, ...],
                features[:, newest_slice, ...],
                station.live_times_s,
            )
            prefix_ll = torch.as_tensor(
                prefix_ll,
                device=total.device,
                dtype=total.dtype,
            )
            expected_prefix_shape = (
                newest_view_count + 1,
                particle_count,
            )
            if tuple(prefix_ll.shape) != expected_prefix_shape:
                raise RuntimeError(
                    "Torch newest-station prefix likelihood shape is invalid."
                )
            result = result + (
                (1.0 - beta) * prefix_ll[prefix_count - 1]
                + beta * prefix_ll[prefix_count]
            )
        invalid_result = torch.stack(
            (
                torch.any(torch.isnan(result)),
                torch.any(torch.isinf(result) & (result > 0.0)),
            )
        ).any()
        if bool(invalid_result.item()):
            raise RuntimeError("Torch joint-history likelihood is numerically invalid.")
        return result

    @staticmethod
    def _joint_birth_proposal_station_digest(
        *,
        filt: IsotopeParticleFilter,
        station: JointStationObservation,
        strength_grid: NDArray[np.float64],
        reference_mean_vb: NDArray[np.float64] | None = None,
    ) -> str:
        """Hash every immutable input to one station proposal-score grid."""
        digest = hashlib.sha256(b"joint_full_spectrum_birth_proposal_station_v1\0")
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
        if reference_mean_vb is None:
            digest.update(b"proposal_reference=physical_background\0")
        else:
            reference = np.ascontiguousarray(
                reference_mean_vb,
                dtype="<f8",
            )
            digest.update(b"proposal_reference=sequential_mean\0")
            digest.update(np.asarray(reference.shape, dtype="<i8").tobytes())
            digest.update(reference.tobytes(order="C"))
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
                for values in (self._joint_birth_proposal_station_score_cache.values())
            )
        )

    def _store_joint_birth_proposal_station_scores(
        self,
        cache_key: tuple[str, str],
        score_grid: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Store one immutable score grid under a strict LRU memory bound."""
        scores = np.ascontiguousarray(score_grid, dtype=np.float64)
        maximum_bytes = int(self.pf_config.structural_rj_proposal_score_cache_max_bytes)
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
            oldest = self._joint_birth_proposal_station_score_cache_order.pop(0)
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
        reference_mean_vb: NDArray[np.float64] | None = None,
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
                reference_mean_vb=reference_mean_vb,
            ),
        )
        cached = self._joint_birth_proposal_station_score_cache.get(cache_key)
        expected_shape = (int(centers.shape[0]), int(strengths.size))
        if cached is not None:
            if cached.shape != expected_shape:
                raise RuntimeError(
                    "Cached birth-proposal score grid has an invalid shape."
                )
            self.last_joint_birth_proposal_cache_hits += 1
            self._joint_birth_proposal_station_score_cache_order = [
                key
                for key in (self._joint_birth_proposal_station_score_cache_order)
                if key != cache_key
            ]
            self._joint_birth_proposal_station_score_cache_order.append(cache_key)
            return cached

        self.last_joint_birth_proposal_cache_misses += 1
        model = self._full_spectrum_model()
        line_count = len(tuple(model.line_identity))
        feature_count = len(tuple(model.transport_feature_order))
        layout = self._joint_line_layout()
        global_columns, local_indices, branching_weights = layout[str(filt.isotope)]
        target_line_mask = np.zeros(line_count, dtype=np.bool_)
        target_line_mask[global_columns] = True
        view_count = int(station.fe_indices.size)
        geometry = self._joint_station_structural_geometry(station)
        score_cg = np.empty(expected_shape, dtype=np.float64)
        batch_size = int(self.pf_config.structural_rj_proposal_chart_batch_size)
        for chart_start in range(0, int(centers.shape[0]), batch_size):
            chart_stop = min(
                chart_start + batch_size,
                int(centers.shape[0]),
            )
            batch_centers = centers[chart_start:chart_stop]
            components = filt._continuous_rj_line_transport_component_columns(
                geometry,
                batch_centers,
                local_indices,
                chart_ids=np.arange(
                    chart_start,
                    chart_stop,
                    dtype=np.int64,
                ),
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
                strengths[None, :, None, None] * branching_weights[None, None, None, :]
            )
            total_local = (
                np.transpose(unit_total, (1, 0, 2))[:, None, :, :] * scale
            ).reshape(
                candidate_count,
                view_count,
                int(local_indices.size),
            )
            uncollided_local = (
                np.transpose(unit_uncollided, (1, 0, 2))[:, None, :, :] * scale
            ).reshape(
                candidate_count,
                view_count,
                int(local_indices.size),
            )
            feature_local = np.broadcast_to(
                np.transpose(unit_features, (1, 0, 2, 3))[:, None, :, :, :],
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
            uncollided[..., global_columns] = uncollided_local[:, :, None, :]
            features[..., global_columns, :] = feature_local[:, :, None, :, :]
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
                    reference_mean_vb=(
                        None
                        if reference_mean_vb is None
                        else torch.as_tensor(
                            reference_mean_vb,
                            dtype=torch.float64,
                            device=device,
                        )
                    ),
                )
                batch_scores = (
                    scores.detach().cpu().numpy().astype(np.float64, copy=False)
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
                        reference_mean_vb=reference_mean_vb,
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

    def _strength_birth_proposal_grid(
        self,
    ) -> tuple[NDArray[np.float64], float]:
        """Return a finite design grid without truncating prior support."""
        prior = StrengthPrior(
            minimum=float(self.pf_config.strength_prior_min_cps_1m),
            maximum=float(self.pf_config.strength_prior_max_cps_1m),
            family=str(self.pf_config.strength_prior_family),
            gamma_shape=float(self.pf_config.strength_prior_gamma_shape),
            gamma_scale=float(self.pf_config.strength_prior_gamma_scale_cps_1m),
        )
        upper = prior.finite_upper_quantile(0.995)
        grid = np.linspace(
            prior.minimum,
            upper,
            int(self.pf_config.structural_rj_strength_proposal_grid_size),
            dtype=np.float64,
        )
        return grid, float(prior.mean)

    def full_spectrum_isotope_detection_score_grids(
        self,
        records: Sequence[Sequence[object]],
        *,
        pose_idx: int,
        generative_contract_hash_sha256: str,
    ) -> Dict[str, NDArray[np.float64]]:
        """Return truth-free chart-by-strength detection scores for one station."""
        if not records:
            raise ValueError(
                "An isotope-detection station must contain at least one view."
            )
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        self._assert_joint_particle_alignment()
        station = self._joint_station_from_spectrum_records(
            records,
            pose_idx=pose_idx,
            station_sequence_id=0,
            generative_contract_hash_sha256=(generative_contract_hash_sha256),
        )
        strength_grid, _ = self._strength_birth_proposal_grid()
        result: Dict[str, NDArray[np.float64]] = {}
        for isotope in self.joint_isotope_order():
            filt = self.filters[isotope]
            atlas = filt._structural_rj_surface_atlas
            if atlas is None:
                raise RuntimeError(
                    "Full-spectrum isotope detection requires a surface atlas."
                )
            centers = np.asarray(
                atlas.geometry.centers_xyz,
                dtype=np.float64,
            )
            result[isotope] = np.asarray(
                self._joint_station_birth_proposal_score_grid(
                    filt=filt,
                    station=station,
                    chart_centers_xyz=centers,
                    strength_grid=strength_grid,
                    reference_mean_vb=None,
                ),
                dtype=np.float64,
            ).copy()
        return result

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
            raise RuntimeError("Joint full-spectrum proposal target is not active.")
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
        strength_grid, midpoint = self._strength_birth_proposal_grid()
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
        active_prefix = self._active_joint_tempering_prefix_count
        if active_prefix is None:
            newest_station_power = beta
        else:
            newest_view_count = int(stations[-1].fe_indices.size)
            newest_station_power = (int(active_prefix) - 1 + beta) / float(
                newest_view_count
            )
        has_active_likelihood = len(stations) > 1 or newest_station_power > 0.0
        if not has_active_likelihood:
            return (
                np.zeros(chart_count, dtype=np.float64),
                np.full(chart_count, midpoint, dtype=np.float64),
                False,
            )
        completed_count = len(self._joint_station_history)
        if (
            len(stations) != completed_count + 1
            or self._joint_birth_proposal_prefix_station_count != completed_count
        ):
            raise RuntimeError(
                "Birth-proposal prefix does not match the completed station history."
            )
        prefix = self._joint_birth_proposal_prefix_scores.get(str(filt.isotope))
        expected_shape = (chart_count, strength_grid.size)
        if prefix is None:
            if completed_count:
                raise RuntimeError(
                    "Birth-proposal prefix is missing for a configured isotope."
                )
            score_cg = np.zeros(expected_shape, dtype=np.float64)
        else:
            score_cg = np.asarray(prefix, dtype=np.float64).copy()
            if score_cg.shape != expected_shape or np.any(~np.isfinite(score_cg)):
                raise RuntimeError("Birth-proposal prefix score grid is invalid.")
        if newest_station_power > 0.0:
            if (
                self._joint_birth_proposal_reference_mean_vb is not None
                and completed_count != 0
            ):
                raise RuntimeError(
                    "Sequential guided residuals are valid only for the first station."
                )
            score_cg += (
                newest_station_power
                * self._joint_station_birth_proposal_score_grid(
                    filt=filt,
                    station=stations[-1],
                    chart_centers_xyz=centers,
                    strength_grid=strength_grid,
                    reference_mean_vb=(self._joint_birth_proposal_reference_mean_vb),
                )
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
        alignment = np.exp(np.clip(best_scores - maximum, -745.0, 0.0))
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
            raise RuntimeError("Birth-proposal prefix promotion is out of sequence.")
        strength_grid, _ = self._strength_birth_proposal_grid()
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
                        "Birth-proposal prefix is missing for a configured isotope."
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
                combined = np.asarray(previous, dtype=np.float64) + np.asarray(
                    station_scores, dtype=np.float64
                )
            if np.any(~np.isfinite(combined)):
                raise RuntimeError("Birth-proposal prefix contains non-finite scores.")
            combined = np.ascontiguousarray(combined, dtype=np.float64)
            combined.setflags(write=False)
            promoted[isotope] = combined
        self._joint_birth_proposal_prefix_scores = promoted
        self._joint_birth_proposal_prefix_station_count = completed_count + 1
        self._joint_birth_proposal_station_score_cache.clear()
        self._joint_birth_proposal_station_score_cache_order.clear()

    @staticmethod
    def _joint_structural_unit_cache_signature(
        *,
        filt: IsotopeParticleFilter,
        data: StructuralGeometryBatch,
        positive_line_indices: NDArray[np.int64],
    ) -> str:
        """Hash all immutable inputs to one unit-transport cache generation."""
        digest = hashlib.sha256(b"joint_continuous_surface_unit_transport_cache_v1\0")
        for text in (
            str(filt.isotope),
            str(filt.structural_rj_surface_atlas_sha256),
            str(id(filt.continuous_kernel)),
        ):
            encoded = text.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        for values in (
            data.detector_positions,
            data.fe_indices,
            data.pb_indices,
            data.station_sequence_ids,
            positive_line_indices,
        ):
            array = np.ascontiguousarray(values)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
            digest.update(array.tobytes(order="C"))
        return digest.hexdigest()

    def _joint_cached_continuous_unit_components_shard(
        self,
        *,
        filt: IsotopeParticleFilter,
        data: StructuralGeometryBatch,
        positions_s3: NDArray[np.float64],
        chart_ids_s: NDArray[np.int64],
        positive_line_indices: NDArray[np.int64],
        prefetched_keys: NDArray[Any] | None = None,
        prefetched_values: tuple[NDArray[np.float64], ...] | None = None,
    ) -> tuple[NDArray[np.float64], ...]:
        """Return exact unit transport for one immutable station shard.

        The cache changes only scheduling.  Keys contain the authoritative
        chart and continuous XYZ coordinates, while values are the exact
        continuous-kernel outputs for one immutable station geometry.
        Missing positions are evaluated together by the batched CPU/GPU
        transport kernel, or consumed from an exact multi-station prefetch.
        """
        positions = np.asarray(positions_s3, dtype=np.float64).reshape(-1, 3)
        raw_chart_ids = np.asarray(chart_ids_s)
        line_indices = np.asarray(
            positive_line_indices,
            dtype=np.int64,
        ).reshape(-1)
        if (
            not np.issubdtype(raw_chart_ids.dtype, np.integer)
            or raw_chart_ids.size != positions.shape[0]
            or np.any(~np.isfinite(positions))
            or line_indices.size == 0
        ):
            raise ValueError(
                "Cached continuous components require aligned finite surface "
                "positions, integer chart IDs, and positive line indices."
            )
        chart_ids = np.asarray(raw_chart_ids, dtype=np.int64).reshape(-1)
        row_count = int(data.row_count)
        line_count = int(line_indices.size)
        request_count = int(positions.shape[0])
        if request_count == 0:
            return tuple(
                np.zeros(
                    (row_count, 0, line_count),
                    dtype=np.float64,
                )
                for _ in JOINT_STRUCTURAL_UNIT_COMPONENT_NAMES
            )
        keys = np.empty(
            request_count,
            dtype=JOINT_STRUCTURAL_UNIT_CACHE_KEY_DTYPE,
        )
        keys["chart"] = chart_ids
        keys["x"] = positions[:, 0]
        keys["y"] = positions[:, 1]
        keys["z"] = positions[:, 2]
        unique_keys, first_indices, inverse = np.unique(
            keys,
            return_index=True,
            return_inverse=True,
        )
        signature = self._joint_structural_unit_cache_signature(
            filt=filt,
            data=data,
            positive_line_indices=line_indices,
        )
        isotope_key = str(filt.isotope)
        isotope_cache = self._joint_structural_unit_transport_cache.setdefault(
            isotope_key,
            {},
        )
        cache = isotope_cache.get(signature)
        if cache is None:
            cache = {
                "signature": signature,
                "keys": np.empty(
                    0,
                    dtype=JOINT_STRUCTURAL_UNIT_CACHE_KEY_DTYPE,
                ),
                "recency": np.empty(0, dtype=np.int64),
                "values": tuple(
                    np.zeros(
                        (0, row_count, line_count),
                        dtype=np.float64,
                    )
                    for _ in JOINT_STRUCTURAL_UNIT_COMPONENT_NAMES
                ),
                "generation": 0,
                "last_access": 0,
            }
        cached_keys = np.asarray(
            cache["keys"],
            dtype=JOINT_STRUCTURAL_UNIT_CACHE_KEY_DTYPE,
        )
        cached_values = tuple(
            np.asarray(values, dtype=np.float64) for values in cache["values"]
        )
        cached_count = int(cached_keys.size)
        lookup = np.searchsorted(cached_keys, unique_keys)
        if cached_count:
            safe_lookup = np.minimum(lookup, cached_count - 1)
            hit = (lookup < cached_count) & (cached_keys[safe_lookup] == unique_keys)
        else:
            safe_lookup = np.zeros(unique_keys.size, dtype=np.int64)
            hit = np.zeros(unique_keys.size, dtype=bool)
        missing = ~hit
        unique_component_values = tuple(
            np.empty(
                (unique_keys.size, row_count, line_count),
                dtype=np.float64,
            )
            for _ in JOINT_STRUCTURAL_UNIT_COMPONENT_NAMES
        )
        if np.any(hit):
            for output, values in zip(
                unique_component_values,
                cached_values,
                strict=True,
            ):
                output[hit] = values[safe_lookup[hit]]
        missing_first_indices = first_indices[missing]
        missing_values: tuple[NDArray[np.float64], ...]
        if missing_first_indices.size:
            if prefetched_keys is not None or prefetched_values is not None:
                if prefetched_keys is None or prefetched_values is None:
                    raise ValueError(
                        "Prefetched transport keys and values must be paired."
                    )
                supplied_keys = np.asarray(
                    prefetched_keys,
                    dtype=JOINT_STRUCTURAL_UNIT_CACHE_KEY_DTYPE,
                ).reshape(-1)
                supplied_values = tuple(
                    np.asarray(values, dtype=np.float64) for values in prefetched_values
                )
                supplied_lookup = np.searchsorted(
                    supplied_keys,
                    unique_keys[missing],
                )
                if (
                    len(supplied_values) != len(JOINT_STRUCTURAL_UNIT_COMPONENT_NAMES)
                    or np.any(supplied_lookup >= supplied_keys.size)
                    or np.any(supplied_keys[supplied_lookup] != unique_keys[missing])
                ):
                    raise RuntimeError(
                        "Exact transport prefetch does not cover shard misses."
                    )
                missing_values = tuple(
                    values[supplied_lookup] for values in supplied_values
                )
            else:
                evaluated = filt._continuous_rj_line_transport_component_columns(
                    data,
                    positions[missing_first_indices],
                    line_indices,
                    chart_ids=chart_ids[missing_first_indices],
                )
                missing_values = tuple(
                    np.transpose(
                        np.asarray(
                            getattr(evaluated, name),
                            dtype=np.float64,
                        ),
                        (1, 0, 2),
                    )
                    for name in JOINT_STRUCTURAL_UNIT_COMPONENT_NAMES
                )
            expected_missing_shape = (
                int(missing_first_indices.size),
                row_count,
                line_count,
            )
            if any(
                values.shape != expected_missing_shape
                or np.any(~np.isfinite(values))
                or np.any(values < 0.0)
                for values in missing_values
            ):
                raise RuntimeError(
                    "Batched unit-transport cache fill returned invalid "
                    "physical components."
                )
            for output, values in zip(
                unique_component_values,
                missing_values,
                strict=True,
            ):
                output[missing] = values
        else:
            missing_values = tuple(
                np.zeros(
                    (0, row_count, line_count),
                    dtype=np.float64,
                )
                for _ in JOINT_STRUCTURAL_UNIT_COMPONENT_NAMES
            )
        # A cache hit normally identifies a source in the currently accepted
        # PF state, whereas a miss is commonly a one-shot rejected proposal.
        # Give hits the newer generation so proposal churn cannot evict the
        # accepted state and force exact old-station transport to be recomputed
        # at every tempering step.
        generation = int(cache["generation"]) + 2
        recency = np.asarray(cache["recency"], dtype=np.int64).copy()
        if np.any(hit):
            recency[safe_lookup[hit]] = generation
        merged_keys = np.concatenate((cached_keys, unique_keys[missing]))
        merged_recency = np.concatenate(
            (
                recency,
                np.full(
                    int(np.count_nonzero(missing)),
                    generation - 1,
                    dtype=np.int64,
                ),
            )
        )
        merged_values = tuple(
            np.concatenate((old, new), axis=0)
            for old, new in zip(
                cached_values,
                missing_values,
                strict=True,
            )
        )
        bytes_per_entry = (
            int(JOINT_STRUCTURAL_UNIT_CACHE_KEY_DTYPE.itemsize)
            + np.dtype(np.int64).itemsize
            + len(JOINT_STRUCTURAL_UNIT_COMPONENT_NAMES)
            * row_count
            * line_count
            * np.dtype(np.float64).itemsize
        )
        byte_capacity = max(
            1,
            JOINT_STRUCTURAL_UNIT_CACHE_MAX_BYTES // max(bytes_per_entry, 1),
        )
        active_state_capacity = max(
            1,
            JOINT_STRUCTURAL_UNIT_CACHE_ACTIVE_STATE_MULTIPLIER
            * int(filt.config.num_particles)
            * int(filt.config.hard_max_sources),
        )
        capacity = min(byte_capacity, active_state_capacity)
        if merged_keys.size > capacity:
            retain = np.argsort(
                merged_recency,
                kind="stable",
            )[-capacity:]
            merged_keys = merged_keys[retain]
            merged_recency = merged_recency[retain]
            merged_values = tuple(values[retain] for values in merged_values)
        order = np.argsort(merged_keys, kind="stable")
        cache = {
            "signature": signature,
            "keys": merged_keys[order],
            "recency": merged_recency[order],
            "values": tuple(values[order] for values in merged_values),
            "generation": generation,
            "last_access": 0,
        }
        self._joint_structural_unit_cache_access_generation += 1
        cache["last_access"] = int(self._joint_structural_unit_cache_access_generation)
        isotope_cache[signature] = cache
        self._trim_joint_structural_unit_transport_cache(
            isotope_key,
            protected_signature=signature,
        )
        self.last_joint_structural_unit_cache_hits += int(np.count_nonzero(hit))
        self.last_joint_structural_unit_cache_misses += int(np.count_nonzero(missing))
        return tuple(
            np.transpose(values[inverse], (1, 0, 2))
            for values in unique_component_values
        )

    @staticmethod
    def _joint_structural_unit_cache_shard_bytes(
        cache: Mapping[str, Any],
    ) -> int:
        """Return the exact NumPy storage owned by one transport-cache shard."""
        return int(
            np.asarray(cache["keys"]).nbytes
            + np.asarray(cache["recency"]).nbytes
            + sum(np.asarray(values).nbytes for values in cache["values"])
        )

    def _trim_joint_structural_unit_transport_cache(
        self,
        isotope: str,
        *,
        protected_signature: str,
    ) -> None:
        """Bound all station shards for one isotope with shard-level LRU."""
        isotope_cache = self._joint_structural_unit_transport_cache[str(isotope)]
        total_bytes = sum(
            self._joint_structural_unit_cache_shard_bytes(cache)
            for cache in isotope_cache.values()
        )
        while total_bytes > JOINT_STRUCTURAL_UNIT_CACHE_MAX_BYTES:
            candidates = [
                (int(cache["last_access"]), signature)
                for signature, cache in isotope_cache.items()
                if signature != protected_signature
            ]
            if not candidates:
                break
            _, signature = min(candidates)
            removed = isotope_cache.pop(signature)
            total_bytes -= self._joint_structural_unit_cache_shard_bytes(removed)

    @staticmethod
    def _joint_structural_station_geometry_shards(
        data: StructuralGeometryBatch,
    ) -> tuple[StructuralGeometryBatch, ...]:
        """Split contiguous immutable station rows without changing their order."""
        station_ids = np.asarray(
            data.station_sequence_ids,
            dtype=np.int64,
        ).reshape(-1)
        boundaries = np.concatenate(
            (
                np.zeros(1, dtype=np.int64),
                np.flatnonzero(station_ids[1:] != station_ids[:-1]) + 1,
                np.asarray([station_ids.size], dtype=np.int64),
            )
        )
        shards = tuple(
            StructuralGeometryBatch(
                detector_positions=data.detector_positions[start:stop],
                fe_indices=data.fe_indices[start:stop],
                pb_indices=data.pb_indices[start:stop],
                live_times=data.live_times[start:stop],
                station_sequence_ids=data.station_sequence_ids[start:stop],
            )
            for start, stop in zip(
                boundaries[:-1].tolist(),
                boundaries[1:].tolist(),
                strict=True,
            )
        )
        observed_ids = [int(shard.station_sequence_ids[0]) for shard in shards]
        if len(set(observed_ids)) != len(observed_ids):
            raise ValueError(
                "Structural geometry station rows must form one contiguous "
                "block per station sequence."
            )
        return shards

    def _joint_cached_continuous_unit_components(
        self,
        *,
        filt: IsotopeParticleFilter,
        data: StructuralGeometryBatch,
        positions_s3: NDArray[np.float64],
        chart_ids_s: NDArray[np.int64],
        positive_line_indices: NDArray[np.int64],
    ) -> tuple[NDArray[np.float64], ...]:
        """Return exact transport with one fused call for station-cache misses."""
        positions = np.asarray(positions_s3, dtype=np.float64).reshape(-1, 3)
        raw_chart_ids = np.asarray(chart_ids_s)
        chart_ids = np.asarray(raw_chart_ids, dtype=np.int64).reshape(-1)
        line_indices = np.asarray(
            positive_line_indices,
            dtype=np.int64,
        ).reshape(-1)
        if (
            not np.issubdtype(raw_chart_ids.dtype, np.integer)
            or raw_chart_ids.size != positions.shape[0]
            or positions.shape[0] != chart_ids.size
            or np.any(~np.isfinite(positions))
            or line_indices.size == 0
        ):
            raise ValueError(
                "Fused continuous transport requires aligned finite states "
                "and positive line indices."
            )
        station_shards = self._joint_structural_station_geometry_shards(data)
        if len(station_shards) == 1 or positions.shape[0] == 0:
            return self._joint_cached_continuous_unit_components_shard(
                filt=filt,
                data=station_shards[0],
                positions_s3=positions,
                chart_ids_s=chart_ids,
                positive_line_indices=line_indices,
            )

        request_keys = np.empty(
            positions.shape[0],
            dtype=JOINT_STRUCTURAL_UNIT_CACHE_KEY_DTYPE,
        )
        request_keys["chart"] = chart_ids
        request_keys["x"] = positions[:, 0]
        request_keys["y"] = positions[:, 1]
        request_keys["z"] = positions[:, 2]
        unique_keys, first_indices = np.unique(
            request_keys,
            return_index=True,
        )
        isotope_cache = self._joint_structural_unit_transport_cache.setdefault(
            str(filt.isotope),
            {},
        )
        active_shard_indices: list[int] = []
        union_missing = np.zeros(unique_keys.size, dtype=np.bool_)
        for shard_index, station_data in enumerate(station_shards):
            signature = self._joint_structural_unit_cache_signature(
                filt=filt,
                data=station_data,
                positive_line_indices=line_indices,
            )
            cache = isotope_cache.get(signature)
            if cache is None:
                missing = np.ones(unique_keys.size, dtype=np.bool_)
            else:
                cached_keys = np.asarray(
                    cache["keys"],
                    dtype=JOINT_STRUCTURAL_UNIT_CACHE_KEY_DTYPE,
                )
                lookup = np.searchsorted(cached_keys, unique_keys)
                if cached_keys.size:
                    safe_lookup = np.minimum(lookup, cached_keys.size - 1)
                    hit = (lookup < cached_keys.size) & (
                        cached_keys[safe_lookup] == unique_keys
                    )
                    missing = ~hit
                else:
                    missing = np.ones(unique_keys.size, dtype=np.bool_)
            if np.any(missing):
                active_shard_indices.append(shard_index)
                union_missing |= missing

        prefetched_keys: NDArray[Any] | None = None
        prefetched_by_shard: dict[
            int,
            tuple[NDArray[np.float64], ...],
        ] = {}
        if active_shard_indices:
            active_shards = [station_shards[index] for index in active_shard_indices]
            fused_geometry = StructuralGeometryBatch(
                detector_positions=np.concatenate(
                    [shard.detector_positions for shard in active_shards],
                    axis=0,
                ),
                fe_indices=np.concatenate(
                    [shard.fe_indices for shard in active_shards],
                ),
                pb_indices=np.concatenate(
                    [shard.pb_indices for shard in active_shards],
                ),
                live_times=np.concatenate(
                    [shard.live_times for shard in active_shards],
                ),
                station_sequence_ids=np.concatenate(
                    [shard.station_sequence_ids for shard in active_shards],
                ),
            )
            prefetched_keys = unique_keys[union_missing]
            prefetched_first_indices = first_indices[union_missing]
            evaluated = filt._continuous_rj_line_transport_component_columns(
                fused_geometry,
                positions[prefetched_first_indices],
                line_indices,
                chart_ids=chart_ids[prefetched_first_indices],
            )
            fused_values = tuple(
                np.transpose(
                    np.asarray(getattr(evaluated, name), dtype=np.float64),
                    (1, 0, 2),
                )
                for name in JOINT_STRUCTURAL_UNIT_COMPONENT_NAMES
            )
            row_start = 0
            for shard_index, station_data in zip(
                active_shard_indices,
                active_shards,
                strict=True,
            ):
                row_stop = row_start + int(station_data.row_count)
                prefetched_by_shard[shard_index] = tuple(
                    values[:, row_start:row_stop, :] for values in fused_values
                )
                row_start = row_stop

        active_shard_set = set(active_shard_indices)
        processing_order = [
            index
            for index in range(len(station_shards))
            if index not in active_shard_set
        ] + active_shard_indices
        station_components_by_index: list[tuple[NDArray[np.float64], ...] | None] = [
            None
        ] * len(station_shards)
        for shard_index in processing_order:
            station_components_by_index[shard_index] = (
                self._joint_cached_continuous_unit_components_shard(
                    filt=filt,
                    data=station_shards[shard_index],
                    positions_s3=positions,
                    chart_ids_s=chart_ids,
                    positive_line_indices=line_indices,
                    prefetched_keys=(
                        prefetched_keys if shard_index in prefetched_by_shard else None
                    ),
                    prefetched_values=prefetched_by_shard.get(shard_index),
                )
            )
        if any(values is None for values in station_components_by_index):
            raise RuntimeError("Fused station transport assembly is incomplete.")
        station_components = [
            values for values in station_components_by_index if values is not None
        ]
        return tuple(
            np.concatenate(
                [components[component_index] for components in station_components],
                axis=0,
            )
            for component_index in range(len(JOINT_STRUCTURAL_UNIT_COMPONENT_NAMES))
        )

    @staticmethod
    def _joint_cached_state_match_torch(
        *,
        filt: IsotopeParticleFilter,
        reference: "torch.Tensor",
        particle_indices: NDArray[np.int64],
        positions_pks: NDArray[np.float64],
        chart_ids_pk: NDArray[np.int64],
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Match proposals to immutable sweep-entry transport columns on Torch.

        The accepted numerical state is updated after every MH decision, while
        the transport tensor itself is refreshed only after an isotope sweep.
        Matching against the immutable sweep-entry mirror therefore prevents a
        moved source from being paired with a stale transport column.
        """
        import torch

        state = getattr(filt, "_structural_rj_device_state", None)
        if state is None and hasattr(
            filt,
            "_initialize_continuous_rj_device_state",
        ):
            filt._initialize_continuous_rj_device_state(reference)
            state = filt._structural_rj_device_state
        if state is None:
            packed = filt._packed_continuous_surface_state_arrays()
            state = {
                "cache_positions": torch.as_tensor(
                    packed[0], device=reference.device, dtype=reference.dtype
                ),
                "cache_strengths": torch.as_tensor(
                    packed[1], device=reference.device, dtype=reference.dtype
                ),
                "cache_mask": torch.as_tensor(
                    packed[2], device=reference.device, dtype=torch.bool
                ),
                "cache_chart_ids": torch.as_tensor(
                    packed[3], device=reference.device, dtype=torch.long
                ),
            }
        indices = torch.as_tensor(
            np.asarray(particle_indices, dtype=np.int64),
            device=reference.device,
            dtype=torch.long,
        )
        cached_positions = torch.index_select(
            state["cache_positions"],
            0,
            indices,
        )
        cached_strengths = torch.index_select(
            state["cache_strengths"],
            0,
            indices,
        )
        cached_mask = torch.index_select(
            state["cache_mask"],
            0,
            indices,
        )
        cached_charts = torch.index_select(
            state["cache_chart_ids"],
            0,
            indices,
        )
        positions = torch.as_tensor(
            positions_pks,
            device=reference.device,
            dtype=reference.dtype,
        )
        charts = torch.as_tensor(
            chart_ids_pk,
            device=reference.device,
            dtype=torch.long,
        )
        matches = (
            cached_mask[:, None, :]
            & (charts[:, :, None] == cached_charts[:, None, :])
            & torch.all(
                positions[:, :, None, :] == cached_positions[:, None, :, :],
                dim=3,
            )
        )
        match_counts = torch.sum(matches, dim=2, dtype=torch.long)
        matched = match_counts == 1
        matched_slots = torch.argmax(matches.to(torch.int8), dim=2)
        accepted_strength = torch.gather(
            cached_strengths,
            1,
            matched_slots,
        )
        invalid = torch.stack(
            (
                torch.any(match_counts > 1),
                torch.any(matched & (accepted_strength <= 0.0)),
            )
        ).any()
        if bool(invalid.item()):
            raise RuntimeError(
                "Cached source matches are ambiguous or have invalid strength."
            )
        return matched, matched_slots, accepted_strength

    def _joint_cuda_accepted_unit_cache_entry(
        self,
        *,
        filt: IsotopeParticleFilter,
        data: StructuralGeometryBatch,
        positive_line_indices: NDArray[np.int64],
        reference: "torch.Tensor",
    ) -> dict[str, object]:
        """Return a fixed-capacity cache for accepted sweep-local geometry."""
        import torch

        line_indices = np.ascontiguousarray(
            positive_line_indices,
            dtype=np.int64,
        )
        signature = (
            f"{id(data)}:{str(reference.device)}:"
            f"{hashlib.sha256(line_indices.tobytes()).hexdigest()}"
        )
        key = (str(filt.isotope), signature)
        cache_by_key = getattr(
            self,
            "_joint_cuda_accepted_unit_transport_cache",
            None,
        )
        if cache_by_key is None:
            cache_by_key = {}
            self._joint_cuda_accepted_unit_transport_cache = cache_by_key
        cached = cache_by_key.get(key)
        if cached is not None:
            return cached
        particle_count = int(reference.shape[0])
        view_count = int(data.row_count)
        maximum = int(filt.config.hard_max_sources or 0)
        line_count = int(np.asarray(positive_line_indices).size)
        feature_count = len(tuple(self._full_spectrum_model().transport_feature_order))
        cached = {
            "mask": torch.zeros(
                (particle_count, maximum),
                device=reference.device,
                dtype=torch.bool,
            ),
            "positions": torch.zeros(
                (particle_count, maximum, 3),
                device=reference.device,
                dtype=reference.dtype,
            ),
            "chart_ids": torch.zeros(
                (particle_count, maximum),
                device=reference.device,
                dtype=torch.long,
            ),
            "total": torch.zeros(
                (particle_count, view_count, maximum, line_count),
                device=reference.device,
                dtype=reference.dtype,
            ),
            "uncollided": torch.zeros(
                (particle_count, view_count, maximum, line_count),
                device=reference.device,
                dtype=reference.dtype,
            ),
            "features": torch.zeros(
                (
                    particle_count,
                    view_count,
                    maximum,
                    line_count,
                    feature_count,
                ),
                device=reference.device,
                dtype=reference.dtype,
            ),
            "pending": None,
        }
        cache_by_key[key] = cached
        return cached

    @staticmethod
    def _joint_promote_pending_cuda_unit_transport(
        *,
        filt: IsotopeParticleFilter,
        cache: dict[str, object],
    ) -> None:
        """Commit pending unit columns only when their proposal was accepted."""
        import torch

        pending = cache.get("pending")
        state = getattr(filt, "_structural_rj_device_state", None)
        if not isinstance(pending, dict) or state is None:
            return
        indices = pending["particle_indices"]
        cardinality = int(pending["cardinality"])
        current_cardinality = torch.index_select(
            state["cardinalities"],
            0,
            indices,
        )
        accepted = current_cardinality == cardinality
        if cardinality:
            current_positions = torch.index_select(
                state["positions"],
                0,
                indices,
            )[:, :cardinality]
            current_charts = torch.index_select(
                state["chart_ids"],
                0,
                indices,
            )[:, :cardinality]
            accepted &= torch.all(
                current_positions == pending["positions"],
                dim=(1, 2),
            )
            accepted &= torch.all(
                current_charts == pending["chart_ids"],
                dim=1,
            )
        accepted_indices = indices[accepted]
        cache["mask"][accepted_indices] = False
        cache["total"][accepted_indices] = 0.0
        cache["uncollided"][accepted_indices] = 0.0
        cache["features"][accepted_indices] = 0.0
        if cardinality:
            cache["mask"][accepted_indices, :cardinality] = True
            cache["positions"][accepted_indices, :cardinality] = pending[
                "positions"
            ][accepted]
            cache["chart_ids"][accepted_indices, :cardinality] = pending[
                "chart_ids"
            ][accepted]
            cache["total"][accepted_indices, :, :cardinality] = pending[
                "total"
            ][accepted]
            cache["uncollided"][accepted_indices, :, :cardinality] = pending[
                "uncollided"
            ][accepted]
            cache["features"][accepted_indices, :, :cardinality] = pending[
                "features"
            ][accepted]
        cache["pending"] = None

    @staticmethod
    def _joint_match_cuda_accepted_unit_transport(
        *,
        cache: dict[str, object],
        particle_indices: NDArray[np.int64],
        positions_pks: NDArray[np.float64],
        chart_ids_pk: NDArray[np.int64],
        reference: "torch.Tensor",
    ) -> tuple["torch.Tensor", "torch.Tensor", tuple["torch.Tensor", ...]]:
        """Match candidates to accepted sweep-local unit columns on CUDA."""
        import torch

        indices = torch.as_tensor(
            particle_indices,
            device=reference.device,
            dtype=torch.long,
        )
        positions = torch.as_tensor(
            positions_pks,
            device=reference.device,
            dtype=reference.dtype,
        )
        charts = torch.as_tensor(
            chart_ids_pk,
            device=reference.device,
            dtype=torch.long,
        )
        cached_mask = torch.index_select(cache["mask"], 0, indices)
        cached_positions = torch.index_select(cache["positions"], 0, indices)
        cached_charts = torch.index_select(cache["chart_ids"], 0, indices)
        matches = (
            cached_mask[:, None, :]
            & (charts[:, :, None] == cached_charts[:, None, :])
            & torch.all(
                positions[:, :, None, :] == cached_positions[:, None, :, :],
                dim=3,
            )
        )
        matched = torch.any(matches, dim=2)
        matched_slots = torch.argmax(matches.to(torch.int8), dim=2)
        view_count = int(cache["total"].shape[1])
        line_count = int(cache["total"].shape[3])
        feature_count = int(cache["features"].shape[4])
        line_gather = matched_slots[:, None, :, None].expand(
            -1,
            view_count,
            -1,
            line_count,
        )
        feature_gather = line_gather[..., None].expand(
            -1,
            -1,
            -1,
            -1,
            feature_count,
        )
        selected_total = torch.index_select(cache["total"], 0, indices)
        selected_uncollided = torch.index_select(
            cache["uncollided"],
            0,
            indices,
        )
        selected_features = torch.index_select(cache["features"], 0, indices)
        gathered = (
            torch.gather(selected_total, 2, line_gather),
            torch.gather(selected_uncollided, 2, line_gather),
            torch.gather(selected_features, 2, feature_gather),
        )
        return matched, matched_slots, gathered

    @staticmethod
    def _joint_stage_cuda_unit_transport(
        *,
        cache: dict[str, object],
        particle_indices: NDArray[np.int64],
        positions_pks: NDArray[np.float64],
        chart_ids_pk: NDArray[np.int64],
        unit_total: "torch.Tensor",
        unit_uncollided: "torch.Tensor",
        unit_features: "torch.Tensor",
    ) -> None:
        """Stage exact proposal columns for acceptance detection next call."""
        import torch

        reference = unit_total
        cache["pending"] = {
            "particle_indices": torch.as_tensor(
                particle_indices,
                device=reference.device,
                dtype=torch.long,
            ),
            "cardinality": int(np.asarray(positions_pks).shape[1]),
            "positions": torch.as_tensor(
                positions_pks,
                device=reference.device,
                dtype=reference.dtype,
            ).clone(),
            "chart_ids": torch.as_tensor(
                chart_ids_pk,
                device=reference.device,
                dtype=torch.long,
            ).clone(),
            "total": unit_total.detach().clone(),
            "uncollided": unit_uncollided.detach().clone(),
            "features": unit_features.detach().clone(),
        }

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
        slots_per_isotope = int(filt.config.hard_max_sources)
        total_slot_count = slots_per_isotope * len(order)
        if (
            tuple(cached_total.shape[1:]) != (total_views, total_slot_count, line_count)
            or tuple(cached_uncollided.shape) != tuple(cached_total.shape)
            or tuple(cached_features.shape)
            != tuple(cached_total.shape) + (feature_count,)
            or np.any(indices < 0)
            or np.any(indices >= int(cached_total.shape[0]))
        ):
            raise RuntimeError("Joint structural transport cache is misaligned.")
        cache_is_torch = hasattr(cached_total, "detach")
        layout = self._joint_line_layout()
        global_columns, local_indices, branching_weights = layout[str(filt.isotope)]
        local_shape = (
            total_views,
            indices.size,
            int(positions.shape[1]),
            int(local_indices.size),
        )
        isotope_index = order.index(str(filt.isotope))
        slot_start = isotope_index * slots_per_isotope
        slot_stop = slot_start + slots_per_isotope
        if cache_is_torch:
            import torch

            index_tensor = torch.as_tensor(
                indices,
                device=cached_total.device,
                dtype=torch.long,
            )
            total = torch.index_select(
                cached_total,
                0,
                index_tensor,
            ).clone()
            uncollided = torch.index_select(
                cached_uncollided,
                0,
                index_tensor,
            ).clone()
            features = torch.index_select(
                cached_features,
                0,
                index_tensor,
            ).clone()
            global_column_selection = torch.as_tensor(
                global_columns,
                device=total.device,
                dtype=torch.long,
            )
            cardinality = int(positions.shape[1])
            matched, matched_slot_tensor, accepted_strength_tensor = (
                self._joint_cached_state_match_torch(
                    filt=filt,
                    reference=cached_total,
                    particle_indices=indices,
                    positions_pks=positions,
                    chart_ids_pk=chart_ids,
                )
            )
            accepted_unit_cache = self._joint_cuda_accepted_unit_cache_entry(
                filt=filt,
                data=data,
                positive_line_indices=local_indices,
                reference=cached_total,
            )
            self._joint_promote_pending_cuda_unit_transport(
                filt=filt,
                cache=accepted_unit_cache,
            )
            accepted_matched, _, accepted_unit_components = (
                self._joint_match_cuda_accepted_unit_transport(
                    cache=accepted_unit_cache,
                    particle_indices=indices,
                    positions_pks=positions,
                    chart_ids_pk=chart_ids,
                    reference=cached_total,
                )
            )
            accepted_matched &= ~matched
            local_line_count = int(local_indices.size)
            candidate_total = torch.zeros(
                (
                    indices.size,
                    total_views,
                    cardinality,
                    local_line_count,
                ),
                device=total.device,
                dtype=total.dtype,
            )
            candidate_uncollided = torch.zeros_like(candidate_total)
            candidate_features = torch.zeros(
                tuple(candidate_total.shape) + (feature_count,),
                device=total.device,
                dtype=total.dtype,
            )
            if cardinality:
                accepted_total = torch.index_select(
                    total[:, :, slot_start:slot_stop, :],
                    3,
                    global_column_selection,
                )
                accepted_uncollided = torch.index_select(
                    uncollided[:, :, slot_start:slot_stop, :],
                    3,
                    global_column_selection,
                )
                accepted_features = torch.index_select(
                    features[:, :, slot_start:slot_stop, :, :],
                    3,
                    global_column_selection,
                )
                line_gather = matched_slot_tensor[:, None, :, None].expand(
                    -1,
                    total_views,
                    -1,
                    local_line_count,
                )
                feature_gather = line_gather[..., None].expand(
                    -1,
                    -1,
                    -1,
                    -1,
                    feature_count,
                )
                gathered_total = torch.gather(
                    accepted_total,
                    2,
                    line_gather,
                )
                gathered_uncollided = torch.gather(
                    accepted_uncollided,
                    2,
                    line_gather,
                )
                gathered_features = torch.gather(
                    accepted_features,
                    2,
                    feature_gather,
                )
                proposed_strength_tensor = torch.as_tensor(
                    strengths,
                    device=total.device,
                    dtype=total.dtype,
                )
                ratio_tensor = torch.where(
                    matched,
                    proposed_strength_tensor / torch.where(
                        matched,
                        accepted_strength_tensor,
                        torch.ones_like(accepted_strength_tensor),
                    ),
                    torch.ones_like(proposed_strength_tensor),
                )[:, None, :, None]
                matched_tensor = matched[:, None, :, None]
                candidate_total = torch.where(
                    matched_tensor,
                    gathered_total * ratio_tensor,
                    candidate_total,
                )
                candidate_uncollided = torch.where(
                    matched_tensor,
                    gathered_uncollided * ratio_tensor,
                    candidate_uncollided,
                )
                candidate_features = torch.where(
                    matched_tensor[..., None],
                    gathered_features,
                    candidate_features,
                )
            if cardinality:
                proposed_strength_tensor = torch.as_tensor(
                    strengths,
                    device=total.device,
                    dtype=total.dtype,
                )[:, None, :, None]
                accepted_tensor = accepted_matched[:, None, :, None]
                candidate_total = torch.where(
                    accepted_tensor,
                    accepted_unit_components[0] * proposed_strength_tensor,
                    candidate_total,
                )
                candidate_uncollided = torch.where(
                    accepted_tensor,
                    accepted_unit_components[1] * proposed_strength_tensor,
                    candidate_uncollided,
                )
                candidate_features = torch.where(
                    accepted_tensor[..., None],
                    accepted_unit_components[2],
                    candidate_features,
                )
            all_matched = matched | accepted_matched
            unmatched_index = torch.nonzero(~all_matched, as_tuple=False)
            unmatched_rows = (
                unmatched_index[:, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64, copy=False)
            )
            unmatched_slots = (
                unmatched_index[:, 1]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64, copy=False)
            )
            if unmatched_rows.size:
                device_components = (
                    filt._continuous_rj_line_transport_component_columns(
                        data,
                        positions[unmatched_rows, unmatched_slots],
                        local_indices,
                        chart_ids=chart_ids[unmatched_rows, unmatched_slots],
                        device_resident=True,
                    )
                )
                if not hasattr(device_components.total_kernel, "detach"):
                    raise RuntimeError(
                        "CUDA structural transport returned host components."
                    )
                branch_tensor = torch.as_tensor(
                    branching_weights,
                    device=total.device,
                    dtype=total.dtype,
                )[None, None, :]
                strength_tensor = torch.as_tensor(
                    strengths[unmatched_rows, unmatched_slots],
                    device=total.device,
                    dtype=total.dtype,
                )[:, None, None]
                unmatched_total = (
                    device_components.total_kernel.permute(1, 0, 2)
                    * strength_tensor
                    * branch_tensor
                )
                unmatched_uncollided = (
                    device_components.uncollided_kernel.permute(1, 0, 2)
                    * strength_tensor
                    * branch_tensor
                )
                unmatched_features = torch.stack(
                    (
                        device_components.tau_fe,
                        device_components.tau_pb,
                        device_components.tau_obstacle,
                        device_components.distance_m,
                    ),
                    dim=-1,
                ).permute(1, 0, 2, 3)
                unmatched_row_tensor = torch.as_tensor(
                    unmatched_rows,
                    device=total.device,
                    dtype=torch.long,
                )
                unmatched_slot_tensor = torch.as_tensor(
                    unmatched_slots,
                    device=total.device,
                    dtype=torch.long,
                )
                candidate_total[
                    unmatched_row_tensor,
                    :,
                    unmatched_slot_tensor,
                    :,
                ] = unmatched_total
                candidate_uncollided[
                    unmatched_row_tensor,
                    :,
                    unmatched_slot_tensor,
                    :,
                ] = unmatched_uncollided
                candidate_features[
                    unmatched_row_tensor,
                    :,
                    unmatched_slot_tensor,
                    :,
                    :,
                ] = unmatched_features
            station_count = len(self._joint_structural_station_geometry_shards(data))
            self.last_joint_structural_unit_cache_hits += int(
                torch.count_nonzero(all_matched).item() * station_count
            )
            self.last_joint_structural_unit_cache_misses += int(
                unmatched_rows.size * station_count
            )
            staged_strength = torch.as_tensor(
                strengths,
                device=total.device,
                dtype=total.dtype,
            )[:, None, :, None]
            self._joint_stage_cuda_unit_transport(
                cache=accepted_unit_cache,
                particle_indices=indices,
                positions_pks=positions,
                chart_ids_pk=chart_ids,
                unit_total=candidate_total / staged_strength,
                unit_uncollided=candidate_uncollided / staged_strength,
                unit_features=candidate_features,
            )
        else:
            (
                component_total,
                component_uncollided,
                component_tau_fe,
                component_tau_pb,
                component_tau_obstacle,
                component_distance,
            ) = self._joint_cached_continuous_unit_components(
                filt=filt,
                data=data,
                positions_s3=positions.reshape(-1, 3),
                chart_ids_s=chart_ids.reshape(-1),
                positive_line_indices=local_indices,
            )
            candidate_total_numpy = np.asarray(
                component_total,
                dtype=np.float64,
            ).reshape(local_shape)
            candidate_uncollided_numpy = np.asarray(
                component_uncollided,
                dtype=np.float64,
            ).reshape(local_shape)
            candidate_features_numpy = np.stack(
                (
                    np.asarray(component_tau_fe, dtype=np.float64),
                    np.asarray(component_tau_pb, dtype=np.float64),
                    np.asarray(component_tau_obstacle, dtype=np.float64),
                    np.asarray(component_distance, dtype=np.float64),
                ),
                axis=-1,
            ).reshape(local_shape + (feature_count,))
            total = np.asarray(
                cached_total[indices],
                dtype=np.float64,
            ).copy()
            uncollided = np.asarray(
                cached_uncollided[indices],
                dtype=np.float64,
            ).copy()
            features = np.asarray(
                cached_features[indices],
                dtype=np.float64,
            ).copy()
            scale = strengths[None, :, :, None] * branching_weights.reshape(1, 1, 1, -1)
            candidate_total = np.transpose(
                candidate_total_numpy * scale,
                (1, 0, 2, 3),
            )
            candidate_uncollided = np.transpose(
                candidate_uncollided_numpy * scale,
                (1, 0, 2, 3),
            )
            candidate_features = np.transpose(
                candidate_features_numpy,
                (1, 0, 2, 3, 4),
            )
            global_column_selection = global_columns
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
            total_subset[..., global_column_selection] = candidate_total
            uncollided_subset[..., global_column_selection] = candidate_uncollided
            feature_subset[..., global_column_selection, :] = candidate_features
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
        if cache_is_torch:
            result = self._joint_history_log_likelihood_torch(
                filt=filt,
                stations=stations,
                total_nvsl=total,
                uncollided_nvsl=uncollided,
                features_nvslf=features,
                target_beta=beta,
                newest_prefix_count=(self._active_joint_tempering_prefix_count),
            )
            return result.detach().cpu().numpy().astype(np.float64, copy=False)
        return self._joint_history_log_likelihood_numpy(
            filt=filt,
            stations=stations,
            total_nvsl=total,
            uncollided_nvsl=uncollided,
            features_nvslf=features,
            target_beta=beta,
            newest_prefix_count=self._active_joint_tempering_prefix_count,
        )

    def _joint_structural_strength_grid_target_evaluator(
        self,
        *,
        filt: IsotopeParticleFilter,
        data: StructuralGeometryBatch,
        positions_pks: NDArray[np.float64],
        chart_ids_pk: NDArray[np.int64],
        strengths_pgk: NDArray[np.float64],
        particle_indices: NDArray[np.int64],
        target_beta: float,
        tempering_start_row: int | None,
    ) -> NDArray[np.float64]:
        """Evaluate a fixed-geometry strength grid with one transport pass.

        Source transport is independent of source strength. The standard
        conditional proposal therefore computes each particle/source unit
        transport once, broadcasts it over the strength-grid axis, and scores
        the resulting exact joint targets in bounded CPU/GPU batches.
        """
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
        strengths = np.asarray(strengths_pgk, dtype=np.float64)
        if (
            positions.ndim != 3
            or positions.shape[0] != indices.size
            or positions.shape[2] != 3
            or not np.issubdtype(raw_chart_ids.dtype, np.integer)
            or raw_chart_ids.shape != positions.shape[:2]
            or strengths.ndim != 3
            or strengths.shape[0] != indices.size
            or strengths.shape[2] != positions.shape[1]
            or strengths.shape[1] < 1
            or np.any(~np.isfinite(strengths))
            or np.any(strengths <= 0.0)
        ):
            raise ValueError(
                "Conditional strength-grid states must share aligned "
                "particle, grid, and source axes."
            )
        chart_ids = np.asarray(raw_chart_ids, dtype=np.int64)
        total_views = sum(int(station.fe_indices.size) for station in stations)
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
        beta = float(target_beta)
        if not np.isfinite(beta) or not 0.0 <= beta <= 1.0:
            raise ValueError("Joint structural target_beta must lie in [0, 1].")
        model = self._full_spectrum_model()
        line_count = len(tuple(model.line_identity))
        feature_count = len(tuple(model.transport_feature_order))
        slots_per_isotope = int(filt.config.hard_max_sources)
        total_slot_count = slots_per_isotope * len(order)
        cached_total, cached_uncollided, cached_features = cache
        if (
            tuple(cached_total.shape[1:]) != (total_views, total_slot_count, line_count)
            or tuple(cached_uncollided.shape) != tuple(cached_total.shape)
            or tuple(cached_features.shape)
            != tuple(cached_total.shape) + (feature_count,)
            or np.any(indices < 0)
            or np.any(indices >= int(cached_total.shape[0]))
        ):
            raise RuntimeError("Joint structural transport cache is misaligned.")
        row_count = int(indices.size)
        grid_count = int(strengths.shape[1])
        if row_count == 0:
            return np.empty((0, grid_count), dtype=np.float64)
        configured_batch_size = int(
            self.pf_config.joint_strength_block_batch_size
        )
        batch_size = configured_batch_size
        maximum_batch_size = configured_batch_size
        cache_key: tuple[object, ...] | None = None
        cache_is_cuda = hasattr(cached_total, "detach") and bool(
            cached_total.is_cuda
        )
        if cache_is_cuda:
            import torch

            free_bytes, _ = torch.cuda.mem_get_info(cached_total.device)
            bytes_per_value = int(cached_total.element_size())
            expanded_values_per_row = (
                grid_count
                * total_views
                * total_slot_count
                * line_count
                * (2 + feature_count)
            )
            working_bytes_per_row = max(
                1,
                2 * expanded_values_per_row * bytes_per_value,
            )
            memory_budget = min(
                2 * 1024**3,
                max(64 * 1024**2, int(free_bytes) // 4),
            )
            maximum_batch_size = min(
                JOINT_STRENGTH_GRID_AUTOTUNE_MAX_BATCH_SIZE,
                max(1, memory_budget // working_bytes_per_row),
            )
            cache_key = self._joint_strength_grid_autotune_key(
                cache_tensor=cached_total,
                total_views=total_views,
                total_slot_count=total_slot_count,
                line_count=line_count,
                feature_count=feature_count,
                grid_count=grid_count,
                cardinality=int(positions.shape[1]),
            )
            batch_cache = getattr(
                self,
                "_joint_strength_grid_batch_size_cache",
                None,
            )
            if batch_cache is None:
                batch_cache = {}
                self._joint_strength_grid_batch_size_cache = batch_cache
            batch_size = batch_cache.get(
                cache_key,
                min(configured_batch_size, maximum_batch_size),
            )
        result = np.empty((row_count, grid_count), dtype=np.float64)

        def _evaluate(start: int, stop: int) -> None:
            """Evaluate and retain one exact contiguous candidate row slab."""
            result[start:stop] = self._joint_structural_strength_grid_target_batch(
                filt=filt,
                data=data,
                stations=stations,
                positions_bks=positions[start:stop],
                chart_ids_bk=chart_ids[start:stop],
                strengths_bgk=strengths[start:stop],
                particle_indices=indices[start:stop],
                target_beta=beta,
            )

        start = 0
        batch_cache = getattr(
            self,
            "_joint_strength_grid_batch_size_cache",
            {},
        )
        if cache_is_cuda and cache_key not in batch_cache:
            import torch

            trials: list[dict[str, float | int | str]] = []
            candidates = self._joint_strength_grid_autotune_candidates(
                configured_batch_size=configured_batch_size,
                maximum_batch_size=maximum_batch_size,
                row_count=row_count,
            )
            for candidate in candidates:
                if row_count - start < candidate:
                    break
                stop = start + candidate
                trial_start = time.perf_counter()
                try:
                    _evaluate(start, stop)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    trials.append(
                        {
                            "batch_size": int(candidate),
                            "rows_per_second": 0.0,
                            "elapsed_s": float("inf"),
                            "status": "cuda_oom",
                        }
                    )
                    break
                elapsed = time.perf_counter() - trial_start
                trials.append(
                    {
                        "batch_size": int(candidate),
                        "rows_per_second": float(candidate / max(elapsed, 1.0e-12)),
                        "elapsed_s": float(elapsed),
                        "status": "ok",
                    }
                )
                start = stop
            successful = [trial for trial in trials if trial["status"] == "ok"]
            if successful:
                selected_trial = max(
                    successful,
                    key=lambda trial: float(trial["rows_per_second"]),
                )
                batch_size = int(selected_trial["batch_size"])
            else:
                batch_size = max(1, min(configured_batch_size, maximum_batch_size))
            batch_cache[cache_key] = batch_size
            self._joint_strength_grid_batch_size_cache = batch_cache
            self.last_joint_strength_grid_batch_diagnostics = {
                "mode": "empirical_cuda_autotune",
                "configured_batch_size": configured_batch_size,
                "memory_limited_max_batch_size": maximum_batch_size,
                "selected_batch_size": batch_size,
                "trials": trials,
                "shape_key": cache_key,
            }
            trial_summary = ",".join(
                f"{int(trial['batch_size'])}:"
                f"{float(trial['rows_per_second']):.3g}row/s"
                for trial in successful
            )
            print(
                "[joint-smc] strength-grid-batch-autotune "
                f"trials={trial_summary or 'none'} "
                f"selected={batch_size}",
                flush=True,
            )
        elif cache_is_cuda:
            self.last_joint_strength_grid_batch_diagnostics = {
                "mode": "cached_cuda_autotune",
                "configured_batch_size": configured_batch_size,
                "memory_limited_max_batch_size": maximum_batch_size,
                "selected_batch_size": batch_size,
                "shape_key": cache_key,
            }
        else:
            self.last_joint_strength_grid_batch_diagnostics = {
                "mode": "fixed_non_cuda",
                "configured_batch_size": configured_batch_size,
                "selected_batch_size": batch_size,
            }
        for start in range(start, row_count, batch_size):
            stop = min(start + batch_size, row_count)
            _evaluate(start, stop)
        if np.any(np.isnan(result)) or np.any(np.isposinf(result)):
            raise ValueError("Joint strength-grid target contains invalid values.")
        return result

    @staticmethod
    def _joint_strength_grid_autotune_candidates(
        *,
        configured_batch_size: int,
        maximum_batch_size: int,
        row_count: int,
    ) -> tuple[int, ...]:
        """Return ascending power-of-two batches for one real-work trial."""
        configured = max(1, int(configured_batch_size))
        maximum = max(1, min(int(maximum_batch_size), int(row_count)))
        first = min(configured, maximum)
        candidates = [first]
        while candidates[-1] < maximum:
            candidate = min(maximum, candidates[-1] * 2)
            if candidate == candidates[-1]:
                break
            candidates.append(candidate)
        return tuple(candidates)

    def _joint_strength_grid_autotune_key(
        self,
        *,
        cache_tensor: object,
        total_views: int,
        total_slot_count: int,
        line_count: int,
        feature_count: int,
        grid_count: int,
        cardinality: int,
    ) -> tuple[object, ...]:
        """Return the reusable performance-shape key for CUDA batch tuning."""
        return (
            str(getattr(cache_tensor, "device", "numpy")),
            str(getattr(cache_tensor, "dtype", "float64")),
            int(total_views),
            int(total_slot_count),
            int(line_count),
            int(feature_count),
            int(grid_count),
            int(cardinality),
        )

    def _joint_structural_strength_grid_target_batch(
        self,
        *,
        filt: IsotopeParticleFilter,
        data: StructuralGeometryBatch,
        stations: Sequence[JointStationObservation],
        positions_bks: NDArray[np.float64],
        chart_ids_bk: NDArray[np.int64],
        strengths_bgk: NDArray[np.float64],
        particle_indices: NDArray[np.int64],
        target_beta: float,
    ) -> NDArray[np.float64]:
        """Score one bounded fixed-geometry strength-grid batch exactly."""
        cache = self._joint_structural_transport_cache
        if cache is None:
            raise RuntimeError("Joint structural target cache is not active.")
        positions = np.asarray(positions_bks, dtype=np.float64)
        chart_ids = np.asarray(chart_ids_bk, dtype=np.int64)
        strengths = np.asarray(strengths_bgk, dtype=np.float64)
        indices = np.asarray(particle_indices, dtype=np.int64).reshape(-1)
        row_count, grid_count, cardinality = strengths.shape
        total_views = sum(int(station.fe_indices.size) for station in stations)
        order = self.joint_isotope_order()
        model = self._full_spectrum_model()
        line_count = len(tuple(model.line_identity))
        feature_count = len(tuple(model.transport_feature_order))
        slots_per_isotope = int(filt.config.hard_max_sources)
        isotope_index = order.index(str(filt.isotope))
        slot_start = isotope_index * slots_per_isotope
        slot_stop = slot_start + slots_per_isotope
        layout = self._joint_line_layout()
        global_columns, local_indices, branching_weights = layout[str(filt.isotope)]
        local_line_count = int(local_indices.size)
        cached_total, cached_uncollided, cached_features = cache
        cache_is_torch = hasattr(cached_total, "detach")
        if cache_is_torch:
            import torch

            index_tensor = torch.as_tensor(
                indices,
                device=cached_total.device,
                dtype=torch.long,
            )
            selected_total = torch.index_select(
                cached_total,
                0,
                index_tensor,
            )
            selected_uncollided = torch.index_select(
                cached_uncollided,
                0,
                index_tensor,
            )
            selected_features = torch.index_select(
                cached_features,
                0,
                index_tensor,
            )
            global_column_selection = torch.as_tensor(
                global_columns,
                device=cached_total.device,
                dtype=torch.long,
            )
            matched, matched_slot_tensor, accepted_strength_tensor = (
                self._joint_cached_state_match_torch(
                    filt=filt,
                    reference=cached_total,
                    particle_indices=indices,
                    positions_pks=positions,
                    chart_ids_pk=chart_ids,
                )
            )
            accepted_unit_cache = self._joint_cuda_accepted_unit_cache_entry(
                filt=filt,
                data=data,
                positive_line_indices=local_indices,
                reference=cached_total,
            )
            self._joint_promote_pending_cuda_unit_transport(
                filt=filt,
                cache=accepted_unit_cache,
            )
            accepted_matched, _, accepted_unit_components = (
                self._joint_match_cuda_accepted_unit_transport(
                    cache=accepted_unit_cache,
                    particle_indices=indices,
                    positions_pks=positions,
                    chart_ids_pk=chart_ids,
                    reference=cached_total,
                )
            )
            accepted_matched &= ~matched
            unit_total = torch.zeros(
                (
                    row_count,
                    total_views,
                    cardinality,
                    local_line_count,
                ),
                device=cached_total.device,
                dtype=cached_total.dtype,
            )
            unit_uncollided = torch.zeros_like(unit_total)
            unit_features = torch.zeros(
                tuple(unit_total.shape) + (feature_count,),
                device=cached_total.device,
                dtype=cached_total.dtype,
            )
            if cardinality:
                accepted_total = torch.index_select(
                    selected_total[:, :, slot_start:slot_stop, :],
                    3,
                    global_column_selection,
                )
                accepted_uncollided = torch.index_select(
                    selected_uncollided[:, :, slot_start:slot_stop, :],
                    3,
                    global_column_selection,
                )
                accepted_features = torch.index_select(
                    selected_features[:, :, slot_start:slot_stop, :, :],
                    3,
                    global_column_selection,
                )
                line_gather = matched_slot_tensor[:, None, :, None].expand(
                    -1,
                    total_views,
                    -1,
                    local_line_count,
                )
                feature_gather = line_gather[..., None].expand(
                    -1,
                    -1,
                    -1,
                    -1,
                    feature_count,
                )
                gathered_total = torch.gather(
                    accepted_total,
                    2,
                    line_gather,
                )
                gathered_uncollided = torch.gather(
                    accepted_uncollided,
                    2,
                    line_gather,
                )
                gathered_features = torch.gather(
                    accepted_features,
                    2,
                    feature_gather,
                )
                safe_strength = torch.where(
                    matched,
                    accepted_strength_tensor,
                    torch.ones_like(accepted_strength_tensor),
                )[:, None, :, None]
                matched_tensor = matched[:, None, :, None]
                unit_total = torch.where(
                    matched_tensor,
                    gathered_total / safe_strength,
                    unit_total,
                )
                unit_uncollided = torch.where(
                    matched_tensor,
                    gathered_uncollided / safe_strength,
                    unit_uncollided,
                )
                unit_features = torch.where(
                    matched_tensor[..., None],
                    gathered_features,
                    unit_features,
                )
            if cardinality:
                accepted_tensor = accepted_matched[:, None, :, None]
                unit_total = torch.where(
                    accepted_tensor,
                    accepted_unit_components[0],
                    unit_total,
                )
                unit_uncollided = torch.where(
                    accepted_tensor,
                    accepted_unit_components[1],
                    unit_uncollided,
                )
                unit_features = torch.where(
                    accepted_tensor[..., None],
                    accepted_unit_components[2],
                    unit_features,
                )
            all_matched = matched | accepted_matched
            unmatched_index = torch.nonzero(~all_matched, as_tuple=False)
            unmatched_rows = (
                unmatched_index[:, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64, copy=False)
            )
            unmatched_slots = (
                unmatched_index[:, 1]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64, copy=False)
            )
            if unmatched_rows.size:
                device_components = (
                    filt._continuous_rj_line_transport_component_columns(
                        data,
                        positions[unmatched_rows, unmatched_slots],
                        local_indices,
                        chart_ids=chart_ids[unmatched_rows, unmatched_slots],
                        device_resident=True,
                    )
                )
                if not hasattr(device_components.total_kernel, "detach"):
                    raise RuntimeError(
                        "CUDA structural transport returned host components."
                    )
                branch_tensor = torch.as_tensor(
                    branching_weights,
                    device=cached_total.device,
                    dtype=cached_total.dtype,
                )[None, None, :]
                unmatched_total = (
                    device_components.total_kernel.permute(1, 0, 2) * branch_tensor
                )
                unmatched_uncollided = (
                    device_components.uncollided_kernel.permute(1, 0, 2) * branch_tensor
                )
                unmatched_features = torch.stack(
                    (
                        device_components.tau_fe,
                        device_components.tau_pb,
                        device_components.tau_obstacle,
                        device_components.distance_m,
                    ),
                    dim=-1,
                ).permute(1, 0, 2, 3)
                unmatched_row_tensor = torch.as_tensor(
                    unmatched_rows,
                    device=cached_total.device,
                    dtype=torch.long,
                )
                unmatched_slot_tensor = torch.as_tensor(
                    unmatched_slots,
                    device=cached_total.device,
                    dtype=torch.long,
                )
                unit_total[
                    unmatched_row_tensor,
                    :,
                    unmatched_slot_tensor,
                    :,
                ] = unmatched_total
                unit_uncollided[
                    unmatched_row_tensor,
                    :,
                    unmatched_slot_tensor,
                    :,
                ] = unmatched_uncollided
                unit_features[
                    unmatched_row_tensor,
                    :,
                    unmatched_slot_tensor,
                    :,
                    :,
                ] = unmatched_features
            station_count = len(self._joint_structural_station_geometry_shards(data))
            self.last_joint_structural_unit_cache_hits += int(
                torch.count_nonzero(all_matched).item() * station_count
            )
            self.last_joint_structural_unit_cache_misses += int(
                unmatched_rows.size * station_count
            )
            strength_tensor = torch.as_tensor(
                strengths,
                device=cached_total.device,
                dtype=cached_total.dtype,
            )[:, :, None, :, None]
            column_selection = global_column_selection
        else:
            (
                component_total,
                component_uncollided,
                component_tau_fe,
                component_tau_pb,
                component_tau_obstacle,
                component_distance,
            ) = self._joint_cached_continuous_unit_components(
                filt=filt,
                data=data,
                positions_s3=positions.reshape(-1, 3),
                chart_ids_s=chart_ids.reshape(-1),
                positive_line_indices=local_indices,
            )
            local_shape = (
                total_views,
                row_count,
                cardinality,
                local_line_count,
            )
            unit_total_numpy = np.transpose(
                np.asarray(component_total, dtype=np.float64).reshape(local_shape),
                (1, 0, 2, 3),
            ) * branching_weights.reshape(1, 1, 1, -1)
            unit_uncollided_numpy = np.transpose(
                np.asarray(component_uncollided, dtype=np.float64).reshape(local_shape),
                (1, 0, 2, 3),
            ) * branching_weights.reshape(1, 1, 1, -1)
            unit_features_numpy = np.transpose(
                np.stack(
                    (
                        np.asarray(component_tau_fe, dtype=np.float64),
                        np.asarray(component_tau_pb, dtype=np.float64),
                        np.asarray(component_tau_obstacle, dtype=np.float64),
                        np.asarray(component_distance, dtype=np.float64),
                    ),
                    axis=-1,
                ).reshape(local_shape + (feature_count,)),
                (1, 0, 2, 3, 4),
            )
            selected_total = np.asarray(
                cached_total[indices],
                dtype=np.float64,
            )
            selected_uncollided = np.asarray(
                cached_uncollided[indices],
                dtype=np.float64,
            )
            selected_features = np.asarray(
                cached_features[indices],
                dtype=np.float64,
            )
            candidate_total = (
                unit_total_numpy[:, None] * strengths[:, :, None, :, None]
            ).reshape(
                row_count * grid_count,
                total_views,
                cardinality,
                local_line_count,
            )
            candidate_uncollided = (
                unit_uncollided_numpy[:, None] * strengths[:, :, None, :, None]
            ).reshape(candidate_total.shape)
            candidate_features = np.broadcast_to(
                unit_features_numpy[:, None],
                (
                    row_count,
                    grid_count,
                    total_views,
                    cardinality,
                    local_line_count,
                    feature_count,
                ),
            ).reshape(
                row_count * grid_count,
                total_views,
                cardinality,
                local_line_count,
                feature_count,
            )
            column_selection = global_columns
        if cardinality > slots_per_isotope:
            raise ValueError(
                "Conditional candidate cardinality exceeds its source slots."
            )
        if cache_is_torch:
            base_active = torch.any(
                (selected_total != 0.0) | (selected_uncollided != 0.0),
                dim=(0, 1, 3),
            )
            before_indices = torch.nonzero(
                base_active[:slot_start],
                as_tuple=False,
            ).reshape(-1)
            after_indices = (
                torch.nonzero(
                    base_active[slot_stop:],
                    as_tuple=False,
                ).reshape(-1)
                + slot_stop
            )

            before_count = int(before_indices.numel())
            after_count = int(after_indices.numel())
            compact_slot_count = before_count + cardinality + after_count
            total_view = torch.zeros(
                (
                    row_count,
                    grid_count,
                    total_views,
                    compact_slot_count,
                    line_count,
                ),
                device=cached_total.device,
                dtype=cached_total.dtype,
            )
            uncollided_view = torch.zeros_like(total_view)
            features_view = torch.zeros(
                tuple(total_view.shape) + (feature_count,),
                device=cached_total.device,
                dtype=cached_total.dtype,
            )

            def _copy_base_block(
                destination: object,
                source: object,
                source_indices: object,
                destination_start: int,
            ) -> None:
                """Broadcast immutable accepted slots into one final tensor."""
                count = int(source_indices.numel())
                if count == 0:
                    return
                selected = torch.index_select(source, 2, source_indices)
                destination[
                    :,
                    :,
                    :,
                    destination_start : destination_start + count,
                    ...,
                ] = selected[:, None, ...]

            _copy_base_block(total_view, selected_total, before_indices, 0)
            _copy_base_block(
                uncollided_view,
                selected_uncollided,
                before_indices,
                0,
            )
            _copy_base_block(
                features_view,
                selected_features,
                before_indices,
                0,
            )
            after_start = before_count + cardinality
            _copy_base_block(
                total_view,
                selected_total,
                after_indices,
                after_start,
            )
            _copy_base_block(
                uncollided_view,
                selected_uncollided,
                after_indices,
                after_start,
            )
            _copy_base_block(
                features_view,
                selected_features,
                after_indices,
                after_start,
            )
            if cardinality:
                target_slice = slice(before_count, after_start)
                total_view[..., target_slice, :][..., column_selection] = (
                    unit_total[:, None] * strength_tensor
                )
                uncollided_view[..., target_slice, :][..., column_selection] = (
                    unit_uncollided[:, None] * strength_tensor
                )
                features_view[..., target_slice, :, :][
                    ..., column_selection, :
                ] = unit_features[:, None]
            total = total_view.reshape(
                row_count * grid_count,
                total_views,
                compact_slot_count,
                line_count,
            )
            uncollided = uncollided_view.reshape_as(total)
            features = features_view.reshape(
                tuple(total.shape) + (feature_count,)
            )
            self.last_joint_strength_grid_source_slots_before = int(
                selected_total.shape[2]
            )
            self.last_joint_strength_grid_source_slots_after = int(total.shape[2])
            target = self._joint_history_log_likelihood_torch(
                filt=filt,
                stations=stations,
                total_nvsl=total,
                uncollided_nvsl=uncollided,
                features_nvslf=features,
                target_beta=float(target_beta),
                newest_prefix_count=(self._active_joint_tempering_prefix_count),
            )
            return (
                target.detach()
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
                .reshape(row_count, grid_count)
            )
        base_active = np.any(
            (selected_total != 0.0) | (selected_uncollided != 0.0),
            axis=(0, 1, 3),
        )
        before_indices = np.flatnonzero(base_active[:slot_start])
        after_indices = np.flatnonzero(base_active[slot_stop:]) + slot_stop

        def _expanded_base_numpy(
            values: NDArray[np.float64],
            slot_indices: NDArray[np.int64],
        ) -> NDArray[np.float64]:
            """Select active immutable slots and broadcast the grid axis."""
            selected = values[:, :, slot_indices, ...]
            return np.broadcast_to(
                selected[:, None],
                (row_count, grid_count) + selected.shape[1:],
            ).reshape(row_count * grid_count, *selected.shape[1:])

        target_total_numpy = np.zeros(
            (
                row_count * grid_count,
                total_views,
                cardinality,
                line_count,
            ),
            dtype=np.float64,
        )
        target_uncollided_numpy = np.zeros_like(target_total_numpy)
        target_features_numpy = np.zeros(
            target_total_numpy.shape + (feature_count,),
            dtype=np.float64,
        )
        if cardinality:
            target_total_numpy[..., column_selection] = candidate_total
            target_uncollided_numpy[..., column_selection] = candidate_uncollided
            target_features_numpy[..., column_selection, :] = candidate_features
        total_numpy = np.concatenate(
            (
                _expanded_base_numpy(selected_total, before_indices),
                target_total_numpy,
                _expanded_base_numpy(selected_total, after_indices),
            ),
            axis=2,
        )
        uncollided_numpy = np.concatenate(
            (
                _expanded_base_numpy(selected_uncollided, before_indices),
                target_uncollided_numpy,
                _expanded_base_numpy(selected_uncollided, after_indices),
            ),
            axis=2,
        )
        features_numpy = np.concatenate(
            (
                _expanded_base_numpy(selected_features, before_indices),
                target_features_numpy,
                _expanded_base_numpy(selected_features, after_indices),
            ),
            axis=2,
        )
        self.last_joint_strength_grid_source_slots_before = int(selected_total.shape[2])
        self.last_joint_strength_grid_source_slots_after = int(total_numpy.shape[2])
        return self._joint_history_log_likelihood_numpy(
            filt=filt,
            stations=stations,
            total_nvsl=total_numpy,
            uncollided_nvsl=uncollided_numpy,
            features_nvslf=features_numpy,
            target_beta=float(target_beta),
            newest_prefix_count=self._active_joint_tempering_prefix_count,
        ).reshape(row_count, grid_count)

    @staticmethod
    def _joint_isotope_state_log_prior(
        filt: IsotopeParticleFilter,
        state: IsotopeState,
    ) -> float:
        """Return the exact labeled continuous-state prior log density."""
        atlas = filt._structural_rj_surface_atlas
        cardinality_prior = filt._structural_rj_cardinality_prior
        if atlas is None or cardinality_prior is None:
            raise RuntimeError("Cross-isotope transfer priors are unavailable.")
        cardinality = int(state.num_sources)
        value = float(cardinality_prior.log_prob(cardinality)) + math.lgamma(
            float(cardinality) + 1.0
        )
        if cardinality:
            value += float(
                np.sum(
                    atlas.log_chart_probabilities[state.surface_chart_ids]
                    + np.asarray(
                        filt._strength_prior.log_prob(state.strengths),
                        dtype=np.float64,
                    )
                )
            )
        return value

    @staticmethod
    def _cross_isotope_transferred_states(
        donor_filter: IsotopeParticleFilter,
        receiver_filter: IsotopeParticleFilter,
        donor_state: IsotopeState,
        receiver_state: IsotopeState,
        transferred_indices: NDArray[np.int64],
    ) -> tuple[IsotopeState, IsotopeState]:
        """Return canonical donor/receiver states after identity transfer."""
        selected = np.asarray(transferred_indices, dtype=np.int64).reshape(-1)
        donor_count = int(donor_state.num_sources)
        if (
            selected.size == 0
            or np.unique(selected).size != selected.size
            or np.any(selected < 0)
            or np.any(selected >= donor_count)
        ):
            raise ValueError("Cross-isotope transferred indices are invalid.")
        keep = np.ones(donor_count, dtype=bool)
        keep[selected] = False
        new_donor = IsotopeState(
            num_sources=int(np.sum(keep)),
            strengths=donor_state.strengths[keep],
            surface_chart_ids=donor_state.surface_chart_ids[keep],
            surface_uv=donor_state.surface_uv[keep],
        )
        new_receiver = IsotopeState(
            num_sources=int(receiver_state.num_sources + selected.size),
            strengths=np.concatenate(
                (receiver_state.strengths, donor_state.strengths[selected])
            ),
            surface_chart_ids=np.concatenate(
                (
                    receiver_state.surface_chart_ids,
                    donor_state.surface_chart_ids[selected],
                )
            ),
            surface_uv=np.concatenate(
                (receiver_state.surface_uv, donor_state.surface_uv[selected]),
                axis=0,
            ),
        )
        donor_filter._canonicalize_structural_rj_state(new_donor)
        receiver_filter._canonicalize_structural_rj_state(new_receiver)
        return new_donor, new_receiver

    def _evaluate_cross_isotope_receiver_states(
        self,
        *,
        stations: Sequence[JointStationObservation],
        receiver_states: Mapping[int, IsotopeState],
        receiver_by_row: NDArray[np.int64],
        isotope_order: tuple[str, ...],
        target_beta: float,
    ) -> NDArray[np.float64]:
        """Evaluate batched receiver states with temporary donor cache active."""
        proposed_target = np.full(
            receiver_by_row.size,
            float("nan"),
            dtype=np.float64,
        )
        for receiver_index, isotope in enumerate(isotope_order):
            rows_for_isotope = np.flatnonzero(receiver_by_row == receiver_index)
            if rows_for_isotope.size == 0:
                continue
            filt = self.filters[isotope]
            evidence = self._joint_history_structural_geometry(
                isotope,
                stations,
            )
            cardinalities = np.asarray(
                [receiver_states[int(row)].num_sources for row in rows_for_isotope],
                dtype=np.int64,
            )
            for cardinality in np.unique(cardinalities).tolist():
                local = np.flatnonzero(cardinalities == int(cardinality))
                rows = rows_for_isotope[local]
                states = [receiver_states[int(row)] for row in rows]
                if int(cardinality):
                    chart_ids = np.stack(
                        [state.surface_chart_ids for state in states],
                        axis=0,
                    )
                    strengths = np.stack(
                        [state.strengths for state in states],
                        axis=0,
                    )
                    positions = np.stack(
                        [filt.continuous_state_positions(state) for state in states],
                        axis=0,
                    )
                else:
                    chart_ids = np.empty((rows.size, 0), dtype=np.int64)
                    strengths = np.empty((rows.size, 0), dtype=np.float64)
                    positions = np.empty((rows.size, 0, 3), dtype=np.float64)
                proposed_target[rows] = self._joint_structural_target_evaluator(
                    filt=filt,
                    data=evidence,
                    positions_pks=positions,
                    chart_ids_pk=chart_ids,
                    strengths_pk=strengths,
                    particle_indices=rows,
                    target_beta=float(target_beta),
                    tempering_start_row=sum(
                        int(station.fe_indices.size) for station in stations[:-1]
                    ),
                )
        evaluated_rows = np.asarray(
            sorted(receiver_states),
            dtype=np.int64,
        )
        if np.any(~np.isfinite(proposed_target[evaluated_rows])):
            raise RuntimeError("Cross-isotope target evaluation was incomplete.")
        return proposed_target

    def _joint_packed_strength_state(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Return all isotope strengths in the fixed joint slot layout."""
        strengths_by_isotope: list[NDArray[np.float64]] = []
        masks_by_isotope: list[NDArray[np.bool_]] = []
        for isotope in self.joint_isotope_order():
            _, strengths, active_mask, _, _ = self.filters[
                isotope
            ]._packed_continuous_surface_state_arrays()
            strengths_by_isotope.append(np.asarray(strengths, dtype=np.float64))
            masks_by_isotope.append(np.asarray(active_mask, dtype=np.bool_))
        return (
            np.concatenate(strengths_by_isotope, axis=1),
            np.concatenate(masks_by_isotope, axis=1),
        )

    def _joint_strength_block_target(
        self,
        stations: Sequence[JointStationObservation],
        *,
        particle_indices: NDArray[np.int64],
        scale_ps: NDArray[np.float64],
        target_beta: float,
    ) -> NDArray[np.float64]:
        """Evaluate one strength-only proposal from the accepted GPU cache.

        The immutable unit transport, surface geometry, and line layout do not
        change when strengths move.  Scaling cached source columns is therefore
        exactly equivalent to transport recomputation and avoids new kernels.
        Rows are evaluated in bounded batches to cap temporary GPU memory.
        """
        cache = self._joint_structural_transport_cache
        if cache is None:
            raise RuntimeError("Joint strength moves require an active cache.")
        indices = np.asarray(particle_indices, dtype=np.int64).reshape(-1)
        scale = np.asarray(scale_ps, dtype=np.float64)
        if scale.shape != (indices.size, int(cache[0].shape[2])):
            raise ValueError("Joint strength scales do not match cache slots.")
        result = np.empty(indices.size, dtype=np.float64)
        batch_size = int(self.pf_config.joint_strength_block_batch_size)
        filt = self.filters[self.joint_isotope_order()[0]]
        cache_is_torch = hasattr(cache[0], "detach")
        for start in range(0, indices.size, batch_size):
            stop = min(start + batch_size, indices.size)
            selected = indices[start:stop]
            selected_scale = scale[start:stop]
            if cache_is_torch:
                import torch

                index_tensor = torch.as_tensor(
                    selected,
                    device=cache[0].device,
                    dtype=torch.long,
                )
                scale_tensor = torch.as_tensor(
                    selected_scale,
                    device=cache[0].device,
                    dtype=cache[0].dtype,
                )[:, None, :, None]
                total = torch.index_select(cache[0], 0, index_tensor) * scale_tensor
                uncollided = (
                    torch.index_select(cache[1], 0, index_tensor) * scale_tensor
                )
                features = torch.index_select(cache[2], 0, index_tensor)
                batch_result = self._joint_history_log_likelihood_torch(
                    filt=filt,
                    stations=stations,
                    total_nvsl=total,
                    uncollided_nvsl=uncollided,
                    features_nvslf=features,
                    target_beta=float(target_beta),
                    newest_prefix_count=(self._active_joint_tempering_prefix_count),
                )
                result[start:stop] = (
                    batch_result.detach().cpu().numpy().astype(np.float64, copy=False)
                )
            else:
                slot_scale = selected_scale[:, None, :, None]
                result[start:stop] = self._joint_history_log_likelihood_numpy(
                    filt=filt,
                    stations=stations,
                    total_nvsl=np.asarray(cache[0][selected]) * slot_scale,
                    uncollided_nvsl=(np.asarray(cache[1][selected]) * slot_scale),
                    features_nvslf=np.asarray(cache[2][selected]),
                    target_beta=float(target_beta),
                    newest_prefix_count=(self._active_joint_tempering_prefix_count),
                )
        return result

    def _joint_active_cache_target(
        self,
        stations: Sequence[JointStationObservation],
        *,
        particle_indices: NDArray[np.int64],
        target_beta: float,
    ) -> NDArray[np.float64]:
        """Evaluate selected rows of the currently active joint GPU cache."""
        cache = self._joint_structural_transport_cache
        if cache is None:
            raise RuntimeError("A joint structural cache is not active.")
        rows = np.asarray(particle_indices, dtype=np.int64).reshape(-1)
        if np.any(rows < 0) or np.any(rows >= int(cache[0].shape[0])):
            raise ValueError("Joint cache particle indices are invalid.")
        filt = self.filters[self.joint_isotope_order()[0]]
        if hasattr(cache[0], "detach"):
            import torch

            selected = torch.as_tensor(
                rows,
                device=cache[0].device,
                dtype=torch.long,
            )
            result = self._joint_history_log_likelihood_torch(
                filt=filt,
                stations=stations,
                total_nvsl=torch.index_select(cache[0], 0, selected),
                uncollided_nvsl=torch.index_select(cache[1], 0, selected),
                features_nvslf=torch.index_select(cache[2], 0, selected),
                target_beta=float(target_beta),
                newest_prefix_count=self._active_joint_tempering_prefix_count,
            )
            return result.detach().cpu().numpy().astype(np.float64, copy=False)
        return self._joint_history_log_likelihood_numpy(
            filt=filt,
            stations=stations,
            total_nvsl=np.asarray(cache[0][rows]),
            uncollided_nvsl=np.asarray(cache[1][rows]),
            features_nvslf=np.asarray(cache[2][rows]),
            target_beta=float(target_beta),
            newest_prefix_count=self._active_joint_tempering_prefix_count,
        )

    def _commit_joint_strength_block(
        self,
        accepted_rows: NDArray[np.int64],
        proposed_strengths_ps: NDArray[np.float64],
        old_strengths_ps: NDArray[np.float64],
    ) -> None:
        """Commit accepted joint strengths and rescale their cache rows."""
        rows = np.asarray(accepted_rows, dtype=np.int64).reshape(-1)
        if rows.size == 0:
            return
        proposed = np.asarray(proposed_strengths_ps, dtype=np.float64)
        old = np.asarray(old_strengths_ps, dtype=np.float64)
        slots_per_isotope = self.pf_config.cardinality_capacity
        for isotope_index, isotope in enumerate(self.joint_isotope_order()):
            filt = self.filters[isotope]
            cardinalities = np.asarray(
                [filt.continuous_particles[int(row)].state.num_sources for row in rows],
                dtype=np.int64,
            )
            slot_start = isotope_index * slots_per_isotope
            for cardinality in np.unique(cardinalities).tolist():
                local = np.flatnonzero(cardinalities == int(cardinality))
                isotope_rows = rows[local]
                charts, uv, positions, _ = filt._continuous_rj_group_arrays(
                    isotope_rows,
                    int(cardinality),
                )
                strengths = proposed[
                    local,
                    slot_start : slot_start + int(cardinality),
                ]
                filt._commit_continuous_rj_states(
                    isotope_rows,
                    np.ones(isotope_rows.size, dtype=np.bool_),
                    charts,
                    uv,
                    positions,
                    strengths,
                )
        cache = self._joint_structural_transport_cache
        if cache is None:
            raise RuntimeError("Joint strength cache disappeared during commit.")
        scale = np.divide(
            proposed,
            old,
            out=np.ones_like(proposed),
            where=old > 0.0,
        )
        if hasattr(cache[0], "detach"):
            import torch

            index_tensor = torch.as_tensor(
                rows,
                device=cache[0].device,
                dtype=torch.long,
            )
            scale_tensor = torch.as_tensor(
                scale,
                device=cache[0].device,
                dtype=cache[0].dtype,
            )[:, None, :, None]
            for cached_values in cache[:2]:
                updated = (
                    torch.index_select(
                        cached_values,
                        0,
                        index_tensor,
                    )
                    * scale_tensor
                )
                cached_values.index_copy_(0, index_tensor, updated)
        else:
            for cached_values in cache[:2]:
                cached_values[rows] *= scale[:, None, :, None]
        self._joint_persistent_structural_transport_cache = cache
        signatures = self._joint_station_cache_signatures(
            self._active_joint_station_history or ()
        )
        if signatures is None:
            self._joint_persistent_structural_state_sha256 = None
        else:
            self._joint_persistent_structural_station_signature = signatures
            self._joint_persistent_structural_state_sha256 = (
                self._joint_structural_state_sha256()
            )

    def _joint_mh_acceptance_mask(
        self,
        log_ratio: NDArray[np.float64],
        *,
        rng: np.random.Generator,
        support: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.bool_]:
        """Draw a joint MH mask on CUDA when the active cache is CUDA-backed."""
        ratios = np.asarray(log_ratio, dtype=np.float64).reshape(-1)
        with np.errstate(divide="ignore"):
            log_uniforms = np.log(rng.random(ratios.size))
        thresholds = np.minimum(ratios, 0.0)
        feasible = None
        if support is not None:
            feasible = np.asarray(support, dtype=np.bool_).reshape(-1)
            if feasible.size != ratios.size:
                raise ValueError("Joint MH support must align with log_ratio.")
        cache = self._joint_structural_transport_cache
        if (
            cache is None
            or not hasattr(cache[0], "detach")
            or not bool(cache[0].is_cuda)
        ):
            accepted = log_uniforms < thresholds
            if feasible is not None:
                accepted &= feasible
            return np.asarray(accepted, dtype=np.bool_)
        import torch

        accepted_tensor = torch.as_tensor(
            log_uniforms,
            device=cache[0].device,
            dtype=torch.float64,
        ) < torch.as_tensor(
            thresholds,
            device=cache[0].device,
            dtype=torch.float64,
        )
        if feasible is not None:
            accepted_tensor &= torch.as_tensor(
                feasible,
                device=cache[0].device,
                dtype=torch.bool,
            )
        self.last_joint_device_mh_acceptance_calls += 1
        self.last_joint_device_mh_acceptance_rows += int(ratios.size)
        return (
            accepted_tensor.detach()
            .cpu()
            .numpy()
            .astype(np.bool_, copy=False)
        )

    def _apply_joint_strength_block(
        self,
        stations: Sequence[JointStationObservation],
        *,
        target_beta: float,
        current_target_log_likelihood: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply one exact all-isotope strength move without conservation.

        Each active source receives a symmetric Gaussian increment in shifted
        log-strength coordinates.  Isotope totals are not constrained or
        transferred.  The shared spectrum likelihood alone decides whether a
        simultaneous decrease in one isotope and increase in another is useful.
        """
        probability = float(self.pf_config.joint_strength_block_probability)
        current_target = np.asarray(
            current_target_log_likelihood,
            dtype=np.float64,
        )
        if probability <= 0.0:
            return current_target
        current_strengths, active_mask = self._joint_packed_strength_state()
        rng = self._joint_random_generator
        attempted = (rng.random(current_strengths.shape[0]) < probability) & np.any(
            active_mask, axis=1
        )
        rows = np.flatnonzero(attempted).astype(np.int64, copy=False)
        weights = self._strict_joint_particle_weights()
        self.last_joint_strength_block_attempted_weight_mass += float(
            np.sum(weights[rows])
        )
        if rows.size == 0:
            return current_target
        minimum = float(self.pf_config.strength_prior_min_cps_1m)
        old = current_strengths[rows]
        mask = active_mask[rows]
        shifted = old - minimum
        valid = np.all(~mask | (shifted > 0.0), axis=1)
        noise = rng.normal(
            loc=0.0,
            scale=float(self.pf_config.joint_strength_block_log_sigma),
            size=old.shape,
        )
        proposed = old.copy()
        proposed[mask] = minimum + shifted[mask] * np.exp(noise[mask])
        log_prior_ratio = np.zeros(rows.size, dtype=np.float64)
        slots_per_isotope = self.pf_config.cardinality_capacity
        for isotope_index, isotope in enumerate(self.joint_isotope_order()):
            slot = slice(
                isotope_index * slots_per_isotope,
                (isotope_index + 1) * slots_per_isotope,
            )
            isotope_mask = mask[:, slot]
            prior = self.filters[isotope]._strength_prior
            safe_proposed = np.where(
                isotope_mask,
                proposed[:, slot],
                prior.mean,
            )
            safe_old = np.where(
                isotope_mask,
                old[:, slot],
                prior.mean,
            )
            log_prior_ratio += np.sum(
                np.where(
                    isotope_mask,
                    np.asarray(prior.log_prob(safe_proposed))
                    - np.asarray(prior.log_prob(safe_old)),
                    0.0,
                ),
                axis=1,
                dtype=np.float64,
            )
        log_proposal_ratio = shifted_log_strength_random_walk_log_reverse_ratio(
            old,
            proposed,
            mask,
            minimum_strength=minimum,
        )
        valid &= np.isfinite(log_prior_ratio) & np.isfinite(log_proposal_ratio)
        scale = np.ones_like(old)
        scale[mask] = proposed[mask] / old[mask]
        proposed_target = np.full(rows.size, float("-inf"), dtype=np.float64)
        valid_local = np.flatnonzero(valid)
        if valid_local.size:
            proposed_target[valid_local] = self._joint_strength_block_target(
                stations,
                particle_indices=rows[valid_local],
                scale_ps=scale[valid_local],
                target_beta=float(target_beta),
            )
        log_ratio = (
            proposed_target
            - current_target[rows]
            + log_prior_ratio
            + log_proposal_ratio
        )
        accepted = self._joint_mh_acceptance_mask(
            log_ratio,
            rng=rng,
            support=valid,
        )
        accepted_rows = rows[accepted]
        if accepted_rows.size:
            self._commit_joint_strength_block(
                accepted_rows,
                proposed[accepted],
                old[accepted],
            )
            current_target = current_target.copy()
            current_target[accepted_rows] = proposed_target[accepted]
            self.last_joint_strength_block_accepted_weight_mass += float(
                np.sum(weights[accepted_rows])
            )
        return current_target

    def _apply_joint_cross_isotope_transfer(
        self,
        stations: Sequence[JointStationObservation],
        *,
        target_beta: float,
        current_target_log_likelihood: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply an exact multi-source isotope-identity transfer proposal.

        A uniformly selected subset of one isotope is relabeled as another
        isotope while retaining its continuous surface coordinates and
        strengths.  The state-dependent subset and group-size probabilities,
        both isotope priors, and the full joint likelihood are all included in
        the MH ratio.  This crosses the likelihood barrier created when a clear
        peak has been assigned to the wrong isotope during early resampling.
        """
        probability = float(self.pf_config.joint_cross_isotope_transfer_probability)
        isotope_order = self.joint_isotope_order()
        if probability <= 0.0 or len(isotope_order) < 2:
            return np.asarray(current_target_log_likelihood, dtype=np.float64)
        particle_count = int(self.pf_config.num_particles)
        weights = self._strict_joint_particle_weights()
        rng = self._joint_random_generator
        attempted = rng.random(particle_count) < probability
        donor_by_row = rng.integers(
            0,
            len(isotope_order),
            size=particle_count,
        )
        receiver_offset = rng.integers(
            1,
            len(isotope_order),
            size=particle_count,
        )
        receiver_by_row = (donor_by_row + receiver_offset) % len(isotope_order)
        cardinalities = np.stack(
            [
                np.asarray(
                    [
                        particle.state.num_sources
                        for particle in self.filters[isotope].continuous_particles
                    ],
                    dtype=np.int64,
                )
                for isotope in isotope_order
            ],
            axis=1,
        )
        row_indices = np.arange(particle_count, dtype=np.int64)
        donor_cardinality = cardinalities[row_indices, donor_by_row]
        receiver_cardinality = cardinalities[row_indices, receiver_by_row]
        maximum_sources = self.pf_config.cardinality_capacity
        maximum_group = np.minimum.reduce(
            (
                donor_cardinality,
                maximum_sources - receiver_cardinality,
                np.full(
                    particle_count,
                    int(self.pf_config.joint_cross_isotope_transfer_max_group),
                    dtype=np.int64,
                ),
            )
        )
        feasible = attempted & (maximum_group > 0)
        attempted_rows = np.flatnonzero(attempted)
        rows = np.flatnonzero(feasible)
        self.last_joint_cross_isotope_attempted_weight_mass += float(
            np.sum(weights[attempted_rows])
        )
        if rows.size == 0:
            self.last_joint_cross_isotope_rejection_diagnostics = {
                "attempted": int(attempted_rows.size),
                "accepted": 0,
                "support_rejected": int(attempted_rows.size),
                "geometry_support_rejected": 0,
                "strength_support_rejected": 0,
                "other_support_rejected": int(attempted_rows.size),
                "nonfinite_rejected": 0,
                "mh_random_rejected": 0,
                "component_quantiles": {},
                "by_isotope_cardinality_transfer": {},
            }
            return np.asarray(current_target_log_likelihood, dtype=np.float64)
        group_sizes = np.ones(particle_count, dtype=np.int64)
        group_sizes[rows] = (
            np.floor(rng.random(rows.size) * maximum_group[rows]).astype(np.int64) + 1
        )
        donor_states: dict[int, IsotopeState] = {}
        receiver_states: dict[int, IsotopeState] = {}
        old_donor_states: dict[int, IsotopeState] = {}
        old_receiver_states: dict[int, IsotopeState] = {}
        log_forward = np.full(particle_count, float("-inf"), dtype=np.float64)
        log_reverse = np.full(particle_count, float("-inf"), dtype=np.float64)
        log_prior_ratio = np.zeros(particle_count, dtype=np.float64)
        for row in rows.tolist():
            donor_filter = self.filters[isotope_order[donor_by_row[row]]]
            receiver_filter = self.filters[isotope_order[receiver_by_row[row]]]
            donor_state = donor_filter.continuous_particles[row].state
            receiver_state = receiver_filter.continuous_particles[row].state
            group_size = int(group_sizes[row])
            selected = np.sort(
                rng.choice(
                    int(donor_state.num_sources),
                    size=group_size,
                    replace=False,
                )
            )
            proposed_donor, proposed_receiver = self._cross_isotope_transferred_states(
                donor_filter,
                receiver_filter,
                donor_state,
                receiver_state,
                selected,
            )
            donor_states[row] = proposed_donor
            receiver_states[row] = proposed_receiver
            old_donor_states[row] = donor_state.copy()
            old_receiver_states[row] = receiver_state.copy()
            log_forward[row] = cross_isotope_transfer_log_proposal(
                donor_cardinality=int(donor_state.num_sources),
                receiver_cardinality=int(receiver_state.num_sources),
                group_size=group_size,
                maximum_sources=maximum_sources,
                maximum_group_size=int(
                    self.pf_config.joint_cross_isotope_transfer_max_group
                ),
            )
            log_reverse[row] = cross_isotope_transfer_log_proposal(
                donor_cardinality=int(proposed_receiver.num_sources),
                receiver_cardinality=int(proposed_donor.num_sources),
                group_size=group_size,
                maximum_sources=maximum_sources,
                maximum_group_size=int(
                    self.pf_config.joint_cross_isotope_transfer_max_group
                ),
            )
            old_prior = self._joint_isotope_state_log_prior(
                donor_filter,
                donor_state,
            ) + self._joint_isotope_state_log_prior(
                receiver_filter,
                receiver_state,
            )
            new_prior = self._joint_isotope_state_log_prior(
                donor_filter,
                proposed_donor,
            ) + self._joint_isotope_state_log_prior(
                receiver_filter,
                proposed_receiver,
            )
            log_prior_ratio[row] = new_prior - old_prior

        touched_donors = sorted(set(donor_by_row[rows].tolist()))
        for row in rows.tolist():
            donor_filter = self.filters[isotope_order[donor_by_row[row]]]
            donor_filter.continuous_particles[row].state = donor_states[row]
        try:
            for isotope_index in touched_donors:
                isotope_rows = rows[donor_by_row[rows] == isotope_index]
                self._refresh_joint_structural_transport_cache_isotope(
                    stations,
                    isotope_order[isotope_index],
                    particle_indices=isotope_rows,
                )
            proposed_target = self._evaluate_cross_isotope_receiver_states(
                stations=stations,
                receiver_states=receiver_states,
                receiver_by_row=np.where(
                    feasible,
                    receiver_by_row,
                    -1,
                ),
                isotope_order=isotope_order,
                target_beta=float(target_beta),
            )
        finally:
            for row in rows.tolist():
                donor_filter = self.filters[isotope_order[donor_by_row[row]]]
                donor_filter.continuous_particles[row].state = old_donor_states[row]
            for isotope_index in touched_donors:
                isotope_rows = rows[donor_by_row[rows] == isotope_index]
                self._refresh_joint_structural_transport_cache_isotope(
                    stations,
                    isotope_order[isotope_index],
                    particle_indices=isotope_rows,
                )
        current_target = np.asarray(
            current_target_log_likelihood,
            dtype=np.float64,
        )
        log_ratio = (
            proposed_target[rows]
            - current_target[rows]
            + log_prior_ratio[rows]
            + log_reverse[rows]
            - log_forward[rows]
        )
        accepted_local = self._joint_mh_acceptance_mask(
            log_ratio,
            rng=rng,
        )
        accepted_rows = rows[accepted_local]
        for row in accepted_rows.tolist():
            donor_filter = self.filters[isotope_order[donor_by_row[row]]]
            receiver_filter = self.filters[isotope_order[receiver_by_row[row]]]
            donor_filter.continuous_particles[row].state = donor_states[row]
            receiver_filter.continuous_particles[row].state = receiver_states[row]
        if accepted_rows.size:
            touched = sorted(
                set(donor_by_row[accepted_rows].tolist())
                | set(receiver_by_row[accepted_rows].tolist())
            )
            for isotope_index in touched:
                isotope_rows = accepted_rows[
                    (donor_by_row[accepted_rows] == isotope_index)
                    | (receiver_by_row[accepted_rows] == isotope_index)
                ]
                self._refresh_joint_structural_transport_cache_isotope(
                    stations,
                    isotope_order[isotope_index],
                    particle_indices=isotope_rows,
                )
            current_target = current_target.copy()
            current_target[accepted_rows] = proposed_target[accepted_rows]
            self.last_joint_cross_isotope_accepted_weight_mass += float(
                np.sum(weights[accepted_rows])
            )
            self._invalidate_posterior_summary_cache()
        diagnostic_delta_likelihood = np.full(
            particle_count,
            float("nan"),
            dtype=np.float64,
        )
        diagnostic_log_ratio = np.full_like(
            diagnostic_delta_likelihood,
            float("nan"),
        )
        diagnostic_delta_likelihood[rows] = (
            proposed_target[rows] - current_target_log_likelihood[rows]
        )
        diagnostic_log_ratio[rows] = log_ratio
        diagnostic_accepted = np.zeros(particle_count, dtype=np.bool_)
        diagnostic_accepted[accepted_rows] = True
        diagnostic_strength_support = np.ones(
            particle_count,
            dtype=np.bool_,
        )
        for row in rows.tolist():
            donor_filter = self.filters[isotope_order[donor_by_row[row]]]
            receiver_filter = self.filters[isotope_order[receiver_by_row[row]]]
            diagnostic_strength_support[row] = bool(
                np.all(
                    donor_filter._strength_prior.in_support(
                        donor_states[row].strengths
                    )
                )
                and np.all(
                    receiver_filter._strength_prior.in_support(
                        receiver_states[row].strengths
                    )
                )
            )
        diagnostic_support = (
            feasible
            & diagnostic_strength_support
            & np.isfinite(log_forward)
            & np.isfinite(log_reverse)
            & np.isfinite(log_prior_ratio)
        )
        diagnostic_proposal_ratio = np.full(
            particle_count,
            float("nan"),
            dtype=np.float64,
        )
        diagnostic_proposal_ratio[rows] = (
            log_reverse[rows] - log_forward[rows]
        )
        self.last_joint_cross_isotope_rejection_diagnostics = (
            self._summarize_joint_cross_isotope_transfer(
                attempted_rows=attempted_rows,
                donor_by_row=donor_by_row,
                receiver_by_row=receiver_by_row,
                donor_cardinality=donor_cardinality,
                receiver_cardinality=receiver_cardinality,
                group_sizes=group_sizes,
                isotope_order=isotope_order,
                delta_log_likelihood=diagnostic_delta_likelihood,
                delta_log_prior=log_prior_ratio,
                log_reverse_minus_forward=diagnostic_proposal_ratio,
                log_acceptance_ratio=diagnostic_log_ratio,
                support_feasible=diagnostic_support,
                strength_support_feasible=diagnostic_strength_support,
                accepted=diagnostic_accepted,
            )
        )
        self._assert_joint_particle_alignment()
        return current_target

    @staticmethod
    def _summarize_joint_cross_isotope_transfer(
        *,
        attempted_rows: NDArray[np.int64],
        donor_by_row: NDArray[np.int64],
        receiver_by_row: NDArray[np.int64],
        donor_cardinality: NDArray[np.int64],
        receiver_cardinality: NDArray[np.int64],
        group_sizes: NDArray[np.int64],
        isotope_order: Sequence[str],
        delta_log_likelihood: NDArray[np.float64],
        delta_log_prior: NDArray[np.float64],
        log_reverse_minus_forward: NDArray[np.float64],
        log_acceptance_ratio: NDArray[np.float64],
        support_feasible: NDArray[np.bool_],
        strength_support_feasible: NDArray[np.bool_],
        accepted: NDArray[np.bool_],
    ) -> dict[str, object]:
        """Summarize exact isotope-transfer MH terms on attempted rows."""
        selected = np.asarray(attempted_rows, dtype=np.int64).reshape(-1)
        quantile_levels = np.asarray(
            [0.0, 0.1, 0.5, 0.9, 1.0],
            dtype=np.float64,
        )
        numeric = {
            "delta_log_likelihood": np.asarray(
                delta_log_likelihood,
                dtype=np.float64,
            ),
            "delta_log_prior": np.asarray(
                delta_log_prior,
                dtype=np.float64,
            ),
            "log_reverse_minus_forward": np.asarray(
                log_reverse_minus_forward,
                dtype=np.float64,
            ),
            "log_jacobian": np.zeros(
                np.asarray(delta_log_likelihood).shape,
                dtype=np.float64,
            ),
            "log_acceptance_ratio": np.asarray(
                log_acceptance_ratio,
                dtype=np.float64,
            ),
        }
        support = np.asarray(support_feasible, dtype=np.bool_)
        strength_support = np.asarray(
            strength_support_feasible,
            dtype=np.bool_,
        )
        acceptance = np.asarray(accepted, dtype=np.bool_)

        def _rows_summary(rows: NDArray[np.int64]) -> dict[str, object]:
            """Summarize one batched subset of transfer attempts."""
            finite_all = support[rows].copy()
            component_quantiles: dict[
                str,
                dict[str, float | int] | None,
            ] = {}
            for name, all_values in numeric.items():
                values = all_values[rows]
                finite = np.isfinite(values)
                finite_all &= finite
                if not np.any(finite):
                    component_quantiles[name] = None
                    continue
                finite_values = values[finite]
                resolved = np.quantile(finite_values, quantile_levels)
                component_quantiles[name] = {
                    "finite_count": int(finite_values.size),
                    "mean": float(np.mean(finite_values)),
                    "std": float(np.std(finite_values)),
                    **{
                        label: float(value)
                        for label, value in zip(
                            ("min", "p10", "median", "p90", "max"),
                            resolved,
                            strict=True,
                        )
                    },
                }
            return {
                "attempted": int(rows.size),
                "accepted": int(np.count_nonzero(acceptance[rows])),
                "support_rejected": int(np.count_nonzero(~support[rows])),
                "geometry_support_rejected": 0,
                "strength_support_rejected": int(
                    np.count_nonzero(~strength_support[rows])
                ),
                "other_support_rejected": int(
                    np.count_nonzero(strength_support[rows] & ~support[rows])
                ),
                "nonfinite_rejected": int(
                    np.count_nonzero(support[rows] & ~finite_all)
                ),
                "mh_random_rejected": int(
                    np.count_nonzero(
                        support[rows]
                        & finite_all
                        & ~acceptance[rows]
                    )
                ),
                "component_quantiles": component_quantiles,
            }

        summary = _rows_summary(selected)
        transitions: dict[str, object] = {}
        # Isotope pairs and group sizes are bounded by the configured isotope
        # set and max transfer group, so this packages batched arrays only.
        if selected.size:
            labels = np.stack(
                (
                    np.asarray(donor_by_row, dtype=np.int64),
                    np.asarray(receiver_by_row, dtype=np.int64),
                    np.asarray(donor_cardinality, dtype=np.int64),
                    np.asarray(receiver_cardinality, dtype=np.int64),
                    np.asarray(group_sizes, dtype=np.int64),
                ),
                axis=1,
            )
            for donor, receiver, donor_k, receiver_k, group_size in np.unique(
                labels[selected],
                axis=0,
            ).tolist():
                matching = selected[
                    np.all(
                        labels[selected]
                        == np.asarray(
                            (
                                donor,
                                receiver,
                                donor_k,
                                receiver_k,
                                group_size,
                            ),
                            dtype=np.int64,
                        ),
                        axis=1,
                    )
                ]
                key = (
                    f"{isotope_order[int(donor)]}:{int(donor_k)}"
                    f"->{int(donor_k) - int(group_size)}|"
                    f"{isotope_order[int(receiver)]}:{int(receiver_k)}"
                    f"->{int(receiver_k) + int(group_size)}"
                )
                transitions[key] = _rows_summary(matching)
        summary["by_isotope_cardinality_transfer"] = transitions
        return summary

    def _apply_joint_cross_isotope_state_block(
        self,
        stations: Sequence[JointStationObservation],
        *,
        target_beta: float,
        current_target_log_likelihood: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Jointly propose independent isotope states without strength transfer.

        Each isotope retains its own cardinality, surface-position, and strength
        priors. The proposal is coupled only by making one simultaneous MH
        decision under the shared full-spectrum likelihood; no activity or
        detector-cps quantity is conserved between isotope labels.
        """
        probability = float(self.pf_config.joint_cross_isotope_state_block_probability)
        current_target = np.asarray(
            current_target_log_likelihood,
            dtype=np.float64,
        )
        if probability <= 0.0:
            return current_target
        particle_count = int(self.pf_config.num_particles)
        rng = self._joint_random_generator
        rows = np.flatnonzero(rng.random(particle_count) < probability).astype(
            np.int64, copy=False
        )
        weights = self._strict_joint_particle_weights()
        self.last_joint_cross_isotope_state_attempted_weight_mass += float(
            np.sum(weights[rows], dtype=np.float64)
        )
        if rows.size == 0:
            return current_target
        isotope_order = self.joint_isotope_order()
        old_states: dict[str, list[IsotopeState]] = {}
        proposed_states: dict[str, list[IsotopeState]] = {}
        current_log_prior = np.zeros(rows.size, dtype=np.float64)
        current_log_proposal = np.zeros(rows.size, dtype=np.float64)
        proposed_log_prior = np.zeros(rows.size, dtype=np.float64)
        proposed_log_proposal = np.zeros(rows.size, dtype=np.float64)
        for isotope in isotope_order:
            filt = self.filters[isotope]
            atlas = filt._structural_rj_surface_atlas
            cardinality_prior = filt._structural_rj_cardinality_prior
            if atlas is None or cardinality_prior is None:
                raise RuntimeError(
                    "Joint isotope-state proposals require exact surface priors."
                )
            evidence = self._joint_history_structural_geometry(
                isotope,
                stations,
            )
            self._active_joint_structural_geometry = evidence
            try:
                filt._structural_rj_position_proposal = (
                    filt._build_continuous_rj_position_proposal(
                        evidence,
                        target_beta=float(target_beta),
                    )
                )
                old = [filt.continuous_particles[int(row)].state.copy() for row in rows]
                old_states[isotope] = old
                old_cardinalities = np.asarray(
                    [state.num_sources for state in old],
                    dtype=np.int64,
                )
                for cardinality in np.unique(old_cardinalities).tolist():
                    selected = np.flatnonzero(old_cardinalities == int(cardinality))
                    states = [old[int(index)] for index in selected]
                    charts = (
                        np.stack(
                            [state.surface_chart_ids for state in states],
                            axis=0,
                        )
                        if int(cardinality)
                        else np.empty((selected.size, 0), dtype=np.int64)
                    )
                    strengths = (
                        np.stack(
                            [state.strengths for state in states],
                            axis=0,
                        )
                        if int(cardinality)
                        else np.empty((selected.size, 0), dtype=np.float64)
                    )
                    prior, proposal = filt._continuous_rj_block_log_densities(
                        charts,
                        strengths,
                    )
                    current_log_prior[selected] += prior
                    if int(cardinality) == 0:
                        current_log_proposal[selected] += proposal
                    else:
                        scalar_strength_proposal = (
                            filt._active_continuous_rj_strength_proposal()
                        )
                        minimum_strength = float(filt._strength_prior.minimum)
                        conditional_locations = minimum_strength + (
                            scalar_strength_proposal.data_locations_by_chart[charts]
                            - minimum_strength
                        ) / float(cardinality)
                        strength_block = ContinuousBlockStrengthProposal(
                            minimum=minimum_strength,
                            maximum=float(filt._strength_prior.maximum),
                            data_locations=conditional_locations,
                            data_sigma=float(scalar_strength_proposal.data_sigma),
                            prior_component_probability=float(
                                scalar_strength_proposal.prior_component_probability
                            ),
                            prior_family=str(filt._strength_prior.family),
                            prior_gamma_shape=float(filt._strength_prior.gamma_shape),
                            prior_gamma_scale=float(filt._strength_prior.gamma_scale),
                        )
                        current_log_proposal[selected] += (
                            float(cardinality_prior.log_prob(int(cardinality)))
                            + math.lgamma(float(cardinality) + 1.0)
                            + np.sum(
                                filt._active_continuous_rj_position_proposal().log_density(
                                    charts
                                ),
                                axis=1,
                            )
                            + strength_block.log_density(strengths)
                        )
                cardinalities = rng.choice(
                    cardinality_prior.probabilities.size,
                    size=rows.size,
                    replace=True,
                    p=cardinality_prior.probabilities,
                ).astype(np.int64, copy=False)
                generated: list[IsotopeState | None] = [None] * rows.size
                for cardinality in np.unique(cardinalities).tolist():
                    selected = np.flatnonzero(cardinalities == int(cardinality))
                    source_count = int(selected.size) * int(cardinality)
                    chart_ids, surface_uv, _ = atlas.sample(
                        source_count,
                        rng=rng,
                        chart_probabilities=(
                            filt._active_continuous_rj_position_proposal().chart_probabilities
                        ),
                    )
                    charts_batch = chart_ids.reshape(
                        selected.size,
                        int(cardinality),
                    )
                    uv_batch = surface_uv.reshape(
                        selected.size,
                        int(cardinality),
                        2,
                    )
                    if int(cardinality) == 0:
                        strength_batch = np.empty(
                            (selected.size, 0),
                            dtype=np.float64,
                        )
                        prior, proposal = filt._continuous_rj_block_log_densities(
                            charts_batch,
                            strength_batch,
                        )
                        proposed_log_prior[selected] += prior
                        proposed_log_proposal[selected] += proposal
                    else:
                        scalar_strength_proposal = (
                            filt._active_continuous_rj_strength_proposal()
                        )
                        minimum_strength = float(filt._strength_prior.minimum)
                        conditional_locations = minimum_strength + (
                            scalar_strength_proposal.data_locations_by_chart[
                                charts_batch
                            ]
                            - minimum_strength
                        ) / float(cardinality)
                        strength_block = ContinuousBlockStrengthProposal(
                            minimum=minimum_strength,
                            maximum=float(filt._strength_prior.maximum),
                            data_locations=conditional_locations,
                            data_sigma=float(scalar_strength_proposal.data_sigma),
                            prior_component_probability=float(
                                scalar_strength_proposal.prior_component_probability
                            ),
                            prior_family=str(filt._strength_prior.family),
                            prior_gamma_shape=float(filt._strength_prior.gamma_shape),
                            prior_gamma_scale=float(filt._strength_prior.gamma_scale),
                        )
                        strength_batch = strength_block.sample(rng=rng)
                        prior, _ = filt._continuous_rj_block_log_densities(
                            charts_batch,
                            strength_batch,
                        )
                        proposed_log_prior[selected] += prior
                        proposed_log_proposal[selected] += (
                            float(cardinality_prior.log_prob(int(cardinality)))
                            + math.lgamma(float(cardinality) + 1.0)
                            + np.sum(
                                filt._active_continuous_rj_position_proposal().log_density(
                                    charts_batch
                                ),
                                axis=1,
                            )
                            + strength_block.log_density(strength_batch)
                        )
                    for local_index, destination in enumerate(selected.tolist()):
                        state = IsotopeState(
                            num_sources=int(cardinality),
                            strengths=strength_batch[local_index],
                            surface_chart_ids=charts_batch[local_index],
                            surface_uv=uv_batch[local_index],
                        )
                        filt._canonicalize_structural_rj_state(state)
                        generated[int(destination)] = state
                if any(state is None for state in generated):
                    raise RuntimeError(
                        "Joint isotope-state proposal generation was incomplete."
                    )
                proposed_states[isotope] = [
                    state for state in generated if isinstance(state, IsotopeState)
                ]
            finally:
                self._active_joint_structural_geometry = None
                filt._structural_rj_position_proposal = None
                filt._structural_rj_strength_proposal = None

        def _assign(candidate: Mapping[str, Sequence[IsotopeState]]) -> None:
            """Assign selected variable-length rows before a batched refresh."""
            for isotope in isotope_order:
                filt = self.filters[isotope]
                states = candidate[isotope]
                for local_index, row in enumerate(rows.tolist()):
                    filt.continuous_particles[int(row)].state = states[
                        local_index
                    ].copy()

        _assign(proposed_states)
        try:
            for isotope in isotope_order:
                self._refresh_joint_structural_transport_cache_isotope(
                    stations,
                    isotope,
                    particle_indices=rows,
                )
            proposed_target = self._joint_active_cache_target(
                stations,
                particle_indices=rows,
                target_beta=float(target_beta),
            )
        finally:
            _assign(old_states)
            for isotope in isotope_order:
                self._refresh_joint_structural_transport_cache_isotope(
                    stations,
                    isotope,
                    particle_indices=rows,
                )
        delta_likelihood = np.full(rows.size, float("-inf"), dtype=np.float64)
        proposed_finite = np.isfinite(proposed_target)
        current_finite = np.isfinite(current_target[rows])
        both_finite = proposed_finite & current_finite
        delta_likelihood[both_finite] = (
            proposed_target[both_finite] - current_target[rows][both_finite]
        )
        delta_likelihood[proposed_finite & ~current_finite] = float("inf")
        delta_prior = proposed_log_prior - current_log_prior
        proposal_ratio = current_log_proposal - proposed_log_proposal
        support = (
            np.isfinite(current_log_prior)
            & np.isfinite(current_log_proposal)
            & np.isfinite(proposed_log_prior)
            & np.isfinite(proposed_log_proposal)
        )
        log_ratio = delta_likelihood + delta_prior + proposal_ratio
        accepted = self._joint_mh_acceptance_mask(
            log_ratio,
            rng=rng,
            support=support,
        )
        accepted_rows = rows[accepted]
        if accepted_rows.size:
            for isotope in isotope_order:
                filt = self.filters[isotope]
                states = proposed_states[isotope]
                for local_index in np.flatnonzero(accepted).tolist():
                    row = int(rows[local_index])
                    filt.continuous_particles[row].state = states[local_index].copy()
                self._refresh_joint_structural_transport_cache_isotope(
                    stations,
                    isotope,
                    particle_indices=accepted_rows,
                )
            current_target = current_target.copy()
            current_target[accepted_rows] = proposed_target[accepted]
            self.last_joint_cross_isotope_state_accepted_weight_mass += float(
                np.sum(weights[accepted_rows], dtype=np.float64)
            )
            self._invalidate_posterior_summary_cache()
        finite_ratio = np.isfinite(log_ratio)
        quantile_levels = np.asarray(
            [0.0, 0.1, 0.5, 0.9, 1.0],
            dtype=np.float64,
        )

        def _quantiles(values: NDArray[np.float64]) -> dict[str, float] | None:
            """Return finite rejection-diagnostic quantiles for one MH term."""
            array = np.asarray(values, dtype=np.float64).reshape(-1)
            finite = np.isfinite(array)
            if not np.any(finite):
                return None
            return {
                label: float(value)
                for label, value in zip(
                    ("min", "p10", "median", "p90", "max"),
                    np.quantile(array[finite], quantile_levels),
                    strict=True,
                )
            }

        self.last_joint_cross_isotope_state_rejection_diagnostics = {
            "attempted": int(rows.size),
            "accepted": int(accepted_rows.size),
            "support_rejected": int(np.count_nonzero(~support)),
            "nonfinite_rejected": int(np.count_nonzero(support & ~finite_ratio)),
            "mh_random_rejected": int(
                np.count_nonzero(support & finite_ratio & ~accepted)
            ),
            "component_quantiles": {
                "delta_log_likelihood": _quantiles(delta_likelihood),
                "delta_log_prior": _quantiles(delta_prior),
                "log_reverse_minus_forward": _quantiles(proposal_ratio),
                "log_jacobian": {
                    label: 0.0 for label in ("min", "p10", "median", "p90", "max")
                },
                "log_acceptance_ratio": _quantiles(log_ratio),
            },
        }
        self._assert_joint_particle_alignment()
        return current_target

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
            isinstance(identity, JointRowIdentity) for identity in parent_identities
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
        persistent_cache = self._joint_persistent_structural_transport_cache
        if persistent_cache is not None:
            if hasattr(persistent_cache[0], "detach"):
                import torch

                index_tensor = torch.as_tensor(
                    indices,
                    device=persistent_cache[0].device,
                    dtype=torch.long,
                )
                reindexed_cache = tuple(
                    torch.index_select(values, 0, index_tensor).contiguous()
                    for values in persistent_cache
                )
            else:
                reindexed_cache = tuple(
                    np.ascontiguousarray(values[indices]) for values in persistent_cache
                )
            self._joint_persistent_structural_transport_cache = reindexed_cache
            self._joint_structural_transport_cache = reindexed_cache
            self.last_joint_persistent_cache_reindex_count += 1
        self._joint_row_generation = new_identities[0].generation
        self.last_joint_resample_indices = np.asarray(
            indices,
            dtype=np.int64,
        )
        self._joint_persistent_structural_state_sha256 = (
            self._joint_structural_state_sha256()
            if self._joint_persistent_structural_transport_cache is not None
            else None
        )
        self._invalidate_posterior_summary_cache()
        self._assert_joint_particle_alignment()
        return self.last_joint_resample_indices

    def _joint_mixing_snapshot(self) -> dict[str, object]:
        """Capture batched joint states for one exact rejuvenation diagnostic."""
        weights = self._strict_joint_particle_weights()
        state_blocks: list[NDArray[np.float64]] = []
        cardinality_columns: list[NDArray[np.int64]] = []
        isotope_states: dict[str, dict[str, NDArray[Any]]] = {}
        transition_mass: dict[str, float] = {}
        for isotope in self.joint_isotope_order():
            filt = self.filters[isotope]
            positions, strengths, mask, chart_ids, surface_uv = (
                filt._packed_continuous_surface_state_arrays()
            )
            cardinalities = np.sum(mask, axis=1, dtype=np.int64)
            cardinality_columns.append(cardinalities)
            state_blocks.extend(
                (
                    chart_ids.astype(np.float64, copy=False),
                    surface_uv.reshape(surface_uv.shape[0], -1),
                    strengths,
                    mask.astype(np.float64, copy=False),
                )
            )
            isotope_states[isotope] = {
                "positions": positions,
                "strengths": strengths,
                "mask": mask,
                "cardinalities": cardinalities,
            }
            for key, value in filt.last_structural_transition_weight_mass.items():
                transition_mass[f"{isotope}.{key}"] = float(value)
        state_matrix = np.ascontiguousarray(
            np.concatenate(
                [
                    np.asarray(block, dtype=np.float64).reshape(
                        weights.size,
                        -1,
                    )
                    for block in state_blocks
                ],
                axis=1,
            ),
            dtype=np.float64,
        )
        cardinality_matrix = np.stack(cardinality_columns, axis=1)
        return {
            "weights": weights,
            "state_matrix": state_matrix,
            "cardinality_matrix": cardinality_matrix,
            "isotope_states": isotope_states,
            "transition_mass": transition_mass,
        }

    @staticmethod
    def _joint_isotope_cache_state(
        filt: IsotopeParticleFilter,
    ) -> tuple[NDArray[Any], ...]:
        """Copy compact row state used to find exact accepted cache deltas."""
        _, strengths, mask, chart_ids, surface_uv = (
            filt._packed_continuous_surface_state_arrays()
        )
        return tuple(
            np.ascontiguousarray(values).copy()
            for values in (strengths, mask, chart_ids, surface_uv)
        )

    @staticmethod
    def _joint_changed_cache_rows(
        before: tuple[NDArray[Any], ...],
        filt: IsotopeParticleFilter,
    ) -> NDArray[np.int64]:
        """Return PF rows whose accepted isotope state changed exactly."""
        after = RotatingShieldPFEstimator._joint_isotope_cache_state(filt)
        if len(before) != len(after) or any(
            first.shape != second.shape
            for first, second in zip(before, after, strict=True)
        ):
            raise RuntimeError("Isotope cache-state shapes changed unexpectedly.")
        changed = np.zeros(after[0].shape[0], dtype=bool)
        for first, second in zip(before, after, strict=True):
            changed |= np.any(
                first.reshape(first.shape[0], -1)
                != second.reshape(second.shape[0], -1),
                axis=1,
            )
        return np.flatnonzero(changed).astype(np.int64, copy=False)

    @staticmethod
    def _weighted_correlation(
        before: NDArray[np.float64],
        after: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> float:
        """Return a finite weighted lag-one correlation diagnostic."""
        first = np.asarray(before, dtype=np.float64).reshape(-1)
        second = np.asarray(after, dtype=np.float64).reshape(-1)
        mass = np.asarray(weights, dtype=np.float64).reshape(-1)
        finite = np.isfinite(first) & np.isfinite(second)
        if not np.any(finite):
            return 0.0
        first = first[finite]
        second = second[finite]
        mass = mass[finite]
        mass_sum = float(np.sum(mass))
        if not np.isfinite(mass_sum) or mass_sum <= 0.0:
            return 0.0
        mass = mass / mass_sum
        first_centered = first - float(np.sum(mass * first))
        second_centered = second - float(np.sum(mass * second))
        first_var = float(np.sum(mass * np.square(first_centered)))
        second_var = float(np.sum(mass * np.square(second_centered)))
        if first_var <= 0.0 or second_var <= 0.0:
            return float(np.array_equal(first, second))
        correlation = float(
            np.sum(mass * first_centered * second_centered)
            / math.sqrt(first_var * second_var)
        )
        return float(np.clip(correlation, -1.0, 1.0))

    @staticmethod
    def _weighted_vector_lag1_correlation(
        before: NDArray[np.float64],
        after: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> float:
        """Return one weighted lag-one correlation for vector-valued states."""
        first = np.asarray(before, dtype=np.float64)
        second = np.asarray(after, dtype=np.float64)
        mass = np.asarray(weights, dtype=np.float64).reshape(-1)
        if first.ndim != 2 or second.shape != first.shape:
            raise ValueError(
                "Vector lag-one correlation requires matching row matrices."
            )
        if first.shape[0] != mass.size:
            raise ValueError(
                "Vector lag-one correlation weights must match matrix rows."
            )
        finite_rows = (
            np.all(np.isfinite(first), axis=1)
            & np.all(np.isfinite(second), axis=1)
            & np.isfinite(mass)
        )
        if not np.any(finite_rows):
            return 0.0
        first = first[finite_rows]
        second = second[finite_rows]
        mass = mass[finite_rows]
        mass_sum = float(np.sum(mass))
        if not np.isfinite(mass_sum) or mass_sum <= 0.0:
            return 0.0
        mass = mass / mass_sum
        first_centered = first - np.sum(
            mass[:, None] * first,
            axis=0,
            keepdims=True,
        )
        second_centered = second - np.sum(
            mass[:, None] * second,
            axis=0,
            keepdims=True,
        )
        first_variance = float(np.sum(mass[:, None] * np.square(first_centered)))
        second_variance = float(np.sum(mass[:, None] * np.square(second_centered)))
        if first_variance <= 0.0 or second_variance <= 0.0:
            return float(np.array_equal(first, second))
        covariance = float(np.sum(mass[:, None] * first_centered * second_centered))
        correlation = covariance / math.sqrt(first_variance * second_variance)
        return float(np.clip(correlation, -1.0, 1.0))

    def _joint_mixing_diagnostics(
        self,
        before: Mapping[str, object],
        after: Mapping[str, object],
        *,
        target_before: NDArray[np.float64],
        target_after: NDArray[np.float64],
    ) -> dict[str, float]:
        """Measure row diversity and movement without using ancestry labels."""
        weights = np.asarray(before["weights"], dtype=np.float64)
        before_matrix = np.asarray(
            before["state_matrix"],
            dtype=np.float64,
        )
        after_matrix = np.asarray(
            after["state_matrix"],
            dtype=np.float64,
        )
        if before_matrix.shape != after_matrix.shape:
            raise RuntimeError(
                "Joint rejuvenation changed the aligned particle layout."
            )
        state_changed = np.any(before_matrix != after_matrix, axis=1)
        before_cardinality = np.asarray(
            before["cardinality_matrix"],
            dtype=np.int64,
        )
        after_cardinality = np.asarray(
            after["cardinality_matrix"],
            dtype=np.int64,
        )
        k_changed = np.any(
            before_cardinality != after_cardinality,
            axis=1,
        )
        _, inverse = np.unique(
            after_cardinality,
            axis=0,
            return_inverse=True,
        )
        k_mass = np.bincount(
            inverse,
            weights=weights,
            minlength=int(np.max(inverse)) + 1,
        )
        positive_k_mass = k_mass[k_mass > 0.0]
        k_entropy = -float(np.sum(positive_k_mass * np.log(positive_k_mass)))
        position_row_esjd = np.zeros(weights.size, dtype=np.float64)
        strength_row_esjd = np.zeros(weights.size, dtype=np.float64)
        before_states = before["isotope_states"]
        after_states = after["isotope_states"]
        if not isinstance(before_states, Mapping) or not isinstance(
            after_states,
            Mapping,
        ):
            raise RuntimeError("Joint mixing snapshots lost isotope states.")
        for isotope in self.joint_isotope_order():
            first = before_states[isotope]
            second = after_states[isotope]
            if not isinstance(first, Mapping) or not isinstance(
                second,
                Mapping,
            ):
                raise RuntimeError("Joint mixing isotope snapshots are malformed.")
            first_mask = np.asarray(first["mask"], dtype=bool)
            second_mask = np.asarray(second["mask"], dtype=bool)
            same_cardinality = np.asarray(
                first["cardinalities"], dtype=np.int64
            ) == np.asarray(second["cardinalities"], dtype=np.int64)
            matched = first_mask & second_mask & same_cardinality[:, None]
            position_delta = np.asarray(
                second["positions"], dtype=np.float64
            ) - np.asarray(first["positions"], dtype=np.float64)
            position_row_esjd += np.sum(
                np.where(
                    matched,
                    np.sum(np.square(position_delta), axis=2),
                    0.0,
                ),
                axis=1,
            )
            first_strength = np.asarray(
                first["strengths"],
                dtype=np.float64,
            )
            second_strength = np.asarray(
                second["strengths"],
                dtype=np.float64,
            )
            log_strength_delta = np.zeros_like(first_strength)
            log_strength_delta[matched] = np.log(second_strength[matched]) - np.log(
                first_strength[matched]
            )
            strength_row_esjd += np.sum(
                np.square(log_strength_delta),
                axis=1,
            )
        transition_before = before["transition_mass"]
        transition_after = after["transition_mass"]
        if not isinstance(transition_before, Mapping) or not isinstance(
            transition_after,
            Mapping,
        ):
            raise RuntimeError("Joint transition-mass snapshots are malformed.")
        transition_delta = {
            str(key): float(value) - float(transition_before.get(key, 0.0))
            for key, value in transition_after.items()
        }
        accepted_structure_mass = float(
            sum(
                value
                for key, value in transition_delta.items()
                if key.endswith(
                    (
                        "birth_accepted_weight_mass",
                        "death_accepted_weight_mass",
                        "split_accepted_weight_mass",
                        "merge_accepted_weight_mass",
                    )
                )
            )
        )
        diagnostics = {
            "distinct_joint_state_count": float(
                np.unique(after_matrix, axis=0).shape[0]
            ),
            "joint_k_vector_count": float(
                np.unique(
                    after_cardinality,
                    axis=0,
                ).shape[0]
            ),
            "joint_k_entropy": k_entropy,
            "state_change_weight_mass": float(np.sum(weights[state_changed])),
            "k_transition_weight_mass": float(np.sum(weights[k_changed])),
            "accepted_structure_weight_mass": accepted_structure_mass,
            "surface_position_esjd_m2": float(np.sum(weights * position_row_esjd)),
            "log_strength_esjd": float(np.sum(weights * strength_row_esjd)),
            "target_log_likelihood_lag1_correlation": (
                self._weighted_correlation(
                    target_before,
                    target_after,
                    weights,
                )
            ),
            "joint_k_vector_lag1_correlation": (
                self._weighted_vector_lag1_correlation(
                    before_cardinality,
                    after_cardinality,
                    weights,
                )
            ),
        }
        ordinary_boundary = np.zeros(weights.size, dtype=np.bool_)
        ordinary_boundary_escape = np.zeros(weights.size, dtype=np.bool_)
        for isotope_index, isotope in enumerate(self.joint_isotope_order()):
            ordinary_maximum = int(self.filters[isotope].config.max_sources or 0)
            at_boundary = before_cardinality[:, isotope_index] >= ordinary_maximum
            escaped = at_boundary & (
                after_cardinality[:, isotope_index]
                < before_cardinality[:, isotope_index]
            )
            ordinary_boundary |= at_boundary
            ordinary_boundary_escape |= escaped
            diagnostics[f"ordinary_boundary_weight_mass.{isotope}"] = float(
                np.sum(weights[at_boundary])
            )
            diagnostics[f"ordinary_boundary_escape_weight_mass.{isotope}"] = float(
                np.sum(weights[escaped])
            )
        diagnostics["ordinary_boundary_weight_mass"] = float(
            np.sum(weights[ordinary_boundary])
        )
        diagnostics["ordinary_boundary_escape_weight_mass"] = float(
            np.sum(weights[ordinary_boundary_escape])
        )
        for isotope_index, isotope in enumerate(self.joint_isotope_order()):
            diagnostics[f"k_lag1_correlation.{isotope}"] = self._weighted_correlation(
                before_cardinality[:, isotope_index],
                after_cardinality[:, isotope_index],
                weights,
            )
        return diagnostics

    def _joint_rejuvenate(
        self,
        stations: Sequence[JointStationObservation],
        *,
        target_beta: float,
        newest_prefix_count: int | None = None,
    ) -> dict[str, float]:
        """Apply conditional exact-RJ sweeps under one prefix bridge target."""
        active = tuple(stations)
        if not active:
            return {}
        state_before = self._joint_mixing_snapshot()
        diagnostics: dict[str, float] = {}
        rejuvenation_start = time.perf_counter()
        cache_hits_start = int(self.last_joint_structural_unit_cache_hits)
        cache_misses_start = int(self.last_joint_structural_unit_cache_misses)
        cross_attempted_start = float(
            self.last_joint_cross_isotope_attempted_weight_mass
        )
        cross_accepted_start = float(self.last_joint_cross_isotope_accepted_weight_mass)
        cross_state_attempted_start = float(
            self.last_joint_cross_isotope_state_attempted_weight_mass
        )
        cross_state_accepted_start = float(
            self.last_joint_cross_isotope_state_accepted_weight_mass
        )
        strength_block_attempted_start = float(
            self.last_joint_strength_block_attempted_weight_mass
        )
        strength_block_accepted_start = float(
            self.last_joint_strength_block_accepted_weight_mass
        )
        print(
            "[joint-smc] rejuvenation-start "
            f"beta={float(target_beta):.12g} "
            f"stations={len(active)} "
            f"particles={int(self.pf_config.num_particles)}",
            flush=True,
        )
        active_station_ids = tuple(id(station) for station in active)
        if active_station_ids != self._joint_torch_context_station_ids:
            self._joint_torch_observation_context_cache.clear()
            self._joint_torch_context_station_ids = active_station_ids
        self._joint_cuda_accepted_unit_transport_cache.clear()
        self._active_joint_station_history = active
        self._active_joint_tempering_prefix_count = newest_prefix_count
        newest_start = sum(int(station.fe_indices.size) for station in active[:-1])
        try:
            self._refresh_joint_structural_transport_cache(active)
            isotope_order = self.joint_isotope_order()
            cache = self._joint_structural_transport_cache
            if cache is None:
                raise RuntimeError(
                    "Joint structural transport cache was not initialized."
                )
            cache_is_torch = hasattr(cache[0], "detach")
            if cache_is_torch:
                current_target_tensor = self._joint_history_log_likelihood_torch(
                    filt=self.filters[isotope_order[0]],
                    stations=active,
                    total_nvsl=cache[0],
                    uncollided_nvsl=cache[1],
                    features_nvslf=cache[2],
                    target_beta=float(target_beta),
                    newest_prefix_count=newest_prefix_count,
                )
                current_target_log_likelihood = (
                    current_target_tensor.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=False)
                )
            else:
                current_target_log_likelihood = (
                    self._joint_history_log_likelihood_numpy(
                        filt=self.filters[isotope_order[0]],
                        stations=active,
                        total_nvsl=cache[0],
                        uncollided_nvsl=cache[1],
                        features_nvslf=cache[2],
                        target_beta=float(target_beta),
                        newest_prefix_count=newest_prefix_count,
                    )
                )
            target_before = np.asarray(
                current_target_log_likelihood,
                dtype=np.float64,
            ).copy()
            last_changed_rows = np.empty(0, dtype=np.int64)
            for isotope_index, isotope in enumerate(isotope_order):
                isotope_start = time.perf_counter()
                print(
                    "[joint-smc] isotope-sweep-start "
                    f"beta={float(target_beta):.12g} "
                    f"isotope={isotope} "
                    f"ordinal={isotope_index + 1}/{len(isotope_order)}",
                    flush=True,
                )
                filt = self.filters[isotope]
                cache_state_before = self._joint_isotope_cache_state(filt)
                evidence = self._joint_history_structural_geometry(
                    isotope,
                    active,
                )
                self._validate_joint_structural_geometry(evidence, active)
                self._active_joint_structural_geometry = evidence
                try:
                    filt._initialize_continuous_rj_device_state(cache[0])
                    filt.apply_structural_moves(
                        evidence,
                        target_beta=float(target_beta),
                        tempering_start_row=int(newest_start),
                        current_target_log_likelihood=(current_target_log_likelihood),
                    )
                finally:
                    self._active_joint_structural_geometry = None
                self._assert_joint_particle_alignment()
                updated_target = filt.last_structural_target_log_likelihood
                if updated_target is None:
                    raise RuntimeError(
                        "Joint structural sweep did not return its target cache."
                    )
                current_target_log_likelihood = np.asarray(
                    updated_target,
                    dtype=np.float64,
                )
                changed_rows = self._joint_changed_cache_rows(
                    cache_state_before,
                    filt,
                )
                last_changed_rows = changed_rows
                timing = filt.last_structural_timing_s
                attempt_count = sum(
                    int(timing.get(key, 0.0))
                    for key in (
                        "rj_birth_attempted",
                        "rj_death_attempted",
                        "rj_global_position_attempted",
                        "rj_local_position_attempted",
                        "rj_strength_attempted",
                        "rj_split_attempted",
                        "rj_merge_attempted",
                    )
                )
                print(
                    "[joint-smc] isotope-sweep-done "
                    f"beta={float(target_beta):.12g} "
                    f"isotope={isotope} "
                    f"elapsed_s={time.perf_counter() - isotope_start:.3f} "
                    f"attempts={attempt_count}",
                    flush=True,
                )
                if isotope_index + 1 < len(isotope_order):
                    self._refresh_joint_structural_transport_cache_isotope(
                        active,
                        isotope,
                        particle_indices=changed_rows,
                    )
            if last_changed_rows.size:
                self._refresh_joint_structural_transport_cache_isotope(
                    active,
                    isotope_order[-1],
                    particle_indices=last_changed_rows,
                )
            if float(self.pf_config.joint_strength_block_probability) > 0.0:
                current_target_log_likelihood = self._apply_joint_strength_block(
                    active,
                    target_beta=float(target_beta),
                    current_target_log_likelihood=(current_target_log_likelihood),
                )
            if float(self.pf_config.joint_cross_isotope_state_block_probability) > 0.0:
                current_target_log_likelihood = (
                    self._apply_joint_cross_isotope_state_block(
                        active,
                        target_beta=float(target_beta),
                        current_target_log_likelihood=(current_target_log_likelihood),
                    )
                )
            if float(self.pf_config.joint_cross_isotope_transfer_probability) > 0.0:
                current_target_log_likelihood = (
                    self._apply_joint_cross_isotope_transfer(
                        active,
                        target_beta=float(target_beta),
                        current_target_log_likelihood=(current_target_log_likelihood),
                    )
                )
            diagnostics = self._joint_mixing_diagnostics(
                state_before,
                self._joint_mixing_snapshot(),
                target_before=target_before,
                target_after=current_target_log_likelihood,
            )
            diagnostics["cross_isotope_attempted_weight_mass"] = float(
                self.last_joint_cross_isotope_attempted_weight_mass
                - cross_attempted_start
            )
            diagnostics["cross_isotope_accepted_weight_mass"] = float(
                self.last_joint_cross_isotope_accepted_weight_mass
                - cross_accepted_start
            )
            diagnostics["cross_isotope_state_attempted_weight_mass"] = float(
                self.last_joint_cross_isotope_state_attempted_weight_mass
                - cross_state_attempted_start
            )
            diagnostics["cross_isotope_state_accepted_weight_mass"] = float(
                self.last_joint_cross_isotope_state_accepted_weight_mass
                - cross_state_accepted_start
            )
            diagnostics["joint_strength_attempted_weight_mass"] = float(
                self.last_joint_strength_block_attempted_weight_mass
                - strength_block_attempted_start
            )
            diagnostics["joint_strength_accepted_weight_mass"] = float(
                self.last_joint_strength_block_accepted_weight_mass
                - strength_block_accepted_start
            )
        finally:
            self._invalidate_posterior_summary_cache()
            for filt in self.filters.values():
                filt._clear_continuous_rj_device_state()
            self._active_joint_structural_geometry = None
            self._joint_structural_transport_cache = None
            self._active_joint_station_history = None
            self._active_joint_tempering_prefix_count = None
            print(
                "[joint-smc] rejuvenation-done "
                f"beta={float(target_beta):.12g} "
                f"elapsed_s={time.perf_counter() - rejuvenation_start:.3f} "
                "unit_transport_cache_hits="
                f"{self.last_joint_structural_unit_cache_hits - cache_hits_start} "
                "unit_transport_cache_misses="
                f"{self.last_joint_structural_unit_cache_misses - cache_misses_start}",
                flush=True,
            )
        return diagnostics

    def _joint_rejuvenate_adaptive(
        self,
        stations: Sequence[JointStationObservation],
        *,
        target_beta: float,
        newest_prefix_count: int,
        station_start_s: float,
    ) -> None:
        """Run exact sweeps until movement is adequate or the soft budget binds."""
        minimum_sweeps = int(self.pf_config.joint_rejuvenation_min_sweeps)
        maximum_sweeps = int(self.pf_config.joint_rejuvenation_max_sweeps)
        soft_budget_s = float(self.pf_config.joint_smc_soft_wall_time_s)
        cumulative_k_transition_mass = 0.0
        cumulative_boundary_escape_mass = 0.0
        self.last_joint_structural_mixing_incomplete = False
        for sweep_index in range(maximum_sweeps):
            diagnostics = self._joint_rejuvenate(
                stations,
                target_beta=target_beta,
                newest_prefix_count=newest_prefix_count,
            )
            if not isinstance(diagnostics, Mapping):
                return
            record = {str(key): float(value) for key, value in diagnostics.items()}
            record.update(
                {
                    "prefix_count": float(newest_prefix_count),
                    "target_beta": float(target_beta),
                    "sweep_index": float(sweep_index + 1),
                    "station_elapsed_s": float(time.perf_counter() - station_start_s),
                }
            )
            self.last_joint_rejuvenation_diagnostics.append(record)
            k_transition_mass = float(diagnostics.get("k_transition_weight_mass", 0.0))
            cumulative_k_transition_mass += k_transition_mass
            boundary_mass = float(diagnostics.get("ordinary_boundary_weight_mass", 0.0))
            cumulative_boundary_escape_mass += float(
                diagnostics.get(
                    "ordinary_boundary_escape_weight_mass",
                    0.0,
                )
            )
            record["cumulative_k_transition_weight_mass"] = float(
                cumulative_k_transition_mass
            )
            record["cumulative_boundary_escape_weight_mass"] = float(
                cumulative_boundary_escape_mass
            )
            if sweep_index + 1 < minimum_sweeps:
                continue
            state_mass = float(diagnostics.get("state_change_weight_mass", 0.0))
            position_esjd = float(diagnostics.get("surface_position_esjd_m2", 0.0))
            strength_esjd = float(diagnostics.get("log_strength_esjd", 0.0))
            continuous_movement_sufficient = state_mass >= float(
                self.pf_config.joint_rejuvenation_min_state_change_weight_mass
            ) and (
                position_esjd
                >= float(self.pf_config.joint_rejuvenation_min_surface_esjd_m2)
                or strength_esjd
                >= float(self.pf_config.joint_rejuvenation_min_log_strength_esjd)
            )
            minimum_k_mass = float(
                self.pf_config.joint_rejuvenation_min_k_transition_weight_mass
            )
            boundary_requires_structure = bool(
                self.pf_config.variable_cardinality
            ) and boundary_mass > float(
                self.pf_config.converge_max_cardinality_boundary_mass
            )
            structural_movement_sufficient = not boundary_requires_structure or (
                cumulative_k_transition_mass >= minimum_k_mass
                and cumulative_boundary_escape_mass >= minimum_k_mass
            )
            record["continuous_movement_sufficient"] = float(
                continuous_movement_sufficient
            )
            record["structural_movement_required"] = float(boundary_requires_structure)
            record["structural_movement_sufficient"] = float(
                structural_movement_sufficient
            )
            if continuous_movement_sufficient and structural_movement_sufficient:
                break
            station_elapsed = time.perf_counter() - station_start_s
            if station_elapsed >= soft_budget_s and not boundary_requires_structure:
                self.last_joint_smc_soft_budget_exceeded = True
                print(
                    "[joint-smc] soft-budget "
                    f"elapsed_s={station_elapsed:.3f} "
                    f"budget_s={soft_budget_s:.3f} "
                    "action=skip-optional-rejuvenation-only",
                    flush=True,
                )
                break
            if sweep_index + 1 == maximum_sweeps:
                self.last_joint_structural_mixing_incomplete = bool(
                    boundary_requires_structure and not structural_movement_sufficient
                )
                if self.last_joint_structural_mixing_incomplete:
                    print(
                        "[joint-smc] structural-mixing-incomplete "
                        f"boundary_mass={boundary_mass:.12g} "
                        "cumulative_k_transition_mass="
                        f"{cumulative_k_transition_mass:.12g} "
                        "cumulative_boundary_escape_mass="
                        f"{cumulative_boundary_escape_mass:.12g} "
                        f"sweeps={maximum_sweeps}",
                        flush=True,
                    )

    def _apply_joint_guided_initialization(
        self,
        station: JointStationObservation,
    ) -> None:
        """Replace the first product-prior draw by an exact full-support IS draw."""
        if (
            self._joint_guided_initialization_applied
            or self._joint_station_history
            or not bool(self.pf_config.joint_guided_initialization)
            or not bool(self.pf_config.variable_cardinality)
        ):
            return
        initial_sha256 = self._joint_initial_product_prior_state_sha256
        current_matrix = np.asarray(
            self._joint_mixing_snapshot()["state_matrix"],
            dtype=np.float64,
        )
        current_sha256 = hashlib.sha256(
            np.ascontiguousarray(current_matrix).tobytes(order="C")
        ).hexdigest()
        if initial_sha256 is None or current_sha256 != initial_sha256:
            self._joint_guided_initialization_applied = True
            print(
                "[joint-smc] guided-initialization skipped "
                "reason=initial-state-was-explicitly-replaced",
                flush=True,
            )
            return
        order = self.joint_isotope_order()
        particle_count = int(self.pf_config.num_particles)
        reference_particles = self.filters[order[0]].continuous_particles
        identities = tuple(
            particle.joint_row_identity for particle in reference_particles
        )
        if len(identities) != particle_count or not all(
            isinstance(identity, JointRowIdentity) for identity in identities
        ):
            raise RuntimeError(
                "Guided initialization requires authenticated joint rows."
            )
        geometry = self._joint_station_structural_geometry(station)
        joint_log_prior_density = np.zeros(
            particle_count,
            dtype=np.float64,
        )
        joint_log_guided_density = np.zeros(
            particle_count,
            dtype=np.float64,
        )
        requested_prior_probability = float(
            self.pf_config.joint_guided_initialization_prior_row_probability
        )
        prior_row_count = min(
            particle_count,
            max(1, int(round(requested_prior_probability * particle_count))),
        )
        realized_prior_probability = prior_row_count / float(particle_count)
        component_rng = named_random_generator(
            self.random_seed,
            "joint_guided_initialization_component",
        )
        prior_rows = np.zeros(particle_count, dtype=bool)
        prior_rows[component_rng.permutation(particle_count)[:prior_row_count]] = True
        particles_by_isotope: dict[str, list[IsotopeParticle]] = {}
        cardinality_priors = tuple(
            self.filters[isotope]._structural_rj_cardinality_prior for isotope in order
        )
        if any(prior is None for prior in cardinality_priors):
            raise RuntimeError(
                "Guided initialization requires every cardinality prior."
            )
        cardinality_rng = named_random_generator(
            self.random_seed,
            "joint_guided_initialization_cardinality_vectors",
        )
        joint_cardinalities = _stratified_joint_cardinality_draws(
            tuple(
                np.asarray(prior.probabilities, dtype=np.float64)
                for prior in cardinality_priors
                if prior is not None
            ),
            particle_count,
            rng=cardinality_rng,
        )
        original_particles = {
            isotope: self.filters[isotope].continuous_particles for isotope in order
        }
        for isotope in order:
            self.filters[isotope].continuous_particles = [
                IsotopeParticle(
                    state=IsotopeState(
                        num_sources=0,
                        strengths=np.zeros(0, dtype=np.float64),
                        surface_chart_ids=np.zeros(0, dtype=np.int64),
                        surface_uv=np.zeros((0, 2), dtype=np.float64),
                    ),
                    log_weight=0.0,
                    joint_row_identity=identities[row],
                )
                for row in range(particle_count)
            ]
        self._active_joint_station_history = (station,)
        try:
            for isotope_index, isotope in enumerate(order):
                filt = self.filters[isotope]
                atlas = filt._structural_rj_surface_atlas
                cardinality_prior = filt._structural_rj_cardinality_prior
                if atlas is None or cardinality_prior is None:
                    raise RuntimeError(
                        "Guided initialization requires continuous surface "
                        "and cardinality priors."
                    )
                rng = named_random_generator(
                    self.random_seed,
                    "joint_guided_initialization",
                    isotope,
                )
                if filt.config.variable_cardinality:
                    cardinalities = joint_cardinalities[:, isotope_index].copy()
                else:
                    cardinalities = np.full(
                        particle_count,
                        int(filt.config.init_num_sources[0]),
                        dtype=np.int64,
                    )
                position_proposal = filt._build_continuous_rj_position_proposal(
                    geometry,
                    target_beta=1.0,
                )
                strength_proposal = filt._active_continuous_rj_strength_proposal()
                offsets = np.concatenate(
                    (
                        np.zeros(1, dtype=np.int64),
                        np.cumsum(cardinalities, dtype=np.int64),
                    )
                )
                total_sources = int(offsets[-1])
                source_rows = np.repeat(
                    np.arange(particle_count, dtype=np.int64),
                    cardinalities,
                )
                source_uses_prior = prior_rows[source_rows]
                chart_ids = np.empty(total_sources, dtype=np.int64)
                surface_uv = np.empty((total_sources, 2), dtype=np.float64)
                strengths = np.empty(total_sources, dtype=np.float64)
                for use_prior in (True, False):
                    source_indices = np.flatnonzero(source_uses_prior == use_prior)
                    if source_indices.size == 0:
                        continue
                    sampled_chart_ids, sampled_uv, _ = atlas.sample(
                        int(source_indices.size),
                        rng=rng,
                        chart_probabilities=(
                            None if use_prior else position_proposal.chart_probabilities
                        ),
                    )
                    chart_ids[source_indices] = sampled_chart_ids
                    surface_uv[source_indices] = sampled_uv
                log_cardinality_density = np.log(
                    cardinality_prior.probabilities[cardinalities]
                )
                isotope_log_prior_density = log_cardinality_density.copy()
                isotope_log_guided_density = log_cardinality_density.copy()
                if total_sources:
                    source_log_prior = atlas.log_chart_probabilities[chart_ids]
                    source_log_guided = position_proposal.log_density(chart_ids)
                    isotope_log_prior_density += np.bincount(
                        source_rows,
                        weights=source_log_prior,
                        minlength=particle_count,
                    )
                    isotope_log_guided_density += np.bincount(
                        source_rows,
                        weights=source_log_guided,
                        minlength=particle_count,
                    )
                    minimum_strength = float(filt._strength_prior.minimum)
                    for cardinality in np.unique(
                        cardinalities[cardinalities > 0]
                    ).tolist():
                        state_rows = np.flatnonzero(cardinalities == int(cardinality))
                        source_indices = (
                            offsets[state_rows, None]
                            + np.arange(
                                int(cardinality),
                                dtype=np.int64,
                            )[None, :]
                        )
                        state_charts = chart_ids[source_indices]
                        single_source_locations = np.asarray(
                            strength_proposal.data_locations_by_chart[state_charts],
                            dtype=np.float64,
                        )
                        conditional_locations = minimum_strength + (
                            single_source_locations - minimum_strength
                        ) / float(cardinality)
                        block_proposal = ContinuousBlockStrengthProposal(
                            minimum=minimum_strength,
                            maximum=float(filt._strength_prior.maximum),
                            data_locations=conditional_locations,
                            data_sigma=float(strength_proposal.data_sigma),
                            prior_component_probability=float(
                                strength_proposal.prior_component_probability
                            ),
                            prior_family=str(filt._strength_prior.family),
                            prior_gamma_shape=float(filt._strength_prior.gamma_shape),
                            prior_gamma_scale=float(filt._strength_prior.gamma_scale),
                        )
                        sampled = block_proposal.sample(rng=rng)
                        state_prior_rows = prior_rows[state_rows]
                        if np.any(state_prior_rows):
                            sampled[state_prior_rows] = np.asarray(
                                filt._strength_prior.sample(
                                    (
                                        int(np.count_nonzero(state_prior_rows)),
                                        int(cardinality),
                                    ),
                                    rng=rng,
                                ),
                                dtype=np.float64,
                            )
                        strengths[source_indices] = sampled
                        isotope_log_prior_density[state_rows] += np.sum(
                            np.asarray(
                                filt._strength_prior.log_prob(sampled),
                                dtype=np.float64,
                            ),
                            axis=1,
                        )
                        isotope_log_guided_density[state_rows] += (
                            block_proposal.log_density(sampled)
                        )
                joint_log_prior_density += isotope_log_prior_density
                joint_log_guided_density += isotope_log_guided_density
                particles: list[IsotopeParticle] = []
                for row, cardinality in enumerate(cardinalities.tolist()):
                    begin = int(offsets[row])
                    end = int(offsets[row + 1])
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
                            log_weight=0.0,
                            joint_row_identity=identities[row],
                        )
                    )
                particles_by_isotope[isotope] = particles
                filt.continuous_particles = particles
                filt._structural_rj_position_proposal = None
                filt._structural_rj_strength_proposal = None
                if isotope_index + 1 < len(order) and realized_prior_probability < 1.0:
                    explained = self._joint_station_expected_means_torch(station)
                    guided_indices = np.flatnonzero(~prior_rows)
                    if guided_indices.size == 0:
                        raise RuntimeError(
                            "Sequential guided initialization has no guided "
                            "rows from which to form a residual."
                        )
                    import torch

                    selected = torch.as_tensor(
                        guided_indices,
                        dtype=torch.long,
                        device=explained.device,
                    )
                    reference = torch.mean(
                        torch.index_select(explained, 0, selected),
                        dim=0,
                    )
                    self._joint_birth_proposal_reference_mean_vb = (
                        reference.detach().cpu().numpy().astype(np.float64, copy=False)
                    )
                    if (
                        self._joint_birth_proposal_reference_mean_vb.shape
                        != station.spectrum_vb.shape
                        or np.any(
                            ~np.isfinite(self._joint_birth_proposal_reference_mean_vb)
                        )
                        or np.any(self._joint_birth_proposal_reference_mean_vb < 0.0)
                    ):
                        raise RuntimeError(
                            "Sequential guided reference mean is invalid."
                        )
                    self._joint_birth_proposal_station_score_cache.clear()
                    self._joint_birth_proposal_station_score_cache_order.clear()
        finally:
            self._active_joint_station_history = None
            self._joint_birth_proposal_reference_mean_vb = None
            for filt in self.filters.values():
                filt._structural_rj_position_proposal = None
                filt._structural_rj_strength_proposal = None
            for isotope, particles in original_particles.items():
                self.filters[isotope].continuous_particles = particles
        if realized_prior_probability >= 1.0:
            joint_log_proposal_density = joint_log_prior_density.copy()
        else:
            joint_log_proposal_density = np.logaddexp(
                math.log(realized_prior_probability) + joint_log_prior_density,
                math.log1p(-realized_prior_probability) + joint_log_guided_density,
            )
        log_importance_ratio = joint_log_prior_density - joint_log_proposal_density
        if np.any(~np.isfinite(log_importance_ratio)):
            raise RuntimeError(
                "Guided initialization importance correction is invalid."
            )
        normalized_log_weights = log_importance_ratio - logsumexp(log_importance_ratio)
        for isotope in order:
            particles = particles_by_isotope[isotope]
            for row, particle in enumerate(particles):
                particle.log_weight = float(normalized_log_weights[row])
            self.filters[isotope].continuous_particles = particles
        normalized_weights = np.exp(normalized_log_weights)
        self.last_joint_guided_initialization_ess = float(
            1.0 / np.sum(np.square(normalized_weights))
        )
        self._joint_guided_initialization_applied = True
        self._invalidate_posterior_summary_cache()
        self._assert_joint_particle_alignment()

    def _joint_tempered_station_update(
        self,
        station: JointStationObservation,
    ) -> None:
        """Assimilate one station with common weights and aligned SMC ancestors."""
        import torch

        all_stations = tuple((*self._joint_station_history, station))
        for filt in self.filters.values():
            filt.reset_step_stats()
        self.last_joint_rejuvenation_diagnostics = []
        self.last_joint_smc_soft_budget_exceeded = False
        self.last_joint_structural_mixing_incomplete = False
        station_start = time.perf_counter()
        self._apply_joint_guided_initialization(station)
        self.last_joint_resample_indices = np.empty(0, dtype=np.int64)
        reference_filter = self.filters[self.joint_isotope_order()[0]]
        common_log_weights = self._assert_joint_particle_alignment()
        prefix_log_likelihood = self._joint_station_prefix_log_likelihood_torch(station)
        device = prefix_log_likelihood.device
        log_weights = torch.as_tensor(
            common_log_weights,
            dtype=torch.float64,
            device=device,
        )
        log_weights = reference_filter._normalized_log_weights_torch(log_weights)
        self._assign_joint_log_weights(log_weights.detach().cpu().numpy())
        initial_ess = reference_filter._ess_from_logw_torch(log_weights)
        target_ess = float(self.pf_config.target_ess_ratio) * int(log_weights.numel())
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
        cumulative_lineage_ids = (
            np.asarray(
                self._joint_cumulative_lineage_ids,
                dtype=np.int64,
            )
            .reshape(-1)
            .copy()
        )
        if cumulative_lineage_ids.shape != (particle_count,):
            raise RuntimeError(
                "Cumulative PF lineage does not match aligned particle rows."
            )
        resamples = 0
        steps: list[dict[str, float]] = []
        view_count = int(station.fe_indices.size)

        def _prefix_likelihood() -> "torch.Tensor":
            """Return exact prefix targets for the current aligned states."""
            return self._joint_station_prefix_log_likelihood_torch(station).to(
                device=device,
                dtype=torch.float64,
            )

        def _prefix_increment(
            values: "torch.Tensor",
            prefix_count: int,
        ) -> "torch.Tensor":
            """Return a numerically defined exact adjacent-prefix increment."""
            previous = values[prefix_count - 1]
            current = values[prefix_count]
            previous_finite = torch.isfinite(previous)
            current_finite = torch.isfinite(current)
            if bool(torch.any(current_finite & ~previous_finite).item()):
                raise RuntimeError(
                    "Adding a view restored target mass after an impossible "
                    "prefix; the full-spectrum prefix contract is invalid."
                )
            increment = torch.full_like(current, float("-inf"))
            both_finite = previous_finite & current_finite
            increment[both_finite] = current[both_finite] - previous[both_finite]
            increment[~previous_finite & ~current_finite] = 0.0
            if bool(torch.any(torch.isnan(increment)).item()) or bool(
                torch.any(torch.isinf(increment) & (increment > 0.0)).item()
            ):
                raise RuntimeError(
                    "Adjacent full-spectrum prefix increment is invalid."
                )
            return increment

        prefix_log_likelihood = prefix_log_likelihood.to(
            device=device,
            dtype=torch.float64,
        )
        print(
            "[joint-smc] station-update-start "
            f"station={len(all_stations) - 1} "
            f"particles={particle_count} "
            f"view_prefixes={view_count} "
            f"initial_ess={initial_ess:.6f} "
            f"target_ess={target_ess:.6f}",
            flush=True,
        )
        if initial_ess <= target_ess + 1.0e-9:
            indices = self._resample_joint_particles(log_weights.detach().cpu().numpy())
            station_ancestor_ids = station_ancestor_ids[indices]
            cumulative_lineage_ids = cumulative_lineage_ids[indices]
            resamples += 1
            self._joint_rejuvenate_adaptive(
                all_stations,
                target_beta=0.0,
                newest_prefix_count=1,
                station_start_s=station_start,
            )
            prefix_log_likelihood = _prefix_likelihood()
            log_weights = torch.full(
                (particle_count,),
                -math.log(max(particle_count, 1)),
                dtype=torch.float64,
                device=device,
            )
        max_steps = int(self.pf_config.max_temper_steps)
        for prefix_count in range(1, view_count + 1):
            prefix_step_start = len(steps)
            beta_total = 0.0
            likelihood = _prefix_increment(
                prefix_log_likelihood,
                prefix_count,
            )
            while beta_total < 1.0 - 1.0e-12:
                prefix_step_count = len(steps) - prefix_step_start
                if prefix_step_count >= max_steps:
                    raise RuntimeError(
                        "Joint view-prefix SMC reached max_temper_steps "
                        f"within prefix {prefix_count}/{view_count} "
                        "before that prefix reached beta=1."
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
                    current_ess = reference_filter._ess_from_logw_torch(log_weights)
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
                    print(
                        "[joint-smc] temper-recovery "
                        f"step={len(steps) + 1} "
                        f"prefix={prefix_count}/{view_count} "
                        f"beta={beta_total:.12g} "
                        f"ess={current_ess:.6f} "
                        f"resampled={int(resampled)}",
                        flush=True,
                    )
                    self._joint_rejuvenate_adaptive(
                        all_stations,
                        target_beta=float(beta_total),
                        newest_prefix_count=prefix_count,
                        station_start_s=station_start,
                    )
                    prefix_log_likelihood = _prefix_likelihood()
                    likelihood = _prefix_increment(
                        prefix_log_likelihood,
                        prefix_count,
                    )
                    recovery_step = {
                        "prefix_count": float(prefix_count),
                        "prefix_view_count": float(view_count),
                        "station_beta": float(
                            (prefix_count - 1 + beta_total) / view_count
                        ),
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
                self._assign_joint_log_weights(log_weights.detach().cpu().numpy())
                beta_total += float(delta_beta)
                step = {
                    "prefix_count": float(prefix_count),
                    "prefix_view_count": float(view_count),
                    "station_beta": float((prefix_count - 1 + beta_total) / view_count),
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
                print(
                    "[joint-smc] temper-step "
                    f"step={len(steps)} "
                    f"prefix={prefix_count}/{view_count} "
                    f"beta={beta_total:.12g} "
                    f"delta_beta={float(delta_beta):.12g} "
                    f"ess={float(ess):.6f}",
                    flush=True,
                )
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
                self._joint_rejuvenate_adaptive(
                    all_stations,
                    target_beta=float(beta_total),
                    newest_prefix_count=prefix_count,
                    station_start_s=station_start,
                )
                prefix_log_likelihood = _prefix_likelihood()
                likelihood = _prefix_increment(
                    prefix_log_likelihood,
                    prefix_count,
                )
                log_weights = torch.full(
                    (particle_count,),
                    -math.log(max(particle_count, 1)),
                    dtype=torch.float64,
                    device=device,
                )
            self._joint_rejuvenate_adaptive(
                all_stations,
                target_beta=1.0,
                newest_prefix_count=prefix_count,
                station_start_s=station_start,
            )
            if prefix_count < view_count:
                prefix_log_likelihood = _prefix_likelihood()
        normalized = self._strict_joint_particle_weights()
        final_ess = 1.0 / float(np.sum(normalized**2))
        if final_ess + 1.0e-9 < target_ess:
            raise RuntimeError("Completed joint tempering did not preserve target ESS.")
        station_unique_ancestors = int(np.unique(station_ancestor_ids).size)
        cumulative_unique_ancestors = int(np.unique(cumulative_lineage_ids).size)
        self._joint_cumulative_lineage_ids = cumulative_lineage_ids
        self.last_joint_temper_steps = steps
        self.last_joint_station_unique_ancestor_count = station_unique_ancestors
        self.last_joint_cumulative_unique_ancestor_count = cumulative_unique_ancestors
        # Backward-compatible field now has the conservative cumulative
        # meaning; station-local ancestry is reported separately.
        self.last_joint_unique_ancestor_count = cumulative_unique_ancestors
        for filt in self.filters.values():
            filt.last_temper_steps = [dict(step) for step in steps]
            filt.last_temper_resample_count = int(resamples)
            filt.last_temper_min_ess = float(
                min((step["ess"] for step in steps), default=final_ess)
            )
            filt.last_station_unique_ancestor_count = station_unique_ancestors
            filt.last_cumulative_unique_ancestor_count = cumulative_unique_ancestors
            filt.last_unique_ancestor_count = cumulative_unique_ancestors
            filt.last_ess_pre = float(initial_ess)
            filt.last_ess = float(final_ess)
            filt.last_ess_post = float(final_ess)
            filt.last_resample_ess = bool(resamples)
        self._promote_joint_birth_proposal_station(station)
        self._joint_station_history.append(station)
        self._assert_joint_particle_alignment()
        print(
            "[joint-smc] station-update-done "
            f"station={len(all_stations) - 1} "
            f"elapsed_s={time.perf_counter() - station_start:.3f} "
            f"temper_steps={len(steps)} "
            f"resamples={resamples} "
            f"final_ess={final_ess:.6f}",
            flush=True,
        )

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
        max_sources = self.pf_config.cardinality_capacity
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
            ) = filt._packed_continuous_surface_state_arrays()
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
                    self.filters[isotope].continuous_particles[int(index)].state.copy()
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
        if reset_filters and (self._joint_station_history or self.measurements):
            raise RuntimeError(
                "Pure PF cannot reset particles after observations have "
                "entered the posterior."
            )
        self.poses.append(np.asarray(pose, dtype=float))
        # Rebuild lazily on the next access.
        self.kernel_cache = None
        if reset_filters:
            self._invalidate_posterior_summary_cache()
            self.filters = {}
            self._joint_particles_initialized = False
            self._joint_row_identity_root_sha256 = None
            self._joint_row_generation = None
            self._joint_cumulative_lineage_ids = None
            self._joint_birth_proposal_prefix_scores = {}
            self._joint_birth_proposal_prefix_station_count = 0
            self._joint_birth_proposal_reference_mean_vb = None

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
            raise IndexError("pose_idx lies outside the registered measurement poses.")
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
            generative_contract_hash_sha256=(generative_contract_hash_sha256),
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
        report_start = time.perf_counter()
        self._record_history_estimate(len(self.measurements))
        report_wall = time.perf_counter() - report_start
        self.last_pair_sequence_stage_wall_s = {
            "normalize_and_validate": float(update_start - sequence_start),
            "joint_smc_and_conditional_rj": float(update_wall),
            "posterior_report": float(report_wall),
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
                [record.detector_position_xyz_m for record in selected_records],
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
        pool_all = np.asarray(self.surface_diagnostic_points, dtype=float).reshape(
            -1, 3
        )
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

    def source_response_signatures(
        self,
        isotope: str,
        positions_xyz_m: NDArray[np.float64],
        *,
        window: int | None = None,
    ) -> NDArray[np.float64]:
        """Return normalized measured-view response columns for source points.

        The signature uses the same physical transport kernel and completed
        measurement geometry as inference.  It is intended for reporting
        whether two nearby components are physically distinguishable; it does
        not alter particles, weights, cardinality, or the PF target.
        """
        positions = np.asarray(positions_xyz_m, dtype=np.float64)
        if positions.size == 0:
            return np.zeros((0, 0), dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions_xyz_m must have shape (source, 3).")
        if np.any(~np.isfinite(positions)):
            raise ValueError("Source-response positions must be finite.")
        filt = self.filters.get(str(isotope))
        if filt is None:
            raise KeyError(f"Unknown PF isotope: {isotope}")
        data = self._structural_geometry_for_records(str(isotope), window)
        if data is None or data.row_count == 0:
            return np.zeros((0, positions.shape[0]), dtype=np.float64)
        counts = np.asarray(
            self._cached_expected_counts_per_source(
                filt=filt,
                isotope=str(isotope),
                data=data,
                sources=positions,
                strengths=np.ones(positions.shape[0], dtype=np.float64),
            ),
            dtype=np.float64,
        )
        if counts.shape != (data.row_count, positions.shape[0]):
            raise RuntimeError("Source-response signature shape is invalid.")
        norms = np.linalg.norm(counts, axis=0)
        return np.divide(
            counts,
            norms[None, :],
            out=np.zeros_like(counts),
            where=norms[None, :] > 0.0,
        )

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
        atlases = [filt._structural_rj_surface_atlas for filt in self.filters.values()]
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
            maximum_hausdorff_bound_m=float(maximum_hausdorff_bound_m),
        )

    def posterior_point_estimate(self) -> Dict[str, PFPointEstimate]:
        """Return deterministic posterior summaries for every isotope.

        The pure runtime subclass overrides this method to select one aligned
        joint-cardinality stratum and one common representative particle row.
        Keeping all report consumers on this virtual method prevents stopping
        and visualization from silently reverting to independent isotope
        marginals.
        """
        cached = self._cached_posterior_point_estimate()
        if cached is not None:
            return cached
        estimates: Dict[str, PFPointEstimate] = {}
        for isotope, filt in self.filters.items():
            filt.validate_continuous_surface_states()
            atlas = getattr(filt, "_structural_rj_surface_atlas", None)
            estimates[isotope] = posterior_point_estimate_from_states(
                [particle.state for particle in filt.continuous_particles],
                np.asarray(filt.continuous_weights, dtype=float),
                max_cardinality=self.pf_config.cardinality_capacity,
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
        return self._store_posterior_point_estimate(estimates)

    def estimates(
        self,
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Return the canonical MAP-cardinality PF posterior projection."""
        return self._project_posterior_point_estimates(self.posterior_point_estimate())

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
                        [mode.position_medoid_xyz for mode in point_estimate.modes],
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
                ) = filt._packed_continuous_surface_state_arrays()
                weights = np.asarray(filt.continuous_weights, dtype=float)
                atlas = getattr(filt, "_structural_rj_surface_atlas", None)
                if atlas is None:
                    raise RuntimeError(
                        "Posterior uncertainty requires the continuous surface atlas."
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
                    if not np.isfinite(reporting_mass) or reporting_mass <= 0.0:
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
                    else [mode.position_medoid_xyz for mode in point_estimate.modes],
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
                        [int(mode.surface_chart_id) for mode in point_estimate.modes],
                        dtype=np.int64,
                    )
                    reported_surface_uv = np.asarray(
                        [mode.surface_uv for mode in point_estimate.modes],
                        dtype=np.float64,
                    )
                else:
                    reported_chart_ids, reported_surface_uv = atlas.locate_positions(
                        positions
                    )

            diagnostics = posterior_mode_uncertainty_batched(
                packed_positions,
                packed_mask,
                weights,
                positions,
                packed_surface_kinds=packed_surface_kinds,
                packed_surface_chart_ids=(None if atlas is None else packed_chart_ids),
                packed_surface_uv=(None if atlas is None else packed_surface_uv),
                reported_surface_chart_ids=(
                    None if atlas is None else reported_chart_ids
                ),
                reported_surface_uv=(None if atlas is None else reported_surface_uv),
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
                    else float(point_estimate_map[isotope].selected_stratum_mass)
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
                    "joint_rejuvenation_diagnostics": [],
                    "joint_smc_soft_budget_exceeded": False,
                    "joint_guided_initialization_ess": None,
                    "joint_cross_isotope_rejection_diagnostics": {},
                    "joint_cross_isotope_state_rejection_diagnostics": {},
                    "joint_transport_cache": {},
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
            maximum_cardinality = self.pf_config.cardinality_capacity
            r_int = np.fromiter(
                (
                    validated_state_cardinality(
                        particle.state,
                        name=f"{iso} particle[{index}]",
                        max_cardinality=maximum_cardinality,
                    )
                    for index, particle in enumerate(filt.continuous_particles)
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
                            "positions": filt.continuous_state_positions(state)[
                                :source_count
                            ],
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
                "structural_rejection_diagnostics": dict(
                    getattr(
                        filt,
                        "last_structural_rejection_diagnostics",
                        {},
                    )
                ),
                "temper_steps": list(getattr(filt, "last_temper_steps", [])),
                "joint_rejuvenation_diagnostics": [
                    dict(entry) for entry in self.last_joint_rejuvenation_diagnostics
                ],
                "joint_smc_soft_budget_exceeded": bool(
                    self.last_joint_smc_soft_budget_exceeded
                ),
                "joint_structural_mixing_incomplete": bool(
                    self.last_joint_structural_mixing_incomplete
                ),
                "joint_guided_initialization_ess": (
                    self.last_joint_guided_initialization_ess
                ),
                "joint_cross_isotope_rejection_diagnostics": dict(
                    self.last_joint_cross_isotope_rejection_diagnostics
                ),
                "joint_cross_isotope_state_rejection_diagnostics": dict(
                    self.last_joint_cross_isotope_state_rejection_diagnostics
                ),
                "joint_transport_cache": {
                    "unit_hits": int(self.last_joint_structural_unit_cache_hits),
                    "unit_misses": int(self.last_joint_structural_unit_cache_misses),
                    "accepted_state_reuses": int(
                        self.last_joint_persistent_cache_reuse_count
                    ),
                    "history_appends": int(
                        self.last_joint_persistent_cache_append_count
                    ),
                    "ancestor_reindexes": int(
                        self.last_joint_persistent_cache_reindex_count
                    ),
                    "resident_device": (
                        "cuda"
                        if self._joint_persistent_structural_transport_cache is not None
                        and hasattr(
                            self._joint_persistent_structural_transport_cache[0],
                            "detach",
                        )
                        else "cpu"
                        if self._joint_persistent_structural_transport_cache is not None
                        else None
                    ),
                },
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
        if weights.size == 0 or np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
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
                "conditional_mark_upper_tail_probability": None,
                "confidence": float(self.pf_config.converge_innovation_confidence),
            }
        station = self._joint_station_history[-1]
        weights = self._strict_joint_particle_weights()
        components = tuple(
            value.detach().cpu().numpy().astype(np.float64, copy=False)
            for value in self._joint_station_transport_components_torch(station)
        )
        raw_result = dict(
            self._full_spectrum_model().posterior_predictive_innovation_numpy(
                station.spectrum_vb,
                components[0],
                components[1],
                components[2],
                station.live_times_s,
                weights,
                confidence=float(self.pf_config.converge_innovation_confidence),
            )
        )
        required_raw = {
            "renewal_total_max_abs_z",
            "renewal_total_within_confidence",
            "conditional_mark_pearson",
            "conditional_mark_degrees_of_freedom",
            "conditional_mark_tail_probability",
            "conditional_mark_upper_tail_probability",
            "confidence",
        }
        if set(raw_result) != required_raw:
            raise RuntimeError(
                "Full-spectrum innovation returned an incompatible diagnostic schema."
            )
        confidence = float(raw_result["confidence"])
        total_z = float(raw_result["renewal_total_max_abs_z"])
        mark_pearson = float(raw_result["conditional_mark_pearson"])
        mark_degrees = raw_result["conditional_mark_degrees_of_freedom"]
        mark_tail_raw = raw_result["conditional_mark_tail_probability"]
        mark_upper_tail_raw = raw_result["conditional_mark_upper_tail_probability"]
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
            raise RuntimeError("Full-spectrum innovation contains invalid diagnostics.")
        mark_tail: float | None
        if mark_tail_raw is None:
            mark_tail = None
        else:
            mark_tail = float(mark_tail_raw)
            if not np.isfinite(mark_tail) or mark_tail < 0.0 or mark_tail > 1.0:
                raise RuntimeError(
                    "Full-spectrum conditional-mark tail probability is invalid."
                )
        mark_upper_tail: float | None
        if mark_upper_tail_raw is None:
            mark_upper_tail = None
        else:
            mark_upper_tail = float(mark_upper_tail_raw)
            if (
                not np.isfinite(mark_upper_tail)
                or mark_upper_tail < 0.0
                or mark_upper_tail > 1.0
            ):
                raise RuntimeError(
                    "Full-spectrum conditional-mark upper-tail probability is invalid."
                )
        mark_passed = bool(
            mark_tail is not None and mark_tail + 1.0e-15 >= 1.0 - confidence
        )
        total_passed = bool(raw_result["renewal_total_within_confidence"])
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
            "conditional_mark_upper_tail_probability": mark_upper_tail,
            "confidence": confidence,
        }

    def posterior_predictive_check(
        self,
        *,
        sample_count: int = 128,
        confidence: float = 0.95,
        worst_bin_count: int = 32,
    ) -> Dict[str, object]:
        """Return a model-native posterior predictive residual audit.

        Every predictive spectrum is sampled through the immutable generative
        model, so detector response, background, dead time, and configured
        discrepancy covariance are preserved. Stations are evaluated one at a
        time because station-shared latent variables must not be coupled across
        acquisition boundaries; particles, views, source slots, lines, and
        energy bins remain batched inside each station call.
        """
        count = _strict_nonnegative_integer(
            sample_count,
            name="posterior predictive sample_count",
        )
        if count < 2:
            raise ValueError(
                "posterior predictive sample_count must be at least two."
            )
        probability = _strict_config_number(
            confidence,
            name="posterior predictive confidence",
        )
        if not 0.0 < probability < 1.0:
            raise ValueError("posterior predictive confidence must lie in (0, 1).")
        maximum_worst_bins = _strict_nonnegative_integer(
            worst_bin_count,
            name="posterior predictive worst_bin_count",
        )
        if not self._joint_station_history:
            return {
                "available": False,
                "sample_count": count,
                "confidence": probability,
                "stations": [],
                "shield_pair_summary": {},
                "obstacle_line_of_sight_summary": {},
                "worst_standardized_bin_residuals": [],
            }
        weights = self._strict_joint_particle_weights()
        rng = named_random_generator(
            self.random_seed,
            "posterior_predictive_residual_audit",
        )
        particle_indices = _stratified_categorical_draws(
            weights,
            count,
            rng=rng,
        )
        model = self._full_spectrum_model()
        feature_names = tuple(model.transport_feature_order)
        obstacle_feature_index = (
            feature_names.index("tau_obstacle")
            if "tau_obstacle" in feature_names
            else None
        )
        alpha = 0.5 * (1.0 - probability)
        station_results: list[dict[str, object]] = []
        pair_values: dict[int, list[NDArray[np.float64]]] = {}
        pair_coverages: dict[int, list[NDArray[np.bool_]]] = {}
        obstacle_values: dict[str, list[NDArray[np.float64]]] = {
            "crosses_obstacle": [],
            "clear_line_of_sight": [],
        }
        isotope_ablation_values: dict[str, list[float]] = {
            isotope: [] for isotope in self.joint_isotope_order()
        }
        worst_rows: list[dict[str, object]] = []
        for station in self._joint_station_history:
            components = tuple(
                value.detach().cpu().numpy().astype(
                    np.float64,
                    copy=False,
                )
                for value in self._joint_station_transport_components_torch(
                    station
                )
            )
            selected_components = tuple(
                value[particle_indices] for value in components
            )
            sampled = np.asarray(
                model.sample_predictive_numpy(
                    selected_components[0],
                    selected_components[1],
                    selected_components[2],
                    station.live_times_s,
                    sample_count=1,
                    rng=rng,
                )
            )
            expected_shape = (
                count,
                1,
                int(station.fe_indices.size),
                int(station.energy_axis_keV.size),
            )
            if (
                sampled.shape != expected_shape
                or not np.issubdtype(sampled.dtype, np.integer)
                or np.any(sampled < 0)
            ):
                raise RuntimeError(
                    "Posterior predictive sampler returned an invalid batch."
                )
            draws = sampled[:, 0].astype(np.float64, copy=False)
            observed = np.asarray(station.spectrum_vb, dtype=np.float64)
            predictive_mean = np.mean(draws, axis=0)
            predictive_std = np.std(draws, axis=0, ddof=1)
            standardized = (observed - predictive_mean) / np.maximum(
                predictive_std,
                1.0,
            )
            lower = np.quantile(draws, alpha, axis=0)
            upper = np.quantile(draws, 1.0 - alpha, axis=0)
            covered = (observed >= lower) & (observed <= upper)
            observed_totals = np.sum(observed, axis=1, dtype=np.float64)
            predictive_totals = np.sum(draws, axis=2, dtype=np.float64)
            predictive_total_mean = np.mean(predictive_totals, axis=0)
            predictive_total_std = np.std(
                predictive_totals,
                axis=0,
                ddof=1,
            )
            total_standardized = (
                observed_totals - predictive_total_mean
            ) / np.maximum(predictive_total_std, 1.0)
            pair_ids = (
                np.asarray(station.fe_indices, dtype=np.int64)
                * int(self.num_orientations)
                + np.asarray(station.pb_indices, dtype=np.int64)
            )
            obstacle_probability = np.zeros(
                int(station.fe_indices.size),
                dtype=np.float64,
            )
            if obstacle_feature_index is not None:
                tau_obstacle = selected_components[2][
                    ..., obstacle_feature_index
                ]
                contributes = selected_components[0] > 0.0
                crosses = np.any(
                    contributes & (tau_obstacle > 1.0e-12),
                    axis=(2, 3),
                )
                obstacle_probability = np.mean(crosses, axis=0)
            view_rows: list[dict[str, object]] = []
            for view_index in range(int(station.fe_indices.size)):
                pair_id = int(pair_ids[view_index])
                pair_values.setdefault(pair_id, []).append(
                    standardized[view_index].copy()
                )
                pair_coverages.setdefault(pair_id, []).append(
                    covered[view_index].copy()
                )
                obstacle_label = (
                    "crosses_obstacle"
                    if obstacle_probability[view_index] >= 0.5
                    else "clear_line_of_sight"
                )
                obstacle_values[obstacle_label].append(
                    standardized[view_index].copy()
                )
                view_rows.append(
                    {
                        "view_index": int(view_index),
                        "fe_orientation_index": int(
                            station.fe_indices[view_index]
                        ),
                        "pb_orientation_index": int(
                            station.pb_indices[view_index]
                        ),
                        "shield_pair_id": pair_id,
                        "observed_total_count": float(
                            observed_totals[view_index]
                        ),
                        "predictive_total_mean": float(
                            predictive_total_mean[view_index]
                        ),
                        "predictive_total_std": float(
                            predictive_total_std[view_index]
                        ),
                        "standardized_total_residual": float(
                            total_standardized[view_index]
                        ),
                        "maximum_abs_standardized_bin_residual": float(
                            np.max(np.abs(standardized[view_index]))
                        ),
                        "p95_abs_standardized_bin_residual": float(
                            np.quantile(
                                np.abs(standardized[view_index]),
                                0.95,
                            )
                        ),
                        "marginal_bin_coverage_fraction": float(
                            np.mean(covered[view_index])
                        ),
                        "posterior_obstacle_crossing_probability": float(
                            obstacle_probability[view_index]
                        ),
                    }
                )
            flat_count = min(
                maximum_worst_bins,
                int(standardized.size),
            )
            if flat_count:
                flat_abs = np.abs(standardized).reshape(-1)
                candidate_flat = np.argpartition(
                    flat_abs,
                    -flat_count,
                )[-flat_count:]
                view_indices, bin_indices = np.unravel_index(
                    candidate_flat,
                    standardized.shape,
                )
                for view_index, bin_index in zip(
                    view_indices.tolist(),
                    bin_indices.tolist(),
                    strict=True,
                ):
                    worst_rows.append(
                        {
                            "station_sequence_id": int(
                                station.station_sequence_id
                            ),
                            "view_index": int(view_index),
                            "shield_pair_id": int(pair_ids[view_index]),
                            "energy_keV": float(
                                station.energy_axis_keV[bin_index]
                            ),
                            "bin_index": int(bin_index),
                            "observed_count": float(
                                observed[view_index, bin_index]
                            ),
                            "predictive_mean": float(
                                predictive_mean[view_index, bin_index]
                            ),
                            "predictive_std": float(
                                predictive_std[view_index, bin_index]
                            ),
                            "standardized_residual": float(
                                standardized[view_index, bin_index]
                            ),
                        }
                    )
            innovation = dict(
                model.posterior_predictive_innovation_numpy(
                    observed,
                    components[0],
                    components[1],
                    components[2],
                    station.live_times_s,
                    weights,
                    confidence=probability,
                )
            )
            full_log_likelihood = np.asarray(
                model.log_likelihood_numpy(
                    observed,
                    components[0],
                    components[1],
                    components[2],
                    station.live_times_s,
                ),
                dtype=np.float64,
            )
            log_weights = np.log(
                np.maximum(weights, np.finfo(np.float64).tiny)
            )
            full_log_predictive_density = float(
                logsumexp(log_weights + full_log_likelihood)
            )
            station_isotope_ablation: dict[str, object] = {}
            line_identity = tuple(model.line_identity)
            for isotope in self.joint_isotope_order():
                isotope_line_mask = np.asarray(
                    [
                        str(payload["isotope"]) == isotope
                        for payload in line_identity
                    ],
                    dtype=np.bool_,
                )
                if not np.any(isotope_line_mask):
                    raise RuntimeError(
                        f"No full-spectrum line columns exist for {isotope}."
                    )
                ablated_total = components[0].copy()
                ablated_uncollided = components[1].copy()
                ablated_total[..., isotope_line_mask] = 0.0
                ablated_uncollided[..., isotope_line_mask] = 0.0
                ablated_log_likelihood = np.asarray(
                    model.log_likelihood_numpy(
                        observed,
                        ablated_total,
                        ablated_uncollided,
                        components[2],
                        station.live_times_s,
                    ),
                    dtype=np.float64,
                )
                ablated_log_predictive_density = float(
                    logsumexp(log_weights + ablated_log_likelihood)
                )
                density_delta = (
                    full_log_predictive_density
                    - ablated_log_predictive_density
                )
                isotope_ablation_values[isotope].append(density_delta)
                ablated_innovation = dict(
                    model.posterior_predictive_innovation_numpy(
                        observed,
                        ablated_total,
                        ablated_uncollided,
                        components[2],
                        station.live_times_s,
                        weights,
                        confidence=probability,
                    )
                )
                station_isotope_ablation[isotope] = {
                    "full_minus_ablation_log_predictive_density": float(
                        density_delta
                    ),
                    "ablated_model_native_innovation": ablated_innovation,
                }
            station_results.append(
                {
                    "station_sequence_id": int(station.station_sequence_id),
                    "pose_index": int(station.pose_idx),
                    "view_count": int(station.fe_indices.size),
                    "energy_bin_count": int(station.energy_axis_keV.size),
                    "observed_total_count": float(np.sum(observed_totals)),
                    "predictive_total_mean": float(
                        np.sum(predictive_total_mean)
                    ),
                    "maximum_abs_standardized_bin_residual": float(
                        np.max(np.abs(standardized))
                    ),
                    "p95_abs_standardized_bin_residual": float(
                        np.quantile(np.abs(standardized), 0.95)
                    ),
                    "marginal_bin_coverage_fraction": float(
                        np.mean(covered)
                    ),
                    "model_native_innovation": innovation,
                    "isotope_response_ablation": station_isotope_ablation,
                    "views": view_rows,
                }
            )

        def _group_summary(
            values: Sequence[NDArray[np.float64]],
            coverage: Sequence[NDArray[np.bool_]] | None = None,
        ) -> dict[str, float | int | None]:
            """Summarize residual arrays for one physical view group."""
            if not values:
                return {
                    "view_count": 0,
                    "mean_standardized_bin_residual": None,
                    "maximum_abs_standardized_bin_residual": None,
                    "p95_abs_standardized_bin_residual": None,
                    "marginal_bin_coverage_fraction": None,
                }
            flattened = np.concatenate(
                [np.asarray(value, dtype=np.float64).reshape(-1) for value in values]
            )
            return {
                "view_count": int(len(values)),
                "mean_standardized_bin_residual": float(np.mean(flattened)),
                "maximum_abs_standardized_bin_residual": float(
                    np.max(np.abs(flattened))
                ),
                "p95_abs_standardized_bin_residual": float(
                    np.quantile(np.abs(flattened), 0.95)
                ),
                "marginal_bin_coverage_fraction": (
                    None
                    if coverage is None
                    else float(
                        np.mean(
                            np.concatenate(
                                [
                                    np.asarray(value, dtype=np.bool_).reshape(-1)
                                    for value in coverage
                                ]
                            )
                        )
                    )
                ),
            }

        pair_summary = {
            str(pair_id): _group_summary(
                pair_values[pair_id],
                pair_coverages[pair_id],
            )
            for pair_id in sorted(pair_values)
        }
        obstacle_summary = {
            label: _group_summary(values)
            for label, values in obstacle_values.items()
        }
        isotope_ablation_summary = {
            isotope: {
                "station_count": int(len(values)),
                "sum_full_minus_ablation_log_predictive_density": float(
                    np.sum(values, dtype=np.float64)
                ),
                "median_full_minus_ablation_log_predictive_density": float(
                    np.median(values)
                ),
                "minimum_full_minus_ablation_log_predictive_density": float(
                    np.min(values)
                ),
                "maximum_full_minus_ablation_log_predictive_density": float(
                    np.max(values)
                ),
            }
            for isotope, values in isotope_ablation_values.items()
            if values
        }
        worst_rows.sort(
            key=lambda row: abs(float(row["standardized_residual"])),
            reverse=True,
        )
        return {
            "available": True,
            "sampling_semantics": (
                "stratified_posterior_particle_draw_then_one_exact_"
                "generative_spectrum_per_draw"
            ),
            "sample_count": count,
            "confidence": probability,
            "candidate_isotopes": list(self.joint_isotope_order()),
            "stations": station_results,
            "shield_pair_summary": pair_summary,
            "obstacle_line_of_sight_summary": obstacle_summary,
            "isotope_response_ablation_summary": isotope_ablation_summary,
            "worst_standardized_bin_residuals": worst_rows[
                :maximum_worst_bins
            ],
            "isotope_response_ablation_semantics": (
                "diagnostic full-spectrum response-column ablation with all "
                "other posterior source states and weights held fixed; this "
                "is not leave-one-isotope-out model evidence"
            ),
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
            ordinary_maximum = int(filt.config.max_sources or 0)
            boundary_mass = float(
                sum(
                    mass
                    for cardinality, mass in distribution.items()
                    if int(cardinality) >= ordinary_maximum
                )
            )
            radii = [
                (
                    None
                    if mode.credible_surface_path_radius_95_m is None
                    else float(mode.credible_surface_path_radius_95_m)
                )
                for mode in point_estimate.modes
            ]
            connected_masses = [
                float(mode.surface_connected_mass) for mode in point_estimate.modes
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
                "current_ess": bool(ess_ratio >= self.pf_config.converge_min_ess_ratio),
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
