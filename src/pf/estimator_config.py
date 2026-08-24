"""Validated configuration contract for the rotating-shield PF estimator."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from pf.defaults import DEFAULT_MAX_SOURCES_PER_ISOTOPE
from pf.strength_prior import StrengthPrior
from pf.structural_rj import (
    POISSON_GEOMETRIC_TAIL_CARDINALITY_PRIOR_POLICY,
    TRUNCATED_POISSON_CARDINALITY_PRIOR_POLICY,
    validate_cardinality_prior_policy,
)


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
    adaptive_stop_minimum_joint_map_cardinality_probability: float = 0.95
    adaptive_stop_maximum_upper_cardinality_mass: float = 0.05
    adaptive_stop_maximum_surface_path_radius_95_m: float = 0.5
    adaptive_stop_innovation_confidence: float = 0.99
    target_ess_ratio: float = 0.5
    max_temper_steps: int = 256
    min_delta_beta: float = 1e-10
    joint_rejuvenation_min_sweeps: int = 1
    joint_rejuvenation_max_sweeps: int = 2
    joint_rejuvenation_min_state_change_weight_mass: float = 0.10
    joint_rejuvenation_min_surface_esjd_m2: float = 1.0e-4
    joint_rejuvenation_min_log_strength_esjd: float = 1.0e-4
    joint_rejuvenation_min_k_transition_weight_mass: float = 1.0e-4
    joint_rejuvenation_boundary_mass_threshold: float = 0.05
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
    position_max: tuple[float, float, float] = (10.0, 10.0, 10.0)
    init_num_sources: tuple[int, int] = (
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

    def __post_init__(self) -> None:
        """Validate and normalize estimator configuration values."""
        integer_fields = (
            ("num_particles", self.num_particles, 1),
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
            "adaptive_stop_minimum_joint_map_cardinality_probability",
            "adaptive_stop_maximum_upper_cardinality_mass",
            "adaptive_stop_maximum_surface_path_radius_95_m",
            "adaptive_stop_innovation_confidence",
            "target_ess_ratio",
            "min_delta_beta",
            "joint_rejuvenation_min_state_change_weight_mass",
            "joint_rejuvenation_min_surface_esjd_m2",
            "joint_rejuvenation_min_log_strength_esjd",
            "joint_rejuvenation_min_k_transition_weight_mass",
            "joint_rejuvenation_boundary_mass_threshold",
            "joint_smc_soft_wall_time_s",
            "joint_guided_initialization_prior_row_probability",
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
            self.adaptive_stop_innovation_confidence,
            name="adaptive_stop_innovation_confidence",
        )
        if not 0.0 < innovation_confidence < 1.0:
            raise ValueError(
                "adaptive_stop_innovation_confidence must lie in (0, 1)."
            )
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
        self.surface_diagnostic_response_cache_max_entries = max(
            0,
            int(self.surface_diagnostic_response_cache_max_entries),
        )
        self.adaptive_stop_maximum_surface_path_radius_95_m = float(
            self.adaptive_stop_maximum_surface_path_radius_95_m
        )
        if (
            not np.isfinite(self.adaptive_stop_maximum_surface_path_radius_95_m)
            or self.adaptive_stop_maximum_surface_path_radius_95_m < 0.0
        ):
            raise ValueError(
                "adaptive_stop_maximum_surface_path_radius_95_m must be finite "
                "and nonnegative."
            )
        for probability_field, lower_inclusive in (
            ("adaptive_stop_minimum_joint_map_cardinality_probability", False),
            ("adaptive_stop_maximum_upper_cardinality_mass", True),
            ("adaptive_stop_innovation_confidence", False),
            ("joint_rejuvenation_boundary_mass_threshold", True),
        ):
            probability = float(getattr(self, probability_field))
            lower_valid = probability >= 0.0 if lower_inclusive else probability > 0.0
            if not np.isfinite(probability) or not lower_valid or probability > 1.0:
                lower_symbol = "[" if lower_inclusive else "("
                raise ValueError(f"{probability_field} must be in {lower_symbol}0, 1].")
            setattr(self, probability_field, probability)
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
