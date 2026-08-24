"""Types and validated configuration for DSS-PP planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from measurement.detector_geometry import DEFAULT_PF_DETECTOR_APERTURE_SAMPLES
from planning.program_types import ShieldProgram


DEFAULT_DSS_PP_LIVE_TIME_S = 30.0
DEFAULT_DSS_PP_ROBOT_SPEED_M_S = 0.5
DEFAULT_DSS_PP_ROTATION_OVERHEAD_S = 0.5


def estimate_lambda_cost(
    information_gains: NDArray[np.float64],
    motion_costs: NDArray[np.float64],
    *,
    method: str = "range",
    scale: float = 1.0,
    eps: float = 1.0e-12,
) -> float:
    """Scale motion cost to the spread of exact joint information gain."""
    gains = np.asarray(information_gains, dtype=float).reshape(-1)
    costs = np.asarray(motion_costs, dtype=float).reshape(-1)
    gains = gains[np.isfinite(gains)]
    costs = costs[np.isfinite(costs)]
    if gains.size == 0 or costs.size == 0:
        return 0.0
    if method == "range":
        gain_scale = float(np.ptp(gains))
        cost_scale = float(np.ptp(costs))
    elif method == "iqr":
        gain_scale = float(np.quantile(gains, 0.75) - np.quantile(gains, 0.25))
        cost_scale = float(np.quantile(costs, 0.75) - np.quantile(costs, 0.25))
    else:
        raise ValueError(f"Unknown lambda_cost method: {method}")
    if gain_scale <= float(eps) or cost_scale <= float(eps):
        return 0.0
    return float(scale) * gain_scale / cost_scale


@dataclass(frozen=True)
class SignatureMode:
    """Represent one posterior source mode used for shield signatures."""

    isotope: str
    position_xyz: NDArray[np.float64]
    strength_cps_1m: float
    weight: float
    spread_m: float
    isotope_presence_probability: float | None = None
    surface_chart_id: int | None = None
    surface_uv: tuple[float, float] | None = None


@dataclass(frozen=True)
class DSSPPConfig:
    """Configuration for Differential Shield-Signature Path Planning."""

    max_programs: int = 40
    program_length: int = 2
    mode_cluster_radius_m: float = 1.5
    max_modes_per_isotope: int = 5
    planning_particles: int | None = None
    planning_method: str | None = None
    live_time_s: float = DEFAULT_DSS_PP_LIVE_TIME_S
    lambda_eig: float = 1.0
    lambda_distance: float | None = None
    lambda_time: float = 0.0
    lambda_horizontal_time: float | None = None
    lambda_mast_vertical_time: float | None = None
    lambda_settling_time: float | None = None
    lambda_rotation: float = 0.0
    lambda_coverage: float = 0.0
    lambda_bearing_diversity: float = 0.0
    lambda_frontier: float = 0.0
    lambda_turn_smoothness: float = 0.0
    lambda_local_orbit: float = 0.0
    lambda_elevation_condition: float = 0.0
    eta_revisit: float = 0.0
    coverage_radius_m: float = 3.0
    coverage_surface_quadrature_max_points: int = 65536
    coverage_surface_max_hausdorff_m: float = 0.75
    coverage_floor_quantile: float = 0.0
    coverage_floor_weight: float = 0.0
    min_station_separation_m: float = 0.0
    detector_aperture_samples: int = DEFAULT_PF_DETECTOR_APERTURE_SAMPLES
    robot_speed_m_s: float = DEFAULT_DSS_PP_ROBOT_SPEED_M_S
    rotation_overhead_s: float = DEFAULT_DSS_PP_ROTATION_OVERHEAD_S
    augment_candidates: bool = True
    max_augmented_candidates: int = 256
    ring_radii_m: tuple[float, ...] = (2.0, 3.5, 5.0)
    ring_angles: int = 12
    local_orbit_sigma_m: float = 0.75
    elevation_pair_z_scale_m: float = 2.0
    elevation_pair_xy_scale_m: float = 4.0
    elevation_angle_threshold_deg: float = 15.0
    forced_program_pair_ids: tuple[int, ...] | None = None
    diagnostic_ranked_node_limit: int = 64
    exact_eig_pose_limit: int = 4
    exact_eig_action_limit: int = 192
    exact_eig_coverage_reserve: int = 1
    exact_eig_program_diversity_reserve: int = 0
    exact_eig_memory_budget_bytes: int = 4 * 1024 * 1024 * 1024
    proxy_memory_budget_bytes: int = 256 * 1024 * 1024
    proxy_planning_particles: int = 16
    proxy_eig_samples: int = 2
    shield_program_search_policy: str = "predeclared_library"
    legacy_program_guard_enabled: bool = True
    conditional_greedy_one_swap: bool = True
    exact_eig_pose_min: int = 8
    exact_eig_pose_max: int = 16
    exact_eig_pose_step: int = 4
    proxy_stability_refinement_pool: int = 24
    proxy_stability_replicates: int = 3
    proxy_boundary_confidence: float = 0.95
    proxy_top_k_jaccard_min: float = 0.75
    shield_view_count_shadow_enabled: bool = False
    shield_view_count_shadow_candidate_counts: tuple[int, ...] = (2, 4, 8)
    shield_view_count_shadow_retention_fraction: float = 0.95
    shield_view_count_shadow_per_comparison_confidence: float = 0.95

    def __post_init__(self) -> None:
        """Validate every planner field before it can affect observations."""

        def _integer(
            value: object,
            name: str,
            *,
            minimum: int,
        ) -> int:
            """Return one exact integer above an inclusive lower bound."""
            if isinstance(value, bool) or not isinstance(
                value,
                (int, np.integer),
            ):
                raise ValueError(f"{name} must be an integer.")
            resolved = int(value)
            if resolved < minimum:
                raise ValueError(f"{name} must be at least {minimum}.")
            return resolved

        def _number(
            value: object,
            name: str,
            *,
            minimum: float,
            maximum: float | None = None,
            strict_minimum: bool = False,
            strict_maximum: bool = False,
        ) -> float:
            """Return one finite numeric planner value inside its domain."""
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float, np.integer, np.floating),
            ):
                raise ValueError(f"{name} must be numeric.")
            resolved = float(value)
            if not np.isfinite(resolved):
                raise ValueError(f"{name} must be finite.")
            below = resolved <= minimum if strict_minimum else resolved < minimum
            if below:
                relation = "greater than" if strict_minimum else "at least"
                raise ValueError(f"{name} must be {relation} {minimum}.")
            if maximum is not None:
                above = resolved >= maximum if strict_maximum else resolved > maximum
                if above:
                    relation = "less than" if strict_maximum else "at most"
                    raise ValueError(f"{name} must be {relation} {maximum}.")
            return resolved

        positive_integer_fields = {
            "max_programs": self.max_programs,
            "program_length": self.program_length,
            "max_modes_per_isotope": self.max_modes_per_isotope,
            "coverage_surface_quadrature_max_points": (
                self.coverage_surface_quadrature_max_points
            ),
            "detector_aperture_samples": self.detector_aperture_samples,
            "max_augmented_candidates": self.max_augmented_candidates,
            "exact_eig_pose_limit": self.exact_eig_pose_limit,
            "exact_eig_action_limit": self.exact_eig_action_limit,
            "exact_eig_memory_budget_bytes": (self.exact_eig_memory_budget_bytes),
            "proxy_memory_budget_bytes": self.proxy_memory_budget_bytes,
            "proxy_eig_samples": self.proxy_eig_samples,
            "exact_eig_pose_min": self.exact_eig_pose_min,
            "exact_eig_pose_max": self.exact_eig_pose_max,
            "exact_eig_pose_step": self.exact_eig_pose_step,
            "proxy_stability_refinement_pool": (self.proxy_stability_refinement_pool),
            "proxy_stability_replicates": self.proxy_stability_replicates,
        }
        for name, value in positive_integer_fields.items():
            _integer(value, name, minimum=1)
        _integer(self.ring_angles, "ring_angles", minimum=4)
        _integer(
            self.proxy_planning_particles,
            "proxy_planning_particles",
            minimum=2,
        )
        for name, value in {
            "diagnostic_ranked_node_limit": self.diagnostic_ranked_node_limit,
            "exact_eig_coverage_reserve": self.exact_eig_coverage_reserve,
            "exact_eig_program_diversity_reserve": (
                self.exact_eig_program_diversity_reserve
            ),
        }.items():
            _integer(value, name, minimum=0)
        if self.planning_particles is not None:
            _integer(
                self.planning_particles,
                "planning_particles",
                minimum=2,
            )
        if self.planning_method not in {None, "resample", "top_weight"}:
            raise ValueError(
                "planning_method must be None, 'resample', or 'top_weight'."
            )
        for name, value in {
            "augment_candidates": self.augment_candidates,
            "legacy_program_guard_enabled": self.legacy_program_guard_enabled,
            "conditional_greedy_one_swap": self.conditional_greedy_one_swap,
            "shield_view_count_shadow_enabled": (self.shield_view_count_shadow_enabled),
        }.items():
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean.")
        if self.shield_program_search_policy not in {
            "predeclared_library",
            "conditional_greedy_shadow",
            "conditional_greedy_all_pairs",
        }:
            raise ValueError(
                "shield_program_search_policy must be predeclared_library, "
                "conditional_greedy_shadow, or conditional_greedy_all_pairs."
            )
        conditional_search_enabled = self.shield_program_search_policy in {
            "conditional_greedy_shadow",
            "conditional_greedy_all_pairs",
        }
        predeclared_search_enabled = (
            self.shield_program_search_policy == "predeclared_library"
        )
        legacy_execution_enabled = self.shield_program_search_policy in {
            "predeclared_library",
            "conditional_greedy_shadow",
        }

        nonnegative_fields = {
            "lambda_eig": self.lambda_eig,
            "lambda_time": self.lambda_time,
            "lambda_rotation": self.lambda_rotation,
            "lambda_coverage": self.lambda_coverage,
            "lambda_bearing_diversity": self.lambda_bearing_diversity,
            "lambda_frontier": self.lambda_frontier,
            "lambda_turn_smoothness": self.lambda_turn_smoothness,
            "lambda_local_orbit": self.lambda_local_orbit,
            "lambda_elevation_condition": self.lambda_elevation_condition,
            "eta_revisit": self.eta_revisit,
            "coverage_radius_m": self.coverage_radius_m,
            "coverage_floor_weight": self.coverage_floor_weight,
            "min_station_separation_m": self.min_station_separation_m,
            "rotation_overhead_s": self.rotation_overhead_s,
        }
        if self.lambda_distance is not None:
            nonnegative_fields["lambda_distance"] = self.lambda_distance
        if self.lambda_horizontal_time is not None:
            nonnegative_fields["lambda_horizontal_time"] = self.lambda_horizontal_time
        if self.lambda_mast_vertical_time is not None:
            nonnegative_fields["lambda_mast_vertical_time"] = (
                self.lambda_mast_vertical_time
            )
        if self.lambda_settling_time is not None:
            nonnegative_fields["lambda_settling_time"] = self.lambda_settling_time
        for name, value in nonnegative_fields.items():
            _number(value, name, minimum=0.0)
        if conditional_search_enabled and float(self.lambda_eig) <= 0.0:
            raise ValueError(
                "Conditional-greedy shield search requires lambda_eig > 0."
            )
        if conditional_search_enabled and int(self.proxy_eig_samples) < 2:
            raise ValueError(
                "Conditional-greedy shield search requires proxy_eig_samples >= 2."
            )
        if float(self.lambda_rotation) != 0.0:
            raise ValueError(
                "lambda_rotation is retired: shield programs must be selected "
                "by exact EIG. Model a measured rotation duration through the "
                "time-cost contract if it becomes physically material."
            )

        positive_fields = {
            "mode_cluster_radius_m": self.mode_cluster_radius_m,
            "live_time_s": self.live_time_s,
            "coverage_surface_max_hausdorff_m": (self.coverage_surface_max_hausdorff_m),
            "robot_speed_m_s": self.robot_speed_m_s,
            "local_orbit_sigma_m": self.local_orbit_sigma_m,
            "elevation_pair_z_scale_m": self.elevation_pair_z_scale_m,
            "elevation_pair_xy_scale_m": self.elevation_pair_xy_scale_m,
        }
        for name, value in positive_fields.items():
            _number(value, name, minimum=0.0, strict_minimum=True)
        _number(
            self.elevation_angle_threshold_deg,
            "elevation_angle_threshold_deg",
            minimum=0.0,
            maximum=180.0,
            strict_minimum=True,
        )
        _number(
            self.coverage_floor_quantile,
            "coverage_floor_quantile",
            minimum=0.0,
            maximum=1.0,
        )
        _number(
            self.proxy_boundary_confidence,
            "proxy_boundary_confidence",
            minimum=0.0,
            maximum=1.0,
            strict_minimum=True,
            strict_maximum=True,
        )
        _number(
            self.proxy_top_k_jaccard_min,
            "proxy_top_k_jaccard_min",
            minimum=0.0,
            maximum=1.0,
        )
        _number(
            self.shield_view_count_shadow_retention_fraction,
            "shield_view_count_shadow_retention_fraction",
            minimum=0.0,
            maximum=1.0,
            strict_minimum=True,
        )
        _number(
            self.shield_view_count_shadow_per_comparison_confidence,
            "shield_view_count_shadow_per_comparison_confidence",
            minimum=0.0,
            maximum=1.0,
            strict_minimum=True,
            strict_maximum=True,
        )
        shadow_counts = tuple(
            _integer(
                value,
                "shield_view_count_shadow_candidate_counts",
                minimum=1,
            )
            for value in self.shield_view_count_shadow_candidate_counts
        )
        if len(shadow_counts) < 2:
            raise ValueError(
                "shield_view_count_shadow_candidate_counts must contain at "
                "least two values."
            )
        if tuple(sorted(set(shadow_counts))) != shadow_counts:
            raise ValueError(
                "shield_view_count_shadow_candidate_counts must be strictly "
                "increasing and unique."
            )
        if self.shield_view_count_shadow_enabled:
            if self.shield_program_search_policy != "conditional_greedy_all_pairs":
                raise ValueError(
                    "Shield view-count shadow audit requires "
                    "conditional_greedy_all_pairs."
                )
            if shadow_counts != (2, 4, 8):
                raise ValueError(
                    "Enabled shield view-count shadow candidates must be "
                    "exactly (2, 4, 8)."
                )
            if int(self.program_length) != int(shadow_counts[-1]):
                raise ValueError(
                    "Executed program_length must equal the shadow reference "
                    "view count."
                )
        if not self.ring_radii_m:
            raise ValueError("ring_radii_m must not be empty.")
        for index, radius in enumerate(self.ring_radii_m):
            _number(
                radius,
                f"ring_radii_m[{index}]",
                minimum=0.0,
                strict_minimum=True,
            )
        if self.forced_program_pair_ids is not None:
            pair_ids = tuple(
                _integer(value, "forced_program_pair_ids", minimum=0)
                for value in self.forced_program_pair_ids
            )
            if not pair_ids:
                raise ValueError(
                    "forced_program_pair_ids must contain at least one pair."
                )
            if len(set(pair_ids)) != len(pair_ids):
                raise ValueError("forced_program_pair_ids must not contain duplicates.")
        if predeclared_search_enabled and int(self.exact_eig_coverage_reserve) > int(
            self.exact_eig_pose_limit
        ):
            raise ValueError(
                "exact_eig_coverage_reserve must fit within exact_eig_pose_limit."
            )
        if int(self.exact_eig_program_diversity_reserve) != 0:
            raise ValueError(
                "exact_eig_program_diversity_reserve is retired because every "
                "program is exactly evaluated at every shortlisted pose."
            )
        if legacy_execution_enabled and int(self.exact_eig_pose_limit) > int(
            self.exact_eig_action_limit
        ):
            raise ValueError(
                "exact_eig_action_limit must accommodate at least one program "
                "for every shortlisted pose."
            )
        if int(self.exact_eig_pose_max) < int(self.exact_eig_pose_min):
            raise ValueError("exact_eig_pose_max must be at least exact_eig_pose_min.")
        if int(self.exact_eig_pose_step) > int(self.exact_eig_pose_max):
            raise ValueError("exact_eig_pose_step must not exceed exact_eig_pose_max.")
        if int(self.proxy_stability_replicates) < 2:
            raise ValueError(
                "proxy_stability_replicates must include an independent "
                "boundary recheck."
            )
        if int(self.proxy_stability_refinement_pool) <= int(self.exact_eig_pose_max):
            raise ValueError(
                "proxy_stability_refinement_pool must include the pose just "
                "outside the maximum exact shortlist."
            )
        if conditional_search_enabled and int(self.exact_eig_coverage_reserve) > int(
            self.exact_eig_pose_min
        ):
            raise ValueError(
                "exact_eig_coverage_reserve must fit within the minimum "
                "conditional-greedy pose shortlist."
            )


@dataclass(frozen=True)
class _JointProgramSpectrumComponents:
    """Store source-resolved full-spectrum inputs for aligned PF particles."""

    total_pnvsl: NDArray[np.float64]
    uncollided_pnvsl: NDArray[np.float64]
    features_pnvslf: NDArray[np.float64]
    live_times_v: NDArray[np.float64]
    contract_hash_sha256: str


@dataclass(frozen=True)
class _DeviceJointProgramSpectrumComponents:
    """Store source-resolved full-spectrum inputs on one Torch device."""

    total_pnvsl: object
    uncollided_pnvsl: object
    features_pnvslf: object
    live_times_v: object
    contract_hash_sha256: str

    def __post_init__(self) -> None:
        """Require aligned float64 Torch tensors on exactly one device."""
        import torch

        values = (
            self.total_pnvsl,
            self.uncollided_pnvsl,
            self.features_pnvslf,
            self.live_times_v,
        )
        if any(not torch.is_tensor(value) for value in values):
            raise TypeError(
                "Device-resident spectrum components must be Torch tensors."
            )
        reference = values[0]
        if any(value.dtype != torch.float64 for value in values):
            raise TypeError(
                "Device-resident spectrum components must use torch.float64."
            )
        if any(value.device != reference.device for value in values[1:]):
            raise ValueError(
                "Device-resident spectrum components must share one device."
            )
        if not str(self.contract_hash_sha256):
            raise ValueError("Device-resident spectrum components need a model hash.")


@dataclass(frozen=True)
class DSSPPNode:
    """Store one candidate station and shield program evaluation."""

    pose_index: int
    pose_xyz: NDArray[np.float64]
    program: ShieldProgram
    score: float
    static_score: float
    distance_weight: float
    information_gain: float
    coverage_gain: float
    revisit_penalty: float
    bearing_diversity_gain: float
    frontier_gain: float
    turn_penalty: float
    local_orbit_gain: float
    elevation_condition_gain: float


@dataclass(frozen=True)
class DSSPPResult:
    """Return the selected one-step DSS-PP action."""

    next_pose: NDArray[np.float64]
    next_pose_index: int
    shield_program: ShieldProgram
    score: float
    sequence: tuple[DSSPPNode, ...]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _PendingDSSPPNode:
    """Store geometry-only terms before exact joint EIG evaluation."""

    pose_index: int
    pose_xyz: NDArray[np.float64]
    program: ShieldProgram
    static_score: float
    coverage_gain: float
    revisit_penalty: float
    bearing_diversity_gain: float
    frontier_gain: float
    turn_penalty: float
    local_orbit_gain: float
    elevation_condition_gain: float
