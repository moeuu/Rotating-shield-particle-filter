"""Differential Shield-Signature Path Planning.

DSS-PP plans over a joint robot-pose and shield-program action. It samples
future spectra from the same validated generative model and evaluates them
with the same sole full-spectrum likelihood used by the online PF.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.spatial import cKDTree

from measurement.continuous_kernels import ContinuousKernel
from measurement.detector_geometry import DEFAULT_PF_DETECTOR_APERTURE_SAMPLES
from pf.estimator import JointPlanningParticles, RotatingShieldPFEstimator
from pf.full_spectrum import validate_full_spectrum_model
from pf.posterior import (
    validated_probability,
    validated_probability_distribution,
)
from pf.randomness import named_random_generator, named_stream_seed
from planning.candidate_generation import sample_low_discrepancy_heights
from planning.traversability import shortest_grid_path_length
from runtime_defaults import (
    DEFAULT_MEASUREMENT_TIME_S,
    DEFAULT_ROBOT_SPEED_M_S,
    DEFAULT_ROTATION_OVERHEAD_S,
)


_DSS_PP_PATH_LENGTH_CACHE: dict[tuple[object, ...], float] = {}
_DSS_PP_PATH_LENGTH_CACHE_MAX = 20000


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
class ShieldProgram:
    """Represent a short sequence of Fe/Pb shield orientation pairs."""

    name: str
    pair_ids: tuple[int, ...]
    kind: str


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
    live_time_s: float = DEFAULT_MEASUREMENT_TIME_S
    lambda_eig: float = 1.0
    lambda_distance: float | None = None
    lambda_time: float = 0.0
    lambda_rotation: float = 0.15
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
    robot_speed_m_s: float = DEFAULT_ROBOT_SPEED_M_S
    rotation_overhead_s: float = DEFAULT_ROTATION_OVERHEAD_S
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
    exact_eig_action_limit: int = 32
    exact_eig_coverage_reserve: int = 4
    exact_eig_program_diversity_reserve: int = 4
    proxy_memory_budget_bytes: int = 256 * 1024 * 1024
    proxy_planning_particles: int = 16
    proxy_eig_samples: int = 2

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
                above = (
                    resolved >= maximum
                    if strict_maximum
                    else resolved > maximum
                )
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
            "exact_eig_action_limit": self.exact_eig_action_limit,
            "proxy_memory_budget_bytes": self.proxy_memory_budget_bytes,
            "proxy_eig_samples": self.proxy_eig_samples,
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
        if not isinstance(self.augment_candidates, bool):
            raise ValueError("augment_candidates must be a boolean.")

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
        for name, value in nonnegative_fields.items():
            _number(value, name, minimum=0.0)

        positive_fields = {
            "mode_cluster_radius_m": self.mode_cluster_radius_m,
            "live_time_s": self.live_time_s,
            "coverage_surface_max_hausdorff_m": (
                self.coverage_surface_max_hausdorff_m
            ),
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
                raise ValueError(
                    "forced_program_pair_ids must not contain duplicates."
                )
        if (
            int(self.exact_eig_coverage_reserve)
            + int(self.exact_eig_program_diversity_reserve)
            > int(self.exact_eig_action_limit)
        ):
            raise ValueError(
                "Exact-EIG reserve counts must fit within exact_eig_action_limit."
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


def _node_diagnostic_payload(node: DSSPPNode, rank: int) -> dict[str, object]:
    """Return a JSON-serializable diagnostic payload for one DSS-PP node."""
    return {
        "rank": int(rank),
        "pose_index": int(node.pose_index),
        "pose_xyz": [float(value) for value in np.asarray(node.pose_xyz, dtype=float)],
        "program_name": str(node.program.name),
        "program_kind": str(node.program.kind),
        "pair_ids": [int(value) for value in node.program.pair_ids],
        "score": float(node.score),
        "static_score": float(node.static_score),
        "distance_weight": float(node.distance_weight),
        "information_gain": float(node.information_gain),
        "coverage_gain": float(node.coverage_gain),
        "revisit_penalty": float(node.revisit_penalty),
        "bearing_diversity_gain": float(node.bearing_diversity_gain),
        "frontier_gain": float(node.frontier_gain),
        "turn_penalty": float(node.turn_penalty),
        "local_orbit_gain": float(node.local_orbit_gain),
        "elevation_condition_gain": float(node.elevation_condition_gain),
    }


def _mode_diagnostic_payload(mode: SignatureMode, index: int) -> dict[str, object]:
    """Return a compact diagnostic payload for one posterior source mode."""
    return {
        "index": int(index),
        "pos": [float(value) for value in np.asarray(mode.position_xyz, dtype=float)],
        "q": float(mode.strength_cps_1m),
        "marginal_existence_probability": float(mode.weight),
        "isotope_presence_probability": _isotope_presence_probability(
            [mode]
        ),
        "spread_m": float(mode.spread_m),
        "surface_chart_id": (
            None
            if mode.surface_chart_id is None
            else int(mode.surface_chart_id)
        ),
        "surface_uv": (
            None
            if mode.surface_uv is None
            else [float(value) for value in mode.surface_uv]
        ),
    }


def _component_leader_payloads(
    nodes: Sequence[DSSPPNode],
) -> dict[str, dict[str, object]]:
    """Return best-node diagnostics for individual DSS-PP score components."""
    node_list = list(nodes)
    if not node_list:
        return {}
    selectors: dict[str, Any] = {
        "score": lambda node: float(node.score),
        "information_gain": lambda node: float(node.information_gain),
        "coverage": lambda node: float(node.coverage_gain),
        "bearing_diversity": lambda node: float(node.bearing_diversity_gain),
        "frontier": lambda node: float(node.frontier_gain),
        "local_orbit": lambda node: float(node.local_orbit_gain),
        "elevation_condition": lambda node: float(node.elevation_condition_gain),
    }
    leaders: dict[str, dict[str, object]] = {}
    for name, selector in selectors.items():
        finite_nodes = [
            node for node in node_list if np.isfinite(float(selector(node)))
        ]
        if not finite_nodes:
            continue
        leader = max(finite_nodes, key=lambda node: float(selector(node)))
        payload = _node_diagnostic_payload(leader, 1)
        payload["component_value"] = float(selector(leader))
        leaders[name] = payload
    return leaders


def _normalise_weights(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return normalized weights and fail on an invalid posterior contract."""
    arr = np.asarray(weights, dtype=float).ravel()
    if arr.size == 0:
        return arr
    if np.any(~np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("Posterior weights must be finite and nonnegative.")
    total = float(np.sum(arr))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Posterior weights must contain positive mass.")
    return arr / total


def _posterior_mode_weights(
    weights: NDArray[np.float64] | Sequence[float],
) -> NDArray[np.float64]:
    """Return marginal existence probabilities without erasing source count."""
    arr = np.asarray(weights, dtype=object).reshape(-1)
    return np.fromiter(
        (
            validated_probability(
                value,
                name=f"Mode existence probability[{index}]",
            )
            for index, value in enumerate(arr)
        ),
        dtype=np.float64,
        count=arr.size,
    )


def _isotope_presence_probability(
    modes: Sequence[SignatureMode],
) -> float | None:
    """Return the shared PF probability that an isotope has at least one source."""
    if not modes:
        return None
    configured = [
        mode.isotope_presence_probability
        for mode in modes
        if mode.isotope_presence_probability is not None
    ]
    if not configured:
        return validated_probability(
            sum(
                validated_probability(
                    mode.weight,
                    name=(
                        "Implicit signature source-mode probability"
                        f"[{index}]"
                    ),
                )
                for index, mode in enumerate(modes)
            ),
            name="Implicit signature-isotope presence probability",
        )
    if len(configured) != len(modes):
        raise ValueError(
            "Modes for one isotope cannot mix explicit and implicit presence."
        )
    values = np.asarray(configured, dtype=object).reshape(-1)
    values = np.fromiter(
        (
            validated_probability(
                value,
                name=f"Signature isotope presence probability[{index}]",
            )
            for index, value in enumerate(values)
        ),
        dtype=np.float64,
        count=values.size,
    )
    if not np.allclose(values, values[:1], rtol=0.0, atol=1.0e-12):
        raise ValueError(
            "Modes for one isotope must share one presence probability."
        )
    return float(values[0])


def _flattened_posterior_mode_weights(
    modes_by_isotope: dict[str, list[SignatureMode]],
) -> NDArray[np.float64]:
    """Return concatenated mode masses while preserving each isotope's K=0 mass."""
    blocks = [
        _posterior_mode_weights([mode.weight for mode in modes])
        for modes in modes_by_isotope.values()
        if modes
    ]
    return np.concatenate(blocks) if blocks else np.zeros(0, dtype=float)


def _presence_weighted_rows(
    rows: Sequence[NDArray[np.float64]],
    presence_masses: Sequence[float],
    *,
    population_size: int,
) -> NDArray[np.float64]:
    """Average feature rows without renormalizing away absent-isotope mass."""
    if not rows:
        return np.zeros(0, dtype=float)
    stacked = np.vstack([np.asarray(row, dtype=float) for row in rows])
    raw_masses = np.asarray(presence_masses, dtype=object)
    masses = np.fromiter(
        (
            validated_probability(
                value,
                name=f"Feature-row presence probability[{index}]",
            )
            for index, value in enumerate(raw_masses)
        ),
        dtype=np.float64,
        count=raw_masses.size,
    )
    if masses.shape != (stacked.shape[0],):
        raise ValueError("Presence masses must match feature rows.")
    if (
        isinstance(population_size, bool)
        or not isinstance(population_size, (int, np.integer))
        or int(population_size) <= 0
    ):
        raise ValueError("population_size must be a positive integer.")
    denominator = int(population_size)
    return np.sum(stacked * masses[:, None], axis=0) / float(denominator)


def _planning_rng(
    rng: np.random.Generator | None,
) -> np.random.Generator:
    """Return the caller-owned persistent planning generator."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError("DSS planning requires an explicit persistent RNG.")
    return rng


def _validate_mode_capacity(
    estimator: RotatingShieldPFEstimator,
    config: DSSPPConfig,
) -> int:
    """Require the planner mode capacity to cover every PF source slot."""
    try:
        pf_config = estimator.pf_config
        configured_max_sources = pf_config.max_sources
    except AttributeError as error:
        raise TypeError(
            "DSS planning requires an estimator with an explicit PF config."
        ) from error
    if configured_max_sources is None:
        raise ValueError("DSS planning requires a finite PF max_sources value.")
    pf_max_sources = int(configured_max_sources)
    if pf_max_sources <= 0:
        raise ValueError("PF max_sources must be a positive integer.")
    configured = int(config.max_modes_per_isotope)
    if pf_max_sources > 0 and configured < pf_max_sources:
        raise ValueError(
            "max_modes_per_isotope must be at least the PF max_sources "
            f"({configured} < {pf_max_sources})."
        )
    return pf_max_sources


def _validate_eig_likelihood_contract(
    estimator: RotatingShieldPFEstimator,
    config: DSSPPConfig,
) -> None:
    """Require the exact full-spectrum model used by the joint PF."""
    if float(config.lambda_eig) <= 0.0:
        return
    model = validate_full_spectrum_model(
        estimator.full_spectrum_generative_model
    )
    if not callable(getattr(model, "cross_log_likelihood_numpy", None)):
        raise RuntimeError(
            "DSS EIG requires vectorized full-spectrum cross likelihoods."
        )
    if not callable(getattr(estimator, "planning_joint_particles", None)):
        raise RuntimeError("DSS EIG requires aligned joint PF particles.")


def _pair_indices(pair_id: int, num_orients: int) -> tuple[int, int]:
    """Return Fe and Pb indices from a flattened pair id."""
    return int(pair_id) // int(num_orients), int(pair_id) % int(num_orients)


def build_shield_program_library(
    normals: NDArray[np.float64],
    *,
    program_length: int = 2,
    max_programs: int = 40,
) -> list[ShieldProgram]:
    """Build balanced, repetition-free programs covering every Fe/Pb pair."""
    normal_arr = np.asarray(normals, dtype=float)
    if (
        normal_arr.ndim != 2
        or normal_arr.shape[1] != 3
        or np.any(~np.isfinite(normal_arr))
    ):
        raise ValueError("normals must be shaped (N, 3).")
    num_orients = int(normal_arr.shape[0])
    if num_orients <= 0:
        raise ValueError("normals must contain at least one orientation.")
    if (
        isinstance(program_length, bool)
        or not isinstance(program_length, (int, np.integer))
        or int(program_length) <= 0
    ):
        raise ValueError("program_length must be a positive integer.")
    if (
        isinstance(max_programs, bool)
        or not isinstance(max_programs, (int, np.integer))
        or int(max_programs) <= 0
    ):
        raise ValueError("max_programs must be a positive integer.")
    length = int(program_length)
    pair_count = num_orients * num_orients
    if length > pair_count:
        raise ValueError(
            "program_length cannot exceed the number of unique Fe/Pb pairs."
        )
    if num_orients == 8 and length == 8:
        orientation_axis = np.arange(num_orients, dtype=np.int64)
        slopes = orientation_axis[
            np.gcd(orientation_axis, num_orients) == 1
        ]
        latin_pb = (
            slopes[:, None, None] * orientation_axis[None, None, :]
            + orientation_axis[None, :, None]
        ) % num_orients
        latin_pairs = (
            orientation_axis[None, None, :] * num_orients + latin_pb
        ).reshape(-1, num_orients)
        fixed_fe_pairs = (
            orientation_axis[:, None] * num_orients
            + orientation_axis[None, :]
        )
        fixed_pb_pairs = (
            orientation_axis[None, :] * num_orients
            + orientation_axis[:, None]
        )
        pair_matrix = np.concatenate(
            (latin_pairs, fixed_fe_pairs, fixed_pb_pairs),
            axis=0,
        )
        partition_names = tuple(
            [f"latin_slope_{int(slope)}" for slope in slopes]
            + ["fixed_fe", "fixed_pb"]
        )
        required_programs = len(partition_names) * num_orients
        if int(max_programs) < required_programs:
            raise ValueError(
                "max_programs is too small for the canonical balanced "
                "multi-partition shield library "
                f"({int(max_programs)} < {required_programs})."
            )
        program_names = tuple(
            f"{partition_name}_{program_index:02d}"
            for partition_name in partition_names
            for program_index in range(num_orients)
        )
        programs = [
            ShieldProgram(
                name=program_name,
                pair_ids=tuple(int(pair_id) for pair_id in pair_row),
                kind="all_pair_balanced_multi_partition",
            )
            for program_name, pair_row in zip(
                program_names,
                pair_matrix,
                strict=True,
            )
        ]
        pair_occurrences = np.bincount(
            pair_matrix.reshape(-1),
            minlength=pair_count,
        )
        if (
            np.any(np.diff(np.sort(pair_matrix, axis=1), axis=1) == 0)
            or np.any(pair_occurrences <= 0)
            or int(np.max(pair_occurrences) - np.min(pair_occurrences)) != 0
        ):
            raise RuntimeError(
                "Canonical shield partitions must be repetition-free and "
                "pair-frequency balanced."
            )
        return programs

    required_programs = int(np.ceil(pair_count / float(length)))
    if int(max_programs) < required_programs:
        raise ValueError(
            "max_programs is too small to expose every Fe/Pb pair without "
            f"within-program repetition ({int(max_programs)} < "
            f"{required_programs})."
        )

    # Latin-diagonal order makes every Fe and Pb orientation occur uniformly
    # before the sequence is split into fixed-size station programs.
    orientation_axis = np.arange(num_orients, dtype=np.int64)
    ordered_pairs = (
        orientation_axis[None, :] * num_orients
        + (
            orientation_axis[None, :]
            + orientation_axis[:, None]
        )
        % num_orients
    ).reshape(-1)
    program_pair_indices = (
        np.arange(required_programs * length, dtype=np.int64) % pair_count
    )
    pair_matrix = ordered_pairs[program_pair_indices].reshape(
        required_programs,
        length,
    )
    if np.any(np.diff(np.sort(pair_matrix, axis=1), axis=1) == 0):
        raise RuntimeError(
            "Balanced shield construction produced a repeated pair."
        )
    programs = [
        ShieldProgram(
            name=f"all_pair_balanced_{program_index:02d}",
            pair_ids=tuple(int(pair_id) for pair_id in pair_row),
            kind="all_pair_balanced",
        )
        for program_index, pair_row in enumerate(pair_matrix)
    ]
    pair_occurrences = np.bincount(
        pair_matrix.reshape(-1),
        minlength=pair_count,
    )
    if (
        np.any(pair_occurrences <= 0)
        or int(np.max(pair_occurrences) - np.min(pair_occurrences)) > 1
    ):
        raise RuntimeError(
            "Balanced shield programs must cover every pair with frequencies "
            "differing by at most one."
        )
    return programs


def _continuous_kernel_for_estimator(
    estimator: RotatingShieldPFEstimator,
    *,
    detector_aperture_samples: int | None = None,
) -> ContinuousKernel:
    """Build a ContinuousKernel matching the estimator."""
    return estimator.continuous_kernel(
        detector_aperture_samples=detector_aperture_samples,
    )


def _weighted_surface_medoid_index(
    positions_xyz: NDArray[np.float64],
    weights: NDArray[np.float64],
    *,
    surface_path_distance: Callable[
        [NDArray[np.float64], NDArray[np.float64]],
        NDArray[np.float64],
    ]
    | None,
    surface_chart_ids: NDArray[np.int64] | None = None,
    surface_uv: NDArray[np.float64] | None = None,
    surface_coordinate_path_distance: Callable[
        [
            NDArray[np.int64],
            NDArray[np.float64],
            NDArray[np.int64],
            NDArray[np.float64],
        ],
        NDArray[np.float64],
    ]
    | None = None,
    row_chunk_size: int = 256,
) -> int:
    """Return the weighted medoid index using intrinsic surface distance."""
    positions = np.asarray(positions_xyz, dtype=float).reshape(-1, 3)
    sample_weights = _normalise_weights(np.asarray(weights, dtype=float).reshape(-1))
    if positions.shape[0] != sample_weights.size or positions.shape[0] == 0:
        raise ValueError("Surface medoid inputs must contain matching samples.")
    if positions.shape[0] == 1:
        return 0
    coordinate_inputs = (
        surface_chart_ids,
        surface_uv,
        surface_coordinate_path_distance,
    )
    coordinates_active = all(value is not None for value in coordinate_inputs)
    if any(value is not None for value in coordinate_inputs) and not coordinates_active:
        raise ValueError(
            "Surface medoids require chart IDs, UV, and the coordinate-distance "
            "function together."
        )
    chart_ids: NDArray[np.int64] | None = None
    uv: NDArray[np.float64] | None = None
    if coordinates_active:
        raw_chart_ids = np.asarray(surface_chart_ids)
        if not np.issubdtype(raw_chart_ids.dtype, np.integer):
            raise TypeError("surface_chart_ids must contain integers.")
        chart_ids = raw_chart_ids.astype(np.int64, copy=False).reshape(-1)
        uv = np.asarray(surface_uv, dtype=np.float64)
        if (
            chart_ids.shape != (positions.shape[0],)
            or uv.shape != (positions.shape[0], 2)
            or np.any(~np.isfinite(uv))
        ):
            raise ValueError(
                "Surface medoid coordinates must match the position samples."
            )
    if (
        isinstance(row_chunk_size, bool)
        or not isinstance(row_chunk_size, (int, np.integer))
        or int(row_chunk_size) <= 0
    ):
        raise ValueError("row_chunk_size must be a positive integer.")
    medoid_costs = np.empty(positions.shape[0], dtype=float)
    chunk_size = int(row_chunk_size)
    for start in range(0, positions.shape[0], chunk_size):
        stop = min(start + chunk_size, positions.shape[0])
        if coordinates_active:
            assert chart_ids is not None
            assert uv is not None
            assert surface_coordinate_path_distance is not None
            distance_matrix = np.asarray(
                surface_coordinate_path_distance(
                    chart_ids[start:stop, None],
                    uv[start:stop, None, :],
                    chart_ids[None, :],
                    uv[None, :, :],
                ),
                dtype=float,
            )
        else:
            left = np.repeat(positions[start:stop], positions.shape[0], axis=0)
            right = np.tile(positions, (stop - start, 1))
            if surface_path_distance is None:
                distance_rows = np.linalg.norm(left - right, axis=1)
            else:
                distance_rows = np.asarray(
                    surface_path_distance(left, right),
                    dtype=float,
                ).reshape(-1)
            distance_matrix = distance_rows.reshape(
                stop - start,
                positions.shape[0],
            )
        if (
            distance_matrix.shape
            != (stop - start, positions.shape[0])
            or np.any(np.isnan(distance_matrix))
            or np.any(distance_matrix < 0.0)
        ):
            raise RuntimeError(
                "Surface medoid calculation returned invalid distances."
            )
        medoid_costs[start:stop] = distance_matrix @ sample_weights
    minimum_cost = float(np.min(medoid_costs))
    tied = np.flatnonzero(
        np.isclose(medoid_costs, minimum_cost, rtol=0.0, atol=1.0e-12)
    )
    return int(tied[np.argmax(sample_weights[tied])])


def _cluster_source_samples(
    isotope: str,
    positions: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    strengths: Sequence[float] | NDArray[np.float64],
    weights: Sequence[float] | NDArray[np.float64],
    *,
    radius_m: float,
    max_modes: int,
    particle_ids: Sequence[int] | None = None,
    isotope_presence_probability: float | None = None,
    surface_path_distance: Callable[
        [NDArray[np.float64], NDArray[np.float64]],
        NDArray[np.float64],
    ]
    | None = None,
    surface_chart_ids: Sequence[int] | NDArray[np.int64] | None = None,
    surface_uv: Sequence[Sequence[float]] | NDArray[np.float64] | None = None,
    surface_coordinate_path_distance: Callable[
        [
            NDArray[np.int64],
            NDArray[np.float64],
            NDArray[np.int64],
            NDArray[np.float64],
        ],
        NDArray[np.float64],
    ]
    | None = None,
) -> list[SignatureMode]:
    """Cluster source samples into marginal-existence surface modes.

    Samples from the same PF particle contribute to a cluster's existence only
    once and may never occupy the same cluster. Thus simultaneous nearby
    sources remain distinct rather than being collapsed into one strong mode.
    Production callers use intrinsic surface-path distances, preventing
    ambient-near points on disconnected or folded surfaces from being merged.
    """
    if len(positions) == 0:
        return []
    pos_arr = np.asarray(positions, dtype=float)
    str_arr = np.asarray(strengths, dtype=float)
    w_arr = np.asarray(weights, dtype=float).reshape(-1)
    if (
        pos_arr.ndim != 2
        or pos_arr.shape[1] != 3
        or str_arr.shape != (pos_arr.shape[0],)
        or w_arr.shape != (pos_arr.shape[0],)
    ):
        raise ValueError(
            "Mode samples must have matching position, strength, and weight."
        )
    if not np.all(np.isfinite(pos_arr)) or not np.all(np.isfinite(str_arr)):
        raise ValueError("Mode sample positions and strengths must be finite.")
    if not np.all(np.isfinite(w_arr)) or np.any(w_arr < 0.0):
        raise ValueError("Mode sample weights must be finite and nonnegative.")
    resolved_radius = float(radius_m)
    if not np.isfinite(resolved_radius) or resolved_radius <= 0.0:
        raise ValueError("radius_m must be finite and positive.")
    if (
        isinstance(max_modes, bool)
        or not isinstance(max_modes, (int, np.integer))
        or int(max_modes) <= 0
    ):
        raise ValueError("max_modes must be a positive integer.")
    # ``max_modes`` is the PF's simultaneous source-slot capacity.  Marginal
    # posterior clustering can legitimately produce more spatial modes than
    # any one particle contains, so it must never be used as a truncation cap.
    _ = int(max_modes)
    coordinate_inputs = (
        surface_chart_ids,
        surface_uv,
        surface_coordinate_path_distance,
    )
    coordinates_active = all(value is not None for value in coordinate_inputs)
    if any(value is not None for value in coordinate_inputs) and not coordinates_active:
        raise ValueError(
            "Surface clustering requires chart IDs, UV, and the coordinate-"
            "distance function together."
        )
    chart_id_arr: NDArray[np.int64] | None = None
    surface_uv_arr: NDArray[np.float64] | None = None
    if coordinates_active:
        raw_chart_ids = np.asarray(surface_chart_ids)
        if not np.issubdtype(raw_chart_ids.dtype, np.integer):
            raise TypeError("surface_chart_ids must contain integers.")
        chart_id_arr = raw_chart_ids.astype(np.int64, copy=False).reshape(-1)
        surface_uv_arr = np.asarray(surface_uv, dtype=np.float64)
        if (
            chart_id_arr.shape != (pos_arr.shape[0],)
            or surface_uv_arr.shape != (pos_arr.shape[0], 2)
            or np.any(chart_id_arr < 0)
            or np.any(~np.isfinite(surface_uv_arr))
            or np.any(surface_uv_arr < 0.0)
            or np.any(surface_uv_arr > 1.0)
        ):
            raise ValueError(
                "Surface chart coordinates must match every source sample."
            )
    if particle_ids is None:
        particle_id_arr = np.arange(pos_arr.shape[0], dtype=np.int64)
    else:
        particle_id_arr = np.asarray(particle_ids, dtype=np.int64).reshape(-1)
        if particle_id_arr.shape != (pos_arr.shape[0],):
            raise ValueError("particle_ids must contain one ID per source sample.")
        if np.any(particle_id_arr < 0):
            raise ValueError("particle_ids must be nonnegative.")
    if float(np.sum(w_arr)) <= 0.0:
        return []
    if isotope_presence_probability is None:
        resolved_presence = validated_probability(
            float(np.sum(w_arr)),
            name="Implicit clustered-isotope presence probability",
        )
    else:
        resolved_presence = validated_probability(
            isotope_presence_probability,
            name="Clustered-isotope presence probability",
        )
    order = np.argsort(w_arr)[::-1]
    clusters: list[list[int]] = []
    centers: list[NDArray[np.float64]] = []
    center_chart_ids: list[int] = []
    center_surface_uv: list[NDArray[np.float64]] = []
    cluster_particle_ids: list[set[int]] = []
    for idx in order:
        pos = pos_arr[int(idx)]
        particle_id = int(particle_id_arr[int(idx)])
        assigned = False
        if centers:
            if coordinates_active:
                assert chart_id_arr is not None
                assert surface_uv_arr is not None
                assert surface_coordinate_path_distance is not None
                center_distances = np.asarray(
                    surface_coordinate_path_distance(
                        np.full(
                            len(centers),
                            chart_id_arr[int(idx)],
                            dtype=np.int64,
                        ),
                        np.broadcast_to(
                            surface_uv_arr[int(idx)],
                            (len(centers), 2),
                        ),
                        np.asarray(center_chart_ids, dtype=np.int64),
                        np.asarray(center_surface_uv, dtype=np.float64),
                    ),
                    dtype=float,
                ).reshape(-1)
            elif surface_path_distance is None:
                center_array = np.asarray(centers, dtype=float).reshape(-1, 3)
                center_distances = np.linalg.norm(
                    center_array - pos[None, :],
                    axis=1,
                )
            else:
                center_array = np.asarray(centers, dtype=float).reshape(-1, 3)
                center_distances = np.asarray(
                    surface_path_distance(
                        np.broadcast_to(pos, center_array.shape),
                        center_array,
                    ),
                    dtype=float,
                ).reshape(-1)
                if (
                    center_distances.shape != (len(centers),)
                    or np.any(np.isnan(center_distances))
                    or np.any(center_distances < 0.0)
                ):
                    raise RuntimeError(
                        "Surface mode clustering returned invalid path distances."
                    )
        else:
            center_distances = np.zeros(0, dtype=float)
        for cluster_idx, distance in enumerate(center_distances):
            if particle_id in cluster_particle_ids[cluster_idx]:
                continue
            if float(distance) <= resolved_radius:
                clusters[cluster_idx].append(int(idx))
                cluster_particle_ids[cluster_idx].add(particle_id)
                assigned = True
                break
        if not assigned:
            clusters.append([int(idx)])
            centers.append(pos.copy())
            if coordinates_active:
                assert chart_id_arr is not None
                assert surface_uv_arr is not None
                center_chart_ids.append(int(chart_id_arr[int(idx)]))
                center_surface_uv.append(surface_uv_arr[int(idx)].copy())
            cluster_particle_ids.append({particle_id})
    modes: list[SignatureMode] = []
    for cluster in clusters:
        cluster_weights = w_arr[cluster]
        cluster_particle_ids = particle_id_arr[cluster]
        particle_order = np.argsort(cluster_particle_ids, kind="stable")
        sorted_particle_ids = cluster_particle_ids[particle_order]
        group_starts = np.flatnonzero(
            np.concatenate(
                (
                    np.asarray([True]),
                    sorted_particle_ids[1:] != sorted_particle_ids[:-1],
                )
            )
        )
        sorted_particle_weights = cluster_weights[particle_order]
        particle_weight_max = np.maximum.reduceat(
            sorted_particle_weights,
            group_starts,
        )
        particle_weight_min = np.minimum.reduceat(
            sorted_particle_weights,
            group_starts,
        )
        if not np.allclose(
            particle_weight_max,
            particle_weight_min,
            rtol=0.0,
            atol=1.0e-15,
        ):
            raise ValueError(
                "All source samples from one PF particle must share its weight."
            )
        cluster_existence = float(np.sum(particle_weight_max))
        if cluster_existence <= 0.0:
            continue
        cluster_existence = validated_probability(
            cluster_existence,
            name="Cluster existence probability",
        )
        representative_local_index = _weighted_surface_medoid_index(
            pos_arr[cluster],
            cluster_weights,
            surface_path_distance=surface_path_distance,
            surface_chart_ids=(
                None if chart_id_arr is None else chart_id_arr[cluster]
            ),
            surface_uv=(
                None if surface_uv_arr is None else surface_uv_arr[cluster]
            ),
            surface_coordinate_path_distance=surface_coordinate_path_distance,
        )
        representative = pos_arr[cluster[representative_local_index]].copy()
        strength = float(
            np.sum(str_arr[cluster] * cluster_weights)
            / cluster_existence
        )
        if not np.isfinite(strength) or strength <= 0.0:
            raise RuntimeError(
                "A supported posterior source mode must have positive strength."
            )
        if coordinates_active:
            assert chart_id_arr is not None
            assert surface_uv_arr is not None
            assert surface_coordinate_path_distance is not None
            representative_sample_index = int(
                cluster[representative_local_index]
            )
            representative_distances = np.asarray(
                surface_coordinate_path_distance(
                    np.full(
                        len(cluster),
                        chart_id_arr[representative_sample_index],
                        dtype=np.int64,
                    ),
                    np.broadcast_to(
                        surface_uv_arr[representative_sample_index],
                        (len(cluster), 2),
                    ),
                    chart_id_arr[cluster],
                    surface_uv_arr[cluster],
                ),
                dtype=float,
            ).reshape(-1)
        elif surface_path_distance is None:
            representative_distances = np.linalg.norm(
                pos_arr[cluster] - representative[None, :],
                axis=1,
            )
        else:
            representative_distances = np.asarray(
                surface_path_distance(
                    np.broadcast_to(representative, (len(cluster), 3)),
                    pos_arr[cluster],
                ),
                dtype=float,
            ).reshape(-1)
        if (
            representative_distances.shape != (len(cluster),)
            or np.any(np.isnan(representative_distances))
            or np.any(representative_distances < 0.0)
        ):
            raise RuntimeError("Surface mode spread returned invalid path distances.")
        spread = float(
            np.sqrt(
                np.average(
                    representative_distances * representative_distances,
                    weights=cluster_weights,
                )
            )
        )
        modes.append(
            SignatureMode(
                isotope=isotope,
                position_xyz=representative.astype(float),
                strength_cps_1m=strength,
                weight=cluster_existence,
                spread_m=spread,
                isotope_presence_probability=resolved_presence,
                surface_chart_id=(
                    None
                    if chart_id_arr is None
                    else int(
                        chart_id_arr[
                            int(cluster[representative_local_index])
                        ]
                    )
                ),
                surface_uv=(
                    None
                    if surface_uv_arr is None
                    else tuple(
                        float(value)
                        for value in surface_uv_arr[
                            int(cluster[representative_local_index])
                        ]
                    )
                ),
            )
        )
    modes.sort(key=lambda mode: mode.weight, reverse=True)
    represented_source_mass = float(
        np.sum([float(mode.weight) for mode in modes], dtype=np.float64)
    )
    expected_source_mass = float(np.sum(w_arr, dtype=np.float64))
    if not np.isclose(
        represented_source_mass,
        expected_source_mass,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "Surface-mode clustering failed to preserve posterior expected "
            "source-count mass."
        )
    return modes


def extract_signature_modes(
    estimator: RotatingShieldPFEstimator,
    *,
    max_particles: int | None = None,
    method: str | None = None,
    mode_cluster_radius_m: float = 1.5,
    max_modes_per_isotope: int = 5,
    rng: np.random.Generator | None = None,
    particles_by_isotope: dict[
        str,
        tuple[Sequence[object], NDArray[np.float64]],
    ]
    | None = None,
    joint_particles: JointPlanningParticles | None = None,
) -> dict[str, list[SignatureMode]]:
    """Extract isotope-wise modes while preserving native joint-PF mass.

    Production planning passes ``joint_particles`` so particle/source slots are
    validated and packed with array operations. ``particles_by_isotope`` is
    retained only for small deterministic oracles that expose state objects.
    """
    planning_rng = _planning_rng(rng)
    if particles_by_isotope is not None and joint_particles is not None:
        raise ValueError(
            "Provide either particles_by_isotope or joint_particles, not both."
        )
    if particles_by_isotope is None and joint_particles is None:
        particles = estimator.planning_particles(
            max_particles=max_particles,
            method=method,
            rng=planning_rng,
        )
    else:
        particles = particles_by_isotope
    if joint_particles is not None:
        if tuple(str(value) for value in joint_particles.isotope_order) != tuple(
            str(value) for value in estimator.isotopes
        ):
            raise ValueError(
                "Joint mode snapshot isotope order differs from the estimator."
            )
        joint_weights = _normalise_weights(
            np.asarray(joint_particles.weights_n, dtype=np.float64)
        )
        joint_particle_count = int(joint_weights.size)
        joint_particle_ids_n1 = np.arange(
            joint_particle_count,
            dtype=np.int64,
        )[:, None]
    modes_by_isotope: dict[str, list[SignatureMode]] = {}
    eps = 1e-12
    for isotope in estimator.isotopes:
        if joint_particles is not None:
            isotope_key = str(isotope)
            try:
                packed_positions_nk3 = np.asarray(
                    joint_particles.positions_nk3_by_isotope[isotope_key],
                    dtype=np.float64,
                )
                packed_strengths_nk = np.asarray(
                    joint_particles.strengths_nk_by_isotope[isotope_key],
                    dtype=np.float64,
                )
                packed_mask_nk = np.asarray(
                    joint_particles.source_mask_nk_by_isotope[isotope_key],
                    dtype=bool,
                )
                packed_chart_ids_nk = np.asarray(
                    joint_particles.surface_chart_ids_nk_by_isotope[
                        isotope_key
                    ]
                )
                packed_surface_uv_nk2 = np.asarray(
                    joint_particles.surface_uv_nk2_by_isotope[isotope_key],
                    dtype=np.float64,
                )
            except KeyError as error:
                raise ValueError(
                    "Joint mode snapshot is missing an estimator isotope."
                ) from error
            expected_slots = packed_mask_nk.shape
            if (
                packed_mask_nk.ndim != 2
                or packed_positions_nk3.shape != expected_slots + (3,)
                or packed_strengths_nk.shape != expected_slots
                or packed_chart_ids_nk.shape != expected_slots
                or not np.issubdtype(
                    packed_chart_ids_nk.dtype,
                    np.integer,
                )
                or packed_surface_uv_nk2.shape != expected_slots + (2,)
                or expected_slots[0] != joint_particle_count
                or np.any(~np.isfinite(packed_positions_nk3))
                or np.any(~np.isfinite(packed_strengths_nk))
                or np.any(~np.isfinite(packed_surface_uv_nk2))
                or np.any(packed_strengths_nk[packed_mask_nk] <= 0.0)
                or np.any(packed_strengths_nk[~packed_mask_nk] != 0.0)
                or np.any(packed_chart_ids_nk[packed_mask_nk] < 0)
                or np.any(packed_surface_uv_nk2[packed_mask_nk] < 0.0)
                or np.any(packed_surface_uv_nk2[packed_mask_nk] > 1.0)
            ):
                raise ValueError(
                    "Joint mode snapshot contains invalid packed source arrays."
                )
            positions = packed_positions_nk3[packed_mask_nk]
            strengths = packed_strengths_nk[packed_mask_nk]
            sample_chart_ids = packed_chart_ids_nk[
                packed_mask_nk
            ].astype(np.int64, copy=False)
            sample_surface_uv = packed_surface_uv_nk2[packed_mask_nk]
            broadcast_weights_nk = np.broadcast_to(
                joint_weights[:, None],
                expected_slots,
            )
            sample_weights = broadcast_weights_nk[packed_mask_nk]
            broadcast_particle_ids_nk = np.broadcast_to(
                joint_particle_ids_n1,
                expected_slots,
            )
            sample_particle_ids = broadcast_particle_ids_nk[packed_mask_nk]
            isotope_presence_probability = float(
                np.sum(
                    joint_weights[np.any(packed_mask_nk, axis=1)],
                    dtype=np.float64,
                )
            )
        else:
            positions_list: list[NDArray[np.float64]] = []
            strengths_list: list[float] = []
            sample_weights_list: list[float] = []
            sample_particle_ids_list: list[int] = []
            sample_chart_ids_list: list[int] = []
            sample_surface_uv_list: list[NDArray[np.float64]] = []
            isotope_presence_probability = 0.0
            if particles is None or isotope not in particles:
                positions = np.zeros((0, 3), dtype=np.float64)
                strengths = np.zeros(0, dtype=np.float64)
                sample_weights = np.zeros(0, dtype=np.float64)
                sample_particle_ids = np.zeros(0, dtype=np.int64)
                sample_chart_ids = None
                sample_surface_uv = None
                modes_by_isotope[isotope] = []
                continue
            states, weights = particles[isotope]
            norm_weights = _normalise_weights(np.asarray(weights, dtype=float))
            for particle_index, (state, particle_weight) in enumerate(
                zip(states, norm_weights)
            ):
                num_sources = int(state.num_sources)
                if num_sources <= 0:
                    continue
                isotope_presence_probability += float(particle_weight)
                state_strengths = np.asarray(
                    state.strengths[:num_sources],
                    dtype=float,
                )
                if (
                    state_strengths.shape != (num_sources,)
                    or np.any(~np.isfinite(state_strengths))
                    or np.any(state_strengths <= 0.0)
                ):
                    raise ValueError(
                        "A positive-cardinality PF state must contain one "
                        "finite positive strength per source."
                    )
                total_strength = float(np.sum(state_strengths))
                if total_strength <= eps:
                    raise ValueError(
                        "A positive-cardinality PF state must contain positive "
                        "source strength."
                    )
                particle_filter = estimator.filters[str(isotope)]
                state_positions = np.asarray(
                    particle_filter.continuous_state_positions(state),
                    dtype=float,
                )
                if state_positions.shape != (num_sources, 3):
                    raise ValueError(
                        "A positive-cardinality PF state must resolve to one "
                        "continuous surface position per source."
                    )
                state_chart_ids = np.asarray(
                    state.surface_chart_ids,
                )
                state_surface_uv = np.asarray(
                    state.surface_uv,
                    dtype=np.float64,
                )
                if (
                    not np.issubdtype(state_chart_ids.dtype, np.integer)
                    or state_chart_ids.shape != (num_sources,)
                    or state_surface_uv.shape != (num_sources, 2)
                    or np.any(state_chart_ids < 0)
                    or np.any(~np.isfinite(state_surface_uv))
                    or np.any(state_surface_uv < 0.0)
                    or np.any(state_surface_uv > 1.0)
                ):
                    raise ValueError(
                        "A positive-cardinality PF state must retain one "
                        "authoritative chart/UV coordinate per source."
                    )
                for pos, strength, chart_id, source_uv in zip(
                    state_positions,
                    state_strengths,
                    state_chart_ids,
                    state_surface_uv,
                ):
                    positions_list.append(np.asarray(pos, dtype=float))
                    strengths_list.append(float(strength))
                    sample_weights_list.append(float(particle_weight))
                    sample_particle_ids_list.append(int(particle_index))
                    sample_chart_ids_list.append(int(chart_id))
                    sample_surface_uv_list.append(
                        np.asarray(source_uv, dtype=np.float64)
                    )
            positions = np.asarray(positions_list, dtype=np.float64).reshape(-1, 3)
            strengths = np.asarray(strengths_list, dtype=np.float64)
            sample_weights = np.asarray(sample_weights_list, dtype=np.float64)
            sample_particle_ids = np.asarray(
                sample_particle_ids_list,
                dtype=np.int64,
            )
            sample_chart_ids = np.asarray(
                sample_chart_ids_list,
                dtype=np.int64,
            )
            sample_surface_uv = np.asarray(
                sample_surface_uv_list,
                dtype=np.float64,
            ).reshape(-1, 2)
        coordinate_distance = None
        if isinstance(estimator, RotatingShieldPFEstimator):
            atlas = estimator.filters[
                str(isotope)
            ]._structural_rj_surface_atlas
            if atlas is None:
                raise RuntimeError(
                    "Production mode extraction requires a continuous surface "
                    "atlas."
                )
            if sample_chart_ids is None or sample_surface_uv is None:
                raise RuntimeError(
                    "Production mode extraction requires authoritative chart "
                    "coordinates for every source sample."
                )
            decoded_positions = np.asarray(
                atlas.positions_xyz(
                    sample_chart_ids,
                    sample_surface_uv,
                ),
                dtype=np.float64,
            ).reshape(-1, 3)
            if (
                decoded_positions.shape != positions.shape
                or not np.allclose(
                    decoded_positions,
                    positions,
                    rtol=0.0,
                    atol=1.0e-10,
                )
            ):
                raise RuntimeError(
                    "Planner XYZ positions differ from their authoritative "
                    "continuous surface chart coordinates."
                )
            coordinate_distance = (
                atlas.surface_coordinate_path_distance_upper_bound_m
            )
        modes = _cluster_source_samples(
            isotope,
            positions,
            strengths,
            sample_weights,
            radius_m=mode_cluster_radius_m,
            max_modes=max_modes_per_isotope,
            particle_ids=np.asarray(sample_particle_ids, dtype=np.int64),
            isotope_presence_probability=isotope_presence_probability,
            surface_path_distance=(
                None
                if not isinstance(estimator, RotatingShieldPFEstimator)
                else estimator.filters[
                    str(isotope)
                ]._structural_rj_surface_atlas.surface_path_distance_upper_bound_m
            ),
            surface_chart_ids=(
                None if coordinate_distance is None else sample_chart_ids
            ),
            surface_uv=(
                None if coordinate_distance is None else sample_surface_uv
            ),
            surface_coordinate_path_distance=coordinate_distance,
        )
        modes_by_isotope[isotope] = modes
    return modes_by_isotope


def _official_signature_modes(
    estimator: RotatingShieldPFEstimator,
    *,
    max_modes_per_isotope: int,
) -> tuple[dict[str, list[SignatureMode]], dict[str, object]]:
    """Return the official point projection for read-only consistency checks.

    Candidate geometry and exact EIG use the full aligned joint posterior with
    unconditional mass. These joint-MAP modes are diagnostic only and must
    never restrict candidate generation or erase the K=0 posterior mass.
    """
    point_estimates = estimator.posterior_point_estimate()
    isotope_order_getter = getattr(estimator, "joint_isotope_order", None)
    isotope_order = (
        tuple(str(value) for value in isotope_order_getter())
        if callable(isotope_order_getter)
        else tuple(sorted(str(value) for value in estimator.isotopes))
    )
    if set(point_estimates) != set(isotope_order):
        raise RuntimeError(
            "Official PF point estimates do not match planner isotopes."
        )
    modes_by_isotope: dict[str, list[SignatureMode]] = {}
    cardinality_vector: list[int] = []
    stratum_masses: list[float] = []
    medoids_by_isotope: dict[str, list[list[float]]] = {}
    for isotope in isotope_order:
        point_estimate = point_estimates[isotope]
        raw_map_cardinality = point_estimate.map_cardinality
        if isinstance(raw_map_cardinality, (bool, np.bool_)) or not isinstance(
            raw_map_cardinality,
            (int, np.integer),
        ):
            raise ValueError("Official PF cardinality must be an integer.")
        map_cardinality = int(raw_map_cardinality)
        if map_cardinality < 0:
            raise ValueError("Official PF cardinality cannot be negative.")
        if map_cardinality > int(max_modes_per_isotope):
            raise ValueError(
                "Official PF cardinality exceeds planner mode capacity."
            )
        if len(point_estimate.modes) != map_cardinality:
            raise RuntimeError(
                "Official PF mode count differs from its MAP cardinality."
            )
        cardinality_vector.append(map_cardinality)
        selected_mass = validated_probability(
            point_estimate.selected_stratum_mass,
            name=f"Official joint-MAP stratum mass[{isotope}]",
        )
        stratum_masses.append(selected_mass)
        distribution: dict[int, object] = {}
        for raw_cardinality, mass in (
            point_estimate.cardinality_distribution.items()
        ):
            if isinstance(raw_cardinality, (bool, np.bool_)) or not isinstance(
                raw_cardinality,
                (int, np.integer),
            ):
                raise ValueError(
                    "Official PF cardinality-distribution keys must be integers."
                )
            cardinality = int(raw_cardinality)
            if cardinality < 0:
                raise ValueError(
                    "Official PF cardinality-distribution keys must be "
                    "nonnegative."
                )
            distribution[cardinality] = mass
        distribution_values = validated_probability_distribution(
            [
                distribution[cardinality]
                for cardinality in sorted(distribution)
            ],
            name=f"Official cardinality distribution[{isotope}]",
        )
        distribution_keys = sorted(distribution)
        presence_probability = validated_probability(
            float(
                np.sum(
                    distribution_values[
                        np.asarray(distribution_keys, dtype=np.int64) > 0
                    ]
                )
            ),
            name=f"Official isotope presence probability[{isotope}]",
        )
        isotope_modes: list[SignatureMode] = []
        medoid_rows: list[list[float]] = []
        for mode in point_estimate.modes:
            medoid = np.asarray(
                mode.position_medoid_xyz,
                dtype=np.float64,
            ).reshape(3)
            if np.any(~np.isfinite(medoid)):
                raise ValueError("Official PF surface medoid must be finite.")
            strength = float(mode.strength_representative_cps_1m)
            if not np.isfinite(strength) or strength <= 0.0:
                raise ValueError(
                    "Official PF source strength must be finite and positive."
                )
            mode_mass = validated_probability(
                mode.posterior_mass,
                name=f"Official source-mode mass[{isotope}]",
            )
            if not np.isclose(
                mode_mass,
                selected_mass,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise RuntimeError(
                    "Official source mode and joint-MAP stratum masses differ."
                )
            surface_radius = mode.credible_surface_path_radius_95_m
            spread = (
                float(mode.credible_radius_95_m)
                if surface_radius is None
                else float(surface_radius)
            )
            if not np.isfinite(spread) or spread < 0.0:
                raise ValueError("Official PF source spread must be nonnegative.")
            isotope_modes.append(
                SignatureMode(
                    isotope=isotope,
                    position_xyz=medoid.copy(),
                    strength_cps_1m=strength,
                    weight=mode_mass,
                    spread_m=spread,
                    isotope_presence_probability=presence_probability,
                    surface_chart_id=(
                        None
                        if mode.surface_chart_id is None
                        else int(mode.surface_chart_id)
                    ),
                    surface_uv=(
                        None
                        if mode.surface_uv is None
                        else tuple(
                            float(value) for value in mode.surface_uv
                        )
                    ),
                )
            )
            medoid_rows.append([float(value) for value in medoid])
        modes_by_isotope[isotope] = isotope_modes
        medoids_by_isotope[isotope] = medoid_rows
    if stratum_masses and not np.allclose(
        stratum_masses,
        stratum_masses[:1],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "Official isotope reports do not share one joint-MAP stratum mass."
        )
    joint_distribution_getter = getattr(
        estimator,
        "posterior_joint_cardinality_distribution",
        None,
    )
    joint_distribution: dict[tuple[int, ...], float] | None = None
    if callable(joint_distribution_getter):
        raw_joint_distribution = joint_distribution_getter()
        if not isinstance(raw_joint_distribution, Mapping) or not (
            raw_joint_distribution
        ):
            raise RuntimeError("Official joint cardinality distribution is empty.")
        joint_entries: list[tuple[tuple[int, ...], object]] = []
        for raw_vector, mass in raw_joint_distribution.items():
            if (
                not isinstance(raw_vector, tuple)
                or len(raw_vector) != len(isotope_order)
                or any(
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer))
                    or int(value) < 0
                    for value in raw_vector
                )
            ):
                raise ValueError(
                    "Official joint cardinality keys must be nonnegative "
                    "integer tuples matching the isotope order."
                )
            joint_entries.append(
                (
                    tuple(int(value) for value in raw_vector),
                    mass,
                )
            )
        joint_masses = validated_probability_distribution(
            [mass for _, mass in joint_entries],
            name="Official joint cardinality distribution",
        )
        joint_distribution = {
            vector: float(mass)
            for (vector, _), mass in zip(
                joint_entries,
                joint_masses,
                strict=True,
            )
        }
        maximum_mass = max(joint_distribution.values())
        tied_vectors = sorted(
            vector
            for vector, mass in joint_distribution.items()
            if np.isclose(
                mass,
                maximum_mass,
                rtol=0.0,
                atol=1.0e-15,
            )
        )
        official_vector = tuple(cardinality_vector)
        if not tied_vectors or official_vector != tied_vectors[0]:
            raise RuntimeError(
                "Planner cardinality vector differs from the official joint MAP."
            )
        if stratum_masses and not np.isclose(
            stratum_masses[0],
            maximum_mass,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                "Planner stratum mass differs from the official joint MAP."
            )
    diagnostics: dict[str, object] = {
        "source": "official_joint_map_posterior_projection",
        "isotope_order": list(isotope_order),
        "joint_map_cardinality_vector": list(cardinality_vector),
        "joint_map_stratum_mass": (
            float(stratum_masses[0]) if stratum_masses else 0.0
        ),
        "position_representative": "common_joint_particle_surface_medoid",
        "medoids_by_isotope": medoids_by_isotope,
        "verified_against_joint_cardinality_distribution": bool(
            joint_distribution is not None
        ),
    }
    return modes_by_isotope, diagnostics


def _free_space_mask_batch(
    map_api: object | None,
    points_xyz: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Return free-space flags, preferring the map's batched runtime path."""
    points = np.asarray(points_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must be shape (N, 3).")
    if map_api is None:
        return np.ones(points.shape[0], dtype=bool)
    for attr in ("is_free_batch", "is_free_space_batch"):
        function = getattr(map_api, attr, None)
        if not callable(function):
            continue
        mask = np.asarray(function(points), dtype=bool).reshape(-1)
        if mask.size != points.shape[0]:
            raise ValueError("Batched free-space checker returned the wrong length.")
        return mask
    raise TypeError(
        "Production planning maps must provide is_free_batch or "
        "is_free_space_batch; unknown workspace APIs cannot be treated as free."
    )


def _cell_centers_batch(
    map_api: object,
    cells_xy: NDArray[np.int64] | Sequence[Sequence[int]],
    z_value: float,
) -> NDArray[np.float64]:
    """Return world-space centers for an integer map-cell batch."""
    raw_cells = np.asarray(cells_xy)
    if raw_cells.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if (
        raw_cells.ndim != 2
        or raw_cells.shape[1] != 2
        or not np.issubdtype(raw_cells.dtype, np.integer)
        or np.any(raw_cells < 0)
    ):
        raise ValueError("Map cells must be a nonnegative N x 2 integer array.")
    cells = raw_cells.astype(np.int64, copy=False)
    if (
        isinstance(z_value, (bool, np.bool_))
        or not isinstance(
            z_value,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(float(z_value))
    ):
        raise ValueError("Map cell-center height must be finite.")
    center_batch = getattr(map_api, "cell_centers_batch", None)
    if callable(center_batch):
        xy_centers = np.asarray(center_batch(cells), dtype=np.float64)
    else:
        if not hasattr(map_api, "origin") or not hasattr(map_api, "cell_size"):
            raise TypeError(
                "A planning grid without cell_centers_batch must define "
                "origin and cell_size for vectorized center construction."
            )
        origin = np.asarray(map_api.origin, dtype=np.float64)
        cell_size = getattr(map_api, "cell_size")
        if (
            origin.shape != (2,)
            or np.any(~np.isfinite(origin))
            or isinstance(cell_size, (bool, np.bool_))
            or not isinstance(
                cell_size,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(float(cell_size))
            or float(cell_size) <= 0.0
        ):
            raise ValueError("Map cell-center geometry is invalid.")
        xy_centers = (
            origin[None, :]
            + (cells.astype(np.float64) + 0.5) * float(cell_size)
        )
    if (
        xy_centers.shape != (cells.shape[0], 2)
        or np.any(~np.isfinite(xy_centers))
    ):
        raise ValueError(
            "cell_centers_batch must return one finite xy center per cell."
        )
    return np.column_stack(
        (
            xy_centers,
            np.full(cells.shape[0], float(z_value), dtype=np.float64),
        )
    )


def _bounds_filter(
    points: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]] | None,
    map_api: object | None,
) -> NDArray[np.float64]:
    """Filter a candidate batch by bounds and traversability."""
    point_array = np.asarray(points, dtype=float)
    if point_array.size == 0:
        return np.zeros((0, 3), dtype=float)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("points must be shape (N, 3).")
    if np.any(~np.isfinite(point_array)):
        raise ValueError("Candidate points must contain finite coordinates.")
    mask = np.ones(point_array.shape[0], dtype=bool)
    if bounds_xyz is None:
        lo = None
        hi = None
    else:
        lo = np.asarray(bounds_xyz[0], dtype=float)
        hi = np.asarray(bounds_xyz[1], dtype=float)
        if (
            lo.shape != (3,)
            or hi.shape != (3,)
            or np.any(~np.isfinite(lo))
            or np.any(~np.isfinite(hi))
            or np.any(hi < lo)
        ):
            raise ValueError("bounds_xyz must contain two (3,) arrays.")
        mask &= np.all((point_array >= lo) & (point_array <= hi), axis=1)
    if not np.any(mask):
        return np.zeros((0, 3), dtype=float)
    bounded = point_array[mask]
    return bounded[_free_space_mask_batch(map_api, bounded)]


def _dedupe_points(
    points: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    *,
    decimals: int = 3,
) -> NDArray[np.float64]:
    """Return unique points while preserving first occurrence order."""
    point_array = np.asarray(points, dtype=float)
    if point_array.size == 0:
        return np.zeros((0, 3), dtype=float)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("points must be shape (N, 3).")
    rounded = np.round(point_array, int(decimals))
    _, first_indices = np.unique(rounded, axis=0, return_index=True)
    return point_array[np.sort(first_indices)].astype(float)


def _bearing_angle_xy(source: NDArray[np.float64], pose: NDArray[np.float64]) -> float:
    """Return the planar bearing angle from source to pose."""
    delta = np.asarray(pose[:2], dtype=float) - np.asarray(source[:2], dtype=float)
    return float(np.arctan2(delta[1], delta[0]))


def _angle_distance_rad(left: float, right: float) -> float:
    """Return wrapped absolute angular distance in radians."""
    return float(abs(np.arctan2(np.sin(left - right), np.cos(left - right))))


def augment_candidate_stations(
    candidate_poses_xyz: NDArray[np.float64],
    *,
    modes_by_isotope: dict[str, list[SignatureMode]],
    current_pose_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    map_api: object | None,
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]] | None,
    config: DSSPPConfig,
    continuous_height_bounds_m: tuple[float, float] | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Add posterior-ring, occlusion-boundary, and cross-bearing candidates."""
    planning_rng = _planning_rng(rng)
    base = np.asarray(candidate_poses_xyz, dtype=float)
    current_pose = np.asarray(current_pose_xyz, dtype=np.float64)
    if (
        base.ndim != 2
        or base.shape[1] != 3
        or np.any(~np.isfinite(base))
    ):
        raise ValueError("candidate_poses_xyz must be shape (N, 3).")
    if current_pose.shape != (3,) or np.any(~np.isfinite(current_pose)):
        raise ValueError("current_pose_xyz must be a finite three-vector.")
    z_value = float(current_pose[2])
    generated_batches: list[NDArray[np.float64]] = [base.copy()]
    all_modes = [
        mode
        for modes in modes_by_isotope.values()
        for mode in modes
        if mode.weight > 0.0
    ]
    all_modes.sort(key=lambda mode: mode.weight, reverse=True)
    augmentation_capacity = int(config.max_augmented_candidates)
    if len(all_modes) > augmentation_capacity:
        raise ValueError(
            "Posterior geometry contains more material surface modes than the "
            "explicit candidate-augmentation capacity "
            f"({len(all_modes)} > {augmentation_capacity}); increase "
            "max_augmented_candidates instead of silently dropping modes."
        )
    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        num=int(config.ring_angles),
        endpoint=False,
    )
    mode_positions = (
        np.vstack(
            [
                np.asarray(mode.position_xyz, dtype=np.float64).reshape(3)
                for mode in all_modes
            ]
        )
        if all_modes
        else np.zeros((0, 3), dtype=np.float64)
    )
    radii = np.asarray(config.ring_radii_m, dtype=np.float64)
    if mode_positions.size and radii.size:
        unit_xy = np.column_stack((np.cos(angles), np.sin(angles)))
        ring_xy_by_mode = (
            mode_positions[:, None, None, :2]
            + radii[None, :, None, None]
            * unit_xy[None, None, :, :]
        )
        # Interleave modes before applying the explicit augmentation budget.
        # Consequently every material mode contributes one proposal before a
        # second proposal is taken from any mode.
        ring_xy = np.transpose(
            ring_xy_by_mode,
            (1, 2, 0, 3),
        ).reshape(-1, 2)
        generated_batches.append(
            np.column_stack(
                (
                    ring_xy,
                    np.full(ring_xy.shape[0], z_value, dtype=np.float64),
                )
            )
        )
    cells = getattr(map_api, "traversable_cells", None)
    if cells is None and hasattr(map_api, "blocked_cells"):
        raw_blocked = np.asarray(tuple(getattr(map_api, "blocked_cells")))
        grid_shape = np.asarray(getattr(map_api, "grid_shape", (0, 0)))
        if raw_blocked.size:
            if (
                raw_blocked.ndim != 2
                or raw_blocked.shape[1] != 2
                or not np.issubdtype(raw_blocked.dtype, np.integer)
                or grid_shape.shape != (2,)
                or not np.issubdtype(grid_shape.dtype, np.integer)
                or np.any(grid_shape <= 0)
            ):
                raise ValueError(
                    "blocked_cells and grid_shape must define a valid "
                    "integer planning grid."
                )
            blocked = raw_blocked.astype(np.int64, copy=False)
            neighbor_offsets = np.asarray(
                ((-1, 0), (1, 0), (0, -1), (0, 1)),
                dtype=np.int64,
            )
            neighbors = (
                blocked[:, None, :] + neighbor_offsets[None, :, :]
            ).reshape(-1, 2)
            in_bounds = np.all(
                (neighbors >= 0) & (neighbors < grid_shape[None, :]),
                axis=1,
            )
            neighbors = np.unique(neighbors[in_bounds], axis=0)
            grid_width = int(grid_shape[1])
            blocked_ids = blocked[:, 0] * grid_width + blocked[:, 1]
            neighbor_ids = neighbors[:, 0] * grid_width + neighbors[:, 1]
            cells = neighbors[~np.isin(neighbor_ids, blocked_ids)]
        else:
            cells = np.zeros((0, 2), dtype=np.int64)
    if cells is not None:
        raw_cells = np.asarray(tuple(cells))
        if raw_cells.size:
            boundary_points = _cell_centers_batch(
                map_api,
                raw_cells,
                z_value,
            )
            if mode_positions.size:
                distances = np.linalg.norm(
                    boundary_points - mode_positions[0][None, :],
                    axis=1,
                )
                boundary_points = boundary_points[
                    np.argsort(distances, kind="stable")
                ]
            generated_batches.append(
                boundary_points[
                    : int(config.max_augmented_candidates) // 2
                ]
            )
    coverage_points = _free_cell_centers(
        map_api,
        z_value=z_value,
        max_cells=int(config.max_augmented_candidates),
        bounds_xyz=bounds_xyz,
    )
    if coverage_points.size:
        visited = _pose_matrix_or_empty(visited_poses_xyz)
        if visited.size:
            distances = np.linalg.norm(
                coverage_points[:, None, :2] - visited[None, :, :2],
                axis=2,
            )
            order = np.argsort(np.min(distances, axis=1))[::-1]
            coverage_points = coverage_points[order]
        generated_batches.append(
            coverage_points[
                : int(config.max_augmented_candidates) // 2
            ].copy()
        )
    visited = _pose_matrix_or_empty(visited_poses_xyz)
    if visited.size and mode_positions.size and radii.size:
        bearing_delta = (
            visited[None, :, :2] - mode_positions[:, None, :2]
        )
        prior_angles = np.arctan2(
            bearing_delta[:, :, 1],
            bearing_delta[:, :, 0],
        )
        bearing_offsets = np.asarray(
            (0.5 * np.pi, -0.5 * np.pi, np.pi),
            dtype=np.float64,
        )
        bearing_angles = (
            prior_angles[:, :, None] + bearing_offsets[None, None, :]
        )
        bearing_unit_xy = np.stack(
            (np.cos(bearing_angles), np.sin(bearing_angles)),
            axis=-1,
        )
        bearing_xy = (
            mode_positions[:, None, None, None, :2]
            + bearing_unit_xy[:, :, :, None, :]
            * radii[None, None, None, :, None]
        ).reshape(-1, 2)
        generated_batches.append(
            np.column_stack(
                (
                    bearing_xy,
                    np.full(
                        bearing_xy.shape[0],
                        z_value,
                        dtype=np.float64,
                    ),
                )
            )
        )
    generated_array = np.concatenate(generated_batches, axis=0)
    if continuous_height_bounds_m is not None:
        lower_z = float(continuous_height_bounds_m[0])
        upper_z = float(continuous_height_bounds_m[1])
        if not np.isfinite(lower_z) or not np.isfinite(upper_z):
            raise ValueError("continuous_height_bounds_m must be finite.")
        if upper_z < lower_z:
            raise ValueError(
                "continuous_height_bounds_m upper bound must be >= lower bound."
            )
        if bounds_xyz is not None:
            bounds_lo = np.asarray(bounds_xyz[0], dtype=float).reshape(3)
            bounds_hi = np.asarray(bounds_xyz[1], dtype=float).reshape(3)
            if lower_z < bounds_lo[2] or upper_z > bounds_hi[2]:
                raise ValueError(
                    "continuous_height_bounds_m must lie within bounds_xyz."
                )
        augmented_count = int(generated_array.shape[0] - base.shape[0])
        if augmented_count > 0:
            generated_array[base.shape[0] :, 2] = sample_low_discrepancy_heights(
                planning_rng,
                (lower_z, upper_z),
                augmented_count,
            )
    filtered = _bounds_filter(generated_array, bounds_xyz, map_api)
    deduped = _dedupe_points(filtered)
    limit = base.shape[0] + augmentation_capacity
    return deduped[:limit]


def _program_pair_id_matrix(
    programs: Sequence[ShieldProgram],
) -> NDArray[np.int64]:
    """Return a padded pair-id matrix for a set of shield programs."""
    if not programs:
        return np.zeros((0, 0), dtype=np.int64)
    pair_rows = tuple(
        np.asarray(program.pair_ids, dtype=np.int64) for program in programs
    )
    lengths = np.fromiter(
        (row.size for row in pair_rows),
        dtype=np.int64,
        count=len(pair_rows),
    )
    max_length = int(np.max(lengths, initial=0))
    if max_length <= 0:
        return np.zeros((len(programs), 0), dtype=np.int64)
    matrix = np.zeros((len(programs), max_length), dtype=np.int64)
    total_values = int(np.sum(lengths))
    flat_values = np.concatenate(pair_rows)
    row_indices = np.repeat(
        np.arange(len(programs), dtype=np.int64),
        lengths,
    )
    starts = np.cumsum(
        np.concatenate(
            (np.zeros(1, dtype=np.int64), lengths[:-1])
        ),
        dtype=np.int64,
    )
    row_starts = np.repeat(starts, lengths)
    column_indices = np.arange(total_values, dtype=np.int64) - row_starts
    matrix[row_indices, column_indices] = flat_values
    return matrix


def _program_view_mask(
    programs: Sequence[ShieldProgram],
    *,
    max_length: int,
) -> NDArray[np.bool_]:
    """Return a mask selecting the physical views in padded programs."""
    if max_length <= 0:
        return np.zeros((len(programs), 0), dtype=bool)
    lengths = np.asarray([len(program.pair_ids) for program in programs], dtype=int)
    return np.arange(max_length, dtype=int)[None, :] < lengths[:, None]


def _finite_sphere_geometric_terms_batched(
    detector_positions: NDArray[np.float64],
    source_positions: NDArray[np.float64],
    *,
    detector_radius_m: float,
) -> NDArray[np.float64]:
    """Return finite-sphere detector geometry for batched positions."""
    detectors = np.asarray(detector_positions, dtype=float)
    sources = np.asarray(source_positions, dtype=float)
    if detectors.ndim != 2 or detectors.shape[1] != 3:
        raise ValueError("detector_positions must be shaped (D, 3).")
    if sources.ndim < 2 or sources.shape[-1] != 3:
        raise ValueError("source_positions must end in a three-vector dimension.")
    source_shape = sources.shape[:-1]
    delta = detectors.reshape(
        (detectors.shape[0],) + (1,) * len(source_shape) + (3,)
    )
    delta = delta - sources.reshape((1,) + source_shape + (3,))
    distance = np.linalg.norm(delta, axis=-1)
    radius = float(detector_radius_m)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("detector_radius_m must be finite and nonnegative.")
    if radius <= 0.0:
        scale = np.zeros_like(distance, dtype=float)
        positive = np.square(distance) > 1.0e-12
        scale[positive] = 1.0 / np.square(distance[positive])
        return scale
    effective_distance = np.maximum(distance, radius)
    ratio = np.clip(
        radius / np.maximum(effective_distance, 1.0e-12),
        0.0,
        1.0,
    )
    fraction = 0.5 * (
        1.0 - np.sqrt(np.maximum(1.0 - np.square(ratio), 0.0))
    )
    reference_distance = max(1.0, radius)
    reference_ratio = min(radius / reference_distance, 1.0)
    reference_fraction = max(
        0.5
        * (
            1.0
            - float(
                np.sqrt(max(1.0 - reference_ratio * reference_ratio, 0.0))
            )
        ),
        1.0e-12,
    )
    scale = fraction / reference_fraction
    return np.where(distance > 1.0e-12, scale, 0.0)


def _information_gain_from_log_likelihood(
    log_likelihood_psn: NDArray[np.float64],
    weights_n: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return program mutual information from sampled particle likelihoods."""
    log_likelihood = np.asarray(log_likelihood_psn, dtype=float)
    weights = _normalise_weights(np.asarray(weights_n, dtype=float))
    if log_likelihood.ndim != 3:
        raise ValueError(
            "log_likelihood_psn must be shaped (program, sample, particle)."
        )
    if weights.shape != (log_likelihood.shape[2],):
        raise ValueError("weights_n must match the particle dimension.")
    if np.any(np.isnan(log_likelihood)) or np.any(
        np.isposinf(log_likelihood)
    ):
        raise ValueError(
            "Program likelihoods may be finite or minus infinity only."
        )
    positive_prior = weights > 0.0
    if not np.any(positive_prior):
        raise ValueError("Program EIG requires positive posterior mass.")
    active_likelihood = log_likelihood[:, :, positive_prior]
    active_weights = weights[positive_prior]
    log_prior = np.log(active_weights)[None, None, :]
    log_joint = active_likelihood + log_prior
    log_evidence = logsumexp(log_joint, axis=2, keepdims=True)
    if np.any(~np.isfinite(log_evidence)):
        raise RuntimeError(
            "A predictive DSS observation is outside every positive-mass "
            "PF state."
        )
    posterior = np.exp(log_joint - log_evidence)
    kl_terms = np.zeros_like(posterior)
    np.multiply(
        posterior,
        active_likelihood - log_evidence,
        out=kl_terms,
        where=posterior > 0.0,
    )
    kl_samples = np.sum(kl_terms, axis=2)
    information_gain = np.mean(kl_samples, axis=1)
    if np.any(~np.isfinite(information_gain)):
        raise ValueError("Program mutual information must be finite.")
    numerical_tolerance = 1.0e-10
    if np.any(information_gain < -numerical_tolerance):
        raise RuntimeError(
            "Program mutual information became materially negative; the "
            "joint likelihood or posterior weights are inconsistent."
        )
    return np.maximum(information_gain, 0.0)


def _finite_sample_information_gain_upper_bound(
    weights_n: NDArray[np.float64],
) -> float:
    """Bound every sampled posterior KL by the smallest positive prior mass.

    The entropy of the prior bounds the *expected* mutual information, but it
    does not bound a finite Monte Carlo average: an unusually diagnostic draw
    from a rare particle can have KL larger than the prior entropy. For any
    posterior supported on the positive-prior particles,
    ``KL(q || p) <= -log(min(p))``. This looser bound is therefore safe for the
    actual finite-sample EIG objective used by adaptive action expansion.
    """
    weights = _normalise_weights(np.asarray(weights_n, dtype=np.float64))
    positive = weights[weights > 0.0]
    if positive.size == 0:
        raise ValueError("Program EIG requires positive posterior mass.")
    return float(-np.log(np.min(positive)))



def _joint_program_action_layout(
    programs_by_pose: Sequence[Sequence[ShieldProgram]],
) -> tuple[
    list[ShieldProgram],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.bool_],
    NDArray[np.int64],
]:
    """Return one flattened action table for all candidate poses."""
    counts = np.asarray(
        [len(programs) for programs in programs_by_pose],
        dtype=np.int64,
    )
    offsets = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)]
    )
    flattened = [
        program
        for programs in programs_by_pose
        for program in programs
    ]
    pose_indices = np.repeat(
        np.arange(len(programs_by_pose), dtype=np.int64),
        counts,
    )
    pair_ids = _program_pair_id_matrix(flattened)
    view_mask = _program_view_mask(
        flattened,
        max_length=int(pair_ids.shape[1]) if pair_ids.ndim == 2 else 0,
    )
    return flattened, pose_indices, pair_ids, view_mask, offsets



def _full_spectrum_joint_program_components(
    estimator: RotatingShieldPFEstimator,
    detector_positions: NDArray[np.float64],
    programs: Sequence[ShieldProgram],
    joint_particles: JointPlanningParticles,
    *,
    live_time_s: float,
    detector_aperture_samples: int,
) -> _JointProgramSpectrumComponents:
    """Build batched source-resolved inputs for the shared spectrum model."""
    detectors = np.asarray(detector_positions, dtype=np.float64)
    if (
        detectors.ndim != 2
        or detectors.shape[1] != 3
        or np.any(~np.isfinite(detectors))
        or len(programs) != int(detectors.shape[0])
        or not programs
    ):
        raise ValueError(
            "Full-spectrum DSS actions require one finite detector position "
            "per nonempty shield program."
        )
    view_count = len(programs[0].pair_ids)
    if view_count <= 0 or any(
        len(program.pair_ids) != view_count for program in programs
    ):
        raise ValueError(
            "One full-spectrum DSS batch requires equal nonzero view counts."
        )
    resolved_live_time = float(live_time_s)
    if not np.isfinite(resolved_live_time) or resolved_live_time <= 0.0:
        raise ValueError("DSS live_time_s must be finite and positive.")
    model = validate_full_spectrum_model(
        estimator.full_spectrum_generative_model
    )
    isotope_order = tuple(str(value) for value in joint_particles.isotope_order)
    if isotope_order != tuple(sorted(str(value) for value in estimator.isotopes)):
        raise ValueError(
            "Joint planning isotope order must equal the estimator order."
        )
    particle_weights = _normalise_weights(
        np.asarray(joint_particles.weights_n, dtype=np.float64)
    )
    particle_count = int(particle_weights.size)
    line_identity = tuple(model.line_identity)
    line_count = len(line_identity)
    feature_order = tuple(str(value) for value in model.transport_feature_order)
    if feature_order != ("tau_fe", "tau_pb", "tau_obstacle", "distance_m"):
        raise ValueError("DSS and PF transport feature orders differ.")
    slot_counts = {
        isotope: int(
            np.asarray(
                joint_particles.strengths_nk_by_isotope[isotope],
                dtype=np.float64,
            ).shape[1]
        )
        for isotope in isotope_order
    }
    source_slot_count = int(sum(slot_counts.values()))
    action_count = int(detectors.shape[0])
    flattened_view_count = action_count * view_count
    total_flat = np.zeros(
        (
            flattened_view_count,
            particle_count,
            source_slot_count,
            line_count,
        ),
        dtype=np.float64,
    )
    uncollided_flat = np.zeros_like(total_flat)
    features_flat = np.zeros(
        total_flat.shape + (len(feature_order),),
        dtype=np.float64,
    )
    pair_ids = _program_pair_id_matrix(programs)
    orientation_count = int(estimator.num_orientations)
    if (
        pair_ids.shape != (action_count, view_count)
        or np.any(pair_ids < 0)
        or np.any(pair_ids >= orientation_count**2)
    ):
        raise ValueError("DSS shield program contains an invalid pair id.")
    fe_indices = (pair_ids.reshape(-1) // orientation_count).astype(
        np.int64,
        copy=False,
    )
    pb_indices = (pair_ids.reshape(-1) % orientation_count).astype(
        np.int64,
        copy=False,
    )
    active_detectors = np.repeat(detectors, view_count, axis=0)
    if (
        isinstance(detector_aperture_samples, bool)
        or not isinstance(detector_aperture_samples, (int, np.integer))
        or int(detector_aperture_samples) < 1
    ):
        raise ValueError("detector_aperture_samples must be a positive integer.")
    kernel = _continuous_kernel_for_estimator(
        estimator,
        detector_aperture_samples=int(detector_aperture_samples),
    )
    particle_axis = np.arange(particle_count, dtype=np.int64)
    flat_view_axis = np.arange(flattened_view_count, dtype=np.int64)
    slot_offset = 0
    for isotope in isotope_order:
        positions = np.asarray(
            joint_particles.positions_nk3_by_isotope[isotope],
            dtype=np.float64,
        )
        raw_chart_ids = np.asarray(
            joint_particles.surface_chart_ids_nk_by_isotope[isotope],
        )
        surface_uv = np.asarray(
            joint_particles.surface_uv_nk2_by_isotope[isotope],
            dtype=np.float64,
        )
        strengths = np.asarray(
            joint_particles.strengths_nk_by_isotope[isotope],
            dtype=np.float64,
        )
        source_mask = np.asarray(
            joint_particles.source_mask_nk_by_isotope[isotope],
            dtype=bool,
        )
        slot_count = slot_counts[isotope]
        if (
            positions.shape != (particle_count, slot_count, 3)
            or strengths.shape != (particle_count, slot_count)
            or source_mask.shape != strengths.shape
            or not np.issubdtype(raw_chart_ids.dtype, np.integer)
            or raw_chart_ids.shape != strengths.shape
            or surface_uv.shape != strengths.shape + (2,)
            or np.any(~np.isfinite(positions))
            or np.any(~np.isfinite(surface_uv))
            or np.any(~np.isfinite(strengths))
            or np.any(strengths < 0.0)
            or np.any(strengths[~source_mask] != 0.0)
        ):
            raise ValueError(
                "Joint full-spectrum planning particles contain an invalid "
                f"state for {isotope!r}."
            )
        global_line_indices = np.asarray(
            [
                index
                for index, metadata in enumerate(line_identity)
                if str(metadata["isotope"]) == isotope
            ],
            dtype=np.int64,
        )
        local_line_indices = np.asarray(
            [
                int(line_identity[int(index)]["transport_line_index"])
                for index in global_line_indices
            ],
            dtype=np.int64,
        )
        branching_weights = np.asarray(
            [
                float(line_identity[int(index)]["branching_weight"])
                for index in global_line_indices
            ],
            dtype=np.float64,
        )
        if (
            global_line_indices.size == 0
            or np.any(local_line_indices < 0)
            or np.any(~np.isfinite(branching_weights))
            or np.any(branching_weights <= 0.0)
        ):
            raise RuntimeError(
                f"Full-spectrum model has no valid positive line for {isotope!r}."
            )
        configured_branching = kernel.line_branching_weights(
            isotope,
            local_line_indices,
        )
        if not np.allclose(
            configured_branching,
            branching_weights,
            rtol=1.0e-12,
            atol=1.0e-15,
        ):
            raise RuntimeError(
                "DSS, PF, and spectrum-model branching weights differ for "
                f"{isotope!r}."
            )
        if slot_count == 0:
            continue
        chart_ids = np.asarray(raw_chart_ids, dtype=np.int64)
        transport_positions = positions.copy()
        if np.any(source_mask):
            transport_positions[source_mask] = (
                estimator.surface_transport_positions(
                    isotope,
                    positions[source_mask],
                    chart_ids[source_mask],
                    surface_uv[source_mask],
                )
            )
        components = (
            kernel.line_transport_components_selected_pairs_for_detectors(
                isotope=isotope,
                detector_positions=active_detectors,
                sources=transport_positions.reshape(
                    particle_count * slot_count,
                    3,
                ),
                fe_indices=fe_indices,
                pb_indices=pb_indices,
                positive_line_indices=local_line_indices,
            )
        )
        expected_local_shape = (
            flattened_view_count,
            particle_count,
            slot_count,
            int(global_line_indices.size),
        )

        def _local_component(field_name: str) -> NDArray[np.float64]:
            """Return one validated reshaped physical component."""
            values = np.asarray(
                getattr(components, field_name),
                dtype=np.float64,
            ).reshape(expected_local_shape)
            if np.any(~np.isfinite(values)) or np.any(values < 0.0):
                raise RuntimeError(
                    f"Full-spectrum component {field_name!r} is invalid."
                )
            return values

        total_local = _local_component("total_kernel")
        uncollided_local = _local_component("uncollided_kernel")
        tau_fe = _local_component("tau_fe")
        tau_pb = _local_component("tau_pb")
        tau_obstacle = _local_component("tau_obstacle")
        distance_m = _local_component("distance_m")
        source_scale = (
            strengths[None, :, :, None]
            * source_mask[None, :, :, None]
            * branching_weights[None, None, None, :]
        )
        total_local *= source_scale
        uncollided_local *= source_scale
        local_features = np.stack(
            (tau_fe, tau_pb, tau_obstacle, distance_m),
            axis=-1,
        )
        source_slots = np.arange(
            slot_offset,
            slot_offset + slot_count,
            dtype=np.int64,
        )
        target = np.ix_(
            flat_view_axis,
            particle_axis,
            source_slots,
            global_line_indices,
        )
        total_flat[target] = total_local
        uncollided_flat[target] = uncollided_local
        feature_target = np.ix_(
            flat_view_axis,
            particle_axis,
            source_slots,
            global_line_indices,
            np.arange(len(feature_order), dtype=np.int64),
        )
        features_flat[feature_target] = local_features
        slot_offset += slot_count
    output_shape = (
        action_count,
        view_count,
        particle_count,
        source_slot_count,
        line_count,
    )
    total = total_flat.reshape(output_shape).transpose(0, 2, 1, 3, 4)
    uncollided = uncollided_flat.reshape(output_shape).transpose(
        0,
        2,
        1,
        3,
        4,
    )
    features = features_flat.reshape(
        output_shape + (len(feature_order),)
    ).transpose(0, 2, 1, 3, 4, 5)
    if np.any(uncollided > total + 1.0e-10):
        raise RuntimeError(
            "Full-spectrum DSS transport violates uncollided <= total."
        )
    return _JointProgramSpectrumComponents(
        total_pnvsl=np.ascontiguousarray(total),
        uncollided_pnvsl=np.ascontiguousarray(uncollided),
        features_pnvslf=np.ascontiguousarray(features),
        live_times_v=np.full(
            view_count,
            resolved_live_time,
            dtype=np.float64,
        ),
        contract_hash_sha256=str(model.contract_hash_sha256),
    )


def _program_information_proxy_for_poses(
    estimator: RotatingShieldPFEstimator,
    detector_positions: NDArray[np.float64],
    programs: Sequence[ShieldProgram],
    *,
    config: DSSPPConfig,
    joint_particles: JointPlanningParticles,
    rng: np.random.Generator,
    eig_call_seed: int,
    diagnostics: dict[str, object] | None = None,
) -> NDArray[np.float64]:
    """Return reduced-posterior EIG using the exact PF spectrum law."""
    detectors = np.asarray(detector_positions, dtype=np.float64)
    if (
        detectors.ndim != 2
        or detectors.shape[1] != 3
        or np.any(~np.isfinite(detectors))
    ):
        raise ValueError("Proxy detector positions must be finite and shaped Px3.")
    if not programs:
        return np.zeros((detectors.shape[0], 0), dtype=np.float64)
    gains_by_pose = _program_information_gains_for_poses(
        estimator,
        detectors,
        [list(programs) for _ in range(detectors.shape[0])],
        config=config,
        rng=rng,
        joint_particles=joint_particles,
        diagnostics=diagnostics,
        sample_count_override=config.proxy_eig_samples,
        eig_call_seed=eig_call_seed,
        memory_budget_bytes_override=config.proxy_memory_budget_bytes,
    )
    output = np.vstack(gains_by_pose)
    if output.shape != (detectors.shape[0], len(programs)):
        raise RuntimeError("Full-spectrum proxy returned an invalid action layout.")
    if np.any(~np.isfinite(output)) or np.any(output < 0.0):
        raise RuntimeError("Program information ranking proxies are invalid.")
    return output


def _full_spectrum_information_gain(
    estimator: RotatingShieldPFEstimator,
    components: _JointProgramSpectrumComponents,
    particle_weights: NDArray[np.float64],
    *,
    sample_count: int,
    rng: np.random.Generator,
    use_gpu: bool,
    gpu_device: str,
    latent_particle_indices: NDArray[np.int64] | None = None,
    action_seeds_a: NDArray[np.int64] | None = None,
) -> NDArray[np.float64]:
    """Estimate full-spectrum mutual information with bounded action scheduling.

    Transport and cross-likelihood tensors are batched. The generative model
    schedules one canonically seeded predictive draw stream per action. Caller
    batching changes only the execution schedule, not any action's physics,
    likelihood, posterior sample, or random stream.
    """
    model = validate_full_spectrum_model(
        estimator.full_spectrum_generative_model
    )
    if str(model.contract_hash_sha256) != str(
        components.contract_hash_sha256
    ):
        raise RuntimeError("DSS spectrum components use a different model hash.")
    total = np.asarray(components.total_pnvsl, dtype=np.float64)
    uncollided = np.asarray(components.uncollided_pnvsl, dtype=np.float64)
    features = np.asarray(components.features_pnvslf, dtype=np.float64)
    live_times = np.asarray(components.live_times_v, dtype=np.float64)
    if (
        total.ndim != 5
        or uncollided.shape != total.shape
        or features.shape != total.shape + (4,)
        or live_times.shape != (total.shape[2],)
    ):
        raise ValueError("DSS full-spectrum component shapes are inconsistent.")
    action_count, particle_count = total.shape[:2]
    action_seeds = None
    if action_seeds_a is not None:
        action_seeds = np.asarray(action_seeds_a)
        if (
            action_seeds.ndim != 1
            or action_seeds.shape != (action_count,)
            or not np.issubdtype(action_seeds.dtype, np.integer)
        ):
            raise ValueError(
                "action_seeds_a must contain one integer seed per DSS action."
            )
    weights = _normalise_weights(
        np.asarray(particle_weights, dtype=np.float64)
    )
    if weights.shape != (particle_count,):
        raise ValueError("DSS particle weights do not match spectrum states.")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, (int, np.integer))
        or int(sample_count) <= 0
    ):
        raise ValueError("sample_count must be a positive integer.")
    resolved_sample_count = int(sample_count)
    if latent_particle_indices is None:
        latent_indices = rng.choice(
            particle_count,
            size=resolved_sample_count,
            replace=True,
            p=weights,
        )
    else:
        latent_indices = np.asarray(
            latent_particle_indices,
            dtype=np.int64,
        ).reshape(-1)
        if (
            latent_indices.shape != (resolved_sample_count,)
            or np.any(latent_indices < 0)
            or np.any(latent_indices >= particle_count)
        ):
            raise ValueError(
                "DSS latent_particle_indices must contain one valid common "
                "posterior-particle index per predictive sample."
            )
    truth_total = total[:, latent_indices]
    truth_uncollided = uncollided[:, latent_indices]
    truth_features = features[:, latent_indices]
    predictive = np.asarray(
        model.sample_predictive_numpy(
            truth_total,
            truth_uncollided,
            truth_features,
            live_times,
            sample_count=1,
            rng=rng,
            action_seeds_a=action_seeds,
        ),
        dtype=np.float64,
    )
    expected_predictive_shape = (
        action_count,
        resolved_sample_count,
        1,
        int(total.shape[2]),
        int(np.asarray(model.energy_axis_keV).size),
    )
    if predictive.shape != expected_predictive_shape:
        raise RuntimeError(
            "Full-spectrum predictive sampler returned an invalid DSS shape."
        )
    observations = np.ascontiguousarray(predictive[:, :, 0])
    if bool(use_gpu):
        import torch

        cross_likelihood = getattr(model, "cross_log_likelihood_torch", None)
        if not callable(cross_likelihood):
            raise RuntimeError(
                "GPU DSS requires vectorized full-spectrum Torch cross likelihood."
            )
        device = torch.device(str(gpu_device))
        log_likelihood = np.asarray(
            cross_likelihood(
                torch.as_tensor(
                    observations,
                    dtype=torch.float64,
                    device=device,
                ),
                torch.as_tensor(total, dtype=torch.float64, device=device),
                torch.as_tensor(uncollided, dtype=torch.float64, device=device),
                torch.as_tensor(features, dtype=torch.float64, device=device),
                torch.as_tensor(live_times, dtype=torch.float64, device=device),
            )
            .detach()
            .cpu()
            .numpy(),
            dtype=np.float64,
        )
    else:
        cross_likelihood = getattr(model, "cross_log_likelihood_numpy", None)
        if not callable(cross_likelihood):
            raise RuntimeError(
                "DSS requires vectorized full-spectrum cross likelihoods."
            )
        log_likelihood = np.asarray(
            cross_likelihood(
                observations,
                total,
                uncollided,
                features,
                live_times,
            ),
            dtype=np.float64,
        )
    expected_log_shape = (
        action_count,
        resolved_sample_count,
        particle_count,
    )
    if log_likelihood.shape != expected_log_shape:
        raise RuntimeError(
            "Full-spectrum cross likelihood returned an invalid DSS shape."
        )
    return _information_gain_from_log_likelihood(log_likelihood, weights)


def _dss_eig_action_batch_size(
    model: object,
    *,
    action_count: int,
    particle_count: int,
    sample_count: int,
    source_slot_count: int,
    view_count: int,
    line_count: int,
    feature_count: int,
    memory_budget_bytes: int,
    diagnostics: dict[str, int] | None = None,
) -> int:
    """Return a conservative action batch using the model workspace contract."""
    counts = {
        "action_count": action_count,
        "particle_count": particle_count,
        "sample_count": sample_count,
        "source_slot_count": source_slot_count,
        "view_count": view_count,
        "line_count": line_count,
        "feature_count": feature_count,
        "memory_budget_bytes": memory_budget_bytes,
    }
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
        for value in counts.values()
    ):
        raise ValueError("DSS EIG batch dimensions and memory budget must be positive.")
    estimator = getattr(
        model,
        "estimate_cross_likelihood_working_set_bytes",
        None,
    )
    if not callable(estimator):
        raise RuntimeError(
            "The full-spectrum model must publish an exact likelihood "
            "working-set estimate for DSS batching."
        )
    model_working_set = int(
        estimator(
            num_actions=int(action_count),
            num_samples=int(sample_count),
            num_particles=int(particle_count),
            num_isotopes=int(source_slot_count),
            num_views=int(view_count),
            dtype_bytes=np.dtype(np.float64).itemsize,
        )
    )
    if model_working_set <= 0:
        raise RuntimeError(
            "The full-spectrum likelihood returned an invalid working-set estimate."
        )
    energy_axis = np.asarray(getattr(model, "energy_axis_keV", ()))
    if energy_axis.ndim != 1 or energy_axis.size <= 0:
        raise RuntimeError("The full-spectrum model has no valid energy axis.")
    float_bytes = np.dtype(np.float64).itemsize
    transport_per_action = (
        int(particle_count)
        * int(view_count)
        * int(source_slot_count)
        * int(line_count)
        * (2 + int(feature_count))
        * float_bytes
    )
    predictive_per_action = (
        int(sample_count)
        * int(view_count)
        * int(energy_axis.size)
        * float_bytes
    )
    likelihood_output_per_action = (
        int(sample_count) * int(particle_count) * float_bytes
    )
    # Account for NumPy storage, Torch device copies, and allocator overlap.
    persistent_per_action = 3 * (
        transport_per_action
        + predictive_per_action
        + likelihood_output_per_action
    )
    available_for_actions = int(memory_budget_bytes) - model_working_set
    if available_for_actions < persistent_per_action:
        raise MemoryError(
            "DSS EIG memory budget cannot hold the model workspace and one "
            "action without violating the declared limit."
        )
    selected_batch_size = min(
        int(action_count),
        int(available_for_actions // persistent_per_action),
    )
    if diagnostics is not None:
        diagnostics.update(
            {
                "requested_action_count": int(action_count),
                "particle_count": int(particle_count),
                "sample_count": int(sample_count),
                "source_slot_count": int(source_slot_count),
                "view_count": int(view_count),
                "line_count": int(line_count),
                "feature_count": int(feature_count),
                "energy_bin_count": int(energy_axis.size),
                "memory_budget_bytes": int(memory_budget_bytes),
                "model_working_set_bytes": int(model_working_set),
                "transport_per_action_bytes": int(transport_per_action),
                "predictive_per_action_bytes": int(predictive_per_action),
                "likelihood_output_per_action_bytes": int(
                    likelihood_output_per_action
                ),
                "persistent_per_action_bytes": int(persistent_per_action),
                "available_for_actions_bytes": int(available_for_actions),
                "initial_action_batch_size": int(selected_batch_size),
            }
        )
    return int(selected_batch_size)


def _is_dss_eig_memory_error(error: BaseException) -> bool:
    """Return whether an exception represents host or accelerator exhaustion."""
    if isinstance(error, MemoryError):
        return True
    error_type = type(error)
    if (
        error_type.__name__ == "OutOfMemoryError"
        and error_type.__module__.startswith("torch")
    ):
        return True
    return "out of memory" in str(error).lower()


def _release_dss_gpu_cache() -> None:
    """Release unused Torch cache blocks after a recoverable DSS OOM."""
    try:
        import torch
    except ImportError:
        return
    if bool(torch.cuda.is_available()):
        torch.cuda.empty_cache()


def _dss_accelerator_memory_snapshot(
    *,
    use_gpu: bool,
    gpu_device: str,
) -> dict[str, object]:
    """Return read-only accelerator memory diagnostics for exact DSS EIG."""
    if not bool(use_gpu):
        return {
            "enabled": False,
            "device": "cpu",
        }
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "GPU DSS EIG was requested but Torch is unavailable."
        ) from error
    device = torch.device(str(gpu_device))
    if device.type != "cuda":
        return {
            "enabled": True,
            "device": str(device),
            "cuda": False,
        }
    if not bool(torch.cuda.is_available()):
        raise RuntimeError(
            "GPU DSS EIG was requested but CUDA is unavailable."
        )
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "enabled": True,
        "device": str(device),
        "cuda": True,
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def _program_information_gains_for_poses(
    estimator: RotatingShieldPFEstimator,
    detector_positions: NDArray[np.float64],
    programs_by_pose: Sequence[Sequence[ShieldProgram]],
    *,
    config: DSSPPConfig,
    rng: np.random.Generator,
    joint_particles: JointPlanningParticles | None = None,
    diagnostics: dict[str, object] | None = None,
    sample_count_override: int | None = None,
    eig_call_seed: int | None = None,
    memory_budget_bytes_override: int | None = None,
) -> list[NDArray[np.float64]]:
    """Return shared full-spectrum EIG for every candidate/program action."""
    pf_config = estimator.pf_config
    isotopes = tuple(sorted(str(value) for value in estimator.isotopes))
    if not isotopes or any(
        isotope not in estimator.filters for isotope in isotopes
    ):
        raise RuntimeError(
            "Pure PF planning requires one initialized filter per isotope."
        )
    model = validate_full_spectrum_model(
        estimator.full_spectrum_generative_model
    )
    detectors = np.asarray(detector_positions, dtype=np.float64)
    if detectors.size == 0:
        detectors = np.zeros((0, 3), dtype=np.float64)
    if (
        detectors.ndim != 2
        or detectors.shape[1] != 3
        or np.any(~np.isfinite(detectors))
    ):
        raise ValueError("detector_positions must be finite and shaped Px3.")
    if len(programs_by_pose) != detectors.shape[0]:
        raise ValueError("programs_by_pose must match detector_positions.")
    outputs = [
        np.zeros(len(programs), dtype=np.float64)
        for programs in programs_by_pose
    ]
    if detectors.shape[0] == 0:
        return outputs
    if not isinstance(rng, np.random.Generator):
        raise TypeError("DSS EIG requires an explicit numpy random generator.")
    if eig_call_seed is None:
        resolved_eig_call_seed = int(
            rng.integers(
                0,
                np.iinfo(np.int64).max,
                endpoint=False,
                dtype=np.int64,
            )
        )
    elif (
        isinstance(eig_call_seed, bool)
        or not isinstance(eig_call_seed, (int, np.integer))
        or int(eig_call_seed) < 0
    ):
        raise ValueError("eig_call_seed must be a nonnegative integer.")
    else:
        resolved_eig_call_seed = int(eig_call_seed)
    if joint_particles is None:
        joint_particles = estimator.planning_joint_particles(
            max_particles=config.planning_particles,
            method=config.planning_method,
            rng=rng,
        )
    if tuple(joint_particles.isotope_order) != isotopes:
        raise ValueError("Joint planning snapshot isotope order is inconsistent.")
    if int(np.asarray(joint_particles.weights_n).size) < 2:
        return outputs
    configured_samples = (
        pf_config.planning_eig_samples
        if sample_count_override is None
        else sample_count_override
    )
    if (
        isinstance(configured_samples, bool)
        or not isinstance(configured_samples, (int, np.integer))
        or int(configured_samples) <= 0
    ):
        raise ValueError("DSS EIG sample count must be a positive integer.")
    snapshot_index = len(estimator.measurements)
    use_gpu = bool(pf_config.use_gpu)
    gpu_device = str(pf_config.gpu_device)
    accelerator_memory_before = _dss_accelerator_memory_snapshot(
        use_gpu=use_gpu,
        gpu_device=gpu_device,
    )
    flattened_programs, action_pose_indices, pair_ids, view_mask, offsets = (
        _joint_program_action_layout(programs_by_pose)
    )
    if not flattened_programs:
        return outputs
    action_lengths = np.sum(view_mask, axis=1, dtype=np.int64)
    flattened_gains = np.zeros(len(flattened_programs), dtype=np.float64)
    particle_count = int(np.asarray(joint_particles.weights_n).size)
    source_slot_count = int(
        sum(
            np.asarray(
                joint_particles.strengths_nk_by_isotope[isotope]
            ).shape[1]
            for isotope in isotopes
        )
    )
    line_count = len(tuple(model.line_identity))
    feature_count = len(tuple(model.transport_feature_order))
    memory_budget_bytes = (
        256 * 1024 * 1024
        if memory_budget_bytes_override is None
        else memory_budget_bytes_override
    )
    if (
        isinstance(memory_budget_bytes, bool)
        or not isinstance(memory_budget_bytes, (int, np.integer))
        or int(memory_budget_bytes) <= 0
    ):
        raise ValueError("DSS EIG memory budget must be a positive integer.")
    memory_budget_bytes = int(memory_budget_bytes)
    sample_count = int(configured_samples)
    latent_rng = named_random_generator(
        resolved_eig_call_seed,
        "dss_pp",
        "joint_full_spectrum_eig",
        int(snapshot_index),
        "common_latent_particles",
    )
    common_latent_indices = latent_rng.choice(
        particle_count,
        size=sample_count,
        replace=True,
        p=_normalise_weights(
            np.asarray(joint_particles.weights_n, dtype=np.float64)
        ),
    ).astype(np.int64, copy=False)
    action_seeds = np.asarray(
        [
            named_stream_seed(
                resolved_eig_call_seed,
                "dss_pp",
                "joint_full_spectrum_eig",
                int(snapshot_index),
                "canonical_action",
                *(float(value).hex() for value in detectors[int(pose_index)]),
                "pairs",
                *(int(pair_id) for pair_id in program.pair_ids),
            )
            & ((1 << 63) - 1)
            for pose_index, program in zip(
                action_pose_indices,
                flattened_programs,
            )
        ],
        dtype=np.int64,
    )
    memory_contracts: list[dict[str, int]] = []
    attempted_action_batch_sizes: list[int] = []
    successful_action_batch_sizes: list[int] = []
    oom_retry_events: list[dict[str, int]] = []
    for view_count_raw in np.unique(action_lengths):
        view_count = int(view_count_raw)
        selected_actions = np.flatnonzero(action_lengths == view_count)
        if view_count <= 0:
            raise ValueError("Every DSS shield program must contain a view.")
        action_detectors = detectors[
            action_pose_indices[selected_actions]
        ]
        action_pairs = pair_ids[selected_actions, :view_count]
        lexicographic_keys = tuple(
            [
                action_pairs[:, column]
                for column in range(view_count - 1, -1, -1)
            ]
            + [
                action_detectors[:, 2],
                action_detectors[:, 1],
                action_detectors[:, 0],
            ]
        )
        selected_actions = selected_actions[
            np.lexsort(lexicographic_keys)
        ]
        memory_contract: dict[str, int] = {}
        action_batch_size = _dss_eig_action_batch_size(
            model,
            action_count=int(selected_actions.size),
            particle_count=particle_count,
            sample_count=sample_count,
            source_slot_count=max(source_slot_count, 1),
            view_count=view_count,
            line_count=max(line_count, 1),
            feature_count=max(feature_count, 1),
            memory_budget_bytes=memory_budget_bytes,
            diagnostics=memory_contract,
        )
        memory_contracts.append(memory_contract)
        action_start = 0
        while action_start < int(selected_actions.size):
            action_stop = min(
                action_start + action_batch_size,
                int(selected_actions.size),
            )
            action_indices = selected_actions[action_start:action_stop]
            attempted_action_batch_sizes.append(int(action_indices.size))
            action_rng = named_random_generator(
                resolved_eig_call_seed,
                "dss_pp",
                "joint_full_spectrum_eig",
                int(snapshot_index),
                "action_seeded_sampler_fallback",
            )
            try:
                components = _full_spectrum_joint_program_components(
                    estimator,
                    detectors[action_pose_indices[action_indices]],
                    [flattened_programs[int(index)] for index in action_indices],
                    joint_particles,
                    live_time_s=float(config.live_time_s),
                    detector_aperture_samples=int(
                        config.detector_aperture_samples
                    ),
                )
                flattened_gains[action_indices] = (
                    _full_spectrum_information_gain(
                        estimator,
                        components,
                        np.asarray(
                            joint_particles.weights_n,
                            dtype=np.float64,
                        ),
                        sample_count=sample_count,
                        rng=action_rng,
                        use_gpu=use_gpu,
                        gpu_device=gpu_device,
                        latent_particle_indices=common_latent_indices,
                        action_seeds_a=action_seeds[action_indices],
                    )
                )
            except Exception as error:
                if not _is_dss_eig_memory_error(error):
                    raise
                if action_stop - action_start <= 1:
                    raise RuntimeError(
                        "DSS exact EIG exhausted memory for one action even "
                        "after batch reduction."
                    ) from error
                _release_dss_gpu_cache()
                reduced_action_batch_size = max(
                    1,
                    (action_stop - action_start) // 2,
                )
                oom_retry_events.append(
                    {
                        "view_count": int(view_count),
                        "failed_action_batch_size": int(
                            action_stop - action_start
                        ),
                        "retry_action_batch_size": int(
                            reduced_action_batch_size
                        ),
                    }
                )
                action_batch_size = int(reduced_action_batch_size)
                continue
            successful_action_batch_sizes.append(int(action_indices.size))
            action_start = action_stop
    for pose_index in range(int(detectors.shape[0])):
        action_start = int(offsets[pose_index])
        action_stop = int(offsets[pose_index + 1])
        outputs[pose_index] = np.asarray(
            flattened_gains[action_start:action_stop],
            dtype=np.float64,
        )
    if diagnostics is not None:
        diagnostics.update(
            {
                "backend": "torch" if use_gpu else "numpy",
                "gpu_device": str(gpu_device) if use_gpu else "cpu",
                "memory_budget_bytes": int(memory_budget_bytes),
                "accelerator_memory_before": accelerator_memory_before,
                "accelerator_memory_after": _dss_accelerator_memory_snapshot(
                    use_gpu=use_gpu,
                    gpu_device=gpu_device,
                ),
                "memory_contracts": memory_contracts,
                "attempted_action_batch_sizes": attempted_action_batch_sizes,
                "successful_action_batch_sizes": successful_action_batch_sizes,
                "oom_retry_count": int(len(oom_retry_events)),
                "oom_retry_events": oom_retry_events,
                "cpu_fallback_used": False,
            }
        )
    return outputs


def _elevation_pair_indices_and_weights(
    modes: Sequence[SignatureMode],
    mode_weights: NDArray[np.float64],
    *,
    config: DSSPPConfig,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Return mode-pair weights emphasizing vertical ambiguity."""
    mode_count = len(modes)
    if mode_count < 2:
        empty_idx = np.zeros(0, dtype=np.int64)
        return empty_idx, empty_idx, np.zeros(0, dtype=float)
    weights = _normalise_weights(np.asarray(mode_weights, dtype=float))
    if weights.size != mode_count:
        raise ValueError("mode_weights must contain one value per mode.")
    positions = np.vstack(
        [np.asarray(mode.position_xyz, dtype=float).reshape(3) for mode in modes]
    )
    left, right = np.triu_indices(mode_count, k=1)
    z_delta = np.abs(positions[left, 2] - positions[right, 2])
    xy_delta = np.linalg.norm(positions[left, :2] - positions[right, :2], axis=1)
    z_scale = float(config.elevation_pair_z_scale_m)
    xy_scale = float(config.elevation_pair_xy_scale_m)
    z_factor = z_delta / (z_delta + z_scale)
    xy_factor = xy_scale / (xy_delta + xy_scale)
    posterior_factor = np.sqrt(np.maximum(weights[left] * weights[right], 0.0))
    pair_weights = posterior_factor * z_factor * xy_factor
    valid = pair_weights > 0.0
    return (
        left[valid].astype(np.int64, copy=False),
        right[valid].astype(np.int64, copy=False),
        pair_weights[valid].astype(float, copy=False),
    )


def _local_orbit_gains_batch(
    candidate_poses_xyz: NDArray[np.float64],
    modes_by_isotope: dict[str, list[SignatureMode]],
    *,
    config: DSSPPConfig,
) -> NDArray[np.float64]:
    """Return local-orbit gains for many candidate stations."""
    candidates = np.asarray(candidate_poses_xyz, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidate_poses_xyz must be shaped (N, 3).")
    radii = np.asarray(
        [float(radius) for radius in config.ring_radii_m if float(radius) > 0.0],
        dtype=float,
    )
    if radii.size == 0 or candidates.shape[0] == 0:
        return np.zeros(candidates.shape[0], dtype=float)
    modes = [
        mode
        for mode_list in modes_by_isotope.values()
        for mode in mode_list
        if float(mode.weight) > 0.0
    ]
    if not modes:
        return np.zeros(candidates.shape[0], dtype=float)
    mode_positions = np.vstack(
        [np.asarray(mode.position_xyz, dtype=float) for mode in modes]
    )
    mode_weights = _flattened_posterior_mode_weights(modes_by_isotope)
    xy_distances = np.linalg.norm(
        candidates[:, None, :2] - mode_positions[None, :, :2],
        axis=2,
    )
    radial_error = np.min(
        np.abs(xy_distances[:, :, None] - radii[None, None, :]), axis=2
    )
    sigma = float(config.local_orbit_sigma_m)
    radial_gain = np.exp(-0.5 * (radial_error / sigma) ** 2)
    isotope_count = len(modes_by_isotope)
    if isotope_count <= 0:
        raise ValueError("modes_by_isotope must contain configured isotopes.")
    return (
        np.sum(radial_gain * mode_weights.reshape(1, -1), axis=1)
        / float(isotope_count)
    )


def _elevation_condition_gains_batch(
    candidate_poses_xyz: NDArray[np.float64],
    modes_by_isotope: dict[str, list[SignatureMode]],
    *,
    config: DSSPPConfig,
) -> NDArray[np.float64]:
    """Return candidate gains for separating posterior modes by elevation angle."""
    candidates = np.asarray(candidate_poses_xyz, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidate_poses_xyz must be shaped (N, 3).")
    gains = np.zeros(candidates.shape[0], dtype=float)
    if candidates.shape[0] == 0:
        return gains
    threshold = np.deg2rad(float(config.elevation_angle_threshold_deg))
    isotope_weight_values: list[float] = []
    isotope_gain_rows: list[NDArray[np.float64]] = []
    for modes in modes_by_isotope.values():
        active = [mode for mode in modes if float(mode.weight) > 0.0]
        if len(active) < 2:
            continue
        weights = _normalise_weights(
            np.asarray([float(mode.weight) for mode in active], dtype=float)
        )
        left, right, pair_weights = _elevation_pair_indices_and_weights(
            active,
            weights,
            config=config,
        )
        if left.size == 0:
            continue
        positions = np.vstack(
            [np.asarray(mode.position_xyz, dtype=float).reshape(3) for mode in active]
        )
        vectors = positions[None, :, :] - candidates[:, None, :]
        horizontal = np.linalg.norm(vectors[:, :, :2], axis=2)
        elevation = np.arctan2(vectors[:, :, 2], np.maximum(horizontal, 1.0e-9))
        pair_contrast = np.abs(elevation[:, left] - elevation[:, right])
        pair_scores = np.minimum(pair_contrast / threshold, 1.0)
        row = np.sum(pair_scores * pair_weights.reshape(1, -1), axis=1) / max(
            float(np.sum(pair_weights)),
            1.0e-12,
        )
        isotope_gain_rows.append(row)
        isotope_weight_values.append(
            float(_isotope_presence_probability(active) or 0.0)
        )
    if not isotope_gain_rows:
        return gains
    return _presence_weighted_rows(
        isotope_gain_rows,
        isotope_weight_values,
        population_size=len(modes_by_isotope),
    )


def _static_station_program_score(
    *,
    coverage_norm: float,
    revisit_penalty: float,
    bearing_gain: float,
    frontier_gain: float,
    turn_penalty: float,
    local_orbit_gain: float,
    elevation_condition_gain: float,
    coverage_floor: float,
    config: DSSPPConfig,
) -> float:
    """Return geometry, route, and coverage utility without count reuse."""
    return float(
        float(config.lambda_coverage) * float(coverage_norm)
        + float(config.lambda_bearing_diversity) * float(bearing_gain)
        + float(config.lambda_frontier) * float(frontier_gain)
        + float(config.lambda_local_orbit) * float(local_orbit_gain)
        + float(config.lambda_elevation_condition)
        * float(np.log1p(max(elevation_condition_gain, 0.0)))
        - float(config.eta_revisit) * float(revisit_penalty)
        - float(config.lambda_turn_smoothness) * float(turn_penalty)
        - float(config.coverage_floor_weight)
        * max(0.0, float(coverage_floor) - float(coverage_norm)) ** 2
    )


def _evaluate_pose_index_from_context(
    pose_index_value: int,
    context: Mapping[str, object],
) -> tuple[int, float, list[_PendingDSSPPNode]]:
    """Materialize all program nodes for one already-vectorized station."""
    pose_index = int(pose_index_value)
    candidate_poses = np.asarray(context["candidate_poses"], dtype=float)
    path_lengths = np.asarray(context["path_lengths"], dtype=float)
    programs = cast(Sequence[ShieldProgram], context["programs"])
    config = cast(DSSPPConfig, context["config"])
    coverage_norm = np.asarray(context["coverage_norm"], dtype=float)
    coverage_raw = np.asarray(context["coverage_raw"], dtype=float)
    revisit_penalties = np.asarray(context["revisit_penalties"], dtype=float)
    bearing_gains = np.asarray(context["bearing_gains"], dtype=float)
    frontier_gains = np.asarray(context["frontier_gains"], dtype=float)
    turn_penalties = np.asarray(context["turn_penalties"], dtype=float)
    local_orbit_gains = np.asarray(context["local_orbit_gains"], dtype=float)
    elevation_condition_gains = np.asarray(
        context["elevation_condition_gains"],
        dtype=float,
    )
    coverage_floor = float(context["coverage_floor"])

    local_pending: list[_PendingDSSPPNode] = []
    local_cheap_score = -np.inf
    pose = candidate_poses[pose_index]
    if not np.isfinite(path_lengths[pose_index]):
        return (
            pose_index,
            local_cheap_score,
            local_pending,
        )
    # Every candidate program is compared by the exact joint EIG below.  The
    # static term is deliberately restricted to geometry, route, and coverage
    # so the same prospective counts cannot be scored a second time.
    for program in programs:
        static_score = _static_station_program_score(
            coverage_norm=float(coverage_norm[pose_index]),
            revisit_penalty=float(revisit_penalties[pose_index]),
            bearing_gain=float(bearing_gains[pose_index]),
            frontier_gain=float(frontier_gains[pose_index]),
            turn_penalty=float(turn_penalties[pose_index]),
            local_orbit_gain=float(local_orbit_gains[pose_index]),
            elevation_condition_gain=float(elevation_condition_gains[pose_index]),
            coverage_floor=coverage_floor,
            config=config,
        )
        local_cheap_score = max(float(local_cheap_score), float(static_score))
        local_pending.append(
            _PendingDSSPPNode(
                pose_index=pose_index,
                pose_xyz=pose.copy(),
                program=program,
                static_score=float(static_score),
                coverage_gain=float(coverage_raw[pose_index]),
                revisit_penalty=float(revisit_penalties[pose_index]),
                bearing_diversity_gain=float(bearing_gains[pose_index]),
                frontier_gain=float(frontier_gains[pose_index]),
                turn_penalty=float(turn_penalties[pose_index]),
                local_orbit_gain=float(local_orbit_gains[pose_index]),
                elevation_condition_gain=float(
                    elevation_condition_gains[pose_index]
                ),
            )
        )
    return (
        pose_index,
        local_cheap_score,
        local_pending,
    )


def _materialize_pose_nodes(
    eval_indices: Sequence[int],
    *,
    context: dict[str, object],
) -> list[tuple[int, float, list[_PendingDSSPPNode]]]:
    """Materialize nodes after all numerical candidate terms were batched."""
    return [
        _evaluate_pose_index_from_context(int(index), context)
        for index in eval_indices
    ]


def _node_path_length(
    map_api: object | None,
    start_xyz: NDArray[np.float64],
    goal_xyz: NDArray[np.float64],
) -> float:
    """Return grid path length when possible, otherwise Euclidean distance."""
    start = np.asarray(start_xyz, dtype=float)
    goal = np.asarray(goal_xyz, dtype=float)
    if map_api is None:
        return float(np.linalg.norm(goal - start))
    motion_waypoints = getattr(map_api, "motion_waypoints", None)
    if callable(motion_waypoints):
        start_key = tuple(float(value) for value in start.reshape(3))
        goal_key = tuple(float(value) for value in goal.reshape(3))
        cache_key = ("motion", id(map_api), start_key, goal_key)
        cached = _DSS_PP_PATH_LENGTH_CACHE.get(cache_key)
        if cached is not None:
            return float(cached)
        path = motion_waypoints(start, goal)
        if path is None:
            length = float("inf")
        else:
            points = np.asarray(path, dtype=float)
            if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
                length = float("inf")
            elif points.shape[0] == 1:
                length = 0.0
            else:
                length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        if len(_DSS_PP_PATH_LENGTH_CACHE) >= _DSS_PP_PATH_LENGTH_CACHE_MAX:
            _DSS_PP_PATH_LENGTH_CACHE.clear()
        _DSS_PP_PATH_LENGTH_CACHE[cache_key] = float(length)
        _DSS_PP_PATH_LENGTH_CACHE[("motion", id(map_api), goal_key, start_key)] = float(
            length
        )
        return float(length)
    cell_index = getattr(map_api, "cell_index", None)
    if not callable(cell_index):
        return float(np.linalg.norm(goal - start))
    start_cell = cell_index(start)
    goal_cell = cell_index(goal)
    if start_cell is None or goal_cell is None:
        return float(np.linalg.norm(goal - start))
    start_key = tuple(float(value) for value in start.reshape(3))
    goal_key = tuple(float(value) for value in goal.reshape(3))
    cache_key = (
        id(map_api),
        tuple(start_cell),
        tuple(goal_cell),
        start_key,
        goal_key,
    )
    cached = _DSS_PP_PATH_LENGTH_CACHE.get(cache_key)
    if cached is not None:
        return float(cached)
    length = shortest_grid_path_length(map_api, start, goal, allow_diagonal=True)
    if len(_DSS_PP_PATH_LENGTH_CACHE) >= _DSS_PP_PATH_LENGTH_CACHE_MAX:
        _DSS_PP_PATH_LENGTH_CACHE.clear()
    _DSS_PP_PATH_LENGTH_CACHE[cache_key] = float(length)
    reverse_key = (
        id(map_api),
        tuple(goal_cell),
        tuple(start_cell),
        goal_key,
        start_key,
    )
    _DSS_PP_PATH_LENGTH_CACHE[reverse_key] = float(length)
    return float(length)


def _node_path_lengths_batch(
    map_api: object | None,
    start_xyz: NDArray[np.float64],
    goals_xyz: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return candidate path lengths, preferring a map-native batch API."""
    start = np.asarray(start_xyz, dtype=float).reshape(-1)
    goals = np.asarray(goals_xyz, dtype=float)
    if start.shape != (3,) or np.any(~np.isfinite(start)):
        raise ValueError("start_xyz must be a finite three-vector.")
    if goals.size == 0:
        return np.zeros(0, dtype=float)
    if goals.ndim != 2 or goals.shape[1] != 3:
        raise ValueError("goals_xyz must be shape (N, 3).")
    if np.any(~np.isfinite(goals)):
        raise ValueError("goals_xyz must contain only finite coordinates.")
    if map_api is None:
        return np.linalg.norm(goals - start[None, :], axis=1)
    batch_function = getattr(map_api, "motion_path_lengths_batch", None)
    if callable(batch_function):
        lengths = np.asarray(
            batch_function(start, goals),
            dtype=float,
        ).reshape(-1)
        if lengths.size != goals.shape[0]:
            raise ValueError(
                "motion_path_lengths_batch returned the wrong number of values."
            )
        return lengths
    return np.fromiter(
        (_node_path_length(map_api, start, goal) for goal in goals),
        dtype=float,
        count=goals.shape[0],
    )


def _filter_path_reachable_stations(
    candidate_poses_xyz: NDArray[np.float64],
    *,
    current_pose_xyz: NDArray[np.float64],
    map_api: object | None,
) -> tuple[NDArray[np.float64], int]:
    """Remove station candidates that have no traversable path from the robot."""
    candidates = np.asarray(candidate_poses_xyz, dtype=float)
    if candidates.size == 0:
        return np.zeros((0, 3), dtype=float), 0
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidate_poses_xyz must be shape (N, 3).")
    motion_path_lengths_batch = getattr(
        map_api,
        "motion_path_lengths_batch",
        None,
    )
    motion_reachable_batch = getattr(
        map_api,
        "is_motion_reachable_batch",
        None,
    )
    if callable(motion_path_lengths_batch):
        path_lengths = _node_path_lengths_batch(
            map_api,
            current_pose_xyz,
            candidates,
        )
        reachable = np.isfinite(path_lengths)
    elif callable(motion_reachable_batch):
        reachable = np.asarray(
            motion_reachable_batch(current_pose_xyz, candidates),
            dtype=bool,
        ).reshape(-1)
        if reachable.size != candidates.shape[0]:
            raise ValueError(
                "is_motion_reachable_batch returned the wrong number of flags."
            )
    else:
        reachable = np.isfinite(
            _node_path_lengths_batch(
                map_api,
                current_pose_xyz,
                candidates,
            )
        )
    removed = int(np.count_nonzero(~reachable))
    if not np.any(reachable):
        return np.zeros((0, 3), dtype=float), removed
    return candidates[reachable], removed


def _free_cell_centers(
    map_api: object | None,
    *,
    z_value: float,
    max_cells: int,
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
) -> NDArray[np.float64]:
    """Return free-cell center positions for coverage scoring."""
    if map_api is None:
        return _bounds_cell_centers(
            bounds_xyz,
            z_value=z_value,
            max_cells=max_cells,
        )
    grid_shape = getattr(map_api, "grid_shape", None)
    if grid_shape is None:
        return _bounds_cell_centers(
            bounds_xyz,
            z_value=z_value,
            max_cells=max_cells,
        )
    traversable_cells = getattr(map_api, "traversable_cells", None)
    if traversable_cells is None:
        return np.zeros((0, 3), dtype=float)
    raw_cells = np.asarray(tuple(traversable_cells))
    if raw_cells.size == 0:
        return np.zeros((0, 3), dtype=float)
    if (
        raw_cells.ndim != 2
        or raw_cells.shape[1] != 2
        or not np.issubdtype(raw_cells.dtype, np.integer)
        or np.any(raw_cells < 0)
    ):
        raise ValueError(
            "traversable_cells must be a nonnegative N x 2 integer array."
        )
    cells = raw_cells.astype(np.int64, copy=False)
    max_count = max(0, int(max_cells))
    if max_count > 0 and cells.shape[0] > max_count:
        indices = np.linspace(
            0,
            cells.shape[0] - 1,
            max_count,
            dtype=np.int64,
        )
        cells = cells[indices]
    return _cell_centers_batch(map_api, cells, z_value)


def _bounds_cell_centers(
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]] | None,
    *,
    z_value: float,
    max_cells: int,
) -> NDArray[np.float64]:
    """Return rectangular free-space samples when no traversability map exists."""
    if bounds_xyz is None:
        return np.zeros((0, 3), dtype=float)
    lo = np.asarray(bounds_xyz[0], dtype=float)
    hi = np.asarray(bounds_xyz[1], dtype=float)
    if lo.shape != (3,) or hi.shape != (3,):
        return np.zeros((0, 3), dtype=float)
    span = np.maximum(hi[:2] - lo[:2], 0.0)
    if float(span[0]) <= 0.0 or float(span[1]) <= 0.0:
        return np.zeros((0, 3), dtype=float)
    target = max(4, int(max_cells))
    aspect = float(span[0]) / max(float(span[1]), 1e-12)
    nx = max(2, int(np.sqrt(float(target) * aspect)))
    ny = max(2, int(np.ceil(float(target) / max(nx, 1))))
    if nx * ny > target:
        scale = np.sqrt(float(target) / float(nx * ny))
        nx = max(2, int(np.floor(nx * scale)))
        ny = max(2, int(np.floor(ny * scale)))
    xs = np.linspace(float(lo[0]), float(hi[0]), num=nx)
    ys = np.linspace(float(lo[1]), float(hi[1]), num=ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    zz = np.full(xx.size, float(z_value), dtype=float)
    return np.column_stack([xx.ravel(), yy.ravel(), zz])


def _coverage_gain_fraction(
    *,
    cell_centers_xyz: NDArray[np.float64],
    candidate_pose_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    radius_m: float,
) -> float:
    """Return newly covered three-dimensional surface-support fraction."""
    centers = np.asarray(cell_centers_xyz, dtype=float)
    if centers.size == 0:
        return 0.0
    radius = float(radius_m)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("radius_m must be finite and nonnegative.")
    if radius <= 0.0:
        return 0.0
    candidate = np.asarray(candidate_pose_xyz, dtype=float).reshape(3)
    candidate_dist = np.linalg.norm(centers - candidate[None, :], axis=1)
    candidate_covered = candidate_dist <= radius
    if not np.any(candidate_covered):
        return 0.0
    visited_covered = np.zeros(centers.shape[0], dtype=bool)
    if visited_poses_xyz is not None:
        visited = np.asarray(visited_poses_xyz, dtype=float)
        if visited.ndim == 1 and visited.size == 3:
            visited = visited.reshape(1, 3)
        if visited.ndim == 2 and visited.shape[1] == 3 and visited.size:
            visited_dist = np.linalg.norm(
                centers[:, None, :] - visited[None, :, :],
                axis=2,
            )
            visited_covered = np.min(visited_dist, axis=1) <= radius
    newly_covered = candidate_covered & ~visited_covered
    return float(np.count_nonzero(newly_covered)) / float(centers.shape[0])


def _pose_matrix_or_empty(poses_xyz: NDArray[np.float64] | None) -> NDArray[np.float64]:
    """Return finite poses as an N x 3 array or an empty absent history."""
    if poses_xyz is None:
        return np.zeros((0, 3), dtype=float)
    poses = np.asarray(poses_xyz, dtype=float)
    if poses.ndim == 1 and poses.size == 3:
        poses = poses.reshape(1, 3)
    if poses.ndim == 2 and poses.shape == (0, 3):
        return np.zeros((0, 3), dtype=float)
    if (
        poses.ndim != 2
        or poses.shape[1] != 3
        or np.any(~np.isfinite(poses))
    ):
        raise ValueError("visited_poses_xyz must be finite and shaped (N, 3).")
    return poses


def _coverage_gain_fractions_batch(
    *,
    cell_centers_xyz: NDArray[np.float64],
    candidate_poses_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    radius_m: float,
) -> NDArray[np.float64]:
    """Return area-sampled 3-D surface coverage gains for candidate stations."""
    centers = np.asarray(cell_centers_xyz, dtype=float)
    candidates = np.asarray(candidate_poses_xyz, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidate_poses_xyz must be shaped (N, 3).")
    if centers.size == 0 or candidates.shape[0] == 0:
        return np.zeros(candidates.shape[0], dtype=float)
    radius = float(radius_m)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("radius_m must be finite and nonnegative.")
    if radius <= 0.0:
        return np.zeros(candidates.shape[0], dtype=float)
    visited = _pose_matrix_or_empty(visited_poses_xyz)
    visited_covered = np.zeros(centers.shape[0], dtype=bool)
    if visited.size:
        visited_dist = np.linalg.norm(
            centers[:, None, :] - visited[None, :, :],
            axis=2,
        )
        visited_covered = np.min(visited_dist, axis=1) <= radius
    candidate_dist = np.linalg.norm(
        candidates[:, None, :] - centers[None, :, :],
        axis=2,
    )
    newly_covered = (candidate_dist <= radius) & ~visited_covered.reshape(1, -1)
    return np.count_nonzero(newly_covered, axis=1).astype(float) / float(
        centers.shape[0]
    )


def _response_equivalent_surface_coverage_masks(
    *,
    kernel: ContinuousKernel,
    estimator: RotatingShieldPFEstimator,
    surface_points_xyz: NDArray[np.float64],
    candidate_poses_xyz: NDArray[np.float64],
    reference_radius_m: float,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Return candidate and visited-station surface-coverage masks.

    Coverage is deliberately a shield-independent spatial exploration term:
    both prospective and visited detector stations use the same finite-detector
    distance-plus-obstacle response before Fe/Pb attenuation. Shield-specific
    evidence is evaluated exactly once by the joint full-spectrum EIG. Keeping
    both sides of this coverage state on the same contract prevents a station
    from being repeatedly rewarded as "new" merely because its executed shield
    pair differs from the optimistic candidate calculation.
    """
    surfaces = np.asarray(surface_points_xyz, dtype=float)
    candidates = np.asarray(candidate_poses_xyz, dtype=float)
    if surfaces.ndim != 2 or surfaces.shape[1] != 3:
        raise ValueError("surface_points_xyz must be shaped (S, 3).")
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidate_poses_xyz must be shaped (C, 3).")
    isotope_names = tuple(str(value) for value in estimator.isotopes)
    if not isotope_names:
        raise ValueError("Surface observability requires configured isotopes.")
    radius = float(reference_radius_m)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("reference_radius_m must be finite and positive.")
    detector_radius = float(kernel.detector_radius_m)
    reference = float(
        _finite_sphere_geometric_terms_batched(
            np.zeros((1, 3), dtype=float),
            np.asarray([[radius, 0.0, 0.0]], dtype=float),
            detector_radius_m=detector_radius,
        )[0, 0]
    )
    if not np.isfinite(reference) or reference <= 0.0:
        raise RuntimeError("Surface observability reference response is invalid.")
    if surfaces.shape[0] == 0:
        return (
            np.zeros((candidates.shape[0], 0), dtype=bool),
            np.zeros(0, dtype=bool),
        )

    candidate_pairs = cKDTree(candidates).sparse_distance_matrix(
        cKDTree(surfaces),
        max_distance=float(np.nextafter(radius, np.inf)),
        output_type="coo_matrix",
    )
    candidate_rows = np.asarray(candidate_pairs.row, dtype=np.int64)
    candidate_surface_ids = np.asarray(
        candidate_pairs.col,
        dtype=np.int64,
    )
    candidate_pair_covered = np.ones(
        candidate_rows.size,
        dtype=bool,
    )
    for isotope in isotope_names:
        values = np.asarray(
            kernel.kernel_values_unshielded_for_detector_source_pairs(
                isotope=isotope,
                detector_positions=candidates[candidate_rows],
                sources=surfaces[candidate_surface_ids],
            ),
            dtype=np.float64,
        ).reshape(-1)
        if (
            values.shape != (candidate_rows.size,)
            or np.any(~np.isfinite(values))
            or np.any(values < 0.0)
        ):
            raise RuntimeError(
                "Surface observability kernel returned invalid matched "
                "unshielded values."
            )
        candidate_pair_covered &= values >= reference
    candidate_covered = np.zeros(
        (candidates.shape[0], surfaces.shape[0]),
        dtype=bool,
    )
    candidate_covered[
        candidate_rows[candidate_pair_covered],
        candidate_surface_ids[candidate_pair_covered],
    ] = True

    records = tuple(estimator.measurements)
    if not records:
        acquired_covered = np.zeros(surfaces.shape[0], dtype=bool)
    else:
        acquired_detectors = np.unique(
            np.asarray(
                [record.detector_position_xyz_m for record in records],
                dtype=float,
            ).reshape(-1, 3),
            axis=0,
        )
        if np.any(~np.isfinite(acquired_detectors)):
            raise ValueError(
                "Acquired detector positions must contain finite coordinates."
            )
        acquired_pairs = cKDTree(acquired_detectors).sparse_distance_matrix(
            cKDTree(surfaces),
            max_distance=float(np.nextafter(radius, np.inf)),
            output_type="coo_matrix",
        )
        acquired_rows = np.asarray(acquired_pairs.row, dtype=np.int64)
        acquired_surface_ids = np.asarray(
            acquired_pairs.col,
            dtype=np.int64,
        )
        acquired_min_best = np.full(surfaces.shape[0], np.inf, dtype=float)
        for isotope in isotope_names:
            isotope_best = np.zeros(surfaces.shape[0], dtype=float)
            values = np.asarray(
                kernel.kernel_values_unshielded_for_detector_source_pairs(
                    isotope=isotope,
                    detector_positions=acquired_detectors[acquired_rows],
                    sources=surfaces[acquired_surface_ids],
                ),
                dtype=np.float64,
            ).reshape(-1)
            if (
                values.shape != (acquired_rows.size,)
                or np.any(~np.isfinite(values))
                or np.any(values < 0.0)
            ):
                raise RuntimeError(
                    "Surface observability kernel returned invalid matched "
                    "unshielded acquired-station values."
                )
            np.maximum.at(
                isotope_best,
                acquired_surface_ids,
                values,
            )
            acquired_min_best = np.minimum(
                acquired_min_best,
                isotope_best / reference,
            )
        acquired_covered = acquired_min_best >= 1.0
    return candidate_covered, acquired_covered


def _response_equivalent_surface_coverage_gains(
    *,
    kernel: ContinuousKernel,
    estimator: RotatingShieldPFEstimator,
    surface_points_xyz: NDArray[np.float64],
    surface_area_weights_m2: NDArray[np.float64],
    candidate_poses_xyz: NDArray[np.float64],
    reference_radius_m: float,
) -> NDArray[np.float64]:
    """Return new physically observable surface-area fractions by candidate."""
    candidate_covered, acquired_covered = (
        _response_equivalent_surface_coverage_masks(
            kernel=kernel,
            estimator=estimator,
            surface_points_xyz=surface_points_xyz,
            candidate_poses_xyz=candidate_poses_xyz,
            reference_radius_m=reference_radius_m,
        )
    )
    if candidate_covered.shape[1] == 0:
        return np.zeros(candidate_covered.shape[0], dtype=float)
    area_weights = np.asarray(
        surface_area_weights_m2,
        dtype=np.float64,
    ).reshape(-1)
    if (
        area_weights.shape != (candidate_covered.shape[1],)
        or np.any(~np.isfinite(area_weights))
        or np.any(area_weights <= 0.0)
    ):
        raise ValueError(
            "Surface coverage requires one finite positive physical area "
            "weight per quadrature point."
        )
    total_area = float(np.sum(area_weights, dtype=np.float64))
    if not np.isfinite(total_area) or total_area <= 0.0:
        raise ValueError("Surface coverage total physical area must be positive.")
    newly_covered = candidate_covered & ~acquired_covered[None, :]
    return np.einsum(
        "cs,s->c",
        newly_covered,
        area_weights,
        optimize=True,
    ) / total_area


def _station_revisit_penalties_batch(
    candidate_poses_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    *,
    min_separation_m: float,
) -> NDArray[np.float64]:
    """Return revisit penalties for many candidate stations."""
    candidates = np.asarray(candidate_poses_xyz, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidate_poses_xyz must be shaped (N, 3).")
    penalties = np.zeros(candidates.shape[0], dtype=float)
    min_sep = float(min_separation_m)
    if not np.isfinite(min_sep) or min_sep < 0.0:
        raise ValueError(
            "min_separation_m must be finite and nonnegative."
        )
    visited = _pose_matrix_or_empty(visited_poses_xyz)
    if min_sep <= 0.0 or visited.size == 0 or candidates.shape[0] == 0:
        return penalties
    distances = np.linalg.norm(
        candidates[:, None, :] - visited[None, :, :],
        axis=2,
    )
    min_dist = np.min(distances, axis=1)
    shortfall = 1.0 - min_dist / max(min_sep, 1.0e-12)
    active = min_dist < min_sep
    penalties[active] = shortfall[active] * shortfall[active]
    return penalties


def _bearing_diversity_gain(
    candidate_pose_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    modes_by_isotope: dict[str, list[SignatureMode]],
) -> float:
    """
    Return an isotope-agnostic gain for new bearings of multi-mode posteriors.

    The term activates only for isotopes with multiple posterior modes. It
    rewards stations that separate those modes angularly and provide bearings
    different from already visited stations, which is the generic observability
    need behind same-isotope source separation.
    """
    candidate = np.asarray(candidate_pose_xyz, dtype=float).reshape(3)
    visited = None
    if visited_poses_xyz is not None:
        visited = np.asarray(visited_poses_xyz, dtype=float)
        if visited.ndim == 1 and visited.size == 3:
            visited = visited.reshape(1, 3)
        if visited.ndim != 2 or visited.shape[1] != 3 or visited.size == 0:
            visited = None
    gains: list[float] = []
    weights: list[float] = []
    for modes in modes_by_isotope.values():
        active = [mode for mode in modes if mode.weight > 0.0]
        if len(active) < 2:
            continue
        candidate_angles = [
            _bearing_angle_xy(mode.position_xyz, candidate) for mode in active
        ]
        pair_separations: list[float] = []
        for idx, left in enumerate(candidate_angles):
            for right in candidate_angles[idx + 1 :]:
                pair_separations.append(_angle_distance_rad(left, right) / np.pi)
        pair_gain = min(pair_separations) if pair_separations else 0.0
        novelty_gain = 0.0
        if visited is not None:
            novelty_terms: list[float] = []
            for mode, cand_angle in zip(active, candidate_angles):
                prior_angles = [
                    _bearing_angle_xy(mode.position_xyz, pose) for pose in visited
                ]
                if prior_angles:
                    novelty_terms.append(
                        min(
                            _angle_distance_rad(cand_angle, prior_angle)
                            for prior_angle in prior_angles
                        )
                        / np.pi
                    )
            novelty_gain = float(np.mean(novelty_terms)) if novelty_terms else 0.0
        gains.append(0.5 * float(pair_gain) + 0.5 * float(novelty_gain))
        weights.append(float(_isotope_presence_probability(active) or 0.0))
    if not gains:
        return 0.0
    weighted = _presence_weighted_rows(
        [np.asarray([gain], dtype=float) for gain in gains],
        weights,
        population_size=len(modes_by_isotope),
    )
    return float(weighted[0])


def _bearing_diversity_gains_batch(
    candidate_poses_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    modes_by_isotope: dict[str, list[SignatureMode]],
) -> NDArray[np.float64]:
    """Return bearing-diversity gains for many candidate stations."""
    candidates = np.asarray(candidate_poses_xyz, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidate_poses_xyz must be shaped (N, 3).")
    total_gains: list[NDArray[np.float64]] = []
    total_weights: list[float] = []
    visited = _pose_matrix_or_empty(visited_poses_xyz)
    for modes in modes_by_isotope.values():
        active = [mode for mode in modes if float(mode.weight) > 0.0]
        if len(active) < 2:
            continue
        positions = np.vstack(
            [np.asarray(mode.position_xyz, dtype=float) for mode in active]
        )
        deltas = candidates[:, None, :2] - positions[None, :, :2]
        candidate_angles = np.arctan2(deltas[:, :, 1], deltas[:, :, 0])
        left, right = np.triu_indices(len(active), k=1)
        pair_distances = (
            np.abs(
                np.arctan2(
                    np.sin(candidate_angles[:, left] - candidate_angles[:, right]),
                    np.cos(candidate_angles[:, left] - candidate_angles[:, right]),
                )
            )
            / np.pi
        )
        pair_gain = (
            np.min(pair_distances, axis=1)
            if pair_distances.size
            else np.zeros(candidates.shape[0], dtype=float)
        )
        novelty_gain = np.zeros(candidates.shape[0], dtype=float)
        if visited.size:
            prior_deltas = visited[:, None, :2] - positions[None, :, :2]
            prior_angles = np.arctan2(prior_deltas[:, :, 1], prior_deltas[:, :, 0])
            bearing_differences = (
                candidate_angles[:, :, None]
                - np.transpose(prior_angles, (1, 0))[None, :, :]
            )
            distances = (
                np.abs(
                    np.arctan2(
                        np.sin(bearing_differences),
                        np.cos(bearing_differences),
                    )
                )
                / np.pi
            )
            novelty_gain = np.mean(np.min(distances, axis=2), axis=1)
        total_gains.append(0.5 * pair_gain + 0.5 * novelty_gain)
        total_weights.append(
            float(_isotope_presence_probability(active) or 0.0)
        )
    if not total_gains:
        return np.zeros(candidates.shape[0], dtype=float)
    return _presence_weighted_rows(
        total_gains,
        total_weights,
        population_size=len(modes_by_isotope),
    )


def _frontier_band_gain(
    candidate_pose_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    *,
    target_radius_m: float,
) -> float:
    """Return a gain for expanding from the current explored frontier."""
    target = max(float(target_radius_m), 1.0e-12)
    if visited_poses_xyz is None:
        return 0.0
    visited = np.asarray(visited_poses_xyz, dtype=float)
    if visited.ndim == 1 and visited.size == 3:
        visited = visited.reshape(1, 3)
    if visited.ndim != 2 or visited.shape[1] != 3 or visited.size == 0:
        return 0.0
    candidate = np.asarray(candidate_pose_xyz, dtype=float).reshape(3)
    nearest = float(
        np.min(np.linalg.norm(visited - candidate[None, :], axis=1))
    )
    return float(np.exp(-(((nearest - target) / target) ** 2)))


def _frontier_band_gains_batch(
    candidate_poses_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    *,
    target_radius_m: float,
) -> NDArray[np.float64]:
    """Return frontier-band gains for many candidate stations."""
    candidates = np.asarray(candidate_poses_xyz, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidate_poses_xyz must be shaped (N, 3).")
    target = max(float(target_radius_m), 1.0e-12)
    visited = _pose_matrix_or_empty(visited_poses_xyz)
    if visited.size == 0 or candidates.shape[0] == 0:
        return np.zeros(candidates.shape[0], dtype=float)
    distances = np.linalg.norm(
        candidates[:, None, :] - visited[None, :, :],
        axis=2,
    )
    nearest = np.min(distances, axis=1)
    return np.exp(-(((nearest - target) / target) ** 2))


def _route_turn_penalty(
    candidate_pose_xyz: NDArray[np.float64],
    current_pose_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
) -> float:
    """Return a normalized penalty for sharp reversals from the previous leg."""
    if visited_poses_xyz is None:
        return 0.0
    visited = np.asarray(visited_poses_xyz, dtype=float)
    if visited.ndim == 1 and visited.size == 3:
        visited = visited.reshape(1, 3)
    if visited.ndim != 2 or visited.shape[1] != 3 or visited.shape[0] < 1:
        return 0.0
    current = np.asarray(current_pose_xyz, dtype=float).reshape(3)
    if (
        visited.shape[0] >= 2
        and float(np.linalg.norm(visited[-1] - current)) < 1.0e-6
    ):
        previous = visited[-2]
    else:
        previous = visited[-1]
    prev_vec = current - previous
    next_vec = (
        np.asarray(candidate_pose_xyz, dtype=float).reshape(3) - current
    )
    prev_norm = float(np.linalg.norm(prev_vec))
    next_norm = float(np.linalg.norm(next_vec))
    if prev_norm <= 1.0e-9 or next_norm <= 1.0e-9:
        return 0.0
    dot = float(
        np.clip(np.dot(prev_vec, next_vec) / (prev_norm * next_norm), -1.0, 1.0)
    )
    return float(0.5 * (1.0 - dot))


def _route_turn_penalties_batch(
    candidate_poses_xyz: NDArray[np.float64],
    current_pose_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Return route-turn penalties for many candidate stations."""
    candidates = np.asarray(candidate_poses_xyz, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidate_poses_xyz must be shaped (N, 3).")
    penalties = np.zeros(candidates.shape[0], dtype=float)
    visited = _pose_matrix_or_empty(visited_poses_xyz)
    if visited.shape[0] < 1 or candidates.shape[0] == 0:
        return penalties
    current = np.asarray(current_pose_xyz, dtype=float).reshape(3)
    if (
        visited.shape[0] >= 2
        and float(np.linalg.norm(visited[-1] - current)) < 1.0e-6
    ):
        previous = visited[-2]
    else:
        previous = visited[-1]
    prev_vec = current - previous
    prev_norm = float(np.linalg.norm(prev_vec))
    next_vecs = candidates - current[None, :]
    next_norms = np.linalg.norm(next_vecs, axis=1)
    active = (prev_norm > 1.0e-9) & (next_norms > 1.0e-9)
    if not np.any(active):
        return penalties
    dots = np.sum(next_vecs[active] * prev_vec.reshape(1, 3), axis=1) / (
        prev_norm * next_norms[active]
    )
    penalties[active] = 0.5 * (1.0 - np.clip(dots, -1.0, 1.0))
    return penalties


def _filter_station_separation(
    candidate_poses_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    *,
    min_separation_m: float,
) -> tuple[NDArray[np.float64], int]:
    """Remove every station that violates the generic 3-D separation rule."""
    min_sep = float(min_separation_m)
    if not np.isfinite(min_sep) or min_sep < 0.0:
        raise ValueError(
            "min_separation_m must be finite and nonnegative."
        )
    candidates = np.asarray(candidate_poses_xyz, dtype=float)
    if (
        candidates.ndim != 2
        or candidates.shape[1] != 3
        or np.any(~np.isfinite(candidates))
    ):
        raise ValueError(
            "candidate_poses_xyz must be finite and shaped (N, 3)."
        )
    if candidates.size == 0 or min_sep <= 0.0 or visited_poses_xyz is None:
        return candidates, 0
    visited = _pose_matrix_or_empty(visited_poses_xyz)
    if visited.size == 0:
        return candidates, 0
    distances = np.linalg.norm(
        candidates[:, None, :] - visited[None, :, :],
        axis=2,
    )
    keep = np.min(distances, axis=1) >= min_sep
    removed = int(np.count_nonzero(~keep))
    if not np.any(keep):
        return np.zeros((0, 3), dtype=float), removed
    return candidates[keep], removed


def _shield_transition_cost(
    normals: NDArray[np.float64],
    from_pair_id: int | None,
    program: ShieldProgram,
) -> float:
    """Return angular shield-transition cost for a program."""
    if not program.pair_ids:
        return 0.0
    normal_arr = np.asarray(normals, dtype=float)
    num_orients = int(normal_arr.shape[0])
    sequence: list[int] = []
    if from_pair_id is not None and int(from_pair_id) >= 0:
        sequence.append(int(from_pair_id))
    sequence.extend(int(pair_id) for pair_id in program.pair_ids)
    if len(sequence) < 2:
        return 0.0
    cost = 0.0
    for prev_id, next_id in zip(sequence[:-1], sequence[1:]):
        prev_fe, prev_pb = _pair_indices(prev_id, num_orients)
        next_fe, next_pb = _pair_indices(next_id, num_orients)
        for prev_idx, next_idx in ((prev_fe, next_fe), (prev_pb, next_pb)):
            dot = float(
                np.clip(
                    np.dot(normal_arr[prev_idx], normal_arr[next_idx]),
                    -1.0,
                    1.0,
                )
            )
            cost += float(np.arccos(dot))
    return cost


def _compose_transition_score(
    *,
    node: DSSPPNode,
    previous_pose_xyz: NDArray[np.float64],
    previous_pair_id: int | None,
    estimator: RotatingShieldPFEstimator,
    map_api: object | None,
    config: DSSPPConfig,
) -> tuple[float, float]:
    """Return node score and path length for a specific predecessor."""
    path_length = _node_path_length(map_api, previous_pose_xyz, node.pose_xyz)
    if not np.isfinite(path_length):
        return -float("inf"), float("inf")
    travel_time = path_length / float(config.robot_speed_m_s)
    time_cost = travel_time + len(node.program.pair_ids) * (
        float(config.rotation_overhead_s) + float(config.live_time_s)
    )
    rotation_cost = _shield_transition_cost(
        estimator.normals,
        previous_pair_id,
        node.program,
    )
    score = (
        float(node.static_score)
        - float(node.distance_weight) * float(path_length)
        - float(config.lambda_time) * float(time_cost)
        - float(config.lambda_rotation) * float(rotation_cost)
    )
    return float(score), float(path_length)


def _stable_descending_indices(values: NDArray[np.float64]) -> NDArray[np.int64]:
    """Return deterministic descending indices with source order as tie-break."""
    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(scores)):
        raise ValueError("Shortlist scores must be finite.")
    return np.lexsort(
        (
            np.arange(scores.size, dtype=np.int64),
            -scores,
        )
    ).astype(np.int64, copy=False)


def _exact_eig_shortlist(
    pending: Sequence[_PendingDSSPPNode],
    programs: Sequence[ShieldProgram],
    proxy_information_scores_pp: NDArray[np.float64],
    *,
    config: DSSPPConfig,
) -> tuple[NDArray[np.int64], NDArray[np.float64], dict[str, int]]:
    """Choose a fixed-budget exact-EIG set with coverage and program reserves."""
    pending_nodes = list(pending)
    if not pending_nodes:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            {
                "global": 0,
                "coverage": 0,
                "program_diversity": 0,
            },
        )
    proxy = np.asarray(proxy_information_scores_pp, dtype=np.float64)
    if (
        proxy.ndim != 2
        or proxy.shape[1] != len(programs)
        or np.any(~np.isfinite(proxy))
        or np.any(proxy < 0.0)
    ):
        raise ValueError("Proxy information scores have an invalid shape.")
    program_index = {
        (
            str(program.name),
            tuple(int(value) for value in program.pair_ids),
            str(program.kind),
        ): index
        for index, program in enumerate(programs)
    }
    if len(program_index) != len(programs):
        raise ValueError("DSS shield programs must be unique.")
    ranking_scores = np.zeros(len(pending_nodes), dtype=np.float64)
    pending_program_indices = np.zeros(len(pending_nodes), dtype=np.int64)
    for index, item in enumerate(pending_nodes):
        key = (
            str(item.program.name),
            tuple(int(value) for value in item.program.pair_ids),
            str(item.program.kind),
        )
        resolved_program_index = program_index.get(key)
        if resolved_program_index is None:
            raise RuntimeError("Pending DSS node references an unknown program.")
        if int(item.pose_index) < 0 or int(item.pose_index) >= proxy.shape[0]:
            raise IndexError("Pending DSS node references an unknown pose.")
        pending_program_indices[index] = int(resolved_program_index)
        ranking_scores[index] = (
            float(item.static_score)
            + float(config.lambda_eig)
            * float(proxy[int(item.pose_index), int(resolved_program_index)])
        )
    limit = min(int(config.exact_eig_action_limit), len(pending_nodes))
    if limit == len(pending_nodes):
        return (
            np.arange(len(pending_nodes), dtype=np.int64),
            ranking_scores,
            {
                "global": int(limit),
                "coverage": 0,
                "program_diversity": 0,
            },
        )
    selected: set[int] = set()
    category_counts = {
        "global": 0,
        "coverage": 0,
        "program_diversity": 0,
    }

    coverage_candidates: list[int] = []
    for pose_index in sorted({int(item.pose_index) for item in pending_nodes}):
        pose_rows = np.asarray(
            [
                index
                for index, item in enumerate(pending_nodes)
                if int(item.pose_index) == pose_index
            ],
            dtype=np.int64,
        )
        pose_best = pose_rows[
            _stable_descending_indices(ranking_scores[pose_rows])[0]
        ]
        coverage_candidates.append(int(pose_best))
    coverage_candidates.sort(
        key=lambda index: (
            -float(pending_nodes[index].coverage_gain),
            -float(ranking_scores[index]),
            int(index),
        )
    )
    for index in coverage_candidates[
        : int(config.exact_eig_coverage_reserve)
    ]:
        if len(selected) >= limit:
            break
        selected.add(int(index))
        category_counts["coverage"] += 1

    program_candidates: list[int] = []
    for resolved_program_index in range(len(programs)):
        program_rows = np.flatnonzero(
            pending_program_indices == resolved_program_index
        )
        if program_rows.size == 0:
            continue
        best = program_rows[
            _stable_descending_indices(ranking_scores[program_rows])[0]
        ]
        program_candidates.append(int(best))
    program_candidates.sort(
        key=lambda index: (-float(ranking_scores[index]), int(index))
    )
    for index in program_candidates[
        : int(config.exact_eig_program_diversity_reserve)
    ]:
        if len(selected) >= limit:
            break
        if int(index) not in selected:
            selected.add(int(index))
            category_counts["program_diversity"] += 1

    for index_raw in _stable_descending_indices(ranking_scores):
        if len(selected) >= limit:
            break
        index = int(index_raw)
        if index not in selected:
            selected.add(index)
            category_counts["global"] += 1
    ordered = np.asarray(
        sorted(selected, key=lambda index: (-ranking_scores[index], index)),
        dtype=np.int64,
    )
    if ordered.size != limit:
        raise RuntimeError("Exact-EIG shortlist did not fill its fixed budget.")
    return ordered, ranking_scores, category_counts


def _build_nodes(
    *,
    estimator: RotatingShieldPFEstimator,
    candidate_poses_xyz: NDArray[np.float64],
    programs: Sequence[ShieldProgram],
    modes_by_isotope: dict[str, list[SignatureMode]],
    current_pose_xyz: NDArray[np.float64],
    current_pair_id: int | None,
    visited_poses_xyz: NDArray[np.float64] | None,
    map_api: object | None,
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]] | None,
    config: DSSPPConfig,
    rng: np.random.Generator,
    joint_particles: JointPlanningParticles,
) -> tuple[list[DSSPPNode], dict[str, object]]:
    """Shortlist all actions cheaply, then exactly evaluate a fixed subset."""
    kernel = _continuous_kernel_for_estimator(
        estimator,
        detector_aperture_samples=int(config.detector_aperture_samples),
    )
    candidate_poses = np.asarray(candidate_poses_xyz, dtype=float)
    if candidate_poses.ndim != 2 or candidate_poses.shape[1] != 3:
        raise ValueError("candidate_poses_xyz must be shape (N, 3).")
    info_gains = np.zeros(candidate_poses.shape[0], dtype=float)
    path_lengths = _node_path_lengths_batch(
        map_api,
        current_pose_xyz,
        candidate_poses,
    )
    surface_quadrature_builder = getattr(
        estimator,
        "surface_atlas_area_quadrature",
        None,
    )
    coverage_quadrature_diagnostics: dict[str, object] | None = None
    if callable(surface_quadrature_builder):
        surface_quadrature = surface_quadrature_builder(
            max_points=int(config.coverage_surface_quadrature_max_points),
            maximum_hausdorff_bound_m=float(
                config.coverage_surface_max_hausdorff_m
            ),
        )
        surface_coverage_points = np.asarray(
            surface_quadrature.positions_s3,
            dtype=np.float64,
        ).reshape(-1, 3)
        surface_area_weights_m2 = np.asarray(
            surface_quadrature.area_weights_m2_s,
            dtype=np.float64,
        ).reshape(-1)
        coverage_raw = _response_equivalent_surface_coverage_gains(
            kernel=kernel,
            estimator=estimator,
            surface_points_xyz=surface_coverage_points,
            surface_area_weights_m2=surface_area_weights_m2,
            candidate_poses_xyz=candidate_poses,
            reference_radius_m=float(config.coverage_radius_m),
        )
        diagnostics_getter = getattr(
            surface_quadrature,
            "diagnostics",
            None,
        )
        if not callable(diagnostics_getter):
            raise TypeError(
                "Surface quadrature must expose completeness diagnostics."
            )
        coverage_quadrature_diagnostics = dict(diagnostics_getter())
        coverage_support = (
            "complete_chart_center_area_weighted_unshielded_station_coverage"
        )
    elif isinstance(estimator, RotatingShieldPFEstimator):
        raise RuntimeError(
            "Production DSS coverage requires the PF continuous physical "
            "surface atlas; an XY/free-cell fallback is forbidden."
        )
    else:
        # A small deterministic oracle remains available only to unit-test
        # score composition without constructing a production estimator.
        surface_coverage_points = _free_cell_centers(
            map_api,
            z_value=float(current_pose_xyz[2]),
            max_cells=int(config.coverage_surface_quadrature_max_points),
            bounds_xyz=bounds_xyz,
        )
        coverage_raw = _coverage_gain_fractions_batch(
            cell_centers_xyz=surface_coverage_points,
            candidate_poses_xyz=candidate_poses,
            visited_poses_xyz=visited_poses_xyz,
            radius_m=float(config.coverage_radius_m),
        )
        coverage_support = "test_only_free_cell_fallback_3d"
    coverage_norm = coverage_raw.copy()
    max_coverage = float(np.max(coverage_norm)) if coverage_norm.size else 0.0
    if max_coverage > 0.0:
        coverage_norm = coverage_norm / max_coverage
    coverage_floor = 0.0
    coverage_floor_quantile = float(config.coverage_floor_quantile)
    if (
        coverage_norm.size
        and float(config.coverage_floor_weight) > 0.0
        and coverage_floor_quantile > 0.0
    ):
        positive_coverage = coverage_norm[coverage_norm > 0.0]
        if positive_coverage.size:
            coverage_floor = float(
                np.quantile(
                    positive_coverage,
                    coverage_floor_quantile,
                )
            )
    revisit_penalties = _station_revisit_penalties_batch(
        candidate_poses,
        visited_poses_xyz,
        min_separation_m=float(config.min_station_separation_m),
    )
    bearing_gains = _bearing_diversity_gains_batch(
        candidate_poses,
        visited_poses_xyz,
        modes_by_isotope,
    )
    frontier_target = max(
        float(config.min_station_separation_m),
        float(config.coverage_radius_m),
    )
    frontier_gains = _frontier_band_gains_batch(
        candidate_poses,
        visited_poses_xyz,
        target_radius_m=frontier_target,
    )
    turn_penalties = _route_turn_penalties_batch(
        candidate_poses,
        current_pose_xyz,
        visited_poses_xyz,
    )
    local_orbit_gains = _local_orbit_gains_batch(
        candidate_poses,
        modes_by_isotope,
        config=config,
    )
    elevation_condition_gains = _elevation_condition_gains_batch(
        candidate_poses,
        modes_by_isotope,
        config=config,
    )
    evaluation_pose_indices = np.arange(candidate_poses.shape[0], dtype=np.int64)
    raw_nodes: list[DSSPPNode] = []
    pending: list[_PendingDSSPPNode] = []
    eval_indices = [int(idx) for idx in evaluation_pose_indices]
    pose_eval_context: dict[str, object] = {
        "candidate_poses": candidate_poses,
        "path_lengths": path_lengths,
        "programs": programs,
        "config": config,
        "coverage_norm": coverage_norm,
        "coverage_raw": coverage_raw,
        "revisit_penalties": revisit_penalties,
        "bearing_gains": bearing_gains,
        "frontier_gains": frontier_gains,
        "turn_penalties": turn_penalties,
        "local_orbit_gains": local_orbit_gains,
        "elevation_condition_gains": elevation_condition_gains,
        "coverage_floor": float(coverage_floor),
        "coverage_support": coverage_support,
        "coverage_quadrature": coverage_quadrature_diagnostics,
    }
    pose_results = _materialize_pose_nodes(
        eval_indices,
        context=pose_eval_context,
    )
    for (
        _pose_index,
        _local_cheap_score,
        local_pending,
    ) in pose_results:
        if local_pending:
            pending.extend(local_pending)
    if not pending:
        return [], {
            "total_action_count": 0,
            "proxy_action_count": 0,
            "exact_action_count": 0,
        }
    total_action_count = len(pending)
    proxy_wall_s = 0.0
    exact_wall_s = 0.0
    proxy_information_scores = np.zeros(
        (candidate_poses.shape[0], len(programs)),
        dtype=np.float64,
    )
    shortlist_indices = np.arange(total_action_count, dtype=np.int64)
    proxy_ranking_scores = np.asarray(
        [float(item.static_score) for item in pending],
        dtype=np.float64,
    )
    shortlist_category_counts = {
        "global": int(total_action_count),
        "coverage": 0,
        "program_diversity": 0,
    }
    proxy_action_count = 0
    proxy_particle_count = 0
    proxy_eig_runtime_diagnostics: dict[str, object] = {}
    exact_eig_runtime_rounds: list[dict[str, object]] = []
    exact_eig_seed = int(
        rng.integers(
            0,
            np.iinfo(np.int64).max,
            endpoint=False,
            dtype=np.int64,
        )
    )
    if (
        float(config.lambda_eig) > 0.0
        and total_action_count > int(config.exact_eig_action_limit)
    ):
        proxy_joint_particles = estimator.planning_joint_particles(
            max_particles=int(config.proxy_planning_particles),
            method="top_weight",
        )
        proxy_particle_count = int(
            np.asarray(proxy_joint_particles.weights_n).size
        )
        proxy_started = time.perf_counter()
        proxy_information_scores = (
            _program_information_proxy_for_poses(
                estimator,
                candidate_poses,
                programs,
                config=config,
                joint_particles=proxy_joint_particles,
                rng=rng,
                eig_call_seed=exact_eig_seed,
                diagnostics=proxy_eig_runtime_diagnostics,
            )
        )
        proxy_wall_s = float(time.perf_counter() - proxy_started)
        proxy_action_count = int(total_action_count)
        (
            shortlist_indices,
            proxy_ranking_scores,
            shortlist_category_counts,
        ) = _exact_eig_shortlist(
            pending,
            programs,
            proxy_information_scores,
            config=config,
        )
    initial_indices = (
        np.asarray(shortlist_indices, dtype=np.int64)
        if float(config.lambda_eig) > 0.0
        else np.arange(total_action_count, dtype=np.int64)
    )
    proxy_order = _stable_descending_indices(proxy_ranking_scores)
    remaining_order = proxy_order[
        ~np.isin(proxy_order, initial_indices, assume_unique=False)
    ]
    evaluation_order = np.concatenate((initial_indices, remaining_order))
    if (
        np.unique(evaluation_order).size != total_action_count
        or np.any(evaluation_order < 0)
        or np.any(evaluation_order >= total_action_count)
    ):
        raise RuntimeError("Adaptive exact-EIG ordering lost a DSS action.")

    program_information_gains = np.full(
        len(pending),
        np.nan,
        dtype=np.float64,
    )
    evaluated_pending_indices = np.zeros(0, dtype=np.int64)

    def _evaluate_exact_indices(
        new_indices: NDArray[np.int64],
    ) -> None:
        """Evaluate one adaptive action batch under a fixed common RNG stream."""
        nonlocal exact_wall_s
        if new_indices.size == 0:
            return
        pending_indices_by_pose: dict[int, list[int]] = {}
        for pending_index_raw in new_indices:
            pending_index = int(pending_index_raw)
            item = pending[pending_index]
            pending_indices_by_pose.setdefault(
                int(item.pose_index),
                [],
            ).append(pending_index)

        eig_indices = sorted(pending_indices_by_pose)
        batched_programs = [
            [
                pending[index].program
                for index in pending_indices_by_pose.get(pose_index, [])
            ]
            for pose_index in eig_indices
        ]
        round_diagnostics: dict[str, object] = {}
        exact_started = time.perf_counter()
        batched_gains = _program_information_gains_for_poses(
            estimator,
            candidate_poses[eig_indices],
            batched_programs,
            config=config,
            rng=rng,
            joint_particles=joint_particles,
            diagnostics=round_diagnostics,
            eig_call_seed=exact_eig_seed,
        )
        round_wall_s = float(time.perf_counter() - exact_started)
        exact_wall_s += round_wall_s
        round_diagnostics["wall_s"] = round_wall_s
        round_diagnostics["action_count"] = int(new_indices.size)
        exact_eig_runtime_rounds.append(round_diagnostics)
        eig_results = [
            (
                pose_index,
                pending_indices_by_pose.get(pose_index, []),
                np.asarray(values, dtype=float),
            )
            for pose_index, values in zip(eig_indices, batched_gains)
        ]
        for _pose_index, pending_indices, values in eig_results:
            if values.size != len(pending_indices):
                raise RuntimeError(
                    "Program EIG result does not match the evaluated programs."
                )
            for pending_index, value in zip(
                pending_indices,
                values,
                strict=True,
            ):
                program_information_gains[pending_index] = float(value)

    normalized_joint_weights = _normalise_weights(
        np.asarray(joint_particles.weights_n, dtype=np.float64)
    )
    positive_joint_weights = normalized_joint_weights[
        normalized_joint_weights > 0.0
    ]
    particle_entropy = float(
        -np.sum(positive_joint_weights * np.log(positive_joint_weights))
    )
    finite_sample_eig_upper = _finite_sample_information_gain_upper_bound(
        normalized_joint_weights
    )
    excluded_universal_upper = float("inf")
    evaluated_objective_lower = -float("inf")
    shortlist_bound_certified = False
    adaptive_round_count = 0
    next_evaluation_offset = 0
    while next_evaluation_offset < total_action_count:
        adaptive_round_count += 1
        next_stop = (
            initial_indices.size
            if next_evaluation_offset == 0
            else min(
                next_evaluation_offset + int(config.exact_eig_action_limit),
                total_action_count,
            )
        )
        if next_stop <= next_evaluation_offset:
            raise RuntimeError("Adaptive exact-EIG batch made no progress.")
        new_indices = evaluation_order[next_evaluation_offset:next_stop]
        if float(config.lambda_eig) > 0.0:
            _evaluate_exact_indices(new_indices)
        else:
            program_information_gains[new_indices] = 0.0
        next_evaluation_offset = int(next_stop)
        evaluated_pending_indices = evaluation_order[:next_evaluation_offset]

        info_gains.fill(0.0)
        for pending_index_raw in evaluated_pending_indices:
            pending_index = int(pending_index_raw)
            item = pending[pending_index]
            info_gains[int(item.pose_index)] = max(
                float(info_gains[int(item.pose_index)]),
                float(program_information_gains[pending_index]),
            )
        finite_path = np.isfinite(path_lengths)
        if config.lambda_distance is None:
            evaluated_pose_mask = np.zeros(
                candidate_poses.shape[0],
                dtype=bool,
            )
            evaluated_pose_mask[
                np.asarray(
                    sorted(
                        {
                            int(pending[int(index)].pose_index)
                            for index in evaluated_pending_indices
                        }
                    ),
                    dtype=np.int64,
                )
            ] = True
            lambda_distance = estimate_lambda_cost(
                info_gains[finite_path & evaluated_pose_mask],
                path_lengths[finite_path & evaluated_pose_mask],
                method="range",
            )
        else:
            lambda_distance = float(config.lambda_distance)

        raw_nodes = []
        for pending_index_raw in evaluated_pending_indices:
            pending_index = int(pending_index_raw)
            item = pending[pending_index]
            info_gain = float(program_information_gains[pending_index])
            base_score = (
                float(item.static_score)
                + float(config.lambda_eig) * info_gain
            )
            placeholder_node = DSSPPNode(
                pose_index=int(item.pose_index),
                pose_xyz=item.pose_xyz,
                program=item.program,
                score=0.0,
                static_score=float(base_score),
                distance_weight=float(lambda_distance),
                information_gain=float(info_gain),
                coverage_gain=float(item.coverage_gain),
                revisit_penalty=float(item.revisit_penalty),
                bearing_diversity_gain=float(item.bearing_diversity_gain),
                frontier_gain=float(item.frontier_gain),
                turn_penalty=float(item.turn_penalty),
                local_orbit_gain=float(item.local_orbit_gain),
                elevation_condition_gain=float(
                    item.elevation_condition_gain
                ),
            )
            score, _ = _compose_transition_score(
                node=placeholder_node,
                previous_pose_xyz=current_pose_xyz,
                previous_pair_id=current_pair_id,
                estimator=estimator,
                map_api=map_api,
                config=config,
            )
            raw_nodes.append(
                DSSPPNode(
                    **{
                        **placeholder_node.__dict__,
                        "score": float(score),
                    }
                )
            )
        raw_nodes.sort(key=lambda node: node.score, reverse=True)

        excluded_mask = np.ones(total_action_count, dtype=bool)
        excluded_mask[evaluated_pending_indices] = False
        if not np.any(excluded_mask):
            excluded_universal_upper = -float("inf")
            shortlist_bound_certified = True
            break
        if config.lambda_distance is None:
            # Auto-scaled distance changes with unseen EIG values, so no safe
            # finite lower/upper objective bracket exists before exhaustion.
            continue
        evaluated_lower_scores: list[float] = []
        for index_raw in evaluated_pending_indices:
            item = pending[int(index_raw)]
            lower_node = DSSPPNode(
                pose_index=int(item.pose_index),
                pose_xyz=item.pose_xyz,
                program=item.program,
                score=0.0,
                static_score=float(item.static_score),
                distance_weight=float(lambda_distance),
                information_gain=0.0,
                coverage_gain=float(item.coverage_gain),
                revisit_penalty=float(item.revisit_penalty),
                bearing_diversity_gain=float(item.bearing_diversity_gain),
                frontier_gain=float(item.frontier_gain),
                turn_penalty=float(item.turn_penalty),
                local_orbit_gain=float(item.local_orbit_gain),
                elevation_condition_gain=float(
                    item.elevation_condition_gain
                ),
            )
            lower_score, _ = _compose_transition_score(
                node=lower_node,
                previous_pose_xyz=current_pose_xyz,
                previous_pair_id=current_pair_id,
                estimator=estimator,
                map_api=map_api,
                config=config,
            )
            evaluated_lower_scores.append(float(lower_score))
        evaluated_objective_lower = float(max(evaluated_lower_scores))
        excluded_upper_scores: list[float] = []
        for index_raw in np.flatnonzero(excluded_mask):
            item = pending[int(index_raw)]
            upper_node = DSSPPNode(
                pose_index=int(item.pose_index),
                pose_xyz=item.pose_xyz,
                program=item.program,
                score=0.0,
                static_score=(
                    float(item.static_score)
                    + float(config.lambda_eig) * finite_sample_eig_upper
                ),
                distance_weight=float(lambda_distance),
                information_gain=float(finite_sample_eig_upper),
                coverage_gain=float(item.coverage_gain),
                revisit_penalty=float(item.revisit_penalty),
                bearing_diversity_gain=float(item.bearing_diversity_gain),
                frontier_gain=float(item.frontier_gain),
                turn_penalty=float(item.turn_penalty),
                local_orbit_gain=float(item.local_orbit_gain),
                elevation_condition_gain=float(
                    item.elevation_condition_gain
                ),
            )
            upper_score, _ = _compose_transition_score(
                node=upper_node,
                previous_pose_xyz=current_pose_xyz,
                previous_pair_id=current_pair_id,
                estimator=estimator,
                map_api=map_api,
                config=config,
            )
            excluded_upper_scores.append(float(upper_score))
        excluded_universal_upper = float(max(excluded_upper_scores))
        shortlist_bound_certified = bool(
            evaluated_objective_lower
            >= excluded_universal_upper - 1.0e-12
        )
        if shortlist_bound_certified:
            break

    best_exact_score = (
        float(raw_nodes[0].score)
        if raw_nodes
        else -float("inf")
    )
    winner_exceeds_universal_excluded_bound = shortlist_bound_certified
    selected_pending_index = -1
    if raw_nodes:
        for index_raw in evaluated_pending_indices:
            index = int(index_raw)
            if (
                int(pending[index].pose_index) == int(raw_nodes[0].pose_index)
                and pending[index].program == raw_nodes[0].program
            ):
                selected_pending_index = index
                break
        if selected_pending_index < 0:
            raise RuntimeError("Selected exact-EIG node lost its pending identity.")
    proxy_order = _stable_descending_indices(proxy_ranking_scores)
    proxy_rank = (
        int(
            np.flatnonzero(proxy_order == selected_pending_index)[0]
            + 1
        )
        if selected_pending_index >= 0
        else 0
    )
    model = validate_full_spectrum_model(
        estimator.full_spectrum_generative_model
    )
    sample_count = int(estimator.pf_config.planning_eig_samples)
    particle_count = int(np.asarray(joint_particles.weights_n).size)
    view_count = max((len(program.pair_ids) for program in programs), default=0)
    energy_bin_count = int(np.asarray(model.energy_axis_keV).size)
    exact_action_count = (
        int(evaluated_pending_indices.size)
        if float(config.lambda_eig) > 0.0
        else 0
    )
    diagnostics: dict[str, object] = {
        "total_action_count": int(total_action_count),
        "proxy_action_count": int(proxy_action_count),
        "proxy_particle_count": int(proxy_particle_count),
        "proxy_eig_samples": int(config.proxy_eig_samples),
        "shared_full_spectrum_detector_aperture_samples": int(
            config.detector_aperture_samples
        ),
        "exact_action_count": int(exact_action_count),
        "exact_eig_action_limit": int(config.exact_eig_action_limit),
        "adaptive_exact_eig_round_count": int(adaptive_round_count),
        "adaptive_exact_eig_exhausted_all_actions": bool(
            exact_action_count == total_action_count
        ),
        "shortlist_category_counts": dict(shortlist_category_counts),
        "proxy_wall_s": float(proxy_wall_s),
        "exact_eig_wall_s": float(exact_wall_s),
        "proxy_eig_runtime": dict(proxy_eig_runtime_diagnostics),
        "exact_eig_runtime": {
            "rounds": list(exact_eig_runtime_rounds),
        },
        "proxy_unique_action_count": int(proxy_action_count),
        "legacy_all_exact_bin_state_operations": int(
            total_action_count
            * max(sample_count, 0)
            * particle_count
            * view_count
            * energy_bin_count
        ),
        "shortlisted_exact_bin_state_operations": int(
            exact_action_count
            * max(sample_count, 0)
            * particle_count
            * view_count
            * energy_bin_count
        ),
        "proxy_full_spectrum_bin_state_operations": int(
            proxy_action_count
            * int(config.proxy_eig_samples)
            * proxy_particle_count
            * view_count
            * energy_bin_count
            if proxy_action_count
            else 0
        ),
        "shortlist_mc_winner_exceeds_universal_excluded_bound": bool(
            winner_exceeds_universal_excluded_bound
        ),
        "shortlist_best_exact_score": float(best_exact_score),
        "shortlist_evaluated_objective_lower_bound": (
            None
            if not np.isfinite(evaluated_objective_lower)
            else float(evaluated_objective_lower)
        ),
        "shortlist_max_excluded_universal_objective_upper_bound": (
            None
            if not np.isfinite(excluded_universal_upper)
            else float(excluded_universal_upper)
        ),
        "shortlist_selected_proxy_rank": int(proxy_rank),
        "proxy_contract": (
            "reduced_particle_and_sample_joint_full_spectrum_generative_eig_"
            "with_identical_background_dead_time_marks_and_likelihood"
        ),
        "posterior_entropy_true_eig_upper_bound_nats": float(
            particle_entropy
        ),
        "finite_sample_mc_eig_upper_bound_nats": float(
            finite_sample_eig_upper
        ),
        "universal_eig_upper_bound_nats": float(
            finite_sample_eig_upper
        ),
        "shortlist_formal_recall_certificate_available": bool(
            shortlist_bound_certified
        ),
        "shortlist_certification_note": (
            "Exact evaluation expands until a zero-EIG evaluated objective "
            "lower bound exceeds every excluded action's finite-sample "
            "-log(min_positive_prior_mass) KL upper bound. Prior entropy is "
            "only a bound on expected mutual information, not on a finite "
            "Monte Carlo estimate. When auto distance scaling prevents a "
            "safe objective bound, every action is evaluated exactly."
        ),
        "eig_shortlist_wall_s": float(proxy_wall_s + exact_wall_s),
    }
    return raw_nodes, diagnostics


def select_dss_pp_next_station(
    estimator: RotatingShieldPFEstimator,
    candidate_poses_xyz: NDArray[np.float64],
    current_pose_xyz: NDArray[np.float64],
    *,
    current_pair_id: int | None = None,
    visited_poses_xyz: NDArray[np.float64] | None = None,
    map_api: object | None = None,
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    continuous_height_bounds_m: tuple[float, float] | None = None,
    config: DSSPPConfig | None = None,
    rng: np.random.Generator | None = None,
) -> DSSPPResult:
    """Select the next station and its actually executed shield program.

    When ``continuous_height_bounds_m`` is provided, newly augmented xy
    stations receive deterministic low-discrepancy heights within that range;
    caller-provided candidate heights remain unchanged. No height/lateral
    alternation constraint is imposed; exact EIG and global surface
    observability decide among all reachable actions.
    """
    cfg = config or DSSPPConfig()
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            "select_dss_pp_next_station requires a persistent explicit rng; "
            "reinitializing a fixed seed per planning call is forbidden."
        )
    planning_rng = rng
    pf_max_sources = _validate_mode_capacity(estimator, cfg)
    _validate_eig_likelihood_contract(estimator, cfg)
    current_pose = np.asarray(current_pose_xyz, dtype=float)
    if current_pose.shape != (3,) or np.any(~np.isfinite(current_pose)):
        raise ValueError("current_pose_xyz must be a finite shape-(3,) vector.")
    if current_pair_id is not None:
        if isinstance(current_pair_id, bool) or not isinstance(
            current_pair_id,
            (int, np.integer),
        ):
            raise ValueError("current_pair_id must be an integer or None.")
        current_pair_count = int(estimator.num_orientations) ** 2
        if not 0 <= int(current_pair_id) < current_pair_count:
            raise ValueError(
                "current_pair_id lies outside the estimator shield-pair "
                f"support [0, {current_pair_count - 1}]."
            )
    joint_particles = estimator.planning_joint_particles(
        max_particles=cfg.planning_particles,
        method=cfg.planning_method,
        rng=planning_rng,
    )
    geometry_joint_particles = estimator.planning_joint_particles(
        max_particles=0,
        method="top_weight",
    )
    modes = extract_signature_modes(
        estimator,
        mode_cluster_radius_m=float(cfg.mode_cluster_radius_m),
        max_modes_per_isotope=int(cfg.max_modes_per_isotope),
        rng=planning_rng,
        joint_particles=geometry_joint_particles,
    )
    _official_modes, official_snapshot_diagnostics = _official_signature_modes(
        estimator,
        max_modes_per_isotope=int(cfg.max_modes_per_isotope),
    )
    candidates = np.asarray(candidate_poses_xyz, dtype=float)
    if cfg.augment_candidates:
        candidates = augment_candidate_stations(
            candidates,
            modes_by_isotope=modes,
            current_pose_xyz=current_pose,
            visited_poses_xyz=visited_poses_xyz,
            map_api=map_api,
            bounds_xyz=bounds_xyz,
            config=cfg,
            continuous_height_bounds_m=continuous_height_bounds_m,
            rng=planning_rng,
        )

    candidates, separation_filtered = _filter_station_separation(
        candidates,
        visited_poses_xyz,
        min_separation_m=float(cfg.min_station_separation_m),
    )
    candidates, path_filtered = _filter_path_reachable_stations(
        candidates,
        current_pose_xyz=current_pose,
        map_api=map_api,
    )
    if candidates.size == 0:
        raise ValueError(
            "DSS-PP received no reachable candidate after the generic 3-D "
            "station-separation contract."
        )
    if cfg.forced_program_pair_ids is None:
        programs = build_shield_program_library(
            estimator.normals,
            program_length=int(cfg.program_length),
            max_programs=int(cfg.max_programs),
        )
    else:
        pair_count = int(estimator.num_orientations) ** 2
        if any(
            int(pair_id) >= pair_count
            for pair_id in cfg.forced_program_pair_ids
        ):
            raise ValueError(
                "forced_program_pair_ids exceed the estimator shield-pair "
                f"support [0, {pair_count - 1}]."
            )
        programs = [
            ShieldProgram(
                name="forced_baseline_shield_program",
                pair_ids=tuple(int(pair_id) for pair_id in cfg.forced_program_pair_ids),
                kind="forced_baseline",
            )
        ]
    candidate_pair_ids = [
        int(pair_id)
        for program in programs
        for pair_id in program.pair_ids
    ]
    pair_occurrences = np.bincount(
        np.asarray(candidate_pair_ids, dtype=np.int64),
        minlength=int(estimator.num_orientations) ** 2,
    )
    positive_occurrences = pair_occurrences[pair_occurrences > 0]
    companion_sets = {
        pair_id: set()
        for pair_id in np.flatnonzero(pair_occurrences)
    }
    for program in programs:
        program_pairs = set(int(pair_id) for pair_id in program.pair_ids)
        for pair_id in program_pairs:
            companion_sets[pair_id].update(program_pairs - {pair_id})
    nodes, shortlist_diagnostics = _build_nodes(
        estimator=estimator,
        candidate_poses_xyz=candidates,
        programs=programs,
        modes_by_isotope=modes,
        current_pose_xyz=current_pose,
        current_pair_id=current_pair_id,
        visited_poses_xyz=visited_poses_xyz,
        map_api=map_api,
        bounds_xyz=bounds_xyz,
        config=cfg,
        rng=planning_rng,
        joint_particles=joint_particles,
    )
    if not nodes:
        raise ValueError("DSS-PP could not evaluate any station-program node.")
    first = nodes[0]
    sequence = (first,)
    best_score = float(first.score)
    mode_count = sum(len(mode_list) for mode_list in modes.values())
    ranked_limit = int(cfg.diagnostic_ranked_node_limit)
    ranked_nodes = (
        sorted(nodes, key=lambda node: float(node.score), reverse=True)[:ranked_limit]
        if ranked_limit > 0
        else []
    )
    diagnostics: dict[str, Any] = {
        "candidate_count": int(candidates.shape[0]),
        "separation_filtered_candidates": int(separation_filtered),
        "path_filtered_candidates": int(path_filtered),
        "program_count": int(len(programs)),
        "program_library_policy": (
            "forced_predeclared_baseline"
            if cfg.forced_program_pair_ids is not None
            else "balanced_multi_partition_predeclared_action_set"
        ),
        "program_library_global_optimality_claimed": False,
        "program_library_exact_eig_over_every_predeclared_action": bool(
            int(shortlist_diagnostics.get("exact_action_count", 0))
            == int(shortlist_diagnostics.get("total_action_count", 0))
        ),
        "program_library_unique_pair_count": int(
            np.count_nonzero(pair_occurrences)
        ),
        "program_library_pair_occurrence_min": int(
            np.min(positive_occurrences)
        ),
        "program_library_pair_occurrence_max": int(
            np.max(positive_occurrences)
        ),
        "program_library_companion_diversity_min": int(
            min(
                (len(companions) for companions in companion_sets.values()),
                default=0,
            )
        ),
        "program_library_companion_diversity_max": int(
            max(
                (len(companions) for companions in companion_sets.values()),
                default=0,
            )
        ),
        "continuous_height_bounds_m": (
            None
            if continuous_height_bounds_m is None
            else [float(value) for value in continuous_height_bounds_m]
        ),
        "evaluated_candidate_count": int(len({int(node.pose_index) for node in nodes})),
        "node_count": int(len(nodes)),
        "mode_count": int(mode_count),
        "max_modes_per_isotope": int(cfg.max_modes_per_isotope),
        "pf_max_sources": int(pf_max_sources),
        "planner_belief_sources": ["pf_posterior"],
        "planner_official_posterior_projection": dict(
            official_snapshot_diagnostics
        ),
        "planner_geometry_mode_projection": {
            "source": "full_aligned_joint_posterior",
            "particle_count": int(
                np.asarray(geometry_joint_particles.weights_n).size
            ),
            "mass_semantics": (
                "unconditional_particle_mass_with_k_zero_preserved"
            ),
            "position_representative": "intrinsic_surface_weighted_medoid",
            "synthetic_xyz_centroids": False,
        },
        "planning_policy": "one_step_joint_eig",
        "first_program_kind": first.program.kind,
        "planning_eig_joint_program_views": True,
        "planning_eig_joint_isotope_vector": True,
        "planning_eig_aligned_joint_posterior_snapshot": True,
        "planning_eig_raw_spectrum_observations": True,
        "planning_eig_persistent_named_rng": True,
        "planning_eig_all_valid_candidates_exact": bool(
            int(shortlist_diagnostics.get("exact_action_count", 0))
            == int(shortlist_diagnostics.get("total_action_count", 0))
        ),
        "planning_eig_batched_source_line_response": True,
        "planning_eig_action_memory_budget_bytes": 256 * 1024 * 1024,
        "planning_eig_likelihood_model": "joint_full_spectrum_generative",
        "planning_eig_contract_hash_sha256": str(
            estimator.full_spectrum_generative_model.contract_hash_sha256
        ),
        "planning_eig_observation_semantics": (
            "same_full_spectrum_predictive_sampler_and_log_likelihood_as_pf"
        ),
        "planning_eig_shortlist": dict(shortlist_diagnostics),
        "first_information_gain": float(first.information_gain),
        "first_coverage_gain": float(first.coverage_gain),
        "coverage_support": str(
            shortlist_diagnostics.get(
                "coverage_support",
                "unavailable",
            )
        ),
        "coverage_quadrature": shortlist_diagnostics.get(
            "coverage_quadrature"
        ),
        "coverage_sample_count": int(
            (
                shortlist_diagnostics.get("coverage_quadrature") or {}
            ).get(
                "sample_count",
                cfg.coverage_surface_quadrature_max_points,
            )
        ),
        "first_revisit_penalty": float(first.revisit_penalty),
        "first_bearing_diversity_gain": float(first.bearing_diversity_gain),
        "first_frontier_gain": float(first.frontier_gain),
        "first_turn_penalty": float(first.turn_penalty),
        "first_local_orbit_gain": float(first.local_orbit_gain),
        "first_elevation_condition_gain": float(first.elevation_condition_gain),
        "diagnostic_ranked_node_limit": int(ranked_limit),
        "component_leaders": _component_leader_payloads(nodes),
        "ranked_nodes": [
            _node_diagnostic_payload(node, rank)
            for rank, node in enumerate(ranked_nodes, start=1)
        ],
    }
    return DSSPPResult(
        next_pose=first.pose_xyz.copy(),
        next_pose_index=int(first.pose_index),
        shield_program=first.program,
        score=best_score,
        sequence=tuple(sequence),
        diagnostics=diagnostics,
    )
