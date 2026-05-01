"""Differential Shield-Signature Path Planning.

DSS-PP plans over a joint robot-pose and shield-program action. The module
uses the same PF expected-count kernel as the online estimator; it does not
generate spectra or replace Geant4 transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from measurement.continuous_kernels import ContinuousKernel
from pf.estimator import RotatingShieldPFEstimator
from planning.pose_selection import (
    _auto_scale_observation_penalty,
    _minimum_observation_feasible_mask,
    estimate_lambda_cost,
    minimum_observation_shortfall,
)
from planning.traversability import shortest_grid_path_length


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


@dataclass(frozen=True)
class DSSPPConfig:
    """Configuration for Differential Shield-Signature Path Planning."""

    horizon: int = 2
    beam_width: int = 8
    max_programs: int = 24
    program_length: int = 2
    mode_cluster_radius_m: float = 1.5
    max_modes_per_isotope: int = 4
    planning_particles: int | None = None
    planning_method: str | None = None
    live_time_s: float = 30.0
    lambda_eig: float = 1.0
    lambda_signature: float = 1.0
    lambda_distance: float | None = None
    lambda_time: float = 0.0
    lambda_rotation: float = 0.15
    lambda_dose: float = 0.0
    lambda_coverage: float = 0.0
    lambda_bearing_diversity: float = 0.0
    lambda_frontier: float = 0.0
    lambda_turn_smoothness: float = 0.0
    eta_observation: float = 1.0
    eta_differential: float = 1.0
    eta_count_balance: float = 0.5
    eta_revisit: float = 0.0
    min_observation_counts: float = 0.0
    enforce_min_observation: bool = True
    signature_std_min_counts: float = 1.0
    count_variance_floor: float = 1.0
    coverage_radius_m: float = 3.0
    coverage_grid_max_cells: int = 5000
    coverage_floor_quantile: float = 0.0
    coverage_floor_weight: float = 0.0
    min_station_separation_m: float = 0.0
    detector_aperture_samples: int = 1
    robot_speed_m_s: float = 0.5
    rotation_overhead_s: float = 0.5
    augment_candidates: bool = True
    max_augmented_candidates: int = 256
    ring_radii_m: tuple[float, ...] = (2.0, 3.5, 5.0)
    ring_angles: int = 12
    rng_seed: int | None = 0
    eig_candidate_limit: int | None = 64


@dataclass(frozen=True)
class DSSPPNode:
    """Store one candidate station and shield program evaluation."""

    pose_index: int
    pose_xyz: NDArray[np.float64]
    program: ShieldProgram
    score: float
    static_score: float
    distance_weight: float
    observation_penalty_weight: float
    information_gain: float
    signature_score: float
    observation_penalty: float
    count_balance_penalty: float
    differential_penalty: float
    dose_score: float
    coverage_gain: float
    revisit_penalty: float
    bearing_diversity_gain: float
    frontier_gain: float
    turn_penalty: float


@dataclass(frozen=True)
class DSSPPResult:
    """Return the selected receding-horizon DSS-PP action."""

    next_pose: NDArray[np.float64]
    next_pose_index: int
    shield_program: ShieldProgram
    score: float
    sequence: tuple[DSSPPNode, ...]
    diagnostics: dict[str, float | int | str]


def _normalise_weights(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return normalized nonnegative weights with a uniform fallback."""
    arr = np.asarray(weights, dtype=float).ravel()
    if arr.size == 0:
        return arr
    arr = np.maximum(arr, 0.0)
    total = float(np.sum(arr))
    if total <= 0.0:
        return np.ones(arr.size, dtype=float) / float(arr.size)
    return arr / total


def _pair_id(fe_index: int, pb_index: int, num_orients: int) -> int:
    """Return the flattened orientation-pair id."""
    return int(fe_index) * int(num_orients) + int(pb_index)


def _pair_indices(pair_id: int, num_orients: int) -> tuple[int, int]:
    """Return Fe and Pb indices from a flattened pair id."""
    return int(pair_id) // int(num_orients), int(pair_id) % int(num_orients)


def _opposite_indices(normals: NDArray[np.float64]) -> list[int]:
    """Return the approximately opposite octant index for each normal."""
    normal_arr = np.asarray(normals, dtype=float)
    opposite: list[int] = []
    for normal in normal_arr:
        dots = normal_arr @ normal
        opposite.append(int(np.argmin(dots)))
    return opposite


def build_shield_program_library(
    normals: NDArray[np.float64],
    *,
    program_length: int = 2,
    max_programs: int = 24,
) -> list[ShieldProgram]:
    """Build bearing, material, and occlusion-test shield programs."""
    normal_arr = np.asarray(normals, dtype=float)
    if normal_arr.ndim != 2 or normal_arr.shape[1] != 3:
        raise ValueError("normals must be shaped (N, 3).")
    num_orients = int(normal_arr.shape[0])
    length = max(1, int(program_length))
    opposite = _opposite_indices(normal_arr)
    programs: list[ShieldProgram] = []
    for idx in range(num_orients):
        opp = opposite[idx]
        blocked = _pair_id(idx, idx, num_orients)
        unblocked = _pair_id(opp, opp, num_orients)
        fe_only = _pair_id(idx, opp, num_orients)
        pb_only = _pair_id(opp, idx, num_orients)
        programs.append(
            ShieldProgram(
                name=f"bearing_split_{idx}",
                pair_ids=(blocked, unblocked)[:length],
                kind="bearing_split",
            )
        )
        programs.append(
            ShieldProgram(
                name=f"material_split_{idx}",
                pair_ids=(fe_only, pb_only)[:length],
                kind="material_split",
            )
        )
        programs.append(
            ShieldProgram(
                name=f"occlusion_test_{idx}",
                pair_ids=(unblocked, blocked, fe_only)[:length],
                kind="occlusion_test",
            )
        )
    deduped: dict[tuple[int, ...], ShieldProgram] = {}
    for program in programs:
        if program.pair_ids:
            deduped.setdefault(program.pair_ids, program)
    return list(deduped.values())[: max(1, int(max_programs))]


def _continuous_kernel_for_estimator(
    estimator: RotatingShieldPFEstimator,
    *,
    detector_aperture_samples: int | None = None,
) -> ContinuousKernel:
    """Build a ContinuousKernel matching the estimator."""
    aperture_samples = (
        int(getattr(estimator, "detector_aperture_samples", 1))
        if detector_aperture_samples is None
        else int(detector_aperture_samples)
    )
    return ContinuousKernel(
        mu_by_isotope=estimator.mu_by_isotope,
        shield_params=estimator.shield_params,
        obstacle_grid=getattr(estimator, "obstacle_grid", None),
        obstacle_height_m=float(getattr(estimator, "obstacle_height_m", 2.0)),
        obstacle_mu_by_isotope=getattr(estimator, "obstacle_mu_by_isotope", None),
        obstacle_buildup_coeff=float(getattr(estimator, "obstacle_buildup_coeff", 0.0)),
        detector_radius_m=float(getattr(estimator, "detector_radius_m", 0.0)),
        detector_aperture_samples=max(1, aperture_samples),
    )


def _cluster_source_samples(
    isotope: str,
    positions: list[NDArray[np.float64]],
    strengths: list[float],
    weights: list[float],
    *,
    radius_m: float,
    max_modes: int,
) -> list[SignatureMode]:
    """Cluster weighted posterior source samples into signature modes."""
    if not positions:
        return []
    pos_arr = np.asarray(positions, dtype=float)
    str_arr = np.asarray(strengths, dtype=float)
    w_arr = _normalise_weights(np.asarray(weights, dtype=float))
    order = np.argsort(w_arr)[::-1]
    clusters: list[list[int]] = []
    centers: list[NDArray[np.float64]] = []
    for idx in order:
        pos = pos_arr[int(idx)]
        assigned = False
        for cluster_idx, center in enumerate(centers):
            if float(np.linalg.norm(pos - center)) <= float(radius_m):
                clusters[cluster_idx].append(int(idx))
                cluster_weights = w_arr[clusters[cluster_idx]]
                centers[cluster_idx] = np.average(
                    pos_arr[clusters[cluster_idx]],
                    axis=0,
                    weights=cluster_weights,
                )
                assigned = True
                break
        if not assigned:
            clusters.append([int(idx)])
            centers.append(pos.copy())
    modes: list[SignatureMode] = []
    for cluster in clusters:
        cluster_weights = w_arr[cluster]
        cluster_weight_sum = float(np.sum(cluster_weights))
        if cluster_weight_sum <= 0.0:
            continue
        center = np.average(pos_arr[cluster], axis=0, weights=cluster_weights)
        strength = float(np.average(str_arr[cluster], weights=cluster_weights))
        spread = float(
            np.sqrt(
                np.average(
                    np.sum((pos_arr[cluster] - center[None, :]) ** 2, axis=1),
                    weights=cluster_weights,
                )
            )
        )
        modes.append(
            SignatureMode(
                isotope=isotope,
                position_xyz=center.astype(float),
                strength_cps_1m=max(strength, 0.0),
                weight=cluster_weight_sum,
                spread_m=spread,
            )
        )
    modes.sort(key=lambda mode: mode.weight, reverse=True)
    return modes[: max(1, int(max_modes))]


def extract_signature_modes(
    estimator: RotatingShieldPFEstimator,
    *,
    max_particles: int | None = None,
    method: str | None = None,
    mode_cluster_radius_m: float = 1.5,
    max_modes_per_isotope: int = 4,
) -> dict[str, list[SignatureMode]]:
    """Extract isotope-wise posterior source modes from PF particles."""
    particles = estimator.planning_particles(
        max_particles=max_particles,
        method=method,
    )
    modes_by_isotope: dict[str, list[SignatureMode]] = {}
    eps = 1e-12
    for isotope in estimator.isotopes:
        if isotope not in particles:
            modes_by_isotope[isotope] = []
            continue
        states, weights = particles[isotope]
        norm_weights = _normalise_weights(np.asarray(weights, dtype=float))
        positions: list[NDArray[np.float64]] = []
        strengths: list[float] = []
        sample_weights: list[float] = []
        for state, particle_weight in zip(states, norm_weights):
            num_sources = int(state.num_sources)
            if num_sources <= 0:
                continue
            state_strengths = np.maximum(
                np.asarray(state.strengths[:num_sources], dtype=float),
                0.0,
            )
            total_strength = float(np.sum(state_strengths))
            if total_strength <= eps:
                rel_strengths = (
                    np.ones(num_sources, dtype=float) / float(num_sources)
                )
            else:
                rel_strengths = state_strengths / total_strength
            for pos, strength, rel_strength in zip(
                state.positions[:num_sources],
                state_strengths,
                rel_strengths,
            ):
                positions.append(np.asarray(pos, dtype=float))
                strengths.append(float(strength))
                sample_weights.append(float(particle_weight) * float(rel_strength))
        modes_by_isotope[isotope] = _cluster_source_samples(
            isotope,
            positions,
            strengths,
            sample_weights,
            radius_m=mode_cluster_radius_m,
            max_modes=max_modes_per_isotope,
        )
    return modes_by_isotope


def _is_free(map_api: object | None, point: NDArray[np.float64]) -> bool:
    """Return True when point is in free space according to map_api."""
    if map_api is None:
        return True
    if callable(map_api):
        return bool(map_api(point))
    for attr in ("is_free", "is_free_space", "is_free_cell"):
        fn = getattr(map_api, attr, None)
        if callable(fn):
            try:
                return bool(fn(point))
            except TypeError:
                continue
    return True


def _cell_center(map_api: object, cell: tuple[int, int], z_value: float) -> NDArray[np.float64]:
    """Return the world-space center of a map cell."""
    fn = getattr(map_api, "cell_center", None)
    if callable(fn):
        x_val, y_val = fn(cell)
    else:
        origin = getattr(map_api, "origin", (0.0, 0.0))
        cell_size = float(getattr(map_api, "cell_size", 1.0))
        x_val = float(origin[0]) + (float(cell[0]) + 0.5) * cell_size
        y_val = float(origin[1]) + (float(cell[1]) + 0.5) * cell_size
    return np.array([float(x_val), float(y_val), float(z_value)], dtype=float)


def _bounds_filter(
    points: list[NDArray[np.float64]],
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]] | None,
    map_api: object | None,
) -> list[NDArray[np.float64]]:
    """Filter points by bounds and traversability."""
    filtered: list[NDArray[np.float64]] = []
    if bounds_xyz is None:
        lo = None
        hi = None
    else:
        lo = np.asarray(bounds_xyz[0], dtype=float)
        hi = np.asarray(bounds_xyz[1], dtype=float)
    for point in points:
        pt = np.asarray(point, dtype=float)
        if pt.shape != (3,):
            continue
        if lo is not None and hi is not None:
            if bool(np.any(pt < lo) or np.any(pt > hi)):
                continue
        if _is_free(map_api, pt):
            filtered.append(pt)
    return filtered


def _dedupe_points(
    points: Sequence[NDArray[np.float64]],
    *,
    decimals: int = 3,
) -> NDArray[np.float64]:
    """Return unique points while preserving first occurrence order."""
    seen: set[tuple[float, float, float]] = set()
    unique: list[NDArray[np.float64]] = []
    for point in points:
        pt = np.asarray(point, dtype=float)
        key = tuple(float(v) for v in np.round(pt, int(decimals)))
        if key in seen:
            continue
        seen.add(key)
        unique.append(pt)
    if not unique:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(unique).astype(float)


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
) -> NDArray[np.float64]:
    """Add posterior-ring, occlusion-boundary, and cross-bearing candidates."""
    base = np.asarray(candidate_poses_xyz, dtype=float)
    if base.ndim != 2 or base.shape[1] != 3:
        raise ValueError("candidate_poses_xyz must be shape (N, 3).")
    z_value = float(current_pose_xyz[2])
    generated: list[NDArray[np.float64]] = [row.copy() for row in base]
    all_modes = [
        mode
        for modes in modes_by_isotope.values()
        for mode in modes
        if mode.weight > 0.0
    ]
    all_modes.sort(key=lambda mode: mode.weight, reverse=True)
    top_modes = all_modes[: max(1, int(config.max_modes_per_isotope) * 2)]
    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        num=max(4, int(config.ring_angles)),
        endpoint=False,
    )
    for mode in top_modes:
        for radius in config.ring_radii_m:
            for angle in angles:
                point = np.array(
                    [
                        mode.position_xyz[0] + float(radius) * np.cos(angle),
                        mode.position_xyz[1] + float(radius) * np.sin(angle),
                        z_value,
                    ],
                    dtype=float,
                )
                generated.append(point)
    cells = getattr(map_api, "traversable_cells", None)
    if cells is None and hasattr(map_api, "blocked_cells"):
        blocked = set(getattr(map_api, "blocked_cells"))
        grid_shape = getattr(map_api, "grid_shape", (0, 0))
        free_neighbors: set[tuple[int, int]] = set()
        for ix, iy in blocked:
            for nb in ((ix - 1, iy), (ix + 1, iy), (ix, iy - 1), (ix, iy + 1)):
                if 0 <= nb[0] < grid_shape[0] and 0 <= nb[1] < grid_shape[1]:
                    if nb not in blocked:
                        free_neighbors.add(nb)
        cells = tuple(sorted(free_neighbors))
    if cells is not None:
        boundary_points = [_cell_center(map_api, tuple(cell), z_value) for cell in cells]
        if top_modes:
            ref = top_modes[0].position_xyz
            boundary_points.sort(key=lambda pt: float(np.linalg.norm(pt - ref)))
        generated.extend(boundary_points[: max(0, int(config.max_augmented_candidates) // 2)])
    coverage_points = _free_cell_centers(
        map_api,
        z_value=z_value,
        max_cells=max(0, int(config.max_augmented_candidates)),
        bounds_xyz=bounds_xyz,
    )
    if coverage_points.size:
        if visited_poses_xyz is not None:
            visited = np.asarray(visited_poses_xyz, dtype=float)
            if visited.ndim == 1 and visited.size == 3:
                visited = visited.reshape(1, 3)
            if visited.ndim == 2 and visited.shape[1] == 3 and visited.size:
                distances = np.linalg.norm(
                    coverage_points[:, None, :2] - visited[None, :, :2],
                    axis=2,
                )
                order = np.argsort(np.min(distances, axis=1))[::-1]
                coverage_points = coverage_points[order]
        generated.extend(
            [
                point.copy()
                for point in coverage_points[: max(0, int(config.max_augmented_candidates) // 2)]
            ]
        )
    if visited_poses_xyz is not None and top_modes:
        visited = np.asarray(visited_poses_xyz, dtype=float)
        if visited.ndim == 2 and visited.shape[1] == 3:
            for mode in top_modes:
                prior_angles = [
                    _bearing_angle_xy(mode.position_xyz, pose) for pose in visited
                ]
                for base_angle in prior_angles:
                    for offset in (0.5 * np.pi, -0.5 * np.pi, np.pi):
                        angle = base_angle + offset
                        for radius in config.ring_radii_m:
                            generated.append(
                                np.array(
                                    [
                                        mode.position_xyz[0]
                                        + float(radius) * np.cos(angle),
                                        mode.position_xyz[1]
                                        + float(radius) * np.sin(angle),
                                        z_value,
                                    ],
                                    dtype=float,
                                )
                            )
    filtered = _bounds_filter(generated, bounds_xyz, map_api)
    deduped = _dedupe_points(filtered)
    limit = max(base.shape[0], int(config.max_augmented_candidates))
    return deduped[:limit]


def _expected_signature(
    *,
    kernel: ContinuousKernel,
    estimator: RotatingShieldPFEstimator,
    mode: SignatureMode,
    pose_xyz: NDArray[np.float64],
    program: ShieldProgram,
    num_orients: int,
    live_time_s: float,
) -> NDArray[np.float64]:
    """Return a source-mode shield-signature count vector."""
    values: list[float] = []
    source_scale = estimator.response_scale_for_isotope(mode.isotope)
    for pair_id in program.pair_ids:
        fe_index, pb_index = _pair_indices(pair_id, num_orients)
        kernel_value = kernel.kernel_value_pair(
            isotope=mode.isotope,
            detector_pos=pose_xyz,
            source_pos=mode.position_xyz,
            fe_index=fe_index,
            pb_index=pb_index,
        )
        count = (
            float(live_time_s)
            * float(mode.strength_cps_1m)
            * float(source_scale)
            * float(kernel_value)
        )
        values.append(max(count, 0.0))
    return np.asarray(values, dtype=float)


def _weighted_mean_signature(
    signatures: list[NDArray[np.float64]],
    weights: list[float],
) -> NDArray[np.float64]:
    """Return the weighted mean signature vector."""
    if not signatures:
        return np.zeros(0, dtype=float)
    sig_arr = np.vstack(signatures)
    weight_arr = _normalise_weights(np.asarray(weights, dtype=float))
    return np.sum(weight_arr[:, None] * sig_arr, axis=0)


def _signature_separation_score(
    signatures: list[NDArray[np.float64]],
    *,
    variance_floor: float,
) -> float:
    """Return the worst-pair Mahalanobis shield-signature separation."""
    if len(signatures) < 2:
        return 0.0
    floor = max(float(variance_floor), 1e-12)
    best_worst = np.inf
    for idx in range(len(signatures)):
        for jdx in range(idx + 1, len(signatures)):
            left = signatures[idx]
            right = signatures[jdx]
            diff = left - right
            variance = np.maximum(left + right, floor)
            distance = float(np.sum(diff * diff / variance))
            best_worst = min(best_worst, distance)
    if not np.isfinite(best_worst):
        return 0.0
    return max(float(best_worst), 0.0)


def _count_balance_penalty(counts: dict[str, float]) -> float:
    """Return an isotope-agnostic penalty for single-isotope dominated programs."""
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


def _score_program(
    *,
    estimator: RotatingShieldPFEstimator,
    kernel: ContinuousKernel,
    modes_by_isotope: dict[str, list[SignatureMode]],
    pose_xyz: NDArray[np.float64],
    program: ShieldProgram,
    config: DSSPPConfig,
    ) -> tuple[float, float, float, float, float]:
    """Return signature, observation, balance, differential, and dose terms."""
    num_orients = int(estimator.num_orientations)
    isotope_weights = estimator.pf_config.alpha_weights or {
        isotope: 1.0 for isotope in estimator.isotopes
    }
    alpha_sum = sum(float(v) for v in isotope_weights.values()) or 1.0
    signature_total = 0.0
    differential_terms: list[float] = []
    observation_counts: dict[str, float] = {}
    balance_counts: dict[str, float] = {}
    dose_score = 0.0
    for isotope in estimator.isotopes:
        modes = modes_by_isotope.get(isotope, [])
        signatures: list[NDArray[np.float64]] = []
        weights: list[float] = []
        for mode in modes:
            signature = _expected_signature(
                kernel=kernel,
                estimator=estimator,
                mode=mode,
                pose_xyz=pose_xyz,
                program=program,
                num_orients=num_orients,
                live_time_s=config.live_time_s,
            )
            signatures.append(signature)
            weights.append(float(mode.weight))
        if signatures:
            signature_total += (
                float(isotope_weights.get(isotope, 1.0))
                / alpha_sum
                * _signature_separation_score(
                    signatures,
                    variance_floor=config.count_variance_floor,
                )
            )
            mean_signature = _weighted_mean_signature(signatures, weights)
            observation_counts[isotope] = (
                float(np.max(mean_signature)) if mean_signature.size else 0.0
            )
            balance_counts[isotope] = (
                float(np.sum(mean_signature)) if mean_signature.size else 0.0
            )
            dose_score += float(np.sum(mean_signature))
            signature_std = float(np.std(mean_signature)) if mean_signature.size else 0.0
        else:
            observation_counts[isotope] = 0.0
            balance_counts[isotope] = 0.0
            signature_std = 0.0
        min_std = max(float(config.signature_std_min_counts), 0.0)
        if min_std > 0.0:
            shortfall = max(0.0, 1.0 - signature_std / min_std)
            differential_terms.append(shortfall * shortfall)
    observation_penalty = minimum_observation_shortfall(
        observation_counts,
        min_counts=float(config.min_observation_counts),
    )
    differential_penalty = (
        float(np.mean(differential_terms)) if differential_terms else 0.0
    )
    count_balance_penalty = _count_balance_penalty(balance_counts)
    return (
        signature_total,
        observation_penalty,
        count_balance_penalty,
        differential_penalty,
        dose_score,
    )


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
    cell_index = getattr(map_api, "cell_index", None)
    if not callable(cell_index):
        return float(np.linalg.norm(goal - start))
    start_cell = cell_index(start)
    goal_cell = cell_index(goal)
    if start_cell is None or goal_cell is None:
        return float(np.linalg.norm(goal - start))
    return shortest_grid_path_length(map_api, start, goal, allow_diagonal=True)


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
    reachable = np.asarray(
        [
            np.isfinite(_node_path_length(map_api, current_pose_xyz, candidate))
            for candidate in candidates
        ],
        dtype=bool,
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
    if traversable_cells is not None:
        cells = [tuple(cell) for cell in traversable_cells]
    else:
        is_free_cell = getattr(map_api, "is_free_cell", None)
        if not callable(is_free_cell):
            is_free_cell = getattr(map_api, "is_cell_free", None)
        if not callable(is_free_cell):
            return np.zeros((0, 3), dtype=float)
        cells = [
            (ix, iy)
            for ix in range(int(grid_shape[0]))
            for iy in range(int(grid_shape[1]))
            if bool(is_free_cell((ix, iy)))
        ]
    if not cells:
        return np.zeros((0, 3), dtype=float)
    max_count = max(0, int(max_cells))
    if max_count > 0 and len(cells) > max_count:
        indices = np.linspace(0, len(cells) - 1, max_count, dtype=int)
        cells = [cells[int(idx)] for idx in indices]
    centers = [_cell_center(map_api, tuple(cell), z_value) for cell in cells]
    return np.vstack(centers).astype(float)


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
    """Return newly covered free-space fraction for a candidate station."""
    centers = np.asarray(cell_centers_xyz, dtype=float)
    if centers.size == 0:
        return 0.0
    radius = max(float(radius_m), 0.0)
    if radius <= 0.0:
        return 0.0
    candidate = np.asarray(candidate_pose_xyz, dtype=float).reshape(3)
    candidate_dist = np.linalg.norm(centers[:, :2] - candidate[None, :2], axis=1)
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
                centers[:, None, :2] - visited[None, :, :2],
                axis=2,
            )
            visited_covered = np.min(visited_dist, axis=1) <= radius
    newly_covered = candidate_covered & ~visited_covered
    return float(np.count_nonzero(newly_covered)) / float(centers.shape[0])


def _station_revisit_penalty(
    candidate_pose_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    *,
    min_separation_m: float,
) -> float:
    """Return a normalized penalty for selecting a near-visited station."""
    min_sep = max(float(min_separation_m), 0.0)
    if min_sep <= 0.0 or visited_poses_xyz is None:
        return 0.0
    visited = np.asarray(visited_poses_xyz, dtype=float)
    if visited.ndim == 1 and visited.size == 3:
        visited = visited.reshape(1, 3)
    if visited.ndim != 2 or visited.shape[1] != 3 or visited.size == 0:
        return 0.0
    candidate = np.asarray(candidate_pose_xyz, dtype=float).reshape(3)
    min_dist = float(np.min(np.linalg.norm(visited[:, :2] - candidate[None, :2], axis=1)))
    if min_dist >= min_sep:
        return 0.0
    shortfall = 1.0 - min_dist / max(min_sep, 1e-12)
    return float(shortfall * shortfall)


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
            _bearing_angle_xy(mode.position_xyz, candidate)
            for mode in active
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
                    _bearing_angle_xy(mode.position_xyz, pose)
                    for pose in visited
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
        weights.append(float(sum(mode.weight for mode in active)))
    if not gains:
        return 0.0
    weight_arr = _normalise_weights(np.asarray(weights, dtype=float))
    return float(np.sum(weight_arr * np.asarray(gains, dtype=float)))


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
    nearest = float(np.min(np.linalg.norm(visited[:, :2] - candidate[None, :2], axis=1)))
    return float(np.exp(-((nearest - target) / target) ** 2))


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
        and float(np.linalg.norm(visited[-1, :2] - current[:2])) < 1.0e-6
    ):
        previous = visited[-2]
    else:
        previous = visited[-1]
    prev_vec = current[:2] - previous[:2]
    next_vec = np.asarray(candidate_pose_xyz, dtype=float).reshape(3)[:2] - current[:2]
    prev_norm = float(np.linalg.norm(prev_vec))
    next_norm = float(np.linalg.norm(next_vec))
    if prev_norm <= 1.0e-9 or next_norm <= 1.0e-9:
        return 0.0
    dot = float(np.clip(np.dot(prev_vec, next_vec) / (prev_norm * next_norm), -1.0, 1.0))
    return float(0.5 * (1.0 - dot))


def _filter_station_separation(
    candidate_poses_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    *,
    min_separation_m: float,
) -> tuple[NDArray[np.float64], int]:
    """Remove near-revisited stations when at least one unvisited option exists."""
    min_sep = max(float(min_separation_m), 0.0)
    candidates = np.asarray(candidate_poses_xyz, dtype=float)
    if candidates.size == 0 or min_sep <= 0.0 or visited_poses_xyz is None:
        return candidates, 0
    visited = np.asarray(visited_poses_xyz, dtype=float)
    if visited.ndim == 1 and visited.size == 3:
        visited = visited.reshape(1, 3)
    if visited.ndim != 2 or visited.shape[1] != 3 or visited.size == 0:
        return candidates, 0
    distances = np.linalg.norm(
        candidates[:, None, :2] - visited[None, :, :2],
        axis=2,
    )
    keep = np.min(distances, axis=1) >= min_sep
    if not np.any(keep):
        return candidates, 0
    removed = int(np.count_nonzero(~keep))
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
    travel_time = path_length / max(float(config.robot_speed_m_s), 1e-9)
    time_cost = (
        travel_time
        + len(node.program.pair_ids)
        * (float(config.rotation_overhead_s) + float(config.live_time_s))
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
        - float(node.observation_penalty_weight) * float(node.observation_penalty)
    )
    return float(score), float(path_length)


def _candidate_information_gain(
    estimator: RotatingShieldPFEstimator,
    pose_xyz: NDArray[np.float64],
    *,
    config: DSSPPConfig,
    rng_seed: int | None,
) -> float:
    """Return candidate-level predicted log-uncertainty reduction."""
    if float(config.lambda_eig) <= 0.0:
        return 0.0
    try:
        current_uncertainty = max(float(estimator.global_uncertainty()), 0.0)
        after_uncertainty = float(
            estimator.expected_uncertainty_after_rotation(
                pose_xyz=pose_xyz,
                live_time_per_rot_s=float(config.live_time_s),
                tau_ig=float(estimator.pf_config.ig_threshold),
                tmax_s=float(config.live_time_s)
                * max(1, int(config.program_length)),
                n_rollouts=0,
                orient_selection="IG",
                rng_seed=rng_seed,
            )
        )
    except RuntimeError:
        return 0.0
    current_information = float(np.log1p(current_uncertainty))
    after_information = float(np.log1p(max(after_uncertainty, 0.0)))
    return max(current_information - after_information, 0.0)


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
) -> list[DSSPPNode]:
    """Evaluate all station-program nodes for the first horizon layer."""
    kernel = _continuous_kernel_for_estimator(
        estimator,
        detector_aperture_samples=max(1, int(config.detector_aperture_samples)),
    )
    candidate_poses = np.asarray(candidate_poses_xyz, dtype=float)
    if candidate_poses.ndim != 2 or candidate_poses.shape[1] != 3:
        raise ValueError("candidate_poses_xyz must be shape (N, 3).")
    info_gains = np.zeros(candidate_poses.shape[0], dtype=float)
    path_lengths = np.asarray(
        [
            _node_path_length(map_api, current_pose_xyz, pose)
            for pose in candidate_poses
        ],
        dtype=float,
    )
    free_cell_centers = _free_cell_centers(
        map_api,
        z_value=float(current_pose_xyz[2]),
        max_cells=int(config.coverage_grid_max_cells),
        bounds_xyz=bounds_xyz,
    )
    coverage_raw = np.asarray(
        [
            _coverage_gain_fraction(
                cell_centers_xyz=free_cell_centers,
                candidate_pose_xyz=pose,
                visited_poses_xyz=visited_poses_xyz,
                radius_m=float(config.coverage_radius_m),
            )
            for pose in candidate_poses
        ],
        dtype=float,
    )
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
                    np.clip(coverage_floor_quantile, 0.0, 1.0),
                )
            )
    revisit_penalties = np.asarray(
        [
            _station_revisit_penalty(
                pose,
                visited_poses_xyz,
                min_separation_m=float(config.min_station_separation_m),
            )
            for pose in candidate_poses
        ],
        dtype=float,
    )
    bearing_gains = np.asarray(
        [
            _bearing_diversity_gain(
                pose,
                visited_poses_xyz,
                modes_by_isotope,
            )
            for pose in candidate_poses
        ],
        dtype=float,
    )
    frontier_target = max(
        float(config.min_station_separation_m),
        float(config.coverage_radius_m),
    )
    frontier_gains = np.asarray(
        [
            _frontier_band_gain(
                pose,
                visited_poses_xyz,
                target_radius_m=frontier_target,
            )
            for pose in candidate_poses
        ],
        dtype=float,
    )
    turn_penalties = np.asarray(
        [
            _route_turn_penalty(
                pose,
                current_pose_xyz,
                visited_poses_xyz,
            )
            for pose in candidate_poses
        ],
        dtype=float,
    )
    finite_path = np.isfinite(path_lengths)
    if config.lambda_distance is None:
        lambda_distance = estimate_lambda_cost(
            -info_gains,
            path_lengths[finite_path] if np.any(finite_path) else path_lengths,
            method="range",
        )
    else:
        lambda_distance = float(config.lambda_distance)
    raw_nodes: list[DSSPPNode] = []
    static_scores: list[float] = []
    observation_penalties: list[float] = []
    cheap_pose_scores = np.full(candidate_poses.shape[0], -np.inf, dtype=float)
    pending: list[
        tuple[
            int,
            NDArray[np.float64],
            ShieldProgram,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
        ]
    ] = []
    for pose_index, pose in enumerate(candidate_poses):
        if not np.isfinite(path_lengths[pose_index]):
            continue
        for program in programs:
            (
                signature_score,
                observation_penalty,
                count_balance_penalty,
                differential_penalty,
                dose_score,
            ) = _score_program(
                estimator=estimator,
                kernel=kernel,
                modes_by_isotope=modes_by_isotope,
                pose_xyz=pose,
                program=program,
                config=config,
            )
            static_score = (
                float(config.lambda_signature)
                * float(np.log1p(max(signature_score, 0.0)))
                + float(config.lambda_coverage) * float(coverage_norm[pose_index])
                + float(config.lambda_bearing_diversity) * float(bearing_gains[pose_index])
                + float(config.lambda_frontier) * float(frontier_gains[pose_index])
                - float(config.lambda_dose) * dose_score
                - float(config.eta_count_balance) * count_balance_penalty
                - float(config.eta_differential) * differential_penalty
                - float(config.eta_revisit) * float(revisit_penalties[pose_index])
                - float(config.lambda_turn_smoothness) * float(turn_penalties[pose_index])
                - float(config.coverage_floor_weight)
                * max(0.0, coverage_floor - float(coverage_norm[pose_index])) ** 2
            )
            cheap_pose_scores[pose_index] = max(
                float(cheap_pose_scores[pose_index]),
                float(static_score),
            )
            static_scores.append(static_score)
            observation_penalties.append(observation_penalty)
            pending.append(
                (
                    pose_index,
                    pose.copy(),
                    program,
                    0.0,
                    signature_score,
                    observation_penalty,
                    count_balance_penalty,
                    differential_penalty,
                    dose_score,
                    float(coverage_raw[pose_index]),
                    float(revisit_penalties[pose_index]),
                    float(bearing_gains[pose_index]),
                    float(frontier_gains[pose_index]),
                    float(turn_penalties[pose_index]),
                )
            )
    if not pending:
        return []
    if float(config.lambda_eig) > 0.0:
        valid_pose_indices = np.flatnonzero(
            np.isfinite(cheap_pose_scores) & np.isfinite(path_lengths)
        )
        eig_limit = config.eig_candidate_limit
        if eig_limit is None or int(eig_limit) <= 0:
            selected_pose_indices = valid_pose_indices
        else:
            limit = min(int(eig_limit), int(valid_pose_indices.size))
            order = np.argsort(cheap_pose_scores[valid_pose_indices])[::-1]
            selected_pose_indices = valid_pose_indices[order[:limit]]
        for pose_index in selected_pose_indices:
            info_gains[int(pose_index)] = _candidate_information_gain(
                estimator,
                candidate_poses[int(pose_index)],
                config=config,
                rng_seed=None
                if config.rng_seed is None
                else int(config.rng_seed) + int(pose_index),
            )
        static_scores = [
            float(score) + float(config.lambda_eig) * float(info_gains[item[0]])
            for score, item in zip(static_scores, pending)
        ]
    obs_arr = np.asarray(observation_penalties, dtype=float)
    feasible_mask = _minimum_observation_feasible_mask(
        obs_arr,
        float(config.min_observation_counts),
    )
    if bool(config.enforce_min_observation) and np.any(feasible_mask):
        keep_indices = [
            idx for idx, feasible in enumerate(feasible_mask) if bool(feasible)
        ]
        pending = [pending[idx] for idx in keep_indices]
        static_scores = [static_scores[idx] for idx in keep_indices]
        observation_penalties = [
            observation_penalties[idx] for idx in keep_indices
        ]
        obs_arr = np.asarray(observation_penalties, dtype=float)
    obs_weight = _auto_scale_observation_penalty(
        np.asarray(static_scores, dtype=float),
        obs_arr,
        scale=float(config.eta_observation),
    )
    for item, base_score, observation_penalty in zip(
        pending,
        static_scores,
        observation_penalties,
    ):
        (
            pose_index,
            pose,
            program,
            _info_gain,
            signature_score,
            _obs_penalty,
            count_balance_penalty,
            differential_penalty,
            dose_score,
            coverage_gain,
            revisit_penalty,
            bearing_gain,
            frontier_gain,
            turn_penalty,
        ) = item
        info_gain = float(info_gains[pose_index])
        placeholder_node = DSSPPNode(
            pose_index=int(pose_index),
            pose_xyz=pose,
            program=program,
            score=0.0,
            static_score=float(base_score),
            distance_weight=float(lambda_distance),
            observation_penalty_weight=float(obs_weight),
            information_gain=float(info_gain),
            signature_score=float(signature_score),
            observation_penalty=float(observation_penalty),
            count_balance_penalty=float(count_balance_penalty),
            differential_penalty=float(differential_penalty),
            dose_score=float(dose_score),
            coverage_gain=float(coverage_gain),
            revisit_penalty=float(revisit_penalty),
            bearing_diversity_gain=float(bearing_gain),
            frontier_gain=float(frontier_gain),
            turn_penalty=float(turn_penalty),
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
                pose_index=int(pose_index),
                pose_xyz=pose,
                program=program,
                score=score,
                static_score=float(base_score),
                distance_weight=float(lambda_distance),
                observation_penalty_weight=float(obs_weight),
                information_gain=float(info_gain),
                signature_score=float(signature_score),
                observation_penalty=float(observation_penalty),
                count_balance_penalty=float(count_balance_penalty),
                differential_penalty=float(differential_penalty),
                dose_score=float(dose_score),
                coverage_gain=float(coverage_gain),
                revisit_penalty=float(revisit_penalty),
                bearing_diversity_gain=float(bearing_gain),
                frontier_gain=float(frontier_gain),
                turn_penalty=float(turn_penalty),
            )
        )
    raw_nodes.sort(key=lambda node: node.score, reverse=True)
    return raw_nodes


def _beam_search_sequence(
    nodes: Sequence[DSSPPNode],
    *,
    current_pose_xyz: NDArray[np.float64],
    current_pair_id: int | None,
    estimator: RotatingShieldPFEstimator,
    map_api: object | None,
    config: DSSPPConfig,
) -> tuple[DSSPPNode, ...]:
    """Return a receding-horizon node sequence using beam search."""
    if not nodes:
        return tuple()
    horizon = max(1, int(config.horizon))
    beam_width = max(1, int(config.beam_width))
    beam: list[tuple[float, tuple[DSSPPNode, ...], NDArray[np.float64], int | None]] = [
        (0.0, tuple(), np.asarray(current_pose_xyz, dtype=float), current_pair_id)
    ]
    expansion_nodes = list(nodes[: max(beam_width * 4, beam_width)])
    for _depth in range(horizon):
        next_beam: list[
            tuple[float, tuple[DSSPPNode, ...], NDArray[np.float64], int | None]
        ] = []
        for cumulative, sequence, prev_pose, prev_pair in beam:
            for node in expansion_nodes:
                transition_score, path_len = _compose_transition_score(
                    node=node,
                    previous_pose_xyz=prev_pose,
                    previous_pair_id=prev_pair,
                    estimator=estimator,
                    map_api=map_api,
                    config=config,
                )
                if not np.isfinite(path_len):
                    continue
                score = cumulative + transition_score
                next_pair = node.program.pair_ids[-1] if node.program.pair_ids else prev_pair
                next_beam.append(
                    (
                        score,
                        sequence + (node,),
                        node.pose_xyz,
                        int(next_pair) if next_pair is not None else None,
                    )
                )
        if not next_beam:
            break
        next_beam.sort(key=lambda item: item[0], reverse=True)
        beam = next_beam[:beam_width]
    if not beam:
        return tuple(nodes[:1])
    return beam[0][1]


def select_dss_pp_next_station(
    estimator: RotatingShieldPFEstimator,
    candidate_poses_xyz: NDArray[np.float64],
    current_pose_xyz: NDArray[np.float64],
    *,
    current_pair_id: int | None = None,
    visited_poses_xyz: NDArray[np.float64] | None = None,
    map_api: object | None = None,
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    config: DSSPPConfig | None = None,
) -> DSSPPResult:
    """Select the next station and shield program using DSS-PP."""
    cfg = config or DSSPPConfig()
    current_pose = np.asarray(current_pose_xyz, dtype=float)
    if current_pose.shape != (3,):
        raise ValueError("current_pose_xyz must be shape (3,).")
    modes = extract_signature_modes(
        estimator,
        max_particles=cfg.planning_particles,
        method=cfg.planning_method,
        mode_cluster_radius_m=float(cfg.mode_cluster_radius_m),
        max_modes_per_isotope=int(cfg.max_modes_per_isotope),
    )
    candidates = np.asarray(candidate_poses_xyz, dtype=float)
    base_candidates = candidates.copy()
    if cfg.augment_candidates:
        candidates = augment_candidate_stations(
            candidates,
            modes_by_isotope=modes,
            current_pose_xyz=current_pose,
            visited_poses_xyz=visited_poses_xyz,
            map_api=map_api,
            bounds_xyz=bounds_xyz,
            config=cfg,
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
    fallback_used = False
    if candidates.size == 0 and base_candidates.size != 0:
        candidates, base_path_filtered = _filter_path_reachable_stations(
            base_candidates,
            current_pose_xyz=current_pose,
            map_api=map_api,
        )
        path_filtered += int(base_path_filtered)
        fallback_used = candidates.size != 0
    programs = build_shield_program_library(
        estimator.normals,
        program_length=int(cfg.program_length),
        max_programs=int(cfg.max_programs),
    )
    nodes = _build_nodes(
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
    )
    if not nodes:
        raise ValueError("DSS-PP could not evaluate any station-program node.")
    sequence = _beam_search_sequence(
        nodes,
        current_pose_xyz=current_pose,
        current_pair_id=current_pair_id,
        estimator=estimator,
        map_api=map_api,
        config=cfg,
    )
    if not sequence:
        sequence = (nodes[0],)
    first = sequence[0]
    best_score = float(sum(node.score for node in sequence))
    mode_count = sum(len(mode_list) for mode_list in modes.values())
    diagnostics: dict[str, float | int | str] = {
        "candidate_count": int(candidates.shape[0]),
        "separation_filtered_candidates": int(separation_filtered),
        "path_filtered_candidates": int(path_filtered),
        "path_fallback_used": int(fallback_used),
        "program_count": int(len(programs)),
        "node_count": int(len(nodes)),
        "mode_count": int(mode_count),
        "horizon": int(max(1, cfg.horizon)),
        "beam_width": int(max(1, cfg.beam_width)),
        "first_program_kind": first.program.kind,
        "first_information_gain": float(first.information_gain),
        "first_signature_score": float(first.signature_score),
        "first_observation_penalty": float(first.observation_penalty),
        "first_count_balance_penalty": float(first.count_balance_penalty),
        "first_differential_penalty": float(first.differential_penalty),
        "first_dose_score": float(first.dose_score),
        "first_coverage_gain": float(first.coverage_gain),
        "first_revisit_penalty": float(first.revisit_penalty),
        "first_bearing_diversity_gain": float(first.bearing_diversity_gain),
        "first_frontier_gain": float(first.frontier_gain),
        "first_turn_penalty": float(first.turn_penalty),
    }
    return DSSPPResult(
        next_pose=first.pose_xyz.copy(),
        next_pose_index=int(first.pose_index),
        shield_program=first.program,
        score=best_score,
        sequence=tuple(sequence),
        diagnostics=diagnostics,
    )
