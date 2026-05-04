"""Differential Shield-Signature Path Planning.

DSS-PP plans over a joint robot-pose and shield-program action. The module
uses the same PF expected-count kernel as the online estimator; it does not
generate spectra or replace Geant4 transport.
"""

from __future__ import annotations

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Sequence, cast

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


_DSS_PP_POSE_EVAL_CONTEXT: dict[str, object] | None = None


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
    lambda_temporal_separation: float = 0.0
    lambda_count_utility: float = 0.0
    lambda_local_orbit: float = 0.0
    lambda_station_condition: float = 0.0
    residual_signature_weight: float = 1.0
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
    count_utility_saturation_counts: float = 250.0
    local_orbit_sigma_m: float = 0.75
    station_condition_ridge: float = 1.0e-3
    rng_seed: int | None = 0
    eig_candidate_limit: int | None = 64
    temporal_cover_weight: float = 1.0
    temporal_logdet_weight: float = 0.25
    temporal_decorrelation_weight: float = 0.5
    temporal_pair_contrast_threshold: float = 0.25
    temporal_logdet_ridge: float = 1.0e-3
    temporal_cover_programs: int = 1
    program_eval_workers: int | None = None


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
    temporal_separation_score: float
    observation_penalty: float
    count_balance_penalty: float
    differential_penalty: float
    dose_score: float
    count_utility: float
    coverage_gain: float
    revisit_penalty: float
    bearing_diversity_gain: float
    frontier_gain: float
    turn_penalty: float
    local_orbit_gain: float
    station_condition_gain: float


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


def _cycle_program_pairs(pair_ids: Sequence[int], length: int) -> tuple[int, ...]:
    """Return a non-empty pair-id sequence repeated to the requested length."""
    base = tuple(int(pair_id) for pair_id in pair_ids)
    target = max(1, int(length))
    if not base:
        return tuple()
    repeats = int(np.ceil(target / float(len(base))))
    return (base * repeats)[:target]


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
                pair_ids=_cycle_program_pairs((blocked, unblocked), length),
                kind="bearing_split",
            )
        )
        programs.append(
            ShieldProgram(
                name=f"material_split_{idx}",
                pair_ids=_cycle_program_pairs((fe_only, pb_only), length),
                kind="material_split",
            )
        )
        programs.append(
            ShieldProgram(
                name=f"occlusion_test_{idx}",
                pair_ids=_cycle_program_pairs((unblocked, blocked, fe_only), length),
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
        use_gpu=bool(estimator.pf_config.use_gpu),
        gpu_device=str(estimator.pf_config.gpu_device),
        gpu_dtype=str(estimator.pf_config.gpu_dtype),
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
    tentative_weight_multiplier: float = 1.0,
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
            tentative_raw = getattr(state, "tentative_sources", None)
            tentative = (
                np.zeros(num_sources, dtype=bool)
                if tentative_raw is None
                else np.asarray(tentative_raw, dtype=bool)
            )
            if tentative.size != num_sources:
                tentative = np.resize(tentative, num_sources)
            failed_raw = getattr(state, "verification_fail_streaks", None)
            failed = (
                np.zeros(num_sources, dtype=int)
                if failed_raw is None
                else np.asarray(failed_raw, dtype=int)
            )
            if failed.size != num_sources:
                failed = np.resize(failed, num_sources)
            quarantine_mask = tentative & (failed > 0)
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
            for source_idx, (pos, strength, rel_strength) in enumerate(
                zip(
                    state.positions[:num_sources],
                    state_strengths,
                    rel_strengths,
                )
            ):
                if bool(quarantine_mask[source_idx]):
                    continue
                positions.append(np.asarray(pos, dtype=float))
                strengths.append(float(strength))
                multiplier = (
                    max(float(tentative_weight_multiplier), 1.0)
                    if bool(tentative[source_idx])
                    else 1.0
                )
                sample_weights.append(
                    float(particle_weight) * float(rel_strength) * multiplier
                )
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


def _build_pair_signature_cache(
    *,
    kernel: ContinuousKernel,
    estimator: RotatingShieldPFEstimator,
    modes_by_isotope: dict[str, list[SignatureMode]],
    pose_xyz: NDArray[np.float64],
    config: DSSPPConfig,
) -> dict[str, tuple[NDArray[np.float64], list[float]]]:
    """Precompute single-posture signatures for every Fe/Pb orientation pair."""
    num_orients = int(estimator.num_orientations)
    num_pairs = num_orients * num_orients
    cache: dict[str, tuple[NDArray[np.float64], list[float]]] = {}
    for isotope in estimator.isotopes:
        modes = modes_by_isotope.get(isotope, [])
        if not modes:
            cache[isotope] = (np.zeros((num_pairs, 0), dtype=float), [])
            continue
        mode_positions = np.vstack(
            [np.asarray(mode.position_xyz, dtype=float).reshape(3) for mode in modes]
        )
        mode_strengths = np.asarray(
            [float(mode.strength_cps_1m) for mode in modes],
            dtype=float,
        )
        source_scale = estimator.response_scale_for_isotope(isotope)
        try:
            kernel_values = kernel.kernel_values_all_pairs(
                isotope=isotope,
                detector_pos=pose_xyz,
                sources=mode_positions,
            )
        except RuntimeError:
            kernel_values = np.asarray(
                [
                    [
                        kernel.kernel_value_pair(
                            isotope=isotope,
                            detector_pos=pose_xyz,
                            source_pos=mode.position_xyz,
                            fe_index=fe_index,
                            pb_index=pb_index,
                        )
                        for mode in modes
                    ]
                    for pair_id in range(num_pairs)
                    for fe_index, pb_index in (
                        _pair_indices(pair_id, num_orients),
                    )
                ],
                dtype=float,
            )
        if kernel_values.shape != (num_pairs, len(modes)):
            matrix = np.zeros((num_pairs, len(modes)), dtype=float)
            for mode_idx, mode in enumerate(modes):
                for pair_id in range(num_pairs):
                    fe_index, pb_index = _pair_indices(pair_id, num_orients)
                    matrix[pair_id, mode_idx] = kernel.kernel_value_pair(
                        isotope=isotope,
                        detector_pos=pose_xyz,
                        source_pos=mode.position_xyz,
                        fe_index=fe_index,
                        pb_index=pb_index,
                    )
            kernel_values = matrix
        matrix = np.maximum(
            float(config.live_time_s)
            * float(source_scale)
            * kernel_values
            * mode_strengths[None, :],
            0.0,
        )
        cache[isotope] = (matrix, [float(mode.weight) for mode in modes])
    return cache


def _build_pair_signature_caches_for_poses(
    *,
    kernel: ContinuousKernel,
    estimator: RotatingShieldPFEstimator,
    modes_by_isotope: dict[str, list[SignatureMode]],
    poses_xyz: NDArray[np.float64],
    config: DSSPPConfig,
) -> list[dict[str, tuple[NDArray[np.float64], list[float]]]]:
    """Precompute all-pair shield signatures for every candidate pose."""
    pose_arr = np.asarray(poses_xyz, dtype=float)
    if pose_arr.ndim != 2 or pose_arr.shape[1] != 3:
        raise ValueError("poses_xyz must be shaped (P, 3).")
    num_orients = int(estimator.num_orientations)
    num_pairs = num_orients * num_orients
    caches: list[dict[str, tuple[NDArray[np.float64], list[float]]]] = [
        {} for _ in range(pose_arr.shape[0])
    ]
    for isotope in estimator.isotopes:
        modes = modes_by_isotope.get(isotope, [])
        if not modes:
            for cache in caches:
                cache[isotope] = (np.zeros((num_pairs, 0), dtype=float), [])
            continue
        mode_positions = np.vstack(
            [np.asarray(mode.position_xyz, dtype=float).reshape(3) for mode in modes]
        )
        mode_strengths = np.asarray(
            [float(mode.strength_cps_1m) for mode in modes],
            dtype=float,
        )
        source_scale = estimator.response_scale_for_isotope(isotope)
        try:
            kernel_values = kernel.kernel_values_all_pairs_for_detectors(
                isotope=isotope,
                detector_positions=pose_arr,
                sources=mode_positions,
            )
        except RuntimeError:
            kernel_values = np.stack(
                [
                    kernel.kernel_values_all_pairs(
                        isotope=isotope,
                        detector_pos=pose,
                        sources=mode_positions,
                    )
                    for pose in pose_arr
                ],
                axis=0,
            )
        expected_shape = (pose_arr.shape[0], num_pairs, len(modes))
        if kernel_values.shape != expected_shape:
            fallback = np.zeros(expected_shape, dtype=float)
            for pose_idx, pose in enumerate(pose_arr):
                for mode_idx, mode in enumerate(modes):
                    for pair_id in range(num_pairs):
                        fe_index, pb_index = _pair_indices(pair_id, num_orients)
                        fallback[pose_idx, pair_id, mode_idx] = (
                            kernel.kernel_value_pair(
                                isotope=isotope,
                                detector_pos=pose,
                                source_pos=mode.position_xyz,
                                fe_index=fe_index,
                                pb_index=pb_index,
                            )
                        )
            kernel_values = fallback
        matrices = np.maximum(
            float(config.live_time_s)
            * float(source_scale)
            * kernel_values
            * mode_strengths[None, None, :],
            0.0,
        )
        weights = [float(mode.weight) for mode in modes]
        for pose_idx, cache in enumerate(caches):
            cache[isotope] = (matrices[pose_idx], weights)
    return caches


def _score_program_from_pair_cache(
    *,
    estimator: RotatingShieldPFEstimator,
    pair_cache: dict[str, tuple[NDArray[np.float64], list[float]]],
    program: ShieldProgram,
    config: DSSPPConfig,
) -> tuple[float, float, float, float, float, float, float]:
    """Score a shield program from cached single-posture signatures."""
    isotope_weights = estimator.pf_config.alpha_weights or {
        isotope: 1.0 for isotope in estimator.isotopes
    }
    alpha_sum = sum(float(v) for v in isotope_weights.values()) or 1.0
    pair_ids = np.asarray(program.pair_ids, dtype=int)
    signature_total = 0.0
    temporal_total = 0.0
    differential_terms: list[float] = []
    observation_counts: dict[str, float] = {}
    balance_counts: dict[str, float] = {}
    dose_score = 0.0
    for isotope in estimator.isotopes:
        matrix, weights = pair_cache.get(
            isotope,
            (np.zeros((0, 0), dtype=float), []),
        )
        signatures: list[NDArray[np.float64]] = []
        if matrix.size and pair_ids.size:
            clipped_pair_ids = np.clip(pair_ids, 0, matrix.shape[0] - 1)
            signatures = [
                np.asarray(matrix[clipped_pair_ids, mode_idx], dtype=float)
                for mode_idx in range(matrix.shape[1])
            ]
        if signatures:
            isotope_weight = float(isotope_weights.get(isotope, 1.0)) / alpha_sum
            signature_total += (
                isotope_weight
                * _signature_separation_score(
                    signatures,
                    variance_floor=config.count_variance_floor,
                )
            )
            temporal_total += isotope_weight * _temporal_separation_score_from_signatures(
                signatures,
                weights,
                config=config,
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
    count_utility = _saturated_count_utility(
        balance_counts,
        saturation_counts=float(config.count_utility_saturation_counts),
    )
    return (
        signature_total,
        temporal_total,
        observation_penalty,
        count_balance_penalty,
        differential_penalty,
        dose_score,
        count_utility,
    )


def _temporal_score_program_from_pair_cache(
    *,
    estimator: RotatingShieldPFEstimator,
    pair_cache: dict[str, tuple[NDArray[np.float64], list[float]]],
    program: ShieldProgram,
    config: DSSPPConfig,
) -> float:
    """Return only the temporal separation term from cached pair signatures."""
    isotope_weights = estimator.pf_config.alpha_weights or {
        isotope: 1.0 for isotope in estimator.isotopes
    }
    alpha_sum = sum(float(value) for value in isotope_weights.values()) or 1.0
    pair_ids = np.asarray(program.pair_ids, dtype=int)
    if pair_ids.size == 0:
        return 0.0
    temporal_total = 0.0
    for isotope in estimator.isotopes:
        matrix, weights = pair_cache.get(
            isotope,
            (np.zeros((0, 0), dtype=float), []),
        )
        if not matrix.size or matrix.shape[1] < 2:
            continue
        clipped_pair_ids = np.clip(pair_ids, 0, matrix.shape[0] - 1)
        signatures = [
            np.asarray(matrix[clipped_pair_ids, mode_idx], dtype=float)
            for mode_idx in range(matrix.shape[1])
        ]
        isotope_weight = float(isotope_weights.get(isotope, 1.0)) / alpha_sum
        temporal_total += isotope_weight * _temporal_separation_score_from_signatures(
            signatures,
            weights,
            config=config,
        )
    return float(temporal_total)


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


def _pairwise_contrast_cover_score(
    response_matrix: NDArray[np.float64],
    mode_weights: NDArray[np.float64],
    *,
    contrast_threshold: float,
) -> float:
    """Return the weighted fraction of mode pairs separated by any posture."""
    matrix = np.maximum(np.asarray(response_matrix, dtype=float), 0.0)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        return 0.0
    weights = _normalise_weights(np.asarray(mode_weights, dtype=float))
    if weights.size != matrix.shape[1]:
        weights = np.ones(matrix.shape[1], dtype=float) / float(matrix.shape[1])
    eps = 1.0e-12
    threshold = max(float(contrast_threshold), eps)
    covered = 0.0
    total_weight = 0.0
    log_matrix = np.log(matrix + eps)
    for idx in range(matrix.shape[1]):
        for jdx in range(idx + 1, matrix.shape[1]):
            pair_weight = float(np.sqrt(max(weights[idx] * weights[jdx], 0.0)))
            if pair_weight <= 0.0:
                continue
            contrast = float(np.max(np.abs(log_matrix[:, idx] - log_matrix[:, jdx])))
            covered += pair_weight * min(1.0, contrast / threshold)
            total_weight += pair_weight
    if total_weight <= 0.0:
        return 0.0
    return float(np.clip(covered / total_weight, 0.0, 1.0))


def _response_logdet_score(
    response_matrix: NDArray[np.float64],
    mode_weights: NDArray[np.float64],
    *,
    ridge: float,
    variance_floor: float,
) -> float:
    """Return a D-optimality score for temporal shield response columns."""
    matrix = np.maximum(np.asarray(response_matrix, dtype=float), 0.0)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        return 0.0
    weights = _normalise_weights(np.asarray(mode_weights, dtype=float))
    if weights.size != matrix.shape[1]:
        weights = np.ones(matrix.shape[1], dtype=float) / float(matrix.shape[1])
    row_variance = np.maximum(np.sum(matrix, axis=1), max(float(variance_floor), 1.0e-12))
    whitened = matrix / np.sqrt(row_variance[:, None])
    weighted = whitened * np.sqrt(np.maximum(weights, 0.0))[None, :]
    ridge_val = max(float(ridge), 1.0e-12)
    gram = weighted.T @ weighted + ridge_val * np.eye(matrix.shape[1], dtype=float)
    sign, logdet = np.linalg.slogdet(gram)
    if sign <= 0 or not np.isfinite(logdet):
        return 0.0
    baseline = float(matrix.shape[1]) * float(np.log(ridge_val))
    return max(float(logdet - baseline), 0.0)


def _maximum_response_correlation(
    response_matrix: NDArray[np.float64],
    *,
    variance_floor: float,
) -> float:
    """Return the largest Poisson-whitened cosine between mode responses."""
    matrix = np.maximum(np.asarray(response_matrix, dtype=float), 0.0)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        return 0.0
    floor = max(float(variance_floor), 1.0e-12)
    max_corr = 0.0
    for idx in range(matrix.shape[1]):
        for jdx in range(idx + 1, matrix.shape[1]):
            left = matrix[:, idx]
            right = matrix[:, jdx]
            variance = np.maximum(left + right, floor)
            weighted_left = left / np.sqrt(variance)
            weighted_right = right / np.sqrt(variance)
            denom = float(np.linalg.norm(weighted_left) * np.linalg.norm(weighted_right))
            if denom <= 0.0:
                continue
            corr = float(np.dot(weighted_left, weighted_right) / denom)
            max_corr = max(max_corr, corr)
    return float(np.clip(max_corr, 0.0, 1.0))


def _temporal_separation_score_from_signatures(
    signatures: list[NDArray[np.float64]],
    weights: list[float],
    *,
    config: DSSPPConfig,
) -> float:
    """Score a shield program by same-isotope temporal-code separability."""
    if len(signatures) < 2:
        return 0.0
    raw_matrix = np.column_stack(
        [np.maximum(np.asarray(sig, dtype=float).ravel(), 0.0) for sig in signatures]
    )
    if raw_matrix.ndim != 2 or raw_matrix.shape[0] == 0 or raw_matrix.shape[1] < 2:
        return 0.0
    column_totals = np.sum(raw_matrix, axis=0)
    active = column_totals > 1.0e-12
    if int(np.sum(active)) < 2:
        return 0.0
    matrix = raw_matrix[:, active] / column_totals[active][None, :]
    weight_arr = _normalise_weights(np.asarray(weights, dtype=float)[active])
    cover = _pairwise_contrast_cover_score(
        matrix,
        weight_arr,
        contrast_threshold=float(config.temporal_pair_contrast_threshold),
    )
    logdet = _response_logdet_score(
        matrix,
        weight_arr,
        ridge=float(config.temporal_logdet_ridge),
        variance_floor=float(config.count_variance_floor),
    )
    max_corr = _maximum_response_correlation(
        matrix,
        variance_floor=float(config.count_variance_floor),
    )
    decorrelation = 1.0 - max_corr
    return max(
        float(config.temporal_cover_weight) * cover
        + float(config.temporal_logdet_weight)
        * decorrelation
        * float(np.log1p(logdet))
        + float(config.temporal_decorrelation_weight) * decorrelation,
        0.0,
    )


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


def _saturated_count_utility(
    counts: dict[str, float],
    *,
    saturation_counts: float,
) -> float:
    """Return a bounded utility for usable counts without rewarding raw proximity."""
    values = np.asarray(
        [max(float(value), 0.0) for value in counts.values()],
        dtype=float,
    )
    if values.size == 0:
        return 0.0
    saturation = max(float(saturation_counts), 1.0e-12)
    utilities = 1.0 - np.exp(-values / saturation)
    return float(np.mean(np.clip(utilities, 0.0, 1.0)))


def _mode_visibility_proxy(
    mode: SignatureMode,
    pose_xyz: NDArray[np.float64],
    *,
    live_time_s: float,
    saturation_counts: float,
) -> float:
    """Return a bounded inverse-square visibility proxy for planning heuristics."""
    pose = np.asarray(pose_xyz, dtype=float).reshape(3)
    distance = max(float(np.linalg.norm(pose - mode.position_xyz)), 0.25)
    expected = float(live_time_s) * max(float(mode.strength_cps_1m), 0.0) / (
        distance * distance
    )
    saturation = max(float(saturation_counts), 1.0e-12)
    return float(np.clip(1.0 - np.exp(-expected / saturation), 0.0, 1.0))


def _local_orbit_gain(
    candidate_pose_xyz: NDArray[np.float64],
    modes_by_isotope: dict[str, list[SignatureMode]],
    *,
    config: DSSPPConfig,
) -> float:
    """Return a gain for near-source annular viewpoints rather than source chasing."""
    radii = tuple(
        float(radius)
        for radius in config.ring_radii_m
        if float(radius) > 0.0
    )
    if not radii:
        return 0.0
    candidate = np.asarray(candidate_pose_xyz, dtype=float).reshape(3)
    sigma = max(float(config.local_orbit_sigma_m), 1.0e-6)
    gains: list[float] = []
    weights: list[float] = []
    for modes in modes_by_isotope.values():
        for mode in modes:
            if float(mode.weight) <= 0.0:
                continue
            distance = float(np.linalg.norm(candidate[:2] - mode.position_xyz[:2]))
            radial_error = min(abs(distance - radius) for radius in radii)
            radial_gain = float(np.exp(-0.5 * (radial_error / sigma) ** 2))
            visibility = _mode_visibility_proxy(
                mode,
                candidate,
                live_time_s=float(config.live_time_s),
                saturation_counts=float(config.count_utility_saturation_counts),
            )
            gains.append(radial_gain * visibility)
            weights.append(float(mode.weight))
    if not gains:
        return 0.0
    weight_arr = _normalise_weights(np.asarray(weights, dtype=float))
    return float(np.sum(weight_arr * np.asarray(gains, dtype=float)))


def _station_response_matrix(
    poses_xyz: NDArray[np.float64],
    modes: Sequence[SignatureMode],
    *,
    live_time_s: float,
) -> NDArray[np.float64]:
    """Return a geometric station-response design matrix for planning only."""
    pose_arr = np.asarray(poses_xyz, dtype=float)
    if pose_arr.ndim == 1 and pose_arr.size == 3:
        pose_arr = pose_arr.reshape(1, 3)
    if pose_arr.ndim != 2 or pose_arr.shape[1] != 3 or not modes:
        return np.zeros((0, 0), dtype=float)
    matrix = np.zeros((pose_arr.shape[0], len(modes)), dtype=float)
    for mode_idx, mode in enumerate(modes):
        deltas = pose_arr - mode.position_xyz[None, :]
        distances = np.maximum(np.linalg.norm(deltas, axis=1), 0.25)
        matrix[:, mode_idx] = (
            float(live_time_s)
            * max(float(mode.strength_cps_1m), 0.0)
            / (distances * distances)
        )
    return matrix


def _station_condition_logdet(
    matrix: NDArray[np.float64],
    *,
    ridge: float,
) -> float:
    """Return a column-normalized D-optimal station-design score."""
    design = np.maximum(np.asarray(matrix, dtype=float), 0.0)
    if design.ndim != 2 or design.shape[0] == 0 or design.shape[1] < 2:
        return 0.0
    column_norms = np.linalg.norm(design, axis=0)
    active = column_norms > 1.0e-12
    if int(np.count_nonzero(active)) < 2:
        return 0.0
    normalized = design[:, active] / column_norms[active][None, :]
    ridge_val = max(float(ridge), 1.0e-12)
    gram = normalized.T @ normalized + ridge_val * np.eye(
        int(np.count_nonzero(active)),
        dtype=float,
    )
    sign, logdet = np.linalg.slogdet(gram)
    if sign <= 0 or not np.isfinite(logdet):
        return 0.0
    baseline = float(np.count_nonzero(active)) * float(np.log(ridge_val))
    return max(float(logdet - baseline), 0.0)


def _station_condition_gain(
    candidate_pose_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    modes_by_isotope: dict[str, list[SignatureMode]],
    *,
    config: DSSPPConfig,
) -> float:
    """Return the added response-matrix conditioning from one candidate station."""
    candidate = np.asarray(candidate_pose_xyz, dtype=float).reshape(1, 3)
    visited = np.zeros((0, 3), dtype=float)
    if visited_poses_xyz is not None:
        visited = np.asarray(visited_poses_xyz, dtype=float)
        if visited.ndim == 1 and visited.size == 3:
            visited = visited.reshape(1, 3)
        if visited.ndim != 2 or visited.shape[1] != 3:
            visited = np.zeros((0, 3), dtype=float)
    gains: list[float] = []
    weights: list[float] = []
    for modes in modes_by_isotope.values():
        active = [mode for mode in modes if float(mode.weight) > 0.0]
        if len(active) < 2:
            continue
        before = _station_response_matrix(
            visited,
            active,
            live_time_s=float(config.live_time_s),
        )
        after = _station_response_matrix(
            np.vstack([visited, candidate]) if visited.size else candidate,
            active,
            live_time_s=float(config.live_time_s),
        )
        before_score = _station_condition_logdet(
            before,
            ridge=float(config.station_condition_ridge),
        )
        after_score = _station_condition_logdet(
            after,
            ridge=float(config.station_condition_ridge),
        )
        gains.append(max(after_score - before_score, 0.0))
        weights.append(float(sum(mode.weight for mode in active)))
    if not gains:
        return 0.0
    weight_arr = _normalise_weights(np.asarray(weights, dtype=float))
    return float(np.sum(weight_arr * np.asarray(gains, dtype=float)))


def _score_program(
    *,
    estimator: RotatingShieldPFEstimator,
    kernel: ContinuousKernel,
    modes_by_isotope: dict[str, list[SignatureMode]],
    pose_xyz: NDArray[np.float64],
    program: ShieldProgram,
    config: DSSPPConfig,
) -> tuple[float, float, float, float, float, float, float]:
    """Return signature, temporal, observation, balance, differential, dose, and count terms."""
    num_orients = int(estimator.num_orientations)
    isotope_weights = estimator.pf_config.alpha_weights or {
        isotope: 1.0 for isotope in estimator.isotopes
    }
    alpha_sum = sum(float(v) for v in isotope_weights.values()) or 1.0
    signature_total = 0.0
    temporal_total = 0.0
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
            isotope_weight = float(isotope_weights.get(isotope, 1.0)) / alpha_sum
            signature_total += (
                isotope_weight
                * _signature_separation_score(
                    signatures,
                    variance_floor=config.count_variance_floor,
                )
            )
            temporal_total += isotope_weight * _temporal_separation_score_from_signatures(
                signatures,
                weights,
                config=config,
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
    count_utility = _saturated_count_utility(
        balance_counts,
        saturation_counts=float(config.count_utility_saturation_counts),
    )
    return (
        signature_total,
        temporal_total,
        observation_penalty,
        count_balance_penalty,
        differential_penalty,
        dose_score,
        count_utility,
    )


def _greedy_pairwise_contrast_program(
    *,
    estimator: RotatingShieldPFEstimator,
    kernel: ContinuousKernel,
    modes_by_isotope: dict[str, list[SignatureMode]],
    pose_xyz: NDArray[np.float64],
    config: DSSPPConfig,
    pair_cache: dict[str, tuple[NDArray[np.float64], list[float]]] | None = None,
) -> ShieldProgram | None:
    """Build a pose-specific temporal-code program by greedy pairwise cover."""
    if float(config.lambda_temporal_separation) <= 0.0:
        return None
    if int(config.temporal_cover_programs) <= 0:
        return None
    has_multi_mode = any(len(modes) >= 2 for modes in modes_by_isotope.values())
    if not has_multi_mode:
        return None
    num_orients = int(estimator.num_orientations)
    program_length = max(1, int(config.program_length))
    remaining = list(range(num_orients * num_orients))
    selected: list[int] = []
    best_score = 0.0
    for _ in range(program_length):
        candidate_best_pair: int | None = None
        candidate_best_score = -np.inf
        for pair_id in remaining:
            program = ShieldProgram(
                name="temporal_cover_probe",
                pair_ids=tuple(selected + [int(pair_id)]),
                kind="pairwise_contrast_cover",
            )
            if pair_cache is None:
                temporal_score = _score_program(
                    estimator=estimator,
                    kernel=kernel,
                    modes_by_isotope=modes_by_isotope,
                    pose_xyz=pose_xyz,
                    program=program,
                    config=config,
                )[1]
            else:
                temporal_score = _temporal_score_program_from_pair_cache(
                    estimator=estimator,
                    pair_cache=pair_cache,
                    program=program,
                    config=config,
                )
            if temporal_score > candidate_best_score:
                candidate_best_score = float(temporal_score)
                candidate_best_pair = int(pair_id)
        if candidate_best_pair is None:
            break
        selected.append(candidate_best_pair)
        remaining.remove(candidate_best_pair)
        best_score = max(best_score, float(candidate_best_score))
    if not selected or best_score <= 0.0:
        return None
    return ShieldProgram(
        name=f"pairwise_contrast_cover_{len(selected)}",
        pair_ids=tuple(selected),
        kind="pairwise_contrast_cover",
    )


def _programs_for_pose(
    *,
    estimator: RotatingShieldPFEstimator,
    kernel: ContinuousKernel,
    modes_by_isotope: dict[str, list[SignatureMode]],
    pose_xyz: NDArray[np.float64],
    base_programs: Sequence[ShieldProgram],
    config: DSSPPConfig,
    include_pose_specific_cover: bool = True,
    pair_cache: dict[str, tuple[NDArray[np.float64], list[float]]] | None = None,
) -> list[ShieldProgram]:
    """Return base programs plus pose-specific temporal-code programs."""
    programs = list(base_programs)
    if not include_pose_specific_cover:
        return programs
    cover_program = _greedy_pairwise_contrast_program(
        estimator=estimator,
        kernel=kernel,
        modes_by_isotope=modes_by_isotope,
        pose_xyz=pose_xyz,
        config=config,
        pair_cache=pair_cache,
    )
    if cover_program is None:
        return programs
    seen = {tuple(program.pair_ids) for program in programs}
    if tuple(cover_program.pair_ids) not in seen:
        programs.append(cover_program)
    return programs


def _static_station_program_score(
    *,
    signature_score: float,
    temporal_separation_score: float,
    count_utility: float,
    count_balance_penalty: float,
    differential_penalty: float,
    dose_score: float,
    coverage_norm: float,
    revisit_penalty: float,
    bearing_gain: float,
    frontier_gain: float,
    turn_penalty: float,
    local_orbit_gain: float,
    station_condition_gain: float,
    coverage_floor: float,
    config: DSSPPConfig,
) -> float:
    """Return the non-transition score for one station-program pair."""
    return float(
        float(config.lambda_signature) * float(np.log1p(max(signature_score, 0.0)))
        + float(config.lambda_temporal_separation)
        * float(np.log1p(max(temporal_separation_score, 0.0)))
        + float(config.lambda_coverage) * float(coverage_norm)
        + float(config.lambda_bearing_diversity) * float(bearing_gain)
        + float(config.lambda_frontier) * float(frontier_gain)
        + float(config.lambda_count_utility) * float(count_utility)
        + float(config.lambda_local_orbit) * float(local_orbit_gain)
        + float(config.lambda_station_condition)
        * float(np.log1p(max(station_condition_gain, 0.0)))
        - float(config.lambda_dose) * float(dose_score)
        - float(config.eta_count_balance) * float(count_balance_penalty)
        - float(config.eta_differential) * float(differential_penalty)
        - float(config.eta_revisit) * float(revisit_penalty)
        - float(config.lambda_turn_smoothness) * float(turn_penalty)
        - float(config.coverage_floor_weight)
        * max(0.0, float(coverage_floor) - float(coverage_norm)) ** 2
    )


def _evaluate_pose_index_from_context(
    pose_index_value: int,
) -> tuple[int, float, list[tuple[Any, ...]], list[float], list[float]]:
    """Evaluate all shield programs for one candidate station from worker context."""
    context = _DSS_PP_POSE_EVAL_CONTEXT
    if context is None:
        raise RuntimeError("DSS-PP pose evaluation context is not initialized.")
    pose_index = int(pose_index_value)
    candidate_poses = np.asarray(context["candidate_poses"], dtype=float)
    path_lengths = np.asarray(context["path_lengths"], dtype=float)
    pair_caches_by_pose = cast(
        list[dict[str, tuple[NDArray[np.float64], list[float]]]],
        context["pair_caches_by_pose"],
    )
    programs = cast(Sequence[ShieldProgram], context["programs"])
    estimator = cast(RotatingShieldPFEstimator, context["estimator"])
    kernel = cast(ContinuousKernel, context["kernel"])
    modes_by_isotope = cast(
        dict[str, list[SignatureMode]],
        context["modes_by_isotope"],
    )
    config = cast(DSSPPConfig, context["config"])
    coverage_norm = np.asarray(context["coverage_norm"], dtype=float)
    coverage_raw = np.asarray(context["coverage_raw"], dtype=float)
    revisit_penalties = np.asarray(context["revisit_penalties"], dtype=float)
    bearing_gains = np.asarray(context["bearing_gains"], dtype=float)
    frontier_gains = np.asarray(context["frontier_gains"], dtype=float)
    turn_penalties = np.asarray(context["turn_penalties"], dtype=float)
    local_orbit_gains = np.asarray(context["local_orbit_gains"], dtype=float)
    station_condition_gains = np.asarray(
        context["station_condition_gains"],
        dtype=float,
    )
    coverage_floor = float(context["coverage_floor"])

    local_pending: list[tuple[Any, ...]] = []
    local_scores: list[float] = []
    local_observation_penalties: list[float] = []
    local_cheap_score = -np.inf
    pose = candidate_poses[pose_index]
    if not np.isfinite(path_lengths[pose_index]):
        return (
            pose_index,
            local_cheap_score,
            local_pending,
            local_scores,
            local_observation_penalties,
        )
    pair_cache = pair_caches_by_pose[pose_index]
    pose_programs = _programs_for_pose(
        estimator=estimator,
        kernel=kernel,
        modes_by_isotope=modes_by_isotope,
        pose_xyz=pose,
        base_programs=programs,
        config=config,
        include_pose_specific_cover=True,
        pair_cache=pair_cache,
    )
    for program in pose_programs:
        (
            signature_score,
            temporal_separation_score,
            observation_penalty,
            count_balance_penalty,
            differential_penalty,
            dose_score,
            count_utility,
        ) = _score_program_from_pair_cache(
            estimator=estimator,
            pair_cache=pair_cache,
            program=program,
            config=config,
        )
        static_score = _static_station_program_score(
            signature_score=signature_score,
            temporal_separation_score=temporal_separation_score,
            count_utility=count_utility,
            count_balance_penalty=count_balance_penalty,
            differential_penalty=differential_penalty,
            dose_score=dose_score,
            coverage_norm=float(coverage_norm[pose_index]),
            revisit_penalty=float(revisit_penalties[pose_index]),
            bearing_gain=float(bearing_gains[pose_index]),
            frontier_gain=float(frontier_gains[pose_index]),
            turn_penalty=float(turn_penalties[pose_index]),
            local_orbit_gain=float(local_orbit_gains[pose_index]),
            station_condition_gain=float(station_condition_gains[pose_index]),
            coverage_floor=coverage_floor,
            config=config,
        )
        local_cheap_score = max(float(local_cheap_score), float(static_score))
        local_scores.append(static_score)
        local_observation_penalties.append(observation_penalty)
        local_pending.append(
            (
                pose_index,
                pose.copy(),
                program,
                0.0,
                signature_score,
                temporal_separation_score,
                observation_penalty,
                count_balance_penalty,
                differential_penalty,
                dose_score,
                count_utility,
                float(coverage_raw[pose_index]),
                float(revisit_penalties[pose_index]),
                float(bearing_gains[pose_index]),
                float(frontier_gains[pose_index]),
                float(turn_penalties[pose_index]),
                float(local_orbit_gains[pose_index]),
                float(station_condition_gains[pose_index]),
            )
        )
    return (
        pose_index,
        local_cheap_score,
        local_pending,
        local_scores,
        local_observation_penalties,
    )


def _evaluate_pose_indices_parallel(
    eval_indices: Sequence[int],
    *,
    context: dict[str, object],
    worker_count: int,
) -> list[tuple[int, float, list[tuple[Any, ...]], list[float], list[float]]]:
    """Evaluate pose-program scores using process workers when available."""
    global _DSS_PP_POSE_EVAL_CONTEXT
    indices = [int(index) for index in eval_indices]
    previous_context = _DSS_PP_POSE_EVAL_CONTEXT
    _DSS_PP_POSE_EVAL_CONTEXT = context
    try:
        if int(worker_count) <= 1 or len(indices) <= 1:
            return [_evaluate_pose_index_from_context(index) for index in indices]
        max_workers = min(len(indices), max(1, int(worker_count)))
        if os.name == "posix":
            try:
                fork_context = mp.get_context("fork")
                with ProcessPoolExecutor(
                    max_workers=max_workers,
                    mp_context=fork_context,
                ) as executor:
                    return list(
                        executor.map(
                            _evaluate_pose_index_from_context,
                            indices,
                            chunksize=1,
                        )
                    )
            except Exception:
                pass
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(_evaluate_pose_index_from_context, indices))
    finally:
        _DSS_PP_POSE_EVAL_CONTEXT = previous_context


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


def _program_evaluation_pose_indices(
    *,
    path_lengths: NDArray[np.float64],
    coverage_norm: NDArray[np.float64],
    revisit_penalties: NDArray[np.float64],
    bearing_gains: NDArray[np.float64],
    frontier_gains: NDArray[np.float64],
    turn_penalties: NDArray[np.float64],
    local_orbit_gains: NDArray[np.float64],
    station_condition_gains: NDArray[np.float64],
    lambda_distance: float,
    config: DSSPPConfig,
) -> NDArray[np.int64]:
    """Return candidate indices that merit full shield-program evaluation."""
    count = int(path_lengths.size)
    if count == 0:
        return np.zeros(0, dtype=np.int64)
    eig_limit = int(config.eig_candidate_limit or 0)
    target = max(
        32,
        int(config.beam_width) * 8,
        eig_limit * 8 if eig_limit > 0 else 0,
        int(config.max_programs),
    )
    if count <= target:
        return np.arange(count, dtype=np.int64)
    finite = np.isfinite(path_lengths)
    if not np.any(finite):
        return np.zeros(0, dtype=np.int64)
    finite_lengths = path_lengths[finite]
    length_scale = max(float(np.max(finite_lengths)), 1.0e-12)
    cheap_score = (
        float(config.lambda_coverage) * np.asarray(coverage_norm, dtype=float)
        + float(config.lambda_bearing_diversity) * np.asarray(bearing_gains, dtype=float)
        + float(config.lambda_frontier) * np.asarray(frontier_gains, dtype=float)
        + float(config.lambda_local_orbit) * np.asarray(local_orbit_gains, dtype=float)
        + float(config.lambda_station_condition)
        * np.log1p(np.maximum(np.asarray(station_condition_gains, dtype=float), 0.0))
        - float(config.eta_revisit) * np.asarray(revisit_penalties, dtype=float)
        - float(config.lambda_turn_smoothness) * np.asarray(turn_penalties, dtype=float)
        - float(lambda_distance) * np.asarray(path_lengths, dtype=float) / length_scale
    )
    cheap_score[~finite] = -np.inf
    selected_count = min(count, target)
    order = np.argsort(cheap_score)[::-1]
    selected = order[:selected_count]
    return np.asarray(selected[np.isfinite(cheap_score[selected])], dtype=np.int64)


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
    local_orbit_gains = np.asarray(
        [
            _local_orbit_gain(
                pose,
                modes_by_isotope,
                config=config,
            )
            for pose in candidate_poses
        ],
        dtype=float,
    )
    station_condition_gains = np.asarray(
        [
            _station_condition_gain(
                pose,
                visited_poses_xyz,
                modes_by_isotope,
                config=config,
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
    evaluation_pose_indices = np.arange(candidate_poses.shape[0], dtype=np.int64)
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
            float,
            float,
            float,
            float,
        ]
    ] = []
    pair_caches_by_pose = _build_pair_signature_caches_for_poses(
        kernel=kernel,
        estimator=estimator,
        modes_by_isotope=modes_by_isotope,
        poses_xyz=candidate_poses,
        config=config,
    )
    eval_indices = [int(idx) for idx in evaluation_pose_indices]
    worker_cfg = config.program_eval_workers
    if worker_cfg is None:
        worker_count = min(len(eval_indices), max(1, os.cpu_count() or 1))
    else:
        worker_count = min(len(eval_indices), max(1, int(worker_cfg)))
    pose_eval_context: dict[str, object] = {
        "candidate_poses": candidate_poses,
        "path_lengths": path_lengths,
        "pair_caches_by_pose": pair_caches_by_pose,
        "programs": programs,
        "estimator": estimator,
        "kernel": kernel,
        "modes_by_isotope": modes_by_isotope,
        "config": config,
        "coverage_norm": coverage_norm,
        "coverage_raw": coverage_raw,
        "revisit_penalties": revisit_penalties,
        "bearing_gains": bearing_gains,
        "frontier_gains": frontier_gains,
        "turn_penalties": turn_penalties,
        "local_orbit_gains": local_orbit_gains,
        "station_condition_gains": station_condition_gains,
        "coverage_floor": float(coverage_floor),
    }
    pose_results = _evaluate_pose_indices_parallel(
        eval_indices,
        context=pose_eval_context,
        worker_count=worker_count,
    )
    for (
        pose_index,
        local_cheap_score,
        local_pending,
        local_scores,
        local_observation_penalties,
    ) in pose_results:
        if local_pending:
            cheap_pose_scores[int(pose_index)] = max(
                float(cheap_pose_scores[int(pose_index)]),
                float(local_cheap_score),
            )
            pending.extend(local_pending)
            static_scores.extend(local_scores)
            observation_penalties.extend(local_observation_penalties)
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

        def _candidate_ig_for_index(pose_index_value: int) -> tuple[int, float]:
            """Return the candidate EIG for one station index."""
            pose_index = int(pose_index_value)
            value = _candidate_information_gain(
                estimator,
                candidate_poses[int(pose_index)],
                config=config,
                rng_seed=None
                if config.rng_seed is None
                else int(config.rng_seed) + int(pose_index),
            )
            return pose_index, float(value)

        eig_indices = [int(index) for index in selected_pose_indices]
        eig_workers = min(
            len(eig_indices),
            max(1, int(worker_count)),
        )
        if eig_workers <= 1 or len(eig_indices) <= 1:
            eig_results = [_candidate_ig_for_index(index) for index in eig_indices]
        else:
            with ThreadPoolExecutor(max_workers=eig_workers) as executor:
                eig_results = list(executor.map(_candidate_ig_for_index, eig_indices))
        for pose_index, value in eig_results:
            info_gains[int(pose_index)] = float(value)
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
            temporal_separation_score,
            _obs_penalty,
            count_balance_penalty,
            differential_penalty,
            dose_score,
            count_utility,
            coverage_gain,
            revisit_penalty,
            bearing_gain,
            frontier_gain,
            turn_penalty,
            local_orbit_gain,
            station_condition_gain,
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
            temporal_separation_score=float(temporal_separation_score),
            observation_penalty=float(observation_penalty),
            count_balance_penalty=float(count_balance_penalty),
            differential_penalty=float(differential_penalty),
            dose_score=float(dose_score),
            count_utility=float(count_utility),
            coverage_gain=float(coverage_gain),
            revisit_penalty=float(revisit_penalty),
            bearing_diversity_gain=float(bearing_gain),
            frontier_gain=float(frontier_gain),
            turn_penalty=float(turn_penalty),
            local_orbit_gain=float(local_orbit_gain),
            station_condition_gain=float(station_condition_gain),
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
                temporal_separation_score=float(temporal_separation_score),
                observation_penalty=float(observation_penalty),
                count_balance_penalty=float(count_balance_penalty),
                differential_penalty=float(differential_penalty),
                dose_score=float(dose_score),
                count_utility=float(count_utility),
                coverage_gain=float(coverage_gain),
                revisit_penalty=float(revisit_penalty),
                bearing_diversity_gain=float(bearing_gain),
                frontier_gain=float(frontier_gain),
                turn_penalty=float(turn_penalty),
                local_orbit_gain=float(local_orbit_gain),
                station_condition_gain=float(station_condition_gain),
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
        tentative_weight_multiplier=1.0 + max(float(cfg.residual_signature_weight), 0.0),
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
    configured_workers = cfg.program_eval_workers
    program_eval_workers = (
        min(int(candidates.shape[0]), max(1, os.cpu_count() or 1))
        if configured_workers is None
        else min(int(candidates.shape[0]), max(1, int(configured_workers)))
    )
    diagnostics: dict[str, float | int | str] = {
        "candidate_count": int(candidates.shape[0]),
        "separation_filtered_candidates": int(separation_filtered),
        "path_filtered_candidates": int(path_filtered),
        "path_fallback_used": int(fallback_used),
        "program_count": int(len(programs)),
        "node_count": int(len(nodes)),
        "mode_count": int(mode_count),
        "program_eval_workers": int(program_eval_workers),
        "horizon": int(max(1, cfg.horizon)),
        "beam_width": int(max(1, cfg.beam_width)),
        "first_program_kind": first.program.kind,
        "first_information_gain": float(first.information_gain),
        "first_signature_score": float(first.signature_score),
        "first_temporal_separation_score": float(first.temporal_separation_score),
        "first_observation_penalty": float(first.observation_penalty),
        "first_count_balance_penalty": float(first.count_balance_penalty),
        "first_differential_penalty": float(first.differential_penalty),
        "first_dose_score": float(first.dose_score),
        "first_count_utility": float(first.count_utility),
        "first_coverage_gain": float(first.coverage_gain),
        "first_revisit_penalty": float(first.revisit_penalty),
        "first_bearing_diversity_gain": float(first.bearing_diversity_gain),
        "first_frontier_gain": float(first.frontier_gain),
        "first_turn_penalty": float(first.turn_penalty),
        "first_local_orbit_gain": float(first.local_orbit_gain),
        "first_station_condition_gain": float(first.station_condition_gain),
    }
    return DSSPPResult(
        next_pose=first.pose_xyz.copy(),
        next_pose_index=int(first.pose_index),
        shield_program=first.program,
        score=best_score,
        sequence=tuple(sequence),
        diagnostics=diagnostics,
    )
