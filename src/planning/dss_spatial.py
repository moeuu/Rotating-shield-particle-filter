"""Batched spatial objectives and route geometry for DSS-PP."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from planning.dss_candidates import (
    _angle_distance_rad,
    _bearing_angle_xy,
    _pose_matrix_or_empty,
)
from planning.dss_modes import (
    _flattened_posterior_mode_weights,
    _isotope_presence_probability,
    _normalise_weights,
    _presence_weighted_rows,
)
from planning.dss_types import DSSPPConfig, SignatureMode


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
    return np.sum(radial_gain * mode_weights.reshape(1, -1), axis=1) / float(
        isotope_count
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


def _node_path_lengths_batch(
    map_api: object | None,
    start_xyz: NDArray[np.float64],
    goals_xyz: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return path lengths through explicit free space or a map-native batch."""
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
        if (
            lengths.size != goals.shape[0]
            or np.any(np.isnan(lengths))
            or np.any(lengths < 0.0)
        ):
            raise ValueError(
                "motion_path_lengths_batch must return one nonnegative "
                "non-NaN path length per goal."
            )
        return lengths
    raise TypeError(
        "A non-None planning map must provide motion_path_lengths_batch; "
        "candidate-by-candidate waypoint evaluation is not a runtime path."
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
    path_lengths = _node_path_lengths_batch(
        map_api,
        current_pose_xyz,
        candidates,
    )
    reachable = np.isfinite(path_lengths)
    removed = int(np.count_nonzero(~reachable))
    if not np.any(reachable):
        return np.zeros((0, 3), dtype=float), removed
    return candidates[reachable], removed


def _align_candidate_values(
    original_poses_xyz: NDArray[np.float64],
    original_values: NDArray[np.float64],
    retained_poses_xyz: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Align one runtime-owned candidate vector after planner filtering."""
    original = np.asarray(original_poses_xyz, dtype=np.float64)
    retained = np.asarray(retained_poses_xyz, dtype=np.float64)
    values = np.asarray(original_values, dtype=np.float64).reshape(-1)
    if original.ndim != 2 or original.shape[1:] != (3,):
        raise ValueError("original_poses_xyz must have shape (N, 3).")
    if retained.ndim != 2 or retained.shape[1:] != (3,):
        raise ValueError("retained_poses_xyz must have shape (M, 3).")
    if values.shape != (original.shape[0],):
        raise ValueError("original_values must align with original poses.")
    matches = np.all(
        np.isclose(
            retained[:, None, :],
            original[None, :, :],
            rtol=0.0,
            atol=1.0e-10,
        ),
        axis=2,
    )
    counts = np.sum(matches, axis=1)
    if np.any(counts != 1):
        raise ValueError(
            "Every retained candidate must match exactly one runtime pose."
        )
    indices = np.argmax(matches, axis=1)
    return np.ascontiguousarray(values[indices], dtype=np.float64)


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
        raise ValueError("min_separation_m must be finite and nonnegative.")
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
        total_weights.append(float(_isotope_presence_probability(active) or 0.0))
    if not total_gains:
        return np.zeros(candidates.shape[0], dtype=float)
    return _presence_weighted_rows(
        total_gains,
        total_weights,
        population_size=len(modes_by_isotope),
    )


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
    if visited.shape[0] >= 2 and float(np.linalg.norm(visited[-1] - current)) < 1.0e-6:
        previous = visited[-2]
    else:
        previous = visited[-1]
    prev_vec = current - previous
    next_vec = np.asarray(candidate_pose_xyz, dtype=float).reshape(3) - current
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
    if visited.shape[0] >= 2 and float(np.linalg.norm(visited[-1] - current)) < 1.0e-6:
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
        raise ValueError("min_separation_m must be finite and nonnegative.")
    candidates = np.asarray(candidate_poses_xyz, dtype=float)
    if (
        candidates.ndim != 2
        or candidates.shape[1] != 3
        or np.any(~np.isfinite(candidates))
    ):
        raise ValueError("candidate_poses_xyz must be finite and shaped (N, 3).")
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
