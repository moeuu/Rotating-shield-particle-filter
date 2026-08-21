"""Batched station-candidate generation for DSS-PP."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from planning.candidate_generation import sample_low_discrepancy_heights
from planning.dss_modes import _planning_rng
from planning.dss_types import DSSPPConfig, SignatureMode


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
        xy_centers = origin[None, :] + (cells.astype(np.float64) + 0.5) * float(
            cell_size
        )
    if xy_centers.shape != (cells.shape[0], 2) or np.any(~np.isfinite(xy_centers)):
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
    if base.ndim != 2 or base.shape[1] != 3 or np.any(~np.isfinite(base)):
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
    # Every material posterior mode must contribute at least one candidate.
    # Treat the configured value as the ordinary augmentation budget, but
    # expand it when the posterior itself contains more distinct modes. This
    # preserves the full posterior geometry without requiring a run-specific
    # capacity guess or silently discarding low-mass alternatives.
    augmentation_capacity = max(
        int(config.max_augmented_candidates),
        len(all_modes),
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
            + radii[None, :, None, None] * unit_xy[None, None, :, :]
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
            neighbors = (blocked[:, None, :] + neighbor_offsets[None, :, :]).reshape(
                -1, 2
            )
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
                boundary_points = boundary_points[np.argsort(distances, kind="stable")]
            generated_batches.append(
                boundary_points[: int(config.max_augmented_candidates) // 2]
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
            coverage_points[: int(config.max_augmented_candidates) // 2].copy()
        )
    visited = _pose_matrix_or_empty(visited_poses_xyz)
    if visited.size and mode_positions.size and radii.size:
        bearing_delta = visited[None, :, :2] - mode_positions[:, None, :2]
        prior_angles = np.arctan2(
            bearing_delta[:, :, 1],
            bearing_delta[:, :, 0],
        )
        bearing_offsets = np.asarray(
            (0.5 * np.pi, -0.5 * np.pi, np.pi),
            dtype=np.float64,
        )
        bearing_angles = prior_angles[:, :, None] + bearing_offsets[None, None, :]
        bearing_unit_xy = np.stack(
            (np.cos(bearing_angles), np.sin(bearing_angles)),
            axis=-1,
        )
        bearing_xy = (
            mode_positions[:, None, None, None, :2]
            + bearing_unit_xy[:, :, :, None, :] * radii[None, None, None, :, None]
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
        raise ValueError("traversable_cells must be a nonnegative N x 2 integer array.")
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


def _pose_matrix_or_empty(poses_xyz: NDArray[np.float64] | None) -> NDArray[np.float64]:
    """Return finite poses as an N x 3 array or an empty absent history."""
    if poses_xyz is None:
        return np.zeros((0, 3), dtype=float)
    poses = np.asarray(poses_xyz, dtype=float)
    if poses.ndim == 1 and poses.size == 3:
        poses = poses.reshape(1, 3)
    if poses.ndim == 2 and poses.shape == (0, 3):
        return np.zeros((0, 3), dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 3 or np.any(~np.isfinite(poses)):
        raise ValueError("visited_poses_xyz must be finite and shaped (N, 3).")
    return poses
