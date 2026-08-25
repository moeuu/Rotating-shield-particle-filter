"""Angular geometry helpers for DSS-PP candidate scoring."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray


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


def _bearing_angle_xy(source: NDArray[np.float64], pose: NDArray[np.float64]) -> float:
    """Return the planar bearing angle from source to pose."""
    delta = np.asarray(pose[:2], dtype=float) - np.asarray(source[:2], dtype=float)
    return float(np.arctan2(delta[1], delta[0]))


def _angle_distance_rad(left: float, right: float) -> float:
    """Return wrapped absolute angular distance in radians."""
    return float(abs(np.arctan2(np.sin(left - right), np.cos(left - right))))


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
