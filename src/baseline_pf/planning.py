"""Measurement planning for the baseline particle filter."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid


def measurement_count(total_time_s: float, measurement_time_s: float = 30.0) -> int:
    """Return the number of measurements implied by the total time."""
    if total_time_s <= 0.0:
        raise ValueError("total_time_s must be positive.")
    if measurement_time_s <= 0.0:
        raise ValueError("measurement_time_s must be positive.")
    count = int(total_time_s // measurement_time_s)
    return max(1, count)


def _grid_centers(
    origin: tuple[float, float],
    cell_size: float,
    grid_shape: tuple[int, int],
    blocked_cells: Iterable[tuple[int, int]] | None = None,
) -> NDArray[np.float64]:
    """Return centers of grid cells, excluding any blocked cells."""
    nx, ny = grid_shape
    xs = origin[0] + (np.arange(nx, dtype=float) + 0.5) * cell_size
    ys = origin[1] + (np.arange(ny, dtype=float) + 0.5) * cell_size
    blocked = set(blocked_cells or [])
    centers: list[list[float]] = []
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            if (ix, iy) in blocked:
                continue
            centers.append([float(x), float(y)])
    if not centers:
        return np.zeros((0, 2), dtype=float)
    return np.asarray(centers, dtype=float)


def _farthest_point_sample(
    points: NDArray[np.float64],
    n_select: int,
) -> NDArray[np.float64]:
    """Select points that maximize the minimum spacing (greedy FPS)."""
    if points.size == 0:
        return points
    if n_select <= 0:
        return np.zeros((0, points.shape[1]), dtype=float)
    if n_select >= points.shape[0]:
        return points.copy()
    centroid = np.mean(points, axis=0)
    start_idx = int(np.argmin(np.linalg.norm(points - centroid, axis=1)))
    selected_indices = [start_idx]
    min_dist = np.linalg.norm(points - points[start_idx], axis=1)
    for _ in range(1, n_select):
        next_idx = int(np.argmax(min_dist))
        selected_indices.append(next_idx)
        dist = np.linalg.norm(points - points[next_idx], axis=1)
        min_dist = np.minimum(min_dist, dist)
    return points[np.array(selected_indices, dtype=int)]


def generate_measurement_positions(
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    total_time_s: float,
    *,
    measurement_time_s: float = 30.0,
) -> NDArray[np.float64]:
    """Return evenly spaced measurement positions that avoid obstacles."""
    num_measurements = measurement_count(
        total_time_s,
        measurement_time_s=measurement_time_s,
    )
    if obstacle_grid is None:
        nx = int(np.floor(env.size_x))
        ny = int(np.floor(env.size_y))
        centers = _grid_centers((0.0, 0.0), 1.0, (nx, ny), blocked_cells=None)
    else:
        centers = _grid_centers(
            obstacle_grid.origin,
            obstacle_grid.cell_size,
            obstacle_grid.grid_shape,
            blocked_cells=obstacle_grid.blocked_cells,
        )
    if centers.size == 0:
        raise ValueError("No free grid cells available for measurement.")
    selected = _farthest_point_sample(
        centers,
        min(num_measurements, centers.shape[0]),
    )
    if num_measurements > selected.shape[0]:
        repeats = num_measurements // selected.shape[0]
        remainder = num_measurements % selected.shape[0]
        tiled = [selected for _ in range(repeats)]
        if remainder:
            tiled.append(selected[:remainder])
        selected = np.vstack(tiled)
    z = float(env.detector()[2])
    positions = np.column_stack(
        [selected, np.full(selected.shape[0], z, dtype=float)]
    )
    return positions
