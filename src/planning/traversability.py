"""Build and render robot traversability maps for path planning."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
import json
import math
from numbers import Real
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import numpy as np

from measurement.obstacles import ObstacleGrid


def _finite_real(
    value: object,
    *,
    field_name: str,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    """Return a finite real value satisfying a physical domain."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a real number.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite.")
    if strictly_positive and parsed <= 0.0:
        raise ValueError(f"{field_name} must be positive.")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}.")
    return parsed


def _json_integer(
    value: object,
    *,
    field_name: str,
    minimum: int = 0,
) -> int:
    """Return an exact JSON integer with a lower bound."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a JSON integer.")
    if value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}.")
    return value


@dataclass(frozen=True)
class TraversabilityMap:
    """Represent floor cells that a robot center can safely occupy."""

    origin: tuple[float, float]
    cell_size: float
    grid_shape: tuple[int, int]
    traversable_cells: tuple[tuple[int, int], ...]
    robot_radius_m: float = 0.0
    source: str = "projected_3d_environment"

    def __post_init__(self) -> None:
        """Normalize inputs and validate traversable cell bounds."""
        if len(self.origin) != 2:
            raise ValueError("origin must have length 2.")
        if len(self.grid_shape) != 2:
            raise ValueError("grid_shape must have length 2.")
        origin = (
            _finite_real(self.origin[0], field_name="origin[0]"),
            _finite_real(self.origin[1], field_name="origin[1]"),
        )
        cell_size = _finite_real(
            self.cell_size,
            field_name="cell_size",
            strictly_positive=True,
        )
        grid_shape = (
            _json_integer(self.grid_shape[0], field_name="grid_shape[0]"),
            _json_integer(self.grid_shape[1], field_name="grid_shape[1]"),
        )
        robot_radius = _finite_real(
            self.robot_radius_m,
            field_name="robot_radius_m",
            minimum=0.0,
        )
        normalized_cells: list[tuple[int, int]] = []
        for index, cell in enumerate(self.traversable_cells):
            if not isinstance(cell, (list, tuple)) or len(cell) != 2:
                raise ValueError(
                    f"traversable_cells[{index}] must contain two integers."
                )
            normalized_cells.append(
                (
                    _json_integer(
                        cell[0],
                        field_name=f"traversable_cells[{index}][0]",
                    ),
                    _json_integer(
                        cell[1],
                        field_name=f"traversable_cells[{index}][1]",
                    ),
                )
            )
        if len(set(normalized_cells)) != len(normalized_cells):
            raise ValueError("traversable_cells must not contain duplicates.")
        traversable = tuple(sorted(normalized_cells))
        for ix, iy in traversable:
            if ix < 0 or iy < 0 or ix >= grid_shape[0] or iy >= grid_shape[1]:
                raise ValueError("traversable_cells entry out of grid bounds.")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be a nonempty string.")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "cell_size", cell_size)
        object.__setattr__(self, "grid_shape", grid_shape)
        object.__setattr__(self, "traversable_cells", traversable)
        object.__setattr__(self, "robot_radius_m", robot_radius)
        object.__setattr__(self, "_traversable_set", frozenset(traversable))

    @property
    def total_cells(self) -> int:
        """Return the total number of cells in the map."""
        return int(self.grid_shape[0] * self.grid_shape[1])

    @property
    def traversable_fraction(self) -> float:
        """Return the fraction of cells marked traversable."""
        if self.total_cells <= 0:
            return 0.0
        return float(len(self.traversable_cells)) / float(self.total_cells)

    def cell_index(self, point: Sequence[float]) -> tuple[int, int] | None:
        """Return the cell index containing a point, or None outside the map."""
        if len(point) < 2:
            raise ValueError("point must have at least two coordinates.")
        rel_x = float(point[0]) - self.origin[0]
        rel_y = float(point[1]) - self.origin[1]
        if rel_x < 0.0 or rel_y < 0.0:
            return None
        ix = int(np.floor(rel_x / self.cell_size))
        iy = int(np.floor(rel_y / self.cell_size))
        if ix < 0 or iy < 0:
            return None
        if ix >= self.grid_shape[0] or iy >= self.grid_shape[1]:
            return None
        return ix, iy

    def cell_center(self, cell: tuple[int, int]) -> tuple[float, float]:
        """Return the world-space center of a cell."""
        return (
            self.origin[0] + (float(cell[0]) + 0.5) * self.cell_size,
            self.origin[1] + (float(cell[1]) + 0.5) * self.cell_size,
        )

    def is_free_cell(self, cell: tuple[int, int]) -> bool:
        """Return True when a cell is traversable."""
        return (int(cell[0]), int(cell[1])) in self._traversable_set

    def is_free(self, point: Sequence[float]) -> bool:
        """Return True when a point lies in a traversable cell."""
        idx = self.cell_index(point)
        if idx is None:
            return False
        return idx in self._traversable_set

    def is_free_batch(self, points: Sequence[Sequence[float]]) -> np.ndarray:
        """Return traversability flags for a batch of world-space points."""
        points_array = np.asarray(points, dtype=float)
        if points_array.size == 0:
            return np.zeros(0, dtype=bool)
        if points_array.ndim != 2 or points_array.shape[1] < 2:
            raise ValueError("points must have shape (N, D) with D >= 2.")
        if np.any(~np.isfinite(points_array[:, :2])):
            raise ValueError("point coordinates must be finite.")
        relative_xy = points_array[:, :2] - np.asarray(
            self.origin,
            dtype=float,
        )[None, :]
        cell_indices = np.floor(relative_xy / float(self.cell_size)).astype(
            np.int64,
        )
        inside = (
            (cell_indices[:, 0] >= 0)
            & (cell_indices[:, 1] >= 0)
            & (cell_indices[:, 0] < int(self.grid_shape[0]))
            & (cell_indices[:, 1] < int(self.grid_shape[1]))
        )
        free = np.zeros(points_array.shape[0], dtype=bool)
        if not np.any(inside) or not self.traversable_cells:
            return free
        cell_codes = (
            cell_indices[:, 0] * int(self.grid_shape[1]) + cell_indices[:, 1]
        )
        traversable = np.asarray(
            self.traversable_cells,
            dtype=np.int64,
        ).reshape(-1, 2)
        traversable_codes = (
            traversable[:, 0] * int(self.grid_shape[1]) + traversable[:, 1]
        )
        free[inside] = np.isin(cell_codes[inside], traversable_codes)
        return free

    def shortest_path_cells(
        self,
        start_point: Sequence[float],
        goal_point: Sequence[float],
        *,
        allow_diagonal: bool = True,
    ) -> tuple[tuple[int, int], ...] | None:
        """Return the shortest free-cell path between two world points."""
        return shortest_grid_path_cells(
            self,
            start_point,
            goal_point,
            allow_diagonal=allow_diagonal,
        )

    def shortest_path_points(
        self,
        start_point: Sequence[float],
        goal_point: Sequence[float],
        *,
        allow_diagonal: bool = True,
    ) -> np.ndarray | None:
        """Return world-space waypoints for the shortest traversable path."""
        return shortest_grid_path_points(
            self,
            start_point,
            goal_point,
            allow_diagonal=allow_diagonal,
        )

    def shortest_path_length(
        self,
        start_point: Sequence[float],
        goal_point: Sequence[float],
        *,
        allow_diagonal: bool = True,
    ) -> float:
        """Return the shortest traversable path length, or inf when disconnected."""
        path = self.shortest_path_points(
            start_point,
            goal_point,
            allow_diagonal=allow_diagonal,
        )
        return _polyline_length(path)

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation of the map."""
        return {
            "version": 1,
            "source": self.source,
            "origin": [self.origin[0], self.origin[1]],
            "cell_size": self.cell_size,
            "grid_shape": [self.grid_shape[0], self.grid_shape[1]],
            "robot_radius_m": self.robot_radius_m,
            "traversable_fraction": self.traversable_fraction,
            "traversable_cells": [list(cell) for cell in self.traversable_cells],
        }

    def save(self, path: Path) -> None:
        """Save the traversability map to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")

    @classmethod
    def from_dict(cls, data: dict) -> TraversabilityMap:
        """Construct a TraversabilityMap from a dictionary payload."""
        if not isinstance(data, dict):
            raise ValueError("Traversability map must be a dict.")
        schema_keys = {
            "version",
            "source",
            "origin",
            "cell_size",
            "grid_shape",
            "robot_radius_m",
            "traversable_fraction",
            "traversable_cells",
        }
        missing = sorted(schema_keys - set(data))
        unknown = sorted(set(data) - schema_keys)
        if missing or unknown:
            raise ValueError(
                "Traversability map schema mismatch: "
                f"missing={missing}, unknown={unknown}."
            )
        version = _json_integer(
            data["version"],
            field_name="version",
            minimum=1,
        )
        if version != 1:
            raise ValueError("Only traversability map version 1 is supported.")
        origin = data["origin"]
        grid_shape = data["grid_shape"]
        if not isinstance(origin, list) or len(origin) != 2:
            raise ValueError("origin must be a two-element JSON array.")
        if not isinstance(grid_shape, list) or len(grid_shape) != 2:
            raise ValueError("grid_shape must be a two-element JSON array.")
        cells = data["traversable_cells"]
        if not isinstance(cells, list):
            raise ValueError("traversable_cells must be a list.")
        result = cls(
            origin=(origin[0], origin[1]),
            cell_size=data["cell_size"],
            grid_shape=(grid_shape[0], grid_shape[1]),
            traversable_cells=tuple(cells),
            robot_radius_m=data["robot_radius_m"],
            source=data["source"],
        )
        declared_fraction = _finite_real(
            data["traversable_fraction"],
            field_name="traversable_fraction",
            minimum=0.0,
        )
        if declared_fraction > 1.0 or not np.isclose(
            declared_fraction,
            result.traversable_fraction,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "traversable_fraction disagrees with traversable_cells."
            )
        return result

    @classmethod
    def load(cls, path: Path) -> TraversabilityMap:
        """Load a traversability map from JSON."""
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)


def _cell_center(
    origin: tuple[float, float],
    cell_size: float,
    cell: tuple[int, int],
) -> tuple[float, float]:
    """Return the center of a cell in world coordinates."""
    return (
        origin[0] + (float(cell[0]) + 0.5) * cell_size,
        origin[1] + (float(cell[1]) + 0.5) * cell_size,
    )


def _disk_intersects_rect(
    center_xy: tuple[float, float],
    radius_m: float,
    rect: tuple[float, float, float, float],
) -> bool:
    """Return True when a disk intersects an axis-aligned rectangle."""
    x, y = center_xy
    x0, x1, y0, y1 = rect
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    return dx * dx + dy * dy <= radius_m * radius_m


def _neighbors(
    cell: tuple[int, int],
    grid_shape: tuple[int, int],
) -> Iterable[tuple[int, int]]:
    """Yield 4-connected neighboring cells inside grid bounds."""
    ix, iy = cell
    nx, ny = grid_shape
    for neighbor in ((ix - 1, iy), (ix + 1, iy), (ix, iy - 1), (ix, iy + 1)):
        if 0 <= neighbor[0] < nx and 0 <= neighbor[1] < ny:
            yield neighbor


def _map_cell_center(map_api: object, cell: tuple[int, int]) -> tuple[float, float]:
    """Return a map cell center for any grid-like map API."""
    fn = getattr(map_api, "cell_center", None)
    if callable(fn):
        center = fn(cell)
        return float(center[0]), float(center[1])
    if not hasattr(map_api, "origin") or not hasattr(map_api, "cell_size"):
        raise TypeError(
            "A grid map without cell_center must define origin and cell_size."
        )
    origin = np.asarray(map_api.origin, dtype=float)
    cell_size = float(map_api.cell_size)
    if (
        origin.shape != (2,)
        or np.any(~np.isfinite(origin))
        or not np.isfinite(cell_size)
        or cell_size <= 0.0
    ):
        raise ValueError("Grid-map origin and cell_size are invalid.")
    return (
        float(origin[0]) + (float(cell[0]) + 0.5) * cell_size,
        float(origin[1]) + (float(cell[1]) + 0.5) * cell_size,
    )


def _free_cell_fn(map_api: object):
    """Return the free-cell predicate for a grid-like map API."""
    for attr in ("is_free_cell", "is_cell_free"):
        fn = getattr(map_api, attr, None)
        if callable(fn):
            return fn
    return None


def _path_neighbors(
    map_api: object,
    cell: tuple[int, int],
    *,
    allow_diagonal: bool,
) -> Iterable[tuple[tuple[int, int], float]]:
    """Yield free neighbor cells and step lengths for A* path planning."""
    grid_shape = getattr(map_api, "grid_shape", None)
    if grid_shape is None:
        return
    is_free_cell = _free_cell_fn(map_api)
    if is_free_cell is None:
        return
    transition_is_free = getattr(map_api, "is_transition_free", None)
    nx, ny = int(grid_shape[0]), int(grid_shape[1])
    if not hasattr(map_api, "cell_size"):
        raise TypeError("A path-planning map must define cell_size.")
    cell_size = float(map_api.cell_size)
    if not np.isfinite(cell_size) or cell_size <= 0.0:
        raise ValueError("Path-planning cell_size must be finite and positive.")
    ix, iy = int(cell[0]), int(cell[1])
    moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
    if allow_diagonal:
        moves.extend(
            [
                (-1, -1, np.sqrt(2.0)),
                (-1, 1, np.sqrt(2.0)),
                (1, -1, np.sqrt(2.0)),
                (1, 1, np.sqrt(2.0)),
            ]
        )
    for dx, dy, multiplier in moves:
        neighbor = (ix + dx, iy + dy)
        if neighbor[0] < 0 or neighbor[1] < 0:
            continue
        if neighbor[0] >= nx or neighbor[1] >= ny:
            continue
        if not bool(is_free_cell(neighbor)):
            continue
        if dx != 0 and dy != 0:
            side_a = (ix + dx, iy)
            side_b = (ix, iy + dy)
            if not bool(is_free_cell(side_a)) or not bool(is_free_cell(side_b)):
                continue
        if callable(transition_is_free) and not bool(
            transition_is_free(cell, neighbor)
        ):
            continue
        yield neighbor, float(multiplier) * cell_size


def _cell_heuristic(
    map_api: object,
    cell: tuple[int, int],
    goal_cell: tuple[int, int],
) -> float:
    """Return the Euclidean grid heuristic for A*."""
    x0, y0 = _map_cell_center(map_api, cell)
    x1, y1 = _map_cell_center(map_api, goal_cell)
    return float(np.hypot(x1 - x0, y1 - y0))


def _reconstruct_path(
    came_from: dict[tuple[int, int], tuple[int, int]],
    goal_cell: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    """Return an ordered cell path from an A* predecessor map."""
    path = [goal_cell]
    current = goal_cell
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return tuple(path)


def shortest_grid_path_cells(
    map_api: object,
    start_point: Sequence[float],
    goal_point: Sequence[float],
    *,
    allow_diagonal: bool = True,
) -> tuple[tuple[int, int], ...] | None:
    """Return an obstacle-aware shortest path over a grid-like map API."""
    cell_index = getattr(map_api, "cell_index", None)
    is_free_cell = _free_cell_fn(map_api)
    grid_shape = getattr(map_api, "grid_shape", None)
    if not callable(cell_index) or is_free_cell is None or grid_shape is None:
        return None
    start_cell = cell_index(start_point)
    goal_cell = cell_index(goal_point)
    if start_cell is None or goal_cell is None:
        return None
    start_cell = (int(start_cell[0]), int(start_cell[1]))
    goal_cell = (int(goal_cell[0]), int(goal_cell[1]))
    if not bool(is_free_cell(start_cell)) or not bool(is_free_cell(goal_cell)):
        return None
    if start_cell == goal_cell:
        return (start_cell,)

    frontier: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(frontier, (0.0, 0.0, start_cell))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    cost_so_far: dict[tuple[int, int], float] = {start_cell: 0.0}
    while frontier:
        _, current_cost, current = heapq.heappop(frontier)
        if current == goal_cell:
            return _reconstruct_path(came_from, goal_cell)
        if current_cost > cost_so_far.get(current, float("inf")):
            continue
        for neighbor, step_cost in _path_neighbors(
            map_api,
            current,
            allow_diagonal=allow_diagonal,
        ):
            new_cost = current_cost + float(step_cost)
            if new_cost >= cost_so_far.get(neighbor, float("inf")):
                continue
            cost_so_far[neighbor] = new_cost
            came_from[neighbor] = current
            priority = new_cost + _cell_heuristic(map_api, neighbor, goal_cell)
            heapq.heappush(frontier, (priority, new_cost, neighbor))
    return None


def _coerce_xyz(point: Sequence[float], z_default: float = 0.0) -> np.ndarray:
    """Return a 3D point array, filling z when only xy is provided."""
    arr = np.asarray(point, dtype=float).ravel()
    if arr.size < 2:
        raise ValueError("point must have at least two coordinates.")
    if arr.size >= 3:
        return arr[:3].astype(float)
    return np.array([arr[0], arr[1], float(z_default)], dtype=float)


def shortest_grid_path_points(
    map_api: object,
    start_point: Sequence[float],
    goal_point: Sequence[float],
    *,
    allow_diagonal: bool = True,
) -> np.ndarray | None:
    """Return world-space waypoints for an obstacle-aware grid path."""
    start = _coerce_xyz(start_point)
    goal = _coerce_xyz(goal_point, z_default=float(start[2]))
    cells = shortest_grid_path_cells(
        map_api,
        start,
        goal,
        allow_diagonal=allow_diagonal,
    )
    if cells is None:
        return None
    if len(cells) <= 1:
        return np.vstack([start, goal]).astype(float)
    z_center = float(start[2])
    centers = []
    for cell in cells[1:-1]:
        x_val, y_val = _map_cell_center(map_api, cell)
        centers.append(np.array([x_val, y_val, z_center], dtype=float))
    points = [start]
    points.extend(centers)
    points.append(goal)
    return _dedupe_path_points(np.vstack(points).astype(float))


def _dedupe_path_points(points: np.ndarray) -> np.ndarray:
    """Remove consecutive duplicate points from a path polyline."""
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return np.zeros((0, 3), dtype=float)
    keep = [0]
    for idx in range(1, arr.shape[0]):
        if float(np.linalg.norm(arr[idx] - arr[keep[-1]])) > 1e-9:
            keep.append(idx)
    return arr[keep].astype(float)


def _polyline_length(points: np.ndarray | None) -> float:
    """Return the length of a 3D polyline, or inf for a missing path."""
    if points is None:
        return float("inf")
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != 3:
        return 0.0
    deltas = np.diff(arr, axis=0)
    return float(np.sum(np.linalg.norm(deltas, axis=1)))


def shortest_grid_path_length(
    map_api: object,
    start_point: Sequence[float],
    goal_point: Sequence[float],
    *,
    allow_diagonal: bool = True,
) -> float:
    """Return the obstacle-aware path length over a grid-like map API."""
    points = shortest_grid_path_points(
        map_api,
        start_point,
        goal_point,
        allow_diagonal=allow_diagonal,
    )
    return _polyline_length(points)


def _reachable_cells(
    free_cells: set[tuple[int, int]],
    *,
    start: tuple[int, int],
    grid_shape: tuple[int, int],
) -> set[tuple[int, int]]:
    """Return the connected free component reachable from start."""
    if start not in free_cells:
        return set()
    visited = {start}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        cell = queue.popleft()
        for neighbor in _neighbors(cell, grid_shape):
            if neighbor in visited or neighbor not in free_cells:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    return visited


def _build_map_from_footprints(
    *,
    obstacle_rects: Iterable[tuple[float, float, float, float]],
    origin: tuple[float, float],
    cell_size: float,
    grid_shape: tuple[int, int],
    robot_radius_m: float,
    reachable_from: Sequence[float] | None,
    source: str,
) -> TraversabilityMap:
    """Build a traversability map from projected obstacle rectangles."""
    radius = _finite_real(
        robot_radius_m,
        field_name="robot_radius_m",
        minimum=0.0,
    )
    normalized_origin = (
        _finite_real(origin[0], field_name="origin[0]"),
        _finite_real(origin[1], field_name="origin[1]"),
    )
    normalized_cell_size = _finite_real(
        cell_size,
        field_name="cell_size",
        strictly_positive=True,
    )
    normalized_grid_shape = (
        _json_integer(grid_shape[0], field_name="grid_shape[0]", minimum=1),
        _json_integer(grid_shape[1], field_name="grid_shape[1]", minimum=1),
    )
    if not isinstance(source, str) or not source:
        raise ValueError("source must be a nonempty string.")
    normalized_rects: list[tuple[float, float, float, float]] = []
    for index, rect in enumerate(obstacle_rects):
        if not isinstance(rect, (list, tuple)) or len(rect) != 4:
            raise ValueError(f"obstacle_rects[{index}] must contain four values.")
        parsed = tuple(
            _finite_real(value, field_name=f"obstacle_rects[{index}]")
            for value in rect
        )
        if parsed[1] < parsed[0] or parsed[3] < parsed[2]:
            raise ValueError("Obstacle footprint bounds must be ordered.")
        normalized_rects.append(parsed)
    rects = tuple(normalized_rects)
    free_cells: set[tuple[int, int]] = set()
    nx, ny = normalized_grid_shape
    for ix in range(nx):
        for iy in range(ny):
            center = _cell_center(
                normalized_origin,
                normalized_cell_size,
                (ix, iy),
            )
            blocked = any(_disk_intersects_rect(center, radius, rect) for rect in rects)
            if not blocked:
                free_cells.add((ix, iy))
    if reachable_from is not None:
        start_point = np.asarray(reachable_from, dtype=float)
        if (
            start_point.ndim != 1
            or start_point.size < 2
            or not np.all(np.isfinite(start_point))
        ):
            raise ValueError("reachable_from must be one finite 2-D or 3-D point.")
        start = _point_to_cell_index(
            start_point,
            normalized_origin,
            normalized_cell_size,
            normalized_grid_shape,
        )
        if start is None or start not in free_cells:
            raise ValueError(
                "reachable_from is outside the grid or blocked by robot clearance."
            )
        free_cells = _reachable_cells(
            free_cells,
            start=start,
            grid_shape=normalized_grid_shape,
        )
    if not free_cells:
        raise ValueError("Traversability geometry contains no reachable free cell.")
    return TraversabilityMap(
        origin=normalized_origin,
        cell_size=normalized_cell_size,
        grid_shape=normalized_grid_shape,
        traversable_cells=tuple(sorted(free_cells)),
        robot_radius_m=radius,
        source=source,
    )


def _point_to_cell_index(
    point: Sequence[float],
    origin: tuple[float, float],
    cell_size: float,
    grid_shape: tuple[int, int],
) -> tuple[int, int] | None:
    """Return the cell index for a point, or None outside the grid."""
    if len(point) < 2:
        raise ValueError("point must have at least two coordinates.")
    rel_x = float(point[0]) - origin[0]
    rel_y = float(point[1]) - origin[1]
    if rel_x < 0.0 or rel_y < 0.0:
        return None
    ix = int(np.floor(rel_x / cell_size))
    iy = int(np.floor(rel_y / cell_size))
    if ix < 0 or iy < 0 or ix >= grid_shape[0] or iy >= grid_shape[1]:
        return None
    return ix, iy


def _quat_rotate(
    vector_xyz: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
) -> np.ndarray:
    """Rotate a vector by a quaternion."""
    vector = np.asarray(vector_xyz, dtype=float)
    w, x, y, z = (float(value) for value in quat_wxyz)
    q_vec = np.asarray((x, y, z), dtype=float)
    return vector + 2.0 * np.cross(q_vec, np.cross(q_vec, vector) + w * vector)


def _finite_vector(
    value: object,
    *,
    field_name: str,
    length: int,
    strictly_positive: bool = False,
) -> tuple[float, ...]:
    """Return an exact-length finite numeric vector."""
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"{field_name} must be a {length}-element vector.")
    if len(value) != length:
        raise ValueError(f"{field_name} must be a {length}-element vector.")
    return tuple(
        _finite_real(
            component,
            field_name=f"{field_name}[{index}]",
            strictly_positive=strictly_positive,
        )
        for index, component in enumerate(value)
    )


def _unit_quaternion(
    value: object,
    *,
    field_name: str,
) -> tuple[float, float, float, float]:
    """Return a finite unit quaternion without accepting invalid geometry."""
    parsed = _finite_vector(
        value,
        field_name=field_name,
        length=4,
    )
    norm = float(np.linalg.norm(np.asarray(parsed, dtype=float)))
    if not np.isclose(norm, 1.0, rtol=0.0, atol=1.0e-6):
        raise ValueError(f"{field_name} must be a unit quaternion.")
    return parsed


def _box_footprint_rect(
    center_xyz: tuple[float, float, float],
    orientation_wxyz: tuple[float, float, float, float],
    size_xyz: tuple[float, float, float],
    blocking_z_range_m: tuple[float, float],
) -> tuple[float, float, float, float] | None:
    """Return the projected footprint of a possibly oriented box."""
    half = np.asarray(size_xyz, dtype=float) * 0.5
    center = np.asarray(center_xyz, dtype=float)
    corners: list[np.ndarray] = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                local = (sx * half[0], sy * half[1], sz * half[2])
                corners.append(center + _quat_rotate(local, orientation_wxyz))
    points = np.vstack(corners)
    if float(np.max(points[:, 2])) < blocking_z_range_m[0]:
        return None
    if float(np.min(points[:, 2])) > blocking_z_range_m[1]:
        return None
    return (
        float(np.min(points[:, 0])),
        float(np.max(points[:, 0])),
        float(np.min(points[:, 1])),
        float(np.max(points[:, 1])),
    )


def _solid_footprint_rects(
    solid: object,
    *,
    blocking_z_range_m: tuple[float, float],
) -> list[tuple[float, float, float, float]]:
    """Return projected blocking footprint rectangles for a stage solid."""
    path = getattr(solid, "path", None)
    if not isinstance(path, str) or not path:
        raise ValueError("Stage solid path must be a nonempty string.")
    shape = getattr(solid, "shape", None)
    if shape not in {"box", "sphere", "mesh"}:
        raise ValueError(
            f"Stage solid {path!r} has unsupported shape {shape!r}."
        )
    pose = getattr(solid, "pose", None)
    if pose is None:
        raise ValueError(f"Stage solid {path!r} must define a world pose.")
    center = _finite_vector(
        getattr(pose, "translation_xyz", None),
        field_name=f"{path}.translation_xyz",
        length=3,
    )
    orientation = _unit_quaternion(
        getattr(pose, "orientation_wxyz", None),
        field_name=f"{path}.orientation_wxyz",
    )
    if shape == "box":
        size = getattr(solid, "size_xyz", None)
        if size is None:
            raise ValueError(f"Box stage solid {path!r} must define size_xyz.")
        parsed_size = _finite_vector(
            size,
            field_name=f"{path}.size_xyz",
            length=3,
            strictly_positive=True,
        )
        rect = _box_footprint_rect(
            center,
            orientation,
            parsed_size,
            blocking_z_range_m,
        )
        return [] if rect is None else [rect]
    if shape == "sphere":
        radius = getattr(solid, "radius_m", None)
        if radius is None:
            raise ValueError(f"Sphere stage solid {path!r} must define radius_m.")
        r = _finite_real(
            radius,
            field_name=f"{path}.radius_m",
            strictly_positive=True,
        )
        if center[2] + r < blocking_z_range_m[0] or center[2] - r > blocking_z_range_m[1]:
            return []
        return [(center[0] - r, center[0] + r, center[1] - r, center[1] + r)]
    triangles = getattr(solid, "triangles_xyz", None)
    if not isinstance(triangles, (list, tuple)) or not triangles:
        raise ValueError(
            f"Mesh stage solid {path!r} must define nonempty triangles_xyz."
        )
    rects: list[tuple[float, float, float, float]] = []
    for index, triangle in enumerate(triangles):
        if not isinstance(triangle, (list, tuple, np.ndarray)):
            raise ValueError(
                f"{path}.triangles_xyz[{index}] must have shape (3, 3)."
            )
        points = np.asarray(triangle, dtype=float)
        if points.shape != (3, 3) or np.any(~np.isfinite(points)):
            raise ValueError(
                f"{path}.triangles_xyz[{index}] must be a finite (3, 3) array."
            )
        twice_area = np.linalg.norm(
            np.cross(points[1] - points[0], points[2] - points[0])
        )
        if not np.isfinite(twice_area) or twice_area <= 1.0e-12:
            raise ValueError(
                f"{path}.triangles_xyz[{index}] must be nondegenerate."
            )
        if float(np.max(points[:, 2])) < blocking_z_range_m[0]:
            continue
        if float(np.min(points[:, 2])) > blocking_z_range_m[1]:
            continue
        rects.append(
            (
                float(np.min(points[:, 0])),
                float(np.max(points[:, 0])),
                float(np.min(points[:, 1])),
                float(np.max(points[:, 1])),
            )
        )
    return rects


def build_traversability_map_from_stage_solids(
    solids: Iterable[object],
    *,
    origin: tuple[float, float],
    cell_size: float,
    grid_shape: tuple[int, int],
    robot_radius_m: float = 0.35,
    reachable_from: Sequence[float] | None = None,
    blocking_z_range_m: tuple[float, float] = (0.05, 2.0),
) -> TraversabilityMap:
    """
    Build a traversability map from USD/Isaac stage solid geometry.

    Stage solids can come from ``StageBackend.list_solid_prims()``. Mesh,
    sphere, and box prims are projected to floor-space obstacle footprints,
    filtered by the robot blocking height range, then converted into robot
    center free cells.
    """
    if (
        not isinstance(blocking_z_range_m, (list, tuple, np.ndarray))
        or len(blocking_z_range_m) != 2
    ):
        raise ValueError("blocking_z_range_m must contain two finite values.")
    blocking_range = (
        _finite_real(
            blocking_z_range_m[0],
            field_name="blocking_z_range_m[0]",
        ),
        _finite_real(
            blocking_z_range_m[1],
            field_name="blocking_z_range_m[1]",
        ),
    )
    if blocking_range[1] <= blocking_range[0]:
        raise ValueError("blocking_z_range_m bounds must be strictly ordered.")
    rects: list[tuple[float, float, float, float]] = []
    for solid in solids:
        rects.extend(
            _solid_footprint_rects(
                solid,
                blocking_z_range_m=blocking_range,
            )
        )
    return _build_map_from_footprints(
        obstacle_rects=rects,
        origin=origin,
        cell_size=cell_size,
        grid_shape=grid_shape,
        robot_radius_m=robot_radius_m,
        reachable_from=reachable_from,
        source="stage_projected_3d_environment",
    )


def build_traversability_map_from_obstacle_grid(
    obstacle_grid: ObstacleGrid,
    *,
    robot_radius_m: float = 0.35,
    reachable_from: Sequence[float] | None = None,
) -> TraversabilityMap:
    """
    Project 3D obstacle cells to a 2D robot-center traversability map.

    The random 3D environment authors each blocked cell as a vertical obstacle
    volume. This function projects those volumes onto the floor plane, rejects
    robot-center cells whose footprint intersects an obstacle footprint, and can
    keep only the free component reachable from the robot start.
    """
    return _build_map_from_footprints(
        obstacle_rects=obstacle_grid.blocked_bounds(),
        origin=obstacle_grid.origin,
        cell_size=obstacle_grid.cell_size,
        grid_shape=obstacle_grid.grid_shape,
        robot_radius_m=robot_radius_m,
        reachable_from=reachable_from,
        source="projected_3d_environment",
    )


def render_traversability_map(
    traversability_map: TraversabilityMap,
    output_path: Path,
) -> None:
    """Render a 2D image showing only robot-traversable cells."""
    xmin, ymin = traversability_map.origin
    xmax = xmin + traversability_map.grid_shape[0] * traversability_map.cell_size
    ymax = ymin + traversability_map.grid_shape[1] * traversability_map.cell_size
    patches: list[Rectangle] = []
    for cell in traversability_map.traversable_cells:
        x, y = traversability_map.cell_center(cell)
        half = 0.5 * traversability_map.cell_size
        patches.append(
            Rectangle(
                (x - half, y - half),
                traversability_map.cell_size,
                traversability_map.cell_size,
            )
        )

    fig, ax = plt.subplots(figsize=(6, 10))
    ax.set_facecolor("#111111")
    if patches:
        collection = PatchCollection(
            patches,
            facecolor="#E8F5E9",
            edgecolor="#81C784",
            linewidth=0.35,
        )
        ax.add_collection(collection)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Robot traversable map")
    ax.grid(True, color="#444444", linewidth=0.5, alpha=0.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
