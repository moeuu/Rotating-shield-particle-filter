"""Build and render robot traversability maps for path planning."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import numpy as np

from measurement.obstacles import ObstacleGrid


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
        origin = (float(self.origin[0]), float(self.origin[1]))
        cell_size = float(self.cell_size)
        grid_shape = (int(self.grid_shape[0]), int(self.grid_shape[1]))
        if cell_size <= 0.0:
            raise ValueError("cell_size must be positive.")
        if grid_shape[0] < 0 or grid_shape[1] < 0:
            raise ValueError("grid_shape must be non-negative.")
        traversable = tuple(
            sorted((int(cell[0]), int(cell[1])) for cell in self.traversable_cells)
        )
        for ix, iy in traversable:
            if ix < 0 or iy < 0 or ix >= grid_shape[0] or iy >= grid_shape[1]:
                raise ValueError("traversable_cells entry out of grid bounds.")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "cell_size", cell_size)
        object.__setattr__(self, "grid_shape", grid_shape)
        object.__setattr__(self, "traversable_cells", traversable)
        object.__setattr__(self, "robot_radius_m", float(self.robot_radius_m))
        object.__setattr__(self, "source", str(self.source))
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
        origin = data.get("origin", (0.0, 0.0))
        grid_shape = data.get("grid_shape")
        if grid_shape is None:
            raise ValueError("Traversability map missing 'grid_shape'.")
        cells = data.get("traversable_cells", [])
        if not isinstance(cells, list):
            raise ValueError("traversable_cells must be a list.")
        return cls(
            origin=(float(origin[0]), float(origin[1])),
            cell_size=float(data.get("cell_size", 1.0)),
            grid_shape=(int(grid_shape[0]), int(grid_shape[1])),
            traversable_cells=tuple((int(cell[0]), int(cell[1])) for cell in cells),
            robot_radius_m=float(data.get("robot_radius_m", 0.0)),
            source=str(data.get("source", "projected_3d_environment")),
        )

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
    radius = max(0.0, float(robot_radius_m))
    rects = tuple(obstacle_rects)
    free_cells: set[tuple[int, int]] = set()
    nx, ny = grid_shape
    for ix in range(nx):
        for iy in range(ny):
            center = _cell_center(origin, cell_size, (ix, iy))
            blocked = any(_disk_intersects_rect(center, radius, rect) for rect in rects)
            if not blocked:
                free_cells.add((ix, iy))
    if reachable_from is not None:
        start = _point_to_cell_index(reachable_from, origin, cell_size, grid_shape)
        free_cells = (
            set()
            if start is None
            else _reachable_cells(free_cells, start=start, grid_shape=grid_shape)
        )
    return TraversabilityMap(
        origin=origin,
        cell_size=cell_size,
        grid_shape=grid_shape,
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
    shape = str(getattr(solid, "shape", "")).lower()
    pose = getattr(solid, "pose")
    center = tuple(float(value) for value in getattr(pose, "translation_xyz"))
    orientation = tuple(float(value) for value in getattr(pose, "orientation_wxyz"))
    if shape == "box":
        size = getattr(solid, "size_xyz", None)
        if size is None:
            return []
        rect = _box_footprint_rect(
            center,
            orientation,
            tuple(float(value) for value in size),
            blocking_z_range_m,
        )
        return [] if rect is None else [rect]
    if shape == "sphere":
        radius = getattr(solid, "radius_m", None)
        if radius is None:
            return []
        r = float(radius)
        if center[2] + r < blocking_z_range_m[0] or center[2] - r > blocking_z_range_m[1]:
            return []
        return [(center[0] - r, center[0] + r, center[1] - r, center[1] + r)]
    if shape != "mesh":
        return []
    rects: list[tuple[float, float, float, float]] = []
    for triangle in getattr(solid, "triangles_xyz", ()) or ():
        points = np.asarray(triangle, dtype=float)
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
    rects: list[tuple[float, float, float, float]] = []
    for solid in solids:
        rects.extend(
            _solid_footprint_rects(
                solid,
                blocking_z_range_m=blocking_z_range_m,
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
