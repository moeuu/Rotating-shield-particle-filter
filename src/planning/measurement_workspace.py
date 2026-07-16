"""Three-dimensional free space for detector measurement and transport poses."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import heapq
import os
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


_GEOMETRY_TOLERANCE_M = 1.0e-9
_DEFAULT_ELEMENT_BUDGET = 4_000_000
_MOTION_CACHE_MAX_ENTRIES = 20_000
_MOTION_CACHE_MISS = object()
_DEFAULT_MOTION_GRID_CELL_SIZE_M = 0.25
_GRID_NEIGHBOR_OFFSETS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


def _validated_xyz_tuple(
    values: Sequence[float],
    *,
    field_name: str,
) -> tuple[float, float, float]:
    """Return a finite three-coordinate tuple."""
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.shape != (3,):
        raise ValueError(f"{field_name} must contain exactly three values.")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{field_name} must contain only finite values.")
    return float(array[0]), float(array[1]), float(array[2])


def _validated_nonnegative_float(value: float, *, field_name: str) -> float:
    """Return a finite non-negative geometry value."""
    normalized = float(value)
    if not np.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative.")
    return normalized


def _validated_positive_float(value: float, *, field_name: str) -> float:
    """Return a finite positive geometry value."""
    normalized = float(value)
    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{field_name} must be finite and positive.")
    return normalized


def _coerce_points_batch(
    points_xyz: Sequence[Sequence[float]] | NDArray[np.float64],
    *,
    field_name: str,
) -> NDArray[np.float64]:
    """Return a validated ``(N, 3)`` point array."""
    points = np.asarray(points_xyz, dtype=float)
    if points.size == 0:
        return np.zeros((0, 3), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{field_name} must be shaped (N, 3).")
    if np.any(~np.isfinite(points)):
        raise ValueError(f"{field_name} must contain only finite values.")
    return points


def _coerce_collision_boxes(
    boxes_m: Sequence[Sequence[float]] | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return validated collision AABBs as ``(x0, y0, z0, x1, y1, z1)`` rows."""
    boxes = np.asarray(boxes_m, dtype=float)
    if boxes.size == 0:
        return np.zeros((0, 6), dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] != 6:
        raise ValueError("collision_boxes_m must be shaped (B, 6).")
    if np.any(~np.isfinite(boxes)):
        raise ValueError("collision_boxes_m must contain only finite values.")
    if np.any(boxes[:, 3:] < boxes[:, :3]):
        raise ValueError(
            "collision_boxes_m entries must be ordered lower-to-upper."
        )
    result = np.array(boxes, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _candidate_chunk_size(
    candidate_count: int,
    box_count: int,
    element_budget: int,
) -> int:
    """Return a bounded candidate chunk size for candidate-by-box operations."""
    if candidate_count <= 0:
        return 1
    return max(
        1,
        min(
            int(candidate_count),
            max(int(element_budget), 1) // max(int(box_count), 1),
        ),
    )


@dataclass(frozen=True)
class AxisAlignedRoomBounds:
    """Describe the closed axis-aligned volume available to the robot assembly."""

    lower_xyz: tuple[float, float, float]
    upper_xyz: tuple[float, float, float]

    def __post_init__(self) -> None:
        """Normalize coordinates and require positive room extent."""
        lower = _validated_xyz_tuple(self.lower_xyz, field_name="lower_xyz")
        upper = _validated_xyz_tuple(self.upper_xyz, field_name="upper_xyz")
        if np.any(np.asarray(upper) <= np.asarray(lower)):
            raise ValueError("upper_xyz must be strictly greater than lower_xyz.")
        object.__setattr__(self, "lower_xyz", lower)
        object.__setattr__(self, "upper_xyz", upper)

    @property
    def size_xyz(self) -> tuple[float, float, float]:
        """Return the room extent along each axis."""
        lower = np.asarray(self.lower_xyz, dtype=float)
        upper = np.asarray(self.upper_xyz, dtype=float)
        size = upper - lower
        return float(size[0]), float(size[1]), float(size[2])


@dataclass(frozen=True)
class DetectorAssemblyGeometry:
    """Describe conservative robot, mast, and detector-head collision geometry."""

    base_radius_m: float
    base_height_m: float
    mast_radius_m: float
    head_radius_m: float

    def __post_init__(self) -> None:
        """Normalize dimensions and validate the nested cylindrical envelope."""
        base_radius = _validated_positive_float(
            self.base_radius_m,
            field_name="base_radius_m",
        )
        base_height = _validated_positive_float(
            self.base_height_m,
            field_name="base_height_m",
        )
        mast_radius = _validated_nonnegative_float(
            self.mast_radius_m,
            field_name="mast_radius_m",
        )
        head_radius = _validated_positive_float(
            self.head_radius_m,
            field_name="head_radius_m",
        )
        if mast_radius > base_radius + _GEOMETRY_TOLERANCE_M:
            raise ValueError("mast_radius_m must not exceed base_radius_m.")
        object.__setattr__(self, "base_radius_m", base_radius)
        object.__setattr__(self, "base_height_m", base_height)
        object.__setattr__(self, "mast_radius_m", mast_radius)
        object.__setattr__(self, "head_radius_m", head_radius)


def _sphere_collision_mask_batch(
    centers_xyz: NDArray[np.float64],
    *,
    radius_m: float,
    boxes_m: NDArray[np.float64],
    element_budget: int,
) -> NDArray[np.bool_]:
    """Return exact sphere-versus-AABB collision flags for many centers."""
    centers = np.asarray(centers_xyz, dtype=float).reshape(-1, 3)
    boxes = np.asarray(boxes_m, dtype=float).reshape(-1, 6)
    collision = np.zeros(centers.shape[0], dtype=bool)
    if centers.shape[0] == 0 or boxes.shape[0] == 0:
        return collision
    radius = max(float(radius_m), 0.0) + _GEOMETRY_TOLERANCE_M
    radius_sq = radius * radius
    lower = boxes[:, :3]
    upper = boxes[:, 3:]
    chunk_size = _candidate_chunk_size(
        centers.shape[0],
        boxes.shape[0],
        element_budget,
    )
    for start in range(0, centers.shape[0], chunk_size):
        stop = min(start + chunk_size, centers.shape[0])
        batch = centers[start:stop, None, :]
        closest = np.minimum(np.maximum(batch, lower[None, :, :]), upper[None, :, :])
        distance_sq = np.sum((batch - closest) ** 2, axis=2)
        collision[start:stop] = np.any(distance_sq <= radius_sq, axis=1)
    return collision


def _vertical_cylinder_collision_mask_batch(
    centers_xy: NDArray[np.float64],
    *,
    z_lower_m: NDArray[np.float64],
    z_upper_m: NDArray[np.float64],
    radius_m: float,
    boxes_m: NDArray[np.float64],
    element_budget: int,
) -> NDArray[np.bool_]:
    """Return exact finite vertical-cylinder versus AABB collision flags."""
    centers = np.asarray(centers_xy, dtype=float).reshape(-1, 2)
    lower_z = np.asarray(z_lower_m, dtype=float).reshape(-1)
    upper_z = np.asarray(z_upper_m, dtype=float).reshape(-1)
    boxes = np.asarray(boxes_m, dtype=float).reshape(-1, 6)
    if lower_z.size != centers.shape[0] or upper_z.size != centers.shape[0]:
        raise ValueError("Cylinder z intervals must align with centers_xy.")
    collision = np.zeros(centers.shape[0], dtype=bool)
    if centers.shape[0] == 0 or boxes.shape[0] == 0:
        return collision
    radius = max(float(radius_m), 0.0) + _GEOMETRY_TOLERANCE_M
    radius_sq = radius * radius
    lower_xy = boxes[:, :2]
    upper_xy = boxes[:, 3:5]
    chunk_size = _candidate_chunk_size(
        centers.shape[0],
        boxes.shape[0],
        element_budget,
    )
    for start in range(0, centers.shape[0], chunk_size):
        stop = min(start + chunk_size, centers.shape[0])
        batch = centers[start:stop, None, :]
        closest = np.minimum(
            np.maximum(batch, lower_xy[None, :, :]),
            upper_xy[None, :, :],
        )
        distance_sq = np.sum((batch - closest) ** 2, axis=2)
        z_overlap = (
            upper_z[start:stop, None]
            >= boxes[None, :, 2] - _GEOMETRY_TOLERANCE_M
        ) & (
            lower_z[start:stop, None]
            <= boxes[None, :, 5] + _GEOMETRY_TOLERANCE_M
        )
        collision[start:stop] = np.any(
            (distance_sq <= radius_sq) & z_overlap,
            axis=1,
        )
    return collision


def _vertical_capsule_collision_mask_batch(
    centers_xy: NDArray[np.float64],
    *,
    z_lower_m: NDArray[np.float64],
    z_upper_m: NDArray[np.float64],
    radius_m: float,
    boxes_m: NDArray[np.float64],
    element_budget: int,
) -> NDArray[np.bool_]:
    """Return exact vertical capsule-versus-AABB collision flags."""
    centers = np.asarray(centers_xy, dtype=float).reshape(-1, 2)
    lower_z = np.asarray(z_lower_m, dtype=float).reshape(-1)
    upper_z = np.asarray(z_upper_m, dtype=float).reshape(-1)
    boxes = np.asarray(boxes_m, dtype=float).reshape(-1, 6)
    if lower_z.size != centers.shape[0] or upper_z.size != centers.shape[0]:
        raise ValueError("Capsule z intervals must align with centers_xy.")
    collision = np.zeros(centers.shape[0], dtype=bool)
    if centers.shape[0] == 0 or boxes.shape[0] == 0:
        return collision
    radius = max(float(radius_m), 0.0) + _GEOMETRY_TOLERANCE_M
    radius_sq = radius * radius
    lower_xy = boxes[:, :2]
    upper_xy = boxes[:, 3:5]
    chunk_size = _candidate_chunk_size(
        centers.shape[0],
        boxes.shape[0],
        element_budget,
    )
    for start in range(0, centers.shape[0], chunk_size):
        stop = min(start + chunk_size, centers.shape[0])
        batch = centers[start:stop, None, :]
        closest = np.minimum(
            np.maximum(batch, lower_xy[None, :, :]),
            upper_xy[None, :, :],
        )
        distance_xy_sq = np.sum((batch - closest) ** 2, axis=2)
        distance_z = np.maximum(
            np.maximum(
                boxes[None, :, 2] - upper_z[start:stop, None],
                lower_z[start:stop, None] - boxes[None, :, 5],
            ),
            0.0,
        )
        collision[start:stop] = np.any(
            distance_xy_sq + distance_z**2 <= radius_sq,
            axis=1,
        )
    return collision


def _segments_intersect_expanded_boxes_batch(
    starts_xyz: NDArray[np.float64],
    ends_xyz: NDArray[np.float64],
    *,
    boxes_m: NDArray[np.float64],
    expansion_xyz_m: Sequence[float],
    element_budget: int,
) -> NDArray[np.bool_]:
    """Return conservative segment collisions using Minkowski-expanded AABBs."""
    starts = np.asarray(starts_xyz, dtype=float).reshape(-1, 3)
    ends = np.asarray(ends_xyz, dtype=float).reshape(-1, 3)
    boxes = np.asarray(boxes_m, dtype=float).reshape(-1, 6)
    if starts.shape != ends.shape:
        raise ValueError("starts_xyz and ends_xyz must have the same shape.")
    expansion = np.asarray(expansion_xyz_m, dtype=float).reshape(-1)
    invalid_expansion = (
        expansion.shape != (3,)
        or np.any(~np.isfinite(expansion))
        or np.any(expansion < 0.0)
    )
    if invalid_expansion:
        raise ValueError("expansion_xyz_m must be a finite non-negative 3-vector.")
    collision = np.zeros(starts.shape[0], dtype=bool)
    if starts.shape[0] == 0 or boxes.shape[0] == 0:
        return collision
    lower = boxes[:, :3] - expansion[None, :]
    upper = boxes[:, 3:] + expansion[None, :]
    chunk_size = _candidate_chunk_size(
        starts.shape[0],
        boxes.shape[0],
        element_budget,
    )
    for start_index in range(0, starts.shape[0], chunk_size):
        stop_index = min(start_index + chunk_size, starts.shape[0])
        segment_start = starts[start_index:stop_index, None, :]
        segment_end = ends[start_index:stop_index, None, :]
        direction = segment_end - segment_start
        t_enter = np.zeros((stop_index - start_index, boxes.shape[0]), dtype=float)
        t_exit = np.ones_like(t_enter)
        possible = np.ones_like(t_enter, dtype=bool)
        for axis in range(3):
            value = segment_start[:, :, axis]
            step = direction[:, :, axis]
            axis_lower = lower[None, :, axis]
            axis_upper = upper[None, :, axis]
            parallel = np.abs(step) <= _GEOMETRY_TOLERANCE_M
            outside = parallel & (
                (value < axis_lower - _GEOMETRY_TOLERANCE_M)
                | (value > axis_upper + _GEOMETRY_TOLERANCE_M)
            )
            safe_step = np.where(parallel, 1.0, step)
            t0 = (axis_lower - value) / safe_step
            t1 = (axis_upper - value) / safe_step
            axis_enter = np.minimum(t0, t1)
            axis_exit = np.maximum(t0, t1)
            axis_enter = np.where(parallel, -np.inf, axis_enter)
            axis_exit = np.where(parallel, np.inf, axis_exit)
            t_enter = np.maximum(t_enter, axis_enter)
            t_exit = np.minimum(t_exit, axis_exit)
            possible &= ~outside
        intersects = possible & (
            t_exit >= t_enter - _GEOMETRY_TOLERANCE_M
        )
        collision[start_index:stop_index] = np.any(intersects, axis=1)
    return collision


def _dedupe_consecutive_points(
    points_xyz: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Remove consecutive duplicate waypoints while preserving path order."""
    points = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    if points.shape[0] <= 1:
        return points.copy()
    separation = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate(
        [np.ones(1, dtype=bool), separation > _GEOMETRY_TOLERANCE_M]
    )
    return points[keep]


def _canonical_edge_codes(first_code: int, second_code: int) -> tuple[int, int]:
    """Return one undirected grid edge in deterministic code order."""
    first = int(first_code)
    second = int(second_code)
    return (first, second) if first <= second else (second, first)


def _batched_neighbor_cell_pairs(
    grid_shape: tuple[int, int],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return every unique cardinal and diagonal grid edge as two cell arrays."""
    nx, ny = int(grid_shape[0]), int(grid_shape[1])
    if nx <= 0 or ny <= 0:
        empty = np.zeros((0, 2), dtype=np.int64)
        return empty, empty.copy()
    ix, iy = np.meshgrid(
        np.arange(nx, dtype=np.int64),
        np.arange(ny, dtype=np.int64),
        indexing="ij",
    )
    starts: list[NDArray[np.int64]] = []
    ends: list[NDArray[np.int64]] = []
    # Four positive directions enumerate each undirected edge exactly once.
    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        end_x = ix + int(dx)
        end_y = iy + int(dy)
        inside = (
            (end_x >= 0)
            & (end_y >= 0)
            & (end_x < nx)
            & (end_y < ny)
        )
        starts.append(np.column_stack([ix[inside], iy[inside]]))
        ends.append(np.column_stack([end_x[inside], end_y[inside]]))
    return np.vstack(starts), np.vstack(ends)


@dataclass(frozen=True)
class _TransportConfigurationGraph:
    """Cache transport-height free nodes and exact safe lattice transitions."""

    origin: tuple[float, float]
    cell_size: float
    grid_shape: tuple[int, int]
    free_cell_codes: frozenset[int]
    overlay_free_cell_codes: frozenset[int]
    safe_edge_codes: frozenset[tuple[int, int]]
    overlay_blocked_boxes_m: tuple[
        tuple[float, float, float, float, float, float], ...
    ]

    def cell_code(self, cell: tuple[int, int]) -> int:
        """Return a compact integer code for one in-bounds cell."""
        return int(cell[0]) * int(self.grid_shape[1]) + int(cell[1])

    def cell_index(self, point: Sequence[float]) -> tuple[int, int] | None:
        """Return the graph cell containing a world point, or ``None``."""
        point_array = np.asarray(point, dtype=float).reshape(-1)
        if point_array.size < 2 or np.any(~np.isfinite(point_array[:2])):
            return None
        relative = point_array[:2] - np.asarray(self.origin, dtype=float)
        if np.any(relative < 0.0):
            return None
        cell = np.floor(relative / float(self.cell_size)).astype(np.int64)
        if (
            cell[0] < 0
            or cell[1] < 0
            or cell[0] >= int(self.grid_shape[0])
            or cell[1] >= int(self.grid_shape[1])
        ):
            return None
        return int(cell[0]), int(cell[1])

    def cell_center(self, cell: tuple[int, int]) -> tuple[float, float]:
        """Return the world-space center of one graph cell."""
        return (
            float(self.origin[0])
            + (float(cell[0]) + 0.5) * float(self.cell_size),
            float(self.origin[1])
            + (float(cell[1]) + 0.5) * float(self.cell_size),
        )

    def is_free_cell(self, cell: tuple[int, int]) -> bool:
        """Return whether a cell center is a valid transport configuration."""
        ix, iy = int(cell[0]), int(cell[1])
        if ix < 0 or iy < 0:
            return False
        if ix >= int(self.grid_shape[0]) or iy >= int(self.grid_shape[1]):
            return False
        return self.cell_code((ix, iy)) in self.free_cell_codes

    def is_overlay_free_cell(self, cell: tuple[int, int]) -> bool:
        """Return whether the wrapped base map permits the graph-cell center."""
        ix, iy = int(cell[0]), int(cell[1])
        if ix < 0 or iy < 0:
            return False
        if ix >= int(self.grid_shape[0]) or iy >= int(self.grid_shape[1]):
            return False
        return self.cell_code((ix, iy)) in self.overlay_free_cell_codes

    def is_transition_free(
        self,
        first_cell: tuple[int, int],
        second_cell: tuple[int, int],
    ) -> bool:
        """Return whether a cached lattice edge passes the swept-envelope test."""
        if not self.is_free_cell(first_cell) or not self.is_free_cell(second_cell):
            return False
        edge = _canonical_edge_codes(
            self.cell_code(first_cell),
            self.cell_code(second_cell),
        )
        return edge in self.safe_edge_codes

    def connector_topology_free(
        self,
        owning_cell: tuple[int, int],
        target_cell: tuple[int, int],
    ) -> bool:
        """Return whether a target is a local overlay-free connector center."""
        dx = int(target_cell[0]) - int(owning_cell[0])
        dy = int(target_cell[1]) - int(owning_cell[1])
        if abs(dx) > 1 or abs(dy) > 1:
            return False
        return self.is_overlay_free_cell(target_cell)

    def iter_neighbors(
        self,
        cell: tuple[int, int],
    ) -> tuple[tuple[tuple[int, int], float], ...]:
        """Return at most eight cached-safe neighbors for sequential A* expansion."""
        neighbors: list[tuple[tuple[int, int], float]] = []
        ix, iy = int(cell[0]), int(cell[1])
        for dx, dy in _GRID_NEIGHBOR_OFFSETS:
            neighbor = (ix + int(dx), iy + int(dy))
            if not self.is_free_cell(neighbor):
                continue
            if dx != 0 and dy != 0:
                if not self.is_free_cell((ix + int(dx), iy)):
                    continue
                if not self.is_free_cell((ix, iy + int(dy))):
                    continue
            if not self.is_transition_free(cell, neighbor):
                continue
            multiplier = np.sqrt(2.0) if dx != 0 and dy != 0 else 1.0
            neighbors.append((neighbor, float(multiplier) * float(self.cell_size)))
        return tuple(neighbors)


@dataclass(frozen=True)
class MeasurementWorkspace:
    """Validate measurement endpoints and safe retract-translate-extend motion."""

    room_bounds: AxisAlignedRoomBounds
    assembly: DetectorAssemblyGeometry
    ground_z_m: float
    detector_transport_world_z_m: float
    collision_boxes_m: Sequence[Sequence[float]] | NDArray[np.float64] = ()
    base_map: object | None = None
    element_budget: int = _DEFAULT_ELEMENT_BUDGET
    motion_worker_count: int = 0
    motion_grid_cell_size_m: float = _DEFAULT_MOTION_GRID_CELL_SIZE_M
    _motion_waypoint_cache: dict[
        tuple[float, ...],
        tuple[tuple[float, float, float], ...] | None,
    ] = field(default_factory=dict, init=False, repr=False, compare=False)
    _transport_graph: _TransportConfigurationGraph | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate room, robot geometry, collision boxes, and map compatibility."""
        if not isinstance(self.room_bounds, AxisAlignedRoomBounds):
            raise TypeError("room_bounds must be an AxisAlignedRoomBounds instance.")
        if not isinstance(self.assembly, DetectorAssemblyGeometry):
            raise TypeError(
                "assembly must be a DetectorAssemblyGeometry instance."
            )
        ground_z = float(self.ground_z_m)
        transport_z = float(self.detector_transport_world_z_m)
        if not np.isfinite(ground_z):
            raise ValueError("ground_z_m must be finite.")
        if not np.isfinite(transport_z):
            raise ValueError("detector_transport_world_z_m must be finite.")
        room_lower = np.asarray(self.room_bounds.lower_xyz, dtype=float)
        room_upper = np.asarray(self.room_bounds.upper_xyz, dtype=float)
        base_top = ground_z + float(self.assembly.base_height_m)
        if ground_z < room_lower[2] - _GEOMETRY_TOLERANCE_M:
            raise ValueError("ground_z_m must lie inside the room.")
        if base_top > room_upper[2] + _GEOMETRY_TOLERANCE_M:
            raise ValueError("The base cylinder must fit inside the room height.")
        minimum_head_z = base_top + float(self.assembly.head_radius_m)
        maximum_head_z = room_upper[2] - float(self.assembly.head_radius_m)
        if (
            transport_z < minimum_head_z - _GEOMETRY_TOLERANCE_M
            or transport_z > maximum_head_z + _GEOMETRY_TOLERANCE_M
            or transport_z - float(self.assembly.head_radius_m)
            < room_lower[2] - _GEOMETRY_TOLERANCE_M
        ):
            raise ValueError(
                "detector_transport_world_z_m must keep the head above the base "
                "and inside the room."
            )
        if (
            float(self.assembly.head_radius_m)
            > float(self.assembly.base_radius_m) + _GEOMETRY_TOLERANCE_M
        ):
            raise ValueError(
                "head_radius_m must not exceed base_radius_m when a 2D base path "
                "is reused for detector transport."
            )
        boxes = _coerce_collision_boxes(self.collision_boxes_m)
        budget = int(self.element_budget)
        if budget <= 0:
            raise ValueError("element_budget must be positive.")
        worker_count = int(self.motion_worker_count)
        if worker_count < 0:
            raise ValueError("motion_worker_count must be non-negative.")
        motion_grid_cell_size = _validated_positive_float(
            self.motion_grid_cell_size_m,
            field_name="motion_grid_cell_size_m",
        )
        if self.base_map is not None:
            batch_checker = getattr(self.base_map, "is_free_batch", None)
            if not callable(batch_checker):
                raise TypeError("base_map must provide a batched is_free_batch method.")
            map_radius = getattr(self.base_map, "robot_radius_m", None)
            if map_radius is not None:
                resolved_map_radius = float(map_radius)
                if not np.isfinite(resolved_map_radius):
                    raise ValueError("base_map.robot_radius_m must be finite.")
                if (
                    resolved_map_radius + _GEOMETRY_TOLERANCE_M
                    < float(self.assembly.base_radius_m)
                ):
                    raise ValueError(
                        "base_map.robot_radius_m must cover assembly.base_radius_m."
                    )
        object.__setattr__(self, "ground_z_m", ground_z)
        object.__setattr__(self, "detector_transport_world_z_m", transport_z)
        object.__setattr__(self, "collision_boxes_m", boxes)
        object.__setattr__(self, "element_budget", budget)
        object.__setattr__(self, "motion_worker_count", worker_count)
        object.__setattr__(
            self,
            "motion_grid_cell_size_m",
            motion_grid_cell_size,
        )
        object.__setattr__(
            self,
            "_transport_graph",
            self._build_transport_configuration_graph(),
        )

    def __getattr__(self, name: str) -> Any:
        """Forward grid and path APIs to the wrapped two-dimensional base map."""
        base_map = object.__getattribute__(self, "base_map")
        if base_map is None:
            raise AttributeError(name)
        try:
            return getattr(base_map, name)
        except AttributeError as exc:
            raise AttributeError(name) from exc

    def _wrapped_base_grid_spec(
        self,
    ) -> tuple[tuple[float, float], float, tuple[int, int]] | None:
        """Return validated native geometry for a wrapped grid-like base map."""
        if self.base_map is None:
            return None
        origin = getattr(self.base_map, "origin", None)
        cell_size = getattr(self.base_map, "cell_size", None)
        grid_shape = getattr(self.base_map, "grid_shape", None)
        if (
            origin is None
            or cell_size is None
            or grid_shape is None
        ):
            return None
        origin_array = np.asarray(origin, dtype=float).reshape(-1)
        shape_array = np.asarray(grid_shape, dtype=np.int64).reshape(-1)
        resolved_cell_size = float(cell_size)
        if (
            origin_array.shape != (2,)
            or np.any(~np.isfinite(origin_array))
            or shape_array.shape != (2,)
            or np.any(shape_array < 0)
            or not np.isfinite(resolved_cell_size)
            or resolved_cell_size <= 0.0
        ):
            return None
        return (
            (float(origin_array[0]), float(origin_array[1])),
            resolved_cell_size,
            (int(shape_array[0]), int(shape_array[1])),
        )

    def _transport_grid_spec(
        self,
    ) -> tuple[tuple[float, float], float, tuple[int, int]] | None:
        """Return the configured fine room grid for exact transport routing."""
        if self.base_map is not None and self._wrapped_base_grid_spec() is None:
            return None
        lower = np.asarray(self.room_bounds.lower_xyz, dtype=float)
        upper = np.asarray(self.room_bounds.upper_xyz, dtype=float)
        cell_size = float(self.motion_grid_cell_size_m)
        extent_xy = np.maximum(upper[:2] - lower[:2], 0.0)
        grid_shape_array = np.ceil(extent_xy / cell_size).astype(np.int64)
        return (
            (float(lower[0]), float(lower[1])),
            cell_size,
            (int(grid_shape_array[0]), int(grid_shape_array[1])),
        )

    def _wrapped_base_blocked_boxes(
        self,
    ) -> NDArray[np.float64]:
        """Project wrapped base-map no-go cells to transport-height AABBs in batch."""
        grid_spec = self._wrapped_base_grid_spec()
        if grid_spec is None:
            return np.zeros((0, 6), dtype=float)
        origin, cell_size, grid_shape = grid_spec
        nx, ny = int(grid_shape[0]), int(grid_shape[1])
        cell_count = nx * ny
        if cell_count <= 0:
            return np.zeros((0, 6), dtype=float)
        ix = np.repeat(np.arange(nx, dtype=np.int64), ny)
        iy = np.tile(np.arange(ny, dtype=np.int64), nx)
        centers = np.column_stack(
            [
                float(origin[0]) + (ix.astype(float) + 0.5) * cell_size,
                float(origin[1]) + (iy.astype(float) + 0.5) * cell_size,
                np.full(
                    cell_count,
                    float(self.detector_transport_world_z_m),
                    dtype=float,
                ),
            ]
        )
        blocked = ~self._base_map_free_mask(centers)
        if not np.any(blocked):
            return np.zeros((0, 6), dtype=float)
        lower_x = float(origin[0]) + ix[blocked].astype(float) * cell_size
        lower_y = float(origin[1]) + iy[blocked].astype(float) * cell_size
        transport_z = float(self.detector_transport_world_z_m)
        return np.column_stack(
            [
                lower_x,
                lower_y,
                np.full(lower_x.size, transport_z, dtype=float),
                lower_x + cell_size,
                lower_y + cell_size,
                np.full(lower_x.size, transport_z, dtype=float),
            ]
        )

    def _base_overlay_segment_free_batch(
        self,
        starts_xyz: NDArray[np.float64],
        ends_xyz: NDArray[np.float64],
        blocked_boxes_m: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """Return exact segment freedom against projected wrapped-map no-go cells."""
        starts = np.asarray(starts_xyz, dtype=float).reshape(-1, 3)
        ends = np.asarray(ends_xyz, dtype=float).reshape(-1, 3)
        if starts.shape != ends.shape:
            raise ValueError("starts_xyz and ends_xyz must have the same shape.")
        if starts.shape[0] == 0:
            return np.zeros(0, dtype=bool)
        boxes = np.asarray(blocked_boxes_m, dtype=float).reshape(-1, 6)
        if boxes.shape[0] == 0:
            return np.ones(starts.shape[0], dtype=bool)
        collision = _segments_intersect_expanded_boxes_batch(
            starts,
            ends,
            boxes_m=boxes,
            expansion_xyz_m=(0.0, 0.0, 0.0),
            element_budget=int(self.element_budget),
        )
        return ~collision

    def _build_transport_configuration_graph(
        self,
    ) -> _TransportConfigurationGraph | None:
        """Build batched transport-height nodes and edges for collision-box routing."""
        boxes = np.asarray(self.collision_boxes_m, dtype=float).reshape(-1, 6)
        if boxes.shape[0] == 0:
            return None
        grid_spec = self._transport_grid_spec()
        if grid_spec is None:
            return None
        origin, cell_size, grid_shape = grid_spec
        overlay_blocked_boxes = self._wrapped_base_blocked_boxes()
        overlay_blocked_box_tuples = tuple(
            tuple(float(value) for value in box)
            for box in overlay_blocked_boxes
        )
        nx, ny = int(grid_shape[0]), int(grid_shape[1])
        cell_count = nx * ny
        if cell_count <= 0:
            return _TransportConfigurationGraph(
                origin=origin,
                cell_size=cell_size,
                grid_shape=grid_shape,
                free_cell_codes=frozenset(),
                overlay_free_cell_codes=frozenset(),
                safe_edge_codes=frozenset(),
                overlay_blocked_boxes_m=overlay_blocked_box_tuples,
            )
        ix = np.repeat(np.arange(nx, dtype=np.int64), ny)
        iy = np.tile(np.arange(ny, dtype=np.int64), nx)
        cells = np.column_stack([ix, iy])
        centers_xy = np.column_stack(
            [
                float(origin[0]) + (ix.astype(float) + 0.5) * cell_size,
                float(origin[1]) + (iy.astype(float) + 0.5) * cell_size,
            ]
        )
        transport_poses = np.column_stack(
            [
                centers_xy,
                np.full(
                    cell_count,
                    float(self.detector_transport_world_z_m),
                    dtype=float,
                ),
            ]
        )
        overlay_free = self._base_map_free_mask(transport_poses)
        node_free = self.is_free_batch(transport_poses)
        cell_codes = cells[:, 0] * ny + cells[:, 1]
        overlay_codes = frozenset(
            int(code) for code in cell_codes[np.asarray(overlay_free, dtype=bool)]
        )
        free_codes = frozenset(
            int(code) for code in cell_codes[np.asarray(node_free, dtype=bool)]
        )

        edge_starts, edge_ends = _batched_neighbor_cell_pairs(grid_shape)
        if edge_starts.shape[0] == 0:
            safe_edges: frozenset[tuple[int, int]] = frozenset()
        else:
            start_codes = edge_starts[:, 0] * ny + edge_starts[:, 1]
            end_codes = edge_ends[:, 0] * ny + edge_ends[:, 1]
            endpoint_free = node_free[start_codes] & node_free[end_codes]
            start_codes = start_codes[endpoint_free]
            end_codes = end_codes[endpoint_free]
            if start_codes.size == 0:
                safe_edges = frozenset()
            else:
                starts_xyz = transport_poses[start_codes]
                ends_xyz = transport_poses[end_codes]
                edge_free = self.is_horizontal_motion_free_batch(
                    starts_xyz,
                    ends_xyz,
                )
                edge_free &= self._base_overlay_segment_free_batch(
                    starts_xyz,
                    ends_xyz,
                    overlay_blocked_boxes,
                )
                safe_edges = frozenset(
                    _canonical_edge_codes(int(first), int(second))
                    for first, second in zip(
                        start_codes[np.asarray(edge_free, dtype=bool)],
                        end_codes[np.asarray(edge_free, dtype=bool)],
                        strict=True,
                    )
                )
        return _TransportConfigurationGraph(
            origin=origin,
            cell_size=cell_size,
            grid_shape=grid_shape,
            free_cell_codes=free_codes,
            overlay_free_cell_codes=overlay_codes,
            safe_edge_codes=safe_edges,
            overlay_blocked_boxes_m=overlay_blocked_box_tuples,
        )

    def _base_map_free_mask(
        self,
        poses_xyz: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """Return the wrapped base-map free-space mask."""
        if self.base_map is None:
            return np.ones(poses_xyz.shape[0], dtype=bool)
        checker = getattr(self.base_map, "is_free_batch")
        result = np.asarray(checker(poses_xyz), dtype=bool).reshape(-1)
        if result.size != poses_xyz.shape[0]:
            raise ValueError("base_map.is_free_batch returned the wrong length.")
        return result

    def _room_and_self_clearance_mask(
        self,
        poses_xyz: NDArray[np.float64],
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """Return room-containment and assembly self-clearance masks."""
        poses = np.asarray(poses_xyz, dtype=float).reshape(-1, 3)
        lower = np.asarray(self.room_bounds.lower_xyz, dtype=float)
        upper = np.asarray(self.room_bounds.upper_xyz, dtype=float)
        base_radius = float(self.assembly.base_radius_m)
        mast_radius = float(self.assembly.mast_radius_m)
        head_radius = float(self.assembly.head_radius_m)
        base_top = float(self.ground_z_m + self.assembly.base_height_m)
        base_xy_inside = (
            (poses[:, 0] - base_radius >= lower[0] - _GEOMETRY_TOLERANCE_M)
            & (poses[:, 0] + base_radius <= upper[0] + _GEOMETRY_TOLERANCE_M)
            & (poses[:, 1] - base_radius >= lower[1] - _GEOMETRY_TOLERANCE_M)
            & (poses[:, 1] + base_radius <= upper[1] + _GEOMETRY_TOLERANCE_M)
        )
        mast_xy_inside = (
            (poses[:, 0] - mast_radius >= lower[0] - _GEOMETRY_TOLERANCE_M)
            & (poses[:, 0] + mast_radius <= upper[0] + _GEOMETRY_TOLERANCE_M)
            & (poses[:, 1] - mast_radius >= lower[1] - _GEOMETRY_TOLERANCE_M)
            & (poses[:, 1] + mast_radius <= upper[1] + _GEOMETRY_TOLERANCE_M)
        )
        head_inside = np.all(
            (poses - head_radius >= lower[None, :] - _GEOMETRY_TOLERANCE_M)
            & (poses + head_radius <= upper[None, :] + _GEOMETRY_TOLERANCE_M),
            axis=1,
        )
        constant_z_inside = bool(
            self.ground_z_m >= lower[2] - _GEOMETRY_TOLERANCE_M
            and base_top <= upper[2] + _GEOMETRY_TOLERANCE_M
        )
        self_clear = (
            poses[:, 2] - head_radius
            >= base_top - _GEOMETRY_TOLERANCE_M
        )
        mast_z_inside = (
            poses[:, 2] >= base_top - _GEOMETRY_TOLERANCE_M
        ) & (poses[:, 2] <= upper[2] + _GEOMETRY_TOLERANCE_M)
        room_clear = (
            base_xy_inside
            & mast_xy_inside
            & head_inside
            & mast_z_inside
            & constant_z_inside
        )
        return room_clear, self_clear

    def endpoint_validity_masks(
        self,
        poses_xyz: Sequence[Sequence[float]] | NDArray[np.float64],
    ) -> dict[str, NDArray[np.bool_]]:
        """Return component-wise endpoint feasibility masks for many poses."""
        poses = _coerce_points_batch(poses_xyz, field_name="poses_xyz")
        count = int(poses.shape[0])
        room_clear, self_clear = self._room_and_self_clearance_mask(poses)
        base_map_free = self._base_map_free_mask(poses)
        base_lower = np.full(count, float(self.ground_z_m), dtype=float)
        base_upper = np.full(
            count,
            float(self.ground_z_m + self.assembly.base_height_m),
            dtype=float,
        )
        mast_lower = base_upper.copy()
        mast_upper = np.maximum(poses[:, 2], mast_lower)
        base_collision = _vertical_cylinder_collision_mask_batch(
            poses[:, :2],
            z_lower_m=base_lower,
            z_upper_m=base_upper,
            radius_m=float(self.assembly.base_radius_m),
            boxes_m=np.asarray(self.collision_boxes_m, dtype=float),
            element_budget=int(self.element_budget),
        )
        mast_collision = _vertical_cylinder_collision_mask_batch(
            poses[:, :2],
            z_lower_m=mast_lower,
            z_upper_m=mast_upper,
            radius_m=float(self.assembly.mast_radius_m),
            boxes_m=np.asarray(self.collision_boxes_m, dtype=float),
            element_budget=int(self.element_budget),
        )
        head_collision = _sphere_collision_mask_batch(
            poses,
            radius_m=float(self.assembly.head_radius_m),
            boxes_m=np.asarray(self.collision_boxes_m, dtype=float),
            element_budget=int(self.element_budget),
        )
        valid = (
            room_clear
            & self_clear
            & base_map_free
            & ~base_collision
            & ~mast_collision
            & ~head_collision
        )
        return {
            "room_clear": room_clear,
            "self_clear": self_clear,
            "base_map_free": base_map_free,
            "base_collision_free": ~base_collision,
            "mast_collision_free": ~mast_collision,
            "head_collision_free": ~head_collision,
            "valid": valid,
        }

    def is_free_batch(
        self,
        points: Sequence[Sequence[float]] | NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """Return complete three-dimensional endpoint feasibility flags."""
        return self.endpoint_validity_masks(points)["valid"]

    def is_free(self, point: Sequence[float]) -> bool:
        """Return whether one detector measurement pose is feasible."""
        pose = np.asarray(point, dtype=float).reshape(-1)
        if pose.shape != (3,):
            raise ValueError("point must contain exactly three coordinates.")
        return bool(self.is_free_batch(pose.reshape(1, 3))[0])

    def vertical_head_sweep_validity_masks(
        self,
        start_poses_xyz: Sequence[Sequence[float]] | NDArray[np.float64],
        end_poses_xyz: Sequence[Sequence[float]] | NDArray[np.float64],
    ) -> dict[str, NDArray[np.bool_]]:
        """Return exact vertical detector-head sweep feasibility masks."""
        starts = _coerce_points_batch(
            start_poses_xyz,
            field_name="start_poses_xyz",
        )
        ends = _coerce_points_batch(end_poses_xyz, field_name="end_poses_xyz")
        if starts.shape != ends.shape:
            raise ValueError(
                "start_poses_xyz and end_poses_xyz must have the same shape."
            )
        vertical = (
            np.linalg.norm(starts[:, :2] - ends[:, :2], axis=1)
            <= _GEOMETRY_TOLERANCE_M
        )
        endpoint_free = self.is_free_batch(starts) & self.is_free_batch(ends)
        lower_z = np.minimum(starts[:, 2], ends[:, 2])
        upper_z = np.maximum(starts[:, 2], ends[:, 2])
        head_collision = _vertical_capsule_collision_mask_batch(
            starts[:, :2],
            z_lower_m=lower_z,
            z_upper_m=upper_z,
            radius_m=float(self.assembly.head_radius_m),
            boxes_m=np.asarray(self.collision_boxes_m, dtype=float),
            element_budget=int(self.element_budget),
        )
        valid = vertical & endpoint_free & ~head_collision
        return {
            "vertical": vertical,
            "endpoints_free": endpoint_free,
            "head_sweep_collision_free": ~head_collision,
            "valid": valid,
        }

    def is_vertical_head_sweep_free_batch(
        self,
        start_poses_xyz: Sequence[Sequence[float]] | NDArray[np.float64],
        end_poses_xyz: Sequence[Sequence[float]] | NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """Return feasibility flags for batched vertical head motions."""
        return self.vertical_head_sweep_validity_masks(
            start_poses_xyz,
            end_poses_xyz,
        )["valid"]

    def is_vertical_head_sweep_free(
        self,
        start_pose_xyz: Sequence[float],
        end_pose_xyz: Sequence[float],
    ) -> bool:
        """Return whether one vertical detector-head motion is feasible."""
        start = np.asarray(start_pose_xyz, dtype=float).reshape(-1)
        end = np.asarray(end_pose_xyz, dtype=float).reshape(-1)
        if start.shape != (3,) or end.shape != (3,):
            raise ValueError("start_pose_xyz and end_pose_xyz must be 3-vectors.")
        return bool(
            self.is_vertical_head_sweep_free_batch(
                start.reshape(1, 3),
                end.reshape(1, 3),
            )[0]
        )

    def horizontal_motion_validity_masks(
        self,
        start_poses_xyz: Sequence[Sequence[float]] | NDArray[np.float64],
        end_poses_xyz: Sequence[Sequence[float]] | NDArray[np.float64],
    ) -> dict[str, NDArray[np.bool_]]:
        """Return conservative swept-envelope checks for horizontal segments."""
        starts = _coerce_points_batch(
            start_poses_xyz,
            field_name="start_poses_xyz",
        )
        ends = _coerce_points_batch(end_poses_xyz, field_name="end_poses_xyz")
        if starts.shape != ends.shape:
            raise ValueError(
                "start_poses_xyz and end_poses_xyz must have the same shape."
            )
        transport_z = float(self.detector_transport_world_z_m)
        horizontal = (
            np.abs(starts[:, 2] - ends[:, 2]) <= _GEOMETRY_TOLERANCE_M
        ) & (
            np.abs(starts[:, 2] - transport_z) <= _GEOMETRY_TOLERANCE_M
        ) & (
            np.abs(ends[:, 2] - transport_z) <= _GEOMETRY_TOLERANCE_M
        )
        endpoint_free = self.is_free_batch(starts) & self.is_free_batch(ends)
        boxes = np.asarray(self.collision_boxes_m, dtype=float)
        base_half_height = 0.5 * float(self.assembly.base_height_m)
        base_center_z = float(self.ground_z_m) + base_half_height
        base_starts = starts.copy()
        base_ends = ends.copy()
        base_starts[:, 2] = base_center_z
        base_ends[:, 2] = base_center_z
        base_collision = _segments_intersect_expanded_boxes_batch(
            base_starts,
            base_ends,
            boxes_m=boxes,
            expansion_xyz_m=(
                float(self.assembly.base_radius_m),
                float(self.assembly.base_radius_m),
                base_half_height,
            ),
            element_budget=int(self.element_budget),
        )
        base_top = float(self.ground_z_m + self.assembly.base_height_m)
        mast_half_height = 0.5 * max(transport_z - base_top, 0.0)
        mast_center_z = base_top + mast_half_height
        mast_starts = starts.copy()
        mast_ends = ends.copy()
        mast_starts[:, 2] = mast_center_z
        mast_ends[:, 2] = mast_center_z
        mast_collision = _segments_intersect_expanded_boxes_batch(
            mast_starts,
            mast_ends,
            boxes_m=boxes,
            expansion_xyz_m=(
                float(self.assembly.mast_radius_m),
                float(self.assembly.mast_radius_m),
                mast_half_height,
            ),
            element_budget=int(self.element_budget),
        )
        head_radius = float(self.assembly.head_radius_m)
        head_collision = _segments_intersect_expanded_boxes_batch(
            starts,
            ends,
            boxes_m=boxes,
            expansion_xyz_m=(head_radius, head_radius, head_radius),
            element_budget=int(self.element_budget),
        )
        valid = (
            horizontal
            & endpoint_free
            & ~base_collision
            & ~mast_collision
            & ~head_collision
        )
        return {
            "horizontal_at_transport_height": horizontal,
            "endpoints_free": endpoint_free,
            "base_sweep_collision_free": ~base_collision,
            "mast_sweep_collision_free": ~mast_collision,
            "head_sweep_collision_free": ~head_collision,
            "valid": valid,
        }

    def is_horizontal_motion_free_batch(
        self,
        start_poses_xyz: Sequence[Sequence[float]] | NDArray[np.float64],
        end_poses_xyz: Sequence[Sequence[float]] | NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """Return swept-envelope feasibility flags for horizontal segments."""
        return self.horizontal_motion_validity_masks(
            start_poses_xyz,
            end_poses_xyz,
        )["valid"]

    def _transport_connector_costs(
        self,
        point_xyz: NDArray[np.float64],
        graph: _TransportConfigurationGraph,
    ) -> dict[tuple[int, int], float]:
        """Return exact-safe local graph connectors and their Euclidean costs."""
        owning_cell = graph.cell_index(point_xyz)
        if owning_cell is None:
            return {}
        offsets = np.asarray(((0, 0),) + _GRID_NEIGHBOR_OFFSETS, dtype=np.int64)
        candidate_array = offsets + np.asarray(owning_cell, dtype=np.int64)[None, :]
        nx, ny = int(graph.grid_shape[0]), int(graph.grid_shape[1])
        inside = (
            (candidate_array[:, 0] >= 0)
            & (candidate_array[:, 1] >= 0)
            & (candidate_array[:, 0] < nx)
            & (candidate_array[:, 1] < ny)
        )
        candidate_array = candidate_array[inside]
        candidate_cells = tuple(
            (int(cell[0]), int(cell[1]))
            for cell in candidate_array
            if graph.is_free_cell((int(cell[0]), int(cell[1])))
            and graph.connector_topology_free(
                owning_cell,
                (int(cell[0]), int(cell[1])),
            )
        )
        if not candidate_cells:
            return {}
        centers = np.asarray(
            [
                (
                    *graph.cell_center(cell),
                    float(self.detector_transport_world_z_m),
                )
                for cell in candidate_cells
            ],
            dtype=float,
        )
        starts = np.repeat(
            np.asarray(point_xyz, dtype=float).reshape(1, 3),
            centers.shape[0],
            axis=0,
        )
        connector_free = self.is_horizontal_motion_free_batch(starts, centers)
        connector_free &= self._base_overlay_segment_free_batch(
            starts,
            centers,
            np.asarray(graph.overlay_blocked_boxes_m, dtype=float),
        )
        distances = np.linalg.norm(centers - starts, axis=1)
        return {
            cell: float(distance)
            for cell, distance, is_free in zip(
                candidate_cells,
                distances,
                connector_free,
                strict=True,
            )
            if bool(is_free)
        }

    @staticmethod
    def _shortest_transport_cell_path(
        graph: _TransportConfigurationGraph,
        start_costs: dict[tuple[int, int], float],
        goal_costs: dict[tuple[int, int], float],
        goal_xyz: NDArray[np.float64],
    ) -> tuple[tuple[int, int], ...] | None:
        """Run multi-source A* over cached, batch-validated transport edges."""
        if not start_costs or not goal_costs:
            return None
        goal_xy = np.asarray(goal_xyz, dtype=float).reshape(3)[:2]
        frontier: list[tuple[float, float, int, int]] = []
        cost_so_far: dict[tuple[int, int], float] = {}
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        for cell, connector_cost in start_costs.items():
            center = np.asarray(graph.cell_center(cell), dtype=float)
            heuristic = float(np.linalg.norm(center - goal_xy))
            resolved_cost = float(connector_cost)
            cost_so_far[cell] = resolved_cost
            heapq.heappush(
                frontier,
                (
                    resolved_cost + heuristic,
                    resolved_cost,
                    int(cell[0]),
                    int(cell[1]),
                ),
            )
        best_goal_cell: tuple[int, int] | None = None
        best_goal_cost = float("inf")
        # A* expansion is sequential, but every expensive geometry edge test was
        # already evaluated in one batch when the immutable graph was constructed.
        while frontier:
            priority, current_cost, ix, iy = heapq.heappop(frontier)
            if priority > best_goal_cost + _GEOMETRY_TOLERANCE_M:
                break
            current = (int(ix), int(iy))
            if current_cost > cost_so_far.get(current, float("inf")):
                continue
            if current in goal_costs:
                complete_cost = current_cost + float(goal_costs[current])
                if complete_cost < best_goal_cost:
                    best_goal_cost = complete_cost
                    best_goal_cell = current
            for neighbor, step_cost in graph.iter_neighbors(current):
                new_cost = current_cost + float(step_cost)
                if new_cost >= cost_so_far.get(neighbor, float("inf")):
                    continue
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                center = np.asarray(graph.cell_center(neighbor), dtype=float)
                heuristic = float(np.linalg.norm(center - goal_xy))
                heapq.heappush(
                    frontier,
                    (
                        new_cost + heuristic,
                        new_cost,
                        int(neighbor[0]),
                        int(neighbor[1]),
                    ),
                )
        if best_goal_cell is None:
            return None
        path = [best_goal_cell]
        while path[-1] in came_from:
            path.append(came_from[path[-1]])
        path.reverse()
        return tuple(path)

    def _transport_graph_path(
        self,
        start_transport_xyz: NDArray[np.float64],
        goal_transport_xyz: NDArray[np.float64],
        graph: _TransportConfigurationGraph,
    ) -> NDArray[np.float64] | None:
        """Return a safe continuous-point path through the cached transport graph."""
        start_costs = self._transport_connector_costs(start_transport_xyz, graph)
        goal_costs = self._transport_connector_costs(goal_transport_xyz, graph)
        cell_path = self._shortest_transport_cell_path(
            graph,
            start_costs,
            goal_costs,
            goal_transport_xyz,
        )
        if cell_path is None:
            return None
        centers = np.asarray(
            [
                (
                    *graph.cell_center(cell),
                    float(self.detector_transport_world_z_m),
                )
                for cell in cell_path
            ],
            dtype=float,
        ).reshape(-1, 3)
        return _dedupe_consecutive_points(
            np.vstack([start_transport_xyz, centers, goal_transport_xyz])
        )

    def _direct_transport_segment_free(
        self,
        start_transport_xyz: NDArray[np.float64],
        goal_transport_xyz: NDArray[np.float64],
        graph: _TransportConfigurationGraph,
    ) -> bool:
        """Return whether a direct segment is exact-safe and cannot cross map no-go."""
        if not bool(
            self.is_horizontal_motion_free_batch(
                np.asarray(start_transport_xyz, dtype=float).reshape(1, 3),
                np.asarray(goal_transport_xyz, dtype=float).reshape(1, 3),
            )[0]
        ):
            return False
        return bool(
            self._base_overlay_segment_free_batch(
                np.asarray(start_transport_xyz, dtype=float).reshape(1, 3),
                np.asarray(goal_transport_xyz, dtype=float).reshape(1, 3),
                np.asarray(graph.overlay_blocked_boxes_m, dtype=float),
            )[0]
        )

    def _horizontal_base_path(
        self,
        start_transport_xyz: NDArray[np.float64],
        goal_transport_xyz: NDArray[np.float64],
    ) -> NDArray[np.float64] | None:
        """Return a base-map path with every waypoint at transport height."""
        transport_graph = self._transport_graph
        if transport_graph is not None:
            if self._direct_transport_segment_free(
                start_transport_xyz,
                goal_transport_xyz,
                transport_graph,
            ):
                path = np.vstack([start_transport_xyz, goal_transport_xyz])
            else:
                path = self._transport_graph_path(
                    start_transport_xyz,
                    goal_transport_xyz,
                    transport_graph,
                )
                if path is None:
                    return None
        elif self.base_map is None:
            path = np.vstack([start_transport_xyz, goal_transport_xyz])
        else:
            path_function = getattr(self.base_map, "shortest_path_points", None)
            if callable(path_function):
                path = path_function(
                    start_transport_xyz,
                    goal_transport_xyz,
                    allow_diagonal=True,
                )
            else:
                from planning.traversability import shortest_grid_path_points

                path = shortest_grid_path_points(
                    self.base_map,
                    start_transport_xyz,
                    goal_transport_xyz,
                    allow_diagonal=True,
                )
            if path is None:
                return None
        path_array = np.asarray(path, dtype=float)
        if path_array.ndim != 2 or path_array.shape[1] != 3:
            raise ValueError("base_map path must be shaped (N, 3).")
        if path_array.shape[0] == 0 or np.any(~np.isfinite(path_array)):
            return None
        path_array = path_array.copy()
        path_array[:, 2] = float(self.detector_transport_world_z_m)
        path_array = np.vstack(
            [start_transport_xyz, path_array, goal_transport_xyz]
        )
        return _dedupe_consecutive_points(path_array)

    @staticmethod
    def _motion_cache_key(
        start_xyz: NDArray[np.float64],
        goal_xyz: NDArray[np.float64],
    ) -> tuple[float, ...]:
        """Return an exact endpoint key for one directed motion request."""
        return tuple(
            float(value)
            for value in np.concatenate([start_xyz, goal_xyz]).reshape(6)
        )

    def _cached_motion_waypoints(
        self,
        cache_key: tuple[float, ...],
    ) -> tuple[bool, NDArray[np.float64] | None]:
        """Return an isolated copy of a cached route when one is present."""
        cached = self._motion_waypoint_cache.get(
            cache_key,
            _MOTION_CACHE_MISS,
        )
        if cached is _MOTION_CACHE_MISS:
            return False, None
        if cached is None:
            return True, None
        return True, np.asarray(cached, dtype=float).reshape(-1, 3).copy()

    def _store_motion_waypoints(
        self,
        cache_key: tuple[float, ...],
        waypoints: NDArray[np.float64] | None,
    ) -> None:
        """Store an immutable route snapshot in the bounded instance cache."""
        cached = None
        if waypoints is not None:
            points = np.asarray(waypoints, dtype=float).reshape(-1, 3)
            cached = tuple(
                (float(point[0]), float(point[1]), float(point[2]))
                for point in points
            )
        if len(self._motion_waypoint_cache) + 2 > _MOTION_CACHE_MAX_ENTRIES:
            self._motion_waypoint_cache.clear()
        reverse_key = cache_key[3:] + cache_key[:3]
        reverse_cached = None if cached is None else tuple(reversed(cached))
        self._motion_waypoint_cache[cache_key] = cached
        self._motion_waypoint_cache[reverse_key] = reverse_cached

    def _compute_motion_waypoints(
        self,
        start: NDArray[np.float64],
        goal: NDArray[np.float64],
    ) -> NDArray[np.float64] | None:
        """Compute one uncached safe retract-translate-extend route."""
        if float(np.linalg.norm(start - goal)) <= _GEOMETRY_TOLERANCE_M:
            return start.reshape(1, 3) if self.is_free(start) else None
        if (
            float(np.linalg.norm(start[:2] - goal[:2]))
            <= _GEOMETRY_TOLERANCE_M
        ):
            if not self.is_vertical_head_sweep_free(start, goal):
                return None
            return np.vstack([start, goal]).astype(float)
        transport_z = float(self.detector_transport_world_z_m)
        start_transport = np.array([start[0], start[1], transport_z], dtype=float)
        goal_transport = np.array([goal[0], goal[1], transport_z], dtype=float)
        vertical_starts = np.vstack([start, goal_transport])
        vertical_ends = np.vstack([start_transport, goal])
        if not np.all(
            self.is_vertical_head_sweep_free_batch(
                vertical_starts,
                vertical_ends,
            )
        ):
            return None
        horizontal_path = self._horizontal_base_path(
            start_transport,
            goal_transport,
        )
        if horizontal_path is None:
            return None
        if not np.all(self.is_free_batch(horizontal_path)):
            return None
        if horizontal_path.shape[0] >= 2 and not np.all(
            self.is_horizontal_motion_free_batch(
                horizontal_path[:-1],
                horizontal_path[1:],
            )
        ):
            return None
        waypoints = _dedupe_consecutive_points(
            np.vstack([start, horizontal_path, goal])
        )
        if not np.all(self.is_free_batch(waypoints)):
            return None
        return waypoints

    def motion_waypoints(
        self,
        start_xyz: Sequence[float],
        goal_xyz: Sequence[float],
    ) -> NDArray[np.float64] | None:
        """Return a cached safe detector-motion path, or ``None``."""
        start = np.asarray(start_xyz, dtype=float).reshape(-1)
        goal = np.asarray(goal_xyz, dtype=float).reshape(-1)
        if start.shape != (3,) or goal.shape != (3,):
            raise ValueError("start_xyz and goal_xyz must be 3-vectors.")
        if np.any(~np.isfinite(start)) or np.any(~np.isfinite(goal)):
            raise ValueError("start_xyz and goal_xyz must contain finite values.")
        cache_key = self._motion_cache_key(start, goal)
        cache_hit, cached = self._cached_motion_waypoints(cache_key)
        if cache_hit:
            return cached
        waypoints = self._compute_motion_waypoints(start, goal)
        self._store_motion_waypoints(cache_key, waypoints)
        if waypoints is None:
            return None
        return np.asarray(waypoints, dtype=float).copy()

    def motion_path_lengths_batch(
        self,
        start_xyz: Sequence[float],
        goals_xyz: Sequence[Sequence[float]] | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return exact safe route lengths using parallel path evaluation."""
        start = np.asarray(start_xyz, dtype=float).reshape(-1)
        if start.shape != (3,) or np.any(~np.isfinite(start)):
            raise ValueError("start_xyz must be a finite three-vector.")
        goals = _coerce_points_batch(goals_xyz, field_name="goals_xyz")
        lengths = np.full(goals.shape[0], float("inf"), dtype=float)
        if goals.shape[0] == 0 or not self.is_free(start):
            return lengths
        endpoint_mask = self.is_free_batch(goals)
        indices = np.flatnonzero(endpoint_mask)
        if indices.size == 0:
            return lengths
        unique_goals, inverse = np.unique(
            goals[indices],
            axis=0,
            return_inverse=True,
        )
        worker_count = int(self.motion_worker_count)
        if worker_count <= 0:
            worker_count = min(max(os.cpu_count() or 1, 1), 32)
        worker_count = min(worker_count, int(unique_goals.shape[0]))

        def _route_length(goal_index: int) -> float:
            """Return the safe polyline length for one unique goal."""
            route = self.motion_waypoints(start, unique_goals[int(goal_index)])
            if route is None:
                return float("inf")
            points = np.asarray(route, dtype=float).reshape(-1, 3)
            if points.shape[0] <= 1:
                return 0.0
            return float(
                np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
            )

        goal_indices = range(unique_goals.shape[0])
        if worker_count <= 1:
            unique_lengths = tuple(_route_length(index) for index in goal_indices)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                unique_lengths = tuple(executor.map(_route_length, goal_indices))
        length_values = np.asarray(unique_lengths, dtype=float)
        lengths[indices] = length_values[inverse]
        return lengths

    def is_motion_reachable_batch(
        self,
        start_xyz: Sequence[float],
        goals_xyz: Sequence[Sequence[float]] | NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """Return route-feasibility flags derived from batched path lengths."""
        return np.isfinite(self.motion_path_lengths_batch(start_xyz, goals_xyz))
