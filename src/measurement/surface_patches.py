"""Area-aware surface patches for distribution reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid


_GEOMETRY_TOLERANCE_M = 1.0e-9
_ADJACENCY_GROUP_BATCH_SIZE = 2_048
_BOX_POINT_BATCH_SIZE = 4_096


@dataclass(frozen=True)
class SurfacePatchDictionary:
    """Describe finite surface patches and their graph-TV adjacency."""

    centers_xyz: NDArray[np.float64]
    areas_m2: NDArray[np.float64]
    kinds: tuple[str, ...]
    face_ids: tuple[str, ...]
    normals_xyz: NDArray[np.float64]
    local_uv_m: NDArray[np.float64]
    adjacency_edges: NDArray[np.int64]
    shared_edge_lengths_m: NDArray[np.float64]
    obstacle_geometry_source: str = "none"
    obstacle_surfaces_available: bool = True
    obstacle_component_count: int = 0
    obstacle_geometry_warning: str | None = None

    def __post_init__(self) -> None:
        """Validate array shapes and physical patch measures."""
        centers = np.asarray(self.centers_xyz, dtype=float)
        areas = np.asarray(self.areas_m2, dtype=float).reshape(-1)
        normals = np.asarray(self.normals_xyz, dtype=float)
        local_uv = np.asarray(self.local_uv_m, dtype=float)
        edges = np.asarray(self.adjacency_edges, dtype=np.int64)
        edge_lengths = np.asarray(self.shared_edge_lengths_m, dtype=float).reshape(-1)
        count = int(areas.size)
        if centers.shape != (count, 3):
            raise ValueError("centers_xyz must be shaped (C, 3).")
        if normals.shape != (count, 3):
            raise ValueError("normals_xyz must be shaped (C, 3).")
        if local_uv.shape != (count, 2):
            raise ValueError("local_uv_m must be shaped (C, 2).")
        if len(self.kinds) != count or len(self.face_ids) != count:
            raise ValueError("kinds and face_ids must have one entry per patch.")
        if np.any(~np.isfinite(areas)) or np.any(areas <= 0.0):
            raise ValueError("areas_m2 must contain finite positive values.")
        if edges.size == 0:
            edges = np.zeros((0, 2), dtype=np.int64)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("adjacency_edges must be shaped (E, 2).")
        if edge_lengths.size != edges.shape[0]:
            raise ValueError("shared_edge_lengths_m must have E entries.")
        if edges.size and (np.min(edges) < 0 or np.max(edges) >= count):
            raise ValueError("adjacency_edges contains an invalid patch index.")
        if np.any(~np.isfinite(edge_lengths)) or np.any(edge_lengths <= 0.0):
            raise ValueError("shared edge lengths must be finite and positive.")
        component_count = int(self.obstacle_component_count)
        if component_count < 0:
            raise ValueError("obstacle_component_count must be non-negative.")
        object.__setattr__(self, "centers_xyz", centers)
        object.__setattr__(self, "areas_m2", areas)
        object.__setattr__(self, "normals_xyz", normals)
        object.__setattr__(self, "local_uv_m", local_uv)
        object.__setattr__(self, "adjacency_edges", edges)
        object.__setattr__(self, "shared_edge_lengths_m", edge_lengths)
        object.__setattr__(
            self,
            "obstacle_geometry_source",
            str(self.obstacle_geometry_source),
        )
        object.__setattr__(
            self,
            "obstacle_surfaces_available",
            bool(self.obstacle_surfaces_available),
        )
        object.__setattr__(self, "obstacle_component_count", component_count)
        warning = self.obstacle_geometry_warning
        object.__setattr__(
            self,
            "obstacle_geometry_warning",
            None if warning is None else str(warning),
        )

    @property
    def patch_count(self) -> int:
        """Return the number of finite surface patches."""
        return int(self.areas_m2.size)

    @property
    def geometry_metadata(self) -> dict[str, object]:
        """Return explicit provenance and completeness of obstacle surfaces."""
        return {
            "obstacle_geometry_source": self.obstacle_geometry_source,
            "obstacle_surfaces_available": self.obstacle_surfaces_available,
            "obstacle_component_count": self.obstacle_component_count,
            "obstacle_geometry_warning": self.obstacle_geometry_warning,
        }


@dataclass(frozen=True)
class _PatchBlock:
    """Store one construction block before global index offsets are applied."""

    centers: NDArray[np.float64]
    areas: NDArray[np.float64]
    kinds: NDArray[np.str_]
    face_ids: NDArray[np.str_]
    normals: NDArray[np.float64]
    local_uv: NDArray[np.float64]
    edges: NDArray[np.int64]
    edge_lengths: NDArray[np.float64]
    boundary_segments: NDArray[np.float64]


def _axis_cell_count(length_m: float, target_spacing_m: float) -> int:
    """Return the interval count without allocating its coordinate array."""
    length = float(length_m)
    spacing = float(target_spacing_m)
    if not np.isfinite(length) or not np.isfinite(spacing):
        raise ValueError("Surface lengths and spacing must be finite.")
    if length <= 0.0 or spacing <= 0.0:
        raise ValueError("Surface lengths and spacing must be positive.")
    return max(1, int(np.ceil(length / spacing)))


def _axis_edges(length_m: float, target_spacing_m: float) -> NDArray[np.float64]:
    """Return exact interval edges with cells no wider than target spacing."""
    length = float(length_m)
    cells = _axis_cell_count(length, target_spacing_m)
    return np.linspace(0.0, length, num=cells + 1, dtype=float)


def _spacing_vector(spacing: float | Sequence[float]) -> NDArray[np.float64]:
    """Return finite positive x, y, and z patch spacings."""
    spacing_array = np.asarray(spacing, dtype=float).reshape(-1)
    if spacing_array.size == 1:
        spacing_array = np.repeat(spacing_array, 3)
    if (
        spacing_array.shape != (3,)
        or np.any(~np.isfinite(spacing_array))
        or np.any(spacing_array <= 0.0)
    ):
        raise ValueError("spacing must be a finite positive scalar or 3-vector.")
    return spacing_array


def _clipped_transport_boxes(
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
) -> NDArray[np.float64]:
    """Return transport components with positive volume inside the room."""
    if obstacle_grid is None or not obstacle_grid.transport_boxes_m:
        return np.zeros((0, 6), dtype=float)
    boxes = np.asarray(obstacle_grid.transport_boxes_m, dtype=float).reshape(-1, 6)
    room_upper = np.asarray(
        [float(env.size_x), float(env.size_y), float(env.size_z)],
        dtype=float,
    )
    lower = np.maximum(boxes[:, :3], 0.0)
    upper = np.minimum(boxes[:, 3:], room_upper[None, :])
    keep = np.all(upper - lower > _GEOMETRY_TOLERANCE_M, axis=1)
    return np.column_stack([lower[keep], upper[keep]])


def _floor_contact_boxes(boxes: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return clipped components whose solid volume starts on the room floor."""
    box_array = np.asarray(boxes, dtype=float).reshape(-1, 6)
    if box_array.shape[0] == 0:
        return box_array
    on_floor = box_array[:, 2] <= _GEOMETRY_TOLERANCE_M
    return box_array[on_floor]


def _floor_obstacle_boundaries(
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return exact floor-contact component x and y boundaries."""
    boxes = _floor_contact_boxes(_clipped_transport_boxes(env, obstacle_grid))
    if boxes.shape[0] == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    return boxes[:, (0, 3)].reshape(-1), boxes[:, (1, 4)].reshape(-1)


def _floor_edges_with_obstacle_boundaries(
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    x_edges_m: NDArray[np.float64],
    y_edges_m: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split floor patches at exact floor-contact component boundaries."""
    x_boundaries, y_boundaries = _floor_obstacle_boundaries(env, obstacle_grid)
    if x_boundaries.size == 0:
        return x_edges_m, y_edges_m
    return (
        np.unique(np.concatenate([x_edges_m, x_boundaries])),
        np.unique(np.concatenate([y_edges_m, y_boundaries])),
    )


def _points_inside_box_union(
    points_xyz: NDArray[np.float64],
    boxes_m: NDArray[np.float64],
    *,
    excluded_box_indices: NDArray[np.int64] | None = None,
) -> NDArray[np.bool_]:
    """Test points against a box union in bounded vectorized batches."""
    points = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    boxes = np.asarray(boxes_m, dtype=float).reshape(-1, 6)
    if points.shape[0] == 0 or boxes.shape[0] == 0:
        return np.zeros(points.shape[0], dtype=bool)
    excluded = None
    if excluded_box_indices is not None:
        excluded = np.asarray(excluded_box_indices, dtype=np.int64).reshape(-1)
        if excluded.size != points.shape[0]:
            raise ValueError("excluded_box_indices must align with points_xyz.")
    result = np.zeros(points.shape[0], dtype=bool)
    for start in range(0, points.shape[0], _BOX_POINT_BATCH_SIZE):
        stop = min(start + _BOX_POINT_BATCH_SIZE, points.shape[0])
        batch = points[start:stop]
        inside = np.all(
            (batch[:, None, :] >= boxes[None, :, :3] - _GEOMETRY_TOLERANCE_M)
            & (batch[:, None, :] <= boxes[None, :, 3:] + _GEOMETRY_TOLERANCE_M),
            axis=2,
        )
        if excluded is not None:
            rows = np.arange(stop - start, dtype=np.int64)
            valid_excluded = (
                (excluded[start:stop] >= 0)
                & (excluded[start:stop] < boxes.shape[0])
            )
            inside[
                rows[valid_excluded],
                excluded[start:stop][valid_excluded],
            ] = False
        result[start:stop] = np.any(inside, axis=1)
    return result


def _floor_footprint_mask(
    points_xyz: NDArray[np.float64],
    boxes_m: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Return patches covered by the exact union of floor-contact footprints."""
    floor_boxes = _floor_contact_boxes(boxes_m)
    if floor_boxes.shape[0] == 0:
        return np.zeros(np.asarray(points_xyz).shape[0], dtype=bool)
    probe_points = np.asarray(points_xyz, dtype=float).reshape(-1, 3).copy()
    probe_points[:, 2] = _GEOMETRY_TOLERANCE_M
    return _points_inside_box_union(probe_points, floor_boxes)


def _face_block(
    *,
    origin_xyz: Sequence[float],
    u_axis_xyz: Sequence[float],
    v_axis_xyz: Sequence[float],
    u_edges_m: NDArray[np.float64],
    v_edges_m: NDArray[np.float64],
    normal_xyz: Sequence[float],
    kind: str,
    face_id: str,
) -> _PatchBlock:
    """Build a rectangular face as finite patches with grid adjacency."""
    origin = np.asarray(origin_xyz, dtype=float).reshape(3)
    u_axis = np.asarray(u_axis_xyz, dtype=float).reshape(3)
    v_axis = np.asarray(v_axis_xyz, dtype=float).reshape(3)
    normal = np.asarray(normal_xyz, dtype=float).reshape(3)
    u_edges = np.asarray(u_edges_m, dtype=float).reshape(-1)
    v_edges = np.asarray(v_edges_m, dtype=float).reshape(-1)
    du = np.diff(u_edges)
    dv = np.diff(v_edges)
    u_centers = 0.5 * (u_edges[:-1] + u_edges[1:])
    v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])
    uu, vv = np.meshgrid(u_centers, v_centers, indexing="ij")
    centers = (
        origin[None, :]
        + uu.reshape(-1, 1) * u_axis[None, :]
        + vv.reshape(-1, 1) * v_axis[None, :]
    )
    scale = float(np.linalg.norm(np.cross(u_axis, v_axis)))
    areas = (du[:, None] * dv[None, :] * scale).reshape(-1)
    ids = np.arange(areas.size, dtype=np.int64).reshape(du.size, dv.size)
    edge_parts: list[NDArray[np.int64]] = []
    length_parts: list[NDArray[np.float64]] = []
    if du.size > 1:
        edge_parts.append(np.column_stack([ids[:-1, :].ravel(), ids[1:, :].ravel()]))
        length_parts.append(
            np.broadcast_to(
                dv[None, :] * np.linalg.norm(v_axis), (du.size - 1, dv.size)
            ).ravel()
        )
    if dv.size > 1:
        edge_parts.append(np.column_stack([ids[:, :-1].ravel(), ids[:, 1:].ravel()]))
        length_parts.append(
            np.broadcast_to(
                du[:, None] * np.linalg.norm(u_axis), (du.size, dv.size - 1)
            ).ravel()
        )
    edges = (
        np.vstack(edge_parts).astype(np.int64, copy=False)
        if edge_parts
        else np.zeros((0, 2), dtype=np.int64)
    )
    edge_lengths = (
        np.concatenate(length_parts).astype(float, copy=False)
        if length_parts
        else np.zeros(0, dtype=float)
    )
    u_lower, v_lower = np.meshgrid(
        u_edges[:-1],
        v_edges[:-1],
        indexing="ij",
    )
    u_upper, v_upper = np.meshgrid(
        u_edges[1:],
        v_edges[1:],
        indexing="ij",
    )
    p00 = (
        origin[None, :]
        + u_lower.reshape(-1, 1) * u_axis[None, :]
        + v_lower.reshape(-1, 1) * v_axis[None, :]
    )
    p10 = (
        origin[None, :]
        + u_upper.reshape(-1, 1) * u_axis[None, :]
        + v_lower.reshape(-1, 1) * v_axis[None, :]
    )
    p11 = (
        origin[None, :]
        + u_upper.reshape(-1, 1) * u_axis[None, :]
        + v_upper.reshape(-1, 1) * v_axis[None, :]
    )
    p01 = (
        origin[None, :]
        + u_lower.reshape(-1, 1) * u_axis[None, :]
        + v_upper.reshape(-1, 1) * v_axis[None, :]
    )
    boundary_segments = np.stack(
        [
            np.stack([p00, p10], axis=1),
            np.stack([p10, p11], axis=1),
            np.stack([p11, p01], axis=1),
            np.stack([p01, p00], axis=1),
        ],
        axis=1,
    )
    return _PatchBlock(
        centers=centers,
        areas=areas,
        kinds=np.repeat(np.asarray([str(kind)], dtype=str), areas.size),
        face_ids=np.repeat(np.asarray([str(face_id)], dtype=str), areas.size),
        normals=np.repeat(normal.reshape(1, 3), areas.size, axis=0),
        local_uv=np.column_stack([uu.ravel(), vv.ravel()]),
        edges=edges,
        edge_lengths=edge_lengths,
        boundary_segments=boundary_segments,
    )


def _batched_axis_intervals(
    lengths_m: NDArray[np.float64],
    spacing_m: float,
    extra_cuts_m: NDArray[np.float64] | None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    """Return ragged regular intervals refined by batched physical cut planes."""
    lengths = np.asarray(lengths_m, dtype=float).reshape(-1)
    cell_counts = np.maximum(
        1,
        np.ceil(lengths / float(spacing_m)).astype(np.int64),
    )
    maximum_edges = int(np.max(cell_counts)) + 1
    edge_ids = np.arange(maximum_edges, dtype=np.int64)
    regular_valid = edge_ids[None, :] <= cell_counts[:, None]
    regular_edges = (
        edge_ids[None, :]
        * lengths[:, None]
        / cell_counts[:, None]
    )
    regular_edges[~regular_valid] = np.inf
    if extra_cuts_m is None:
        all_edges = regular_edges
    else:
        extras = np.asarray(extra_cuts_m, dtype=float)
        if extras.ndim != 2 or extras.shape[0] != lengths.size:
            raise ValueError("extra axis cuts must be shaped (F, K).")
        valid_extras = np.isfinite(extras)
        clipped_extras = np.clip(extras, 0.0, lengths[:, None])
        clipped_extras[~valid_extras] = np.inf
        all_edges = np.concatenate([regular_edges, clipped_extras], axis=1)
    sorted_edges = np.sort(all_edges, axis=1)
    lower_edges = sorted_edges[:, :-1]
    upper_edges = sorted_edges[:, 1:]
    finite_pairs = np.isfinite(lower_edges) & np.isfinite(upper_edges)
    differences = np.zeros_like(lower_edges, dtype=float)
    np.subtract(
        upper_edges,
        lower_edges,
        out=differences,
        where=finite_pairs,
    )
    interval_mask = finite_pairs & (differences > _GEOMETRY_TOLERANCE_M)
    counts = np.count_nonzero(interval_mask, axis=1).astype(np.int64)
    offsets = np.cumsum(
        np.concatenate([np.zeros(1, dtype=np.int64), counts[:-1]])
    )
    return (
        lower_edges[interval_mask],
        upper_edges[interval_mask],
        counts,
        offsets,
    )


def _batched_rectangular_face_block(
    *,
    origins_xyz: NDArray[np.float64],
    u_lengths_m: NDArray[np.float64],
    v_lengths_m: NDArray[np.float64],
    u_axis_xyz: Sequence[float],
    v_axis_xyz: Sequence[float],
    normal_xyz: Sequence[float],
    u_spacing_m: float,
    v_spacing_m: float,
    kinds: NDArray[np.str_],
    face_ids: NDArray[np.str_],
    component_indices: NDArray[np.int64],
    u_extra_cuts_m: NDArray[np.float64] | None = None,
    v_extra_cuts_m: NDArray[np.float64] | None = None,
) -> tuple[_PatchBlock, NDArray[np.int64]]:
    """Tessellate unequal rectangular component faces in vectorized batches."""
    origins = np.asarray(origins_xyz, dtype=float).reshape(-1, 3)
    u_lengths = np.asarray(u_lengths_m, dtype=float).reshape(-1)
    v_lengths = np.asarray(v_lengths_m, dtype=float).reshape(-1)
    kind_array = np.asarray(kinds, dtype=str).reshape(-1)
    name_array = np.asarray(face_ids, dtype=str).reshape(-1)
    components = np.asarray(component_indices, dtype=np.int64).reshape(-1)
    face_count = origins.shape[0]
    if not (
        u_lengths.size
        == v_lengths.size
        == kind_array.size
        == name_array.size
        == components.size
        == face_count
    ):
        raise ValueError("Batched rectangular face inputs must have matching lengths.")
    if face_count == 0:
        return _empty_block(), np.zeros(0, dtype=np.int64)
    if np.any(u_lengths <= 0.0) or np.any(v_lengths <= 0.0):
        raise ValueError("Batched rectangular faces must have positive lengths.")

    u_axis = np.asarray(u_axis_xyz, dtype=float).reshape(3)
    v_axis = np.asarray(v_axis_xyz, dtype=float).reshape(3)
    normal = np.asarray(normal_xyz, dtype=float).reshape(3)
    u_interval_lower, u_interval_upper, u_cells, u_offsets = (
        _batched_axis_intervals(
            u_lengths,
            float(u_spacing_m),
            u_extra_cuts_m,
        )
    )
    v_interval_lower, v_interval_upper, v_cells, v_offsets = (
        _batched_axis_intervals(
            v_lengths,
            float(v_spacing_m),
            v_extra_cuts_m,
        )
    )
    patch_counts = u_cells * v_cells
    patch_offsets = np.cumsum(
        np.concatenate([np.zeros(1, dtype=np.int64), patch_counts[:-1]])
    )
    owners = np.repeat(np.arange(face_count, dtype=np.int64), patch_counts)
    local_ids = np.arange(int(np.sum(patch_counts)), dtype=np.int64) - np.repeat(
        patch_offsets,
        patch_counts,
    )
    owner_v_cells = v_cells[owners]
    u_ids = local_ids // owner_v_cells
    v_ids = local_ids % owner_v_cells
    u_interval_ids = u_offsets[owners] + u_ids
    v_interval_ids = v_offsets[owners] + v_ids
    u_lower = u_interval_lower[u_interval_ids]
    u_upper = u_interval_upper[u_interval_ids]
    v_lower = v_interval_lower[v_interval_ids]
    v_upper = v_interval_upper[v_interval_ids]
    du = u_upper - u_lower
    dv = v_upper - v_lower
    u_centers = 0.5 * (u_lower + u_upper)
    v_centers = 0.5 * (v_lower + v_upper)
    centers = (
        origins[owners]
        + u_centers[:, None] * u_axis[None, :]
        + v_centers[:, None] * v_axis[None, :]
    )
    scale = float(np.linalg.norm(np.cross(u_axis, v_axis)))

    u_edge_counts = np.maximum(u_cells - 1, 0) * v_cells
    u_edge_owners = np.repeat(
        np.arange(face_count, dtype=np.int64),
        u_edge_counts,
    )
    u_edge_offsets = np.cumsum(
        np.concatenate([np.zeros(1, dtype=np.int64), u_edge_counts[:-1]])
    )
    u_edge_local = np.arange(int(np.sum(u_edge_counts)), dtype=np.int64) - np.repeat(
        u_edge_offsets,
        u_edge_counts,
    )
    u_edge_v_cells = v_cells[u_edge_owners]
    u_edge_left = (
        patch_offsets[u_edge_owners]
        + (u_edge_local // u_edge_v_cells) * u_edge_v_cells
        + u_edge_local % u_edge_v_cells
    )
    u_edges = np.column_stack(
        [u_edge_left, u_edge_left + u_edge_v_cells]
    ).astype(np.int64, copy=False)
    u_edge_v_ids = u_edge_local % u_edge_v_cells
    u_edge_interval_ids = v_offsets[u_edge_owners] + u_edge_v_ids
    u_edge_lengths = (
        v_interval_upper[u_edge_interval_ids]
        - v_interval_lower[u_edge_interval_ids]
    ) * np.linalg.norm(v_axis)

    v_edge_counts = u_cells * np.maximum(v_cells - 1, 0)
    v_edge_owners = np.repeat(
        np.arange(face_count, dtype=np.int64),
        v_edge_counts,
    )
    v_edge_offsets = np.cumsum(
        np.concatenate([np.zeros(1, dtype=np.int64), v_edge_counts[:-1]])
    )
    v_edge_local = np.arange(int(np.sum(v_edge_counts)), dtype=np.int64) - np.repeat(
        v_edge_offsets,
        v_edge_counts,
    )
    v_neighbor_count = np.maximum(v_cells[v_edge_owners] - 1, 1)
    v_edge_left = (
        patch_offsets[v_edge_owners]
        + (v_edge_local // v_neighbor_count) * v_cells[v_edge_owners]
        + v_edge_local % v_neighbor_count
    )
    v_edges = np.column_stack([v_edge_left, v_edge_left + 1]).astype(
        np.int64,
        copy=False,
    )
    v_edge_u_ids = v_edge_local // v_neighbor_count
    v_edge_interval_ids = u_offsets[v_edge_owners] + v_edge_u_ids
    v_edge_lengths = (
        u_interval_upper[v_edge_interval_ids]
        - u_interval_lower[v_edge_interval_ids]
    ) * np.linalg.norm(u_axis)
    edge_parts = [part for part in (u_edges, v_edges) if part.size]
    edge_length_parts = [
        part for part in (u_edge_lengths, v_edge_lengths) if part.size
    ]

    patch_origins = origins[owners]
    p00 = (
        patch_origins
        + u_lower[:, None] * u_axis[None, :]
        + v_lower[:, None] * v_axis[None, :]
    )
    p10 = (
        patch_origins
        + u_upper[:, None] * u_axis[None, :]
        + v_lower[:, None] * v_axis[None, :]
    )
    p11 = (
        patch_origins
        + u_upper[:, None] * u_axis[None, :]
        + v_upper[:, None] * v_axis[None, :]
    )
    p01 = (
        patch_origins
        + u_lower[:, None] * u_axis[None, :]
        + v_upper[:, None] * v_axis[None, :]
    )
    boundary_segments = np.stack(
        [
            np.stack([p00, p10], axis=1),
            np.stack([p10, p11], axis=1),
            np.stack([p11, p01], axis=1),
            np.stack([p01, p00], axis=1),
        ],
        axis=1,
    )
    return (
        _PatchBlock(
            centers=centers,
            areas=du * dv * scale,
            kinds=kind_array[owners],
            face_ids=name_array[owners],
            normals=np.repeat(normal.reshape(1, 3), centers.shape[0], axis=0),
            local_uv=np.column_stack([u_centers, v_centers]),
            edges=(
                np.vstack(edge_parts).astype(np.int64, copy=False)
                if edge_parts
                else np.zeros((0, 2), dtype=np.int64)
            ),
            edge_lengths=(
                np.concatenate(edge_length_parts).astype(float, copy=False)
                if edge_length_parts
                else np.zeros(0, dtype=float)
            ),
            boundary_segments=boundary_segments,
        ),
        components[owners],
    )


def _component_face_touch_mask(
    boxes_m: NDArray[np.float64],
    component_indices: NDArray[np.int64],
    *,
    face_axis: int,
    upper_face: bool,
    u_axis: int,
    v_axis: int,
) -> NDArray[np.bool_]:
    """Return components touching each selected face over positive area."""
    boxes = np.asarray(boxes_m, dtype=float).reshape(-1, 6)
    selected = np.asarray(component_indices, dtype=np.int64).reshape(-1)
    lower = boxes[:, :3]
    upper = boxes[:, 3:]
    face_plane = upper[selected, face_axis] if upper_face else lower[selected, face_axis]
    neighbor_plane = lower[:, face_axis] if upper_face else upper[:, face_axis]
    coplanar = (
        np.abs(face_plane[:, None] - neighbor_plane[None, :])
        <= _GEOMETRY_TOLERANCE_M
    )
    overlap_u = (
        np.minimum(upper[selected, u_axis, None], upper[None, :, u_axis])
        - np.maximum(lower[selected, u_axis, None], lower[None, :, u_axis])
        > _GEOMETRY_TOLERANCE_M
    )
    overlap_v = (
        np.minimum(upper[selected, v_axis, None], upper[None, :, v_axis])
        - np.maximum(lower[selected, v_axis, None], lower[None, :, v_axis])
        > _GEOMETRY_TOLERANCE_M
    )
    not_self = selected[:, None] != np.arange(boxes.shape[0], dtype=np.int64)[None, :]
    return coplanar & overlap_u & overlap_v & not_self


def _transport_component_surface_blocks(
    env: EnvironmentConfig,
    boxes_m: NDArray[np.float64],
    spacing_xyz_m: NDArray[np.float64],
) -> list[_PatchBlock]:
    """Build exposed surfaces of known component boxes in six batched passes."""
    boxes = np.asarray(boxes_m, dtype=float).reshape(-1, 6)
    if boxes.shape[0] == 0:
        return []
    lower = boxes[:, :3]
    upper = boxes[:, 3:]
    lengths = upper - lower
    room_upper = np.asarray(
        [float(env.size_x), float(env.size_y), float(env.size_z)],
        dtype=float,
    )
    face_specs = (
        (0, False, 1, 2, (-1.0, 0.0, 0.0), "obstacle_side", "x0"),
        (0, True, 1, 2, (1.0, 0.0, 0.0), "obstacle_side", "x1"),
        (1, False, 0, 2, (0.0, -1.0, 0.0), "obstacle_side", "y0"),
        (1, True, 0, 2, (0.0, 1.0, 0.0), "obstacle_side", "y1"),
        (2, False, 0, 1, (0.0, 0.0, -1.0), "obstacle_bottom", "z0"),
        (2, True, 0, 1, (0.0, 0.0, 1.0), "obstacle_top", "z1"),
    )
    basis = np.eye(3, dtype=float)
    blocks: list[_PatchBlock] = []
    component_ids = np.arange(boxes.shape[0], dtype=np.int64)
    for axis, upper_face, u_axis, v_axis, normal, kind, suffix in face_specs:
        plane = upper[:, axis] if upper_face else lower[:, axis]
        inside_room = (
            plane < room_upper[axis] - _GEOMETRY_TOLERANCE_M
            if upper_face
            else plane > _GEOMETRY_TOLERANCE_M
        )
        selected = np.flatnonzero(inside_room)
        if selected.size == 0:
            continue
        origins = lower[selected].copy()
        origins[:, axis] = plane[selected]
        component_names = np.char.add(
            np.char.add("transport_component_", component_ids[selected].astype(str)),
            f"_{suffix}",
        )
        touching = _component_face_touch_mask(
            boxes,
            component_ids[selected],
            face_axis=axis,
            upper_face=upper_face,
            u_axis=u_axis,
            v_axis=v_axis,
        )
        u_neighbor_bounds = np.stack(
            [lower[:, u_axis], upper[:, u_axis]],
            axis=1,
        )
        v_neighbor_bounds = np.stack(
            [lower[:, v_axis], upper[:, v_axis]],
            axis=1,
        )
        u_extra_cuts = np.where(
            touching[:, :, None],
            u_neighbor_bounds[None, :, :] - lower[selected, u_axis, None, None],
            np.nan,
        ).reshape(selected.size, -1)
        v_extra_cuts = np.where(
            touching[:, :, None],
            v_neighbor_bounds[None, :, :] - lower[selected, v_axis, None, None],
            np.nan,
        ).reshape(selected.size, -1)
        block, patch_components = _batched_rectangular_face_block(
            origins_xyz=origins,
            u_lengths_m=lengths[selected, u_axis],
            v_lengths_m=lengths[selected, v_axis],
            u_axis_xyz=basis[u_axis],
            v_axis_xyz=basis[v_axis],
            normal_xyz=normal,
            u_spacing_m=float(spacing_xyz_m[u_axis]),
            v_spacing_m=float(spacing_xyz_m[v_axis]),
            kinds=np.repeat(np.asarray([kind], dtype=str), selected.size),
            face_ids=component_names,
            component_indices=component_ids[selected],
            u_extra_cuts_m=u_extra_cuts,
            v_extra_cuts_m=v_extra_cuts,
        )
        outward_probe = block.centers + (
            16.0 * _GEOMETRY_TOLERANCE_M
        ) * np.asarray(normal, dtype=float)[None, :]
        obscured = _points_inside_box_union(
            outward_probe,
            boxes,
            excluded_box_indices=patch_components,
        )
        blocks.append(_filter_patch_block(block, ~obscured))
    return blocks


def _filter_patch_block(
    block: _PatchBlock, keep_mask: NDArray[np.bool_]
) -> _PatchBlock:
    """Filter patches and remap graph edges in batch."""
    keep = np.asarray(keep_mask, dtype=bool).reshape(-1)
    if keep.size != block.areas.size:
        raise ValueError("keep_mask must have one entry per patch.")
    remap = np.full(keep.size, -1, dtype=np.int64)
    remap[keep] = np.arange(np.count_nonzero(keep), dtype=np.int64)
    if block.edges.size:
        edge_keep = keep[block.edges[:, 0]] & keep[block.edges[:, 1]]
        edges = remap[block.edges[edge_keep]]
        edge_lengths = block.edge_lengths[edge_keep]
    else:
        edges = np.zeros((0, 2), dtype=np.int64)
        edge_lengths = np.zeros(0, dtype=float)
    return _PatchBlock(
        centers=block.centers[keep],
        areas=block.areas[keep],
        kinds=block.kinds[keep],
        face_ids=block.face_ids[keep],
        normals=block.normals[keep],
        local_uv=block.local_uv[keep],
        edges=edges,
        edge_lengths=edge_lengths,
        boundary_segments=block.boundary_segments[keep],
    )


def _grid_component_neighbor_pairs(
    cells: NDArray[np.int64],
    grid_shape: Sequence[int],
    *,
    axis: int,
) -> NDArray[np.int64]:
    """Return batched positive-axis neighbor pairs in component-index space."""
    cell_array = np.asarray(cells, dtype=np.int64).reshape(-1, 2)
    shape = np.asarray(grid_shape, dtype=np.int64).reshape(-1)
    if shape.shape != (2,) or np.any(shape < 0):
        raise ValueError("grid_shape must be a non-negative two-vector.")
    if int(axis) not in {0, 1}:
        raise ValueError("axis must be zero or one.")
    if cell_array.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if np.any(cell_array < 0) or np.any(cell_array >= shape[None, :]):
        raise ValueError("cells must lie inside grid_shape.")

    stride = int(shape[1]) if int(axis) == 0 else 1
    codes = cell_array[:, 0] * int(shape[1]) + cell_array[:, 1]
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    lower_indices = np.flatnonzero(cell_array[:, int(axis)] + 1 < shape[int(axis)])
    if lower_indices.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    target_codes = codes[lower_indices] + stride
    matched_positions = np.searchsorted(sorted_codes, target_codes)
    in_range = matched_positions < sorted_codes.size
    safe_positions = np.minimum(matched_positions, sorted_codes.size - 1)
    matched = in_range & (sorted_codes[safe_positions] == target_codes)
    if not np.any(matched):
        return np.zeros((0, 2), dtype=np.int64)
    return np.column_stack(
        [
            lower_indices[matched],
            order[safe_positions[matched]],
        ]
    ).astype(np.int64, copy=False)


def _component_boundary_edges(
    component_pairs: NDArray[np.int64] | None,
    *,
    face_count: int,
    patches_per_face: int,
    lower_boundary_ids: NDArray[np.int64],
    upper_boundary_ids: NDArray[np.int64],
    shared_lengths_m: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Expand component neighbors into aligned patch-boundary graph edges."""
    if component_pairs is None:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=float)
    pairs = np.asarray(component_pairs, dtype=np.int64)
    if pairs.size == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=float)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("component neighbor pairs must be shaped (N, 2).")
    if np.any(pairs < 0) or np.any(pairs >= int(face_count)):
        raise ValueError("component neighbor pair index is out of range.")
    lower_ids = np.asarray(lower_boundary_ids, dtype=np.int64).reshape(-1)
    upper_ids = np.asarray(upper_boundary_ids, dtype=np.int64).reshape(-1)
    lengths = np.asarray(shared_lengths_m, dtype=float).reshape(-1)
    if lower_ids.size != upper_ids.size or lower_ids.size != lengths.size:
        raise ValueError("boundary patch IDs and shared lengths must align.")
    left = pairs[:, 0, None] * int(patches_per_face) + lower_ids[None, :]
    right = pairs[:, 1, None] * int(patches_per_face) + upper_ids[None, :]
    return (
        np.stack([left, right], axis=-1).reshape(-1, 2),
        np.broadcast_to(lengths[None, :], left.shape).reshape(-1).copy(),
    )


def _repeated_face_blocks(
    *,
    origins_xyz: NDArray[np.float64],
    face_ids: NDArray[np.str_],
    u_axis_xyz: Sequence[float],
    v_axis_xyz: Sequence[float],
    u_edges_m: NDArray[np.float64],
    v_edges_m: NDArray[np.float64],
    normal_xyz: Sequence[float],
    kind: str,
    u_component_neighbors: NDArray[np.int64] | None = None,
    v_component_neighbors: NDArray[np.int64] | None = None,
) -> _PatchBlock:
    """Build equal faces and connect contiguous component boundaries in batch."""
    origins = np.asarray(origins_xyz, dtype=float).reshape(-1, 3)
    names = np.asarray(face_ids, dtype=str).reshape(-1)
    if origins.shape[0] != names.size:
        raise ValueError("origins_xyz and face_ids must have matching lengths.")
    if origins.shape[0] == 0:
        return _empty_block()
    template = _face_block(
        origin_xyz=(0.0, 0.0, 0.0),
        u_axis_xyz=u_axis_xyz,
        v_axis_xyz=v_axis_xyz,
        u_edges_m=u_edges_m,
        v_edges_m=v_edges_m,
        normal_xyz=normal_xyz,
        kind=kind,
        face_id="template",
    )
    face_count = int(origins.shape[0])
    patch_count = int(template.areas.size)
    centers = (origins[:, None, :] + template.centers[None, :, :]).reshape(-1, 3)
    offsets = np.arange(face_count, dtype=np.int64) * patch_count
    internal_edges = (template.edges[None, :, :] + offsets[:, None, None]).reshape(
        -1, 2
    )
    u_edges = np.asarray(u_edges_m, dtype=float).reshape(-1)
    v_edges = np.asarray(v_edges_m, dtype=float).reshape(-1)
    u_patch_count = int(u_edges.size - 1)
    v_patch_count = int(v_edges.size - 1)
    patch_ids = np.arange(patch_count, dtype=np.int64).reshape(
        u_patch_count,
        v_patch_count,
    )
    cross_u_edges, cross_u_lengths = _component_boundary_edges(
        u_component_neighbors,
        face_count=face_count,
        patches_per_face=patch_count,
        lower_boundary_ids=patch_ids[-1, :],
        upper_boundary_ids=patch_ids[0, :],
        shared_lengths_m=np.diff(v_edges) * np.linalg.norm(v_axis_xyz),
    )
    cross_v_edges, cross_v_lengths = _component_boundary_edges(
        v_component_neighbors,
        face_count=face_count,
        patches_per_face=patch_count,
        lower_boundary_ids=patch_ids[:, -1],
        upper_boundary_ids=patch_ids[:, 0],
        shared_lengths_m=np.diff(u_edges) * np.linalg.norm(u_axis_xyz),
    )
    edge_parts = [
        part for part in (internal_edges, cross_u_edges, cross_v_edges) if part.size
    ]
    length_parts = [
        part
        for part in (
            np.tile(template.edge_lengths, face_count),
            cross_u_lengths,
            cross_v_lengths,
        )
        if part.size
    ]
    edges = (
        np.vstack(edge_parts).astype(np.int64, copy=False)
        if edge_parts
        else np.zeros((0, 2), dtype=np.int64)
    )
    edge_lengths = (
        np.concatenate(length_parts).astype(float, copy=False)
        if length_parts
        else np.zeros(0, dtype=float)
    )
    return _PatchBlock(
        centers=centers,
        areas=np.tile(template.areas, face_count),
        kinds=np.repeat(
            np.asarray([str(kind)], dtype=str),
            face_count * patch_count,
        ),
        face_ids=np.repeat(names, patch_count),
        normals=np.tile(template.normals, (face_count, 1)),
        local_uv=np.tile(template.local_uv, (face_count, 1)),
        edges=edges.astype(np.int64, copy=False),
        edge_lengths=edge_lengths,
        boundary_segments=(
            origins[:, None, None, None, :]
            + template.boundary_segments[None, :, :, :, :]
        ).reshape(-1, 4, 2, 3),
    )


def _empty_block() -> _PatchBlock:
    """Return an empty patch-construction block."""
    return _PatchBlock(
        centers=np.zeros((0, 3), dtype=float),
        areas=np.zeros(0, dtype=float),
        kinds=np.zeros(0, dtype=str),
        face_ids=np.zeros(0, dtype=str),
        normals=np.zeros((0, 3), dtype=float),
        local_uv=np.zeros((0, 2), dtype=float),
        edges=np.zeros((0, 2), dtype=np.int64),
        edge_lengths=np.zeros(0, dtype=float),
        boundary_segments=np.zeros((0, 4, 2, 3), dtype=float),
    )


def _cross_face_boundary_adjacency(
    boundary_segments: NDArray[np.float64],
    face_ids: NDArray[np.str_],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Match exact collinear patch boundaries in bounded vectorized groups."""
    boundaries = np.asarray(boundary_segments, dtype=float)
    names = np.asarray(face_ids, dtype=str).reshape(-1)
    if boundaries.shape != (names.size, 4, 2, 3):
        raise ValueError("boundary_segments must be shaped (C, 4, 2, 3).")
    if names.size == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=float)
    segments = boundaries.reshape(-1, 2, 3)
    patch_indices = np.repeat(np.arange(names.size, dtype=np.int64), 4)
    _, patch_face_codes = np.unique(names, return_inverse=True)
    segment_face_codes = np.repeat(patch_face_codes.astype(np.int64), 4)
    delta = np.abs(segments[:, 1, :] - segments[:, 0, :])
    axes = np.argmax(delta, axis=1).astype(np.int64)
    lengths = delta[np.arange(delta.shape[0]), axes]
    residual = delta.copy()
    residual[np.arange(delta.shape[0]), axes] = 0.0
    valid = (lengths > _GEOMETRY_TOLERANCE_M) & np.all(
        residual <= _GEOMETRY_TOLERANCE_M,
        axis=1,
    )
    if not np.all(valid):
        raise ValueError("Surface patch boundaries must be non-degenerate and axis-aligned.")

    midpoint = np.mean(segments, axis=1)
    fixed = np.rint(midpoint / _GEOMETRY_TOLERANCE_M).astype(np.int64)
    fixed[np.arange(fixed.shape[0]), axes] = 0
    order = np.lexsort((fixed[:, 2], fixed[:, 1], fixed[:, 0], axes))
    ordered_axes = axes[order]
    ordered_fixed = fixed[order]
    group_break = np.ones(order.size, dtype=bool)
    group_break[1:] = (
        (ordered_axes[1:] != ordered_axes[:-1])
        | np.any(ordered_fixed[1:] != ordered_fixed[:-1], axis=1)
    )
    group_starts = np.flatnonzero(group_break)
    group_stops = np.concatenate(
        [group_starts[1:], np.asarray([order.size], dtype=np.int64)]
    )
    group_sizes = group_stops - group_starts
    active = group_sizes >= 2
    starts = group_starts[active]
    sizes = group_sizes[active]
    if starts.size == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=float)

    lower = np.minimum(
        segments[np.arange(segments.shape[0]), 0, axes],
        segments[np.arange(segments.shape[0]), 1, axes],
    )
    upper = np.maximum(
        segments[np.arange(segments.shape[0]), 0, axes],
        segments[np.arange(segments.shape[0]), 1, axes],
    )
    pair_parts: list[NDArray[np.int64]] = []
    length_parts: list[NDArray[np.float64]] = []
    batch_start = 0
    while batch_start < starts.size:
        tentative_stop = min(
            batch_start + _ADJACENCY_GROUP_BATCH_SIZE,
            starts.size,
        )
        maximum_size = int(np.max(sizes[batch_start:tentative_stop]))
        pair_budget = 2_000_000
        groups_per_batch = max(1, pair_budget // max(maximum_size**2, 1))
        batch_stop = min(tentative_stop, batch_start + groups_per_batch)
        batch_starts = starts[batch_start:batch_stop]
        batch_sizes = sizes[batch_start:batch_stop]
        maximum_size = int(np.max(batch_sizes))
        offsets = np.arange(maximum_size, dtype=np.int64)
        valid_offsets = offsets[None, :] < batch_sizes[:, None]
        ordered_positions = batch_starts[:, None] + offsets[None, :]
        safe_positions = np.minimum(ordered_positions, order.size - 1)
        segment_indices = order[safe_positions]
        group_patches = patch_indices[segment_indices]
        group_faces = segment_face_codes[segment_indices]
        group_lower = lower[segment_indices]
        group_upper = upper[segment_indices]
        overlap = np.minimum(
            group_upper[:, :, None],
            group_upper[:, None, :],
        ) - np.maximum(
            group_lower[:, :, None],
            group_lower[:, None, :],
        )
        pair_mask = (
            valid_offsets[:, :, None]
            & valid_offsets[:, None, :]
            & (group_patches[:, :, None] != group_patches[:, None, :])
            & (group_faces[:, :, None] != group_faces[:, None, :])
            & (overlap > _GEOMETRY_TOLERANCE_M)
        )
        pair_mask &= np.triu(
            np.ones((maximum_size, maximum_size), dtype=bool),
            k=1,
        )[None, :, :]
        group_ids, left_ids, right_ids = np.nonzero(pair_mask)
        if group_ids.size:
            left = group_patches[group_ids, left_ids]
            right = group_patches[group_ids, right_ids]
            pair_parts.append(
                np.column_stack([np.minimum(left, right), np.maximum(left, right)])
            )
            length_parts.append(overlap[group_ids, left_ids, right_ids])
        batch_start = batch_stop

    if not pair_parts:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=float)
    pairs = np.vstack(pair_parts).astype(np.int64, copy=False)
    overlap_lengths = np.concatenate(length_parts).astype(float, copy=False)
    pair_order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs = pairs[pair_order]
    overlap_lengths = overlap_lengths[pair_order]
    first = np.ones(pairs.shape[0], dtype=bool)
    first[1:] = np.any(pairs[1:] != pairs[:-1], axis=1)
    starts = np.flatnonzero(first)
    return pairs[starts], np.add.reduceat(overlap_lengths, starts)


def _deduplicate_adjacency(
    edges: NDArray[np.int64],
    lengths_m: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Canonicalize graph edges and keep one physical length per patch pair."""
    edge_array = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    lengths = np.asarray(lengths_m, dtype=float).reshape(-1)
    if edge_array.shape[0] == 0:
        return edge_array, lengths
    canonical = np.sort(edge_array, axis=1)
    order = np.lexsort((canonical[:, 1], canonical[:, 0]))
    canonical = canonical[order]
    lengths = lengths[order]
    first = np.ones(canonical.shape[0], dtype=bool)
    first[1:] = np.any(canonical[1:] != canonical[:-1], axis=1)
    starts = np.flatnonzero(first)
    return canonical[starts], np.maximum.reduceat(lengths, starts)


def _combine_patch_blocks(
    blocks: Sequence[_PatchBlock],
    *,
    obstacle_geometry_source: str,
    obstacle_surfaces_available: bool,
    obstacle_component_count: int,
    obstacle_geometry_warning: str | None,
) -> SurfacePatchDictionary:
    """Combine blocks and connect every exactly shared physical boundary."""
    active = [block for block in blocks if block.areas.size]
    if not active:
        raise ValueError("Surface patch dictionary is empty.")
    offsets = np.cumsum([0, *[int(block.areas.size) for block in active[:-1]]])
    internal_edges = [
        block.edges + int(offset)
        for block, offset in zip(active, offsets)
        if block.edges.size
    ]
    internal_lengths = [
        block.edge_lengths for block in active if block.edges.size
    ]
    face_ids = np.concatenate([block.face_ids for block in active])
    cross_edges, cross_lengths = _cross_face_boundary_adjacency(
        np.concatenate([block.boundary_segments for block in active], axis=0),
        face_ids,
    )
    edge_parts = [*internal_edges]
    length_parts = [*internal_lengths]
    if cross_edges.size:
        edge_parts.append(cross_edges)
        length_parts.append(cross_lengths)
    edges, edge_lengths = _deduplicate_adjacency(
        np.vstack(edge_parts) if edge_parts else np.zeros((0, 2), dtype=np.int64),
        np.concatenate(length_parts) if length_parts else np.zeros(0, dtype=float),
    )
    return SurfacePatchDictionary(
        centers_xyz=np.vstack([block.centers for block in active]),
        areas_m2=np.concatenate([block.areas for block in active]),
        kinds=tuple(np.concatenate([block.kinds for block in active]).tolist()),
        face_ids=tuple(face_ids.tolist()),
        normals_xyz=np.vstack([block.normals for block in active]),
        local_uv_m=np.vstack([block.local_uv for block in active]),
        adjacency_edges=edges,
        shared_edge_lengths_m=edge_lengths,
        obstacle_geometry_source=obstacle_geometry_source,
        obstacle_surfaces_available=obstacle_surfaces_available,
        obstacle_component_count=obstacle_component_count,
        obstacle_geometry_warning=obstacle_geometry_warning,
    )


def estimate_surface_patch_count_upper_bound(
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    spacing: float | Sequence[float],
    *,
    obstacle_height_m: float = 2.0,
) -> int:
    """Estimate a conservative patch count without constructing patch arrays."""
    del obstacle_height_m
    spacing_array = _spacing_vector(spacing)
    x_cells = _axis_cell_count(float(env.size_x), float(spacing_array[0]))
    y_cells = _axis_cell_count(float(env.size_y), float(spacing_array[1]))
    z_cells = _axis_cell_count(float(env.size_z), float(spacing_array[2]))
    boxes = _clipped_transport_boxes(env, obstacle_grid)
    floor_boxes = _floor_contact_boxes(boxes)
    x_boundaries = floor_boxes[:, (0, 3)].reshape(-1)
    y_boundaries = floor_boxes[:, (1, 4)].reshape(-1)
    interior_x_boundaries = np.unique(
        x_boundaries[
            (x_boundaries > 0.0) & (x_boundaries < float(env.size_x))
        ]
    )
    interior_y_boundaries = np.unique(
        y_boundaries[
            (y_boundaries > 0.0) & (y_boundaries < float(env.size_y))
        ]
    )
    floor_x_cells_upper = x_cells + int(interior_x_boundaries.size)
    floor_y_cells_upper = y_cells + int(interior_y_boundaries.size)
    floor_cells = floor_x_cells_upper * floor_y_cells_upper
    room_patch_count = (
        floor_cells
        + x_cells * y_cells
        + 2 * y_cells * z_cells
        + 2 * x_cells * z_cells
    )
    if boxes.shape[0] == 0:
        return int(room_patch_count)
    dimensions = boxes[:, 3:] - boxes[:, :3]
    component_cells = np.maximum(
        1,
        np.ceil(dimensions / spacing_array[None, :]).astype(np.int64),
    )
    x_count = component_cells[:, 0]
    y_count = component_cells[:, 1]
    z_count = component_cells[:, 2]
    component_patch_count = np.sum(
        2 * (x_count * y_count + x_count * z_count + y_count * z_count),
        dtype=np.int64,
    )
    return int(room_patch_count + component_patch_count)


def build_surface_patch_dictionary(
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    spacing: float | Sequence[float],
    *,
    obstacle_height_m: float = 2.0,
) -> SurfacePatchDictionary:
    """Build area-aware patches with physical shared-boundary adjacency."""
    del obstacle_height_m
    spacing_arr = _spacing_vector(spacing)
    transport_boxes = _clipped_transport_boxes(env, obstacle_grid)
    x_edges = _axis_edges(float(env.size_x), float(spacing_arr[0]))
    y_edges = _axis_edges(float(env.size_y), float(spacing_arr[1]))
    z_edges = _axis_edges(float(env.size_z), float(spacing_arr[2]))
    floor_x_edges, floor_y_edges = _floor_edges_with_obstacle_boundaries(
        env,
        obstacle_grid,
        x_edges,
        y_edges,
    )
    floor = _face_block(
        origin_xyz=(0.0, 0.0, 0.0),
        u_axis_xyz=(1.0, 0.0, 0.0),
        v_axis_xyz=(0.0, 1.0, 0.0),
        u_edges_m=floor_x_edges,
        v_edges_m=floor_y_edges,
        normal_xyz=(0.0, 0.0, 1.0),
        kind="floor",
        face_id="room_floor",
    )
    if transport_boxes.shape[0]:
        floor = _filter_patch_block(
            floor,
            ~_floor_footprint_mask(floor.centers, transport_boxes),
        )
    blocks: list[_PatchBlock] = [
        floor,
        _face_block(
            origin_xyz=(0.0, 0.0, float(env.size_z)),
            u_axis_xyz=(1.0, 0.0, 0.0),
            v_axis_xyz=(0.0, 1.0, 0.0),
            u_edges_m=x_edges,
            v_edges_m=y_edges,
            normal_xyz=(0.0, 0.0, -1.0),
            kind="ceiling",
            face_id="room_ceiling",
        ),
        _face_block(
            origin_xyz=(0.0, 0.0, 0.0),
            u_axis_xyz=(0.0, 1.0, 0.0),
            v_axis_xyz=(0.0, 0.0, 1.0),
            u_edges_m=y_edges,
            v_edges_m=z_edges,
            normal_xyz=(1.0, 0.0, 0.0),
            kind="wall",
            face_id="room_wall_x0",
        ),
        _face_block(
            origin_xyz=(float(env.size_x), 0.0, 0.0),
            u_axis_xyz=(0.0, 1.0, 0.0),
            v_axis_xyz=(0.0, 0.0, 1.0),
            u_edges_m=y_edges,
            v_edges_m=z_edges,
            normal_xyz=(-1.0, 0.0, 0.0),
            kind="wall",
            face_id="room_wall_x1",
        ),
        _face_block(
            origin_xyz=(0.0, 0.0, 0.0),
            u_axis_xyz=(1.0, 0.0, 0.0),
            v_axis_xyz=(0.0, 0.0, 1.0),
            u_edges_m=x_edges,
            v_edges_m=z_edges,
            normal_xyz=(0.0, 1.0, 0.0),
            kind="wall",
            face_id="room_wall_y0",
        ),
        _face_block(
            origin_xyz=(0.0, float(env.size_y), 0.0),
            u_axis_xyz=(1.0, 0.0, 0.0),
            v_axis_xyz=(0.0, 0.0, 1.0),
            u_edges_m=x_edges,
            v_edges_m=z_edges,
            normal_xyz=(0.0, -1.0, 0.0),
            kind="wall",
            face_id="room_wall_y1",
        ),
    ]
    blocks.extend(
        _transport_component_surface_blocks(env, transport_boxes, spacing_arr)
    )

    blocked_cells_exist = bool(
        obstacle_grid is not None and obstacle_grid.blocked_cells
    )
    if transport_boxes.shape[0]:
        geometry_source = "transport_boxes_m"
        surfaces_available = True
        geometry_warning = None
    elif blocked_cells_exist:
        geometry_source = "blocked_cells_only"
        surfaces_available = False
        geometry_warning = (
            "Obstacle cells constrain navigation, but no component geometry is "
            "available; synthetic uniform-box source surfaces were omitted."
        )
    else:
        geometry_source = "none"
        surfaces_available = True
        geometry_warning = None
    return _combine_patch_blocks(
        blocks,
        obstacle_geometry_source=geometry_source,
        obstacle_surfaces_available=surfaces_available,
        obstacle_component_count=int(transport_boxes.shape[0]),
        obstacle_geometry_warning=geometry_warning,
    )
