"""Continuous rectangular surface charts for source-state inference.

The physical surface builder tessellates exposed room and obstacle faces into
rectangles.  Those rectangles are used here only as an atlas: a source state
stores a chart identifier and continuous unit-square coordinates inside that
chart.  The tessellation therefore does not quantize source positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from measurement.surface_charts import SurfaceChartGeometry


_SURFACE_TOLERANCE_M = 1.0e-8
_DIRECTION_TOLERANCE = 1.0e-12
_LOCATE_POINT_BATCH_SIZE = 1_024
_U_LOWER = 0
_U_UPPER = 1
_V_LOWER = 2
_V_UPPER = 3
_EDGE_INDEX_TO_SIDE = np.asarray(
    [_V_LOWER, _U_UPPER, _V_UPPER, _U_LOWER],
    dtype=np.int8,
)


def _rectangle_edge_segments(
    vertices_xyz: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return four directed boundary segments for every rectangle."""
    vertices = np.asarray(vertices_xyz, dtype=np.float64)
    return np.stack(
        (
            vertices[:, (0, 1), :],
            vertices[:, (1, 2), :],
            vertices[:, (2, 3), :],
            vertices[:, (3, 0), :],
        ),
        axis=1,
    )


def _build_portal_table(
    vertices_xyz: NDArray[np.float64],
    adjacency_edges: NDArray[np.int64],
    *,
    tolerance_m: float,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int8],
    NDArray[np.int8],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Build padded directed portals from exactly shared boundary segments.

    Geometry matching is batched over all adjacency pairs and all sixteen
    rectangle-edge combinations.  An adjacency contributes portals only when
    exactly one collinear, positive-length overlap is found.  This deliberately
    leaves malformed or geometrically ambiguous adjacency pairs disconnected.
    """
    vertices = np.asarray(vertices_xyz, dtype=np.float64)
    adjacency = np.asarray(adjacency_edges, dtype=np.int64).reshape(-1, 2)
    chart_count = int(vertices.shape[0])
    if adjacency.shape[0] == 0:
        return (
            np.full((chart_count, 0), -1, dtype=np.int64),
            np.zeros((chart_count, 0), dtype=np.int8),
            np.zeros((chart_count, 0), dtype=np.int8),
            np.zeros((chart_count, 0, 3), dtype=np.float64),
            np.zeros((chart_count, 0, 3), dtype=np.float64),
        )

    segments = _rectangle_edge_segments(vertices)
    left_segments = segments[adjacency[:, 0]]
    right_segments = segments[adjacency[:, 1]]
    left_start = left_segments[:, :, None, 0, :]
    right_start = right_segments[:, None, :, 0, :]
    left_delta = left_segments[:, :, 1, :] - left_segments[:, :, 0, :]
    right_delta = right_segments[:, :, 1, :] - right_segments[:, :, 0, :]
    left_length = np.linalg.norm(left_delta, axis=2)
    right_length = np.linalg.norm(right_delta, axis=2)
    left_axis = left_delta / left_length[:, :, None]
    right_axis = right_delta / right_length[:, :, None]

    parallel_error = np.linalg.norm(
        np.cross(left_axis[:, :, None, :], right_axis[:, None, :, :]),
        axis=3,
    )
    offset = right_start - left_start
    offset_projection = np.sum(
        offset * left_axis[:, :, None, :],
        axis=3,
    )
    line_error = np.linalg.norm(
        offset - offset_projection[:, :, :, None]
        * left_axis[:, :, None, :],
        axis=3,
    )
    right_end_offset = (
        right_segments[:, None, :, 1, :] - left_start
    )
    right_start_projection = offset_projection
    right_end_projection = np.sum(
        right_end_offset * left_axis[:, :, None, :],
        axis=3,
    )
    overlap_lower = np.maximum(
        0.0,
        np.minimum(right_start_projection, right_end_projection),
    )
    overlap_upper = np.minimum(
        left_length[:, :, None],
        np.maximum(right_start_projection, right_end_projection),
    )
    valid = (
        (parallel_error <= _DIRECTION_TOLERANCE)
        & (line_error <= tolerance_m)
        & (overlap_upper - overlap_lower > tolerance_m)
    )
    unique = np.sum(valid, axis=(1, 2)) == 1
    adjacency_rows, left_edges, right_edges = np.nonzero(
        valid & unique[:, None, None]
    )
    if adjacency_rows.size == 0:
        return (
            np.full((chart_count, 0), -1, dtype=np.int64),
            np.zeros((chart_count, 0), dtype=np.int8),
            np.zeros((chart_count, 0), dtype=np.int8),
            np.zeros((chart_count, 0, 3), dtype=np.float64),
            np.zeros((chart_count, 0, 3), dtype=np.float64),
        )

    overlap_origins = left_segments[
        adjacency_rows,
        left_edges,
        0,
    ]
    overlap_axes = left_axis[adjacency_rows, left_edges]
    lower = overlap_lower[adjacency_rows, left_edges, right_edges]
    upper = overlap_upper[adjacency_rows, left_edges, right_edges]
    portal_start = overlap_origins + lower[:, None] * overlap_axes
    portal_end = overlap_origins + upper[:, None] * overlap_axes
    left_ids = adjacency[adjacency_rows, 0]
    right_ids = adjacency[adjacency_rows, 1]
    directed_from = np.concatenate((left_ids, right_ids))
    directed_to = np.concatenate((right_ids, left_ids))
    directed_source_side = np.concatenate(
        (
            _EDGE_INDEX_TO_SIDE[left_edges],
            _EDGE_INDEX_TO_SIDE[right_edges],
        )
    )
    directed_destination_side = np.concatenate(
        (
            _EDGE_INDEX_TO_SIDE[right_edges],
            _EDGE_INDEX_TO_SIDE[left_edges],
        )
    )
    directed_start = np.concatenate((portal_start, portal_start), axis=0)
    directed_end = np.concatenate((portal_end, portal_end), axis=0)
    order = np.lexsort(
        (
            directed_destination_side,
            directed_source_side,
            directed_to,
            directed_from,
        )
    )
    directed_from = directed_from[order]
    directed_to = directed_to[order]
    directed_source_side = directed_source_side[order]
    directed_destination_side = directed_destination_side[order]
    directed_start = directed_start[order]
    directed_end = directed_end[order]

    counts = np.bincount(directed_from, minlength=chart_count)
    maximum_count = int(np.max(counts, initial=0))
    group_starts = np.cumsum(counts) - counts
    slots = np.arange(directed_from.size, dtype=np.int64) - np.repeat(
        group_starts,
        counts,
    )
    neighbor_ids = np.full(
        (chart_count, maximum_count),
        -1,
        dtype=np.int64,
    )
    source_sides = np.zeros(
        (chart_count, maximum_count),
        dtype=np.int8,
    )
    destination_sides = np.zeros(
        (chart_count, maximum_count),
        dtype=np.int8,
    )
    starts = np.zeros(
        (chart_count, maximum_count, 3),
        dtype=np.float64,
    )
    ends = np.zeros_like(starts)
    neighbor_ids[directed_from, slots] = directed_to
    source_sides[directed_from, slots] = directed_source_side
    destination_sides[directed_from, slots] = directed_destination_side
    starts[directed_from, slots] = directed_start
    ends[directed_from, slots] = directed_end
    return neighbor_ids, source_sides, destination_sides, starts, ends


def _outward_tangent_axes(
    sides: NDArray[np.int8],
    u_axes_xyz: NDArray[np.float64],
    v_axes_xyz: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the outward in-surface unit axis for each rectangle side."""
    side_values = np.asarray(sides, dtype=np.int8).reshape(-1)
    u_axes = np.asarray(u_axes_xyz, dtype=np.float64).reshape(-1, 3)
    v_axes = np.asarray(v_axes_xyz, dtype=np.float64).reshape(-1, 3)
    return np.where(
        (side_values == _U_LOWER)[:, None],
        -u_axes,
        np.where(
            (side_values == _U_UPPER)[:, None],
            u_axes,
            np.where(
                (side_values == _V_LOWER)[:, None],
                -v_axes,
                v_axes,
            ),
        ),
    )


@dataclass
class ContinuousSurfaceAtlas:
    """Map continuous chart coordinates to the shared exposed surface geometry."""

    geometry: SurfaceChartGeometry
    _origins_xyz: NDArray[np.float64] = field(init=False, repr=False)
    _u_edges_xyz: NDArray[np.float64] = field(init=False, repr=False)
    _v_edges_xyz: NDArray[np.float64] = field(init=False, repr=False)
    _u_lengths_m: NDArray[np.float64] = field(init=False, repr=False)
    _v_lengths_m: NDArray[np.float64] = field(init=False, repr=False)
    _u_axes_xyz: NDArray[np.float64] = field(init=False, repr=False)
    _v_axes_xyz: NDArray[np.float64] = field(init=False, repr=False)
    _chart_probabilities: NDArray[np.float64] = field(init=False, repr=False)
    _log_chart_probabilities: NDArray[np.float64] = field(
        init=False,
        repr=False,
    )
    _portal_neighbor_ids: NDArray[np.int64] = field(
        init=False,
        repr=False,
    )
    _portal_source_sides: NDArray[np.int8] = field(
        init=False,
        repr=False,
    )
    _portal_destination_sides: NDArray[np.int8] = field(
        init=False,
        repr=False,
    )
    _portal_starts_xyz: NDArray[np.float64] = field(
        init=False,
        repr=False,
    )
    _portal_ends_xyz: NDArray[np.float64] = field(
        init=False,
        repr=False,
    )
    _chart_path_graph: csr_matrix = field(init=False, repr=False)
    _center_tree: cKDTree = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate chart geometry and precompute batched affine mappings."""
        vertices = np.asarray(self.geometry.vertices_xyz, dtype=np.float64)
        if vertices.ndim != 3 or vertices.shape[1:] != (4, 3):
            raise ValueError("Surface chart rectangles must be shaped C x 4 x 3.")
        origins = vertices[:, 0]
        u_edges = vertices[:, 1] - origins
        v_edges = vertices[:, 3] - origins
        u_lengths = np.linalg.norm(u_edges, axis=1)
        v_lengths = np.linalg.norm(v_edges, axis=1)
        areas = np.asarray(self.geometry.areas_m2, dtype=np.float64)
        total_area = float(np.sum(areas, dtype=np.float64))
        if (
            areas.size == 0
            or not np.isfinite(total_area)
            or total_area <= 0.0
            or np.any(~np.isfinite(u_lengths))
            or np.any(~np.isfinite(v_lengths))
            or np.any(u_lengths <= 0.0)
            or np.any(v_lengths <= 0.0)
        ):
            raise ValueError("Surface charts must have finite positive area.")
        u_axes = u_edges / u_lengths[:, None]
        v_axes = v_edges / v_lengths[:, None]
        if (
            np.any(
                np.abs(np.sum(u_axes * v_axes, axis=1))
                > _DIRECTION_TOLERANCE
            )
            or not np.allclose(
                areas,
                u_lengths * v_lengths,
                rtol=1.0e-10,
                atol=_SURFACE_TOLERANCE_M,
            )
        ):
            raise ValueError(
                "Continuous surface charts must be physical rectangles."
            )
        probabilities = areas / total_area
        (
            portal_neighbor_ids,
            portal_source_sides,
            portal_destination_sides,
            portal_starts_xyz,
            portal_ends_xyz,
        ) = _build_portal_table(
            vertices,
            self.geometry.adjacency_edges,
            tolerance_m=_SURFACE_TOLERANCE_M,
        )
        self._origins_xyz = origins
        self._u_edges_xyz = u_edges
        self._v_edges_xyz = v_edges
        self._u_lengths_m = u_lengths
        self._v_lengths_m = v_lengths
        self._u_axes_xyz = u_axes
        self._v_axes_xyz = v_axes
        self._chart_probabilities = probabilities
        self._log_chart_probabilities = np.log(probabilities)
        self._portal_neighbor_ids = portal_neighbor_ids
        self._portal_source_sides = portal_source_sides
        self._portal_destination_sides = portal_destination_sides
        self._portal_starts_xyz = portal_starts_xyz
        self._portal_ends_xyz = portal_ends_xyz
        chart_centers = np.asarray(
            self.geometry.centers_xyz,
            dtype=np.float64,
        )
        maximum_portals = int(portal_neighbor_ids.shape[1])
        if maximum_portals == 0:
            self._chart_path_graph = csr_matrix(
                (self.chart_count, self.chart_count),
                dtype=np.float64,
            )
        else:
            source_ids = np.repeat(
                np.arange(self.chart_count, dtype=np.int64),
                maximum_portals,
            )
            destination_ids = portal_neighbor_ids.reshape(-1)
            valid_portals = destination_ids >= 0
            source_ids = source_ids[valid_portals]
            destination_ids = destination_ids[valid_portals]
            portal_midpoints = 0.5 * (
                portal_starts_xyz.reshape(-1, 3)[valid_portals]
                + portal_ends_xyz.reshape(-1, 3)[valid_portals]
            )
            edge_lengths = (
                np.linalg.norm(
                    chart_centers[source_ids] - portal_midpoints,
                    axis=1,
                )
                + np.linalg.norm(
                    chart_centers[destination_ids] - portal_midpoints,
                    axis=1,
                )
            )
            edge_keys = source_ids * self.chart_count + destination_ids
            order = np.argsort(edge_keys, kind="stable")
            sorted_keys = edge_keys[order]
            group_starts = np.flatnonzero(
                np.concatenate(
                    (
                        np.asarray([True]),
                        sorted_keys[1:] != sorted_keys[:-1],
                    )
                )
            )
            unique_keys = sorted_keys[group_starts]
            minimum_lengths = np.minimum.reduceat(
                edge_lengths[order],
                group_starts,
            )
            self._chart_path_graph = csr_matrix(
                (
                    minimum_lengths,
                    (
                        unique_keys // self.chart_count,
                        unique_keys % self.chart_count,
                    ),
                ),
                shape=(self.chart_count, self.chart_count),
                dtype=np.float64,
            )
        self._center_tree = cKDTree(
            chart_centers
        )

    @property
    def chart_count(self) -> int:
        """Return the number of rectangular coordinate charts."""
        return int(self._chart_probabilities.size)

    @property
    def total_area_m2(self) -> float:
        """Return the total selectable physical surface area."""
        return float(
            np.sum(self.geometry.areas_m2, dtype=np.float64)
        )

    @property
    def chart_probabilities(self) -> NDArray[np.float64]:
        """Return area-weighted chart probabilities for unit-square coordinates."""
        return self._chart_probabilities

    @property
    def log_chart_probabilities(self) -> NDArray[np.float64]:
        """Return log position-prior density in chart/unit-square coordinates."""
        return self._log_chart_probabilities

    def validate_coordinates(
        self,
        chart_ids: ArrayLike,
        uv: ArrayLike,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Return validated broadcast-compatible chart identifiers and UV values."""
        raw_ids = np.asarray(chart_ids)
        if not np.issubdtype(raw_ids.dtype, np.integer):
            raise TypeError("surface chart identifiers must be integers.")
        ids = np.asarray(raw_ids, dtype=np.int64)
        coordinates = np.asarray(uv, dtype=np.float64)
        if coordinates.shape != ids.shape + (2,):
            raise ValueError("surface_uv must have shape chart_ids.shape + (2,).")
        if (
            np.any(ids < 0)
            or np.any(ids >= self.chart_count)
            or np.any(~np.isfinite(coordinates))
            or np.any(coordinates < 0.0)
            or np.any(coordinates > 1.0)
        ):
            raise ValueError(
                "Surface chart coordinates must be finite and inside their charts."
            )
        return ids, coordinates

    def positions_xyz(
        self,
        chart_ids: ArrayLike,
        uv: ArrayLike,
    ) -> NDArray[np.float64]:
        """Map a batch of chart/unit-square coordinates to continuous XYZ."""
        ids, coordinates = self.validate_coordinates(chart_ids, uv)
        flat_ids = ids.reshape(-1)
        flat_uv = coordinates.reshape(-1, 2)
        positions = (
            self._origins_xyz[flat_ids]
            + flat_uv[:, :1] * self._u_edges_xyz[flat_ids]
            + flat_uv[:, 1:] * self._v_edges_xyz[flat_ids]
        )
        return positions.reshape(ids.shape + (3,))

    def air_facing_normals_xyz(
        self,
        chart_ids: ArrayLike,
    ) -> NDArray[np.float64]:
        """Return semantic air-facing normals for authoritative chart IDs."""
        raw_ids = np.asarray(chart_ids)
        if not np.issubdtype(raw_ids.dtype, np.integer):
            raise TypeError("surface chart identifiers must be integers.")
        ids = np.asarray(raw_ids, dtype=np.int64)
        if np.any(ids < 0) or np.any(ids >= self.chart_count):
            raise ValueError("surface chart identifier is outside the atlas.")
        normals = np.asarray(
            self.geometry.normals_xyz,
            dtype=np.float64,
        )[ids.reshape(-1)]
        return normals.reshape(ids.shape + (3,))

    def sample(
        self,
        sample_count: int,
        *,
        rng: np.random.Generator,
        chart_probabilities: ArrayLike | None = None,
    ) -> tuple[
        NDArray[np.int64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Draw continuous chart coordinates in one batched RNG operation."""
        if isinstance(sample_count, bool) or not isinstance(
            sample_count,
            (int, np.integer),
        ):
            raise TypeError("sample_count must be an integer.")
        count = int(sample_count)
        if count < 0:
            raise ValueError("sample_count must be non-negative.")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator.")
        if chart_probabilities is None:
            probabilities = self._chart_probabilities
        else:
            probabilities = np.asarray(
                chart_probabilities,
                dtype=np.float64,
            ).reshape(-1)
            total = float(np.sum(probabilities, dtype=np.float64))
            if (
                probabilities.size != self.chart_count
                or np.any(~np.isfinite(probabilities))
                or np.any(probabilities <= 0.0)
                or not np.isfinite(total)
                or total <= 0.0
            ):
                raise ValueError(
                    "chart_probabilities must contain positive finite mass "
                    "for every surface chart."
                )
            probabilities = probabilities / total
        if count == 0:
            return (
                np.zeros(0, dtype=np.int64),
                np.zeros((0, 2), dtype=np.float64),
                np.zeros((0, 3), dtype=np.float64),
            )
        chart_ids = np.asarray(
            rng.choice(self.chart_count, size=count, p=probabilities),
            dtype=np.int64,
        )
        uv = np.asarray(rng.random((count, 2)), dtype=np.float64)
        return chart_ids, uv, self.positions_xyz(chart_ids, uv)

    def trace_tangent_displacements(
        self,
        chart_ids: ArrayLike,
        uv: ArrayLike,
        displacement_m: ArrayLike,
        *,
        max_crossings: int = 128,
    ) -> tuple[
        NDArray[np.int64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.bool_],
        NDArray[np.int64],
    ]:
        """Trace batched tangent displacements through shared-edge portals.

        The displacement components are physical metres in the starting
        chart's orthonormal U/V frame.  At a shared edge, the tangent vector is
        unfolded isometrically into the neighboring chart.  A move that reaches
        a vertex, a missing portal, an ambiguous/non-manifold portal, or the
        crossing limit becomes a self-transition; the original draw is never
        resampled.

        Returns
        -------
        proposed_chart_ids, proposed_uv, reverse_displacement_m, valid, crossings:
            The traced endpoint, the exact reverse displacement expressed in
            its endpoint chart, a validity mask, and successful portal counts.
            Invalid rows contain their original state and zero crossing count.
        """
        ids, coordinates = self.validate_coordinates(chart_ids, uv)
        displacements = np.asarray(displacement_m, dtype=np.float64)
        if displacements.shape != coordinates.shape:
            raise ValueError(
                "displacement_m must have shape chart_ids.shape + (2,)."
            )
        if np.any(~np.isfinite(displacements)):
            raise ValueError("displacement_m must contain finite values.")
        if isinstance(max_crossings, bool) or not isinstance(
            max_crossings,
            (int, np.integer),
        ):
            raise TypeError("max_crossings must be an integer.")
        crossing_limit = int(max_crossings)
        if crossing_limit < 0:
            raise ValueError("max_crossings must be non-negative.")

        original_ids = ids.reshape(-1).copy()
        original_uv = coordinates.reshape(-1, 2).copy()
        original_displacements = displacements.reshape(-1, 2)
        row_count = int(original_ids.size)
        if row_count == 0:
            return (
                original_ids.reshape(ids.shape),
                original_uv.reshape(coordinates.shape),
                original_displacements.reshape(displacements.shape),
                np.ones(ids.shape, dtype=bool),
                np.zeros(ids.shape, dtype=np.int64),
            )

        displacement_lengths = np.linalg.norm(
            original_displacements,
            axis=1,
        )
        start_u_axes = self._u_axes_xyz[original_ids]
        start_v_axes = self._v_axes_xyz[original_ids]
        displacement_xyz = (
            original_displacements[:, :1] * start_u_axes
            + original_displacements[:, 1:] * start_v_axes
        )
        directions_xyz = np.zeros((row_count, 3), dtype=np.float64)
        moving = displacement_lengths > 0.0
        directions_xyz[moving] = (
            displacement_xyz[moving]
            / displacement_lengths[moving, None]
        )
        current_ids = original_ids.copy()
        current_uv = original_uv.copy()
        remaining_m = displacement_lengths.copy()
        active = moving.copy()
        valid = np.ones(row_count, dtype=bool)
        crossing_counts = np.zeros(row_count, dtype=np.int64)

        for _ in range(crossing_limit + 1):
            if not np.any(active):
                break
            active_rows = np.flatnonzero(active)
            active_ids = current_ids[active_rows]
            active_uv = current_uv[active_rows]
            active_directions = directions_xyz[active_rows]
            active_remaining = remaining_m[active_rows]
            u_axes = self._u_axes_xyz[active_ids]
            v_axes = self._v_axes_xyz[active_ids]
            u_rates = (
                np.sum(active_directions * u_axes, axis=1)
                / self._u_lengths_m[active_ids]
            )
            v_rates = (
                np.sum(active_directions * v_axes, axis=1)
                / self._v_lengths_m[active_ids]
            )
            u_boundary_distance = np.full(
                active_rows.size,
                np.inf,
                dtype=np.float64,
            )
            v_boundary_distance = np.full_like(u_boundary_distance, np.inf)
            positive_u = u_rates > _DIRECTION_TOLERANCE
            negative_u = u_rates < -_DIRECTION_TOLERANCE
            positive_v = v_rates > _DIRECTION_TOLERANCE
            negative_v = v_rates < -_DIRECTION_TOLERANCE
            u_boundary_distance[positive_u] = (
                (1.0 - active_uv[positive_u, 0]) / u_rates[positive_u]
            )
            u_boundary_distance[negative_u] = (
                -active_uv[negative_u, 0] / u_rates[negative_u]
            )
            v_boundary_distance[positive_v] = (
                (1.0 - active_uv[positive_v, 1]) / v_rates[positive_v]
            )
            v_boundary_distance[negative_v] = (
                -active_uv[negative_v, 1] / v_rates[negative_v]
            )
            u_boundary_distance = np.maximum(u_boundary_distance, 0.0)
            v_boundary_distance = np.maximum(v_boundary_distance, 0.0)
            boundary_distance = np.minimum(
                u_boundary_distance,
                v_boundary_distance,
            )

            finishes_inside = (
                active_remaining
                < boundary_distance - _SURFACE_TOLERANCE_M
            )
            if np.any(finishes_inside):
                rows = active_rows[finishes_inside]
                ending_uv = active_uv[finishes_inside].copy()
                ending_uv[:, 0] += (
                    active_remaining[finishes_inside]
                    * u_rates[finishes_inside]
                )
                ending_uv[:, 1] += (
                    active_remaining[finishes_inside]
                    * v_rates[finishes_inside]
                )
                current_uv[rows] = np.clip(ending_uv, 0.0, 1.0)
                remaining_m[rows] = 0.0
                active[rows] = False

            ends_on_boundary = (
                np.abs(active_remaining - boundary_distance)
                <= _SURFACE_TOLERANCE_M
            )
            invalid_direction = ~np.isfinite(boundary_distance)
            invalid_now = (
                (~finishes_inside)
                & (ends_on_boundary | invalid_direction)
            )
            if np.any(invalid_now):
                rows = active_rows[invalid_now]
                valid[rows] = False
                active[rows] = False

            crosses = (
                (~finishes_inside)
                & (~invalid_now)
                & (
                    active_remaining
                    > boundary_distance + _SURFACE_TOLERANCE_M
                )
            )
            unclassified = ~(finishes_inside | invalid_now | crosses)
            if np.any(unclassified):
                rows = active_rows[unclassified]
                valid[rows] = False
                active[rows] = False
            if not np.any(crosses):
                continue

            crossing_rows = active_rows[crosses]
            crossing_u_distance = u_boundary_distance[crosses]
            crossing_v_distance = v_boundary_distance[crosses]
            crossing_distance = boundary_distance[crosses]
            crossing_u_rates = u_rates[crosses]
            crossing_v_rates = v_rates[crosses]
            vertex_hit = (
                np.abs(crossing_u_distance - crossing_v_distance)
                <= _SURFACE_TOLERANCE_M
            )
            exceeds_limit = crossing_counts[crossing_rows] >= crossing_limit
            rejected_crossing = vertex_hit | exceeds_limit
            if np.any(rejected_crossing):
                rows = crossing_rows[rejected_crossing]
                valid[rows] = False
                active[rows] = False

            portal_candidates = ~rejected_crossing
            if not np.any(portal_candidates):
                continue
            portal_rows = crossing_rows[portal_candidates]
            portal_ids = current_ids[portal_rows]
            portal_directions = directions_xyz[portal_rows]
            portal_uv = current_uv[portal_rows].copy()
            portal_distance = crossing_distance[portal_candidates]
            portal_u_rates = crossing_u_rates[portal_candidates]
            portal_v_rates = crossing_v_rates[portal_candidates]
            crosses_u = (
                crossing_u_distance[portal_candidates]
                < crossing_v_distance[portal_candidates]
            )
            source_sides = np.where(
                crosses_u,
                np.where(
                    portal_u_rates > 0.0,
                    _U_UPPER,
                    _U_LOWER,
                ),
                np.where(
                    portal_v_rates > 0.0,
                    _V_UPPER,
                    _V_LOWER,
                ),
            ).astype(np.int8)
            portal_uv[:, 0] += portal_distance * portal_u_rates
            portal_uv[:, 1] += portal_distance * portal_v_rates
            portal_uv[crosses_u, 0] = np.where(
                source_sides[crosses_u] == _U_UPPER,
                1.0,
                0.0,
            )
            portal_uv[~crosses_u, 1] = np.where(
                source_sides[~crosses_u] == _V_UPPER,
                1.0,
                0.0,
            )
            crossing_xyz = (
                self._origins_xyz[portal_ids]
                + portal_uv[:, :1] * self._u_edges_xyz[portal_ids]
                + portal_uv[:, 1:] * self._v_edges_xyz[portal_ids]
            )

            if self._portal_neighbor_ids.shape[1] == 0:
                valid[portal_rows] = False
                active[portal_rows] = False
                continue
            candidate_neighbors = self._portal_neighbor_ids[portal_ids]
            candidate_source_sides = self._portal_source_sides[portal_ids]
            candidate_starts = self._portal_starts_xyz[portal_ids]
            candidate_ends = self._portal_ends_xyz[portal_ids]
            candidate_segments = candidate_ends - candidate_starts
            candidate_length_sq = np.sum(
                candidate_segments * candidate_segments,
                axis=2,
            )
            relative_points = crossing_xyz[:, None, :] - candidate_starts
            along = np.divide(
                np.sum(relative_points * candidate_segments, axis=2),
                candidate_length_sq,
                out=np.zeros_like(candidate_length_sq),
                where=candidate_length_sq > 0.0,
            )
            closest = candidate_starts + along[:, :, None] * candidate_segments
            line_distance = np.linalg.norm(
                crossing_xyz[:, None, :] - closest,
                axis=2,
            )
            portal_lengths = np.sqrt(candidate_length_sq)
            endpoint_margin = np.divide(
                _SURFACE_TOLERANCE_M,
                portal_lengths,
                out=np.full_like(portal_lengths, np.inf),
                where=portal_lengths > 0.0,
            )
            matches = (
                (candidate_neighbors >= 0)
                & (candidate_source_sides == source_sides[:, None])
                & (line_distance <= _SURFACE_TOLERANCE_M)
                & (along > endpoint_margin)
                & (along < 1.0 - endpoint_margin)
            )
            match_counts = np.sum(matches, axis=1)
            unique_match = match_counts == 1
            if np.any(~unique_match):
                rows = portal_rows[~unique_match]
                valid[rows] = False
                active[rows] = False
            if not np.any(unique_match):
                continue

            traversing_rows = portal_rows[unique_match]
            match_slots = np.argmax(matches[unique_match], axis=1)
            source_ids = current_ids[traversing_rows]
            destination_ids = candidate_neighbors[
                unique_match,
                match_slots,
            ]
            destination_sides = self._portal_destination_sides[source_ids][
                np.arange(traversing_rows.size, dtype=np.int64),
                match_slots,
            ]
            selected_starts = candidate_starts[unique_match, match_slots]
            selected_ends = candidate_ends[unique_match, match_slots]
            edge_axes = selected_ends - selected_starts
            edge_axes /= np.linalg.norm(edge_axes, axis=1)[:, None]
            source_outward = _outward_tangent_axes(
                source_sides[unique_match],
                self._u_axes_xyz[source_ids],
                self._v_axes_xyz[source_ids],
            )
            destination_outward = _outward_tangent_axes(
                destination_sides,
                self._u_axes_xyz[destination_ids],
                self._v_axes_xyz[destination_ids],
            )
            directions = portal_directions[unique_match]
            along_edge = np.sum(directions * edge_axes, axis=1)
            across_edge = np.sum(directions * source_outward, axis=1)
            transported = (
                along_edge[:, None] * edge_axes
                - across_edge[:, None] * destination_outward
            )
            transported_norm = np.linalg.norm(transported, axis=1)
            transport_valid = (
                (across_edge > _DIRECTION_TOLERANCE)
                & np.isfinite(transported_norm)
                & (transported_norm > _DIRECTION_TOLERANCE)
            )
            if np.any(~transport_valid):
                rows = traversing_rows[~transport_valid]
                valid[rows] = False
                active[rows] = False
            if not np.any(transport_valid):
                continue

            accepted_rows = traversing_rows[transport_valid]
            accepted_destination_ids = destination_ids[transport_valid]
            accepted_xyz = crossing_xyz[unique_match][transport_valid]
            accepted_delta = (
                accepted_xyz
                - self._origins_xyz[accepted_destination_ids]
            )
            accepted_uv = np.column_stack(
                (
                    np.sum(
                        accepted_delta
                        * self._u_edges_xyz[accepted_destination_ids],
                        axis=1,
                    )
                    / self._u_lengths_m[accepted_destination_ids] ** 2,
                    np.sum(
                        accepted_delta
                        * self._v_edges_xyz[accepted_destination_ids],
                        axis=1,
                    )
                    / self._v_lengths_m[accepted_destination_ids] ** 2,
                )
            )
            accepted_destination_sides = destination_sides[transport_valid]
            accepted_uv[
                accepted_destination_sides == _U_LOWER,
                0,
            ] = 0.0
            accepted_uv[
                accepted_destination_sides == _U_UPPER,
                0,
            ] = 1.0
            accepted_uv[
                accepted_destination_sides == _V_LOWER,
                1,
            ] = 0.0
            accepted_uv[
                accepted_destination_sides == _V_UPPER,
                1,
            ] = 1.0
            current_ids[accepted_rows] = accepted_destination_ids
            current_uv[accepted_rows] = np.clip(accepted_uv, 0.0, 1.0)
            directions_xyz[accepted_rows] = (
                transported[transport_valid]
                / transported_norm[transport_valid, None]
            )
            remaining_m[accepted_rows] -= portal_distance[unique_match][
                transport_valid
            ]
            crossing_counts[accepted_rows] += 1

        if np.any(active):
            valid[active] = False
            active[active] = False

        proposed_ids = np.where(valid, current_ids, original_ids)
        proposed_uv = np.where(
            valid[:, None],
            current_uv,
            original_uv,
        )
        endpoint_u_axes = self._u_axes_xyz[proposed_ids]
        endpoint_v_axes = self._v_axes_xyz[proposed_ids]
        reverse_xyz = (
            -directions_xyz * displacement_lengths[:, None]
        )
        reverse_displacements = np.column_stack(
            (
                np.sum(reverse_xyz * endpoint_u_axes, axis=1),
                np.sum(reverse_xyz * endpoint_v_axes, axis=1),
            )
        )
        reverse_displacements[~valid] = -original_displacements[~valid]
        crossing_counts = np.where(valid, crossing_counts, 0)
        return (
            proposed_ids.reshape(ids.shape),
            proposed_uv.reshape(coordinates.shape),
            reverse_displacements.reshape(displacements.shape),
            valid.reshape(ids.shape),
            crossing_counts.reshape(ids.shape),
        )

    def tangent_geodesic_portal_proposal(
        self,
        chart_ids: ArrayLike,
        uv: ArrayLike,
        *,
        sigma_m: float,
        rng: np.random.Generator,
        max_crossings: int = 128,
    ) -> tuple[
        NDArray[np.int64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Draw one symmetric Gaussian move and trace it across chart portals.

        The portal unfolding is a piecewise physical isometry with absolute
        physical-area Jacobian one.  State coordinates use normalized U/V,
        however, so their coordinate density gains the destination chart area.
        The returned log reverse/forward proposal-density ratio is therefore
        ``log(A_source / A_destination)``.  This exactly cancels the uniform
        physical-surface prior's normalized-chart mass ratio.  Invalid traces
        are explicit self-transitions and consume no replacement random draws.
        """
        ids, coordinates = self.validate_coordinates(chart_ids, uv)
        sigma = float(sigma_m)
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma_m must be finite and positive.")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator.")
        displacement_m = np.asarray(
            rng.normal(0.0, sigma, size=(ids.size, 2)),
            dtype=np.float64,
        ).reshape(coordinates.shape)
        proposed_ids, proposed_uv, _, _, _ = (
            self.trace_tangent_displacements(
                ids,
                coordinates,
                displacement_m,
                max_crossings=max_crossings,
            )
        )
        return (
            proposed_ids,
            proposed_uv,
            (
                self._log_chart_probabilities[ids]
                - self._log_chart_probabilities[proposed_ids]
            ),
        )

    def surface_path_distance_upper_bound_m(
        self,
        first_positions_xyz: ArrayLike,
        second_positions_xyz: ArrayLike,
    ) -> NDArray[np.float64]:
        """Return fail-closed upper bounds on paths along connected surfaces.

        A same-chart path is the exact in-chart straight line.  A path across
        one shared portal minimizes the two straight chart segments over that
        portal.  Longer paths use chart centers and shared-portal midpoints,
        which is a realizable surface path and therefore an upper bound on the
        intrinsic geodesic distance.  Points on disconnected surface
        components return positive infinity instead of being compared through
        free space.
        """
        first = np.asarray(first_positions_xyz, dtype=np.float64)
        second = np.asarray(second_positions_xyz, dtype=np.float64)
        if first.shape[-1:] != (3,) or second.shape[-1:] != (3,):
            raise ValueError("Surface path positions must end in XYZ coordinates.")
        first, second = np.broadcast_arrays(first, second)
        if first.reshape(-1, 3).shape[0] == 0:
            return np.zeros(first.shape[:-1], dtype=np.float64)
        first_ids, first_uv = self.locate_positions(first)
        second_ids, second_uv = self.locate_positions(second)
        return self.surface_coordinate_path_distance_upper_bound_m(
            first_ids,
            first_uv,
            second_ids,
            second_uv,
        )

    def surface_coordinate_path_distance_upper_bound_m(
        self,
        first_chart_ids: ArrayLike,
        first_uv: ArrayLike,
        second_chart_ids: ArrayLike,
        second_uv: ArrayLike,
    ) -> NDArray[np.float64]:
        """Return intrinsic path bounds from authoritative chart coordinates.

        Unlike the XYZ convenience wrapper, this method never has to infer a
        chart at a shared edge or at two geometrically close faces.  Posterior
        association and evaluation should therefore use this method whenever
        the continuous PF chart/UV state is available.
        """
        raw_first_ids = np.asarray(first_chart_ids)
        raw_second_ids = np.asarray(second_chart_ids)
        if not np.issubdtype(raw_first_ids.dtype, np.integer):
            raise TypeError("first_chart_ids must contain integers.")
        if not np.issubdtype(raw_second_ids.dtype, np.integer):
            raise TypeError("second_chart_ids must contain integers.")
        first_ids, second_ids = np.broadcast_arrays(
            raw_first_ids.astype(np.int64, copy=False),
            raw_second_ids.astype(np.int64, copy=False),
        )
        first_coordinates = np.asarray(first_uv, dtype=np.float64)
        second_coordinates = np.asarray(second_uv, dtype=np.float64)
        expected_first_shape = np.asarray(first_chart_ids).shape + (2,)
        expected_second_shape = np.asarray(second_chart_ids).shape + (2,)
        if first_coordinates.shape != expected_first_shape:
            raise ValueError(
                "first_uv must have shape first_chart_ids.shape + (2,)."
            )
        if second_coordinates.shape != expected_second_shape:
            raise ValueError(
                "second_uv must have shape second_chart_ids.shape + (2,)."
            )
        first_coordinates = np.broadcast_to(
            first_coordinates,
            first_ids.shape + (2,),
        )
        second_coordinates = np.broadcast_to(
            second_coordinates,
            second_ids.shape + (2,),
        )
        first_ids, first_coordinates = self.validate_coordinates(
            first_ids,
            first_coordinates,
        )
        second_ids, second_coordinates = self.validate_coordinates(
            second_ids,
            second_coordinates,
        )
        first = self.positions_xyz(first_ids, first_coordinates)
        second = self.positions_xyz(second_ids, second_coordinates)
        return self._surface_path_distance_from_coordinates(
            first,
            first_ids,
            second,
            second_ids,
        )

    def _surface_path_distance_from_coordinates(
        self,
        first_positions_xyz: NDArray[np.float64],
        first_chart_ids: NDArray[np.int64],
        second_positions_xyz: NDArray[np.float64],
        second_chart_ids: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Return path bounds for validated, broadcast chart-coordinate rows."""
        first = np.asarray(first_positions_xyz, dtype=np.float64)
        second = np.asarray(second_positions_xyz, dtype=np.float64)
        first_ids = np.asarray(first_chart_ids, dtype=np.int64)
        second_ids = np.asarray(second_chart_ids, dtype=np.int64)
        if (
            first.shape != first_ids.shape + (3,)
            or second.shape != second_ids.shape + (3,)
            or first.shape != second.shape
            or first_ids.shape != second_ids.shape
        ):
            raise ValueError(
                "Surface path positions and chart identifiers must align."
            )
        flat_first = first.reshape(-1, 3)
        flat_second = second.reshape(-1, 3)
        flat_first_ids = first_ids.reshape(-1)
        flat_second_ids = second_ids.reshape(-1)
        if flat_first.shape[0] == 0:
            return np.zeros(first_ids.shape, dtype=np.float64)
        direct_distance = np.linalg.norm(flat_first - flat_second, axis=1)
        result = np.full(flat_first.shape[0], np.inf, dtype=np.float64)
        same_chart = flat_first_ids == flat_second_ids
        result[same_chart] = direct_distance[same_chart]

        portal_slots = self._portal_neighbor_ids[flat_first_ids]
        portal_matches = portal_slots == flat_second_ids[:, None]
        if portal_matches.shape[1] > 0 and np.any(portal_matches):
            pair_indices, portal_indices = np.nonzero(portal_matches)
            portal_starts = self._portal_starts_xyz[
                flat_first_ids[pair_indices],
                portal_indices,
            ]
            portal_deltas = (
                self._portal_ends_xyz[
                    flat_first_ids[pair_indices],
                    portal_indices,
                ]
                - portal_starts
            )
            lower = np.zeros(pair_indices.size, dtype=np.float64)
            upper = np.ones(pair_indices.size, dtype=np.float64)
            first_expanded = flat_first[pair_indices]
            second_expanded = flat_second[pair_indices]
            for _ in range(48):
                first_fraction = (2.0 * lower + upper) / 3.0
                second_fraction = (lower + 2.0 * upper) / 3.0
                first_portal_points = (
                    portal_starts
                    + first_fraction[:, None] * portal_deltas
                )
                second_portal_points = (
                    portal_starts
                    + second_fraction[:, None] * portal_deltas
                )
                first_cost = np.linalg.norm(
                    first_expanded - first_portal_points,
                    axis=1,
                ) + np.linalg.norm(
                    second_expanded - first_portal_points,
                    axis=1,
                )
                second_cost = np.linalg.norm(
                    first_expanded - second_portal_points,
                    axis=1,
                ) + np.linalg.norm(
                    second_expanded - second_portal_points,
                    axis=1,
                )
                choose_left = first_cost <= second_cost
                upper = np.where(choose_left, second_fraction, upper)
                lower = np.where(choose_left, lower, first_fraction)
            optimum_fraction = 0.5 * (lower + upper)
            optimum_points = (
                portal_starts
                + optimum_fraction[:, None] * portal_deltas
            )
            portal_costs = np.linalg.norm(
                first_expanded - optimum_points,
                axis=1,
            ) + np.linalg.norm(
                second_expanded - optimum_points,
                axis=1,
            )
            best_portal_cost = np.full(
                flat_first.shape[0],
                np.inf,
                dtype=np.float64,
            )
            np.minimum.at(
                best_portal_cost,
                pair_indices,
                portal_costs,
            )
            result = np.minimum(result, best_portal_cost)

        unresolved = ~np.isfinite(result)
        if np.any(unresolved):
            source_ids = flat_first_ids[unresolved]
            destination_ids = flat_second_ids[unresolved]
            unique_sources, source_inverse = np.unique(
                source_ids,
                return_inverse=True,
            )
            graph_distances = np.asarray(
                dijkstra(
                    self._chart_path_graph,
                    directed=True,
                    indices=unique_sources,
                    return_predecessors=False,
                ),
                dtype=np.float64,
            )
            if graph_distances.ndim == 1:
                graph_distances = graph_distances[None, :]
            center_distances = graph_distances[
                source_inverse,
                destination_ids,
            ]
            chart_centers = np.asarray(
                self.geometry.centers_xyz,
                dtype=np.float64,
            )
            endpoint_costs = (
                np.linalg.norm(
                    flat_first[unresolved] - chart_centers[source_ids],
                    axis=1,
                )
                + np.linalg.norm(
                    flat_second[unresolved] - chart_centers[destination_ids],
                    axis=1,
                )
            )
            result[unresolved] = center_distances + endpoint_costs
        return result.reshape(first_ids.shape)

    def canonical_order(
        self,
        chart_ids: ArrayLike,
        uv: ArrayLike,
    ) -> NDArray[np.int64]:
        """Return deterministic source order by chart, U, then V."""
        ids, coordinates = self.validate_coordinates(chart_ids, uv)
        if ids.ndim != 1:
            raise ValueError("canonical_order expects one source-state row.")
        return np.lexsort((coordinates[:, 1], coordinates[:, 0], ids))

    def locate_positions(
        self,
        positions_xyz: ArrayLike,
        *,
        tolerance_m: float = _SURFACE_TOLERANCE_M,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Resolve known surface XYZ points to deterministic chart coordinates.

        PF-created states already carry chart coordinates.  This resolver exists
        for replay/test states and fails fast if a point is not on the shared
        physical surface instead of projecting an invalid position silently.
        """
        raw_positions = np.asarray(positions_xyz)
        if raw_positions.dtype.kind not in {"f", "i", "u"}:
            raise ValueError(
                "positions_xyz must contain real numeric values without "
                "boolean, string, object, or complex coercion."
            )
        positions = raw_positions.astype(np.float64, copy=False)
        if positions.shape[-1:] != (3,):
            raise ValueError("positions_xyz must have final dimension 3.")
        if np.any(~np.isfinite(positions)):
            raise ValueError("positions_xyz must contain finite values.")
        if (
            isinstance(tolerance_m, (bool, np.bool_))
            or not isinstance(tolerance_m, Real)
        ):
            raise ValueError("tolerance_m must be a real number.")
        tolerance = float(tolerance_m)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance_m must be finite and non-negative.")
        flat = positions.reshape(-1, 3)
        if flat.shape[0] == 0:
            return (
                np.zeros(positions.shape[:-1], dtype=np.int64),
                np.zeros(positions.shape[:-1] + (2,), dtype=np.float64),
            )
        chart_centers = np.asarray(
            self.geometry.centers_xyz,
            dtype=np.float64,
        )
        chart_vertices = np.asarray(
            self.geometry.vertices_xyz,
            dtype=np.float64,
        )
        maximum_chart_radius = float(
            np.max(
                np.linalg.norm(
                    chart_vertices - chart_centers[:, None, :],
                    axis=2,
                )
            )
        )
        search_radius = (
            maximum_chart_radius
            + np.sqrt(3.0) * tolerance
            + np.finfo(np.float64).eps
        )
        chosen_ids = np.empty(flat.shape[0], dtype=np.int64)
        chosen_uv = np.empty((flat.shape[0], 2), dtype=np.float64)
        # Candidate discovery and all rectangle tests are vectorized within
        # bounded point batches. The radius is a geometric guarantee, unlike a
        # fixed nearest-neighbor count, while batching prevents dense
        # multi-surface scenes from allocating a P-by-C temporary array.
        for start in range(0, flat.shape[0], _LOCATE_POINT_BATCH_SIZE):
            stop = min(start + _LOCATE_POINT_BATCH_SIZE, flat.shape[0])
            batch = flat[start:stop]
            candidate_counts = np.asarray(
                self._center_tree.query_ball_point(
                    batch,
                    r=search_radius,
                    return_length=True,
                ),
                dtype=np.int64,
            ).reshape(-1)
            neighbor_count = min(
                max(int(np.max(candidate_counts, initial=0)), 1),
                self.chart_count,
            )
            _, candidates = self._center_tree.query(
                batch,
                k=neighbor_count,
            )
            candidate_ids = np.asarray(candidates, dtype=np.int64)
            if candidate_ids.ndim == 1:
                candidate_ids = candidate_ids[:, None]
            origins = self._origins_xyz[candidate_ids]
            delta = batch[:, None, :] - origins
            u_edges = self._u_edges_xyz[candidate_ids]
            v_edges = self._v_edges_xyz[candidate_ids]
            u_norm_sq = np.sum(u_edges * u_edges, axis=2)
            v_norm_sq = np.sum(v_edges * v_edges, axis=2)
            u = np.sum(delta * u_edges, axis=2) / u_norm_sq
            v = np.sum(delta * v_edges, axis=2) / v_norm_sq
            reconstructed = (
                origins
                + u[:, :, None] * u_edges
                + v[:, :, None] * v_edges
            )
            residual = np.linalg.norm(
                reconstructed - batch[:, None, :],
                axis=2,
            )
            u_margin = tolerance / np.sqrt(u_norm_sq)
            v_margin = tolerance / np.sqrt(v_norm_sq)
            valid = (
                (residual <= tolerance)
                & (u >= -u_margin)
                & (u <= 1.0 + u_margin)
                & (v >= -v_margin)
                & (v <= 1.0 + v_margin)
            )
            if np.any(~np.any(valid, axis=1)):
                bad = np.flatnonzero(~np.any(valid, axis=1))
                raise ValueError(
                    "A source position is outside the continuous physical "
                    "surface atlas (first invalid row "
                    f"{int(start + bad[0])})."
                )
            valid_ids = np.where(valid, candidate_ids, self.chart_count)
            chosen_slot = np.argmin(valid_ids, axis=1)
            rows = np.arange(batch.shape[0], dtype=np.int64)
            chosen_ids[start:stop] = candidate_ids[rows, chosen_slot]
            chosen_uv[start:stop] = np.column_stack(
                (
                    np.clip(u[rows, chosen_slot], 0.0, 1.0),
                    np.clip(v[rows, chosen_slot], 0.0, 1.0),
                )
            )
        return (
            chosen_ids.reshape(positions.shape[:-1]),
            chosen_uv.reshape(positions.shape[:-1] + (2,)),
        )
