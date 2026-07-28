"""Surface-constrained source placement utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from measurement.model import EnvironmentConfig, PointSource
from measurement.obstacles import ObstacleGrid
from measurement.source_boundary import (
    surface_emission_policy_sha256,
    surface_transport_positions,
    validate_air_facing_surface_normals,
)
from measurement.surface_charts import (
    SurfaceChartGeometry,
    build_surface_chart_geometry,
    sample_continuous_surface_coordinates,
)

SourceSurfaceKind = Literal[
    "floor",
    "ceiling",
    "wall",
    "obstacle_side",
    "obstacle_top",
    "obstacle_bottom",
]

SOURCE_SURFACE_KINDS: tuple[SourceSurfaceKind, ...] = (
    "floor",
    "ceiling",
    "wall",
    "obstacle_side",
    "obstacle_top",
    "obstacle_bottom",
)
SOURCE_SURFACE_REPORT_LABELS = (*SOURCE_SURFACE_KINDS, "off_surface")
REMOVED_SOURCE_SELECTION_CONFIG_KEYS = frozenset(
    {
        "random_source_clear_path_max_m",
        "random_source_max_ceiling_sources",
        "random_source_min_visible_fraction",
        "random_source_preferred_max_z_m",
        "random_source_response_condition_max",
        "random_source_response_max_pairwise_corr",
        "random_source_response_max_set_attempts",
        "random_source_response_observability_filter",
        "random_source_same_isotope_min_distance_m",
        "random_source_visibility_batch_size",
        "random_source_visibility_filter",
        "random_source_visibility_max_attempts_per_source",
    }
)


def _nonnegative_finite_tolerance(value: object) -> float:
    """Return a strict finite nonnegative geometric tolerance."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("tolerance_m must be a real number.")
    tolerance = float(value)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance_m must be finite and nonnegative.")
    return tolerance


def validate_area_uniform_source_config(config: Mapping[str, object]) -> str:
    """Validate the non-configurable continuous truth-position contract."""
    removed = sorted(REMOVED_SOURCE_SELECTION_CONFIG_KEYS.intersection(config))
    if removed:
        raise ValueError(
            "Source truth selection/conditioning options were removed; random "
            "truth positions are continuous area-uniform over all eligible "
            f"physical surfaces. Remove these keys: {removed}."
        )
    raw_measure = config.get(
        "random_source_surface_sampling_measure",
        "continuous_area_uniform",
    )
    if not isinstance(raw_measure, str):
        raise TypeError(
            "random_source_surface_sampling_measure must be a JSON string."
        )
    sampling_measure = raw_measure
    if sampling_measure != "continuous_area_uniform":
        raise ValueError(
            "random_source_surface_sampling_measure must be "
            "'continuous_area_uniform'."
        )
    return sampling_measure


def transport_interior_mask(
    positions: NDArray[np.float64],
    obstacle_grid: ObstacleGrid | None,
    *,
    tolerance_m: float = 1.0e-6,
) -> NDArray[np.bool_]:
    """Return True for positions strictly inside known obstacle transport boxes."""
    points = np.asarray(positions, dtype=float).reshape(-1, 3)
    if obstacle_grid is None or not getattr(obstacle_grid, "has_transport_model", False):
        return np.zeros(points.shape[0], dtype=bool)
    boxes = np.asarray(obstacle_grid.transport_boxes(), dtype=float).reshape(-1, 6)
    if boxes.size == 0:
        return np.zeros(points.shape[0], dtype=bool)
    tol = _nonnegative_finite_tolerance(tolerance_m)
    lower = boxes[:, :3] + tol
    upper = boxes[:, 3:] - tol
    valid_boxes = np.all(upper > lower, axis=1)
    if not np.any(valid_boxes):
        return np.zeros(points.shape[0], dtype=bool)
    inside_lower = points[:, None, :] > lower[None, valid_boxes, :]
    inside_upper = points[:, None, :] < upper[None, valid_boxes, :]
    return np.any(np.all(inside_lower & inside_upper, axis=2), axis=1)


def _clipped_source_transport_boxes(
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid,
) -> NDArray[np.float64]:
    """Return positive-volume transport components clipped inside the room."""
    boxes = np.asarray(obstacle_grid.transport_boxes_m, dtype=float).reshape(-1, 6)
    if boxes.size == 0:
        return boxes
    upper_room = np.asarray(
        [float(env.size_x), float(env.size_y), float(env.size_z)],
        dtype=float,
    )
    lower = np.maximum(boxes[:, :3], 0.0)
    upper = np.minimum(boxes[:, 3:], upper_room[None, :])
    keep = np.all(upper - lower > 1.0e-9, axis=1)
    return np.column_stack([lower[keep], upper[keep]])


def _points_inside_transport_box_union(
    points_xyz: NDArray[np.float64],
    boxes_m: NDArray[np.float64],
    *,
    tolerance_m: float,
) -> NDArray[np.bool_]:
    """Return strict interior membership for a clipped transport-box union."""
    points = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    boxes = np.asarray(boxes_m, dtype=float).reshape(-1, 6)
    inside_any = np.zeros(points.shape[0], dtype=bool)
    if boxes.size == 0 or points.size == 0:
        return inside_any
    tol = _nonnegative_finite_tolerance(tolerance_m)
    for start in range(0, boxes.shape[0], 128):
        chunk = boxes[start : start + 128]
        inside = np.all(
            (points[:, None, :] > chunk[None, :, :3] + tol)
            & (points[:, None, :] < chunk[None, :, 3:] - tol),
            axis=2,
        )
        inside_any |= np.any(inside, axis=1)
    return inside_any


def _transport_floor_footprint_mask(
    points_xy: NDArray[np.float64],
    boxes_m: NDArray[np.float64],
    *,
    tolerance_m: float,
) -> NDArray[np.bool_]:
    """Return points covered by floor-contact transport components."""
    points = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    boxes = np.asarray(boxes_m, dtype=float).reshape(-1, 6)
    if boxes.size == 0:
        return np.zeros(points.shape[0], dtype=bool)
    tol = _nonnegative_finite_tolerance(tolerance_m)
    floor_boxes = boxes[boxes[:, 2] <= tol]
    covered = np.zeros(points.shape[0], dtype=bool)
    for start in range(0, floor_boxes.shape[0], 128):
        chunk = floor_boxes[start : start + 128]
        inside = (
            (points[:, None, 0] >= chunk[None, :, 0] - tol)
            & (points[:, None, 0] <= chunk[None, :, 3] + tol)
            & (points[:, None, 1] >= chunk[None, :, 1] - tol)
            & (points[:, None, 1] <= chunk[None, :, 4] + tol)
        )
        covered |= np.any(inside, axis=1)
    return covered


def _transport_room_face_covered_mask(
    points_xyz: NDArray[np.float64],
    boxes_m: NDArray[np.float64],
    *,
    inward_normal_xyz: Sequence[float],
    tolerance_m: float,
) -> NDArray[np.bool_]:
    """Return room-boundary points hidden by a touching transport solid."""
    points = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    boxes = np.asarray(boxes_m, dtype=float).reshape(-1, 6)
    if points.shape[0] == 0 or boxes.shape[0] == 0:
        return np.zeros(points.shape[0], dtype=bool)
    inward = np.asarray(inward_normal_xyz, dtype=float).reshape(3)
    norm = float(np.linalg.norm(inward))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("inward_normal_xyz must be finite and nonzero.")
    tol = _nonnegative_finite_tolerance(tolerance_m)
    step = max(4.0 * tol, 1.0e-8)
    probes = points + (step / norm) * inward[None, :]
    return _points_inside_transport_box_union(
        probes,
        boxes,
        tolerance_m=tol,
    )


def _transport_component_surface_kinds(
    points_xyz: NDArray[np.float64],
    boxes_m: NDArray[np.float64],
    *,
    room_upper_xyz: Sequence[float],
    tolerance_m: float,
) -> NDArray[np.object_]:
    """Classify points on attached transport-component box surfaces."""
    points = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    boxes = np.asarray(boxes_m, dtype=float).reshape(-1, 6)
    kinds = np.full(points.shape[0], None, dtype=object)
    if boxes.size == 0 or points.size == 0:
        return kinds
    tol = _nonnegative_finite_tolerance(tolerance_m)
    room_upper = np.asarray(room_upper_xyz, dtype=float).reshape(3)
    if np.any(~np.isfinite(room_upper)) or np.any(room_upper <= 0.0):
        raise ValueError("room_upper_xyz must contain finite positive values.")
    eligible = ~_points_inside_transport_box_union(
        points,
        boxes,
        tolerance_m=tol,
    )
    for start in range(0, boxes.shape[0], 128):
        active = eligible & np.equal(kinds, None)
        if not np.any(active):
            break
        chunk = boxes[start : start + 128]
        selected = points[active]
        within = (
            (selected[:, None, :] >= chunk[None, :, :3] - tol)
            & (selected[:, None, :] <= chunk[None, :, 3:] + tol)
        )
        within_all = np.all(within, axis=2)
        on_lower = np.abs(
            selected[:, None, :] - chunk[None, :, :3]
        ) <= tol
        on_upper = np.abs(
            selected[:, None, :] - chunk[None, :, 3:]
        ) <= tol
        probe_step = max(4.0 * tol, 1.0e-8)
        exposed_lower = np.zeros((selected.shape[0], 3), dtype=bool)
        exposed_upper = np.zeros((selected.shape[0], 3), dtype=bool)
        for axis in range(3):
            lower_probe = selected.copy()
            lower_probe[:, axis] -= probe_step
            upper_probe = selected.copy()
            upper_probe[:, axis] += probe_step
            exposed_lower[:, axis] = ~_points_inside_transport_box_union(
                lower_probe,
                boxes,
                tolerance_m=tol,
            )
            exposed_upper[:, axis] = ~_points_inside_transport_box_union(
                upper_probe,
                boxes,
                tolerance_m=tol,
            )
            exposed_lower[:, axis] &= selected[:, axis] > tol
            exposed_upper[:, axis] &= (
                selected[:, axis] < room_upper[axis] - tol
            )
        lower_faces = (
            within_all[:, :, None]
            & on_lower
            & exposed_lower[:, None, :]
        )
        upper_faces = (
            within_all[:, :, None]
            & on_upper
            & exposed_upper[:, None, :]
        )
        top = upper_faces[:, :, 2]
        bottom = lower_faces[:, :, 2]
        side = np.any(
            np.concatenate(
                [lower_faces[:, :, :2], upper_faces[:, :, :2]],
                axis=2,
            ),
            axis=2,
        )
        active_indices = np.flatnonzero(active)
        top_rows = active_indices[np.any(top, axis=1)]
        kinds[top_rows] = "obstacle_top"
        remaining = np.equal(kinds[active_indices], None)
        bottom_rows = active_indices[remaining & np.any(bottom, axis=1)]
        kinds[bottom_rows] = "obstacle_bottom"
        remaining = np.equal(kinds[active_indices], None)
        side_rows = active_indices[remaining & np.any(side, axis=1)]
        kinds[side_rows] = "obstacle_side"
    return kinds


def source_surface_kind(
    position: Sequence[float],
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None = None,
    *,
    obstacle_height_m: float = 2.0,
    tolerance_m: float = 1.0e-6,
) -> SourceSurfaceKind | None:
    """Return the allowed source surface kind for a position, or None."""
    if len(position) != 3:
        raise ValueError("position must be a 3-element vector.")
    return source_surface_kinds(
        np.asarray(position, dtype=float).reshape(1, 3),
        env,
        obstacle_grid,
        obstacle_height_m=obstacle_height_m,
        tolerance_m=tolerance_m,
    )[0]


def source_surface_kinds(
    positions: NDArray[np.float64],
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None = None,
    *,
    obstacle_height_m: float = 2.0,
    tolerance_m: float = 1.0e-6,
) -> NDArray[np.object_]:
    """Return vectorized allowed source surface kinds for batched positions."""
    del obstacle_height_m
    arr = np.asarray(positions, dtype=float)
    if arr.shape[-1:] != (3,):
        raise ValueError("positions must have a final dimension of 3.")
    if arr.size == 0:
        return np.zeros(arr.reshape(-1, 3).shape[0], dtype=object)
    points = arr.reshape(-1, 3)
    tol = _nonnegative_finite_tolerance(tolerance_m)
    kinds = np.full(points.shape[0], None, dtype=object)
    valid = (
        (points[:, 0] >= -tol)
        & (points[:, 1] >= -tol)
        & (points[:, 2] >= -tol)
        & (points[:, 0] <= float(env.size_x) + tol)
        & (points[:, 1] <= float(env.size_y) + tol)
        & (points[:, 2] <= float(env.size_z) + tol)
    )
    if not np.any(valid):
        return kinds

    use_transport_surfaces = bool(
        obstacle_grid is not None and obstacle_grid.has_transport_model
    )
    if use_transport_surfaces:
        transport_boxes = _clipped_source_transport_boxes(
            env,
            obstacle_grid,
        )
        valid &= ~_points_inside_transport_box_union(
            points,
            transport_boxes,
            tolerance_m=tol,
        )
        blocked = _transport_floor_footprint_mask(
            points[:, :2],
            transport_boxes,
            tolerance_m=tol,
        )
    else:
        transport_boxes = np.zeros((0, 6), dtype=float)
        blocked = np.zeros(points.shape[0], dtype=bool)
    floor = valid & (np.abs(points[:, 2]) <= tol) & ~blocked
    kinds[floor] = "floor"

    unset = valid & np.equal(kinds, None)
    ceiling = unset & (np.abs(points[:, 2] - float(env.size_z)) <= tol)
    if use_transport_surfaces and np.any(ceiling):
        ceiling &= ~_transport_room_face_covered_mask(
            points,
            transport_boxes,
            inward_normal_xyz=(0.0, 0.0, -1.0),
            tolerance_m=tol,
        )
    kinds[ceiling] = "ceiling"

    unset = valid & np.equal(kinds, None)
    wall = np.zeros(points.shape[0], dtype=bool)
    wall_specs = (
        (np.abs(points[:, 0]) <= tol, (1.0, 0.0, 0.0)),
        (
            np.abs(points[:, 0] - float(env.size_x)) <= tol,
            (-1.0, 0.0, 0.0),
        ),
        (np.abs(points[:, 1]) <= tol, (0.0, 1.0, 0.0)),
        (
            np.abs(points[:, 1] - float(env.size_y)) <= tol,
            (0.0, -1.0, 0.0),
        ),
    )
    for on_face, inward_normal in wall_specs:
        candidate = unset & on_face
        if use_transport_surfaces and np.any(candidate):
            candidate &= ~_transport_room_face_covered_mask(
                points,
                transport_boxes,
                inward_normal_xyz=inward_normal,
                tolerance_m=tol,
            )
        wall |= candidate
    kinds[wall] = "wall"

    if not use_transport_surfaces:
        return kinds
    unset = valid & np.equal(kinds, None)
    if np.any(unset):
        component_kinds = _transport_component_surface_kinds(
            points[unset],
            transport_boxes,
            room_upper_xyz=(
                float(env.size_x),
                float(env.size_y),
                float(env.size_z),
            ),
            tolerance_m=tol,
        )
        kinds[np.flatnonzero(unset)] = component_kinds
    return kinds


def source_surface_kind_counts(
    positions: NDArray[np.float64],
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None = None,
    *,
    obstacle_height_m: float = 2.0,
    tolerance_m: float = 1.0e-6,
) -> dict[str, int]:
    """Return surface-kind counts for batched source positions."""
    kinds = source_surface_kinds(
        positions,
        env,
        obstacle_grid,
        obstacle_height_m=obstacle_height_m,
        tolerance_m=tolerance_m,
    )
    counts = {label: 0 for label in SOURCE_SURFACE_REPORT_LABELS}
    for label in SOURCE_SURFACE_REPORT_LABELS[:-1]:
        counts[label] = int(np.count_nonzero(kinds == label))
    counts["off_surface"] = int(np.count_nonzero(np.equal(kinds, None)))
    return counts


def is_allowed_source_surface_position(
    position: Sequence[float],
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None = None,
    *,
    obstacle_height_m: float = 2.0,
    tolerance_m: float = 1.0e-6,
) -> bool:
    """Return True when a source position lies on an allowed physical surface."""
    point = np.asarray(position, dtype=float).reshape(1, 3)
    if bool(transport_interior_mask(point, obstacle_grid, tolerance_m=tolerance_m)[0]):
        return False
    return (
        source_surface_kind(
            position,
            env,
            obstacle_grid,
            obstacle_height_m=obstacle_height_m,
            tolerance_m=tolerance_m,
        )
        is not None
    )


def _build_source_surface_atlas(
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    *,
    obstacle_height_m: float,
    chart_max_edge_m: float = 1.0,
) -> SurfaceChartGeometry:
    """Build the shared exact physical-surface atlas used for truth sampling."""
    if isinstance(chart_max_edge_m, (bool, np.bool_)) or not isinstance(
        chart_max_edge_m,
        Real,
    ):
        raise TypeError("chart_max_edge_m must be a real number.")
    maximum_edge = float(chart_max_edge_m)
    if not np.isfinite(maximum_edge) or maximum_edge <= 0.0:
        raise ValueError("chart_max_edge_m must be finite and positive.")
    charts = build_surface_chart_geometry(
        env,
        obstacle_grid,
        max_edge_m=maximum_edge,
        obstacle_height_m=obstacle_height_m,
    )
    if not charts.obstacle_surfaces_available:
        raise ValueError(
            "Random source placement requires transport_boxes_m when blocked "
            "obstacle cells are present; synthetic blocked-cell surfaces are "
            "not physical Geant4 source support."
        )
    return charts


def generate_surface_sources(
    *,
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    isotopes: Sequence[str],
    intensity_cps_1m: float | Sequence[float] | Mapping[str, float | Sequence[float]],
    rng: np.random.Generator,
    count: int | None = None,
    obstacle_height_m: float = 2.0,
    chart_max_edge_m: float = 1.0,
) -> list[PointSource]:
    """Generate independent points uniformly over eligible physical area.

    Source locations are independent continuous draws from the normalized
    physical surface-area measure.  No detector visibility, height, ceiling
    quota, isotope separation, PF state, or response-conditioning criterion
    can alter the truth-position distribution.
    """
    if isinstance(isotopes, (str, bytes)) or not isinstance(
        isotopes,
        Sequence,
    ):
        raise TypeError("isotopes must be a sequence of JSON strings.")
    isotope_names = tuple(isotopes)
    if not isotope_names:
        raise ValueError("At least one isotope is required.")
    if any(
        not isinstance(isotope, str) or not isotope
        for isotope in isotope_names
    ):
        raise TypeError("isotopes must contain nonempty JSON strings.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")
    if count is None:
        source_count = len(isotopes)
    else:
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
            raise TypeError("count must be a positive integer.")
        source_count = int(count)
        if source_count <= 0:
            raise ValueError("count must be a positive integer.")
    surface_atlas = _build_source_surface_atlas(
        env,
        obstacle_grid,
        obstacle_height_m=obstacle_height_m,
        chart_max_edge_m=chart_max_edge_m,
    )
    validate_air_facing_surface_normals(surface_atlas)
    positions, chart_indices, surface_uv = sample_continuous_surface_coordinates(
        surface_atlas,
        source_count,
        rng,
    )
    if positions.shape != (source_count, 3):
        raise RuntimeError(
            "Continuous source sampler returned an invalid position array."
        )
    if np.any(transport_interior_mask(positions, obstacle_grid)):
        raise RuntimeError(
            "Continuous source sampler returned a transport-interior point."
        )
    sampled_kinds = source_surface_kinds(
        positions,
        env,
        obstacle_grid,
        obstacle_height_m=obstacle_height_m,
    )
    if np.any(np.equal(sampled_kinds, None)):
        raise RuntimeError(
            "Continuous source sampler returned a point outside physical "
            "surface support."
        )
    sources: list[PointSource] = []
    normals = np.asarray(
        surface_atlas.normals_xyz[chart_indices],
        dtype=np.float64,
    )
    transport_positions = surface_transport_positions(positions, normals)
    policy_sha256 = surface_emission_policy_sha256()
    for idx in range(source_count):
        isotope = isotope_names[idx % len(isotope_names)]
        intensity = _sample_intensity_cps_1m(
            intensity_cps_1m,
            isotope=isotope,
            rng=rng,
        )
        sampled = positions[idx]
        position = (
            float(sampled[0]),
            float(sampled[1]),
            float(sampled[2]),
        )
        sources.append(
            PointSource(
                isotope=isotope,
                position=position,
                intensity_cps_1m=intensity,
                surface_chart_id=int(chart_indices[idx]),
                surface_uv=(
                    float(surface_uv[idx, 0]),
                    float(surface_uv[idx, 1]),
                ),
                surface_normal=(
                    float(normals[idx, 0]),
                    float(normals[idx, 1]),
                    float(normals[idx, 2]),
                ),
                transport_position=(
                    float(transport_positions[idx, 0]),
                    float(transport_positions[idx, 1]),
                    float(transport_positions[idx, 2]),
                ),
                surface_emission_policy_sha256=policy_sha256,
            )
        )
    return sources


def _sample_intensity_cps_1m(
    intensity_cps_1m: float | Sequence[float] | Mapping[str, float | Sequence[float]],
    *,
    isotope: str,
    rng: np.random.Generator,
) -> float:
    """Return a fixed or uniformly sampled detector-cps@1m source strength."""
    def _positive_finite_strength(value: object, *, name: str) -> float:
        """Return one strict positive finite detector-cps@1m value."""
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number.")
        resolved = float(value)
        if not np.isfinite(resolved) or resolved <= 0.0:
            raise ValueError(f"{name} must be positive and finite.")
        return resolved

    raw_value: object
    if isinstance(intensity_cps_1m, Mapping):
        raw_value = intensity_cps_1m[isotope]
    else:
        raw_value = intensity_cps_1m
    if isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes)):
        if len(raw_value) != 2:
            raise ValueError("intensity range must contain exactly two values.")
        lo = _positive_finite_strength(
            raw_value[0],
            name="intensity range minimum",
        )
        hi = _positive_finite_strength(
            raw_value[1],
            name="intensity range maximum",
        )
        if hi < lo:
            raise ValueError("intensity range maximum must be >= minimum.")
        if hi == lo:
            return lo
        return float(rng.uniform(lo, hi))
    return _positive_finite_strength(
        raw_value,
        name="intensity_cps_1m",
    )
