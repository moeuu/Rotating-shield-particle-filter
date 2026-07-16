"""Tests for the three-dimensional detector measurement workspace."""

from __future__ import annotations

import pickle
from typing import Sequence

import numpy as np
import pytest

from measurement.obstacles import ObstacleGrid
from planning.measurement_workspace import (
    AxisAlignedRoomBounds,
    DetectorAssemblyGeometry,
    MeasurementWorkspace,
)


class DummyBaseMap:
    """Provide a small deterministic 2-D map API for workspace tests."""

    robot_radius_m = 0.3
    origin = (0.0, 0.0)
    grid_shape = (5, 5)

    def __init__(
        self,
        *,
        disconnected: bool = False,
        direct_path: bool = False,
    ) -> None:
        """Store the desired deterministic path behavior."""
        self.disconnected = bool(disconnected)
        self.direct_path = bool(direct_path)
        self.path_calls = 0

    def is_free_batch(self, points: Sequence[Sequence[float]]) -> np.ndarray:
        """Mark points inside a conservative square footprint as free."""
        array = np.asarray(points, dtype=float)
        return (
            (array[:, 0] >= 0.3)
            & (array[:, 0] <= 4.7)
            & (array[:, 1] >= 0.3)
            & (array[:, 1] <= 4.7)
        )

    def cell_center(self, cell: tuple[int, int]) -> tuple[float, float]:
        """Return a deterministic cell center for forwarding checks."""
        return float(cell[0]) + 0.5, float(cell[1]) + 0.5

    def shortest_path_points(
        self,
        start_point: Sequence[float],
        goal_point: Sequence[float],
        *,
        allow_diagonal: bool = True,
    ) -> np.ndarray | None:
        """Return a direct or one-elbow path between the supplied points."""
        assert allow_diagonal
        self.path_calls += 1
        if self.disconnected:
            return None
        start = np.asarray(start_point, dtype=float)
        goal = np.asarray(goal_point, dtype=float)
        if self.direct_path:
            return np.vstack([start, goal])
        elbow = np.array([start[0], goal[1], start[2]], dtype=float)
        return np.vstack([start, elbow, goal])


class UndersizedBaseMap(DummyBaseMap):
    """Represent a map whose clearance radius cannot cover the assembly."""

    robot_radius_m = 0.2


def _room() -> AxisAlignedRoomBounds:
    """Return the standard room used by focused geometry tests."""
    return AxisAlignedRoomBounds((0.0, 0.0, 0.0), (5.0, 5.0, 3.0))


def _assembly() -> DetectorAssemblyGeometry:
    """Return a conservative base, mast, and detector-head envelope."""
    return DetectorAssemblyGeometry(
        base_radius_m=0.3,
        base_height_m=0.3,
        mast_radius_m=0.05,
        head_radius_m=0.2,
    )


def _workspace(
    boxes: Sequence[Sequence[float]] = (),
    *,
    base_map: object | None = None,
    element_budget: int = 1_000_000,
    motion_worker_count: int = 0,
    motion_grid_cell_size_m: float = 0.25,
) -> MeasurementWorkspace:
    """Build the standard three-dimensional workspace."""
    return MeasurementWorkspace(
        room_bounds=_room(),
        assembly=_assembly(),
        ground_z_m=0.0,
        detector_transport_world_z_m=0.6,
        collision_boxes_m=boxes,
        base_map=base_map,
        element_budget=element_budget,
        motion_worker_count=motion_worker_count,
        motion_grid_cell_size_m=motion_grid_cell_size_m,
    )


def _point_aabb_distance_sq(point: np.ndarray, box: np.ndarray) -> float:
    """Return the squared Euclidean distance from a point to one AABB."""
    closest = np.minimum(np.maximum(point, box[:3]), box[3:])
    return float(np.sum((point - closest) ** 2))


def _scalar_sphere_collision(
    center: np.ndarray,
    radius_m: float,
    boxes: np.ndarray,
) -> bool:
    """Return a scalar sphere-versus-AABB collision oracle result."""
    return any(
        _point_aabb_distance_sq(center, box) <= radius_m**2
        for box in boxes
    )


def _scalar_cylinder_collision(
    center_xy: np.ndarray,
    z_lower_m: float,
    z_upper_m: float,
    radius_m: float,
    boxes: np.ndarray,
) -> bool:
    """Return a scalar vertical-cylinder versus AABB collision result."""
    for box in boxes:
        if z_upper_m < box[2] or z_lower_m > box[5]:
            continue
        closest_xy = np.minimum(
            np.maximum(center_xy, box[:2]),
            box[3:5],
        )
        if float(np.sum((center_xy - closest_xy) ** 2)) <= radius_m**2:
            return True
    return False


def _scalar_vertical_capsule_collision(
    center_xy: np.ndarray,
    z_lower_m: float,
    z_upper_m: float,
    radius_m: float,
    boxes: np.ndarray,
) -> bool:
    """Return a scalar vertical-capsule versus AABB collision result."""
    for box in boxes:
        closest_xy = np.minimum(
            np.maximum(center_xy, box[:2]),
            box[3:5],
        )
        distance_xy_sq = float(np.sum((center_xy - closest_xy) ** 2))
        distance_z = max(box[2] - z_upper_m, z_lower_m - box[5], 0.0)
        if distance_xy_sq + distance_z**2 <= radius_m**2:
            return True
    return False


def _scalar_segment_expanded_aabb_collision(
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    expansion_xyz_m: np.ndarray,
    boxes: np.ndarray,
) -> bool:
    """Return a scalar segment collision against expanded AABBs."""
    direction = end_xyz - start_xyz
    for box in boxes:
        lower = box[:3] - expansion_xyz_m
        upper = box[3:] + expansion_xyz_m
        t_enter = 0.0
        t_exit = 1.0
        possible = True
        for axis in range(3):
            if abs(direction[axis]) <= 1.0e-12:
                if start_xyz[axis] < lower[axis] or start_xyz[axis] > upper[axis]:
                    possible = False
                    break
                continue
            t0 = (lower[axis] - start_xyz[axis]) / direction[axis]
            t1 = (upper[axis] - start_xyz[axis]) / direction[axis]
            t_enter = max(t_enter, min(t0, t1))
            t_exit = min(t_exit, max(t0, t1))
        if possible and t_exit >= t_enter:
            return True
    return False


def _random_boxes(
    rng: np.random.Generator,
    count: int,
) -> np.ndarray:
    """Return deterministic non-degenerate random collision boxes."""
    lower = rng.uniform((0.4, 0.4, 0.02), (4.4, 4.4, 2.5), size=(count, 3))
    widths = rng.uniform(0.03, 0.35, size=(count, 3))
    return np.hstack([lower, lower + widths])


def test_geometry_dataclasses_reject_invalid_physical_dimensions() -> None:
    """Physical geometry types reject degenerate rooms and envelopes."""
    with pytest.raises(ValueError, match="strictly greater"):
        AxisAlignedRoomBounds((0.0, 0.0, 0.0), (1.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="base_radius_m"):
        DetectorAssemblyGeometry(0.0, 0.3, 0.05, 0.2)
    with pytest.raises(ValueError, match="mast_radius_m"):
        DetectorAssemblyGeometry(0.3, 0.3, 0.4, 0.2)


def test_workspace_rejects_invalid_transport_boxes_and_map_clearance() -> None:
    """Workspace construction enforces transport and map prerequisites."""
    with pytest.raises(ValueError, match="transport"):
        MeasurementWorkspace(_room(), _assembly(), 0.0, 0.4)
    with pytest.raises(ValueError, match="ordered"):
        _workspace(((1.0, 1.0, 1.0, 0.0, 2.0, 2.0),))
    with pytest.raises(ValueError, match="head_radius_m"):
        MeasurementWorkspace(
            _room(),
            DetectorAssemblyGeometry(0.15, 0.3, 0.05, 0.2),
            0.0,
            0.6,
        )
    with pytest.raises(ValueError, match="robot_radius_m"):
        _workspace(base_map=UndersizedBaseMap())
    with pytest.raises(ValueError, match="motion_grid_cell_size_m"):
        _workspace(motion_grid_cell_size_m=0.0)


@pytest.mark.parametrize(
    ("boxes", "failed_component"),
    [
        (
            ((1.25, 0.98, 0.10, 1.35, 1.02, 0.20),),
            "base_collision_free",
        ),
        (
            ((1.04, 0.98, 0.60, 1.08, 1.02, 0.70),),
            "mast_collision_free",
        ),
        (
            ((1.15, 0.98, 1.18, 1.25, 1.02, 1.22),),
            "head_collision_free",
        ),
    ],
)
def test_endpoint_checks_isolate_each_assembly_component(
    boxes: Sequence[Sequence[float]],
    failed_component: str,
) -> None:
    """Base, mast, and detector-head collisions are reported separately."""
    masks = _workspace(boxes).endpoint_validity_masks(((1.0, 1.0, 1.2),))

    assert not bool(masks[failed_component][0])
    assert not bool(masks["valid"][0])
    for component in (
        "base_collision_free",
        "mast_collision_free",
        "head_collision_free",
    ):
        if component != failed_component:
            assert bool(masks[component][0])


def test_endpoint_checks_room_self_clearance_and_overhead_space() -> None:
    """Room walls, ceiling, base overlap, and clear overhead are handled."""
    clear_workspace = _workspace()
    points = np.array(
        [
            [0.3, 1.0, 1.0],
            [0.299, 1.0, 1.0],
            [1.0, 1.0, 2.8],
            [1.0, 1.0, 2.801],
            [1.0, 1.0, 0.5],
            [1.0, 1.0, 0.499],
        ],
        dtype=float,
    )
    np.testing.assert_array_equal(
        clear_workspace.is_free_batch(points),
        np.array([True, False, True, False, True, False]),
    )

    overhead_box = ((0.9, 0.9, 1.6, 1.1, 1.1, 1.7),)
    assert _workspace(overhead_box).is_free((1.0, 1.0, 1.2))


def test_batched_endpoint_geometry_matches_scalar_oracle() -> None:
    """Batched exact endpoint checks match independent scalar geometry."""
    rng = np.random.default_rng(713)
    boxes = _random_boxes(rng, 37)
    poses = rng.uniform((0.2, 0.2, 0.35), (4.8, 4.8, 2.9), size=(193, 3))
    masks = _workspace(boxes).endpoint_validity_masks(poses)

    expected_base_free = []
    expected_mast_free = []
    expected_head_free = []
    for pose in poses:
        expected_base_free.append(
            not _scalar_cylinder_collision(
                pose[:2],
                0.0,
                0.3,
                0.3,
                boxes,
            )
        )
        expected_mast_free.append(
            not _scalar_cylinder_collision(
                pose[:2],
                0.3,
                max(float(pose[2]), 0.3),
                0.05,
                boxes,
            )
        )
        expected_head_free.append(
            not _scalar_sphere_collision(pose, 0.2, boxes)
        )

    np.testing.assert_array_equal(masks["base_collision_free"], expected_base_free)
    np.testing.assert_array_equal(masks["mast_collision_free"], expected_mast_free)
    np.testing.assert_array_equal(masks["head_collision_free"], expected_head_free)
    chunked_masks = _workspace(boxes, element_budget=41).endpoint_validity_masks(
        poses
    )
    for key, values in masks.items():
        np.testing.assert_array_equal(chunked_masks[key], values)


def test_vertical_head_sweep_rejects_intermediate_collision() -> None:
    """A box between two free heights blocks the exact vertical head sweep."""
    obstacle = ((2.10, 1.98, 1.20, 2.14, 2.02, 1.30),)
    workspace = _workspace(obstacle)
    start = np.array([2.0, 2.0, 0.8], dtype=float)
    end = np.array([2.0, 2.0, 1.8], dtype=float)

    assert workspace.is_free(start)
    assert workspace.is_free(end)
    assert not workspace.is_vertical_head_sweep_free(start, end)
    assert workspace.is_vertical_head_sweep_free(
        (3.0, 3.0, 0.8),
        (3.0, 3.0, 1.8),
    )
    assert not workspace.is_vertical_head_sweep_free(
        (3.0, 3.0, 0.8),
        (3.1, 3.0, 1.8),
    )


def test_batched_vertical_sweeps_match_scalar_oracle() -> None:
    """Batched vertical-capsule checks match an independent scalar oracle."""
    rng = np.random.default_rng(1701)
    boxes = _random_boxes(rng, 29)
    xy = rng.uniform((0.4, 0.4), (4.6, 4.6), size=(151, 2))
    start_z = rng.uniform(0.5, 2.7, size=151)
    end_z = rng.uniform(0.5, 2.7, size=151)
    starts = np.column_stack([xy, start_z])
    ends = np.column_stack([xy, end_z])
    workspace = _workspace(boxes)

    masks = workspace.vertical_head_sweep_validity_masks(starts, ends)
    expected_collision_free = np.array(
        [
            not _scalar_vertical_capsule_collision(
                xy[index],
                min(float(start_z[index]), float(end_z[index])),
                max(float(start_z[index]), float(end_z[index])),
                0.2,
                boxes,
            )
            for index in range(starts.shape[0])
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(
        masks["head_sweep_collision_free"],
        expected_collision_free,
    )
    np.testing.assert_array_equal(
        masks["valid"],
        masks["endpoints_free"] & expected_collision_free,
    )
    chunked_masks = _workspace(
        boxes,
        element_budget=37,
    ).vertical_head_sweep_validity_masks(starts, ends)
    for key, values in masks.items():
        np.testing.assert_array_equal(chunked_masks[key], values)


def test_batched_horizontal_sweeps_match_scalar_expanded_box_oracle() -> None:
    """Batched horizontal envelopes match scalar expanded-AABB checks."""
    rng = np.random.default_rng(2903)
    boxes = _random_boxes(rng, 23)
    starts = np.column_stack(
        [
            rng.uniform(0.4, 4.6, size=(97, 2)),
            np.full(97, 0.6, dtype=float),
        ]
    )
    ends = np.column_stack(
        [
            rng.uniform(0.4, 4.6, size=(97, 2)),
            np.full(97, 0.6, dtype=float),
        ]
    )
    masks = _workspace(boxes).horizontal_motion_validity_masks(starts, ends)
    base_starts = starts.copy()
    base_ends = ends.copy()
    base_starts[:, 2] = 0.15
    base_ends[:, 2] = 0.15
    mast_starts = starts.copy()
    mast_ends = ends.copy()
    mast_starts[:, 2] = 0.45
    mast_ends[:, 2] = 0.45

    expected_base_free = np.array(
        [
            not _scalar_segment_expanded_aabb_collision(
                base_starts[index],
                base_ends[index],
                np.array([0.3, 0.3, 0.15]),
                boxes,
            )
            for index in range(starts.shape[0])
        ]
    )
    expected_mast_free = np.array(
        [
            not _scalar_segment_expanded_aabb_collision(
                mast_starts[index],
                mast_ends[index],
                np.array([0.05, 0.05, 0.15]),
                boxes,
            )
            for index in range(starts.shape[0])
        ]
    )
    expected_head_free = np.array(
        [
            not _scalar_segment_expanded_aabb_collision(
                starts[index],
                ends[index],
                np.full(3, 0.2),
                boxes,
            )
            for index in range(starts.shape[0])
        ]
    )

    np.testing.assert_array_equal(
        masks["base_sweep_collision_free"],
        expected_base_free,
    )
    np.testing.assert_array_equal(
        masks["mast_sweep_collision_free"],
        expected_mast_free,
    )
    np.testing.assert_array_equal(
        masks["head_sweep_collision_free"],
        expected_head_free,
    )
    chunked_masks = _workspace(
        boxes,
        element_budget=31,
    ).horizontal_motion_validity_masks(starts, ends)
    for key, values in masks.items():
        np.testing.assert_array_equal(chunked_masks[key], values)


def test_workspace_forwards_map_api_and_applies_horizontal_filter() -> None:
    """Grid APIs are forwarded while endpoint checks apply base-map freedom."""
    workspace = _workspace(base_map=DummyBaseMap())

    assert workspace.origin == (0.0, 0.0)
    assert workspace.grid_shape == (5, 5)
    assert workspace.cell_center((1, 2)) == (1.5, 2.5)
    assert workspace.is_free((1.0, 1.0, 1.0))
    assert not workspace.is_free((4.8, 1.0, 1.0))


def test_motion_waypoints_retract_translate_and_extend() -> None:
    """Safe motion uses vertical moves around a base-map horizontal path."""
    workspace = _workspace(base_map=DummyBaseMap())
    waypoints = workspace.motion_waypoints((1.0, 1.0, 1.2), (3.0, 2.0, 1.4))

    assert waypoints is not None
    np.testing.assert_allclose(
        waypoints,
        np.array(
            [
                [1.0, 1.0, 1.2],
                [1.0, 1.0, 0.6],
                [1.0, 2.0, 0.6],
                [3.0, 2.0, 0.6],
                [3.0, 2.0, 1.4],
            ]
        ),
    )
    assert np.all(workspace.is_free_batch(waypoints))


def test_motion_waypoints_use_direct_vertical_height_change_at_same_xy() -> None:
    """A height-only action should not retract before extending at fixed xy."""
    workspace = _workspace(base_map=DummyBaseMap())

    stationary = workspace.motion_waypoints((1.0, 1.0, 1.2), (1.0, 1.0, 1.2))
    height_change = workspace.motion_waypoints(
        (1.0, 1.0, 1.2),
        (1.0, 1.0, 1.4),
    )

    assert stationary is not None
    np.testing.assert_allclose(stationary, np.array([[1.0, 1.0, 1.2]]))
    assert height_change is not None
    np.testing.assert_allclose(
        height_change,
        np.array([[1.0, 1.0, 1.2], [1.0, 1.0, 1.4]]),
    )


def test_parallel_motion_path_lengths_match_serial_routes() -> None:
    """Parallel path lengths and reachability match the scalar route oracle."""
    goals = np.array(
        [
            [3.0, 2.0, 1.4],
            [4.8, 2.0, 1.4],
            [2.0, 3.0, 1.1],
            [3.0, 2.0, 1.4],
        ],
        dtype=float,
    )
    start = np.array([1.0, 1.0, 1.2], dtype=float)
    serial_workspace = _workspace(
        base_map=DummyBaseMap(),
        motion_worker_count=1,
    )
    expected_lengths = []
    for goal in goals:
        route = serial_workspace.motion_waypoints(start, goal)
        if route is None:
            expected_lengths.append(float("inf"))
        elif route.shape[0] <= 1:
            expected_lengths.append(0.0)
        else:
            expected_lengths.append(
                float(np.sum(np.linalg.norm(np.diff(route, axis=0), axis=1)))
            )
    expected = np.asarray(expected_lengths, dtype=float)
    workspace = _workspace(base_map=DummyBaseMap(), motion_worker_count=2)

    actual = workspace.motion_path_lengths_batch(start, goals)

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(
        workspace.is_motion_reachable_batch(start, goals),
        np.isfinite(expected),
    )


def test_motion_waypoint_cache_uses_immutable_route_snapshots() -> None:
    """Repeated route requests reuse an immutable cached waypoint snapshot."""
    base_map = DummyBaseMap()
    workspace = _workspace(base_map=base_map, motion_worker_count=1)
    start = np.array([1.0, 1.0, 1.2], dtype=float)
    goal = np.array([3.0, 2.0, 1.4], dtype=float)

    first = workspace.motion_waypoints(start, goal)
    assert first is not None
    first[0, 0] = 99.0
    second = workspace.motion_waypoints(start, goal)
    reverse = workspace.motion_waypoints(goal, start)

    assert second is not None
    assert reverse is not None
    assert base_map.path_calls == 1
    assert second[0, 0] == pytest.approx(1.0)
    np.testing.assert_allclose(reverse, second[::-1])
    restored = pickle.loads(pickle.dumps(workspace))
    restored_route = restored.motion_waypoints(start, goal)
    assert restored_route is not None
    np.testing.assert_allclose(restored_route, second)


@pytest.mark.parametrize(
    "base_map",
    [
        None,
        ObstacleGrid(
            origin=(0.0, 0.0),
            cell_size=1.0,
            grid_shape=(5, 5),
            blocked_cells=(),
        ),
    ],
    ids=("room_grid", "empty_native_grid"),
)
def test_collision_box_only_route_uses_transport_graph_detour(
    base_map: object | None,
) -> None:
    """A box absent from the base map should trigger a safe horizontal detour."""
    obstacle = ((2.3, 2.2, 0.0, 2.7, 2.8, 0.8),)
    workspace = _workspace(obstacle, base_map=base_map)
    start = np.array([1.0, 2.5, 1.2], dtype=float)
    goal = np.array([4.0, 2.5, 1.2], dtype=float)
    transport_start = np.array([[1.0, 2.5, 0.6]], dtype=float)
    transport_goal = np.array([[4.0, 2.5, 0.6]], dtype=float)

    assert not bool(
        workspace.is_horizontal_motion_free_batch(
            transport_start,
            transport_goal,
        )[0]
    )
    route = workspace.motion_waypoints(start, goal)

    assert route is not None
    assert np.any(np.abs(route[:, 1] - 2.5) > 0.25)
    horizontal = np.isclose(route[:-1, 2], 0.6) & np.isclose(
        route[1:, 2],
        0.6,
    )
    assert np.all(
        workspace.is_horizontal_motion_free_batch(
            route[:-1][horizontal],
            route[1:][horizontal],
        )
    )


def test_collision_box_wall_disconnects_transport_graph() -> None:
    """A floor-to-mast wall spanning the room should have no horizontal route."""
    wall = ((2.3, 0.0, 0.0, 2.7, 5.0, 0.8),)
    workspace = _workspace(wall, base_map=None)

    route = workspace.motion_waypoints(
        (1.0, 2.5, 1.2),
        (4.0, 2.5, 1.2),
    )

    assert route is None


def test_fine_graph_preserves_room_edge_detour_with_grid_like_base_map() -> None:
    """A coarse all-free base map must not erase a narrow safe room-edge route."""
    base_map = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(5, 5),
        blocked_cells=(),
    )
    wall_with_bottom_gap = ((2.3, 0.7, 0.0, 2.7, 5.0, 0.8),)
    workspace = _workspace(
        wall_with_bottom_gap,
        base_map=base_map,
        motion_grid_cell_size_m=0.25,
    )

    route = workspace.motion_waypoints(
        (1.13, 2.47, 1.2),
        (3.87, 2.53, 1.2),
    )

    assert route is not None
    assert workspace._transport_graph is not None
    assert workspace._transport_graph.cell_size == pytest.approx(0.25)
    assert np.min(route[:, 1]) == pytest.approx(0.375)
    assert np.min(route[:, 1]) >= 0.3
    horizontal = np.isclose(route[:-1, 2], 0.6) & np.isclose(
        route[1:, 2],
        0.6,
    )
    assert np.all(
        workspace.is_horizontal_motion_free_batch(
            route[:-1][horizontal],
            route[1:][horizontal],
        )
    )


def test_box_above_transport_envelope_keeps_direct_fast_path() -> None:
    """A box above the transport envelope should not add lattice waypoints."""
    overhead = ((2.3, 2.2, 1.0, 2.7, 2.8, 1.1),)
    workspace = _workspace(overhead, base_map=None)

    route = workspace.motion_waypoints(
        (1.0, 2.5, 1.2),
        (4.0, 2.5, 1.2),
    )

    assert route is not None
    np.testing.assert_allclose(
        route,
        np.array(
            [
                [1.0, 2.5, 1.2],
                [1.0, 2.5, 0.6],
                [4.0, 2.5, 0.6],
                [4.0, 2.5, 1.2],
            ]
        ),
    )


def test_transport_graph_overlays_native_base_map_no_go_cells() -> None:
    """Wrapped grid no-go cells must remain blocked by the private graph."""
    base_map = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(5, 5),
        blocked_cells=((2, 2),),
    )
    overhead = ((2.3, 2.2, 1.0, 2.7, 2.8, 1.1),)
    workspace = _workspace(
        overhead,
        base_map=base_map,
        motion_grid_cell_size_m=0.25,
    )

    route = workspace.motion_waypoints(
        (1.5, 2.5, 1.2),
        (3.5, 2.5, 1.2),
    )

    assert route is not None
    assert workspace._transport_graph is not None
    assert workspace._transport_graph.cell_size == pytest.approx(0.25)
    assert all(base_map.cell_index(point) != (2, 2) for point in route)
    assert np.any(np.abs(route[:, 1] - 2.5) > 0.5)
    graph = workspace._transport_graph
    horizontal = np.isclose(route[:-1, 2], 0.6) & np.isclose(
        route[1:, 2],
        0.6,
    )
    assert np.all(
        workspace._base_overlay_segment_free_batch(
            route[:-1][horizontal],
            route[1:][horizontal],
            np.asarray(graph.overlay_blocked_boxes_m, dtype=float),
        )
    )


def test_base_overlay_rejects_diagonal_no_go_corner_cut() -> None:
    """A fine direct edge must not squeeze through a blocked coarse-grid corner."""
    base_map = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(5, 5),
        blocked_cells=((0, 1), (1, 0)),
    )
    overhead = ((3.0, 3.0, 1.0, 3.2, 3.2, 1.1),)
    workspace = _workspace(overhead, base_map=base_map)
    graph = workspace._transport_graph
    assert graph is not None
    start = np.array([0.5, 0.5, 0.6], dtype=float)
    goal = np.array([1.5, 1.5, 0.6], dtype=float)

    assert workspace.is_free(start)
    assert workspace.is_free(goal)
    assert not workspace._direct_transport_segment_free(start, goal, graph)
    assert not bool(
        workspace._base_overlay_segment_free_batch(
            start.reshape(1, 3),
            goal.reshape(1, 3),
            np.asarray(graph.overlay_blocked_boxes_m, dtype=float),
        )[0]
    )


def test_arbitrary_endpoint_connects_when_fine_cell_center_is_no_go() -> None:
    """Exact connectors should escape a partial fine cell with a blocked center."""
    base_map = ObstacleGrid(
        origin=(0.1, 0.1),
        cell_size=1.0,
        grid_shape=(5, 5),
        blocked_cells=((1, 1),),
    )
    overhead = ((4.0, 4.0, 1.0, 4.2, 4.2, 1.1),)
    workspace = _workspace(overhead, base_map=base_map)
    graph = workspace._transport_graph
    assert graph is not None
    endpoint = np.array([1.05, 1.5, 0.6], dtype=float)
    owning_cell = graph.cell_index(endpoint)
    assert owning_cell is not None

    assert workspace.is_free(endpoint)
    assert not graph.is_overlay_free_cell(owning_cell)
    connectors = workspace._transport_connector_costs(endpoint, graph)
    assert connectors
    assert all(graph.is_overlay_free_cell(cell) for cell in connectors)


def test_batched_base_overlay_segments_match_scalar_aabb_oracle() -> None:
    """Batched coarse-map overlay checks must match scalar segment geometry."""
    base_map = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(5, 5),
        blocked_cells=((1, 1), (2, 3), (3, 2)),
    )
    overhead = ((4.0, 4.0, 1.0, 4.2, 4.2, 1.1),)
    workspace = _workspace(overhead, base_map=base_map)
    graph = workspace._transport_graph
    assert graph is not None
    boxes = np.asarray(graph.overlay_blocked_boxes_m, dtype=float)
    rng = np.random.default_rng(9321)
    starts = np.column_stack(
        [rng.uniform(0.31, 4.69, size=(83, 2)), np.full(83, 0.6)]
    )
    ends = np.column_stack(
        [rng.uniform(0.31, 4.69, size=(83, 2)), np.full(83, 0.6)]
    )
    expected = np.asarray(
        [
            not _scalar_segment_expanded_aabb_collision(
                start,
                end,
                np.zeros(3, dtype=float),
                boxes,
            )
            for start, end in zip(starts, ends, strict=True)
        ],
        dtype=bool,
    )

    actual = workspace._base_overlay_segment_free_batch(starts, ends, boxes)

    np.testing.assert_array_equal(actual, expected)


def test_transport_graph_batched_edges_match_scalar_queries() -> None:
    """Cached batched edge flags must equal one-edge swept-envelope queries."""
    obstacle = ((2.3, 2.2, 0.0, 2.7, 2.8, 0.8),)
    workspace = _workspace(
        obstacle,
        base_map=None,
        motion_grid_cell_size_m=0.5,
    )
    graph = workspace._transport_graph
    assert graph is not None
    pairs = []
    for ix in range(graph.grid_shape[0]):
        for iy in range(graph.grid_shape[1]):
            for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
                neighbor = (ix + dx, iy + dy)
                if (
                    0 <= neighbor[0] < graph.grid_shape[0]
                    and 0 <= neighbor[1] < graph.grid_shape[1]
                ):
                    pairs.append(((ix, iy), neighbor))
    starts = np.asarray(
        [(*graph.cell_center(first), 0.6) for first, _ in pairs],
        dtype=float,
    )
    ends = np.asarray(
        [(*graph.cell_center(second), 0.6) for _, second in pairs],
        dtype=float,
    )
    scalar = np.asarray(
        [
            bool(
                workspace.is_horizontal_motion_free_batch(
                    starts[index : index + 1],
                    ends[index : index + 1],
                )[0]
            )
            for index in range(len(pairs))
        ],
        dtype=bool,
    )
    cached = np.asarray(
        [graph.is_transition_free(first, second) for first, second in pairs],
        dtype=bool,
    )

    np.testing.assert_array_equal(cached, scalar)


def test_motion_waypoints_reject_vertical_horizontal_and_disconnected_paths() -> None:
    """Motion fails for swept collisions or a disconnected horizontal map."""
    vertical_box = ((1.12, 0.98, 0.85, 1.16, 1.02, 0.95),)
    vertical_workspace = _workspace(
        vertical_box,
        base_map=DummyBaseMap(direct_path=True),
    )
    assert vertical_workspace.motion_waypoints(
        (1.0, 1.0, 1.2),
        (3.0, 1.0, 1.2),
    ) is None

    horizontal_box = ((1.95, 1.12, 0.55, 2.05, 1.16, 0.65),)
    horizontal_workspace = _workspace(
        horizontal_box,
        base_map=DummyBaseMap(direct_path=True),
    )
    assert horizontal_workspace.motion_waypoints(
        (1.0, 1.0, 1.2),
        (3.0, 1.0, 1.2),
    ) is None

    disconnected_workspace = _workspace(
        base_map=DummyBaseMap(disconnected=True),
    )
    assert disconnected_workspace.motion_waypoints(
        (1.0, 1.0, 1.2),
        (3.0, 1.0, 1.2),
    ) is None
