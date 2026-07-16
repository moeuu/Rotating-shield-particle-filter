"""Tests for robot traversability map generation."""

from pathlib import Path

import numpy as np

from measurement.obstacles import ObstacleGrid
from planning.traversability import (
    TraversabilityMap,
    build_traversability_map_from_obstacle_grid,
    build_traversability_map_from_stage_solids,
    render_traversability_map,
    shortest_grid_path_points,
)
from sim.isaacsim_app.stage_backend import PrimPose, StageSolidPrim


class TransitionAwareMap:
    """Wrap a traversability map with one forbidden graph transition."""

    def __init__(
        self,
        base_map: TraversabilityMap,
        forbidden_edge: tuple[tuple[int, int], tuple[int, int]],
    ) -> None:
        """Store the wrapped map and undirected forbidden transition."""
        self.base_map = base_map
        self.forbidden_edge = frozenset(forbidden_edge)

    def __getattr__(self, name: str) -> object:
        """Forward standard grid APIs to the wrapped map."""
        return getattr(self.base_map, name)

    def is_transition_free(
        self,
        first_cell: tuple[int, int],
        second_cell: tuple[int, int],
    ) -> bool:
        """Reject exactly one undirected graph edge."""
        return frozenset((first_cell, second_cell)) != self.forbidden_edge


def test_traversability_map_projects_obstacle_footprints() -> None:
    """A robot footprint should reject cells too close to projected obstacles."""
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(3, 3),
        blocked_cells=((1, 1),),
    )

    narrow_robot = build_traversability_map_from_obstacle_grid(
        grid,
        robot_radius_m=0.35,
    )
    wide_robot = build_traversability_map_from_obstacle_grid(
        grid,
        robot_radius_m=0.6,
    )

    assert not narrow_robot.is_free((1.5, 1.5))
    assert narrow_robot.is_free((0.5, 1.5))
    assert not wide_robot.is_free((0.5, 1.5))


def test_traversability_shortest_path_avoids_blocked_cells() -> None:
    """Shortest paths should route around obstacle cells instead of drawing a chord."""
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(5, 3),
        blocked_cells=((2, 0), (2, 1)),
    )
    traversable = build_traversability_map_from_obstacle_grid(
        grid,
        robot_radius_m=0.0,
    )

    path = traversable.shortest_path_points(
        (0.5, 0.5, 0.0),
        (4.5, 0.5, 0.0),
    )

    assert path is not None
    assert np.max(path[:, 1]) > 2.0
    assert sum(np.linalg.norm(delta) for delta in np.diff(path, axis=0)) > 4.0
    for waypoint in path:
        assert traversable.is_free(waypoint)


def test_traversability_shortest_path_reports_disconnected_cells() -> None:
    """Disconnected free components should return no path."""
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(5, 3),
        blocked_cells=((2, 0), (2, 1), (2, 2)),
    )
    traversable = build_traversability_map_from_obstacle_grid(
        grid,
        robot_radius_m=0.0,
    )

    path = traversable.shortest_path_points(
        (0.5, 0.5, 0.0),
        (4.5, 0.5, 0.0),
    )

    assert path is None


def test_traversability_shortest_path_honors_transition_hook() -> None:
    """A map transition hook should force A* around an unsafe free-node edge."""
    base_map = TraversabilityMap(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(3, 2),
        traversable_cells=tuple(
            (ix, iy) for ix in range(3) for iy in range(2)
        ),
    )
    wrapped = TransitionAwareMap(base_map, ((0, 0), (1, 0)))

    path = base_map.shortest_path_points((0.5, 0.5), (2.5, 0.5))
    rerouted = shortest_grid_path_points(
        wrapped,
        (0.5, 0.5),
        (2.5, 0.5),
    )

    assert path is not None
    assert rerouted is not None
    assert np.max(rerouted[:, 1]) > np.max(path[:, 1])


def test_traversability_map_keeps_reachable_component_only() -> None:
    """Reachable filtering should remove free cells isolated from the robot start."""
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(5, 3),
        blocked_cells=((2, 0), (2, 1), (2, 2)),
    )

    traversable = build_traversability_map_from_obstacle_grid(
        grid,
        robot_radius_m=0.35,
        reachable_from=(0.5, 1.5),
    )

    assert traversable.is_free((0.5, 1.5))
    assert not traversable.is_free((4.5, 1.5))


def test_traversability_map_roundtrip_and_render(tmp_path: Path) -> None:
    """Traversability maps should round-trip and render a PNG for inspection."""
    traversable = TraversabilityMap(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(2, 2),
        traversable_cells=((0, 0), (1, 0)),
        robot_radius_m=0.35,
    )
    json_path = tmp_path / "map.json"
    png_path = tmp_path / "map.png"

    traversable.save(json_path)
    render_traversability_map(traversable, png_path)

    loaded = TraversabilityMap.load(json_path)
    assert loaded == traversable
    assert png_path.exists()
    assert loaded.is_free((0.5, 0.5))
    assert not loaded.is_free((0.5, 1.5))


def test_traversability_batch_free_space_matches_scalar() -> None:
    """Batched traversability lookup should reject points outside the map."""
    traversable = TraversabilityMap(
        origin=(1.0, 2.0),
        cell_size=0.5,
        grid_shape=(3, 2),
        traversable_cells=((0, 0), (2, 1)),
    )
    points = np.asarray(
        [
            [0.9, 2.1, 0.5],
            [1.1, 2.1, 0.5],
            [1.6, 2.1, 1.5],
            [2.1, 2.6, 0.5],
            [2.6, 2.6, 0.5],
        ],
        dtype=float,
    )

    batch = traversable.is_free_batch(points)
    scalar = np.asarray(
        [traversable.is_free(point) for point in points],
        dtype=bool,
    )

    assert np.array_equal(batch, scalar)


def test_traversability_map_projects_stage_solids() -> None:
    """USD/Isaac solid geometry should project into the same map API."""
    solids = [
        StageSolidPrim(
            path="/World/Environment/Floor",
            shape="box",
            pose=PrimPose(translation_xyz=(1.5, 1.5, -0.05)),
            size_xyz=(3.0, 3.0, 0.05),
        ),
        StageSolidPrim(
            path="/World/Environment/Wall",
            shape="box",
            pose=PrimPose(translation_xyz=(1.5, 1.5, 0.75)),
            size_xyz=(1.0, 1.0, 1.5),
        ),
        StageSolidPrim(
            path="/World/Environment/MeshCrate",
            shape="mesh",
            pose=PrimPose(),
            triangles_xyz=(
                ((2.0, 0.0, 0.2), (2.8, 0.0, 0.2), (2.0, 0.8, 0.8)),
            ),
        ),
    ]

    traversable = build_traversability_map_from_stage_solids(
        solids,
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(3, 3),
        robot_radius_m=0.35,
        reachable_from=(0.5, 0.5),
        blocking_z_range_m=(0.05, 2.0),
    )

    assert traversable.source == "stage_projected_3d_environment"
    assert traversable.is_free((0.5, 0.5))
    assert not traversable.is_free((1.5, 1.5))
    assert not traversable.is_free((2.5, 0.5))
