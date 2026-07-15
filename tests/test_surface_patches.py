"""Tests for finite-area source-surface patch dictionaries."""

from __future__ import annotations

import numpy as np
import pytest

import measurement.surface_patches as surface_patches_module
from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid
from measurement.surface_patches import (
    _grid_component_neighbor_pairs,
    build_surface_patch_dictionary,
    estimate_surface_patch_count_upper_bound,
)


def test_room_patch_areas_cover_each_surface_exactly() -> None:
    """Room patch areas should preserve exact physical face areas."""
    env = EnvironmentConfig(size_x=4.0, size_y=3.0, size_z=2.0)

    patches = build_surface_patch_dictionary(env, None, spacing=1.0)

    kinds = np.asarray(patches.kinds, dtype=str)
    assert np.sum(patches.areas_m2[kinds == "floor"]) == pytest.approx(12.0)
    assert np.sum(patches.areas_m2[kinds == "ceiling"]) == pytest.approx(12.0)
    assert np.sum(patches.areas_m2[kinds == "wall"]) == pytest.approx(28.0)
    assert np.sum(patches.areas_m2) == pytest.approx(52.0)
    assert np.allclose(np.linalg.norm(patches.normals_xyz, axis=1), 1.0)


def test_patch_adjacency_has_physical_shared_edge_lengths() -> None:
    """A two-by-two floor should have four unit-length TV graph edges."""
    env = EnvironmentConfig(size_x=2.0, size_y=2.0, size_z=1.0)
    patches = build_surface_patch_dictionary(env, None, spacing=1.0)
    face_ids = np.asarray(patches.face_ids, dtype=str)
    floor_indices = np.flatnonzero(face_ids == "room_floor")
    floor_mask = np.isin(patches.adjacency_edges[:, 0], floor_indices) & np.isin(
        patches.adjacency_edges[:, 1],
        floor_indices,
    )

    assert np.count_nonzero(floor_mask) == 4
    assert np.allclose(patches.shared_edge_lengths_m[floor_mask], 1.0)


def test_room_faces_connect_across_every_shared_boundary() -> None:
    """Floor, walls, and ceiling should form one physical room-surface graph."""
    env = EnvironmentConfig(size_x=2.0, size_y=2.0, size_z=1.0)
    patches = build_surface_patch_dictionary(env, None, spacing=1.0)
    kinds = np.asarray(patches.kinds, dtype=str)
    face_ids = np.asarray(patches.face_ids, dtype=str)
    edges = patches.adjacency_edges

    def _cross_kind_length(first: str, second: str) -> float:
        """Return total edge length joining two surface kinds."""
        selected = (
            ((kinds[edges[:, 0]] == first) & (kinds[edges[:, 1]] == second))
            | ((kinds[edges[:, 0]] == second) & (kinds[edges[:, 1]] == first))
        )
        return float(np.sum(patches.shared_edge_lengths_m[selected]))

    wall_corners = (
        (kinds[edges[:, 0]] == "wall")
        & (kinds[edges[:, 1]] == "wall")
        & (face_ids[edges[:, 0]] != face_ids[edges[:, 1]])
    )
    assert _cross_kind_length("floor", "wall") == pytest.approx(8.0)
    assert _cross_kind_length("ceiling", "wall") == pytest.approx(8.0)
    assert np.sum(patches.shared_edge_lengths_m[wall_corners]) == pytest.approx(4.0)


def test_transport_components_define_obstacle_surfaces_and_floor_footprint() -> None:
    """Known component geometry should replace synthetic blocked-cell boxes."""
    env = EnvironmentConfig(size_x=3.0, size_y=3.0, size_z=3.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(3, 3),
        blocked_cells=((1, 1),),
        transport_boxes_m=((1.0, 1.0, 0.0, 2.0, 2.0, 2.0),),
    )

    patches = build_surface_patch_dictionary(
        env,
        grid,
        spacing=1.0,
        obstacle_height_m=2.0,
    )

    kinds = np.asarray(patches.kinds, dtype=str)
    face_ids = np.asarray(patches.face_ids, dtype=str)
    assert np.sum(patches.areas_m2[kinds == "floor"]) == pytest.approx(8.0)
    assert np.sum(patches.areas_m2[kinds == "obstacle_top"]) == pytest.approx(1.0)
    assert np.sum(patches.areas_m2[kinds == "obstacle_side"]) == pytest.approx(8.0)
    assert "transport_component_0_z1" in set(face_ids.tolist())
    assert {
        "transport_component_0_x0",
        "transport_component_0_x1",
        "transport_component_0_y0",
        "transport_component_0_y1",
    }.issubset(set(face_ids.tolist()))
    assert patches.geometry_metadata == {
        "obstacle_geometry_source": "transport_boxes_m",
        "obstacle_surfaces_available": True,
        "obstacle_component_count": 1,
        "obstacle_geometry_warning": None,
    }


def test_blocked_cells_without_geometry_do_not_create_synthetic_surfaces() -> None:
    """Navigation occupancy alone must not invent uniform obstacle source faces."""
    env = EnvironmentConfig(size_x=3.0, size_y=3.0, size_z=3.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(3, 3),
        blocked_cells=((1, 1),),
    )

    patches = build_surface_patch_dictionary(env, grid, spacing=1.0)

    kinds = np.asarray(patches.kinds, dtype=str)
    assert set(kinds.tolist()) == {"floor", "wall", "ceiling"}
    assert np.sum(patches.areas_m2[kinds == "floor"]) == pytest.approx(9.0)
    assert patches.obstacle_geometry_source == "blocked_cells_only"
    assert not patches.obstacle_surfaces_available
    assert patches.obstacle_component_count == 0
    assert patches.obstacle_geometry_warning is not None


@pytest.mark.parametrize("spacing", [2.0, 3.0])
def test_floor_area_is_exact_when_spacing_misses_obstacle_edges(
    spacing: float,
) -> None:
    """Floor edges should split at obstacle bounds before footprint removal."""
    env = EnvironmentConfig(size_x=3.0, size_y=3.0, size_z=3.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(3, 3),
        blocked_cells=((0, 0),),
        transport_boxes_m=((0.0, 0.0, 0.0, 1.0, 1.0, 2.0),),
    )

    patches = build_surface_patch_dictionary(env, grid, spacing=spacing)

    kinds = np.asarray(patches.kinds, dtype=str)
    assert np.sum(patches.areas_m2[kinds == "floor"]) == pytest.approx(8.0)


def test_patch_count_upper_bound_covers_constructed_dictionary() -> None:
    """The lightweight count estimate must conservatively cover construction."""
    env = EnvironmentConfig(size_x=3.0, size_y=3.0, size_z=3.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(3, 3),
        blocked_cells=((0, 0), (1, 1)),
        transport_boxes_m=(
            (0.1, 0.2, 0.0, 0.9, 0.8, 1.2),
            (1.1, 1.2, 0.0, 1.8, 1.9, 2.1),
        ),
    )

    upper_bound = estimate_surface_patch_count_upper_bound(
        env,
        grid,
        spacing=(2.0, 1.4, 0.8),
        obstacle_height_m=2.0,
    )
    patches = build_surface_patch_dictionary(
        env,
        grid,
        spacing=(2.0, 1.4, 0.8),
        obstacle_height_m=2.0,
    )

    assert upper_bound >= patches.patch_count


def test_patch_count_estimate_does_not_allocate_axis_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Very fine spacing should be countable before any coordinate-grid allocation."""

    def _fail_axis_allocation(*_args: object, **_kwargs: object) -> object:
        """Fail if the lightweight count path allocates coordinate edges."""
        raise AssertionError("axis edges should not be allocated")

    monkeypatch.setattr(
        surface_patches_module,
        "_axis_edges",
        _fail_axis_allocation,
    )

    upper_bound = estimate_surface_patch_count_upper_bound(
        EnvironmentConfig(size_x=2.0, size_y=2.0, size_z=2.0),
        None,
        spacing=1.0e-6,
    )

    assert upper_bound == 24_000_000_000_000


def test_batched_component_neighbors_match_scalar_grid_oracle() -> None:
    """Vectorized component lookup should equal exact four-neighbor topology."""
    cells = np.asarray(
        [[3, 1], [0, 0], [1, 2], [1, 1], [2, 1]],
        dtype=np.int64,
    )
    lookup = {tuple(cell): index for index, cell in enumerate(cells.tolist())}

    for axis in (0, 1):
        expected = []
        for component_index, cell in enumerate(cells.tolist()):
            neighbor = list(cell)
            neighbor[axis] += 1
            neighbor_index = lookup.get(tuple(neighbor))
            if neighbor_index is not None:
                expected.append((component_index, neighbor_index))
        actual = _grid_component_neighbor_pairs(
            cells,
            grid_shape=(4, 4),
            axis=axis,
        )

        assert {tuple(pair) for pair in actual.tolist()} == set(expected)


def test_adjacent_component_tops_have_exact_graph_tv_edges() -> None:
    """Known component tops should connect every shared physical edge once."""
    env = EnvironmentConfig(size_x=3.0, size_y=3.0, size_z=2.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(3, 3),
        blocked_cells=((0, 0), (1, 0), (1, 1)),
        transport_boxes_m=(
            (0.25, 0.25, 0.0, 1.25, 1.25, 1.0),
            (1.25, 0.25, 0.0, 2.25, 1.25, 1.0),
            (1.25, 1.25, 0.0, 2.25, 2.25, 1.0),
        ),
    )

    patches = build_surface_patch_dictionary(
        env,
        grid,
        spacing=1.0,
        obstacle_height_m=1.0,
    )

    kinds = np.asarray(patches.kinds, dtype=str)
    face_ids = np.asarray(patches.face_ids, dtype=str)
    edges = np.asarray(patches.adjacency_edges, dtype=np.int64)
    top_edges = (kinds[edges[:, 0]] == "obstacle_top") & (
        kinds[edges[:, 1]] == "obstacle_top"
    )
    top_face_pairs = {
        frozenset((face_ids[left], face_ids[right])) for left, right in edges[top_edges]
    }

    assert np.count_nonzero(top_edges) == 2
    assert top_face_pairs == {
        frozenset(("transport_component_0_z1", "transport_component_1_z1")),
        frozenset(("transport_component_1_z1", "transport_component_2_z1")),
    }
    assert patches.shared_edge_lengths_m[top_edges] == pytest.approx([1.0, 1.0])


def test_coplanar_component_sides_connect_aligned_subpatch_boundaries() -> None:
    """Contiguous side faces should connect each aligned vertical subpatch once."""
    env = EnvironmentConfig(size_x=3.0, size_y=3.0, size_z=2.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(3, 3),
        blocked_cells=((0, 0), (1, 0)),
        transport_boxes_m=(
            (0.25, 0.5, 0.0, 1.25, 1.5, 1.0),
            (1.25, 0.5, 0.0, 2.25, 1.5, 1.0),
        ),
    )

    patches = build_surface_patch_dictionary(
        env,
        grid,
        spacing=0.5,
        obstacle_height_m=1.0,
    )

    kinds = np.asarray(patches.kinds, dtype=str)
    face_ids = np.asarray(patches.face_ids, dtype=str)
    edges = np.asarray(patches.adjacency_edges, dtype=np.int64)
    first_component = np.char.startswith(face_ids, "transport_component_0_")
    second_component = np.char.startswith(face_ids, "transport_component_1_")
    cross_side_edges = (
        (kinds[edges[:, 0]] == "obstacle_side")
        & (kinds[edges[:, 1]] == "obstacle_side")
        & (
            (first_component[edges[:, 0]] & second_component[edges[:, 1]])
            | (second_component[edges[:, 0]] & first_component[edges[:, 1]])
        )
    )
    selected_edges = edges[cross_side_edges]
    selected_pairs = [
        frozenset((face_ids[left], face_ids[right])) for left, right in selected_edges
    ]
    lower_z = np.sort(patches.centers_xyz[selected_edges[:, 0], 2])
    upper_z = np.sort(patches.centers_xyz[selected_edges[:, 1], 2])

    assert np.count_nonzero(cross_side_edges) == 4
    assert (
        selected_pairs.count(
            frozenset(
                (
                    "transport_component_0_y0",
                    "transport_component_1_y0",
                )
            )
        )
        == 2
    )
    assert (
        selected_pairs.count(
            frozenset(
                (
                    "transport_component_0_y1",
                    "transport_component_1_y1",
                )
            )
        )
        == 2
    )
    assert patches.shared_edge_lengths_m[cross_side_edges] == pytest.approx(
        np.full(4, 0.5)
    )
    assert lower_z == pytest.approx([0.25, 0.25, 0.75, 0.75])
    assert upper_z == pytest.approx([0.25, 0.25, 0.75, 0.75])


def test_component_top_side_and_floor_side_edges_use_physical_lengths() -> None:
    """A floor-mounted box should connect its top and footprint perimeter exactly."""
    env = EnvironmentConfig(size_x=3.0, size_y=3.0, size_z=3.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(3, 3),
        blocked_cells=((1, 1),),
        transport_boxes_m=((1.0, 1.0, 0.0, 1.4, 1.6, 1.2),),
    )

    patches = build_surface_patch_dictionary(env, grid, spacing=0.3)
    kinds = np.asarray(patches.kinds, dtype=str)
    edges = patches.adjacency_edges

    def _edge_length(first: str, second: str) -> float:
        """Return total graph length between two patch kinds."""
        selected = (
            ((kinds[edges[:, 0]] == first) & (kinds[edges[:, 1]] == second))
            | ((kinds[edges[:, 0]] == second) & (kinds[edges[:, 1]] == first))
        )
        return float(np.sum(patches.shared_edge_lengths_m[selected]))

    expected_perimeter = 2.0 * (0.4 + 0.6)
    assert _edge_length("obstacle_top", "obstacle_side") == pytest.approx(
        expected_perimeter
    )
    assert _edge_length("floor", "obstacle_side") == pytest.approx(
        expected_perimeter
    )


def test_nearby_non_touching_components_are_not_tv_neighbors() -> None:
    """Center proximity must not connect surfaces separated by a physical gap."""
    env = EnvironmentConfig(size_x=4.0, size_y=4.0, size_z=3.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(4, 4),
        blocked_cells=((1, 1), (2, 1)),
        transport_boxes_m=(
            (1.0, 1.0, 0.0, 2.0, 2.0, 1.0),
            (2.01, 1.0, 0.0, 3.01, 2.0, 1.0),
        ),
    )

    patches = build_surface_patch_dictionary(env, grid, spacing=0.5)
    face_ids = np.asarray(patches.face_ids, dtype=str)
    edges = patches.adjacency_edges
    first_component = np.char.startswith(face_ids, "transport_component_0_")
    second_component = np.char.startswith(face_ids, "transport_component_1_")
    between_components = (
        (first_component[edges[:, 0]] & second_component[edges[:, 1]])
        | (second_component[edges[:, 0]] & first_component[edges[:, 1]])
    )

    assert not np.any(between_components)
