"""Tests for surface-constrained random source placement."""

from __future__ import annotations

import numpy as np

from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid
from measurement.source_surfaces import (
    build_surface_candidate_sources,
    generate_surface_sources,
    is_allowed_source_surface_position,
    project_positions_to_allowed_surfaces,
    source_surface_kind_counts,
    source_surface_kind,
    source_surface_kinds,
)


def test_generate_surface_sources_never_places_sources_in_air_or_obstacles() -> None:
    """Random source generation should only place sources on allowed surfaces."""
    env = EnvironmentConfig(size_x=10.0, size_y=20.0, size_z=10.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(10, 20),
        blocked_cells=((3, 4), (4, 4), (3, 5)),
    )
    sources = generate_surface_sources(
        env=env,
        obstacle_grid=grid,
        isotopes=("Cs-137", "Co-60", "Eu-154"),
        intensity_cps_1m=30000.0,
        rng=np.random.default_rng(4),
        count=200,
        obstacle_height_m=2.0,
    )

    assert len(sources) == 200
    for source in sources:
        assert is_allowed_source_surface_position(
            source.position,
            env,
            grid,
            obstacle_height_m=2.0,
        )


def test_source_surface_kind_rejects_air_and_obstacle_interior() -> None:
    """Surface classification should reject unsupported 3D positions."""
    env = EnvironmentConfig(size_x=10.0, size_y=20.0, size_z=10.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(10, 20),
        blocked_cells=((3, 4),),
    )

    assert source_surface_kind((5.0, 5.0, 5.0), env, grid) is None
    assert source_surface_kind((3.5, 4.5, 1.0), env, grid) is None
    assert source_surface_kind((3.5, 4.5, 2.0), env, grid) == "obstacle_top"
    assert source_surface_kind((3.0, 4.5, 1.0), env, grid) == "obstacle_side"
    assert source_surface_kind((1.5, 1.5, 0.0), env, grid) == "floor"
    assert source_surface_kind((5.0, 20.0, 4.0), env, grid) == "wall"


def test_source_surface_kinds_matches_scalar_classification() -> None:
    """Vectorized surface classification should match the scalar oracle."""
    env = EnvironmentConfig(size_x=4.0, size_y=4.0, size_z=3.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(4, 4),
        blocked_cells=((2, 2),),
    )
    points = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 4.0, 2.0],
            [2.5, 2.5, 1.0],
            [2.0, 2.5, 0.5],
            [2.5, 2.5, 0.5],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )

    vectorized = source_surface_kinds(
        points,
        env,
        grid,
        obstacle_height_m=1.0,
    )
    scalar = np.array(
        [
            source_surface_kind(point, env, grid, obstacle_height_m=1.0)
            for point in points
        ],
        dtype=object,
    )
    counts = source_surface_kind_counts(
        points,
        env,
        grid,
        obstacle_height_m=1.0,
    )

    assert vectorized.tolist() == scalar.tolist()
    assert counts["floor"] == 1
    assert counts["wall"] == 1
    assert counts["obstacle_top"] == 1
    assert counts["obstacle_side"] == 1
    assert counts["off_surface"] == 2


def test_build_surface_candidate_sources_only_returns_allowed_surfaces() -> None:
    """Surface candidate generation should cover known room and obstacle surfaces."""
    env = EnvironmentConfig(size_x=2.0, size_y=2.0, size_z=2.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(2, 2),
        blocked_cells=((1, 1),),
    )

    candidates = build_surface_candidate_sources(
        env,
        grid,
        spacing=(0.5, 0.5, 0.5),
        obstacle_height_m=1.0,
    )

    assert candidates.shape[1] == 3
    assert candidates.shape[0] > 0
    assert not np.any(np.all(np.isclose(candidates, (1.5, 1.5, 0.0)), axis=1))
    kinds = {
        source_surface_kind(point, env, grid, obstacle_height_m=1.0)
        for point in candidates
    }
    assert None not in kinds
    assert "wall" in kinds
    assert "floor" in kinds
    assert "ceiling" in kinds
    assert "obstacle_top" in kinds or "obstacle_side" in kinds


def test_project_positions_to_allowed_surfaces_uses_nearest_known_surface() -> None:
    """Position projection should snap off-surface positions to allowed surfaces."""
    env = EnvironmentConfig(size_x=4.0, size_y=4.0, size_z=3.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(4, 4),
        blocked_cells=((2, 2),),
    )
    positions = np.array(
        [
            [2.5, 2.5, 0.6],
            [3.9, 2.0, 1.0],
            [1.2, 1.4, 1.4],
        ],
        dtype=float,
    )

    projected = project_positions_to_allowed_surfaces(
        positions,
        env,
        grid,
        obstacle_height_m=1.0,
    )

    assert projected.shape == positions.shape
    assert np.allclose(projected[0], [2.5, 2.5, 1.0])
    for point in projected:
        assert is_allowed_source_surface_position(
            point,
            env,
            grid,
            obstacle_height_m=1.0,
        )
