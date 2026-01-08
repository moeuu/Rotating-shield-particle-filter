"""Tests for obstacle grid generation and serialization."""

from pathlib import Path

import numpy as np

from measurement.obstacles import (
    ObstacleGrid,
    generate_obstacle_grid,
    load_or_generate_obstacle_grid,
)


def test_obstacle_grid_roundtrip_and_is_free(tmp_path: Path) -> None:
    """Obstacle grids should round-trip and block expected cells."""
    rng = np.random.default_rng(0)
    grid = generate_obstacle_grid(
        size_x=4.0,
        size_y=4.0,
        cell_size=1.0,
        blocked_fraction=0.5,
        rng=rng,
    )
    path = tmp_path / "layout.json"
    grid.save(path)
    loaded = ObstacleGrid.load(path)
    assert loaded == grid
    assert loaded.blocked_cells
    ix, iy = loaded.blocked_cells[0]
    x = loaded.origin[0] + ix * loaded.cell_size + 0.1
    y = loaded.origin[1] + iy * loaded.cell_size + 0.1
    assert loaded.is_free((x, y, 0.0)) is False
    assert loaded.is_free((-1.0, -1.0, 0.0)) is True


def test_generate_obstacle_grid_respects_keep_free_points() -> None:
    """Keep-free points should never be blocked."""
    rng = np.random.default_rng(1)
    grid = generate_obstacle_grid(
        size_x=3.0,
        size_y=3.0,
        cell_size=1.0,
        blocked_fraction=0.6,
        rng=rng,
        keep_free_points=[(0.2, 0.2)],
    )
    assert (0, 0) not in grid.blocked_cells


def test_load_or_generate_obstacle_grid_creates_file(tmp_path: Path) -> None:
    """Missing obstacle layouts should be generated and saved."""
    path = tmp_path / "generated.json"
    grid = load_or_generate_obstacle_grid(
        path,
        size_x=2.0,
        size_y=2.0,
        cell_size=1.0,
        blocked_fraction=0.5,
        rng_seed=0,
    )
    assert path.exists()
    loaded = ObstacleGrid.load(path)
    assert loaded == grid
