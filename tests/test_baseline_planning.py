"""Tests for baseline PF measurement planning."""

import numpy as np

from baseline_pf.planning import generate_measurement_positions, measurement_count
from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid


def test_measurement_count_rounds_down() -> None:
    """Measurement count should use full 30 s blocks with a minimum of one."""
    assert measurement_count(30.0) == 1
    assert measurement_count(59.0) == 1
    assert measurement_count(60.0) == 2


def test_generate_measurement_positions_avoids_obstacles() -> None:
    """Generated positions should not fall inside blocked grid cells."""
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(3, 3),
        blocked_cells=((0, 0), (1, 1)),
    )
    env = EnvironmentConfig(
        size_x=3.0,
        size_y=3.0,
        size_z=1.0,
        detector_position=(0.5, 0.5, 0.0),
    )
    positions = generate_measurement_positions(
        env,
        grid,
        total_time_s=90.0,
        measurement_time_s=30.0,
    )
    assert positions.shape == (3, 3)
    for pos in positions:
        assert grid.is_free(pos)
        assert np.isclose(pos[2], env.detector()[2])
