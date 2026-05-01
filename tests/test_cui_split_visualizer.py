"""Tests for CUI split visualization outputs."""

from __future__ import annotations

import numpy as np

from measurement.obstacles import ObstacleGrid
from visualization.realtime_viz import CUISplitPFVisualizer, PFFrame


def test_cui_split_visualizer_writes_robot_and_pf_views(tmp_path) -> None:
    """The CUI split visualizer should write latest and step-specific PNGs."""
    obstacle_grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(3, 3),
        blocked_cells=((1, 1),),
    )
    frame = PFFrame(
        step_index=3,
        time=12.5,
        robot_position=np.array([1.0, 2.0, 0.5], dtype=float),
        robot_orientation=None,
        RFe=np.eye(3, dtype=float),
        RPb=np.eye(3, dtype=float),
        duration=2.0,
        counts_by_isotope={"Cs-137": 10.0},
        particle_positions={"Cs-137": np.array([[0.5, 0.5, 0.5], [2.5, 2.5, 0.5]])},
        particle_weights={"Cs-137": np.array([0.2, 0.8], dtype=float)},
        estimated_sources={"Cs-137": np.array([[2.0, 2.0, 0.5]], dtype=float)},
        estimated_strengths={"Cs-137": np.array([1000.0], dtype=float)},
        spectrum_energy_keV=np.array([100.0, 102.0, 104.0], dtype=float),
        spectrum_counts=np.array([1.0, 4.0, 2.0], dtype=float),
        spectrum_components_by_isotope={
            "Cs-137": np.array([0.0, 3.0, 0.0], dtype=float),
        },
    )
    visualizer = CUISplitPFVisualizer(
        isotopes=["Cs-137"],
        output_dir=tmp_path,
        world_bounds=(0.0, 3.0, 0.0, 3.0, 0.0, 2.0),
        true_sources={"Cs-137": np.array([[2.0, 2.0, 0.5]], dtype=float)},
        obstacle_grid=obstacle_grid,
    )

    visualizer.update(frame)

    assert (tmp_path / "index.html").exists()
    assert (tmp_path / "latest_robot_2d.png").exists()
    assert (tmp_path / "latest_pf_3d.png").exists()
    assert (tmp_path / "latest_spectrum.png").exists()
    assert (tmp_path / "robot_2d_step_0003.png").exists()
    assert (tmp_path / "pf_3d_step_0003.png").exists()
    assert (tmp_path / "spectrum_step_0003.png").exists()


def test_cui_split_visualizer_tracks_stations_and_path_waypoints(tmp_path) -> None:
    """The CUI view should not confuse repeated rotations with new stations."""
    base_frame = PFFrame(
        step_index=0,
        time=0.0,
        robot_position=np.array([1.0, 1.0, 0.5], dtype=float),
        robot_orientation=None,
        RFe=np.eye(3, dtype=float),
        RPb=np.eye(3, dtype=float),
        duration=1.0,
        counts_by_isotope={},
        particle_positions={"Cs-137": np.zeros((0, 3), dtype=float)},
        particle_weights={"Cs-137": np.zeros(0, dtype=float)},
        estimated_sources={"Cs-137": np.zeros((0, 3), dtype=float)},
        estimated_strengths={"Cs-137": np.zeros(0, dtype=float)},
    )
    visualizer = CUISplitPFVisualizer(
        isotopes=["Cs-137"],
        output_dir=tmp_path,
        world_bounds=(0.0, 4.0, 0.0, 4.0, 0.0, 2.0),
    )
    repeated_frame = base_frame
    moved_frame = PFFrame(
        step_index=2,
        time=2.0,
        robot_position=np.array([3.0, 3.0, 0.5], dtype=float),
        robot_orientation=None,
        RFe=np.eye(3, dtype=float),
        RPb=np.eye(3, dtype=float),
        duration=1.0,
        counts_by_isotope={},
        particle_positions={"Cs-137": np.zeros((0, 3), dtype=float)},
        particle_weights={"Cs-137": np.zeros(0, dtype=float)},
        estimated_sources={"Cs-137": np.zeros((0, 3), dtype=float)},
        estimated_strengths={"Cs-137": np.zeros(0, dtype=float)},
        path_waypoints_xyz=np.array(
            [[1.0, 1.0, 0.5], [2.0, 2.0, 0.5], [3.0, 3.0, 0.5]],
            dtype=float,
        ),
    )

    visualizer.update(base_frame)
    visualizer.update(repeated_frame)
    visualizer.update(moved_frame)

    assert len(visualizer.measurement_points) == 2
    assert visualizer.measurement_visit_counts == [2, 1]
    assert visualizer._unique_path_waypoints().shape[0] == 1
