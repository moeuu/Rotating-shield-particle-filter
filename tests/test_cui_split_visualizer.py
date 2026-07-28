"""Tests for CUI split visualization outputs."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib.axes
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D

from measurement.obstacles import ObstacleGrid
from pf.particle_filter import IsotopeParticle, PFConfig
from pf.state import IsotopeState
from visualization.realtime_viz import CUISplitPFVisualizer, PFFrame, build_frame_from_pf


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
        particle_positions={"Cs-137": np.array([[0.5, 0.5, 0.5], [2.5, 2.5, 0.5]])},
        particle_weights={"Cs-137": np.array([0.2, 0.8], dtype=float)},
        estimated_sources={"Cs-137": np.array([[2.0, 2.0, 0.5]], dtype=float)},
        estimated_strengths={"Cs-137": np.array([1000.0], dtype=float)},
        spectrum_energy_keV=np.array([100.0, 102.0, 104.0], dtype=float),
        spectrum_counts=np.array([1.0, 4.0, 2.0], dtype=float),
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
    assert "RA-L experiment overview" in (tmp_path / "index.html").read_text(
        encoding="utf-8",
    )
    assert (tmp_path / "latest_robot_2d.png").exists()
    assert (tmp_path / "latest_experiment_overview.png").exists()
    assert (tmp_path / "latest_pf_3d.png").exists()
    assert (tmp_path / "latest_spectrum.png").exists()
    assert (tmp_path / "robot_2d_step_0003.png").exists()
    assert (tmp_path / "experiment_overview_step_0003.png").exists()
    assert (tmp_path / "pf_3d_step_0003.png").exists()
    assert (tmp_path / "spectrum_step_0003.png").exists()


def test_cui_split_visualizer_preserves_metric_aspect(
    tmp_path,
    monkeypatch,
) -> None:
    """The paper overview and 3-D view should use metric-equal axes."""
    aspect_calls: list[
        tuple[tuple[float, float], tuple[float, float], object]
    ] = []
    box_aspect_calls: list[tuple[float, float, float]] = []
    xtick_calls: list[np.ndarray] = []
    ytick_calls: list[np.ndarray] = []
    original_set_aspect = matplotlib.axes.Axes.set_aspect
    original_set_box_aspect = Axes3D.set_box_aspect
    original_set_xticks = matplotlib.axes.Axes.set_xticks
    original_set_yticks = matplotlib.axes.Axes.set_yticks

    def record_set_aspect(self, aspect, *args, **kwargs):
        """Record 2-D aspect requests before delegating to Matplotlib."""
        aspect_calls.append((self.get_xlim(), self.get_ylim(), aspect))
        return original_set_aspect(self, aspect, *args, **kwargs)

    def record_set_box_aspect(self, aspect, *args, **kwargs):
        """Record 3-D box aspect requests before delegating to Matplotlib."""
        if aspect is not None:
            box_aspect_calls.append(tuple(float(value) for value in aspect))
        return original_set_box_aspect(self, aspect, *args, **kwargs)

    def record_set_xticks(self, ticks, *args, **kwargs):
        """Record x-axis ticks before delegating to Matplotlib."""
        xtick_calls.append(np.asarray(ticks, dtype=float))
        return original_set_xticks(self, ticks, *args, **kwargs)

    def record_set_yticks(self, ticks, *args, **kwargs):
        """Record y-axis ticks before delegating to Matplotlib."""
        ytick_calls.append(np.asarray(ticks, dtype=float))
        return original_set_yticks(self, ticks, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_aspect", record_set_aspect)
    monkeypatch.setattr(Axes3D, "set_box_aspect", record_set_box_aspect)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_xticks", record_set_xticks)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_yticks", record_set_yticks)

    frame = PFFrame(
        step_index=0,
        time=0.0,
        robot_position=np.array([5.0, 10.0, 0.5], dtype=float),
        robot_orientation=None,
        RFe=np.eye(3, dtype=float),
        RPb=np.eye(3, dtype=float),
        duration=1.0,
        particle_positions={"Cs-137": np.array([[5.0, 10.0, 5.0]])},
        particle_weights={"Cs-137": np.array([1.0], dtype=float)},
        estimated_sources={"Cs-137": np.array([[5.0, 10.0, 5.0]])},
        estimated_strengths={"Cs-137": np.array([1000.0], dtype=float)},
    )
    visualizer = CUISplitPFVisualizer(
        isotopes=["Cs-137"],
        output_dir=tmp_path,
        world_bounds=(0.0, 10.0, 0.0, 20.0, 0.0, 10.0),
        true_sources={"Cs-137": np.array([[5.0, 10.0, 5.0]], dtype=float)},
    )

    visualizer.update(frame)

    assert any(
        np.allclose(xlim, (0.0, 10.0))
        and np.allclose(ylim, (0.0, 20.0))
        and aspect == "equal"
        for xlim, ylim, aspect in aspect_calls
    )
    assert any(
        np.allclose(xlim, (0.0, 10.0))
        and np.allclose(ylim, (0.0, 10.0))
        and aspect == "equal"
        for xlim, ylim, aspect in aspect_calls
    )
    assert any(
        np.allclose(aspect, (10.0, 20.0, 10.0))
        for aspect in box_aspect_calls
    )
    assert any(np.allclose(ticks, np.arange(0.0, 10.1, 2.0)) for ticks in xtick_calls)
    assert any(np.allclose(ticks, np.arange(0.0, 20.1, 2.0)) for ticks in ytick_calls)


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


def test_build_frame_from_pf_uses_continuous_surface_state_positions() -> None:
    """PF screenshots must resolve XYZ from the authoritative chart state."""
    config = PFConfig(
        num_particles=2,
        use_gpu=False,
        variable_cardinality=False,
        init_num_sources=(1, 1),
    )
    filt = SimpleNamespace(config=config)
    resolved_positions = {
        0: np.array([1.0, 2.0, 3.0], dtype=float),
        1: np.array([2.0, 3.0, 4.0], dtype=float),
    }
    filt.continuous_state_positions = lambda state: np.vstack(
        [
            resolved_positions[int(chart_id)]
            for chart_id in state.surface_chart_ids
        ]
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                strengths=np.array([100.0], dtype=float),
                surface_chart_ids=np.array([0], dtype=np.int64),
                surface_uv=np.array([[0.25, 0.75]], dtype=float),
            ),
            log_weight=np.log(0.5),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                strengths=np.array([110.0], dtype=float),
                surface_chart_ids=np.array([1], dtype=np.int64),
                surface_uv=np.array([[0.75, 0.25]], dtype=float),
            ),
            log_weight=np.log(0.5),
        ),
    ]
    filt.continuous_weights = np.array([0.5, 0.5], dtype=float)
    pf = SimpleNamespace(
        filters={"Cs-137": filt},
        estimate_all=lambda: {
            "Cs-137": (
                np.zeros((0, 3), dtype=float),
                np.zeros(0, dtype=float),
            )
        },
    )

    frame = build_frame_from_pf(
        pf,
        step_index=0,
        time_sec=0.0,
        detector_position=np.array([0.0, 0.0, 0.0], dtype=float),
        live_time_s=1.0,
    )

    displayed = frame.particle_positions["Cs-137"]
    assert displayed.shape == (2, 3)
    assert np.allclose(displayed, [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    assert np.allclose(frame.particle_weights["Cs-137"], [0.5, 0.5])


def test_build_frame_uses_one_official_joint_projection_and_surface_medoids() -> None:
    """Live frames must not reconstruct independent isotope point estimates."""
    chart_positions = {
        0: np.array([0.0, 0.0, 0.0], dtype=float),
        1: np.array([2.0, 0.0, 0.0], dtype=float),
        2: np.array([0.0, 2.0, 0.0], dtype=float),
    }

    def make_filter(chart_ids: list[int]) -> SimpleNamespace:
        """Return a minimal filter with one continuous-surface particle."""
        filt = SimpleNamespace()
        filt.continuous_state_positions = lambda state: np.vstack(
            [
                chart_positions[int(chart_id)]
                for chart_id in state.surface_chart_ids
            ]
        )
        state = IsotopeState(
            num_sources=len(chart_ids),
            strengths=np.full(len(chart_ids), 100.0, dtype=float),
            surface_chart_ids=np.asarray(chart_ids, dtype=np.int64),
            surface_uv=np.full((len(chart_ids), 2), 0.5, dtype=float),
        )
        filt.continuous_particles = [
            IsotopeParticle(state=state, log_weight=0.0)
        ]
        filt.continuous_weights = np.ones(1, dtype=float)
        return filt

    official = {
        "Cs-137": (
            np.array([[9.0, 8.0, 7.0]], dtype=float),
            np.array([901.0], dtype=float),
        ),
        "Co-60": (
            np.array([[6.0, 5.0, 4.0]], dtype=float),
            np.array([902.0], dtype=float),
        ),
    }
    calls: list[str] = []

    def estimate_all() -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Return a recognizable coherent joint-posterior projection."""
        calls.append("estimate_all")
        return official

    pf = SimpleNamespace(
        filters={
            "Cs-137": make_filter([0, 1]),
            "Co-60": make_filter([2]),
        },
        estimate_all=estimate_all,
    )

    frame = build_frame_from_pf(
        pf,
        step_index=0,
        time_sec=0.0,
        detector_position=np.zeros(3, dtype=float),
        live_time_s=1.0,
    )

    assert calls == ["estimate_all"]
    for isotope, (positions, strengths) in official.items():
        assert np.array_equal(frame.estimated_sources[isotope], positions)
        assert np.array_equal(frame.estimated_strengths[isotope], strengths)
    cs_representative = frame.particle_representative_positions["Cs-137"]
    assert np.array_equal(cs_representative, [[0.0, 0.0, 0.0]])
    assert not np.array_equal(cs_representative, [[1.0, 0.0, 0.0]])
