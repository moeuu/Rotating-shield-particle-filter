"""Tests for truth-safe CUI evaluation rendering."""

from __future__ import annotations

from pathlib import Path
import pickle
import queue

import matplotlib.pyplot as plt
import numpy as np
import pytest
from measurement.obstacles import ObstacleGrid
from runtime.cui import CUIRoute, cui_route_from_records
from runtime.cui_components import write_cui_index
from runtime.measurement_log import MeasurementLogRecord

from visualization.realtime_viz import (
    AsyncCUISplitPFVisualizer,
    CUISplitPFVisualizer,
    PFFrame,
    atomic_copy_file,
    pf_cui_panel_specs,
)


def _route_record(
    step_id: int,
    station_id: int,
    *,
    pose_xyz: tuple[float, float, float],
) -> MeasurementLogRecord:
    """Return one truth-free record for CUI route regression tests."""
    return MeasurementLogRecord(
        step_id=step_id,
        action_id=step_id,
        station_id=station_id,
        detector_pose_xyz=pose_xyz,
        detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        fe_orientation_index=step_id % 8,
        pb_orientation_index=(step_id * 3) % 8,
        live_time_s=1.0,
        travel_time_s=0.0,
        shield_actuation_time_s=0.0,
        energy_bin_edges_keV=np.asarray([0.0, 1.0, 2.0]),
        spectrum_counts=np.asarray([2, 3], dtype=np.int64),
        metadata={"full_spectrum_contract_hash_sha256": "d" * 64},
    )


def _empty_frame(step_id: int, route: CUIRoute) -> PFFrame:
    """Return a minimal PF frame carrying one cumulative route snapshot."""
    return PFFrame(
        step_index=step_id,
        time=float(step_id),
        robot_position=np.asarray(route.current_pose_xyz, dtype=np.float64),
        robot_orientation=None,
        RFe=np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        RPb=np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        duration=1.0,
        particle_positions={"Cs-137": np.zeros((0, 3), dtype=np.float64)},
        particle_weights={"Cs-137": np.zeros(0, dtype=np.float64)},
        estimated_sources={"Cs-137": np.zeros((0, 3), dtype=np.float64)},
        estimated_strengths={"Cs-137": np.zeros(0, dtype=np.float64)},
        cui_route=route,
    )


class _AliveProcess:
    """Stand in for a live renderer process in queue-only tests."""

    def is_alive(self) -> bool:
        """Report that the synthetic renderer process is alive."""
        return True


def test_cui_truth_is_hidden_until_explicit_evaluation_update(
    tmp_path: Path,
) -> None:
    """CUI truth attachment must be explicit and copy evaluation arrays."""
    visualizer = CUISplitPFVisualizer(
        isotopes=["Cs-137"],
        output_dir=tmp_path,
        true_sources={},
        true_strengths={},
    )
    assert visualizer.true_sources == {}
    hidden_html = visualizer.index_path.read_text(encoding="utf-8")
    assert "truth" not in hidden_html.lower()
    reference_index = write_cui_index(
        tmp_path / "reference-shell",
        pf_cui_panel_specs(),
        title="Rotating Shield PF CUI View",
        refresh_interval_ms=2000,
    )
    assert hidden_html == reference_index.read_text(encoding="utf-8")
    panels = pf_cui_panel_specs()
    assert tuple(panel.panel_id for panel in panels[2:-1]) == (
        "pf-particle-posterior",
        "pf-particle-labels",
    )
    assert tuple(panel.title for panel in panels[2:-1]) == (
        "PF particle posterior 3D",
        "PF particle posterior with source labels",
    )
    assert tuple(panel.image_filename for panel in panels[2:-1]) == (
        "latest_pf_3d.png",
        "latest_pf_3d_labeled.png",
    )
    assert tuple(panel.column_span for panel in panels[2:-1]) == (1, 2)
    assert "grid" not in " ".join(panel.title.lower() for panel in panels[2:-1])
    source = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64)
    strength = np.asarray([400_000.0], dtype=np.float64)

    visualizer.set_truth(
        {"Cs-137": source},
        {"Cs-137": strength},
    )
    source[:] = -1.0
    strength[:] = -1.0

    np.testing.assert_array_equal(
        visualizer.true_sources["Cs-137"],
        np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        visualizer.true_strengths["Cs-137"],
        np.asarray([400_000.0], dtype=np.float64),
    )
    assert (
        "truth: visible (evaluation overlay only; not provided to PF/planner)"
        in visualizer.index_path.read_text(encoding="utf-8")
    )


def test_cui_scene_preserves_asymmetric_obstacle_xy_order(tmp_path: Path) -> None:
    """Canonical scene footprints must not transpose asymmetric grid cells."""
    obstacle_grid = ObstacleGrid(
        origin=(0.25, 1.5),
        cell_size=0.5,
        grid_shape=(10, 12),
        blocked_cells=((6, 3),),
    )
    visualizer = CUISplitPFVisualizer(
        isotopes=["Cs-137"],
        output_dir=tmp_path,
        world_bounds=(0.0, 10.0, 0.0, 20.0, 0.0, 3.0),
        obstacle_grid=obstacle_grid,
    )

    expected_xy = np.asarray(
        [[3.25, 3.0], [3.75, 3.0], [3.75, 3.5], [3.25, 3.5]]
    )
    np.testing.assert_array_equal(
        visualizer.cui_scene.obstacle_footprints_xy[0],
        expected_xy,
    )
    figure, axis = plt.subplots()
    try:
        visualizer._draw_obstacles_2d(axis)
        np.testing.assert_array_equal(
            axis.patches[0].get_xy()[:4],
            expected_xy,
        )
    finally:
        plt.close(figure)


def test_cui_writes_plain_and_neighborhood_labeled_pf_images(
    tmp_path: Path,
) -> None:
    """CUI output must retain plain PF images and add source-labeled images."""
    visualizer = CUISplitPFVisualizer(
        isotopes=["Co-60"],
        output_dir=tmp_path,
        world_bounds=(0.0, 10.0, 0.0, 10.0, 0.0, 3.0),
        true_sources={
            "Co-60": np.asarray(
                [[1.0, 1.0, 0.5], [5.0, 5.0, 0.5]],
                dtype=float,
            )
        },
        true_strengths={"Co-60": np.asarray([1.0, 1.0], dtype=float)},
        source_label_neighborhood_m=1.0,
    )
    frame = PFFrame(
        step_index=3,
        time=120.0,
        robot_position=np.asarray([2.0, 2.0, 0.5], dtype=float),
        robot_orientation=None,
        RFe=np.asarray([1.0, 0.0, 0.0], dtype=float),
        RPb=np.asarray([0.0, 1.0, 0.0], dtype=float),
        duration=30.0,
        particle_positions={"Co-60": np.zeros((0, 3), dtype=float)},
        particle_weights={"Co-60": np.zeros(0, dtype=float)},
        estimated_sources={
            "Co-60": np.asarray(
                [
                    [1.2, 1.0, 0.5],
                    [1.4, 1.0, 0.5],
                    [8.0, 8.0, 0.5],
                ],
                dtype=float,
            )
        },
        estimated_strengths={
            "Co-60": np.asarray([1.0, 1.0, 1.0], dtype=float)
        },
        cui_route=CUIRoute(
            measurement_stations_xyz=np.asarray([[2.0, 2.0, 0.5]]),
            measurement_station_ids=np.asarray([0], dtype=np.int64),
            measurement_step_ids=np.asarray([3], dtype=np.int64),
            measurement_visit_counts=np.asarray([1], dtype=np.int64),
            current_detector_position_xyz=np.asarray([2.0, 2.0, 0.5]),
            latest_step_id=3,
        ),
    )

    truth_entries, estimate_entries = visualizer._source_label_entries(
        frame,
        "Co-60",
    )
    assert [label for _, label in truth_entries] == ["Co-1 T", "Co-2 T"]
    assert [label for _, label in estimate_entries] == [
        "Co-1 E1",
        "Co-1 E2",
        "Co remote-1",
    ]

    visualizer.update(frame)

    for latest_path in (
        visualizer.latest_robot_path,
        visualizer.latest_overview_path,
        visualizer.latest_pf_path,
        visualizer.latest_pf_labeled_path,
        visualizer.latest_spectrum_path,
    ):
        assert latest_path.is_file()
    assert not tuple(tmp_path.glob("*_step_*.png"))
    assert not tuple(tmp_path.glob(".*.render.png"))
    assert "latest_pf_3d_labeled.png" in visualizer.index_path.read_text(
        encoding="utf-8"
    )


def test_cui_step_history_requires_explicit_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """History mode should retain each rendered panel only when requested."""
    route = cui_route_from_records(
        (_route_record(0, 0, pose_xyz=(1.0, 2.0, 0.5)),)
    )
    visualizer = CUISplitPFVisualizer(
        isotopes=["Cs-137"],
        output_dir=tmp_path,
        save_step_history=True,
    )

    def save_panel(frame: PFFrame, path: Path) -> None:
        """Write one small stand-in panel for output-routing tests."""
        del frame
        path.write_bytes(b"panel")

    def save_pf_panels(
        frame: PFFrame,
        path: Path,
        *,
        labeled_output_path: Path,
    ) -> None:
        """Write small stand-ins for both PF panels."""
        del frame
        path.write_bytes(b"pf")
        labeled_output_path.write_bytes(b"pf-labeled")

    monkeypatch.setattr(visualizer, "_save_robot_2d", save_panel)
    monkeypatch.setattr(visualizer, "_save_experiment_overview", save_panel)
    monkeypatch.setattr(visualizer, "_save_pf_3d", save_pf_panels)
    monkeypatch.setattr(visualizer, "_save_spectrum", save_panel)

    visualizer.update(_empty_frame(4, route))

    expected_history = {
        "robot_2d_step_0004.png",
        "experiment_overview_step_0004.png",
        "pf_3d_step_0004.png",
        "pf_3d_labeled_step_0004.png",
        "spectrum_step_0004.png",
    }
    assert {path.name for path in tmp_path.glob("*_step_*.png")} == expected_history
    for filename in (
        "latest_robot_2d.png",
        "latest_experiment_overview.png",
        "latest_pf_3d.png",
        "latest_pf_3d_labeled.png",
        "latest_spectrum.png",
    ):
        assert (tmp_path / filename).is_file()


def test_latest_image_copy_is_atomic_on_copy_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed latest-image copy must preserve the previously published file."""
    source = tmp_path / "step.png"
    target = tmp_path / "latest.png"
    source.write_bytes(b"complete-new-image")
    target.write_bytes(b"previous-image")

    def fail_after_partial_copy(source_handle: object, target_handle: object) -> None:
        """Write an incomplete staging file before simulating copy failure."""
        del source_handle
        target_handle.write(b"partial")
        raise OSError("synthetic copy failure")

    monkeypatch.setattr(
        "runtime.artifacts.shutil.copyfileobj",
        fail_after_partial_copy,
    )

    with pytest.raises(OSError, match="synthetic copy failure"):
        atomic_copy_file(source, target)

    assert target.read_bytes() == b"previous-image"
    assert not tuple(tmp_path.glob(".latest.png.*.tmp"))


def test_cui_route_keeps_same_pose_stations_and_redraw_is_idempotent(
    tmp_path: Path,
) -> None:
    """Distinct station IDs at one pose must survive posterior-only redraws."""
    records = (
        _route_record(0, 0, pose_xyz=(1.0, 1.0, 0.5)),
        _route_record(1, 0, pose_xyz=(1.0, 1.0, 0.5)),
        _route_record(2, 1, pose_xyz=(1.0, 1.0, 0.5)),
    )
    route = cui_route_from_records(records)
    visualizer = CUISplitPFVisualizer(
        isotopes=["Cs-137"],
        output_dir=tmp_path,
    )

    visualizer._apply_route(route)
    visualizer._apply_route(route)

    assert len(visualizer.measurement_points) == 2
    np.testing.assert_array_equal(
        visualizer.measurement_points,
        np.asarray([[1.0, 1.0, 0.5], [1.0, 1.0, 0.5]]),
    )
    assert visualizer.measurement_station_ids == [0, 1]
    assert visualizer.measurement_steps == [0, 2]
    assert visualizer.measurement_visit_counts == [2, 1]
    assert visualizer._station_label(0) == "0(2)"
    assert visualizer._station_label(1) == "1"


def test_async_cui_frame_drop_retains_cumulative_route() -> None:
    """Dropping an old frame must retain all visits in the newest snapshot."""
    records = (
        _route_record(0, 0, pose_xyz=(1.0, 1.0, 0.5)),
        _route_record(1, 0, pose_xyz=(1.0, 1.0, 0.5)),
        _route_record(2, 1, pose_xyz=(2.0, 1.0, 0.5)),
    )
    first_route = cui_route_from_records(records[:1])
    latest_route = cui_route_from_records(records)
    visualizer = AsyncCUISplitPFVisualizer.__new__(AsyncCUISplitPFVisualizer)
    visualizer._closed = False
    visualizer._process = _AliveProcess()
    visualizer._queue = queue.Queue(maxsize=1)

    visualizer.update(_empty_frame(0, first_route))
    visualizer.update(_empty_frame(2, latest_route))

    message, payload = visualizer._queue.get_nowait()
    retained_frame = pickle.loads(payload)
    assert message == "frame"
    assert retained_frame.step_index == 2
    np.testing.assert_array_equal(
        retained_frame.cui_route.measurement_station_ids,
        np.asarray([0, 1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        retained_frame.cui_route.measurement_visit_counts,
        np.asarray([2, 1], dtype=np.int64),
    )
