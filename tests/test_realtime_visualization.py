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
from runtime.cui_truth_overlay import CUITruthOverlaySocketServer
from runtime.measurement_log import MeasurementLogRecord

from visualization import realtime_viz
from visualization.realtime_viz import (
    AsyncCUISplitPFVisualizer,
    CUISplitPFVisualizer,
    PFFrame,
    atomic_copy_file,
    build_frame_from_pf,
    pf_cui_panel_specs,
)
from visualization.obstacle_geometry import axis_aligned_box_faces


def test_frame_builder_rejects_legacy_estimate_interfaces() -> None:
    """Visualization must require its one canonical estimator snapshot API."""

    class LegacyOnlyPF:
        """Expose only the retired visualization estimate method."""

        filters: dict[str, object] = {}

        def estimates(self) -> dict[str, object]:
            """Return an empty retired estimate payload."""
            return {}

    with pytest.raises(TypeError, match="must implement visualization_estimates"):
        build_frame_from_pf(
            LegacyOnlyPF(),
            0,
            0.0,
            detector_position=np.zeros(3, dtype=np.float64),
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
        robot_position=np.asarray(
            route.current_detector_position_xyz,
            dtype=np.float64,
        ),
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


class _ControlledProcess:
    """Provide deterministic process lifecycle hooks for protocol tests."""

    def __init__(self, join_hook: object | None = None) -> None:
        """Create a live process that can emit statuses while joining."""
        self._alive = True
        self.exitcode: int | None = None
        self.join_hook = join_hook
        self.terminate_calls = 0
        self.join_calls = 0

    def is_alive(self) -> bool:
        """Return the synthetic process liveness state."""
        return self._alive

    def terminate(self) -> None:
        """Record forced termination and expose a signal-like exit code."""
        self.terminate_calls += 1
        self._alive = False
        self.exitcode = -15

    def join(self, timeout: float | None = None) -> None:
        """Complete the worker and invoke its deterministic status hook."""
        del timeout
        self.join_calls += 1
        if not self._alive:
            return
        hook = self.join_hook
        if callable(hook):
            hook()
        self._alive = False
        self.exitcode = 0


def _queue_only_async_visualizer() -> AsyncCUISplitPFVisualizer:
    """Return a started synthetic async visualizer without a child process."""
    visualizer = AsyncCUISplitPFVisualizer.__new__(AsyncCUISplitPFVisualizer)
    visualizer._closed = False
    visualizer._process = _ControlledProcess()
    visualizer._queue = queue.Queue(maxsize=2)
    visualizer._status_queue = queue.Queue()
    visualizer._run_token = "test-run"
    visualizer._next_operation_id = 0
    visualizer._last_enqueued_operation_id = -1
    visualizer._last_acknowledged_operation_id = -1
    visualizer._operation_kinds = {}
    visualizer._ready_acknowledged = True
    visualizer._close_operation_id = None
    visualizer._close_acknowledged = False
    visualizer._worker_error = None
    return visualizer


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


def test_async_renderer_loads_truth_directly_from_owner_socket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The renderer worker must consume truth without a PF-frame message."""
    endpoint = tmp_path / "cui-truth.sock"
    server = CUITruthOverlaySocketServer(
        endpoint,
        {
            "schema_version": 1,
            "semantics": "evaluation_cui_overlay_only_not_estimator_input",
            "true_sources": {"Cs-137": [[1.0, 2.0, 0.5]]},
            "true_strengths": {"Cs-137": [400_000.0]},
        },
    )
    instances: list[object] = []

    class _RendererProbe:
        """Capture private truth received inside the renderer worker."""

        def __init__(self, **kwargs: object) -> None:
            """Record truth-free renderer construction arguments."""
            self.kwargs = dict(kwargs)
            self.true_sources: dict[str, np.ndarray] = {}
            self.true_strengths: dict[str, np.ndarray] = {}
            instances.append(self)

        def set_truth(
            self,
            true_sources: dict[str, np.ndarray],
            true_strengths: dict[str, np.ndarray],
        ) -> None:
            """Record the overlay attached directly by the worker."""
            self.true_sources = dict(true_sources)
            self.true_strengths = dict(true_strengths)

    monkeypatch.setattr(realtime_viz, "CUISplitPFVisualizer", _RendererProbe)
    frame_queue: queue.Queue[tuple[str, int, object]] = queue.Queue()
    status_queue: queue.Queue[tuple[str, int, str, str]] = queue.Queue()
    frame_queue.put(("close", 0, None))
    try:
        realtime_viz._async_cui_split_worker(
            {
                "isotopes": ["Cs-137"],
                "output_dir": tmp_path / "cui",
                "truth_overlay_socket_path": endpoint,
            },
            frame_queue,
            status_queue,
            "renderer-run",
        )
    finally:
        server.close()

    probe = instances[0]
    assert "truth_overlay_socket_path" not in probe.kwargs
    np.testing.assert_array_equal(
        probe.true_sources["Cs-137"],
        np.asarray([[1.0, 2.0, 0.5]]),
    )
    np.testing.assert_array_equal(
        probe.true_strengths["Cs-137"],
        np.asarray([400_000.0]),
    )
    assert status_queue.get_nowait() == (
        "ready",
        -1,
        "startup",
        "renderer-run",
    )
    assert status_queue.get_nowait() == (
        "closed",
        0,
        "close",
        "renderer-run",
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

    expected_xy = np.asarray([[3.25, 3.0], [3.75, 3.0], [3.75, 3.5], [3.25, 3.5]])
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
    figure, axis = plt.subplots()
    try:
        visualizer._draw_navigation_occupancy_2d(axis)
        occupancy = axis.patches[0]
        assert occupancy.get_x() == pytest.approx(3.25)
        assert occupancy.get_y() == pytest.approx(3.0)
        assert occupancy.get_width() == pytest.approx(0.5)
        assert occupancy.get_height() == pytest.approx(0.5)
    finally:
        plt.close(figure)


def test_cui_3d_obstacles_preserve_physical_component_height(
    tmp_path: Path,
) -> None:
    """A CUI overview must render exact component heights, not floor patches."""
    component = (1.2, 2.3, 0.4, 1.8, 3.1, 2.7)
    obstacle_grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(5, 5),
        blocked_cells=((1, 2),),
        transport_boxes_m=(component,),
    )
    visualizer = CUISplitPFVisualizer(
        isotopes=["Cs-137"],
        output_dir=tmp_path,
        world_bounds=(0.0, 5.0, 0.0, 5.0, 0.0, 3.0),
        obstacle_grid=obstacle_grid,
    )

    faces = axis_aligned_box_faces(visualizer.cui_scene.obstacle_boxes_xyz)
    figure, axis = plt.subplots()
    try:
        visualizer._draw_obstacles_xz(axis)
        elevation_rectangle = axis.patches[0]
        assert elevation_rectangle.get_x() == pytest.approx(1.2)
        assert elevation_rectangle.get_y() == pytest.approx(0.4)
        assert elevation_rectangle.get_width() == pytest.approx(0.6)
        assert elevation_rectangle.get_height() == pytest.approx(2.3)
    finally:
        plt.close(figure)

    assert len(faces) == 6
    assert {point[2] for face in faces for point in face} == {0.4, 2.7}
    assert any(len({point[2] for point in face}) == 2 for face in faces)


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
        estimated_strengths={"Co-60": np.asarray([1.0, 1.0, 1.0], dtype=float)},
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
    assert [label for _, label in truth_entries] == [
        "Co-1 T\n(1.00, 1.00, 0.50) m",
        "Co-2 T\n(5.00, 5.00, 0.50) m",
    ]
    assert [label for _, label in estimate_entries] == [
        "Co-1 E1",
        "Co-1 E2",
        "Co remote-1",
    ]

    figure, axis = plt.subplots()
    try:
        visualizer._plot_true_sources_2d(axis)
        assert [value.get_text() for value in axis.texts] == [
            "Co-1 T",
            "Co-2 T",
        ]
        assert [(value.get_ha(), value.get_va()) for value in axis.texts] == [
            ("left", "bottom"),
            ("right", "top"),
        ]
    finally:
        plt.close(figure)

    figure, axis = plt.subplots()
    try:
        visualizer._plot_true_sources_xz(axis)
        assert [value.get_text() for value in axis.texts] == [
            "Co-1 T",
            "Co-2 T",
        ]
    finally:
        plt.close(figure)
    assert visualizer._truth_inventory_text() == (
        "True sources (evaluation overlay)\n"
        "Co-1 T  (1.00, 1.00, 0.50) m\n"
        "Co-2 T  (5.00, 5.00, 0.50) m"
    )

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
    route = cui_route_from_records((_route_record(0, 0, pose_xyz=(1.0, 2.0, 0.5)),))
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
    visualizer._status_queue = queue.Queue()
    visualizer._run_token = "test-run"
    visualizer._next_operation_id = 0
    visualizer._last_enqueued_operation_id = -1
    visualizer._last_acknowledged_operation_id = -1
    visualizer._operation_kinds = {}
    visualizer._ready_acknowledged = True
    visualizer._close_operation_id = None
    visualizer._close_acknowledged = False
    visualizer._worker_error = None

    visualizer.update(_empty_frame(0, first_route))
    visualizer.update(_empty_frame(2, latest_route))

    message, operation_id, payload = visualizer._queue.get_nowait()
    retained_frame = pickle.loads(payload)
    assert message == "frame"
    assert operation_id == 1
    assert retained_frame.step_index == 2
    np.testing.assert_array_equal(
        retained_frame.cui_route.measurement_station_ids,
        np.asarray([0, 1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        retained_frame.cui_route.measurement_visit_counts,
        np.asarray([2, 1], dtype=np.int64),
    )


def test_async_cui_startup_rejects_mismatched_ack_and_reaps_worker() -> None:
    """A malformed startup ACK must fail construction and reap the child."""
    visualizer = AsyncCUISplitPFVisualizer.__new__(AsyncCUISplitPFVisualizer)
    process = _ControlledProcess()
    visualizer._closed = False
    visualizer._process = process
    visualizer._status_queue = queue.Queue()
    visualizer._status_queue.put(("ready", -1, "wrong-phase", "test-run"))
    visualizer._run_token = "test-run"
    visualizer._ready_acknowledged = False
    visualizer._operation_kinds = {}
    visualizer._close_operation_id = None
    visualizer._worker_error = None

    with pytest.raises(RuntimeError, match="mismatched startup"):
        visualizer._await_worker_ready(timeout_s=0.1)

    assert visualizer._closed is True
    assert process.terminate_calls == 1
    assert process.join_calls == 1


def test_async_cui_process_start_failure_reaps_a_partially_started_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A process-start exception must not leave a renderer child alive."""

    class _FailingStartProcess(_ControlledProcess):
        """Raise after exposing a synthetic child PID."""

        pid = 12345

        def start(self) -> None:
            """Simulate a partially completed multiprocessing start."""
            raise RuntimeError("synthetic renderer process start failure")

    class _FailingStartContext:
        """Construct deterministic queues and the failing process."""

        def __init__(self, process: _FailingStartProcess) -> None:
            """Retain the process inspected after construction fails."""
            self.process = process

        def Queue(self, maxsize: int = 0) -> queue.Queue[object]:
            """Return an in-memory queue matching the requested capacity."""
            return queue.Queue(maxsize=maxsize)

        def Process(self, **kwargs: object) -> _FailingStartProcess:
            """Return the one synthetic renderer process."""
            del kwargs
            return self.process

    process = _FailingStartProcess()
    context = _FailingStartContext(process)
    monkeypatch.setattr(
        "visualization.realtime_viz.mp.get_context",
        lambda method: context,
    )

    with pytest.raises(RuntimeError, match="process start failure"):
        AsyncCUISplitPFVisualizer(
            isotopes=["Cs-137"],
            output_dir=tmp_path,
        )

    assert process.terminate_calls == 1
    assert process.join_calls == 1
    assert process.is_alive() is False


def test_async_cui_update_propagates_worker_render_failure() -> None:
    """A child render error must fail before another frame can be accepted."""
    route = cui_route_from_records((_route_record(0, 0, pose_xyz=(1.0, 1.0, 0.5)),))
    visualizer = _queue_only_async_visualizer()
    visualizer.update(_empty_frame(0, route))
    visualizer._status_queue.put(
        ("error", 0, "ValueError: synthetic render failure", "test-run")
    )

    with pytest.raises(RuntimeError, match="synthetic render failure"):
        visualizer.update(_empty_frame(1, route))

    assert visualizer._next_operation_id == 1
    assert visualizer._queue.qsize() == 1
    assert visualizer._closed is True
    assert visualizer._process.terminate_calls == 1


def test_async_cui_update_rejects_mismatched_operation_ack() -> None:
    """An ACK for the wrong operation kind must never advance CUI state."""
    route = cui_route_from_records((_route_record(0, 0, pose_xyz=(1.0, 1.0, 0.5)),))
    visualizer = _queue_only_async_visualizer()
    visualizer.update(_empty_frame(0, route))
    visualizer._status_queue.put(("ack", 0, "truth", "test-run"))

    with pytest.raises(RuntimeError, match="mismatched operation"):
        visualizer.update(_empty_frame(1, route))

    assert visualizer.last_acknowledged_operation_id == -1
    assert visualizer._closed is True
    assert visualizer._process.terminate_calls == 1


def test_async_cui_close_requires_latest_frame_ack() -> None:
    """A close ACK cannot conceal an unacknowledged final render operation."""
    route = cui_route_from_records((_route_record(0, 0, pose_xyz=(1.0, 1.0, 0.5)),))
    visualizer = _queue_only_async_visualizer()
    visualizer.update(_empty_frame(0, route))

    def close_without_frame_ack() -> None:
        """Acknowledge only close while omitting the queued frame ACK."""
        while True:
            try:
                kind, operation_id, _payload = visualizer._queue.get_nowait()
            except queue.Empty:
                return
            if kind == "close":
                visualizer._status_queue.put(
                    ("closed", operation_id, "close", "test-run")
                )

    process = _ControlledProcess(close_without_frame_ack)
    visualizer._process = process

    with pytest.raises(RuntimeError, match="latest queued operation"):
        visualizer.close(timeout_s=0.1)

    assert visualizer._closed is True
    assert visualizer._close_acknowledged is True
    assert process.join_calls == 2


def test_async_cui_close_rejects_mismatched_close_ack() -> None:
    """A stale or future close ACK must fail even after the frame was rendered."""
    route = cui_route_from_records((_route_record(0, 0, pose_xyz=(1.0, 1.0, 0.5)),))
    visualizer = _queue_only_async_visualizer()
    visualizer.update(_empty_frame(0, route))

    def emit_mismatched_close_ack() -> None:
        """Emit the valid frame ACK followed by a future close operation ID."""
        while True:
            try:
                kind, operation_id, _payload = visualizer._queue.get_nowait()
            except queue.Empty:
                return
            if kind == "frame":
                visualizer._status_queue.put(("ack", operation_id, "frame", "test-run"))
            elif kind == "close":
                visualizer._status_queue.put(
                    ("closed", operation_id + 1, "close", "test-run")
                )

    process = _ControlledProcess(emit_mismatched_close_ack)
    visualizer._process = process

    with pytest.raises(RuntimeError, match="mismatched close"):
        visualizer.close(timeout_s=0.1)

    assert visualizer._closed is True
    assert process.join_calls == 2
