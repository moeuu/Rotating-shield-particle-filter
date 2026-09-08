"""Tests for the shared CUI and manuscript metric-scene renderer."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from visualization.metric_scene import (
    draw_measurement_stations,
    draw_obstacle_boxes,
    draw_route_segments,
    format_metric_projection_axis,
    project_positions,
    station_label,
    station_label_offsets,
)


def test_project_positions_preserves_requested_metric_axes() -> None:
    """Each named projection must select the matching world coordinates."""
    positions = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    np.testing.assert_array_equal(
        project_positions(positions, "xy"),
        np.asarray([[1.0, 2.0], [4.0, 5.0]]),
    )
    np.testing.assert_array_equal(
        project_positions(positions, "xz"),
        np.asarray([[1.0, 3.0], [4.0, 6.0]]),
    )
    np.testing.assert_array_equal(
        project_positions(positions, "yz"),
        np.asarray([[2.0, 3.0], [5.0, 6.0]]),
    )
    with pytest.raises(ValueError, match="Unsupported metric projection"):
        project_positions(positions, "zx")


def test_station_labels_are_deterministic_at_repeated_positions() -> None:
    """Repeated station locations must retain distinct deterministic labels."""
    points = np.asarray([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    offsets = station_label_offsets(points, radius=0.2)

    np.testing.assert_array_equal(offsets[0], np.zeros(2))
    assert np.linalg.norm(offsets[1]) == pytest.approx(0.2)
    assert np.linalg.norm(offsets[2]) == pytest.approx(0.2)
    assert station_label(0, station_ids=(4,), visit_counts=(2,)) == "4(2)"


def test_shared_drawers_show_saved_route_and_acquisition_order() -> None:
    """The common renderer must draw saved segments and station-order text."""
    figure, axis = plt.subplots()
    route = (np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),)
    stations = np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])

    draw_route_segments(
        axis,
        route,
        projection="xz",
        color="black",
        linewidth=1.0,
        alpha=1.0,
    )
    draw_measurement_stations(
        axis,
        stations,
        projection="xz",
        station_ids=(7, 8),
    )

    np.testing.assert_array_equal(axis.lines[0].get_xdata(), [0.0, 3.0])
    np.testing.assert_array_equal(axis.lines[0].get_ydata(), [2.0, 5.0])
    assert [text.get_text() for text in axis.texts] == ["7", "8"]
    plt.close(figure)


def test_shared_projection_draws_unique_physical_obstacle_extents() -> None:
    """Obstacle footprints and elevations must use the same XYZXYZ boxes."""
    boxes = np.asarray(
        [
            [1.0, 2.0, 0.5, 3.0, 4.0, 2.5],
            [1.0, 6.0, 0.5, 3.0, 8.0, 2.5],
        ]
    )
    figure, (floor_axis, elevation_axis) = plt.subplots(1, 2)

    assert draw_obstacle_boxes(floor_axis, boxes, projection="xy") == 2
    assert draw_obstacle_boxes(elevation_axis, boxes, projection="xz") == 1
    elevation = elevation_axis.patches[0]
    assert elevation.get_x() == pytest.approx(1.0)
    assert elevation.get_y() == pytest.approx(0.5)
    assert elevation.get_width() == pytest.approx(2.0)
    assert elevation.get_height() == pytest.approx(2.0)
    format_metric_projection_axis(
        elevation_axis,
        bounds_xyz=(0.0, 10.0, 0.0, 15.0, 0.0, 5.0),
        projection="xz",
        title="Elevation",
    )
    assert elevation_axis.get_aspect() == pytest.approx(1.0)
    assert elevation_axis.get_xlabel() == "x [m]"
    assert elevation_axis.get_ylabel() == "z [m]"
    plt.close(figure)
