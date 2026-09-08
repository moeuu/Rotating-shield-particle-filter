"""Shared metric route and station drawing for CUI and paper figures."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.patheffects as path_effects
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
import numpy as np
from numpy.typing import NDArray


def draw_obstacle_boxes(
    ax: Axes,
    obstacle_boxes_xyz: NDArray[np.float64],
    *,
    projection: str,
    facecolor: str = "#747c84",
    edgecolor: str = "#343a40",
    linewidth: float = 0.30,
    alpha: float = 0.48,
    label: str | None = "physical obstacle",
    zorder: float = 0.0,
) -> int:
    """Draw unique projections of finite XYZXYZ physical obstacle boxes."""
    boxes = np.asarray(obstacle_boxes_xyz, dtype=np.float64)
    if boxes.size == 0:
        return 0
    if (
        boxes.ndim != 2
        or boxes.shape[1] != 6
        or np.any(~np.isfinite(boxes))
        or np.any(boxes[:, 3:] <= boxes[:, :3])
    ):
        raise ValueError("Obstacle boxes must be finite increasing XYZXYZ rows.")
    indices = {
        "xy": (0, 1, 3, 4),
        "xz": (0, 2, 3, 5),
        "yz": (1, 2, 4, 5),
    }.get(projection)
    if indices is None:
        raise ValueError(f"Unsupported obstacle projection {projection!r}.")
    seen: set[tuple[float, float, float, float]] = set()
    label_pending = label
    for values in boxes:
        rectangle = (
            round(float(values[indices[0]]), 9),
            round(float(values[indices[1]]), 9),
            round(float(values[indices[2]] - values[indices[0]]), 9),
            round(float(values[indices[3]] - values[indices[1]]), 9),
        )
        if rectangle in seen:
            continue
        seen.add(rectangle)
        ax.add_patch(
            Rectangle(
                rectangle[:2],
                rectangle[2],
                rectangle[3],
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=float(linewidth),
                alpha=float(alpha),
                label=label_pending,
                zorder=float(zorder),
            )
        )
        label_pending = None
    return len(seen)


def format_metric_projection_axis(
    ax: Axes,
    *,
    bounds_xyz: Sequence[float],
    projection: str,
    title: str,
    tick_step: float = 2.0,
    padding_fraction: float = 0.025,
    title_size: float | None = None,
    label_size: float | None = None,
    tick_size: float | None = None,
    title_weight: str = "normal",
) -> None:
    """Apply shared metric bounds, ticks, aspect, labels, and grid styling."""
    bounds = np.asarray(bounds_xyz, dtype=np.float64)
    if bounds.shape != (6,) or np.any(~np.isfinite(bounds)):
        raise ValueError("Metric scene bounds must contain six finite values.")
    axis_indices = {
        "xy": (0, 1, 2, 3, "x [m]", "y [m]"),
        "xz": (0, 1, 4, 5, "x [m]", "z [m]"),
        "yz": (2, 3, 4, 5, "y [m]", "z [m]"),
    }.get(projection)
    if axis_indices is None:
        raise ValueError(f"Unsupported metric projection {projection!r}.")
    x_min = float(bounds[axis_indices[0]])
    x_max = float(bounds[axis_indices[1]])
    y_min = float(bounds[axis_indices[2]])
    y_max = float(bounds[axis_indices[3]])
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Metric scene bounds must increase on every projected axis.")
    fraction = float(padding_fraction)
    if not np.isfinite(fraction) or fraction < 0.0:
        raise ValueError("Metric projection padding_fraction must be nonnegative.")
    x_padding = fraction * (x_max - x_min)
    y_padding = fraction * (y_max - y_min)
    spacing = float(tick_step)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Metric projection tick_step must be positive and finite.")
    ax.set_facecolor("#fbfbfb")
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("N")
    ax.set_xticks(np.arange(np.ceil(x_min / spacing) * spacing, x_max + 0.1, spacing))
    ax.set_yticks(np.arange(np.ceil(y_min / spacing) * spacing, y_max + 0.1, spacing))
    ax.grid(True, linewidth=0.30, alpha=0.30)
    ax.set_xlabel(axis_indices[4], fontsize=label_size)
    ax.set_ylabel(axis_indices[5], fontsize=label_size)
    ax.set_title(
        title,
        fontsize=title_size,
        fontweight=title_weight,
        pad=3,
    )
    ax.tick_params(labelsize=tick_size)


def project_positions(
    positions_xyz: NDArray[np.float64],
    projection: str,
) -> NDArray[np.float64]:
    """Project finite XYZ positions onto one named metric plane."""
    positions = np.asarray(positions_xyz, dtype=np.float64)
    if positions.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Metric scene positions must have shape (N, 3).")
    if np.any(~np.isfinite(positions)):
        raise ValueError("Metric scene positions must be finite.")
    indices = {
        "xy": (0, 1),
        "xz": (0, 2),
        "yz": (1, 2),
    }.get(projection)
    if indices is None:
        raise ValueError(f"Unsupported metric projection {projection!r}.")
    return positions[:, indices].copy()


def station_label_offsets(
    projected_points: NDArray[np.float64],
    *,
    radius: float = 0.16,
) -> NDArray[np.float64]:
    """Return deterministic offsets for station labels at repeated locations."""
    points = np.asarray(projected_points, dtype=np.float64)
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Projected station points must have shape (N, 2).")
    if np.any(~np.isfinite(points)):
        raise ValueError("Projected station points must be finite.")
    offsets = np.zeros_like(points)
    used_counts: dict[tuple[float, float], int] = {}
    for index, point in enumerate(points):
        key = tuple(float(value) for value in np.round(point, 3))
        repeat_index = used_counts.get(key, 0)
        used_counts[key] = repeat_index + 1
        if repeat_index == 0:
            continue
        angle = 2.0 * np.pi * float(repeat_index - 1) / 6.0
        offsets[index] = (
            float(radius) * np.cos(angle),
            float(radius) * np.sin(angle),
        )
    return offsets


def station_label(
    station_index: int,
    *,
    station_ids: Sequence[int] | None = None,
    visit_counts: Sequence[int] | None = None,
) -> str:
    """Return a compact station-order label with an optional visit count."""
    station_id = (
        int(station_ids[station_index])
        if station_ids is not None and station_index < len(station_ids)
        else int(station_index)
    )
    visits = (
        int(visit_counts[station_index])
        if visit_counts is not None and station_index < len(visit_counts)
        else 1
    )
    return str(station_id) if visits <= 1 else f"{station_id}({visits})"


def draw_route_segments(
    ax: Axes,
    route_segments_xyz: Sequence[NDArray[np.float64]],
    *,
    projection: str,
    color: str,
    linewidth: float,
    alpha: float,
    label: str | None = None,
    zorder: float = 2.0,
) -> None:
    """Draw persisted metric route segments with one shared projection rule."""
    label_pending = label
    for segment in route_segments_xyz:
        projected = project_positions(np.asarray(segment), projection)
        if len(projected) < 2:
            continue
        ax.plot(
            projected[:, 0],
            projected[:, 1],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            label=label_pending,
            zorder=zorder,
        )
        label_pending = None


def draw_measurement_stations(
    ax: Axes,
    station_positions_xyz: NDArray[np.float64],
    *,
    projection: str,
    station_ids: Sequence[int] | None = None,
    visit_counts: Sequence[int] | None = None,
    show_labels: bool = True,
    marker_size: float = 48.0,
    facecolor: str = "white",
    edgecolor: str = "#009eae",
    linewidth: float = 0.9,
    alpha: float = 0.92,
    label: str | None = "measurement station",
    font_size: float = 8.0,
    label_color: str = "black",
    label_offset_radius: float = 0.16,
    label_stroke_width: float = 1.8,
    zorder: float = 9.0,
) -> NDArray[np.float64]:
    """Draw stations and their acquisition-order labels on a metric axis."""
    projected = project_positions(station_positions_xyz, projection)
    if not len(projected):
        return projected
    ax.scatter(
        projected[:, 0],
        projected[:, 1],
        s=marker_size,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
        zorder=zorder,
    )
    if not show_labels:
        return projected
    offsets = station_label_offsets(
        projected,
        radius=label_offset_radius,
    )
    for index, point in enumerate(projected):
        text = ax.text(
            point[0] + offsets[index, 0],
            point[1] + offsets[index, 1],
            station_label(
                index,
                station_ids=station_ids,
                visit_counts=visit_counts,
            ),
            color=label_color,
            fontsize=font_size,
            ha="center",
            va="center",
            zorder=zorder + 1,
        )
        text.set_path_effects(
            [
                path_effects.withStroke(
                    linewidth=label_stroke_width,
                    foreground="white",
                )
            ]
        )
    return projected
