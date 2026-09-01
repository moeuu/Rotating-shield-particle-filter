"""Deterministic geometry helpers for obstacle-aware scientific figures."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


_BOX_FACE_INDICES = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
)


def validated_axis_aligned_boxes(
    boxes_xyz: object,
) -> NDArray[np.float64]:
    """Return finite increasing ``(x0, y0, z0, x1, y1, z1)`` boxes."""
    try:
        boxes = np.asarray(boxes_xyz, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Obstacle boxes must be a numeric array.") from exc
    if boxes.size == 0:
        return np.zeros((0, 6), dtype=np.float64)
    if (
        boxes.ndim != 2
        or boxes.shape[1:] != (6,)
        or np.any(~np.isfinite(boxes))
        or np.any(boxes[:, 3:] <= boxes[:, :3])
    ):
        raise ValueError("Obstacle boxes must contain finite increasing XYZXYZ rows.")
    return np.asarray(boxes, dtype=np.float64)


def axis_aligned_box_faces(
    boxes_xyz: object,
) -> list[list[tuple[float, float, float]]]:
    """Return six exact quadrilateral faces for every axis-aligned box."""
    boxes = validated_axis_aligned_boxes(boxes_xyz)
    faces: list[list[tuple[float, float, float]]] = []
    for x0, y0, z0, x1, y1, z1 in boxes:
        vertices = (
            (float(x0), float(y0), float(z0)),
            (float(x1), float(y0), float(z0)),
            (float(x1), float(y1), float(z0)),
            (float(x0), float(y1), float(z0)),
            (float(x0), float(y0), float(z1)),
            (float(x1), float(y0), float(z1)),
            (float(x1), float(y1), float(z1)),
            (float(x0), float(y1), float(z1)),
        )
        faces.extend(
            [[vertices[index] for index in indices] for indices in _BOX_FACE_INDICES]
        )
    return faces


def blocked_cell_boxes(
    blocked_cells: Sequence[Sequence[int]],
    *,
    origin_xy: Sequence[float],
    cell_size_m: float,
    z_bounds_m: tuple[float, float],
) -> NDArray[np.float64]:
    """Extrude navigation cells only when physical component boxes are absent."""
    origin = np.asarray(origin_xy, dtype=np.float64)
    if origin.shape != (2,) or np.any(~np.isfinite(origin)):
        raise ValueError("Obstacle-grid origin must be one finite XY pair.")
    cell_size = float(cell_size_m)
    z0, z1 = (float(value) for value in z_bounds_m)
    if (
        not np.isfinite(cell_size)
        or cell_size <= 0.0
        or not np.isfinite(z0)
        or not np.isfinite(z1)
        or z1 <= z0
    ):
        raise ValueError("Obstacle-grid cell size and height must be increasing.")
    boxes: list[tuple[float, float, float, float, float, float]] = []
    for index, raw_cell in enumerate(blocked_cells):
        cell = np.asarray(raw_cell)
        if (
            cell.shape != (2,)
            or np.issubdtype(cell.dtype, np.bool_)
            or not np.issubdtype(cell.dtype, np.integer)
        ):
            raise ValueError(
                f"blocked_cells[{index}] must contain exactly two integers."
            )
        x0 = float(origin[0] + int(cell[0]) * cell_size)
        y0 = float(origin[1] + int(cell[1]) * cell_size)
        boxes.append((x0, y0, z0, x0 + cell_size, y0 + cell_size, z1))
    return validated_axis_aligned_boxes(boxes)
