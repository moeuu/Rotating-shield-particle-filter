"""Obstacle layout utilities for blocking measurement positions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np


@dataclass(frozen=True)
class ObstacleGrid:
    """Represent blocked 1 m grid cells on the z=0 plane."""

    origin: tuple[float, float]
    cell_size: float
    grid_shape: tuple[int, int]
    blocked_cells: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        """Normalize inputs and validate bounds."""
        if len(self.origin) != 2:
            raise ValueError("origin must have length 2.")
        if len(self.grid_shape) != 2:
            raise ValueError("grid_shape must have length 2.")
        origin = (float(self.origin[0]), float(self.origin[1]))
        cell_size = float(self.cell_size)
        grid_shape = (int(self.grid_shape[0]), int(self.grid_shape[1]))
        if cell_size <= 0.0:
            raise ValueError("cell_size must be positive.")
        if grid_shape[0] < 0 or grid_shape[1] < 0:
            raise ValueError("grid_shape must be non-negative.")
        blocked = tuple(
            sorted((int(cell[0]), int(cell[1])) for cell in self.blocked_cells)
        )
        for cell in blocked:
            if cell[0] < 0 or cell[1] < 0:
                raise ValueError("blocked_cells indices must be non-negative.")
            if cell[0] >= grid_shape[0] or cell[1] >= grid_shape[1]:
                raise ValueError("blocked_cells entry out of grid bounds.")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "cell_size", cell_size)
        object.__setattr__(self, "grid_shape", grid_shape)
        object.__setattr__(self, "blocked_cells", blocked)
        object.__setattr__(self, "_blocked_set", frozenset(blocked))

    @property
    def total_cells(self) -> int:
        """Return total number of grid cells."""
        return int(self.grid_shape[0] * self.grid_shape[1])

    @property
    def blocked_fraction(self) -> float:
        """Return the fraction of blocked cells."""
        total = self.total_cells
        if total == 0:
            return 0.0
        return float(len(self.blocked_cells)) / float(total)

    def cell_index(self, point: Sequence[float]) -> tuple[int, int] | None:
        """Return the (ix, iy) cell index for a point, or None if outside."""
        if len(point) < 2:
            raise ValueError("point must have at least two coordinates.")
        x = float(point[0])
        y = float(point[1])
        rel_x = x - self.origin[0]
        rel_y = y - self.origin[1]
        if rel_x < 0.0 or rel_y < 0.0:
            return None
        ix = int(np.floor(rel_x / self.cell_size))
        iy = int(np.floor(rel_y / self.cell_size))
        if ix < 0 or iy < 0:
            return None
        if ix >= self.grid_shape[0] or iy >= self.grid_shape[1]:
            return None
        return ix, iy

    def is_free(self, point: Sequence[float]) -> bool:
        """Return True if the point is not inside a blocked cell."""
        idx = self.cell_index(point)
        if idx is None:
            return True
        return idx not in self._blocked_set

    def blocked_bounds(self) -> list[tuple[float, float, float, float]]:
        """Return (x0, x1, y0, y1) bounds for blocked cells."""
        bounds: list[tuple[float, float, float, float]] = []
        for ix, iy in self.blocked_cells:
            x0 = self.origin[0] + ix * self.cell_size
            y0 = self.origin[1] + iy * self.cell_size
            bounds.append((x0, x0 + self.cell_size, y0, y0 + self.cell_size))
        return bounds

    def blocked_polygons(
        self, z: float = 0.0
    ) -> list[list[tuple[float, float, float]]]:
        """Return polygons for blocked cells at the given z-plane."""
        polygons: list[list[tuple[float, float, float]]] = []
        for x0, x1, y0, y1 in self.blocked_bounds():
            polygons.append([(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)])
        return polygons

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation of the grid."""
        return {
            "version": 1,
            "origin": [self.origin[0], self.origin[1]],
            "cell_size": self.cell_size,
            "grid_shape": [self.grid_shape[0], self.grid_shape[1]],
            "blocked_cells": [list(cell) for cell in self.blocked_cells],
            "blocked_fraction": self.blocked_fraction,
        }

    def save(self, path: Path) -> None:
        """Save the obstacle layout to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict) -> ObstacleGrid:
        """Construct an ObstacleGrid from a dictionary payload."""
        if not isinstance(data, dict):
            raise ValueError("Obstacle layout must be a dict.")
        origin = data.get("origin", (0.0, 0.0))
        cell_size = data.get("cell_size", 1.0)
        grid_shape = data.get("grid_shape")
        blocked_cells = data.get("blocked_cells", [])
        if grid_shape is None:
            raise ValueError("Obstacle layout missing 'grid_shape'.")
        if not isinstance(blocked_cells, list):
            raise ValueError("blocked_cells must be a list.")
        return cls(
            origin=(float(origin[0]), float(origin[1])),
            cell_size=float(cell_size),
            grid_shape=(int(grid_shape[0]), int(grid_shape[1])),
            blocked_cells=tuple((int(cell[0]), int(cell[1])) for cell in blocked_cells),
        )

    @classmethod
    def load(cls, path: Path) -> ObstacleGrid:
        """Load an obstacle layout from a JSON file."""
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)


def _point_to_cell_index(
    point: Sequence[float],
    origin: tuple[float, float],
    cell_size: float,
    grid_shape: tuple[int, int],
) -> tuple[int, int] | None:
    """Return the grid cell index for a point or None if outside."""
    if len(point) < 2:
        raise ValueError("point must have at least two coordinates.")
    x = float(point[0])
    y = float(point[1])
    rel_x = x - origin[0]
    rel_y = y - origin[1]
    if rel_x < 0.0 or rel_y < 0.0:
        return None
    ix = int(np.floor(rel_x / cell_size))
    iy = int(np.floor(rel_y / cell_size))
    if ix < 0 or iy < 0:
        return None
    if ix >= grid_shape[0] or iy >= grid_shape[1]:
        return None
    return ix, iy


def generate_obstacle_grid(
    size_x: float,
    size_y: float,
    *,
    cell_size: float = 1.0,
    blocked_fraction: float = 0.4,
    origin: tuple[float, float] = (0.0, 0.0),
    rng: np.random.Generator | None = None,
    keep_free_points: Iterable[Sequence[float]] | None = None,
) -> ObstacleGrid:
    """
    Generate a random obstacle layout by blocking a fraction of grid cells.

    The grid is defined on the z=0 plane with 1 m x 1 m cells by default.
    """
    if cell_size <= 0.0:
        raise ValueError("cell_size must be positive.")
    if blocked_fraction < 0.0 or blocked_fraction > 1.0:
        raise ValueError("blocked_fraction must be between 0 and 1.")
    extent_x = max(0.0, float(size_x) - float(origin[0]))
    extent_y = max(0.0, float(size_y) - float(origin[1]))
    nx = int(np.floor(extent_x / cell_size))
    ny = int(np.floor(extent_y / cell_size))
    if nx <= 0 or ny <= 0:
        raise ValueError("Environment size is too small for the requested grid.")
    total = int(nx * ny)
    target = int(np.round(total * blocked_fraction))
    target = max(0, min(target, total))
    rng = np.random.default_rng() if rng is None else rng
    keep_free_cells: set[tuple[int, int]] = set()
    if keep_free_points is not None:
        for pt in keep_free_points:
            idx = _point_to_cell_index(pt, origin, cell_size, (nx, ny))
            if idx is not None:
                keep_free_cells.add(idx)
    all_indices = np.arange(total)
    if keep_free_cells:
        keep_flat = np.array(
            [cell[0] + cell[1] * nx for cell in keep_free_cells], dtype=int
        )
        mask = np.ones(total, dtype=bool)
        mask[keep_flat] = False
        available = all_indices[mask]
    else:
        available = all_indices
    target = min(target, int(available.size))
    if target > 0:
        selected = rng.choice(available, size=target, replace=False)
    else:
        selected = np.array([], dtype=int)
    blocked_cells = [(int(idx % nx), int(idx // nx)) for idx in selected]
    blocked_cells.sort()
    return ObstacleGrid(
        origin=origin,
        cell_size=cell_size,
        grid_shape=(nx, ny),
        blocked_cells=tuple(blocked_cells),
    )


def load_or_generate_obstacle_grid(
    path: Path,
    *,
    size_x: float,
    size_y: float,
    cell_size: float = 1.0,
    blocked_fraction: float = 0.4,
    rng_seed: int | None = None,
    keep_free_points: Iterable[Sequence[float]] | None = None,
) -> ObstacleGrid:
    """
    Load a layout from disk, or generate and save one when missing.
    """
    if path.exists():
        return ObstacleGrid.load(path)
    rng = np.random.default_rng(rng_seed)
    grid = generate_obstacle_grid(
        size_x=size_x,
        size_y=size_y,
        cell_size=cell_size,
        blocked_fraction=blocked_fraction,
        origin=(0.0, 0.0),
        rng=rng,
        keep_free_points=keep_free_points,
    )
    grid.save(path)
    return grid


def build_obstacle_grid(
    *,
    mode: Literal["fixed", "random"],
    path: Path | None,
    size_x: float,
    size_y: float,
    cell_size: float = 1.0,
    blocked_fraction: float = 0.4,
    rng_seed: int | None = None,
    keep_free_points: Iterable[Sequence[float]] | None = None,
) -> ObstacleGrid:
    """
    Build an obstacle grid in fixed or random mode.

    Fixed mode keeps the current JSON-backed workflow by loading the layout from
    disk or generating it once when the file does not exist. Random mode always
    creates a fresh in-memory layout for the current run and never writes it to
    disk.
    """
    normalized_mode = mode.strip().lower()
    if normalized_mode == "fixed":
        if path is None:
            raise ValueError("path is required when mode is 'fixed'.")
        return load_or_generate_obstacle_grid(
            path,
            size_x=size_x,
            size_y=size_y,
            cell_size=cell_size,
            blocked_fraction=blocked_fraction,
            rng_seed=rng_seed,
            keep_free_points=keep_free_points,
        )
    if normalized_mode == "random":
        rng = np.random.default_rng(rng_seed)
        return generate_obstacle_grid(
            size_x=size_x,
            size_y=size_y,
            cell_size=cell_size,
            blocked_fraction=blocked_fraction,
            origin=(0.0, 0.0),
            rng=rng,
            keep_free_points=keep_free_points,
        )
    raise ValueError(f"Unknown obstacle grid mode: {mode}")
