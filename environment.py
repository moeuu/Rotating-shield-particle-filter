"""Render a 3D room with obstacle cells on the floor for documentation figures."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from measurement.obstacles import ObstacleGrid, load_or_generate_obstacle_grid


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for environment rendering."""
    parser = argparse.ArgumentParser(
        description="Render a 3D environment with obstacle cells on the floor."
    )
    parser.add_argument(
        "--obstacle-config",
        type=Path,
        default=Path("obstacle_layouts/demo_obstacles.json"),
        help="Obstacle layout JSON path (relative to repo root if not absolute).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/environment.png"),
        help="Output PNG path (relative to repo root if not absolute).",
    )
    parser.add_argument(
        "--size-x",
        type=float,
        default=10.0,
        help="Room size along x (meters) when generating a new layout.",
    )
    parser.add_argument(
        "--size-y",
        type=float,
        default=20.0,
        help="Room size along y (meters) when generating a new layout.",
    )
    parser.add_argument(
        "--size-z",
        type=float,
        default=10.0,
        help="Room size along z (meters).",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=1.0,
        help="Obstacle grid cell size (meters).",
    )
    parser.add_argument(
        "--blocked-fraction",
        type=float,
        default=0.4,
        help="Fraction of blocked cells when generating a new layout.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used when creating a new obstacle layout file.",
    )
    return parser


def _resolve_path(path: Path) -> Path:
    """Resolve a path relative to the repo root."""
    return path if path.is_absolute() else (ROOT / path).resolve()


def _axis_line_style(ax: plt.Axes) -> tuple[str, float]:
    """Return line color and width that match the axis lines."""
    color = "black"
    linewidth = 1.2
    axis_line = getattr(ax.xaxis, "line", None)
    if axis_line is not None:
        color = axis_line.get_color()
        try:
            axis_width = float(axis_line.get_linewidth())
        except (TypeError, ValueError):
            axis_width = linewidth
        if axis_width > 0:
            linewidth = axis_width
    return color, linewidth


def _draw_room_bounds(
    ax: plt.Axes,
    bounds: tuple[float, float, float, float, float, float],
) -> None:
    """Draw a solid line box showing the room boundaries."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    color, linewidth = _axis_line_style(ax)
    edges = [
        ((xmin, ymin, zmin), (xmax, ymin, zmin)),
        ((xmin, ymax, zmin), (xmax, ymax, zmin)),
        ((xmin, ymin, zmax), (xmax, ymin, zmax)),
        ((xmin, ymax, zmax), (xmax, ymax, zmax)),
        ((xmin, ymin, zmin), (xmin, ymax, zmin)),
        ((xmax, ymin, zmin), (xmax, ymax, zmin)),
        ((xmin, ymin, zmax), (xmin, ymax, zmax)),
        ((xmax, ymin, zmax), (xmax, ymax, zmax)),
        ((xmin, ymin, zmin), (xmin, ymin, zmax)),
        ((xmax, ymin, zmin), (xmax, ymin, zmax)),
        ((xmin, ymax, zmin), (xmin, ymax, zmax)),
        ((xmax, ymax, zmin), (xmax, ymax, zmax)),
    ]
    for start, end in edges:
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=color,
            linewidth=linewidth,
        )


def _draw_obstacles(ax: plt.Axes, grid: ObstacleGrid) -> None:
    """Draw obstacle cells as black patches on the floor."""
    polygons = grid.blocked_polygons(z=0.0)
    if not polygons:
        return
    collection = Poly3DCollection(polygons, facecolors="black", edgecolors="none", alpha=0.85)
    ax.add_collection3d(collection)


def render_environment(
    grid: ObstacleGrid,
    output_path: Path,
    size_z: float,
) -> None:
    """Render the 3D environment and save it to a PNG file."""
    xmin, ymin = grid.origin
    xmax = xmin + grid.grid_shape[0] * grid.cell_size
    ymax = ymin + grid.grid_shape[1] * grid.cell_size
    bounds = (xmin, xmax, ymin, ymax, 0.0, size_z)

    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(0.0, size_z)
    ax.set_box_aspect((xmax - xmin, ymax - ymin, size_z))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    _draw_room_bounds(ax, bounds)
    _draw_obstacles(ax, grid)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved environment image to {output_path}")


def main() -> None:
    """Entry point for rendering the obstacle environment image."""
    parser = build_parser()
    args = parser.parse_args()
    obstacle_path = _resolve_path(args.obstacle_config)
    output_path = _resolve_path(args.output)
    grid = load_or_generate_obstacle_grid(
        obstacle_path,
        size_x=args.size_x,
        size_y=args.size_y,
        cell_size=args.cell_size,
        blocked_fraction=args.blocked_fraction,
        rng_seed=args.seed,
    )
    render_environment(grid, output_path, args.size_z)


if __name__ == "__main__":
    main()
