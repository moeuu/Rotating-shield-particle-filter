"""Render an obstacle layout as a 2D environment image."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from measurement.obstacles import load_or_generate_obstacle_grid, ObstacleGrid


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Render a 2D environment map with obstacles.")
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
        help="Environment size in x (meters) when creating a new layout.",
    )
    parser.add_argument(
        "--size-y",
        type=float,
        default=20.0,
        help="Environment size in y (meters) when creating a new layout.",
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
        help="Fraction of blocked grid cells when creating a new layout.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for layout generation if the file does not exist.",
    )
    return parser


def _resolve_path(path: Path) -> Path:
    """Resolve a repo-relative path to an absolute path."""
    return path if path.is_absolute() else (ROOT / path).resolve()


def _draw_obstacles(ax: plt.Axes, grid: ObstacleGrid) -> None:
    """Draw blocked cells as black rectangles."""
    patches: list[Rectangle] = []
    for x0, x1, y0, y1 in grid.blocked_bounds():
        patches.append(Rectangle((x0, y0), x1 - x0, y1 - y0))
    if not patches:
        return
    collection = PatchCollection(
        patches,
        facecolor="black",
        edgecolor="none",
        alpha=0.9,
    )
    ax.add_collection(collection)


def render_environment(grid: ObstacleGrid, output_path: Path) -> None:
    """Render the obstacle grid and save it to a PNG file."""
    xmin, ymin = grid.origin
    xmax = xmin + grid.grid_shape[0] * grid.cell_size
    ymax = ymin + grid.grid_shape[1] * grid.cell_size

    fig, ax = plt.subplots(figsize=(6, 10))
    _draw_obstacles(ax, grid)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xticks(range(int(xmin), int(xmax) + 1, 2))
    ax.set_yticks(range(int(ymin), int(ymax) + 1, 2))
    ax.grid(True, color="#CCCCCC", linewidth=0.6, alpha=0.6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved environment image to {output_path}")


def main() -> None:
    """Entry point for rendering the obstacle environment."""
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
    render_environment(grid, output_path)


if __name__ == "__main__":
    main()
