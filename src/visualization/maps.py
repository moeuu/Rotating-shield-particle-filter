"""Render estimated source locations and strengths as heatmaps overlaid on trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from numpy.typing import NDArray

from measurement.continuous_kernels import geometric_term
from measurement.shielding import OctantShield, octant_index_from_normal


def _make_grid(
    env_bounds: Tuple[float, float, float, float], resolution: float
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create 2D grid (X,Y) covering the environment bounds."""
    xmin, xmax, ymin, ymax = env_bounds
    xs = np.arange(xmin, xmax + resolution, resolution)
    ys = np.arange(ymin, ymax + resolution, resolution)
    X, Y = np.meshgrid(xs, ys)
    return X, Y


def _expected_intensity_at_points(
    grid_points: NDArray[np.float64],
    estimates: Dict[str, Dict[str, NDArray[np.float64]]],
    shield_normal: Optional[NDArray[np.float64]] = None,
    attenuation_block: float = 0.1,
) -> NDArray[np.float64]:
    """
    Compute expected intensity (sum over isotopes) at given grid_points (N,2).

    estimates: dict[iso] -> {"positions": (M,3), "strengths": (M,)}
    """
    octant = OctantShield()
    if shield_normal is not None:
        orient_idx = octant_index_from_normal(shield_normal)
    else:
        orient_idx = None
    intensities = np.zeros(grid_points.shape[0], dtype=float)
    detector_z = 0.0  # assume ground plane rendering
    for iso, est in estimates.items():
        pos3d = est.get("positions", np.zeros((0, 3)))
        strengths = est.get("strengths", np.zeros(0))
        for src, q in zip(pos3d, strengths):
            src = np.asarray(src, dtype=float)
            for i, (x, y) in enumerate(grid_points):
                detector = np.array([x, y, detector_z], dtype=float)
                geom = geometric_term(detector, src)
                att = 1.0
                if orient_idx is not None:
                    if octant.blocks_ray(detector_position=detector, source_position=src, octant_index=orient_idx):
                        att = attenuation_block
                intensities[i] += geom * att * float(q)
    return intensities


def render_heatmap(
    env_bounds: Tuple[float, float, float, float],
    trajectory: NDArray[np.float64],
    estimates: Dict[str, Dict[str, NDArray[np.float64]]],
    resolution: float = 0.5,
    shield_normal: Optional[NDArray[np.float64]] = None,
    obstacles: Optional[Iterable[Tuple[float, float]]] = None,
    cmap: str = "inferno",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Render a 2D heatmap of expected radiation intensity over the environment.

    Args:
        env_bounds: (xmin, xmax, ymin, ymax)
        trajectory: (T,2) robot path
        estimates: dict[iso] -> {"positions": (M,3), "strengths": (M,)}
        resolution: grid spacing (m)
        shield_normal: optional shielding orientation to model attenuation
        obstacles: iterable of (x,y) points to mark obstacles
        cmap: matplotlib colormap name
        output_path: if provided, save the figure to this path
    """
    X, Y = _make_grid(env_bounds, resolution)
    points = np.column_stack([X.ravel(), Y.ravel()])
    intensities = _expected_intensity_at_points(points, estimates, shield_normal=shield_normal)
    Z = intensities.reshape(X.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        Z,
        extent=env_bounds,
        origin="lower",
        cmap=cmap,
        aspect="auto",
    )
    plt.colorbar(im, ax=ax, label="Expected intensity (a.u.)")
    if trajectory is not None and trajectory.size:
        ax.plot(trajectory[:, 0], trajectory[:, 1], "-o", color="cyan", label="Trajectory")
    if obstacles:
        obs = np.array(list(obstacles))
        ax.scatter(obs[:, 0], obs[:, 1], color="red", marker="x", label="Obstacles")
    # Plot estimated sources
    for iso, est in estimates.items():
        pos3d = est.get("positions", np.zeros((0, 3)))
        strengths = est.get("strengths", np.zeros(0))
        if pos3d.size == 0:
            continue
        ax.scatter(pos3d[:, 0], pos3d[:, 1], s=40, label=f"{iso} est.")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Estimated radiation heatmap")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
    return fig
