"""Data structures for capturing PF state per time step for visualization."""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import queue
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy.typing import NDArray
from runtime.artifacts import atomic_copy_file
from runtime.cui import CUIRoute
from runtime.cui_components import (
    CUIPanelSpec,
    CUIScene,
    shared_cui_panel_specs,
    write_cui_index,
)

from visualization.frame import PFFrame

DEFAULT_ISOTOPE_COLORS = {
    "Cs-137": "tab:red",
    "Co-60": "tab:blue",
    "Eu-154": "tab:green",
    "Eu-155": "tab:green",
}

PF_RESULT_PANEL_SPECS = (
    CUIPanelSpec(
        "pf-particle-posterior",
        "PF particle posterior 3D",
        "latest_pf_3d.png",
    ),
    CUIPanelSpec(
        "pf-particle-labels",
        "PF particle posterior with source labels",
        "latest_pf_3d_labeled.png",
        2,
    ),
)


def pf_cui_panel_specs() -> tuple[CUIPanelSpec, ...]:
    """Return the shared context shell with PF-owned particle result panels."""
    return shared_cui_panel_specs(PF_RESULT_PANEL_SPECS)


def _temporary_png_path(target: Path) -> Path:
    """Reserve a unique render path beside one atomically published PNG."""
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".render.png",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    return temporary


def _normalize_weights(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return validated normalized posterior weights."""
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return w
    if np.any(~np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError(
            "PF visualization requires finite nonnegative posterior weights."
        )
    total = float(np.sum(w))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("PF visualization requires strictly positive posterior mass.")
    return w / total


def _metric_ticks(vmin: float, vmax: float, step: float = 2.0) -> NDArray[np.float64]:
    """Return regularly spaced metric ticks covering an axis interval."""
    lo = float(vmin)
    hi = float(vmax)
    spacing = max(float(step), 1.0e-9)
    start = np.ceil(lo / spacing) * spacing
    stop = np.floor(hi / spacing) * spacing
    if stop < start:
        return np.asarray([lo, hi], dtype=float)
    ticks = np.arange(start, stop + 0.5 * spacing, spacing, dtype=float)
    if ticks.size == 0:
        return np.asarray([lo, hi], dtype=float)
    return ticks


def _apply_metric_ticks_2d(
    ax: plt.Axes,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    step: float = 2.0,
) -> None:
    """Apply consistent metric tick spacing to a 2-D axis."""
    ax.set_xticks(_metric_ticks(float(xlim[0]), float(xlim[1]), step))
    ax.set_yticks(_metric_ticks(float(ylim[0]), float(ylim[1]), step))


def _apply_metric_ticks_3d(
    ax: plt.Axes,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
    step: float = 2.0,
) -> None:
    """Apply consistent metric tick spacing to a 3-D axis."""
    ax.set_xticks(_metric_ticks(float(xlim[0]), float(xlim[1]), step))
    ax.set_yticks(_metric_ticks(float(ylim[0]), float(ylim[1]), step))
    ax.set_zticks(_metric_ticks(float(zlim[0]), float(zlim[1]), step))


def _active_display_positions(
    particle_filter: Any,
    state: Any,
) -> NDArray[np.float64]:
    """Resolve and validate active positions from the continuous surface state."""
    num_sources = max(0, int(getattr(state, "num_sources", 0)))
    if num_sources <= 0:
        return np.zeros((0, 3), dtype=float)
    resolver = getattr(particle_filter, "continuous_state_positions", None)
    if not callable(resolver):
        raise RuntimeError(
            "PF visualization requires the continuous surface-state resolver."
        )
    positions = np.asarray(resolver(state), dtype=float)
    if positions.shape != (num_sources, 3) or np.any(~np.isfinite(positions)):
        raise RuntimeError(
            "PF visualization received an invalid continuous surface state."
        )
    return positions


def _active_surface_source_medoid(
    active_positions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return one existing active source nearest the discrete XYZ medoid.

    This helper is only for the compact particle-cloud display. Selecting an
    existing source keeps every marker on the continuous surface; averaging
    source coordinates could create a marker inside an obstacle or in free
    space.
    """
    positions = np.asarray(active_positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
        raise ValueError(
            "An active particle representative requires finite XYZ sources."
        )
    if np.any(~np.isfinite(positions)):
        raise ValueError(
            "An active particle representative requires finite XYZ sources."
        )
    pairwise = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :],
        axis=2,
    )
    return positions[int(np.argmin(np.sum(pairwise, axis=1)))].copy()


def _batched_particle_display_arrays(
    particle_filter: Any,
    particle_weights: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Build exact particle-cloud arrays through the fixed-slot batch path."""
    packer = getattr(
        particle_filter,
        "_packed_continuous_surface_state_arrays",
        None,
    )
    particles = getattr(particle_filter, "continuous_particles", [])
    weights = np.asarray(particle_weights, dtype=float).reshape(-1)
    if not callable(packer):
        flat_positions: list[NDArray[np.float64]] = []
        flat_weights: list[float] = []
        representative_positions: list[NDArray[np.float64]] = []
        representative_weights: list[float] = []
        for particle, weight in zip(particles, weights, strict=True):
            active_positions = _active_display_positions(
                particle_filter,
                particle.state,
            )
            if active_positions.size == 0:
                continue
            representative_positions.append(
                _active_surface_source_medoid(active_positions)
            )
            representative_weights.append(float(weight))
            flat_positions.extend(active_positions)
            flat_weights.extend([float(weight)] * int(active_positions.shape[0]))
        return (
            np.vstack(flat_positions)
            if flat_positions
            else np.zeros((0, 3), dtype=float),
            np.asarray(flat_weights, dtype=float),
            np.vstack(representative_positions)
            if representative_positions
            else np.zeros((0, 3), dtype=float),
            np.asarray(representative_weights, dtype=float),
        )

    packed_positions, _strengths, active_mask, _chart_ids, _surface_uv = packer()
    positions = np.asarray(packed_positions, dtype=float)
    mask = np.asarray(active_mask, dtype=bool)
    if (
        positions.ndim != 3
        or positions.shape[2] != 3
        or mask.shape != positions.shape[:2]
        or weights.shape != positions.shape[:1]
        or np.any(~np.isfinite(positions))
        or np.any(~np.isfinite(weights))
    ):
        raise RuntimeError(
            "PF visualization received invalid batched continuous states."
        )
    source_counts = np.sum(mask, axis=1, dtype=np.int64)
    nonempty = source_counts > 0
    flat_positions = np.asarray(positions[mask], dtype=float)
    flat_weights = np.repeat(weights, source_counts)
    if not np.any(nonempty):
        return (
            flat_positions,
            flat_weights,
            np.zeros((0, 3), dtype=float),
            np.zeros(0, dtype=float),
        )
    differences = positions[:, :, None, :] - positions[:, None, :, :]
    pairwise = np.linalg.norm(differences, axis=3)
    valid_pairs = mask[:, :, None] & mask[:, None, :]
    candidate_cost = np.sum(
        np.where(valid_pairs, pairwise, 0.0),
        axis=2,
    )
    candidate_cost = np.where(mask, candidate_cost, np.inf)
    representative_slots = np.argmin(candidate_cost, axis=1)
    row_indices = np.arange(positions.shape[0], dtype=np.int64)
    representative_positions = positions[
        row_indices[nonempty],
        representative_slots[nonempty],
    ]
    return (
        flat_positions,
        flat_weights,
        np.asarray(representative_positions, dtype=float),
        weights[nonempty],
    )




class CUISplitPFVisualizer:
    """
    Save CUI-friendly split visualizations as independent image files.

    The renderer writes a 2D robot/trajectory panel and a 3D PF particle panel
    after every update. It also writes a small auto-refresh HTML page so the
    latest CUI state can be inspected in a browser without starting Isaac Sim
    or an interactive matplotlib window.
    """

    def __init__(
        self,
        isotopes: list[str],
        output_dir: str | Path,
        *,
        world_bounds: tuple[float, float, float, float, float, float] | None = None,
        true_sources: dict[str, NDArray[np.float64]] | None = None,
        true_strengths: dict[str, float | Sequence[float]] | None = None,
        obstacle_grid: ObstacleGrid | None = None,
        max_particles_per_isotope: int | None = None,
        source_label_neighborhood_m: float = 1.0,
        save_step_history: bool = False,
    ) -> None:
        """Initialize output paths and static scene metadata."""
        self.isotopes = list(isotopes)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.world_bounds = world_bounds or (0.0, 10.0, 0.0, 10.0, 0.0, 3.0)
        self.true_sources = true_sources or {}
        self.true_strengths = true_strengths or {}
        self.obstacle_grid = obstacle_grid
        xmin, xmax, ymin, ymax, zmin, zmax = self.world_bounds
        self.cui_scene = CUIScene.from_environment(
            EnvironmentConfig(
                size_x=float(xmax - xmin),
                size_y=float(ymax - ymin),
                size_z=float(zmax - zmin),
            ),
            obstacle_grid,
            obstacle_height_m=min(2.0, float(zmax - zmin)),
        )
        self.max_particles_per_isotope = (
            None
            if max_particles_per_isotope is None
            else max(1, int(max_particles_per_isotope))
        )
        self.source_label_neighborhood_m = float(source_label_neighborhood_m)
        if (
            not np.isfinite(self.source_label_neighborhood_m)
            or self.source_label_neighborhood_m <= 0.0
        ):
            raise ValueError("source_label_neighborhood_m must be positive and finite.")
        if not isinstance(save_step_history, bool):
            raise TypeError("save_step_history must be a boolean.")
        self.save_step_history = save_step_history
        self.path_segments: list[NDArray[np.float64]] = []
        self.measurement_points: list[NDArray[np.float64]] = []
        self.measurement_station_ids: list[int] = []
        self.measurement_steps: list[int] = []
        self.measurement_visit_counts: list[int] = []
        self.update_index = 0
        cmap = plt.get_cmap("tab10")
        self.colors = {
            iso: DEFAULT_ISOTOPE_COLORS.get(iso, cmap(i % 10))
            for i, iso in enumerate(self.isotopes)
        }
        self.latest_robot_path = self.output_dir / "latest_robot_2d.png"
        self.latest_overview_path = self.output_dir / "latest_experiment_overview.png"
        self.latest_pf_path = self.output_dir / "latest_pf_3d.png"
        self.latest_pf_labeled_path = self.output_dir / "latest_pf_3d_labeled.png"
        self.latest_spectrum_path = self.output_dir / "latest_spectrum.png"
        self.index_path = self.output_dir / "index.html"
        self._write_index_html()
        if not self.latest_overview_path.exists():
            self._save_overview_placeholder(self.latest_overview_path)
        if not self.latest_spectrum_path.exists():
            self._save_spectrum_placeholder(self.latest_spectrum_path)

    def set_truth(
        self,
        true_sources: dict[str, NDArray[np.float64]],
        true_strengths: dict[str, float | Sequence[float]],
    ) -> None:
        """Attach evaluation-only truth without exposing it to the estimator."""
        self.true_sources = {
            str(isotope): np.asarray(values, dtype=np.float64).copy()
            for isotope, values in true_sources.items()
        }
        self.true_strengths = {
            str(isotope): np.asarray(values, dtype=np.float64).copy()
            for isotope, values in true_strengths.items()
        }
        self._write_index_html()

    def update(self, frame: PFFrame) -> None:
        """Render and save the split CUI views for one PF frame."""
        if not isinstance(frame.cui_route, CUIRoute):
            raise TypeError("PF CUI frames require a cumulative runtime CUIRoute.")
        self.update_index += 1
        setattr(frame, "_cui_update_index", int(self.update_index))
        self._apply_route(frame.cui_route)
        step = max(0, int(frame.step_index))
        rendered_paths = (
            self._frame_render_path(
                self.latest_robot_path,
                f"robot_2d_step_{step:04d}.png",
            ),
            self._frame_render_path(
                self.latest_overview_path,
                f"experiment_overview_step_{step:04d}.png",
            ),
            self._frame_render_path(
                self.latest_pf_path,
                f"pf_3d_step_{step:04d}.png",
            ),
            self._frame_render_path(
                self.latest_pf_labeled_path,
                f"pf_3d_labeled_step_{step:04d}.png",
            ),
            self._frame_render_path(
                self.latest_spectrum_path,
                f"spectrum_step_{step:04d}.png",
            ),
        )
        (
            robot_path,
            overview_path,
            pf_path,
            pf_labeled_path,
            spectrum_path,
        ) = rendered_paths
        try:
            self._save_robot_2d(frame, robot_path)
            atomic_copy_file(robot_path, self.latest_robot_path)
            self._save_experiment_overview(frame, overview_path)
            atomic_copy_file(overview_path, self.latest_overview_path)
            self._save_pf_3d(
                frame,
                pf_path,
                labeled_output_path=pf_labeled_path,
            )
            atomic_copy_file(pf_path, self.latest_pf_path)
            atomic_copy_file(pf_labeled_path, self.latest_pf_labeled_path)
            self._save_spectrum(frame, spectrum_path)
            if spectrum_path.exists():
                atomic_copy_file(spectrum_path, self.latest_spectrum_path)
        finally:
            if not self.save_step_history:
                for rendered_path in rendered_paths:
                    rendered_path.unlink(missing_ok=True)

    def _frame_render_path(self, latest_path: Path, history_name: str) -> Path:
        """Return a persistent history path or an ephemeral staging path."""
        if self.save_step_history:
            return self.output_dir / history_name
        return _temporary_png_path(latest_path)

    def _apply_route(self, route: CUIRoute) -> None:
        """Replace renderer route state from one cumulative shared snapshot."""
        self.path_segments = [
            np.asarray(segment, dtype=np.float64).copy()
            for segment in route.travel_path_segments_xyz
        ]
        self.measurement_points = [
            np.asarray(point, dtype=np.float64).copy()
            for point in route.measurement_stations_xyz
        ]
        self.measurement_station_ids = [
            int(value) for value in route.measurement_station_ids
        ]
        self.measurement_steps = [
            int(value) for value in route.measurement_step_ids
        ]
        self.measurement_visit_counts = [
            int(value) for value in route.measurement_visit_counts
        ]

    def _unique_path_waypoints(self) -> NDArray[np.float64]:
        """Return traversed path waypoints that are not measurement stations."""
        waypoints: list[NDArray[np.float64]] = []
        station_arr = (
            np.vstack(self.measurement_points)
            if self.measurement_points
            else np.zeros((0, 3), dtype=float)
        )
        for segment in self.path_segments:
            if segment.shape[0] <= 2:
                continue
            for point in segment[1:-1]:
                if station_arr.size:
                    distances = np.linalg.norm(station_arr - point[None, :], axis=1)
                    if float(np.min(distances)) <= 1e-6:
                        continue
                if any(
                    float(np.linalg.norm(point - existing)) <= 1e-6
                    for existing in waypoints
                ):
                    continue
                waypoints.append(np.asarray(point, dtype=float).reshape(3).copy())
        if not waypoints:
            return np.zeros((0, 3), dtype=float)
        return np.vstack(waypoints).astype(float)

    def _station_label_offsets(
        self, points: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return small deterministic xy offsets for overlapping station labels."""
        point_arr = np.asarray(points, dtype=float)
        if point_arr.size == 0:
            return np.zeros((0, 2), dtype=float)
        offsets = np.zeros((point_arr.shape[0], 2), dtype=float)
        used_counts: dict[tuple[float, float], int] = {}
        radius = 0.16
        for idx, point in enumerate(point_arr):
            key = tuple(float(v) for v in np.round(point[:2], 3))
            repeat_idx = used_counts.get(key, 0)
            used_counts[key] = repeat_idx + 1
            if repeat_idx == 0:
                continue
            angle = 2.0 * np.pi * float(repeat_idx - 1) / 6.0
            offsets[idx, 0] = radius * np.cos(angle)
            offsets[idx, 1] = radius * np.sin(angle)
        return offsets

    def _station_label(self, station_index: int) -> str:
        """Return a compact station label including repeated shield visits."""
        station_id = (
            self.measurement_station_ids[station_index]
            if station_index < len(self.measurement_station_ids)
            else station_index
        )
        visits = (
            self.measurement_visit_counts[station_index]
            if station_index < len(self.measurement_visit_counts)
            else 1
        )
        if visits <= 1:
            return str(station_id)
        return f"{station_id}({visits})"

    def _frame_progress_label(self, frame: PFFrame) -> str:
        """Return a title suffix that separates measurement, render, and station progress."""
        update_idx = int(getattr(frame, "_cui_update_index", 0))
        station_id = (
            self.measurement_station_ids[-1]
            if self.measurement_station_ids
            else 0
        )
        visit_count = (
            int(self.measurement_visit_counts[-1])
            if self.measurement_visit_counts
            else 0
        )
        return (
            f"measurement={int(frame.step_index)} "
            f"update={update_idx} "
            f"station={station_id} visit={visit_count} "
            f"t={frame.time:.1f}s"
        )

    def _write_index_html(self) -> None:
        """Publish the shared context shell with PF particle result panels."""
        title = "Rotating Shield PF CUI View"
        if self.true_sources:
            title += (
                " — truth: visible "
                "(evaluation overlay only; not provided to PF/planner)"
            )
        self.index_path = write_cui_index(
            self.output_dir,
            pf_cui_panel_specs(),
            title=title,
            refresh_interval_ms=2000,
        )

    def _save_overview_placeholder(self, output_path: Path) -> None:
        """Save a placeholder RA-L overview panel until the first frame arrives."""
        fig, ax = plt.subplots(figsize=(12.0, 5.2))
        ax.text(
            0.5,
            0.5,
            "RA-L experiment overview will appear after the first measurement",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_axis_off()
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)

    def _save_spectrum_placeholder(self, output_path: Path) -> None:
        """Save a placeholder spectrum panel until the first measurement arrives."""
        fig, ax = plt.subplots(figsize=(10.0, 4.8))
        ax.text(
            0.5,
            0.5,
            "Spectrum will appear after the first measurement",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_axis_off()
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)

    def _draw_obstacles_2d(self, ax: plt.Axes) -> None:
        """Draw canonical runtime obstacle footprints on a 2-D axis."""
        from matplotlib.patches import Polygon

        for footprint in self.cui_scene.obstacle_footprints_xy:
            ax.add_patch(
                Polygon(
                    footprint,
                    closed=True,
                    facecolor="black",
                    edgecolor="none",
                    alpha=0.75,
                )
            )

    def _draw_obstacles_3d(self, ax: plt.Axes) -> None:
        """Draw canonical runtime obstacles as flat 3-D floor patches."""
        patches = []
        z0 = float(self.world_bounds[4])
        for footprint in self.cui_scene.obstacle_footprints_xy:
            patches.append(
                [(float(x), float(y), z0) for x, y in footprint]
            )
        if not patches:
            return
        collection = Poly3DCollection(
            patches,
            facecolor="black",
            edgecolor="none",
            alpha=0.25,
        )
        ax.add_collection3d(collection)

    def _plot_true_sources_2d(self, ax: plt.Axes) -> None:
        """Plot numbered true sources and XYZ labels on a top-down view."""
        xmin, xmax, ymin, ymax, _, _ = self.world_bounds
        for iso, positions in self.true_sources.items():
            pos = np.asarray(positions, dtype=float)
            if pos.size == 0:
                continue
            pos = pos.reshape((-1, 3))
            ax.scatter(
                pos[:, 0],
                pos[:, 1],
                marker="*",
                s=90,
                color=self.colors.get(iso, "black"),
                edgecolor="white",
                linewidth=0.6,
                label=f"true {iso}",
                zorder=18,
            )
            for index, position in enumerate(pos):
                offset_x, offset_y, horizontal_alignment, vertical_alignment = (
                    self._inward_annotation_offset(
                        float(position[0]),
                        float(position[1]),
                        x_bounds=(float(xmin), float(xmax)),
                        y_bounds=(float(ymin), float(ymax)),
                        index=index,
                    )
                )
                ax.annotate(
                    self._truth_source_label(iso, index, position),
                    xy=(float(position[0]), float(position[1])),
                    xytext=(offset_x, offset_y),
                    textcoords="offset points",
                    ha=horizontal_alignment,
                    va=vertical_alignment,
                    fontsize=6.2,
                    fontweight="bold",
                    color=self.colors.get(iso, "black"),
                    bbox={
                        "boxstyle": "round,pad=0.16",
                        "facecolor": "white",
                        "edgecolor": self.colors.get(iso, "black"),
                        "linewidth": 0.55,
                        "alpha": 0.86,
                    },
                    zorder=20,
                )

    def _plot_estimated_sources_2d(self, ax: plt.Axes, frame: PFFrame) -> None:
        """Plot estimated source positions on a 2D map view."""
        for iso in self.isotopes:
            est = frame.estimated_sources.get(iso, np.zeros((0, 3), dtype=float))
            if np.asarray(est, dtype=float).size == 0:
                continue
            est_arr = np.asarray(est, dtype=float).reshape((-1, 3))
            strengths = np.asarray(
                frame.estimated_strengths.get(iso, np.zeros(0, dtype=float)),
                dtype=float,
            ).reshape(-1)
            sizes = 110.0 + 0.02 * np.clip(strengths, 0.0, 5000.0)
            if sizes.size != est_arr.shape[0]:
                sizes = np.full(est_arr.shape[0], 130.0, dtype=float)
            ax.scatter(
                est_arr[:, 0],
                est_arr[:, 1],
                marker="x",
                s=sizes,
                color=self.colors.get(iso, "black"),
                linewidths=2.2,
                label=f"estimate {iso}",
                zorder=12,
            )

    def _plot_source_match_segments_2d(self, ax: plt.Axes, frame: PFFrame) -> None:
        """Draw nearest truth-to-estimate links for same-isotope source matches."""
        for iso, truth_raw in self.true_sources.items():
            truth = np.asarray(truth_raw, dtype=float)
            est = np.asarray(
                frame.estimated_sources.get(iso, np.zeros((0, 3), dtype=float)),
                dtype=float,
            )
            if truth.size == 0 or est.size == 0:
                continue
            truth = truth.reshape((-1, 3))
            est = est.reshape((-1, 3))
            color = self.colors.get(iso, "black")
            used_label = False
            for src in truth:
                distances = np.linalg.norm(est - src[None, :], axis=1)
                nearest = est[int(np.argmin(distances))]
                ax.plot(
                    [src[0], nearest[0]],
                    [src[1], nearest[1]],
                    "--",
                    color=color,
                    alpha=0.45,
                    linewidth=1.0,
                    label="truth-estimate link" if not used_label else None,
                    zorder=4,
                )
                used_label = True

    def _plot_true_sources_xz(self, ax: plt.Axes) -> None:
        """Plot numbered true sources and XYZ labels in x-z projection."""
        xmin, xmax, _, _, zmin, zmax = self.world_bounds
        for iso, positions in self.true_sources.items():
            pos = np.asarray(positions, dtype=float)
            if pos.size == 0:
                continue
            pos = pos.reshape((-1, 3))
            ax.scatter(
                pos[:, 0],
                pos[:, 2],
                marker="*",
                s=90,
                color=self.colors.get(iso, "black"),
                edgecolor="white",
                linewidth=0.6,
                label=f"true {iso}",
                zorder=18,
            )
            for index, position in enumerate(pos):
                offset_x, offset_y, horizontal_alignment, vertical_alignment = (
                    self._inward_annotation_offset(
                        float(position[0]),
                        float(position[2]),
                        x_bounds=(float(xmin), float(xmax)),
                        y_bounds=(float(zmin), float(zmax)),
                        index=index,
                    )
                )
                ax.annotate(
                    self._truth_source_id_label(iso, index),
                    xy=(float(position[0]), float(position[2])),
                    xytext=(offset_x, offset_y),
                    textcoords="offset points",
                    ha=horizontal_alignment,
                    va=vertical_alignment,
                    fontsize=5.8,
                    fontweight="bold",
                    color=self.colors.get(iso, "black"),
                    bbox={
                        "boxstyle": "round,pad=0.14",
                        "facecolor": "white",
                        "edgecolor": self.colors.get(iso, "black"),
                        "linewidth": 0.5,
                        "alpha": 0.84,
                    },
                    zorder=20,
                )

    def _plot_estimated_sources_xz(self, ax: plt.Axes, frame: PFFrame) -> None:
        """Plot estimated source positions in an x-z elevation projection."""
        for iso in self.isotopes:
            est = np.asarray(
                frame.estimated_sources.get(iso, np.zeros((0, 3), dtype=float)),
                dtype=float,
            )
            if est.size == 0:
                continue
            est = est.reshape((-1, 3))
            ax.scatter(
                est[:, 0],
                est[:, 2],
                marker="x",
                s=120,
                color=self.colors.get(iso, "black"),
                linewidths=2.2,
                label=f"estimate {iso}",
                zorder=11,
            )

    def _plot_source_match_segments_xz(self, ax: plt.Axes, frame: PFFrame) -> None:
        """Draw nearest same-isotope truth-to-estimate links in x-z projection."""
        for iso, truth_raw in self.true_sources.items():
            truth = np.asarray(truth_raw, dtype=float)
            est = np.asarray(
                frame.estimated_sources.get(iso, np.zeros((0, 3), dtype=float)),
                dtype=float,
            )
            if truth.size == 0 or est.size == 0:
                continue
            truth = truth.reshape((-1, 3))
            est = est.reshape((-1, 3))
            color = self.colors.get(iso, "black")
            for src in truth:
                distances = np.linalg.norm(est - src[None, :], axis=1)
                nearest = est[int(np.argmin(distances))]
                ax.plot(
                    [src[0], nearest[0]],
                    [src[2], nearest[2]],
                    "--",
                    color=color,
                    alpha=0.45,
                    linewidth=1.0,
                    zorder=4,
                )

    def _overview_summary_text(self, frame: PFFrame) -> str:
        """Return a compact textual source-count summary for the overview panel."""
        lines = [self._frame_progress_label(frame)]
        truth_visible = bool(self.true_sources)
        for iso in self.isotopes:
            truth_label = "hidden"
            if iso in self.true_sources:
                truth_label = str(
                    int(
                        np.asarray(
                            self.true_sources[iso],
                            dtype=float,
                        )
                        .reshape((-1, 3))
                        .shape[0]
                    )
                )
            elif truth_visible:
                truth_label = "0"
            est_count = int(
                np.asarray(
                    frame.estimated_sources.get(iso, np.zeros((0, 3), dtype=float)),
                    dtype=float,
                )
                .reshape((-1, 3))
                .shape[0]
            )
            lines.append(f"{iso}: truth={truth_label} estimate={est_count}")
        return "\n".join(lines)

    def _plot_true_sources_3d(self, ax: plt.Axes) -> None:
        """Plot true source positions on the 3D PF view when available."""
        for iso, positions in self.true_sources.items():
            pos = np.asarray(positions, dtype=float)
            if pos.size == 0:
                continue
            pos = pos.reshape((-1, 3))
            ax.scatter(
                pos[:, 0],
                pos[:, 1],
                pos[:, 2],
                marker="*",
                s=100,
                color=self.colors.get(iso, "black"),
                edgecolor="white",
                linewidth=0.7,
                depthshade=False,
                label=f"true {iso}",
                zorder=25,
            )

    @staticmethod
    def _isotope_source_prefix(isotope: str) -> str:
        """Return a compact isotope prefix for per-source plot labels."""
        prefix = str(isotope).split("-", maxsplit=1)[0].strip()
        return prefix or str(isotope)

    @staticmethod
    def _inward_annotation_offset(
        x: float,
        y: float,
        *,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        index: int,
    ) -> tuple[int, int, str, str]:
        """Return an annotation offset directed toward the plot interior."""
        x_midpoint = 0.5 * (float(x_bounds[0]) + float(x_bounds[1]))
        y_midpoint = 0.5 * (float(y_bounds[0]) + float(y_bounds[1]))
        horizontal_offset = -5 if float(x) >= x_midpoint else 5
        vertical_magnitude = 5 + 4 * (int(index) % 2)
        vertical_offset = (
            -vertical_magnitude if float(y) >= y_midpoint else vertical_magnitude
        )
        return (
            horizontal_offset,
            vertical_offset,
            "right" if horizontal_offset < 0 else "left",
            "top" if vertical_offset < 0 else "bottom",
        )

    @classmethod
    def _truth_source_label(
        cls,
        isotope: str,
        source_index: int,
        position_xyz_m: NDArray[np.float64],
    ) -> str:
        """Return one numbered truth label with its full XYZ coordinates."""
        position = np.asarray(position_xyz_m, dtype=np.float64).reshape(3)
        return (
            f"{cls._truth_source_id_label(isotope, source_index)}\n"
            f"({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}) m"
        )

    @classmethod
    def _truth_source_id_label(cls, isotope: str, source_index: int) -> str:
        """Return one compact isotope-local truth-source identifier."""
        prefix = cls._isotope_source_prefix(isotope)
        return f"{prefix}-{int(source_index) + 1} T"

    def _truth_inventory_text(self) -> str:
        """Return the complete numbered XYZ truth inventory for an overview."""
        if not self.true_sources:
            return "True sources: hidden"
        lines = ["True sources (evaluation overlay)"]
        for isotope in self.isotopes:
            raw_positions = np.asarray(
                self.true_sources.get(isotope, np.zeros((0, 3), dtype=float)),
                dtype=float,
            )
            if raw_positions.size == 0:
                continue
            for index, position in enumerate(raw_positions.reshape((-1, 3))):
                lines.append(
                    self._truth_source_label(isotope, index, position).replace(
                        "\n",
                        "  ",
                    )
                )
        return "\n".join(lines)

    def _source_label_entries(
        self,
        frame: PFFrame,
        isotope: str,
    ) -> tuple[
        list[tuple[NDArray[np.float64], str]],
        list[tuple[NDArray[np.float64], str]],
    ]:
        """Return truth and estimate labels using same-isotope neighborhoods.

        Estimate labels are assigned to the nearest same-isotope truth within
        ``source_label_neighborhood_m``. Multiple estimates may therefore carry
        the same source identifier with deterministic ``E1``, ``E2``, ...
        suffixes. Estimates outside every neighborhood remain explicit remotes.
        This helper is visualization-only and never changes PF state.
        """
        prefix = self._isotope_source_prefix(isotope)
        truth_raw = np.asarray(
            self.true_sources.get(isotope, np.zeros((0, 3), dtype=float)),
            dtype=float,
        )
        estimate_raw = np.asarray(
            frame.estimated_sources.get(
                isotope,
                np.zeros((0, 3), dtype=float),
            ),
            dtype=float,
        )
        truth = (
            truth_raw.reshape((-1, 3))
            if truth_raw.size
            else np.zeros((0, 3), dtype=float)
        )
        estimates = (
            estimate_raw.reshape((-1, 3))
            if estimate_raw.size
            else np.zeros((0, 3), dtype=float)
        )
        truth_entries = [
            (position, self._truth_source_label(isotope, index, position))
            for index, position in enumerate(truth)
        ]
        if not self.true_sources:
            estimate_entries = [
                (position, f"{prefix} E{index + 1}")
                for index, position in enumerate(estimates)
            ]
            return truth_entries, estimate_entries

        assignments: list[int | None] = []
        for estimate in estimates:
            if not truth.size:
                assignments.append(None)
                continue
            distances = np.linalg.norm(truth - estimate[None, :], axis=1)
            nearest_index = int(np.argmin(distances))
            if float(distances[nearest_index]) <= self.source_label_neighborhood_m:
                assignments.append(nearest_index)
            else:
                assignments.append(None)

        assignment_counts: dict[int, int] = {}
        for assignment in assignments:
            if assignment is None:
                continue
            assignment_counts[assignment] = assignment_counts.get(assignment, 0) + 1
        assignment_occurrences: dict[int, int] = {}
        remote_index = 0
        estimate_entries: list[tuple[NDArray[np.float64], str]] = []
        for estimate, assignment in zip(estimates, assignments):
            if assignment is None:
                remote_index += 1
                label = f"{prefix} remote-{remote_index}"
            else:
                assignment_occurrences[assignment] = (
                    assignment_occurrences.get(assignment, 0) + 1
                )
                estimate_suffix = "E"
                if assignment_counts[assignment] > 1:
                    estimate_suffix += str(assignment_occurrences[assignment])
                label = f"{prefix}-{assignment + 1} {estimate_suffix}"
            estimate_entries.append((estimate, label))
        return truth_entries, estimate_entries

    def _annotate_source_labels_3d(
        self,
        ax: plt.Axes,
        frame: PFFrame,
        *,
        include_truth: bool,
        include_estimates: bool,
    ) -> None:
        """Annotate selected truth or estimate markers on one 3-D axis."""
        estimate_label_index = 0
        for isotope in self.isotopes:
            truth_entries, estimate_entries = self._source_label_entries(
                frame,
                isotope,
            )
            color = self.colors.get(isotope, "black")
            visible_truth_entries = truth_entries if include_truth else []
            for index, (position, label) in enumerate(visible_truth_entries):
                projected_x, projected_y, _ = proj3d.proj_transform(
                    *position,
                    ax.get_proj(),
                )
                offset_x = -4 if projected_x >= 0.0 else 4
                offset_y_magnitude = 5 + 3 * (index % 2)
                offset_y = (
                    -offset_y_magnitude
                    if projected_y >= 0.0
                    else offset_y_magnitude
                )
                annotation = ax.annotate(
                    label,
                    xy=(projected_x, projected_y),
                    xytext=(offset_x, offset_y),
                    textcoords="offset points",
                    ha="right" if offset_x < 0 else "left",
                    va="top" if offset_y < 0 else "bottom",
                    fontsize=6.2,
                    fontweight="bold",
                    color=color,
                    bbox={
                        "boxstyle": "round,pad=0.18",
                        "facecolor": "white",
                        "edgecolor": color,
                        "linewidth": 0.65,
                        "alpha": 0.88,
                    },
                    zorder=30,
                )
                annotation.set_path_effects(
                    [path_effects.withStroke(linewidth=0.6, foreground="white")]
                )
            visible_estimate_entries = estimate_entries if include_estimates else []
            for position, label in visible_estimate_entries:
                projected_x, projected_y, _ = proj3d.proj_transform(
                    *position,
                    ax.get_proj(),
                )
                offset_candidates = (
                    (6, -10),
                    (6, 8),
                    (-6, -10),
                    (-6, 8),
                    (12, 0),
                    (-12, 0),
                )
                if label.endswith(" E1"):
                    offset_x, offset_y = offset_candidates[0]
                elif label.endswith(" E2"):
                    offset_x, offset_y = offset_candidates[1]
                else:
                    offset_x, offset_y = offset_candidates[
                        estimate_label_index % len(offset_candidates)
                    ]
                estimate_label_index += 1
                annotation = ax.annotate(
                    label,
                    xy=(projected_x, projected_y),
                    xytext=(offset_x, offset_y),
                    textcoords="offset points",
                    ha="left" if offset_x >= 0 else "right",
                    va="bottom" if offset_y >= 0 else "top",
                    fontsize=5.8,
                    color=color,
                    bbox={
                        "boxstyle": "round,pad=0.14",
                        "facecolor": "white",
                        "edgecolor": color,
                        "linewidth": 0.5,
                        "alpha": 0.78,
                    },
                    arrowprops={
                        "arrowstyle": "-",
                        "color": color,
                        "linewidth": 0.45,
                        "alpha": 0.7,
                        "shrinkA": 1.5,
                        "shrinkB": 1.5,
                    },
                    zorder=29,
                )
                annotation.set_path_effects(
                    [path_effects.withStroke(linewidth=0.5, foreground="white")]
                )

    def _save_robot_2d(self, frame: PFFrame, output_path: Path) -> None:
        """Save the current robot position and trajectory as a 2D PNG."""
        xmin, xmax, ymin, ymax, _, _ = self.world_bounds
        fig, ax = plt.subplots(figsize=(7.0, 6.0))
        self._draw_obstacles_2d(ax)
        self._plot_true_sources_2d(ax)
        for idx, segment in enumerate(self.path_segments):
            if segment.shape[0] < 2:
                continue
            ax.plot(
                segment[:, 0],
                segment[:, 1],
                "-",
                color="cyan",
                linewidth=2.0,
                alpha=0.75,
                label="traversed path" if idx == 0 else None,
            )
        path_waypoints = self._unique_path_waypoints()
        if path_waypoints.size:
            ax.scatter(
                path_waypoints[:, 0],
                path_waypoints[:, 1],
                s=18,
                color="cyan",
                edgecolor="black",
                linewidth=0.3,
                alpha=0.55,
                marker=".",
                label="path waypoint",
                zorder=6,
            )
        if self.measurement_points:
            points = np.vstack(self.measurement_points)
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=55,
                color="white",
                edgecolor="cyan",
                linewidth=1.0,
                label="measurement station",
                zorder=9,
            )
            offsets = self._station_label_offsets(points)
            for idx, point in enumerate(points):
                label = self._station_label(idx)
                text = ax.text(
                    point[0] + offsets[idx, 0],
                    point[1] + offsets[idx, 1],
                    label,
                    color="black",
                    fontsize=8,
                    ha="center",
                    va="center",
                    zorder=10,
                )
                text.set_path_effects(
                    [
                        path_effects.withStroke(
                            linewidth=1.8,
                            foreground="white",
                        )
                    ]
                )
        robot = np.asarray(frame.robot_position, dtype=float)
        ax.scatter(
            [robot[0]],
            [robot[1]],
            s=130,
            color="cyan",
            edgecolor="black",
            linewidth=1.0,
            label="robot",
            zorder=10,
        )
        self._plot_estimated_sources_2d(ax, frame)
        ax.set_xlim(float(xmin), float(xmax))
        ax.set_ylim(float(ymin), float(ymax))
        _apply_metric_ticks_2d(
            ax,
            xlim=(float(xmin), float(xmax)),
            ylim=(float(ymin), float(ymax)),
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"Robot 2D position - {self._frame_progress_label(frame)}")
        ax.grid(True, alpha=0.25)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize=8,
        )
        fig.subplots_adjust(right=0.74)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def _save_experiment_overview(self, frame: PFFrame, output_path: Path) -> None:
        """Save a paper-oriented overview with map, estimates, and elevation view."""
        xmin, xmax, ymin, ymax, zmin, zmax = self.world_bounds
        fig = plt.figure(figsize=(11.2, 8.0))
        grid = fig.add_gridspec(
            2,
            2,
            width_ratios=(1.0, 1.0),
            height_ratios=(1.0, 1.0),
        )
        map_ax = fig.add_subplot(grid[:, 0])
        elev_ax = fig.add_subplot(grid[0, 1])
        info_ax = fig.add_subplot(grid[1, 1])
        info_ax.axis("off")
        self._draw_obstacles_2d(map_ax)
        self._plot_source_match_segments_2d(map_ax, frame)
        self._plot_true_sources_2d(map_ax)
        for idx, segment in enumerate(self.path_segments):
            if segment.shape[0] < 2:
                continue
            map_ax.plot(
                segment[:, 0],
                segment[:, 1],
                "-",
                color="cyan",
                linewidth=2.0,
                alpha=0.72,
                label="traversed path" if idx == 0 else None,
                zorder=5,
            )
        if self.measurement_points:
            points = np.vstack(self.measurement_points)
            map_ax.scatter(
                points[:, 0],
                points[:, 1],
                s=48,
                color="white",
                edgecolor="cyan",
                linewidth=1.0,
                label="measurement station",
                zorder=9,
            )
        robot = np.asarray(frame.robot_position, dtype=float).reshape(3)
        map_ax.scatter(
            [robot[0]],
            [robot[1]],
            s=125,
            color="cyan",
            edgecolor="black",
            linewidth=1.0,
            label="robot",
            zorder=13,
        )
        self._plot_estimated_sources_2d(map_ax, frame)
        map_ax.set_xlim(float(xmin), float(xmax))
        map_ax.set_ylim(float(ymin), float(ymax))
        _apply_metric_ticks_2d(
            map_ax,
            xlim=(float(xmin), float(xmax)),
            ylim=(float(ymin), float(ymax)),
        )
        map_ax.set_aspect("equal", adjustable="box")
        map_ax.set_xlabel("x [m]")
        map_ax.set_ylabel("y [m]")
        map_ax.set_title("Top-down map: obstacles, path, truth, and estimates")
        map_ax.grid(True, alpha=0.25)

        self._plot_source_match_segments_xz(elev_ax, frame)
        self._plot_true_sources_xz(elev_ax)
        self._plot_estimated_sources_xz(elev_ax, frame)
        if self.measurement_points:
            points = np.vstack(self.measurement_points)
            elev_ax.scatter(
                points[:, 0],
                points[:, 2],
                s=28,
                color="cyan",
                edgecolor="black",
                linewidth=0.4,
                alpha=0.55,
                label="station height",
                zorder=6,
            )
        elev_ax.axhline(float(zmin), color="black", linewidth=0.8, alpha=0.45)
        elev_ax.axhline(float(zmax), color="black", linewidth=0.8, alpha=0.25)
        elev_ax.set_xlim(float(xmin), float(xmax))
        elev_ax.set_ylim(float(zmin), float(zmax))
        _apply_metric_ticks_2d(
            elev_ax,
            xlim=(float(xmin), float(xmax)),
            ylim=(float(zmin), float(zmax)),
        )
        elev_ax.set_aspect("equal", adjustable="box")
        elev_ax.set_xlabel("x [m]")
        elev_ax.set_ylabel("z [m]")
        elev_ax.set_title("Elevation projection: height ambiguity")
        elev_ax.grid(True, alpha=0.25)
        handles, labels = map_ax.get_legend_handles_labels()
        elev_handles, elev_labels = elev_ax.get_legend_handles_labels()
        legend_by_label = dict(zip(labels + elev_labels, handles + elev_handles))
        if legend_by_label:
            info_ax.legend(
                legend_by_label.values(),
                legend_by_label.keys(),
                loc="upper left",
                fontsize=7,
                frameon=True,
            )
        info_ax.text(
            0.0,
            0.52,
            self._truth_inventory_text(),
            ha="left",
            va="top",
            fontsize=7.2,
            transform=info_ax.transAxes,
        )
        info_ax.text(
            0.0,
            0.02,
            self._overview_summary_text(frame),
            ha="left",
            va="bottom",
            fontsize=8,
            transform=info_ax.transAxes,
            wrap=True,
        )
        fig.suptitle("RA-L experiment overview", fontsize=13, fontweight="bold")
        fig.subplots_adjust(
            left=0.06,
            right=0.98,
            top=0.90,
            bottom=0.08,
            wspace=0.25,
            hspace=0.32,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def _particle_subset(
        self,
        positions: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return a deterministic particle subset for display if a cap is set."""
        pts = np.asarray(positions, dtype=float)
        w = np.asarray(weights, dtype=float)
        if pts.size == 0:
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        if w.shape != (pts.shape[0],):
            raise ValueError(
                "PF visualization requires one posterior weight per displayed particle."
            )
        if (
            self.max_particles_per_isotope is None
            or pts.shape[0] <= self.max_particles_per_isotope
        ):
            return pts, w
        indices = np.argsort(w)[::-1][: self.max_particles_per_isotope]
        return pts[indices], w[indices]

    def _particle_style(
        self,
        weights: NDArray[np.float64],
        color: object,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Map display particle weights to marker sizes and RGBA colors."""
        w = np.asarray(weights, dtype=float)
        if w.size == 0:
            return np.full(0, 6.0), np.zeros((0, 4), dtype=float)
        w_norm = _normalize_weights(w)
        if float(np.max(w_norm) - np.min(w_norm)) > 1e-12:
            w_norm = (w_norm - np.min(w_norm)) / (np.max(w_norm) - np.min(w_norm))
        else:
            w_norm = np.ones_like(w_norm)
        sizes = 8.0 + 36.0 * w_norm
        rgba = np.tile(mcolors.to_rgba(color), (w_norm.size, 1))
        rgba[:, 3] = 0.18 + 0.62 * w_norm
        return sizes, rgba

    def _representative_particle_style(
        self,
        weights: NDArray[np.float64],
        color: object,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Map per-PF-particle representative weights to compact display markers."""
        w = np.asarray(weights, dtype=float)
        if w.size == 0:
            return np.full(0, 3.0), np.zeros((0, 4), dtype=float)
        w_norm = _normalize_weights(w)
        if float(np.max(w_norm) - np.min(w_norm)) > 1.0e-12:
            w_norm = (w_norm - np.min(w_norm)) / (np.max(w_norm) - np.min(w_norm))
        else:
            w_norm = np.ones_like(w_norm)
        sizes = 2.2 + 8.0 * w_norm
        rgba = np.tile(mcolors.to_rgba(color), (w_norm.size, 1))
        rgba[:, 3] = 0.07 + 0.25 * w_norm
        return sizes, rgba

    def _draw_pf_scene_context(
        self,
        ax: plt.Axes,
        frame: PFFrame,
        *,
        show_legend_context: bool,
    ) -> None:
        """Draw static PF scene context shared by source-sample and particle panels."""
        self._draw_obstacles_3d(ax)
        for idx, segment in enumerate(self.path_segments):
            if segment.shape[0] < 2:
                continue
            ax.plot(
                segment[:, 0],
                segment[:, 1],
                segment[:, 2],
                "-",
                color="cyan",
                linewidth=1.6,
                alpha=0.68,
                label="traversed path" if idx == 0 and show_legend_context else None,
            )
        path_waypoints = self._unique_path_waypoints()
        if path_waypoints.size:
            ax.scatter(
                path_waypoints[:, 0],
                path_waypoints[:, 1],
                path_waypoints[:, 2],
                s=9,
                color="cyan",
                edgecolor="black",
                linewidth=0.25,
                alpha=0.28,
                marker=".",
                depthshade=False,
                label="path waypoint" if show_legend_context else None,
            )
        if self.measurement_points:
            points = np.vstack(self.measurement_points)
            ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                s=34,
                color="white",
                edgecolor="cyan",
                linewidth=0.8,
                depthshade=False,
                label="measurement station" if show_legend_context else None,
            )
        robot = np.asarray(frame.robot_position, dtype=float)
        ax.scatter(
            [robot[0]],
            [robot[1]],
            [robot[2]],
            s=70,
            color="cyan",
            edgecolor="black",
            linewidth=0.7,
            depthshade=False,
            label="robot" if show_legend_context else None,
        )
        self._plot_true_sources_3d(ax)

    def _plot_estimates_3d(self, ax: plt.Axes, frame: PFFrame) -> None:
        """Plot current source estimates on one PF 3D axis."""
        for iso in self.isotopes:
            est = frame.estimated_sources.get(iso, np.zeros((0, 3), dtype=float))
            strengths = frame.estimated_strengths.get(iso, np.zeros(0, dtype=float))
            if not np.asarray(est, dtype=float).size:
                continue
            est_arr = np.asarray(est, dtype=float).reshape((-1, 3))
            sizes = 110.0 + 0.018 * np.clip(
                np.asarray(strengths, dtype=float),
                0.0,
                5000.0,
            )
            if sizes.size != est_arr.shape[0]:
                sizes = np.full(est_arr.shape[0], 130.0, dtype=float)
            ax.scatter(
                est_arr[:, 0],
                est_arr[:, 1],
                est_arr[:, 2],
                marker="x",
                s=sizes,
                color=self.colors.get(iso, "gray"),
                linewidths=1.9,
                depthshade=False,
                label=f"{iso} estimate",
            )

    def _format_pf_axis(self, ax: plt.Axes, title: str) -> None:
        """Apply shared bounds, metric ticks, labels, and camera to a PF axis."""
        xmin, xmax, ymin, ymax, zmin, zmax = self.world_bounds
        ax.set_xlim(float(xmin), float(xmax))
        ax.set_ylim(float(ymin), float(ymax))
        ax.set_zlim(float(zmin), float(zmax))
        _apply_metric_ticks_3d(
            ax,
            xlim=(float(xmin), float(xmax)),
            ylim=(float(ymin), float(ymax)),
            zlim=(float(zmin), float(zmax)),
        )
        try:
            ax.set_box_aspect(
                (
                    max(float(xmax) - float(xmin), 1.0e-9),
                    max(float(ymax) - float(ymin), 1.0e-9),
                    max(float(zmax) - float(zmin), 1.0e-9),
                )
            )
        except AttributeError:
            pass
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title(title, fontsize=10)
        ax.view_init(elev=26.0, azim=-58.0)

    @staticmethod
    def _point_count(points_by_isotope: dict[str, NDArray[np.float64]]) -> int:
        """Return the total number of display points across isotope arrays."""
        total = 0
        for points in points_by_isotope.values():
            arr = np.asarray(points, dtype=float)
            if arr.size:
                total += int(arr.reshape((-1, 3)).shape[0])
        return total

    def _save_pf_3d(
        self,
        frame: PFFrame,
        output_path: Path,
        *,
        labeled_output_path: Path | None = None,
    ) -> None:
        """Save plain and optionally source-labeled PF 3-D PNGs."""
        fig = plt.figure(figsize=(14.2, 6.6))
        ax_samples = fig.add_subplot(121, projection="3d")
        ax_representatives = fig.add_subplot(122, projection="3d")
        representative_positions_by_iso = (
            frame.particle_representative_positions
            if frame.particle_representative_positions is not None
            else frame.particle_positions
        )
        representative_weights_by_iso = (
            frame.particle_representative_weights
            if frame.particle_representative_weights is not None
            else frame.particle_weights
        )
        self._draw_pf_scene_context(ax_samples, frame, show_legend_context=True)
        self._draw_pf_scene_context(
            ax_representatives,
            frame,
            show_legend_context=False,
        )
        for iso in self.isotopes:
            color = self.colors.get(iso, "gray")
            pts, weights = self._particle_subset(
                frame.particle_positions.get(iso, np.zeros((0, 3), dtype=float)),
                frame.particle_weights.get(iso, np.zeros(0, dtype=float)),
            )
            if pts.size:
                sizes, rgba = self._particle_style(weights, color)
                ax_samples.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    s=sizes,
                    c=rgba,
                    marker=".",
                    depthshade=False,
                    label=f"{iso} source samples",
                )
            rep_positions = representative_positions_by_iso.get(
                iso,
                np.zeros((0, 3), dtype=float),
            )
            rep_weights = representative_weights_by_iso.get(
                iso,
                np.zeros(0, dtype=float),
            )
            reps, rep_w = self._particle_subset(rep_positions, rep_weights)
            if reps.size:
                sizes, rgba = self._representative_particle_style(rep_w, color)
                ax_representatives.scatter(
                    reps[:, 0],
                    reps[:, 1],
                    reps[:, 2],
                    s=sizes,
                    c=rgba,
                    marker="o",
                    edgecolors="none",
                    alpha=None,
                    depthshade=False,
                    label=f"{iso} PF particles",
                )
        self._plot_estimates_3d(ax_samples, frame)
        self._plot_estimates_3d(ax_representatives, frame)
        sample_count = self._point_count(frame.particle_positions)
        representative_count = self._point_count(representative_positions_by_iso)
        self._format_pf_axis(
            ax_samples,
            f"Source-slot samples (N={sample_count})",
        )
        self._format_pf_axis(
            ax_representatives,
            f"PF particle active-source medoids (N={representative_count})",
        )
        fig.suptitle(
            f"Particle filter 3D - {self._frame_progress_label(frame)}",
            fontsize=11,
        )
        handles, _labels = ax_representatives.get_legend_handles_labels()
        if handles:
            ax_representatives.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                fontsize=6.5,
            )
        fig.subplots_adjust(left=0.02, right=0.86, top=0.88, bottom=0.04, wspace=0.06)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        if labeled_output_path is not None:
            self._annotate_source_labels_3d(
                ax_samples,
                frame,
                include_truth=True,
                include_estimates=False,
            )
            self._annotate_source_labels_3d(
                ax_representatives,
                frame,
                include_truth=False,
                include_estimates=True,
            )
            truth_title = "True-source labels (T)"
            if not self.true_sources:
                truth_title = "Truth hidden"
            ax_samples.set_title(
                f"Source-slot samples (N={sample_count})\n{truth_title}",
                fontsize=9.5,
            )
            estimate_title = "Estimate labels (E; isotope-local indices)"
            if self.true_sources:
                estimate_title = (
                    "Estimate labels (E; nearest same-isotope truth within "
                    f"{self.source_label_neighborhood_m:.1f} m"
                    ")"
                )
            ax_representatives.set_title(
                "PF particle active-source medoids "
                f"(N={representative_count})\n{estimate_title}",
                fontsize=9.5,
            )
            labeled_output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(labeled_output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def _save_spectrum(self, frame: PFFrame, output_path: Path) -> None:
        """Save the current raw native full-spectrum observation."""
        energy = getattr(frame, "spectrum_energy_keV", None)
        counts = getattr(frame, "spectrum_counts", None)
        if energy is None or counts is None:
            return
        energy_arr = np.asarray(energy, dtype=float)
        counts_arr = np.asarray(counts, dtype=float)
        if energy_arr.size == 0 or counts_arr.size == 0:
            return
        size = min(energy_arr.size, counts_arr.size)
        energy_arr = energy_arr[:size]
        counts_arr = np.clip(counts_arr[:size], a_min=0.0, a_max=None)
        fig, ax = plt.subplots(figsize=(10.0, 4.8))
        ax.plot(
            energy_arr,
            counts_arr,
            color="black",
            linewidth=1.0,
            label="raw native detector spectrum",
        )
        ax.set_xlabel("Energy [keV]")
        ax.set_ylabel("Counts / bin")
        ax.set_title(f"Full spectrum - {self._frame_progress_label(frame)}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)


def _async_cui_split_worker(
    config: dict[str, Any],
    frame_queue: Any,
    status_queue: Any,
    run_token: str,
) -> None:
    """Render CUI split-view frames in a dedicated worker process."""
    operation_id = -1
    try:
        truth_overlay_socket_path = config.pop(
            "truth_overlay_socket_path",
            None,
        )
        visualizer = CUISplitPFVisualizer(**config)
        if truth_overlay_socket_path is not None:
            from runtime.cui_truth_overlay import load_cui_truth_overlay

            overlay = load_cui_truth_overlay(truth_overlay_socket_path)
            visualizer.set_truth(
                dict(overlay.true_sources),
                dict(overlay.true_strengths),
            )
        status_queue.put(("ready", operation_id, "startup", run_token))
        last_frame: PFFrame | None = None
        while True:
            message, operation_id, payload = frame_queue.get()
            if message == "close":
                status_queue.put(
                    ("closed", operation_id, "close", run_token)
                )
                return
            if message == "truth":
                true_sources, true_strengths = pickle.loads(payload)
                visualizer.set_truth(true_sources, true_strengths)
                if last_frame is not None:
                    visualizer.update(last_frame)
                status_queue.put(
                    ("ack", operation_id, "truth", run_token)
                )
                continue
            if message != "frame":
                raise RuntimeError(
                    f"Unknown asynchronous CUI message {message!r}."
                )
            frame = pickle.loads(payload)
            last_frame = frame
            visualizer.update(frame)
            status_queue.put(("ack", operation_id, "frame", run_token))
    except BaseException as exc:  # pragma: no cover - child-process boundary.
        status_queue.put(
            (
                "error",
                operation_id,
                f"{type(exc).__name__}: {exc}",
                run_token,
            )
        )
        raise


class AsyncCUISplitPFVisualizer:
    """Non-blocking process-backed wrapper for CUI split visualization."""

    def __init__(
        self,
        isotopes: list[str],
        output_dir: str | Path,
        *,
        world_bounds: tuple[float, float, float, float, float, float] | None = None,
        true_sources: dict[str, NDArray[np.float64]] | None = None,
        true_strengths: dict[str, float | Sequence[float]] | None = None,
        obstacle_grid: ObstacleGrid | None = None,
        max_particles_per_isotope: int | None = None,
        source_label_neighborhood_m: float = 1.0,
        truth_overlay_socket_path: str | Path | None = None,
        queue_size: int = 2,
        save_step_history: bool = False,
    ) -> None:
        """Start a renderer process that consumes latest PF frames asynchronously."""
        if not isinstance(save_step_history, bool):
            raise TypeError("save_step_history must be a boolean.")
        if (
            isinstance(queue_size, bool)
            or not isinstance(queue_size, int)
            or queue_size < 1
        ):
            raise ValueError("queue_size must be a positive integer.")
        self.isotopes = list(isotopes)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.output_dir / "index.html"
        self.latest_robot_path = self.output_dir / "latest_robot_2d.png"
        self.latest_overview_path = self.output_dir / "latest_experiment_overview.png"
        self.latest_pf_path = self.output_dir / "latest_pf_3d.png"
        self.latest_pf_labeled_path = self.output_dir / "latest_pf_3d_labeled.png"
        self.latest_spectrum_path = self.output_dir / "latest_spectrum.png"
        self._closed = False
        self._run_token = uuid4().hex
        self._next_operation_id = 0
        self._last_enqueued_operation_id = -1
        self._last_acknowledged_operation_id = -1
        self._operation_kinds: dict[int, str] = {}
        self._ready_acknowledged = False
        self._close_operation_id: int | None = None
        self._close_acknowledged = False
        self._worker_error: RuntimeError | None = None
        self._ctx = mp.get_context("spawn")
        self._queue = self._ctx.Queue(maxsize=queue_size)
        self._status_queue = self._ctx.Queue()
        config = {
            "isotopes": self.isotopes,
            "output_dir": self.output_dir,
            "world_bounds": world_bounds,
            "true_sources": true_sources,
            "true_strengths": true_strengths,
            "obstacle_grid": obstacle_grid,
            "max_particles_per_isotope": max_particles_per_isotope,
            "source_label_neighborhood_m": source_label_neighborhood_m,
            "truth_overlay_socket_path": (
                None
                if truth_overlay_socket_path is None
                else Path(truth_overlay_socket_path).expanduser().resolve()
            ),
            "save_step_history": save_step_history,
        }
        self._process = self._ctx.Process(
            target=_async_cui_split_worker,
            args=(config, self._queue, self._status_queue, self._run_token),
            daemon=True,
        )
        try:
            self._process.start()
        except BaseException as exc:
            self._closed = True
            try:
                if self._process.pid is not None:
                    self._terminate_worker()
            except BaseException as cleanup_exc:
                exc.add_note(
                    "Secondary async CUI startup cleanup failure: "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
            raise
        self._await_worker_ready(timeout_s=10.0)

    @property
    def run_token(self) -> str:
        """Return the opaque identity of this renderer generation."""
        return self._run_token

    @property
    def last_acknowledged_operation_id(self) -> int:
        """Return the last operation proven rendered by the worker."""
        return int(self._last_acknowledged_operation_id)

    def _accept_status(self, status: object) -> str:
        """Validate one worker status and update acknowledged state."""
        if (
            not isinstance(status, tuple)
            or len(status) != 4
            or status[3] != self._run_token
        ):
            raise RuntimeError("Async CUI worker returned an invalid status.")
        kind, raw_operation_id, detail, _token = status
        if isinstance(raw_operation_id, bool) or not isinstance(
            raw_operation_id,
            int,
        ):
            raise RuntimeError("Async CUI worker returned an invalid operation ID.")
        operation_id = int(raw_operation_id)
        if not isinstance(kind, str) or not isinstance(detail, str):
            raise RuntimeError("Async CUI worker returned an invalid status payload.")
        if kind == "error":
            known_operation = (
                (operation_id == -1 and not self._ready_acknowledged)
                or operation_id in self._operation_kinds
                or operation_id == self._close_operation_id
            )
            if not known_operation or not detail:
                raise RuntimeError(
                    "Async CUI worker reported an invalid failure status."
                )
            self._worker_error = RuntimeError(
                "Async CUI worker failed at operation "
                f"{operation_id}: {detail}"
            )
            raise self._worker_error
        if kind == "ready":
            if (
                self._ready_acknowledged
                or operation_id != -1
                or detail != "startup"
            ):
                raise RuntimeError(
                    "Async CUI worker returned a mismatched startup acknowledgement."
                )
            self._ready_acknowledged = True
            return kind
        if kind == "ack":
            if not self._ready_acknowledged:
                raise RuntimeError(
                    "Async CUI worker acknowledged work before startup."
                )
            expected_kind = self._operation_kinds.get(operation_id)
            if expected_kind is None or detail != expected_kind:
                raise RuntimeError(
                    "Async CUI worker returned a mismatched operation acknowledgement."
                )
            if operation_id <= self._last_acknowledged_operation_id:
                raise RuntimeError("Async CUI acknowledgements are not monotonic.")
            self._last_acknowledged_operation_id = operation_id
            self._operation_kinds.pop(operation_id)
            return kind
        if kind == "closed":
            if (
                not self._ready_acknowledged
                or
                self._close_acknowledged
                or self._close_operation_id is None
                or operation_id != self._close_operation_id
                or detail != "close"
            ):
                raise RuntimeError(
                    "Async CUI worker returned a mismatched close acknowledgement."
                )
            self._close_acknowledged = True
            return kind
        if kind not in {"ready", "ack", "closed"}:
            raise RuntimeError(f"Unknown async CUI worker status {kind!r}.")
        raise AssertionError("Async CUI status validation fell through.")

    def _drain_worker_status(self) -> set[str]:
        """Consume every immediately available worker status."""
        kinds: set[str] = set()
        while True:
            try:
                status = self._status_queue.get_nowait()
            except queue.Empty:
                return kinds
            kinds.add(self._accept_status(status))

    def _await_worker_ready(self, *, timeout_s: float) -> None:
        """Require an explicit startup acknowledgement from the child."""
        try:
            try:
                status = self._status_queue.get(timeout=float(timeout_s))
            except queue.Empty as exc:
                raise RuntimeError(
                    "Async CUI worker did not acknowledge startup."
                ) from exc
            if self._accept_status(status) != "ready":
                raise RuntimeError("Async CUI worker did not start cleanly.")
        except BaseException as exc:
            self._terminate_after_failure(exc)
            raise

    def _terminate_worker(self) -> None:
        """Stop and reap the renderer worker after a protocol failure."""
        if self._process.is_alive():
            self._process.terminate()
        self._process.join(timeout=2.0)

    def _terminate_after_failure(self, exc: BaseException) -> None:
        """Reap the worker without allowing cleanup to replace its failure."""
        self._closed = True
        try:
            self._terminate_worker()
        except BaseException as cleanup_exc:
            exc.add_note(
                "Secondary async CUI cleanup failure: "
                f"{type(cleanup_exc).__name__}: {cleanup_exc}"
            )

    def _collect_close_status(self, *, timeout_s: float) -> set[str]:
        """Collect terminal statuses through the explicit close acknowledgement."""
        kinds: set[str] = set()
        deadline = time.monotonic() + min(max(0.1, float(timeout_s)), 1.0)
        while not self._close_acknowledged:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            try:
                status = self._status_queue.get(timeout=remaining)
            except queue.Empty:
                break
            kinds.add(self._accept_status(status))
        kinds.update(self._drain_worker_status())
        return kinds

    def _require_live_worker(self) -> None:
        """Raise when the renderer has failed, closed, or exited."""
        try:
            self._drain_worker_status()
            if self._worker_error is not None:
                raise self._worker_error
            if self._closed:
                raise RuntimeError("Async CUI visualizer is already closed.")
            if not self._process.is_alive():
                raise RuntimeError(
                    "Async CUI worker exited before acknowledging queued work."
                )
        except BaseException as exc:
            self._terminate_after_failure(exc)
            raise

    def _enqueue_latest(self, kind: str, payload: bytes) -> None:
        """Queue one operation, replacing only an older unprocessed operation."""
        try:
            self._require_live_worker()
            operation_id = int(self._next_operation_id)
            message = (kind, operation_id, payload)
            while True:
                try:
                    self._queue.put_nowait(message)
                    self._operation_kinds[operation_id] = kind
                    self._next_operation_id += 1
                    self._last_enqueued_operation_id = operation_id
                    return
                except queue.Full:
                    try:
                        dropped = self._queue.get_nowait()
                    except queue.Empty as exc:
                        raise RuntimeError(
                            "Async CUI queue reported full but yielded no operation."
                        ) from exc
                    if (
                        not isinstance(dropped, tuple)
                        or len(dropped) != 3
                        or isinstance(dropped[1], bool)
                        or not isinstance(dropped[1], int)
                    ):
                        raise RuntimeError(
                            "Async CUI queue contained an invalid operation."
                        )
                    self._operation_kinds.pop(int(dropped[1]), None)
        except BaseException as exc:
            if not self._closed:
                self._terminate_after_failure(exc)
            raise

    def update(self, frame: PFFrame) -> None:
        """Queue the latest frame and fail if the renderer is unavailable."""
        payload = pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL)
        self._enqueue_latest("frame", payload)

    def set_truth(
        self,
        true_sources: dict[str, NDArray[np.float64]],
        true_strengths: dict[str, float | Sequence[float]],
    ) -> None:
        """Queue evaluation truth for post-run rendering only."""
        payload = pickle.dumps(
            (true_sources, true_strengths),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        self._enqueue_latest("truth", payload)

    def close(self, timeout_s: float = 10.0) -> None:
        """Ask the renderer process to finish queued work and stop."""
        if self._closed:
            return
        try:
            self._require_live_worker()
            close_operation_id = int(self._next_operation_id)
            self._close_operation_id = close_operation_id
            try:
                self._queue.put(
                    ("close", close_operation_id, None),
                    timeout=max(0.1, float(timeout_s)),
                )
            except queue.Full as exc:
                raise RuntimeError(
                    "Async CUI worker did not accept close."
                ) from exc
            self._closed = True
            self._process.join(timeout=max(0.1, float(timeout_s)))
            if self._process.is_alive():
                raise RuntimeError(
                    "Async CUI worker did not close before its deadline."
                )
            status_kinds = self._collect_close_status(timeout_s=timeout_s)
            if self._process.exitcode != 0:
                raise RuntimeError(
                    f"Async CUI worker exited with code {self._process.exitcode}."
                )
            if "closed" not in status_kinds or not self._close_acknowledged:
                raise RuntimeError(
                    "Async CUI worker omitted its close acknowledgement."
                )
            if (
                self._last_enqueued_operation_id >= 0
                and self._last_acknowledged_operation_id
                != self._last_enqueued_operation_id
            ):
                raise RuntimeError(
                    "Async CUI worker did not acknowledge the latest queued operation."
                )
        except BaseException as exc:
            self._terminate_after_failure(exc)
            raise


def build_frame_from_pf(
    pf,
    step_index: int,
    time_sec: float,
    *,
    detector_position: NDArray[np.float64],
    spectrum_energy_keV: NDArray[np.float64] | None = None,
    spectrum_counts: NDArray[np.int64] | None = None,
) -> PFFrame:
    """
    Construct a PFFrame snapshot from the joint PF and raw spectrum.

    Args:
        pf: Sequential estimator exposing ``filters`` and cached estimates.
        step_index: integer step
        time_sec: cumulative time in seconds
        detector_position: Physical detector XYZ used by the likelihood.
        spectrum_energy_keV: Native incident-energy bin axis.
        spectrum_counts: Raw nonnegative native histogram.
    """
    visualization_estimates = getattr(pf, "visualization_estimates", None)
    if not callable(visualization_estimates):
        raise TypeError("PF must implement visualization_estimates().")
    est: dict[str, object] = visualization_estimates()
    if not isinstance(est, dict):
        raise TypeError("PF visualization_estimates() must return a dictionary.")
    particle_positions: dict[str, NDArray[np.float64]] = {}
    particle_weights: dict[str, NDArray[np.float64]] = {}
    particle_representative_positions: dict[str, NDArray[np.float64]] = {}
    particle_representative_weights: dict[str, NDArray[np.float64]] = {}
    estimated_sources: dict[str, NDArray[np.float64]] = {}
    estimated_strengths: dict[str, NDArray[np.float64]] = {}

    for iso, filt in pf.filters.items():
        cont_particles = getattr(filt, "continuous_particles", [])
        cont_weights = getattr(filt, "continuous_weights", np.zeros(0))
        if cont_particles and len(cont_weights) == len(cont_particles):
            (
                particle_positions[iso],
                particle_weights[iso],
                particle_representative_positions[iso],
                particle_representative_weights[iso],
            ) = _batched_particle_display_arrays(
                filt,
                np.asarray(cont_weights, dtype=float),
            )
        else:
            particle_positions[iso] = np.zeros((0, 3), dtype=float)
            particle_weights[iso] = np.zeros(0, dtype=float)
            particle_representative_positions[iso] = np.zeros(
                (0, 3),
                dtype=float,
            )
            particle_representative_weights[iso] = np.zeros(
                0,
                dtype=float,
            )
        if iso in est:
            value = est[iso]
            if isinstance(value, tuple) and len(value) == 2:
                est_pos = np.asarray(value[0], dtype=float)
                est_str = np.asarray(value[1], dtype=float)
            else:
                raise TypeError(
                    "PF visualization_estimates() must return "
                    "(positions, strengths) "
                    f"for isotope {iso}."
                )
        else:
            est_pos = np.zeros((0, 3), dtype=float)
            est_str = np.zeros(0, dtype=float)
        est_pos = np.asarray(est_pos, dtype=float).reshape(-1, 3)
        est_str = np.asarray(est_str, dtype=float).reshape(-1)
        if est_pos.shape[0] != est_str.shape[0]:
            raise ValueError(
                "PF visualization_estimates() must return one strength per estimated "
                f"source for isotope {iso}."
            )
        if np.any(~np.isfinite(est_pos)) or np.any(~np.isfinite(est_str)):
            raise ValueError(
                "PF visualization_estimates() returned non-finite values "
                f"for {iso}."
            )
        estimated_sources[iso] = est_pos
        estimated_strengths[iso] = est_str

    robot_pos = np.asarray(detector_position, dtype=float)
    if robot_pos.shape != (3,) or np.any(~np.isfinite(robot_pos)):
        raise ValueError("detector_position must be a finite XYZ vector.")
    return PFFrame(
        step_index=step_index,
        time=time_sec,
        robot_position=robot_pos,
        particle_positions=particle_positions,
        particle_weights=particle_weights,
        estimated_sources=estimated_sources,
        estimated_strengths=estimated_strengths,
        spectrum_energy_keV=(
            None
            if spectrum_energy_keV is None
            else np.asarray(spectrum_energy_keV, dtype=float)
        ),
        spectrum_counts=(
            None
            if spectrum_counts is None
            else np.asarray(spectrum_counts, dtype=np.int64)
        ),
        particle_representative_positions=particle_representative_positions,
        particle_representative_weights=particle_representative_weights,
    )
