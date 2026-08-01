"""Data structures for capturing PF state per time step for visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any, Sequence

from pathlib import Path
import multiprocessing as mp
import pickle
import queue
import shutil
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from measurement.obstacles import ObstacleGrid
@dataclass
class PFFrame:
    """
    Snapshot of the PF state and measurement at one time step.

    - step_index: integer step
    - time: cumulative measurement time (s)
    - robot_position: detector position q_k (3,)
    - robot_orientation: optional robot orientation (e.g., quaternion or R)
    - RFe, RPb: rotation matrices for iron/lead shields (3x3)
    - duration: acquisition time T_k
    - particle_positions: isotope -> source-slot sample positions (N_points, 3)
    - particle_weights: isotope -> source-slot sample weights (N_points,)
    - estimated_sources: isotope -> (N_est, 3)
    - estimated_strengths: isotope -> (N_est,)
    - path_waypoints_xyz: optional obstacle-aware robot path segment (M, 3)
    - spectrum_energy_keV/spectrum_counts: optional raw native spectrum data
    """

    step_index: int
    time: float
    robot_position: NDArray[np.float64]
    robot_orientation: Optional[NDArray[np.float64]]
    RFe: NDArray[np.float64]
    RPb: NDArray[np.float64]
    duration: float
    particle_positions: Dict[str, NDArray[np.float64]]
    particle_weights: Dict[str, NDArray[np.float64]]
    estimated_sources: Dict[str, NDArray[np.float64]]
    estimated_strengths: Dict[str, NDArray[np.float64]]
    path_waypoints_xyz: Optional[NDArray[np.float64]] = None
    spectrum_energy_keV: Optional[NDArray[np.float64]] = None
    spectrum_counts: Optional[NDArray[np.float64]] = None
    particle_representative_positions: Optional[Dict[str, NDArray[np.float64]]] = None
    particle_representative_weights: Optional[Dict[str, NDArray[np.float64]]] = None


def frame_to_isaac_pf_payload(
    frame: PFFrame,
    *,
    max_particles_per_isotope: int | None = None,
) -> Dict[str, Any]:
    """Return a JSON-serializable PF marker payload for Isaac Sim."""
    max_particles = (
        None
        if max_particles_per_isotope is None
        else max(0, int(max_particles_per_isotope))
    )
    particle_positions: Dict[str, list[list[float]]] = {}
    particle_weights: Dict[str, list[float]] = {}
    for isotope, positions_raw in frame.particle_positions.items():
        positions = np.asarray(positions_raw, dtype=float).reshape((-1, 3))
        weights = np.asarray(
            frame.particle_weights.get(isotope, np.ones(positions.shape[0])),
            dtype=float,
        ).reshape(-1)
        if weights.size != positions.shape[0]:
            weights = np.ones(positions.shape[0], dtype=float)
        if max_particles is not None and max_particles > 0 and positions.shape[0] > max_particles:
            order = np.argsort(weights)[::-1][:max_particles]
            positions = positions[order]
            weights = weights[order]
        particle_positions[isotope] = _array2_to_list(positions)
        particle_weights[isotope] = [float(value) for value in weights]
    estimated_sources = {
        isotope: _array2_to_list(np.asarray(positions, dtype=float).reshape((-1, 3)))
        for isotope, positions in frame.estimated_sources.items()
    }
    estimated_strengths = {
        isotope: [float(value) for value in np.asarray(strengths, dtype=float).reshape(-1)]
        for isotope, strengths in frame.estimated_strengths.items()
    }
    payload: Dict[str, Any] = {
        "step_index": int(frame.step_index),
        "time_s": float(frame.time),
        "robot_position": [float(value) for value in np.asarray(frame.robot_position, dtype=float).reshape(-1)[:3]],
        "particle_positions": particle_positions,
        "particle_weights": particle_weights,
        "estimated_sources": estimated_sources,
        "estimated_strengths": estimated_strengths,
    }
    if frame.particle_representative_positions is not None:
        payload["particle_representative_positions"] = {
            isotope: _array2_to_list(np.asarray(positions, dtype=float).reshape((-1, 3)))
            for isotope, positions in frame.particle_representative_positions.items()
        }
    if frame.particle_representative_weights is not None:
        payload["particle_representative_weights"] = {
            isotope: [
                float(value)
                for value in np.asarray(weights, dtype=float).reshape(-1)
            ]
            for isotope, weights in frame.particle_representative_weights.items()
        }
    waypoints = _coerce_path_waypoints(frame)
    if waypoints.size:
        payload["path_waypoints_xyz"] = _array2_to_list(waypoints)
    return payload


def _array2_to_list(values: NDArray[np.float64]) -> list[list[float]]:
    """Convert a two-dimensional numeric array to a JSON list."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return []
    arr = arr.reshape((-1, arr.shape[-1]))
    return [[float(component) for component in row] for row in arr]


@dataclass(frozen=True)
class LayoutGeometry:
    """Figure size and axes positions for the PF visualization layout."""

    fig_size: Tuple[float, float]
    pf_pos: Tuple[float, float, float, float]
    counts_pos: Tuple[float, float, float, float] | None
    labels_pos: Tuple[float, float, float, float]


DEFAULT_ISOTOPE_COLORS = {
    "Cs-137": "tab:red",
    "Co-60": "tab:blue",
    "Eu-154": "tab:green",
    "Eu-155": "tab:green",
}


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
        raise ValueError(
            "PF visualization requires strictly positive posterior mass."
        )
    return w / total


def _coerce_path_waypoints(frame: PFFrame) -> NDArray[np.float64]:
    """Return a valid path waypoint array from a PFFrame."""
    waypoints = getattr(frame, "path_waypoints_xyz", None)
    if waypoints is None:
        return np.zeros((0, 3), dtype=float)
    arr = np.asarray(waypoints, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return np.zeros((0, 3), dtype=float)
    return arr


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


def _extend_trajectory_history(
    history: list[NDArray[np.float64]],
    frame: PFFrame,
) -> None:
    """Append obstacle-aware waypoints or the current robot pose to history."""
    waypoints = _coerce_path_waypoints(frame)
    if waypoints.size == 0:
        waypoints = np.asarray(frame.robot_position, dtype=float).reshape(1, 3)
    for waypoint in waypoints:
        point = np.asarray(waypoint, dtype=float).reshape(3)
        if history and float(np.linalg.norm(point - history[-1])) <= 1e-9:
            continue
        history.append(point.copy())


def _format_pos(pos: NDArray[np.float64]) -> str:
    """Format a position vector with two decimal places."""
    coords = ", ".join(f"{val:.2f}" for val in np.asarray(pos, dtype=float).ravel())
    return f"[{coords}]"


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
    if (
        positions.shape != (num_sources, 3)
        or np.any(~np.isfinite(positions))
    ):
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
            flat_weights.extend(
                [float(weight)] * int(active_positions.shape[0])
            )
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

    packed_positions, _strengths, active_mask, _chart_ids, _surface_uv = (
        packer()
    )
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


class RealTimePFVisualizer:
    """
    Simple matplotlib-based 3D visualizer for the PF state.

    - update(frame) redraws particles, estimates, and the joint-PF label panel.
    - save_final(path) saves the current figure.
    - save_estimates_only(path) saves a view with only estimate markers visible.
    """

    _BASE_FIGSIZE = (15.0, 6.0)
    _BASE_LAYOUT_FRACS = {
        "left": 0.02,
        "right": 0.02,
        "gap": 0.005,
        "pf": 0.54,
    }
    _VERTICAL_LAYOUT = {
        "bottom": 0.17,
        "top": 0.94,
        "labels_frac": 0.57,
        "counts_frac": 0.43,
    }
    _MIN_SIDE_FRAC = 0.26
    _PF_PANEL_SCALE = 1.0
    _PF_PLOT_ZOOM = 1.35
    _X_TICK_OFFSET_PX = 2.0
    _X_LABEL_OFFSET_PX = 8.0
    _X_TICK_MIN_AX_Y = 0.0
    _X_LABEL_MIN_AX_Y = 0.0
    _ESTIMATE_TEXT_OFFSET = 0.2
    _ESTIMATE_TEXT_PAD_PX = 6.0
    _LABEL_LINE_SPACING = 1.3

    def __init__(
        self,
        isotopes: List[str],
        world_bounds: Optional[Tuple[float, float, float, float, float, float]] = None,
        true_sources: Optional[Dict[str, NDArray[np.float64]]] = None,
        true_strengths: Optional[Dict[str, float | Sequence[float]]] = None,
        obstacle_grid: ObstacleGrid | None = None,
    ) -> None:
        """Initialize the visualizer and optional obstacle overlay."""
        self.isotopes = isotopes
        self.world_bounds = world_bounds or (0, 10, 0, 10, 0, 3)
        self.true_sources = true_sources or {}
        self.true_strengths = true_strengths or {}
        self.obstacle_grid = obstacle_grid
        self._fig_width = self._BASE_FIGSIZE[0]
        self._fig_height = self._BASE_FIGSIZE[1]
        layout = self._layout_geometry()
        self.fig = plt.figure(figsize=layout.fig_size)
        self.ax3d = self.fig.add_axes(layout.pf_pos, projection="3d")
        self.ax_labels = self.fig.add_axes(layout.labels_pos)
        cmap = plt.get_cmap("tab10")
        self.colors = {}
        for i, iso in enumerate(isotopes):
            if iso in DEFAULT_ISOTOPE_COLORS:
                self.colors[iso] = DEFAULT_ISOTOPE_COLORS[iso]
            else:
                self.colors[iso] = cmap(i % 10)
        self._label_title_fontsize = 16
        self._label_section_fontsize = 14
        self._label_text_fontsize = 13
        self._label_text_x = 0.16
        self._label_marker_line = (0.02, 0.1)
        self._label_marker_point = 0.06
        self._x_label_artist = None
        self._x_tick_artists: list = []
        self._x_label_cid = None
        self._init_axes()
        self._init_label_axis()
        self._apply_layout()
        self._attach_draw_handler()
        self._particle_artists: Dict[str, Any] = {}
        self._est_artists: Dict[str, Any] = {}
        self._estimate_text_artists: Dict[str, list] = {}
        self._estimate_text_positions: Dict[str, NDArray[np.float64]] = {}
        self._true_text_artists: Dict[str, list] = {}
        self._true_text_positions: Dict[str, NDArray[np.float64]] = {}
        self._true_halo_artists: list = []
        self._robot_artist = None
        self._traj_line = None
        self._shield_arrows: Dict[str, Any] = {}
        self._traj_history: list[NDArray[np.float64]] = []
        self._last_frame: PFFrame | None = None
        self._true_artists: list = []
        self._projection_artists: list = []
        self._true_projection_artists: list = []
        self._obstacle_artist = None
        self._particle_size_range = (0.8, 10.0)
        self._particle_alpha_range = (0.05, 0.95)
        self._particle_weight_exponent = 0.7
        self._projection_linewidth = 1.8
        self.estimate_colors = {}
        self._active_isotopes: set[str] | None = None
        # Plot true sources once if provided (as legend entries)
        for iso, pos in self.true_sources.items():
            if pos.size:
                strength = self.true_strengths.get(iso, None)
                label = f"True {iso}"
                if strength is not None and not isinstance(strength, (list, tuple, np.ndarray)):
                    label = f"{label} pos={_format_pos(pos)} q={strength:.1f} cps@1m"
                halo = self.ax3d.scatter(
                    pos[:, 0],
                    pos[:, 1],
                    pos[:, 2],
                    marker="*",
                    s=140,
                    color="white",
                    edgecolors="white",
                    linewidths=1.5,
                    alpha=0.85,
                    label="_nolegend_",
                    depthshade=False,
                    zorder=26,
                )
                self._true_halo_artists.append(halo)
                art = self.ax3d.scatter(
                    pos[:, 0],
                    pos[:, 1],
                    pos[:, 2],
                    marker="*",
                    s=100,
                    color=self.colors.get(iso, "black"),
                    label=label,
                    depthshade=False,
                    zorder=27,
                )
                self._true_artists.append(art)
                self._true_projection_artists.extend(self._axis_projection_lines(pos, self.colors.get(iso, "black")))
                self._update_true_texts(iso, pos, self.colors.get(iso, "black"))
        for iso in self.isotopes:
            self.estimate_colors[iso] = self._estimate_color(self.colors.get(iso, "black"))

    def set_active_isotopes(self, isotopes: Sequence[str] | None) -> None:
        """Restrict legend/label reporting to the given isotopes."""
        if isotopes is None:
            self._active_isotopes = None
            return
        self._active_isotopes = set(isotopes)

    def _iter_active_isotopes(self) -> List[str]:
        """Return the list of isotopes to display in legends/labels."""
        if self._active_isotopes is None:
            return list(self.isotopes)
        return [iso for iso in self.isotopes if iso in self._active_isotopes]

    def _layout_geometry(self) -> LayoutGeometry:
        """Return figure size and axes positions with fixed margins."""
        fig_width = self._fig_width
        fig_height = self._fig_height
        left = self._BASE_LAYOUT_FRACS["left"]
        right = self._BASE_LAYOUT_FRACS["right"]
        gap = self._BASE_LAYOUT_FRACS["gap"]
        base_pf = self._BASE_LAYOUT_FRACS["pf"]
        available = 1.0 - left - right - gap
        min_side = min(self._MIN_SIDE_FRAC, available)
        pf_width = base_pf * self._PF_PANEL_SCALE
        if pf_width > available - min_side:
            pf_width = max(available - min_side, 0.0)
        side_width = max(available - pf_width, 0.0)
        side_left = left + pf_width + gap
        bottom = self._VERTICAL_LAYOUT["bottom"]
        top = self._VERTICAL_LAYOUT["top"]
        pf_height = max(top - bottom, 0.0)
        pf_pos = (left, bottom, pf_width, pf_height)
        counts_pos = None
        labels_pos = (side_left, bottom, side_width, pf_height)
        return LayoutGeometry(
            fig_size=(fig_width, fig_height),
            pf_pos=pf_pos,
            counts_pos=counts_pos,
            labels_pos=labels_pos,
        )

    def _axis_line_style(self) -> Tuple[str, float]:
        """Return line color and width that match the axis lines."""
        color = "black"
        linewidth = 1.2
        axis_line = None
        if self.ax3d is not None:
            axis_line = getattr(self.ax3d.xaxis, "line", None)
        if axis_line is not None:
            color = axis_line.get_color()
            try:
                axis_width = float(axis_line.get_linewidth())
            except (TypeError, ValueError):
                axis_width = linewidth
            if axis_width > 0:
                linewidth = axis_width
        return color, linewidth

    def _tune_axis_style(self) -> None:
        """Apply axis pane and tick styling for consistent visibility."""
        if self.ax3d is None:
            return
        for axis in (self.ax3d.xaxis, self.ax3d.yaxis, self.ax3d.zaxis):
            pane = axis.pane
            pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
            pane.set_alpha(0.0)
        self.ax3d.computed_zorder = False
        y_tick_pad = 3.5
        if self.ax3d.yaxis.majorTicks:
            y_tick_pad = float(self.ax3d.yaxis.majorTicks[0].get_pad())
        self.ax3d.tick_params(axis="x", pad=y_tick_pad)
        self.ax3d.grid(True, alpha=0.35)

    def _ensure_x_label(self) -> None:
        """Ensure the default x-axis label is visible."""
        if self.ax3d is None:
            return
        self.ax3d.set_xlabel("x [m]")
        if self._x_label_artist is not None:
            self._x_label_artist.set_visible(False)
        for art in self._x_tick_artists:
            art.set_visible(False)

    def _project_to_axes(self, pos: NDArray[np.float64]) -> tuple[float, float] | None:
        """Project a 3D point into axes coordinates for 2D annotations."""
        if self.ax3d is None:
            return None
        x2, y2, _ = proj3d.proj_transform(
            float(pos[0]),
            float(pos[1]),
            float(pos[2] + self._ESTIMATE_TEXT_OFFSET),
            self.ax3d.get_proj(),
        )
        x_disp, y_disp = self.ax3d.transData.transform((x2, y2))
        x_ax, y_ax = self.ax3d.transAxes.inverted().transform((x_disp, y_disp))
        return float(x_ax), float(y_ax)

    def _update_x_label_position(self, event: Any | None = None) -> None:
        """No-op: use the default matplotlib x-axis label and ticks."""
        return

    def _attach_draw_handler(self) -> None:
        """Attach a draw callback to keep the x label aligned with ticks."""
        if self._x_label_cid is None:
            self._x_label_cid = self.fig.canvas.mpl_connect("draw_event", self._on_draw)

    def _on_draw(self, event: Any) -> None:
        """Update custom label placement after draw events."""
        self._ensure_x_label()
        self._update_x_label_position(event)
        self._position_all_estimate_texts()

    def _set_box_aspect(self, aspect: Tuple[float, float, float]) -> None:
        """Set the 3D box aspect ratio with the configured zoom."""
        if self.ax3d is None:
            return
        try:
            self.ax3d.set_box_aspect(aspect, zoom=self._PF_PLOT_ZOOM)
        except TypeError:
            self.ax3d.set_box_aspect(aspect)

    def _draw_room_bounds(self) -> None:
        """Draw the environment bounds as solid edges."""
        if self.ax3d is None:
            return
        xmin, xmax, ymin, ymax, zmin, zmax = self.world_bounds
        color, linewidth = self._axis_line_style()
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
            line = self.ax3d.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color=color,
                linewidth=linewidth,
            )[0]
            line.set_clip_on(False)
            line.set_alpha(1.0)
            line.set_zorder(10)

    def _init_axes(self) -> None:
        """Initialize 3-D axis limits, ticks, labels, and aspect ratio."""
        xmin, xmax, ymin, ymax, zmin, zmax = self.world_bounds
        self.ax3d.set_xlim(xmin, xmax)
        self.ax3d.set_ylim(ymin, ymax)
        self.ax3d.set_zlim(zmin, zmax)
        self.ax3d.set_yticks(np.arange(ymin, ymax + 1e-6, 2.0))
        self._set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))
        self.ax3d.set_ylabel("y [m]")
        self.ax3d.set_zlabel("z [m]")
        self.ax3d.set_yticks(np.arange(ymin, ymax + 1e-6, 2.0))
        self._tune_axis_style()
        self._ensure_x_label()
        self._draw_room_bounds()
        if self.obstacle_grid is not None:
            self._draw_obstacle_grid()

    def _draw_obstacle_grid(self) -> None:
        """Draw obstacle cells as black squares on the z=0 plane."""
        if self.obstacle_grid is None:
            return
        polygons = self.obstacle_grid.blocked_polygons(z=0.0)
        if not polygons:
            return
        collection = Poly3DCollection(polygons, facecolors="black", edgecolors="none", alpha=0.75)
        collection.set_zorder(0)
        collection.set_clip_on(False)
        try:
            collection.set_zsort("average")
        except AttributeError:
            pass
        self.ax3d.add_collection3d(collection)
        self._obstacle_artist = collection

    def _init_label_axis(self) -> None:
        """Initialize the label panel axis."""
        if self.ax_labels is None:
            return
        self.ax_labels.set_title(
            "Legend / Estimates",
            fontsize=self._label_title_fontsize,
            loc="left",
        )
        self.ax_labels.axis("off")

    def _apply_layout(self) -> None:
        """Apply explicit axes positions for the PF/legend layout."""
        layout = self._layout_geometry()
        self.fig.set_size_inches(*layout.fig_size, forward=True)
        if self.ax3d is not None:
            self.ax3d.set_position(layout.pf_pos)
        if self.ax_labels is not None:
            self.ax_labels.set_position(layout.labels_pos)

    def _legend_lines(self) -> List[Tuple[str, str, str, str]]:
        """Build legend-style label lines with matching colors and markers."""
        lines: List[Tuple[str, str, str, str]] = []
        active = set(self._iter_active_isotopes())
        for iso, pos in self.true_sources.items():
            if iso not in active:
                continue
            if pos.size:
                positions = np.atleast_2d(pos)
                strengths = self._true_strengths_for_iso(iso, positions.shape[0])
                for idx, pos_row in enumerate(positions):
                    label = f"True {iso} pos={_format_pos(pos_row)}"
                    strength = strengths[idx]
                    if strength is not None:
                        label = f"{label} q={strength:.1f} cps@1m"
                    lines.append((label, self.colors.get(iso, "black"), "*", "None"))
        lines.append(("trajectory", "cyan", "o", "-"))
        lines.append(("robot", "cyan", "o", "None"))
        if self.obstacle_grid is not None and self.obstacle_grid.blocked_cells:
            lines.append(("obstacles", "black", "s", "None"))
        for iso in self._iter_active_isotopes():
            color = self.colors.get(iso, "black")
            lines.append((f"{iso} particles", color, ".", "None"))
            lines.append((f"{iso} est", self.estimate_colors.get(iso, color), "x", "None"))
        return lines

    def _legend_lines_estimates_only(self) -> List[Tuple[str, str, str, str]]:
        """Build legend lines for the estimates-only view."""
        lines: List[Tuple[str, str, str, str]] = []
        active = set(self._iter_active_isotopes())
        for iso, pos in self.true_sources.items():
            if iso not in active:
                continue
            if pos.size:
                positions = np.atleast_2d(pos)
                strengths = self._true_strengths_for_iso(iso, positions.shape[0])
                for idx, pos_row in enumerate(positions):
                    label = f"True {iso} pos={_format_pos(pos_row)}"
                    strength = strengths[idx]
                    if strength is not None:
                        label = f"{label} q={strength:.1f} cps@1m"
                    lines.append((label, self.colors.get(iso, "black"), "*", "None"))
        lines.append(("trajectory", "cyan", "o", "-"))
        lines.append(("robot", "cyan", "o", "None"))
        if self.obstacle_grid is not None and self.obstacle_grid.blocked_cells:
            lines.append(("obstacles", "black", "s", "None"))
        for iso in self._iter_active_isotopes():
            color = self.estimate_colors.get(iso, self.colors.get(iso, "black"))
            lines.append((f"{iso} est", color, "x", "None"))
        return lines

    def _true_strengths_for_iso(self, iso: str, count: int) -> List[float | None]:
        """Return per-source true strengths for an isotope."""
        strengths = self.true_strengths.get(iso, None)
        if strengths is None:
            return [None] * count
        if isinstance(strengths, np.ndarray):
            values = strengths.reshape(-1).tolist()
        elif isinstance(strengths, (list, tuple)):
            values = list(strengths)
        else:
            values = [float(strengths)]
        if len(values) < count:
            values.extend([None] * (count - len(values)))
        return [float(v) if v is not None else None for v in values[:count]]

    def _estimate_lines(self, frame: PFFrame) -> List[Tuple[str, str]]:
        """Build estimate text lines for the strongest source per isotope."""
        lines: List[Tuple[str, str]] = []
        for iso in self._iter_active_isotopes():
            est_pos = frame.estimated_sources.get(iso, np.zeros((0, 3)))
            strengths = frame.estimated_strengths.get(iso, np.zeros(0))
            if strengths.size and est_pos.size:
                idx = int(np.argmax(strengths))
                pos = est_pos[idx]
                strength = float(strengths[idx])
                text = f"{iso}: pos={_format_pos(pos)} q={strength:.1f} cps@1m"
            else:
                text = f"{iso}: no estimate"
            lines.append((text, self.estimate_colors.get(iso, self.colors.get(iso, "black"))))
        return lines

    def _estimate_lines_all(self, frame: PFFrame) -> List[Tuple[str, str]]:
        """Build estimate text lines for all sources per isotope."""
        lines: List[Tuple[str, str]] = []
        for iso in self._iter_active_isotopes():
            est_pos = frame.estimated_sources.get(iso, np.zeros((0, 3)))
            strengths = frame.estimated_strengths.get(iso, np.zeros(0))
            color = self.estimate_colors.get(iso, self.colors.get(iso, "black"))
            if strengths.size and est_pos.size:
                for idx, (pos, strength) in enumerate(zip(est_pos, strengths)):
                    text = f"{iso}[{idx}]: pos={_format_pos(pos)} q={float(strength):.1f} cps@1m"
                    lines.append((text, color))
            else:
                lines.append((f"{iso}: no estimate", color))
        return lines

    def _estimate_color(self, base_color: str) -> Tuple[float, float, float]:
        """Return a darker variant of the base color for estimate markers."""
        rgb = np.array(mcolors.to_rgb(base_color))
        hsv = mcolors.rgb_to_hsv(rgb)
        hsv[1] = min(1.0, hsv[1] * 1.1 + 0.2)
        hsv[2] = max(0.2, hsv[2] * 0.6)
        return tuple(mcolors.hsv_to_rgb(hsv))

    def _update_estimate_texts(self, iso: str, positions: NDArray[np.float64], color: str) -> None:
        """Update estimate position text above markers for one isotope."""
        if self.ax3d is None:
            return
        self._estimate_text_positions[iso] = positions.copy()
        artists = self._estimate_text_artists.setdefault(iso, [])
        for idx, pos in enumerate(positions):
            coords = [f"{val:.2f}" for val in pos.tolist()]
            text = f"[{', '.join(coords)}]"
            if idx >= len(artists):
                art = self.ax3d.text2D(
                    0.0,
                    0.0,
                    text,
                    transform=self.ax3d.transAxes,
                    color=color,
                    fontsize=self._label_text_fontsize,
                    ha="center",
                    va="bottom",
                    rotation=0,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.35, boxstyle="round,pad=0.15"),
                )
                art.set_path_effects([])
                art.set_clip_on(False)
                artists.append(art)
            else:
                art = artists[idx]
                art.set_text(text)
                art.set_color(color)
                art.set_visible(True)
                art.set_path_effects([])
            art.set_zorder(30)
        for extra in artists[len(positions) :]:
            extra.set_visible(False)

    def _update_true_texts(self, iso: str, positions: NDArray[np.float64], color: str) -> None:
        """Update true position text above markers for one isotope."""
        if self.ax3d is None:
            return
        self._true_text_positions[iso] = positions.copy()
        artists = self._true_text_artists.setdefault(iso, [])
        for idx, pos in enumerate(positions):
            coords = [f"{val:.2f}" for val in pos.tolist()]
            text = f"True[{', '.join(coords)}]"
            if idx >= len(artists):
                art = self.ax3d.text2D(
                    0.0,
                    0.0,
                    text,
                    transform=self.ax3d.transAxes,
                    color=color,
                    fontsize=self._label_text_fontsize,
                    ha="center",
                    va="bottom",
                    rotation=0,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.35, boxstyle="round,pad=0.15"),
                )
                art.set_path_effects(
                    [
                        path_effects.withStroke(linewidth=2.5, foreground="white"),
                        path_effects.Normal(),
                    ]
                )
                art.set_clip_on(False)
                artists.append(art)
            else:
                art = artists[idx]
                art.set_text(text)
                art.set_color(color)
                art.set_visible(True)
                art.set_path_effects(
                    [
                        path_effects.withStroke(linewidth=2.5, foreground="white"),
                        path_effects.Normal(),
                    ]
                )
            art.set_zorder(29)
        for extra in artists[len(positions) :]:
            extra.set_visible(False)

    def _position_all_estimate_texts(self) -> None:
        """Update positions for all estimate text labels."""
        if self.ax3d is None:
            return
        renderer = self.fig.canvas.get_renderer()
        items: list[dict[str, Any]] = []
        for artists_map, positions_map in (
            (self._estimate_text_artists, self._estimate_text_positions),
            (self._true_text_artists, self._true_text_positions),
        ):
            for iso, artists in artists_map.items():
                positions = positions_map.get(iso, np.zeros((0, 3)))
                for pos, art in zip(positions, artists):
                    coords = self._project_to_axes(pos)
                    if coords is None:
                        art.set_visible(False)
                        continue
                    art.set_position(coords)
                    art.set_visible(True)
                    x_disp, y_disp = self.ax3d.transAxes.transform(coords)
                    bbox = art.get_window_extent(renderer=renderer)
                    items.append(
                        {
                            "artist": art,
                            "x_disp": float(x_disp),
                            "y_disp": float(y_disp),
                            "bbox": bbox,
                        }
                    )
        placed = []
        for item in sorted(items, key=lambda i: i["bbox"].y0, reverse=True):
            bbox = item["bbox"]
            shift = 0.0
            while any(bbox.overlaps(prev) for prev in placed):
                shift += self._ESTIMATE_TEXT_PAD_PX
                bbox = item["bbox"].translated(0, shift)
            x_disp = item["x_disp"]
            y_disp = item["y_disp"] + shift
            x_ax, y_ax = self.ax3d.transAxes.inverted().transform((x_disp, y_disp))
            item["artist"].set_position((float(x_ax), float(y_ax)))
            placed.append(bbox)

    def _axis_projection_lines(
        self,
        points: NDArray[np.float64],
        color: str,
        alpha: float = 0.35,
    ) -> list:
        """Draw thin dotted projection lines from points to each axis plane."""
        if points.size == 0:
            return []
        x0 = 0.0
        y0 = 0.0
        z0 = 0.0
        artists: list = []
        for x, y, z in points:
            artists.append(
                self.ax3d.plot(
                    [x, x],
                    [y, y],
                    [z, z0],
                    linestyle=":",
                    linewidth=self._projection_linewidth,
                    color=color,
                    alpha=alpha,
                )[0]
            )
            artists.append(
                self.ax3d.plot(
                    [x, x],
                    [y, y0],
                    [z, z],
                    linestyle=":",
                    linewidth=self._projection_linewidth,
                    color=color,
                    alpha=alpha,
                )[0]
            )
            artists.append(
                self.ax3d.plot(
                    [x, x0],
                    [y, y],
                    [z, z],
                    linestyle=":",
                    linewidth=self._projection_linewidth,
                    color=color,
                    alpha=alpha,
                )[0]
            )
        for art in artists:
            art.set_clip_on(False)
            art.set_zorder(4)
        return artists

    def _ensure_label_height(self, total_lines: int) -> None:
        """Expand the figure height so label lines keep readable spacing."""
        if self.ax_labels is None or total_lines <= 0:
            return
        layout = self._layout_geometry()
        label_height_frac = layout.labels_pos[3]
        if label_height_frac <= 0.0:
            return
        current_height = self._fig_height
        line_height_in = (self._label_text_fontsize / 72.0) * self._LABEL_LINE_SPACING
        required_axis_in = line_height_in * total_lines / 0.95
        required_fig_height = required_axis_in / label_height_frac
        if required_fig_height > current_height + 1e-3:
            self._fig_height = required_fig_height
            self._apply_layout()

    def _update_labels(
        self,
        frame: PFFrame,
        legend_lines: List[Tuple[str, str, str, str]] | None = None,
        estimate_lines: List[Tuple[str, str]] | None = None,
    ) -> None:
        """Update the label panel with legend entries and estimates."""
        if self.ax_labels is None:
            return
        self.ax_labels.cla()
        if frame.step_index < 0:
            step_text = "Initialize"
        else:
            step_text = f"Step {frame.step_index} t={frame.time:.2f}s"
        self.ax_labels.set_title(
            step_text,
            fontsize=self._label_title_fontsize,
            loc="left",
        )
        self.ax_labels.axis("off")
        legend_lines = self._legend_lines() if legend_lines is None else legend_lines
        estimate_lines = self._estimate_lines_all(frame) if estimate_lines is None else estimate_lines
        gap_lines = 1
        total_lines = len(legend_lines) + len(estimate_lines) + 2 + gap_lines
        self._ensure_label_height(total_lines)
        line_height = 0.95 / max(total_lines, 1)
        y = 0.96
        self.ax_labels.text(
            0.0,
            y,
            "Legend",
            transform=self.ax_labels.transAxes,
            va="top",
            ha="left",
            fontsize=self._label_section_fontsize,
            fontweight="bold",
            color="black",
        )
        y -= line_height
        for text, color, marker, linestyle in legend_lines:
            self.ax_labels.text(
                self._label_text_x,
                y,
                text,
                transform=self.ax_labels.transAxes,
                va="top",
                ha="left",
                fontsize=self._label_text_fontsize,
                color=color,
            )
            if linestyle != "None":
                self.ax_labels.plot(
                    list(self._label_marker_line),
                    [y - 0.012, y - 0.012],
                    transform=self.ax_labels.transAxes,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=7,
                    linewidth=1.0,
                )
            else:
                self.ax_labels.plot(
                    [self._label_marker_point],
                    [y - 0.012],
                    transform=self.ax_labels.transAxes,
                    color=color,
                    linestyle="None",
                    marker=marker,
                    markersize=7,
                )
            y -= line_height
        y -= line_height
        self.ax_labels.text(
            0.0,
            y,
            "Estimates",
            transform=self.ax_labels.transAxes,
            va="top",
            ha="left",
            fontsize=self._label_section_fontsize,
            fontweight="bold",
            color="black",
        )
        y -= line_height
        for text, color in estimate_lines:
            self.ax_labels.text(
                0.0,
                y,
                text,
                transform=self.ax_labels.transAxes,
                va="top",
                ha="left",
                fontsize=self._label_text_fontsize,
                color=color,
            )
            y -= line_height

    def _particle_style(self, weights: NDArray[np.float64], base_color: str) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Map particle weights to marker sizes and RGBA colors."""
        if weights.size == 0:
            return np.zeros(0), np.zeros((0, 4))
        w = np.asarray(weights, dtype=float)
        w_min = float(np.min(w))
        w_max = float(np.max(w))
        denom = w_max - w_min
        if denom <= 1e-12:
            w = np.ones_like(w)
        else:
            w = (w - w_min) / denom
        w = np.clip(w, 0.0, 1.0) ** self._particle_weight_exponent
        min_size, max_size = self._particle_size_range
        min_alpha, max_alpha = self._particle_alpha_range
        sizes = min_size + (max_size - min_size) * w
        alphas = min_alpha + (max_alpha - min_alpha) * w
        base_rgba = mcolors.to_rgba(base_color)
        colors = np.tile(base_rgba, (len(w), 1))
        colors[:, 3] = alphas
        return sizes, colors

    def update(self, frame: PFFrame) -> None:
        """Redraw the scene for the given PFFrame."""
        self._last_frame = frame
        if self.ax3d is not None:
            self.ax3d.computed_zorder = False
        init_frame = frame.step_index < 0
        # maintain trajectory history
        _extend_trajectory_history(self._traj_history, frame)
        # Robot and trajectory
        traj_arr = np.vstack(self._traj_history)
        if self._traj_line is None:
            (self._traj_line,) = self.ax3d.plot(
                traj_arr[:, 0],
                traj_arr[:, 1],
                traj_arr[:, 2],
                "-o",
                color="cyan",
                label="trajectory",
                zorder=20,
            )
        else:
            self._traj_line.set_data(traj_arr[:, 0], traj_arr[:, 1])
            self._traj_line.set_3d_properties(traj_arr[:, 2])
            self._traj_line.set_zorder(20)
        if self._robot_artist is None:
            self._robot_artist = self.ax3d.scatter(
                frame.robot_position[0],
                frame.robot_position[1],
                frame.robot_position[2],
                color="cyan",
                marker="o",
                s=80,
                label="robot",
                depthshade=False,
                zorder=21,
            )
        else:
            self._robot_artist._offsets3d = (
                np.array([frame.robot_position[0]]),
                np.array([frame.robot_position[1]]),
                np.array([frame.robot_position[2]]),
            )
            self._robot_artist.set_zorder(21)
        # Shields as arrows
        for arr in self._shield_arrows.values():
            arr.remove()
        self._shield_arrows = {}
        origin = frame.robot_position
        arrow_specs = {
            "Fe": (frame.RFe[:, 2], "magenta"),
            "Pb": (frame.RPb[:, 2], "green"),
        }
        for name, (normal, color) in arrow_specs.items():
            arr = self.ax3d.quiver(
                origin[0],
                origin[1],
                origin[2],
                normal[0],
                normal[1],
                normal[2],
                length=1.0,
                color=color,
                normalize=True,
                label=f"{name} shield",
            )
            arr.set_zorder(19)
            self._shield_arrows[name] = arr
        # Estimated sources and particles
        for iso in self.isotopes:
            pts = frame.particle_positions.get(iso, np.zeros((0, 3)))
            weights = frame.particle_weights.get(iso, np.zeros(0))
            color = self.colors.get(iso, None)
            if init_frame and pts.size:
                _, max_size = self._particle_size_range
                _, max_alpha = self._particle_alpha_range
                sizes = np.full(pts.shape[0], max_size, dtype=float)
                base_rgba = mcolors.to_rgba(color)
                colors = np.tile(base_rgba, (pts.shape[0], 1))
                colors[:, 3] = max_alpha
            else:
                sizes, colors = self._particle_style(weights, color)
            if iso not in self._particle_artists:
                if pts.size:
                    self._particle_artists[iso] = self.ax3d.scatter(
                        pts[:, 0],
                        pts[:, 1],
                        pts[:, 2],
                        s=sizes if sizes.size else 5,
                        c=colors if colors.size else color,
                        label=f"{iso} particles",
                        depthshade=False,
                        zorder=3,
                    )
            else:
                art = self._particle_artists[iso]
                if pts.size:
                    art._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
                    if sizes.size:
                        art.set_sizes(sizes)
                    if colors.size:
                        art.set_facecolors(colors)
                        art.set_edgecolors(colors)
                    art.set_zorder(3)
                else:
                    art._offsets3d = ([], [], [])
            est_pos = frame.estimated_sources.get(iso, np.zeros((0, 3)))
            est_color = self.estimate_colors.get(iso, color)
            if iso not in self._est_artists:
                if est_pos.size:
                    self._est_artists[iso] = self.ax3d.scatter(
                        est_pos[:, 0],
                        est_pos[:, 1],
                        est_pos[:, 2],
                        marker="x",
                        s=180,
                        color=est_color,
                        linewidths=2.5,
                        label=f"{iso} est",
                        depthshade=False,
                        zorder=28,
                    )
            else:
                art = self._est_artists[iso]
                if est_pos.size:
                    art._offsets3d = (est_pos[:, 0], est_pos[:, 1], est_pos[:, 2])
                    art.set_color(est_color)
                    art.set_zorder(28)
                else:
                    art._offsets3d = ([], [], [])
            if est_pos.size:
                self._update_estimate_texts(iso, est_pos, est_color)
            else:
                self._update_estimate_texts(iso, np.zeros((0, 3)), est_color)
        for art in self._projection_artists:
            art.remove()
        self._projection_artists = []
        for iso in self.isotopes:
            est_pos = frame.estimated_sources.get(iso, np.zeros((0, 3)))
            est_color = self.estimate_colors.get(iso, self.colors.get(iso, "black"))
            self._projection_artists.extend(self._axis_projection_lines(est_pos, est_color))
        self.ax3d.set_title("")
        self._ensure_x_label()
        self._update_x_label_position()
        self._position_all_estimate_texts()
        self._update_labels(frame)
        self.fig.canvas.draw_idle()

    def save_final(self, path: str = "result.png") -> None:
        """Save the current figure."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # If we have a last frame, ensure markers are up to date
        if self._last_frame is not None:
            self.update(self._last_frame)
        self.fig.savefig(out, dpi=200)
        self.fig.canvas.draw_idle()

    def save_estimates_only(self, path: str = "result_estimates.png") -> None:
        """Save a figure with only estimate markers visible on the 3D axis."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if self._last_frame is not None:
            self.update(self._last_frame)

        hidden: list[tuple[Any, bool]] = []

        def _hide(artist: Any) -> None:
            """Hide one artist while recording its previous visibility."""
            if artist is None:
                return
            if hasattr(artist, "get_visible") and hasattr(artist, "set_visible"):
                hidden.append((artist, artist.get_visible()))
                artist.set_visible(False)

        for art in self._particle_artists.values():
            _hide(art)
        for art in self._shield_arrows.values():
            _hide(art)

        for art in self._est_artists.values():
            if hasattr(art, "set_visible"):
                art.set_visible(True)

        if self._last_frame is not None:
            self._update_labels(
                self._last_frame,
                legend_lines=self._legend_lines_estimates_only(),
                estimate_lines=self._estimate_lines_all(self._last_frame),
            )
        self.fig.savefig(out, dpi=200)
        for art, vis in hidden:
            art.set_visible(vis)
        if self._last_frame is not None:
            self._update_labels(self._last_frame)
        self.fig.canvas.draw_idle()


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
        isotopes: List[str],
        output_dir: str | Path,
        *,
        world_bounds: Optional[Tuple[float, float, float, float, float, float]] = None,
        true_sources: Optional[Dict[str, NDArray[np.float64]]] = None,
        true_strengths: Optional[Dict[str, float | Sequence[float]]] = None,
        obstacle_grid: ObstacleGrid | None = None,
        max_particles_per_isotope: int | None = None,
    ) -> None:
        """Initialize output paths and static scene metadata."""
        self.isotopes = list(isotopes)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.world_bounds = world_bounds or (0.0, 10.0, 0.0, 10.0, 0.0, 3.0)
        self.true_sources = true_sources or {}
        self.true_strengths = true_strengths or {}
        self.obstacle_grid = obstacle_grid
        self.max_particles_per_isotope = (
            None
            if max_particles_per_isotope is None
            else max(1, int(max_particles_per_isotope))
        )
        self.trajectory: list[NDArray[np.float64]] = []
        self.path_segments: list[NDArray[np.float64]] = []
        self.measurement_points: list[NDArray[np.float64]] = []
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
        self.latest_spectrum_path = self.output_dir / "latest_spectrum.png"
        self.index_path = self.output_dir / "index.html"
        self._write_index_html()
        if not self.latest_overview_path.exists():
            self._save_overview_placeholder(self.latest_overview_path)
        if not self.latest_spectrum_path.exists():
            self._save_spectrum_placeholder(self.latest_spectrum_path)

    def update(self, frame: PFFrame) -> None:
        """Render and save the split CUI views for one PF frame."""
        self.update_index += 1
        setattr(frame, "_cui_update_index", int(self.update_index))
        _extend_trajectory_history(self.trajectory, frame)
        self._record_path_segment(frame)
        self._record_measurement_point(frame)
        step = max(0, int(frame.step_index))
        robot_step_path = self.output_dir / f"robot_2d_step_{step:04d}.png"
        overview_step_path = self.output_dir / f"experiment_overview_step_{step:04d}.png"
        pf_step_path = self.output_dir / f"pf_3d_step_{step:04d}.png"
        spectrum_step_path = self.output_dir / f"spectrum_step_{step:04d}.png"
        self._save_robot_2d(frame, robot_step_path)
        shutil.copyfile(robot_step_path, self.latest_robot_path)
        self._save_experiment_overview(frame, overview_step_path)
        shutil.copyfile(overview_step_path, self.latest_overview_path)
        self._save_pf_3d(frame, pf_step_path)
        shutil.copyfile(pf_step_path, self.latest_pf_path)
        self._save_spectrum(frame, spectrum_step_path)
        if spectrum_step_path.exists():
            shutil.copyfile(spectrum_step_path, self.latest_spectrum_path)

    def _record_path_segment(self, frame: PFFrame) -> None:
        """Store the obstacle-aware segment associated with this frame, if any."""
        waypoints = _coerce_path_waypoints(frame)
        if waypoints.shape[0] < 2:
            return
        if self.path_segments:
            prev = self.path_segments[-1]
            if prev.shape == waypoints.shape and np.allclose(prev, waypoints):
                return
        self.path_segments.append(waypoints.copy())

    def _record_measurement_point(self, frame: PFFrame) -> None:
        """Store measurement stations and repeated shield visits for display."""
        point = np.asarray(frame.robot_position, dtype=float).reshape(3)
        if self.measurement_points:
            if float(np.linalg.norm(point - self.measurement_points[-1])) <= 1e-6:
                self.measurement_visit_counts[-1] += 1
                return
        self.measurement_points.append(point.copy())
        self.measurement_steps.append(int(frame.step_index))
        self.measurement_visit_counts.append(1)

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
                if any(float(np.linalg.norm(point - existing)) <= 1e-6 for existing in waypoints):
                    continue
                waypoints.append(np.asarray(point, dtype=float).reshape(3).copy())
        if not waypoints:
            return np.zeros((0, 3), dtype=float)
        return np.vstack(waypoints).astype(float)

    def _station_label_offsets(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
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
        visits = (
            self.measurement_visit_counts[station_index]
            if station_index < len(self.measurement_visit_counts)
            else 1
        )
        if visits <= 1:
            return str(station_index)
        return f"{station_index}({visits})"

    def _frame_progress_label(self, frame: PFFrame) -> str:
        """Return a title suffix that separates measurement, render, and station progress."""
        update_idx = int(getattr(frame, "_cui_update_index", 0))
        station_idx = max(0, len(self.measurement_points) - 1)
        visit_count = (
            int(self.measurement_visit_counts[-1])
            if self.measurement_visit_counts
            else 0
        )
        return (
            f"measurement={int(frame.step_index)} "
            f"update={update_idx} "
            f"station={station_idx} visit={visit_count} "
            f"t={frame.time:.1f}s"
        )

    def _write_index_html(self) -> None:
        """Write the browser page that auto-refreshes the latest PNG files."""
        html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Rotating Shield PF CUI View</title>
  <style>
    body { margin: 0; background: #111; color: #eee; font-family: sans-serif; }
    header { padding: 10px 16px; background: #1d1d1d; border-bottom: 1px solid #333; }
    main { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; padding: 10px; }
    section { background: #181818; border: 1px solid #333; padding: 8px; }
    h2 { margin: 0 0 8px; font-size: 16px; font-weight: 600; }
    img { width: 100%; height: calc(50vh - 70px); object-fit: contain; background: #fff; }
    .wide { grid-column: 1 / span 2; }
  </style>
</head>
<body>
  <header>Rotating Shield PF CUI View - auto refresh every 2 s</header>
  <main>
    <section class="wide overview"><h2>RA-L experiment overview</h2><img id="overview" src="latest_experiment_overview.png"></section>
    <section><h2>Robot position 2D</h2><img id="robot" src="latest_robot_2d.png"></section>
    <section><h2>Particle filter 3D</h2><img id="pf" src="latest_pf_3d.png"></section>
    <section class="wide"><h2>Raw native full spectrum</h2><img id="spectrum" src="latest_spectrum.png"></section>
  </main>
  <script>
    function refresh() {
      const t = Date.now();
      document.getElementById("overview").src = "latest_experiment_overview.png?t=" + t;
      document.getElementById("robot").src = "latest_robot_2d.png?t=" + t;
      document.getElementById("pf").src = "latest_pf_3d.png?t=" + t;
      document.getElementById("spectrum").src = "latest_spectrum.png?t=" + t;
    }
    setInterval(refresh, 2000);
  </script>
</body>
</html>
"""
        self.index_path.write_text(html, encoding="utf-8")

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
        """Draw the obstacle grid as 2D filled rectangles."""
        if self.obstacle_grid is None:
            return
        from matplotlib.patches import Rectangle

        for x0, x1, y0, y1 in self.obstacle_grid.blocked_bounds():
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    facecolor="black",
                    edgecolor="none",
                    alpha=0.75,
                )
            )

    def _draw_obstacles_3d(self, ax: plt.Axes) -> None:
        """Draw obstacle cells as flat dark floor patches in the 3D PF view."""
        if self.obstacle_grid is None:
            return
        patches = []
        z0 = float(self.world_bounds[4])
        for x0, x1, y0, y1 in self.obstacle_grid.blocked_bounds():
            patches.append(
                [
                    (x0, y0, z0),
                    (x1, y0, z0),
                    (x1, y1, z0),
                    (x0, y1, z0),
                ]
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
        """Plot true source positions on the 2D robot view when available."""
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
        """Plot true source positions in an x-z elevation projection."""
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
                zorder=10,
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
        for iso in self.isotopes:
            truth_count = int(
                np.asarray(self.true_sources.get(iso, np.zeros((0, 3))), dtype=float)
                .reshape((-1, 3))
                .shape[0]
            ) if iso in self.true_sources else 0
            est_count = int(
                np.asarray(
                    frame.estimated_sources.get(iso, np.zeros((0, 3), dtype=float)),
                    dtype=float,
                )
                .reshape((-1, 3))
                .shape[0]
            )
            lines.append(
                f"{iso}: truth={truth_count} estimate={est_count}"
            )
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
                "PF visualization requires one posterior weight per displayed "
                "particle."
            )
        if self.max_particles_per_isotope is None or pts.shape[0] <= self.max_particles_per_isotope:
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
                label="traversed path"
                if idx == 0 and show_legend_context
                else None,
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
    def _point_count(points_by_isotope: Dict[str, NDArray[np.float64]]) -> int:
        """Return the total number of display points across isotope arrays."""
        total = 0
        for points in points_by_isotope.values():
            arr = np.asarray(points, dtype=float)
            if arr.size:
                total += int(arr.reshape((-1, 3)).shape[0])
        return total

    def _save_pf_3d(self, frame: PFFrame, output_path: Path) -> None:
        """Save the current PF particles and estimates as a 3D PNG."""
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
) -> None:
    """Render CUI split-view frames in a dedicated worker process."""
    visualizer = CUISplitPFVisualizer(**config)
    while True:
        message, payload = frame_queue.get()
        if message == "close":
            return
        if message != "frame":
            continue
        try:
            frame = pickle.loads(payload)
            visualizer.update(frame)
        except Exception as exc:  # pragma: no cover - worker-side diagnostics only.
            print(f"Async CUI split visualization worker error: {exc}", flush=True)


class AsyncCUISplitPFVisualizer:
    """Non-blocking process-backed wrapper for CUI split visualization."""

    def __init__(
        self,
        isotopes: List[str],
        output_dir: str | Path,
        *,
        world_bounds: Optional[Tuple[float, float, float, float, float, float]] = None,
        true_sources: Optional[Dict[str, NDArray[np.float64]]] = None,
        true_strengths: Optional[Dict[str, float | Sequence[float]]] = None,
        obstacle_grid: ObstacleGrid | None = None,
        max_particles_per_isotope: int | None = None,
        queue_size: int = 2,
    ) -> None:
        """Start a renderer process that consumes latest PF frames asynchronously."""
        self.isotopes = list(isotopes)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.output_dir / "index.html"
        self.latest_robot_path = self.output_dir / "latest_robot_2d.png"
        self.latest_overview_path = self.output_dir / "latest_experiment_overview.png"
        self.latest_pf_path = self.output_dir / "latest_pf_3d.png"
        self.latest_spectrum_path = self.output_dir / "latest_spectrum.png"
        self._closed = False
        self._ctx = mp.get_context("spawn")
        self._queue = self._ctx.Queue(maxsize=max(1, int(queue_size)))
        config = {
            "isotopes": self.isotopes,
            "output_dir": self.output_dir,
            "world_bounds": world_bounds,
            "true_sources": true_sources,
            "true_strengths": true_strengths,
            "obstacle_grid": obstacle_grid,
            "max_particles_per_isotope": max_particles_per_isotope,
        }
        self._process = self._ctx.Process(
            target=_async_cui_split_worker,
            args=(config, self._queue),
            daemon=True,
        )
        self._process.start()

    def update(self, frame: PFFrame) -> None:
        """Queue the latest frame for asynchronous rendering without blocking."""
        if self._closed or not self._process.is_alive():
            return
        payload = pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL)
        while True:
            try:
                self._queue.put_nowait(("frame", payload))
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    return

    def close(self, timeout_s: float = 10.0) -> None:
        """Ask the renderer process to finish queued work and stop."""
        if self._closed:
            return
        self._closed = True
        if self._process.is_alive():
            try:
                self._queue.put(("close", None), timeout=1.0)
            except queue.Full:
                pass
            self._process.join(timeout=max(0.1, float(timeout_s)))
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2.0)


def build_frame_from_pf(
    pf,
    step_index: int,
    time_sec: float,
    *,
    detector_position: NDArray[np.float64],
    live_time_s: float,
    RFe: NDArray[np.float64] | None = None,
    RPb: NDArray[np.float64] | None = None,
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
        live_time_s: Fixed observation live time in seconds.
        RFe: Iron-shield world rotation.
        RPb: Lead-shield world rotation.
        spectrum_energy_keV: Native incident-energy bin axis.
        spectrum_counts: Raw nonnegative native histogram.
    """
    if hasattr(pf, "visualization_estimates"):
        est: Dict[str, object] = pf.visualization_estimates()
    elif hasattr(pf, "estimate_all"):
        est = pf.estimate_all()
    else:
        est = pf.estimates()  # type: ignore[attr-defined]
    particle_positions: Dict[str, NDArray[np.float64]] = {}
    particle_weights: Dict[str, NDArray[np.float64]] = {}
    particle_representative_positions: Dict[str, NDArray[np.float64]] = {}
    particle_representative_weights: Dict[str, NDArray[np.float64]] = {}
    estimated_sources: Dict[str, NDArray[np.float64]] = {}
    estimated_strengths: Dict[str, NDArray[np.float64]] = {}

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
            if hasattr(value, "positions"):
                est_pos = np.asarray(value.positions, dtype=float)
                est_str = np.asarray(value.strengths, dtype=float)
            elif isinstance(value, tuple) and len(value) == 2:
                est_pos = np.asarray(value[0], dtype=float)
                est_str = np.asarray(value[1], dtype=float)
            else:
                raise TypeError(
                    "PF estimate_all() must return (positions, strengths) "
                    f"for isotope {iso}."
                )
        else:
            est_pos = np.zeros((0, 3), dtype=float)
            est_str = np.zeros(0, dtype=float)
        est_pos = np.asarray(est_pos, dtype=float).reshape(-1, 3)
        est_str = np.asarray(est_str, dtype=float).reshape(-1)
        if est_pos.shape[0] != est_str.shape[0]:
            raise ValueError(
                "PF estimate_all() must return one strength per estimated "
                f"source for isotope {iso}."
            )
        if np.any(~np.isfinite(est_pos)) or np.any(~np.isfinite(est_str)):
            raise ValueError(
                f"PF estimate_all() returned non-finite values for {iso}."
            )
        estimated_sources[iso] = est_pos
        estimated_strengths[iso] = est_str

    robot_pos = np.asarray(detector_position, dtype=float)
    if robot_pos.shape != (3,) or np.any(~np.isfinite(robot_pos)):
        raise ValueError("detector_position must be a finite XYZ vector.")
    duration = float(live_time_s)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("live_time_s must be finite and positive.")
    rotation_fe = (
        np.eye(3, dtype=float)
        if RFe is None
        else np.asarray(RFe, dtype=float)
    )
    rotation_pb = (
        np.eye(3, dtype=float)
        if RPb is None
        else np.asarray(RPb, dtype=float)
    )

    return PFFrame(
        step_index=step_index,
        time=time_sec,
        robot_position=robot_pos,
        robot_orientation=None,
        RFe=rotation_fe,
        RPb=rotation_pb,
        duration=duration,
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
