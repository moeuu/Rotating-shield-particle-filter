"""Data structures for capturing PF state per time step for visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any, Sequence

from pathlib import Path
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
    - counts_by_isotope: z_{k,h} from spectrum unfolding (Sec. 2.5.7)
    - particle_positions: isotope -> (N_points, 3)
    - particle_weights: isotope -> (N_points,)
    - estimated_sources: isotope -> (N_est, 3)
    - estimated_strengths: isotope -> (N_est,)
    """

    step_index: int
    time: float
    robot_position: NDArray[np.float64]
    robot_orientation: Optional[NDArray[np.float64]]
    RFe: NDArray[np.float64]
    RPb: NDArray[np.float64]
    duration: float
    counts_by_isotope: Dict[str, float]
    particle_positions: Dict[str, NDArray[np.float64]]
    particle_weights: Dict[str, NDArray[np.float64]]
    estimated_sources: Dict[str, NDArray[np.float64]]
    estimated_strengths: Dict[str, NDArray[np.float64]]


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
    """Return normalized weights with a uniform fallback when the sum is zero."""
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return w
    total = float(np.sum(w))
    if total <= 0.0:
        return np.ones_like(w) / w.size
    return w / total


def _existence_probabilities(states: Sequence[Any], weights: NDArray[np.float64], max_r: int) -> NDArray[np.float64]:
    """Return per-slot existence probabilities for a list of particle states."""
    if max_r <= 0 or not states:
        return np.zeros(0, dtype=float)
    w = _normalize_weights(weights)
    probs = np.zeros(max_r, dtype=float)
    for wi, st in zip(w, states):
        num_sources = int(getattr(st, "num_sources", 0))
        if num_sources > 0:
            probs[:num_sources] += wi
    return probs


def _format_pos(pos: NDArray[np.float64]) -> str:
    """Format a position vector with two decimal places."""
    coords = ", ".join(f"{val:.2f}" for val in np.asarray(pos, dtype=float).ravel())
    return f"[{coords}]"


def _mmse_estimate_by_slot(
    states: Sequence[Any],
    weights: NDArray[np.float64],
    *,
    max_r: int | None = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Compute per-slot MMSE estimates for positions/strengths and a slot-valid mask."""
    if not states:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=bool)
    if max_r is None:
        max_r = max(int(getattr(st, "num_sources", 0)) for st in states)
    max_r = int(max_r)
    if max_r < 0:
        max_r = 0
    if max_r <= 0:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=bool)
    positions = np.zeros((max_r, 3), dtype=float)
    strengths = np.zeros(max_r, dtype=float)
    slot_valid = np.zeros(max_r, dtype=bool)
    w = _normalize_weights(weights)
    for j in range(max_r):
        pos_stack: list[NDArray[np.float64]] = []
        str_stack: list[float] = []
        w_stack: list[float] = []
        for wi, st in zip(w, states):
            if int(getattr(st, "num_sources", 0)) > j:
                pos_stack.append(st.positions[j])
                str_stack.append(float(st.strengths[j]))
                w_stack.append(float(wi))
        if not w_stack:
            continue
        slot_valid[j] = True
        wj = _normalize_weights(np.asarray(w_stack, dtype=float))
        pos_arr = np.vstack(pos_stack)
        str_arr = np.asarray(str_stack, dtype=float)
        positions[j] = np.sum(wj[:, None] * pos_arr, axis=0)
        strengths[j] = float(np.sum(wj * str_arr))
    return positions, strengths, slot_valid


def _filter_estimates(
    positions: NDArray[np.float64],
    strengths: NDArray[np.float64],
    slot_valid: NDArray[np.bool_],
    existence_probs: NDArray[np.float64],
    min_strength: float | None,
    min_existence_prob: float | None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Filter estimates by strength and existence probability thresholds."""
    if strengths.size == 0:
        return positions, strengths
    mask = slot_valid.copy() if slot_valid.size else np.ones(strengths.shape[0], dtype=bool)
    mask_strength = np.ones(strengths.shape[0], dtype=bool)
    mask_exist = np.ones(strengths.shape[0], dtype=bool)
    if min_existence_prob is not None and existence_probs.size:
        mask_exist = existence_probs[: strengths.shape[0]] >= min_existence_prob
    if min_strength is not None:
        mask_strength = strengths >= min_strength
    mask = mask & mask_exist & mask_strength
    if not np.any(mask) and min_existence_prob is not None and existence_probs.size:
        idx = int(np.argmax(existence_probs[: strengths.shape[0]]))
        if min_strength is None or strengths[idx] >= float(min_strength):
            mask = np.zeros_like(mask, dtype=bool)
            mask[idx] = True
    return positions[mask], strengths[mask]


class RealTimePFVisualizer:
    """
    Simple matplotlib-based 3D visualizer for the PF state.

    - update(frame) redraws particles, estimates, counts, and label panel.
    - save_final(path) saves the current figure.
    - save_estimates_only(path) saves a view with only estimate markers visible.
    """

    _BASE_FIGSIZE = (15.0, 6.0)
    _BASE_LAYOUT_FRACS = {
        "left": 0.02,
        "right": 0.02,
        "gap": 0.02,
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
        show_counts: bool = True,
    ) -> None:
        """Initialize the visualizer and optional obstacle overlay."""
        self.isotopes = isotopes
        self.world_bounds = world_bounds or (0, 10, 0, 10, 0, 3)
        self.true_sources = true_sources or {}
        self.true_strengths = true_strengths or {}
        self.obstacle_grid = obstacle_grid
        self.show_counts = show_counts
        self._fig_width = self._BASE_FIGSIZE[0]
        self._fig_height = self._BASE_FIGSIZE[1]
        layout = self._layout_geometry()
        self.fig = plt.figure(figsize=layout.fig_size)
        if self.show_counts:
            self.ax3d = self.fig.add_axes(layout.pf_pos, projection="3d")
            if layout.counts_pos is None:
                raise ValueError("Counts axis position missing for counts layout.")
            self.ax_counts = self.fig.add_axes(layout.counts_pos)
            self.ax_labels = self.fig.add_axes(layout.labels_pos)
        else:
            self.ax3d = self.fig.add_axes(layout.pf_pos, projection="3d")
            self.ax_counts = None
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
        self._counts_bars = None
        # Pre-create bar containers with zeros
        if self.ax_counts is not None and self.isotopes:
            zeros = [0.0 for _ in self.isotopes]
            self._counts_bars = self.ax_counts.bar(self.isotopes, zeros, color=[self.colors.get(n, "gray") for n in self.isotopes])
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
        if self.show_counts:
            labels_height = pf_height * self._VERTICAL_LAYOUT["labels_frac"]
            counts_height = pf_height * self._VERTICAL_LAYOUT["counts_frac"]
            counts_bottom = bottom + labels_height
            counts_pos = (side_left, counts_bottom, side_width, counts_height)
            labels_pos = (side_left, bottom, side_width, labels_height)
        else:
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
        xmin, xmax, ymin, ymax, zmin, zmax = self.world_bounds
        self.ax3d.set_xlim(xmin, xmax)
        self.ax3d.set_ylim(ymin, ymax)
        self.ax3d.set_zlim(zmin, zmax)
        self.ax3d.set_yticks(np.arange(ymin, ymax + 1e-6, 2.0))
        self._set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))
        self.ax3d.set_ylabel("y [m]")
        self.ax3d.set_zlabel("z [m]")
        self.ax3d.set_yticks(np.arange(ymin, ymax + 1e-6, 2.0))
        if self.ax_counts is not None:
            self.ax_counts.set_ylabel("Counts")
            self.ax_counts.set_title("Isotope-wise counts")
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
        if self.show_counts and self.ax_counts is not None and self.ax_labels is not None:
            if layout.counts_pos is None:
                raise ValueError("Counts axis position missing for counts layout.")
            self.ax_counts.set_position(layout.counts_pos)
            self.ax_labels.set_position(layout.labels_pos)
        elif self.ax_labels is not None:
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
        # maintain trajectory history
        self._traj_history.append(frame.robot_position)
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
        # Counts bar (reuse)
        if self.ax_counts is not None and frame.counts_by_isotope:
            names = list(frame.counts_by_isotope.keys())
            vals = [frame.counts_by_isotope[n] for n in names]
            # Update pre-created bars in the same order as isotopes; fallback if new isotope appears
            name_to_idx = {n: i for i, n in enumerate(self.isotopes)}
            for bar in self._counts_bars:
                bar.set_height(0.0)
            for n, v in zip(names, vals):
                if n in name_to_idx:
                    self._counts_bars[name_to_idx[n]].set_height(v)
                else:
                    # new isotope -> append a bar
                    new_bar = self.ax_counts.bar([n], [v], color=self.colors.get(n, "gray"))[0]
                    self._counts_bars.append(new_bar)
            self.ax_counts.set_ylabel("Counts")
            self.ax_counts.set_title("Unfolded counts z_{k,h}")
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


def build_frame_from_pf(
    pf,
    measurement,
    step_index: int,
    time_sec: float,
    *,
    estimate_mode: str = "mmse",
    min_est_strength: float | None = None,
    min_existence_prob: float | None = None,
) -> PFFrame:
    """
    Construct a PFFrame snapshot from a PF (ParallelIsotopePF-like) and a Measurement.

    Args:
        pf: ParallelIsotopePF or compatible with .filters and .estimate_all()
        measurement: Measurement with counts_by_isotope, pose_idx/orient_idx/RFe/RPb (optional)
        step_index: integer step
        time_sec: cumulative time in seconds
        estimate_mode: "mmse" for weighted mean or "map" for max-weight particle
        min_est_strength: optional minimum strength threshold for displayed estimates
        min_existence_prob: optional minimum existence probability for displayed estimates
    """
    if hasattr(pf, "estimate_all"):
        est = pf.estimate_all()
    else:
        est = pf.estimates()  # type: ignore[attr-defined]
    particle_positions: Dict[str, NDArray[np.float64]] = {}
    particle_weights: Dict[str, NDArray[np.float64]] = {}
    estimated_sources: Dict[str, NDArray[np.float64]] = {}
    estimated_strengths: Dict[str, NDArray[np.float64]] = {}

    mode = estimate_mode.lower()
    if mode not in {"mmse", "map"}:
        raise ValueError(f"Unknown estimate_mode: {estimate_mode}")

    for iso, filt in pf.filters.items():
        positions: list[NDArray[np.float64]] = []
        weights: list[float] = []
        cont_particles = getattr(filt, "continuous_particles", [])
        cont_weights = getattr(filt, "continuous_weights", np.zeros(0))
        if cont_particles and len(cont_weights) == len(cont_particles):
            for p, w in zip(cont_particles, cont_weights):
                if p.state.positions.size == 0:
                    continue
                for pos in p.state.positions:
                    positions.append(pos)
                    weights.append(float(w))
        particle_positions[iso] = np.vstack(positions) if positions else np.zeros((0, 3))
        particle_weights[iso] = np.asarray(weights, dtype=float)
        if cont_particles and len(cont_weights) == len(cont_particles):
            states = [p.state for p in cont_particles]
            weights_arr = np.asarray(cont_weights, dtype=float)
            use_clustered = bool(
                getattr(filt, "config", None)
                and getattr(filt.config, "birth_enable", False)
                and getattr(filt.config, "use_clustered_output", False)
            )
            if use_clustered and hasattr(filt, "estimate_clustered"):
                est_pos, est_str = filt.estimate_clustered()
                if min_est_strength is not None and est_str.size:
                    mask = est_str >= min_est_strength
                    est_pos = est_pos[mask]
                    est_str = est_str[mask]
            elif mode == "map":
                best = max(cont_particles, key=lambda p: p.log_weight).state
                best_r = int(getattr(best, "num_sources", 0))
                if best_r <= 0:
                    est_pos = np.zeros((0, 3), dtype=float)
                    est_str = np.zeros(0, dtype=float)
                else:
                    est_pos = best.positions[:best_r].copy()
                    est_str = best.strengths[:best_r].copy()
                exist_probs_full = (
                    _existence_probabilities(states, weights_arr, best_r)
                    if min_existence_prob is not None and best_r > 0
                    else np.zeros(0, dtype=float)
                )
                exist_probs = exist_probs_full[: est_str.shape[0]] if exist_probs_full.size else np.zeros(0)
                slot_valid = np.ones(len(est_str), dtype=bool)
                est_pos, est_str = _filter_estimates(
                    est_pos,
                    est_str,
                    slot_valid,
                    exist_probs,
                    min_strength=min_est_strength,
                    min_existence_prob=min_existence_prob,
                )
            else:
                max_r_all = max(int(getattr(st, "num_sources", 0)) for st in states) if states else 0
                est_pos, est_str, slot_valid = _mmse_estimate_by_slot(
                    states,
                    weights_arr,
                    max_r=max_r_all,
                )
                exist_probs_full = (
                    _existence_probabilities(states, weights_arr, max_r_all)
                    if min_existence_prob is not None and max_r_all > 0
                    else np.zeros(0, dtype=float)
                )
                exist_probs = exist_probs_full[: est_str.shape[0]] if exist_probs_full.size else np.zeros(0)
                est_pos, est_str = _filter_estimates(
                    est_pos,
                    est_str,
                    slot_valid,
                    exist_probs,
                    min_strength=min_est_strength,
                    min_existence_prob=min_existence_prob,
                )
            estimated_sources[iso] = est_pos
            estimated_strengths[iso] = est_str
        else:
            if iso in est:
                val = est[iso]
                if hasattr(val, "positions"):
                    est_pos = val.positions
                    est_str = val.strengths
                elif isinstance(val, tuple) and len(val) == 2:
                    est_pos = val[0]
                    est_str = val[1]
                else:
                    est_pos = np.zeros((0, 3))
                    est_str = np.zeros(0)
            else:
                est_pos = np.zeros((0, 3))
                est_str = np.zeros(0)
            if min_est_strength is not None and est_str.size:
                mask = est_str >= min_est_strength
                est_pos = est_pos[mask]
                est_str = est_str[mask]
            estimated_sources[iso] = est_pos
            estimated_strengths[iso] = est_str

    RFe = getattr(measurement, "RFe", np.eye(3))
    RPb = getattr(measurement, "RPb", np.eye(3))
    if getattr(measurement, "detector_position", None) is not None:
        robot_pos = np.asarray(measurement.detector_position, dtype=float)
    else:
        robot_pos = np.zeros(3)

    return PFFrame(
        step_index=step_index,
        time=time_sec,
        robot_position=robot_pos,
        robot_orientation=None,
        RFe=RFe,
        RPb=RPb,
        duration=measurement.live_time_s,
        counts_by_isotope=measurement.counts_by_isotope,
        particle_positions=particle_positions,
        particle_weights=particle_weights,
        estimated_sources=estimated_sources,
        estimated_strengths=estimated_strengths,
    )
