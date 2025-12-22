"""Data structures for capturing PF state per time step for visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any

from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


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


DEFAULT_ISOTOPE_COLORS = {
    "Cs-137": "tab:red",
    "Co-60": "tab:blue",
    "Eu-154": "tab:green",
    "Eu-155": "tab:green",
}


class RealTimePFVisualizer:
    """
    Simple matplotlib-based 3D visualizer for the PF state.

    - update(frame) redraws particles, estimates, counts, and label panel.
    - save_final(path) saves the current figure.
    - save_estimates_only(path) saves a view with only estimate markers visible.
    """

    def __init__(
        self,
        isotopes: List[str],
        world_bounds: Optional[Tuple[float, float, float, float, float, float]] = None,
        true_sources: Optional[Dict[str, NDArray[np.float64]]] = None,
        true_strengths: Optional[Dict[str, float]] = None,
        show_counts: bool = True,
    ) -> None:
        self.isotopes = isotopes
        self.world_bounds = world_bounds or (0, 10, 0, 10, 0, 3)
        self.true_sources = true_sources or {}
        self.true_strengths = true_strengths or {}
        self.show_counts = show_counts
        self.fig = plt.figure(figsize=(12, 6))
        if self.show_counts:
            layout = self.fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1])
            self.ax3d = self.fig.add_subplot(layout[:, 0], projection="3d")
            self.ax_counts = self.fig.add_subplot(layout[0, 1])
            self.ax_labels = self.fig.add_subplot(layout[1, 1])
        else:
            layout = self.fig.add_gridspec(1, 2, width_ratios=[3, 1])
            self.ax3d = self.fig.add_subplot(layout[0, 0], projection="3d")
            self.ax_counts = None
            self.ax_labels = self.fig.add_subplot(layout[0, 1])
        cmap = plt.get_cmap("tab10")
        self.colors = {}
        for i, iso in enumerate(isotopes):
            if iso in DEFAULT_ISOTOPE_COLORS:
                self.colors[iso] = DEFAULT_ISOTOPE_COLORS[iso]
            else:
                self.colors[iso] = cmap(i % 10)
        self._init_axes()
        self._init_label_axis()
        plt.tight_layout()
        self._particle_artists: Dict[str, Any] = {}
        self._est_artists: Dict[str, Any] = {}
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
        self._particle_size_range = (0.6, 18.0)
        self._particle_alpha_range = (0.05, 0.95)
        self._particle_weight_exponent = 1.0
        self._projection_linewidth = 1.8
        self.estimate_colors = {}
        # Plot true sources once if provided (as legend entries)
        for iso, pos in self.true_sources.items():
            if pos.size:
                strength = self.true_strengths.get(iso, None)
                label = f"True {iso}"
                if strength is not None:
                    label = f"{label} pos={pos.round(2).tolist()} q={strength:.1f} cps@1m"
                art = self.ax3d.scatter(
                    pos[:, 0],
                    pos[:, 1],
                    pos[:, 2],
                    marker="*",
                    s=100,
                    color=self.colors.get(iso, "black"),
                    label=label,
                )
                self._true_artists.append(art)
                self._true_projection_artists.extend(self._axis_projection_lines(pos, self.colors.get(iso, "black")))
        for iso in self.isotopes:
            self.estimate_colors[iso] = self._estimate_color(self.colors.get(iso, "black"))

    def _init_axes(self) -> None:
        xmin, xmax, ymin, ymax, zmin, zmax = self.world_bounds
        self.ax3d.set_xlim(xmin, xmax)
        self.ax3d.set_ylim(ymin, ymax)
        self.ax3d.set_zlim(zmin, zmax)
        self.ax3d.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))
        self.ax3d.set_xlabel("x [m]")
        self.ax3d.set_ylabel("y [m]")
        self.ax3d.set_zlabel("z [m]")
        if self.ax_counts is not None:
            self.ax_counts.set_ylabel("Counts")
            self.ax_counts.set_title("Isotope-wise counts")
        # Draw wireframe cube for bounds
        xs = [xmin, xmax]
        ys = [ymin, ymax]
        zs = [zmin, zmax]
        for z in zs:
            X, Y = np.meshgrid(xs, ys)
            Z = np.full_like(X, z)
            self.ax3d.plot_wireframe(X, Y, Z, color="gray", alpha=0.2)
        for x in xs:
            X, Z = np.meshgrid([x], zs)
            Y = np.array([[ymin, ymin], [ymax, ymax]])
            self.ax3d.plot_wireframe(X, Y, Z, color="gray", alpha=0.2)
        for y in ys:
            Y, Z = np.meshgrid([y], zs)
            X = np.array([[xmin, xmin], [xmax, xmax]])
            self.ax3d.plot_wireframe(X, Y, Z, color="gray", alpha=0.2)

    def _init_label_axis(self) -> None:
        """Initialize the label panel axis."""
        if self.ax_labels is None:
            return
        self.ax_labels.set_title("Legend / Estimates")
        self.ax_labels.axis("off")

    def _legend_lines(self) -> List[Tuple[str, str, str, str]]:
        """Build legend-style label lines with matching colors and markers."""
        lines: List[Tuple[str, str, str, str]] = []
        for iso, pos in self.true_sources.items():
            if pos.size:
                strength = self.true_strengths.get(iso, None)
                label = f"True {iso}"
                if strength is not None:
                    label = f"{label} pos={pos.round(2).tolist()} q={strength:.1f} cps@1m"
                lines.append((label, self.colors.get(iso, "black"), "*", "None"))
        lines.append(("trajectory", "cyan", "o", "-"))
        lines.append(("robot", "cyan", "o", "None"))
        for iso in self.isotopes:
            color = self.colors.get(iso, "black")
            lines.append((f"{iso} particles", color, ".", "None"))
            lines.append((f"{iso} est", self.estimate_colors.get(iso, color), "x", "None"))
        lines.append(("Fe shield", "magenta", ">", "None"))
        lines.append(("Pb shield", "green", "<", "None"))
        return lines

    def _legend_lines_estimates_only(self) -> List[Tuple[str, str, str, str]]:
        """Build legend lines for the estimates-only view."""
        lines: List[Tuple[str, str, str, str]] = []
        for iso, pos in self.true_sources.items():
            if pos.size:
                strength = self.true_strengths.get(iso, None)
                label = f"True {iso}"
                if strength is not None:
                    label = f"{label} pos={pos.round(2).tolist()} q={strength:.1f} cps@1m"
                lines.append((label, self.colors.get(iso, "black"), "*", "None"))
        lines.append(("trajectory", "cyan", "o", "-"))
        lines.append(("robot", "cyan", "o", "None"))
        for iso in self.isotopes:
            color = self.estimate_colors.get(iso, self.colors.get(iso, "black"))
            lines.append((f"{iso} est", color, "x", "None"))
        return lines

    def _estimate_lines(self, frame: PFFrame) -> List[Tuple[str, str]]:
        """Build estimate text lines for the strongest source per isotope."""
        lines: List[Tuple[str, str]] = []
        for iso in self.isotopes:
            est_pos = frame.estimated_sources.get(iso, np.zeros((0, 3)))
            strengths = frame.estimated_strengths.get(iso, np.zeros(0))
            if strengths.size and est_pos.size:
                idx = int(np.argmax(strengths))
                pos = est_pos[idx]
                strength = float(strengths[idx])
                text = f"{iso}: pos={pos.round(2).tolist()} q={strength:.1f} cps@1m"
            else:
                text = f"{iso}: no estimate"
            lines.append((text, self.estimate_colors.get(iso, self.colors.get(iso, "black"))))
        return lines

    def _estimate_lines_all(self, frame: PFFrame) -> List[Tuple[str, str]]:
        """Build estimate text lines for all sources per isotope."""
        lines: List[Tuple[str, str]] = []
        for iso in self.isotopes:
            est_pos = frame.estimated_sources.get(iso, np.zeros((0, 3)))
            strengths = frame.estimated_strengths.get(iso, np.zeros(0))
            color = self.estimate_colors.get(iso, self.colors.get(iso, "black"))
            if strengths.size and est_pos.size:
                for idx, (pos, strength) in enumerate(zip(est_pos, strengths)):
                    text = f"{iso}[{idx}]: pos={pos.round(2).tolist()} q={float(strength):.1f} cps@1m"
                    lines.append((text, color))
            else:
                lines.append((f"{iso}: no estimate", color))
        return lines

    def _estimate_color(self, base_color: str) -> Tuple[float, float, float]:
        """Return a lighter variant of the base color for estimate markers."""
        rgb = np.array(mcolors.to_rgb(base_color))
        hsv = mcolors.rgb_to_hsv(rgb)
        hsv[0] = (hsv[0] + 0.12) % 1.0
        hsv[1] = min(1.0, hsv[1] * 0.7 + 0.3)
        hsv[2] = min(1.0, hsv[2] * 0.8 + 0.2)
        return tuple(mcolors.hsv_to_rgb(hsv))

    def _axis_projection_lines(
        self,
        points: NDArray[np.float64],
        color: str,
        alpha: float = 0.35,
    ) -> list:
        """Draw thin dotted projection lines from points to each axis."""
        if points.size == 0:
            return []
        xmin, _, ymin, _, zmin, _ = self.world_bounds
        artists: list = []
        for x, y, z in points:
            artists.append(
                self.ax3d.plot(
                    [x, x],
                    [y, ymin],
                    [z, zmin],
                    linestyle=":",
                    linewidth=self._projection_linewidth,
                    color=color,
                    alpha=alpha,
                )[0]
            )
            artists.append(
                self.ax3d.plot(
                    [x, xmin],
                    [y, y],
                    [z, zmin],
                    linestyle=":",
                    linewidth=self._projection_linewidth,
                    color=color,
                    alpha=alpha,
                )[0]
            )
            artists.append(
                self.ax3d.plot(
                    [x, xmin],
                    [y, ymin],
                    [z, z],
                    linestyle=":",
                    linewidth=self._projection_linewidth,
                    color=color,
                    alpha=alpha,
                )[0]
            )
        return artists

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
        self.ax_labels.set_title("Legend / Estimates")
        self.ax_labels.axis("off")
        legend_lines = self._legend_lines() if legend_lines is None else legend_lines
        estimate_lines = self._estimate_lines(frame) if estimate_lines is None else estimate_lines
        gap_lines = 1
        total_lines = len(legend_lines) + len(estimate_lines) + 2 + gap_lines
        line_height = 0.95 / max(total_lines, 1)
        y = 0.98
        self.ax_labels.text(
            0.0,
            y,
            "Legend",
            transform=self.ax_labels.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            fontweight="bold",
            color="black",
        )
        y -= line_height
        for text, color, marker, linestyle in legend_lines:
            self.ax_labels.text(
                0.1,
                y,
                text,
                transform=self.ax_labels.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                color=color,
            )
            if linestyle != "None":
                self.ax_labels.plot(
                    [0.02, 0.06],
                    [y - 0.005, y - 0.005],
                    transform=self.ax_labels.transAxes,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=6,
                    linewidth=1.0,
                )
            else:
                self.ax_labels.plot(
                    [0.03],
                    [y - 0.005],
                    transform=self.ax_labels.transAxes,
                    color=color,
                    linestyle="None",
                    marker=marker,
                    markersize=6,
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
            fontsize=9,
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
                fontsize=8,
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
        # maintain trajectory history
        self._traj_history.append(frame.robot_position)
        # Robot and trajectory
        traj_arr = np.vstack(self._traj_history)
        if self._traj_line is None:
            (self._traj_line,) = self.ax3d.plot(traj_arr[:, 0], traj_arr[:, 1], traj_arr[:, 2], "-o", color="cyan", label="trajectory")
        else:
            self._traj_line.set_data(traj_arr[:, 0], traj_arr[:, 1])
            self._traj_line.set_3d_properties(traj_arr[:, 2])
        if self._robot_artist is None:
            self._robot_artist = self.ax3d.scatter(
                frame.robot_position[0], frame.robot_position[1], frame.robot_position[2], color="cyan", marker="o", s=80, label="robot"
            )
        else:
            self._robot_artist._offsets3d = (
                np.array([frame.robot_position[0]]),
                np.array([frame.robot_position[1]]),
                np.array([frame.robot_position[2]]),
            )
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
                        zorder=1,
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
                        zorder=10,
                    )
            else:
                art = self._est_artists[iso]
                if est_pos.size:
                    art._offsets3d = (est_pos[:, 0], est_pos[:, 1], est_pos[:, 2])
                    art.set_color(est_color)
                    art.set_zorder(10)
                else:
                    art._offsets3d = ([], [], [])
        for art in self._projection_artists:
            art.remove()
        self._projection_artists = []
        for iso in self.isotopes:
            est_pos = frame.estimated_sources.get(iso, np.zeros((0, 3)))
            est_color = self.estimate_colors.get(iso, self.colors.get(iso, "black"))
            self._projection_artists.extend(self._axis_projection_lines(est_pos, est_color))
        self.ax3d.set_title(f"Step {frame.step_index} t={frame.time:.2f}s")
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
) -> PFFrame:
    """
    Construct a PFFrame snapshot from a PF (ParallelIsotopePF-like) and a Measurement.

    Args:
        pf: ParallelIsotopePF or compatible with .filters and .estimate_all()
        measurement: Measurement with counts_by_isotope, pose_idx/orient_idx/RFe/RPb (optional)
        step_index: integer step
        time_sec: cumulative time in seconds
    """
    if hasattr(pf, "estimate_all"):
        est = pf.estimate_all()
    else:
        est = pf.estimates()  # type: ignore[attr-defined]
    particle_positions: Dict[str, NDArray[np.float64]] = {}
    particle_weights: Dict[str, NDArray[np.float64]] = {}
    estimated_sources: Dict[str, NDArray[np.float64]] = {}
    estimated_strengths: Dict[str, NDArray[np.float64]] = {}

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
        if iso in est:
            val = est[iso]
            if hasattr(val, "positions"):
                estimated_sources[iso] = val.positions
                estimated_strengths[iso] = val.strengths
            elif isinstance(val, tuple) and len(val) == 2:
                estimated_sources[iso] = val[0]
                estimated_strengths[iso] = val[1]
            else:
                estimated_sources[iso] = np.zeros((0, 3))
                estimated_strengths[iso] = np.zeros(0)
        else:
            estimated_sources[iso] = np.zeros((0, 3))
            estimated_strengths[iso] = np.zeros(0)

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
