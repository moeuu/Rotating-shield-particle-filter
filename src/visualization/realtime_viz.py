"""Data structures for capturing PF state per time step for visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


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
    - particle_positions: isotope -> (N_particles, 3)
    - particle_weights: isotope -> (N_particles,)
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
    "Eu-155": "tab:green",
}


class RealTimePFVisualizer:
    """
    Simple matplotlib-based 3D visualizer for the PF state.

    - update(frame) redraws particles, estimates, and counts for the given PFFrame.
    - save_final(path) saves the current figure.
    """

    def __init__(
        self,
        isotopes: List[str],
        world_bounds: Optional[Tuple[float, float, float, float, float, float]] = None,
        true_sources: Optional[Dict[str, NDArray[np.float64]]] = None,
        show_counts: bool = True,
    ) -> None:
        self.isotopes = isotopes
        self.world_bounds = world_bounds or (0, 10, 0, 10, 0, 3)
        self.true_sources = true_sources or {}
        self.show_counts = show_counts
        self.fig = plt.figure(figsize=(10, 6))
        if self.show_counts:
            self.ax3d = self.fig.add_subplot(121, projection="3d")
            self.ax_counts = self.fig.add_subplot(122)
        else:
            self.ax3d = self.fig.add_subplot(111, projection="3d")
            self.ax_counts = None
        cmap = plt.get_cmap("tab10")
        self.colors = {}
        for i, iso in enumerate(isotopes):
            if iso in DEFAULT_ISOTOPE_COLORS:
                self.colors[iso] = DEFAULT_ISOTOPE_COLORS[iso]
            else:
                self.colors[iso] = cmap(i % 10)
        self._init_axes()
        plt.tight_layout()
        self._particle_artists: Dict[str, any] = {}
        self._est_artists: Dict[str, any] = {}
        self._robot_artist = None
        self._traj_line = None
        self._shield_arrows: Dict[str, any] = {}
        self._counts_bars = None
        # Pre-create bar containers with zeros
        if self.ax_counts is not None and self.isotopes:
            zeros = [0.0 for _ in self.isotopes]
            self._counts_bars = self.ax_counts.bar(self.isotopes, zeros, color=[self.colors.get(n, "gray") for n in self.isotopes])
        self._traj_history: list[NDArray[np.float64]] = []
        self._last_frame: PFFrame | None = None
        self._true_artists: list = []
        # Plot true sources once if provided
        for iso, pos in self.true_sources.items():
            if pos.size:
                art = self.ax3d.scatter(pos[:, 0], pos[:, 1], pos[:, 2], marker="*", s=100, color=self.colors.get(iso, "black"), label=f"True {iso}")
                self._true_artists.append(art)

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
            color = self.colors.get(iso, None)
            if iso not in self._particle_artists:
                if pts.size:
                    self._particle_artists[iso] = self.ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5, alpha=0.15, color=color, label=f"{iso} particles")
            else:
                art = self._particle_artists[iso]
                if pts.size:
                    art._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
                else:
                    art._offsets3d = ([], [], [])
            est_pos = frame.estimated_sources.get(iso, np.zeros((0, 3)))
            if iso not in self._est_artists:
                if est_pos.size:
                    self._est_artists[iso] = self.ax3d.scatter(
                        est_pos[:, 0], est_pos[:, 1], est_pos[:, 2], marker="x", s=80, color=color, label=f"{iso} est"
                    )
            else:
                art = self._est_artists[iso]
                if est_pos.size:
                    art._offsets3d = (est_pos[:, 0], est_pos[:, 1], est_pos[:, 2])
                else:
                    art._offsets3d = ([], [], [])
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
        self.ax3d.legend(loc="upper right", fontsize=8)
        self.fig.canvas.draw_idle()

    def save_final(self, path: str = "result.png") -> None:
        """Save the current figure."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # If we have a last frame, ensure markers are up to date
        if self._last_frame is not None:
            self.update(self._last_frame)
        # Annotate estimated sources with isotope/strength
        if self._last_frame is not None:
            for iso in self.isotopes:
                est_pos = self._last_frame.estimated_sources.get(iso, np.zeros((0, 3)))
                est_str = self._last_frame.estimated_strengths.get(iso, np.zeros(0))
                for pos, s in zip(est_pos, est_str):
                    self.ax3d.text(pos[0], pos[1], pos[2], f"{iso} {s:.1f}", color=self.colors.get(iso, "black"), fontsize=8)
        self.fig.savefig(out, dpi=200)


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
        positions = [p.state.positions for p in getattr(filt, "continuous_particles", []) if p.state.positions.size]
        particle_positions[iso] = np.vstack(positions) if positions else np.zeros((0, 3))
        particle_weights[iso] = getattr(filt, "continuous_weights", np.zeros(0))
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
