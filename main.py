"""Real-time demo for the rotating-shield particle filter with visualization.

Run `python main.py` to simulate a robot moving through a simple environment, measuring
isotope-wise counts from synthetic spectra (Chapter 2.5.7) with rotating 1/8-shell
shields, updating the PF (Chapter 3), and visualizing the particle clouds and estimates.

Notes:
- PF observations are always the isotope-wise counts from spectrum unfolding (no direct
  geometric shortcuts).
- The PF algorithm itself is unchanged; this script focuses on live visualization and
  playback. Final snapshots are saved to results/result_pf.png (PF) and
  results/result_spectrum.png (final spectrum).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Ensure src/ is on sys.path for direct script execution.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RESULTS_DIR = ROOT / "results"

from measurement.model import EnvironmentConfig, PointSource
from measurement.shielding import generate_octant_orientations, generate_octant_rotation_matrices
from measurement.kernels import ShieldParams
from spectrum.library import Nuclide
from spectrum.peak_detection import detect_peaks
from spectrum.pipeline import SpectralDecomposer
from pf.parallel import Measurement
from pf.estimator import RotatingShieldPFEstimator, RotatingShieldPFConfig
from planning.shield_rotation import select_best_orientation, select_top_k_orientations
from planning.pose_selection import select_next_pose
from visualization.realtime_viz import DEFAULT_ISOTOPE_COLORS, RealTimePFVisualizer, build_frame_from_pf


def _build_demo_sources() -> list[PointSource]:
    """Define a small set of synthetic sources inside the environment."""
    return [
        PointSource("Cs-137", position=(5.0, 10.0, 5.0), intensity_cps_1m=50000.0),
        PointSource("Co-60", position=(2.0, 15.0, 7.0), intensity_cps_1m=20000.0),
        PointSource("Eu-154", position=(7.0, 5.0, 3.0), intensity_cps_1m=30000.0),
    ]


def _build_isotope_colors(isotopes: list[str]) -> dict[str, str]:
    """Return a consistent color mapping for isotope-specific plots."""
    cmap = plt.get_cmap("tab10")
    colors: dict[str, str] = {}
    for i, iso in enumerate(isotopes):
        if iso in DEFAULT_ISOTOPE_COLORS:
            colors[iso] = DEFAULT_ISOTOPE_COLORS[iso]
        else:
            colors[iso] = cmap(i % 10)
    return colors


def _assign_peak_indices(
    energy_axis: np.ndarray,
    peak_indices: np.ndarray,
    library: dict[str, Nuclide],
    tolerance_keV: float,
) -> tuple[dict[str, list[int]], list[int]]:
    """Assign detected peak indices to isotopes based on closest library lines."""
    peaks_by_iso: dict[str, list[int]] = {iso: [] for iso in library}
    unassigned: list[int] = []
    line_energies = {
        iso: np.array([line.energy_keV for line in nuclide.lines], dtype=float)
        for iso, nuclide in library.items()
    }
    for idx in peak_indices:
        energy = float(energy_axis[int(idx)])
        best_iso = None
        best_diff = float("inf")
        for iso, energies in line_energies.items():
            if energies.size == 0:
                continue
            diff = float(np.min(np.abs(energies - energy)))
            if diff < best_diff:
                best_diff = diff
                best_iso = iso
        if best_iso is not None and best_diff <= tolerance_keV:
            peaks_by_iso[best_iso].append(int(idx))
        else:
            unassigned.append(int(idx))
    return peaks_by_iso, unassigned


def _save_spectrum_plot(
    decomposer: SpectralDecomposer,
    spectrum: np.ndarray,
    output_path: Path,
    peak_tolerance_keV: float = 10.0,
) -> None:
    """Save the final measurement spectrum with nuclide lines and colored peaks."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    energy_axis = decomposer.energy_axis
    library = decomposer.library
    colors = _build_isotope_colors(list(library.keys()))
    corrected = decomposer.preprocess(spectrum)
    peak_indices = detect_peaks(corrected, prominence=0.05, distance=5)
    peaks_by_iso, unassigned = _assign_peak_indices(
        energy_axis,
        peak_indices,
        library,
        tolerance_keV=peak_tolerance_keV,
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(energy_axis, spectrum, color="black", linewidth=1.0, label="Spectrum")
    for iso, nuclide in library.items():
        color = colors.get(iso, "gray")
        labeled = False
        for line in nuclide.lines:
            label = iso if not labeled else None
            ax.axvline(
                line.energy_keV,
                color=color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
                label=label,
            )
            labeled = True
    for iso, idxs in peaks_by_iso.items():
        if idxs:
            ax.scatter(energy_axis[idxs], spectrum[idxs], color=colors.get(iso, "gray"), s=28, zorder=3)
    if unassigned:
        ax.scatter(energy_axis[unassigned], spectrum[unassigned], color="gray", s=20, zorder=3, alpha=0.6)
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")
    ax.set_title("Final measurement spectrum")
    ax.grid(True, alpha=0.3)
    if library:
        ax.legend(loc="upper right", fontsize=8, title="Nuclide lines")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_live_pf(live: bool = True, steps: int = 24, all_orientations: bool = False) -> None:
    """Run a simple PF loop with live visualization (active pose/orientation selection)."""
    env = EnvironmentConfig(size_x=10.0, size_y=20.0, size_z=10.0, detector_position=(1.0, 1.0, 0.5))
    sources = [
        PointSource("Cs-137", position=(5.0, 10.0, 5.0), intensity_cps_1m=50000.0),
        PointSource("Co-60", position=(2.0, 15.0, 7.0), intensity_cps_1m=20000.0),
        PointSource("Eu-154", position=(7.0, 5.0, 3.0), intensity_cps_1m=30000.0),
    ]
    decomposer = SpectralDecomposer()
    normals = generate_octant_orientations()
    rot_mats = generate_octant_rotation_matrices()
    num_orients = len(rot_mats)

    # Candidate sources: coarse grid inside environment
    xs_src = np.linspace(0.5, env.size_x - 0.5, 4)
    ys_src = np.linspace(0.5, env.size_y - 0.5, 4)
    zs_src = np.linspace(0.5, env.size_z - 0.5, 2)
    grid = np.array([[x, y, z] for x in xs_src for y in ys_src for z in zs_src], dtype=float)

    # Candidate measurement poses (z fixed at 0.5); PF selects next pose actively.
    xs_pose = np.linspace(1.0, env.size_x - 1.0, 5)
    ys_pose = np.linspace(2.0, env.size_y - 2.0, 6)
    poses_arr = np.array([[x, y, 0.5] for x in xs_pose for y in ys_pose], dtype=float)

    isotopes = ["Cs-137", "Co-60", "Eu-154"]
    # Use a moderate particle count for the demo (previous default was 200)
    num_particles = 2000
    shield_params = ShieldParams()
    mu_by_isotope = {
        iso: {"fe": shield_params.mu_fe, "pb": shield_params.mu_pb} for iso in isotopes
    }
    pf_conf = RotatingShieldPFConfig(
        num_particles=num_particles,
        resample_threshold=0.5,
        min_strength=0.01,
        p_birth=0.05,
        short_time_s=30.0,
        position_min=(0.0, 0.0, 0.0),
        position_max=(env.size_x, env.size_y, env.size_z),
    )
    estimator = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=grid,
        shield_normals=normals,
        mu_by_isotope=mu_by_isotope,
        pf_config=pf_conf,
    )
    # Register all candidate poses
    for pose in poses_arr:
        estimator.add_measurement_pose(pose)

    # Build true sources dict for visualization
    true_src = {}
    true_strengths = {}
    for iso in isotopes:
        positions = [np.array(src.position, dtype=float) for src in sources if src.isotope == iso]
        strengths = [src.intensity_cps_1m for src in sources if src.isotope == iso]
        if positions:
            true_src[iso] = np.vstack(positions)
        if strengths:
            true_strengths[iso] = float(np.max(strengths))
    viz = RealTimePFVisualizer(
        isotopes=isotopes,
        world_bounds=(0, env.size_x, 0, env.size_y, 0, env.size_z),
        true_sources=true_src,
        true_strengths=true_strengths,
        show_counts=False,
    )
    if live:
        plt.ion()
        plt.show(block=False)
        plt.pause(0.1)

    elapsed = 0.0
    step_counter = 0
    live_time = 30.0
    current_pose_idx = 0
    recent_poses: list[int] = []
    unvisited: set[int] = set(range(len(poses_arr)))
    total_pairs = num_orients * num_orients
    min_steps = 21  # ensure we exceed 20 measurements before stopping
    min_unique_poses = 21  # require >20 unique measurement locations
    target_steps = max(steps, min_steps)
    visited_poses: set[int] = set()
    last_spectrum: np.ndarray | None = None
    while (step_counter < target_steps) or (len(visited_poses) < min_unique_poses):
        pose = poses_arr[current_pose_idx]
        available_orients = list(range(total_pairs))
        # Decide which orientation pairs to evaluate at this pose
        if all_orientations:
            top_pairs = available_orients  # all 64 pairs
        else:
            top_pairs = select_top_k_orientations(
                estimator,
                pose_idx=current_pose_idx,
                live_time_s=live_time,
                allowed_indices=available_orients,
            )
        # Rotate shield for each selected pair
        for best_pair_idx in top_pairs:
            if (step_counter >= target_steps) and (len(visited_poses) >= min_unique_poses):
                break
            fe_idx = best_pair_idx // num_orients
            pb_idx = best_pair_idx % num_orients
            RFe_sel = rot_mats[fe_idx]
            RPb_sel = rot_mats[pb_idx]
            env_step = EnvironmentConfig(detector_position=tuple(pose))
            spectrum, _ = decomposer.simulate_spectrum(
                sources=sources,
                environment=env_step,
                acquisition_time=live_time,
                rng=np.random.default_rng(123 + step_counter),
                fe_shield_orientation=normals[fe_idx],
                pb_shield_orientation=normals[pb_idx],
                mu_by_isotope=mu_by_isotope,
                shield_params=shield_params,
            )
            last_spectrum = spectrum.copy()
            z_k = decomposer.isotope_counts(spectrum)
            meas = Measurement(
                counts_by_isotope=z_k,
                pose_idx=current_pose_idx,
                orient_idx=best_pair_idx,
                live_time_s=live_time,
                fe_index=fe_idx,
                pb_index=pb_idx,
                RFe=RFe_sel,
                RPb=RPb_sel,
                detector_position=pose,
            )
            estimator.update_pair(z_k=z_k, pose_idx=current_pose_idx, fe_index=fe_idx, pb_index=pb_idx, live_time_s=live_time)
            elapsed += live_time
            frame = build_frame_from_pf(estimator, meas, step_index=step_counter, time_sec=elapsed)
            viz.update(frame)
            rfe_dir = RFe_sel[:, 2].tolist()
            rpb_dir = RPb_sel[:, 2].tolist()
            # Log only measurement point and shield orientations
            print(
                f"[step {step_counter}] pose={pose.tolist()} orient_pair={best_pair_idx} "
                f"fe_idx={fe_idx} pb_idx={pb_idx} RFe_dir={rfe_dir} RPb_dir={rpb_dir}"
            )
            if live:
                plt.pause(0.05)
            step_counter += 1
            visited_poses.add(current_pose_idx)
            unvisited.discard(current_pose_idx)
            # Convergence check per Chapter 3.6 (change + uncertainty + IG thresholds)
            if (
                step_counter >= target_steps
                and len(visited_poses) >= min_unique_poses
                and estimator.should_stop_shield_rotation(
                    pose_idx=current_pose_idx,
                    ig_threshold=estimator.pf_config.ig_threshold,
                    fisher_threshold=1e-3,
                    change_tol=1e-2,
                    uncertainty_tol=1e-3,
                    live_time_s=live_time,
                )
            ):
                break
        if (step_counter >= target_steps) and (len(visited_poses) >= min_unique_poses):
            # Optional convergence check; if not converged, still stop to avoid infinite loop
            if estimator.should_stop_exploration(
                ig_threshold=estimator.pf_config.ig_threshold, uncertainty_tol=1e-3, change_tol=1e-2
            ):
                print("Converged; stopping exploration.")
            else:
                print("Stopping after reaching target steps/poses (convergence not met).")
            break
        cand_idx = np.array(list(unvisited)) if unvisited else np.arange(len(poses_arr))
        if cand_idx.size == 0:
            cand_idx = np.arange(len(poses_arr))
        current_pose_idx = select_next_pose(
            estimator=estimator,
            candidate_pose_indices=cand_idx,
            current_pose_idx=current_pose_idx,
        )
        recent_poses.append(current_pose_idx)

    # Save final snapshots
    pf_out_path = RESULTS_DIR / "result_pf.png"
    spectrum_out_path = RESULTS_DIR / "result_spectrum.png"
    estimates_out_path = RESULTS_DIR / "result_estimates.png"
    pf_out_path.parent.mkdir(parents=True, exist_ok=True)
    viz.save_final(pf_out_path.as_posix())
    viz.save_estimates_only(estimates_out_path.as_posix())
    if last_spectrum is not None:
        _save_spectrum_plot(decomposer, last_spectrum, spectrum_out_path)
    total_meas_time = step_counter * live_time
    print(f"Final PF visualization saved to: {pf_out_path}")
    print(f"Final estimates-only visualization saved to: {estimates_out_path}")
    if last_spectrum is not None:
        print(f"Final spectrum saved to: {spectrum_out_path}")
    print(f"Total measurements: {step_counter}, total live time (simulated): {total_meas_time:.1f} s")
    if live:
        plt.ioff()
        plt.pause(0.1)
    plt.close("all")


def run_realtime_pf() -> None:
    """Entry point for real-time PF + visualization with built-in demo settings."""
    run_live_pf(live=True, steps=10)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time rotating-shield PF visualization demo.")
    parser.add_argument("--steps", type=int, default=24, help="Number of measurement steps to simulate.")
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="Disable interactive updating (still saves results/result_pf.png and results/result_spectrum.png).",
    )
    parser.add_argument(
        "--all-orientations", action="store_true", help="Measure all 64 Fe/Pb orientation pairs at each pose."
    )
    args = parser.parse_args()
    # For now scenario loading is a placeholder; default to built-in demo
    run_live_pf(live=not args.no_live, steps=args.steps, all_orientations=args.all_orientations)


if __name__ == "__main__":
    main()
