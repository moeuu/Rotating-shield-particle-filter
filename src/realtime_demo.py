"""Real-time demo for the rotating-shield particle filter with visualization."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib


def _configure_matplotlib() -> None:
    """Configure matplotlib backend for interactive or headless use."""
    headless = "--headless" in sys.argv
    if headless:
        matplotlib.use("Agg")
    else:
        try:
            matplotlib.use("TkAgg")
        except Exception:
            matplotlib.use("Agg")


_configure_matplotlib()

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from measurement.model import EnvironmentConfig, PointSource
from measurement.shielding import (
    HVL_TVL_TABLE_MM,
    generate_octant_orientations,
    generate_octant_rotation_matrices,
    mu_by_isotope_from_tvl_mm,
)
from measurement.kernels import ShieldParams
from spectrum.library import Nuclide
from spectrum.peak_detection import detect_peaks
from spectrum.pipeline import SpectralDecomposer
from pf.parallel import Measurement
from pf.estimator import RotatingShieldPFEstimator, RotatingShieldPFConfig
from planning.candidate_generation import generate_candidate_poses
from planning.pose_selection import select_next_pose_from_candidates
from planning.shield_rotation import select_best_orientation
from visualization.realtime_viz import DEFAULT_ISOTOPE_COLORS, RealTimePFVisualizer, build_frame_from_pf

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
SPECTRUM_DIR = RESULTS_DIR / "spetrum"
PRUNE_INTERVAL = 10
PRUNE_MIN_STRENGTH_ABS = 5.0
PRUNE_MIN_STRENGTH_RATIO = 0.001
PRUNE_TAU_MIX = 0.6
DEFAULT_SOURCE_CONFIG = ROOT / "source_layouts" / "demo_sources.json"


def _build_demo_sources() -> list[PointSource]:
    """Define a small set of synthetic sources inside the environment."""
    return [
        PointSource("Cs-137", position=(5.0, 10.0, 5.0), intensity_cps_1m=50000.0),
        PointSource("Co-60", position=(2.0, 15.0, 7.0), intensity_cps_1m=20000.0),
        PointSource("Eu-154", position=(7.0, 5.0, 3.0), intensity_cps_1m=30000.0),
    ]


def load_sources_from_json(path: Path) -> list[PointSource]:
    """Load point sources from a JSON configuration file."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        entries = data.get("sources", [])
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError("Source config must be a list or include a 'sources' list.")
    if not isinstance(entries, list):
        raise ValueError("Source config 'sources' must be a list.")
    sources: list[PointSource] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Source entry {idx} must be an object.")
        isotope = entry.get("isotope")
        position = entry.get("position")
        intensity = entry.get("intensity_cps_1m")
        if intensity is None:
            intensity = entry.get("strength_cps_1m")
        if intensity is None:
            intensity = entry.get("intensity")
        if isotope is None or position is None or intensity is None:
            raise ValueError(
                "Each source must include 'isotope', 'position', and 'intensity_cps_1m'."
            )
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            raise ValueError(f"Source entry {idx} position must be a 3-element list.")
        sources.append(
            PointSource(
                isotope=str(isotope),
                position=(float(position[0]), float(position[1]), float(position[2])),
                intensity_cps_1m=float(intensity),
            )
        )
    return sources


def _update_detection_counts(
    candidates: set[str],
    detect_counts: dict[str, int],
    consecutive: int,
) -> set[str]:
    """
    Update consecutive-detection counters and return isotopes that meet the streak.
    """
    detected: set[str] = set()
    for iso in detect_counts:
        if iso in candidates:
            detect_counts[iso] += 1
        else:
            detect_counts[iso] = 0
        if detect_counts[iso] >= consecutive:
            detected.add(iso)
    return detected


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


def _default_use_gpu() -> bool:
    """Return True if CUDA is available for torch acceleration."""
    try:
        from pf import gpu_utils
    except ImportError:
        return False
    return gpu_utils.torch_available()


def _save_spectrum_plot(
    decomposer: SpectralDecomposer,
    spectrum: np.ndarray,
    output_path: Path,
    peak_tolerance_keV: float = 10.0,
    highlight_isotopes: set[str] | None = None,
) -> None:
    """Save the final measurement spectrum with nuclide lines and colored peaks."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    energy_axis = decomposer.energy_axis
    library = decomposer.library
    if highlight_isotopes is not None:
        library = {iso: library[iso] for iso in library if iso in highlight_isotopes}
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
        if highlight_isotopes is not None and iso not in highlight_isotopes:
            continue
        if idxs:
            ax.scatter(energy_axis[idxs], spectrum[idxs], color=colors.get(iso, "gray"), s=28, zorder=3)
    if unassigned and highlight_isotopes is None:
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


def run_live_pf(
    live: bool = True,
    max_steps: int | None = None,
    all_orientations: bool = False,
    sources: list[PointSource] | None = None,
    detect_threshold_abs: float = 0.1,
    detect_threshold_rel: float = 0.2,
    detect_consecutive: int = 2,
    detect_min_steps: int | None = None,
) -> None:
    """
    Run a simple PF loop with live visualization (active pose/orientation selection).

    If max_steps is None, run until the information-gain threshold is met.
    """
    env = EnvironmentConfig(size_x=10.0, size_y=20.0, size_z=10.0, detector_position=(1.0, 1.0, 0.5))
    sources = _build_demo_sources() if sources is None else sources
    decomposer = SpectralDecomposer()
    normals = generate_octant_orientations()
    rot_mats = generate_octant_rotation_matrices()
    num_orients = len(rot_mats)

    # Candidate sources: coarse grid inside environment
    xs_src = np.linspace(0.5, env.size_x - 0.5, 4)
    ys_src = np.linspace(0.5, env.size_y - 0.5, 4)
    zs_src = np.linspace(0.5, env.size_z - 0.5, 2)
    grid = np.array([[x, y, z] for x in xs_src for y in ys_src for z in zs_src], dtype=float)

    bounds_lo = np.array([0.0, 0.0, env.detector_position[2]], dtype=float)
    bounds_hi = np.array([env.size_x, env.size_y, env.detector_position[2]], dtype=float)

    isotopes = list(decomposer.isotope_names)
    detect_min_steps = detect_consecutive if detect_min_steps is None else detect_min_steps
    detect_counts = {iso: 0 for iso in isotopes}
    detected_isotopes: set[str] = set()
    detection_locked = False
    last_candidates: set[str] = set()
    # Use a moderate particle count for the demo (previous default was 200)
    num_particles = 2000
    shield_params = ShieldParams()
    mu_by_isotope = mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM, isotopes=isotopes)
    if not mu_by_isotope:
        mu_by_isotope = {
            iso: {"fe": shield_params.mu_fe, "pb": shield_params.mu_pb} for iso in isotopes
        }
    use_gpu = _default_use_gpu()
    pf_conf = RotatingShieldPFConfig(
        num_particles=num_particles,
        resample_threshold=0.5,
        min_strength=0.01,
        p_birth=0.05,
        short_time_s=30.0,
        max_dwell_time_s=10000.0,
        position_min=(0.0, 0.0, 0.0),
        position_max=(env.size_x, env.size_y, env.size_z),
        use_gpu=use_gpu,
        gpu_device="cuda",
        gpu_dtype="float32",
    )
    estimator = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=grid,
        shield_normals=normals,
        mu_by_isotope=mu_by_isotope,
        pf_config=pf_conf,
    )
    current_pose = np.array(env.detector_position, dtype=float)
    estimator.add_measurement_pose(current_pose)
    current_pose_idx = len(estimator.poses) - 1

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
    estimate_mode = "mmse"
    estimate_min_strength = 1.0
    estimate_min_existence_prob = 0.2
    if live:
        plt.ion()
        plt.show(block=False)
        plt.pause(0.1)

    elapsed = 0.0
    step_counter = 0
    live_time = 30.0
    total_pairs = num_orients * num_orients
    visited_poses: list[NDArray[np.float64]] = []
    last_spectrum: np.ndarray | None = None
    last_max_ig: float | None = None
    if max_steps is not None and max_steps <= 0:
        max_steps = None
    while True:
        pose = current_pose
        stop_run = False
        pose_elapsed = 0.0
        orientation_sequence = list(range(total_pairs)) if all_orientations else None
        while True:
            if orientation_sequence is not None:
                if not orientation_sequence:
                    break
                best_pair_idx = orientation_sequence.pop(0)
                ig_score = None
            else:
                best_pair_idx, ig_score = select_best_orientation(
                    estimator,
                    pose_idx=current_pose_idx,
                    live_time_s=live_time,
                )
                last_max_ig = float(ig_score)
                if ig_score < estimator.pf_config.ig_threshold:
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
            z_full, candidates = decomposer.isotope_counts_with_detection(
                spectrum,
                detect_threshold_abs=detect_threshold_abs,
                detect_threshold_rel=detect_threshold_rel,
            )
            last_candidates = set(candidates)
            if detect_consecutive > 0:
                detected_isotopes = _update_detection_counts(
                    candidates,
                    detect_counts,
                    consecutive=detect_consecutive,
                )
                if not detection_locked:
                    if step_counter + 1 >= detect_min_steps and detected_isotopes:
                        estimator.restrict_isotopes(sorted(detected_isotopes))
                        detection_locked = True
                        print(f"Detected isotopes locked to: {sorted(detected_isotopes)}")
                else:
                    new_isotopes = set(detected_isotopes) - set(estimator.isotopes)
                    if new_isotopes:
                        estimator.add_isotopes(sorted(new_isotopes))
                        print(f"Detected isotopes expanded to: {sorted(estimator.isotopes)}")
            if detection_locked:
                viz.set_active_isotopes(estimator.isotopes)
            else:
                viz.set_active_isotopes(sorted(candidates))
            z_k = {iso: float(z_full.get(iso, 0.0)) for iso in estimator.isotopes} if detection_locked else z_full
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
            if detection_locked and (step_counter + 1) % PRUNE_INTERVAL == 0:
                estimator.prune_spurious_sources(
                    tau_mix=PRUNE_TAU_MIX,
                    min_strength_abs=PRUNE_MIN_STRENGTH_ABS,
                    min_strength_ratio=PRUNE_MIN_STRENGTH_RATIO,
                )
            elapsed += live_time
            frame = build_frame_from_pf(
                estimator,
                meas,
                step_index=step_counter,
                time_sec=elapsed,
                estimate_mode=estimate_mode,
                min_est_strength=estimate_min_strength,
                min_existence_prob=estimate_min_existence_prob,
            )
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
            if last_spectrum is not None and step_counter % 10 == 0:
                highlight = set(estimator.isotopes) if detection_locked else last_candidates
                spectrum_path = SPECTRUM_DIR / f"spectrum_step_{step_counter:04d}.png"
                _save_spectrum_plot(decomposer, last_spectrum, spectrum_path, highlight_isotopes=highlight)
            if max_steps is not None and step_counter >= max_steps:
                stop_run = True
                break
            pose_elapsed += live_time
            if pose_elapsed >= estimator.pf_config.max_dwell_time_s:
                break
        if stop_run:
            print(f"Reached max steps ({max_steps}); stopping exploration.")
            break
        if orientation_sequence is not None:
            last_max_ig = estimator.max_orientation_information_gain(
                pose_idx=current_pose_idx,
                live_time_s=live_time,
            )
        if last_max_ig is not None and last_max_ig < estimator.pf_config.ig_threshold:
            print("Converged; stopping exploration (IG threshold reached).")
            break
        visited_poses.append(pose.copy())
        visited_arr = np.vstack(visited_poses) if visited_poses else None
        candidates = generate_candidate_poses(
            current_pose_xyz=pose,
            n_candidates=1024,
            strategy="free_space_sobol",
            min_dist_from_visited=0.5,
            visited_poses_xyz=visited_arr,
            bounds_xyz=(bounds_lo, bounds_hi),
        )
        next_idx = select_next_pose_from_candidates(
            estimator=estimator,
            candidate_poses_xyz=candidates,
            current_pose_xyz=pose,
        )
        current_pose = candidates[next_idx]
        estimator.add_measurement_pose(current_pose, reset_filters=False)
        current_pose_idx = len(estimator.poses) - 1

    # Save final snapshots
    pf_out_path = RESULTS_DIR / "result_pf.png"
    spectrum_out_path = RESULTS_DIR / "result_spectrum.png"
    estimates_out_path = RESULTS_DIR / "result_estimates.png"
    pf_out_path.parent.mkdir(parents=True, exist_ok=True)
    viz.save_final(pf_out_path.as_posix())
    viz.save_estimates_only(estimates_out_path.as_posix())
    if last_spectrum is not None:
        highlight = set(estimator.isotopes) if detection_locked else last_candidates
        _save_spectrum_plot(decomposer, last_spectrum, spectrum_out_path, highlight_isotopes=highlight)
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
    run_live_pf(live=True, max_steps=10)
