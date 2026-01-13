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
from measurement.obstacles import load_or_generate_obstacle_grid
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
from planning.pose_selection import (
    DEFAULT_PLANNING_ROLLOUTS,
    select_next_pose_from_candidates,
)
from planning.shield_rotation import select_best_orientation
from visualization.realtime_viz import (
    DEFAULT_ISOTOPE_COLORS,
    RealTimePFVisualizer,
    build_frame_from_pf,
)
from evaluation_metrics import compute_metrics, print_metrics_report

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
SPECTRUM_DIR = RESULTS_DIR / "spetrum"
PF_DIR = RESULTS_DIR / "pf"
OBSTACLE_LAYOUT_DIR = ROOT / "obstacle_layouts"
PRUNE_MIN_STRENGTH_ABS = 5.0
PRUNE_MIN_STRENGTH_RATIO = 0.001
PRUNE_TAU_MIX = 0.6
PRUNE_MIN_SUPPORT = 2
PRUNE_MIN_OBS_COUNT = 0.0
PRUNE_MIN_MEASUREMENTS = 10
DETECT_MIN_PEAKS_BY_ISOTOPE = {"Eu-154": 2, "Co-60": 2}
DETECT_REL_THRESH_BY_ISOTOPE = {"Co-60": 0.2}
DETECT_CONSECUTIVE_BY_ISOTOPE = {"Eu-154": 5}
DETECT_MISS_AFTER_LOCK = 30
DEFAULT_SOURCE_CONFIG = ROOT / "source_layouts" / "demo_sources.json"
DEFAULT_OBSTACLE_CONFIG = OBSTACLE_LAYOUT_DIR / "demo_obstacles.json"


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


def _update_detection_hysteresis(
    candidates: set[str],
    detect_counts: dict[str, int],
    miss_counts: dict[str, int],
    active_isotopes: set[str],
    consecutive: int,
    miss_consecutive: int | None = None,
    consecutive_by_isotope: dict[str, int] | None = None,
) -> set[str]:
    """
    Update detection state with consecutive hit/miss hysteresis.

    Isotopes are activated after `consecutive` hits and deactivated after
    `miss_consecutive` misses (defaults to `consecutive`).
    """
    updated = set(active_isotopes)
    miss_required = consecutive if miss_consecutive is None else miss_consecutive
    for iso in detect_counts:
        hit_required = consecutive
        if consecutive_by_isotope and iso in consecutive_by_isotope:
            hit_required = int(consecutive_by_isotope[iso])
        if iso in candidates:
            detect_counts[iso] += 1
            miss_counts[iso] = 0
        else:
            miss_counts[iso] += 1
            detect_counts[iso] = 0
        if detect_counts[iso] >= hit_required:
            updated.add(iso)
        if miss_counts[iso] >= miss_required:
            updated.discard(iso)
    return updated


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


def _resolve_ig_threshold(
    mode: str,
    ig_floor: float,
    ig_rel: float,
    ig_max_global: float,
    ig_max_pose: float,
) -> float:
    """Return the active IG threshold for the selected mode."""
    mode = mode.lower()
    if mode == "absolute":
        return float(ig_floor)
    if mode == "relative_max":
        return float(max(ig_floor, ig_rel * ig_max_global))
    if mode == "relative_pose":
        return float(max(ig_floor, ig_rel * ig_max_pose))
    raise ValueError(f"Unknown IG threshold mode: {mode}")


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
    max_poses: int | None = 10,
    sources: list[PointSource] | None = None,
    detect_threshold_abs: float = 50.0,
    detect_threshold_rel: float = 0.3,
    detect_consecutive: int = 10,
    detect_min_steps: int | None = None,
    min_peaks_by_isotope: dict[str, int] | None = None,
    ig_threshold_mode: str = "relative_pose",
    ig_threshold_rel: float = 0.02,
    ig_threshold_min: float | None = None,
    obstacle_layout_path: str | None = DEFAULT_OBSTACLE_CONFIG.as_posix(),
    obstacle_seed: int | None = None,
    eval_match_radius_m: float = 0.5,
) -> None:
    """
    Run a simple PF loop with live visualization (active pose/orientation selection).

    If max_steps is None, run until the information-gain threshold is met.
    If max_poses is None, run without a pose-count limit.
    If obstacle_layout_path is provided, blocked grid cells are excluded and shown
    in black.
    """
    env = EnvironmentConfig(size_x=10.0, size_y=20.0, size_z=10.0, detector_position=(1.0, 1.0, 0.5))
    sources = _build_demo_sources() if sources is None else sources
    decomposer = SpectralDecomposer()
    if min_peaks_by_isotope is None:
        min_peaks_by_isotope = dict(DETECT_MIN_PEAKS_BY_ISOTOPE)
    detect_threshold_rel_by_isotope = dict(DETECT_REL_THRESH_BY_ISOTOPE)
    obstacle_grid = None
    if obstacle_layout_path:
        obstacle_path = Path(obstacle_layout_path)
        if not obstacle_path.is_absolute():
            obstacle_path = (ROOT / obstacle_path).resolve()
        keep_free = None
        if env.detector_position is not None:
            keep_free = [(env.detector_position[0], env.detector_position[1])]
        obstacle_grid = load_or_generate_obstacle_grid(
            obstacle_path,
            size_x=env.size_x,
            size_y=env.size_y,
            cell_size=1.0,
            blocked_fraction=0.4,
            rng_seed=obstacle_seed,
            keep_free_points=keep_free,
        )
    normals = generate_octant_orientations()
    rot_mats = generate_octant_rotation_matrices()
    num_orients = len(rot_mats)
    PF_DIR.mkdir(parents=True, exist_ok=True)

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
    miss_counts = {iso: 0 for iso in isotopes}
    detected_isotopes: set[str] = set()
    detection_locked = False
    active_isotopes: set[str] = set()
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
    background_by_isotope = {iso: 5.0 for iso in isotopes}
    pf_conf = RotatingShieldPFConfig(
        num_particles=num_particles,
        min_particles=num_particles,
        max_particles=num_particles * 2,
        resample_threshold=0.7,
        position_sigma=0.1,
        background_level=background_by_isotope,
        min_strength=5.0,
        p_birth=0.01,
        p_kill=0.2,
        short_time_s=30.0,
        max_dwell_time_s=10000.0,
        position_min=(0.0, 0.0, 0.0),
        position_max=(env.size_x, env.size_y, env.size_z),
        orientation_k=16,
        planning_eig_samples=50,
        planning_rollout_particles=256,
        planning_rollout_method="top_weight",
        use_fast_gpu_rollout=True,
        use_gpu=use_gpu,
        gpu_device="cuda",
        gpu_dtype="float32",
    )
    if ig_threshold_min is not None:
        pf_conf.ig_threshold = float(ig_threshold_min)
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
        obstacle_grid=obstacle_grid,
        show_counts=False,
    )
    estimate_mode = "mmse"
    estimate_min_strength = 100.0
    estimate_min_existence_prob = None
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
    if max_poses is not None and max_poses <= 0:
        max_poses = None
    gpu_status = "enabled" if estimator._gpu_enabled() else "disabled"
    print(
        "Rotation IG threshold: "
        f"mode={ig_threshold_mode}, floor={estimator.pf_config.ig_threshold:.6g}, "
        f"rel={ig_threshold_rel:.6g}"
    )
    print(
        "Planning rollout settings: "
        f"eig_samples={estimator.pf_config.planning_eig_samples}, "
        f"particles={estimator.pf_config.planning_rollout_particles}, "
        f"method={estimator.pf_config.planning_rollout_method}, "
        f"rollouts={DEFAULT_PLANNING_ROLLOUTS}"
    )
    print(
        "GPU acceleration: "
        f"{gpu_status} (device={estimator.pf_config.gpu_device}, dtype={estimator.pf_config.gpu_dtype})"
    )
    ig_max_global = 0.0
    pose_counter = 0
    while True:
        pose = current_pose
        stop_run = False
        pose_elapsed = 0.0
        remaining_orientations = set(range(total_pairs))
        rotation_limit = max(1, int(estimator.pf_config.orientation_k))
        rotation_count = 0
        ig_max_pose = 0.0
        ig_threshold_current = estimator.pf_config.ig_threshold
        while True:
            if rotation_count >= rotation_limit:
                print(f"Reached max rotations per pose ({rotation_limit}); moving to the next pose.")
                break
            if not remaining_orientations:
                print("All orientation pairs exhausted; moving to the next pose.")
                break
            best_pair_idx, ig_score = select_best_orientation(
                estimator,
                pose_idx=current_pose_idx,
                live_time_s=live_time,
                allowed_indices=remaining_orientations,
            )
            if best_pair_idx < 0:
                print("No valid orientation candidates; moving to the next pose.")
                break
            ig_val = float(ig_score)
            last_max_ig = ig_val
            ig_max_global = max(ig_max_global, ig_val)
            ig_max_pose = max(ig_max_pose, ig_val)
            ig_threshold_current = _resolve_ig_threshold(
                mode=ig_threshold_mode,
                ig_floor=estimator.pf_config.ig_threshold,
                ig_rel=ig_threshold_rel,
                ig_max_global=ig_max_global,
                ig_max_pose=ig_max_pose,
            )
            if ig_val < ig_threshold_current:
                print(
                    "Stopping rotation at this pose "
                    f"(max IG {ig_val:.6g} < threshold {ig_threshold_current:.6g})."
                )
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
                detect_threshold_rel_by_isotope=detect_threshold_rel_by_isotope,
                min_peaks_by_isotope=min_peaks_by_isotope,
            )
            if candidates:
                nnls_counts = decomposer.decompose_subset(spectrum, isotopes=sorted(candidates))
                z_full = {iso: float(nnls_counts.get(iso, 0.0)) for iso in decomposer.isotope_names}
            last_candidates = set(candidates)
            if detect_consecutive > 0:
                miss_required = DETECT_MISS_AFTER_LOCK if detection_locked else detect_consecutive
                active_isotopes = _update_detection_hysteresis(
                    candidates,
                    detect_counts,
                    miss_counts,
                    active_isotopes,
                    consecutive=detect_consecutive,
                    miss_consecutive=miss_required,
                    consecutive_by_isotope=DETECT_CONSECUTIVE_BY_ISOTOPE,
                )
                detected_isotopes = set(active_isotopes)
                last_candidates = set(detected_isotopes)
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
                    removed_isotopes = set(estimator.isotopes) - set(detected_isotopes)
                    if removed_isotopes:
                        remaining = [iso for iso in estimator.isotopes if iso not in removed_isotopes]
                        if remaining:
                            estimator.restrict_isotopes(remaining)
                            print(f"Detected isotopes pruned to: {sorted(estimator.isotopes)}")
            if detection_locked:
                viz.set_active_isotopes(estimator.isotopes)
            else:
                viz.set_active_isotopes(sorted(detected_isotopes))
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
            if (
                detection_locked
                and estimator.measurements
                and len(estimator.measurements) >= PRUNE_MIN_MEASUREMENTS
            ):
                estimator.prune_spurious_sources(
                    tau_mix=PRUNE_TAU_MIX,
                    min_support=PRUNE_MIN_SUPPORT,
                    min_obs_count=PRUNE_MIN_OBS_COUNT,
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
                f"ig={ig_val:.6g} ig_threshold={ig_threshold_current:.6g} "
                f"fe_idx={fe_idx} pb_idx={pb_idx} RFe_dir={rfe_dir} RPb_dir={rpb_dir}"
            )
            if live:
                plt.pause(0.05)
            step_counter += 1
            rotation_count += 1
            remaining_orientations.discard(best_pair_idx)
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
        if estimator.measurements and estimator.measurements[-1].pose_idx == current_pose_idx:
            pf_step = current_pose_idx + 1
            pf_path = PF_DIR / f"pf_step_{pf_step:03d}.png"
            viz.save_final(pf_path.as_posix())
        if stop_run:
            print(f"Reached max steps ({max_steps}); stopping exploration.")
            break
        if last_max_ig is not None and last_max_ig < ig_threshold_current:
            print(
                "Converged; stopping exploration "
                f"(max IG {last_max_ig:.6g} < threshold {ig_threshold_current:.6g})."
            )
            break
        visited_poses.append(pose.copy())
        pose_counter += 1
        if max_poses is not None and pose_counter >= max_poses:
            print(f"Reached max poses ({max_poses}); stopping exploration.")
            break
        visited_arr = np.vstack(visited_poses) if visited_poses else None
        print("Generating candidate poses for next measurement point...")
        candidates = generate_candidate_poses(
            current_pose_xyz=pose,
            map_api=obstacle_grid,
            n_candidates=16,
            strategy="free_space_sobol",
            min_dist_from_visited=3.0,
            visited_poses_xyz=visited_arr,
            bounds_xyz=(bounds_lo, bounds_hi),
        )
        print(f"Generated {len(candidates)} candidate poses. Computing best next pose...")
        next_idx = select_next_pose_from_candidates(
            estimator=estimator,
            candidate_poses_xyz=candidates,
            current_pose_xyz=pose,
            verbose=True,
            progress_every=1,
            auto_lambda_cost=True,
            num_rollouts=DEFAULT_PLANNING_ROLLOUTS,
        )
        current_pose = candidates[next_idx]
        estimator.add_measurement_pose(current_pose, reset_filters=False)
        current_pose_idx = len(estimator.poses) - 1

    if (
        detection_locked
        and estimator.measurements
        and len(estimator.measurements) >= PRUNE_MIN_MEASUREMENTS
    ):
        estimator.prune_spurious_sources(
            tau_mix=PRUNE_TAU_MIX,
            min_support=PRUNE_MIN_SUPPORT,
            min_obs_count=PRUNE_MIN_OBS_COUNT,
            min_strength_abs=PRUNE_MIN_STRENGTH_ABS,
            min_strength_ratio=PRUNE_MIN_STRENGTH_RATIO,
        )

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
    gt_by_iso: dict[str, list[dict[str, float | list[float]]]] = {}
    for src in sources:
        gt_by_iso.setdefault(src.isotope, []).append(
            {
                "pos": [
                    float(src.position[0]),
                    float(src.position[1]),
                    float(src.position[2]),
                ],
                "strength": float(src.intensity_cps_1m),
            }
        )
    est_by_iso: dict[str, list[dict[str, float | list[float]]]] = {}
    for iso, estimate in estimator.estimates().items():
        positions = np.asarray(estimate[0], dtype=float)
        strengths = np.asarray(estimate[1], dtype=float)
        est_list: list[dict[str, float | list[float]]] = []
        for pos, strength in zip(positions, strengths):
            est_list.append(
                {
                    "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                    "strength": float(strength),
                }
            )
        est_by_iso[iso] = est_list
    metrics = compute_metrics(
        gt_by_iso,
        est_by_iso,
        match_radius_m=eval_match_radius_m,
    )
    print_metrics_report(metrics)
    if live:
        plt.ioff()
        plt.pause(0.1)
    plt.close("all")


def run_realtime_pf() -> None:
    """Entry point for real-time PF + visualization with built-in demo settings."""
    run_live_pf(live=True, max_steps=10)
