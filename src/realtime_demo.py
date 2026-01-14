"""Real-time demo for the rotating-shield particle filter with visualization."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time

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
from measurement.continuous_kernels import ContinuousKernel
from spectrum.library import get_detection_lines_keV
from spectrum.peak_detection import detect_peaks
from spectrum.pipeline import SpectralDecomposer
from spectrum.baseline import baseline_als
from spectrum.smoothing import gaussian_smooth
from pf.parallel import Measurement
from pf.estimator import RotatingShieldPFEstimator, RotatingShieldPFConfig
from pf.particle_filter import PFConfig
from planning.candidate_generation import generate_candidate_poses
from planning.pose_selection import (
    DEFAULT_PLANNING_ROLLOUTS,
    select_next_pose_from_candidates,
)
from visualization.realtime_viz import (
    DEFAULT_ISOTOPE_COLORS,
    RealTimePFVisualizer,
    build_frame_from_pf,
)
from visualization.ig_shield_geometry import render_octant_grid
from evaluation_metrics import compute_metrics, print_metrics_report

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
SPECTRUM_DIR = RESULTS_DIR / "spetrum"
PF_DIR = RESULTS_DIR / "pf"
IG_DIR = RESULTS_DIR / "IG"
OBSTACLE_LAYOUT_DIR = ROOT / "obstacle_layouts"
PRUNE_MIN_STRENGTH_ABS = 5.0
PRUNE_MIN_STRENGTH_RATIO = 0.001
PRUNE_TAU_MIX = 0.6
PRUNE_MIN_SUPPORT = 2
PRUNE_MIN_OBS_COUNT = 0.0
PRUNE_MIN_MEASUREMENTS = 10
DETECT_MIN_PEAKS_BY_ISOTOPE = {"Eu-154": 2, "Co-60": 2}
DETECT_REL_THRESH_BY_ISOTOPE = {"Co-60": 0.2}
DETECT_CONSECUTIVE_BY_ISOTOPE = {"Cs-137": 3, "Co-60": 3, "Eu-154": 5}
DETECT_MISS_AFTER_LOCK = 30
DEFAULT_SOURCE_CONFIG = ROOT / "source_layouts" / "demo_sources.json"
DEFAULT_OBSTACLE_CONFIG = OBSTACLE_LAYOUT_DIR / "demo_obstacles.json"
CANDIDATE_GRID_SPACING = (0.5, 0.5, 0.5)
CANDIDATE_GRID_MARGIN = 0.5
HEALTH_LOG_TOP_K = 3


def _build_demo_sources() -> list[PointSource]:
    """Define a small set of synthetic sources inside the environment."""
    return [
        PointSource("Cs-137", position=(5.0, 10.0, 5.0), intensity_cps_1m=50000.0),
        PointSource("Co-60", position=(2.0, 15.0, 7.0), intensity_cps_1m=20000.0),
        PointSource("Eu-154", position=(7.0, 5.0, 3.0), intensity_cps_1m=30000.0),
    ]


def _candidate_axis_points(start: float, stop: float, step: float) -> NDArray[np.float64]:
    """Return evenly spaced axis points within [start, stop] using the given step."""
    if step <= 0:
        raise ValueError("step must be positive.")
    if stop < start:
        return np.zeros(0, dtype=float)
    count = int(np.floor((stop - start) / step)) + 1
    if count <= 0:
        return np.zeros(0, dtype=float)
    return start + step * np.arange(count, dtype=float)


def _build_candidate_sources(
    env: EnvironmentConfig,
    spacing: tuple[float, float, float],
    margin: float,
) -> NDArray[np.float64]:
    """Create a dense 3D grid of candidate sources inside the environment bounds."""
    xs = _candidate_axis_points(margin, env.size_x - margin, spacing[0])
    ys = _candidate_axis_points(margin, env.size_y - margin, spacing[1])
    zs = _candidate_axis_points(margin, env.size_z - margin, spacing[2])
    if xs.size == 0 or ys.size == 0 or zs.size == 0:
        raise ValueError("Candidate grid is empty; check spacing and margin values.")
    return np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=float)


def _initial_particle_nearby_probability(
    num_particles: int,
    position_min: tuple[float, float, float],
    position_max: tuple[float, float, float],
    radius_m: float,
    init_num_sources: tuple[int, int],
) -> float:
    """
    Return the probability that at least one initial source lies within radius_m of a target.

    Assumes each source position is uniformly sampled within the bounding box.
    """
    if radius_m <= 0.0 or num_particles <= 0:
        return 0.0
    bounds_lo = np.asarray(position_min, dtype=float)
    bounds_hi = np.asarray(position_max, dtype=float)
    span = bounds_hi - bounds_lo
    volume = float(np.prod(span)) if np.all(span > 0.0) else 0.0
    if volume <= 0.0:
        return 0.0
    p = (4.0 / 3.0) * np.pi * (radius_m**3) / volume
    p = float(np.clip(p, 0.0, 1.0))
    r_min, r_max = init_num_sources
    r_min, r_max = (int(r_min), int(r_max)) if r_min <= r_max else (int(r_max), int(r_min))
    per_particle = sum((1.0 - p) ** r for r in range(r_min, r_max + 1)) / (r_max - r_min + 1)
    return float(1.0 - per_particle**num_particles)


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


def _detect_isotopes_from_counts(
    counts: dict[str, float],
    detect_threshold_abs: float,
    detect_threshold_rel: float,
    detect_threshold_rel_by_isotope: dict[str, float] | None,
) -> set[str]:
    """Return isotopes detected from counts using absolute/relative thresholds."""
    max_c = max(counts.values()) if counts else 0.0
    detected: set[str] = set()
    rel_by_iso = detect_threshold_rel_by_isotope or {}
    for iso, val in counts.items():
        rel_thresh = float(rel_by_iso.get(iso, detect_threshold_rel))
        if val >= detect_threshold_abs and (max_c <= 0.0 or val / max_c >= rel_thresh):
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


def _fmt_pos(pos: NDArray[np.float64]) -> str:
    """Format a position vector for logging."""
    return np.array2string(np.asarray(pos, dtype=float), precision=2, floatmode="fixed", separator=", ")


def _fmt_counts(counts: dict[str, float]) -> str:
    """Format a count dict for logging."""
    items = ", ".join(f"{iso}: {float(val):.1f}" for iso, val in sorted(counts.items()))
    return "{" + items + "}"


def _fmt_sources(positions: NDArray[np.float64], strengths: NDArray[np.float64]) -> str:
    """Format a list of source positions/strengths for logging."""
    positions = np.asarray(positions, dtype=float)
    strengths = np.asarray(strengths, dtype=float)
    if positions.size == 0 or strengths.size == 0:
        return "[]"
    chunks = []
    for pos, strength in zip(positions, strengths):
        pos_str = np.array2string(pos, precision=2, floatmode="fixed", separator=", ")
        chunks.append(f"{pos_str}|{float(strength):.2f}")
    return "[" + ", ".join(chunks) + "]"


def _fmt_top_k(entries: list[dict[str, object]]) -> str:
    """Format top-k particle summaries for logging."""
    chunks = []
    for entry in entries:
        weight = float(entry.get("weight", 0.0))
        num_sources = int(entry.get("num_sources", 0))
        positions = np.asarray(
            entry.get("positions", np.zeros((0, 3))),
            dtype=float,
        )
        strengths = np.asarray(entry.get("strengths", np.zeros(0)), dtype=float)
        sources = _fmt_sources(positions, strengths)
        chunks.append(f"(w={weight:.3f}, r={num_sources}, sources={sources})")
    return "[" + "; ".join(chunks) + "]"


def _fmt_optional_float(value: float | None, precision: int = 2) -> str:
    """Format an optional float for logging."""
    if value is None:
        return "NA"
    return f"{float(value):.{precision}f}"


def _log_pf_diagnostics(
    estimator: RotatingShieldPFEstimator,
    step_index: int,
    top_k: int = HEALTH_LOG_TOP_K,
) -> None:
    """Log per-step PF diagnostics for each isotope."""
    diagnostics = estimator.step_diagnostics(top_k=top_k)
    if not diagnostics:
        print(f"[step {step_index}] pf_diagnostics: no active filters")
        return
    for iso, stats in diagnostics.items():
        ess_pre = float(stats["ess_pre"])
        resampled = bool(stats["resampled"])
        ess_post = stats["ess_post"]
        n_after_adapt = int(stats["n_after_adapt"])
        resamples = int(stats["resample_count"])
        births = int(stats["birth_count"])
        kills = int(stats["kill_count"])
        r_mean = float(stats["r_mean"])
        r_var = float(stats["r_var"])
        map_pos, map_str = stats["map"]
        mmse_pos, mmse_str = stats["mmse"]
        top_entries = stats["top_k"]
        print(
            f"[step {step_index}] pf[{iso}] ess_pre={ess_pre:.2f} resampled={resampled} "
            f"ess_post={_fmt_optional_float(ess_post)} n_after={n_after_adapt} "
            f"resamples={resamples} births={births} kills={kills} "
            f"r_mean={r_mean:.2f} r_var={r_var:.2f}"
        )
        print(
            f"[step {step_index}] pf[{iso}] map={_fmt_sources(map_pos, map_str)} "
            f"mmse={_fmt_sources(mmse_pos, mmse_str)}"
        )
        if top_entries:
            print(f"[step {step_index}] pf[{iso}] top_k={_fmt_top_k(top_entries)}")


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


def _default_use_gpu() -> bool:
    """Return True if CUDA is available for torch acceleration."""
    try:
        from pf import gpu_utils
    except ImportError:
        return False
    return gpu_utils.torch_available()


def _resolve_ig_workers(ig_workers: int | None) -> int:
    """Return a worker count for IG evaluation (0 or None means auto)."""
    if ig_workers is None:
        return 1
    workers = int(ig_workers)
    if workers <= 0:
        cpu_count = os.cpu_count() or 1
        return max(1, min(4, cpu_count))
    return workers


def _compute_ig_grid(
    estimator: RotatingShieldPFEstimator,
    rot_mats: Sequence[np.ndarray],
    *,
    pose_idx: int,
    live_time_s: float,
    planning_isotopes: Sequence[str] | None = None,
) -> np.ndarray:
    """
    Compute expected IG over all Fe/Pb orientation pairs for the current PF state.
    """
    eig_samples = estimator.pf_config.planning_eig_samples
    if eig_samples is None:
        eig_samples = estimator.pf_config.eig_num_samples
    rollout_particles = estimator.pf_config.planning_rollout_particles
    if rollout_particles is None:
        rollout_particles = estimator.pf_config.planning_particles
    rollout_method = estimator.pf_config.planning_rollout_method or estimator.pf_config.planning_method
    particles_by_iso = estimator.planning_particles(
        max_particles=rollout_particles,
        method=rollout_method,
    )
    alpha_weights = estimator.pf_config.alpha_weights
    if planning_isotopes is not None:
        planning_set = set(planning_isotopes)
        particles_by_iso = {
            iso: val for iso, val in particles_by_iso.items() if iso in planning_set
        }
        if alpha_weights is None:
            alpha_weights = {iso: 1.0 for iso in planning_set}
        else:
            alpha_weights = {
                iso: float(alpha_weights.get(iso, 1.0)) for iso in planning_set
            }
    size = len(rot_mats)
    scores = np.zeros((size, size), dtype=float)

    def _ig_for_pair(fe_idx: int, pb_idx: int, RFe: np.ndarray, RPb: np.ndarray) -> float:
        """Compute expected IG for a single Fe/Pb orientation pair."""
        return float(
            estimator.orientation_expected_information_gain(
                pose_idx=pose_idx,
                RFe=RFe,
                RPb=RPb,
                live_time_s=live_time_s,
                num_samples=eig_samples,
                alpha_by_isotope=alpha_weights,
                particles_by_isotope=particles_by_iso,
            )
        )

    total_pairs = size * size
    workers = _resolve_ig_workers(getattr(estimator.pf_config, "ig_workers", None))
    if workers <= 1 or total_pairs <= 1:
        for fe_idx, RFe in enumerate(rot_mats):
            for pb_idx, RPb in enumerate(rot_mats):
                scores[fe_idx, pb_idx] = _ig_for_pair(fe_idx, pb_idx, RFe, RPb)
        return scores

    max_workers = min(workers, total_pairs)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for fe_idx, RFe in enumerate(rot_mats):
            for pb_idx, RPb in enumerate(rot_mats):
                future = executor.submit(_ig_for_pair, fe_idx, pb_idx, RFe, RPb)
                futures[future] = (fe_idx, pb_idx)
        for future in as_completed(futures):
            fe_idx, pb_idx = futures[future]
            scores[fe_idx, pb_idx] = float(future.result())
    return scores


def _select_best_pair_from_scores(
    scores: NDArray[np.float64],
    allowed_indices: set[int] | None,
) -> tuple[int, float]:
    """Return the best (fe,pb) pair index and score from a full IG grid."""
    if scores.size == 0:
        return -1, 0.0
    size = int(scores.shape[0])
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must be a square 2D array.")
    if allowed_indices is None:
        allowed_iter = range(size * size)
    else:
        allowed_iter = sorted(allowed_indices)
    best_idx = -1
    best_score = -np.inf
    for oid in allowed_iter:
        fe_idx = int(oid) // size
        pb_idx = int(oid) % size
        score = float(scores[fe_idx, pb_idx])
        if np.isnan(score):
            continue
        if score > best_score:
            best_score = score
            best_idx = int(oid)
    if best_idx < 0:
        return -1, 0.0
    return best_idx, float(best_score)


def _save_spectrum_plot(
    decomposer: SpectralDecomposer,
    spectrum: np.ndarray,
    output_path: Path,
    peak_tolerance_keV: float = 10.0,
    highlight_isotopes: set[str] | None = None,
    counts_by_isotope: dict[str, float] | None = None,
    use_detection_lines: bool = True,
    window_keV: float | None = None,
    window_sigma: float = 3.0,
) -> None:
    """Save the measurement spectrum with nuclide lines and colored count windows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    energy_axis = decomposer.energy_axis
    library = decomposer.library
    if highlight_isotopes is not None:
        library = {iso: library[iso] for iso in library if iso in highlight_isotopes}
    line_map: dict[str, list[float]] = {}
    for iso, nuclide in library.items():
        if use_detection_lines:
            lines = get_detection_lines_keV(iso)
        else:
            lines = [line.energy_keV for line in nuclide.lines]
        if lines:
            line_map[iso] = lines
    colors = _build_isotope_colors(list(library.keys()))
    smoothed = gaussian_smooth(
        spectrum,
        sigma_bins=2.0,
        use_gpu=decomposer.use_gpu,
        gpu_device=decomposer.gpu_device,
        gpu_dtype=decomposer.gpu_dtype,
    )
    baseline = baseline_als(
        smoothed,
        lam=decomposer.config.baseline_lam,
        p=decomposer.config.baseline_p,
        niter=decomposer.config.baseline_niter,
    )
    corrected = np.clip(smoothed - baseline, a_min=0.0, a_max=None)
    peak_indices = detect_peaks(corrected, prominence=0.05, distance=5)
    line_energies = {iso: np.array(lines, dtype=float) for iso, lines in line_map.items()} if use_detection_lines else None
    peaks_by_iso, unassigned = decomposer._assign_peak_indices(
        energy_axis,
        peak_indices,
        library,
        tolerance_keV=peak_tolerance_keV,
        line_energies=line_energies,
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(energy_axis, spectrum, color="black", linewidth=1.0, label="Spectrum")
    for iso, nuclide in library.items():
        if iso not in line_map:
            continue
        if counts_by_isotope is not None and counts_by_isotope.get(iso, 0.0) <= 0.0:
            continue
        color = colors.get(iso, "gray")
        for line_keV in line_map[iso]:
            half_width = window_keV
            if half_width is None:
                sigma = float(decomposer.resolution_fn(line_keV))
                sigma_width = max(window_sigma * sigma, 1e-6)
                if use_detection_lines:
                    half_width = max(float(decomposer.config.detect_half_window_keV), sigma_width)
                else:
                    half_width = sigma_width
            mask = np.abs(energy_axis - line_keV) <= float(half_width)
            if np.any(mask):
                ax.fill_between(
                    energy_axis[mask],
                    baseline[mask],
                    smoothed[mask],
                    color=color,
                    alpha=0.2,
                    linewidth=0.0,
                )
    for iso, nuclide in library.items():
        if iso not in line_map:
            continue
        color = colors.get(iso, "gray")
        labeled = False
        for line_keV in line_map[iso]:
            label = iso if not labeled else None
            ax.axvline(
                line_keV,
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


def _expected_counts(
    kernel: ContinuousKernel,
    sources: list[PointSource],
    isotopes: list[str],
    detector_pos: NDArray[np.float64],
    fe_index: int,
    pb_index: int,
    live_time_s: float,
) -> dict[str, float]:
    """Compute inverse-square + shield-attenuated expected counts per isotope."""
    counts: dict[str, float] = {}
    for iso in isotopes:
        iso_sources = [src for src in sources if src.isotope == iso]
        if not iso_sources:
            counts[iso] = 0.0
            continue
        positions = np.vstack([src.position_array() for src in iso_sources])
        strengths = np.array([src.intensity_cps_1m for src in iso_sources], dtype=float)
        counts[iso] = float(
            kernel.expected_counts_pair(
                isotope=iso,
                detector_pos=detector_pos,
                sources=positions,
                strengths=strengths,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
                background=0.0,
            )
        )
    return counts




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
    candidate_grid_spacing: tuple[float, float, float] | None = None,
    candidate_grid_margin: float = CANDIDATE_GRID_MARGIN,
    count_mode: str = "spectrum",
    pf_config_overrides: dict[str, object] | None = None,
    save_outputs: bool = True,
    return_state: bool = False,
) -> RotatingShieldPFEstimator | None:
    """
    Run a simple PF loop with live visualization (active pose/orientation selection).

    If max_steps is None, run until the information-gain threshold is met.
    If max_poses is None, run without a pose-count limit.
    If obstacle_layout_path is provided, blocked grid cells are excluded and shown
    in black.
    count_mode controls the per-isotope counts passed to the PF:
    - "spectrum": use spectrum-derived counts (default).
    - "expected": use expected counts from the kernel.

    Args:
        pf_config_overrides: Optional overrides applied to the PF configuration.
        save_outputs: When False, skip writing plots and snapshot images.
        return_state: When True, return the estimator for inspection/testing.
        candidate_grid_spacing: Optional (x, y, z) spacing for birth candidate grid.
        candidate_grid_margin: Margin from the environment bounds for candidate sources.
    """
    count_mode = count_mode.strip().lower()
    if count_mode not in {"spectrum", "expected"}:
        raise ValueError(f"Unknown count_mode: {count_mode}")
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
    if save_outputs:
        PF_DIR.mkdir(parents=True, exist_ok=True)

    # Candidate sources: dense grid inside environment (used by birth proposals).
    spacing = candidate_grid_spacing or CANDIDATE_GRID_SPACING
    grid = _build_candidate_sources(env, spacing=spacing, margin=float(candidate_grid_margin))

    bounds_lo = np.array([0.0, 0.0, env.detector_position[2]], dtype=float)
    bounds_hi = np.array([env.size_x, env.size_y, env.detector_position[2]], dtype=float)

    isotopes = list(decomposer.isotope_names)
    detect_min_steps = detect_consecutive if detect_min_steps is None else detect_min_steps
    detect_counts = {iso: 0 for iso in isotopes}
    miss_counts = {iso: 0 for iso in isotopes}
    detected_isotopes: set[str] = set()
    detection_locked = False
    locked_isotopes_for_planning: set[str] = set()
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
    expected_kernel = ContinuousKernel(
        mu_by_isotope=mu_by_isotope,
        shield_params=shield_params,
        use_gpu=use_gpu,
        gpu_device="cuda",
        gpu_dtype="float64",
    )
    background_by_isotope = {iso: 5.0 for iso in isotopes}
    pf_conf = RotatingShieldPFConfig(
        num_particles=num_particles,
        min_particles=num_particles,
        max_particles=num_particles,
        max_sources=1,
        resample_threshold=0.7,
        position_sigma=0.5,
        background_level=background_by_isotope,
        min_strength=5.0,
        p_birth=0.0,
        p_kill=0.0,
        short_time_s=30.0,
        max_dwell_time_s=10000.0,
        position_min=(0.0, 0.0, 0.0),
        position_max=(env.size_x, env.size_y, env.size_z),
        init_num_sources=(1, 1),
        split_prob=0.0,
        merge_prob=0.0,
        orientation_k=16,
        planning_eig_samples=50,
        planning_rollout_particles=256,
        planning_rollout_method="top_weight",
        use_fast_gpu_rollout=True,
        use_gpu=use_gpu,
        gpu_device="cuda",
        gpu_dtype="float64",
    )
    if pf_config_overrides:
        for key, value in pf_config_overrides.items():
            if not hasattr(pf_conf, key):
                raise ValueError(f"Unknown PF config override: {key}")
            setattr(pf_conf, key, value)
    if ig_threshold_min is not None:
        pf_conf.ig_threshold = float(ig_threshold_min)
    init_pf = PFConfig()
    init_pf.init_num_sources = pf_conf.init_num_sources
    init_support_prob = _initial_particle_nearby_probability(
        num_particles=int(pf_conf.num_particles),
        position_min=pf_conf.position_min,
        position_max=pf_conf.position_max,
        radius_m=float(eval_match_radius_m),
        init_num_sources=init_pf.init_num_sources,
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
            true_strengths[iso] = [float(val) for val in strengths]
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
    last_counts: dict[str, float] | None = None
    last_max_ig: float | None = None
    if max_steps is not None and max_steps <= 0:
        max_steps = None
    if max_poses is not None and max_poses <= 0:
        max_poses = None
    gpu_status = "enabled" if estimator._gpu_enabled() else "disabled"
    cfg = decomposer.config
    if save_outputs:
        IG_DIR.mkdir(parents=True, exist_ok=True)
    print(
        "Spectrum config: "
        f"bin_width_keV={cfg.bin_width_keV}, live_time_s={live_time}, "
        f"smooth_sigma_bins={cfg.smooth_sigma_bins}, "
        f"als_lambda={cfg.als_lambda}, als_p={cfg.als_p}, als_niter={cfg.als_niter}, "
        f"resolution_a={cfg.resolution_a}, resolution_b={cfg.resolution_b}, "
        f"peak_window_sigma={cfg.peak_window_sigma}, dead_time_tau_s={cfg.dead_time_tau_s}"
    )
    print(
        "Rotation IG threshold: "
        f"mode={ig_threshold_mode}, floor={estimator.pf_config.ig_threshold:.6g}, "
        f"rel={ig_threshold_rel:.6g}"
    )
    ig_workers = _resolve_ig_workers(estimator.pf_config.ig_workers)
    print(f"IG grid workers: {ig_workers}")
    print(
        "Candidate grid: "
        f"spacing={spacing} margin={float(candidate_grid_margin):.2f} "
        f"points={grid.shape[0]}"
    )
    print(
        "Init support probability: "
        f"radius={float(eval_match_radius_m):.2f}m "
        f"prob≈{init_support_prob:.3f} "
        f"(init_num_sources={init_pf.init_num_sources})"
    )
    print(
        "PF init prior: "
        f"init_num_sources={init_pf.init_num_sources}, "
        f"init_strength_log_mean={init_pf.init_strength_log_mean:.2f}, "
        f"init_strength_log_sigma={init_pf.init_strength_log_sigma:.2f}, "
        f"max_sources={pf_conf.max_sources}"
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
            planning_isotopes = None
            if detection_locked and locked_isotopes_for_planning:
                planning_isotopes = sorted(locked_isotopes_for_planning)
            ig_start = time.perf_counter()
            ig_scores = _compute_ig_grid(
                estimator,
                rot_mats,
                pose_idx=current_pose_idx,
                live_time_s=live_time,
                planning_isotopes=planning_isotopes,
            )
            ig_elapsed = time.perf_counter() - ig_start
            best_pair_idx, ig_score = _select_best_pair_from_scores(
                ig_scores,
                remaining_orientations,
            )
            if best_pair_idx < 0:
                print("No valid orientation candidates; moving to the next pose.")
                break
            ig_val = max(float(ig_score), 0.0)
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
            if save_outputs and (step_counter + 1) % 10 == 0:
                ig_path = IG_DIR / f"ig_grid_step_{step_counter:04d}.png"
                render_octant_grid(
                    ig_path,
                    ig_scores=ig_scores,
                    highlight_idx=(fe_idx, pb_idx),
                    highlight_max=False,
                    font_size=12,
                )
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
            expected_counts = _expected_counts(
                expected_kernel,
                sources,
                isotopes,
                detector_pos=pose,
                fe_index=fe_idx,
                pb_index=pb_idx,
                live_time_s=live_time,
            )
            if count_mode == "expected":
                z_detected = expected_counts
                detected = _detect_isotopes_from_counts(
                    expected_counts,
                    detect_threshold_abs=detect_threshold_abs,
                    detect_threshold_rel=detect_threshold_rel,
                    detect_threshold_rel_by_isotope=detect_threshold_rel_by_isotope,
                )
            else:
                z_detected, detected = decomposer.isotope_counts_with_detection(
                    spectrum,
                    live_time_s=live_time,
                    # Always detect across the full library so the lock can expand.
                    active_isotopes=None,
                    detect_threshold_abs=detect_threshold_abs,
                    detect_threshold_rel=detect_threshold_rel,
                    detect_threshold_rel_by_isotope=detect_threshold_rel_by_isotope,
                    min_peaks_by_isotope=min_peaks_by_isotope,
                )
            last_counts = {iso: float(val) for iso, val in z_detected.items()}
            last_candidates = set(detected)
            if detect_consecutive > 0:
                miss_required = DETECT_MISS_AFTER_LOCK if detection_locked else detect_consecutive
                active_isotopes = _update_detection_hysteresis(
                    set(detected),
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
                    should_lock = step_counter + 1 >= detect_min_steps and detected_isotopes
                    allow_lock = any(iso != "Eu-154" for iso in detected_isotopes)
                    if should_lock and allow_lock:
                        locked_isotopes_for_planning |= set(detected_isotopes)
                        detection_locked = True
                        print(
                            "Detected isotopes locked to: "
                            f"{sorted(locked_isotopes_for_planning)}"
                        )
                else:
                    new_locked = locked_isotopes_for_planning | set(detected_isotopes)
                    if new_locked != locked_isotopes_for_planning:
                        locked_isotopes_for_planning = new_locked
                        print(
                            "pes expanded to: "
                            f"{sorted(locked_isotopes_for_planning)}"
                        )
            counts_for_pf = expected_counts if count_mode == "expected" else z_detected
            z_k_full = {iso: float(counts_for_pf.get(iso, 0.0)) for iso in isotopes}
            z_counts = z_k_full
            z_k = z_k_full
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
            pf_start = time.perf_counter()
            estimator.update_pair(
                z_k=z_k,
                pose_idx=current_pose_idx,
                fe_index=fe_idx,
                pb_index=pb_idx,
                live_time_s=live_time,
            )
            pf_elapsed = time.perf_counter() - pf_start
            elapsed += live_time
            viz_elapsed = 0.0
            viz_start = time.perf_counter()
            frame = build_frame_from_pf(
                estimator,
                meas,
                step_index=step_counter,
                time_sec=elapsed,
                estimate_mode=estimate_mode,
                min_est_strength=estimate_min_strength,
                min_existence_prob=estimate_min_existence_prob,
            )
            viz_elapsed += time.perf_counter() - viz_start
            prune_start = time.perf_counter()
            pruned = estimator.pruned_estimates(
                method="legacy",
                tau_mix=PRUNE_TAU_MIX,
                min_support=PRUNE_MIN_SUPPORT,
                min_obs_count=PRUNE_MIN_OBS_COUNT,
                min_strength_abs=PRUNE_MIN_STRENGTH_ABS,
                min_strength_ratio=PRUNE_MIN_STRENGTH_RATIO,
            )
            prune_elapsed = time.perf_counter() - prune_start
            viz_start = time.perf_counter()
            if hasattr(frame, "estimated_sources") and hasattr(frame, "estimated_strengths"):
                frame.estimated_sources = {}
                frame.estimated_strengths = {}
                for iso in isotopes:
                    pos, strg = pruned.get(iso, (np.zeros((0, 3)), np.zeros(0)))
                    if estimate_min_strength is not None and strg.size:
                        mask = strg >= estimate_min_strength
                        pos = pos[mask]
                        strg = strg[mask]
                    frame.estimated_sources[iso] = pos
                    frame.estimated_strengths[iso] = strg
            viz.update(frame)
            # Log only measurement point and shield orientations
            print(
                f"[step {step_counter}] pose={_fmt_pos(pose)} orient_pair={best_pair_idx} "
                f"ig={ig_val:.6g} ig_threshold={ig_threshold_current:.6g} "
                f"fe_idx={fe_idx} pb_idx={pb_idx} "
                f"live_time_s={live_time:.1f} z_keys={sorted(z_k.keys())} "
                f"z_obs={_fmt_counts(z_counts)} "
                f"expected={_fmt_counts(expected_counts)}"
            )
            if live:
                plt.pause(0.05)
            viz_elapsed += time.perf_counter() - viz_start
            _log_pf_diagnostics(estimator, step_counter)
            print(
                f"[timing step {step_counter}] ig={ig_elapsed:.3f}s pf={pf_elapsed:.3f}s "
                f"prune={prune_elapsed:.3f}s viz={viz_elapsed:.3f}s"
            )
            step_counter += 1
            rotation_count += 1
            remaining_orientations.discard(best_pair_idx)
            if save_outputs and last_spectrum is not None and step_counter % 10 == 0:
                highlight = (
                    set(locked_isotopes_for_planning)
                    if detection_locked
                    else last_candidates
                )
                spectrum_path = SPECTRUM_DIR / f"spectrum_step_{step_counter:04d}.png"
                _save_spectrum_plot(
                    decomposer,
                    last_spectrum,
                    spectrum_path,
                    highlight_isotopes=highlight,
                    counts_by_isotope=last_counts,
                )
            if max_steps is not None and step_counter >= max_steps:
                stop_run = True
                break
            pose_elapsed += live_time
            if pose_elapsed >= estimator.pf_config.max_dwell_time_s:
                break
        if save_outputs and estimator.measurements and estimator.measurements[-1].pose_idx == current_pose_idx:
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

    # Save final snapshots
    if save_outputs:
        pf_out_path = RESULTS_DIR / "result_pf.png"
        spectrum_out_path = RESULTS_DIR / "result_spectrum.png"
        estimates_out_path = RESULTS_DIR / "result_estimates.png"
        pf_out_path.parent.mkdir(parents=True, exist_ok=True)
        viz.save_final(pf_out_path.as_posix())
        viz.save_estimates_only(estimates_out_path.as_posix())
        if last_spectrum is not None:
            highlight = (
                set(locked_isotopes_for_planning)
                if detection_locked
                else last_candidates
            )
            _save_spectrum_plot(
                decomposer,
                last_spectrum,
                spectrum_out_path,
                highlight_isotopes=highlight,
                counts_by_isotope=last_counts,
            )
    total_meas_time = step_counter * live_time
    if save_outputs:
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
    estimates = estimator.pruned_estimates(
        method="legacy",
        tau_mix=PRUNE_TAU_MIX,
        min_support=PRUNE_MIN_SUPPORT,
        min_obs_count=PRUNE_MIN_OBS_COUNT,
        min_strength_abs=PRUNE_MIN_STRENGTH_ABS,
        min_strength_ratio=PRUNE_MIN_STRENGTH_RATIO,
    )
    est_by_iso: dict[str, list[dict[str, float | list[float]]]] = {}
    for iso, estimate in estimates.items():
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
    if return_state:
        return estimator
    return None


def run_realtime_pf() -> None:
    """Entry point for real-time PF + visualization with built-in demo settings."""
    run_live_pf(live=True, max_steps=10)
