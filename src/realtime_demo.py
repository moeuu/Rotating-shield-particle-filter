"""Real-time demo for the rotating-shield particle filter with visualization."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time

import matplotlib


def _has_display() -> bool:
    """Return True when a GUI display is likely available."""
    if sys.platform.startswith("linux"):
        return bool(
            os.environ.get("DISPLAY")
            or os.environ.get("WAYLAND_DISPLAY")
            or os.environ.get("MIR_SOCKET")
        )
    return True


def _argv_requests_cui(argv: list[str] | None = None) -> bool:
    """Return True when command-line arguments request non-interactive CUI mode."""
    args = sys.argv[1:] if argv is None else argv
    if "--matplotlib-live" in args:
        return False

    def _is_run_mode_value(value: str) -> bool:
        """Return True when a CLI value names a non-Matplotlib run mode."""
        mode = value.strip().lower()
        return mode in {"gui", "cui"} or mode.endswith(("-gui", "-cui"))

    for index, arg in enumerate(args):
        if arg in {
            "--headless",
            "--no-live",
            "--cui",
            "--python-gui",
            "--geant4-isaacsim-gui",
            "--python-cui",
            "--geant4-cui",
        }:
            return True
        if arg in {"--mode", "--ui-mode"}:
            if index + 1 >= len(args):
                continue
            if _is_run_mode_value(args[index + 1]):
                return True
        if arg.startswith("--mode=") and _is_run_mode_value(arg.split("=", 1)[1]):
            return True
        if (
            arg.startswith("--ui-mode=")
            and _is_run_mode_value(arg.split("=", 1)[1])
        ):
            return True
    return False


def _configure_matplotlib() -> None:
    """Configure matplotlib backend for interactive or headless use."""
    headless = _argv_requests_cui()
    if headless or not _has_display():
        matplotlib.use("Agg")
        return
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")


_configure_matplotlib()

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from measurement.model import EnvironmentConfig, PointSource
from measurement.obstacles import build_obstacle_grid
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
from planning.candidate_generation import generate_candidate_poses
from planning.pose_selection import (
    DEFAULT_PLANNING_ROLLOUTS,
    select_next_pose_from_candidates,
)
from planning.traversability import (
    TraversabilityMap,
    build_traversability_map_from_obstacle_grid,
    render_traversability_map,
)
from visualization.realtime_viz import (
    DEFAULT_ISOTOPE_COLORS,
    PFFrame,
    RealTimePFVisualizer,
    build_frame_from_pf,
)
from visualization.ig_shield_geometry import render_octant_grid
from evaluation_metrics import compute_metrics, print_metrics_report
from piplup_notify import PiplupNotificationConfig, PiplupNotifier
from sim import (
    SimulationCommand,
    SimulationObservation,
    SimulationRuntime,
    create_simulation_runtime,
    load_runtime_config,
)
from sim.blender_environment import generate_blender_environment_usd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
SPECTRUM_DIR = RESULTS_DIR / "spectrum"
PF_DIR = RESULTS_DIR / "pf"
IG_DIR = RESULTS_DIR / "IG"
BLENDER_ENV_DIR = RESULTS_DIR / "blender_environments"
SAVE_IG_GRIDS = False
OBSTACLE_LAYOUT_DIR = ROOT / "obstacle_layouts"
PRUNE_MIN_STRENGTH_ABS = 5.0
PRUNE_MIN_STRENGTH_RATIO = 0.001
PRUNE_TAU_MIX = 0.6
PRUNE_METHOD = "legacy"
PRUNE_DELTALL_MIN = 0.0
FINAL_ESTIMATE_MIN_STRENGTH_ABS = 500.0
FINAL_MERGE_DISTANCE_M = 1.5
PRUNE_MIN_SUPPORT = 2
PRUNE_MIN_OBS_COUNT = 0.0
PRUNE_MIN_MEASUREMENTS = 10
DETECT_MIN_PEAKS_BY_ISOTOPE = {"Eu-154": 2, "Co-60": 2}
DETECT_REL_THRESH_BY_ISOTOPE = {"Co-60": 0.1}
DETECT_CONSECUTIVE_BY_ISOTOPE = {"Cs-137": 3, "Co-60": 3, "Eu-154": 5}
DETECT_MISS_AFTER_LOCK = 30
DEFAULT_SOURCE_CONFIG = ROOT / "source_layouts" / "demo_sources.json"
DEFAULT_OBSTACLE_CONFIG = OBSTACLE_LAYOUT_DIR / "demo_obstacles.json"
CANDIDATE_GRID_SPACING = (0.5, 0.5, 0.5)
CANDIDATE_GRID_MARGIN = 0.5
HEALTH_LOG_TOP_K = 3
ADAPTIVE_STEP_ID_STRIDE = 100000


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


def _resolve_config_relative_path(
    path_value: object,
    config_path: str | None,
) -> Path | None:
    """Resolve a config path value relative to its JSON file."""
    if path_value in (None, ""):
        return None
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    if config_path:
        base_path = Path(config_path).expanduser()
        if not base_path.is_absolute():
            base_path = (ROOT / base_path).resolve()
        return (base_path.parent / path).resolve()
    return (ROOT / path).resolve()


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


def _fmt_counts(counts: dict[str, float] | None) -> str:
    """Format a count dict for logging."""
    if counts is None:
        return "{}"
    items = ", ".join(f"{iso}: {float(val):.1f}" for iso, val in sorted(counts.items()))
    return "{" + items + "}"


def _thin_spectrum_for_notification(
    energy_keV: NDArray[np.float64],
    counts: NDArray[np.float64],
    max_bins: int,
) -> tuple[list[float], list[float]]:
    """Return spectrum arrays thinned to a notification-friendly size."""
    energy = np.asarray(energy_keV, dtype=float).reshape(-1)
    values = np.asarray(counts, dtype=float).reshape(-1)
    if energy.size != values.size:
        size = min(energy.size, values.size)
        energy = energy[:size]
        values = values[:size]
    limit = int(max_bins)
    if limit > 0 and values.size > limit:
        nonzero = np.flatnonzero(values > 0.0)
        if nonzero.size >= limit:
            ranked = nonzero[np.argsort(values[nonzero])[-limit:]]
            indices = np.sort(ranked)
        else:
            base = np.linspace(0, values.size - 1, limit, dtype=int)
            indices = np.unique(np.concatenate([base, nonzero]))
            if indices.size > limit:
                ranked = indices[np.argsort(values[indices])[-limit:]]
                indices = np.sort(ranked)
        energy = energy[indices]
        values = values[indices]
    return (
        [round(float(value), 3) for value in energy],
        [round(float(value), 6) for value in values],
    )


def _build_spectrum_notification_payload(
    *,
    decomposer: SpectralDecomposer,
    spectrum: NDArray[np.float64],
    step_index: int,
    pose_xyz: NDArray[np.float64],
    fe_index: int,
    pb_index: int,
    live_time_s: float,
    counts_by_isotope: dict[str, float],
    detected_isotopes: set[str],
    count_method: str,
    max_bins: int,
) -> dict[str, object]:
    """Build a compact spectrum payload for piplup/Railway display."""
    spectrum_values = np.asarray(spectrum, dtype=float)
    energy_keV, spectrum_counts = _thin_spectrum_for_notification(
        np.asarray(decomposer.energy_axis, dtype=float),
        spectrum_values,
        max_bins,
    )
    return {
        "step_index": int(step_index),
        "pose_xyz": [float(v) for v in np.asarray(pose_xyz, dtype=float)],
        "fe_index": int(fe_index),
        "pb_index": int(pb_index),
        "live_time_s": float(live_time_s),
        "count_method": str(count_method),
        "counts_by_isotope": {
            iso: float(value) for iso, value in sorted(counts_by_isotope.items())
        },
        "count_variance_by_isotope": {
            iso: float(value)
            for iso, value in sorted(decomposer.last_count_variances.items())
        },
        "detected_isotopes": sorted(detected_isotopes),
        "total_spectrum_counts": float(np.sum(spectrum_values)),
        "max_bin_count": float(np.max(spectrum_values)) if spectrum_values.size else 0.0,
        "energy_keV": energy_keV,
        "spectrum_counts": spectrum_counts,
    }


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
        filt = estimator.filters.get(iso)
        ess_pre = float(stats["ess_pre"])
        resampled = bool(stats["resampled"])
        ess_post = stats["ess_post"]
        n_after_adapt = int(stats["n_after_adapt"])
        resamples = int(stats["resample_count"])
        births = int(stats["birth_count"])
        kills = int(stats["kill_count"])
        temper_steps = stats.get("temper_steps", [])
        temper_resamples = int(stats.get("temper_resamples", 0))
        r_mean = float(stats["r_mean"])
        r_var = float(stats["r_var"])
        map_pos, map_str = stats["map"]
        mmse_pos, mmse_str = stats["mmse"]
        top_entries = stats["top_k"]
        converged = bool(stats.get("converged", False))
        updates_skipped = int(stats.get("updates_skipped", 0))
        birth_enabled = bool(getattr(getattr(filt, "config", None), "birth_enable", False))
        max_sources = getattr(getattr(filt, "config", None), "max_sources", None)
        p_birth = float(getattr(getattr(filt, "config", None), "p_birth", 0.0))
        print(
            f"[step {step_index}] pf[{iso}] ess_pre={ess_pre:.2f} resampled={resampled} "
            f"ess_post={_fmt_optional_float(ess_post)} n_after={n_after_adapt} "
            f"resamples={resamples} births={births} kills={kills} "
            f"r_mean={r_mean:.2f} r_var={r_var:.2f} "
            f"converged={converged} skipped={updates_skipped} "
            f"birth_enabled={birth_enabled} max_sources={max_sources} p_birth={p_birth:.3f}"
        )
        if temper_steps:
            temper_str = ", ".join(
                f"(beta={s['beta_total']:.3f},db={s['delta_beta']:.3f},ess={s['ess']:.1f})"
                for s in temper_steps
            )
            print(
                f"[step {step_index}] pf[{iso}] temper={temper_str} "
                f"temper_resamples={temper_resamples}"
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


def _coerce_live_visualization(live: bool) -> bool:
    """Return whether live Matplotlib updates can run in this process."""
    if not live:
        return False
    backend = str(matplotlib.get_backend()).lower()
    if "agg" in backend or not _has_display():
        print("GUI display unavailable; running in CUI/headless mode.")
        return False
    return True


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
    title: str = "Final measurement spectrum",
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
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if library:
        ax.legend(loc="upper right", fontsize=8, title="Nuclide lines")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _validate_measurement_timing(
    *,
    measurement_time_s: float,
    adaptive_dwell: bool,
    adaptive_dwell_chunk_s: float,
    adaptive_min_dwell_s: float,
    adaptive_ready_min_counts: float,
    adaptive_ready_min_isotopes: int,
) -> None:
    """Validate fixed and adaptive dwell-time settings."""
    if measurement_time_s <= 0.0:
        raise ValueError("measurement_time_s must be positive.")
    if not adaptive_dwell:
        return
    if adaptive_dwell_chunk_s <= 0.0:
        raise ValueError("adaptive_dwell_chunk_s must be positive.")
    if adaptive_min_dwell_s <= 0.0:
        raise ValueError("adaptive_min_dwell_s must be positive.")
    if adaptive_min_dwell_s > measurement_time_s:
        raise ValueError("adaptive_min_dwell_s cannot exceed measurement_time_s.")
    if adaptive_ready_min_counts < 0.0:
        raise ValueError("adaptive_ready_min_counts cannot be negative.")
    if adaptive_ready_min_isotopes < 0:
        raise ValueError("adaptive_ready_min_isotopes cannot be negative.")


def _observation_spectrum_array(
    observation: SimulationObservation,
    decomposer: SpectralDecomposer,
) -> NDArray[np.float64]:
    """Return a validated spectrum array from a simulator observation."""
    spectrum = np.asarray(observation.spectrum_counts, dtype=float)
    if spectrum.shape != decomposer.energy_axis.shape:
        raise ValueError(
            "Simulator returned an unexpected spectrum shape: "
            f"{spectrum.shape} != {decomposer.energy_axis.shape}"
        )
    return spectrum


def _metadata_spectrum_variance(
    metadata: dict[str, object],
    expected_shape: tuple[int, ...],
) -> NDArray[np.float64] | None:
    """Return a validated per-bin spectrum variance array from metadata."""
    raw = metadata.get("spectrum_count_variance")
    if raw is None:
        return None
    variance = np.asarray(raw, dtype=float)
    if variance.shape != expected_shape:
        return None
    return np.clip(variance, a_min=0.0, a_max=None)


def _evaluate_spectrum_counts(
    decomposer: SpectralDecomposer,
    spectrum: NDArray[np.float64],
    *,
    live_time_s: float,
    spectrum_count_method: str,
    detect_threshold_abs: float,
    detect_threshold_rel: float,
    detect_threshold_rel_by_isotope: dict[str, float],
    min_peaks_by_isotope: dict[str, int] | None,
    spectrum_variance: NDArray[np.float64] | None = None,
) -> tuple[dict[str, float], dict[str, float], set[str]]:
    """Extract isotope counts, count variances, and detected labels."""
    counts, detected = decomposer.isotope_counts_with_detection(
        spectrum,
        live_time_s=live_time_s,
        count_method=spectrum_count_method,
        active_isotopes=None,
        detect_threshold_abs=detect_threshold_abs,
        detect_threshold_rel=detect_threshold_rel,
        detect_threshold_rel_by_isotope=detect_threshold_rel_by_isotope,
        min_peaks_by_isotope=min_peaks_by_isotope,
    )
    counts_out = {iso: float(val) for iso, val in counts.items()}
    variances = {
        iso: float(max(decomposer.last_count_variances.get(iso, max(val, 1.0)), 1.0))
        for iso, val in counts_out.items()
    }
    if spectrum_variance is not None:
        variance_floor = decomposer.estimate_count_variances_from_spectrum_variance(
            spectrum_variance,
            isotopes=list(counts_out.keys()),
        )
        variances = {
            iso: float(max(variances.get(iso, 1.0), variance_floor.get(iso, 1.0)))
            for iso in counts_out
        }
    return counts_out, variances, set(detected)


def _is_adaptive_spectrum_ready(
    counts_by_isotope: dict[str, float],
    detected_isotopes: set[str],
    *,
    live_time_s: float,
    min_live_time_s: float,
    min_counts_per_detected_isotope: float,
    min_detected_isotopes: int,
) -> tuple[bool, str]:
    """Return whether an accumulated spectrum is usable for isotope counts."""
    if live_time_s + 1e-12 < min_live_time_s:
        return False, "below_min_live_time"
    if min_detected_isotopes <= 0:
        return True, "min_live_time_reached"
    usable = [
        iso
        for iso in detected_isotopes
        if float(counts_by_isotope.get(iso, 0.0)) >= min_counts_per_detected_isotope
    ]
    if len(usable) >= min_detected_isotopes:
        return True, "detected_isotope_counts_ready"
    return (
        False,
        "insufficient_detected_isotope_counts:"
        f"{len(usable)}/{int(min_detected_isotopes)}",
    )


def _merge_adaptive_observation_chunks(
    *,
    logical_step_id: int,
    observations: list[SimulationObservation],
    chunk_live_times_s: list[float],
    ready_reason: str,
    counts_by_isotope: dict[str, float],
    count_variance_by_isotope: dict[str, float],
    detected_isotopes: set[str],
) -> SimulationObservation:
    """Combine multiple simulator observations into one logical measurement."""
    if not observations:
        raise ValueError("At least one observation chunk is required.")
    first = observations[0]
    edge_ref = np.asarray(first.energy_bin_edges_keV, dtype=float)
    spectrum_total = np.zeros_like(np.asarray(first.spectrum_counts, dtype=float))
    spectrum_variance_total = np.zeros_like(spectrum_total, dtype=float)
    has_spectrum_variance = False
    for observation in observations:
        edges = np.asarray(observation.energy_bin_edges_keV, dtype=float)
        if edges.shape != edge_ref.shape or not np.allclose(edges, edge_ref):
            raise ValueError("Adaptive dwell chunks returned inconsistent energy bins.")
        spectrum_chunk = np.asarray(observation.spectrum_counts, dtype=float)
        spectrum_total += spectrum_chunk
        chunk_variance = _metadata_spectrum_variance(
            observation.metadata,
            spectrum_chunk.shape,
        )
        if chunk_variance is not None:
            spectrum_variance_total += chunk_variance
            has_spectrum_variance = True
    metadata = dict(observations[-1].metadata)
    metadata.update(
        {
            "adaptive_dwell": True,
            "adaptive_dwell_chunks": int(len(observations)),
            "adaptive_dwell_child_step_ids": [
                int(observation.step_id) for observation in observations
            ],
            "adaptive_dwell_chunk_live_times_s": [
                float(value) for value in chunk_live_times_s
            ],
            "adaptive_dwell_live_time_s": float(sum(chunk_live_times_s)),
            "adaptive_dwell_ready_reason": str(ready_reason),
            "adaptive_dwell_detected_isotopes": sorted(detected_isotopes),
            "adaptive_dwell_counts_by_isotope": {
                iso: float(value) for iso, value in counts_by_isotope.items()
            },
            "adaptive_dwell_count_variance_by_isotope": {
                iso: float(value) for iso, value in count_variance_by_isotope.items()
            },
        }
    )
    if has_spectrum_variance:
        metadata["spectrum_count_variance"] = spectrum_variance_total.tolist()
        metadata["spectrum_count_variance_total"] = float(np.sum(spectrum_variance_total))
    return SimulationObservation(
        step_id=int(logical_step_id),
        detector_pose_xyz=first.detector_pose_xyz,
        detector_quat_wxyz=first.detector_quat_wxyz,
        fe_orientation_index=first.fe_orientation_index,
        pb_orientation_index=first.pb_orientation_index,
        spectrum_counts=spectrum_total.tolist(),
        energy_bin_edges_keV=edge_ref.tolist(),
        metadata=metadata,
    )


def _acquire_spectrum_observation(
    *,
    simulation_runtime: SimulationRuntime,
    decomposer: SpectralDecomposer,
    step_id: int,
    pose_xyz: NDArray[np.float64],
    fe_idx: int,
    pb_idx: int,
    live_time_s: float,
    travel_time_s: float,
    shield_actuation_time_s: float,
    adaptive_dwell: bool,
    adaptive_dwell_chunk_s: float,
    adaptive_min_dwell_s: float,
    adaptive_ready_min_counts: float,
    adaptive_ready_min_isotopes: int,
    spectrum_count_method: str,
    detect_threshold_abs: float,
    detect_threshold_rel: float,
    detect_threshold_rel_by_isotope: dict[str, float],
    min_peaks_by_isotope: dict[str, int] | None,
) -> tuple[
    SimulationObservation,
    float,
    dict[str, float],
    dict[str, float],
    set[str],
    str,
    int,
]:
    """Acquire one logical spectrum, optionally stopping adaptive dwell early."""
    target_pose = tuple(float(v) for v in pose_xyz)
    if not adaptive_dwell:
        observation = simulation_runtime.step(
            SimulationCommand(
                step_id=int(step_id),
                target_pose_xyz=target_pose,
                target_base_yaw_rad=0.0,
                fe_orientation_index=int(fe_idx),
                pb_orientation_index=int(pb_idx),
                dwell_time_s=float(live_time_s),
                travel_time_s=float(travel_time_s),
                shield_actuation_time_s=float(shield_actuation_time_s),
            )
        )
        spectrum = _observation_spectrum_array(observation, decomposer)
        spectrum_variance = _metadata_spectrum_variance(
            observation.metadata,
            spectrum.shape,
        )
        counts, variances, detected = _evaluate_spectrum_counts(
            decomposer,
            spectrum,
            live_time_s=float(live_time_s),
            spectrum_count_method=spectrum_count_method,
            detect_threshold_abs=detect_threshold_abs,
            detect_threshold_rel=detect_threshold_rel,
            detect_threshold_rel_by_isotope=detect_threshold_rel_by_isotope,
            min_peaks_by_isotope=min_peaks_by_isotope,
            spectrum_variance=spectrum_variance,
        )
        return (
            observation,
            float(live_time_s),
            counts,
            variances,
            detected,
            "fixed_dwell",
            1,
        )

    observations: list[SimulationObservation] = []
    chunk_live_times_s: list[float] = []
    accumulated_spectrum = np.zeros_like(decomposer.energy_axis, dtype=float)
    accumulated_spectrum_variance = np.zeros_like(decomposer.energy_axis, dtype=float)
    has_spectrum_variance = False
    accumulated_live_time_s = 0.0
    last_counts: dict[str, float] = {}
    last_variances: dict[str, float] = {}
    last_detected: set[str] = set()
    ready_reason = "max_dwell_reached"
    chunk_index = 0
    while accumulated_live_time_s + 1e-12 < live_time_s:
        remaining_s = float(live_time_s) - accumulated_live_time_s
        chunk_live_time_s = min(float(adaptive_dwell_chunk_s), remaining_s)
        internal_step_id = int(step_id) * ADAPTIVE_STEP_ID_STRIDE + int(chunk_index)
        observation = simulation_runtime.step(
            SimulationCommand(
                step_id=internal_step_id,
                target_pose_xyz=target_pose,
                target_base_yaw_rad=0.0,
                fe_orientation_index=int(fe_idx),
                pb_orientation_index=int(pb_idx),
                dwell_time_s=chunk_live_time_s,
                travel_time_s=float(travel_time_s) if chunk_index == 0 else 0.0,
                shield_actuation_time_s=(
                    float(shield_actuation_time_s) if chunk_index == 0 else 0.0
                ),
            )
        )
        spectrum = _observation_spectrum_array(observation, decomposer)
        spectrum_variance = _metadata_spectrum_variance(
            observation.metadata,
            spectrum.shape,
        )
        observations.append(observation)
        chunk_live_times_s.append(chunk_live_time_s)
        accumulated_spectrum += spectrum
        if spectrum_variance is not None:
            accumulated_spectrum_variance += spectrum_variance
            has_spectrum_variance = True
        accumulated_live_time_s += chunk_live_time_s
        last_counts, last_variances, last_detected = _evaluate_spectrum_counts(
            decomposer,
            accumulated_spectrum,
            live_time_s=accumulated_live_time_s,
            spectrum_count_method=spectrum_count_method,
            detect_threshold_abs=detect_threshold_abs,
            detect_threshold_rel=detect_threshold_rel,
            detect_threshold_rel_by_isotope=detect_threshold_rel_by_isotope,
            min_peaks_by_isotope=min_peaks_by_isotope,
            spectrum_variance=(
                accumulated_spectrum_variance if has_spectrum_variance else None
            ),
        )
        ready, reason = _is_adaptive_spectrum_ready(
            last_counts,
            last_detected,
            live_time_s=accumulated_live_time_s,
            min_live_time_s=float(adaptive_min_dwell_s),
            min_counts_per_detected_isotope=float(adaptive_ready_min_counts),
            min_detected_isotopes=int(adaptive_ready_min_isotopes),
        )
        ready_reason = reason
        if ready:
            break
        chunk_index += 1
    reached_dwell_cap = accumulated_live_time_s + 1e-12 >= live_time_s
    if reached_dwell_cap and not ready_reason.startswith("detected_"):
        ready_reason = f"max_dwell_reached:{ready_reason}"
    observation = _merge_adaptive_observation_chunks(
        logical_step_id=int(step_id),
        observations=observations,
        chunk_live_times_s=chunk_live_times_s,
        ready_reason=ready_reason,
        counts_by_isotope=last_counts,
        count_variance_by_isotope=last_variances,
        detected_isotopes=last_detected,
    )
    return (
        observation,
        float(accumulated_live_time_s),
        last_counts,
        last_variances,
        last_detected,
        ready_reason,
        len(observations),
    )


def run_live_pf(
    live: bool = True,
    max_steps: int | None = None,
    max_poses: int | None = 15,
    sources: list[PointSource] | None = None,
    detect_threshold_abs: float = 50.0,
    detect_threshold_rel: float = 0.3,
    detect_consecutive: int = 10,
    detect_min_steps: int | None = None,
    min_peaks_by_isotope: dict[str, int] | None = None,
    ig_threshold_mode: str = "relative_pose",
    ig_threshold_rel: float = 0.02,
    ig_threshold_min: float | None = None,
    environment_mode: str = "fixed",
    obstacle_layout_path: str | None = DEFAULT_OBSTACLE_CONFIG.as_posix(),
    obstacle_seed: int | None = None,
    eval_match_radius_m: float = 0.5,
    candidate_grid_spacing: tuple[float, float, float] | None = None,
    candidate_grid_margin: float = CANDIDATE_GRID_MARGIN,
    birth_enabled: bool = False,
    num_particles: int = 2000,
    pf_config_overrides: dict[str, object] | None = None,
    save_outputs: bool = True,
    output_tag: str | None = None,
    pose_candidates: int = 64,
    pose_min_dist: float = 3.0,
    return_state: bool = False,
    converge: bool = False,
    sim_backend: str = "analytic",
    sim_config_path: str | None = None,
    blender_executable: str | None = None,
    blender_output_path: str | None = None,
    blender_timeout_s: float = 120.0,
    passage_width_m: float = 1.0,
    robot_radius_m: float = 0.35,
    nominal_motion_speed_m_s: float = 0.5,
    rotation_overhead_s: float = 0.5,
    measurement_time_s: float = 30.0,
    adaptive_dwell: bool = False,
    adaptive_dwell_chunk_s: float = 2.0,
    adaptive_min_dwell_s: float = 2.0,
    adaptive_ready_min_counts: float = 100.0,
    adaptive_ready_min_isotopes: int = 1,
    adaptive_strength_prior: bool = True,
    adaptive_strength_prior_steps: int = 3,
    adaptive_strength_prior_min_counts: float = 3.0,
    adaptive_strength_prior_log_sigma: float = 0.7,
    pose_min_observation_counts: float | None = None,
    pose_min_observation_penalty_scale: float = 1.0,
    pose_min_observation_aggregate: str = "max",
    notification_config: PiplupNotificationConfig | None = None,
    notify_spectrum: bool = False,
    notify_spectrum_every: int = 1,
    notify_spectrum_max_bins: int = 800,
) -> RotatingShieldPFEstimator | None:
    """
    Run a simple PF loop with live visualization (active pose/orientation selection).

    If max_steps is None, run until the information-gain threshold is met.
    If max_poses is None, run without a pose-count limit.
    If obstacle_layout_path is provided, blocked grid cells are excluded and shown
    in black.

    Args:
        pf_config_overrides: Optional overrides applied to the PF configuration.
        save_outputs: When False, skip writing plots and snapshot images.
        output_tag: Optional tag appended to result output filenames.
        pose_candidates: Number of pose candidates to generate per step.
        pose_min_dist: Minimum distance from visited poses for candidates (meters).
        return_state: When True, return the estimator for inspection/testing.
        candidate_grid_spacing: Optional (x, y, z) spacing for birth candidate grid.
        candidate_grid_margin: Margin from the environment bounds for candidate sources.
        birth_enabled: Enable birth/death/split/merge moves.
        num_particles: Particle count used by each isotope filter.
        converge: Enable per-isotope convergence gating.
        environment_mode: Obstacle environment mode ("fixed" or "random").
        sim_backend: Simulation backend name ("analytic", "isaacsim", or "geant4").
        sim_config_path: Optional JSON config for the selected simulation backend.
        blender_executable: Optional Blender executable path for random mode.
        blender_output_path: Optional USD path written by Blender in random mode.
        blender_timeout_s: Timeout for Blender environment generation.
        passage_width_m: Minimum reserved corridor width in random mode.
        robot_radius_m: Robot footprint radius used for 2D traversability maps.
        nominal_motion_speed_m_s: Nominal robot speed used for mission-time estimates.
        rotation_overhead_s: Fixed shield-actuation overhead per measurement.
        measurement_time_s: Fixed dwell time or adaptive maximum dwell time.
        adaptive_dwell: Stop each measurement once isotope counts are reliable enough.
        adaptive_dwell_chunk_s: Geant4 dwell duration for each adaptive chunk.
        adaptive_min_dwell_s: Minimum accumulated dwell before early stopping.
        adaptive_ready_min_counts: Minimum count estimate per detected isotope.
        adaptive_ready_min_isotopes: Required number of detected isotopes for readiness.
        adaptive_strength_prior: Rescale early PF strengths from observed counts.
        adaptive_strength_prior_steps: Number of first measurements used for strength rescaling.
        adaptive_strength_prior_min_counts: Count floor for zero/weak observations.
        adaptive_strength_prior_log_sigma: Proposal spread around count-matched strengths.
        pose_min_observation_counts: Minimum posterior-predicted counts per isotope
            used as a soft pose-selection constraint; None uses runtime config
            or adaptive_strength_prior_min_counts.
        pose_min_observation_penalty_scale: Relative weight of the pose
            observability soft constraint.
        pose_min_observation_aggregate: Orientation aggregation for pose
            observability ("max" or "mean").
        notification_config: Optional piplup-notify delivery settings.
        notify_spectrum: Send per-measurement spectrum events through piplup.
        notify_spectrum_every: Send one spectrum event every N measurements.
        notify_spectrum_max_bins: Maximum number of spectrum bins per event.
    """
    _validate_measurement_timing(
        measurement_time_s=float(measurement_time_s),
        adaptive_dwell=bool(adaptive_dwell),
        adaptive_dwell_chunk_s=float(adaptive_dwell_chunk_s),
        adaptive_min_dwell_s=float(adaptive_min_dwell_s),
        adaptive_ready_min_counts=float(adaptive_ready_min_counts),
        adaptive_ready_min_isotopes=int(adaptive_ready_min_isotopes),
    )
    notifier = PiplupNotifier(notification_config)
    live = _coerce_live_visualization(live)
    runtime_config = load_runtime_config(sim_config_path)
    env = EnvironmentConfig(size_x=10.0, size_y=20.0, size_z=10.0, detector_position=(1.0, 1.0, 0.5))
    sources = _build_demo_sources() if sources is None else sources
    decomposer = SpectralDecomposer()
    default_count_method = "photopeak_nnls"
    spectrum_count_method = str(
        runtime_config.get("spectrum_count_method", default_count_method)
    ).strip().lower()
    allowed_runtime_count_methods = {"photopeak_nnls", "response_poisson"}
    if spectrum_count_method not in allowed_runtime_count_methods:
        raise ValueError(
            "spectrum_count_method must be 'photopeak_nnls' or "
            "'response_poisson' for runtime simulations."
        )
    if min_peaks_by_isotope is None:
        min_peaks_by_isotope = dict(DETECT_MIN_PEAKS_BY_ISOTOPE)
    detect_threshold_rel_by_isotope = dict(DETECT_REL_THRESH_BY_ISOTOPE)
    obstacle_grid = None
    normalized_environment_mode = environment_mode.strip().lower()
    if normalized_environment_mode not in {"fixed", "random"}:
        raise ValueError(f"Unknown environment_mode: {environment_mode}")
    if obstacle_layout_path is not None:
        obstacle_path: Path | None = None
        if obstacle_layout_path:
            obstacle_path = Path(obstacle_layout_path)
            if not obstacle_path.is_absolute():
                obstacle_path = (ROOT / obstacle_path).resolve()
        keep_free = None
        if env.detector_position is not None:
            keep_free = [(env.detector_position[0], env.detector_position[1])]
        obstacle_grid = build_obstacle_grid(
            mode=normalized_environment_mode,
            path=obstacle_path,
            size_x=env.size_x,
            size_y=env.size_y,
            cell_size=1.0,
            blocked_fraction=0.4,
            rng_seed=obstacle_seed,
            keep_free_points=keep_free,
            passage_width_m=(
                float(passage_width_m)
                if normalized_environment_mode == "random"
                else 0.0
            ),
        )
        mode_message = f"Obstacle environment mode: {normalized_environment_mode}"
        if normalized_environment_mode == "fixed" and obstacle_path is not None:
            mode_message += f" ({obstacle_path})"
        if obstacle_seed is not None:
            mode_message += f", seed={int(obstacle_seed)}"
        if normalized_environment_mode == "random":
            mode_message += f", passage_width_m={float(passage_width_m):.2f}"
        mode_message += f", blocked_fraction={obstacle_grid.blocked_fraction:.3f}"
        print(mode_message)
    normals = generate_octant_orientations()
    rot_mats = generate_octant_rotation_matrices()
    num_orients = len(rot_mats)
    if save_outputs:
        PF_DIR.mkdir(parents=True, exist_ok=True)
    output_suffix = ""
    cleaned_tag = ""
    if output_tag:
        cleaned_tag = output_tag.strip().replace(" ", "_")
        cleaned_tag = cleaned_tag.replace("/", "_").replace("\\", "_")
        cleaned_tag = cleaned_tag.lstrip("_")
        if cleaned_tag:
            output_suffix = f"_{cleaned_tag}"
    generated_blender_usd_path: Path | None = None
    traversability_map: TraversabilityMap | None = None
    traversability_map_path: Path | None = None
    traversability_map_png_path: Path | None = None
    if obstacle_grid is not None and normalized_environment_mode == "random":
        if blender_output_path:
            generated_output_path = Path(blender_output_path)
            if not generated_output_path.is_absolute():
                generated_output_path = (ROOT / generated_output_path).resolve()
        else:
            if obstacle_seed is None:
                path_token = f"random_{int(time.time() * 1000)}"
            else:
                path_token = f"random_seed_{int(obstacle_seed)}"
            if cleaned_tag:
                path_token = f"{path_token}_{cleaned_tag}"
            generated_output_path = BLENDER_ENV_DIR / f"{path_token}.usda"
        base_usd_path = _resolve_config_relative_path(
            runtime_config.get(
                "random_environment_base_usd_path",
                runtime_config.get("usd_path"),
            ),
            sim_config_path,
        )
        traversability_map_path = generated_output_path.with_suffix(
            ".traversability.json"
        )
        traversability_map_png_path = generated_output_path.with_suffix(
            ".traversability.png"
        )
        generated_blender_usd_path = generate_blender_environment_usd(
            grid=obstacle_grid,
            output_path=generated_output_path,
            room_size_xyz=(env.size_x, env.size_y, env.size_z),
            obstacle_height_m=float(runtime_config.get("obstacle_height_m", 2.0)),
            obstacle_material="concrete",
            base_usd_path=base_usd_path,
            traversability_output_path=traversability_map_path,
            robot_radius_m=float(robot_radius_m),
            traversability_reachable_from_xy=(
                float(env.detector_position[0]),
                float(env.detector_position[1]),
            ),
            blender_executable=blender_executable,
            timeout_s=blender_timeout_s,
        )
        print(f"Generated Blender random environment: {generated_blender_usd_path}")
        if traversability_map_path.exists():
            traversability_map = TraversabilityMap.load(traversability_map_path)
        else:
            traversability_map = build_traversability_map_from_obstacle_grid(
                obstacle_grid,
                robot_radius_m=float(robot_radius_m),
                reachable_from=env.detector_position,
            )
            traversability_map.save(traversability_map_path)
        render_traversability_map(traversability_map, traversability_map_png_path)
        print(
            "Generated 2D robot traversability map: "
            f"{traversability_map_path} "
            f"(free_fraction={traversability_map.traversable_fraction:.3f}, "
            f"robot_radius_m={float(robot_radius_m):.2f})"
        )
    planning_map = traversability_map if traversability_map is not None else obstacle_grid

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
    active_isotopes: set[str] = set()
    last_candidates: set[str] = set()
    num_particles = max(1, int(num_particles))
    shield_params = ShieldParams()
    mu_by_isotope = mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM, isotopes=isotopes)
    if not mu_by_isotope:
        mu_by_isotope = {
            iso: {"fe": shield_params.mu_fe, "pb": shield_params.mu_pb} for iso in isotopes
        }
    use_gpu = _default_use_gpu()
    background_by_isotope = {iso: 5.0 for iso in isotopes}
    live_time = float(measurement_time_s)
    if pose_min_observation_counts is None:
        pose_min_observation_counts_resolved = runtime_config.get(
            "pose_min_observation_counts",
            adaptive_strength_prior_min_counts,
        )
    else:
        pose_min_observation_counts_resolved = pose_min_observation_counts
    pose_min_observation_counts_resolved = max(
        float(pose_min_observation_counts_resolved),
        0.0,
    )
    pose_min_observation_penalty_scale = max(
        float(
            runtime_config.get(
                "pose_min_observation_penalty_scale",
                pose_min_observation_penalty_scale,
            )
        ),
        0.0,
    )
    pose_min_observation_aggregate = str(
        runtime_config.get(
            "pose_min_observation_aggregate",
            pose_min_observation_aggregate,
        )
    ).strip().lower()
    pose_min_observation_max_particles = runtime_config.get(
        "pose_min_observation_max_particles",
        None,
    )
    if pose_min_observation_max_particles is not None:
        pose_min_observation_max_particles = int(pose_min_observation_max_particles)
    likelihood_runtime = runtime_config.get("pf_count_likelihood", {})
    if not isinstance(likelihood_runtime, dict):
        likelihood_runtime = {}
    geant4_likelihood_defaults = sim_backend.strip().lower() == "geant4"
    spectrum_estimate_likelihood_defaults = spectrum_count_method in {
        "photopeak_nnls",
        "response_poisson",
    }

    def _likelihood_config_value(key: str, default: object) -> object:
        """Read a PF likelihood setting from nested or legacy runtime config keys."""
        legacy_key = f"pf_{key}"
        if key in likelihood_runtime:
            return likelihood_runtime[key]
        return runtime_config.get(legacy_key, default)

    count_likelihood_model = str(
        _likelihood_config_value(
            "count_likelihood_model",
            "student_t"
            if geant4_likelihood_defaults or spectrum_estimate_likelihood_defaults
            else "poisson",
        )
    )
    transport_model_rel_sigma = _likelihood_config_value(
        "transport_model_rel_sigma",
        0.30 if geant4_likelihood_defaults else 0.0,
    )
    spectrum_count_rel_sigma = _likelihood_config_value(
        "spectrum_count_rel_sigma",
        0.15 if geant4_likelihood_defaults else 0.0,
    )
    spectrum_count_abs_sigma = _likelihood_config_value(
        "spectrum_count_abs_sigma",
        3.0 if geant4_likelihood_defaults else 0.0,
    )
    count_likelihood_df = float(
        _likelihood_config_value(
            "count_likelihood_df",
            5.0,
        )
    )
    simulation_runtime = create_simulation_runtime(
        sim_backend,
        sources=sources,
        decomposer=decomposer,
        mu_by_isotope=mu_by_isotope,
        shield_params=shield_params,
        runtime_config=runtime_config,
        runtime_config_path=sim_config_path,
    )
    pf_conf = RotatingShieldPFConfig(
        num_particles=num_particles,
        min_particles=num_particles,
        max_particles=num_particles,
        max_sources=1,
        resample_threshold=0.7,
        position_sigma=0.5,
        background_level=background_by_isotope,
        count_likelihood_model=count_likelihood_model,
        transport_model_rel_sigma=transport_model_rel_sigma,
        spectrum_count_rel_sigma=spectrum_count_rel_sigma,
        spectrum_count_abs_sigma=spectrum_count_abs_sigma,
        count_likelihood_df=count_likelihood_df,
        min_strength=5.0,
        p_birth=0.05,
        p_kill=0.1,
        short_time_s=live_time,
        max_dwell_time_s=10000.0,
        position_min=(0.0, 0.0, 0.0),
        position_max=(env.size_x, env.size_y, env.size_z),
        init_num_sources=(1, 1),
        init_grid_spacing_m=1.0,
        adaptive_strength_prior=bool(adaptive_strength_prior),
        adaptive_strength_prior_steps=int(adaptive_strength_prior_steps),
        adaptive_strength_prior_min_counts=float(adaptive_strength_prior_min_counts),
        adaptive_strength_prior_log_sigma=float(adaptive_strength_prior_log_sigma),
        pose_min_observation_counts=pose_min_observation_counts_resolved,
        pose_min_observation_penalty_scale=pose_min_observation_penalty_scale,
        pose_min_observation_aggregate=pose_min_observation_aggregate,
        pose_min_observation_max_particles=pose_min_observation_max_particles,
        split_prob=0.05,
        merge_prob=0.05,
        orientation_k=2,
        planning_eig_samples=50,
        planning_rollout_particles=256,
        planning_rollout_method="top_weight",
        use_fast_gpu_rollout=True,
        use_gpu=use_gpu,
        gpu_device="cuda",
        gpu_dtype="float64",
        ig_workers=1,
    )
    pf_conf.use_tempering = True
    pf_conf.max_temper_steps = 8
    pf_conf.min_delta_beta = 0.01
    pf_conf.target_ess_ratio = 0.4
    pf_conf.converge_enable = bool(converge)
    if pf_config_overrides:
        for key, value in pf_config_overrides.items():
            if not hasattr(pf_conf, key):
                raise ValueError(f"Unknown PF config override: {key}")
            setattr(pf_conf, key, value)
    pf_conf.birth_enable = bool(birth_enabled)
    if birth_enabled:
        if not pf_config_overrides or "p_birth" not in pf_config_overrides:
            if pf_conf.p_birth <= 0.0:
                pf_conf.p_birth = 0.05
        if not pf_config_overrides or "p_kill" not in pf_config_overrides:
            if pf_conf.p_kill <= 0.0:
                pf_conf.p_kill = 0.1
        if not pf_config_overrides or "split_prob" not in pf_config_overrides:
            if pf_conf.split_prob <= 0.0:
                pf_conf.split_prob = 0.05
        if not pf_config_overrides or "merge_prob" not in pf_config_overrides:
            if pf_conf.merge_prob <= 0.0:
                pf_conf.merge_prob = 0.05
    if not birth_enabled:
        pf_conf.p_birth = 0.0
        pf_conf.p_kill = 0.0
        pf_conf.split_prob = 0.0
        pf_conf.merge_prob = 0.0
        pf_conf.max_sources = 1
        if not pf_config_overrides or "init_num_sources" not in pf_config_overrides:
            pf_conf.init_num_sources = (1, 1)
    if ig_threshold_min is not None:
        pf_conf.ig_threshold = float(ig_threshold_min)
    init_support_prob = _initial_particle_nearby_probability(
        num_particles=int(pf_conf.num_particles),
        position_min=pf_conf.position_min,
        position_max=pf_conf.position_max,
        radius_m=float(eval_match_radius_m),
        init_num_sources=pf_conf.init_num_sources,
    )

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
    estimate_mode = "mmse"
    estimate_min_strength = 100.0
    estimate_min_existence_prob = None
    final_estimate_min_strength = max(
        estimate_min_strength,
        FINAL_ESTIMATE_MIN_STRENGTH_ABS,
    )
    prune_min_obs_count = PRUNE_MIN_OBS_COUNT
    if background_by_isotope:
        background_level = float(np.median(list(background_by_isotope.values())))
        prune_live_time = float(adaptive_min_dwell_s) if adaptive_dwell else live_time
        prune_min_obs_count = max(prune_min_obs_count, background_level * prune_live_time)

    def _build_estimator() -> tuple[RotatingShieldPFEstimator, NDArray[np.float64], int]:
        """Create a fresh estimator and register the initial pose."""
        estimator_local = RotatingShieldPFEstimator(
            isotopes=isotopes,
            candidate_sources=grid,
            shield_normals=normals,
            mu_by_isotope=mu_by_isotope,
            pf_config=pf_conf,
            obstacle_grid=obstacle_grid,
            obstacle_height_m=float(runtime_config.get("obstacle_height_m", 2.0)),
        )
        pose_local = np.array(env.detector_position, dtype=float)
        estimator_local.add_measurement_pose(pose_local)
        pose_idx_local = len(estimator_local.poses) - 1
        return estimator_local, pose_local, pose_idx_local

    def _build_visualizer() -> RealTimePFVisualizer:
        """Create a new PF visualizer."""
        return RealTimePFVisualizer(
            isotopes=isotopes,
            world_bounds=(0, env.size_x, 0, env.size_y, 0, env.size_z),
            true_sources=true_src,
            true_strengths=true_strengths,
            obstacle_grid=obstacle_grid,
            show_counts=False,
        )

    def _grid_centers() -> NDArray[np.float64]:
        """Return 1m grid-center positions for the environment bounds."""
        spacing = 1.0
        xs = np.arange(0.5, env.size_x, spacing)
        ys = np.arange(0.5, env.size_y, spacing)
        zs = np.arange(0.5, env.size_z, spacing)
        grid_pos = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
        return grid_pos.reshape(-1, 3)

    def _apply_display_thresholds(
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
        min_strength: float | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Filter estimates using the same min-strength cutoff as the display."""
        if min_strength is None or strengths.size == 0:
            return positions, strengths
        mask = strengths >= float(min_strength)
        return positions[mask], strengths[mask]

    def _merge_close_estimates(
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
        max_distance: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Merge nearby estimates by summing strengths and weighted-average positions.

        The merge uses a greedy agglomeration sorted by strength, combining any
        points within max_distance into a single estimate.
        """
        if positions.size == 0 or strengths.size == 0:
            return positions, strengths
        if max_distance <= 0.0:
            return positions, strengths
        order = np.argsort(strengths)[::-1]
        merged_pos: list[NDArray[np.float64]] = []
        merged_strengths: list[float] = []
        for idx in order:
            pos = positions[int(idx)]
            strength = float(strengths[int(idx)])
            merged = False
            for j, center in enumerate(merged_pos):
                if float(np.linalg.norm(pos - center)) <= max_distance:
                    total = merged_strengths[j] + strength
                    if total > 0.0:
                        merged_pos[j] = (center * merged_strengths[j] + pos * strength) / total
                    merged_strengths[j] = total
                    merged = True
                    break
            if not merged:
                merged_pos.append(pos.copy())
                merged_strengths.append(strength)
        pos_out = np.vstack(merged_pos) if merged_pos else np.zeros((0, 3), dtype=float)
        str_out = np.asarray(merged_strengths, dtype=float) if merged_strengths else np.zeros(0, dtype=float)
        return pos_out, str_out

    def _build_final_estimates(
        estimator_final: RotatingShieldPFEstimator,
        isotope_list: list[str],
        min_strength: float | None,
        min_obs_count: float,
        use_pruning: bool = True,
    ) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """
        Build final estimates using LLR pruning with a legacy fallback per isotope.

        If LLR pruning removes all sources for an isotope, fall back to legacy
        pruning. If that is also empty but raw estimates exist, keep the strongest
        raw estimate to avoid empty outputs. When use_pruning is False,
        return the raw PF estimates without pruning or thresholding.
        """
        if not use_pruning:
            raw_estimates = estimator_final.estimates()
            final_estimates: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
            for iso in isotope_list:
                pos, strg = raw_estimates.get(
                    iso, (np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float))
                )
                pos_arr = np.asarray(pos, dtype=float)
                str_arr = np.asarray(strg, dtype=float)
                final_estimates[iso] = (pos_arr, str_arr)
            return final_estimates
        llr_pruned = estimator_final.pruned_estimates(
            method="deltall",
            params={"deltaLL_min": PRUNE_DELTALL_MIN},
            tau_mix=PRUNE_TAU_MIX,
            min_support=PRUNE_MIN_SUPPORT,
            min_obs_count=min_obs_count,
            min_strength_abs=PRUNE_MIN_STRENGTH_ABS,
            min_strength_ratio=PRUNE_MIN_STRENGTH_RATIO,
        )
        legacy_pruned = estimator_final.pruned_estimates(
            method="legacy",
            params=None,
            tau_mix=PRUNE_TAU_MIX,
            min_support=PRUNE_MIN_SUPPORT,
            min_obs_count=min_obs_count,
            min_strength_abs=PRUNE_MIN_STRENGTH_ABS,
            min_strength_ratio=PRUNE_MIN_STRENGTH_RATIO,
        )
        raw_estimates = estimator_final.estimates()
        final_estimates: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
        for iso in isotope_list:
            pos, strg = llr_pruned.get(iso, (np.zeros((0, 3)), np.zeros(0)))
            pos, strg = _apply_display_thresholds(pos, strg, min_strength)
            if pos.size == 0:
                pos, strg = legacy_pruned.get(iso, (np.zeros((0, 3)), np.zeros(0)))
                pos, strg = _apply_display_thresholds(pos, strg, min_strength)
            if pos.size == 0:
                raw_pos, raw_strg = raw_estimates.get(
                    iso, (np.zeros((0, 3)), np.zeros(0))
                )
                if raw_strg.size:
                    best_idx = int(np.argmax(raw_strg))
                    pos = raw_pos[[best_idx]]
                    strg = np.array([raw_strg[best_idx]], dtype=float)
            pos, strg = _merge_close_estimates(pos, strg, FINAL_MERGE_DISTANCE_M)
            pos, strg = _apply_display_thresholds(pos, strg, min_strength)
            final_estimates[iso] = (pos, strg)
        return final_estimates

    if live:
        plt.ion()
        plt.show(block=False)
        plt.pause(0.1)
        preview_pose = np.array(env.detector_position, dtype=float)
        preview_viz = RealTimePFVisualizer(
            isotopes=["Cs-137"],
            world_bounds=(0, env.size_x, 0, env.size_y, 0, env.size_z),
            true_sources={},
            true_strengths={},
            obstacle_grid=obstacle_grid,
            show_counts=False,
        )
        grid_pos = _grid_centers()
        preview_frame = PFFrame(
            step_index=-1,
            time=0.0,
            robot_position=preview_pose,
            robot_orientation=None,
            RFe=np.eye(3),
            RPb=np.eye(3),
            duration=0.0,
            counts_by_isotope={},
            particle_positions={"Cs-137": grid_pos},
            particle_weights={"Cs-137": np.ones(grid_pos.shape[0], dtype=float)},
            estimated_sources={"Cs-137": np.zeros((0, 3), dtype=float)},
            estimated_strengths={"Cs-137": np.zeros(0, dtype=float)},
        )
        preview_viz.update(preview_frame)
        preview_viz.fig.canvas.draw()
        if hasattr(preview_viz.fig.canvas, "flush_events"):
            preview_viz.fig.canvas.flush_events()
        plt.pause(5.0)
        plt.close(preview_viz.fig)

    estimator, current_pose, current_pose_idx = _build_estimator()
    viz = _build_visualizer()
    if live:
        plt.ion()
        plt.show(block=False)
        plt.pause(0.1)

    elapsed = 0.0
    last_frame: PFFrame | None = None
    step_counter = 0
    total_pairs = num_orients * num_orients
    visited_poses: list[NDArray[np.float64]] = []
    last_spectrum: np.ndarray | None = None
    last_counts: dict[str, float] | None = None
    representative_spectrum: np.ndarray | None = None
    representative_counts: dict[str, float] | None = None
    representative_candidates: set[str] = set()
    representative_step_index: int | None = None
    representative_total_counts = -np.inf
    last_max_ig: float | None = None
    total_motion_distance_m = 0.0
    total_motion_time_s = 0.0
    total_rotation_time_s = 0.0
    pending_motion_distance_m = 0.0
    pending_motion_time_s = 0.0
    pending_path_segment: dict[str, object] | None = None
    path_segments: list[dict[str, object]] = []
    measurement_live_times_s: list[float] = []
    total_ig_wall_s = 0.0
    total_pf_wall_s = 0.0
    total_prune_wall_s = 0.0
    total_viz_wall_s = 0.0
    ig_wall_samples_s: list[float] = []
    pf_wall_samples_s: list[float] = []
    run_wall_start = time.perf_counter()
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
        "Dwell control: "
        f"adaptive={bool(adaptive_dwell)} "
        f"cap_s={live_time:.3f} "
        f"chunk_s={float(adaptive_dwell_chunk_s):.3f} "
        f"min_s={float(adaptive_min_dwell_s):.3f} "
        f"ready_min_counts={float(adaptive_ready_min_counts):.3f} "
        f"ready_min_isotopes={int(adaptive_ready_min_isotopes)}"
    )
    print(
        "Strength prior adaptation: "
        f"enabled={bool(pf_conf.adaptive_strength_prior)} "
        f"steps={int(pf_conf.adaptive_strength_prior_steps)} "
        f"min_counts={float(pf_conf.adaptive_strength_prior_min_counts):.3f} "
        f"log_sigma={float(pf_conf.adaptive_strength_prior_log_sigma):.3f}"
    )
    print(
        "Pose observability constraint: "
        f"min_counts={float(pf_conf.pose_min_observation_counts):.3f} "
        f"penalty_scale={float(pf_conf.pose_min_observation_penalty_scale):.3f} "
        f"aggregate={pf_conf.pose_min_observation_aggregate}"
    )
    print(
        "PF count likelihood: "
        f"model={pf_conf.count_likelihood_model} "
        f"transport_rel_sigma={pf_conf.transport_model_rel_sigma} "
        f"spectrum_rel_sigma={pf_conf.spectrum_count_rel_sigma} "
        f"spectrum_abs_sigma={pf_conf.spectrum_count_abs_sigma} "
        f"df={float(pf_conf.count_likelihood_df):.2f}"
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
        f"(init_num_sources={pf_conf.init_num_sources})"
    )
    print(
        "PF init prior: "
        f"init_num_sources={pf_conf.init_num_sources}, "
        f"init_strength_log_mean={pf_conf.init_strength_log_mean:.2f}, "
        f"init_strength_log_sigma={pf_conf.init_strength_log_sigma:.2f}, "
        f"max_sources={pf_conf.max_sources}"
    )
    print(
        "Birth moves: "
        f"enabled={birth_enabled} "
        f"p_birth={pf_conf.p_birth:.3f} p_kill={pf_conf.p_kill:.3f} "
        f"split_prob={pf_conf.split_prob:.3f} merge_prob={pf_conf.merge_prob:.3f} "
        f"max_sources={pf_conf.max_sources}"
    )
    print(
        "Tempering settings: "
        f"max_resamples_per_observation={pf_conf.max_resamples_per_observation} "
        f"disable_regularize_on_temper_resample={pf_conf.disable_regularize_on_temper_resample}"
    )
    print(
        "Roughening settings: "
        f"k={pf_conf.roughening_k:.3f} "
        f"min_sigma_pos={pf_conf.min_sigma_pos:.3f} "
        f"max_sigma_pos={pf_conf.max_sigma_pos:.3f}"
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
    print(f"Simulation backend: {sim_backend}")
    print(
        "Mission timing model: "
        f"robot_speed={float(nominal_motion_speed_m_s):.3f}m/s "
        f"shield_overhead={float(rotation_overhead_s):.3f}s/measurement "
        "mission_time=travel+shield+live"
    )
    print(
        "Convergence gating: "
        f"enabled={estimator.pf_config.converge_enable} "
        f"window={estimator.pf_config.converge_window} "
        f"min_steps={estimator.pf_config.converge_min_steps} "
        f"map_eps={estimator.pf_config.converge_map_move_eps_m:.3f} "
        f"ess_ratio_high={estimator.pf_config.converge_ess_ratio_high:.2f} "
        f"ll_improve_eps={estimator.pf_config.converge_ll_improve_eps:.3f} "
        f"require_all={estimator.pf_config.converge_require_all}"
    )
    simulation_runtime.reset(
        {
            "usd_path": None
            if generated_blender_usd_path is None
            else generated_blender_usd_path.as_posix(),
            "room_size_xyz": [env.size_x, env.size_y, env.size_z],
            "source_count": len(sources),
            "sources": [
                {
                    "isotope": source.isotope,
                    "position": [
                        float(source.position[0]),
                        float(source.position[1]),
                        float(source.position[2]),
                    ],
                    "intensity_cps_1m": float(source.intensity_cps_1m),
                }
                for source in sources
            ],
            "obstacle_origin_xy": (
                [0.0, 0.0]
                if obstacle_grid is None
                else list(obstacle_grid.origin)
            ),
            "obstacle_cell_size_m": 1.0
            if obstacle_grid is None
            else float(obstacle_grid.cell_size),
            "obstacle_material": "concrete",
            "obstacle_grid_shape": [0, 0]
            if obstacle_grid is None
            else list(obstacle_grid.grid_shape),
            "obstacle_cells": [] if obstacle_grid is None else list(obstacle_grid.blocked_cells),
            "traversability_map_path": None
            if traversability_map_path is None
            else traversability_map_path.as_posix(),
            "traversability_map_png_path": None
            if traversability_map_png_path is None
            else traversability_map_png_path.as_posix(),
            "robot_radius_m": float(robot_radius_m),
            "author_obstacle_prims": generated_blender_usd_path is None,
        }
    )
    notifier.notify_started(
        {
            "backend": sim_backend,
            "sim_config_path": sim_config_path,
            "max_steps": max_steps,
            "max_poses": max_poses,
            "environment_mode": normalized_environment_mode,
            "obstacle_layout_path": obstacle_layout_path,
            "obstacle_seed": obstacle_seed,
            "obstacle_blocked_fraction": (
                None if obstacle_grid is None else float(obstacle_grid.blocked_fraction)
            ),
            "source_count": len(sources),
            "sources": [
                {
                    "isotope": source.isotope,
                    "position": [
                        float(source.position[0]),
                        float(source.position[1]),
                        float(source.position[2]),
                    ],
                    "intensity_cps_1m": float(source.intensity_cps_1m),
                }
                for source in sources
            ],
            "isotopes": isotopes,
            "birth_enabled": birth_enabled,
            "converge": converge,
            "pose_candidates": int(pose_candidates),
            "pose_min_dist_m": float(pose_min_dist),
            "candidate_grid_points": int(grid.shape[0]),
            "pf_num_particles": int(pf_conf.num_particles),
            "pf_max_sources": int(pf_conf.max_sources),
            "robot_speed_m_s": float(nominal_motion_speed_m_s),
            "rotation_overhead_s": float(rotation_overhead_s),
            "measurement_time_s": float(live_time),
            "adaptive_dwell": bool(adaptive_dwell),
            "adaptive_dwell_chunk_s": float(adaptive_dwell_chunk_s),
            "adaptive_min_dwell_s": float(adaptive_min_dwell_s),
            "adaptive_ready_min_counts": float(adaptive_ready_min_counts),
            "adaptive_ready_min_isotopes": int(adaptive_ready_min_isotopes),
            "pose_min_observation_counts": float(pf_conf.pose_min_observation_counts),
            "pose_min_observation_penalty_scale": float(
                pf_conf.pose_min_observation_penalty_scale
            ),
            "pose_min_observation_aggregate": pf_conf.pose_min_observation_aggregate,
        }
    )
    ig_max_global = 0.0
    pose_counter = 0
    try:
        while True:
            pose = current_pose
            stop_run = False
            pose_elapsed = 0.0
            zero_ig_override = False
            remaining_orientations = set(range(total_pairs))
            rotation_limit = max(1, int(estimator.pf_config.orientation_k))
            min_rotations_this_pose = min(
                rotation_limit,
                max(0, int(estimator.pf_config.min_rotations_per_pose)),
            )
            rotation_count = 0
            ig_max_pose = 0.0
            ig_threshold_current = estimator.pf_config.ig_threshold
            while True:
                if rotation_count >= rotation_limit:
                    print(
                        f"Reached max rotations per pose ({rotation_limit}); "
                        "moving to the next pose."
                    )
                    break
                if not remaining_orientations:
                    print("All orientation pairs exhausted; moving to the next pose.")
                    break
                ig_start = time.perf_counter()
                ig_scores = _compute_ig_grid(
                    estimator,
                    rot_mats,
                    pose_idx=current_pose_idx,
                    live_time_s=live_time,
                    planning_isotopes=None,
                )
                ig_elapsed = time.perf_counter() - ig_start
                total_ig_wall_s += ig_elapsed
                ig_wall_samples_s.append(float(ig_elapsed))
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
                if ig_max_pose <= 0.0 and ig_threshold_current > 0.0:
                    if not zero_ig_override:
                        print("IG grid returned zero; forcing rotation despite threshold.")
                        zero_ig_override = True
                    ig_threshold_current = 0.0
                if (
                    ig_val < ig_threshold_current
                    and rotation_count >= min_rotations_this_pose
                ):
                    print(
                        "Stopping rotation at this pose "
                        f"(max IG {ig_val:.6g} < threshold {ig_threshold_current:.6g})."
                    )
                    break
                fe_idx = best_pair_idx // num_orients
                pb_idx = best_pair_idx % num_orients
                RFe_sel = rot_mats[fe_idx]
                RPb_sel = rot_mats[pb_idx]
                if save_outputs and SAVE_IG_GRIDS and (step_counter + 1) % 10 == 0:
                    ig_path = IG_DIR / f"ig_grid_step_{step_counter:04d}.png"
                    render_octant_grid(
                        ig_path,
                        ig_scores=ig_scores,
                        highlight_idx=(fe_idx, pb_idx),
                        highlight_max=False,
                        font_size=12,
                    )
                step_motion_distance_m = float(pending_motion_distance_m)
                step_motion_time_s = float(pending_motion_time_s)
                step_rotation_time_s = float(rotation_overhead_s)
                (
                    observation,
                    actual_live_time_s,
                    z_detected,
                    z_detected_variance,
                    detected,
                    dwell_ready_reason,
                    dwell_chunks,
                ) = _acquire_spectrum_observation(
                    simulation_runtime=simulation_runtime,
                    decomposer=decomposer,
                    step_id=step_counter,
                    pose_xyz=pose,
                    fe_idx=fe_idx,
                    pb_idx=pb_idx,
                    live_time_s=live_time,
                    travel_time_s=step_motion_time_s,
                    shield_actuation_time_s=step_rotation_time_s,
                    adaptive_dwell=bool(adaptive_dwell),
                    adaptive_dwell_chunk_s=float(adaptive_dwell_chunk_s),
                    adaptive_min_dwell_s=float(adaptive_min_dwell_s),
                    adaptive_ready_min_counts=float(adaptive_ready_min_counts),
                    adaptive_ready_min_isotopes=int(adaptive_ready_min_isotopes),
                    spectrum_count_method=spectrum_count_method,
                    detect_threshold_abs=detect_threshold_abs,
                    detect_threshold_rel=detect_threshold_rel,
                    detect_threshold_rel_by_isotope=detect_threshold_rel_by_isotope,
                    min_peaks_by_isotope=min_peaks_by_isotope,
                )
                pending_motion_distance_m = 0.0
                pending_motion_time_s = 0.0
                step_path_segment = pending_path_segment
                pending_path_segment = None
                total_motion_distance_m += step_motion_distance_m
                total_motion_time_s += step_motion_time_s
                total_rotation_time_s += step_rotation_time_s
                if step_path_segment is not None:
                    path_segments.append(step_path_segment)
                elapsed += step_motion_time_s + step_rotation_time_s
                spectrum = _observation_spectrum_array(observation, decomposer)
                last_spectrum = spectrum.copy()
                last_counts = {iso: float(val) for iso, val in z_detected.items()}
                last_candidates = set(detected)
                spectrum_total_counts = float(np.sum(spectrum))
                if spectrum_total_counts > representative_total_counts:
                    representative_total_counts = spectrum_total_counts
                    representative_spectrum = spectrum.copy()
                    representative_counts = dict(last_counts)
                    representative_candidates = set(last_candidates)
                    representative_step_index = int(step_counter)
                spectrum_notify_every = max(1, int(notify_spectrum_every))
                if notify_spectrum and step_counter % spectrum_notify_every == 0:
                    notifier.notify_spectrum(
                        step_counter,
                        _build_spectrum_notification_payload(
                            decomposer=decomposer,
                            spectrum=spectrum,
                            step_index=step_counter,
                            pose_xyz=np.asarray(observation.detector_pose_xyz, dtype=float),
                            fe_index=fe_idx,
                            pb_index=pb_idx,
                            live_time_s=actual_live_time_s,
                            counts_by_isotope=last_counts,
                            detected_isotopes=set(detected),
                            count_method=spectrum_count_method,
                            max_bins=int(notify_spectrum_max_bins),
                        ),
                    )
                if detect_consecutive > 0:
                    previous_active_isotopes = set(active_isotopes)
                    active_isotopes = _update_detection_hysteresis(
                        set(detected),
                        detect_counts,
                        miss_counts,
                        active_isotopes,
                        consecutive=detect_consecutive,
                        miss_consecutive=DETECT_MISS_AFTER_LOCK,
                        consecutive_by_isotope=DETECT_CONSECUTIVE_BY_ISOTOPE,
                    )
                    detected_isotopes = set(active_isotopes)
                    last_candidates = set(detected_isotopes)
                    should_report_detection = (
                        step_counter + 1 >= detect_min_steps
                        and active_isotopes != previous_active_isotopes
                    )
                    if should_report_detection:
                        print(
                            "Detected isotope diagnostics active: "
                            f"{sorted(active_isotopes)} "
                            "(PF and planning still keep all candidate isotopes)."
                        )
                pf_isotopes = list(estimator.isotopes)
                z_k_full = {iso: float(z_detected.get(iso, 0.0)) for iso in pf_isotopes}
                z_variance_full = {
                    iso: float(
                        max(
                            z_detected_variance.get(iso, max(z_k_full[iso], 1.0)),
                            1.0,
                        )
                    )
                    for iso in pf_isotopes
                }
                z_counts = z_k_full
                z_k = z_k_full
                pose_for_pf = np.asarray(observation.detector_pose_xyz, dtype=float)
                meas = Measurement(
                    counts_by_isotope=z_k,
                    count_variance_by_isotope=z_variance_full,
                    pose_idx=current_pose_idx,
                    orient_idx=best_pair_idx,
                    live_time_s=actual_live_time_s,
                    fe_index=fe_idx,
                    pb_index=pb_idx,
                    RFe=RFe_sel,
                    RPb=RPb_sel,
                    detector_position=pose_for_pf,
                )
                pf_start = time.perf_counter()
                estimator.update_pair(
                    z_k=z_k,
                    pose_idx=current_pose_idx,
                    fe_index=fe_idx,
                    pb_index=pb_idx,
                    live_time_s=actual_live_time_s,
                    z_variance_k=z_variance_full,
                )
                if estimator.last_strength_prior_diagnostics:
                    for iso, stats in sorted(estimator.last_strength_prior_diagnostics.items()):
                        print(
                            f"[step {step_counter}] strength_prior[{iso}] "
                            f"z={stats['observed_counts']:.2f} "
                            f"target={stats['target_counts']:.2f} "
                            f"median_before={stats['before_median_strength']:.2f} "
                            f"median_after={stats['after_median_strength']:.2f} "
                            f"particles={int(stats['particles_changed'])}"
                        )
                pf_elapsed = time.perf_counter() - pf_start
                total_pf_wall_s += pf_elapsed
                pf_wall_samples_s.append(float(pf_elapsed))
                measurement_live_times_s.append(float(actual_live_time_s))
                elapsed += actual_live_time_s
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
                    method=PRUNE_METHOD,
                    params={"deltaLL_min": PRUNE_DELTALL_MIN},
                    tau_mix=PRUNE_TAU_MIX,
                    min_support=PRUNE_MIN_SUPPORT,
                    min_obs_count=prune_min_obs_count,
                    min_strength_abs=PRUNE_MIN_STRENGTH_ABS,
                    min_strength_ratio=PRUNE_MIN_STRENGTH_RATIO,
                )
                prune_elapsed = time.perf_counter() - prune_start
                total_prune_wall_s += prune_elapsed
                viz_start = time.perf_counter()
                if hasattr(frame, "estimated_sources") and hasattr(frame, "estimated_strengths"):
                    frame.estimated_sources = {}
                    frame.estimated_strengths = {}
                    for iso in isotopes:
                        pos, strg = pruned.get(iso, (np.zeros((0, 3)), np.zeros(0)))
                        pos, strg = _apply_display_thresholds(pos, strg, estimate_min_strength)
                        frame.estimated_sources[iso] = pos
                        frame.estimated_strengths[iso] = strg
                viz.update(frame)
                last_frame = frame
                print(
                    f"[step {step_counter}] pose={_fmt_pos(pose_for_pf)} "
                    f"orient_pair={best_pair_idx} "
                    f"ig={ig_val:.6g} ig_threshold={ig_threshold_current:.6g} "
                    f"fe_idx={fe_idx} pb_idx={pb_idx} "
                    f"travel_distance_m={step_motion_distance_m:.3f} "
                    f"travel_time_s={step_motion_time_s:.1f} "
                    f"shield_time_s={step_rotation_time_s:.1f} "
                    f"live_time_s={actual_live_time_s:.1f}/{live_time:.1f} "
                    f"dwell_chunks={dwell_chunks} "
                    f"dwell_reason={dwell_ready_reason} "
                    f"mission_time_s={elapsed:.1f} "
                    f"z_keys={sorted(z_k.keys())} "
                    f"z_obs={_fmt_counts(z_counts)}"
                )
                if live:
                    plt.pause(0.05)
                viz_elapsed += time.perf_counter() - viz_start
                total_viz_wall_s += viz_elapsed
                _log_pf_diagnostics(estimator, step_counter)
                print(
                    f"[timing step {step_counter}] ig={ig_elapsed:.3f}s pf={pf_elapsed:.3f}s "
                    f"prune={prune_elapsed:.3f}s viz={viz_elapsed:.3f}s "
                    f"travel={step_motion_time_s:.1f}s "
                    f"shield={step_rotation_time_s:.1f}s "
                    f"live={actual_live_time_s:.1f}s"
                )
                step_counter += 1
                rotation_count += 1
                remaining_orientations.discard(best_pair_idx)
                if save_outputs and last_spectrum is not None and step_counter % 10 == 0:
                    highlight = set(last_candidates)
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
                pose_elapsed += actual_live_time_s + step_rotation_time_s
                if pose_elapsed >= estimator.pf_config.max_dwell_time_s:
                    break
            if (
                save_outputs
                and estimator.measurements
                and estimator.measurements[-1].pose_idx == current_pose_idx
            ):
                pf_step = current_pose_idx + 1
                pf_path = PF_DIR / f"pf_step_{pf_step:03d}.png"
                viz.save_final(pf_path.as_posix())
            if stop_run:
                print(f"Reached max steps ({max_steps}); stopping exploration.")
                break
            if last_max_ig is not None and last_max_ig < ig_threshold_current:
                if estimator.should_stop_exploration(
                    ig_threshold=ig_threshold_current,
                    live_time_s=live_time,
                ):
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
                map_api=planning_map,
                n_candidates=pose_candidates,
                strategy="free_space_sobol",
                min_dist_from_visited=pose_min_dist,
                visited_poses_xyz=visited_arr,
                bounds_xyz=(bounds_lo, bounds_hi),
            )
            if candidates.size == 0 and pose_min_dist > 0.0:
                relaxed_dist = max(pose_min_dist * 0.5, 0.5)
                print(
                    "No candidates with current spacing; retrying with min_dist="
                    f"{relaxed_dist:.2f}."
                )
                candidates = generate_candidate_poses(
                    current_pose_xyz=pose,
                    map_api=planning_map,
                    n_candidates=max(pose_candidates * 2, pose_candidates),
                    strategy="free_space_sobol",
                    min_dist_from_visited=relaxed_dist,
                    visited_poses_xyz=visited_arr,
                    bounds_xyz=(bounds_lo, bounds_hi),
                )
            if candidates.size == 0:
                print("No candidate poses available; stopping exploration.")
                break
            print(f"Generated {len(candidates)} candidate poses. Computing best next pose...")
            next_idx = select_next_pose_from_candidates(
                estimator=estimator,
                candidate_poses_xyz=candidates,
                current_pose_xyz=pose,
                criterion="after_rotation",
                t_max_s=float(rotation_limit) * float(live_time),
                verbose=True,
                progress_every=0,
                auto_lambda_cost=True,
                num_rollouts=DEFAULT_PLANNING_ROLLOUTS,
                min_observation_counts=float(pf_conf.pose_min_observation_counts),
                min_observation_penalty_scale=float(
                    pf_conf.pose_min_observation_penalty_scale
                ),
                min_observation_aggregate=pf_conf.pose_min_observation_aggregate,
                min_observation_max_particles=pf_conf.pose_min_observation_max_particles,
            )
            next_pose = candidates[next_idx]
            motion_distance_m = float(np.linalg.norm(next_pose - pose))
            motion_time_s = motion_distance_m / max(float(nominal_motion_speed_m_s), 1e-9)
            pending_motion_distance_m = motion_distance_m
            pending_motion_time_s = motion_time_s
            pending_path_segment = {
                "from_pose_xyz": [float(v) for v in pose],
                "to_pose_xyz": [float(v) for v in next_pose],
                "distance_m": float(motion_distance_m),
                "travel_time_s": float(motion_time_s),
                "speed_m_s": float(nominal_motion_speed_m_s),
            }
            print(
                "Robot travel segment: "
                f"distance={motion_distance_m:.3f}m "
                f"time={motion_time_s:.1f}s "
                f"speed={float(nominal_motion_speed_m_s):.3f}m/s"
            )
            current_pose = next_pose
            estimator.add_measurement_pose(current_pose, reset_filters=False)
            current_pose_idx = len(estimator.poses) - 1
    except Exception as exc:
        notifier.notify_failed(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "measurements_completed": int(step_counter),
                "mission_time_s": float(elapsed),
                "last_counts": last_counts,
                "last_max_ig": last_max_ig,
            }
        )
        raise
    finally:
        simulation_runtime.close()

    # Save final snapshots
    result_paths: dict[str, str] = {}
    if save_outputs:
        pf_out_path = RESULTS_DIR / f"result_pf{output_suffix}.png"
        spectrum_out_path = RESULTS_DIR / f"result_spectrum{output_suffix}.png"
        last_spectrum_out_path = RESULTS_DIR / f"result_spectrum_last{output_suffix}.png"
        estimates_out_path = RESULTS_DIR / f"result_estimates{output_suffix}.png"
        result_paths = {
            "pf_plot": pf_out_path.as_posix(),
            "estimates_plot": estimates_out_path.as_posix(),
            "spectrum_plot": spectrum_out_path.as_posix(),
            "last_spectrum_plot": last_spectrum_out_path.as_posix(),
        }
        pf_out_path.parent.mkdir(parents=True, exist_ok=True)
        raw_estimates = None
        if last_frame is not None:
            last_frame.step_index = max(0, int(step_counter) - 1)
            last_frame.time = float(elapsed)
            final_estimates = _build_final_estimates(
                estimator,
                isotopes,
                final_estimate_min_strength,
                prune_min_obs_count,
                use_pruning=True,
            )
            raw_estimates = _build_final_estimates(
                estimator,
                isotopes,
                final_estimate_min_strength,
                prune_min_obs_count,
                use_pruning=False,
            )
            last_frame.estimated_sources = {
                iso: pos for iso, (pos, _) in final_estimates.items()
            }
            last_frame.estimated_strengths = {
                iso: strg for iso, (_, strg) in final_estimates.items()
            }
            viz.update(last_frame)
        viz.save_final(pf_out_path.as_posix())
        if last_frame is not None:
            viz.save_estimates_only(estimates_out_path.as_posix())
        if last_frame is not None and raw_estimates is not None:
            last_frame.estimated_sources = {
                iso: pos for iso, (pos, _) in raw_estimates.items()
            }
            last_frame.estimated_strengths = {
                iso: strg for iso, (_, strg) in raw_estimates.items()
            }
            viz.update(last_frame)
            if estimator.poses:
                pf_step = len(estimator.poses)
                estimates_step_path = PF_DIR / f"estimates_step_{pf_step:03d}.png"
                viz.save_estimates_only(estimates_step_path.as_posix())
        if representative_spectrum is not None:
            highlight = set(representative_candidates)
            title = "Representative measurement spectrum"
            if representative_step_index is not None:
                title = f"{title} (step {representative_step_index})"
            _save_spectrum_plot(
                decomposer,
                representative_spectrum,
                spectrum_out_path,
                highlight_isotopes=highlight,
                counts_by_isotope=representative_counts,
                title=title,
            )
        if last_spectrum is not None:
            highlight = set(last_candidates)
            _save_spectrum_plot(
                decomposer,
                last_spectrum,
                last_spectrum_out_path,
                highlight_isotopes=highlight,
                counts_by_isotope=last_counts,
                title="Last measurement spectrum",
            )
    total_meas_time = float(sum(measurement_live_times_s))
    total_mission_time_s = float(
        total_meas_time + total_motion_time_s + total_rotation_time_s
    )
    mean_live_time_s = (
        float(np.mean(measurement_live_times_s)) if measurement_live_times_s else 0.0
    )
    min_live_time_s = (
        float(np.min(measurement_live_times_s)) if measurement_live_times_s else 0.0
    )
    max_live_time_s = (
        float(np.max(measurement_live_times_s)) if measurement_live_times_s else 0.0
    )
    mean_ig_wall_s = float(np.mean(ig_wall_samples_s)) if ig_wall_samples_s else 0.0
    max_ig_wall_s = float(np.max(ig_wall_samples_s)) if ig_wall_samples_s else 0.0
    mean_pf_wall_s = float(np.mean(pf_wall_samples_s)) if pf_wall_samples_s else 0.0
    max_pf_wall_s = float(np.max(pf_wall_samples_s)) if pf_wall_samples_s else 0.0
    total_compute_time_s = float(
        total_ig_wall_s
        + total_pf_wall_s
        + total_prune_wall_s
        + total_viz_wall_s
    )
    wall_clock_runtime_s = float(time.perf_counter() - run_wall_start)
    mission_metrics = {
        "total_measurements": int(step_counter),
        "total_live_time_s": float(total_meas_time),
        "measurement_live_times_s": [
            float(value) for value in measurement_live_times_s
        ],
        "mean_live_time_s": mean_live_time_s,
        "min_live_time_s": min_live_time_s,
        "max_live_time_s": max_live_time_s,
        "measurement_time_cap_s": float(live_time),
        "adaptive_dwell_enabled": bool(adaptive_dwell),
        "adaptive_dwell_chunk_s": float(adaptive_dwell_chunk_s),
        "adaptive_min_dwell_s": float(adaptive_min_dwell_s),
        "adaptive_ready_min_counts": float(adaptive_ready_min_counts),
        "adaptive_ready_min_isotopes": int(adaptive_ready_min_isotopes),
        "total_motion_distance_m": float(total_motion_distance_m),
        "nominal_motion_speed_m_s": float(nominal_motion_speed_m_s),
        "total_travel_time_s": float(total_motion_time_s),
        "estimated_motion_time_s": float(total_motion_time_s),
        "rotation_overhead_s_per_measurement": float(rotation_overhead_s),
        "total_shield_actuation_time_s": float(total_rotation_time_s),
        "estimated_rotation_time_s": float(total_rotation_time_s),
        "total_mission_time_s": float(total_mission_time_s),
        "estimated_end_to_end_time_s": float(total_mission_time_s),
        "total_move_measure_time_s": float(total_mission_time_s),
        "path_segments": path_segments,
        "num_motion_segments": int(len(path_segments)),
        "total_compute_time_s": total_compute_time_s,
        "ig_compute_time_s": float(total_ig_wall_s),
        "mean_orientation_selection_time_s": mean_ig_wall_s,
        "max_orientation_selection_time_s": max_ig_wall_s,
        "pf_compute_time_s": float(total_pf_wall_s),
        "mean_pf_update_time_s": mean_pf_wall_s,
        "max_pf_update_time_s": max_pf_wall_s,
        "prune_time_s": float(total_prune_wall_s),
        "viz_time_s": float(total_viz_wall_s),
        "wall_clock_runtime_s": wall_clock_runtime_s,
    }
    setattr(estimator, "mission_metrics", mission_metrics)
    if save_outputs:
        print(f"Final PF visualization saved to: {pf_out_path}")
        print(f"Final estimates-only visualization saved to: {estimates_out_path}")
        if representative_spectrum is not None:
            print(f"Representative spectrum saved to: {spectrum_out_path}")
        if last_spectrum is not None:
            print(f"Last spectrum saved to: {last_spectrum_out_path}")
    print(
        f"Total measurements: {step_counter}, "
        f"live={total_meas_time:.1f}s, "
        f"travel={total_motion_time_s:.1f}s, "
        f"shield={total_rotation_time_s:.1f}s, "
        f"mission={total_mission_time_s:.1f}s"
    )
    print(
        "Mission timing summary: "
        f"distance={total_motion_distance_m:.2f}m "
        f"motion={total_motion_time_s:.1f}s "
        f"rotation={total_rotation_time_s:.1f}s "
        f"end_to_end={mission_metrics['estimated_end_to_end_time_s']:.1f}s "
        f"compute={total_compute_time_s:.3f}s "
        f"ig_mean={mean_ig_wall_s:.3f}s "
        f"pf_mean={mean_pf_wall_s:.3f}s "
        f"wall_clock={wall_clock_runtime_s:.2f}s"
    )
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
    estimates = _build_final_estimates(
        estimator,
        isotopes,
        final_estimate_min_strength,
        prune_min_obs_count,
        use_pruning=False,
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
    notifier.notify_finished(
        {
            "summary": (
                f"{step_counter} measurements, "
                f"mission_time_s={total_mission_time_s:.1f}, "
                f"wall_clock_s={wall_clock_runtime_s:.2f}"
            ),
            "measurements_completed": int(step_counter),
            "mission_metrics": {
                key: value
                for key, value in mission_metrics.items()
                if key != "path_segments"
            },
            "match_metrics": metrics,
            "estimated_sources": est_by_iso,
            "ground_truth_sources": gt_by_iso,
            "last_counts": last_counts,
            "output_paths": result_paths,
            "backend": sim_backend,
            "sim_config_path": sim_config_path,
            "environment_mode": normalized_environment_mode,
        }
    )
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
