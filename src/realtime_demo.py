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
from spectrum.pipeline import SpectralDecomposer, SpectrumConfig
from spectrum.baseline import baseline_als
from spectrum.smoothing import gaussian_smooth
from pf.parallel import Measurement
from pf.estimator import RotatingShieldPFEstimator, RotatingShieldPFConfig
from planning.candidate_generation import generate_candidate_poses
from planning.pose_selection import (
    DEFAULT_PLANNING_ROLLOUTS,
    minimum_observation_shortfall,
    select_next_pose_from_candidates,
)
from planning.dss_pp import DSSPPConfig, select_dss_pp_next_station
from planning.traversability import (
    TraversabilityMap,
    build_traversability_map_from_obstacle_grid,
    render_traversability_map,
    shortest_grid_path_points,
)
from visualization.realtime_viz import (
    CUISplitPFVisualizer,
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
from sim.shield_geometry import resolve_shield_thickness_config

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
SPECTRUM_DIR = RESULTS_DIR / "spectrum"
PF_DIR = RESULTS_DIR / "pf"
IG_DIR = RESULTS_DIR / "IG"
CUI_VIEW_DIR = RESULTS_DIR / "cui_view"
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
    position_min: tuple[float, float, float] | None = None,
    position_max: tuple[float, float, float] | None = None,
) -> NDArray[np.float64]:
    """Create a dense 3D grid of candidate sources inside the environment bounds."""
    lo = (
        np.array([0.0, 0.0, 0.0], dtype=float)
        if position_min is None
        else np.asarray(position_min, dtype=float)
    )
    hi = (
        np.array([env.size_x, env.size_y, env.size_z], dtype=float)
        if position_max is None
        else np.asarray(position_max, dtype=float)
    )
    if lo.shape != (3,) or hi.shape != (3,):
        raise ValueError("Candidate source bounds must be 3D vectors.")
    room_lo = np.array([0.0, 0.0, 0.0], dtype=float)
    room_hi = np.array([env.size_x, env.size_y, env.size_z], dtype=float)
    lo = np.maximum(lo, room_lo)
    hi = np.minimum(hi, room_hi)
    if bool(np.any(hi <= lo)):
        raise ValueError("Candidate source bounds are empty.")
    xs = _candidate_axis_points(lo[0] + margin, hi[0] - margin, spacing[0])
    ys = _candidate_axis_points(lo[1] + margin, hi[1] - margin, spacing[1])
    zs = _candidate_axis_points(lo[2] + margin, hi[2] - margin, spacing[2])
    if xs.size == 0 or ys.size == 0 or zs.size == 0:
        raise ValueError("Candidate grid is empty; check spacing and margin values.")
    return np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=float)


def _resolve_source_position_bounds(
    env: EnvironmentConfig,
    runtime_config: dict[str, object],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Resolve the PF source-position support from runtime config and room bounds."""
    lo = np.array([0.0, 0.0, 0.0], dtype=float)
    hi = np.array([env.size_x, env.size_y, env.size_z], dtype=float)
    raw_min = runtime_config.get("source_position_min")
    raw_max = runtime_config.get("source_position_max")
    if raw_min is not None:
        arr = np.asarray(raw_min, dtype=float)
        if arr.shape != (3,):
            raise ValueError("source_position_min must be a 3-element vector.")
        lo = arr
    if raw_max is not None:
        arr = np.asarray(raw_max, dtype=float)
        if arr.shape != (3,):
            raise ValueError("source_position_max must be a 3-element vector.")
        hi = arr
    if "source_z_min_m" in runtime_config:
        lo[2] = float(runtime_config["source_z_min_m"])
    if "source_z_max_m" in runtime_config:
        hi[2] = float(runtime_config["source_z_max_m"])
    room_lo = np.array([0.0, 0.0, 0.0], dtype=float)
    room_hi = np.array([env.size_x, env.size_y, env.size_z], dtype=float)
    lo = np.maximum(lo, room_lo)
    hi = np.minimum(hi, room_hi)
    if bool(np.any(hi <= lo)):
        raise ValueError("Resolved source-position support is empty.")
    return tuple(float(v) for v in lo), tuple(float(v) for v in hi)


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


def _metadata_float(metadata: dict[str, object], key: str) -> float | None:
    """Read one numeric metadata field when present."""
    value = metadata.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_count_map(counts: dict[str, float], precision: int = 2) -> str:
    """Format isotope count maps for compact step diagnostics."""
    if not counts:
        return "{}"
    chunks = [
        f"{iso}:{float(value):.{precision}f}"
        for iso, value in sorted(counts.items())
    ]
    return "{" + ", ".join(chunks) + "}"


def _source_equivalent_counts_from_metadata(
    metadata: dict[str, object],
    isotopes: list[str] | tuple[str, ...],
) -> dict[str, float]:
    """Extract Geant4 source-equivalent counts from observation metadata."""
    counts: dict[str, float] = {}
    for isotope in isotopes:
        value = _metadata_float(metadata, f"source_equivalent_counts_{isotope}")
        if value is not None:
            counts[str(isotope)] = value
    return counts


def _transport_detected_counts_from_metadata(
    metadata: dict[str, object],
    isotopes: list[str] | tuple[str, ...],
) -> dict[str, float]:
    """Extract source-labeled Geant4 detector-entry counts for diagnostics."""
    counts: dict[str, float] = {}
    for isotope in isotopes:
        value = _metadata_float(metadata, f"transport_detected_counts_{isotope}")
        if value is not None:
            counts[str(isotope)] = value
    return counts


def _response_poisson_counts_for_diagnostics(
    diagnostic_decomposer: SpectralDecomposer,
    spectrum: NDArray[np.float64],
    isotopes: list[str] | tuple[str, ...],
) -> dict[str, float]:
    """Compute response-Poisson counts without mutating the main decomposer state."""
    return {
        iso: float(value)
        for iso, value in diagnostic_decomposer.compute_response_poisson_counts(
            spectrum,
            isotopes=list(isotopes),
            include_background=True,
        ).items()
    }


def _log_geant4_transport_decomposition_diagnostics(
    *,
    step_index: int,
    metadata: dict[str, object],
    spectrum_total_counts: float,
    selected_count_method: str,
    selected_counts: dict[str, float],
    response_poisson_counts: dict[str, float],
    source_equivalent_counts: dict[str, float],
    transport_detected_counts: dict[str, float],
) -> None:
    """Log Geant4 transport and spectrum-decomposition diagnostics for one step."""
    if str(metadata.get("backend", "")).lower() != "geant4":
        return
    weighted_effective = _metadata_float(metadata, "weighted_spectrum_effective_entries")
    primaries = _metadata_float(metadata, "num_primaries")
    run_time_s = _metadata_float(metadata, "run_time_s")
    primaries_per_sec = _metadata_float(metadata, "primaries_per_sec")
    effective_per_sec = _metadata_float(metadata, "effective_entries_per_sec")
    total_steps = _metadata_float(metadata, "total_track_steps")
    detector_hit_events = _metadata_float(metadata, "detector_hit_events")
    detector_hit_steps = _metadata_float(metadata, "detector_hit_steps")
    secondaries = _metadata_float(metadata, "secondary_count")
    killed_non_gamma = _metadata_float(metadata, "killed_non_gamma_secondary_count")
    compton = _metadata_float(metadata, "process_count_compton")
    rayleigh = _metadata_float(metadata, "process_count_rayleigh")
    photoelectric = _metadata_float(metadata, "process_count_photoelectric")
    print(
        f"[step {step_index}] geant4_transport "
        f"primaries={_fmt_optional_float(primaries, 0)} "
        f"run={_fmt_optional_float(run_time_s, 3)}s "
        f"primaries_per_sec={_fmt_optional_float(primaries_per_sec, 1)} "
        f"track_steps={_fmt_optional_float(total_steps, 0)} "
        f"detector_hit_events={_fmt_optional_float(detector_hit_events, 0)} "
        f"detector_hit_steps={_fmt_optional_float(detector_hit_steps, 0)} "
        f"secondaries={_fmt_optional_float(secondaries, 0)} "
        f"killed_non_gamma={_fmt_optional_float(killed_non_gamma, 0)} "
        f"compton={_fmt_optional_float(compton, 0)} "
        f"rayleigh={_fmt_optional_float(rayleigh, 0)} "
        f"photoelectric={_fmt_optional_float(photoelectric, 0)} "
        f"effective_entries={_fmt_optional_float(weighted_effective, 1)} "
        f"effective_entries_per_sec={_fmt_optional_float(effective_per_sec, 1)}"
    )
    print(
        f"[step {step_index}] geant4_decomposition "
        f"source_equivalent_unattenuated={_fmt_count_map(source_equivalent_counts)} "
        f"transport_detected={_fmt_count_map(transport_detected_counts)} "
        f"total_spectrum_counts={float(spectrum_total_counts):.2f} "
        f"{selected_count_method}={_fmt_count_map(selected_counts)} "
        f"response_poisson={_fmt_count_map(response_poisson_counts)}"
    )
    volume_top = str(metadata.get("volume_step_counts_top", "")).strip()
    if volume_top and volume_top != "-":
        print(f"[step {step_index}] geant4_volume_steps_top {volume_top}")


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
        birth_gate_passed = bool(stats.get("birth_residual_gate_passed", False))
        birth_gate_support = int(stats.get("birth_residual_support", 0))
        birth_gate_distinct = int(stats.get("birth_residual_distinct_poses", 0))
        birth_gate_stations = int(stats.get("birth_residual_distinct_stations", 0))
        birth_gate_chi2 = float(stats.get("birth_residual_chi2", 0.0))
        birth_gate_p = float(stats.get("birth_residual_p_value", 1.0))
        birth_refit_fraction = float(stats.get("birth_residual_refit_fraction", 1.0))
        birth_refit_gate = bool(stats.get("birth_residual_refit_gate_passed", True))
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
            f"birth_enabled={birth_enabled} max_sources={max_sources} p_birth={p_birth:.3f} "
            f"birth_gate={birth_gate_passed} "
            f"birth_residual_support={birth_gate_support} "
            f"birth_residual_distinct_poses={birth_gate_distinct} "
            f"birth_residual_distinct_stations={birth_gate_stations} "
            f"birth_residual_chi2={birth_gate_chi2:.2f} "
            f"birth_residual_p={birth_gate_p:.3g} "
            f"birth_refit_gate={birth_refit_gate} "
            f"birth_refit_fraction={birth_refit_fraction:.3f}"
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


def _spectrum_config_from_runtime_config(
    runtime_config: dict[str, object],
) -> SpectrumConfig:
    """Build a spectrum config from runtime keys while preserving defaults."""
    config = SpectrumConfig()
    scoring_mode = str(runtime_config.get("detector_scoring_mode", "")).strip().lower()
    source_rate_model = str(runtime_config.get("source_rate_model", "")).strip().lower()
    if scoring_mode == "incident_gamma_energy":
        if "response_efficiency_model" not in runtime_config:
            config.response_efficiency_model = "unit"
        config.use_incident_gamma_response_matrix = True
    if source_rate_model == "detector_cps_1m":
        config.normalize_line_intensities = True
    field_names = set(SpectrumConfig.__dataclass_fields__.keys())
    for key, value in runtime_config.items():
        if key not in field_names or value is None:
            continue
        current = getattr(config, key)
        if isinstance(current, bool):
            setattr(config, key, bool(value))
        elif isinstance(current, int) and not isinstance(current, bool):
            setattr(config, key, int(value))
        elif isinstance(current, float):
            setattr(config, key, float(value))
        else:
            setattr(config, key, value)
    config.__post_init__()
    return config


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


def _transition_cost_for_pair(
    rot_mats: Sequence[np.ndarray],
    current_pair_id: int | None,
    candidate_pair_id: int,
) -> float:
    """Return a normalized Fe/Pb angular transition cost for one pair."""
    if current_pair_id is None or int(current_pair_id) < 0:
        return 0.0
    num_orients = len(rot_mats)
    if num_orients <= 0:
        return 0.0
    prev_fe = int(current_pair_id) // num_orients
    prev_pb = int(current_pair_id) % num_orients
    next_fe = int(candidate_pair_id) // num_orients
    next_pb = int(candidate_pair_id) % num_orients
    normals = np.asarray([np.asarray(mat, dtype=float)[:, 2] for mat in rot_mats])
    cost = 0.0
    for prev_idx, next_idx in ((prev_fe, next_fe), (prev_pb, next_pb)):
        dot = float(np.clip(np.dot(normals[prev_idx], normals[next_idx]), -1.0, 1.0))
        cost += float(np.arccos(dot) / np.pi)
    return cost


def _log_utility_grid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a nonnegative log-scaled utility grid for planner score terms."""
    arr = np.asarray(values, dtype=float)
    finite = np.where(np.isfinite(arr), arr, 0.0)
    return np.log1p(np.maximum(finite, 0.0))


def _isotope_count_balance_penalty(counts: dict[str, float]) -> float:
    """Return an isotope-agnostic penalty for single-isotope dominated counts."""
    values = np.asarray(
        [max(float(value), 0.0) for value in counts.values()],
        dtype=float,
    )
    if values.size <= 1:
        return 0.0
    total = float(np.sum(values))
    if total <= 0.0:
        return 1.0
    probabilities = values / total
    positive = probabilities > 0.0
    entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
    normalized_entropy = entropy / max(float(np.log(values.size)), 1e-12)
    return float(np.clip(1.0 - normalized_entropy, 0.0, 1.0))


def _compute_shield_selection_grid(
    estimator: RotatingShieldPFEstimator,
    rot_mats: Sequence[np.ndarray],
    *,
    pose_idx: int,
    live_time_s: float,
    ig_scores: NDArray[np.float64],
    current_pair_id: int | None,
    min_observation_counts: float,
    signature_weight: float,
    low_count_penalty_weight: float,
    count_balance_weight: float,
    rotation_cost_weight: float,
    variance_floor: float,
    max_particles: int | None,
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    """
    Score shield postures by EIG, signature separation, observability, and rotation cost.

    The score is a planner utility only; PF updates still use measured spectra.
    """
    size = int(len(rot_mats))
    scores = np.asarray(ig_scores, dtype=float).copy()
    signature_grid = np.zeros((size, size), dtype=float)
    signature_utility_grid = np.zeros((size, size), dtype=float)
    penalty_grid = np.zeros((size, size), dtype=float)
    balance_grid = np.zeros((size, size), dtype=float)
    rotation_grid = np.zeros((size, size), dtype=float)
    planning_particles = estimator.planning_particles(
        max_particles=max_particles,
        method=estimator.pf_config.planning_method,
    )
    for fe_idx in range(size):
        for pb_idx in range(size):
            pair_id = fe_idx * size + pb_idx
            if signature_weight > 0.0:
                signature_grid[fe_idx, pb_idx] = estimator.orientation_signature_separation_score(
                    pose_idx=pose_idx,
                    fe_index=fe_idx,
                    pb_index=pb_idx,
                    live_time_s=live_time_s,
                    particles_by_isotope=planning_particles,
                    alpha_by_isotope=estimator.pf_config.alpha_weights,
                    variance_floor=variance_floor,
                )
            counts = None
            if min_observation_counts > 0.0 or count_balance_weight > 0.0:
                counts = estimator.expected_observation_counts_by_isotope_at_pair(
                    pose_idx=pose_idx,
                    fe_index=fe_idx,
                    pb_index=pb_idx,
                    live_time_s=live_time_s,
                    max_particles=max_particles,
                )
            if min_observation_counts > 0.0 and counts is not None:
                penalty_grid[fe_idx, pb_idx] = minimum_observation_shortfall(
                    counts,
                    min_counts=float(min_observation_counts),
                )
            if count_balance_weight > 0.0 and counts is not None:
                balance_grid[fe_idx, pb_idx] = _isotope_count_balance_penalty(counts)
            rotation_grid[fe_idx, pb_idx] = _transition_cost_for_pair(
                rot_mats,
                current_pair_id,
                pair_id,
            )
    signature_utility_grid = _log_utility_grid(signature_grid)
    scores += float(signature_weight) * signature_utility_grid
    scores -= float(low_count_penalty_weight) * penalty_grid
    scores -= float(count_balance_weight) * balance_grid
    scores -= float(rotation_cost_weight) * rotation_grid
    diagnostics = {
        "eig": np.asarray(ig_scores, dtype=float),
        "signature": signature_grid,
        "signature_utility": signature_utility_grid,
        "low_count_penalty": penalty_grid,
        "count_balance_penalty": balance_grid,
        "rotation_cost": rotation_grid,
    }
    return scores, diagnostics


def _polyline_distance(points_xyz: NDArray[np.float64] | None) -> float:
    """Return the total length of a 3D polyline."""
    if points_xyz is None:
        return float("inf")
    arr = np.asarray(points_xyz, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] != 3:
        return float("inf")
    if arr.shape[0] < 2:
        return 0.0
    deltas = np.diff(arr, axis=0)
    return float(np.sum(np.linalg.norm(deltas, axis=1)))


def _obstacle_aware_waypoints(
    map_api: object | None,
    start_xyz: NDArray[np.float64],
    goal_xyz: NDArray[np.float64],
) -> tuple[NDArray[np.float64], bool]:
    """Return travel waypoints, preferring a grid path when available."""
    start = np.asarray(start_xyz, dtype=float).reshape(3)
    goal = np.asarray(goal_xyz, dtype=float).reshape(3)
    if map_api is not None:
        path = shortest_grid_path_points(map_api, start, goal, allow_diagonal=True)
        if path is not None and path.shape[0] >= 2:
            return np.asarray(path, dtype=float), True
        if _supports_grid_path(map_api):
            return np.zeros((0, 3), dtype=float), True
    return np.vstack([start, goal]).astype(float), False


def _supports_grid_path(map_api: object | None) -> bool:
    """Return True when a map API supports grid path planning."""
    if map_api is None:
        return False
    cell_index = getattr(map_api, "cell_index", None)
    grid_shape = getattr(map_api, "grid_shape", None)
    has_cell_free = any(
        callable(getattr(map_api, attr, None))
        for attr in ("is_free_cell", "is_cell_free")
    )
    return callable(cell_index) and grid_shape is not None and has_cell_free


def _filter_reachable_candidates(
    *,
    current_pose_xyz: NDArray[np.float64],
    map_api: object | None,
    candidates: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Keep only candidates connected to the current pose on the traversability grid."""
    candidate_arr = np.asarray(candidates, dtype=float)
    if candidate_arr.size == 0 or not _supports_grid_path(map_api):
        return candidate_arr
    reachable_mask = [
        shortest_grid_path_points(map_api, current_pose_xyz, candidate) is not None
        for candidate in candidate_arr
    ]
    return candidate_arr[np.asarray(reachable_mask, dtype=bool)]


def _build_robot_path_segment(
    *,
    map_api: object | None,
    from_pose_xyz: NDArray[np.float64],
    to_pose_xyz: NDArray[np.float64],
    nominal_motion_speed_m_s: float,
    path_planner: str,
    planned_shield_program: tuple[int, ...] | None,
    dss_diagnostics: dict[str, float | int | str] | None,
) -> dict[str, object]:
    """Build an obstacle-aware robot travel segment for timing and rendering."""
    waypoints, obstacle_aware = _obstacle_aware_waypoints(
        map_api,
        np.asarray(from_pose_xyz, dtype=float),
        np.asarray(to_pose_xyz, dtype=float),
    )
    distance_m = _polyline_distance(waypoints)
    if not np.isfinite(distance_m):
        raise RuntimeError(
            "Selected robot travel segment is not connected on the traversability grid."
        )
    motion_time_s = distance_m / max(float(nominal_motion_speed_m_s), 1e-9)
    return {
        "from_pose_xyz": [float(v) for v in np.asarray(from_pose_xyz, dtype=float)],
        "to_pose_xyz": [float(v) for v in np.asarray(to_pose_xyz, dtype=float)],
        "waypoints_xyz": [
            [float(coord) for coord in waypoint]
            for waypoint in np.asarray(waypoints, dtype=float)
        ],
        "distance_m": float(distance_m),
        "euclidean_distance_m": float(
            np.linalg.norm(
                np.asarray(to_pose_xyz, dtype=float) - np.asarray(from_pose_xyz, dtype=float)
            )
        ),
        "travel_time_s": float(motion_time_s),
        "speed_m_s": float(nominal_motion_speed_m_s),
        "obstacle_aware": bool(obstacle_aware),
        "path_planner": path_planner,
        "planned_shield_program": None
        if planned_shield_program is None
        else [int(v) for v in planned_shield_program],
        "dss_diagnostics": dss_diagnostics,
    }


def _estimate_best_next_pose_gain_rate(
    estimator: RotatingShieldPFEstimator,
    *,
    candidates: NDArray[np.float64],
    current_pose_xyz: NDArray[np.float64],
    map_api: object | None,
    live_time_s: float,
    rotation_limit: int,
    nominal_motion_speed_m_s: float,
    rotation_overhead_s: float,
    max_candidates: int,
) -> tuple[float, float, int]:
    """Estimate the best next-pose information gain rate for rotation stopping."""
    if candidates.size == 0:
        return 0.0, 0.0, -1
    candidate_arr = np.asarray(candidates, dtype=float)
    if candidate_arr.ndim != 2 or candidate_arr.shape[1] != 3:
        return 0.0, 0.0, -1
    limit = min(max(1, int(max_candidates)), int(candidate_arr.shape[0]))
    current_uncertainty = max(float(estimator.global_uncertainty()), 0.0)
    current_information = float(np.log1p(current_uncertainty))
    best_rate = 0.0
    best_gain = 0.0
    best_idx = -1
    for idx, candidate in enumerate(candidate_arr[:limit]):
        try:
            after_uncertainty = float(
                estimator.expected_uncertainty_after_rotation(
                    pose_xyz=candidate,
                    live_time_per_rot_s=live_time_s,
                    tau_ig=float(estimator.pf_config.ig_threshold),
                    tmax_s=float(max(1, rotation_limit)) * float(live_time_s),
                    n_rollouts=0,
                    orient_selection="IG",
                    rng_seed=idx,
                )
            )
        except RuntimeError:
            continue
        after_information = float(np.log1p(max(after_uncertainty, 0.0)))
        gain = max(current_information - after_information, 0.0)
        waypoints, _ = _obstacle_aware_waypoints(map_api, current_pose_xyz, candidate)
        travel = _polyline_distance(waypoints)
        if not np.isfinite(travel):
            continue
        travel_time = travel / max(float(nominal_motion_speed_m_s), 1e-9)
        cost_time = travel_time + float(rotation_overhead_s) + float(live_time_s)
        rate = gain / max(cost_time, 1e-9)
        if rate > best_rate:
            best_rate = float(rate)
            best_gain = float(gain)
            best_idx = int(idx)
    return best_rate, best_gain, best_idx


def _generate_planning_candidates(
    *,
    current_pose_xyz: NDArray[np.float64],
    map_api: object | None,
    n_candidates: int,
    min_dist_from_visited: float,
    visited_poses_xyz: NDArray[np.float64] | None,
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> tuple[NDArray[np.float64], bool, float]:
    """Generate next-pose candidates with one relaxed-spacing retry."""
    min_dist = max(float(min_dist_from_visited), 0.0)
    candidates = generate_candidate_poses(
        current_pose_xyz=current_pose_xyz,
        map_api=map_api,
        n_candidates=int(n_candidates),
        strategy="free_space_sobol",
        min_dist_from_visited=min_dist,
        visited_poses_xyz=visited_poses_xyz,
        bounds_xyz=bounds_xyz,
    )
    candidates = _filter_reachable_candidates(
        current_pose_xyz=current_pose_xyz,
        map_api=map_api,
        candidates=candidates,
    )
    if candidates.size != 0 or min_dist <= 0.0:
        return candidates, False, min_dist
    relaxed_dist = max(min_dist * 0.5, 0.5)
    candidates = generate_candidate_poses(
        current_pose_xyz=current_pose_xyz,
        map_api=map_api,
        n_candidates=max(int(n_candidates) * 2, int(n_candidates)),
        strategy="free_space_sobol",
        min_dist_from_visited=relaxed_dist,
        visited_poses_xyz=visited_poses_xyz,
        bounds_xyz=bounds_xyz,
    )
    candidates = _filter_reachable_candidates(
        current_pose_xyz=current_pose_xyz,
        map_api=map_api,
        candidates=candidates,
    )
    return candidates, True, relaxed_dist


def _coverage_fraction_for_poses(
    map_api: object | None,
    poses_xyz: Sequence[NDArray[np.float64]],
    *,
    radius_m: float,
) -> float:
    """Return traversable-map coverage by measurement poses within a radius."""
    if map_api is None or not poses_xyz:
        return 0.0
    grid_shape = getattr(map_api, "grid_shape", None)
    cell_size = getattr(map_api, "cell_size", None)
    origin = getattr(map_api, "origin", None)
    if grid_shape is None or cell_size is None or origin is None:
        return 0.0
    cell_center = getattr(map_api, "cell_center", None)
    traversable_cells = getattr(map_api, "traversable_cells", None)
    if traversable_cells is not None:
        cells = list(traversable_cells)
    else:
        is_cell_free = getattr(map_api, "is_free_cell", None)
        if not callable(is_cell_free):
            is_cell_free = getattr(map_api, "is_cell_free", None)
        if not callable(is_cell_free):
            return 0.0
        cells = [
            (ix, iy)
            for ix in range(int(grid_shape[0]))
            for iy in range(int(grid_shape[1]))
            if bool(is_cell_free((ix, iy)))
        ]
    if not cells:
        return 0.0
    pose_xy = np.asarray([np.asarray(pose, dtype=float)[:2] for pose in poses_xyz])
    radius = max(float(radius_m), 0.0)
    covered = 0
    for cell in cells:
        if callable(cell_center):
            center_xy = np.asarray(cell_center(cell), dtype=float)
        else:
            center_xy = np.asarray(
                [
                    float(origin[0]) + (float(cell[0]) + 0.5) * float(cell_size),
                    float(origin[1]) + (float(cell[1]) + 0.5) * float(cell_size),
                ],
                dtype=float,
            )
        if float(np.min(np.linalg.norm(pose_xy - center_xy, axis=1))) <= radius:
            covered += 1
    return float(covered) / float(len(cells))


def _adaptive_mission_stop_reason(
    estimator: RotatingShieldPFEstimator,
    *,
    current_pose_idx: int,
    visited_poses_xyz: Sequence[NDArray[np.float64]],
    map_api: object | None,
    min_poses: int,
    coverage_radius_m: float,
    coverage_fraction_threshold: float,
    ig_threshold: float,
    planning_live_time_s: float,
    require_quiet_birth_residual: bool = True,
    birth_residual_min_support: int = 1,
    require_pf_convergence_for_coverage: bool = False,
) -> str | None:
    """Return an adaptive mission-stop reason when exploration is sufficiently complete."""
    if len(visited_poses_xyz) < max(1, int(min_poses)):
        return None
    if estimator.should_stop_exploration(
        ig_threshold=float(ig_threshold),
        live_time_s=float(planning_live_time_s),
    ):
        return "pf_converged_low_information_gain"
    coverage = _coverage_fraction_for_poses(
        map_api,
        visited_poses_xyz,
        radius_m=float(coverage_radius_m),
    )
    if coverage >= float(coverage_fraction_threshold):
        if bool(require_quiet_birth_residual) and _has_birth_residual_evidence(
            estimator,
            min_support=int(birth_residual_min_support),
        ):
            return None
        if bool(require_pf_convergence_for_coverage) and not estimator.should_stop_exploration(
            ig_threshold=float(ig_threshold),
            live_time_s=float(planning_live_time_s),
        ):
            return None
        return f"environment_coverage:{coverage:.3f}"
    if current_pose_idx >= 0 and estimator.should_stop_shield_rotation(
        pose_idx=int(current_pose_idx),
        ig_threshold=float(ig_threshold),
        live_time_s=float(planning_live_time_s),
    ):
        return "current_pose_converged"
    return None


def _has_birth_residual_evidence(
    estimator: RotatingShieldPFEstimator,
    *,
    min_support: int,
) -> bool:
    """Return True when any isotope still has residual evidence for a new source."""
    support_floor = max(1, int(min_support))
    filters = getattr(estimator, "filters", {})
    for filt in filters.values():
        gate_passed = bool(getattr(filt, "last_birth_residual_gate_passed", False))
        support = int(getattr(filt, "last_birth_residual_support", 0))
        if gate_passed and support >= support_floor:
            return True
    return False


def _signature_vector_is_dependent(
    vector: NDArray[np.float64],
    previous_vectors: Sequence[NDArray[np.float64]],
    *,
    cosine_threshold: float,
    min_norm: float = 1e-9,
) -> bool:
    """Return True when a shield signature vector adds little new direction."""
    threshold = float(cosine_threshold)
    if threshold <= 0.0 or threshold >= 1.0:
        return False
    candidate = np.asarray(vector, dtype=float).ravel()
    candidate_norm = float(np.linalg.norm(candidate))
    if candidate_norm <= float(min_norm):
        return False
    for previous in previous_vectors:
        prev = np.asarray(previous, dtype=float).ravel()
        if prev.shape != candidate.shape:
            continue
        prev_norm = float(np.linalg.norm(prev))
        if prev_norm <= float(min_norm):
            continue
        cosine = float(np.dot(candidate, prev) / max(candidate_norm * prev_norm, min_norm))
        if abs(cosine) >= threshold:
            return True
    return False


def _save_spectrum_plot(
    decomposer: SpectralDecomposer,
    spectrum: np.ndarray,
    output_path: Path,
    peak_tolerance_keV: float = 10.0,
    highlight_isotopes: set[str] | None = None,
    counts_by_isotope: dict[str, float] | None = None,
    component_spectra_by_isotope: dict[str, NDArray[np.float64]] | None = None,
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
    component_labels: list[str] = []
    component_values: list[NDArray[np.float64]] = []
    component_colors: list[object] = []
    if component_spectra_by_isotope:
        for iso in library:
            component_raw = component_spectra_by_isotope.get(iso)
            if component_raw is None:
                continue
            component = np.clip(
                np.asarray(component_raw, dtype=float)[: energy_axis.size],
                a_min=0.0,
                a_max=None,
            )
            if component.size != energy_axis.size or float(np.sum(component)) <= 0.0:
                continue
            component_values.append(component)
            component_colors.append(colors.get(iso, "gray"))
            component_labels.append(f"{iso} photopeak={float(np.sum(component)):.1f}")
    if component_values:
        ax.stackplot(
            energy_axis,
            component_values,
            labels=component_labels,
            colors=component_colors,
            alpha=0.45,
        )
    ax.plot(energy_axis, smoothed, color="black", linewidth=1.0, label="Processed spectrum")
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
    adaptive_ready_min_snr: float,
) -> None:
    """Validate fixed and adaptive dwell-time settings."""
    if measurement_time_s <= 0.0 and not adaptive_dwell:
        raise ValueError("measurement_time_s must be positive for fixed dwell.")
    if not adaptive_dwell:
        return
    if adaptive_dwell_chunk_s <= 0.0:
        raise ValueError("adaptive_dwell_chunk_s must be positive.")
    if adaptive_min_dwell_s <= 0.0:
        raise ValueError("adaptive_min_dwell_s must be positive.")
    has_dwell_cap = measurement_time_s > 0.0 and np.isfinite(measurement_time_s)
    if has_dwell_cap and adaptive_min_dwell_s > measurement_time_s:
        raise ValueError("adaptive_min_dwell_s cannot exceed measurement_time_s.")
    if adaptive_ready_min_counts < 0.0:
        raise ValueError("adaptive_ready_min_counts cannot be negative.")
    if adaptive_ready_min_isotopes < 0:
        raise ValueError("adaptive_ready_min_isotopes cannot be negative.")
    if adaptive_ready_min_snr < 0.0:
        raise ValueError("adaptive_ready_min_snr cannot be negative.")


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


def _should_fold_incident_gamma_detector_response(
    observation: SimulationObservation,
    decomposer: SpectralDecomposer,
) -> bool:
    """Return whether fast Geant4 incident-gamma spectra need detector response folding."""
    if not bool(decomposer.config.apply_incident_gamma_detector_response):
        return False
    metadata = observation.metadata
    scoring_mode = str(metadata.get("detector_scoring_mode", "")).strip().lower()
    fast_scoring = str(metadata.get("detector_fast_scoring", "")).strip().lower()
    return scoring_mode == "incident_gamma_energy" or fast_scoring == "true"


def _analysis_spectrum_array(
    observation: SimulationObservation,
    decomposer: SpectralDecomposer,
) -> NDArray[np.float64]:
    """Return the pulse-height spectrum used for display and isotope count extraction."""
    spectrum = _observation_spectrum_array(observation, decomposer)
    if _should_fold_incident_gamma_detector_response(observation, decomposer):
        return decomposer.fold_incident_gamma_spectrum(spectrum)
    return spectrum


def _analysis_spectrum_variance(
    observation: SimulationObservation,
    decomposer: SpectralDecomposer,
) -> NDArray[np.float64] | None:
    """Return the variance of the analysis spectrum when simulator metadata provides it."""
    raw_spectrum = _observation_spectrum_array(observation, decomposer)
    variance = _metadata_spectrum_variance(observation.metadata, raw_spectrum.shape)
    if variance is None:
        return None
    if _should_fold_incident_gamma_detector_response(observation, decomposer):
        return decomposer.fold_incident_gamma_spectrum_variance(variance)
    return variance


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
    transport_metadata: dict[str, object] | None = None,
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
    effective_entries = None
    if spectrum_variance is not None:
        total_sum = float(np.sum(np.clip(spectrum, a_min=0.0, a_max=None)))
        total_variance = float(np.sum(np.clip(spectrum_variance, a_min=0.0, a_max=None)))
        if total_sum > 0.0 and total_variance > 0.0:
            effective_entries = (total_sum * total_sum) / total_variance
    metadata_effective_entries = (
        None
        if transport_metadata is None
        else _metadata_float(transport_metadata, "weighted_spectrum_effective_entries")
    )
    if effective_entries is None:
        effective_entries = metadata_effective_entries
    if effective_entries is not None and effective_entries > 0.0:
        variances = {
            iso: float(
                max(
                    variances.get(iso, 1.0),
                    (max(float(count), 0.0) ** 2) / max(float(effective_entries), 1.0),
                )
            )
            for iso, count in counts_out.items()
        }
    return counts_out, variances, set(detected)


def _is_adaptive_spectrum_ready(
    counts_by_isotope: dict[str, float],
    count_variance_by_isotope: dict[str, float] | None = None,
    *,
    live_time_s: float,
    min_live_time_s: float,
    min_counts_per_detected_isotope: float,
    min_detected_isotopes: int,
    candidate_isotopes: list[str] | tuple[str, ...] | None = None,
    min_snr: float = 3.0,
    total_spectrum_counts: float | None = None,
    informative_low_total_factor: float = 20.0,
    informative_low_fraction: float = 0.01,
    informative_low_min_live_s: float = 20.0,
    informative_low_count_fraction: float = 0.5,
    allow_informative_low: bool = True,
    allow_low_signal_stop: bool = False,
    low_signal_min_live_s: float = 120.0,
    low_signal_upper_sigma: float = 3.0,
    low_signal_count_fraction: float = 0.05,
    low_signal_projected_live_factor: float = 4.0,
) -> tuple[bool, str]:
    """Return whether an accumulated spectrum is usable for isotope counts."""
    if live_time_s + 1e-12 < min_live_time_s:
        return False, "below_min_live_time"
    if min_detected_isotopes <= 0:
        return True, "min_live_time_reached"
    if candidate_isotopes is None:
        candidate_isotopes = tuple(sorted(counts_by_isotope))
    required = min(int(min_detected_isotopes), len(candidate_isotopes))
    if required <= 0:
        return True, "min_live_time_reached"
    min_counts = float(min_counts_per_detected_isotope)
    min_snr = max(float(min_snr), 0.0)
    total_counts = (
        None
        if total_spectrum_counts is None
        else max(float(total_spectrum_counts), 0.0)
    )
    informative_total_floor = max(min_counts * float(informative_low_total_factor), 0.0)
    informative_fraction = max(float(informative_low_fraction), 0.0)
    informative_count_ceiling = max(
        1.0,
        min_counts * max(float(informative_low_count_fraction), 0.0),
    )
    enough_live_for_low = live_time_s + 1e-12 >= max(
        min_live_time_s,
        float(informative_low_min_live_s),
    )
    variances = count_variance_by_isotope or {}
    usable = []
    informative_low = []
    unresolved_upper_below = []
    unresolved_count_below = []
    projected_unproductive = []
    low_count_floor = max(min_counts * max(float(low_signal_count_fraction), 0.0), 0.0)
    projected_limit = max(
        live_time_s,
        max(min_live_time_s, float(low_signal_min_live_s))
        * max(float(low_signal_projected_live_factor), 1.0),
    )
    for iso in candidate_isotopes:
        count = max(float(counts_by_isotope.get(iso, 0.0)), 0.0)
        variance = max(float(variances.get(iso, max(count, 1.0))), 1.0)
        snr = count / np.sqrt(variance)
        count_ready = count >= min_counts
        snr_ready = True if min_counts <= 0.0 else snr >= min_snr
        if count_ready and snr_ready:
            usable.append(str(iso))
            continue
        upper_bound = count + max(float(low_signal_upper_sigma), 0.0) * np.sqrt(variance)
        if min_counts > 0.0 and upper_bound < min_counts:
            unresolved_upper_below.append(str(iso))
        if min_counts > 0.0 and count <= low_count_floor:
            unresolved_count_below.append(str(iso))
        if allow_informative_low:
            has_spectrum_evidence = (
                total_counts is not None
                and total_counts >= informative_total_floor
                and total_counts > 0.0
            )
            low_fraction = (
                enough_live_for_low
                and has_spectrum_evidence
                and count < min_counts
                and count <= informative_count_ceiling
                and (count / max(total_counts, 1.0)) <= informative_fraction
            )
            if low_fraction:
                informative_low.append(str(iso))
                continue
        if (
            allow_low_signal_stop
            and live_time_s + 1e-12 >= max(min_live_time_s, float(low_signal_min_live_s))
        ):
            projected_live_s = 0.0
            if min_counts > 0.0 and not count_ready:
                projected_live_s = (
                    float("inf")
                    if count <= 0.0
                    else live_time_s * min_counts / max(count, 1.0e-12)
                )
            elif count_ready and not snr_ready and min_snr > 0.0:
                projected_live_s = (
                    float("inf")
                    if snr <= 0.0
                    else live_time_s * (min_snr / max(snr, 1.0e-12)) ** 2
                )
            if projected_live_s > projected_limit:
                projected_unproductive.append(
                    (str(iso), float(count), float(projected_live_s))
                )
    if len(usable) + len(informative_low) >= required:
        if informative_low:
            return (
                True,
                "isotope_count_estimates_ready:"
                f"positive={len(usable)},informative_low={len(informative_low)}",
            )
        return True, "isotope_count_estimates_ready"
    unresolved_count = len(candidate_isotopes) - len(usable)
    if (
        allow_low_signal_stop
        and min_counts > 0.0
        and unresolved_count > 0
        and live_time_s + 1e-12 >= max(min_live_time_s, float(low_signal_min_live_s))
        and (
            len(unresolved_upper_below) >= unresolved_count
            or len(unresolved_count_below) >= unresolved_count
        )
    ):
        if len(unresolved_upper_below) >= unresolved_count:
            reason_kind = "upper_bound"
            evidence_count = len(unresolved_upper_below)
        else:
            reason_kind = "count_floor"
            evidence_count = len(unresolved_count_below)
        return (
            True,
            f"low_signal_{reason_kind}:"
            f"positive={len(usable)},below={evidence_count}",
        )
    if (
        allow_low_signal_stop
        and min_counts > 0.0
        and live_time_s + 1e-12 >= max(min_live_time_s, float(low_signal_min_live_s))
        and projected_unproductive
    ):
        projected_blocked = {iso for iso, _, _ in projected_unproductive}
        available_count = sum(
            1
            for iso in candidate_isotopes
            if str(iso) not in projected_blocked
        )
        if available_count < required:
            best_iso, best_count, best_projected = max(
                projected_unproductive,
                key=lambda item: item[1],
            )
            return (
                True,
                "low_signal_projected_time:"
                f"positive={len(usable)},available={available_count},"
                f"blocked={len(projected_blocked)},best={best_count:.3f},"
                f"best_iso={best_iso},projected={best_projected:.1f}",
            )
    if (
        allow_low_signal_stop
        and min_counts > 0.0
        and not usable
        and live_time_s + 1e-12 >= max(min_live_time_s, float(low_signal_min_live_s))
    ):
        max_count = max(
            (
                max(float(counts_by_isotope.get(iso, 0.0)), 0.0)
                for iso in candidate_isotopes
            ),
            default=0.0,
        )
        if max_count <= 0.0:
            return (
                True,
                "low_signal_projected_time:positive=0,best=0.000,projected=inf",
            )
        projected_live_s = live_time_s * min_counts / max(max_count, 1e-12)
        if projected_live_s > projected_limit:
            return (
                True,
                "low_signal_projected_time:"
                f"positive=0,best={max_count:.3f},projected={projected_live_s:.1f}",
            )
    return (
        False,
        "insufficient_isotope_count_estimates:"
        f"{len(usable) + len(informative_low)}/{required}",
    )


def _adaptive_dwell_progress_message(
    *,
    step_id: int,
    chunk_index: int,
    live_time_s: float,
    counts_by_isotope: dict[str, float],
    count_variance_by_isotope: dict[str, float],
    reason: str,
) -> str:
    """Return a concise adaptive-dwell progress diagnostic message."""
    snr_by_isotope = {}
    for iso, count in counts_by_isotope.items():
        variance = max(float(count_variance_by_isotope.get(iso, max(count, 1.0))), 1.0)
        snr_by_isotope[iso] = float(max(count, 0.0) / np.sqrt(variance))
    return (
        f"[adaptive dwell step {step_id}] "
        f"chunks={chunk_index + 1} live={live_time_s:.1f}s "
        f"counts={_fmt_counts(counts_by_isotope)} "
        f"snr={_fmt_counts(snr_by_isotope)} "
        f"reason={reason}"
    )


def _inflate_low_signal_variances(
    counts_by_isotope: dict[str, float],
    count_variance_by_isotope: dict[str, float],
    *,
    min_counts_per_detected_isotope: float,
    ready_reason: str,
) -> dict[str, float]:
    """Return count variances with censored low-signal observations softened."""
    reason = str(ready_reason)
    should_soften_subthreshold = (
        reason.startswith("low_signal_")
        or reason.startswith("max_dwell_reached")
        or reason == "isotope_count_estimates_ready"
    )
    if not should_soften_subthreshold:
        return {
            iso: float(max(var, 1.0))
            for iso, var in count_variance_by_isotope.items()
        }
    threshold_var = max(float(min_counts_per_detected_isotope), 1.0) ** 2
    inflated: dict[str, float] = {}
    for isotope, count in counts_by_isotope.items():
        base_var = float(count_variance_by_isotope.get(isotope, max(float(count), 1.0)))
        if max(float(count), 0.0) < float(min_counts_per_detected_isotope):
            inflated[isotope] = float(max(base_var, threshold_var, 1.0))
        else:
            inflated[isotope] = float(max(base_var, 1.0))
    return inflated


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
    additive_metadata_keys = {
        "num_primaries",
        "expected_physical_primaries",
        "expected_detector_equivalent_primaries",
        "expected_sampled_primaries",
        "total_spectrum_counts",
        "total_track_steps",
        "detector_hit_events",
        "detector_hit_steps",
        "secondary_count",
        "killed_non_gamma_secondary_count",
        "process_count_compton",
        "process_count_rayleigh",
        "process_count_photoelectric",
        "weighted_spectrum_sumw2",
        "run_time_s",
    }
    additive_metadata_keys.update(
        key
        for observation in observations
        for key in observation.metadata
        if str(key).startswith("source_equivalent_counts_")
    )
    additive_metadata_keys.update(
        key
        for observation in observations
        for key in observation.metadata
        if str(key).startswith("transport_detected_counts_")
    )
    for key in sorted(additive_metadata_keys):
        values = [
            _metadata_float(observation.metadata, key)
            for observation in observations
        ]
        finite_values = [value for value in values if value is not None]
        if finite_values:
            metadata[key] = float(sum(finite_values))
    spectrum_total_sum = float(np.sum(np.clip(spectrum_total, a_min=0.0, a_max=None)))
    metadata["total_spectrum_counts"] = spectrum_total_sum
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
        variance_total = float(np.sum(spectrum_variance_total))
        metadata["spectrum_count_variance_total"] = variance_total
        if spectrum_total_sum > 0.0 and variance_total > 0.0:
            metadata["weighted_spectrum_effective_entries"] = float(
                (spectrum_total_sum * spectrum_total_sum) / variance_total
            )
    run_time_total = _metadata_float(metadata, "run_time_s")
    if run_time_total is not None and run_time_total > 0.0:
        primaries_total = _metadata_float(metadata, "num_primaries")
        effective_entries = _metadata_float(metadata, "weighted_spectrum_effective_entries")
        if primaries_total is not None:
            metadata["primaries_per_sec"] = float(primaries_total / run_time_total)
        if effective_entries is not None:
            metadata["effective_entries_per_sec"] = float(
                effective_entries / run_time_total
            )
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
    adaptive_ready_min_snr: float,
    spectrum_count_method: str,
    detect_threshold_abs: float,
    detect_threshold_rel: float,
    detect_threshold_rel_by_isotope: dict[str, float],
    min_peaks_by_isotope: dict[str, int] | None,
    adaptive_progress_every_chunks: int = 0,
    adaptive_ready_allow_informative_low: bool = True,
    adaptive_allow_low_signal_stop: bool = False,
    adaptive_low_signal_min_live_s: float = 120.0,
    adaptive_low_signal_upper_sigma: float = 3.0,
    adaptive_low_signal_count_fraction: float = 0.05,
    adaptive_low_signal_projected_live_factor: float = 4.0,
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
        spectrum = _analysis_spectrum_array(observation, decomposer)
        spectrum_variance = _analysis_spectrum_variance(observation, decomposer)
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
            transport_metadata=observation.metadata,
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
    has_dwell_cap = np.isfinite(float(live_time_s)) and float(live_time_s) > 0.0
    ready_reason = "max_dwell_reached" if has_dwell_cap else "uncapped_dwell_not_ready"
    last_ready = False
    chunk_index = 0
    while True:
        if has_dwell_cap and accumulated_live_time_s + 1e-12 >= float(live_time_s):
            break
        remaining_s = (
            float(live_time_s) - accumulated_live_time_s
            if has_dwell_cap
            else float("inf")
        )
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
        spectrum = _analysis_spectrum_array(observation, decomposer)
        spectrum_variance = _analysis_spectrum_variance(observation, decomposer)
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
            transport_metadata=observation.metadata,
        )
        ready, reason = _is_adaptive_spectrum_ready(
            last_counts,
            last_variances,
            live_time_s=accumulated_live_time_s,
            min_live_time_s=float(adaptive_min_dwell_s),
            min_counts_per_detected_isotope=float(adaptive_ready_min_counts),
            min_detected_isotopes=int(adaptive_ready_min_isotopes),
            candidate_isotopes=list(decomposer.isotope_names),
            min_snr=float(adaptive_ready_min_snr),
            total_spectrum_counts=float(np.sum(np.clip(accumulated_spectrum, 0.0, None))),
            allow_informative_low=bool(adaptive_ready_allow_informative_low),
            allow_low_signal_stop=bool(adaptive_allow_low_signal_stop),
            low_signal_min_live_s=float(adaptive_low_signal_min_live_s),
            low_signal_upper_sigma=float(adaptive_low_signal_upper_sigma),
            low_signal_count_fraction=float(adaptive_low_signal_count_fraction),
            low_signal_projected_live_factor=float(
                adaptive_low_signal_projected_live_factor
            ),
        )
        last_ready = ready
        ready_reason = reason
        progress_every = int(adaptive_progress_every_chunks)
        should_log_progress = (
            progress_every > 0
            and (
                chunk_index == 0
                or ready
                or (chunk_index + 1) % progress_every == 0
            )
        )
        if should_log_progress:
            print(
                _adaptive_dwell_progress_message(
                    step_id=step_id,
                    chunk_index=chunk_index,
                    live_time_s=accumulated_live_time_s,
                    counts_by_isotope=last_counts,
                    count_variance_by_isotope=last_variances,
                    reason=reason,
                ),
                flush=True,
            )
        if ready:
            break
        chunk_index += 1
    reached_dwell_cap = has_dwell_cap and accumulated_live_time_s + 1e-12 >= live_time_s
    if reached_dwell_cap and not last_ready:
        ready_reason = f"max_dwell_reached:{ready_reason}"
    last_variances = _inflate_low_signal_variances(
        last_counts,
        last_variances,
        min_counts_per_detected_isotope=float(adaptive_ready_min_counts),
        ready_reason=ready_reason,
    )
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
    adaptive_ready_min_snr: float = 0.0,
    adaptive_strength_prior: bool = True,
    adaptive_strength_prior_steps: int = 3,
    adaptive_strength_prior_min_counts: float = 3.0,
    adaptive_strength_prior_log_sigma: float = 0.7,
    pose_min_observation_counts: float | None = None,
    pose_min_observation_penalty_scale: float = 1.0,
    pose_min_observation_aggregate: str = "max",
    path_planner: str | None = None,
    dss_horizon: int | None = None,
    dss_beam_width: int | None = None,
    dss_program_length: int | None = None,
    dss_signature_weight: float | None = None,
    dss_differential_weight: float | None = None,
    dss_rotation_weight: float | None = None,
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
            Values <= 0 remove the adaptive dwell cap.
        adaptive_dwell: Stop each measurement once isotope counts are reliable enough.
        adaptive_dwell_chunk_s: Geant4 dwell duration for each adaptive chunk.
        adaptive_min_dwell_s: Minimum accumulated dwell before early stopping.
        adaptive_ready_min_counts: Minimum count estimate per detected isotope.
        adaptive_ready_min_isotopes: Required number of detected isotopes for readiness.
        adaptive_ready_min_snr: Optional minimum count-estimate SNR for dwell readiness.
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
        path_planner: Pose planner name. Use "one_step" for the original
            selector or "dss_pp" for joint pose-shield planning.
        dss_horizon: DSS-PP receding-horizon length.
        dss_beam_width: DSS-PP beam width.
        dss_program_length: Number of shield postures in each DSS program.
        dss_signature_weight: DSS shield-signature separation weight.
        dss_differential_weight: DSS differential-observability penalty weight.
        dss_rotation_weight: DSS shield-transition penalty weight.
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
        adaptive_ready_min_snr=float(adaptive_ready_min_snr),
    )
    notifier = PiplupNotifier(notification_config)
    live = _coerce_live_visualization(live)
    runtime_config = load_runtime_config(sim_config_path)
    adaptive_ready_allow_informative_low = bool(
        runtime_config.get("adaptive_ready_allow_informative_low", False)
    )
    adaptive_allow_low_signal_stop = bool(
        runtime_config.get("adaptive_allow_low_signal_stop", False)
    )
    adaptive_low_signal_min_live_s = max(
        0.0,
        float(runtime_config.get("adaptive_low_signal_min_live_s", 120.0)),
    )
    adaptive_low_signal_upper_sigma = max(
        0.0,
        float(runtime_config.get("adaptive_low_signal_upper_sigma", 3.0)),
    )
    adaptive_low_signal_count_fraction = max(
        0.0,
        float(runtime_config.get("adaptive_low_signal_count_fraction", 0.05)),
    )
    adaptive_low_signal_projected_live_factor = max(
        1.0,
        float(runtime_config.get("adaptive_low_signal_projected_live_factor", 4.0)),
    )
    env = EnvironmentConfig(size_x=10.0, size_y=20.0, size_z=10.0, detector_position=(1.0, 1.0, 0.5))
    sources = _build_demo_sources() if sources is None else sources
    spectrum_config = _spectrum_config_from_runtime_config(runtime_config)
    decomposer = SpectralDecomposer(spectrum_config=spectrum_config)
    diagnostic_decomposer = SpectralDecomposer(
        spectrum_config=spectrum_config,
        library=decomposer.library,
    )
    default_count_method = "response_poisson"
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
    cui_split_view_enabled = bool(
        runtime_config.get("cui_split_view", bool(save_outputs and not live))
    )
    cui_split_view_dir_raw = runtime_config.get("cui_split_view_dir")
    if cui_split_view_dir_raw is None:
        cui_split_view_dir = CUI_VIEW_DIR / "latest"
    else:
        cui_split_view_dir = Path(str(cui_split_view_dir_raw)).expanduser()
        if not cui_split_view_dir.is_absolute():
            cui_split_view_dir = ROOT / cui_split_view_dir
    cui_split_max_particles_raw = runtime_config.get(
        "cui_split_view_max_particles_per_isotope",
        None,
    )
    cui_split_max_particles = (
        None
        if cui_split_max_particles_raw is None
        else int(cui_split_max_particles_raw)
    )
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
    source_position_min, source_position_max = _resolve_source_position_bounds(
        env,
        runtime_config,
    )
    grid = _build_candidate_sources(
        env,
        spacing=spacing,
        margin=float(candidate_grid_margin),
        position_min=source_position_min,
        position_max=source_position_max,
    )

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
    shield_thickness = resolve_shield_thickness_config(runtime_config)
    buildup_runtime = runtime_config.get("pf_buildup", {})
    if not isinstance(buildup_runtime, dict):
        buildup_runtime = {}
    shield_params = ShieldParams(
        thickness_fe_cm=float(shield_thickness.thickness_fe_cm),
        thickness_pb_cm=float(shield_thickness.thickness_pb_cm),
        buildup_fe_coeff=float(
            buildup_runtime.get(
                "fe_coeff",
                runtime_config.get("pf_buildup_fe_coeff", 0.0),
            )
        ),
        buildup_pb_coeff=float(
            buildup_runtime.get(
                "pb_coeff",
                runtime_config.get("pf_buildup_pb_coeff", 0.0),
            )
        ),
    )
    obstacle_buildup_coeff = float(
        buildup_runtime.get(
            "obstacle_coeff",
            runtime_config.get("pf_obstacle_buildup_coeff", 0.0),
        )
    )
    print(
        "Shield thickness model: "
        f"scale={float(shield_thickness.thickness_scale):.6g} "
        f"target_transmission={shield_thickness.transmission_target} "
        f"Fe={float(shield_params.thickness_fe_cm):.4f}cm "
        f"Pb={float(shield_params.thickness_pb_cm):.4f}cm "
        f"buildup=(Fe {shield_params.buildup_fe_coeff:.3g}, "
        f"Pb {shield_params.buildup_pb_coeff:.3g}, "
        f"obstacle {obstacle_buildup_coeff:.3g})"
    )
    mu_by_isotope = mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM, isotopes=isotopes)
    if not mu_by_isotope:
        mu_by_isotope = {
            iso: {"fe": shield_params.mu_fe, "pb": shield_params.mu_pb} for iso in isotopes
        }
    use_gpu = _default_use_gpu()
    background_by_isotope = {iso: 5.0 for iso in isotopes}
    live_time = float(measurement_time_s)
    has_live_time_cap = np.isfinite(live_time) and live_time > 0.0
    planning_live_time = (
        live_time
        if has_live_time_cap
        else max(float(adaptive_min_dwell_s), float(adaptive_dwell_chunk_s))
    )
    observation_snr_floor = max(
        0.0,
        float(runtime_config.get("pose_min_observation_snr", 5.0)),
    )
    default_min_observation_counts = observation_snr_floor * observation_snr_floor
    if pose_min_observation_counts is None:
        pose_min_observation_counts_resolved = runtime_config.get(
            "pose_min_observation_counts",
            default_min_observation_counts,
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
    pose_min_observation_quantile = float(
        runtime_config.get("pose_min_observation_quantile", 0.25)
    )
    path_planner_resolved = str(
        path_planner
        if path_planner is not None
        else runtime_config.get("path_planner", "one_step")
    ).strip().lower()
    if path_planner_resolved in {"dss", "dss-pp", "dsspp"}:
        path_planner_resolved = "dss_pp"
    if path_planner_resolved not in {"one_step", "dss_pp"}:
        raise ValueError("path_planner must be 'one_step' or 'dss_pp'.")
    dss_runtime = runtime_config.get("dss_pp", {})
    if not isinstance(dss_runtime, dict):
        dss_runtime = {}

    def _dss_value(key: str, default: object) -> object:
        """Read a DSS-PP setting from CLI override or runtime config."""
        return dss_runtime.get(key, runtime_config.get(f"dss_{key}", default))

    dss_horizon_resolved = int(
        dss_horizon
        if dss_horizon is not None
        else _dss_value("horizon", 2)
    )
    dss_beam_width_resolved = int(
        dss_beam_width
        if dss_beam_width is not None
        else _dss_value("beam_width", 8)
    )
    dss_program_length_resolved = int(
        dss_program_length
        if dss_program_length is not None
        else _dss_value("program_length", 2)
    )
    dss_signature_weight_resolved = float(
        dss_signature_weight
        if dss_signature_weight is not None
        else _dss_value("signature_weight", 1.0)
    )
    dss_differential_weight_resolved = float(
        dss_differential_weight
        if dss_differential_weight is not None
        else _dss_value("differential_weight", 1.0)
    )
    dss_rotation_weight_resolved = float(
        dss_rotation_weight
        if dss_rotation_weight is not None
        else _dss_value("rotation_weight", 0.15)
    )
    dss_planning_particles_resolved = _dss_value(
        "planning_particles",
        runtime_config.get("planning_rollout_particles", 512),
    )
    dss_planning_method_resolved = _dss_value(
        "planning_method",
        runtime_config.get("planning_rollout_method", "resample"),
    )
    dss_config = DSSPPConfig(
        horizon=max(1, dss_horizon_resolved),
        beam_width=max(1, dss_beam_width_resolved),
        max_programs=max(1, int(_dss_value("max_programs", 24))),
        program_length=max(1, dss_program_length_resolved),
        mode_cluster_radius_m=float(_dss_value("mode_cluster_radius_m", 1.5)),
        max_modes_per_isotope=max(1, int(_dss_value("max_modes_per_isotope", 4))),
        planning_particles=(
            None
            if dss_planning_particles_resolved is None
            else int(dss_planning_particles_resolved)
        ),
        planning_method=(
            None
            if dss_planning_method_resolved is None
            else str(dss_planning_method_resolved)
        ),
        live_time_s=planning_live_time,
        lambda_eig=float(_dss_value("eig_weight", 1.0)),
        lambda_signature=max(0.0, dss_signature_weight_resolved),
        lambda_distance=(
            None
            if _dss_value("distance_weight", None) is None
            else float(_dss_value("distance_weight", 0.0))
        ),
        lambda_time=max(0.0, float(_dss_value("time_weight", 0.0))),
        lambda_rotation=max(0.0, dss_rotation_weight_resolved),
        lambda_dose=max(0.0, float(_dss_value("dose_weight", 0.0))),
        lambda_coverage=max(0.0, float(_dss_value("coverage_weight", 0.0))),
        lambda_bearing_diversity=max(
            0.0,
            float(_dss_value("bearing_diversity_weight", 0.0)),
        ),
        lambda_frontier=max(
            0.0,
            float(_dss_value("frontier_weight", 0.0)),
        ),
        lambda_turn_smoothness=max(
            0.0,
            float(_dss_value("turn_smoothness_weight", 0.0)),
        ),
        eta_observation=max(
            0.0,
            float(
                _dss_value(
                    "observation_weight",
                    pose_min_observation_penalty_scale,
                )
            ),
        ),
        eta_differential=max(0.0, dss_differential_weight_resolved),
        eta_count_balance=max(
            0.0,
            float(
                _dss_value(
                    "count_balance_weight",
                    runtime_config.get("shield_count_balance_weight", 0.5),
                )
            ),
        ),
        eta_revisit=max(
            0.0,
            float(_dss_value("revisit_penalty_weight", 0.0)),
        ),
        min_observation_counts=pose_min_observation_counts_resolved,
        enforce_min_observation=bool(
            _dss_value("enforce_min_observation", True)
        ),
        signature_std_min_counts=max(
            0.0,
            float(_dss_value("signature_std_min_counts", 1.0)),
        ),
        count_variance_floor=max(
            1e-12,
            float(_dss_value("count_variance_floor", 1.0)),
        ),
        coverage_radius_m=max(
            0.0,
            float(_dss_value("coverage_radius_m", 3.0)),
        ),
        coverage_grid_max_cells=max(
            0,
            int(_dss_value("coverage_grid_max_cells", 5000)),
        ),
        coverage_floor_quantile=float(
            _dss_value("coverage_floor_quantile", 0.0)
        ),
        coverage_floor_weight=max(
            0.0,
            float(_dss_value("coverage_floor_weight", 0.0)),
        ),
        min_station_separation_m=max(
            0.0,
            float(_dss_value("min_station_separation_m", pose_min_dist)),
        ),
        detector_aperture_samples=max(
            1,
            int(_dss_value("detector_aperture_samples", 1)),
        ),
        robot_speed_m_s=float(nominal_motion_speed_m_s),
        rotation_overhead_s=float(rotation_overhead_s),
        augment_candidates=bool(_dss_value("augment_candidates", True)),
        max_augmented_candidates=max(
            pose_candidates,
            int(_dss_value("max_augmented_candidates", 256)),
        ),
        eig_candidate_limit=(
            None
            if _dss_value("eig_candidate_limit", 64) is None
            else int(_dss_value("eig_candidate_limit", 64))
        ),
        rng_seed=obstacle_seed,
    )
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
        0.50 if geant4_likelihood_defaults else 0.0,
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
    adaptive_mission_stop = bool(runtime_config.get("adaptive_mission_stop", False))
    mission_stop_min_poses = max(
        1,
        int(runtime_config.get("mission_stop_min_poses", 3)),
    )
    mission_stop_coverage_radius_m = max(
        0.0,
        float(runtime_config.get("mission_stop_coverage_radius_m", 4.0)),
    )
    mission_stop_coverage_fraction = float(
        np.clip(
            float(runtime_config.get("mission_stop_coverage_fraction", 0.85)),
            0.0,
            1.0,
        )
    )
    mission_stop_require_quiet_birth_residual = bool(
        runtime_config.get("mission_stop_require_quiet_birth_residual", True)
    )
    mission_stop_require_pf_convergence_for_coverage = bool(
        runtime_config.get("mission_stop_require_pf_convergence_for_coverage", False)
    )
    mission_stop_birth_residual_min_support = max(
        1,
        int(runtime_config.get("mission_stop_birth_residual_min_support", 1)),
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
    orientation_limit_resolved = max(
        1,
        int(
            runtime_config.get(
                "orientation_k",
                runtime_config.get("rotations_per_pose", 2),
            )
        ),
    )
    min_rotations_resolved = max(
        0,
        int(
            runtime_config.get(
                "min_rotations_per_pose",
                min(2, orientation_limit_resolved),
            )
        ),
    )
    pf_max_sources_raw = runtime_config.get("pf_max_sources", 3)
    pf_max_sources = (
        None
        if pf_max_sources_raw is None
        else max(1, int(pf_max_sources_raw))
    )
    init_num_sources_raw = runtime_config.get("init_num_sources", None)
    if isinstance(init_num_sources_raw, (list, tuple)) and len(init_num_sources_raw) == 2:
        init_num_sources = (
            max(0, int(init_num_sources_raw[0])),
            max(0, int(init_num_sources_raw[1])),
        )
    else:
        default_init_max = min(3, pf_max_sources if pf_max_sources is not None else 3)
        init_num_sources = (
            max(0, int(runtime_config.get("init_num_sources_min", 0 if birth_enabled else 1))),
            max(0, int(runtime_config.get("init_num_sources_max", default_init_max))),
        )
    if init_num_sources[1] < init_num_sources[0]:
        init_num_sources = (init_num_sources[1], init_num_sources[0])
    if pf_max_sources is not None:
        init_num_sources = (
            min(init_num_sources[0], pf_max_sources),
            min(init_num_sources[1], pf_max_sources),
        )
    if init_num_sources[1] <= 0 and not birth_enabled:
        init_num_sources = (1, 1)
    parallel_isotope_workers_raw = runtime_config.get("parallel_isotope_workers", None)
    parallel_isotope_workers = (
        None
        if parallel_isotope_workers_raw is None
        else max(1, int(parallel_isotope_workers_raw))
    )
    birth_jitter_topk_raw = runtime_config.get("birth_jitter_topk_candidates", 512)
    birth_jitter_topk_candidates = (
        None
        if birth_jitter_topk_raw is None
        else max(1, int(birth_jitter_topk_raw))
    )
    structural_proposal_topk_raw = runtime_config.get(
        "structural_proposal_topk_particles",
        None,
    )
    structural_proposal_topk_particles = (
        None
        if structural_proposal_topk_raw is None
        else max(1, int(structural_proposal_topk_raw))
    )
    pf_conf = RotatingShieldPFConfig(
        num_particles=num_particles,
        min_particles=num_particles,
        max_particles=num_particles,
        max_sources=pf_max_sources,
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
        short_time_s=planning_live_time,
        max_dwell_time_s=10000.0,
        position_min=source_position_min,
        position_max=source_position_max,
        init_num_sources=init_num_sources,
        init_grid_spacing_m=1.0,
        init_grid_repeats=max(1, int(runtime_config.get("init_grid_repeats", 1))),
        adaptive_strength_prior=bool(adaptive_strength_prior),
        adaptive_strength_prior_steps=int(adaptive_strength_prior_steps),
        adaptive_strength_prior_min_counts=float(adaptive_strength_prior_min_counts),
        adaptive_strength_prior_log_sigma=float(adaptive_strength_prior_log_sigma),
        pose_min_observation_counts=pose_min_observation_counts_resolved,
        pose_min_observation_penalty_scale=pose_min_observation_penalty_scale,
        pose_min_observation_aggregate=pose_min_observation_aggregate,
        pose_min_observation_max_particles=pose_min_observation_max_particles,
        pose_min_observation_quantile=pose_min_observation_quantile,
        split_prob=max(0.0, float(runtime_config.get("split_prob", 0.05))),
        split_residual_guided=bool(
            runtime_config.get("split_residual_guided", True)
        ),
        split_residual_candidate_count=max(
            1,
            int(runtime_config.get("split_residual_candidate_count", 8)),
        ),
        merge_prob=max(0.0, float(runtime_config.get("merge_prob", 0.05))),
        merge_distance_max=max(
            0.0,
            float(runtime_config.get("merge_distance_max", 0.5)),
        ),
        merge_delta_ll_threshold=float(
            runtime_config.get("merge_delta_ll_threshold", 0.0)
        ),
        merge_response_corr_min=float(
            np.clip(
                float(runtime_config.get("merge_response_corr_min", 0.995)),
                0.0,
                1.0,
            )
        ),
        merge_search_topk_pairs=max(
            1,
            int(runtime_config.get("merge_search_topk_pairs", 8)),
        ),
        structural_proposal_topk_particles=structural_proposal_topk_particles,
        structural_update_min_counts=float(
            runtime_config.get("structural_update_min_counts", 0.0)
        ),
        birth_max_per_update=(
            None
            if runtime_config.get("birth_max_per_update", None) is None
            else int(runtime_config.get("birth_max_per_update", 0))
        ),
        birth_min_distinct_poses=max(
            1,
            int(runtime_config.get("birth_min_distinct_poses", 1)),
        ),
        birth_residual_min_support=max(
            1,
            int(runtime_config.get("birth_residual_min_support", 2)),
        ),
        birth_residual_support_sigma=max(
            0.0,
            float(runtime_config.get("birth_residual_support_sigma", 1.0)),
        ),
        birth_min_distinct_stations=max(
            1,
            int(runtime_config.get("birth_min_distinct_stations", 1)),
        ),
        source_detector_exclusion_m=max(
            0.0,
            float(runtime_config.get("source_detector_exclusion_m", 0.0)),
        ),
        birth_residual_gate_p_value=float(
            runtime_config.get("birth_residual_gate_p_value", 0.05)
        ),
        birth_candidate_support_fraction=float(
            runtime_config.get("birth_candidate_support_fraction", 0.05)
        ),
        birth_refit_residual_gate=bool(
            runtime_config.get("birth_refit_residual_gate", True)
        ),
        birth_refit_residual_min_fraction=max(
            0.0,
            float(runtime_config.get("birth_refit_residual_min_fraction", 0.5)),
        ),
        birth_jitter_topk_candidates=birth_jitter_topk_candidates,
        weak_source_prune_min_expected_count=max(
            0.0,
            float(runtime_config.get("weak_source_prune_min_expected_count", 3.0)),
        ),
        weak_source_prune_min_fraction=max(
            0.0,
            float(runtime_config.get("weak_source_prune_min_fraction", 0.001)),
        ),
        conditional_strength_refit=bool(
            runtime_config.get("conditional_strength_refit", True)
        ),
        conditional_strength_refit_window=max(
            1,
            int(runtime_config.get("conditional_strength_refit_window", 10)),
        ),
        conditional_strength_refit_iters=max(
            1,
            int(runtime_config.get("conditional_strength_refit_iters", 3)),
        ),
        conditional_strength_refit_reweight=bool(
            runtime_config.get("conditional_strength_refit_reweight", False)
        ),
        conditional_strength_refit_reweight_clip=max(
            0.0,
            float(runtime_config.get("conditional_strength_refit_reweight_clip", 50.0)),
        ),
        conditional_strength_refit_min_count=max(
            0.0,
            float(runtime_config.get("conditional_strength_refit_min_count", 5.0)),
        ),
        conditional_strength_refit_min_snr=max(
            0.0,
            float(runtime_config.get("conditional_strength_refit_min_snr", 1.0)),
        ),
        conditional_strength_refit_prior_weight=max(
            0.0,
            float(runtime_config.get("conditional_strength_refit_prior_weight", 0.0)),
        ),
        conditional_strength_refit_prior_rel_sigma=max(
            1.0e-6,
            float(runtime_config.get("conditional_strength_refit_prior_rel_sigma", 2.0)),
        ),
        report_strength_refit=bool(
            runtime_config.get("report_strength_refit", False)
        ),
        report_strength_refit_iters=max(
            1,
            int(runtime_config.get("report_strength_refit_iters", 64)),
        ),
        report_strength_refit_eps=max(
            1.0e-15,
            float(runtime_config.get("report_strength_refit_eps", 1.0e-9)),
        ),
        orientation_k=orientation_limit_resolved,
        min_rotations_per_pose=min_rotations_resolved,
        planning_eig_samples=int(runtime_config.get("planning_eig_samples", 50)),
        planning_rollout_particles=(
            None
            if runtime_config.get("planning_rollout_particles", 512) is None
            else int(runtime_config.get("planning_rollout_particles", 512))
        ),
        planning_rollout_method=str(
            runtime_config.get("planning_rollout_method", "resample")
        ),
        use_fast_gpu_rollout=True,
        use_gpu=use_gpu,
        gpu_device="cuda",
        gpu_dtype="float64",
        ig_workers=1,
        parallel_isotope_updates=bool(
            runtime_config.get("parallel_isotope_updates", True)
        ),
        parallel_isotope_workers=parallel_isotope_workers,
    )
    pf_conf.use_tempering = True
    pf_conf.max_temper_steps = 8
    pf_conf.min_delta_beta = 0.01
    pf_conf.target_ess_ratio = 0.4
    pf_conf.converge_enable = bool(
        converge
        or runtime_config.get("converge_enable", False)
        or adaptive_mission_stop
    )
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
    delayed_resample_update = bool(runtime_config.get("delayed_resample_update", True))
    joint_observation_update = (
        bool(runtime_config.get("joint_observation_update", False))
        and not delayed_resample_update
    )
    strict_planned_shield_program = bool(
        runtime_config.get(
            "strict_planned_shield_program",
            path_planner_resolved == "dss_pp",
        )
    )
    shield_signature_weight = max(
        0.0,
        float(runtime_config.get("shield_signature_weight", 0.25)),
    )
    shield_low_count_penalty_weight = max(
        0.0,
        float(runtime_config.get("shield_low_count_penalty_weight", 1.0)),
    )
    shield_count_balance_weight = max(
        0.0,
        float(runtime_config.get("shield_count_balance_weight", 0.25)),
    )
    shield_rotation_cost_weight = max(
        0.0,
        float(runtime_config.get("shield_rotation_cost_weight", 0.05)),
    )
    shield_signature_variance_floor = max(
        1e-12,
        float(runtime_config.get("shield_signature_variance_floor", 1.0)),
    )
    shield_stop_min_gain = max(
        0.0,
        float(runtime_config.get("shield_stop_min_gain", 0.0)),
    )
    shield_stop_compare_next_pose = bool(
        runtime_config.get("shield_stop_compare_next_pose", True)
    )
    shield_stop_pose_candidates = max(
        1,
        int(runtime_config.get("shield_stop_pose_candidates", min(pose_candidates, 16))),
    )
    shield_stop_rate_margin = max(
        0.0,
        float(runtime_config.get("shield_stop_rate_margin", 1.0)),
    )
    shield_stop_signature_cosine = float(
        runtime_config.get("shield_stop_signature_cosine", 0.995)
    )
    shield_selection_max_particles_raw = runtime_config.get(
        "shield_selection_max_particles",
        None,
    )
    if shield_selection_max_particles_raw is None:
        shield_selection_max_particles = (
            pf_conf.pose_min_observation_max_particles
            if pf_conf.pose_min_observation_max_particles is not None
            else (
                pf_conf.planning_rollout_particles
                or pf_conf.planning_particles
                or 256
            )
        )
    else:
        shield_selection_max_particles = int(shield_selection_max_particles_raw)
    if shield_selection_max_particles is not None:
        shield_selection_max_particles = max(1, int(shield_selection_max_particles))
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
    detector_model_payload = runtime_config.get("detector_model", {})
    if not isinstance(detector_model_payload, dict):
        detector_model_payload = {}
    pf_detector_radius_m = float(detector_model_payload.get("crystal_radius_m", 0.0)) + float(
        detector_model_payload.get("housing_thickness_m", 0.0)
    )
    pf_detector_aperture_samples = int(runtime_config.get("pf_detector_aperture_samples", 121))

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
            obstacle_buildup_coeff=obstacle_buildup_coeff,
            detector_radius_m=pf_detector_radius_m,
            detector_aperture_samples=pf_detector_aperture_samples,
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

    def _build_cui_split_visualizer() -> CUISplitPFVisualizer | None:
        """Create the CUI split visualizer when enabled."""
        if not cui_split_view_enabled:
            return None
        split_viz = CUISplitPFVisualizer(
            isotopes=isotopes,
            output_dir=cui_split_view_dir,
            world_bounds=(0, env.size_x, 0, env.size_y, 0, env.size_z),
            true_sources=true_src,
            true_strengths=true_strengths,
            obstacle_grid=obstacle_grid,
            max_particles_per_isotope=cui_split_max_particles,
        )
        print(
            "CUI split visualization enabled: "
            f"{split_viz.index_path.as_posix()} "
            "(latest_robot_2d.png, latest_pf_3d.png)"
        )
        return split_viz

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
    cui_split_viz = _build_cui_split_visualizer()
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
    last_spectrum_components: dict[str, NDArray[np.float64]] = {}
    last_counts: dict[str, float] | None = None
    representative_spectrum: np.ndarray | None = None
    representative_spectrum_components: dict[str, NDArray[np.float64]] = {}
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
    dwell_cap_label = f"{live_time:.3f}" if has_live_time_cap else "unbounded"
    dwell_step_label = f"{live_time:.1f}" if has_live_time_cap else "unbounded"
    if save_outputs:
        IG_DIR.mkdir(parents=True, exist_ok=True)
    print(
        "Spectrum config: "
        f"bin_width_keV={cfg.bin_width_keV}, live_time_s={dwell_cap_label}, "
        f"smooth_sigma_bins={cfg.smooth_sigma_bins}, "
        f"als_lambda={cfg.als_lambda}, als_p={cfg.als_p}, als_niter={cfg.als_niter}, "
        f"resolution_a={cfg.resolution_a}, resolution_b={cfg.resolution_b}, "
        f"peak_window_sigma={cfg.peak_window_sigma}, dead_time_tau_s={cfg.dead_time_tau_s}, "
        f"response_continuum_to_peak={cfg.response_continuum_to_peak}, "
        f"response_backscatter_fraction={cfg.response_backscatter_fraction}, "
        f"apply_incident_gamma_detector_response={cfg.apply_incident_gamma_detector_response}"
    )
    print(
        "Dwell control: "
        f"adaptive={bool(adaptive_dwell)} "
        f"cap_s={dwell_cap_label} "
        f"chunk_s={float(adaptive_dwell_chunk_s):.3f} "
        f"min_s={float(adaptive_min_dwell_s):.3f} "
        f"ready_min_counts={float(adaptive_ready_min_counts):.3f} "
        f"ready_min_isotopes={int(adaptive_ready_min_isotopes)} "
        f"ready_min_snr={float(adaptive_ready_min_snr):.3f} "
        f"allow_informative_low={bool(adaptive_ready_allow_informative_low)} "
        f"allow_low_signal_stop={bool(adaptive_allow_low_signal_stop)} "
        f"low_signal_min_s={float(adaptive_low_signal_min_live_s):.1f} "
        f"low_signal_sigma={float(adaptive_low_signal_upper_sigma):.1f} "
        f"low_signal_count_fraction={float(adaptive_low_signal_count_fraction):.3f} "
        "low_signal_projected_live_factor="
        f"{float(adaptive_low_signal_projected_live_factor):.2f}"
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
        f"aggregate={pf_conf.pose_min_observation_aggregate} "
        f"quantile={float(pf_conf.pose_min_observation_quantile):.3f}"
    )
    print(
        "Path planner: "
        f"mode={path_planner_resolved} "
        f"dss_horizon={int(dss_config.horizon)} "
        f"dss_beam={int(dss_config.beam_width)} "
        f"dss_program_len={int(dss_config.program_length)} "
        f"signature_w={float(dss_config.lambda_signature):.3f} "
        f"differential_w={float(dss_config.eta_differential):.3f} "
        f"count_balance_w={float(dss_config.eta_count_balance):.3f} "
        f"rotation_w={float(dss_config.lambda_rotation):.3f} "
        f"coverage_w={float(dss_config.lambda_coverage):.3f} "
        f"bearing_w={float(dss_config.lambda_bearing_diversity):.3f} "
        f"frontier_w={float(dss_config.lambda_frontier):.3f} "
        f"turn_w={float(dss_config.lambda_turn_smoothness):.3f} "
        f"revisit_w={float(dss_config.eta_revisit):.3f} "
        f"min_station_sep={float(dss_config.min_station_separation_m):.2f}m "
        f"enforce_min_obs={bool(dss_config.enforce_min_observation)}"
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
        "PF cardinality birth gate: "
        f"p_value={float(pf_conf.birth_residual_gate_p_value):.3g} "
        f"min_support={int(pf_conf.birth_residual_min_support)} "
        f"min_distinct_poses={int(pf_conf.birth_min_distinct_poses)} "
        f"min_distinct_stations={int(pf_conf.birth_min_distinct_stations)} "
        f"source_detector_exclusion_m={float(pf_conf.source_detector_exclusion_m):.3f} "
        f"support_sigma={float(pf_conf.birth_residual_support_sigma):.2f} "
        f"candidate_support_fraction={float(pf_conf.birth_candidate_support_fraction):.2f} "
        f"refit_gate={bool(pf_conf.birth_refit_residual_gate)} "
        f"refit_min_fraction={float(pf_conf.birth_refit_residual_min_fraction):.2f} "
        f"jitter_topk={pf_conf.birth_jitter_topk_candidates} "
        f"max_per_update={pf_conf.birth_max_per_update}"
    )
    print(
        "PF conditional strength refit: "
        f"enabled={bool(pf_conf.conditional_strength_refit)} "
        f"window={int(pf_conf.conditional_strength_refit_window)} "
        f"iters={int(pf_conf.conditional_strength_refit_iters)} "
        f"reweight={bool(pf_conf.conditional_strength_refit_reweight)} "
        f"reweight_clip={float(pf_conf.conditional_strength_refit_reweight_clip):.3f} "
        f"min_count={float(pf_conf.conditional_strength_refit_min_count):.3f} "
        f"min_snr={float(pf_conf.conditional_strength_refit_min_snr):.3f} "
        f"prior_weight={float(pf_conf.conditional_strength_refit_prior_weight):.3f} "
        f"prior_rel_sigma={float(pf_conf.conditional_strength_refit_prior_rel_sigma):.3f} "
        f"report_refit={bool(pf_conf.report_strength_refit)} "
        f"report_refit_iters={int(pf_conf.report_strength_refit_iters)} "
        f"weak_prune_min_counts={float(pf_conf.weak_source_prune_min_expected_count):.3f} "
        f"weak_prune_min_fraction={float(pf_conf.weak_source_prune_min_fraction):.4f}"
    )
    print(
        "PF shield-program update: "
        f"delayed_resample={bool(delayed_resample_update)} "
        f"legacy_joint={bool(joint_observation_update)}"
    )
    print(
        "Shield posture selector: "
        f"signature_w={float(shield_signature_weight):.3f} "
        f"low_count_w={float(shield_low_count_penalty_weight):.3f} "
        f"count_balance_w={float(shield_count_balance_weight):.3f} "
        f"rotation_w={float(shield_rotation_cost_weight):.3f} "
        f"max_particles={shield_selection_max_particles} "
        f"stop_min_gain={float(shield_stop_min_gain):.6g} "
        f"compare_next_pose={bool(shield_stop_compare_next_pose)} "
        f"rate_margin={float(shield_stop_rate_margin):.3f} "
        f"signature_cosine_stop={float(shield_stop_signature_cosine):.3f}"
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
        "PF source-position support: "
        f"min={tuple(round(float(v), 3) for v in source_position_min)} "
        f"max={tuple(round(float(v), 3) for v in source_position_max)}"
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
        f"init_grid_spacing_m={pf_conf.init_grid_spacing_m}, "
        f"init_grid_repeats={int(pf_conf.init_grid_repeats)}, "
        f"init_strength_log_mean={pf_conf.init_strength_log_mean:.2f}, "
        f"init_strength_log_sigma={pf_conf.init_strength_log_sigma:.2f}, "
        f"max_sources={pf_conf.max_sources}"
    )
    print(
        "Birth moves: "
        f"enabled={birth_enabled} "
        f"p_birth={pf_conf.p_birth:.3f} p_kill={pf_conf.p_kill:.3f} "
        f"split_prob={pf_conf.split_prob:.3f} merge_prob={pf_conf.merge_prob:.3f} "
        f"split_residual_guided={bool(pf_conf.split_residual_guided)} "
        f"split_candidates={int(pf_conf.split_residual_candidate_count)} "
        f"merge_corr_min={float(pf_conf.merge_response_corr_min):.3f} "
        f"merge_pairs={int(pf_conf.merge_search_topk_pairs)} "
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
    print(
        "PF parallelism: "
        f"parallel_isotope_updates={bool(estimator.pf_config.parallel_isotope_updates)} "
        f"parallel_isotope_workers={estimator.pf_config.parallel_isotope_workers}"
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
    print(
        "Adaptive mission stop: "
        f"enabled={adaptive_mission_stop} "
        f"min_poses={mission_stop_min_poses} "
        f"coverage_radius={mission_stop_coverage_radius_m:.2f}m "
        f"coverage_threshold={mission_stop_coverage_fraction:.3f} "
        f"quiet_birth_residual={mission_stop_require_quiet_birth_residual} "
        f"coverage_requires_pf_convergence="
        f"{mission_stop_require_pf_convergence_for_coverage} "
        f"birth_residual_min_support={mission_stop_birth_residual_min_support}"
    )
    reset_usd_path = (
        generated_blender_usd_path.as_posix()
        if generated_blender_usd_path is not None
        else ("" if obstacle_grid is None else None)
    )
    simulation_runtime.reset(
        {
            "usd_path": reset_usd_path,
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
            "use_config_usd_fallback": not (
                generated_blender_usd_path is None and obstacle_grid is None
            ),
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
            "birth_max_per_update": pf_conf.birth_max_per_update,
            "converge": converge,
            "pose_candidates": int(pose_candidates),
            "pose_min_dist_m": float(pose_min_dist),
            "candidate_grid_points": int(grid.shape[0]),
            "pf_num_particles": int(pf_conf.num_particles),
            "pf_max_sources": int(pf_conf.max_sources),
            "robot_speed_m_s": float(nominal_motion_speed_m_s),
            "rotation_overhead_s": float(rotation_overhead_s),
            "measurement_time_s": float(live_time),
            "measurement_time_cap_s": float(live_time) if has_live_time_cap else None,
            "adaptive_dwell": bool(adaptive_dwell),
            "adaptive_dwell_chunk_s": float(adaptive_dwell_chunk_s),
            "adaptive_min_dwell_s": float(adaptive_min_dwell_s),
            "adaptive_ready_min_counts": float(adaptive_ready_min_counts),
            "adaptive_ready_min_isotopes": int(adaptive_ready_min_isotopes),
            "adaptive_ready_min_snr": float(adaptive_ready_min_snr),
            "adaptive_ready_allow_informative_low": bool(
                adaptive_ready_allow_informative_low
            ),
            "adaptive_allow_low_signal_stop": bool(adaptive_allow_low_signal_stop),
            "adaptive_low_signal_min_live_s": float(adaptive_low_signal_min_live_s),
            "adaptive_low_signal_upper_sigma": float(
                adaptive_low_signal_upper_sigma
            ),
            "adaptive_low_signal_count_fraction": float(
                adaptive_low_signal_count_fraction
            ),
            "adaptive_low_signal_projected_live_factor": float(
                adaptive_low_signal_projected_live_factor
            ),
            "delayed_resample_update": bool(delayed_resample_update),
            "joint_observation_update": bool(joint_observation_update),
            "strict_planned_shield_program": bool(strict_planned_shield_program),
            "shield_signature_weight": float(shield_signature_weight),
            "shield_low_count_penalty_weight": float(
                shield_low_count_penalty_weight
            ),
            "shield_count_balance_weight": float(shield_count_balance_weight),
            "shield_rotation_cost_weight": float(shield_rotation_cost_weight),
            "shield_selection_max_particles": None
            if shield_selection_max_particles is None
            else int(shield_selection_max_particles),
            "shield_signature_variance_floor": float(
                shield_signature_variance_floor
            ),
            "shield_stop_min_gain": float(shield_stop_min_gain),
            "shield_stop_compare_next_pose": bool(shield_stop_compare_next_pose),
            "shield_stop_pose_candidates": int(shield_stop_pose_candidates),
            "shield_stop_rate_margin": float(shield_stop_rate_margin),
            "shield_stop_signature_cosine": float(shield_stop_signature_cosine),
            "pose_min_observation_counts": float(pf_conf.pose_min_observation_counts),
            "pose_min_observation_penalty_scale": float(
                pf_conf.pose_min_observation_penalty_scale
            ),
            "pose_min_observation_aggregate": pf_conf.pose_min_observation_aggregate,
            "pose_min_observation_quantile": float(
                pf_conf.pose_min_observation_quantile
            ),
            "path_planner": path_planner_resolved,
            "dss_horizon": int(dss_config.horizon),
            "dss_beam_width": int(dss_config.beam_width),
            "dss_program_length": int(dss_config.program_length),
            "dss_signature_weight": float(dss_config.lambda_signature),
            "dss_differential_weight": float(dss_config.eta_differential),
            "dss_rotation_weight": float(dss_config.lambda_rotation),
            "dss_coverage_weight": float(dss_config.lambda_coverage),
            "dss_revisit_penalty_weight": float(dss_config.eta_revisit),
            "dss_min_station_separation_m": float(
                dss_config.min_station_separation_m
            ),
        }
    )
    ig_max_global = 0.0
    pose_counter = 0
    current_shield_pair_id: int | None = None
    pending_shield_program: tuple[int, ...] | None = None
    try:
        while True:
            pose = current_pose
            stop_run = False
            pose_elapsed = 0.0
            zero_ig_override = False
            active_shield_program = pending_shield_program
            pending_shield_program = None
            if active_shield_program:
                print(
                    "Executing planned DSS-PP shield program at this pose: "
                    f"{list(active_shield_program)}"
                )
            remaining_orientations = set(range(total_pairs))
            rotation_limit = max(1, int(estimator.pf_config.orientation_k))
            joint_update_records: list[
                tuple[dict[str, float], int, int, float, dict[str, float] | None]
            ] = []
            deferred_update_records = 0
            executed_signature_vectors: list[NDArray[np.float64]] = []
            if delayed_resample_update:
                estimator.begin_deferred_pose_update()
            min_rotations_this_pose = min(
                rotation_limit,
                max(2, int(estimator.pf_config.min_rotations_per_pose)),
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
                    live_time_s=planning_live_time,
                    planning_isotopes=None,
                )
                ig_elapsed = time.perf_counter() - ig_start
                total_ig_wall_s += ig_elapsed
                ig_wall_samples_s.append(float(ig_elapsed))
                print(f"IG grid computed in {ig_elapsed:.3f}s.")
                shield_selection_start = time.perf_counter()
                shield_scores, shield_score_parts = _compute_shield_selection_grid(
                    estimator,
                    rot_mats,
                    pose_idx=current_pose_idx,
                    live_time_s=planning_live_time,
                    ig_scores=ig_scores,
                    current_pair_id=current_shield_pair_id,
                    min_observation_counts=float(pf_conf.pose_min_observation_counts),
                    signature_weight=float(shield_signature_weight),
                    low_count_penalty_weight=float(shield_low_count_penalty_weight),
                    count_balance_weight=float(shield_count_balance_weight),
                    rotation_cost_weight=float(shield_rotation_cost_weight),
                    variance_floor=float(shield_signature_variance_floor),
                    max_particles=shield_selection_max_particles,
                )
                shield_selection_elapsed = time.perf_counter() - shield_selection_start
                print(
                    "Shield selection grid computed in "
                    f"{shield_selection_elapsed:.3f}s."
                )
                planned_pair = None
                if (
                    strict_planned_shield_program
                    and active_shield_program
                    and rotation_count < len(active_shield_program)
                ):
                    candidate_pair = int(active_shield_program[rotation_count])
                    if candidate_pair in remaining_orientations:
                        planned_pair = candidate_pair
                if planned_pair is None:
                    best_pair_idx, shield_score = _select_best_pair_from_scores(
                        shield_scores,
                        remaining_orientations,
                    )
                    using_planned_pair = False
                else:
                    best_pair_idx = int(planned_pair)
                    fe_tmp = best_pair_idx // num_orients
                    pb_tmp = best_pair_idx % num_orients
                    shield_score = float(shield_scores[fe_tmp, pb_tmp])
                    using_planned_pair = True
                if best_pair_idx < 0:
                    print("No valid orientation candidates; moving to the next pose.")
                    break
                fe_for_score = best_pair_idx // num_orients
                pb_for_score = best_pair_idx % num_orients
                raw_ig_val = max(float(ig_scores[fe_for_score, pb_for_score]), 0.0)
                signature_val = float(
                    shield_score_parts["signature"][fe_for_score, pb_for_score]
                )
                signature_utility_val = float(
                    shield_score_parts["signature_utility"][fe_for_score, pb_for_score]
                )
                low_count_penalty = float(
                    shield_score_parts["low_count_penalty"][fe_for_score, pb_for_score]
                )
                count_balance_penalty = float(
                    shield_score_parts["count_balance_penalty"][
                        fe_for_score,
                        pb_for_score,
                    ]
                )
                rotation_cost = float(
                    shield_score_parts["rotation_cost"][fe_for_score, pb_for_score]
                )
                shield_gain = float(shield_score)
                ig_val = max(raw_ig_val, shield_gain, 0.0)
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
                predicted_counts = estimator.expected_observation_counts_by_isotope_at_pair(
                    pose_idx=current_pose_idx,
                    fe_index=fe_for_score,
                    pb_index=pb_for_score,
                    live_time_s=planning_live_time,
                    max_particles=shield_selection_max_particles,
                )
                signature_vector = np.asarray(
                    [
                        float(predicted_counts.get(iso, 0.0))
                        for iso in estimator.isotopes
                    ],
                    dtype=float,
                )
                signature_dependent = _signature_vector_is_dependent(
                    signature_vector,
                    executed_signature_vectors,
                    cosine_threshold=float(shield_stop_signature_cosine),
                )
                shield_gain_rate = max(shield_gain, 0.0) / max(
                    float(planning_live_time) + float(rotation_overhead_s),
                    1e-9,
                )
                next_pose_gain_rate = 0.0
                next_pose_gain = 0.0
                next_pose_candidate_idx = -1
                stop_reason = "continue"
                if (
                    rotation_count >= min_rotations_this_pose
                    and not using_planned_pair
                    and shield_gain < float(shield_stop_min_gain)
                ):
                    stop_reason = "shield_gain_below_absolute_threshold"
                elif (
                    ig_val < ig_threshold_current
                    and rotation_count >= min_rotations_this_pose
                    and not using_planned_pair
                ):
                    stop_reason = "shield_gain_below_ig_threshold"
                elif (
                    signature_dependent
                    and rotation_count >= min_rotations_this_pose
                    and not using_planned_pair
                ):
                    stop_reason = "signature_linearly_dependent"
                elif (
                    shield_stop_compare_next_pose
                    and rotation_count >= min_rotations_this_pose
                    and not using_planned_pair
                    and (max_poses is None or pose_counter + 1 < max_poses)
                ):
                    visited_for_stop = list(visited_poses) + [pose.copy()]
                    visited_stop_arr = (
                        np.vstack(visited_for_stop) if visited_for_stop else None
                    )
                    stop_candidates, _, _ = _generate_planning_candidates(
                        current_pose_xyz=pose,
                        map_api=planning_map,
                        n_candidates=int(shield_stop_pose_candidates),
                        min_dist_from_visited=pose_min_dist,
                        visited_poses_xyz=visited_stop_arr,
                        bounds_xyz=(bounds_lo, bounds_hi),
                    )
                    (
                        next_pose_gain_rate,
                        next_pose_gain,
                        next_pose_candidate_idx,
                    ) = _estimate_best_next_pose_gain_rate(
                        estimator,
                        candidates=stop_candidates,
                        current_pose_xyz=pose,
                        map_api=planning_map,
                        live_time_s=planning_live_time,
                        rotation_limit=rotation_limit,
                        nominal_motion_speed_m_s=nominal_motion_speed_m_s,
                        rotation_overhead_s=rotation_overhead_s,
                        max_candidates=int(shield_stop_pose_candidates),
                    )
                    if (
                        next_pose_gain_rate > 0.0
                        and shield_gain_rate
                        < next_pose_gain_rate * float(shield_stop_rate_margin)
                    ):
                        stop_reason = "next_pose_gain_rate_higher"
                if stop_reason != "continue":
                    print(
                        "Stopping rotation at this pose "
                        f"(reason={stop_reason}, "
                        f"shield_gain={shield_gain:.6g}, "
                        f"raw_ig={raw_ig_val:.6g}, "
                        f"threshold={ig_threshold_current:.6g}, "
                        f"shield_gain_rate={shield_gain_rate:.6g}, "
                        f"next_pose_gain={next_pose_gain:.6g}, "
                        f"next_pose_gain_rate={next_pose_gain_rate:.6g}, "
                        f"next_pose_candidate={next_pose_candidate_idx})."
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
                    adaptive_ready_min_snr=float(adaptive_ready_min_snr),
                    spectrum_count_method=spectrum_count_method,
                    detect_threshold_abs=detect_threshold_abs,
                    detect_threshold_rel=detect_threshold_rel,
                    detect_threshold_rel_by_isotope=detect_threshold_rel_by_isotope,
                    min_peaks_by_isotope=min_peaks_by_isotope,
                    adaptive_progress_every_chunks=(
                        10 if bool(adaptive_dwell) and not has_live_time_cap else 0
                    ),
                    adaptive_ready_allow_informative_low=(
                        adaptive_ready_allow_informative_low
                    ),
                    adaptive_allow_low_signal_stop=adaptive_allow_low_signal_stop,
                    adaptive_low_signal_min_live_s=adaptive_low_signal_min_live_s,
                    adaptive_low_signal_upper_sigma=adaptive_low_signal_upper_sigma,
                    adaptive_low_signal_count_fraction=(
                        adaptive_low_signal_count_fraction
                    ),
                    adaptive_low_signal_projected_live_factor=(
                        adaptive_low_signal_projected_live_factor
                    ),
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
                spectrum = _analysis_spectrum_array(observation, decomposer)
                last_spectrum = spectrum.copy()
                last_counts = {iso: float(val) for iso, val in z_detected.items()}
                last_spectrum_components = {
                    iso: np.asarray(component, dtype=float).copy()
                    for iso, component in getattr(
                        decomposer,
                        "last_response_poisson_components",
                        {},
                    ).items()
                }
                last_candidates = set(detected)
                spectrum_total_counts = float(np.sum(spectrum))
                if spectrum_total_counts > representative_total_counts:
                    representative_total_counts = spectrum_total_counts
                    representative_spectrum = spectrum.copy()
                    representative_spectrum_components = {
                        iso: component.copy()
                        for iso, component in last_spectrum_components.items()
                    }
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
                if str(observation.metadata.get("backend", "")).lower() == "geant4":
                    if spectrum_count_method == "response_poisson":
                        response_poisson_counts = {
                            iso: float(z_k_full.get(iso, 0.0))
                            for iso in pf_isotopes
                        }
                    else:
                        response_poisson_counts = _response_poisson_counts_for_diagnostics(
                            diagnostic_decomposer,
                            spectrum,
                            pf_isotopes,
                        )
                    source_equivalent_counts = _source_equivalent_counts_from_metadata(
                        observation.metadata,
                        pf_isotopes,
                    )
                    transport_detected_counts = _transport_detected_counts_from_metadata(
                        observation.metadata,
                        pf_isotopes,
                    )
                    _log_geant4_transport_decomposition_diagnostics(
                        step_index=step_counter,
                        metadata=observation.metadata,
                        spectrum_total_counts=spectrum_total_counts,
                        selected_count_method=spectrum_count_method,
                        selected_counts=z_k_full,
                        response_poisson_counts=response_poisson_counts,
                        source_equivalent_counts=source_equivalent_counts,
                        transport_detected_counts=transport_detected_counts,
                    )
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
                if joint_observation_update:
                    joint_update_records.append(
                        (
                            dict(z_k),
                            int(fe_idx),
                            int(pb_idx),
                            float(actual_live_time_s),
                            dict(z_variance_full),
                        )
                    )
                else:
                    estimator.update_pair(
                        z_k=z_k,
                        pose_idx=current_pose_idx,
                        fe_index=fe_idx,
                        pb_index=pb_idx,
                        live_time_s=actual_live_time_s,
                        z_variance_k=z_variance_full,
                    )
                    if delayed_resample_update:
                        deferred_update_records += 1
                if (
                    not joint_observation_update
                    and estimator.last_strength_prior_diagnostics
                ):
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
                if not joint_observation_update:
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
                if step_path_segment is not None:
                    waypoints_payload = step_path_segment.get("waypoints_xyz")
                    if waypoints_payload is not None:
                        waypoint_array = np.asarray(
                            waypoints_payload,
                            dtype=float,
                        )
                        if isinstance(frame, dict):
                            frame["path_waypoints_xyz"] = waypoint_array
                        else:
                            frame.path_waypoints_xyz = waypoint_array
                spectrum_components_payload = {
                    iso: component.copy()
                    for iso, component in last_spectrum_components.items()
                }
                if isinstance(frame, dict):
                    frame["spectrum_energy_keV"] = decomposer.energy_axis.copy()
                    frame["spectrum_counts"] = spectrum.copy()
                    frame["spectrum_components_by_isotope"] = spectrum_components_payload
                else:
                    frame.spectrum_energy_keV = decomposer.energy_axis.copy()
                    frame.spectrum_counts = spectrum.copy()
                    frame.spectrum_components_by_isotope = spectrum_components_payload
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
                if cui_split_viz is not None:
                    cui_split_viz.update(frame)
                last_frame = frame
                print(
                    f"[step {step_counter}] pose={_fmt_pos(pose_for_pf)} "
                    f"orient_pair={best_pair_idx} "
                    f"planned_pair={using_planned_pair} "
                    f"shield_gain={shield_gain:.6g} raw_ig={raw_ig_val:.6g} "
                    f"signature={signature_val:.6g} "
                    f"signature_utility={signature_utility_val:.6g} "
                    f"low_count_penalty={low_count_penalty:.6g} "
                    f"count_balance_penalty={count_balance_penalty:.6g} "
                    f"rotation_cost={rotation_cost:.6g} "
                    f"shield_gain_rate={shield_gain_rate:.6g} "
                    f"next_pose_gain={next_pose_gain:.6g} "
                    f"next_pose_gain_rate={next_pose_gain_rate:.6g} "
                    f"stop_reason={stop_reason} "
                    f"ig_threshold={ig_threshold_current:.6g} "
                    f"fe_idx={fe_idx} pb_idx={pb_idx} "
                    f"travel_distance_m={step_motion_distance_m:.3f} "
                    f"travel_time_s={step_motion_time_s:.1f} "
                    f"shield_time_s={step_rotation_time_s:.1f} "
                    f"live_time_s={actual_live_time_s:.1f}/{dwell_step_label} "
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
                current_shield_pair_id = int(best_pair_idx)
                executed_signature_vectors.append(signature_vector.copy())
                if save_outputs and last_spectrum is not None:
                    highlight = set(last_candidates)
                    spectrum_path = SPECTRUM_DIR / f"spectrum_step_{step_counter:04d}.png"
                    _save_spectrum_plot(
                        decomposer,
                        last_spectrum,
                        spectrum_path,
                        highlight_isotopes=highlight,
                        counts_by_isotope=last_counts,
                        component_spectra_by_isotope=last_spectrum_components,
                        title=f"Processed measurement spectrum (step {step_counter})",
                    )
                if max_steps is not None and step_counter >= max_steps:
                    stop_run = True
                    break
                pose_elapsed += actual_live_time_s + step_rotation_time_s
                if pose_elapsed >= estimator.pf_config.max_dwell_time_s:
                    break
            if delayed_resample_update:
                pf_start = time.perf_counter()
                finalized_measurements = estimator.finalize_deferred_pose_update()
                pf_elapsed = time.perf_counter() - pf_start
                if finalized_measurements > 0:
                    total_pf_wall_s += pf_elapsed
                    per_measurement_pf = pf_elapsed / max(finalized_measurements, 1)
                    pf_wall_samples_s.extend(
                        [float(per_measurement_pf)] * finalized_measurements
                    )
                    print(
                        f"[pose {current_pose_idx}] delayed_pf_finalize "
                        f"measurements={finalized_measurements} "
                        f"likelihood_updates={deferred_update_records} "
                        f"pf={pf_elapsed:.3f}s "
                        f"per_measurement={per_measurement_pf:.3f}s"
                    )
                    _log_pf_diagnostics(estimator, max(step_counter - 1, 0))
            elif joint_observation_update and joint_update_records:
                pf_start = time.perf_counter()
                if len(joint_update_records) == 1:
                    (
                        z_joint,
                        fe_joint,
                        pb_joint,
                        live_joint,
                        var_joint,
                    ) = joint_update_records[0]
                    estimator.update_pair(
                        z_k=z_joint,
                        pose_idx=current_pose_idx,
                        fe_index=fe_joint,
                        pb_index=pb_joint,
                        live_time_s=live_joint,
                        z_variance_k=var_joint,
                    )
                else:
                    estimator.update_pair_sequence(
                        joint_update_records,
                        pose_idx=current_pose_idx,
                    )
                if estimator.last_strength_prior_diagnostics:
                    for iso, stats in sorted(estimator.last_strength_prior_diagnostics.items()):
                        print(
                            f"[pose {current_pose_idx}] strength_prior[{iso}] "
                            f"z={stats['observed_counts']:.2f} "
                            f"target={stats['target_counts']:.2f} "
                            f"median_before={stats['before_median_strength']:.2f} "
                            f"median_after={stats['after_median_strength']:.2f} "
                            f"particles={int(stats['particles_changed'])}"
                        )
                pf_elapsed = time.perf_counter() - pf_start
                total_pf_wall_s += pf_elapsed
                per_measurement_pf = pf_elapsed / max(len(joint_update_records), 1)
                pf_wall_samples_s.extend(
                    [float(per_measurement_pf)] * len(joint_update_records)
                )
                print(
                    f"[pose {current_pose_idx}] joint_pf_update "
                    f"measurements={len(joint_update_records)} "
                    f"pf={pf_elapsed:.3f}s "
                    f"per_measurement={per_measurement_pf:.3f}s"
                )
                _log_pf_diagnostics(estimator, max(step_counter - 1, 0))
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
                    live_time_s=planning_live_time,
                ):
                    print(
                        "Converged; stopping exploration "
                        f"(max IG {last_max_ig:.6g} < threshold {ig_threshold_current:.6g})."
                    )
                    break
            visited_poses.append(pose.copy())
            pose_counter += 1
            if adaptive_mission_stop:
                stop_reason = _adaptive_mission_stop_reason(
                    estimator,
                    current_pose_idx=current_pose_idx,
                    visited_poses_xyz=visited_poses,
                    map_api=planning_map,
                    min_poses=mission_stop_min_poses,
                    coverage_radius_m=mission_stop_coverage_radius_m,
                    coverage_fraction_threshold=mission_stop_coverage_fraction,
                    ig_threshold=ig_threshold_current,
                    planning_live_time_s=planning_live_time,
                    require_quiet_birth_residual=mission_stop_require_quiet_birth_residual,
                    birth_residual_min_support=mission_stop_birth_residual_min_support,
                    require_pf_convergence_for_coverage=(
                        mission_stop_require_pf_convergence_for_coverage
                    ),
                )
                if stop_reason is not None:
                    print(f"Adaptive mission stop: {stop_reason}.")
                    break
            if max_poses is not None and pose_counter >= max_poses:
                print(f"Reached max poses ({max_poses}); stopping exploration.")
                break
            visited_arr = np.vstack(visited_poses) if visited_poses else None
            print("Generating candidate poses for next measurement point...")
            candidates, relaxed_retry, candidate_min_dist = _generate_planning_candidates(
                current_pose_xyz=pose,
                map_api=planning_map,
                n_candidates=pose_candidates,
                min_dist_from_visited=pose_min_dist,
                visited_poses_xyz=visited_arr,
                bounds_xyz=(bounds_lo, bounds_hi),
            )
            if relaxed_retry:
                print(
                    "No candidates with current spacing; retrying with min_dist="
                    f"{candidate_min_dist:.2f}."
                )
            if candidates.size == 0:
                print("No candidate poses available; stopping exploration.")
                break
            print(f"Generated {len(candidates)} candidate poses. Computing best next pose...")
            planned_program_for_next: tuple[int, ...] | None = None
            dss_diagnostics: dict[str, float | int | str] | None = None
            if path_planner_resolved == "dss_pp":
                dss_start = time.perf_counter()
                dss_result = select_dss_pp_next_station(
                    estimator=estimator,
                    candidate_poses_xyz=candidates,
                    current_pose_xyz=pose,
                    current_pair_id=current_shield_pair_id,
                    visited_poses_xyz=visited_arr,
                    map_api=planning_map,
                    bounds_xyz=(bounds_lo, bounds_hi),
                    config=dss_config,
                )
                dss_elapsed = time.perf_counter() - dss_start
                next_pose = dss_result.next_pose
                planned_program_for_next = tuple(
                    int(pair_id) for pair_id in dss_result.shield_program.pair_ids
                )
                dss_diagnostics = dict(dss_result.diagnostics)
                print(
                    "DSS-PP selected next station: "
                    f"pose={next_pose.tolist()} "
                    f"program={dss_result.shield_program.name} "
                    f"pairs={list(planned_program_for_next)} "
                    f"score={float(dss_result.score):.6g} "
                    f"signature={float(dss_result.sequence[0].signature_score):.6g} "
                    f"obs_penalty={float(dss_result.sequence[0].observation_penalty):.6g} "
                    f"diff_penalty={float(dss_result.sequence[0].differential_penalty):.6g} "
                    f"coverage_gain={float(dss_result.sequence[0].coverage_gain):.6g} "
                    f"revisit_penalty={float(dss_result.sequence[0].revisit_penalty):.6g} "
                    f"bearing_gain={float(dss_result.sequence[0].bearing_diversity_gain):.6g} "
                    f"frontier_gain={float(dss_result.sequence[0].frontier_gain):.6g} "
                    f"turn_penalty={float(dss_result.sequence[0].turn_penalty):.6g} "
                    f"compute={dss_elapsed:.3f}s"
                )
            else:
                next_idx = select_next_pose_from_candidates(
                    estimator=estimator,
                    candidate_poses_xyz=candidates,
                    current_pose_xyz=pose,
                    criterion="after_rotation",
                    t_max_s=float(rotation_limit) * float(planning_live_time),
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
            pending_path_segment = _build_robot_path_segment(
                map_api=planning_map,
                from_pose_xyz=pose,
                to_pose_xyz=next_pose,
                nominal_motion_speed_m_s=nominal_motion_speed_m_s,
                path_planner=path_planner_resolved,
                planned_shield_program=planned_program_for_next,
                dss_diagnostics=dss_diagnostics,
            )
            motion_distance_m = float(pending_path_segment["distance_m"])
            motion_time_s = float(pending_path_segment["travel_time_s"])
            pending_motion_distance_m = motion_distance_m
            pending_motion_time_s = motion_time_s
            pending_shield_program = planned_program_for_next
            print(
                "Robot travel segment: "
                f"distance={motion_distance_m:.3f}m "
                f"euclidean={float(pending_path_segment['euclidean_distance_m']):.3f}m "
                f"time={motion_time_s:.1f}s "
                f"speed={float(nominal_motion_speed_m_s):.3f}m/s "
                f"obstacle_aware={bool(pending_path_segment['obstacle_aware'])}"
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
        if cui_split_viz is not None:
            result_paths.update(
                {
                    "cui_split_view": cui_split_viz.index_path.as_posix(),
                    "cui_robot_2d_latest": cui_split_viz.latest_robot_path.as_posix(),
                    "cui_pf_3d_latest": cui_split_viz.latest_pf_path.as_posix(),
                }
            )
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
            if cui_split_viz is not None:
                cui_split_viz.update(last_frame)
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
                component_spectra_by_isotope=representative_spectrum_components,
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
                component_spectra_by_isotope=last_spectrum_components,
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
        "measurement_time_cap_s": float(live_time) if has_live_time_cap else None,
        "adaptive_dwell_enabled": bool(adaptive_dwell),
        "adaptive_dwell_chunk_s": float(adaptive_dwell_chunk_s),
        "adaptive_min_dwell_s": float(adaptive_min_dwell_s),
        "adaptive_ready_min_counts": float(adaptive_ready_min_counts),
        "adaptive_ready_min_isotopes": int(adaptive_ready_min_isotopes),
        "adaptive_ready_min_snr": float(adaptive_ready_min_snr),
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
        "path_planner": path_planner_resolved,
        "dss_horizon": int(dss_config.horizon),
        "dss_beam_width": int(dss_config.beam_width),
        "dss_program_length": int(dss_config.program_length),
        "dss_signature_weight": float(dss_config.lambda_signature),
        "dss_differential_weight": float(dss_config.eta_differential),
        "dss_rotation_weight": float(dss_config.lambda_rotation),
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
        if cui_split_viz is not None:
            print(f"CUI split view saved to: {cui_split_viz.index_path}")
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
        use_pruning=True,
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
